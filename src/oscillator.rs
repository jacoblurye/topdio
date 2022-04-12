use rand::Rng;
use std::{
    fmt::Debug,
    iter::Sum,
    sync::{
        mpsc::{channel, sync_channel, Sender},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
};

use anyhow::{Context, Result};
use cpal::{
    traits::{DeviceTrait, StreamTrait},
    Device, Sample, Stream, StreamConfig,
};

use crate::topdio::{TopdioMessage, TopdioSubscriber};

const TWO_PI: f32 = 2.0 * std::f32::consts::PI;
const BUFFER_SIZE: usize = 1024;

/// Describes all parameters needed to determine an [`Wave`]'s next sample.
#[derive(Debug, Clone, Copy)]
pub struct OscillatorState {
    pub freq: f32,
    pub gain: f32,
    pub phase: f32,
}

pub fn sine<S: Sample>(state: &OscillatorState) -> S {
    let sample = state.gain * (TWO_PI * state.phase).sin();
    Sample::from(&sample)
}

pub fn triangle<S: Sample>(state: &OscillatorState) -> S {
    let sample = 4.0 * state.gain * (state.phase - (state.phase + 0.5).floor()).abs() - state.gain;
    Sample::from(&sample)
}

pub type Wave<S> = fn(&OscillatorState) -> S;

/// Represents a piecewise function that decreases or increases linearly
/// at a given rate to a given target value.
struct Ramp {
    rate: f32,
    target: f32,
}

impl Ramp {
    fn new(target: f32, rate: f32) -> Ramp {
        Ramp { target, rate }
    }

    fn next(&self, val: f32) -> f32 {
        if (val - self.target).abs() < f32::EPSILON {
            val
        } else if val > self.target {
            (val - self.rate).max(self.target)
        } else {
            (val + self.rate).min(self.target)
        }
    }
}

/// An [`Oscillator`] generates samples for a given wave function. It supports
/// smooth frequency updates.
pub struct Oscillator<S>
where
    S: Sample + Send + 'static,
{
    wave: Wave<S>,
    state: OscillatorState,
    gain_ramp: Ramp,
    freq_ramp: Ramp,
    sample_rate: f32,
}

impl<S> Oscillator<S>
where
    S: Sample + Send + 'static,
{
    pub fn new(wave: Wave<S>, freq: f32, gain: f32, sample_rate: f32) -> Oscillator<S> {
        let freq = freq.max(0.0);
        let gain = gain.clamp(0.0, 1.0);

        Oscillator {
            wave,
            sample_rate,
            state: OscillatorState {
                freq,
                gain: 0.,
                phase: 0.,
            },
            gain_ramp: Ramp::new(gain, gain / sample_rate),
            freq_ramp: Ramp::new(freq, 0.0),
        }
    }

    pub fn next_sample(&mut self) -> S {
        let mut state = &mut self.state;
        state.phase = (state.phase + state.freq / self.sample_rate) % 1.0;
        state.freq = self.freq_ramp.next(state.freq);
        state.gain = self.gain_ramp.next(state.gain);
        (self.wave)(state)
    }

    pub fn set_freq(&mut self, freq: f32) {
        let rate = ((freq - self.state.freq) / (self.sample_rate)).abs();
        self.freq_ramp = Ramp::new(freq, rate)
    }
}

struct StreamHandle {
    stop_tx: Sender<()>,
    handle: JoinHandle<()>,
    _stream: Stream,
}

/// An [`OscillatorManager`] controls a group of oscillators and their associated
/// audio output stream. It can act as a [`TopdioSubscriber`], translating system
/// statistics into changes that affect the oscillators.
pub struct OscillatorManager<S>
where
    S: Sample + PartialEq + Sum + Debug + Send + 'static,
{
    oscillators: Arc<Mutex<Vec<Oscillator<S>>>>,
    config: Arc<StreamConfig>,
    device: Arc<Device>,
    freq_scale: f32,
    stream_handle: Option<StreamHandle>,
}

impl<S> OscillatorManager<S>
where
    S: Sample + PartialEq + Sum + Debug + Send + 'static,
{
    /// Create a new [`OscillatorManager`] with the given number of oscillators, all with the
    /// given wave function. Calling this function does not create an audio output stream.
    pub fn new(
        wave: Wave<S>,
        num_oscillators: usize,
        freq_scale: f32,
        config: Arc<StreamConfig>,
        device: Arc<Device>,
    ) -> OscillatorManager<S> {
        let mut oscillators = vec![];
        let mut rng = rand::thread_rng();
        let normalized_gain = 1. / (num_oscillators as f32);
        let sample_rate = config.sample_rate.0 as f32;
        for _ in 0..num_oscillators {
            // Randomly initialize the oscillator frequency between 0 and 1 to avoid weird
            // constructive interference when the stream starts.
            let freq = rng.gen::<f32>();
            let oscillator = Oscillator::new(wave, freq, normalized_gain, sample_rate);
            oscillators.push(oscillator);
        }
        let oscillators = Arc::new(Mutex::new(oscillators));

        OscillatorManager {
            oscillators,
            config,
            device,
            freq_scale,
            stream_handle: None,
        }
    }

    /// Create and start an audio output stream for the managed oscillators. Calling this
    /// function more than once without calling [`stop()`] is a no-op.
    pub fn play(&mut self) -> Result<()> {
        if self.stream_handle.is_some() {
            return Ok(());
        }

        // Start a thread that gets the next sample from every oscillator, sums
        // them together, and sends the summed sample to a bounded channel.
        let oscillators = self.oscillators.clone();
        let (summed_tx, summed_rx) = sync_channel::<S>(BUFFER_SIZE);
        let (stop_tx, stop_rx) = channel::<()>();
        let handle = thread::spawn(move || loop {
            if stop_rx.try_recv().is_ok() {
                break;
            }
            let sum: S = {
                let mut oscillators = oscillators.lock().unwrap();
                oscillators
                    .iter_mut()
                    .map(|oscillator| oscillator.next_sample())
                    .sum()
            };
            if summed_tx.send(sum).is_err() {
                break;
            }
        });

        // Create the audio output stream. The stream callback receives new samples
        // from the summed channel and writes them to the output buffer.
        let stream = self.device.build_output_stream(
            &self.config.clone(),
            move |buf: &mut [S], _| {
                for frame in buf.chunks_mut(2) {
                    let value = summed_rx.recv().unwrap();
                    for sample in frame.iter_mut() {
                        *sample = value;
                    }
                }
            },
            |err| panic!("{:?}", err),
        )?;
        stream.play()?;

        self.stream_handle = Some(StreamHandle {
            stop_tx,
            handle,
            // Ensure the stream isn't dropped until the StreamHandle is dropped.
            _stream: stream,
        });

        Ok(())
    }

    /// Stop a running audio output stream, if there is one. If there isn't one
    /// calling this function is a no-op.
    pub fn stop(&mut self) -> Result<()> {
        if let Some(stream_handle) = self.stream_handle.take() {
            stream_handle.stop_tx.send(())?;
            stream_handle.handle.join().unwrap();
        };
        // stream_handle._stream is stopped automatically when it's dropped.
        Ok(())
    }
}

impl<S> TopdioSubscriber for OscillatorManager<S>
where
    S: Sample + PartialEq + Sum + Debug + Send + 'static,
{
    fn handle(&mut self, message: &TopdioMessage) -> Result<()> {
        match message {
            TopdioMessage::Stats { processes } => {
                // Acquire lock on oscillators
                {
                    let mut oscillators = self.oscillators.lock().unwrap();
                    let num_oscillators = oscillators.len();

                    // Update each oscillator's frequency based on the process CPU usage statistics.
                    for (index, process) in processes.iter().take(num_oscillators).enumerate() {
                        oscillators[index].set_freq(self.freq_scale * process.cpu_usage);
                    }
                } // Release lock on oscillators

                // Start the output stream if it hasn't already been started.
                if self.stream_handle.is_none() {
                    self.play().context("oscillator manager failed to play")?;
                }
            }
            TopdioMessage::Stop => {
                self.stop()
                    .context("oscillator manager failed while stopping")?;
            }
        };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use cpal::{traits::HostTrait, Host};

    use crate::topdio::ProcessInfo;

    use super::*;

    #[test]
    fn test_ramp() {
        let ramp = Ramp::new(1.0, 0.5);
        assert_eq!(ramp.next(1.0), 1.0);
        // Ramp up.
        assert_eq!(ramp.next(0.1), 0.6);
        assert_eq!(ramp.next(0.6), 1.0);
        // Ramp down.
        assert_eq!(ramp.next(0.1), 0.6);
        assert_eq!(ramp.next(0.6), 1.0);
    }

    fn next_n_samples(oscillator: &mut Oscillator<f32>, n: usize) -> Vec<f32> {
        let mut samples = Vec::<f32>::new();
        for _ in 0..n {
            let rounded = (oscillator.next_sample() * 100.).round() / 100.;
            samples.push(rounded);
        }
        samples
    }

    #[test]
    fn test_oscillator_sine() {
        let mut oscillator = Oscillator::new(sine::<f32>, 2.0, 1.0, 16.);

        assert_eq!(
            next_n_samples(&mut oscillator, 32),
            vec![
                0.04, 0.13, 0.13, -0.0, -0.22, -0.38, -0.31, 0.0, 0.4, 0.63, 0.49, -0.0, -0.57,
                -0.88, -0.66, 0.0, 0.71, 1.0, 0.71, -0.0, -0.71, -1.0, -0.71, 0.0, 0.71, 1.0, 0.71,
                -0.0, -0.71, -1.0, -0.71, 0.0
            ],
            "gain ramp mismatch"
        );

        oscillator.set_freq(1.0);
        assert_eq!(
            next_n_samples(&mut oscillator, 48),
            vec![
                0.71, 1.0, 0.76, 0.15, -0.51, -0.93, -0.96, -0.63, -0.1, 0.45, 0.84, 1.0, 0.9,
                0.62, 0.22, -0.2, -0.56, -0.83, -0.98, -0.98, -0.83, -0.56, -0.2, 0.2, 0.56, 0.83,
                0.98, 0.98, 0.83, 0.56, 0.2, -0.2, -0.56, -0.83, -0.98, -0.98, -0.83, -0.56, -0.2,
                0.2, 0.56, 0.83, 0.98, 0.98, 0.83, 0.56, 0.2, -0.2
            ],
            "frequency ramp mismatch"
        );
    }

    #[test]
    fn test_oscillator_triangle() {
        let mut oscillator = Oscillator::new(triangle::<f32>, 2.0, 1.0, 16.);

        assert_eq!(
            next_n_samples(&mut oscillator, 32),
            vec![
                -0.03, 0.0, 0.09, 0.25, 0.16, 0.0, -0.22, -0.5, -0.28, 0.0, 0.34, 0.75, 0.41, 0.0,
                -0.47, -1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 1.0, 0.5,
                0.0, -0.5, -1.0
            ],
            "gain ramp mismatch"
        );

        oscillator.set_freq(1.0);
        assert_eq!(
            next_n_samples(&mut oscillator, 48),
            vec![
                -0.5, -0.02, 0.45, 0.91, 0.66, 0.23, -0.17, -0.56, -0.94, -0.7, -0.36, -0.03, 0.28,
                0.58, 0.86, 0.88, 0.63, 0.38, 0.13, -0.13, -0.38, -0.63, -0.88, -0.88, -0.63,
                -0.38, -0.13, 0.13, 0.38, 0.63, 0.88, 0.88, 0.63, 0.38, 0.13, -0.13, -0.38, -0.63,
                -0.88, -0.88, -0.63, -0.38, -0.13, 0.13, 0.38, 0.63, 0.88, 0.88
            ],
            "frequency ramp mismatch"
        );
    }

    #[test]
    fn test_oscillator_manager() {
        let wave = |_: &OscillatorState| 0.;
        let host: Host = cpal::default_host();
        let output: Device = host.default_output_device().unwrap();
        let config: StreamConfig = output.default_output_config().unwrap().into();

        let mut oscillator_manager =
            OscillatorManager::new(wave, 3, 1., Arc::new(config), Arc::new(output));

        // Handle a stats message.
        let message = TopdioMessage::Stats {
            processes: vec![
                ProcessInfo {
                    pid: 1,
                    name: "p1".to_string(),
                    cpu_usage: 5.0,
                },
                ProcessInfo {
                    pid: 2,
                    name: "p2".to_string(),
                    cpu_usage: 2.0,
                },
            ],
        };
        oscillator_manager.handle(&message).unwrap();
        assert!(oscillator_manager.stream_handle.is_some());

        // Handle a stop message.
        oscillator_manager.handle(&TopdioMessage::Stop).unwrap();
        assert!(oscillator_manager.stream_handle.is_none());

        // play and stop are idempotent.
        oscillator_manager.play().unwrap();
        oscillator_manager.play().unwrap();
        assert!(oscillator_manager.stream_handle.is_some());
        oscillator_manager.stop().unwrap();
        oscillator_manager.stop().unwrap();
        assert!(oscillator_manager.stream_handle.is_none());
    }
}
