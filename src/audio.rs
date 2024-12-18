use rand::Rng;
use std::{
    fmt::Debug,
    sync::mpsc::{channel, Receiver, Sender},
};

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize, Device, FromSample, Host, Sample, SampleFormat, SizedSample, Stream, StreamConfig,
    SupportedBufferSize, SupportedStreamConfig,
};

use crate::{
    cli::TopdioArgs,
    topdio::{TopdioMessage, TopdioSubscriber},
};

const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct WaveState {
    freq: f32,
    gain: f32,
    phase: f32,
}

fn summed_sines(state: &WaveState, harmonics: &[u8]) -> f32 {
    let summed_harmonics = harmonics
        .iter()
        .map(|h| (TWO_PI * state.phase / *h as f32).sin())
        .sum::<f32>();
    state.gain * summed_harmonics / harmonics.len() as f32
}

fn sine<S: SizedSample + FromSample<f32>>(state: &WaveState) -> S {
    let raw = summed_sines(state, &[1]);
    raw.to_sample::<S>()
}

fn triangle<S: SizedSample + FromSample<f32>>(state: &WaveState) -> S {
    let raw = 4.0 * state.gain * (state.phase - (state.phase + 0.5).floor()).abs() - state.gain;
    raw.to_sample::<S>()
}

fn saw<S: SizedSample + FromSample<f32>>(state: &WaveState) -> S {
    let raw = state.gain * (state.phase - (state.phase + 0.5).floor());
    raw.to_sample::<S>()
}

fn square<S: SizedSample + FromSample<f32>>(state: &WaveState) -> S {
    // Only use the first two harmonics - this generates a fuller-sounding square wave approximation.
    let raw = summed_sines(state, &[1, 3, 5]);
    raw.to_sample::<S>()
}

#[derive(Clone, ValueEnum)]
pub enum Wave {
    Sine,
    Triangle,
    Saw,
    Square,
}

impl Wave {
    fn call<S: SizedSample + FromSample<f32>>(&self, state: &WaveState) -> S {
        let wave_fn = match self {
            Wave::Sine => sine,
            Wave::Triangle => triangle,
            Wave::Saw => saw,
            Wave::Square => square,
        };
        wave_fn(state)
    }
}

/// Represents a piecewise function that decreases or increases linearly
/// at a given rate to a given target value.
#[derive(Clone)]
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

pub const A0: f32 = 27.5;

fn quantize_overtone(freq: f32) -> f32 {
    let root = A0;
    let mut harmonic = 1.0;
    while root * harmonic < freq {
        harmonic += 1.0;
    }
    root * (harmonic - 1.0)
}

const PENTATONIC_RATIOS: [f32; 5] = [1., 9. / 8., 5. / 4., 3. / 2., 5. / 3.];

fn quantize_pentatonic(freq: f32) -> f32 {
    let root = A0;
    let mut octave = 1.0;
    let mut step = 0;
    while root * octave * PENTATONIC_RATIOS[step] < freq {
        if step == PENTATONIC_RATIOS.len() - 1 {
            step = 0;
            octave *= 2.0;
        } else {
            step += 1;
        }
    }
    root * octave * PENTATONIC_RATIOS[step]
}

#[derive(Clone, ValueEnum)]
pub enum Quantize {
    Overtone,
    Pentatonic,
    None,
}

impl Quantize {
    fn call(&self, sample: f32) -> f32 {
        let quantize_fn = match self {
            Quantize::Overtone => quantize_overtone,
            Quantize::Pentatonic => quantize_pentatonic,
            Quantize::None => |f: f32| f,
        };
        quantize_fn(sample)
    }
}

#[derive(Clone)]
struct OscillatorConfig {
    wave: Wave,
    quantize: Quantize,
    freq_scale: f32,
    freq_floor: f32,
    glide_secs: f32,
}

#[derive(Clone)]
struct Oscillator {
    state: WaveState,
    config: OscillatorConfig,
    gain_ramp: Ramp,
    freq_ramp: Ramp,
    sample_rate: u32,
    pan: f32,
}

impl Oscillator {
    fn new(
        initial_freq: f32,
        initial_gain: f32,
        config: OscillatorConfig,
        sample_rate: u32,
        pan: f32,
    ) -> Oscillator {
        let initial_freq = initial_freq.max(0.0);
        let initial_gain = initial_gain.clamp(0.0, 1.0);
        Oscillator {
            state: WaveState {
                freq: initial_freq,
                gain: 0.,
                phase: 0.,
            },
            config,
            gain_ramp: Ramp::new(initial_gain, initial_gain / sample_rate as f32),
            freq_ramp: Ramp::new(initial_freq, 0.0),
            sample_rate,
            pan,
        }
    }

    pub fn next_sample<S: SizedSample + FromSample<f32>>(&mut self) -> (S, S) {
        let state = &mut self.state;
        state.phase = (state.phase + state.freq / self.sample_rate as f32) % 1.0;
        state.freq = self.freq_ramp.next(state.freq);
        state.gain = self.gain_ramp.next(state.gain);

        let sample = self.config.wave.call::<S>(state);
        let left = sample.mul_amp(Sample::from_sample(1.0 - self.pan));
        let right = sample.mul_amp(Sample::from_sample(1.0 + self.pan));

        (left, right)
    }

    pub fn set_freq_from_stat(&mut self, usage_stat: f32) {
        let raw_freq = usage_stat * self.config.freq_scale + self.config.freq_floor;
        let freq = self.config.quantize.call(raw_freq);
        let rate =
            ((freq - self.state.freq) / (self.sample_rate as f32 * self.config.glide_secs)).abs();
        self.freq_ramp = Ramp::new(freq, rate);
    }
}

struct StreamHandle {
    stats_tx: Sender<Vec<f32>>,
    stream: Stream,
}

/// An [`Synth`] controls a group of oscillators and their associated
/// audio output stream. It can act as a [`TopdioSubscriber`].
pub struct Synth {
    oscillators: Vec<Oscillator>,
    output: Device,
    stream_config: StreamConfig,
    sample_format: SampleFormat,
    stream_handle: Option<StreamHandle>,
}

impl Synth {
    pub fn from_args(args: &TopdioArgs) -> Result<Self> {
        Self::new(
            args.num_oscillators as usize,
            OscillatorConfig {
                wave: args.wave.clone(),
                quantize: args.quantize.clone(),
                freq_scale: args.freq_scale,
                freq_floor: args.freq_floor,
                glide_secs: args.glide * args.refresh_rate,
            },
            args.buffer_size,
        )
    }

    /// Create a new [`Synth`] with the given number of oscillators, all with the given
    /// config. Calling this function does not create an audio output stream.
    fn new(
        num_oscillators: usize,
        oscillator_config: OscillatorConfig,
        buffer_size: u32,
    ) -> Result<Synth> {
        let host: Host = cpal::default_host();
        let output: Device = host
            .default_output_device()
            .context("no default output device available")?;
        let supported_stream_config: SupportedStreamConfig = output
            .default_output_config()
            .context("no default output config")?;

        let stream_config = {
            let mut stream_config = supported_stream_config.config();
            stream_config.buffer_size = match supported_stream_config.buffer_size() {
                SupportedBufferSize::Range { max, .. } => BufferSize::Fixed(*max.min(&buffer_size)),
                _ => BufferSize::Fixed(buffer_size),
            };
            stream_config.channels = stream_config.channels.min(2);
            stream_config
        };

        let mut rng = rand::thread_rng();
        let normalized_gain = 1. / (num_oscillators as f32);
        let pan_wideness_factor = 1.0;
        let oscillators = (0..num_oscillators)
            .map(|i| {
                // Randomly initialize the oscillator frequency between 0 and 50hz to avoid weird
                // constructive interference when the stream starts.
                let freq = rng.gen::<f32>() * 50.0;

                // Set up pans, starting from the middle and flip-flopping outwards.
                let pan = if i % 2 == 0 {
                    0.0 - ((i as f32 + pan_wideness_factor)
                        / (num_oscillators as f32 + pan_wideness_factor))
                } else {
                    i as f32 / num_oscillators as f32
                };

                Oscillator::new(
                    freq,
                    normalized_gain,
                    oscillator_config.clone(),
                    stream_config.sample_rate.0,
                    pan,
                )
            })
            .collect();

        Ok(Synth {
            oscillators,
            output,
            stream_config,
            sample_format: supported_stream_config.sample_format(),
            stream_handle: None,
        })
    }

    /// Start the audio output stream. This function is idempotent.
    fn play(&mut self) -> Result<()> {
        if self.stream_handle.is_some() {
            return Ok(());
        }

        // Create the audio output stream. The stream callback receives stats data from the
        // channel and updates the oscillators' target frequencies accordingly.
        let (stats_tx, stats_rx) = channel::<Vec<f32>>();
        let stream = match self.sample_format {
            SampleFormat::F32 => self.build_output_stream::<f32>(stats_rx)?,
            SampleFormat::F64 => self.build_output_stream::<f64>(stats_rx)?,
            SampleFormat::I8 => self.build_output_stream::<i8>(stats_rx)?,
            SampleFormat::I16 => self.build_output_stream::<i16>(stats_rx)?,
            SampleFormat::I32 => self.build_output_stream::<i32>(stats_rx)?,
            SampleFormat::I64 => self.build_output_stream::<i64>(stats_rx)?,
            SampleFormat::U8 => self.build_output_stream::<u8>(stats_rx)?,
            SampleFormat::U16 => self.build_output_stream::<u16>(stats_rx)?,
            SampleFormat::U32 => self.build_output_stream::<u32>(stats_rx)?,
            SampleFormat::U64 => self.build_output_stream::<u64>(stats_rx)?,
            _ => bail!("unsupported sample format: {:?}", self.sample_format),
        };
        stream.play()?;

        self.stream_handle = Some(StreamHandle { stats_tx, stream });

        Ok(())
    }

    fn build_output_stream<S: SizedSample + FromSample<f32> + 'static>(
        &mut self,
        stats_rx: Receiver<Vec<f32>>,
    ) -> Result<Stream> {
        let mut oscillators = self.oscillators.clone();
        let channels = self.stream_config.channels as usize;
        let stream = self.output.build_output_stream(
            &self.stream_config,
            move |buf: &mut [S], _| {
                for frame in buf.chunks_mut(channels) {
                    // See if new stats are available, and update the oscillators' target
                    // frequencies if so.
                    if let Ok(stats) = stats_rx.try_recv() {
                        stats.iter().take(oscillators.len()).enumerate().for_each(
                            |(index, stat)| {
                                oscillators[index].set_freq_from_stat(*stat);
                            },
                        );
                    }

                    // Compute the next sample for each oscillator and sum them.
                    let (left, right): (S, S) = oscillators
                        .iter_mut()
                        .map(|oscillator| oscillator.next_sample::<S>())
                        .fold(
                            (Sample::from_sample(0.0), Sample::from_sample(0.0)),
                            |(l_acc, r_acc), (l_osc, r_osc)| {
                                (
                                    l_acc.add_amp(l_osc.to_signed_sample()),
                                    r_acc.add_amp(r_osc.to_signed_sample()),
                                )
                            },
                        );

                    // Only stereo and mono output are currently supported. If mono,
                    // the left and right channels are averaged.
                    if channels == 2 {
                        frame[0] = left;
                        frame[1] = right;
                    } else {
                        frame[0] = left
                            .add_amp(right.to_signed_sample())
                            .mul_amp(Sample::from_sample(0.5));
                    }
                }
            },
            |err| panic!("{:?}", err),
            None,
        )?;
        Ok(stream)
    }
}

impl TopdioSubscriber for Synth {
    fn handle(&mut self, message: &TopdioMessage) -> Result<()> {
        // Make sure the stream is playing – this is a no-op if it's already playing.
        self.play().context("synth failed to play")?;

        match self.stream_handle {
            Some(ref mut stream_handle) => match message {
                TopdioMessage::Stats { processes } => {
                    let stats: Vec<f32> = processes.iter().map(|p| p.cpu_usage).collect();
                    stream_handle
                        .stats_tx
                        .send(stats)
                        .context("synth failed to send stats")?;
                }
                TopdioMessage::Stop => {
                    // TODO: ideally, we should fade the oscillators out gracefully.
                    stream_handle.stream.pause()?;
                }
            },
            None => bail!("synth stream handle was not created"),
        };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
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
}
