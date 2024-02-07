use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::{ArgEnum, Parser};
use cpal::{
    traits::{DeviceTrait, HostTrait},
    Device, FromSample, Host, Sample, StreamConfig,
};
use topdio::{
    oscillator::{sine, triangle, OscillatorManager, Wave},
    topdio::{CrosstermQuitter, Topdio},
    ui::UI,
};

#[derive(Parser)]
#[clap(version)]
struct Args {
    #[clap(
        short,
        long,
        help = "Scaling factor for oscillator frequency (freq = freq_scale * cpu_usage)",
        default_value_t = 20.
    )]
    freq_scale: f32,
    #[clap(
        short,
        long,
        help = "Number of oscillators to run simultaneously",
        default_value_t = 10
    )]
    num_oscillators: usize,
    #[clap(short, long, arg_enum, help = "Synth wave type", default_value_t = WaveType::Triangle)]
    wave_type: WaveType,
    #[clap(
        short,
        long,
        help = "How often to refresh system stats in milliseconds",
        default_value_t = 1000
    )]
    refresh_rate: u64,
}

#[derive(Clone, ArgEnum)]
pub enum WaveType {
    Triangle,
    Sine,
}

impl WaveType {
    pub fn wave<S>(&self) -> Wave<S>
    where
        S: Sample + FromSample<f32>,
    {
        match self {
            WaveType::Triangle => triangle::<S>,
            WaveType::Sine => sine::<S>,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.freq_scale <= 0. {
        bail!("--freq-scale must be greater than 0.")
    }

    let host: Host = cpal::default_host();
    let output: Device = host
        .default_output_device()
        .context("no default output device available")?;
    let config: StreamConfig = output
        .default_output_config()
        .context("no default output config")?
        .into();

    let oscillator_manager = OscillatorManager::<f32>::new(
        args.wave_type.wave(),
        args.num_oscillators,
        args.freq_scale,
        Arc::new(config),
        Arc::new(output),
    );

    let ui = UI::new().unwrap();

    let mut topdio = Topdio::new(
        vec![Box::new(oscillator_manager), Box::new(ui)],
        args.refresh_rate,
    );
    topdio
        .run(&CrosstermQuitter::new()?)
        .context("topdio failed")
}
