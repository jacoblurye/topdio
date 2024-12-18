use anyhow::{bail, Result};
use clap::Parser;

use crate::{
    audio::{Quantize, Synth, Wave, A0},
    topdio::Topdio,
    ui::UI,
};

#[derive(Parser)]
#[clap(version)]
pub struct TopdioArgs {
    #[clap(
        short = 's',
        long,
        help = "Scaling factor for oscillator frequency (freq = freq_scale * cpu_usage + freq_floor)",
        default_value_t = 10.
    )]
    pub freq_scale: f32,
    #[clap(
        short = 'f',
        long,
        help = "Lowest allowed frequency for an oscillator to have (freq = freq_scale * cpu_usage + freq_floor)",
        default_value_t = A0,
    )]
    pub freq_floor: f32,
    #[clap(
        short,
        long,
        help = "Ratio relative to the refresh rate controlling time it takes oscillators to move between frequencies",
        default_value_t = 1.
    )]
    pub glide: f32,
    #[clap(
        short,
        long,
        help = "Number of oscillators to run simultaneously",
        default_value_t = 10
    )]
    pub num_oscillators: u8,
    #[clap(
        short,
        long,
        help = "How often to refresh system stats in seconds",
        default_value_t = 1.
    )]
    pub refresh_rate: f32,
    #[clap(short, long, value_enum, help = "Synth wave kind", default_value_t = Wave::Triangle)]
    pub wave: Wave,
    #[clap(
        short,
        long,
        value_enum,
        help = "How to quantize oscillator frequencies",
        default_value_t = Quantize::None
    )]
    pub quantize: Quantize,
    #[clap(
        short,
        long,
        help = "Buffer size for the audio stream (if greater than max supported buffer size, it will be silently capped).",
        default_value_t = 1024
    )]
    pub buffer_size: u32,
}

pub fn cli() -> Result<()> {
    let args = TopdioArgs::parse();
    if args.freq_scale <= 0.0 {
        bail!("freq_scale must be greater than 0")
    }
    if args.freq_floor <= 0.0 {
        bail!("freq_floor must be greater than 0")
    }
    if args.glide <= 0.0 {
        bail!("glide must be greater than 0")
    }
    if args.num_oscillators == 0 {
        bail!("num_oscillators must be greater than 0")
    }
    if args.refresh_rate <= 0.0 {
        bail!("refresh_rate must be greater than 0")
    }
    if args.buffer_size == 0 {
        bail!("buffer_size must be greater than 0")
    }

    let ui = UI::new()?;
    let synth = Synth::from_args(&args)?;

    let mut topdio = Topdio::new(vec![Box::new(ui), Box::new(synth)], args.refresh_rate);
    topdio.run()
}
