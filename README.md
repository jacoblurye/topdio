# topdio

![crates.io](https://img.shields.io/crates/v/topdio.svg)

Like [top](https://man7.org/linux/man-pages/man1/top.1.html), but with audio. Listen to your computer's resource usage statistics, just because.

## Installation

[Install Rust](https://www.rust-lang.org/tools/install), then run:

```bash
cargo install topdio --locked
```

## Usage

> **!!!! Volume Warning !!!!**    
> topdio starts playing audio immediately. Make sure your volume is at a reasonable level before running.

Default usage:
```bash
topdio
```

For more musical results, try the `--quantize` flag. For example:
```bash
# Quantize to a pentatonic scale with a very fast refresh rate.
# (tip: try different waveforms with --wave)
topdio --quantize pentatonic --glide 0.2 --refresh-rate 0.1 --freq-floor 200 --num-oscillators 5

# Quantize to an overtone series with a very slow refresh rate.
topdio -q overtone -w sine -g 0.9 -r 10 -f 150
```

For smoother transitions without decreasing the refresh rate, try a `--glide` above 1.
```bash
topdio --glide 10
```

If you're hearing clicks and pops, try increasing the buffer size with `--buffer-size`.
```bash
topdio --buffer-size 4096
```
