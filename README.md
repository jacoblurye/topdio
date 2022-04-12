# topdio

![crates.io](https://img.shields.io/crates/v/topdio.svg)

Like [top](https://man7.org/linux/man-pages/man1/top.1.html), but with audio. Listen to your computer's resource usage statistics.

## Installation

[Install Rust](https://www.rust-lang.org/tools/install) if you haven't already, then run:

```bash
cargo install topdio
```

## Usage

```bash
topdio
# or, with non-default arguments:
topdio --num-oscillators 30 --wave-type sine --freq-scale 50 --refresh-rate 500
```
