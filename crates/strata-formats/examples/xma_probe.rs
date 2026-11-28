//! Decodes the XMA packets of a RIFF `.xma` file with Strata's WMA Pro port
//! and reports how many frames decode, plus level statistics; optionally
//! writes a 16-bit WAV for listening.
//!
//! `cargo run -p strata-formats --example xma_probe -- <file.xma> <channels> [out.wav|out.f32]`
//!
//! Validation: the public Xbox 360 samples at
//! https://samples.ffmpeg.org/game-formats/xma/ (big-endian RIFF headers),
//! re-wrapped with a standard XMA2WAVEFORMATEX header and decoded by FFmpeg
//! 7.1 (`ffmpeg -i x.wav -f f32le ref.f32`), match this decoder's `.f32`
//! output at 100-133 dB SNR (max error ~7e-5); Strata's output runs 576
//! samples later (the primed first frame plus FFmpeg's 64-sample start trim).

use strata_formats::wmapro::{SAMPLES_PER_FRAME, WmaProDecoder};
use strata_formats::xma::{PACKET_BYTES, parse_packet_header, read_frame};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let file = std::fs::read(&args[1]).expect("read");
    let channels: usize = args[2].parse().expect("channels");
    // RIFF: find the "data" chunk (chunk sizes little-endian).
    let mut at = 12;
    let mut data = &file[..0];
    let mut rate = 44100u32;
    while at + 8 <= file.len() {
        let id = &file[at..at + 4];
        let len = u32::from_le_bytes(file[at + 4..at + 8].try_into().unwrap()) as usize;
        let body = &file[at + 8..(at + 8 + len).min(file.len())];
        if id == b"fmt " && body.len() >= 16 {
            // Xbox 360 headers are big-endian: the rate is a BE u32 at +12.
            rate = u32::from_be_bytes(body[12..16].try_into().unwrap());
        }
        if id == b"data" {
            data = body;
            break;
        }
        at += 8 + len + (len & 1);
    }
    let packets = data.len() / PACKET_BYTES;
    let mut dec = WmaProDecoder::new(channels, rate).expect("decoder");
    let (mut ok, mut failed) = (0usize, 0usize);
    let mut pcm = Vec::new();
    // Every packet: walk its frames from the first frame offset.
    let mut pkt = 0;
    let mut start = None;
    while pkt < packets {
        let h = parse_packet_header(data, pkt).unwrap();
        if h.frame_count > 0 {
            start = Some(pkt * PACKET_BYTES * 8 + 32 + h.first_frame_bit_offset as usize);
            break;
        }
        pkt += 1;
    }
    let mut cur = start;
    let mut first_err = None;
    while let Some(bit) = cur {
        let Some(frame) = read_frame(data, bit) else {
            break;
        };
        match dec.decode_frame(&frame.data, frame.bit_len, &mut pcm) {
            Ok(_) => ok += 1,
            Err(e) => {
                failed += 1;
                first_err.get_or_insert((ok + failed, e));
            }
        }
        cur = frame.next_bit;
    }
    let peak = pcm.iter().fold(0f32, |m, s| m.max(s.abs()));
    let rms = (pcm
        .iter()
        .map(|s| f64::from(*s) * f64::from(*s))
        .sum::<f64>()
        / pcm.len().max(1) as f64)
        .sqrt();
    let clipped = pcm.iter().filter(|s| s.abs() > 1.0).count();
    println!(
        "{}: rate {rate}, {packets} packets, {} frames ok, {failed} failed (first: {first_err:?}), {} samples/ch, peak {peak:.4}, rms {rms:.4}, clipped {clipped}",
        args[1],
        ok,
        pcm.len() / channels
    );
    let _ = SAMPLES_PER_FRAME;
    if let Some(out) = args.get(3).filter(|o| o.ends_with(".f32")) {
        let raw: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        std::fs::write(out, raw).unwrap();
        return;
    }
    if let Some(out) = args.get(3) {
        let mut w = Vec::new();
        let n = (pcm.len() * 2) as u32;
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + n).to_le_bytes());
        w.extend_from_slice(b"WAVEfmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes());
        w.extend_from_slice(&(channels as u16).to_le_bytes());
        w.extend_from_slice(&rate.to_le_bytes());
        w.extend_from_slice(&(rate * 2 * channels as u32).to_le_bytes());
        w.extend_from_slice(&((2 * channels) as u16).to_le_bytes());
        w.extend_from_slice(&16u16.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&n.to_le_bytes());
        for s in &pcm {
            w.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
        }
        std::fs::write(out, w).unwrap();
    }
}
