//! Decodes the Opus packets of an Ogg Opus file with a Strata Opus backend
//! and writes raw interleaved `f32` PCM (48 kHz) for comparison with a
//! reference decoder.
//!
//! `cargo run -p strata-hle --example opus_probe -- <in.opus> <native|mf|libopus> <out.f32>`

use strata_formats::opus::OpusDecoder;

fn ogg_packets(data: &[u8]) -> (u8, Vec<Vec<u8>>) {
    let mut packets = Vec::new();
    let mut cur = Vec::new();
    let mut at = 0;
    while at + 27 <= data.len() && &data[at..at + 4] == b"OggS" {
        let segs = data[at + 26] as usize;
        let table = &data[at + 27..at + 27 + segs];
        let mut body = at + 27 + segs;
        for &lace in table {
            cur.extend_from_slice(&data[body..body + lace as usize]);
            body += lace as usize;
            if lace < 255 {
                packets.push(std::mem::take(&mut cur));
            }
        }
        at = body;
    }
    // packets[0] = OpusHead (channel count at byte 9), [1] = OpusTags.
    let channels = packets.first().map_or(1, |h| h[9]);
    (channels, packets.into_iter().skip(2).collect())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).expect("read");
    let (channels, packets) = ogg_packets(&data);
    let name = args[2].as_str();
    let backend = if name == "native" {
        strata_formats::opus_native::backend(usize::from(channels), 48_000)
    } else {
        #[cfg(windows)]
        {
            strata_hle::opus_host::backend(name, usize::from(channels), 48_000)
        }
        #[cfg(not(windows))]
        {
            None
        }
    }
    .unwrap_or_else(|| panic!("backend {name} unavailable"));
    let name = backend.name();
    let mut decoder = OpusDecoder::with_backend(u16::from(channels), 48_000, backend);
    let mut pcm = Vec::new();
    let mut malformed = 0;
    for packet in &packets {
        match decoder.decode_packet(packet) {
            Some(s) => pcm.extend(s),
            None => malformed += 1,
        }
    }
    let failed = malformed + decoder.packets_failed;
    println!(
        "{name}: {} packets ({failed} failed), {channels} ch, {} samples/ch",
        packets.len(),
        pcm.len() / channels as usize
    );
    let raw: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(&args[3], raw).unwrap();
}
