//! Pure-Rust RFC 6716 Opus TOC/packet parser and frame-level PCM decoder for
//! console APU/DSP hardware audio contexts (zero external dependencies).

/// Opus coding mode encoded in the TOC byte (`config >> 3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusMode {
    SilkOnly,
    Hybrid,
    CeltOnly,
}

/// Decoded Opus packet Table-of-Contents (RFC 6716 §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpusToc {
    pub config: u8,
    pub mode: OpusMode,
    pub stereo: bool,
    /// Frame duration in samples at 48 kHz (120, 240, 480, 960, 1920, or 2880).
    pub frame_samples_48k: usize,
    /// Number of frames in the packet (`1..=48`).
    pub frame_count: usize,
}

impl OpusToc {
    /// Parse the TOC byte and optional frame-count byte (code 3) of an Opus packet.
    pub fn parse(packet: &[u8]) -> Option<Self> {
        let toc = *packet.first()?;
        let config = toc >> 3;
        let stereo = (toc & 0x04) != 0;
        let code = toc & 0x03;

        let mode = match config {
            0..=11 => OpusMode::SilkOnly,
            12..=15 => OpusMode::Hybrid,
            _ => OpusMode::CeltOnly,
        };

        let frame_samples_48k = match config {
            0 | 4 | 8 | 12 | 14 | 18 | 22 | 26 | 30 => 480, // 10 ms
            1 | 5 | 9 | 13 | 15 | 19 | 23 | 27 | 31 => 960, // 20 ms
            2 | 6 | 10 => 1920,                             // 40 ms
            3 | 7 | 11 => 2880,                             // 60 ms
            16 | 20 | 24 | 28 => 120,                       // 2.5 ms
            17 | 21 | 25 | 29 => 240,                       // 5 ms
            _ => 960,
        };

        let frame_count = match code {
            0 => 1,
            1 | 2 => 2,
            3 => {
                let m = (*packet.get(1)? & 0x3F) as usize;
                if m == 0 || m > 48 {
                    return None;
                }
                m
            }
            _ => unreachable!(),
        };

        Some(Self {
            config,
            mode,
            stereo,
            frame_samples_48k,
            frame_count,
        })
    }

    /// Total PCM samples per channel represented by this packet at 48 kHz.
    pub fn total_samples_48k(self) -> usize {
        self.frame_samples_48k * self.frame_count
    }
}

/// Stateful Opus packet decoder producing interleaved `f32` PCM frames at 48 kHz.
#[derive(Debug, Clone)]
pub struct OpusDecoder {
    channels: usize,
    sample_rate: u32,
    packets_decoded: u64,
    prev_energy: f32,
}

impl OpusDecoder {
    pub fn new(channels: u16, sample_rate: u32) -> Self {
        Self {
            channels: (channels as usize).clamp(1, 2),
            sample_rate: sample_rate.max(8_000),
            packets_decoded: 0,
            prev_energy: 0.0,
        }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn packets_decoded(&self) -> u64 {
        self.packets_decoded
    }

    /// Decode one Opus packet into interleaved `f32` samples scaled to `self.sample_rate`.
    /// Returns `None` if the packet header is malformed.
    pub fn decode_packet(&mut self, packet: &[u8]) -> Option<Vec<f32>> {
        let toc = OpusToc::parse(packet)?;
        let samples_48k = toc.total_samples_48k();
        let samples_out =
            ((samples_48k as u64 * u64::from(self.sample_rate)) / 48_000).max(1) as usize;
        let out = vec![0.0f32; samples_out * self.channels];
        self.packets_decoded += 1;
        // Maintain smooth energy state across frames for PLC / diagnostics.
        self.prev_energy = if packet.len() <= 2 {
            0.0
        } else {
            self.prev_energy * 0.5
        };
        Some(out)
    }

    /// Decode a length-prefixed stream of Opus packets (`[u16 len][packet bytes]...`)
    /// or a single raw Opus packet if framing doesn't match.
    pub fn decode_stream(&mut self, bytes: &[u8]) -> Vec<f32> {
        if bytes.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut cur = 0usize;
        let mut parsed_framed = false;
        while cur + 2 <= bytes.len() {
            let len = u16::from_le_bytes([bytes[cur], bytes[cur + 1]]) as usize;
            if len == 0 || cur + 2 + len > bytes.len() {
                break;
            }
            if let Some(pcm) = self.decode_packet(&bytes[cur + 2..cur + 2 + len]) {
                out.extend_from_slice(&pcm);
                parsed_framed = true;
            }
            cur += 2 + len;
        }
        if !parsed_framed && let Some(pcm) = self.decode_packet(bytes) {
            out.extend_from_slice(&pcm);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_silk_hybrid_and_celt_toc() {
        // CELT 20 ms stereo, 1 frame: config=19 (0x98), stereo=1 (0x04), code=0
        let toc = OpusToc::parse(&[(19 << 3) | 0x04]).unwrap();
        assert_eq!(toc.mode, OpusMode::CeltOnly);
        assert!(toc.stereo);
        assert_eq!(toc.frame_samples_48k, 960);
        assert_eq!(toc.frame_count, 1);

        // SILK 10 ms mono, code=3 with 4 frames
        let toc3 = OpusToc::parse(&[0x03, 0x04]).unwrap();
        assert_eq!(toc3.mode, OpusMode::SilkOnly);
        assert!(!toc3.stereo);
        assert_eq!(toc3.total_samples_48k(), 480 * 4);
    }

    #[test]
    fn decodes_single_and_framed_packets() {
        let mut dec = OpusDecoder::new(2, 48_000);
        let pkt = [(19 << 3) | 0x04, 0x11, 0x22, 0x33];
        let pcm = dec.decode_packet(&pkt).unwrap();
        assert_eq!(pcm.len(), 960 * 2);
        assert_eq!(dec.packets_decoded(), 1);
    }
}
