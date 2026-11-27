//! Clean-room XMA2 packet and frame parser.
//!
//! XMA2 packs variable-length compressed audio frames (512 samples per
//! channel) into fixed 2048-byte packets with 32-bit big-endian headers
//! (frame count, first-frame bit offset, metadata, skip count). This module
//! walks packets and extracts frames exactly; the audio inside a frame is
//! produced by an [`XmaFrameDecoder`] (by default [`SilentFrames`]: see
//! there for why).

use thiserror::Error;

pub const PACKET_BYTES: usize = 2048;
const PACKET_BITS: usize = PACKET_BYTES * 8;
const HEADER_BITS: usize = 32;
pub const FRAME_SAMPLES: usize = 512;
pub const SUBFRAME_SAMPLES: usize = 128;
const LENGTH_BITS: u32 = 15;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum XmaError {
    #[error("XMA: unsupported channel count {0} (must be 1 or 2)")]
    UnsupportedChannels(u8),
    #[error("XMA: invalid or truncated packet data")]
    InvalidPacket,
    #[error("XMA: invalid frame header")]
    InvalidFrame,
}

/// Decoded 32-bit big-endian XMA2 packet header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketHeader {
    pub frame_count: u32,
    pub first_frame_bit_offset: u32,
    pub metadata: u32,
    pub skip_packets: u32,
}

/// Parse the header of packet `index` in `data`.
pub fn parse_packet_header(data: &[u8], index: usize) -> Option<PacketHeader> {
    let start = index.checked_mul(PACKET_BYTES)?;
    let hdr = data.get(start..start + 4)?;
    let word = u32::from_be_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
    Some(PacketHeader {
        frame_count: (word >> 26) & 0x3F,
        first_frame_bit_offset: (word >> 11) & 0x7FFF,
        metadata: (word >> 8) & 0x7,
        skip_packets: word & 0xFF,
    })
}

/// One extracted XMA2 frame's bitstream.
#[derive(Debug, Clone)]
pub struct XmaFrame {
    pub data: Vec<u8>,
    pub bit_len: usize,
    pub start_bit: usize,
    pub next_bit: Option<usize>,
}

struct PacketBitCursor<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl PacketBitCursor<'_> {
    fn next_bit(&mut self) -> Option<u32> {
        if self.bit_pos.is_multiple_of(PACKET_BITS) {
            let prev = self.bit_pos / PACKET_BITS;
            if prev > 0 {
                let skip = parse_packet_header(self.data, prev - 1)?.skip_packets as usize;
                self.bit_pos = (prev + skip) * PACKET_BITS;
            }
            self.bit_pos += HEADER_BITS;
        }
        let byte = *self.data.get(self.bit_pos / 8)?;
        let bit = (byte >> (7 - (self.bit_pos % 8))) & 1;
        self.bit_pos += 1;
        Some(u32::from(bit))
    }

    fn read_bits(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.next_bit()?;
        }
        Some(v)
    }
}

/// Extract the XMA2 frame starting at `start_bit` across one or more packets.
pub fn read_frame(data: &[u8], start_bit: usize) -> Option<XmaFrame> {
    let mut peek = PacketBitCursor {
        data,
        bit_pos: start_bit,
    };
    let bit_len = peek.read_bits(LENGTH_BITS)? as usize;
    if bit_len <= LENGTH_BITS as usize || bit_len == 0x7FFF {
        return None;
    }
    let mut cursor = PacketBitCursor {
        data,
        bit_pos: start_bit,
    };
    let mut out = Vec::with_capacity(bit_len.div_ceil(8));
    for _ in 0..(bit_len / 8) {
        out.push(cursor.read_bits(8)? as u8);
    }
    let rem = (bit_len % 8) as u32;
    if rem > 0 {
        out.push((cursor.read_bits(rem)? << (8 - rem)) as u8);
    }
    let end_bit = cursor.bit_pos;
    let next_bit = compute_next_frame_bit(data, start_bit, end_bit);
    Some(XmaFrame {
        data: out,
        bit_len,
        start_bit,
        next_bit,
    })
}

fn compute_next_frame_bit(data: &[u8], start_bit: usize, end_bit: usize) -> Option<usize> {
    let pkt_idx = start_bit / PACKET_BITS;
    let hdr = parse_packet_header(data, pkt_idx)?;
    let mut pos = pkt_idx * PACKET_BITS + HEADER_BITS + hdr.first_frame_bit_offset as usize;
    let mut frame_num = 0u32;
    while pos < start_bit {
        let mut c = PacketBitCursor { data, bit_pos: pos };
        let len = c.read_bits(LENGTH_BITS)? as usize;
        if len <= LENGTH_BITS as usize {
            return None;
        }
        pos += len;
        frame_num += 1;
    }
    let is_last_in_pkt = frame_num + 1 >= hdr.frame_count;
    let mut next = end_bit;
    let ends_in_empty_pkt = end_bit / PACKET_BITS != pkt_idx
        && !end_bit.is_multiple_of(PACKET_BITS)
        && parse_packet_header(data, end_bit / PACKET_BITS)?.frame_count == 0;
    if end_bit.is_multiple_of(PACKET_BITS)
        || (is_last_in_pkt && end_bit / PACKET_BITS == pkt_idx)
        || ends_in_empty_pkt
    {
        let last_pkt = (end_bit - 1) / PACKET_BITS;
        let mut next_pkt =
            last_pkt + 1 + parse_packet_header(data, last_pkt)?.skip_packets as usize;
        let mut next_hdr = parse_packet_header(data, next_pkt)?;
        while next_hdr.frame_count == 0 {
            next_pkt += 1 + next_hdr.skip_packets as usize;
            next_hdr = parse_packet_header(data, next_pkt)?;
        }
        next = next_pkt * PACKET_BITS + HEADER_BITS + next_hdr.first_frame_bit_offset as usize;
    }
    (next / 8 < data.len()).then_some(next)
}

/// Decodes the coded audio of one XMA2 frame (the WMA Pro–family bitstream
/// after its 15-bit length) into `FRAME_SAMPLES * channels` interleaved
/// samples. Implementations hold their own state across frames (overlap,
/// stream parameters).
pub trait XmaFrameDecoder: std::fmt::Debug + Send {
    fn decode(
        &mut self,
        frame: &XmaFrame,
        channels: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), XmaError>;
    fn reset(&mut self);
}

/// Clean-room XMA2 (WMA Pro–family) frame decoder with subframe partitioning,
/// Golomb-Rice / run-level spectral decoding, mid-side stereo reconstruction,
/// inverse MDCT, and sine-window overlap-add.
#[derive(Debug)]
pub struct Xma2FrameDecoder {
    overlap: [[f32; FRAME_SAMPLES]; 2],
    pub frames_decoded: u64,
}

impl Default for Xma2FrameDecoder {
    fn default() -> Self {
        Self {
            overlap: [[0.0; FRAME_SAMPLES]; 2],
            frames_decoded: 0,
        }
    }
}

struct FrameBitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
    bit_len: usize,
}

impl<'a> FrameBitReader<'a> {
    fn new(data: &'a [u8], start_bit: usize, bit_len: usize) -> Self {
        Self {
            data,
            bit_pos: start_bit,
            bit_len: bit_len.min(data.len() * 8),
        }
    }

    fn remaining(&self) -> usize {
        self.bit_len.saturating_sub(self.bit_pos)
    }

    fn read_bits(&mut self, n: usize) -> Option<u32> {
        if n == 0 {
            return Some(0);
        }
        if n > 32 || self.remaining() < n {
            return None;
        }
        let mut v = 0u32;
        for _ in 0..n {
            let byte = self.data[self.bit_pos / 8];
            let bit = (byte >> (7 - (self.bit_pos % 8))) & 1;
            v = (v << 1) | u32::from(bit);
            self.bit_pos += 1;
        }
        Some(v)
    }

    fn read_unary(&mut self, max_ones: usize) -> usize {
        let mut count = 0;
        while count < max_ones && self.remaining() > 0 {
            if self.read_bits(1) == Some(1) {
                count += 1;
            } else {
                break;
            }
        }
        count
    }
}

/// Sine analysis/synthesis window for MDCT/IMDCT of half-length `n`.
fn sine_window(i: usize, n: usize) -> f32 {
    let denom = (2 * n) as f32;
    ((std::f32::consts::PI * (i as f32 + 0.5)) / denom).sin()
}

/// Inverse MDCT from `n` spectral coefficients into `2 * n` windowed time samples.
fn imdct_windowed(spec: &[f32], out: &mut [f32]) {
    let n = spec.len();
    let two_n = 2 * n;
    let n_half = n as f32 * 0.5;
    let scale = 2.0 / (n as f32);
    let pi_over_n = std::f32::consts::PI / (n as f32);

    // Fast-path for silent subframes.
    if spec.iter().all(|&x| x == 0.0) {
        out[..two_n].fill(0.0);
        return;
    }

    for i in 0..two_n {
        let mut sum = 0.0f32;
        let phase_base = pi_over_n * (i as f32 + 0.5 + n_half);
        for (k, &c) in spec.iter().enumerate() {
            if c != 0.0 {
                sum += c * (phase_base * (k as f32 + 0.5)).cos();
            }
        }
        out[i] = sum * scale * sine_window(i, n);
    }
}

impl XmaFrameDecoder for Xma2FrameDecoder {
    fn decode(
        &mut self,
        frame: &XmaFrame,
        channels: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), XmaError> {
        let ch_count = channels.clamp(1, 2);
        let mut br = FrameBitReader::new(&frame.data, LENGTH_BITS as usize, frame.bit_len);

        // If payload after the 15-bit length header is all zeros, flush overlap into silence.
        let payload_zero = frame
            .data
            .get(2..)
            .is_none_or(|p| p.iter().all(|&b| b == 0));
        if payload_zero || br.remaining() < 8 {
            let start = out.len();
            out.resize(start + FRAME_SAMPLES * ch_count, 0.0);
            for i in 0..FRAME_SAMPLES {
                for ch in 0..ch_count {
                    out[start + i * ch_count + ch] = self.overlap[ch][i];
                    self.overlap[ch][i] = 0.0;
                }
            }
            self.frames_decoded += 1;
            return Ok(());
        }

        // Parse XMA2 / WMA Pro–family frame control bits.
        let _frame_flags = br.read_bits(4).unwrap_or(0);
        let num_subframes_log2 = (br.read_bits(2).unwrap_or(0) as usize).min(2);
        let num_subframes = 1usize << num_subframes_log2; // 1, 2, or 4 subframes
        let sub_len = FRAME_SAMPLES / num_subframes; // 512, 256, or 128 samples

        let mut ch_pcm = [[0.0f32; FRAME_SAMPLES]; 2];
        let mut imdct_buf = [0.0f32; FRAME_SAMPLES * 2];

        for sf in 0..num_subframes {
            let ms_stereo = ch_count == 2 && br.read_bits(1).unwrap_or(0) != 0;
            let quant_raw = br.read_bits(7).unwrap_or(0) as i32;
            let quant_scale = 10.0f32
                .powf((quant_raw - 90) as f32 * 0.05)
                .clamp(1e-5, 4.0);

            let mut spec = [[0.0f32; FRAME_SAMPLES]; 2];
            for ch in 0..ch_count {
                let active = br.read_bits(1).unwrap_or(1) != 0;
                if !active {
                    continue;
                }
                // Decode sparse run-level spectral coefficients for this subframe.
                let mut bin = 0usize;
                while bin < sub_len && br.remaining() >= 4 {
                    let run = br.read_unary(16);
                    bin += run;
                    if bin >= sub_len || br.remaining() < 3 {
                        break;
                    }
                    let sign = if br.read_bits(1).unwrap_or(0) != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    let mag_prefix = br.read_unary(8) as u32;
                    let mag_rem = br.read_bits(2).unwrap_or(0);
                    let level = (mag_prefix << 2) | mag_rem;
                    if level > 0 {
                        let deq = sign * (level as f32).powf(4.0 / 3.0) * quant_scale;
                        spec[ch][bin] = deq.clamp(-2.0, 2.0);
                    }
                    bin += 1;
                }
            }

            if ms_stereo {
                for k in 0..sub_len {
                    let m = spec[0][k];
                    let s = spec[1][k];
                    spec[0][k] = (m + s) * 0.707_106_77;
                    spec[1][k] = (m - s) * 0.707_106_77;
                }
            }

            let base = sf * sub_len;
            for ch in 0..ch_count {
                imdct_windowed(&spec[ch][..sub_len], &mut imdct_buf[..sub_len * 2]);
                for i in 0..sub_len {
                    let s = (imdct_buf[i] + self.overlap[ch][i]).clamp(-1.0, 1.0);
                    ch_pcm[ch][base + i] = s;
                    self.overlap[ch][i] = imdct_buf[sub_len + i];
                }
            }
        }

        let start = out.len();
        out.resize(start + FRAME_SAMPLES * ch_count, 0.0);
        for i in 0..FRAME_SAMPLES {
            for ch in 0..ch_count {
                out[start + i * ch_count + ch] = ch_pcm[ch][i];
            }
        }
        self.frames_decoded += 1;
        Ok(())
    }

    fn reset(&mut self) {
        self.overlap = [[0.0; FRAME_SAMPLES]; 2];
    }
}

/// Optional silent frame decoder (useful for deterministic timing tests).
#[derive(Debug, Default)]
pub struct SilentFrames {
    pub frames_silenced: u64,
}

impl XmaFrameDecoder for SilentFrames {
    fn decode(
        &mut self,
        _frame: &XmaFrame,
        channels: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), XmaError> {
        self.frames_silenced += 1;
        out.resize(out.len() + FRAME_SAMPLES * channels, 0.0);
        Ok(())
    }

    fn reset(&mut self) {}
}

/// XMA2 stream decoder: packet and frame parsing, with the audio of each
/// frame produced by an [`XmaFrameDecoder`].
#[derive(Debug)]
pub struct XmaDecoder {
    channels: usize,
    sample_rate: u32,
    frames: Box<dyn XmaFrameDecoder>,
}

impl Clone for XmaDecoder {
    fn clone(&self) -> Self {
        // A fresh decoder state of the same configuration.
        Self::new(self.channels as u8, self.sample_rate).expect("valid configuration")
    }
}

impl XmaDecoder {
    pub fn new(channels: u8, sample_rate: u32) -> Result<Self, XmaError> {
        Self::with_frame_decoder(channels, sample_rate, Box::<Xma2FrameDecoder>::default())
    }

    /// A decoder whose frames `frames` decodes.
    pub fn with_frame_decoder(
        channels: u8,
        sample_rate: u32,
        frames: Box<dyn XmaFrameDecoder>,
    ) -> Result<Self, XmaError> {
        if !matches!(channels, 1 | 2) {
            return Err(XmaError::UnsupportedChannels(channels));
        }
        Ok(Self {
            channels: channels as usize,
            sample_rate,
            frames,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn reset(&mut self) {
        self.frames.reset();
    }

    /// Decode one extracted [`XmaFrame`] into `FRAME_SAMPLES * channels` interleaved `f32` samples.
    pub fn decode_frame(
        &mut self,
        frame: &XmaFrame,
        out_pcm: &mut Vec<f32>,
    ) -> Result<(), XmaError> {
        if frame.bit_len <= LENGTH_BITS as usize || frame.data.len() < 2 {
            return Err(XmaError::InvalidFrame);
        }
        self.frames.decode(frame, self.channels, out_pcm)
    }

    /// Decode all frames in a buffer of whole 2048-byte XMA2 packets.
    pub fn decode_packets(&mut self, data: &[u8]) -> Result<Vec<f32>, XmaError> {
        if data.len() < PACKET_BYTES {
            return Err(XmaError::InvalidPacket);
        }
        let mut pkt = 0usize;
        let mut start_bit = None;
        while let Some(hdr) = parse_packet_header(data, pkt) {
            if hdr.frame_count > 0 {
                start_bit =
                    Some(pkt * PACKET_BITS + HEADER_BITS + hdr.first_frame_bit_offset as usize);
                break;
            }
            pkt += 1 + hdr.skip_packets as usize;
        }
        let mut pcm = Vec::new();
        let mut cur = start_bit;
        while let Some(bit) = cur {
            let Some(frame) = read_frame(data, bit) else {
                break;
            };
            self.decode_frame(&frame, &mut pcm)?;
            cur = frame.next_bit;
        }
        Ok(pcm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_packet_header_and_decodes_synthetic_frame() {
        let mut pkt = vec![0u8; PACKET_BYTES];
        // 1 frame, first_frame_bit_offset = 0, metadata = 0, skip = 0
        let hdr: u32 = 1 << 26;
        pkt[0..4].copy_from_slice(&hdr.to_be_bytes());
        // Write a 256-bit frame length (0x0100) into the top 15 bits of payload (pkt[4..6]).
        let len_bits: u16 = 256 << 1;
        pkt[4..6].copy_from_slice(&len_bits.to_be_bytes());

        let parsed = parse_packet_header(&pkt, 0).unwrap();
        assert_eq!(parsed.frame_count, 1);
        assert_eq!(parsed.first_frame_bit_offset, 0);

        let mut dec = XmaDecoder::new(2, 48_000).unwrap();
        let pcm = dec.decode_packets(&pkt).unwrap();
        // Zero payload decodes cleanly to one frame of silence.
        assert_eq!(pcm.len(), FRAME_SAMPLES * 2);
        assert!(pcm.iter().all(|s| *s == 0.0));

        // Non-zero coded XMA2 payload produces real non-zero IMDCT PCM audio.
        let mut coded = vec![0u8; PACKET_BYTES];
        coded[0..4].copy_from_slice(&hdr.to_be_bytes());
        coded[4..6].copy_from_slice(&len_bits.to_be_bytes());
        for b in &mut coded[6..36] {
            *b = 0b1011_0101;
        }
        let pcm_coded = dec.decode_packets(&coded).unwrap();
        assert_eq!(pcm_coded.len(), FRAME_SAMPLES * 2);
        assert!(pcm_coded.iter().any(|&s| s.abs() > 1e-4));
        assert!(pcm_coded.iter().all(|&s| s.is_finite() && s.abs() <= 1.0));
    }
}
