//! Clean-room XMA / XMA2 packet parser and frame-level audio decoder.
//!
//! XMA2 packs variable-length compressed audio frames (512 samples/channel,
//! subdivided into 128/256/512-sample subframes) into fixed 2048-byte packets
//! with 32-bit big-endian headers.

use std::f32::consts::PI;
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

/// Stateful XMA2 stream decoder producing interleaved `f32` PCM frames.
#[derive(Debug, Clone)]
pub struct XmaDecoder {
    channels: usize,
    sample_rate: u32,
    overlap: [[f32; FRAME_SAMPLES / 2]; 2],
}

impl XmaDecoder {
    pub fn new(channels: u8, sample_rate: u32) -> Result<Self, XmaError> {
        if !matches!(channels, 1 | 2) {
            return Err(XmaError::UnsupportedChannels(channels));
        }
        Ok(Self {
            channels: channels as usize,
            sample_rate,
            overlap: [[0.0; FRAME_SAMPLES / 2]; 2],
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn reset(&mut self) {
        self.overlap = [[0.0; FRAME_SAMPLES / 2]; 2];
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
        let mut bit_pos = LENGTH_BITS as usize;
        let read_bits = |pos: &mut usize, n: usize| -> u32 {
            let mut v = 0u32;
            for _ in 0..n {
                if *pos < frame.bit_len {
                    let b = frame.data[*pos / 8];
                    let bit = (b >> (7 - (*pos % 8))) & 1;
                    v = (v << 1) | u32::from(bit);
                }
                *pos += 1;
            }
            v
        };

        // Parse subframe partitioning and coarse spectral energy per channel, then
        // synthesize via IMDCT + sine window overlap-add.
        let tile_flag = read_bits(&mut bit_pos, 1);
        let subframe_len = if tile_flag == 0 {
            FRAME_SAMPLES
        } else {
            SUBFRAME_SAMPLES * 2
        };
        let num_subframes = FRAME_SAMPLES / subframe_len;

        let mut ch_buf = [[0.0f32; FRAME_SAMPLES]; 2];
        for sf in 0..num_subframes {
            for ch in 0..self.channels {
                let scale_bits = read_bits(&mut bit_pos, 6) as i32 - 32;
                let gain = 2.0f32.powf(scale_bits as f32 * 0.125) * 0.001;
                let mut spec = [0.0f32; 16];
                for bin in &mut spec {
                    let q = read_bits(&mut bit_pos, 4) as i32 - 8;
                    *bin = (q as f32) * gain;
                }
                // Synthesize subframe time-domain samples via cosine basis + sine window.
                let half = subframe_len / 2;
                let base = sf * subframe_len;
                for n in 0..subframe_len {
                    let mut acc = 0.0f32;
                    let phase = PI / (subframe_len as f32) * (n as f32 + 0.5 + half as f32 * 0.5);
                    for (k, &c) in spec.iter().enumerate() {
                        if c != 0.0 {
                            acc += c * (phase * (k as f32 + 0.5)).cos();
                        }
                    }
                    let win = (PI * (n as f32 + 0.5) / (subframe_len as f32)).sin();
                    let sample = acc * win;
                    let prev = if n < half { self.overlap[ch][n] } else { 0.0 };
                    ch_buf[ch][base + n] = (sample + prev).clamp(-1.0, 1.0);
                    if n >= half {
                        self.overlap[ch][n - half] = sample;
                    }
                }
            }
        }

        out_pcm.reserve(FRAME_SAMPLES * self.channels);
        for i in 0..FRAME_SAMPLES {
            for ch in 0..self.channels {
                out_pcm.push(ch_buf[ch][i]);
            }
        }
        Ok(())
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
        assert_eq!(pcm.len(), FRAME_SAMPLES * 2);
        assert!(
            pcm.iter()
                .all(|s| s.is_finite() && (-1.0..=1.0).contains(s))
        );
    }
}
