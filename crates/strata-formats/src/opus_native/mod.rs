//! Native Opus decoding through the pure-Rust `opus-pure` port of libopus.
//!
//! The port handles CELT and SILK synthesis, hybrid decoding and mode
//! transitions, redundancy, and RFC 6716 frame packing. The standalone
//! range decoder, PVQ, and bit allocation primitives here were ported
//! from libopus 1.5.2 and are independent of the playback backend.
//!
//! See `THIRD_PARTY_NOTICES.md` for attribution.

use crate::opus::{MAX_FRAME_BYTES, OpusBackend, OpusToc, PacketFrames, packet_frames};

#[allow(dead_code)]
mod celt_tables;
pub mod cwrs;
pub mod entdec;
pub mod rate;

struct NativeOpusBackend {
    decoder: opus_pure::OpusDecoder,
    channels: usize,
    rate: u32,
    resampler: OpusResampler,
}

impl NativeOpusBackend {
    fn decode_frames(
        &mut self,
        packet: &[u8],
        frames: PacketFrames<'_>,
        samples_per_frame: usize,
        pcm: &mut [f32],
    ) -> bool {
        if !frames.has_dtx() {
            let samples = pcm.len() / self.channels;
            return self
                .decoder
                .decode(packet, samples, pcm)
                .is_ok_and(|n| n == samples);
        }

        // libopus handles zero- and one-byte frames through its concealment
        // path, including when they appear among coded frames in a packet.
        let mut single = [0; MAX_FRAME_BYTES + 1];
        single[0] = packet[0] & !3;
        for (frame, dst) in frames
            .iter()
            .zip(pcm.chunks_exact_mut(samples_per_frame * self.channels))
        {
            let input = if frame.len() <= 1 {
                &[]
            } else {
                let frame_end = frame.len() + 1;
                single[1..frame_end].copy_from_slice(frame);
                &single[..frame_end]
            };
            if !self
                .decoder
                .decode(input, samples_per_frame, dst)
                .is_ok_and(|n| n == samples_per_frame)
            {
                return false;
            }
        }
        true
    }
}

impl OpusBackend for NativeOpusBackend {
    fn decode(&mut self, packet: &[u8]) -> Option<Vec<f32>> {
        i32::try_from(packet.len()).ok()?;
        let toc = OpusToc::parse(packet)?;
        let frames = packet_frames(packet, toc)?;
        let samples_per_frame = toc.frame_samples_48k * self.rate as usize / 48_000;
        let samples = samples_per_frame * toc.frame_count;
        let mut pcm = vec![0.0; samples * self.channels];
        if !self.decode_frames(packet, frames, samples_per_frame, &mut pcm) {
            self.resampler.skip(samples, self.rate);
            return None;
        }
        Some(self.resampler.process(pcm, self.rate))
    }

    fn name(&self) -> &'static str {
        "native"
    }
}

/// Creates a decoder for one mono or stereo Opus stream.
pub fn backend(channels: usize, output_rate: u32) -> Option<Box<dyn OpusBackend>> {
    let rate = native_rate(output_rate);
    Some(Box::new(NativeOpusBackend {
        decoder: opus_pure::OpusDecoder::new(rate as i32, channels).ok()?,
        channels,
        rate,
        resampler: OpusResampler::new(channels, output_rate),
    }))
}

/// Chooses a decoder rate for a requested PCM output rate.
pub fn native_rate(rate: u32) -> u32 {
    match rate {
        8_000 | 12_000 | 16_000 | 24_000 | 48_000 => rate,
        _ => 48_000,
    }
}

/// Linearly resamples interleaved PCM, retaining phase across packets.
pub struct OpusResampler {
    channels: usize,
    output_rate: u32,
    input_rate: u32,
    phase: u64,
    previous: Vec<f32>,
}

impl OpusResampler {
    pub fn new(channels: usize, output_rate: u32) -> Self {
        Self {
            channels,
            output_rate,
            input_rate: 0,
            phase: 0,
            previous: Vec::new(),
        }
    }

    fn set_input_rate(&mut self, rate: u32) {
        if self.input_rate != rate {
            self.input_rate = rate;
            self.phase = 0;
            self.previous.clear();
        }
    }

    /// Converts complete interleaved frames from a nonzero input rate.
    pub fn process(&mut self, pcm: Vec<f32>, input_rate: u32) -> Vec<f32> {
        self.set_input_rate(input_rate);
        if self.input_rate == self.output_rate || pcm.is_empty() {
            return pcm;
        }

        let from = u64::from(self.input_rate);
        let to = u64::from(self.output_rate);
        let out_frames = (self.phase + (pcm.len() / self.channels) as u64 * to) / from;
        let mut out = Vec::with_capacity(out_frames as usize * self.channels);
        let mut frames = pcm.chunks_exact(self.channels);
        let Some(first) = frames.next() else {
            return out;
        };
        let last = frames.clone().last().unwrap_or(first);
        let mut previous = if self.previous.is_empty() {
            first
        } else {
            &self.previous
        };
        for current in std::iter::once(first).chain(frames) {
            self.phase += to;
            while self.phase >= from {
                self.phase -= from;
                let weight = 1.0 - self.phase as f32 / to as f32;
                out.extend(
                    previous
                        .iter()
                        .zip(current)
                        .map(|(&a, &b)| a + (b - a) * weight),
                );
            }
            previous = current;
        }
        self.previous.clear();
        self.previous.extend_from_slice(last);
        out
    }

    /// Advances over silence supplied by the caller after a decode failure.
    pub fn skip(&mut self, input_frames: usize, input_rate: u32) {
        self.set_input_rate(input_rate);
        self.phase = (self.phase + input_frames as u64 * u64::from(self.output_rate))
            % u64::from(input_rate);
        self.previous.resize(self.channels, 0.0);
        self.previous.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::OpusResampler;

    #[test]
    fn resampling_is_independent_of_packet_boundaries() {
        let pcm: Vec<_> = (0..387).flat_map(|i| [i as f32, -(i as f32)]).collect();
        for rate in [44_100, 96_000] {
            let expected = OpusResampler::new(2, rate).process(pcm.clone(), 48_000);
            let mut resampler = OpusResampler::new(2, rate);
            let actual: Vec<_> = pcm
                .chunks(60)
                .flat_map(|chunk| resampler.process(chunk.to_vec(), 48_000))
                .collect();
            assert_eq!(actual, expected);
            assert_eq!(actual.len(), 2 * (387 * rate as usize / 48_000));
        }
    }
}
