//! Pure-Rust RFC 6716 Opus TOC/packet parser and frame-level PCM decoder for
//! console APU/DSP hardware audio contexts.

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

        let toc = Self {
            config,
            mode,
            stereo,
            frame_samples_48k,
            frame_count,
        };
        (toc.total_samples_48k() <= opus_pure::MAX_PACKET_SAMPLES).then_some(toc)
    }

    /// Total PCM samples per channel represented by this packet at 48 kHz.
    pub fn total_samples_48k(self) -> usize {
        self.frame_samples_48k * self.frame_count
    }
}

pub(super) const MAX_FRAME_BYTES: usize = 1275;

pub(super) struct PacketFrames<'a> {
    payload: &'a [u8],
    sizes: [u16; 48],
    count: usize,
}

impl<'a> PacketFrames<'a> {
    pub(super) fn has_dtx(&self) -> bool {
        self.sizes[..self.count].iter().any(|&size| size <= 1)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &'a [u8]> + '_ {
        let mut payload = self.payload;
        self.sizes[..self.count].iter().map(move |&size| {
            let (frame, rest) = payload.split_at(usize::from(size));
            payload = rest;
            frame
        })
    }
}

/// Validate the frame lengths and padding from RFC 6716 §3.2.
pub(super) fn packet_frames(packet: &[u8], toc: OpusToc) -> Option<PacketFrames<'_>> {
    let code = packet[0] & 3;
    let (payload, vbr) = match code {
        3 => {
            let control = packet[1];
            let mut payload = &packet[2..];
            if control & 0x40 != 0 {
                payload = strip_packet_padding(payload)?;
            }
            (payload, control & 0x80 != 0)
        }
        _ => (&packet[1..], code == 2),
    };

    let mut sizes = [0; 48];
    let mut frames = payload;
    let count = toc.frame_count;
    if vbr {
        let mut frame_bytes = 0;
        for size in &mut sizes[..count - 1] {
            (*size, frames) = match frames {
                [lo @ 252..=255, hi, rest @ ..] => (u16::from(*lo) + 4 * u16::from(*hi), rest),
                [size @ 0..=251, rest @ ..] => (u16::from(*size), rest),
                _ => return None,
            };
            frame_bytes += usize::from(*size);
        }
        let last_size = frames.len().checked_sub(frame_bytes)?;
        if last_size > MAX_FRAME_BYTES {
            return None;
        }
        sizes[count - 1] = last_size as u16;
    } else {
        let size = frames.len() / count;
        if !frames.len().is_multiple_of(count) || size > MAX_FRAME_BYTES {
            return None;
        }
        sizes[..count].fill(size as u16);
    }
    Some(PacketFrames {
        payload: frames,
        sizes,
        count,
    })
}

fn strip_packet_padding(mut payload: &[u8]) -> Option<&[u8]> {
    loop {
        let (&byte, rest) = payload.split_first()?;
        let padding = usize::from(if byte == 255 { 254 } else { byte });
        payload = &rest[..rest.len().checked_sub(padding)?];
        if byte != 255 {
            return Some(payload);
        }
    }
}

/// A real Opus decoder for one stream (1 or 2 channels): libopus, the
/// Windows Media Foundation decoder, or the native SILK/CELT decoder.
pub trait OpusBackend: Send {
    /// Decodes one packet into interleaved `f32` samples at the backend's
    /// output rate (`None`: the packet could not be decoded).
    fn decode(&mut self, packet: &[u8]) -> Option<Vec<f32>>;
    /// A short name for logs (`"libopus"`, `"MSOpusDecoder"`, ...).
    fn name(&self) -> &'static str;
}

/// Makes a backend for `(channels, sample_rate)`, or `None` when this
/// factory's decoder is unavailable on the host.
pub type OpusBackendFactory = fn(usize, u32) -> Option<Box<dyn OpusBackend>>;

static FACTORIES: std::sync::Mutex<Vec<OpusBackendFactory>> = std::sync::Mutex::new(Vec::new());

/// Registers a backend factory; [`OpusDecoder::new`] tries factories in
/// registration order before using the native decoder.
pub fn register_backend(f: OpusBackendFactory) {
    let mut v = FACTORIES.lock().unwrap_or_else(|e| e.into_inner());
    if !v.iter().any(|&g| g as usize == f as usize) {
        v.push(f);
    }
}

/// Stateful Opus packet decoder producing interleaved `f32` PCM at the
/// configured rate. Mono and stereo use the native decoder by default.
/// The silence fallback preserves timing when no backend is available or
/// when a structurally valid packet cannot be decoded.
pub struct OpusDecoder {
    channels: usize,
    sample_rate: u32,
    sample_remainder: u64,
    packets_decoded: u64,
    backend: Option<Box<dyn OpusBackend>>,
    pub packets_failed: u64,
}

impl Clone for OpusDecoder {
    /// A fresh decoder of the same configuration (backend state is not
    /// copyable).
    fn clone(&self) -> Self {
        if self.backend.is_some() {
            Self::new(self.channels as u16, self.sample_rate)
        } else {
            Self::without_backend(self.channels as u16, self.sample_rate)
        }
    }
}

impl std::fmt::Debug for OpusDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpusDecoder")
            .field("channels", &self.channels)
            .field("sample_rate", &self.sample_rate)
            .field("backend", &self.backend_name())
            .finish()
    }
}

impl OpusDecoder {
    pub fn new(channels: u16, sample_rate: u32) -> Self {
        let mut decoder = Self::without_backend(channels, sample_rate);
        if decoder.channels <= 2 {
            let fs = FACTORIES.lock().unwrap_or_else(|e| e.into_inner()).clone();
            decoder.backend = fs
                .into_iter()
                .find_map(|f| f(decoder.channels, decoder.sample_rate))
                .or_else(|| crate::opus_native::backend(decoder.channels, decoder.sample_rate));
        }
        decoder
    }

    /// Preserves packet duration with silent PCM while decoding is disabled.
    pub fn without_backend(channels: u16, sample_rate: u32) -> Self {
        Self {
            channels: (channels as usize).clamp(1, 8),
            sample_rate: sample_rate.max(8_000),
            sample_remainder: 0,
            packets_decoded: 0,
            backend: None,
            packets_failed: 0,
        }
    }

    /// A decoder using `backend` explicitly.
    pub fn with_backend(channels: u16, sample_rate: u32, backend: Box<dyn OpusBackend>) -> Self {
        let mut decoder = Self::without_backend(channels.min(2), sample_rate);
        decoder.backend = Some(backend);
        decoder
    }

    /// The active backend, or `None` (silence).
    pub fn backend_name(&self) -> Option<&'static str> {
        self.backend.as_ref().map(|b| b.name())
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
    /// Returns `None` if the packet framing is malformed.
    pub fn decode_packet(&mut self, packet: &[u8]) -> Option<Vec<f32>> {
        // The native packet parser and the libopus ABI use signed 32-bit lengths.
        i32::try_from(packet.len()).ok()?;
        let toc = OpusToc::parse(packet)?;
        packet_frames(packet, toc)?;
        self.packets_decoded += 1;
        let output_ticks =
            self.sample_remainder + toc.total_samples_48k() as u64 * u64::from(self.sample_rate);
        self.sample_remainder = output_ticks % 48_000;
        if let Some(b) = self.backend.as_mut() {
            match b.decode(packet) {
                Some(pcm) => return Some(pcm),
                None => self.packets_failed += 1,
            }
        }
        let samples_out = (output_ticks / 48_000) as usize;
        Some(vec![0.0; samples_out * self.channels])
    }

    /// Decode a length-prefixed stream of Opus packets (`[u16 len][packet bytes]...`)
    /// or a single raw Opus packet if the buffer has no complete framed stream
    /// containing a valid packet.
    pub fn decode_stream(&mut self, bytes: &[u8]) -> Vec<f32> {
        // A raw TOC and first payload byte can also look like a length prefix.
        // Check the entire envelope before feeding any packet to the decoder.
        let mut rest = bytes;
        while let Some((_, remaining)) = split_length_prefixed(rest) {
            rest = remaining;
        }
        if !rest.is_empty() {
            return self.decode_packet(bytes).unwrap_or_default();
        }

        let mut out = Vec::new();
        let mut decoded = false;
        rest = bytes;
        while let Some((packet, remaining)) = split_length_prefixed(rest) {
            if let Some(pcm) = self.decode_packet(packet) {
                out.extend(pcm);
                decoded = true;
            }
            rest = remaining;
        }
        if decoded {
            out
        } else {
            self.decode_packet(bytes).unwrap_or_default()
        }
    }
}

fn split_length_prefixed(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let [lo, hi, rest @ ..] = bytes else {
        return None;
    };
    let len = usize::from(u16::from_le_bytes([*lo, *hi]));
    if len == 0 {
        return None;
    }
    rest.split_at_checked(len)
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
    fn accepts_all_frame_codes_and_padding() {
        let mut dec = OpusDecoder::without_backend(1, 48_000);
        let cases: &[(&[u8], usize)] = &[
            (&[0x98], 960),                                // Code 0, DTX.
            (&[0x99, 1, 2], 1920),                         // Code 1, equal-size frames.
            (&[0x9a, 1, 1, 2, 3], 1920),                   // Code 2, unequal sizes.
            (&[0x9b, 3, 1, 2, 3], 2880),                   // Code 3, CBR.
            (&[0x9b, 0xc3, 2, 1, 0, 1, 2, 3, 0, 0], 2880), // Code 3, padded VBR.
            (&[0x83, 48], 5760), // 48 DTX frames and the maximum duration.
        ];
        for &(packet, frames) in cases {
            assert_eq!(
                dec.decode_packet(packet).unwrap().len(),
                frames,
                "{packet:x?}"
            );
        }

        let mut packet = vec![0x9b, 0x41, 255, 1, 1];
        packet.extend([0; 255]);
        assert_eq!(dec.decode_packet(&packet).unwrap().len(), 960);

        let mut packet = vec![0x9a, 252, 1];
        packet.extend([0; 256]);
        assert_eq!(dec.decode_packet(&packet).unwrap().len(), 1920);
    }

    #[test]
    fn rejects_malformed_packets_before_decoding() {
        let mut dec = OpusDecoder::new(1, 48_000);
        let cases: &[&[u8]] = &[
            &[],
            &[0x99, 1],                // Odd-length CBR payload.
            &[0x9a, 252],              // Incomplete two-byte size.
            &[0x9a, 5, 1],             // Size exceeds payload.
            &[0x9b, 0],                // Empty code 3 packet.
            &[0x9b, 7],                // More than 120 ms.
            &[0x9b, 0x43, 6, 1, 2, 3], // Padding exceeds packet.
            &[0x9b, 0x41, 255],        // Truncated padding continuation.
            &[0x9b, 3, 1, 2],          // CBR payload does not divide evenly.
            &[0x9b, 0x83, 0],          // VBR frame sizes are missing.
        ];
        for &packet in cases {
            assert!(dec.decode_packet(packet).is_none(), "{packet:x?}");
        }

        let mut oversized = vec![0; 1277];
        oversized[0] = 0x98;
        assert!(dec.decode_packet(&oversized).is_none());
        assert_eq!(dec.packets_decoded(), 0);
    }

    #[test]
    fn raw_packet_with_partial_length_prefix_decodes_once() {
        let mut dec = OpusDecoder::without_backend(1, 48_000);
        // The first 10 bytes look like an eight-byte prefixed packet.
        // The full buffer is a raw 20 ms SILK packet.
        let packet = [0x08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        assert_eq!(dec.decode_stream(&packet).len(), 960);
        assert_eq!(dec.packets_decoded(), 1);
    }

    #[test]
    fn false_length_prefix_with_invalid_inner_packet_falls_back_to_raw() {
        let mut dec = OpusDecoder::without_backend(1, 48_000);
        // Interpreted as length-prefixed, the inner packet would be code 1
        // with an odd payload size. Interpreted as raw, this is 20 ms SILK.
        let packet = [0x08, 0, 0x99, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(dec.decode_stream(&packet).len(), 960);
        assert_eq!(dec.packets_decoded(), 1);
    }

    #[test]
    fn short_packets_preserve_fractional_pcm_frames() {
        let mut native = OpusDecoder::new(1, 44_100);
        let mut muted = OpusDecoder::without_backend(1, 44_100);
        for expected in [110, 110, 110, 111, 110, 110, 110, 111] {
            // CELT 2.5 ms DTX packet, whose duration is 110.25 samples at 44.1 kHz.
            assert_eq!(native.decode_packet(&[0xe0]).unwrap().len(), expected);
            assert_eq!(muted.decode_packet(&[0xe0]).unwrap().len(), expected);
        }
        assert_eq!(native.packets_failed, 0);
    }

    #[test]
    fn native_decoder_matches_libopus_in_each_mode() {
        // Packets encoded by FFmpeg with libopus from a synthetic signal.
        // Reference samples are signed 16-bit PCM at the untrimmed offsets.
        let cases: &[(OpusMode, &[u8], usize, [i16; 6])] = &[
            (
                OpusMode::SilkOnly,
                &[
                    0x48, 0x82, 0xe2, 0xfc, 0x36, 0x08, 0x9f, 0xc2, 0x01, 0x36, 0x7c, 0xce, 0xe8,
                    0xd5, 0xac, 0x81, 0x95, 0xd3, 0x77, 0x09, 0x24, 0x0e, 0x4d, 0x98, 0xe5, 0xe2,
                    0x7a, 0x9f, 0x66, 0xf2, 0x1e, 0x4a, 0x0b, 0x3c, 0x39, 0xc1, 0xb0, 0xd2, 0xb3,
                    0xee, 0x9c, 0xa8, 0xa8, 0xdd, 0xda, 0x20,
                ],
                382,
                [-6154, -6370, -6629, -6885, -7062, -7113],
            ),
            (
                OpusMode::Hybrid,
                &[
                    0x78, 0x82, 0x01, 0xb7, 0x6c, 0x9e, 0x99, 0xac, 0x00, 0x00, 0x08, 0x56, 0x30,
                    0xa7, 0x25, 0xcb, 0xd0, 0x9b, 0x20, 0x92, 0x6d, 0x41, 0x26, 0x0e, 0xbc, 0xf3,
                    0xa6, 0x41, 0xfa, 0x15, 0x59, 0xa4, 0xec, 0x36, 0x41, 0x69, 0x1a, 0xc2, 0x62,
                    0x0a, 0x6e, 0x0d, 0xd0, 0x0e, 0xa1, 0xdd, 0xec, 0xf5, 0x35, 0xce, 0xac, 0xfe,
                    0xe8, 0x6c, 0x6a, 0x9f, 0xde, 0x27, 0x31, 0xf8, 0x38, 0x07, 0x4d, 0xc2, 0x84,
                    0x2d, 0x6f, 0x94, 0xe7, 0x06, 0xbd, 0x07, 0xe9, 0x0d, 0xf4, 0x10, 0xa9, 0x03,
                    0xfe, 0x4b, 0x93, 0xe0, 0x6c, 0x8c, 0xef, 0xc1, 0x2c, 0xbf, 0x0d, 0x8c, 0x31,
                    0x01, 0x0b, 0x0e, 0xbd, 0x53, 0xdd, 0xf3, 0x21, 0x55, 0x54, 0x03, 0x04, 0xcf,
                    0x73, 0x0b, 0x27, 0xd6, 0xff, 0x97,
                ],
                382,
                [-8906, -7455, -5646, -6464, -8842, -9367],
            ),
            (
                OpusMode::CeltOnly,
                &[
                    0xf0, 0xb5, 0x3f, 0x78, 0xe7, 0x35, 0xfe, 0xa2, 0x98, 0xc4, 0x91, 0xde, 0xe9,
                    0xff, 0xea, 0xce, 0x1e, 0xd1, 0xa8, 0xea, 0x62, 0x4f, 0xbf, 0x8f, 0xdb, 0xe0,
                    0x33, 0x60, 0x24, 0xa6, 0x12, 0x4a, 0x2e, 0x86, 0x03, 0xd5, 0x83, 0x04, 0x73,
                    0x16, 0xe0, 0x5d, 0xf3, 0x29,
                ],
                190,
                [-6092, -5570, -5577, -6805, -7550, -7149],
            ),
        ];

        let mut framed = Vec::new();
        let mut sequential = OpusDecoder::new(1, 48_000);
        let mut expected = Vec::new();
        for &(mode, packet, offset, reference) in cases {
            let toc = OpusToc::parse(packet).unwrap();
            assert_eq!(toc.mode, mode);
            let mut decoder = OpusDecoder::new(1, 48_000);
            assert_eq!(decoder.backend_name(), Some("native"));
            let pcm = decoder.decode_packet(packet).unwrap();
            assert_eq!(decoder.packets_failed, 0);
            assert_eq!(pcm.len(), toc.total_samples_48k());
            for (actual, sample) in pcm[offset..][..reference.len()].iter().zip(reference) {
                let sample = f32::from(sample) / 32_768.0;
                assert!(
                    (actual - sample).abs() < 0.000_03,
                    "{mode:?}: {actual} vs {sample}"
                );
            }
            check_dtx_mixed_with_coded_frames(packet, toc.frame_samples_48k);
            for rate in [8_000, 12_000, 16_000, 24_000, 44_100, 48_000] {
                let mut decoder = OpusDecoder::new(2, rate);
                let pcm = decoder.decode_packet(packet).unwrap();
                assert_eq!(decoder.packets_failed, 0, "{mode:?} at {rate} Hz");
                assert_eq!(
                    pcm.len(),
                    (toc.total_samples_48k() * rate as usize / 48_000) * 2
                );
                assert!(pcm.iter().all(|s| s.is_finite()));
                assert!(pcm.chunks_exact(2).all(|frame| frame[0] == frame[1]));
                assert!(pcm.iter().any(|s| s.abs() > 0.01));
            }

            framed.extend((packet.len() as u16).to_le_bytes());
            framed.extend(packet);
            expected.extend(sequential.decode_packet(packet).unwrap());
        }

        let mut decoder = OpusDecoder::new(1, 48_000);
        assert_eq!(decoder.decode_stream(&framed), expected);
        assert_eq!(decoder.packets_failed, 0);
        assert_eq!(decoder.packets_decoded(), cases.len() as u64);
    }

    fn check_dtx_mixed_with_coded_frames(packet: &[u8], samples: usize) {
        let mut reference = opus_pure::OpusDecoder::new(48_000, 1).unwrap();
        let mut expected = vec![0.0; samples * 4];
        for (input, output) in [packet, &[], &[], packet]
            .into_iter()
            .zip(expected.chunks_exact_mut(samples))
        {
            assert_eq!(reference.decode(input, samples, output).unwrap(), samples);
        }

        // Pack a zero-byte and a one-byte DTX frame between two coded frames.
        let mut combined = vec![packet[0] | 3, 0x84, (packet.len() - 1) as u8, 0, 1];
        combined.extend(&packet[1..]);
        combined.push(0);
        combined.extend(&packet[1..]);
        let mut decoder = OpusDecoder::new(1, 48_000);
        assert_eq!(decoder.decode_packet(&combined).unwrap(), expected);
        assert_eq!(decoder.packets_failed, 0);
    }
}
