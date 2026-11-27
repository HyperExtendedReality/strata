//! Console Audio HLE: XAudio2 voice graph/mixer, XAPU/AcpHal hardware XMA2
//! context pool (backed by `strata_formats::xma::XmaDecoder`), and DSP chain.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};

use strata_formats::xma::{FRAME_SAMPLES, XmaDecoder};

/// Audio sample format submitted to a source voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveFormat {
    Pcm16 { channels: u16, sample_rate: u32 },
    PcmF32 { channels: u16, sample_rate: u32 },
    Xma2 { channels: u16, sample_rate: u32 },
}

impl WaveFormat {
    pub fn channels(self) -> usize {
        match self {
            Self::Pcm16 { channels, .. }
            | Self::PcmF32 { channels, .. }
            | Self::Xma2 { channels, .. } => (channels as usize).clamp(1, 8),
        }
    }

    pub fn sample_rate(self) -> u32 {
        match self {
            Self::Pcm16 { sample_rate, .. }
            | Self::PcmF32 { sample_rate, .. }
            | Self::Xma2 { sample_rate, .. } => sample_rate.max(8000),
        }
    }
}

/// Queued audio buffer for an XAudio2 source voice.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub data: Vec<u8>,
    pub loop_count: u32,
    pub end_of_stream: bool,
}

#[derive(Debug, Clone)]
pub struct SourceVoice {
    pub id: u32,
    pub format: WaveFormat,
    pub volume: f32,
    pub pitch_ratio: f32,
    pub playing: bool,
    pub submix_target: Option<u32>,
    pub queue: VecDeque<AudioBuffer>,
    pub decoded_pcm: VecDeque<f32>,
    pub samples_played: u64,
    xma: Option<XmaDecoder>,
}

impl SourceVoice {
    pub fn new(id: u32, format: WaveFormat) -> Self {
        let xma = match format {
            WaveFormat::Xma2 {
                channels,
                sample_rate,
            } => XmaDecoder::new((channels as u8).clamp(1, 2), sample_rate).ok(),
            _ => None,
        };
        Self {
            id,
            format,
            volume: 1.0,
            pitch_ratio: 1.0,
            playing: false,
            submix_target: None,
            queue: VecDeque::new(),
            decoded_pcm: VecDeque::new(),
            samples_played: 0,
            xma,
        }
    }

    pub fn submit_buffer(&mut self, buf: AudioBuffer) {
        self.queue.push_back(buf);
    }

    pub fn flush(&mut self) {
        self.queue.clear();
        self.decoded_pcm.clear();
        if let Some(dec) = &mut self.xma {
            dec.reset();
        }
    }

    fn refill_decoded(&mut self) {
        while self.decoded_pcm.len() < FRAME_SAMPLES * 2 {
            let Some(buf) = self.queue.pop_front() else {
                break;
            };
            match self.format {
                WaveFormat::Pcm16 { .. } => {
                    for chunk in buf.data.chunks_exact(2) {
                        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                        self.decoded_pcm.push_back(f32::from(s) / 32768.0);
                    }
                }
                WaveFormat::PcmF32 { .. } => {
                    for chunk in buf.data.chunks_exact(4) {
                        let s = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        self.decoded_pcm.push_back(s.clamp(-1.0, 1.0));
                    }
                }
                WaveFormat::Xma2 { .. } => {
                    if let Some(dec) = &mut self.xma
                        && let Ok(pcm) = dec.decode_packets(&buf.data)
                    {
                        self.decoded_pcm.extend(pcm);
                    }
                }
            }
            if buf.loop_count > 0 {
                let mut next = buf;
                if next.loop_count != u32::MAX {
                    next.loop_count -= 1;
                }
                self.queue.push_front(next);
            }
        }
    }

    /// Mix `frames` stereo frames into `out` (length `frames * 2`), applying volume.
    pub fn mix_into(&mut self, out: &mut [f32], frames: usize) {
        if !self.playing {
            return;
        }
        self.refill_decoded();
        let ch = self.format.channels();
        for f in 0..frames {
            if self.decoded_pcm.len() < ch {
                break;
            }
            let l = self.decoded_pcm.pop_front().unwrap_or(0.0);
            let r = if ch >= 2 {
                let r_s = self.decoded_pcm.pop_front().unwrap_or(l);
                for _ in 2..ch {
                    let _ = self.decoded_pcm.pop_front();
                }
                r_s
            } else {
                l
            };
            out[f * 2] += l * self.volume;
            out[f * 2 + 1] += r * self.volume;
            self.samples_played += 1;
        }
    }
}

/// Optional DSP effect stage applied to a submix or mastering voice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DspEffect {
    Gain(f32),
    SoftLimiter {
        threshold: f32,
    },
    LowPass {
        alpha: f32,
        prev_l: f32,
        prev_r: f32,
    },
}

impl DspEffect {
    pub fn process_stereo(&mut self, buf: &mut [f32]) {
        match self {
            Self::Gain(g) => {
                for s in buf.iter_mut() {
                    *s *= *g;
                }
            }
            Self::SoftLimiter { threshold } => {
                let t = threshold.max(0.1);
                for s in buf.iter_mut() {
                    if s.abs() > t {
                        *s = s.signum()
                            * (t + (1.0 - t) * ((s.abs() - t) / (1.0 - t + 1e-4)).tanh());
                    }
                }
            }
            Self::LowPass {
                alpha,
                prev_l,
                prev_r,
            } => {
                let a = alpha.clamp(0.0, 1.0);
                for frame in buf.chunks_exact_mut(2) {
                    *prev_l = *prev_l + a * (frame[0] - *prev_l);
                    *prev_r = *prev_r + a * (frame[1] - *prev_r);
                    frame[0] = *prev_l;
                    frame[1] = *prev_r;
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubmixVoice {
    pub id: u32,
    pub volume: f32,
    pub effects: Vec<DspEffect>,
}

/// XAudio2 engine state managing mastering, submix, and source voices.
#[derive(Debug)]
pub struct XAudio2Engine {
    next_id: AtomicU32,
    pub master_volume: f32,
    pub master_sample_rate: u32,
    pub master_effects: Vec<DspEffect>,
    pub submixes: HashMap<u32, SubmixVoice>,
    pub sources: HashMap<u32, SourceVoice>,
    pub frames_mixed: u64,
}

impl Default for XAudio2Engine {
    fn default() -> Self {
        Self::new(48_000)
    }
}

impl XAudio2Engine {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            next_id: AtomicU32::new(1),
            master_volume: 1.0,
            master_sample_rate: sample_rate,
            master_effects: vec![DspEffect::SoftLimiter { threshold: 0.95 }],
            submixes: HashMap::new(),
            sources: HashMap::new(),
            frames_mixed: 0,
        }
    }

    pub fn create_submix_voice(&mut self) -> u32 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.submixes.insert(
            id,
            SubmixVoice {
                id,
                volume: 1.0,
                effects: Vec::new(),
            },
        );
        id
    }

    pub fn create_source_voice(&mut self, format: WaveFormat) -> u32 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.sources.insert(id, SourceVoice::new(id, format));
        id
    }

    pub fn destroy_voice(&mut self, id: u32) {
        self.sources.remove(&id);
        self.submixes.remove(&id);
    }

    /// Render `frames` interleaved stereo `f32` samples across the full voice graph.
    pub fn render_stereo(&mut self, frames: usize) -> Vec<f32> {
        let mut master = vec![0.0f32; frames * 2];
        let mut submix_bufs: HashMap<u32, Vec<f32>> = self
            .submixes
            .keys()
            .map(|&id| (id, vec![0.0f32; frames * 2]))
            .collect();

        for voice in self.sources.values_mut() {
            if let Some(sub_id) = voice.submix_target
                && let Some(buf) = submix_bufs.get_mut(&sub_id)
            {
                voice.mix_into(buf, frames);
                continue;
            }
            voice.mix_into(&mut master, frames);
        }

        for (id, sub) in &mut self.submixes {
            if let Some(buf) = submix_bufs.get_mut(id) {
                for fx in &mut sub.effects {
                    fx.process_stereo(buf);
                }
                for (m, &s) in master.iter_mut().zip(buf.iter()) {
                    *m += s * sub.volume;
                }
            }
        }

        for s in &mut master {
            *s *= self.master_volume;
        }
        for fx in &mut self.master_effects {
            fx.process_stereo(&mut master);
        }
        for s in &mut master {
            *s = s.clamp(-1.0, 1.0);
        }
        self.frames_mixed += frames as u64;
        master
    }
}

/// Hardware XAPU / AcpHal XMA2 context slot.
#[derive(Debug)]
pub struct XmaHardwareContext {
    pub id: u32,
    pub enabled: bool,
    pub channels: u8,
    pub sample_rate: u32,
    pub loop_count: u8,
    pub decoder: XmaDecoder,
    pub output_ring: VecDeque<f32>,
}

/// XAPU / AcpHal hardware audio processor manager.
#[derive(Debug, Default)]
pub struct XapuHal {
    next_ctx: u32,
    pub contexts: HashMap<u32, XmaHardwareContext>,
    pub apu_allocations: HashMap<usize, Vec<u8>>,
}

impl XapuHal {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apu_alloc(&mut self, size: usize) -> usize {
        let mut buf = vec![0u8; size.max(64)];
        let ptr = buf.as_mut_ptr() as usize;
        self.apu_allocations.insert(ptr, buf);
        ptr
    }

    pub fn apu_free(&mut self, ptr: usize) -> bool {
        self.apu_allocations.remove(&ptr).is_some()
    }

    pub fn create_xma_context(&mut self, channels: u8, sample_rate: u32) -> Option<u32> {
        let dec = XmaDecoder::new(channels, sample_rate).ok()?;
        let id = self.next_ctx;
        self.next_ctx += 1;
        self.contexts.insert(
            id,
            XmaHardwareContext {
                id,
                enabled: false,
                channels,
                sample_rate,
                loop_count: 0,
                decoder: dec,
                output_ring: VecDeque::new(),
            },
        );
        Some(id)
    }

    pub fn submit_xma_packets(
        &mut self,
        ctx_id: u32,
        packets: &[u8],
    ) -> Result<usize, strata_formats::xma::XmaError> {
        let ctx = self
            .contexts
            .get_mut(&ctx_id)
            .ok_or(strata_formats::xma::XmaError::InvalidPacket)?;
        let pcm = ctx.decoder.decode_packets(packets)?;
        let count = pcm.len();
        ctx.output_ring.extend(pcm);
        Ok(count)
    }
}

/// Live Windows `winmm.dll` (`waveOut`) PCM audio output sink that converts
/// mixed stereo `f32` frames from [`XAudio2Engine`] into 16-bit PCM and
/// plays them on the host's default audio device (zero external crates).
pub struct WinmmAudioSink {
    hwo: isize,
    pub sample_rate: u32,
    pub channels: u16,
    pub frames_written: u64,
}

#[repr(C)]
struct WaveFormatEx {
    w_format_tag: u16,
    n_channels: u16,
    n_samples_per_sec: u32,
    n_avg_bytes_per_sec: u32,
    n_block_align: u16,
    w_bits_per_sample: u16,
    cb_size: u16,
}

#[repr(C)]
struct WaveHdr {
    lp_data: *mut u8,
    dw_buffer_length: u32,
    dw_bytes_recorded: u32,
    dw_user: usize,
    dw_flags: u32,
    dw_loops: u32,
    lp_next: *mut WaveHdr,
    reserved: usize,
}

#[cfg(windows)]
#[link(name = "winmm")]
unsafe extern "system" {
    fn waveOutOpen(
        phwo: *mut isize,
        u_device_id: u32,
        pwfx: *const WaveFormatEx,
        dw_callback: usize,
        dw_instance: usize,
        fdw_open: u32,
    ) -> u32;
    fn waveOutPrepareHeader(hwo: isize, pwh: *mut WaveHdr, cbwh: u32) -> u32;
    fn waveOutWrite(hwo: isize, pwh: *mut WaveHdr, cbwh: u32) -> u32;
    fn waveOutUnprepareHeader(hwo: isize, pwh: *mut WaveHdr, cbwh: u32) -> u32;
    fn waveOutReset(hwo: isize) -> u32;
    fn waveOutClose(hwo: isize) -> u32;
}

impl WinmmAudioSink {
    /// Open the default Windows waveOut stereo 16-bit PCM device.
    #[cfg(windows)]
    pub fn open(sample_rate: u32) -> Result<Self, u32> {
        let channels = 2u16;
        let bits = 16u16;
        let block_align = channels * (bits / 8);
        let fmt = WaveFormatEx {
            w_format_tag: 1, // WAVE_FORMAT_PCM
            n_channels: channels,
            n_samples_per_sec: sample_rate,
            n_avg_bytes_per_sec: sample_rate * u32::from(block_align),
            n_block_align: block_align,
            w_bits_per_sample: bits,
            cb_size: 0,
        };
        let mut hwo: isize = 0;
        let rc = unsafe {
            waveOutOpen(&mut hwo, u32::MAX /* WAVE_MAPPER */, &fmt, 0, 0, 0)
        };
        if rc != 0 {
            return Err(rc);
        }
        Ok(Self {
            hwo,
            sample_rate,
            channels,
            frames_written: 0,
        })
    }

    /// Convert interleaved stereo `f32` samples `[-1.0, 1.0]` to i16 LE bytes.
    pub fn f32_to_pcm16_bytes(samples: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Submit one mixed stereo `f32` buffer to the host waveOut device.
    #[cfg(windows)]
    pub fn write_f32(&mut self, stereo_samples: &[f32]) -> Result<(), u32> {
        if stereo_samples.is_empty() {
            return Ok(());
        }
        let mut pcm = Self::f32_to_pcm16_bytes(stereo_samples);
        let mut hdr = WaveHdr {
            lp_data: pcm.as_mut_ptr(),
            dw_buffer_length: pcm.len() as u32,
            dw_bytes_recorded: 0,
            dw_user: 0,
            dw_flags: 0,
            dw_loops: 0,
            lp_next: std::ptr::null_mut(),
            reserved: 0,
        };
        let sz = std::mem::size_of::<WaveHdr>() as u32;
        let rc = unsafe { waveOutPrepareHeader(self.hwo, &mut hdr, sz) };
        if rc != 0 {
            return Err(rc);
        }
        let rc = unsafe { waveOutWrite(self.hwo, &mut hdr, sz) };
        if rc == 0 {
            // Wait briefly for the buffer to finish before unpreparing stack memory.
            while (unsafe { std::ptr::read_volatile(&hdr.dw_flags) } & 1/* WHDR_DONE */) == 0 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            self.frames_written += (stereo_samples.len() / self.channels as usize) as u64;
        }
        unsafe {
            let _ = waveOutUnprepareHeader(self.hwo, &mut hdr, sz);
        }
        if rc != 0 { Err(rc) } else { Ok(()) }
    }
}

#[cfg(windows)]
impl Drop for WinmmAudioSink {
    fn drop(&mut self) {
        if self.hwo != 0 {
            unsafe {
                let _ = waveOutReset(self.hwo);
                let _ = waveOutClose(self.hwo);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xaudio2_mixes_pcm_voice_through_submix_and_dsp() {
        let mut eng = XAudio2Engine::new(48_000);
        let sub = eng.create_submix_voice();
        eng.submixes
            .get_mut(&sub)
            .unwrap()
            .effects
            .push(DspEffect::Gain(0.5));
        let v = eng.create_source_voice(WaveFormat::PcmF32 {
            channels: 1,
            sample_rate: 48_000,
        });
        {
            let voice = eng.sources.get_mut(&v).unwrap();
            voice.submix_target = Some(sub);
            voice.playing = true;
            let samples: Vec<u8> = [0.5f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
            voice.submit_buffer(AudioBuffer {
                data: samples,
                loop_count: 0,
                end_of_stream: true,
            });
        }
        let out = eng.render_stereo(16);
        assert_eq!(out.len(), 32);
        assert!((out[0] - 0.25).abs() < 1e-3);
        assert_eq!(eng.sources[&v].samples_played, 16);
    }

    #[test]
    fn xapu_decodes_xma2_into_context_ring() {
        let mut xapu = XapuHal::new();
        let ctx = xapu.create_xma_context(2, 48_000).unwrap();
        let mut pkt = vec![0u8; strata_formats::xma::PACKET_BYTES];
        pkt[0..4].copy_from_slice(&(1u32 << 26).to_be_bytes());
        pkt[4..6].copy_from_slice(&(256u16 << 1).to_be_bytes());
        let samples = xapu.submit_xma_packets(ctx, &pkt).unwrap();
        assert_eq!(samples, FRAME_SAMPLES * 2);
        assert_eq!(xapu.contexts[&ctx].output_ring.len(), FRAME_SAMPLES * 2);
    }
}
