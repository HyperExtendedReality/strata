//! Console Audio HLE: the XAudio2 voice graph and mixer, XAPU/AcpHal
//! hardware XMA2 contexts (backed by `strata_formats::xma`), DSP chain, and
//! host output (WASAPI shared mode, `waveOut` as a fallback).
//!
//! Source buffers are decoded when submitted (PCM16, float, XMA2), played
//! through their play/loop ranges at the source rate times the frequency
//! ratio, resampled to the device rate, and reported back as the voice
//! callback events (`OnBufferStart` / `OnBufferEnd` / `OnLoopEnd` /
//! `OnStreamEnd` / pass start and end) XAudio2 titles stream with.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};

use strata_formats::opus::OpusDecoder;
#[cfg(test)]
use strata_formats::xma::FRAME_SAMPLES;
use strata_formats::xma::XmaDecoder;

/// Audio sample format submitted to a source voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveFormat {
    Pcm16 { channels: u16, sample_rate: u32 },
    PcmF32 { channels: u16, sample_rate: u32 },
    Xma2 { channels: u16, sample_rate: u32 },
    Opus { channels: u16, sample_rate: u32 },
}

impl WaveFormat {
    pub fn channels(self) -> usize {
        match self {
            Self::Pcm16 { channels, .. }
            | Self::PcmF32 { channels, .. }
            | Self::Xma2 { channels, .. }
            | Self::Opus { channels, .. } => (channels as usize).clamp(1, 8),
        }
    }

    pub fn sample_rate(self) -> u32 {
        match self {
            Self::Pcm16 { sample_rate, .. }
            | Self::PcmF32 { sample_rate, .. }
            | Self::Xma2 { sample_rate, .. }
            | Self::Opus { sample_rate, .. } => sample_rate.max(1000),
        }
    }

    /// The format of a `WAVEFORMATEX` (PCM, IEEE float, `WAVE_FORMAT_EXTENSIBLE`
    /// of either, or XMA2), if Strata plays it.
    pub fn from_waveformatex(b: &[u8]) -> Option<Self> {
        let u16_at = |o: usize| b.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]]));
        let u32_at = |o: usize| {
            b.get(o..o + 4)
                .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        };
        let tag = u16_at(0)?;
        let channels = u16_at(2)?.clamp(1, 8);
        let sample_rate = u32_at(4)?;
        let bits = u16_at(14)?;
        let tag = if tag == 0xFFFE {
            // EXTENSIBLE: the sub-format GUID's first dword is the real tag.
            u32_at(24)? as u16
        } else {
            tag
        };
        Some(match (tag, bits) {
            (1, 16) => Self::Pcm16 {
                channels,
                sample_rate,
            },
            (3, 32) => Self::PcmF32 {
                channels,
                sample_rate,
            },
            (0x166, _) => Self::Xma2 {
                channels,
                sample_rate,
            },
            (0x704F, _) => Self::Opus {
                channels,
                sample_rate,
            },
            _ => return None,
        })
    }
}

/// Queued audio buffer for an XAudio2 source voice (`XAUDIO2_BUFFER`).
#[derive(Debug, Clone, Default)]
pub struct AudioBuffer {
    pub data: Vec<u8>,
    pub loop_count: u32,
    pub end_of_stream: bool,
    /// `PlayBegin` / `PlayLength` (0: to the end), in frames.
    pub play_begin: u32,
    pub play_length: u32,
    /// `LoopBegin` / `LoopLength` (0: to the end of the play range), in frames.
    pub loop_begin: u32,
    pub loop_length: u32,
    /// `pContext`, handed back in the buffer callbacks.
    pub context: u64,
}

/// A buffer decoded to interleaved float frames, with its play position.
#[derive(Debug, Clone)]
struct Playing {
    pcm: Vec<f32>,
    pos: usize,
    end: usize,
    loop_start: usize,
    loop_end: usize,
    loops_left: u32,
    started: bool,
    end_of_stream: bool,
    context: u64,
}

/// What a render pass reports to a voice's `IXAudio2VoiceCallback`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceEvent {
    PassStart { bytes_required: u32 },
    PassEnd,
    BufferStart(u64),
    BufferEnd(u64),
    LoopEnd(u64),
    StreamEnd,
}

/// An event for the callback registered with a voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallbackEvent {
    pub callback: usize,
    pub event: VoiceEvent,
}

#[derive(Debug, Clone)]
pub struct SourceVoice {
    pub id: u32,
    pub format: WaveFormat,
    pub volume: f32,
    pub pitch_ratio: f32,
    pub playing: bool,
    pub submix_target: Option<u32>,
    /// `IXAudio2VoiceCallback*` (0: none).
    pub callback: usize,
    queue: VecDeque<Playing>,
    pub samples_played: u64,
    /// Fractional source position for resampling, and the last frame.
    phase: f64,
    a: [f32; 8],
    b: [f32; 8],
    a_valid: bool,
    b_valid: bool,
    primed: bool,
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
            callback: 0,
            queue: VecDeque::new(),
            samples_played: 0,
            phase: 0.0,
            a: [0.0; 8],
            b: [0.0; 8],
            a_valid: false,
            b_valid: false,
            primed: false,
            xma,
        }
    }

    /// Decodes and queues a buffer.
    pub fn submit_buffer(&mut self, buf: AudioBuffer) {
        let ch = self.format.channels();
        let pcm: Vec<f32> = match self.format {
            WaveFormat::Pcm16 { .. } => buf
                .data
                .chunks_exact(2)
                .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
                .collect(),
            WaveFormat::PcmF32 { .. } => buf
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).clamp(-1.0, 1.0))
                .collect(),
            WaveFormat::Xma2 { .. } => match &mut self.xma {
                Some(dec) => dec.decode_packets(&buf.data).unwrap_or_default(),
                None => Vec::new(),
            },
            WaveFormat::Opus {
                channels,
                sample_rate,
            } => {
                let mut dec = OpusDecoder::new(channels, sample_rate);
                dec.decode_stream(&buf.data)
            }
        };
        let frames = pcm.len() / ch;
        let begin = (buf.play_begin as usize).min(frames);
        let end = if buf.play_length == 0 {
            frames
        } else {
            (begin + buf.play_length as usize).min(frames)
        };
        let loop_start = (buf.loop_begin as usize).clamp(begin, end);
        let loop_end = if buf.loop_length == 0 {
            end
        } else {
            (loop_start + buf.loop_length as usize).min(end)
        };
        self.queue.push_back(Playing {
            pcm,
            pos: begin,
            end,
            loop_start,
            loop_end,
            loops_left: buf.loop_count,
            started: false,
            end_of_stream: buf.end_of_stream,
            context: buf.context,
        });
    }

    /// Drops every queued buffer, reporting each as ended.
    pub fn flush(&mut self, events: &mut Vec<CallbackEvent>) {
        for b in self.queue.drain(..) {
            if self.callback != 0 {
                events.push(CallbackEvent {
                    callback: self.callback,
                    event: VoiceEvent::BufferEnd(b.context),
                });
            }
        }
        if let Some(dec) = &mut self.xma {
            dec.reset();
        }
        self.primed = false;
    }

    /// `ExitLoop`: the current buffer stops looping after this pass.
    pub fn exit_loop(&mut self) {
        if let Some(b) = self.queue.front_mut() {
            b.loops_left = 0;
        }
    }

    /// `Discontinuity`: the last queued buffer ends the stream.
    pub fn discontinuity(&mut self) {
        if let Some(b) = self.queue.back_mut() {
            b.end_of_stream = true;
        }
    }

    pub fn buffers_queued(&self) -> u32 {
        self.queue.len() as u32
    }

    /// The context of the buffer being played (0: none).
    pub fn current_context(&self) -> u64 {
        self.queue.front().map_or(0, |b| b.context)
    }

    /// The next source frame, advancing through buffers, loops and their
    /// callbacks; `None` once the queue runs dry.
    fn next_frame(&mut self, events: &mut Vec<CallbackEvent>) -> Option<[f32; 8]> {
        let ch = self.format.channels();
        let cb = self.callback;
        let mut push = |e: VoiceEvent| {
            if cb != 0 {
                events.push(CallbackEvent {
                    callback: cb,
                    event: e,
                });
            }
        };
        loop {
            let b = self.queue.front_mut()?;
            if !b.started {
                b.started = true;
                push(VoiceEvent::BufferStart(b.context));
            }
            let limit = if b.loops_left > 0 { b.loop_end } else { b.end };
            if b.pos < limit {
                let mut f = [0.0f32; 8];
                f[..ch].copy_from_slice(&b.pcm[b.pos * ch..b.pos * ch + ch]);
                b.pos += 1;
                return Some(f);
            }
            if b.loops_left > 0 && b.loop_end > b.loop_start {
                if b.loops_left != 255 {
                    b.loops_left -= 1;
                }
                b.pos = b.loop_start;
                push(VoiceEvent::LoopEnd(b.context));
                continue;
            }
            let done = self.queue.pop_front().expect("front");
            push(VoiceEvent::BufferEnd(done.context));
            if done.end_of_stream {
                push(VoiceEvent::StreamEnd);
            }
        }
    }

    /// Mix `frames` stereo frames at `out_rate` into `out` (`frames * 2`
    /// samples), resampling linearly from the source rate times the
    /// frequency ratio.
    pub fn mix_into(
        &mut self,
        out: &mut [f32],
        frames: usize,
        out_rate: u32,
        events: &mut Vec<CallbackEvent>,
    ) {
        if !self.playing {
            return;
        }
        let ch = self.format.channels();
        if self.callback != 0 {
            let step = f64::from(self.format.sample_rate()) * f64::from(self.pitch_ratio)
                / f64::from(out_rate.max(1));
            let bytes = match self.format {
                WaveFormat::Pcm16 { .. } => 2,
                _ => 4,
            };
            let needed = (frames as f64 * step).ceil() as usize * ch * bytes;
            let queued: usize = self
                .queue
                .iter()
                .map(|b| (b.end.saturating_sub(b.pos)) * ch * bytes)
                .sum();
            events.push(CallbackEvent {
                callback: self.callback,
                event: VoiceEvent::PassStart {
                    bytes_required: needed.saturating_sub(queued) as u32,
                },
            });
        }
        let step = f64::from(self.format.sample_rate()) * f64::from(self.pitch_ratio.max(0.0005))
            / f64::from(out_rate.max(1));
        // Source position = a frame index plus `phase`; output is the
        // linear blend of the frames on either side (`a`, `b`).
        if !self.primed {
            match self.next_frame(events) {
                Some(f) => {
                    self.a = f;
                    self.samples_played += 1;
                    self.b = match self.next_frame(events) {
                        Some(f) => {
                            self.samples_played += 1;
                            self.b_valid = true;
                            f
                        }
                        None => {
                            self.b_valid = false;
                            [0.0; 8]
                        }
                    };
                    self.a_valid = true;
                    self.phase = 0.0;
                    self.primed = true;
                }
                None => self.a_valid = false,
            }
        }
        for f in 0..frames {
            if !self.a_valid {
                // Starved: silence until more buffers arrive (then re-prime).
                self.primed = false;
                break;
            }
            let t = self.phase as f32;
            let at = |k: usize| self.a[k] + (self.b[k] - self.a[k]) * t;
            let (l, r) = if ch >= 2 {
                (at(0), at(1))
            } else {
                (at(0), at(0))
            };
            out[f * 2] += l * self.volume;
            out[f * 2 + 1] += r * self.volume;
            self.phase += step;
            while self.phase >= 1.0 {
                self.phase -= 1.0;
                self.a = self.b;
                self.a_valid = self.b_valid;
                match self.next_frame(events) {
                    Some(fr) => {
                        self.b = fr;
                        self.b_valid = true;
                        self.samples_played += 1;
                    }
                    None => {
                        self.b = [0.0; 8];
                        self.b_valid = false;
                    }
                }
            }
        }
        if self.callback != 0 {
            events.push(CallbackEvent {
                callback: self.callback,
                event: VoiceEvent::PassEnd,
            });
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
    /// The mastering voice's id (0: none yet).
    pub mastering: u32,
    /// `StartEngine` / `StopEngine`.
    pub running: bool,
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
            mastering: 0,
            running: true,
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
        for v in self.sources.values_mut() {
            if v.submix_target == Some(id) {
                v.submix_target = None;
            }
        }
    }

    /// Render `frames` interleaved stereo `f32` samples across the full voice graph.
    pub fn render_stereo(&mut self, frames: usize) -> Vec<f32> {
        self.render(frames).0
    }

    /// As [`Self::render_stereo`], with the callback events the pass raised
    /// (to be delivered without the engine locked: callbacks submit buffers).
    pub fn render(&mut self, frames: usize) -> (Vec<f32>, Vec<CallbackEvent>) {
        let mut master = vec![0.0f32; frames * 2];
        let mut events = Vec::new();
        if !self.running {
            return (master, events);
        }
        let mut submix_bufs: HashMap<u32, Vec<f32>> = self
            .submixes
            .keys()
            .map(|&id| (id, vec![0.0f32; frames * 2]))
            .collect();
        let rate = self.master_sample_rate;
        let mut ids: Vec<u32> = self.sources.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            let voice = self.sources.get_mut(&id).expect("listed");
            match voice
                .submix_target
                .filter(|t| *t != self.mastering)
                .and_then(|t| submix_bufs.get_mut(&t))
            {
                Some(buf) => voice.mix_into(buf, frames, rate, &mut events),
                None => voice.mix_into(&mut master, frames, rate, &mut events),
            }
        }

        for (id, sub) in &mut self.submixes {
            if *id == self.mastering {
                continue;
            }
            if let Some(buf) = submix_bufs.get_mut(id) {
                for fx in &mut sub.effects {
                    fx.process_stereo(buf);
                }
                for (m, &s) in master.iter_mut().zip(buf.iter()) {
                    *m += s * sub.volume;
                }
            }
        }

        let mv = self.master_volume * self.submixes.get(&self.mastering).map_or(1.0, |m| m.volume);
        for s in &mut master {
            *s *= mv;
        }
        for fx in &mut self.master_effects {
            fx.process_stereo(&mut master);
        }
        for s in &mut master {
            *s = s.clamp(-1.0, 1.0);
        }
        self.frames_mixed += frames as u64;
        (master, events)
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

/// Live WASAPI shared-mode output on the default render endpoint: 32-bit
/// float stereo at the engine rate (Windows converts to the device mix
/// format: `AUTOCONVERTPCM | SRC_DEFAULT_QUALITY`), written as space frees
/// in a 100 ms buffer.
pub struct WasapiSink {
    client: *mut std::ffi::c_void,
    render: *mut std::ffi::c_void,
    pub buffer_frames: u32,
    pub sample_rate: u32,
    pub frames_written: u64,
}

// SAFETY: used only by the audio thread that made it.
unsafe impl Send for WasapiSink {}

#[cfg(windows)]
mod wasapi_ffi {
    use std::ffi::c_void;

    #[repr(C)]
    pub struct Guid(pub u32, pub u16, pub u16, pub [u8; 8]);

    pub const CLSID_MM_DEVICE_ENUMERATOR: Guid = Guid(
        0xBCDE_0395,
        0xE52F,
        0x467C,
        [0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E],
    );
    pub const IID_IMM_DEVICE_ENUMERATOR: Guid = Guid(
        0xA956_64D2,
        0x9614,
        0x4F35,
        [0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6],
    );
    pub const IID_IAUDIO_CLIENT: Guid = Guid(
        0x1CB9_AD4C,
        0xDBFA,
        0x4C32,
        [0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2],
    );
    pub const IID_IAUDIO_RENDER_CLIENT: Guid = Guid(
        0xF294_ACFC,
        0x3146,
        0x4483,
        [0xA7, 0xBF, 0xAD, 0xDC, 0xA7, 0xC2, 0x60, 0xE2],
    );
    pub const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: Guid = Guid(
        0x0000_0003,
        0x0000,
        0x0010,
        [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
    );

    #[repr(C, packed(2))]
    pub struct WaveFormatExtensible {
        pub tag: u16,
        pub channels: u16,
        pub rate: u32,
        pub avg_bytes: u32,
        pub block_align: u16,
        pub bits: u16,
        pub cb_size: u16,
        pub valid_bits: u16,
        pub channel_mask: u32,
        pub sub_format: Guid,
    }

    #[link(name = "ole32")]
    unsafe extern "system" {
        pub fn CoInitializeEx(reserved: *mut c_void, coinit: u32) -> i32;
        pub fn CoCreateInstance(
            clsid: *const Guid,
            outer: *mut c_void,
            ctx: u32,
            iid: *const Guid,
            out: *mut *mut c_void,
        ) -> i32;
    }

    /// Slot `n` of a COM object's vtable.
    pub unsafe fn slot<T: Copy>(obj: *mut c_void, n: usize) -> T {
        unsafe {
            let vt = *(obj as *const *const *const ());
            std::mem::transmute_copy(&*vt.add(n))
        }
    }

    pub unsafe fn release(obj: *mut c_void) {
        if !obj.is_null() {
            unsafe {
                let f: unsafe extern "system" fn(*mut c_void) -> u32 = slot(obj, 2);
                f(obj);
            }
        }
    }
}

impl WasapiSink {
    /// Opens the default render endpoint for float stereo at `sample_rate`.
    #[cfg(windows)]
    pub fn open(sample_rate: u32) -> Result<Self, i32> {
        use std::ffi::c_void;
        use wasapi_ffi::*;
        unsafe {
            // Multithreaded apartment; S_FALSE / RPC_E_CHANGED_MODE are fine.
            let _ = CoInitializeEx(std::ptr::null_mut(), 0);
            let mut enumerator = std::ptr::null_mut();
            let hr = CoCreateInstance(
                &CLSID_MM_DEVICE_ENUMERATOR,
                std::ptr::null_mut(),
                0x17, // CLSCTX_ALL
                &IID_IMM_DEVICE_ENUMERATOR,
                &mut enumerator,
            );
            if hr < 0 {
                return Err(hr);
            }
            let get_default: unsafe extern "system" fn(
                *mut c_void,
                u32,
                u32,
                *mut *mut c_void,
            ) -> i32 = slot(enumerator, 4);
            let mut device = std::ptr::null_mut();
            let hr = get_default(
                enumerator,
                0, /* eRender */
                0, /* eConsole */
                &mut device,
            );
            release(enumerator);
            if hr < 0 {
                return Err(hr);
            }
            let activate: unsafe extern "system" fn(
                *mut c_void,
                *const Guid,
                u32,
                *mut c_void,
                *mut *mut c_void,
            ) -> i32 = slot(device, 3);
            let mut client = std::ptr::null_mut();
            let hr = activate(
                device,
                &IID_IAUDIO_CLIENT,
                0x17,
                std::ptr::null_mut(),
                &mut client,
            );
            release(device);
            if hr < 0 {
                return Err(hr);
            }
            let fmt = WaveFormatExtensible {
                tag: 0xFFFE,
                channels: 2,
                rate: sample_rate,
                avg_bytes: sample_rate * 8,
                block_align: 8,
                bits: 32,
                cb_size: 22,
                valid_bits: 32,
                channel_mask: 3, // front left | front right
                sub_format: KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
            };
            let initialize: unsafe extern "system" fn(
                *mut c_void,
                u32,
                u32,
                i64,
                i64,
                *const WaveFormatExtensible,
                *const Guid,
            ) -> i32 = slot(client, 3);
            const AUTOCONVERTPCM: u32 = 0x8000_0000;
            const SRC_DEFAULT_QUALITY: u32 = 0x0800_0000;
            let hr = initialize(
                client,
                0, // AUDCLNT_SHAREMODE_SHARED
                AUTOCONVERTPCM | SRC_DEFAULT_QUALITY,
                1_000_000, // 100 ms, in 100 ns units
                0,
                &fmt,
                std::ptr::null(),
            );
            if hr < 0 {
                release(client);
                return Err(hr);
            }
            let get_size: unsafe extern "system" fn(*mut c_void, *mut u32) -> i32 = slot(client, 4);
            let mut buffer_frames = 0u32;
            get_size(client, &mut buffer_frames);
            let get_service: unsafe extern "system" fn(
                *mut c_void,
                *const Guid,
                *mut *mut c_void,
            ) -> i32 = slot(client, 14);
            let mut render = std::ptr::null_mut();
            let hr = get_service(client, &IID_IAUDIO_RENDER_CLIENT, &mut render);
            if hr < 0 {
                release(client);
                return Err(hr);
            }
            let start: unsafe extern "system" fn(*mut c_void) -> i32 = slot(client, 10);
            start(client);
            Ok(Self {
                client,
                render,
                buffer_frames,
                sample_rate,
                frames_written: 0,
            })
        }
    }

    #[cfg(not(windows))]
    pub fn open(_sample_rate: u32) -> Result<Self, i32> {
        Err(0x8000_4001u32 as i32)
    }

    /// Frames the device buffer can take now.
    #[cfg(windows)]
    pub fn writable_frames(&self) -> u32 {
        unsafe {
            let padding: unsafe extern "system" fn(*mut std::ffi::c_void, *mut u32) -> i32 =
                wasapi_ffi::slot(self.client, 6);
            let mut p = 0u32;
            if padding(self.client, &mut p) < 0 {
                return 0;
            }
            self.buffer_frames.saturating_sub(p)
        }
    }

    #[cfg(not(windows))]
    pub fn writable_frames(&self) -> u32 {
        0
    }

    /// Writes interleaved stereo frames (at most [`Self::writable_frames`]).
    #[cfg(windows)]
    pub fn write_f32(&mut self, stereo: &[f32]) -> Result<(), i32> {
        let frames = (stereo.len() / 2) as u32;
        if frames == 0 {
            return Ok(());
        }
        unsafe {
            let get: unsafe extern "system" fn(*mut std::ffi::c_void, u32, *mut *mut f32) -> i32 =
                wasapi_ffi::slot(self.render, 3);
            let rel: unsafe extern "system" fn(*mut std::ffi::c_void, u32, u32) -> i32 =
                wasapi_ffi::slot(self.render, 4);
            let mut data = std::ptr::null_mut();
            let hr = get(self.render, frames, &mut data);
            if hr < 0 {
                return Err(hr);
            }
            std::ptr::copy_nonoverlapping(stereo.as_ptr(), data, frames as usize * 2);
            rel(self.render, frames, 0);
        }
        self.frames_written += u64::from(frames);
        Ok(())
    }

    #[cfg(not(windows))]
    pub fn write_f32(&mut self, _stereo: &[f32]) -> Result<(), i32> {
        Ok(())
    }
}

#[cfg(windows)]
impl Drop for WasapiSink {
    fn drop(&mut self) {
        unsafe {
            let stop: unsafe extern "system" fn(*mut std::ffi::c_void) -> i32 =
                wasapi_ffi::slot(self.client, 11);
            stop(self.client);
            wasapi_ffi::release(self.render);
            wasapi_ffi::release(self.client);
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
                end_of_stream: true,
                ..Default::default()
            });
        }
        let out = eng.render_stereo(16);
        assert_eq!(out.len(), 32);
        assert!((out[0] - 0.25).abs() < 1e-3);
        assert_eq!(eng.sources[&v].samples_played, 16);
    }

    #[test]
    fn buffers_raise_callbacks_and_resample_to_the_device_rate() {
        let mut eng = XAudio2Engine::new(48_000);
        // 24 kHz mono source: every source frame lasts two device frames.
        let v = eng.create_source_voice(WaveFormat::Pcm16 {
            channels: 1,
            sample_rate: 24_000,
        });
        let voice = eng.sources.get_mut(&v).unwrap();
        voice.callback = 0x1000;
        voice.playing = true;
        let data: Vec<u8> = (0..8i16).flat_map(|s| (s * 1000).to_le_bytes()).collect();
        voice.submit_buffer(AudioBuffer {
            data: data.clone(),
            context: 7,
            ..Default::default()
        });
        voice.submit_buffer(AudioBuffer {
            data,
            context: 8,
            end_of_stream: true,
            loop_count: 1,
            ..Default::default()
        });
        let (out, events) = eng.render(64);
        // 8 + 8 x 2 (one loop) = 24 source frames -> 48 device frames.
        assert_eq!(eng.sources[&v].samples_played, 24);
        assert!(out[2 * 47].abs() > 0.0 && out[2 * 50] == 0.0);
        let kinds: Vec<VoiceEvent> = events
            .iter()
            .map(|e| e.event)
            .filter(|e| !matches!(e, VoiceEvent::PassStart { .. } | VoiceEvent::PassEnd))
            .collect();
        assert_eq!(
            kinds,
            [
                VoiceEvent::BufferStart(7),
                VoiceEvent::BufferEnd(7),
                VoiceEvent::BufferStart(8),
                VoiceEvent::LoopEnd(8),
                VoiceEvent::BufferEnd(8),
                VoiceEvent::StreamEnd,
            ]
        );
        assert_eq!(eng.sources[&v].buffers_queued(), 0);
    }

    #[test]
    fn waveformatex_tags_map_to_formats() {
        let mut w = [0u8; 40];
        w[0..2].copy_from_slice(&0xFFFEu16.to_le_bytes());
        w[2..4].copy_from_slice(&2u16.to_le_bytes());
        w[4..8].copy_from_slice(&44_100u32.to_le_bytes());
        w[14..16].copy_from_slice(&32u16.to_le_bytes());
        w[24..28].copy_from_slice(&3u32.to_le_bytes()); // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
        assert_eq!(
            WaveFormat::from_waveformatex(&w),
            Some(WaveFormat::PcmF32 {
                channels: 2,
                sample_rate: 44_100
            })
        );
        w[0..2].copy_from_slice(&0x166u16.to_le_bytes());
        assert!(matches!(
            WaveFormat::from_waveformatex(&w),
            Some(WaveFormat::Xma2 { .. })
        ));
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
