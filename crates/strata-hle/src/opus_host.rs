//! Host Opus decoders for [`strata_formats::opus`]: libopus loaded at run
//! time (`opus.dll`, BSD-3-Clause, not shipped with Strata) and Windows'
//! own "Microsoft Opus Audio Decoder" Media Foundation transform
//! (`MSOpusDecoder.dll`). Both are loaded dynamically, so Strata has no
//! link-time dependency on either.
//!
//! `STRATA_OPUS_BACKEND=libopus|mf|native|none` selects the voice decoder.
//! The default is native. The optional libopus DLL is searched for in
//! `STRATA_OPUS_DIR`, next to the executable, then on the DLL search path.

use std::ffi::c_void;
use strata_formats::opus::{OpusBackend, OpusDecoder, OpusToc, register_backend};
use strata_formats::opus_native::{self, OpusResampler, native_rate};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(m: *mut c_void, name: *const i8) -> *const c_void;
}

/// `GetProcAddress` for a NUL-terminated byte name.
unsafe fn proc(m: *mut c_void, name: &[u8]) -> *mut c_void {
    unsafe { GetProcAddress(m, name.as_ptr().cast()) as *mut c_void }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

// ---------------------------------------------------------------------------
// libopus

type OpusCreate = unsafe extern "C" fn(i32, i32, *mut i32) -> *mut c_void;
type OpusDecodeFloat = unsafe extern "C" fn(*mut c_void, *const u8, i32, *mut f32, i32, i32) -> i32;
type OpusDestroy = unsafe extern "C" fn(*mut c_void);

struct LibOpus {
    create: OpusCreate,
    decode: OpusDecodeFloat,
    destroy: OpusDestroy,
}

fn libopus() -> Option<&'static LibOpus> {
    static L: std::sync::OnceLock<Option<LibOpus>> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        let mut candidates = Vec::new();
        if let Ok(dir) = std::env::var("STRATA_OPUS_DIR") {
            candidates.push(std::path::Path::new(&dir).join("opus.dll"));
        }
        if let Ok(exe) = std::env::current_exe()
            && let Some(dir) = exe.parent()
        {
            candidates.push(dir.join("opus.dll"));
        }
        candidates.push("opus.dll".into());
        for c in candidates {
            // SAFETY: loading a library by path; symbols checked below.
            let m = unsafe { LoadLibraryW(wide(&c.to_string_lossy()).as_ptr()) };
            if m.is_null() {
                continue;
            }
            let sym = |n: &[u8]| unsafe { proc(m, n) };
            let (c, d, x) = (
                sym(b"opus_decoder_create\0"),
                sym(b"opus_decode_float\0"),
                sym(b"opus_decoder_destroy\0"),
            );
            if c.is_null() || d.is_null() || x.is_null() {
                continue;
            }
            // SAFETY: the libopus 1.x C ABI.
            return Some(unsafe {
                LibOpus {
                    create: std::mem::transmute::<*mut c_void, OpusCreate>(c),
                    decode: std::mem::transmute::<*mut c_void, OpusDecodeFloat>(d),
                    destroy: std::mem::transmute::<*mut c_void, OpusDestroy>(x),
                }
            });
        }
        None
    })
    .as_ref()
}

struct LibOpusBackend {
    lib: &'static LibOpus,
    st: *mut c_void,
    channels: usize,
    rate: u32,
    resampler: OpusResampler,
}

// SAFETY: the decoder state is owned and only used through `&mut self`.
unsafe impl Send for LibOpusBackend {}

impl Drop for LibOpusBackend {
    fn drop(&mut self) {
        // SAFETY: created by opus_decoder_create.
        unsafe { (self.lib.destroy)(self.st) };
    }
}

impl OpusBackend for LibOpusBackend {
    fn decode(&mut self, packet: &[u8]) -> Option<Vec<f32>> {
        const MAX: usize = 5760; // 120 ms at 48 kHz
        let mut pcm = vec![0f32; MAX * self.channels];
        // SAFETY: `pcm` holds MAX frames of `channels` samples.
        let n = unsafe {
            (self.lib.decode)(
                self.st,
                packet.as_ptr(),
                packet.len() as i32,
                pcm.as_mut_ptr(),
                MAX as i32,
                0,
            )
        };
        if n < 0 {
            if let Some(toc) = OpusToc::parse(packet) {
                let frames = toc.total_samples_48k() * self.rate as usize / 48_000;
                self.resampler.skip(frames, self.rate);
            }
            return None;
        }
        pcm.truncate(n as usize * self.channels);
        Some(self.resampler.process(pcm, self.rate))
    }

    fn name(&self) -> &'static str {
        "libopus"
    }
}

fn libopus_factory(channels: usize, out_rate: u32) -> Option<Box<dyn OpusBackend>> {
    let lib = libopus()?;
    let rate = native_rate(out_rate);
    let mut err = 0i32;
    // SAFETY: libopus C ABI.
    let st = unsafe { (lib.create)(rate as i32, channels as i32, &mut err) };
    if st.is_null() || err != 0 {
        return None;
    }
    Some(Box::new(LibOpusBackend {
        lib,
        st,
        channels,
        rate,
        resampler: OpusResampler::new(channels, out_rate),
    }))
}

// ---------------------------------------------------------------------------
// Media Foundation (MSOpusDecoder.dll)

#[repr(C)]
#[derive(Clone, Copy)]
struct Guid(u32, u16, u16, [u8; 8]);

const fn media_subtype(fourcc: u32) -> Guid {
    Guid(
        fourcc,
        0,
        0x10,
        [0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71],
    )
}

const CLSID_MS_OPUS_DECODER: Guid = Guid(
    0x63e17c10,
    0x2d43,
    0x4c42,
    [0x8f, 0xe3, 0x8d, 0x8b, 0x63, 0xe4, 0x6a, 0x6a],
);
const IID_IMF_TRANSFORM: Guid = Guid(
    0xbf94c121,
    0x5b05,
    0x4e6f,
    [0x80, 0x00, 0xba, 0x59, 0x89, 0x61, 0x41, 0x4d],
);
const IID_ICLASS_FACTORY: Guid = Guid(0x00000001, 0, 0, [0xc0, 0, 0, 0, 0, 0, 0, 0x46]);
const MF_MT_MAJOR_TYPE: Guid = Guid(
    0x48eba18e,
    0xf8c9,
    0x4687,
    [0xbf, 0x11, 0x0a, 0x74, 0xc9, 0xf9, 0x6a, 0x8f],
);
const MF_MT_SUBTYPE: Guid = Guid(
    0xf7e34c9a,
    0x42e8,
    0x4714,
    [0xb7, 0x4b, 0xcb, 0x29, 0xd7, 0x2c, 0x35, 0xe5],
);
const MF_MT_AUDIO_NUM_CHANNELS: Guid = Guid(
    0x37e48bf5,
    0x645e,
    0x4c5b,
    [0x89, 0xde, 0xad, 0xa9, 0xe2, 0x9b, 0x69, 0x6a],
);
const MF_MT_AUDIO_SAMPLES_PER_SECOND: Guid = Guid(
    0x5faeeae7,
    0x0290,
    0x4c31,
    [0x9e, 0x8a, 0xc5, 0x34, 0xf6, 0x8d, 0x9d, 0xba],
);
const MF_MT_USER_DATA: Guid = Guid(
    0xb6bc765f,
    0x4c3b,
    0x40a4,
    [0xbd, 0x51, 0x25, 0x35, 0xb6, 0x6f, 0xe0, 0x9d],
);
const MF_MEDIA_TYPE_AUDIO: Guid = media_subtype(0x7364_7561); // 'auds'
const MF_AUDIO_FORMAT_OPUS: Guid = media_subtype(0x704f);

const MF_VERSION: u32 = 0x0002_0070;
const MFSTARTUP_LITE: u32 = 1;
const COINIT_MULTITHREADED: u32 = 0;
const CLSCTX_INPROC_SERVER: u32 = 1;
const MF_E_TRANSFORM_NEED_MORE_INPUT: i32 = 0xC00D_6D72_u32 as i32;
const MF_E_TRANSFORM_STREAM_CHANGE: i32 = 0xC00D_6D61_u32 as i32;
const MFT_OUTPUT_STREAM_PROVIDES_SAMPLES: u32 = 0x100;
const MFT_MESSAGE_NOTIFY_BEGIN_STREAMING: u32 = 0x1000_0000;
const MFT_MESSAGE_NOTIFY_START_OF_STREAM: u32 = 0x1000_0003;

#[link(name = "ole32")]
unsafe extern "system" {
    fn CoInitializeEx(reserved: *mut c_void, coinit: u32) -> i32;
    fn CoCreateInstance(
        clsid: *const Guid,
        outer: *mut c_void,
        ctx: u32,
        iid: *const Guid,
        out: *mut *mut c_void,
    ) -> i32;
}

type MfStartup = unsafe extern "system" fn(u32, u32) -> i32;
type MfCreateMediaType = unsafe extern "system" fn(*mut *mut c_void) -> i32;
type MfCreateSample = unsafe extern "system" fn(*mut *mut c_void) -> i32;
type MfCreateMemoryBuffer = unsafe extern "system" fn(u32, *mut *mut c_void) -> i32;

struct MfApi {
    create_media_type: MfCreateMediaType,
    create_sample: MfCreateSample,
    create_memory_buffer: MfCreateMemoryBuffer,
}

fn mf() -> Option<&'static MfApi> {
    static M: std::sync::OnceLock<Option<MfApi>> = std::sync::OnceLock::new();
    M.get_or_init(|| unsafe {
        let m = LoadLibraryW(wide("mfplat.dll").as_ptr());
        if m.is_null() {
            return None;
        }
        let sym = |n: &[u8]| proc(m, n);
        let (st, mt, sa, bu) = (
            sym(b"MFStartup\0"),
            sym(b"MFCreateMediaType\0"),
            sym(b"MFCreateSample\0"),
            sym(b"MFCreateMemoryBuffer\0"),
        );
        if st.is_null() || mt.is_null() || sa.is_null() || bu.is_null() {
            return None;
        }
        let _ = CoInitializeEx(std::ptr::null_mut(), COINIT_MULTITHREADED);
        if std::mem::transmute::<*mut c_void, MfStartup>(st)(MF_VERSION, MFSTARTUP_LITE) < 0 {
            return None;
        }
        Some(MfApi {
            create_media_type: std::mem::transmute::<*mut c_void, MfCreateMediaType>(mt),
            create_sample: std::mem::transmute::<*mut c_void, MfCreateSample>(sa),
            create_memory_buffer: std::mem::transmute::<*mut c_void, MfCreateMemoryBuffer>(bu),
        })
    })
    .as_ref()
}

/// Calls COM method `slot` of `obj` (vtable index, `IUnknown` = 0..=2).
macro_rules! com {
    ($obj:expr, $slot:expr, fn($($t:ty),*) -> $r:ty $(, $a:expr)*) => {{
        let obj: *mut c_void = $obj;
        let vt = *(obj as *const *const usize);
        let f: unsafe extern "system" fn(*mut c_void $(, $t)*) -> $r =
            std::mem::transmute(*vt.add($slot));
        f(obj $(, $a)*)
    }};
}

unsafe fn release(obj: *mut c_void) {
    if !obj.is_null() {
        unsafe { com!(obj, 2, fn() -> u32) };
    }
}

// IMFAttributes slots (after IUnknown): SetUINT32 21, SetGUID 24, SetBlob 26.
unsafe fn set_u32(a: *mut c_void, k: &Guid, v: u32) -> i32 {
    unsafe { com!(a, 21, fn(*const Guid, u32) -> i32, k, v) }
}
unsafe fn set_guid(a: *mut c_void, k: &Guid, v: &Guid) -> i32 {
    unsafe { com!(a, 24, fn(*const Guid, *const Guid) -> i32, k, v) }
}
unsafe fn set_blob(a: *mut c_void, k: &Guid, data: &[u8]) -> i32 {
    unsafe {
        com!(
            a,
            26,
            fn(*const Guid, *const u8, u32) -> i32,
            k,
            data.as_ptr(),
            data.len() as u32
        )
    }
}

#[repr(C)]
struct OutputDataBuffer {
    stream_id: u32,
    sample: *mut c_void,
    status: u32,
    events: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct OutputStreamInfo {
    flags: u32,
    size: u32,
    alignment: u32,
}

struct MfOpusBackend {
    api: &'static MfApi,
    mft: *mut c_void,
    channels: usize,
    out_rate: u32,
    resampler: OpusResampler,
    provides_samples: bool,
    out_size: u32,
    time: i64,
    /// The decoder's chosen output: float or 16-bit PCM, channels, rate.
    out_float: bool,
    dec_channels: usize,
    dec_rate: u32,
}

// SAFETY: the transform is created multithreaded and used through `&mut self`.
unsafe impl Send for MfOpusBackend {}

impl Drop for MfOpusBackend {
    fn drop(&mut self) {
        // SAFETY: an owned COM reference.
        unsafe { release(self.mft) };
    }
}

impl MfOpusBackend {
    fn new(channels: usize, out_rate: u32) -> Option<Self> {
        let api = mf()?;
        // SAFETY: COM calls on interfaces created here, per the MF headers.
        unsafe {
            let mut mft = std::ptr::null_mut();
            let hr = CoCreateInstance(
                &CLSID_MS_OPUS_DECODER,
                std::ptr::null_mut(),
                CLSCTX_INPROC_SERVER,
                &IID_IMF_TRANSFORM,
                &mut mft,
            );
            if std::env::var_os("STRATA_OPUS_DEBUG").is_some() {
                eprintln!("MF CoCreateInstance hr={hr:#x}");
            }
            if hr < 0 || mft.is_null() {
                mft = Self::from_dll()?;
            }
            let mut me = Self {
                api,
                mft,
                channels,
                out_rate,
                resampler: OpusResampler::new(channels, out_rate),
                provides_samples: false,
                out_size: 0,
                time: 0,
                out_float: true,
                dec_channels: channels,
                dec_rate: 48000,
            };
            me.configure().then_some(me)
        }
    }

    /// Fallback when the class is not registered: `DllGetClassObject`.
    unsafe fn from_dll() -> Option<*mut c_void> {
        type GetClassObject =
            unsafe extern "system" fn(*const Guid, *const Guid, *mut *mut c_void) -> i32;
        unsafe {
            let m = LoadLibraryW(wide("MSOpusDecoder.dll").as_ptr());
            if m.is_null() {
                return None;
            }
            let f = proc(m, b"DllGetClassObject\0");
            if f.is_null() {
                return None;
            }
            let f: GetClassObject = std::mem::transmute(f);
            let mut cf = std::ptr::null_mut();
            if f(&CLSID_MS_OPUS_DECODER, &IID_ICLASS_FACTORY, &mut cf) < 0 {
                return None;
            }
            let mut mft = std::ptr::null_mut();
            // IClassFactory::CreateInstance
            let hr = com!(
                cf,
                3,
                fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
                std::ptr::null_mut(),
                &IID_IMF_TRANSFORM,
                &mut mft
            );
            release(cf);
            (hr >= 0 && !mft.is_null()).then_some(mft)
        }
    }

    unsafe fn configure(&mut self) -> bool {
        unsafe {
            let ch = self.channels as u32;
            let rate = native_rate(self.out_rate);
            // Input: Opus, with an RFC 7845 OpusHead as the user data.
            let mut it = std::ptr::null_mut();
            if (self.api.create_media_type)(&mut it) < 0 {
                return false;
            }
            let mut head = Vec::from(*b"OpusHead");
            head.push(1);
            head.push(ch as u8);
            head.extend_from_slice(&0u16.to_le_bytes()); // pre-skip
            head.extend_from_slice(&48000u32.to_le_bytes());
            head.extend_from_slice(&0i16.to_le_bytes()); // gain
            head.push(0); // mapping family
            set_guid(it, &MF_MT_MAJOR_TYPE, &MF_MEDIA_TYPE_AUDIO);
            set_guid(it, &MF_MT_SUBTYPE, &MF_AUDIO_FORMAT_OPUS);
            set_u32(it, &MF_MT_AUDIO_NUM_CHANNELS, ch);
            set_u32(it, &MF_MT_AUDIO_SAMPLES_PER_SECOND, 48000);
            set_blob(it, &MF_MT_USER_DATA, &head);
            // IMFTransform::SetInputType (15)
            let hr = com!(self.mft, 15, fn(u32, *mut c_void, u32) -> i32, 0, it, 0);
            if std::env::var_os("STRATA_OPUS_DEBUG").is_some() {
                eprintln!("MF SetInputType hr={hr:#x}");
            }
            release(it);
            if hr < 0 {
                return false;
            }
            // Output: one of the decoder's own offered types, float
            // preferred, else 16-bit PCM (converted here).
            let debug = std::env::var_os("STRATA_OPUS_DEBUG").is_some();
            let mut chosen = None;
            for idx in 0..32u32 {
                let mut t = std::ptr::null_mut();
                // IMFTransform::GetOutputAvailableType (14)
                if com!(
                    self.mft,
                    14,
                    fn(u32, u32, *mut *mut c_void) -> i32,
                    0,
                    idx,
                    &mut t
                ) < 0
                {
                    break;
                }
                let mut sub = Guid(0, 0, 0, [0; 8]);
                // IMFAttributes::GetGUID (10) / GetUINT32 (7)
                com!(
                    t,
                    10,
                    fn(*const Guid, *mut Guid) -> i32,
                    &MF_MT_SUBTYPE,
                    &mut sub
                );
                let (mut ch_out, mut rate_out) = (0u32, 0u32);
                com!(
                    t,
                    7,
                    fn(*const Guid, *mut u32) -> i32,
                    &MF_MT_AUDIO_NUM_CHANNELS,
                    &mut ch_out
                );
                com!(
                    t,
                    7,
                    fn(*const Guid, *mut u32) -> i32,
                    &MF_MT_AUDIO_SAMPLES_PER_SECOND,
                    &mut rate_out
                );
                if debug {
                    eprintln!(
                        "MF output type {idx}: subtype {:#x} ch {ch_out} rate {rate_out}",
                        sub.0
                    );
                }
                let float = sub.0 == 3 && sub.1 == 0;
                let pcm = sub.0 == 1 && sub.1 == 0;
                let score = match (float, pcm) {
                    (true, _) => 2,
                    (_, true) => 1,
                    _ => 0,
                } + if ch_out == ch { 4 } else { 0 }
                    + if rate_out == rate { 8 } else { 0 };
                if (float || pcm)
                    && chosen
                        .as_ref()
                        .is_none_or(|c: &(u32, *mut c_void, bool, u32, u32)| score > c.0)
                {
                    if let Some((_, old, ..)) = chosen.take() {
                        release(old);
                    }
                    chosen = Some((score, t, float, ch_out.max(1), rate_out.max(8000)));
                } else {
                    release(t);
                }
            }
            let Some((_, ot, float, ch_out, rate_out)) = chosen else {
                return false;
            };
            let hr = com!(self.mft, 16, fn(u32, *mut c_void, u32) -> i32, 0, ot, 0);
            if debug {
                eprintln!("MF SetOutputType hr={hr:#x}");
            }
            release(ot);
            if hr < 0 {
                return false;
            }
            self.out_float = float;
            self.dec_channels = ch_out as usize;
            self.dec_rate = rate_out;
            let mut info = OutputStreamInfo::default();
            com!(
                self.mft,
                7,
                fn(u32, *mut OutputStreamInfo) -> i32,
                0,
                &mut info
            );
            self.provides_samples = info.flags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES != 0;
            self.out_size = info.size.max(5760 * 4 * ch);
            com!(
                self.mft,
                23,
                fn(u32, usize) -> i32,
                MFT_MESSAGE_NOTIFY_BEGIN_STREAMING,
                0
            );
            com!(
                self.mft,
                23,
                fn(u32, usize) -> i32,
                MFT_MESSAGE_NOTIFY_START_OF_STREAM,
                0
            );
            true
        }
    }

    unsafe fn sample_with(&self, bytes: &[u8], capacity: u32) -> Option<*mut c_void> {
        unsafe {
            let mut buf = std::ptr::null_mut();
            if (self.api.create_memory_buffer)(capacity.max(1), &mut buf) < 0 {
                return None;
            }
            if !bytes.is_empty() {
                let mut p: *mut u8 = std::ptr::null_mut();
                let (mut max, mut cur) = (0u32, 0u32);
                // IMFMediaBuffer::Lock (3) / Unlock (4) / SetCurrentLength (6)
                if com!(
                    buf,
                    3,
                    fn(*mut *mut u8, *mut u32, *mut u32) -> i32,
                    &mut p,
                    &mut max,
                    &mut cur
                ) < 0
                {
                    release(buf);
                    return None;
                }
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
                com!(buf, 4, fn() -> i32);
                com!(buf, 6, fn(u32) -> i32, bytes.len() as u32);
            }
            let mut s = std::ptr::null_mut();
            if (self.api.create_sample)(&mut s) < 0 {
                release(buf);
                return None;
            }
            // IMFSample::AddBuffer (42)
            com!(s, 42, fn(*mut c_void) -> i32, buf);
            release(buf);
            Some(s)
        }
    }

    /// Appends the samples of every buffer in `sample` to `out` as `f32`.
    unsafe fn read_sample(sample: *mut c_void, out: &mut Vec<f32>, float: bool) {
        unsafe {
            let mut buf = std::ptr::null_mut();
            // IMFSample::ConvertToContiguousBuffer (41)
            if com!(sample, 41, fn(*mut *mut c_void) -> i32, &mut buf) < 0 {
                return;
            }
            let mut p: *mut u8 = std::ptr::null_mut();
            let (mut max, mut cur) = (0u32, 0u32);
            if com!(
                buf,
                3,
                fn(*mut *mut u8, *mut u32, *mut u32) -> i32,
                &mut p,
                &mut max,
                &mut cur
            ) >= 0
            {
                if float {
                    let n = cur as usize / 4;
                    out.extend_from_slice(std::slice::from_raw_parts(p as *const f32, n));
                } else {
                    let n = cur as usize / 2;
                    let s = std::slice::from_raw_parts(p as *const i16, n);
                    out.extend(s.iter().map(|&v| f32::from(v) / 32768.0));
                }
                com!(buf, 4, fn() -> i32);
            }
            release(buf);
        }
    }
}

impl OpusBackend for MfOpusBackend {
    fn decode(&mut self, packet: &[u8]) -> Option<Vec<f32>> {
        // SAFETY: COM calls on the owned transform and samples made here.
        unsafe {
            let input = self.sample_with(packet, packet.len() as u32)?;
            // IMFSample::SetSampleTime (36)
            com!(input, 36, fn(i64) -> i32, self.time);
            // IMFTransform::ProcessInput (24)
            let hr = com!(self.mft, 24, fn(u32, *mut c_void, u32) -> i32, 0, input, 0);
            release(input);
            if hr < 0 {
                return None;
            }
            let mut pcm = Vec::new();
            loop {
                let own = if self.provides_samples {
                    std::ptr::null_mut()
                } else {
                    match self.sample_with(&[], self.out_size) {
                        Some(s) => s,
                        None => break,
                    }
                };
                let mut db = OutputDataBuffer {
                    stream_id: 0,
                    sample: own,
                    status: 0,
                    events: std::ptr::null_mut(),
                };
                let mut status = 0u32;
                // IMFTransform::ProcessOutput (25)
                let hr = com!(
                    self.mft,
                    25,
                    fn(u32, u32, *mut OutputDataBuffer, *mut u32) -> i32,
                    0,
                    1,
                    &mut db,
                    &mut status
                );
                if hr >= 0 && !db.sample.is_null() {
                    Self::read_sample(db.sample, &mut pcm, self.out_float);
                }
                if !db.sample.is_null() {
                    release(db.sample);
                } else {
                    release(own);
                }
                release(db.events);
                if hr == MF_E_TRANSFORM_STREAM_CHANGE {
                    if !self.configure() {
                        return None;
                    }
                    continue;
                }
                if hr < 0 {
                    // MF_E_TRANSFORM_NEED_MORE_INPUT ends a packet's output.
                    if hr != MF_E_TRANSFORM_NEED_MORE_INPUT && pcm.is_empty() {
                        return None;
                    }
                    break;
                }
            }
            // Match the requested channel count (mono <-> stereo).
            let dc = self.dec_channels.max(1);
            let channels = self.channels;
            let frames = pcm.len() / dc;
            let pcm = if dc == channels {
                pcm
            } else {
                (0..frames)
                    .flat_map(|f| {
                        let fr = &pcm[f * dc..f * dc + dc];
                        let mono = fr.iter().sum::<f32>() / dc as f32;
                        (0..channels).map(move |c| {
                            if channels == 1 {
                                mono
                            } else {
                                fr.get(c).copied().unwrap_or(mono)
                            }
                        })
                    })
                    .collect()
            };
            self.time += frames as i64 * 10_000_000 / i64::from(self.dec_rate);
            Some(self.resampler.process(pcm, self.dec_rate))
        }
    }

    fn name(&self) -> &'static str {
        "MSOpusDecoder"
    }
}

fn mf_factory(channels: usize, out_rate: u32) -> Option<Box<dyn OpusBackend>> {
    MfOpusBackend::new(channels, out_rate).map(|b| Box::new(b) as Box<dyn OpusBackend>)
}

/// Registers the optional host Opus factories globally for callers that
/// want `OpusDecoder::new` to prefer them.
pub fn register() {
    match std::env::var("STRATA_OPUS_BACKEND")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("none") | Some("native") => {}
        Some("mf") => register_backend(mf_factory),
        Some("libopus") => register_backend(libopus_factory),
        _ => {
            register_backend(libopus_factory);
            register_backend(mf_factory);
        }
    }
}

/// Creates a voice decoder, honoring the Windows backend override per voice.
pub(crate) fn decoder(channels: u16, rate: u32) -> OpusDecoder {
    let choice = std::env::var("STRATA_OPUS_BACKEND").unwrap_or_default();
    let name = match choice.trim() {
        "libopus" => "libopus",
        "mf" => "mf",
        "native" => "native",
        "none" => return OpusDecoder::without_backend(channels, rate),
        _ => return OpusDecoder::new(channels, rate),
    };
    if channels <= 2
        && let Some(backend) = backend(name, usize::from(channels.max(1)), rate.max(8_000))
    {
        return OpusDecoder::with_backend(channels, rate, backend);
    }
    OpusDecoder::without_backend(channels, rate)
}

/// A backend by name, for tests and tools (`"native"`, `"libopus"`, or `"mf"`).
pub fn backend(name: &str, channels: usize, rate: u32) -> Option<Box<dyn OpusBackend>> {
    match name {
        "native" => opus_native::backend(channels, rate),
        "libopus" => libopus_factory(channels, rate),
        "mf" => mf_factory(channels, rate),
        _ => None,
    }
}
