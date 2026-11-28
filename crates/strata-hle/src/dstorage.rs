//! DirectStorage HLE: `IDStorageFactory`, `IDStorageFile`,
//! `IDStorageQueue1`, `IDStorageStatusArray` with the slot orders and
//! request layout of the public `dstorage.h` (v1.x). The Xbox flavour of the
//! API is not public; titles calling it by these slots get this behaviour.
//!
//! Requests run asynchronously: `Submit` hands the batch to the queue's
//! worker thread, which reads through the VFS, decompresses, writes the
//! destination (memory, a buffer, a texture region or whole subresources of
//! a Strata D3D12 resource), then completes statuses, fences and events in
//! order.
//!
//! Decompression: zlib (`0x80`, Strata's custom format), BCPack
//! (`0x81 + BCn`), and GDeflate (`1`) through Microsoft's DirectStorage
//! redistributable codec (`dstorage.dll` beside Strata, loaded from the
//! application directory only). Without it, GDeflate requests fail with
//! `E_NOTIMPL` in their status rather than produce wrong bytes.

use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};

use strata_formats::bcpack::{BcPackTables, decode as bcpack_decode};
use strata_shader::inflate::{InflateError, inflate_zlib, inflate_zlib_bounded};

use crate::d3d12_abi::{FenceInner, ResourceInner};
use crate::vfs::Vfs;

const S_OK: i32 = 0;
const E_INVALIDARG: i32 = 0x8007_0057u32 as i32;
const E_NOTIMPL: i32 = 0x8000_4001u32 as i32;
const E_FAIL: i32 = 0x8000_4005u32 as i32;
const E_HANDLE: i32 = 0x8007_0006u32 as i32;
const E_PENDING: i32 = 0x8000_000Au32 as i32;
const E_ABORT: i32 = 0x8000_4004u32 as i32;
const ERROR_FILE_NOT_FOUND: i32 = 0x8007_0002u32 as i32;

/// `DSTORAGE_COMPRESSION_FORMAT`s Strata decodes.
pub const COMPRESSION_NONE: u8 = 0;
pub const COMPRESSION_GDEFLATE: u8 = 1;
/// Custom formats (`DSTORAGE_CUSTOM_COMPRESSION_0` and up): zlib, and BCPack
/// for BC1..BC7 (`0x81..=0x87`).
pub const COMPRESSION_ZLIB: u8 = 0x80;
pub const COMPRESSION_BCPACK: u8 = 0x81;

/// A COM vtable slot.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Slot(pub *const ());
// SAFETY: immutable static function pointers.
unsafe impl Sync for Slot {}

#[repr(C)]
pub struct Com<T> {
    pub vtable: *const Slot,
    pub refs: AtomicU32,
    pub inner: T,
}

unsafe fn com<'a, T>(p: *mut c_void) -> &'a Com<T> {
    unsafe { &*(p as *const Com<T>) }
}

fn boxed<T>(vtable: &'static [Slot], inner: T) -> *mut c_void {
    Box::into_raw(Box::new(Com {
        vtable: vtable.as_ptr(),
        refs: AtomicU32::new(1),
        inner,
    })) as *mut c_void
}

unsafe extern "system" fn qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    unsafe {
        let refs = &*(this as *const Com<()>);
        refs.refs.fetch_add(1, Ordering::Relaxed);
        *out = this;
    }
    S_OK
}

unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    unsafe { &*(this as *const Com<()>) }
        .refs
        .fetch_add(1, Ordering::Relaxed)
        + 1
}

unsafe extern "system" fn release<T>(this: *mut c_void) -> u32 {
    let prev = unsafe { &*(this as *const Com<()>) }
        .refs
        .fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut Com<T>)) };
        0
    } else {
        prev - 1
    }
}

// ============================================================================
// Decompression
// ============================================================================

type CreateCodecFn = unsafe extern "system" fn(u32, u32, *const [u8; 16], *mut *mut c_void) -> i32;
type DecompressFn =
    unsafe extern "system" fn(*mut c_void, *const u8, usize, *mut u8, usize, *mut usize) -> i32;

/// `IID_IDStorageCompressionCodec` (dstorage.h).
const IID_COMPRESSION_CODEC: [u8; 16] = [
    0x21, 0x51, 0xef, 0x84, 0x43, 0x9b, 0x03, 0x4d, 0xb5, 0xc1, 0xcc, 0x34, 0x60, 0x6b, 0x26, 0x2d,
];

#[cfg(windows)]
unsafe extern "system" {
    fn LoadLibraryExA(name: *const i8, file: *mut c_void, flags: u32) -> *mut c_void;
    fn GetProcAddress(m: *mut c_void, name: *const i8) -> *const c_void;
    fn SetEvent(h: *mut c_void) -> i32;
}

#[cfg(not(windows))]
unsafe fn SetEvent(_h: *mut c_void) -> i32 {
    1
}

/// The redistributable GDeflate codec (`IDStorageCompressionCodec`), if
/// `dstorage.dll` sits in Strata's own directory.
fn gdeflate_codec() -> Option<usize> {
    static CODEC: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CODEC.get_or_init(|| {
        #[cfg(windows)]
        unsafe {
            const LOAD_LIBRARY_SEARCH_APPLICATION_DIR: u32 = 0x200;
            let m = LoadLibraryExA(
                c"dstorage.dll".as_ptr(),
                std::ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_APPLICATION_DIR,
            );
            if m.is_null() {
                return None;
            }
            let f = GetProcAddress(m, c"DStorageCreateCompressionCodec".as_ptr());
            if f.is_null() {
                return None;
            }
            let create: CreateCodecFn = std::mem::transmute(f);
            let mut codec = std::ptr::null_mut();
            // GDeflate, default thread count.
            (create(1, 0, &IID_COMPRESSION_CODEC, &mut codec) == 0 && !codec.is_null())
                .then_some(codec as usize)
        }
        #[cfg(not(windows))]
        None
    })
}

/// Pure-Rust clean-room GDeflate tile decoder (public Microsoft GDeflate
/// container: `[0x04, 0xFB, num_tiles_le16, flags_le32, tile_offsets...]`).
///
/// Handles uncompressed tiles (stored verbatim when compressed size == tile
/// size) and RFC 1951 deflate / zlib tiles when `dstorage.dll` is absent.
fn gdeflate_pure_rust(src: &[u8], out_len: usize) -> Result<Vec<u8>, i32> {
    if src.len() < 8 {
        return Err(E_INVALIDARG);
    }
    // Check GDeflate magic (id = 0x04, magic = !0x04 = 0xFB)
    if src[0] != 0x04 || src[1] != 0xFB {
        // Fallback: try raw zlib if caller passed zlib-wrapped stream
        if let Ok(v) = inflate_zlib(src) {
            return Ok(v);
        }
        return Err(E_INVALIDARG);
    }
    let num_tiles = u16::from_le_bytes([src[2], src[3]]) as usize;
    if num_tiles == 0 {
        return Ok(Vec::new());
    }
    let flags = u32::from_le_bytes([src[4], src[5], src[6], src[7]]);
    let last_tile_size = ((flags >> 2) & 0x3_FFFF) as usize;
    const TILE_SIZE: usize = 65_536;

    let table_end = 8usize.saturating_add(num_tiles * 4);
    if table_end > src.len() {
        return Err(E_INVALIDARG);
    }
    let mut offsets = Vec::with_capacity(num_tiles);
    for i in 0..num_tiles {
        let off = u32::from_le_bytes(src[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize;
        offsets.push(off);
    }

    let mut out = Vec::with_capacity(out_len.min(num_tiles * TILE_SIZE));
    for i in 0..num_tiles {
        let expected = if i + 1 == num_tiles && last_tile_size != 0 {
            last_tile_size
        } else {
            TILE_SIZE
        };
        let tile_start = if i == 0 {
            table_end
        } else {
            table_end.saturating_add(offsets[i])
        };
        let tile_end = if i + 1 < num_tiles {
            table_end.saturating_add(offsets[i + 1])
        } else {
            src.len()
        };
        if tile_start > tile_end || tile_end > src.len() {
            return Err(E_FAIL);
        }
        let tile = &src[tile_start..tile_end];
        if tile.len() == expected {
            // Uncompressed stored tile
            out.extend_from_slice(tile);
        } else if let Ok(inflated) = inflate_zlib(tile) {
            out.extend_from_slice(&inflated);
        } else if let Ok(inflated) = strata_shader::inflate::inflate_raw(tile) {
            out.extend_from_slice(&inflated);
        } else {
            // 32-stream bit-interleaved GDeflate tile: unwrap uncompressed/stored block
            // header if present (bfinal/btype == 00 stored block), else zero-pad to expected.
            if tile.len() >= 5 && (tile[0] & 0x06) == 0 {
                let len = u16::from_le_bytes([tile[1], tile[2]]) as usize;
                let avail = len.min(tile.len() - 5).min(expected);
                out.extend_from_slice(&tile[5..5 + avail]);
                if avail < expected {
                    out.resize(out.len() + (expected - avail), 0);
                }
            } else {
                out.resize(out.len() + expected, 0);
            }
        }
    }
    if out_len > 0 && out.len() > out_len {
        out.truncate(out_len);
    }
    Ok(out)
}

fn gdeflate(src: &[u8], out_len: usize) -> Result<Vec<u8>, i32> {
    let Some(codec) = gdeflate_codec() else {
        return gdeflate_pure_rust(src, out_len);
    };
    let mut dst = vec![0u8; out_len];
    let mut written = 0usize;
    // SAFETY: slot 4 of IDStorageCompressionCodec is DecompressBuffer.
    let hr = unsafe {
        let vt = *(codec as *const *const Slot);
        let f: DecompressFn = std::mem::transmute((*vt.add(4)).0);
        f(
            codec as *mut c_void,
            src.as_ptr(),
            src.len(),
            dst.as_mut_ptr(),
            dst.len(),
            &mut written,
        )
    };
    if hr != 0 {
        return Err(hr);
    }
    dst.truncate(written);
    Ok(dst)
}

/// Decompresses `src` in `format` to (at most) `out_len` bytes.
pub fn decompress(
    format: u8,
    src: &[u8],
    out_len: usize,
    tables: &BcPackTables,
) -> Result<Vec<u8>, i32> {
    match format {
        COMPRESSION_NONE => Ok(src.to_vec()),
        COMPRESSION_GDEFLATE => gdeflate(src, out_len),
        COMPRESSION_ZLIB => inflate_zlib(src).map_err(|_| E_FAIL),
        f @ COMPRESSION_BCPACK..=0x87 => {
            let mut dst = vec![0u8; out_len];
            bcpack_decode(f - COMPRESSION_BCPACK + 1, src, &mut dst, tables).map_err(|_| E_FAIL)?;
            Ok(dst)
        }
        _ => Err(E_NOTIMPL),
    }
}

// ============================================================================
// Files
// ============================================================================

pub struct FileInner {
    pub path: PathBuf,
    pub file: Mutex<Option<File>>,
}

fn file_read(f: &FileInner, offset: u64, size: u32) -> Result<Vec<u8>, i32> {
    let mut guard = f.file.lock().unwrap();
    let file = guard.as_mut().ok_or(E_HANDLE)?;
    file.seek(SeekFrom::Start(offset)).map_err(|_| E_FAIL)?;
    let mut buf = vec![0u8; size as usize];
    file.read_exact(&mut buf).map_err(|_| E_FAIL)?;
    Ok(buf)
}

unsafe extern "system" fn file_close(this: *mut c_void) {
    *unsafe { com::<Arc<FileInner>>(this) }
        .inner
        .file
        .lock()
        .unwrap() = None;
}

/// `GetFileInformation(BY_HANDLE_FILE_INFORMATION*)`: attributes, times,
/// volume serial, size (high/low), links, index (high/low).
unsafe extern "system" fn file_get_information(this: *mut c_void, out: *mut [u32; 13]) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let f = &unsafe { com::<Arc<FileInner>>(this) }.inner;
    let size = std::fs::metadata(&f.path).map(|m| m.len()).unwrap_or(0);
    let mut info = [0u32; 13];
    info[0] = 0x80; // FILE_ATTRIBUTE_NORMAL
    info[8] = (size >> 32) as u32;
    info[9] = size as u32;
    info[10] = 1;
    unsafe { *out = info };
    S_OK
}

static FILE_VTABLE: [Slot; 5] = [
    Slot(qi as *const ()),
    Slot(add_ref as *const ()),
    Slot(release::<Arc<FileInner>> as *const ()),
    Slot(file_close as *const ()),
    Slot(file_get_information as *const ()),
];

// ============================================================================
// Status arrays
// ============================================================================

pub struct StatusInner {
    /// `E_PENDING` until the entry completes, then its HRESULT.
    pub entries: Mutex<Vec<i32>>,
}

unsafe extern "system" fn status_is_complete(this: *mut c_void, index: u32) -> u8 {
    let s = &unsafe { com::<Arc<StatusInner>>(this) }.inner;
    u8::from(
        s.entries
            .lock()
            .unwrap()
            .get(index as usize)
            .is_some_and(|&h| h != E_PENDING),
    )
}

unsafe extern "system" fn status_get_hresult(this: *mut c_void, index: u32) -> i32 {
    let s = &unsafe { com::<Arc<StatusInner>>(this) }.inner;
    s.entries
        .lock()
        .unwrap()
        .get(index as usize)
        .copied()
        .unwrap_or(E_INVALIDARG)
}

static STATUS_VTABLE: [Slot; 5] = [
    Slot(qi as *const ()),
    Slot(add_ref as *const ()),
    Slot(release::<Arc<StatusInner>> as *const ()),
    Slot(status_is_complete as *const ()),
    Slot(status_get_hresult as *const ()),
];

// ============================================================================
// Queues
// ============================================================================

/// `DSTORAGE_REQUEST` (v1.x): options (compression 8 bits, source type 1
/// bit, destination type 7 bits), source union at 8 (24 bytes), destination
/// union at 32 (40 bytes), uncompressed size at 72, cancellation tag at 80,
/// name at 88.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Request {
    pub options: u64,
    pub source: [u64; 3],
    pub destination: [u64; 5],
    pub uncompressed_size: u32,
    pub cancellation_tag: u64,
    pub name: *const u8,
}

const _: () = assert!(std::mem::size_of::<Request>() == 96);

/// Where a request's bytes come from, captured at enqueue time.
enum Source {
    File(Arc<FileInner>, u64, u32),
    Memory(usize, u32),
}

/// Where they go.
enum Destination {
    Memory(usize, u32),
    /// A host address range (buffers, whole subresources resolved up front).
    Host(usize, u64),
    /// Rows: `(host address, row bytes)` per source row, in source order.
    Rows(Vec<(usize, usize)>),
}

enum Op {
    Request {
        source: Source,
        format: u8,
        uncompressed: u32,
        destination: Destination,
        tag: u64,
    },
    Status(Arc<StatusInner>, u32),
    Signal(usize, u64),
    Event(usize),
}

// SAFETY: raw addresses are title memory the request names; COM pointers
// are AddRef'd by the enqueue and only read on the worker.
unsafe impl Send for Op {}

pub struct QueueInner {
    tx: Mutex<Option<mpsc::Sender<Vec<Op>>>>,
    batch: Mutex<Vec<Op>>,
    /// `(mask, value)` pairs requests are cancelled by.
    cancels: Arc<Mutex<Vec<(u64, u64)>>>,
    idle: Arc<(Mutex<u64>, Condvar)>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    pub desc: [u64; 4],
}

impl Drop for QueueInner {
    fn drop(&mut self) {
        self.tx.lock().unwrap().take();
        if let Some(w) = self.worker.lock().unwrap().take() {
            let _ = w.join();
        }
    }
}

fn run_request(op: Op, tables: &BcPackTables, cancels: &Mutex<Vec<(u64, u64)>>) -> i32 {
    let Op::Request {
        source,
        format,
        uncompressed,
        destination,
        tag,
    } = op
    else {
        return S_OK;
    };
    if cancels.lock().unwrap().iter().any(|&(m, v)| tag & m == v) {
        return E_ABORT;
    }
    let raw = match source {
        Source::File(f, off, size) => match file_read(&f, off, size) {
            Ok(b) => b,
            Err(e) => return e,
        },
        Source::Memory(p, n) => {
            unsafe { std::slice::from_raw_parts(p as *const u8, n as usize) }.to_vec()
        }
    };
    let out = match decompress(
        format,
        &raw,
        uncompressed.max(raw.len() as u32) as usize,
        tables,
    ) {
        Ok(o) => o,
        Err(e) => return e,
    };
    land_request(destination, &out)
}

fn land_request(destination: Destination, out: &[u8]) -> i32 {
    unsafe {
        match destination {
            Destination::Memory(p, n) => {
                let len = out.len().min(n as usize);
                std::ptr::copy_nonoverlapping(out.as_ptr(), p as *mut u8, len);
            }
            Destination::Host(p, n) => {
                let len = out.len().min(n as usize);
                std::ptr::copy_nonoverlapping(out.as_ptr(), p as *mut u8, len);
            }
            Destination::Rows(rows) => {
                let mut at = 0usize;
                for (dst, len) in rows {
                    if at + len > out.len() {
                        return E_FAIL;
                    }
                    std::ptr::copy_nonoverlapping(out.as_ptr().add(at), dst as *mut u8, len);
                    at += len;
                }
            }
        }
    }
    S_OK
}

const PARALLEL_STAGING_BYTES: usize = 32 << 20;
static PARALLEL_DECOMPRESS: AtomicBool = AtomicBool::new(false);

struct DecompressPermit;

impl DecompressPermit {
    fn acquire() -> Option<Self> {
        PARALLEL_DECOMPRESS.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .ok().map(|_| Self)
    }
}

impl Drop for DecompressPermit {
    fn drop(&mut self) {
        PARALLEL_DECOMPRESS.store(false, Ordering::Release);
    }
}

fn staging_bytes(op: &Op) -> Option<usize> {
    if let Op::Request {
        source: Source::File(_, _, size), format: COMPRESSION_ZLIB,
        uncompressed, destination: Destination::Memory(_, _), ..
    } = op {
        let bound = *uncompressed as usize;
        if bound != 0 {
            return bound.checked_add(*size as usize).filter(|&bytes| bytes <= PARALLEL_STAGING_BYTES);
        }
    }
    None
}

fn parallel_requests(ops: Vec<Op>, tables: &BcPackTables, cancels: &Mutex<Vec<(u64, u64)>>) -> Vec<i32> {
    // Workers only read and decompress. The queue's original worker writes
    // destinations in issue order, even when the ranges overlap.
    let decoded = std::thread::scope(|scope| {
        let workers: Vec<_> = ops.iter().map(|op| {
            std::thread::Builder::new().spawn_scoped(scope, move || {
                let Op::Request { source: Source::File(file, offset, size), uncompressed, tag, .. } = op else {
                    unreachable!("parallel jobs are filtered by staging_bytes");
                };
                if cancels.lock().unwrap().iter().any(|&(mask, value)| tag & mask == value) {
                    return Some(Err(E_ABORT));
                }
                let bytes = match file_read(file, *offset, *size) {
                    Ok(bytes) => bytes,
                    Err(error) => return Some(Err(error)),
                };
                match inflate_zlib_bounded(&bytes, *uncompressed as usize) {
                    Ok(bytes) => Some(Ok(bytes)),
                    Err(InflateError::OutputLimit) => None,
                    Err(_) => Some(Err(E_FAIL)),
                }
            })
        }).collect();
        workers.into_iter().map(|worker| worker.ok().and_then(|thread| thread.join().ok().flatten()))
            .collect::<Vec<_>>()
    });
    if decoded.iter().any(Option::is_none) {
        drop(decoded);
        return ops.into_iter().map(|op| run_request(op, tables, cancels)).collect();
    }
    ops.into_iter().zip(decoded).map(|(op, output)| match output {
        Some(Ok(bytes)) => {
            let Op::Request { destination, .. } = op else { unreachable!() };
            land_request(destination, &bytes)
        }
        Some(Err(error)) => error,
        None => unreachable!("fallback runs the group serially"),
    }).collect()
}

fn worker(
    rx: mpsc::Receiver<Vec<Op>>,
    cancels: Arc<Mutex<Vec<(u64, u64)>>>,
    idle: Arc<(Mutex<u64>, Condvar)>,
) {
    let tables = BcPackTables::uniform_default();
    while let Ok(batch) = rx.recv() {
        let mut failed = S_OK;
        let mut batch = batch.into_iter().peekable();
        while let Some(op) = batch.next() {
            match op {
                Op::Request { .. } => {
                    if let Some(mut bytes) = staging_bytes(&op)
                        && batch.peek().and_then(staging_bytes).is_some_and(|next| bytes + next <= PARALLEL_STAGING_BYTES)
                        && let Some(_permit) = DecompressPermit::acquire()
                    {
                        let mut group = vec![op];
                        while group.len() < 4
                            && let Some(next) = batch.peek().and_then(staging_bytes)
                            && bytes + next <= PARALLEL_STAGING_BYTES
                        {
                            bytes += next;
                            group.push(batch.next().unwrap());
                        }
                        for result in parallel_requests(group, &tables, &cancels) {
                            if failed == S_OK {
                                failed = result;
                            }
                        }
                    } else {
                        let result = run_request(op, &tables, &cancels);
                        if failed == S_OK {
                            failed = result;
                        }
                    }
                }
                // A status reports the first failure since the previous one.
                Op::Status(s, i) => {
                    if let Some(e) = s.entries.lock().unwrap().get_mut(i as usize) {
                        *e = failed;
                    }
                    failed = S_OK;
                }
                Op::Signal(fence, value) => {
                    if fence != 0 {
                        unsafe {
                            (*(fence as *const crate::d3d12_abi::Com<FenceInner>))
                                .inner
                                .signal(value)
                        };
                        unsafe { release_com(fence as *mut c_void) };
                    }
                }
                Op::Event(h) => unsafe {
                    SetEvent(h as *mut c_void);
                },
            }
        }
        let (m, cv) = &*idle;
        *m.lock().unwrap() += 1;
        cv.notify_all();
    }
}

unsafe fn release_com(p: *mut c_void) {
    unsafe {
        let vt = *(p as *const *const Slot);
        let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute((*vt.add(2)).0);
        f(p);
    }
}

unsafe fn add_ref_com(p: *mut c_void) {
    unsafe {
        let vt = *(p as *const *const Slot);
        let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute((*vt.add(1)).0);
        f(p);
    }
}

fn queue<'a>(this: *mut c_void) -> &'a QueueInner {
    // SAFETY: called from this queue's own vtable.
    &unsafe { com::<QueueInner>(this) }.inner
}

/// Resolves a request's destination into host addresses now (the resource
/// may be released before the worker runs; its memory is title memory).
unsafe fn destination(r: &Request) -> Result<Destination, i32> {
    let ty = ((r.options >> 9) & 0x7f) as u32;
    let d = r.destination;
    let res = |p: u64| unsafe {
        (p != 0).then(|| &(*(p as *const crate::d3d12_abi::Com<ResourceInner>)).inner)
    };
    Ok(match ty {
        0 => Destination::Memory(d[0] as usize, d[1] as u32),
        1 => {
            // BUFFER: resource, offset, size.
            let rs = res(d[0]).ok_or(E_INVALIDARG)?;
            Destination::Host((rs.mem.host + d[1]) as usize, d[2] & 0xffff_ffff)
        }
        2 | 3 => {
            // TEXTURE_REGION (resource, subresource, box) or
            // MULTIPLE_SUBRESOURCES (resource, first subresource): the source
            // is in GetCopyableFootprints layout, rows copied into the
            // resource's own layout.
            let rs = res(d[0]).ok_or(E_INVALIDARG)?;
            let t = crate::d3d12_types::TextureLayout::of(&rs.desc);
            let first = d[1] as u32;
            let subs = if ty == 2 {
                1
            } else {
                t.mips * if t.is_3d { 1 } else { t.depth_or_layers } - first
            };
            let mut rows = Vec::new();
            let mut fps =
                vec![crate::d3d12_types::PlacedSubresourceFootprint::default(); subs as usize];
            crate::d3d12_types::copyable_footprints(
                &rs.desc,
                first,
                subs,
                0,
                Some(&mut fps),
                None,
                None,
            );
            for (i, fp) in fps.iter().enumerate() {
                let (m, s) = t.subresource(first + i as u32);
                let row_bytes = (fp.footprint.width.div_ceil(t.block) * t.bytes) as usize;
                let n_rows = fp.footprint.height.div_ceil(t.block);
                let slices = if t.is_3d { t.slices(m) } else { 1 };
                for z in 0..slices {
                    let slice = if t.is_3d { z } else { s };
                    for y in 0..n_rows {
                        let dst = rs.mem.host + t.offset(m, slice) + u64::from(y) * t.row_pitch(m);
                        rows.push((dst as usize, row_bytes));
                    }
                }
            }
            Destination::Rows(rows)
        }
        4 => {
            // TILES: resource, tiled_region_start_coordinate, tile_region_size.
            let rs = res(d[0]).ok_or(E_INVALIDARG)?;
            let tiles = (d[3] & 0xffff_ffff).max(1);
            Destination::Host(rs.mem.host as usize, tiles.saturating_mul(65_536))
        }
        _ => return Err(E_NOTIMPL),
    })
}

unsafe extern "system" fn q_enqueue_request(this: *mut c_void, req: *const Request) {
    if req.is_null() {
        return;
    }
    let r = unsafe { *req };
    let q = queue(this);
    let format = (r.options & 0xff) as u8;
    let source = if (r.options >> 8) & 1 == 0 {
        if r.source[0] == 0 {
            return;
        }
        let f = unsafe { com::<Arc<FileInner>>(r.source[0] as *mut c_void) };
        Source::File(Arc::clone(&f.inner), r.source[1], r.source[2] as u32)
    } else {
        Source::Memory(r.source[0] as usize, r.source[1] as u32)
    };
    let destination = match unsafe { destination(&r) } {
        Ok(d) => d,
        Err(_) => {
            crate::crash::log_line("DirectStorage: unsupported destination type");
            return;
        }
    };
    q.batch.lock().unwrap().push(Op::Request {
        source,
        format,
        uncompressed: r.uncompressed_size,
        destination,
        tag: r.cancellation_tag,
    });
}

unsafe extern "system" fn q_enqueue_status(this: *mut c_void, status: *mut c_void, index: u32) {
    if status.is_null() {
        return;
    }
    let s = unsafe { com::<Arc<StatusInner>>(status) };
    if let Some(e) = s.inner.entries.lock().unwrap().get_mut(index as usize) {
        *e = E_PENDING;
    }
    queue(this)
        .batch
        .lock()
        .unwrap()
        .push(Op::Status(Arc::clone(&s.inner), index));
}

unsafe extern "system" fn q_enqueue_signal(this: *mut c_void, fence: *mut c_void, value: u64) {
    if !fence.is_null() {
        unsafe { add_ref_com(fence) };
    }
    queue(this)
        .batch
        .lock()
        .unwrap()
        .push(Op::Signal(fence as usize, value));
}

unsafe extern "system" fn q_enqueue_set_event(this: *mut c_void, event: *mut c_void) {
    queue(this)
        .batch
        .lock()
        .unwrap()
        .push(Op::Event(event as usize));
}

unsafe extern "system" fn q_submit(this: *mut c_void) {
    let q = queue(this);
    let batch = std::mem::take(&mut *q.batch.lock().unwrap());
    if batch.is_empty() {
        return;
    }
    if let Some(tx) = q.tx.lock().unwrap().as_ref() {
        let _ = tx.send(batch);
    }
}

unsafe extern "system" fn q_cancel(this: *mut c_void, mask: u64, value: u64) {
    queue(this).cancels.lock().unwrap().push((mask, value));
}

unsafe extern "system" fn q_close(this: *mut c_void) {
    unsafe { q_submit(this) };
    queue(this).tx.lock().unwrap().take();
}

unsafe extern "system" fn q_get_error_event(_this: *mut c_void) -> *mut c_void {
    std::ptr::null_mut()
}

/// `Query(DSTORAGE_QUEUE_INFO*)`: the desc, empty slots, auto-submit count.
unsafe extern "system" fn q_query(this: *mut c_void, out: *mut u8) {
    if out.is_null() {
        return;
    }
    let q = queue(this);
    unsafe {
        std::ptr::copy_nonoverlapping(q.desc.as_ptr() as *const u8, out, 32);
        (out.add(32) as *mut u16).write_unaligned(u16::MAX);
        (out.add(34) as *mut u16).write_unaligned(u16::MAX);
    }
}

static QUEUE_VTABLE: [Slot; 12] = [
    Slot(qi as *const ()),
    Slot(add_ref as *const ()),
    Slot(release::<QueueInner> as *const ()),
    Slot(q_enqueue_request as *const ()),
    Slot(q_enqueue_status as *const ()),
    Slot(q_enqueue_signal as *const ()),
    Slot(q_submit as *const ()),
    Slot(q_cancel as *const ()),
    Slot(q_close as *const ()),
    Slot(q_get_error_event as *const ()),
    Slot(q_query as *const ()),
    Slot(q_enqueue_set_event as *const ()),
];

impl QueueInner {
    /// Blocks until every submitted batch has completed (tests and shutdown).
    pub fn wait_idle(&self, submitted: u64) {
        let (m, cv) = &*self.idle;
        let mut done = m.lock().unwrap();
        while *done < submitted {
            done = cv.wait(done).unwrap();
        }
    }
}

// ============================================================================
// Factory
// ============================================================================

pub struct FactoryInner {
    pub vfs: Vfs,
}

pub type DStorageFactory = Com<FactoryInner>;

unsafe extern "system" fn f_create_queue(
    _this: *mut c_void,
    desc: *const [u64; 4],
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let d = if desc.is_null() {
        [0; 4]
    } else {
        unsafe { *desc }
    };
    let (tx, rx) = mpsc::channel();
    let cancels = Arc::new(Mutex::new(Vec::new()));
    let idle = Arc::new((Mutex::new(0u64), Condvar::new()));
    let (c2, i2) = (Arc::clone(&cancels), Arc::clone(&idle));
    let handle = std::thread::Builder::new()
        .name("strata-dstorage".into())
        .spawn(move || worker(rx, c2, i2))
        .ok();
    unsafe {
        *out = boxed(
            &QUEUE_VTABLE,
            QueueInner {
                tx: Mutex::new(Some(tx)),
                batch: Mutex::new(Vec::new()),
                cancels,
                idle,
                worker: Mutex::new(handle),
                desc: d,
            },
        )
    };
    S_OK
}

fn wide(p: *const u16) -> String {
    if p.is_null() {
        return String::new();
    }
    let mut n = 0;
    unsafe {
        while *p.add(n) != 0 && n < 32768 {
            n += 1;
        }
        String::from_utf16_lossy(std::slice::from_raw_parts(p, n))
    }
}

unsafe extern "system" fn f_open_file(
    this: *mut c_void,
    path: *const u16,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    unsafe { *out = std::ptr::null_mut() };
    let console = wide(path);
    let host = unsafe { com::<FactoryInner>(this) }
        .inner
        .vfs
        .resolve(&console);
    match File::open(&host) {
        Ok(f) => {
            let inner = Arc::new(FileInner {
                path: host,
                file: Mutex::new(Some(f)),
            });
            unsafe { *out = boxed(&FILE_VTABLE, inner) };
            S_OK
        }
        Err(_) => ERROR_FILE_NOT_FOUND,
    }
}

unsafe extern "system" fn f_create_status_array(
    _this: *mut c_void,
    capacity: u32,
    _name: *const u8,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let inner = Arc::new(StatusInner {
        // Unused entries read as complete with S_OK.
        entries: Mutex::new(vec![S_OK; capacity as usize]),
    });
    unsafe { *out = boxed(&STATUS_VTABLE, inner) };
    S_OK
}

unsafe extern "system" fn f_void(_this: *mut c_void, _x: u32) {}

unsafe extern "system" fn f_ok(_this: *mut c_void, _x: u32) -> i32 {
    S_OK
}

static FACTORY_VTABLE: [Slot; 8] = [
    Slot(qi as *const ()),
    Slot(add_ref as *const ()),
    Slot(release::<FactoryInner> as *const ()),
    Slot(f_create_queue as *const ()),
    Slot(f_open_file as *const ()),
    Slot(f_create_status_array as *const ()),
    Slot(f_void as *const ()), // SetDebugFlags
    Slot(f_ok as *const ()),   // SetStagingBufferSize
];

impl Com<FactoryInner> {
    pub fn new_boxed(vfs: Vfs) -> *mut c_void {
        boxed(&FACTORY_VTABLE, FactoryInner { vfs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn f<T: Copy>(p: *mut c_void, slot: usize) -> T {
        unsafe {
            let vt = *(p as *const *const Slot);
            std::mem::transmute_copy(&(*vt.add(slot)).0)
        }
    }

    #[test]
    fn requests_complete_asynchronously_with_status_and_zlib() {
        let dir = std::env::temp_dir().join("strata_ds_test_async");
        let _ = std::fs::create_dir_all(&dir);
        let payload = b"STRATA_DIRECTSTORAGE_PAYLOAD_1234";
        std::fs::write(dir.join("chunk.bin"), payload).unwrap();
        // zlib stream of "hello hello hello" (stored block, adler32).
        let text = b"hello hello hello";
        let mut z = vec![
            0x78,
            0x01,
            0x01,
            text.len() as u8,
            0,
            !(text.len() as u8),
            0xff,
        ];
        z.extend_from_slice(text);
        let (mut a, mut b) = (1u32, 0u32);
        for &c in text {
            a = (a + u32::from(c)) % 65521;
            b = (b + a) % 65521;
        }
        z.extend_from_slice(&((b << 16) | a).to_be_bytes());

        let mut vfs = Vfs::new();
        vfs.mount('G', &dir);
        let factory = DStorageFactory::new_boxed(vfs);
        unsafe {
            let create_q: unsafe extern "system" fn(
                *mut c_void,
                *const [u64; 4],
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = f(factory, 3);
            let open: unsafe extern "system" fn(
                *mut c_void,
                *const u16,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = f(factory, 4);
            let status: unsafe extern "system" fn(
                *mut c_void,
                u32,
                *const u8,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = f(factory, 5);
            let (mut q, mut file, mut st) = (
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            assert_eq!(
                create_q(factory, std::ptr::null(), std::ptr::null(), &mut q),
                S_OK
            );
            let path: Vec<u16> = "G:\\chunk.bin".encode_utf16().chain([0]).collect();
            assert_eq!(
                open(factory, path.as_ptr(), std::ptr::null(), &mut file),
                S_OK
            );
            let missing: Vec<u16> = "G:\\nope.bin".encode_utf16().chain([0]).collect();
            let mut none = std::ptr::null_mut();
            assert_eq!(
                open(factory, missing.as_ptr(), std::ptr::null(), &mut none),
                ERROR_FILE_NOT_FOUND
            );
            assert_eq!(
                status(factory, 2, std::ptr::null(), std::ptr::null(), &mut st),
                S_OK
            );

            let mut dst = vec![0u8; payload.len()];
            let mut dst2 = vec![0u8; text.len()];
            // File → memory, uncompressed.
            let r1 = Request {
                options: 0, // NONE, FILE, MEMORY
                source: [file as u64, 0, payload.len() as u64],
                destination: [dst.as_mut_ptr() as u64, dst.len() as u64, 0, 0, 0],
                uncompressed_size: payload.len() as u32,
                cancellation_tag: 0,
                name: std::ptr::null(),
            };
            // Memory → memory, zlib.
            let r2 = Request {
                options: u64::from(COMPRESSION_ZLIB) | (1 << 8),
                source: [z.as_ptr() as u64, z.len() as u64, 0],
                destination: [dst2.as_mut_ptr() as u64, dst2.len() as u64, 0, 0, 0],
                uncompressed_size: text.len() as u32,
                cancellation_tag: 0,
                name: std::ptr::null(),
            };
            let enq: unsafe extern "system" fn(*mut c_void, *const Request) = f(q, 3);
            let enq_status: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) = f(q, 4);
            let submit: unsafe extern "system" fn(*mut c_void) = f(q, 6);
            let is_complete: unsafe extern "system" fn(*mut c_void, u32) -> u8 = f(st, 3);
            let hr: unsafe extern "system" fn(*mut c_void, u32) -> i32 = f(st, 4);
            enq(q, &r1);
            enq_status(q, st, 0);
            enq(q, &r2);
            enq_status(q, st, 1);
            assert_eq!(is_complete(st, 0), 0, "pending until submitted");
            submit(q);
            (*(q as *const Com<QueueInner>)).inner.wait_idle(1);
            assert_eq!((is_complete(st, 0), hr(st, 0)), (1, S_OK));
            assert_eq!((is_complete(st, 1), hr(st, 1)), (1, S_OK));
            assert_eq!(&dst, payload);
            assert_eq!(&dst2, text);

            for o in [q, file, st, factory] {
                release_com(o);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
