//! PE32+ title loader: Windows maps the image, Strata binds and starts it.
//!
//! The title's bytes are patched into a DLL that the Windows loader maps but
//! does not run ([`prepare_image`]), written once per build to a cache file,
//! and loaded with `LoadLibraryExW`. Mapping through the OS loader (rather
//! than copying sections into `VirtualAlloc` memory) is what makes the image
//! a real module:
//!
//! - its `.pdata` is registered, so C++ `throw`, SEH and stack unwinding
//!   work, and `RtlPcToFileHeader` (which `_CxxThrowException` relies on)
//!   finds its base;
//! - its static TLS block exists on every thread, including ones created
//!   later ([`crate::tls`] handles the index-0 assumption of exe code);
//! - relocations and section protections are applied by the OS.
//!
//! Strata then binds the original import table itself: console DLLs to HLE
//! thunks, host DLLs to the host, and anything unresolved to a logged stub.

use std::ffi::{CString, c_void};
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use thiserror::Error;

use crate::pe::{Pe64Info, PeError, parse_pe64, rva_to_offset, u16_at, u32_at, u64_at};
use crate::runtime::{DllClass, classify_dll};
use crate::thunks::{logged_unimplemented_stub, resolve_hle};
use crate::tls::{TLS_DIR_CALLBACKS_OFFSET, TlsSetup};
use crate::veh::VehGuard;

#[cfg(windows)]
unsafe extern "system" {
    fn LoadLibraryA(name: *const i8) -> *mut c_void;
    fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn GetModuleHandleW(name: *const u16) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *const c_void;
    fn GetLastError() -> u32;
    fn VirtualProtect(
        addr: *mut c_void,
        size: usize,
        new_protect: u32,
        old_protect: *mut u32,
    ) -> i32;
}

const PAGE_EXECUTE_READWRITE: u32 = 0x40;
pub const DLL_PROCESS_DETACH: u32 = 0;
pub const DLL_PROCESS_ATTACH: u32 = 1;
pub const DLL_THREAD_ATTACH: u32 = 2;
pub const DLL_THREAD_DETACH: u32 = 3;

const FILE_CHARACTERISTICS: usize = 22;
const FILE_DLL: u16 = 0x2000;
const OPT_ENTRY_POINT: usize = 16;
const OPT_CHECKSUM: usize = 64;
const OPT_DLL_CHARACTERISTICS: usize = 70;
const OPT_NUM_DIRS: usize = 108;
const OPT_DIRS: usize = 112;
const DLLCHAR_FORCE_INTEGRITY: u16 = 0x0080;
const DIR_IMPORT: usize = 1;
const DIR_SECURITY: usize = 4;
const DIR_BOUND_IMPORT: usize = 11;
const REL_BASED_DIR64: u16 = 10;
const IMPORT_DESCRIPTOR_LEN: usize = 20;

#[derive(Debug, Error)]
pub enum MapError {
    #[error(transparent)]
    Pe(#[from] PeError),
    #[error("cannot prepare image: {0}")]
    Prepare(&'static str),
    #[error("cannot write prepared image {}: {source}", path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{} is already loaded in this process", path.display())]
    AlreadyLoaded { path: PathBuf },
    #[error("Windows loader rejected {} (error {code})", path.display())]
    LoadFailed { path: PathBuf, code: u32 },
    #[error("static TLS setup failed: {0}")]
    Tls(&'static str),
    #[error("entry point execution failed: {0}")]
    ExecFailed(String),
}

#[derive(Debug, Default)]
pub struct BindReport {
    pub resolved_hle: usize,
    pub resolved_host: usize,
    /// Bound to an export of a DLL the title ships.
    pub resolved_title: usize,
    pub deferred_title_local: usize,
    pub unresolved: Vec<String>,
}

/// A PE32+ title mapped by the Windows loader, with its IAT bound to Strata
/// HLE and the host.
pub struct MappedImage {
    pub module: *mut c_void,
    pub base: *mut u8,
    pub size: usize,
    pub entry_point: usize,
    /// DIR64 relocations the OS applied (0 when mapped at its preferred base).
    pub relocations_applied: usize,
    pub bind_report: BindReport,
    pub tls: Option<TlsSetup>,
    /// The patched copy handed to the loader.
    pub path: PathBuf,
    /// `IMAGE_FILE_DLL`: `entry_point` is `DllMain`.
    pub is_dll: bool,
    pub exports: Vec<crate::pe::ExportFunc>,
    /// The Windows loader tells this image of every thread (entry thunk in place).
    pub thread_notices: bool,
    attached: std::sync::Arc<AtomicBool>,
}

// SAFETY: the pointers name process-global memory (the mapped image); the
// value carries no thread affinity.
unsafe impl Send for MappedImage {}
unsafe impl Sync for MappedImage {}

impl MappedImage {
    /// Prepare `bytes`, map them with the Windows loader, bind the IAT and
    /// set up static TLS. Nothing in the title runs yet. Title-local DLLs
    /// are not looked for (see [`crate::title::load_title`]).
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, MapError> {
        Self::load_with(bytes, &mut HostResolver)
    }

    /// Address of an export (`name` or `ordinal#N`).
    pub fn export(&self, func: &str) -> Option<usize> {
        let rva = if let Some(ord) = func.strip_prefix("ordinal#") {
            let ord: u16 = ord.parse().ok()?;
            self.exports.iter().find(|e| e.ordinal == ord)?.rva
        } else {
            self.exports.iter().find(|e| e.name == func)?.rva
        };
        (rva != 0).then(|| self.base as usize + rva as usize)
    }

    /// As [`Self::load_from_bytes`], with `resolver` deciding every import.
    pub fn load_with(bytes: &[u8], resolver: &mut dyn ImportResolver) -> Result<Self, MapError> {
        let pe = parse_pe64(bytes)?;
        let prepared = prepare_image(bytes, &pe)?;
        let original_callbacks = prepared.original_callbacks;
        let path = prepared_image_path(bytes);
        write_if_changed(&path, &prepared.bytes)?;

        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: `wide` is a NUL-terminated path.
        if !unsafe { GetModuleHandleW(wide.as_ptr()) }.is_null() {
            // Its TLS index and IAT were already rewritten by the first load.
            return Err(MapError::AlreadyLoaded { path });
        }
        // SAFETY: as above. The image has no imports and no entry point, so
        // the loader runs none of its code.
        let module = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), 0) };
        if module.is_null() {
            // SAFETY: no preconditions.
            let code = unsafe { GetLastError() };
            return Err(MapError::LoadFailed { path, code });
        }

        let base = module as *mut u8;
        let size = pe.size_of_image as usize;
        let relocated = base as u64 != pe.image_base;
        let mut image = Self {
            module,
            base,
            size,
            entry_point: if pe.entry_rva != 0 {
                base as usize + pe.entry_rva as usize
            } else {
                0
            },
            relocations_applied: if relocated {
                count_dir64_relocs(bytes, &pe)
            } else {
                0
            },
            bind_report: BindReport::default(),
            tls: None,
            path,
            is_dll: pe.is_dll,
            exports: pe.exports.clone(),
            thread_notices: false,
            attached: std::sync::Arc::new(AtomicBool::new(false)),
        };
        resolver.mapped(&image, &pe);
        // SAFETY: the image is mapped at `base` and nothing runs it yet.
        image.bind_report = unsafe { patch_iat(&pe, base, size, resolver) };
        // SAFETY: as above.
        image.tls = unsafe { TlsSetup::init_for_loaded_image(base, &pe, original_callbacks) }
            .map_err(MapError::Tls)?;
        if let Some(slot) = prepared.notify_slot_rva {
            notify_targets()
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert(
                    base as usize,
                    NotifyTarget {
                        module: module as usize,
                        callbacks: image
                            .tls
                            .as_ref()
                            .map(|t| t.callbacks.clone())
                            .unwrap_or_default(),
                        dll_main: (pe.is_dll && image.entry_point != 0)
                            .then_some(image.entry_point),
                        attached: image.attached.clone(),
                    },
                );
            // SAFETY: the slot is inside the image's `.strata` section, which
            // lies past the original `SizeOfImage`.
            unsafe {
                write_iat_slots(
                    base,
                    slot as usize + 8,
                    slot as usize,
                    &[entry_notify as *const () as usize],
                );
            }
            image.thread_notices = true;
        }
        Ok(image)
    }

    /// Run the image's TLS callbacks with `DLL_PROCESS_ATTACH` on the calling
    /// thread, the first time only (as the Windows loader would before the
    /// entry point). [`Self::execute_entry_point`] calls this itself.
    ///
    /// # Safety
    /// The callbacks are title code: the IAT must be bound to what they need.
    ///
    /// For a DLL this is the whole Windows-loader attach: TLS callbacks, then
    /// `DllMain(DLL_PROCESS_ATTACH)`. Returns `false` if `DllMain` refused.
    pub unsafe fn attach(&self) -> bool {
        if self.attached.swap(true, Ordering::SeqCst) {
            return true;
        }
        // SAFETY: forwarded from the caller.
        unsafe { self.notify(DLL_PROCESS_ATTACH) }
    }

    /// Tells the image of a thread or process event as the Windows loader
    /// would: TLS callbacks, then (for a DLL with an entry point) `DllMain`.
    /// Returns `DllMain`'s answer (`true` when there is none).
    ///
    /// # Safety
    /// The image's imports must be bound.
    pub unsafe fn notify(&self, reason: u32) -> bool {
        if let Some(tls) = &self.tls {
            // SAFETY: forwarded from the caller.
            unsafe { run_tls_callbacks(self.module as usize, &tls.callbacks, reason) };
        }
        if self.is_dll && self.entry_point != 0 {
            type DllMain = unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32;
            // SAFETY: a DLL's entry point is its DllMain.
            let main: DllMain = unsafe { std::mem::transmute(self.entry_point) };
            return unsafe { main(self.module, reason, std::ptr::null_mut()) } != 0;
        }
        true
    }

    /// Whether [`Self::attach`] has run.
    pub fn is_attached(&self) -> bool {
        self.attached.load(Ordering::SeqCst)
    }

    /// Runs the TLS callbacks for `DLL_PROCESS_ATTACH` (once) on the calling
    /// thread, without `DllMain`. For an executable's main thread.
    ///
    /// # Safety
    /// The image's imports must be bound.
    pub unsafe fn attach_main_thread(&self) {
        if self.attached.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Some(tls) = &self.tls {
            // SAFETY: forwarded from the caller.
            unsafe { run_tls_callbacks(self.module as usize, &tls.callbacks, DLL_PROCESS_ATTACH) };
        }
    }

    /// Execute the mapped image's entry point on a dedicated thread with
    /// Strata's console instruction exception handler (`VehGuard`) installed,
    /// after running its TLS callbacks there.
    ///
    /// # Safety
    /// The mapped image must contain valid x86-64 machine code at `entry_point`
    /// using the Windows x64 ABI (`extern "system" fn() -> i32`).
    pub unsafe fn execute_entry_point(&self) -> Result<i32, MapError> {
        if self.entry_point == 0 || self.base.is_null() {
            return Err(MapError::ExecFailed(
                "image has no valid entry point".into(),
            ));
        }
        let ep = self.entry_point;
        let module = self.module as usize;
        let callbacks = if self.attached.swap(true, Ordering::SeqCst) {
            Vec::new()
        } else {
            self.tls
                .as_ref()
                .map(|t| t.callbacks.clone())
                .unwrap_or_default()
        };
        let handle = std::thread::Builder::new()
            .name("strata-guest-main".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let _veh = VehGuard::install();
                // SAFETY: forwarded from the caller.
                unsafe { run_tls_callbacks(module, &callbacks, DLL_PROCESS_ATTACH) };
                let entry_fn: unsafe extern "system" fn() -> i32 =
                    unsafe { std::mem::transmute(ep) };
                unsafe { entry_fn() }
            })
            .map_err(|e| MapError::ExecFailed(e.to_string()))?;

        handle
            .join()
            .map_err(|_| MapError::ExecFailed("guest entry thread panicked".into()))
    }
}

impl Drop for MappedImage {
    fn drop(&mut self) {
        notify_targets()
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(self.base as usize));
        if !self.module.is_null() {
            // SAFETY: loaded by `load_from_bytes`; `tls` (and its thunk page)
            // is dropped after this, once no code of the image remains.
            unsafe { FreeLibrary(self.module) };
        }
    }
}

/// # Safety
/// Each callback is title code taking `(module, reason, reserved)`.
unsafe fn run_tls_callbacks(module: usize, callbacks: &[usize], reason: u32) {
    type TlsCallback = unsafe extern "system" fn(*mut c_void, u32, *mut c_void);
    for &cb in callbacks {
        // SAFETY: forwarded from the caller.
        let callback: TlsCallback = unsafe { std::mem::transmute(cb) };
        unsafe { callback(module as *mut c_void, reason, std::ptr::null_mut()) };
    }
}

/// A patched copy of a title image and what Strata needs from the original.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub bytes: Vec<u8>,
    /// `AddressOfCallBacks` as the original had it (a VA; 0 if none).
    pub original_callbacks: u64,
    /// RVA of the entry thunk's target slot (see [`add_notify_thunk`]); `None`
    /// when the headers had no room for one more section.
    pub notify_slot_rva: Option<u32>,
}

/// Patch a copy of `bytes` into a DLL the Windows loader maps but does not
/// run. Returns the copy and the original `AddressOfCallBacks` VA (0 if none).
///
/// - `IMAGE_FILE_DLL` set and entry point cleared: mapped like a DLL, nothing called;
/// - import directory pointed at an empty descriptor, bound imports cleared:
///   Strata binds the original slots;
/// - TLS callback pointer cleared (and its relocation made a no-op so the 0
///   survives a rebase): the OS still allocates static TLS, but callbacks run
///   only when Strata calls them, after binding;
/// - `FORCE_INTEGRITY`, checksum and certificate cleared: the copy is unsigned.
pub fn prepare_image(bytes: &[u8], pe: &Pe64Info) -> Result<Prepared, MapError> {
    let mut out = bytes.to_vec();
    let pe_off = u32_at(bytes, 0x3c) as usize;
    let opt = pe_off + 24;

    let chars = u16_at(&out, pe_off + FILE_CHARACTERISTICS) | FILE_DLL;
    put(
        &mut out,
        pe_off + FILE_CHARACTERISTICS,
        &chars.to_le_bytes(),
    )?;
    put(&mut out, opt + OPT_ENTRY_POINT, &0u32.to_le_bytes())?;
    put(&mut out, opt + OPT_CHECKSUM, &0u32.to_le_bytes())?;
    let dll_chars = u16_at(&out, opt + OPT_DLL_CHARACTERISTICS) & !DLLCHAR_FORCE_INTEGRITY;
    put(
        &mut out,
        opt + OPT_DLL_CHARACTERISTICS,
        &dll_chars.to_le_bytes(),
    )?;

    let num_dirs = u32_at(&out, opt + OPT_NUM_DIRS) as usize;
    if num_dirs <= DIR_IMPORT {
        return Err(MapError::Prepare("no import directory slot"));
    }
    for dir in [DIR_SECURITY, DIR_BOUND_IMPORT] {
        if dir < num_dirs {
            put(&mut out, opt + OPT_DIRS + dir * 8, &[0; 8])?;
        }
    }
    // The import directory points at an empty descriptor rather than being
    // cleared: Windows skips an image's static TLS setup when it has no
    // import directory at all.
    let empty = find_empty_import_descriptor(bytes, pe).ok_or(MapError::Prepare(
        "no 20 zero bytes for an empty import descriptor",
    ))?;
    put(
        &mut out,
        opt + OPT_DIRS + DIR_IMPORT * 8,
        &empty.to_le_bytes(),
    )?;
    put(
        &mut out,
        opt + OPT_DIRS + DIR_IMPORT * 8 + 4,
        &(IMPORT_DESCRIPTOR_LEN as u32).to_le_bytes(),
    )?;

    let mut original_callbacks = 0;
    if pe.tls_dir_rva != 0 {
        let field_rva = pe.tls_dir_rva + TLS_DIR_CALLBACKS_OFFSET as u32;
        let field = rva_to_offset(&pe.sections, field_rva)
            .filter(|&o| o + 8 <= out.len())
            .ok_or(MapError::Prepare("TLS directory is outside the file"))?;
        original_callbacks = u64_at(&out, field);
        put(&mut out, field, &0u64.to_le_bytes())?;
        neutralize_dir64_reloc(&mut out, pe, field_rva);
    }
    let notify_slot_rva = add_notify_thunk(&mut out, pe)?;
    Ok(Prepared {
        bytes: out,
        original_callbacks,
        notify_slot_rva,
    })
}

/// Entry thunk: `DllMain`-shaped, jumps to the 64-bit target in the slot at
/// +0x20, or returns TRUE while the slot is still 0 (the loader's own
/// `DLL_PROCESS_ATTACH`, before Strata fills the slot in).
///
/// ```text
/// mov rax, [rip+0x19]  ; 48 8B 05 19 00 00 00
/// test rax, rax        ; 48 85 C0
/// jz +2                ; 74 02
/// jmp rax              ; FF E0
/// mov eax, 1           ; B8 01 00 00 00
/// ret                  ; C3
/// ```
const NOTIFY_THUNK: [u8; 20] = [
    0x48, 0x8B, 0x05, 0x19, 0x00, 0x00, 0x00, 0x48, 0x85, 0xC0, 0x74, 0x02, 0xFF, 0xE0, 0xB8, 0x01,
    0x00, 0x00, 0x00, 0xC3,
];
const NOTIFY_SLOT_OFFSET: u32 = 0x20;
const SCN_CODE_EXEC_READ: u32 = 0x6000_0020;

/// Appends a `.strata` section holding [`NOTIFY_THUNK`] and makes it the
/// image's entry point, so the Windows loader tells the image of every
/// thread that starts or ends in the process (host-created ones too), as it
/// would a DLL. Returns the slot's RVA, or `None` (no thunk) when the
/// headers have no free room for another section header.
fn add_notify_thunk(out: &mut Vec<u8>, pe: &Pe64Info) -> Result<Option<u32>, MapError> {
    let pe_off = u32_at(out, 0x3c) as usize;
    let opt = pe_off + 24;
    let nsec = u16_at(out, pe_off + 6) as usize;
    let table = opt + u16_at(out, pe_off + 20) as usize;
    let new_hdr = table + nsec * 40;
    let headers_end = (pe.size_of_headers as usize).min(out.len());
    let first_raw = pe
        .sections
        .iter()
        .map(|s| s.raw_offset as usize)
        .filter(|&o| o != 0)
        .min()
        .unwrap_or(headers_end);
    if new_hdr + 40 > headers_end.min(first_raw)
        || out[new_hdr..new_hdr + 40].iter().any(|&b| b != 0)
    {
        return Ok(None);
    }
    let section_align = u32_at(out, opt + 32).max(0x1000);
    let file_align = u32_at(out, opt + 36).max(0x200);
    let va = pe.size_of_image.next_multiple_of(section_align);
    let raw = (out.len() as u32).next_multiple_of(file_align);

    out.resize(raw as usize, 0);
    out.extend_from_slice(&NOTIFY_THUNK);
    out.resize((raw + file_align) as usize, 0);

    let mut hdr = [0u8; 40];
    hdr[0..7].copy_from_slice(b".strata");
    hdr[8..12].copy_from_slice(&0x40u32.to_le_bytes()); // VirtualSize
    hdr[12..16].copy_from_slice(&va.to_le_bytes());
    hdr[16..20].copy_from_slice(&file_align.to_le_bytes());
    hdr[20..24].copy_from_slice(&raw.to_le_bytes());
    hdr[36..40].copy_from_slice(&SCN_CODE_EXEC_READ.to_le_bytes());
    put(out, new_hdr, &hdr)?;
    put(out, pe_off + 6, &((nsec + 1) as u16).to_le_bytes())?;
    put(out, opt + 56, &(va + section_align).to_le_bytes())?; // SizeOfImage
    put(out, opt + OPT_ENTRY_POINT, &va.to_le_bytes())?;
    Ok(Some(va + NOTIFY_SLOT_OFFSET))
}

/// What the entry thunk forwards thread events to, per image base.
#[derive(Clone)]
struct NotifyTarget {
    module: usize,
    callbacks: Vec<usize>,
    /// The original `DllMain`, for DLLs.
    dll_main: Option<usize>,
    attached: std::sync::Arc<AtomicBool>,
}

fn notify_targets() -> &'static std::sync::RwLock<std::collections::HashMap<usize, NotifyTarget>> {
    static T: std::sync::OnceLock<
        std::sync::RwLock<std::collections::HashMap<usize, NotifyTarget>>,
    > = std::sync::OnceLock::new();
    T.get_or_init(Default::default)
}

/// The entry thunk's target: a thread starts or ends somewhere in the
/// process. Forwards `DLL_THREAD_ATTACH`/`DETACH` to an attached image's TLS
/// callbacks and `DllMain`. Process events are not forwarded: process attach
/// is [`MappedImage::attach`]'s, and title code is not run at host exit.
unsafe extern "system" fn entry_notify(
    module: *mut c_void,
    reason: u32,
    _reserved: *mut c_void,
) -> i32 {
    if reason != DLL_THREAD_ATTACH && reason != DLL_THREAD_DETACH {
        return 1;
    }
    let target = notify_targets()
        .read()
        .ok()
        .and_then(|t| t.get(&(module as usize)).cloned());
    let Some(t) = target.filter(|t| t.attached.load(Ordering::SeqCst)) else {
        return 1;
    };
    // SAFETY: the image's imports are bound (it is attached).
    unsafe { run_tls_callbacks(t.module, &t.callbacks, reason) };
    if let Some(main) = t.dll_main {
        type DllMain = unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32;
        // SAFETY: the original entry point of a DLL is its DllMain.
        let f: DllMain = unsafe { std::mem::transmute(main) };
        unsafe { f(module, reason, std::ptr::null_mut()) };
    }
    1
}

/// RVA of 20 zero bytes in the file-backed part of a section. (Windows
/// ignores an import directory inside the headers, and with it the TLS setup.)
fn find_empty_import_descriptor(bytes: &[u8], pe: &Pe64Info) -> Option<u32> {
    pe.sections.iter().find_map(|s| {
        let start = s.raw_offset as usize;
        let len = (s.raw_size as usize).min(s.virtual_size.max(s.raw_size) as usize);
        let end = start.saturating_add(len).min(bytes.len());
        (start.next_multiple_of(4)..end.saturating_sub(IMPORT_DESCRIPTOR_LEN - 1))
            .step_by(4)
            .find(|&off| {
                bytes[off..off + IMPORT_DESCRIPTOR_LEN]
                    .iter()
                    .all(|&b| b == 0)
            })
            .and_then(|off| u32::try_from(s.virtual_addr as usize + (off - start)).ok())
    })
}

fn put(out: &mut [u8], off: usize, value: &[u8]) -> Result<(), MapError> {
    out.get_mut(off..off + value.len())
        .ok_or(MapError::Prepare("header field is outside the file"))?
        .copy_from_slice(value);
    Ok(())
}

/// Walks the base relocation blocks in file `bytes`, calling `f` with the
/// file offset and target RVA of every DIR64 entry.
fn for_each_dir64_reloc(bytes: &[u8], pe: &Pe64Info, mut f: impl FnMut(usize, u32)) {
    if pe.reloc_dir_rva == 0 || pe.reloc_dir_size < 8 {
        return;
    }
    let Some(start) = rva_to_offset(&pe.sections, pe.reloc_dir_rva) else {
        return;
    };
    let end = (start + pe.reloc_dir_size as usize).min(bytes.len());
    let mut off = start;
    while off + 8 <= end {
        let page_rva = u32_at(bytes, off);
        let block_size = u32_at(bytes, off + 4) as usize;
        if block_size < 8 || off + block_size > end {
            break;
        }
        for entry_off in (off + 8..off + block_size - 1).step_by(2) {
            let entry = u16_at(bytes, entry_off);
            if entry >> 12 == REL_BASED_DIR64 {
                f(entry_off, page_rva + u32::from(entry & 0x0fff));
            }
        }
        off += block_size;
    }
}

fn neutralize_dir64_reloc(out: &mut [u8], pe: &Pe64Info, target_rva: u32) {
    let mut hits = Vec::new();
    for_each_dir64_reloc(out, pe, |off, rva| {
        if rva == target_rva {
            hits.push(off);
        }
    });
    for off in hits {
        // Type 0 (`IMAGE_REL_BASED_ABSOLUTE`) is padding the loader skips.
        out[off..off + 2].copy_from_slice(&0u16.to_le_bytes());
    }
}

fn count_dir64_relocs(bytes: &[u8], pe: &Pe64Info) -> usize {
    let mut n = 0;
    for_each_dir64_reloc(bytes, pe, |_, _| n += 1);
    n
}

/// Cache path of the prepared copy, unique per original build.
fn prepared_image_path(original: &[u8]) -> PathBuf {
    // FNV-1a: a file name, not a security boundary.
    let hash = original.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    });
    std::env::temp_dir()
        .join("strata-images")
        .join(format!("strata-image-{hash:016x}-{:x}.dll", original.len()))
}

fn write_if_changed(path: &Path, bytes: &[u8]) -> Result<(), MapError> {
    let io = |source| MapError::Io {
        path: path.to_owned(),
        source,
    };
    if std::fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(io)?;
    }
    std::fs::write(path, bytes).map_err(io)
}

/// Looks up `func` (a name or `ordinal#N`) in a loaded host module.
fn host_proc(hmod: *mut c_void, func: &str) -> *const c_void {
    if hmod.is_null() {
        return std::ptr::null();
    }
    if let Some(ord_str) = func.strip_prefix("ordinal#") {
        let ord = ord_str.parse::<usize>().unwrap_or(0);
        // SAFETY: ordinal lookup (a value below 0x10000 in place of a name).
        unsafe { GetProcAddress(hmod, ord as *const i8) }
    } else if let Ok(c_fn) = CString::new(func) {
        // SAFETY: NUL-terminated name.
        unsafe { GetProcAddress(hmod, c_fn.as_ptr()) }
    } else {
        std::ptr::null()
    }
}

fn load_host_dll(dll_name: &str) -> *mut c_void {
    match CString::new(dll_name) {
        // SAFETY: NUL-terminated name.
        Ok(s) => unsafe { LoadLibraryA(s.as_ptr()) },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Where an import was bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    /// A Strata implementation (HLE module or an override of a host export).
    Hle(usize),
    /// The host's own export.
    Host(usize),
    /// An export of a DLL the title ships.
    Title(usize),
    /// Left to the caller (title DLLs when no title folder is known).
    Deferred,
    Unresolved,
}

/// Decides what each import of an image is bound to.
pub trait ImportResolver {
    /// Called once the image is mapped, before any of its imports is bound
    /// (so a DLL that imports back from it can find its exports).
    fn mapped(&mut self, _image: &MappedImage, _pe: &Pe64Info) {}
    fn resolve(&mut self, dll: &str, func: &str) -> Resolution;
    /// Module handle a delay-load descriptor of `dll` should hold.
    fn module_handle(&mut self, _dll: &str) -> Option<usize> {
        None
    }
}

/// Resolution without title DLLs: overrides, HLE, host. Title-local DLLs are
/// deferred.
pub struct HostResolver;

impl ImportResolver for HostResolver {
    fn resolve(&mut self, dll: &str, func: &str) -> Resolution {
        resolve_without_title(dll, func)
    }
    fn module_handle(&mut self, dll: &str) -> Option<usize> {
        let h = load_host_dll(dll);
        (!h.is_null()).then_some(h as usize)
    }
}

/// Host modules a `kernelx.dll` import falls back to, in order: the console's
/// kernel exports the same Win32 API the PC's does.
const KERNELX_HOSTS: [&str; 3] = ["kernelbase.dll", "kernel32.dll", "ntdll.dll"];

/// Every step of import resolution except the title's own DLLs.
pub fn resolve_without_title(dll: &str, func: &str) -> Resolution {
    if let Some(addr) = crate::title::import_override(dll, func) {
        return Resolution::Hle(addr);
    }
    match classify_dll(dll) {
        DllClass::StrataHle => {
            if let Some(addr) = resolve_hle(dll, func) {
                return Resolution::Hle(addr);
            }
            if dll.eq_ignore_ascii_case("kernelx.dll") {
                for host in KERNELX_HOSTS {
                    let addr = host_proc(load_host_dll(host), func);
                    if !addr.is_null() {
                        return Resolution::Host(addr as usize);
                    }
                }
            }
            Resolution::Unresolved
        }
        DllClass::HostPassthrough => {
            let addr = host_proc(load_host_dll(dll), func);
            if addr.is_null() {
                Resolution::Unresolved
            } else {
                Resolution::Host(addr as usize)
            }
        }
        DllClass::TitleLocal => Resolution::Deferred,
        DllClass::NeedsStub => Resolution::Unresolved,
    }
}

/// Address of a host export, or 0.
pub fn host_export(dll: &str, func: &str) -> usize {
    host_proc(load_host_dll(dll), func) as usize
}

/// The stub an unresolved `dll!func` is bound to: its own named stub, or the
/// shared logged one if no executable memory could be had.
pub fn unresolved_stub(dll: &str, func: &str) -> usize {
    crate::stubs::stub_for(dll, func).unwrap_or(logged_unimplemented_stub as *const () as usize)
}

/// # Safety
/// `base` is a mapped image of `size` bytes that `pe` describes.
unsafe fn patch_iat(
    pe: &Pe64Info,
    base: *mut u8,
    size: usize,
    resolver: &mut dyn ImportResolver,
) -> BindReport {
    let mut report = BindReport::default();

    for dll in &pe.imports {
        let mut slots: Vec<usize> = Vec::with_capacity(dll.functions.len());
        for func in &dll.functions {
            let addr = match resolver.resolve(&dll.dll_name, func) {
                Resolution::Hle(a) => {
                    report.resolved_hle += 1;
                    a
                }
                Resolution::Host(a) => {
                    report.resolved_host += 1;
                    a
                }
                Resolution::Title(a) => {
                    report.resolved_title += 1;
                    a
                }
                Resolution::Deferred => {
                    report.deferred_title_local += 1;
                    // Left alone, the slot would hold the name's RVA and a
                    // call through it would jump to a low address.
                    unresolved_stub(&dll.dll_name, func)
                }
                Resolution::Unresolved => {
                    report.unresolved.push(format!("{}!{}", dll.dll_name, func));
                    unresolved_stub(&dll.dll_name, func)
                }
            };
            slots.push(addr);
        }
        // SAFETY: forwarded from the caller.
        unsafe { write_iat_slots(base, size, dll.first_thunk_rva as usize, &slots) };
        // Delay-load slots are bound eagerly above, so the title's delay
        // helper never runs; its handle slot still gets the module.
        if let Some(handle_rva) = dll.delay_module_handle_rva
            && let Some(handle) = resolver.module_handle(&dll.dll_name)
        {
            // SAFETY: forwarded from the caller.
            unsafe { write_iat_slots(base, size, handle_rva as usize, &[handle]) };
        }
    }

    report
}

/// Writes consecutive 8-byte IAT slots, lifting the OS-applied protection
/// (the IAT usually sits in read-only `.rdata`) for the span and restoring it.
///
/// # Safety
/// `base` is a mapped image of `size` bytes.
unsafe fn write_iat_slots(base: *mut u8, size: usize, offset: usize, values: &[usize]) {
    let len = values.len() * 8;
    if values.is_empty() || offset.checked_add(len).is_none_or(|end| end > size) {
        return;
    }
    // SAFETY: the span is inside the image (checked above).
    let at = unsafe { base.add(offset) };
    let mut old = 0u32;
    // Writable *and* executable while patching: the page may hold code another
    // thread runs meanwhile (the entry thunk on thread start, merged sections).
    if unsafe { VirtualProtect(at as *mut c_void, len, PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
        return;
    }
    for (i, &value) in values.iter().enumerate() {
        // SAFETY: inside the span made writable above.
        unsafe { (at.add(i * 8) as *mut u64).write_unaligned(value as u64) };
    }
    // SAFETY: restoring what VirtualProtect reported.
    unsafe { VirtualProtect(at as *mut c_void, len, old, &mut old) };
}

/// Dry-run import check without mapping the executable into memory.
pub fn check_import_binding(pe: &Pe64Info) -> BindReport {
    let mut report = BindReport::default();

    for dll in &pe.imports {
        match classify_dll(&dll.dll_name) {
            DllClass::StrataHle => {
                for func in &dll.functions {
                    if resolve_hle(&dll.dll_name, func).is_some() {
                        report.resolved_hle += 1;
                    } else {
                        report.unresolved.push(format!("{}!{}", dll.dll_name, func));
                    }
                }
            }
            DllClass::HostPassthrough => {
                let hmod = load_host_dll(&dll.dll_name);
                for func in &dll.functions {
                    if host_proc(hmod, func).is_null() {
                        report.unresolved.push(format!("{}!{}", dll.dll_name, func));
                    } else {
                        report.resolved_host += 1;
                    }
                }
            }
            DllClass::TitleLocal => {
                report.deferred_title_local += dll.functions.len();
            }
            DllClass::NeedsStub => {
                for func in &dll.functions {
                    report.unresolved.push(format!("{}!{}", dll.dll_name, func));
                }
            }
        }
    }

    report
}

#[cfg(all(test, windows, target_arch = "x86_64"))]
mod tests {
    use super::*;

    use crate::testpe::*;

    #[test]
    fn maps_and_executes_real_pe64_entry_point() {
        let pe = TestPe {
            text: vec![0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3],
            ..Default::default()
        };
        let mapped = MappedImage::load_from_bytes(&pe.build()).expect("failed to map test PE");
        let code = unsafe { mapped.execute_entry_point() }.expect("entry execution failed");
        assert_eq!(code, 42);
    }

    #[test]
    fn prepared_image_hides_imports_entry_and_tls_callbacks_from_the_os() {
        let mut pe = TestPe {
            text: vec![0xC3],
            ..Default::default()
        };
        at(&mut pe.rdata, 0x18, &(BASE + 0x3020).to_le_bytes());
        pe.dirs = vec![(1, RDATA + 0x100, 40), (9, RDATA, 40)];
        pe.relocs = vec![RDATA + 0x18, DATA];
        let bytes = pe.build();
        let info = parse_pe64(&bytes).unwrap();
        let prepared = prepare_image(&bytes, &info).unwrap();
        let (out, callbacks) = (prepared.bytes, prepared.original_callbacks);
        let patched = parse_pe64(&out).unwrap();

        assert_eq!(callbacks, BASE + 0x3020);
        let thunk = patched.sections.last().unwrap();
        assert_eq!(thunk.name, ".strata");
        assert_eq!(
            patched.entry_rva, thunk.virtual_addr,
            "entry is the notify thunk"
        );
        assert_eq!(prepared.notify_slot_rva, Some(thunk.virtual_addr + 0x20));
        assert_eq!(u16_at(&out, 0x40 + 22) & FILE_DLL, FILE_DLL);
        assert!(patched.imports.is_empty(), "the OS sees no imports");
        assert_eq!(u32_at(&out, 0x40 + 24 + OPT_DIRS + 12), 20);
        let field = rva_to_offset(&patched.sections, RDATA + 0x18).unwrap();
        assert_eq!(u64_at(&out, field), 0);
        let mut targets = Vec::new();
        for_each_dir64_reloc(&out, &patched, |_, rva| targets.push(rva));
        assert_eq!(
            targets,
            [DATA],
            "only the callback pointer's relocation is dropped"
        );
    }

    /// Two ways title code reaches its TLS block, plus a TLS callback.
    fn tls_test_pe() -> TestPe {
        let mut pe = TestPe::default();
        // 0x1000 (entry): exe-style, assumes index 0.
        //   mov rax, gs:[58h] ; mov rax, [rax] ; mov eax, [rax] ; ret
        at(
            &mut pe.text,
            0x00,
            &[
                0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0, 0x48, 0x8B, 0x00, 0x8B, 0x00, 0xC3,
            ],
        );
        // 0x1010: CRT-style, reads `_tls_index`.
        //   mov ecx, [rip+_tls_index] ; mov rax, gs:[58h] ; mov rax, [rax+rcx*8] ; mov eax, [rax] ; ret
        at(&mut pe.text, 0x10, &[0x8B, 0x0D]);
        at(&mut pe.text, 0x12, &rel(TEXT + 0x10, 6, DATA + 0x10));
        at(
            &mut pe.text,
            0x16,
            &[
                0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0, 0x48, 0x8B, 0x04, 0xC8, 0x8B, 0x00,
                0xC3,
            ],
        );
        // 0x1030: TLS callback: bts [rip+marker], edx ; ret  (one bit per reason seen)
        at(&mut pe.text, 0x30, &[0x0F, 0xAB, 0x15]);
        at(&mut pe.text, 0x33, &rel(TEXT + 0x30, 7, DATA + 0x18));
        at(&mut pe.text, 0x37, &[0xC3]);

        // .data: template, _tls_index, marker, callback array.
        at(&mut pe.data, 0x00, &0x5EED_1234u32.to_le_bytes());
        at(
            &mut pe.data,
            0x20,
            &(BASE + u64::from(TEXT) + 0x30).to_le_bytes(),
        );
        at(&mut pe.data, 0x28, &0u64.to_le_bytes());

        // .rdata: IMAGE_TLS_DIRECTORY64.
        for (i, va) in [DATA, DATA + 4, DATA + 0x10, DATA + 0x20]
            .into_iter()
            .enumerate()
        {
            at(&mut pe.rdata, i * 8, &(BASE + u64::from(va)).to_le_bytes());
        }
        at(&mut pe.rdata, 32, &4u32.to_le_bytes()); // zero fill
        pe.dirs = vec![(9, RDATA, 40)];
        pe.relocs = vec![RDATA, RDATA + 8, RDATA + 16, RDATA + 24, DATA + 0x20];
        pe
    }

    #[test]
    fn static_tls_reaches_the_titles_block_on_every_thread() {
        let mapped = MappedImage::load_from_bytes(&tls_test_pe().build()).expect("map");
        let tls = mapped.tls.as_ref().expect("TLS directory");
        if tls.os_index != 0 {
            assert_eq!(tls.patched_loads, 2);
        }
        assert_eq!(tls.callbacks, [mapped.base as usize + 0x1030]);
        let marker = unsafe { mapped.base.add(0x3018) as *const u32 };
        assert_eq!(
            unsafe { marker.read_volatile() },
            0,
            "the OS must not run TLS callbacks before binding"
        );

        // The guest thread is created after the load.
        assert_eq!(
            unsafe { mapped.execute_entry_point() }.unwrap() as u32,
            0x5EED_1234
        );
        // The callback ran with PROCESS_ATTACH on the guest thread, then the
        // loader told it the thread ended.
        let seen = unsafe { marker.read_volatile() };
        assert_ne!(
            seen & (1 << DLL_PROCESS_ATTACH),
            0,
            "process attach: {seen:#b}"
        );
        assert_ne!(
            seen & (1 << DLL_THREAD_DETACH),
            0,
            "thread detach: {seen:#b}"
        );

        // This thread existed before the load.
        let crt: extern "system" fn() -> u32 =
            unsafe { std::mem::transmute(mapped.base.add(0x1010)) };
        let exe: extern "system" fn() -> u32 = unsafe { std::mem::transmute(mapped.entry_point) };
        assert_eq!(crt(), 0x5EED_1234);
        assert_eq!(exe(), 0x5EED_1234);
        // And another thread, created later, gets a fresh block.
        assert_eq!(
            std::thread::spawn(move || (crt(), exe())).join().unwrap(),
            (0x5EED_1234, 0x5EED_1234)
        );
    }

    #[test]
    fn threads_the_host_creates_reach_an_attached_title_dll() {
        let dll = MappedImage::load_from_bytes(&crate::testpe::probe_helper_dll()).expect("map");
        assert!(dll.thread_notices);
        let counts = unsafe { std::slice::from_raw_parts(dll.base.add(0x3000) as *const u32, 4) };
        std::thread::spawn(|| ()).join().unwrap();
        assert_eq!(
            unsafe { std::ptr::read_volatile(&counts[2]) },
            0,
            "no notices before attach"
        );
        assert!(unsafe { dll.attach() });
        assert_eq!(unsafe { std::ptr::read_volatile(&counts[1]) }, 1);
        std::thread::spawn(|| ()).join().unwrap();
        assert!(
            unsafe { std::ptr::read_volatile(&counts[2]) } >= 1,
            "DLL_THREAD_ATTACH"
        );
        assert!(
            unsafe { std::ptr::read_volatile(&counts[3]) } >= 1,
            "DLL_THREAD_DETACH"
        );
    }

    #[repr(C)]
    struct RuntimeFunction {
        begin: u32,
        end: u32,
        unwind: u32,
    }

    #[link(name = "ntdll", kind = "raw-dylib")]
    unsafe extern "system" {
        fn RtlLookupFunctionEntry(
            pc: u64,
            image_base: *mut u64,
            history: *mut c_void,
        ) -> *mut c_void;
        fn RtlPcToFileHeader(pc: *const c_void, base: *mut *mut c_void) -> *mut c_void;
    }

    #[test]
    fn unwind_tables_are_registered_with_the_os() {
        let mut pe = TestPe {
            text: vec![0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3],
            ..Default::default()
        };
        // RUNTIME_FUNCTION for 0x1000..0x1006, UNWIND_INFO v1 with no codes.
        at(
            &mut pe.rdata,
            0,
            &[TEXT, TEXT + 6, RDATA + 0x10]
                .map(u32::to_le_bytes)
                .concat(),
        );
        at(&mut pe.rdata, 0x10, &[0x01, 0x00, 0x00, 0x00]);
        pe.dirs = vec![(3, RDATA, 12)];
        let mapped = MappedImage::load_from_bytes(&pe.build()).expect("map");

        let mut image_base = 0u64;
        let entry = unsafe {
            RtlLookupFunctionEntry(
                mapped.entry_point as u64 + 2,
                &mut image_base,
                std::ptr::null_mut(),
            )
        } as *const RuntimeFunction;
        assert!(
            !entry.is_null(),
            "no unwind entry: C++ exceptions and SEH cannot cross title frames"
        );
        assert_eq!(image_base, mapped.base as u64);
        assert_eq!(unsafe { ((*entry).begin, (*entry).end) }, (TEXT, TEXT + 6));

        let mut header = std::ptr::null_mut();
        let found = unsafe { RtlPcToFileHeader(mapped.entry_point as *const c_void, &mut header) };
        assert_eq!(found as usize, mapped.base as usize);
    }

    #[test]
    fn missing_host_export_is_bound_to_its_named_stub() {
        let _policy = crate::stubs::policy_lock(crate::stubs::UnimplementedPolicy::ReturnNotImpl);
        let mut pe = TestPe::default();
        // Descriptor (+ null terminator), ILT at 0x40, IAT at 0x60, names after.
        let desc = [RDATA + 0x40, 0, 0, RDATA + 0x80, RDATA + 0x60]
            .map(u32::to_le_bytes)
            .concat();
        at(&mut pe.rdata, 0x00, &desc);
        for table in [0x40, 0x60] {
            at(
                &mut pe.rdata,
                table,
                &[u64::from(RDATA + 0x90), u64::from(RDATA + 0xB0), 0]
                    .map(u64::to_le_bytes)
                    .concat(),
            );
        }
        at(&mut pe.rdata, 0x80, b"kernel32.dll\0");
        at(&mut pe.rdata, 0x92, b"GetCurrentProcessId\0");
        at(&mut pe.rdata, 0xB2, b"StrataNoSuchExport\0");
        pe.dirs = vec![(1, RDATA, 40), (12, RDATA + 0x60, 24)];
        // Entry: jmp [rip+slot1] — a tail call into the missing import.
        at(&mut pe.text, 0, &[0xFF, 0x25]);
        at(&mut pe.text, 2, &rel(TEXT, 6, RDATA + 0x68));

        let mapped = MappedImage::load_from_bytes(&pe.build()).expect("map");
        assert_eq!(mapped.bind_report.resolved_host, 1);
        assert_eq!(
            mapped.bind_report.unresolved,
            ["kernel32.dll!StrataNoSuchExport"]
        );

        let slots =
            unsafe { std::slice::from_raw_parts(mapped.base.add(0x2060) as *const usize, 2) };
        let kernel32 = load_host_dll("kernel32.dll");
        assert_eq!(
            slots[0],
            host_proc(kernel32, "GetCurrentProcessId") as usize
        );
        let site = crate::stubs::site_of(slots[1]).expect("a named stub");
        assert_eq!(
            (site.dll.as_str(), site.symbol.as_str()),
            ("kernel32.dll", "StrataNoSuchExport")
        );
        assert_eq!(
            unsafe { mapped.execute_entry_point() }.unwrap() as u32,
            0x8000_4001
        );
    }
}
