//! PE32+ In-Memory Image Mapper, Base Relocator & Live IAT Patcher.

use std::ffi::CString;

use thiserror::Error;

use crate::pe::{Pe64Info, PeError, parse_pe64};
use crate::runtime::{DllClass, classify_dll};
use crate::thunks::resolve_hle;
use crate::tls::TlsSetup;

#[cfg(windows)]
unsafe extern "system" {
    fn LoadLibraryA(name: *const i8) -> *mut core::ffi::c_void;
    fn GetProcAddress(module: *mut core::ffi::c_void, name: *const i8) -> *const core::ffi::c_void;
    fn VirtualAlloc(
        addr: *mut core::ffi::c_void,
        size: usize,
        alloc_type: u32,
        protect: u32,
    ) -> *mut core::ffi::c_void;
    fn VirtualFree(addr: *mut core::ffi::c_void, size: usize, free_type: u32) -> i32;
    fn VirtualProtect(
        addr: *mut core::ffi::c_void,
        size: usize,
        new_protect: u32,
        old_protect: *mut u32,
    ) -> i32;
}

const MEM_COMMIT_RESERVE: u32 = 0x3000;
const MEM_RELEASE: u32 = 0x8000;
const PAGE_READONLY: u32 = 0x02;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_EXECUTE_READ: u32 = 0x20;
const PAGE_EXECUTE_READWRITE: u32 = 0x40;

#[derive(Debug, Error)]
pub enum MapError {
    #[error(transparent)]
    Pe(#[from] PeError),
    #[error("VirtualAlloc failed for {size} bytes")]
    AllocFailed { size: u32 },
}

#[derive(Debug, Default)]
pub struct BindReport {
    pub resolved_hle: usize,
    pub resolved_host: usize,
    pub deferred_title_local: usize,
    pub unresolved: Vec<String>,
}

/// A PE32+ executable mapped into live Windows virtual memory with base
/// relocations applied and its IAT patched to Strata HLE + Host Windows.
pub struct MappedImage {
    pub base: *mut u8,
    pub size: usize,
    pub entry_point: usize,
    pub relocations_applied: usize,
    pub bind_report: BindReport,
    pub tls: Option<TlsSetup>,
}

impl MappedImage {
    /// Map a PE32+ binary into memory, apply DIR64 relocations, patch its live
    /// IAT, and apply per-section R/W/X page protections.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, MapError> {
        let pe = parse_pe64(bytes)?;
        let size = pe.size_of_image as usize;

        unsafe {
            // Try preferred ImageBase first; fall back to OS-chosen base if occupied.
            let mut base = VirtualAlloc(
                pe.image_base as *mut core::ffi::c_void,
                size,
                MEM_COMMIT_RESERVE,
                PAGE_READWRITE,
            ) as *mut u8;
            if base.is_null() {
                base = VirtualAlloc(
                    std::ptr::null_mut(),
                    size,
                    MEM_COMMIT_RESERVE,
                    PAGE_READWRITE,
                ) as *mut u8;
            }
            if base.is_null() {
                return Err(MapError::AllocFailed { size: pe.size_of_image });
            }

            // Copy PE headers
            let hdr_len = (pe.size_of_headers as usize).min(bytes.len()).min(size);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), base, hdr_len);

            // Copy sections into their virtual addresses
            for sec in &pe.sections {
                let raw_off = sec.raw_offset as usize;
                let raw_len = sec.raw_size as usize;
                let va = sec.virtual_addr as usize;
                if raw_len > 0 && raw_off + raw_len <= bytes.len() && va + raw_len <= size {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr().add(raw_off), base.add(va), raw_len);
                }
            }

            // Apply IMAGE_REL_BASED_DIR64 base relocations if loaded away from ImageBase
            let delta = (base as u64).wrapping_sub(pe.image_base);
            let mut relocations_applied = 0usize;
            if delta != 0 && pe.reloc_dir_rva != 0 && pe.reloc_dir_size >= 8 {
                let mut off = pe.reloc_dir_rva as usize;
                let end = (off + pe.reloc_dir_size as usize).min(size);
                while off + 8 <= end {
                    let page_rva = u32::from_le_bytes(*(base.add(off) as *const [u8; 4])) as usize;
                    let block_size = u32::from_le_bytes(*(base.add(off + 4) as *const [u8; 4])) as usize;
                    if block_size < 8 || off + block_size > end {
                        break;
                    }
                    let count = (block_size - 8) / 2;
                    for i in 0..count {
                        let entry = u16::from_le_bytes(*(base.add(off + 8 + i * 2) as *const [u8; 2]));
                        let kind = entry >> 12;
                        let page_off = (entry & 0x0fff) as usize;
                        if kind == 10 {
                            // IMAGE_REL_BASED_DIR64
                            let target = page_rva + page_off;
                            if target + 8 <= size {
                                let p = base.add(target) as *mut u64;
                                *p = (*p).wrapping_add(delta);
                                relocations_applied += 1;
                            }
                        }
                    }
                    off += block_size;
                }
            }

            // Patch live IAT entries in memory
            let bind_report = patch_iat(&pe, base, size);

            // Initialize static TLS slot if the PE defines IMAGE_DIRECTORY_ENTRY_TLS
            let tls = TlsSetup::init_for_mapped_image(base, size, pe.tls_dir_rva);

            // Apply per-section memory protections (R / RW / RX / RWX)
            for sec in &pe.sections {
                let va = sec.virtual_addr as usize;
                let len = (sec.virtual_size.max(sec.raw_size) as usize).min(size.saturating_sub(va));
                if len == 0 {
                    continue;
                }
                let exec = (sec.characteristics & 0x2000_0000) != 0;
                let write = (sec.characteristics & 0x8000_0000) != 0;
                let prot = match (exec, write) {
                    (true, true) => PAGE_EXECUTE_READWRITE,
                    (true, false) => PAGE_EXECUTE_READ,
                    (false, true) => PAGE_READWRITE,
                    (false, false) => PAGE_READONLY,
                };
                let mut old = 0u32;
                VirtualProtect(base.add(va) as *mut core::ffi::c_void, len, prot, &mut old);
            }

            Ok(Self {
                base,
                size,
                entry_point: (base as usize) + (pe.entry_rva as usize),
                relocations_applied,
                bind_report,
                tls,
            })
        }
    }
}

impl Drop for MappedImage {
    fn drop(&mut self) {
        if !self.base.is_null() {
            unsafe {
                VirtualFree(self.base as *mut core::ffi::c_void, 0, MEM_RELEASE);
            }
        }
    }
}

unsafe fn patch_iat(pe: &Pe64Info, base: *mut u8, size: usize) -> BindReport {
    let mut report = BindReport::default();

    for dll in &pe.imports {
        let iat_base = dll.first_thunk_rva as usize;
        match classify_dll(&dll.dll_name) {
            DllClass::StrataHle => {
                for (idx, func) in dll.functions.iter().enumerate() {
                    if let Some(addr) = resolve_hle(&dll.dll_name, func) {
                        write_iat_slot(base, size, iat_base + idx * 8, addr as u64);
                        report.resolved_hle += 1;
                    } else {
                        report.unresolved.push(format!("{}!{}", dll.dll_name, func));
                    }
                }
            }
            DllClass::HostPassthrough => {
                let c_dll = CString::new(dll.dll_name.as_str()).ok();
                let hmod = c_dll
                    .as_ref()
                    .map(|s| unsafe { LoadLibraryA(s.as_ptr()) })
                    .unwrap_or(std::ptr::null_mut());
                for (idx, func) in dll.functions.iter().enumerate() {
                    if hmod.is_null() {
                        report.unresolved.push(format!("{}!{}", dll.dll_name, func));
                        continue;
                    }
                    let addr = if let Some(ord_str) = func.strip_prefix("ordinal#") {
                        let ord = ord_str.parse::<usize>().unwrap_or(0);
                        unsafe { GetProcAddress(hmod, ord as *const i8) }
                    } else if let Ok(c_fn) = CString::new(func.as_str()) {
                        unsafe { GetProcAddress(hmod, c_fn.as_ptr()) }
                    } else {
                        std::ptr::null()
                    };
                    if !addr.is_null() {
                        write_iat_slot(base, size, iat_base + idx * 8, addr as u64);
                        report.resolved_host += 1;
                    } else {
                        report.unresolved.push(format!("{}!{}", dll.dll_name, func));
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

fn write_iat_slot(base: *mut u8, size: usize, offset: usize, value: u64) {
    if offset + 8 <= size {
        unsafe {
            *(base.add(offset) as *mut u64) = value;
        }
    }
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
                let c_dll = CString::new(dll.dll_name.as_str()).ok();
                let hmod = c_dll
                    .as_ref()
                    .map(|s| unsafe { LoadLibraryA(s.as_ptr()) })
                    .unwrap_or(std::ptr::null_mut());
                for func in &dll.functions {
                    if hmod.is_null() {
                        report.unresolved.push(format!("{}!{}", dll.dll_name, func));
                        continue;
                    }
                    let addr = if let Some(ord_str) = func.strip_prefix("ordinal#") {
                        let ord = ord_str.parse::<usize>().unwrap_or(0);
                        unsafe { GetProcAddress(hmod, ord as *const i8) }
                    } else if let Ok(c_fn) = CString::new(func.as_str()) {
                        unsafe { GetProcAddress(hmod, c_fn.as_ptr()) }
                    } else {
                        std::ptr::null()
                    };
                    if !addr.is_null() {
                        report.resolved_host += 1;
                    } else {
                        report.unresolved.push(format!("{}!{}", dll.dll_name, func));
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
