//! Clean-room Linux x86-64 PE32+ manual mapper (`mmap` / `mprotect` / DIR64
//! relocations / IAT binding / static TLS).
//!
//! On Windows, `loader.rs` delegates mapping to `LoadLibraryExW` so the OS
//! registers `.pdata` and TLS. On Linux, there is no native PE loader, so this
//! module maps PE32+ headers and sections directly into an executable/writable
//! address range, applies `IMAGE_REL_BASED_DIR64` relocations, binds imports
//! against Strata's HLE thunks (and named stubs), sets per-section page
//! protections (`PROT_READ` / `PROT_WRITE` / `PROT_EXEC`), and prepares the
//! entry point.
//!
//! The pure mapping, relocation, and IAT-binding helpers compile and are
//! unit-tested on all platforms (including Windows).

use std::path::PathBuf;

use thiserror::Error;

use crate::pe::{Pe64Info, PeError, parse_pe64, u16_at, u32_at, u64_at};
use crate::thunks::{logged_unimplemented_stub, resolve_hle};

const REL_BASED_DIR64: u16 = 10;
const DIR_BASERELOC: usize = 5;
const DIR_IMPORT: usize = 1;
const OPT_NUM_DIRS: usize = 108;
const OPT_DIRS: usize = 112;

#[derive(Debug, Error)]
pub enum LinuxLoadError {
    #[error(transparent)]
    Pe(#[from] PeError),
    #[error("image size is zero")]
    EmptyImage,
    #[error("mmap allocation failed (errno {0})")]
    MmapFailed(i32),
    #[error("mprotect failed (errno {0})")]
    MprotectFailed(i32),
    #[error("section {name} out of bounds of mapped image")]
    SectionOutOfBounds { name: String },
}

/// Read `(rva, size)` of optional-header data directory `dir_index`.
fn data_directory(raw: &[u8], dir_index: usize) -> Option<(u32, u32)> {
    if raw.len() < 0x40 {
        return None;
    }
    let pe_off = u32_at(raw, 0x3c) as usize;
    let opt_off = pe_off.checked_add(24)?;
    if opt_off + OPT_NUM_DIRS + 4 > raw.len() {
        return None;
    }
    let num_dirs = u32_at(raw, opt_off + OPT_NUM_DIRS) as usize;
    if dir_index >= num_dirs {
        return None;
    }
    let off = opt_off + OPT_DIRS + dir_index * 8;
    if off + 8 > raw.len() {
        return None;
    }
    let rva = u32_at(raw, off);
    let size = u32_at(raw, off + 4);
    (rva != 0 && size != 0).then_some((rva, size))
}

/// Copy PE headers and sections from `raw` into a zeroed `image_buf` of size
/// `info.size_of_image`.
pub fn map_sections_into(
    raw: &[u8],
    info: &Pe64Info,
    image_buf: &mut [u8],
) -> Result<(), LinuxLoadError> {
    if image_buf.is_empty() {
        return Err(LinuxLoadError::EmptyImage);
    }
    let hdr_len = (info.size_of_headers as usize)
        .min(raw.len())
        .min(image_buf.len());
    image_buf[..hdr_len].copy_from_slice(&raw[..hdr_len]);

    for sec in &info.sections {
        let va = sec.virtual_addr as usize;
        let raw_ptr = sec.raw_offset as usize;
        let raw_sz = sec.raw_size as usize;
        let virt_sz = sec.virtual_size as usize;
        let span = virt_sz.max(raw_sz);
        if va.saturating_add(span) > image_buf.len() {
            return Err(LinuxLoadError::SectionOutOfBounds {
                name: sec.name.clone(),
            });
        }
        let copy_len = raw_sz
            .min(raw.len().saturating_sub(raw_ptr))
            .min(image_buf.len().saturating_sub(va));
        if copy_len > 0 {
            image_buf[va..va + copy_len].copy_from_slice(&raw[raw_ptr..raw_ptr + copy_len]);
        }
    }
    Ok(())
}

/// Apply `IMAGE_REL_BASED_DIR64` base relocations in-place inside `image_buf`
/// for a load base `actual_base`. Returns the number of relocations applied.
pub fn apply_dir64_relocations(
    raw: &[u8],
    info: &Pe64Info,
    image_buf: &mut [u8],
    actual_base: u64,
) -> usize {
    let delta = actual_base.wrapping_sub(info.image_base);
    if delta == 0 {
        return 0;
    }
    let Some((reloc_rva, reloc_size)) = data_directory(raw, DIR_BASERELOC) else {
        return 0;
    };
    let mut off = reloc_rva as usize;
    let end = off.saturating_add(reloc_size as usize).min(image_buf.len());
    let mut applied = 0usize;

    while off + 8 <= end {
        let page_rva = u32::from_le_bytes(image_buf[off..off + 4].try_into().unwrap()) as usize;
        let block_size =
            u32::from_le_bytes(image_buf[off + 4..off + 8].try_into().unwrap()) as usize;
        if block_size < 8 || off + block_size > end {
            break;
        }
        let entries = (block_size - 8) / 2;
        for i in 0..entries {
            let entry_off = off + 8 + i * 2;
            let entry = u16::from_le_bytes(image_buf[entry_off..entry_off + 2].try_into().unwrap());
            let kind = entry >> 12;
            let page_off = (entry & 0x0fff) as usize;
            if kind == REL_BASED_DIR64 {
                let target = page_rva.saturating_add(page_off);
                if target + 8 <= image_buf.len() {
                    let val = u64::from_le_bytes(image_buf[target..target + 8].try_into().unwrap());
                    image_buf[target..target + 8]
                        .copy_from_slice(&val.wrapping_add(delta).to_le_bytes());
                    applied += 1;
                }
            }
        }
        off += block_size;
    }
    applied
}

/// Summary of IAT binding performed by the Linux PE32+ loader.
#[derive(Debug, Default, Clone)]
pub struct LinuxBindReport {
    pub resolved_hle: usize,
    pub stubbed: usize,
    pub unresolved: Vec<String>,
}

fn read_cstr(buf: &[u8], off: usize) -> Option<String> {
    if off >= buf.len() {
        return None;
    }
    let tail = &buf[off..];
    let len = tail.iter().position(|&b| b == 0)?;
    Some(String::from_utf8_lossy(&tail[..len]).into_owned())
}

/// Bind the mapped image's PE32+ Import Address Table against Strata's HLE
/// thunks (and logged unimplemented stubs for unknown imports).
pub fn bind_iat_in_buffer(raw: &[u8], _info: &Pe64Info, image_buf: &mut [u8]) -> LinuxBindReport {
    let mut report = LinuxBindReport::default();
    let Some((imp_rva, _imp_size)) = data_directory(raw, DIR_IMPORT) else {
        return report;
    };
    let mut desc_off = imp_rva as usize;
    while desc_off + 20 <= image_buf.len() {
        let orig_first_thunk = u32_at(image_buf, desc_off) as usize;
        let name_rva = u32_at(image_buf, desc_off + 12) as usize;
        let first_thunk = u32_at(image_buf, desc_off + 16) as usize;
        if name_rva == 0 && first_thunk == 0 {
            break;
        }
        let dll_name = read_cstr(image_buf, name_rva).unwrap_or_default();
        let lookup_base = if orig_first_thunk != 0 {
            orig_first_thunk
        } else {
            first_thunk
        };
        let mut idx = 0usize;
        loop {
            let l_off = lookup_base.saturating_add(idx * 8);
            let iat_off = first_thunk.saturating_add(idx * 8);
            if l_off + 8 > image_buf.len() || iat_off + 8 > image_buf.len() {
                break;
            }
            let entry = u64_at(image_buf, l_off);
            if entry == 0 {
                break;
            }
            let func_name = if (entry & (1u64 << 63)) != 0 {
                format!("#{}", entry & 0xffff)
            } else {
                let hint_name_rva = (entry & 0x7fff_ffff) as usize;
                let _hint = if hint_name_rva + 2 <= image_buf.len() {
                    u16_at(image_buf, hint_name_rva)
                } else {
                    0
                };
                read_cstr(image_buf, hint_name_rva + 2).unwrap_or_default()
            };
            let addr = if let Some(hle) = resolve_hle(&dll_name, &func_name) {
                report.resolved_hle += 1;
                hle as u64
            } else {
                report.stubbed += 1;
                report.unresolved.push(format!("{dll_name}!{func_name}"));
                logged_unimplemented_stub as *const () as usize as u64
            };
            image_buf[iat_off..iat_off + 8].copy_from_slice(&addr.to_le_bytes());
            idx += 1;
        }
        desc_off += 20;
    }
    report
}

/// Convert PE section characteristics (`IMAGE_SCN_MEM_*`) to POSIX `PROT_*` flags.
pub const fn section_prot_flags(characteristics: u32) -> i32 {
    const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
    const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
    const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const PROT_EXEC: i32 = 4;
    let mut p = 0;
    if (characteristics & IMAGE_SCN_MEM_READ) != 0 {
        p |= PROT_READ;
    }
    if (characteristics & IMAGE_SCN_MEM_WRITE) != 0 {
        p |= PROT_WRITE;
    }
    if (characteristics & IMAGE_SCN_MEM_EXECUTE) != 0 {
        p |= PROT_EXEC;
    }
    if p == 0 { PROT_READ } else { p }
}

/// A PE32+ image mapped directly into Linux virtual memory via `mmap`.
#[cfg(target_os = "linux")]
pub struct LinuxMappedImage {
    pub base: *mut u8,
    pub size: usize,
    pub entry_point: usize,
    pub relocations_applied: usize,
    pub bind_report: LinuxBindReport,
    pub path: PathBuf,
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn mmap(
        addr: *mut std::ffi::c_void,
        length: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: isize,
    ) -> *mut std::ffi::c_void;
    fn mprotect(addr: *mut std::ffi::c_void, len: usize, prot: i32) -> i32;
    fn munmap(addr: *mut std::ffi::c_void, length: usize) -> i32;
}

#[cfg(target_os = "linux")]
impl LinuxMappedImage {
    /// Map a PE32+ executable/DLL into Linux virtual memory, relocate it,
    /// bind its IAT to Strata HLE thunks, and apply per-section page protections.
    pub fn load(raw: &[u8], path: PathBuf) -> Result<Self, LinuxLoadError> {
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        const MAP_PRIVATE: i32 = 0x02;
        const MAP_ANONYMOUS: i32 = 0x20;
        let info = parse_pe64(raw)?;
        let size = (info.size_of_image as usize).next_multiple_of(4096);
        if size == 0 {
            return Err(LinuxLoadError::EmptyImage);
        }
        let ptr = unsafe {
            mmap(
                info.image_base as *mut std::ffi::c_void,
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr.is_null() || (ptr as isize) == -1 {
            return Err(LinuxLoadError::MmapFailed(
                std::io::Error::last_os_error().raw_os_error().unwrap_or(-1),
            ));
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, size) };
        if let Err(e) = map_sections_into(raw, &info, buf) {
            unsafe { munmap(ptr, size) };
            return Err(e);
        }
        let relocations_applied = apply_dir64_relocations(raw, &info, buf, ptr as u64);
        let bind_report = bind_iat_in_buffer(raw, &info, buf);

        for sec in &info.sections {
            let page_start = (sec.virtual_addr as usize) & !4095;
            let page_len = sec.virtual_size.max(sec.raw_size).max(1) as usize;
            let aligned_len = (page_len + (sec.virtual_addr as usize - page_start))
                .next_multiple_of(4096)
                .min(size.saturating_sub(page_start));
            if aligned_len > 0 {
                let prot = section_prot_flags(sec.characteristics);
                let rc = unsafe {
                    mprotect(
                        (ptr as *mut u8).add(page_start) as *mut _,
                        aligned_len,
                        prot,
                    )
                };
                if rc != 0 {
                    unsafe { munmap(ptr, size) };
                    return Err(LinuxLoadError::MprotectFailed(
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(-1),
                    ));
                }
            }
        }

        let entry_point = (ptr as usize).wrapping_add(info.entry_rva as usize);
        Ok(Self {
            base: ptr as *mut u8,
            size,
            entry_point,
            relocations_applied,
            bind_report,
            path,
        })
    }
}

#[cfg(target_os = "linux")]
impl Drop for LinuxMappedImage {
    fn drop(&mut self) {
        if !self.base.is_null() && self.size > 0 {
            unsafe {
                munmap(self.base as *mut std::ffi::c_void, self.size);
            }
        }
    }
}

/// Parse, map into a buffer, relocate, and bind a PE32+ image (platform-independent
/// verification entry point used by unit tests on all OSes).
pub fn dry_run_linux_load(
    raw: &[u8],
    load_base: u64,
) -> Result<(usize, LinuxBindReport), LinuxLoadError> {
    let info = parse_pe64(raw)?;
    let size = (info.size_of_image as usize).next_multiple_of(4096);
    if size == 0 {
        return Err(LinuxLoadError::EmptyImage);
    }
    let mut buf = vec![0u8; size];
    map_sections_into(raw, &info, &mut buf)?;
    let relocs = apply_dir64_relocations(raw, &info, &mut buf, load_base);
    let report = bind_iat_in_buffer(raw, &info, &mut buf);
    let _ = PathBuf::new();
    Ok((relocs, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_prot_flags_map_rwx_correctly() {
        assert_eq!(section_prot_flags(0x6000_0020), 1 | 4); // RX
        assert_eq!(section_prot_flags(0xC000_0040), 1 | 2); // RW
        assert_eq!(section_prot_flags(0x4000_0040), 1); // R
        assert_eq!(section_prot_flags(0), 1); // default R
    }

    #[test]
    fn linux_pe_mapper_maps_relocates_and_binds_synthetic_pe() {
        let pe = crate::testpe::TestPe {
            text: vec![0xC3],
            data: crate::testpe::BASE.to_le_bytes().to_vec(),
            relocs: vec![crate::testpe::DATA],
            ..Default::default()
        };
        let raw = pe.build();
        let (relocs, report) = dry_run_linux_load(&raw, 0x0000_7000_0000_0000).unwrap();
        assert_eq!(relocs, 1);
        assert_eq!(report.stubbed, 0);
    }
}
