//! Static TLS of a title executable that the Windows loader maps as a DLL.
//!
//! The loader allocates the image's TLS block for every thread, existing and
//! future, and writes the index it chose into the image's `_tls_index`. Two
//! things are left to us:
//!
//! - An executable always owns TLS index 0, so the compiler may reach the
//!   thread's block as `mov r64, gs:[0x58]` followed by `[r64]` without ever
//!   reading `_tls_index`. Mapped beside the host executable (which owns
//!   index 0 itself), the title gets another index. Every such load is
//!   turned into a call to a small thunk that yields `&array[index]`, and
//!   `_tls_index` is set to 0 so CRT code that does read it (and adds it to
//!   the array pointer) lands on the same entry.
//! - TLS callbacks. The patched image hands the loader no callback list, so
//!   nothing in the title runs before its imports are bound; the callbacks
//!   are collected here and run by [`crate::loader::MappedImage`].

use std::ffi::c_void;

use crate::pe::Pe64Info;

#[cfg(windows)]
unsafe extern "system" {
    fn VirtualAlloc(addr: *mut c_void, size: usize, alloc_type: u32, protect: u32) -> *mut c_void;
    fn VirtualFree(addr: *mut c_void, size: usize, free_type: u32) -> i32;
    fn VirtualProtect(addr: *mut c_void, size: usize, new_protect: u32, old: *mut u32) -> i32;
    fn FlushInstructionCache(process: *mut c_void, addr: *const c_void, size: usize) -> i32;
    fn GetCurrentProcess() -> *mut c_void;
}

const MEM_COMMIT_RESERVE: u32 = 0x3000;
const MEM_RELEASE: u32 = 0x8000;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_EXECUTE_READ: u32 = 0x20;
const PAGE_EXECUTE_READWRITE: u32 = 0x40;
const SCN_MEM_EXECUTE: u32 = 0x2000_0000;

/// `mov r64, gs:[0x58]` is `65 REX.W 8B ModRM(00,reg,100) SIB(25) 58 00 00 00`.
const GS_LOAD_LEN: usize = 9;
const THUNK_SLOT: usize = 32;
const THUNK_PAGE: usize = 16 * THUNK_SLOT;
/// Allocation granularity: thunk pages are searched for in steps of this.
const GRANULARITY: usize = 0x1_0000;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ImageTlsDirectory64 {
    pub start_address_of_raw_data: u64,
    pub end_address_of_raw_data: u64,
    pub address_of_index: u64,
    pub address_of_callbacks: u64,
    pub size_of_zero_fill: u32,
    pub characteristics: u32,
}

/// Offset of `AddressOfCallBacks` inside `IMAGE_TLS_DIRECTORY64`.
pub(crate) const TLS_DIR_CALLBACKS_OFFSET: usize = 24;

#[derive(Debug)]
pub struct TlsSetup {
    /// TLS index the Windows loader gave the image.
    pub os_index: u32,
    /// `mov r64, gs:[0x58]` loads redirected to `&array[os_index]`.
    pub patched_loads: usize,
    pub raw_template_bytes: usize,
    pub zero_fill_bytes: usize,
    /// Absolute addresses of the image's TLS callbacks, in order.
    pub callbacks: Vec<usize>,
    thunks: *mut u8,
}

// SAFETY: `thunks` is an executable page owned by this value; it is only
// freed in `Drop` and never written after setup.
unsafe impl Send for TlsSetup {}

impl TlsSetup {
    /// Reads the TLS directory of an image the Windows loader has mapped at
    /// `base`, redirects its index-0 TLS loads, and collects its callbacks.
    ///
    /// `original_callbacks` is `AddressOfCallBacks` as the unpatched image
    /// had it (a VA relative to `pe.image_base`).
    ///
    /// # Safety
    /// `base` is the image `pe` describes, mapped by the OS loader, with no
    /// thread running its code yet.
    #[cfg(windows)]
    pub unsafe fn init_for_loaded_image(
        base: *mut u8,
        pe: &Pe64Info,
        original_callbacks: u64,
    ) -> Result<Option<Self>, &'static str> {
        let size = pe.size_of_image as usize;
        let rva = pe.tls_dir_rva as usize;
        if rva == 0 || rva + std::mem::size_of::<ImageTlsDirectory64>() > size {
            return Ok(None);
        }
        let base_addr = base as u64;
        let end_addr = base_addr + size as u64;
        let in_image = |va: u64, len: u64| va >= base_addr && va.saturating_add(len) <= end_addr;

        // SAFETY: the directory lies inside the mapped image (checked above);
        // the OS loader has relocated its pointers.
        let dir = unsafe { (base.add(rva) as *const ImageTlsDirectory64).read_unaligned() };

        let os_index = if in_image(dir.address_of_index, 4) {
            // SAFETY: `_tls_index` is inside the image; the loader wrote it.
            unsafe { (dir.address_of_index as *const u32).read_unaligned() }
        } else {
            0
        };

        let mut thunks = std::ptr::null_mut();
        let mut patched_loads = 0;
        if os_index != 0 {
            // SAFETY: forwarded from the caller.
            match unsafe { redirect_gs_loads(base, pe, os_index) } {
                Some((page, count)) => {
                    thunks = page;
                    patched_loads = count;
                }
                None => return Err("no free memory within 2 GiB of the image for TLS thunks"),
            }
            // CRT code reads `_tls_index` and indexes the (now redirected)
            // array pointer with it; 0 keeps it on the image's own entry.
            // SAFETY: checked inside the image above; `.data` is writable.
            unsafe { (dir.address_of_index as *mut u32).write_unaligned(0) };
        }

        let mut callbacks = Vec::new();
        if original_callbacks != 0 {
            let array = original_callbacks
                .wrapping_sub(pe.image_base)
                .wrapping_add(base_addr);
            let mut at = array;
            while in_image(at, 8) {
                // SAFETY: inside the image; entries were relocated by the OS.
                let cb = unsafe { (at as *const u64).read_unaligned() };
                if cb == 0 {
                    break;
                }
                callbacks.push(cb as usize);
                at += 8;
            }
        }

        Ok(Some(Self {
            os_index,
            patched_loads,
            raw_template_bytes: dir
                .end_address_of_raw_data
                .saturating_sub(dir.start_address_of_raw_data)
                as usize,
            zero_fill_bytes: dir.size_of_zero_fill as usize,
            callbacks,
            thunks,
        }))
    }
}

impl Drop for TlsSetup {
    fn drop(&mut self) {
        #[cfg(windows)]
        if !self.thunks.is_null() {
            // SAFETY: allocated by `alloc_near`; the image is unloaded first.
            unsafe { VirtualFree(self.thunks as *mut c_void, 0, MEM_RELEASE) };
        }
    }
}

/// Finds `mov r64, gs:[0x58]` in the image's code. Returns `(offset, reg)`.
pub(crate) fn find_gs_tls_loads(code: &[u8]) -> Vec<(usize, u8)> {
    let mut found = Vec::new();
    let mut i = 0;
    while i + GS_LOAD_LEN <= code.len() {
        let w = &code[i..i + GS_LOAD_LEN];
        let rex_ok = w[1] & 0xFB == 0x48; // REX.W, optionally REX.R; no X/B
        if w[0] == 0x65
            && rex_ok
            && w[2] == 0x8B
            && w[3] & 0xC7 == 0x04
            && w[4] == 0x25
            && w[5..9] == [0x58, 0, 0, 0]
        {
            let reg = ((w[3] >> 3) & 7) | if w[1] & 0x04 != 0 { 8 } else { 0 };
            // `rsp` as the target would make the call below meaningless.
            if reg != 4 {
                found.push((i, reg));
                i += GS_LOAD_LEN;
                continue;
            }
        }
        i += 1;
    }
    found
}

/// Thunk for one register: `mov reg, gs:[0x58]; lea reg, [reg + index*8]; ret`.
/// `lea` leaves the flags alone, as the original `mov` did.
fn encode_thunk(reg: u8, index: u32) -> Vec<u8> {
    let (rex_rb, low) = (if reg >= 8 { 0x05 } else { 0 }, reg & 7);
    let mut t = vec![
        0x65,
        0x48 | (rex_rb & 0x04),
        0x8B,
        0x04 | (low << 3),
        0x25,
        0x58,
        0,
        0,
        0,
    ];
    t.extend_from_slice(&[0x48 | rex_rb, 0x8D, 0x80 | (low << 3) | low]);
    if low == 4 {
        t.push(0x24); // SIB: base r12, no index
    }
    t.extend_from_slice(&(index.wrapping_mul(8)).to_le_bytes());
    t.push(0xC3);
    t
}

/// # Safety
/// `base` is the mapped image `pe` describes, with no thread running it.
#[cfg(windows)]
unsafe fn redirect_gs_loads(base: *mut u8, pe: &Pe64Info, index: u32) -> Option<(*mut u8, usize)> {
    let size = pe.size_of_image as usize;
    let code_sections: Vec<(usize, usize)> = pe
        .sections
        .iter()
        .filter(|s| s.characteristics & SCN_MEM_EXECUTE != 0)
        .map(|s| {
            let va = s.virtual_addr as usize;
            (va, (s.virtual_size as usize).min(size.saturating_sub(va)))
        })
        .filter(|&(_, len)| len > 0)
        .collect();

    // SAFETY: each span lies inside the mapped image and is readable.
    let sites: Vec<Vec<(usize, u8)>> = code_sections
        .iter()
        .map(|&(va, len)| {
            find_gs_tls_loads(unsafe { std::slice::from_raw_parts(base.add(va), len) })
        })
        .collect();
    let total: usize = sites.iter().map(Vec::len).sum();
    if total == 0 {
        return Some((std::ptr::null_mut(), 0));
    }

    // SAFETY: plain allocation near the image.
    let page = unsafe { alloc_near(base as usize, size, THUNK_PAGE) }?;
    for reg in 0..16u8 {
        let thunk = encode_thunk(reg, index);
        // SAFETY: the page is THUNK_PAGE bytes of fresh RW memory.
        unsafe {
            std::ptr::copy_nonoverlapping(
                thunk.as_ptr(),
                page.add(reg as usize * THUNK_SLOT),
                thunk.len(),
            );
        }
    }
    let mut old = 0;
    // SAFETY: the page is ours.
    unsafe {
        VirtualProtect(page as *mut c_void, THUNK_PAGE, PAGE_EXECUTE_READ, &mut old);
        FlushInstructionCache(GetCurrentProcess(), page as *const c_void, THUNK_PAGE);
    }

    let mut patched = 0;
    for (&(va, len), sites) in code_sections.iter().zip(&sites) {
        if sites.is_empty() {
            continue;
        }
        // SAFETY: the section is part of the image; protection is restored below.
        let section = unsafe { base.add(va) };
        let mut prev = 0;
        if unsafe {
            VirtualProtect(
                section as *mut c_void,
                len,
                PAGE_EXECUTE_READWRITE,
                &mut prev,
            )
        } == 0
        {
            continue;
        }
        for &(off, reg) in sites {
            let site = section as usize + off;
            let target = page as usize + reg as usize * THUNK_SLOT;
            let Ok(rel) = i32::try_from(target as i64 - (site as i64 + 5)) else {
                continue;
            };
            // `call rel32` then a 4-byte nop, the same 9 bytes as the load.
            let mut call = [0xE8, 0, 0, 0, 0, 0x0F, 0x1F, 0x40, 0x00];
            call[1..5].copy_from_slice(&rel.to_le_bytes());
            // SAFETY: `site..site+9` is the matched load inside this section.
            unsafe { std::ptr::copy_nonoverlapping(call.as_ptr(), site as *mut u8, GS_LOAD_LEN) };
            patched += 1;
        }
        // SAFETY: restoring what VirtualProtect reported.
        unsafe {
            VirtualProtect(section as *mut c_void, len, prev, &mut prev);
            FlushInstructionCache(GetCurrentProcess(), section as *const c_void, len);
        }
    }
    Some((page, patched))
}

/// Allocates `len` bytes of RW memory within ±2 GiB of the image, so a
/// `call rel32` from any of its code reaches it.
///
/// # Safety
/// Plain Win32 calls; the caller owns the returned memory.
#[cfg(windows)]
unsafe fn alloc_near(base: usize, size: usize, len: usize) -> Option<*mut u8> {
    const REACH: usize = 0x7000_0000;
    let above = (base + size).div_ceil(GRANULARITY) * GRANULARITY;
    let below = base & !(GRANULARITY - 1);
    let mut step = 0;
    while step <= REACH {
        let down = below.checked_sub(GRANULARITY + step).filter(|&a| a != 0);
        for addr in std::iter::once(above + step).chain(down) {
            // SAFETY: asking for a specific free range; null if taken.
            let p = unsafe {
                VirtualAlloc(addr as *mut c_void, len, MEM_COMMIT_RESERVE, PAGE_READWRITE)
            };
            if !p.is_null() {
                return Some(p as *mut u8);
            }
        }
        step += GRANULARITY;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_gs_tls_loads_for_low_and_high_registers() {
        // mov rax, gs:[58h] ; mov r9, gs:[58h] ; mov rsp, gs:[58h] (skipped)
        let code = [
            0x90, 0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0, //
            0x65, 0x4C, 0x8B, 0x0C, 0x25, 0x58, 0, 0, 0, //
            0x65, 0x48, 0x8B, 0x24, 0x25, 0x58, 0, 0, 0,
        ];
        assert_eq!(find_gs_tls_loads(&code), vec![(1, 0), (10, 9)]);
    }

    #[test]
    fn thunks_encode_lea_for_every_register() {
        assert_eq!(
            encode_thunk(0, 3),
            [
                0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0, 0x48, 0x8D, 0x80, 24, 0, 0, 0, 0xC3
            ]
        );
        // r12 needs a SIB byte; r13 does not.
        assert_eq!(
            &encode_thunk(12, 1)[9..],
            [0x4D, 0x8D, 0xA4, 0x24, 8, 0, 0, 0, 0xC3]
        );
        assert_eq!(
            &encode_thunk(13, 1)[9..],
            [0x4D, 0x8D, 0xAD, 8, 0, 0, 0, 0xC3]
        );
        assert!((0..16).all(|r| encode_thunk(r, u32::MAX).len() <= THUNK_SLOT));
    }
}
