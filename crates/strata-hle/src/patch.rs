//! Safe load-time code patching and hot write-site / descriptor-site
//! instrumentation registry.

use std::collections::HashSet;
use std::ffi::c_void;

#[cfg(windows)]
unsafe extern "system" {
    fn VirtualProtect(
        addr: *mut c_void,
        size: usize,
        new_protect: u32,
        old_protect: *mut u32,
    ) -> i32;
    fn FlushInstructionCache(process: *mut c_void, base: *const c_void, size: usize) -> i32;
    fn GetCurrentProcess() -> *mut c_void;
}

const PAGE_EXECUTE_READWRITE: u32 = 0x40;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchResult {
    Applied,
    ByteMismatch,
    ProtectFailed,
}

/// Safely patches `replacement` bytes at `addr` only if the existing bytes
/// match `expected` (when non-empty), temporarily granting RWX protection and
/// flushing the CPU instruction cache.
///
/// # Safety
/// `addr` must point to at least `replacement.len()` readable/executable bytes.
pub unsafe fn write_code(addr: *mut u8, expected: &[u8], replacement: &[u8]) -> PatchResult {
    if addr.is_null() || replacement.is_empty() {
        return PatchResult::ByteMismatch;
    }
    if !expected.is_empty() {
        let current = unsafe { std::slice::from_raw_parts(addr, expected.len()) };
        if current != expected {
            return PatchResult::ByteMismatch;
        }
    }
    #[cfg(windows)]
    {
        let mut old = 0u32;
        let ok = unsafe {
            VirtualProtect(
                addr.cast(),
                replacement.len(),
                PAGE_EXECUTE_READWRITE,
                &mut old,
            )
        };
        if ok == 0 {
            return PatchResult::ProtectFailed;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(replacement.as_ptr(), addr, replacement.len());
            let mut dummy = 0u32;
            VirtualProtect(addr.cast(), replacement.len(), old, &mut dummy);
            FlushInstructionCache(GetCurrentProcess(), addr.cast(), replacement.len());
        }
    }
    #[cfg(not(windows))]
    unsafe {
        std::ptr::copy_nonoverlapping(replacement.as_ptr(), addr, replacement.len());
    }
    PatchResult::Applied
}

/// Tracks hot title-code RVAs that repeatedly write unchanged bytes to guarded
/// pages or copy descriptor records into shader-visible heaps.
#[derive(Debug, Default, Clone)]
pub struct SiteTable {
    sites: HashSet<(String, usize)>,
}

impl SiteTable {
    pub fn record(&mut self, module: impl Into<String>, rva: usize) -> bool {
        self.sites.insert((module.into(), rva))
    }

    pub fn contains(&self, module: &str, rva: usize) -> bool {
        self.sites.contains(&(module.to_string(), rva))
    }

    pub fn len(&self) -> usize {
        self.sites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// Parses `<module> <hex-rva>` lines from a write-sites file.
    pub fn parse_lines(text: &str) -> Self {
        let mut t = Self::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            if let (Some(m), Some(hex)) = (parts.next(), parts.next())
                && let Ok(rva) = usize::from_str_radix(hex.trim_start_matches("0x"), 16)
            {
                t.record(m, rva);
            }
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_code_verifies_expected_bytes_and_site_table_parses() {
        let mut buf = [0x90u8, 0x90, 0xC3, 0x00];
        assert_eq!(
            unsafe { write_code(buf.as_mut_ptr(), &[0xAA, 0xBB], &[0xCC, 0xDD]) },
            PatchResult::ByteMismatch
        );
        assert_eq!(
            unsafe { write_code(buf.as_mut_ptr(), &[0x90, 0x90], &[0x31, 0xC0]) },
            PatchResult::Applied
        );
        assert_eq!(&buf[0..3], &[0x31, 0xC0, 0xC3]);

        let t = SiteTable::parse_lines("# hot sites\ntitle.exe 0x1234\ntitle.exe 5678\n");
        assert_eq!(t.len(), 2);
        assert!(t.contains("title.exe", 0x1234));
        assert!(t.contains("title.exe", 0x5678));
    }
}
