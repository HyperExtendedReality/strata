//! Host pages whose CPU writes Windows records (`MEM_WRITE_WATCH`).
//!
//! The console's CPU and GPU share memory; a PC's do not. A Tier 2 mirror
//! keeps the title's bytes in host pages like these and a copy in VRAM:
//! before GPU work only the pages the title wrote since the last look are
//! uploaded, page-exact, instead of hashing or re-sending whole buffers.

use std::ffi::c_void;

#[cfg(windows)]
unsafe extern "system" {
    fn VirtualAlloc(addr: *mut c_void, size: usize, alloc_type: u32, protect: u32) -> *mut c_void;
    fn VirtualFree(addr: *mut c_void, size: usize, free_type: u32) -> i32;
    fn GetWriteWatch(
        flags: u32,
        base: *mut c_void,
        size: usize,
        addresses: *mut *mut c_void,
        count: *mut usize,
        granularity: *mut u32,
    ) -> u32;
    fn ResetWriteWatch(base: *mut c_void, size: usize) -> u32;
}

const MEM_COMMIT_RESERVE_WATCH: u32 = 0x1000 | 0x2000 | 0x20_0000;
const MEM_RELEASE: u32 = 0x8000;
const PAGE_READWRITE: u32 = 0x04;
const WRITE_WATCH_FLAG_RESET: u32 = 1;
pub const PAGE: usize = 4096;

/// A block of write-watched, committed, read-write host memory.
pub struct WatchedPages {
    ptr: *mut u8,
    size: usize,
}

// SAFETY: plain process memory; access is synchronised by the owner.
unsafe impl Send for WatchedPages {}

impl WatchedPages {
    /// `size` rounded up to whole pages, zeroed.
    #[cfg(windows)]
    pub fn new(size: usize) -> Option<Self> {
        let size = size.max(1).next_multiple_of(PAGE);
        // SAFETY: a fresh allocation.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                size,
                MEM_COMMIT_RESERVE_WATCH,
                PAGE_READWRITE,
            )
        };
        (!ptr.is_null()).then(|| Self {
            ptr: ptr as *mut u8,
            size,
        })
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Byte ranges `(offset, len)` the CPU wrote since the last reset,
    /// adjacent pages joined. `reset` clears the record as it reads it.
    #[cfg(windows)]
    pub fn dirty_ranges(&self, reset: bool) -> Vec<(usize, usize)> {
        let pages = self.size / PAGE;
        let mut addrs: Vec<*mut c_void> = vec![std::ptr::null_mut(); pages];
        let mut count = pages;
        let mut granularity = 0u32;
        // SAFETY: `addrs` holds `count` entries; the range is ours.
        let failed = unsafe {
            GetWriteWatch(
                if reset { WRITE_WATCH_FLAG_RESET } else { 0 },
                self.ptr as *mut c_void,
                self.size,
                addrs.as_mut_ptr(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        if failed {
            // Unknown: treat everything as written.
            return vec![(0, self.size)];
        }
        let page = (granularity as usize).max(PAGE);
        let mut offsets: Vec<usize> = addrs[..count]
            .iter()
            .map(|&a| a as usize - self.ptr as usize)
            .collect();
        offsets.sort_unstable();
        join_pages(&offsets, page)
    }

    /// Forgets writes in `offset..offset+len` (e.g. bytes Strata itself
    /// copied in from the GPU).
    #[cfg(windows)]
    pub fn reset_range(&self, offset: usize, len: usize) {
        let start = offset / PAGE * PAGE;
        let end = (offset + len).next_multiple_of(PAGE).min(self.size);
        if start < end {
            // SAFETY: a sub-range of our allocation.
            unsafe { ResetWriteWatch(self.ptr.add(start) as *mut c_void, end - start) };
        }
    }
}

impl Drop for WatchedPages {
    fn drop(&mut self) {
        #[cfg(windows)]
        // SAFETY: allocated by `new`.
        unsafe {
            VirtualFree(self.ptr as *mut c_void, 0, MEM_RELEASE);
        }
    }
}

/// Sorted page offsets -> `(offset, len)` runs of adjacent pages.
pub fn join_pages(offsets: &[usize], page: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for &off in offsets {
        match out.last_mut() {
            Some((start, len)) if *start + *len == off => *len += page,
            _ => out.push((off, page)),
        }
    }
    out
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn reports_exactly_the_pages_written() {
        let pages = WatchedPages::new(8 * PAGE).unwrap();
        assert!(pages.dirty_ranges(true).is_empty() || pages.dirty_ranges(true).is_empty());
        unsafe {
            *pages.as_ptr() = 1;
            *pages.as_ptr().add(PAGE + 5) = 2;
            *pages.as_ptr().add(5 * PAGE) = 3;
        }
        assert_eq!(pages.dirty_ranges(false), [(0, 2 * PAGE), (5 * PAGE, PAGE)]);
        pages.reset_range(5 * PAGE, 1);
        assert_eq!(pages.dirty_ranges(true), [(0, 2 * PAGE)]);
        assert!(pages.dirty_ranges(false).is_empty());
    }

    #[test]
    fn joins_adjacent_pages_only() {
        assert_eq!(
            join_pages(&[0, 4096, 12288], 4096),
            [(0, 8192), (12288, 4096)]
        );
        assert!(join_pages(&[], 4096).is_empty());
    }
}
