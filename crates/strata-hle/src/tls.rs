//! Static TLS Directory initialization for mapped PE32+ console executables.
//!
//! A manually mapped PE executable cannot use the host process's static TLS slot 0.
//! This module reads `IMAGE_TLS_DIRECTORY64` from the mapped image, allocates a
//! dedicated host TLS slot (`TlsAlloc`), writes the slot index into `*AddressOfIndex`,
//! and provides per-thread template initialization.

#[cfg(windows)]
unsafe extern "system" {
    fn TlsAlloc() -> u32;
    fn TlsFree(index: u32) -> i32;
    fn TlsSetValue(index: u32, val: *mut core::ffi::c_void) -> i32;
}

const TLS_OUT_OF_INDEXES: u32 = 0xFFFF_FFFF;

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

#[derive(Debug)]
pub struct TlsSetup {
    pub slot_index: u32,
    pub raw_template_bytes: usize,
    pub zero_fill_bytes: usize,
    pub callback_count: usize,
}

impl TlsSetup {
    /// Inspect and initialize the mapped image's TLS directory if present.
    pub unsafe fn init_for_mapped_image(base: *mut u8, size: usize, tls_dir_rva: u32) -> Option<Self> {
        if tls_dir_rva == 0 || (tls_dir_rva as usize) + std::mem::size_of::<ImageTlsDirectory64>() > size {
            return None;
        }
        let dir = unsafe { *(base.add(tls_dir_rva as usize) as *const ImageTlsDirectory64) };
        let raw_len = dir
            .end_address_of_raw_data
            .saturating_sub(dir.start_address_of_raw_data) as usize;

        let slot = unsafe { TlsAlloc() };
        if slot == TLS_OUT_OF_INDEXES {
            return None;
        }

        // Write allocated host TLS slot index into the image's `_tls_index` variable.
        let base_addr = base as u64;
        if dir.address_of_index >= base_addr && dir.address_of_index + 4 <= base_addr + (size as u64) {
            unsafe { *(dir.address_of_index as *mut u32) = slot };
        }

        // Count null-terminated TLS callbacks (without invoking them during map-test).
        let mut callback_count = 0usize;
        if dir.address_of_callbacks >= base_addr && dir.address_of_callbacks < base_addr + (size as u64) {
            let mut ptr = dir.address_of_callbacks as *const u64;
            while (ptr as u64) + 8 <= base_addr + (size as u64) {
                let cb = unsafe { *ptr };
                if cb == 0 {
                    break;
                }
                callback_count += 1;
                ptr = unsafe { ptr.add(1) };
            }
        }

        // Bind a zero-initialized TLS block for the current thread.
        let total = raw_len + (dir.size_of_zero_fill as usize);
        if total > 0 && raw_len > 0 && dir.start_address_of_raw_data >= base_addr
            && dir.end_address_of_raw_data <= base_addr + (size as u64)
        {
            let mut block = vec![0u8; total].into_boxed_slice();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    dir.start_address_of_raw_data as *const u8,
                    block.as_mut_ptr(),
                    raw_len,
                );
                let raw_block = Box::into_raw(block) as *mut core::ffi::c_void;
                TlsSetValue(slot, raw_block);
            }
        }

        Some(Self {
            slot_index: slot,
            raw_template_bytes: raw_len,
            zero_fill_bytes: dir.size_of_zero_fill as usize,
            callback_count,
        })
    }
}

impl Drop for TlsSetup {
    fn drop(&mut self) {
        if self.slot_index != TLS_OUT_OF_INDEXES {
            unsafe {
                TlsFree(self.slot_index);
            }
        }
    }
}
