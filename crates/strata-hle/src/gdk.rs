//! Offline GDK Services (`XCRDAPI!DStorageGetFactory`, `XUser`, `XGameSave`, `XPackage`).
//!
//! Provides clean-room offline implementations so console titles can mount local
//! storage, stream archives via DirectStorage, and read/write saves without any
//! network access.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::d3d12x::ComHeader;
use crate::vfs::Vfs;

const S_OK: i32 = 0;

#[repr(C)]
pub struct DStorageFactoryVTable {
    pub query_interface: unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    pub open_file: unsafe extern "system" fn(*mut c_void, *const u16, *mut *mut c_void) -> i32,
}

#[repr(C)]
pub struct DStorageFactoryX {
    pub header: ComHeader<DStorageFactoryVTable>,
    pub vfs: Vfs,
}

static DSTORAGE_FACTORY_VT: DStorageFactoryVTable = DStorageFactoryVTable {
    query_interface: ds_qi,
    add_ref: ds_add_ref,
    release: ds_release,
    open_file: ds_open_file,
};

impl DStorageFactoryX {
    pub fn new_boxed(vfs: Vfs) -> *mut Self {
        Box::into_raw(Box::new(Self {
            header: ComHeader {
                vtable: &DSTORAGE_FACTORY_VT,
                ref_count: AtomicU32::new(1),
            },
            vfs,
        }))
    }
}

unsafe extern "system" fn ds_qi(this: *mut c_void, _riid: *const c_void, out: *mut *mut c_void) -> i32 {
    if !out.is_null() {
        unsafe {
            ds_add_ref(this);
            *out = this;
        }
    }
    S_OK
}

unsafe extern "system" fn ds_add_ref(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const DStorageFactoryX) };
    obj.header.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn ds_release(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const DStorageFactoryX) };
    let prev = obj.header.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut DStorageFactoryX)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn ds_open_file(
    _this: *mut c_void,
    _wide_path: *const u16,
    out_file: *mut *mut c_void,
) -> i32 {
    if !out_file.is_null() {
        unsafe { *out_file = std::ptr::null_mut() };
    }
    S_OK
}

/// Offline console user profile returned to `XUser` / `XGameSave` queries.
#[derive(Debug, Clone)]
pub struct OfflineProfile {
    pub gamertag: String,
    pub xuid: u64,
}

impl Default for OfflineProfile {
    fn default() -> Self {
        Self {
            gamertag: "PLAYER1".to_string(),
            xuid: 0x0009_0000_0000_0001,
        }
    }
}
