//! Clean-room `XboxBvhBuilder.dll` HLE bridge.
//!
//! Retail Xbox Series titles import 7 C++ exports from `XboxBvhBuilder.dll` to
//! build, copy, query prebuild sizes, and rehydrate static BVH4 acceleration
//! structures. This module provides the ABI-compatible entry points so titles
//! can run with native Strata acceleration-structure management (or opt-in
//! `STRATA_NATIVE_RT=1`) without crashing on missing C++ mangled symbols.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const S_OK: i32 = 0;

const SYM_CTOR: &str = "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z";
const SYM_DTOR: &str = "??1XboxBvhBuilder@@QEAA@XZ";
const SYM_BUILD_BATCHED_2X: &str = "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z";
const SYM_COPY: &str = "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z";
const SYM_PREBUILD: &str = "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z";
const SYM_REHYDRATE_BATCHED_2X: &str = "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z";
const SYM_SET_64BIT_MORTON: &str = "?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z";

static USE_64BIT_MORTON: AtomicBool = AtomicBool::new(false);
static BUILD_CALLS: AtomicU64 = AtomicU64::new(0);
static REHYDRATE_CALLS: AtomicU64 = AtomicU64::new(0);
static COPY_CALLS: AtomicU64 = AtomicU64::new(0);

#[repr(C)]
pub struct PrebuildInfo {
    pub result_data_max_size_in_bytes: u64,
    pub scratch_data_size_in_bytes: u64,
    pub update_scratch_data_size_in_bytes: u64,
}

#[repr(C)]
pub struct BuildInputsHeader {
    pub as_type: u32,
    pub flags: u32,
    pub num_descs: u32,
    pub descs_layout: u32,
}

unsafe extern "system" fn bvh_ctor(
    this: *mut c_void,
    _device: *mut c_void,
    _use_async: u8,
) -> *mut c_void {
    this
}

unsafe extern "system" fn bvh_dtor(_this: *mut c_void) {}

unsafe extern "system" fn bvh_build_batched_2x(
    _this: *mut c_void,
    _cmd_list: *const c_void,
    count: u32,
    _descs: *const c_void,
) {
    BUILD_CALLS.fetch_add(u64::from(count), Ordering::Relaxed);
}

unsafe extern "system" fn bvh_copy(
    _this: *mut c_void,
    _cmd_list: *const c_void,
    _dst_va: u64,
    _src_va: u64,
    _mode: u32,
) {
    COPY_CALLS.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "system" fn bvh_prebuild(
    _this: *mut c_void,
    inputs: *const BuildInputsHeader,
    _stream_desc: *const c_void,
    out: *mut PrebuildInfo,
) {
    if out.is_null() {
        return;
    }
    let (prims, leaf) = if inputs.is_null() {
        (1u64, crate::d3d12_rt::TRIANGLE_BYTES)
    } else {
        let h = unsafe { &*inputs };
        let leaf = if h.as_type == 0 {
            crate::d3d12_rt::INSTANCE_BYTES
        } else {
            crate::d3d12_rt::TRIANGLE_BYTES
        };
        (u64::from(h.num_descs.max(1)), leaf)
    };
    let result_bytes = crate::d3d12_rt::max_size(prims, leaf).next_multiple_of(256);
    let scratch_bytes = (result_bytes / 2).max(65_536).next_multiple_of(256);
    unsafe {
        *out = PrebuildInfo {
            result_data_max_size_in_bytes: result_bytes,
            scratch_data_size_in_bytes: scratch_bytes,
            update_scratch_data_size_in_bytes: scratch_bytes,
        };
    }
}

unsafe extern "system" fn bvh_rehydrate_batched_2x(
    _this: *mut c_void,
    _cmd_list: *const c_void,
    count: u32,
    _descs: *const c_void,
    _src_vas: *const u64,
) -> i32 {
    REHYDRATE_CALLS.fetch_add(u64::from(count), Ordering::Relaxed);
    S_OK
}

unsafe extern "system" fn bvh_set_64bit_morton(_this: *mut c_void, enable: u8) {
    USE_64BIT_MORTON.store(enable != 0, Ordering::Relaxed);
}

pub fn resolve_bvh_builder_hle(dll_lower: &str, func_name: &str) -> Option<usize> {
    if dll_lower != "xboxbvhbuilder.dll" {
        return None;
    }
    Some(match func_name {
        SYM_CTOR => bvh_ctor as *const () as usize,
        SYM_DTOR => bvh_dtor as *const () as usize,
        SYM_BUILD_BATCHED_2X => bvh_build_batched_2x as *const () as usize,
        SYM_COPY => bvh_copy as *const () as usize,
        SYM_PREBUILD => bvh_prebuild as *const () as usize,
        SYM_REHYDRATE_BATCHED_2X => bvh_rehydrate_batched_2x as *const () as usize,
        SYM_SET_64BIT_MORTON => bvh_set_64bit_morton as *const () as usize,
        _ => return None,
    })
}

pub fn stats() -> (u64, u64, u64, bool) {
    (
        BUILD_CALLS.load(Ordering::Relaxed),
        REHYDRATE_CALLS.load(Ordering::Relaxed),
        COPY_CALLS.load(Ordering::Relaxed),
        USE_64BIT_MORTON.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_all_seven_exports_and_computes_prebuild_info() {
        for sym in [
            SYM_CTOR,
            SYM_DTOR,
            SYM_BUILD_BATCHED_2X,
            SYM_COPY,
            SYM_PREBUILD,
            SYM_REHYDRATE_BATCHED_2X,
            SYM_SET_64BIT_MORTON,
        ] {
            assert!(resolve_bvh_builder_hle("xboxbvhbuilder.dll", sym).is_some());
        }
        let inputs = BuildInputsHeader {
            as_type: 1,
            flags: 0,
            num_descs: 128,
            descs_layout: 0,
        };
        let mut info = PrebuildInfo {
            result_data_max_size_in_bytes: 0,
            scratch_data_size_in_bytes: 0,
            update_scratch_data_size_in_bytes: 0,
        };
        unsafe { bvh_prebuild(std::ptr::null_mut(), &inputs, std::ptr::null(), &mut info) };
        assert!(info.result_data_max_size_in_bytes >= 256);
        assert_eq!(info.result_data_max_size_in_bytes % 256, 0);
        unsafe { bvh_set_64bit_morton(std::ptr::null_mut(), 1) };
        assert!(stats().3);
    }
}
