//! Standard Direct3D 12 COM vtable slot layouts matching the published
//! Microsoft `d3d12.h` ABI slot indices (`IUnknown` -> `ID3D12Object` ->
//! `ID3D12DeviceChild` -> `ID3D12Fence` / `ID3D12Resource` /
//! `ID3D12CommandQueue` / `ID3D12GraphicsCommandList` / `ID3D12Device`).
//!
//! Each vtable is a fixed-size `[usize; N]` slot table with documented slot
//! index constants so retail callers invoking virtual methods by slot index
//! dispatch to the exact expected entry point.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const S_OK: i32 = 0;
const E_NOINTERFACE: i32 = 0x8000_4002u32 as i32;

/// `IUnknown` slots (0..3).
pub const SLOT_QUERY_INTERFACE: usize = 0;
pub const SLOT_ADD_REF: usize = 1;
pub const SLOT_RELEASE: usize = 2;

/// `ID3D12Object` slots (3..7).
pub const SLOT_GET_PRIVATE_DATA: usize = 3;
pub const SLOT_SET_PRIVATE_DATA: usize = 4;
pub const SLOT_SET_PRIVATE_DATA_INTERFACE: usize = 5;
pub const SLOT_SET_NAME: usize = 6;

/// `ID3D12DeviceChild` slot (7).
pub const SLOT_GET_DEVICE: usize = 7;

/// `ID3D12Fence` slots (8..11).
pub const FENCE_SLOT_GET_COMPLETED_VALUE: usize = 8;
pub const FENCE_SLOT_SET_EVENT_ON_COMPLETION: usize = 9;
pub const FENCE_SLOT_SIGNAL: usize = 10;
pub const FENCE_VTABLE_LEN: usize = 11;

/// `ID3D12Resource` slots (8..15).
pub const RESOURCE_SLOT_MAP: usize = 8;
pub const RESOURCE_SLOT_UNMAP: usize = 9;
pub const RESOURCE_SLOT_GET_DESC: usize = 10;
pub const RESOURCE_SLOT_GET_GPU_VA: usize = 11;
pub const RESOURCE_SLOT_WRITE_TO_SUBRESOURCE: usize = 12;
pub const RESOURCE_SLOT_READ_FROM_SUBRESOURCE: usize = 13;
pub const RESOURCE_SLOT_GET_HEAP_PROPERTIES: usize = 14;
pub const RESOURCE_VTABLE_LEN: usize = 15;

/// `ID3D12CommandQueue` slots (8..19).
pub const QUEUE_SLOT_UPDATE_TILE_MAPPINGS: usize = 8;
pub const QUEUE_SLOT_COPY_TILE_MAPPINGS: usize = 9;
pub const QUEUE_SLOT_EXECUTE_COMMAND_LISTS: usize = 10;
pub const QUEUE_SLOT_SET_MARKER: usize = 11;
pub const QUEUE_SLOT_BEGIN_EVENT: usize = 12;
pub const QUEUE_SLOT_END_EVENT: usize = 13;
pub const QUEUE_SLOT_SIGNAL: usize = 14;
pub const QUEUE_SLOT_WAIT: usize = 15;
pub const QUEUE_SLOT_GET_TIMESTAMP_FREQUENCY: usize = 16;
pub const QUEUE_SLOT_GET_CLOCK_CALIBRATION: usize = 17;
pub const QUEUE_SLOT_GET_DESC: usize = 18;
pub const QUEUE_VTABLE_LEN: usize = 19;

/// `ID3D12GraphicsCommandList` key slots (8..60).
pub const CMDLIST_SLOT_GET_TYPE: usize = 8;
pub const CMDLIST_SLOT_CLOSE: usize = 9;
pub const CMDLIST_SLOT_RESET: usize = 10;
pub const CMDLIST_SLOT_CLEAR_STATE: usize = 11;
pub const CMDLIST_SLOT_DRAW_INSTANCED: usize = 12;
pub const CMDLIST_SLOT_DRAW_INDEXED_INSTANCED: usize = 13;
pub const CMDLIST_SLOT_DISPATCH: usize = 14;
pub const CMDLIST_SLOT_COPY_BUFFER_REGION: usize = 15;
pub const CMDLIST_SLOT_COPY_TEXTURE_REGION: usize = 16;
pub const CMDLIST_SLOT_COPY_RESOURCE: usize = 17;
pub const CMDLIST_SLOT_SET_PIPELINE_STATE: usize = 25;
pub const CMDLIST_SLOT_RESOURCE_BARRIER: usize = 26;
pub const CMDLIST_SLOT_SET_DESCRIPTOR_HEAPS: usize = 28;
pub const CMDLIST_SLOT_SET_COMPUTE_ROOT_SIGNATURE: usize = 29;
pub const CMDLIST_SLOT_SET_GRAPHICS_ROOT_SIGNATURE: usize = 30;
pub const CMDLIST_SLOT_SET_COMPUTE_ROOT_32BIT_CONSTANTS: usize = 34;
pub const CMDLIST_SLOT_OM_SET_RENDER_TARGETS: usize = 46;
pub const CMDLIST_SLOT_EXECUTE_INDIRECT: usize = 59;
pub const CMDLIST_VTABLE_LEN: usize = 60;

/// `ID3D12Device` key slots (7..44).
pub const DEVICE_SLOT_GET_NODE_COUNT: usize = 7;
pub const DEVICE_SLOT_CREATE_COMMAND_QUEUE: usize = 8;
pub const DEVICE_SLOT_CREATE_COMMAND_ALLOCATOR: usize = 9;
pub const DEVICE_SLOT_CREATE_GRAPHICS_PSO: usize = 10;
pub const DEVICE_SLOT_CREATE_COMPUTE_PSO: usize = 11;
pub const DEVICE_SLOT_CREATE_COMMAND_LIST: usize = 12;
pub const DEVICE_SLOT_CHECK_FEATURE_SUPPORT: usize = 13;
pub const DEVICE_SLOT_CREATE_DESCRIPTOR_HEAP: usize = 14;
pub const DEVICE_SLOT_GET_DESCRIPTOR_HANDLE_INC: usize = 15;
pub const DEVICE_SLOT_CREATE_ROOT_SIGNATURE: usize = 16;
pub const DEVICE_SLOT_CREATE_COMMITTED_RESOURCE: usize = 27;
pub const DEVICE_SLOT_CREATE_FENCE: usize = 36;
pub const DEVICE_SLOT_GET_DEVICE_REMOVED_REASON: usize = 37;
pub const DEVICE_VTABLE_LEN: usize = 44;

unsafe extern "system" fn stub_ok(_this: *mut c_void) -> i32 {
    S_OK
}

unsafe extern "system" fn stub_void(_this: *mut c_void) {}

/// A single ABI-compatible COM vtable function pointer slot.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct VTableSlot(pub *const ());

// SAFETY: VTableSlot only holds immutable static function pointers.
unsafe impl Sync for VTableSlot {}
unsafe impl Send for VTableSlot {}

/// Standard-ABI `ID3D12Fence` COM object with exact `d3d12.h` slot indices (0..11).
#[repr(C)]
pub struct StandardFence {
    pub vtable: &'static [VTableSlot; FENCE_VTABLE_LEN],
    pub ref_count: AtomicU32,
    pub value: AtomicU64,
}

unsafe extern "system" fn std_fence_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_NOINTERFACE;
    }
    unsafe {
        std_fence_add_ref(this);
        *out = this;
    }
    S_OK
}

unsafe extern "system" fn std_fence_add_ref(this: *mut c_void) -> u32 {
    let f = unsafe { &*(this as *const StandardFence) };
    f.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn std_fence_release(this: *mut c_void) -> u32 {
    let f = unsafe { &*(this as *const StandardFence) };
    let prev = f.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut StandardFence)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn std_fence_get_completed(this: *mut c_void) -> u64 {
    let f = unsafe { &*(this as *const StandardFence) };
    f.value.load(Ordering::Acquire)
}

unsafe extern "system" fn std_fence_signal(this: *mut c_void, val: u64) -> i32 {
    let f = unsafe { &*(this as *const StandardFence) };
    f.value.store(val, Ordering::Release);
    S_OK
}

static STANDARD_FENCE_VTABLE: [VTableSlot; FENCE_VTABLE_LEN] = [
    VTableSlot(std_fence_qi as *const ()),      // 0: QueryInterface
    VTableSlot(std_fence_add_ref as *const ()), // 1: AddRef
    VTableSlot(std_fence_release as *const ()), // 2: Release
    VTableSlot(stub_ok as *const ()),           // 3: GetPrivateData
    VTableSlot(stub_ok as *const ()),           // 4: SetPrivateData
    VTableSlot(stub_ok as *const ()),           // 5: SetPrivateDataInterface
    VTableSlot(stub_ok as *const ()),           // 6: SetName
    VTableSlot(stub_ok as *const ()),           // 7: GetDevice
    VTableSlot(std_fence_get_completed as *const ()), // 8: GetCompletedValue
    VTableSlot(stub_ok as *const ()),           // 9: SetEventOnCompletion
    VTableSlot(std_fence_signal as *const ()),  // 10: Signal
];

impl StandardFence {
    pub fn new_boxed(initial: u64) -> *mut Self {
        Box::into_raw(Box::new(Self {
            vtable: &STANDARD_FENCE_VTABLE,
            ref_count: AtomicU32::new(1),
            value: AtomicU64::new(initial),
        }))
    }
}

/// Build a standard-ABI slot table for `ID3D12CommandQueue` (19 slots) where
/// unimplemented slots safely return `S_OK` / no-op instead of crashing.
pub fn build_standard_queue_vtable(
    qi: *const (),
    add_ref: *const (),
    release: *const (),
    execute_command_lists: *const (),
    signal: *const (),
) -> [VTableSlot; QUEUE_VTABLE_LEN] {
    let mut vt = [VTableSlot(stub_ok as *const ()); QUEUE_VTABLE_LEN];
    vt[SLOT_QUERY_INTERFACE] = VTableSlot(qi);
    vt[SLOT_ADD_REF] = VTableSlot(add_ref);
    vt[SLOT_RELEASE] = VTableSlot(release);
    vt[QUEUE_SLOT_EXECUTE_COMMAND_LISTS] = VTableSlot(execute_command_lists);
    vt[QUEUE_SLOT_SET_MARKER] = VTableSlot(stub_void as *const ());
    vt[QUEUE_SLOT_BEGIN_EVENT] = VTableSlot(stub_void as *const ());
    vt[QUEUE_SLOT_END_EVENT] = VTableSlot(stub_void as *const ());
    vt[QUEUE_SLOT_SIGNAL] = VTableSlot(signal);
    vt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_fence_dispatches_by_exact_d3d12_vtable_slot_indices() {
        let fence = StandardFence::new_boxed(10);
        unsafe {
            let vt = *(*fence).vtable;
            let get_val: unsafe extern "system" fn(*mut c_void) -> u64 =
                std::mem::transmute(vt[FENCE_SLOT_GET_COMPLETED_VALUE].0);
            let signal: unsafe extern "system" fn(*mut c_void, u64) -> i32 =
                std::mem::transmute(vt[FENCE_SLOT_SIGNAL].0);
            let release: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(vt[SLOT_RELEASE].0);

            assert_eq!(get_val(fence.cast()), 10);
            assert_eq!(signal(fence.cast(), 99), S_OK);
            assert_eq!(get_val(fence.cast()), 99);
            assert_eq!(release(fence.cast()), 0);
        }
    }
}
