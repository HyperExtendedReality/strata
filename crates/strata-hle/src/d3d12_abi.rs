//! The title-facing Direct3D 12 objects: COM objects whose vtables follow
//! the published `d3d12.h` slot order (`IUnknown` → `ID3D12Object` →
//! `ID3D12DeviceChild` → each interface and its numbered successors), so
//! titles calling methods by slot reach real implementations.
//!
//! Coverage: `ID3D12Device` through `ID3D12Device5` (65 slots),
//! `ID3D12GraphicsCommandList` through `ID3D12GraphicsCommandList6` (80
//! slots), command queues, fences (with events and waits), resources
//! (committed, placed, reserved), descriptor heaps, root signatures,
//! pipeline states (desc and stream forms), command allocators, heaps, query
//! heaps and command signatures.
//!
//! No-API model: every resource, heap and descriptor lives in title memory
//! the GPU addresses directly (UMA, 64-bit BDA). Views are console
//! descriptor dwords written into heap memory (see [`crate::d3d12_types`]);
//! command lists record [`ThinCommand`]s; queues hand them to the engine
//! ([`crate::d3d12x::D3D12CommandQueueX`]), which runs translated shaders
//! with dynamic rendering, one bindless set and push constants. Nothing
//! Vulkan exists per D3D12 object.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use strata_gpu::graphics::GraphicsPsoState;
use strata_gpu::runner::ComputeRunner;
use strata_gpu::texture::TextureCache;
use strata_gpu::thin_list::{
    AccelBuild, AccelGeometry, ClearAspects, IndirectArg, IndirectLayout, PipelineKey, Surface,
    TextureCopy, ThinCommand, ThinCommandList,
};
use strata_gpu::uma::UmaManager;
use strata_shader::descriptor::ImageKind;

use crate::d3d12_types::*;
use crate::d3d12x::{D3D12CommandQueueX, PipelineTable, SharedTextures};

pub const S_OK: i32 = 0;
pub const E_NOINTERFACE: i32 = 0x8000_4002u32 as i32;
pub const E_NOTIMPL: i32 = 0x8000_4001u32 as i32;
const E_FAIL: i32 = 0x8000_4005u32 as i32;
pub const E_INVALIDARG: i32 = 0x8007_0057u32 as i32;
pub const E_OUTOFMEMORY: i32 = 0x8007_000Eu32 as i32;
pub const DXGI_ERROR_NOT_FOUND: i32 = 0x887A_0002u32 as i32;
pub const DXGI_ERROR_UNSUPPORTED: i32 = 0x887A_0004u32 as i32;

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
pub const QUEUE_SLOT_PRESENT_X: usize = 32;
pub const QUEUE_VTABLE_LEN: usize = QUEUE_SLOT_PRESENT_X + 1;

/// `ID3D12GraphicsCommandList` slots (8..60), then `…List1`..`…List6`.
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
pub const CMDLIST_SLOT_COPY_TILES: usize = 18;
pub const CMDLIST_SLOT_RESOLVE_SUBRESOURCE: usize = 19;
pub const CMDLIST_SLOT_IA_SET_PRIMITIVE_TOPOLOGY: usize = 20;
pub const CMDLIST_SLOT_RS_SET_VIEWPORTS: usize = 21;
pub const CMDLIST_SLOT_RS_SET_SCISSOR_RECTS: usize = 22;
pub const CMDLIST_SLOT_OM_SET_BLEND_FACTOR: usize = 23;
pub const CMDLIST_SLOT_OM_SET_STENCIL_REF: usize = 24;
pub const CMDLIST_SLOT_SET_PIPELINE_STATE: usize = 25;
pub const CMDLIST_SLOT_RESOURCE_BARRIER: usize = 26;
pub const CMDLIST_SLOT_EXECUTE_BUNDLE: usize = 27;
pub const CMDLIST_SLOT_SET_DESCRIPTOR_HEAPS: usize = 28;
pub const CMDLIST_SLOT_SET_COMPUTE_ROOT_SIGNATURE: usize = 29;
pub const CMDLIST_SLOT_SET_GRAPHICS_ROOT_SIGNATURE: usize = 30;
pub const CMDLIST_SLOT_SET_COMPUTE_ROOT_DESCRIPTOR_TABLE: usize = 31;
pub const CMDLIST_SLOT_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE: usize = 32;
pub const CMDLIST_SLOT_SET_COMPUTE_ROOT_32BIT_CONSTANT: usize = 33;
pub const CMDLIST_SLOT_SET_GRAPHICS_ROOT_32BIT_CONSTANT: usize = 34;
pub const CMDLIST_SLOT_SET_COMPUTE_ROOT_32BIT_CONSTANTS: usize = 35;
pub const CMDLIST_SLOT_SET_GRAPHICS_ROOT_32BIT_CONSTANTS: usize = 36;
pub const CMDLIST_SLOT_SET_COMPUTE_ROOT_CBV: usize = 37;
pub const CMDLIST_SLOT_SET_GRAPHICS_ROOT_CBV: usize = 38;
pub const CMDLIST_SLOT_SET_COMPUTE_ROOT_SRV: usize = 39;
pub const CMDLIST_SLOT_SET_GRAPHICS_ROOT_SRV: usize = 40;
pub const CMDLIST_SLOT_SET_COMPUTE_ROOT_UAV: usize = 41;
pub const CMDLIST_SLOT_SET_GRAPHICS_ROOT_UAV: usize = 42;
pub const CMDLIST_SLOT_IA_SET_INDEX_BUFFER: usize = 43;
pub const CMDLIST_SLOT_IA_SET_VERTEX_BUFFERS: usize = 44;
pub const CMDLIST_SLOT_SO_SET_TARGETS: usize = 45;
pub const CMDLIST_SLOT_OM_SET_RENDER_TARGETS: usize = 46;
pub const CMDLIST_SLOT_CLEAR_DEPTH_STENCIL_VIEW: usize = 47;
pub const CMDLIST_SLOT_CLEAR_RENDER_TARGET_VIEW: usize = 48;
pub const CMDLIST_SLOT_CLEAR_UAV_UINT: usize = 49;
pub const CMDLIST_SLOT_CLEAR_UAV_FLOAT: usize = 50;
pub const CMDLIST_SLOT_DISCARD_RESOURCE: usize = 51;
pub const CMDLIST_SLOT_BEGIN_QUERY: usize = 52;
pub const CMDLIST_SLOT_END_QUERY: usize = 53;
pub const CMDLIST_SLOT_RESOLVE_QUERY_DATA: usize = 54;
pub const CMDLIST_SLOT_SET_PREDICATION: usize = 55;
pub const CMDLIST_SLOT_SET_MARKER: usize = 56;
pub const CMDLIST_SLOT_BEGIN_EVENT: usize = 57;
pub const CMDLIST_SLOT_END_EVENT: usize = 58;
pub const CMDLIST_SLOT_EXECUTE_INDIRECT: usize = 59;
pub const CMDLIST_SLOT_ATOMIC_COPY_BUFFER_UINT: usize = 60;
pub const CMDLIST_SLOT_ATOMIC_COPY_BUFFER_UINT64: usize = 61;
pub const CMDLIST_SLOT_OM_SET_DEPTH_BOUNDS: usize = 62;
pub const CMDLIST_SLOT_SET_SAMPLE_POSITIONS: usize = 63;
pub const CMDLIST_SLOT_RESOLVE_SUBRESOURCE_REGION: usize = 64;
pub const CMDLIST_SLOT_SET_VIEW_INSTANCE_MASK: usize = 65;
pub const CMDLIST_SLOT_WRITE_BUFFER_IMMEDIATE: usize = 66;
pub const CMDLIST_SLOT_SET_PROTECTED_RESOURCE_SESSION: usize = 67;
pub const CMDLIST_SLOT_BEGIN_RENDER_PASS: usize = 68;
pub const CMDLIST_SLOT_END_RENDER_PASS: usize = 69;
pub const CMDLIST_SLOT_INITIALIZE_META_COMMAND: usize = 70;
pub const CMDLIST_SLOT_EXECUTE_META_COMMAND: usize = 71;
pub const CMDLIST_SLOT_BUILD_RAYTRACING_AS: usize = 72;
pub const CMDLIST_SLOT_EMIT_RAYTRACING_AS_POSTBUILD_INFO: usize = 73;
pub const CMDLIST_SLOT_COPY_RAYTRACING_AS: usize = 74;
pub const CMDLIST_SLOT_SET_PIPELINE_STATE1: usize = 75;
pub const CMDLIST_SLOT_DISPATCH_RAYS: usize = 76;
pub const CMDLIST_SLOT_RS_SET_SHADING_RATE: usize = 77;
pub const CMDLIST_SLOT_RS_SET_SHADING_RATE_IMAGE: usize = 78;
pub const CMDLIST_SLOT_DISPATCH_MESH: usize = 79;
pub const CMDLIST_VTABLE_LEN: usize = 80;

/// `ID3D12Device` slots (7..44), then `ID3D12Device1`..`ID3D12Device5`.
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
pub const DEVICE_SLOT_CREATE_CBV: usize = 17;
pub const DEVICE_SLOT_CREATE_SRV: usize = 18;
pub const DEVICE_SLOT_CREATE_UAV: usize = 19;
pub const DEVICE_SLOT_CREATE_RTV: usize = 20;
pub const DEVICE_SLOT_CREATE_DSV: usize = 21;
pub const DEVICE_SLOT_CREATE_SAMPLER: usize = 22;
pub const DEVICE_SLOT_COPY_DESCRIPTORS: usize = 23;
pub const DEVICE_SLOT_COPY_DESCRIPTORS_SIMPLE: usize = 24;
pub const DEVICE_SLOT_GET_RESOURCE_ALLOCATION_INFO: usize = 25;
pub const DEVICE_SLOT_GET_CUSTOM_HEAP_PROPERTIES: usize = 26;
pub const DEVICE_SLOT_CREATE_COMMITTED_RESOURCE: usize = 27;
pub const DEVICE_SLOT_CREATE_HEAP: usize = 28;
pub const DEVICE_SLOT_CREATE_PLACED_RESOURCE: usize = 29;
pub const DEVICE_SLOT_CREATE_RESERVED_RESOURCE: usize = 30;
pub const DEVICE_SLOT_CREATE_SHARED_HANDLE: usize = 31;
pub const DEVICE_SLOT_OPEN_SHARED_HANDLE: usize = 32;
pub const DEVICE_SLOT_OPEN_SHARED_HANDLE_BY_NAME: usize = 33;
pub const DEVICE_SLOT_MAKE_RESIDENT: usize = 34;
pub const DEVICE_SLOT_EVICT: usize = 35;
pub const DEVICE_SLOT_CREATE_FENCE: usize = 36;
pub const DEVICE_SLOT_GET_DEVICE_REMOVED_REASON: usize = 37;
pub const DEVICE_SLOT_GET_COPYABLE_FOOTPRINTS: usize = 38;
pub const DEVICE_SLOT_CREATE_QUERY_HEAP: usize = 39;
pub const DEVICE_SLOT_SET_STABLE_POWER_STATE: usize = 40;
pub const DEVICE_SLOT_CREATE_COMMAND_SIGNATURE: usize = 41;
pub const DEVICE_SLOT_GET_RESOURCE_TILING: usize = 42;
pub const DEVICE_SLOT_GET_ADAPTER_LUID: usize = 43;
pub const DEVICE_SLOT_CREATE_PIPELINE_LIBRARY: usize = 44;
pub const DEVICE_SLOT_SET_EVENT_ON_MULTIPLE_FENCE_COMPLETION: usize = 45;
pub const DEVICE_SLOT_SET_RESIDENCY_PRIORITY: usize = 46;
pub const DEVICE_SLOT_CREATE_PIPELINE_STATE: usize = 47;
pub const DEVICE_SLOT_OPEN_EXISTING_HEAP_FROM_ADDRESS: usize = 48;
pub const DEVICE_SLOT_OPEN_EXISTING_HEAP_FROM_FILE_MAPPING: usize = 49;
pub const DEVICE_SLOT_ENQUEUE_MAKE_RESIDENT: usize = 50;
pub const DEVICE_SLOT_CREATE_COMMAND_LIST1: usize = 51;
pub const DEVICE_SLOT_CREATE_PROTECTED_RESOURCE_SESSION: usize = 52;
pub const DEVICE_SLOT_CREATE_COMMITTED_RESOURCE1: usize = 53;
pub const DEVICE_SLOT_CREATE_HEAP1: usize = 54;
pub const DEVICE_SLOT_CREATE_RESERVED_RESOURCE1: usize = 55;
pub const DEVICE_SLOT_GET_RESOURCE_ALLOCATION_INFO1: usize = 56;
pub const DEVICE_SLOT_CREATE_LIFETIME_TRACKER: usize = 57;
pub const DEVICE_SLOT_REMOVE_DEVICE: usize = 58;
pub const DEVICE_SLOT_ENUMERATE_META_COMMANDS: usize = 59;
pub const DEVICE_SLOT_ENUMERATE_META_COMMAND_PARAMETERS: usize = 60;
pub const DEVICE_SLOT_CREATE_META_COMMAND: usize = 61;
pub const DEVICE_SLOT_CREATE_STATE_OBJECT: usize = 62;
pub const DEVICE_SLOT_GET_RAYTRACING_AS_PREBUILD_INFO: usize = 63;
pub const DEVICE_SLOT_CHECK_DRIVER_MATCHING_IDENTIFIER: usize = 64;
pub const DEVICE_VTABLE_LEN: usize = 65;

/// Legacy names kept for callers of the earlier ABI module.
pub type D3D12ResourceDesc = ResourceDesc;
pub type D3D12HeapProperties = HeapProperties;

/// ABI-compatible `D3D12_COMMAND_QUEUE_DESC`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct D3D12CommandQueueDesc {
    pub list_type: u32,
    pub priority: i32,
    pub flags: u32,
    pub node_mask: u32,
}

/// A single ABI-compatible COM vtable function pointer slot.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct VTableSlot(pub *const ());

// SAFETY: VTableSlot only holds immutable static function pointers.
unsafe impl Sync for VTableSlot {}
unsafe impl Send for VTableSlot {}

macro_rules! slot {
    ($f:expr) => {
        VTableSlot($f as *const ())
    };
}

unsafe extern "system" fn stub_ok(_this: *mut c_void) -> i32 {
    S_OK
}

unsafe extern "system" fn stub_void(_this: *mut c_void) {}

unsafe extern "system" fn stub_notimpl(_this: *mut c_void) -> i32 {
    E_NOTIMPL
}

/// Logs `what` once per process (unsupported paths a title reaches).
fn log_once(what: &'static str) {
    static SEEN: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap();
    if !seen.contains(&what) {
        seen.push(what);
        crate::crash::log_line(what);
    }
}

// ============================================================================
// Shared COM plumbing
// ============================================================================

/// `ID3D12Object` private data and debug name.
#[derive(Default)]
pub struct ObjectMeta {
    pub name: String,
    pub private_data: HashMap<[u8; 16], Vec<u8>>,
    /// `SetPrivateDataInterface` objects (AddRef'd, released with us).
    pub interfaces: HashMap<[u8; 16], usize>,
}

/// The header every Strata D3D12 object starts with.
#[repr(C)]
pub struct ComBase {
    pub vtable: *const VTableSlot,
    pub refs: AtomicU32,
    /// The `ID3D12Device` that made the object (`GetDevice`).
    pub device: *mut c_void,
    pub meta: Mutex<ObjectMeta>,
    drop_fn: unsafe fn(*mut c_void),
}

/// A COM object: the header, then the interface's own state.
#[repr(C)]
pub struct Com<T> {
    pub base: ComBase,
    pub inner: T,
}

unsafe fn drop_com<T>(p: *mut c_void) {
    unsafe { drop(Box::from_raw(p as *mut Com<T>)) };
}

impl<T> Com<T> {
    pub fn boxed(vtable: &'static [VTableSlot], device: *mut c_void, inner: T) -> *mut Self {
        if !device.is_null() {
            // Children keep their device alive (GetDevice hands it out).
            unsafe { com_add_ref(device) };
        }
        Box::into_raw(Box::new(Self {
            base: ComBase {
                vtable: vtable.as_ptr(),
                refs: AtomicU32::new(1),
                device,
                meta: Mutex::new(ObjectMeta::default()),
                drop_fn: drop_com::<T>,
            },
            inner,
        }))
    }

    /// # Safety
    /// `p` is a live `Com<T>`.
    pub unsafe fn from<'a>(p: *mut c_void) -> &'a Self {
        unsafe { &*(p as *const Self) }
    }

    /// The object's vtable as a slice of `n` slots (tests).
    ///
    /// # Safety
    /// The vtable has at least `n` slots.
    pub unsafe fn vtable(&self, n: usize) -> &[VTableSlot] {
        unsafe { std::slice::from_raw_parts(self.base.vtable, n) }
    }
}

/// `inner` of a `Com<T>` behind `p`, or `None` for null.
unsafe fn inner<'a, T>(p: *mut c_void) -> Option<&'a T> {
    (!p.is_null()).then(|| unsafe { &Com::<T>::from(p).inner })
}

unsafe extern "system" fn com_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    unsafe {
        com_add_ref(this);
        *out = this;
    }
    S_OK
}

unsafe extern "system" fn com_add_ref(this: *mut c_void) -> u32 {
    let b = unsafe { &*(this as *const ComBase) };
    b.refs.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn com_release(this: *mut c_void) -> u32 {
    let b = unsafe { &*(this as *const ComBase) };
    let prev = b.refs.fetch_sub(1, Ordering::Release);
    if prev != 1 {
        return prev - 1;
    }
    std::sync::atomic::fence(Ordering::Acquire);
    let device = b.device;
    let interfaces: Vec<usize> = b
        .meta
        .lock()
        .unwrap()
        .interfaces
        .values()
        .copied()
        .collect();
    let drop_fn = b.drop_fn;
    unsafe {
        drop_fn(this);
        for i in interfaces {
            release(i as *mut c_void);
        }
        if !device.is_null() {
            release(device);
        }
    }
    0
}

/// Calls `Release` (slot 2) on any COM object.
unsafe fn release(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    unsafe {
        let vt = *(p as *const *const VTableSlot);
        let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute((*vt.add(2)).0);
        f(p);
    }
}

/// Calls `AddRef` (slot 1) on any COM object.
unsafe fn add_ref(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    unsafe {
        let vt = *(p as *const *const VTableSlot);
        let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute((*vt.add(1)).0);
        f(p);
    }
}

unsafe extern "system" fn com_get_private_data(
    this: *mut c_void,
    guid: *const [u8; 16],
    data_size: *mut u32,
    out_data: *mut c_void,
) -> i32 {
    if guid.is_null() || data_size.is_null() {
        return E_INVALIDARG;
    }
    let b = unsafe { &*(this as *const ComBase) };
    let key = unsafe { *guid };
    let guard = b.meta.lock().unwrap();
    let bytes: Vec<u8> = if let Some(bytes) = guard.private_data.get(&key) {
        bytes.clone()
    } else if let Some(&iface) = guard.interfaces.get(&key) {
        if !out_data.is_null() {
            unsafe { add_ref(iface as *mut c_void) };
        }
        (iface as u64).to_le_bytes().to_vec()
    } else {
        unsafe { *data_size = 0 };
        return DXGI_ERROR_NOT_FOUND;
    };
    let cap = unsafe { *data_size } as usize;
    unsafe { *data_size = bytes.len() as u32 };
    if !out_data.is_null() {
        if cap < bytes.len() {
            return 0x887A_0003u32 as i32; // DXGI_ERROR_MORE_DATA
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_data.cast::<u8>(), bytes.len())
        };
    }
    S_OK
}

unsafe extern "system" fn com_set_private_data(
    this: *mut c_void,
    guid: *const [u8; 16],
    data_size: u32,
    data: *const c_void,
) -> i32 {
    if guid.is_null() {
        return E_INVALIDARG;
    }
    let b = unsafe { &*(this as *const ComBase) };
    let key = unsafe { *guid };
    let mut guard = b.meta.lock().unwrap();
    if data.is_null() || data_size == 0 {
        guard.private_data.remove(&key);
    } else {
        let slice = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), data_size as usize) };
        guard.private_data.insert(key, slice.to_vec());
    }
    S_OK
}

unsafe extern "system" fn com_set_private_data_interface(
    this: *mut c_void,
    guid: *const [u8; 16],
    iface: *mut c_void,
) -> i32 {
    if guid.is_null() {
        return E_INVALIDARG;
    }
    let b = unsafe { &*(this as *const ComBase) };
    let key = unsafe { *guid };
    let old = {
        let mut guard = b.meta.lock().unwrap();
        if iface.is_null() {
            guard.interfaces.remove(&key)
        } else {
            unsafe { add_ref(iface) };
            guard.interfaces.insert(key, iface as usize)
        }
    };
    if let Some(o) = old {
        unsafe { release(o as *mut c_void) };
    }
    S_OK
}

unsafe extern "system" fn com_set_name(this: *mut c_void, name: *const u16) -> i32 {
    if name.is_null() {
        return E_INVALIDARG;
    }
    let b = unsafe { &*(this as *const ComBase) };
    let mut len = 0usize;
    unsafe {
        while *name.add(len) != 0 && len < 4096 {
            len += 1;
        }
        b.meta.lock().unwrap().name =
            String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
    }
    S_OK
}

unsafe extern "system" fn com_get_device(
    this: *mut c_void,
    riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let b = unsafe { &*(this as *const ComBase) };
    if b.device.is_null() {
        unsafe { *out = std::ptr::null_mut() };
        return E_NOINTERFACE;
    }
    unsafe { com_qi(b.device, riid, out) }
}

/// Slots 0..8 every device child shares.
const fn base_slots<const N: usize>() -> [VTableSlot; N] {
    let mut vt = [VTableSlot(stub_notimpl as *const ()); N];
    vt[SLOT_QUERY_INTERFACE] = slot!(com_qi);
    vt[SLOT_ADD_REF] = slot!(com_add_ref);
    vt[SLOT_RELEASE] = slot!(com_release);
    vt[SLOT_GET_PRIVATE_DATA] = slot!(com_get_private_data);
    vt[SLOT_SET_PRIVATE_DATA] = slot!(com_set_private_data);
    vt[SLOT_SET_PRIVATE_DATA_INTERFACE] = slot!(com_set_private_data_interface);
    vt[SLOT_SET_NAME] = slot!(com_set_name);
    if N > SLOT_GET_DEVICE {
        vt[SLOT_GET_DEVICE] = slot!(com_get_device);
    }
    vt
}

/// Writes a freshly made object to `out` (`QueryInterface` semantics: `out`
/// null only checks parameters).
unsafe fn hand_out<T>(obj: *mut Com<T>, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        unsafe { com_release(obj.cast()) };
        return 1; // S_FALSE: parameters valid, nothing created
    }
    unsafe { *out = obj.cast() };
    S_OK
}

// ============================================================================
// The device backend
// ============================================================================

/// What a device shares with every object it makes: the GPU, console
/// memory, and the pipeline and texture tables its queues execute with.
pub struct Backend {
    pub runner: Arc<ComputeRunner>,
    pub uma: Arc<Mutex<UmaManager>>,
    pub pipelines: PipelineTable,
    pub textures: SharedTextures,
}

// SAFETY: the runner is internally synchronised; the rest is behind mutexes.
unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

/// Title-visible memory: a console allocation, its host pointer and its GPU
/// address.
#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub console_va: u64,
    pub host: u64,
    pub gpu_va: u64,
    pub size: u64,
    /// Freed when its owner goes (false for placed resources and views).
    pub owned: bool,
    /// From the GPU's UMA (false: plain host memory without a GPU address).
    pub uma: bool,
}

impl Backend {
    /// GPU-visible memory of `size` bytes, zeroed.
    fn alloc(&self, size: u64) -> Option<Block> {
        let mut u = self.uma.lock().unwrap();
        let a = u.allocate_resource(&self.runner, size.max(256)).ok()?;
        let host = a.cpu_ptr()? as u64;
        Some(Block {
            console_va: a.console_va,
            host,
            gpu_va: a.host_bda,
            size: a.size,
            owned: true,
            uma: true,
        })
    }

    fn free(&self, b: &Block) {
        if b.owned && b.uma {
            self.uma
                .lock()
                .unwrap()
                .free_allocation(&self.runner, b.console_va);
        }
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if let Err(error) = self.runner.flush_pipeline_cache() {
            crate::crash::log_line(&format!("pipeline cache: {error}"));
        }
        crate::pipeline_cache::released(self);
    }
}

/// Memory an object owns, freed through its device when the object goes.
pub struct Owned {
    pub dev: *mut c_void,
    pub mem: Block,
}

// SAFETY: the device pointer is a COM reference the owner keeps alive.
unsafe impl Send for Owned {}
unsafe impl Sync for Owned {}

impl Drop for Owned {
    fn drop(&mut self) {
        free_block(self.dev, &self.mem);
    }
}

/// Plain host memory for a device without a GPU (headless tests): the GPU
/// address is the host address.
fn host_layout(size: u64) -> std::alloc::Layout {
    // 64 KiB alignment, like D3D12 resource placement (and T# base addresses
    // need 256 bytes).
    std::alloc::Layout::from_size_align(size.max(256).next_multiple_of(65536) as usize, 65536)
        .expect("valid layout")
}

fn host_block(size: u64) -> Block {
    let layout = host_layout(size);
    // SAFETY: a non-zero-sized layout.
    let p = unsafe { std::alloc::alloc_zeroed(layout) } as u64;
    Block {
        console_va: p,
        host: p,
        gpu_va: p,
        size: layout.size() as u64,
        owned: true,
        uma: false,
    }
}

fn free_host_block(b: &Block) {
    if b.owned && !b.uma && b.host != 0 {
        // SAFETY: allocated by host_block with this layout.
        unsafe { std::alloc::dealloc(b.host as *mut u8, host_layout(b.size)) };
    }
}

/// `ID3D12Device` state.
pub struct DeviceInner {
    pub backend: Option<Arc<Backend>>,
    next_pso: AtomicU64,
}

pub type StandardDevice = Com<DeviceInner>;

fn device_inner<'a>(dev: *mut c_void) -> &'a DeviceInner {
    // SAFETY: callers pass the device that made them (a live StandardDevice).
    unsafe { &Com::<DeviceInner>::from(dev).inner }
}

fn alloc_block(dev: *mut c_void, size: u64) -> Option<Block> {
    if dev.is_null() {
        return Some(host_block(size));
    }
    match &device_inner(dev).backend {
        Some(b) => b.alloc(size),
        None => Some(host_block(size)),
    }
}

fn free_block(dev: *mut c_void, b: &Block) {
    if !b.uma {
        free_host_block(b);
        return;
    }
    if !dev.is_null()
        && let Some(be) = &device_inner(dev).backend
    {
        be.free(b);
    }
}

// ============================================================================
// ID3D12Fence (11 slots)
// ============================================================================

pub struct FenceInner {
    pub value: AtomicU64,
    /// `(value, event)` pairs `SetEventOnCompletion` is waiting on.
    pub waiters: Mutex<Vec<(u64, usize)>>,
    pub cv: Condvar,
    pub cv_lock: Mutex<()>,
}

pub type StandardFence = Com<FenceInner>;

#[cfg(windows)]
unsafe extern "system" {
    fn SetEvent(h_event: *mut c_void) -> i32;
    fn QueryPerformanceCounter(count: *mut i64) -> i32;
}

#[cfg(not(windows))]
unsafe fn SetEvent(_h: *mut c_void) -> i32 {
    1
}

impl FenceInner {
    /// Completes the fence to `v`: fires the events waiting for it and wakes
    /// blocked waiters.
    pub fn signal(&self, v: u64) {
        self.value.store(v, Ordering::Release);
        let ready: Vec<usize> = {
            let mut w = self.waiters.lock().unwrap();
            let (fire, keep): (Vec<_>, Vec<_>) = w.drain(..).partition(|(t, _)| *t <= v);
            *w = keep;
            fire.into_iter().map(|(_, e)| e).collect()
        };
        for e in ready {
            unsafe { SetEvent(e as *mut c_void) };
        }
        let _g = self.cv_lock.lock().unwrap();
        self.cv.notify_all();
    }

    /// Blocks until the fence reaches `v` (or `timeout` passes; `false` then).
    pub fn wait(&self, v: u64, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        let mut g = self.cv_lock.lock().unwrap();
        while self.value.load(Ordering::Acquire) < v {
            let now = std::time::Instant::now();
            if now >= deadline {
                return false;
            }
            g = self.cv.wait_timeout(g, deadline - now).unwrap().0;
        }
        true
    }
}

unsafe extern "system" fn fence_get_completed(this: *mut c_void) -> u64 {
    unsafe { Com::<FenceInner>::from(this) }
        .inner
        .value
        .load(Ordering::Acquire)
}

unsafe extern "system" fn fence_set_event_on_completion(
    this: *mut c_void,
    value: u64,
    h_event: *mut c_void,
) -> i32 {
    let f = unsafe { &Com::<FenceInner>::from(this).inner };
    if h_event.is_null() {
        // A null event blocks until the value is reached.
        f.wait(value, std::time::Duration::from_secs(10));
        return S_OK;
    }
    if f.value.load(Ordering::Acquire) >= value {
        unsafe { SetEvent(h_event) };
    } else {
        f.waiters.lock().unwrap().push((value, h_event as usize));
        // Signalled between the check and the push: fire now.
        if f.value.load(Ordering::Acquire) >= value {
            f.signal(f.value.load(Ordering::Acquire));
        }
    }
    S_OK
}

unsafe extern "system" fn fence_signal(this: *mut c_void, val: u64) -> i32 {
    unsafe { Com::<FenceInner>::from(this) }.inner.signal(val);
    S_OK
}

static FENCE_VTABLE: [VTableSlot; FENCE_VTABLE_LEN] = {
    let mut vt = base_slots::<FENCE_VTABLE_LEN>();
    vt[FENCE_SLOT_GET_COMPLETED_VALUE] = slot!(fence_get_completed);
    vt[FENCE_SLOT_SET_EVENT_ON_COMPLETION] = slot!(fence_set_event_on_completion);
    vt[FENCE_SLOT_SIGNAL] = slot!(fence_signal);
    vt
};

impl Com<FenceInner> {
    pub fn new_boxed(initial: u64) -> *mut Self {
        Self::new_boxed_with_device(initial, std::ptr::null_mut())
    }

    pub fn new_boxed_with_device(initial: u64, device: *mut c_void) -> *mut Self {
        Com::boxed(
            &FENCE_VTABLE,
            device,
            FenceInner {
                value: AtomicU64::new(initial),
                waiters: Mutex::new(Vec::new()),
                cv: Condvar::new(),
                cv_lock: Mutex::new(()),
            },
        )
    }
}

// ============================================================================
// ID3D12Resource (15 slots)
// ============================================================================

pub struct ResourceInner {
    pub desc: ResourceDesc,
    pub heap_props: HeapProperties,
    pub mem: Block,
    /// A placed resource's heap (kept alive while the resource lives).
    pub heap: *mut c_void,
    /// The resource's own memory (committed and reserved resources).
    _owned: Option<Owned>,
}

// SAFETY: the heap pointer is a COM object reference only.
unsafe impl Send for ResourceInner {}
unsafe impl Sync for ResourceInner {}

impl Drop for ResourceInner {
    fn drop(&mut self) {
        if !self.heap.is_null() {
            unsafe { release(self.heap) };
        }
    }
}

pub type StandardResource = Com<ResourceInner>;

/// Recognises a resource produced by our device before interpreting its COM body.
pub(crate) unsafe fn is_present_resource(p: *mut c_void) -> bool {
    crate::d3d12x::readable(p as u64, size_of::<StandardResource>())
        && unsafe { p.cast::<*const VTableSlot>().read_unaligned() } == RESOURCE_VTABLE.as_ptr()
}

impl ResourceInner {
    fn layout(&self) -> TextureLayout {
        TextureLayout::of(&self.desc)
    }

    /// Byte offset of subresource `sub` (0 for buffers).
    fn subresource_offset(&self, sub: u32) -> u64 {
        if self.desc.dimension == DIMENSION_BUFFER {
            return 0;
        }
        let t = self.layout();
        let (m, s) = t.subresource(sub);
        t.offset(m, s)
    }

    pub fn view(&self) -> ViewResource {
        ViewResource {
            gpu_va: self.mem.gpu_va,
            desc: self.desc,
        }
    }
}

unsafe fn resource<'a>(p: *mut c_void) -> Option<&'a ResourceInner> {
    unsafe { inner::<ResourceInner>(p) }
}

unsafe extern "system" fn res_map(
    this: *mut c_void,
    sub: u32,
    read_range: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    let resource = unsafe { Com::<ResourceInner>::from(this) };
    let r = &resource.inner;
    if !out.is_null() {
        if r.mem.uma && !resource.base.device.is_null() {
            let start = r.subresource_offset(sub);
            let length = r.mem.size.saturating_sub(start);
            let [begin, end] = if read_range.is_null() {
                [0, length as usize]
            } else {
                unsafe { read_range.cast::<[usize; 2]>().read_unaligned() }
            };
            let end = (end as u64).min(length);
            let device = unsafe { Com::<DeviceInner>::from(resource.base.device) };
            if end > begin as u64
                && let Some(backend) = &device.inner.backend
                && let Err(error) = backend.uma.lock().unwrap().readback(
                    &backend.runner,
                    r.mem.console_va + start + begin as u64,
                    end - begin as u64,
                )
            {
                crate::crash::log_line(&format!("resource map readback: {error}"));
                return E_FAIL;
            }
        }
        unsafe { *out = (r.mem.host + r.subresource_offset(sub)) as *mut c_void };
    }
    S_OK
}

unsafe extern "system" fn res_unmap(_this: *mut c_void, _sub: u32, _range: *const c_void) {}

unsafe extern "system" fn res_get_desc(
    this: *mut c_void,
    out: *mut ResourceDesc,
) -> *mut ResourceDesc {
    let r = unsafe { &Com::<ResourceInner>::from(this).inner };
    if !out.is_null() {
        unsafe { *out = r.desc };
    }
    out
}

unsafe extern "system" fn res_get_gpu_va(this: *mut c_void) -> u64 {
    unsafe { Com::<ResourceInner>::from(this) }.inner.mem.gpu_va
}

/// Copies between a subresource and caller memory, row by row.
#[allow(clippy::too_many_arguments)]
unsafe fn subresource_copy(
    r: &ResourceInner,
    sub: u32,
    bx: *const Box3,
    user: *mut u8,
    row_pitch: u32,
    depth_pitch: u32,
    to_resource: bool,
) -> i32 {
    if user.is_null() {
        return E_INVALIDARG;
    }
    if r.desc.dimension == DIMENSION_BUFFER {
        let (start, end) = if bx.is_null() {
            (0, r.mem.size)
        } else {
            let b = unsafe { *bx };
            (u64::from(b.left), u64::from(b.right).min(r.mem.size))
        };
        let n = end.saturating_sub(start) as usize;
        unsafe {
            let res = (r.mem.host + start) as *mut u8;
            if to_resource {
                std::ptr::copy_nonoverlapping(user, res, n);
            } else {
                std::ptr::copy_nonoverlapping(res, user, n);
            }
        }
        return S_OK;
    }
    let t = r.layout();
    let (m, s) = t.subresource(sub);
    let (w, h) = t.extent(m);
    let b = if bx.is_null() {
        Box3 {
            left: 0,
            top: 0,
            front: 0,
            right: w,
            bottom: h,
            back: if t.is_3d { t.slices(m) } else { 1 },
        }
    } else {
        unsafe { *bx }
    };
    let bpe = u64::from(t.bytes);
    let blk = t.block;
    let row_bytes = u64::from(b.right.saturating_sub(b.left).div_ceil(blk)) * bpe;
    let rows = b.bottom.saturating_sub(b.top).div_ceil(blk);
    let pitch = t.row_pitch(m);
    for z in b.front..b.back {
        let slice = if t.is_3d { z } else { s };
        let base = r.mem.host + t.offset(m, slice);
        for y in 0..rows {
            let res = base + u64::from(b.top / blk + y) * pitch + u64::from(b.left / blk) * bpe;
            let usr = unsafe {
                user.add(
                    ((z - b.front) as usize) * depth_pitch as usize
                        + y as usize * row_pitch as usize,
                )
            };
            unsafe {
                if to_resource {
                    std::ptr::copy_nonoverlapping(usr, res as *mut u8, row_bytes as usize);
                } else {
                    std::ptr::copy_nonoverlapping(res as *const u8, usr, row_bytes as usize);
                }
            }
        }
    }
    S_OK
}

unsafe extern "system" fn res_write_to_subresource(
    this: *mut c_void,
    dst_sub: u32,
    dst_box: *const Box3,
    src: *const c_void,
    src_row_pitch: u32,
    src_depth_pitch: u32,
) -> i32 {
    let r = unsafe { &Com::<ResourceInner>::from(this).inner };
    unsafe {
        subresource_copy(
            r,
            dst_sub,
            dst_box,
            src as *mut u8,
            src_row_pitch,
            src_depth_pitch,
            true,
        )
    }
}

unsafe extern "system" fn res_read_from_subresource(
    this: *mut c_void,
    dst: *mut c_void,
    dst_row_pitch: u32,
    dst_depth_pitch: u32,
    src_sub: u32,
    src_box: *const Box3,
) -> i32 {
    let r = unsafe { &Com::<ResourceInner>::from(this).inner };
    unsafe {
        subresource_copy(
            r,
            src_sub,
            src_box,
            dst as *mut u8,
            dst_row_pitch,
            dst_depth_pitch,
            false,
        )
    }
}

unsafe extern "system" fn res_get_heap_props(
    this: *mut c_void,
    out_props: *mut HeapProperties,
    out_flags: *mut u32,
) -> i32 {
    let r = unsafe { &Com::<ResourceInner>::from(this).inner };
    if !out_props.is_null() {
        unsafe { *out_props = r.heap_props };
    }
    if !out_flags.is_null() {
        unsafe { *out_flags = 0 };
    }
    S_OK
}

static RESOURCE_VTABLE: [VTableSlot; RESOURCE_VTABLE_LEN] = {
    let mut vt = base_slots::<RESOURCE_VTABLE_LEN>();
    vt[RESOURCE_SLOT_MAP] = slot!(res_map);
    vt[RESOURCE_SLOT_UNMAP] = slot!(res_unmap);
    vt[RESOURCE_SLOT_GET_DESC] = slot!(res_get_desc);
    vt[RESOURCE_SLOT_GET_GPU_VA] = slot!(res_get_gpu_va);
    vt[RESOURCE_SLOT_WRITE_TO_SUBRESOURCE] = slot!(res_write_to_subresource);
    vt[RESOURCE_SLOT_READ_FROM_SUBRESOURCE] = slot!(res_read_from_subresource);
    vt[RESOURCE_SLOT_GET_HEAP_PROPERTIES] = slot!(res_get_heap_props);
    vt
};

impl Com<ResourceInner> {
    /// A committed resource: its own memory, sized by Strata's layout.
    pub fn new_boxed(
        desc: ResourceDesc,
        heap_props: HeapProperties,
        device: *mut c_void,
    ) -> *mut Self {
        let size = resource_size(&desc);
        let Some(mem) = alloc_block(device, size) else {
            return std::ptr::null_mut();
        };
        Self::with_memory(desc, heap_props, device, mem, std::ptr::null_mut())
    }

    fn with_memory(
        desc: ResourceDesc,
        heap_props: HeapProperties,
        device: *mut c_void,
        mem: Block,
        heap: *mut c_void,
    ) -> *mut Self {
        Com::boxed(
            &RESOURCE_VTABLE,
            device,
            ResourceInner {
                desc,
                heap_props,
                mem,
                heap,
                _owned: mem.owned.then_some(Owned { dev: device, mem }),
            },
        )
    }
}

// ============================================================================
// Simple device children
// ============================================================================

pub struct DescriptorHeapInner {
    pub desc: DescriptorHeapDesc,
    pub mem: Block,
    _owned: Owned,
}

unsafe extern "system" fn heap_get_desc(
    this: *mut c_void,
    out: *mut DescriptorHeapDesc,
) -> *mut DescriptorHeapDesc {
    if !out.is_null() {
        unsafe { *out = Com::<DescriptorHeapInner>::from(this).inner.desc };
    }
    out
}

unsafe extern "system" fn heap_cpu_start(this: *mut c_void, out: *mut u64) -> *mut u64 {
    if !out.is_null() {
        unsafe { *out = Com::<DescriptorHeapInner>::from(this).inner.mem.host };
    }
    out
}

unsafe extern "system" fn heap_gpu_start(this: *mut c_void, out: *mut u64) -> *mut u64 {
    if !out.is_null() {
        unsafe { *out = Com::<DescriptorHeapInner>::from(this).inner.mem.gpu_va };
    }
    out
}

static DESCRIPTOR_HEAP_VTABLE: [VTableSlot; 11] = {
    let mut vt = base_slots::<11>();
    vt[8] = slot!(heap_get_desc);
    vt[9] = slot!(heap_cpu_start);
    vt[10] = slot!(heap_gpu_start);
    vt
};

pub struct RootSignatureInner {
    pub layout: Arc<RootLayout>,
    /// The static samplers' S# table.
    pub samplers: Option<Block>,
    _owned: Option<Owned>,
}

static ROOT_SIGNATURE_VTABLE: [VTableSlot; 8] = base_slots::<8>();

pub struct PipelineStateInner {
    pub key: PipelineKey,
    pub compute: bool,
    pub root: Option<Arc<RootLayout>>,
}

unsafe extern "system" fn pso_get_cached_blob(_this: *mut c_void, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    unsafe { *out = crate::thunks::ComBlob::new_boxed(Vec::new()) as *mut c_void };
    S_OK
}

static PIPELINE_STATE_VTABLE: [VTableSlot; 9] = {
    let mut vt = base_slots::<9>();
    vt[8] = slot!(pso_get_cached_blob);
    vt
};

pub struct AllocatorInner {
    pub list_type: u32,
}

static ALLOCATOR_VTABLE: [VTableSlot; 9] = {
    let mut vt = base_slots::<9>();
    vt[8] = slot!(stub_ok); // Reset
    vt
};

pub struct HeapInner {
    pub desc: HeapDesc,
    pub mem: Block,
    _owned: Owned,
}

unsafe extern "system" fn d3d_heap_get_desc(
    this: *mut c_void,
    out: *mut HeapDesc,
) -> *mut HeapDesc {
    if !out.is_null() {
        unsafe { *out = Com::<HeapInner>::from(this).inner.desc };
    }
    out
}

static HEAP_VTABLE: [VTableSlot; 9] = {
    let mut vt = base_slots::<9>();
    vt[8] = slot!(d3d_heap_get_desc);
    vt
};

pub struct QueryHeapInner {
    pub desc: QueryHeapDesc,
    pub mem: Block,
    pub stride: u64,
    _owned: Owned,
}

static QUERY_HEAP_VTABLE: [VTableSlot; 8] = base_slots::<8>();

pub struct CommandSignatureInner {
    pub layout: Arc<IndirectLayout>,
}

static COMMAND_SIGNATURE_VTABLE: [VTableSlot; 8] = base_slots::<8>();

// ============================================================================
// ID3D12GraphicsCommandList (80 slots)
// ============================================================================

/// What the root of one pipeline type (graphics or compute) holds.
#[derive(Clone)]
struct RootState {
    layout: Option<Arc<RootLayout>>,
    shadow: [u32; 256],
    sampler_table: u64,
}

impl Default for RootState {
    fn default() -> Self {
        Self {
            layout: None,
            shadow: [0; 256],
            sampler_table: 0,
        }
    }
}

#[derive(Default)]
struct ListState {
    graphics: RootState,
    compute: RootState,
    /// Which root the engine last received (graphics or compute), and
    /// whether it changed since.
    pushed: Option<bool>,
    dirty: bool,
    /// Ending-access resolves recorded in `BeginRenderPass`, flushed by `EndRenderPass`.
    render_pass_resolves: Vec<ThinCommand>,
}

pub struct CommandListInner {
    pub list_type: u32,
    pub closed: AtomicBool,
    pub thin: Mutex<ThinCommandList>,
    state: Mutex<ListState>,
}

pub type StandardCommandList = Com<CommandListInner>;

fn cl<'a>(this: *mut c_void) -> &'a CommandListInner {
    // SAFETY: called from this list's own vtable methods.
    unsafe { &Com::<CommandListInner>::from(this).inner }
}

fn record(this: *mut c_void, cmd: ThinCommand) {
    cl(this).thin.lock().unwrap().push(cmd);
}

/// Before a draw (`graphics`) or dispatch: the engine gets that root's
/// SGPRs and layout if they changed or the other root was pushed last.
fn flush_root(this: *mut c_void, graphics: bool) {
    let l = cl(this);
    let mut st = l.state.lock().unwrap();
    if st.pushed == Some(graphics) && !st.dirty {
        return;
    }
    let root = if graphics { &st.graphics } else { &st.compute };
    let (dwords, spilled, vt) = root.layout.as_ref().map_or((32, false, 30), |r| {
        (r.dwords.max(32), r.spilled, r.vertex_table_sgpr)
    });
    let values = root.shadow[..dwords.min(256) as usize].to_vec();
    let mut thin = l.thin.lock().unwrap();
    thin.push(ThinCommand::SetRootLayout {
        dwords,
        spilled,
        vertex_table_sgpr: vt,
    });
    thin.push_sgprs(0, &values);
    st.pushed = Some(graphics);
    st.dirty = false;
}

/// Writes root values: `sgpr` onward of the chosen root.
fn set_root(this: *mut c_void, graphics: bool, sgpr: usize, values: &[u32]) {
    let mut st = cl(this).state.lock().unwrap();
    let root = if graphics {
        &mut st.graphics
    } else {
        &mut st.compute
    };
    for (i, &v) in values.iter().enumerate() {
        if let Some(s) = root.shadow.get_mut(sgpr + i) {
            *s = v;
        }
    }
    st.dirty = true;
}

/// The user SGPR of root parameter `idx` of the chosen root.
fn param_sgpr(this: *mut c_void, graphics: bool, idx: u32) -> Option<(usize, RootParam)> {
    let st = cl(this).state.lock().unwrap();
    let root = if graphics { &st.graphics } else { &st.compute };
    let p = root.layout.as_ref()?.params.get(idx as usize)?.clone();
    Some((p.sgpr() as usize, p))
}

unsafe extern "system" fn cl_get_type(this: *mut c_void) -> u32 {
    cl(this).list_type
}

unsafe extern "system" fn cl_close(this: *mut c_void) -> i32 {
    cl(this).closed.store(true, Ordering::Release);
    S_OK
}

unsafe extern "system" fn cl_reset(
    this: *mut c_void,
    _alloc: *mut c_void,
    pso: *mut c_void,
) -> i32 {
    let l = cl(this);
    l.closed.store(false, Ordering::Release);
    l.thin.lock().unwrap().reset();
    *l.state.lock().unwrap() = ListState::default();
    if !pso.is_null() {
        unsafe { cl_set_pso(this, pso) };
    }
    S_OK
}

unsafe extern "system" fn cl_clear_state(this: *mut c_void, pso: *mut c_void) {
    *cl(this).state.lock().unwrap() = ListState::default();
    if !pso.is_null() {
        unsafe { cl_set_pso(this, pso) };
    }
}

unsafe extern "system" fn cl_draw_instanced(this: *mut c_void, vc: u32, ic: u32, sv: u32, si: u32) {
    flush_root(this, true);
    cl(this).thin.lock().unwrap().draw_instanced(vc, ic, sv, si);
}

unsafe extern "system" fn cl_draw_indexed_instanced(
    this: *mut c_void,
    index_count: u32,
    ic: u32,
    start_index: u32,
    base_vertex: i32,
    si: u32,
) {
    flush_root(this, true);
    cl(this).thin.lock().unwrap().draw_indexed_instanced(
        index_count,
        ic,
        start_index,
        base_vertex,
        si,
    );
}

unsafe extern "system" fn cl_dispatch(this: *mut c_void, x: u32, y: u32, z: u32) {
    flush_root(this, false);
    cl(this)
        .thin
        .lock()
        .unwrap()
        .dispatch(x, y, z, [1, 1, 1], false);
}

unsafe extern "system" fn cl_copy_buffer_region(
    this: *mut c_void,
    dst: *mut c_void,
    dst_offset: u64,
    src: *mut c_void,
    src_offset: u64,
    num_bytes: u64,
) {
    let (Some(d), Some(s)) = (unsafe { resource(dst) }, unsafe { resource(src) }) else {
        return;
    };
    record(
        this,
        ThinCommand::CopyMemory {
            dst: d.mem.gpu_va + dst_offset,
            src: s.mem.gpu_va + src_offset,
            len: num_bytes,
        },
    );
}

/// A copy location as a surface: address, pitches, element size, and the
/// subresource's extent in elements.
fn copy_surface(loc: &TextureCopyLocation) -> Option<(Surface, [u32; 3], u32)> {
    let r = unsafe { resource(loc.resource) }?;
    if loc.copy_type == 1 {
        // Placed footprint: offset, then format, width, height, depth, row pitch.
        let offset = u64::from(loc.u[0]) | (u64::from(loc.u[1]) << 32);
        let (fmt, w, h, d, pitch) = (loc.u[2], loc.u[3], loc.u[4], loc.u[5], loc.u[6]);
        let (bytes, block) = dxgi_bytes(fmt);
        let rows = h.div_ceil(block);
        Some((
            Surface {
                va: r.mem.gpu_va + offset,
                row_pitch: u64::from(pitch),
                slice_pitch: u64::from(pitch) * u64::from(rows),
                bpe: bytes,
            },
            [w.div_ceil(block), rows, d.max(1)],
            block,
        ))
    } else if r.desc.dimension == DIMENSION_BUFFER {
        None
    } else {
        let t = TextureLayout::of(&r.desc);
        let (m, s) = t.subresource(loc.u[0]);
        let (w, h) = t.extent(m);
        Some((
            Surface {
                va: r.mem.gpu_va + t.offset(m, if t.is_3d { 0 } else { s }),
                row_pitch: t.row_pitch(m),
                slice_pitch: t.slice_bytes(m),
                bpe: t.bytes,
            },
            [
                w.div_ceil(t.block),
                h.div_ceil(t.block),
                if t.is_3d { t.slices(m) } else { 1 },
            ],
            t.block,
        ))
    }
}

unsafe extern "system" fn cl_copy_texture_region(
    this: *mut c_void,
    dst: *const TextureCopyLocation,
    dx: u32,
    dy: u32,
    dz: u32,
    src: *const TextureCopyLocation,
    src_box: *const Box3,
) {
    if dst.is_null() || src.is_null() {
        return;
    }
    let (Some((ds, _, dblk)), Some((ss, sext, sblk))) = (
        copy_surface(unsafe { &*dst }),
        copy_surface(unsafe { &*src }),
    ) else {
        log_once("CopyTextureRegion: buffer-to-buffer or unknown resource location");
        return;
    };
    let (origin, extent) = if src_box.is_null() {
        ([0, 0, 0], sext)
    } else {
        let b = unsafe { *src_box };
        (
            [b.left / sblk, b.top / sblk, b.front],
            [
                b.right.saturating_sub(b.left).div_ceil(sblk),
                b.bottom.saturating_sub(b.top).div_ceil(sblk),
                b.back.saturating_sub(b.front).max(1),
            ],
        )
    };
    record(
        this,
        ThinCommand::CopyTexture(Box::new(TextureCopy {
            dst: ds,
            dst_origin: [dx / dblk, dy / dblk, dz],
            src: ss,
            src_origin: origin,
            extent,
        })),
    );
}

unsafe extern "system" fn cl_copy_resource(this: *mut c_void, dst: *mut c_void, src: *mut c_void) {
    let (Some(d), Some(s)) = (unsafe { resource(dst) }, unsafe { resource(src) }) else {
        return;
    };
    // Same desc, same layout: the bytes copy as they are.
    let len = if s.desc.dimension == DIMENSION_BUFFER {
        s.desc.width
    } else {
        TextureLayout::of(&s.desc).total_bytes()
    };
    record(
        this,
        ThinCommand::CopyMemory {
            dst: d.mem.gpu_va,
            src: s.mem.gpu_va,
            len: len.min(d.mem.size).min(s.mem.size),
        },
    );
}

unsafe extern "system" fn cl_resolve_subresource(
    this: *mut c_void,
    dst: *mut c_void,
    dst_sub: u32,
    src: *mut c_void,
    src_sub: u32,
    _format: u32,
) {
    // Strata renders single-sampled: a resolve is a subresource copy.
    let (Some(d), Some(s)) = (unsafe { resource(dst) }, unsafe { resource(src) }) else {
        return;
    };
    let t = TextureLayout::of(&s.desc);
    let (m, _) = t.subresource(src_sub);
    record(
        this,
        ThinCommand::CopyMemory {
            dst: d.mem.gpu_va + d.subresource_offset(dst_sub),
            src: s.mem.gpu_va + s.subresource_offset(src_sub),
            len: t.slice_bytes(m),
        },
    );
}

unsafe extern "system" fn cl_ia_set_topology(this: *mut c_void, t: u32) {
    // Console D3D_PRIMITIVE_TOPOLOGY_RECTLIST (13) maps to Vulkan TRIANGLE_LIST (3)
    // with vertex expansion handled at draw time via `strata_shader::rect`.
    let vk_t = if t == strata_shader::rect::RECTLIST_TOPOLOGY_ID {
        3 // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    } else {
        vk_topology(t)
    };
    record(this, ThinCommand::SetTopology(vk_t));
}

unsafe extern "system" fn cl_rs_set_viewports(this: *mut c_void, n: u32, vps: *const Viewport) {
    if n == 0 || vps.is_null() {
        return;
    }
    let v = unsafe { *vps };
    record(
        this,
        ThinCommand::SetViewport(
            [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth].map(f32::to_bits),
        ),
    );
}

unsafe extern "system" fn cl_rs_set_scissor_rects(this: *mut c_void, n: u32, rects: *const Rect) {
    if n == 0 || rects.is_null() {
        return;
    }
    let r = unsafe { *rects };
    record(
        this,
        ThinCommand::SetScissor([r.left, r.top, r.right, r.bottom]),
    );
}

unsafe extern "system" fn cl_om_set_blend_factor(this: *mut c_void, f: *const f32) {
    let v = if f.is_null() {
        [1.0; 4]
    } else {
        unsafe { std::ptr::read_unaligned(f as *const [f32; 4]) }
    };
    record(this, ThinCommand::SetBlendFactor(v.map(f32::to_bits)));
}

unsafe extern "system" fn cl_om_set_stencil_ref(this: *mut c_void, r: u32) {
    record(this, ThinCommand::SetStencilRef(r));
}

unsafe extern "system" fn cl_set_pso(this: *mut c_void, pso: *mut c_void) {
    let Some(p) = (unsafe { inner::<PipelineStateInner>(pso) }) else {
        return;
    };
    cl(this).thin.lock().unwrap().bind_pipeline(p.key);
}

unsafe extern "system" fn cl_resource_barrier(
    this: *mut c_void,
    n: u32,
    barriers: *const ResourceBarrier,
) {
    // In-order synchronous execution leaves barriers nothing to order;
    // they are recorded for the statistics only.
    if n > 0 && !barriers.is_null() {
        let b = unsafe { *barriers };
        let va = unsafe { resource(b.resource) }.map_or(0, |r| r.mem.gpu_va);
        cl(this)
            .thin
            .lock()
            .unwrap()
            .resource_barrier(va, b.u[1], b.u[2]);
    }
}

unsafe extern "system" fn cl_execute_bundle(this: *mut c_void, bundle: *mut c_void) {
    let Some(b) = (unsafe { inner::<CommandListInner>(bundle) }) else {
        return;
    };
    let cmds: Vec<ThinCommand> = b.thin.lock().unwrap().commands().to_vec();
    let mut t = cl(this).thin.lock().unwrap();
    for c in cmds {
        t.push(c);
    }
    // The bundle's binds may have changed the engine's root.
    cl(this).state.lock().unwrap().dirty = true;
}

unsafe extern "system" fn cl_set_descriptor_heaps(
    _this: *mut c_void,
    _n: u32,
    _heaps: *const *mut c_void,
) {
    // Descriptor handles are absolute addresses: nothing to bind.
}

fn set_root_signature(this: *mut c_void, rs: *mut c_void, graphics: bool) {
    let Some(r) = (unsafe { inner::<RootSignatureInner>(rs) }) else {
        return;
    };
    let mut st = cl(this).state.lock().unwrap();
    let root = if graphics {
        &mut st.graphics
    } else {
        &mut st.compute
    };
    if root
        .layout
        .as_ref()
        .is_some_and(|l| Arc::ptr_eq(l, &r.layout))
    {
        return;
    }
    root.layout = Some(Arc::clone(&r.layout));
    root.shadow = [0; 256];
    root.sampler_table = r.samplers.map_or(0, |b| b.gpu_va);
    if let (Some(s), true) = (r.layout.sampler_table_sgpr, root.sampler_table != 0) {
        let a = root.sampler_table;
        root.shadow[s as usize] = a as u32;
        root.shadow[s as usize + 1] = (a >> 32) as u32;
    }
    st.dirty = true;
}

unsafe extern "system" fn cl_set_compute_root_signature(this: *mut c_void, rs: *mut c_void) {
    set_root_signature(this, rs, false);
}

unsafe extern "system" fn cl_set_graphics_root_signature(this: *mut c_void, rs: *mut c_void) {
    set_root_signature(this, rs, true);
}

fn root_address(this: *mut c_void, graphics: bool, idx: u32, va: u64) {
    match param_sgpr(this, graphics, idx) {
        Some((s, _)) => set_root(this, graphics, s, &[va as u32, (va >> 32) as u32]),
        None => log_once("root parameter set without a root signature"),
    }
}

unsafe extern "system" fn cl_set_compute_root_table(this: *mut c_void, idx: u32, handle: u64) {
    root_address(this, false, idx, handle);
}

unsafe extern "system" fn cl_set_graphics_root_table(this: *mut c_void, idx: u32, handle: u64) {
    root_address(this, true, idx, handle);
}

fn root_constants(this: *mut c_void, graphics: bool, idx: u32, values: &[u32], offset: u32) {
    match param_sgpr(this, graphics, idx) {
        Some((s, _)) => set_root(this, graphics, s + offset as usize, values),
        // No root signature yet (the D3D12.X test path): constants go to SGPRs from 0.
        None => set_root(this, graphics, offset as usize, values),
    }
}

unsafe extern "system" fn cl_set_compute_root_constant(
    this: *mut c_void,
    idx: u32,
    v: u32,
    off: u32,
) {
    root_constants(this, false, idx, &[v], off);
}

unsafe extern "system" fn cl_set_graphics_root_constant(
    this: *mut c_void,
    idx: u32,
    v: u32,
    off: u32,
) {
    root_constants(this, true, idx, &[v], off);
}

unsafe fn slice_u32<'a>(p: *const u32, n: u32) -> &'a [u32] {
    if p.is_null() || n == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(p, n as usize) }
    }
}

unsafe extern "system" fn cl_set_compute_root_constants(
    this: *mut c_void,
    idx: u32,
    n: u32,
    src: *const u32,
    off: u32,
) {
    root_constants(this, false, idx, unsafe { slice_u32(src, n) }, off);
}

unsafe extern "system" fn cl_set_graphics_root_constants(
    this: *mut c_void,
    idx: u32,
    n: u32,
    src: *const u32,
    off: u32,
) {
    root_constants(this, true, idx, unsafe { slice_u32(src, n) }, off);
}

unsafe extern "system" fn cl_set_compute_root_view(this: *mut c_void, idx: u32, va: u64) {
    root_address(this, false, idx, va);
}

unsafe extern "system" fn cl_set_graphics_root_view(this: *mut c_void, idx: u32, va: u64) {
    root_address(this, true, idx, va);
}

unsafe extern "system" fn cl_ia_set_index_buffer(this: *mut c_void, view: *const IndexBufferView) {
    let cmd = if view.is_null() {
        ThinCommand::SetIndexBuffer {
            va: 0,
            size: 0,
            index32: false,
        }
    } else {
        let v = unsafe { *view };
        ThinCommand::SetIndexBuffer {
            va: v.location,
            size: v.size,
            index32: v.format == 42,
        }
    };
    record(this, cmd);
}

unsafe extern "system" fn cl_ia_set_vertex_buffers(
    this: *mut c_void,
    start: u32,
    n: u32,
    views: *const VertexBufferView,
) {
    let views = if views.is_null() {
        vec![(0, 0, 0); n as usize]
    } else {
        unsafe { std::slice::from_raw_parts(views, n as usize) }
            .iter()
            .map(|v| (v.location, v.size, v.stride))
            .collect()
    };
    record(
        this,
        ThinCommand::SetVertexBuffers {
            start_slot: start,
            views,
        },
    );
}

unsafe extern "system" fn cl_so_set_targets(
    this: *mut c_void,
    start: u32,
    n: u32,
    views: *const c_void,
) {
    if n == 0 || views.is_null() {
        record(
            this,
            ThinCommand::SetStreamOutputTargets {
                start_slot: start,
                views: Vec::new(),
            },
        );
        return;
    }
    // D3D12_STREAM_OUTPUT_BUFFER_VIEW: BufferLocation (u64), SizeInBytes (u64), BufferFilledSizeLocation (u64)
    let raw = unsafe { std::slice::from_raw_parts(views as *const [u64; 3], n as usize) };
    let out = raw.iter().map(|v| (v[0], v[1], v[2])).collect();
    record(
        this,
        ThinCommand::SetStreamOutputTargets {
            start_slot: start,
            views: out,
        },
    );
}

/// The 8 dwords a CPU descriptor handle points at.
fn read_descriptor(handle: u64) -> Option<[u32; 8]> {
    (handle != 0).then(|| unsafe { std::ptr::read_unaligned(handle as *const [u32; 8]) })
}

unsafe extern "system" fn cl_om_set_render_targets(
    this: *mut c_void,
    n: u32,
    rtvs: *const u64,
    single_range: i32,
    dsv: *const u64,
) {
    let colors: Vec<[u32; 8]> = (0..n as usize)
        .filter_map(|i| {
            if rtvs.is_null() {
                return None;
            }
            let h = if single_range != 0 {
                (unsafe { *rtvs }) + 32 * i as u64
            } else {
                unsafe { *rtvs.add(i) }
            };
            read_descriptor(h)
        })
        .collect();
    let depth = if dsv.is_null() {
        None
    } else {
        read_descriptor(unsafe { *dsv })
    };
    record(this, ThinCommand::BindTargets { colors, depth });
}

unsafe extern "system" fn cl_clear_dsv(
    this: *mut c_void,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    _n: u32,
    _rects: *const Rect,
) {
    let Some(words) = read_descriptor(handle) else {
        return;
    };
    record(
        this,
        ThinCommand::ClearImage {
            words,
            aspects: ClearAspects {
                color: false,
                depth: flags & 1 != 0,
                stencil: flags & 2 != 0,
            },
            color: [0; 4],
            depth: depth.to_bits(),
            stencil: u32::from(stencil),
        },
    );
}

unsafe extern "system" fn cl_clear_rtv(
    this: *mut c_void,
    handle: u64,
    color: *const f32,
    _n: u32,
    _rects: *const Rect,
) {
    let Some(words) = read_descriptor(handle) else {
        return;
    };
    let c = if color.is_null() {
        [0.0; 4]
    } else {
        unsafe { std::ptr::read_unaligned(color as *const [f32; 4]) }
    };
    record(
        this,
        ThinCommand::ClearImage {
            words,
            aspects: ClearAspects {
                color: true,
                ..Default::default()
            },
            color: c.map(f32::to_bits),
            depth: 0,
            stencil: 0,
        },
    );
}

/// Strata marks buffer views it writes into heaps: `'SB'` in dword 5's top
/// half, the byte length in dwords 6–7.
const BUFFER_VIEW_MAGIC: u32 = 0x5342_0000;

fn clear_uav(this: *mut c_void, cpu_handle: u64, values: [u32; 4]) {
    let Some(w) = read_descriptor(cpu_handle) else {
        return;
    };
    if w[5] & 0xffff_0000 == BUFFER_VIEW_MAGIC {
        let va = u64::from(w[0]) | (u64::from(w[1] & 0xffff) << 32);
        let len = u64::from(w[6]) | (u64::from(w[7]) << 32);
        record(
            this,
            ThinCommand::FillMemory {
                va,
                len,
                pattern: [values[0]; 4],
                dwords: 1,
            },
        );
    } else {
        record(
            this,
            ThinCommand::ClearImage {
                words: w,
                aspects: ClearAspects {
                    color: true,
                    ..Default::default()
                },
                color: values,
                depth: 0,
                stencil: 0,
            },
        );
    }
}

unsafe extern "system" fn cl_clear_uav_uint(
    this: *mut c_void,
    _gpu: u64,
    cpu: u64,
    _res: *mut c_void,
    values: *const u32,
    _n: u32,
    _rects: *const Rect,
) {
    let v = if values.is_null() {
        [0; 4]
    } else {
        unsafe { std::ptr::read_unaligned(values as *const [u32; 4]) }
    };
    // Images are float views: the integers clear as their float values.
    let w = read_descriptor(cpu);
    let as_image = w.is_some_and(|w| w[5] & 0xffff_0000 != BUFFER_VIEW_MAGIC);
    clear_uav(
        this,
        cpu,
        if as_image {
            v.map(|x| (x as f32).to_bits())
        } else {
            v
        },
    );
}

unsafe extern "system" fn cl_clear_uav_float(
    this: *mut c_void,
    _gpu: u64,
    cpu: u64,
    _res: *mut c_void,
    values: *const f32,
    _n: u32,
    _rects: *const Rect,
) {
    let v = if values.is_null() {
        [0.0; 4]
    } else {
        unsafe { std::ptr::read_unaligned(values as *const [f32; 4]) }
    };
    clear_uav(this, cpu, v.map(f32::to_bits));
}

unsafe extern "system" fn cl_discard_resource(
    _this: *mut c_void,
    _res: *mut c_void,
    _region: *const c_void,
) {
}

unsafe extern "system" fn cl_begin_query(this: *mut c_void, heap: *mut c_void, t: u32, i: u32) {
    let Some(h) = (unsafe { inner::<QueryHeapInner>(heap) }) else {
        return;
    };
    let va = h.mem.gpu_va + h.stride * u64::from(i);
    record(this, ThinCommand::BeginQuery { va, query_type: t });
}

unsafe extern "system" fn cl_end_query(this: *mut c_void, heap: *mut c_void, t: u32, i: u32) {
    let Some(h) = (unsafe { inner::<QueryHeapInner>(heap) }) else {
        return;
    };
    let va = h.mem.gpu_va + h.stride * u64::from(i);
    record(this, ThinCommand::EndQuery { va, query_type: t });
}

unsafe extern "system" fn cl_resolve_query_data(
    this: *mut c_void,
    heap: *mut c_void,
    _t: u32,
    start: u32,
    n: u32,
    dst: *mut c_void,
    dst_offset: u64,
) {
    let (Some(h), Some(d)) = (unsafe { inner::<QueryHeapInner>(heap) }, unsafe {
        resource(dst)
    }) else {
        return;
    };
    record(
        this,
        ThinCommand::CopyMemory {
            dst: d.mem.gpu_va + dst_offset,
            src: h.mem.gpu_va + h.stride * u64::from(start),
            len: h.stride * u64::from(n),
        },
    );
}

unsafe extern "system" fn cl_set_predication(
    this: *mut c_void,
    buf: *mut c_void,
    offset: u64,
    op: u32,
) {
    let va = unsafe { resource(buf) }.map_or(0, |r| r.mem.gpu_va + offset);
    record(
        this,
        ThinCommand::SetPredication {
            va,
            skip_if_zero: op == 0,
        },
    );
}

unsafe extern "system" fn cl_marker(
    _this: *mut c_void,
    _meta: u32,
    _data: *const c_void,
    _size: u32,
) {
}

unsafe extern "system" fn cl_end_event(_this: *mut c_void) {}

unsafe extern "system" fn cl_execute_indirect(
    this: *mut c_void,
    sig: *mut c_void,
    max_count: u32,
    args: *mut c_void,
    args_offset: u64,
    count: *mut c_void,
    count_offset: u64,
) {
    let (Some(s), Some(a)) = (unsafe { inner::<CommandSignatureInner>(sig) }, unsafe {
        resource(args)
    }) else {
        return;
    };
    let graphics = s.layout.args.iter().any(|x| {
        matches!(
            x,
            IndirectArg::Draw | IndirectArg::DrawIndexed | IndirectArg::DispatchMesh
        )
    });
    flush_root(this, graphics);
    let count_va = unsafe { resource(count) }.map_or(0, |c| c.mem.gpu_va + count_offset);
    record(
        this,
        ThinCommand::ExecuteIndirect {
            layout: Arc::clone(&s.layout),
            max_count,
            args_va: a.mem.gpu_va + args_offset,
            count_va,
        },
    );
}

unsafe extern "system" fn cl_atomic_copy_buffer(
    this: *mut c_void,
    dst: *mut c_void,
    dst_offset: u64,
    src: *mut c_void,
    src_offset: u64,
    _n_deps: u32,
    _deps: *const *mut c_void,
    _ranges: *const c_void,
) {
    // Execution is serial: a plain copy is atomic with respect to the GPU.
    let (Some(d), Some(s)) = (unsafe { resource(dst) }, unsafe { resource(src) }) else {
        return;
    };
    record(
        this,
        ThinCommand::CopyMemory {
            dst: d.mem.gpu_va + dst_offset,
            src: s.mem.gpu_va + src_offset,
            len: 8,
        },
    );
}

unsafe extern "system" fn cl_om_set_depth_bounds(this: *mut c_void, min: f32, max: f32) {
    record(
        this,
        ThinCommand::SetDepthBounds {
            min: min.to_bits(),
            max: max.to_bits(),
        },
    );
}

unsafe extern "system" fn cl_copy_tiles(
    this: *mut c_void,
    tiled_resource: *mut c_void,
    _tile_region_start_coordinate: *const c_void,
    tile_region_size: *const u32,
    buffer: *mut c_void,
    buffer_start_offset: u64,
    flags: u32,
) {
    let (Some(tr), Some(buf)) = (unsafe { resource(tiled_resource) }, unsafe {
        resource(buffer)
    }) else {
        return;
    };
    let num_tiles = if tile_region_size.is_null() {
        1
    } else {
        unsafe { (*tile_region_size).max(1) }
    };
    // D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE = 2
    let to_tiled = (flags & 2) != 0;
    record(
        this,
        ThinCommand::CopyTiles {
            tiled_va: tr.mem.gpu_va,
            buffer_va: buf.mem.gpu_va + buffer_start_offset,
            tile_count: num_tiles,
            tile_bytes: 65_536,
            to_tiled,
        },
    );
}

unsafe extern "system" fn cl_resolve_subresource_region(
    this: *mut c_void,
    dst: *mut c_void,
    dst_sub: u32,
    dst_x: u32,
    dst_y: u32,
    src: *mut c_void,
    src_sub: u32,
    src_rect: *const Rect,
    _format: u32,
    _mode: u32,
) {
    let (Some(d), Some(s)) = (unsafe { resource(dst) }, unsafe { resource(src) }) else {
        return;
    };
    let dt = TextureLayout::of(&d.desc);
    let st = TextureLayout::of(&s.desc);
    let (dm, _) = dt.subresource(dst_sub);
    let (sm, _) = st.subresource(src_sub);
    let (sw, sh) = st.extent(sm);
    let [sx, sy, rw, rh] = if src_rect.is_null() {
        [0, 0, sw, sh]
    } else {
        let r = unsafe { *src_rect };
        [
            r.left.max(0) as u32,
            r.top.max(0) as u32,
            (r.right - r.left).max(0) as u32,
            (r.bottom - r.top).max(0) as u32,
        ]
    };
    record(
        this,
        ThinCommand::ResolveSubresourceRegion {
            dst: Surface {
                va: d.mem.gpu_va + d.subresource_offset(dst_sub),
                row_pitch: dt.row_pitch(dm),
                slice_pitch: dt.slice_bytes(dm),
                bpe: dt.bytes,
            },
            dst_origin: [dst_x, dst_y],
            src: Surface {
                va: s.mem.gpu_va + s.subresource_offset(src_sub),
                row_pitch: st.row_pitch(sm),
                slice_pitch: st.slice_bytes(sm),
                bpe: st.bytes,
            },
            src_rect: [sx, sy, rw, rh],
            sample_count: s.desc.sample_count.max(1),
        },
    );
}

unsafe extern "system" fn cl_end_render_pass(this: *mut c_void) {
    let resolves = {
        let Some(cl) = (unsafe { inner::<CommandListInner>(this) }) else {
            return;
        };
        let mut s = cl.state.lock().unwrap();
        std::mem::take(&mut s.render_pass_resolves)
    };
    for cmd in resolves {
        record(this, cmd);
    }
}

unsafe extern "system" fn cl_write_buffer_immediate(
    this: *mut c_void,
    n: u32,
    params: *const [u32; 4],
    _modes: *const u32,
) {
    if params.is_null() {
        return;
    }
    // D3D12_WRITEBUFFERIMMEDIATE_PARAMETER: Dest (u64), Value (u32), padding.
    for p in unsafe { std::slice::from_raw_parts(params, n as usize) } {
        let va = u64::from(p[0]) | (u64::from(p[1]) << 32);
        record(
            this,
            ThinCommand::FillMemory {
                va,
                len: 4,
                pattern: [p[2]; 4],
                dwords: 1,
            },
        );
    }
}

/// `BeginRenderPass`: binds the targets and runs the `CLEAR` beginning
/// accesses. Layouts: `D3D12_RENDER_PASS_RENDER_TARGET_DESC` is 88 bytes
/// (handle, beginning access type at 8, clear format at 12, colour at 16);
/// `D3D12_RENDER_PASS_DEPTH_STENCIL_DESC` is 168 bytes (handle, depth
/// beginning access at 8 with its depth at 16, stencil access at 32 with its
/// stencil at 44).
unsafe extern "system" fn cl_begin_render_pass(
    this: *mut c_void,
    n: u32,
    rts: *const u8,
    ds: *const u8,
    _flags: u32,
) {
    let rd32 = |p: *const u8, o: usize| unsafe { (p.add(o) as *const u32).read_unaligned() };
    let rd64 = |p: *const u8, o: usize| unsafe { (p.add(o) as *const u64).read_unaligned() };
    let mut colors = Vec::new();
    for i in 0..n as usize {
        if rts.is_null() {
            break;
        }
        let d = unsafe { rts.add(88 * i) };
        let handle = rd64(d, 0);
        let Some(words) = read_descriptor(handle) else {
            continue;
        };
        if rd32(d, 8) == 2 {
            let c = [rd32(d, 16), rd32(d, 20), rd32(d, 24), rd32(d, 28)];
            record(
                this,
                ThinCommand::ClearImage {
                    words,
                    aspects: ClearAspects {
                        color: true,
                        ..Default::default()
                    },
                    color: c,
                    depth: 0,
                    stencil: 0,
                },
            );
        }
        colors.push(words);
    }
    let depth = if ds.is_null() {
        None
    } else {
        let words = read_descriptor(rd64(ds, 0));
        if let Some(w) = words {
            let (cd, cs) = (rd32(ds, 8) == 2, rd32(ds, 32) == 2);
            if cd || cs {
                record(
                    this,
                    ThinCommand::ClearImage {
                        words: w,
                        aspects: ClearAspects {
                            color: false,
                            depth: cd,
                            stencil: cs,
                        },
                        color: [0; 4],
                        depth: rd32(ds, 16),
                        stencil: rd32(ds, 44) & 0xff,
                    },
                );
            }
        }
        words
    };
    record(this, ThinCommand::BindTargets { colors, depth });
}

/// The geometries of a bottom-level build, from title memory.
unsafe fn build_inputs(inputs: &BuildRaytracingInputs) -> (Vec<AccelGeometry>, (u64, u32)) {
    if inputs.as_type == 0 {
        // TOP_LEVEL: `descs` is the instance array's GPU address.
        return (Vec::new(), (inputs.descs, inputs.num_descs));
    }
    let mut geoms = Vec::new();
    for i in 0..inputs.num_descs as usize {
        let d: RaytracingGeometryDesc = unsafe {
            if inputs.descs_layout == 1 {
                // ARRAY_OF_POINTERS
                let pp = *(inputs.descs as *const *const RaytracingGeometryDesc).add(i);
                *pp
            } else {
                *(inputs.descs as *const RaytracingGeometryDesc).add(i)
            }
        };
        let q = |k: usize| u64::from(d.u[k]) | (u64::from(d.u[k + 1]) << 32);
        let opaque = d.flags & 1 != 0;
        geoms.push(if d.geometry_type == 0 {
            AccelGeometry::Triangles {
                transform: q(0),
                index_format: d.u[2],
                vertex_format: d.u[3],
                index_count: d.u[4],
                vertex_count: d.u[5],
                index_va: q(6),
                vertex_va: q(8),
                vertex_stride: q(10),
                opaque,
            }
        } else {
            AccelGeometry::Aabbs {
                count: q(0),
                va: q(2),
                stride: q(4),
                opaque,
            }
        });
    }
    (geoms, (0, 0))
}

unsafe extern "system" fn cl_build_raytracing_as(
    this: *mut c_void,
    desc: *const BuildRaytracingDesc,
    n_post: u32,
    post: *const [u64; 2],
) {
    if desc.is_null() {
        return;
    }
    let d = unsafe { *desc };
    let (geometries, instances) = unsafe { build_inputs(&d.inputs) };
    let prims: u64 = if d.inputs.as_type == 0 {
        u64::from(d.inputs.num_descs)
    } else {
        geometries
            .iter()
            .map(|g| match g {
                AccelGeometry::Triangles {
                    index_count,
                    vertex_count,
                    index_format,
                    ..
                } => u64::from(
                    if *index_format == 0 {
                        *vertex_count
                    } else {
                        *index_count
                    } / 3,
                ),
                AccelGeometry::Aabbs { count, .. } => *count,
            })
            .sum()
    };
    let leaf = if d.inputs.as_type == 0 {
        crate::d3d12_rt::INSTANCE_BYTES
    } else {
        crate::d3d12_rt::TRIANGLE_BYTES
    };
    record(
        this,
        ThinCommand::BuildAccel(Box::new(AccelBuild {
            dest: d.dest,
            dest_size: crate::d3d12_rt::max_size(prims, leaf),
            top_level: d.inputs.as_type == 0,
            geometries,
            instances,
        })),
    );
    if n_post > 0 && !post.is_null() {
        let srcs = [d.dest];
        unsafe { cl_emit_postbuild(this, post as *const u8, 1, srcs.as_ptr()) };
    }
}

/// `EmitRaytracingAccelerationStructurePostbuildInfo`: every info type is
/// answered from the structure's used size (header bytes 8..16).
unsafe extern "system" fn cl_emit_postbuild(
    this: *mut c_void,
    desc: *const u8,
    n: u32,
    srcs: *const u64,
) {
    if desc.is_null() || srcs.is_null() {
        return;
    }
    let dest = unsafe { (desc as *const u64).read_unaligned() };
    let info = unsafe { (desc.add(8) as *const u32).read_unaligned() };
    for i in 0..n as u64 {
        let src = unsafe { *srcs.add(i as usize) };
        let stride = if info == 2 { 16 } else { 8 };
        record(
            this,
            ThinCommand::CopyMemory {
                dst: dest + i * stride,
                src: src + 8,
                len: 8,
            },
        );
        if info == 2 {
            record(
                this,
                ThinCommand::WriteU64 {
                    va: dest + i * stride + 8,
                    value: 0,
                },
            );
        }
    }
}

unsafe extern "system" fn cl_copy_raytracing_as(this: *mut c_void, dst: u64, src: u64, _mode: u32) {
    record(this, ThinCommand::CopyAccel { dst, src });
}

unsafe extern "system" fn cl_set_pipeline_state1(this: *mut c_void, so: *mut c_void) {
    if let Some(s) = unsafe { inner::<StateObjectInner>(so) } {
        record(this, ThinCommand::BindPipeline(s.key));
    }
}

unsafe extern "system" fn cl_dispatch_rays(this: *mut c_void, desc: *const c_void) {
    if desc.is_null() {
        return;
    }
    // D3D12_DISPATCH_RAYS_DESC (104 bytes = 26 u32s):
    //   0..4:   RayGenerationShaderRecord { StartAddress: u64, SizeInBytes: u64 }
    //   4..10:  MissShaderTable { StartAddress: u64, SizeInBytes: u64, StrideInBytes: u64 }
    //   10..16: HitGroupTable { StartAddress: u64, SizeInBytes: u64, StrideInBytes: u64 }
    //   16..22: CallableShaderTable { StartAddress: u64, SizeInBytes: u64, StrideInBytes: u64 }
    //   22..25: Width, Height, Depth (u32 each)
    let d = unsafe { std::slice::from_raw_parts(desc as *const u32, 26) };
    let q = |k: usize| u64::from(d[k]) | (u64::from(d[k + 1]) << 32);
    flush_root(this, false);
    record(
        this,
        ThinCommand::DispatchRays {
            raygen: (q(0), q(2)),
            miss: (q(4), q(6), q(8)),
            hit: (q(10), q(12), q(14)),
            callable: (q(16), q(18), q(20)),
            width: d[22],
            height: d[23],
            depth: d[24],
        },
    );
}

unsafe extern "system" fn cl_rs_set_shading_rate(
    this: *mut c_void,
    rate: u32,
    combiners: *const u32,
) {
    // D3D12_SHADING_RATE: X shift in bits [3:2], Y shift in bits [1:0].
    let width = 1u32 << ((rate >> 2) & 0x3);
    let height = 1u32 << (rate & 0x3);
    let comb = if combiners.is_null() {
        [0, 0]
    } else {
        unsafe { [*combiners, *combiners.add(1)] }
    };
    record(
        this,
        ThinCommand::SetShadingRate {
            width,
            height,
            combiners: comb,
        },
    );
}

unsafe extern "system" fn cl_rs_set_shading_rate_image(this: *mut c_void, image: *mut c_void) {
    let va = unsafe { resource(image) }.map_or(0, |r| r.mem.gpu_va);
    record(this, ThinCommand::SetShadingRateImage { va });
}

unsafe extern "system" fn cl_dispatch_mesh(this: *mut c_void, x: u32, y: u32, z: u32) {
    flush_root(this, true);
    record(this, ThinCommand::DispatchMesh { x, y, z });
}

static CMDLIST_VTABLE: [VTableSlot; CMDLIST_VTABLE_LEN] = {
    let mut vt = base_slots::<CMDLIST_VTABLE_LEN>();
    let mut i = 8;
    while i < CMDLIST_VTABLE_LEN {
        vt[i] = slot!(stub_void);
        i += 1;
    }
    vt[CMDLIST_SLOT_GET_TYPE] = slot!(cl_get_type);
    vt[CMDLIST_SLOT_CLOSE] = slot!(cl_close);
    vt[CMDLIST_SLOT_RESET] = slot!(cl_reset);
    vt[CMDLIST_SLOT_CLEAR_STATE] = slot!(cl_clear_state);
    vt[CMDLIST_SLOT_DRAW_INSTANCED] = slot!(cl_draw_instanced);
    vt[CMDLIST_SLOT_DRAW_INDEXED_INSTANCED] = slot!(cl_draw_indexed_instanced);
    vt[CMDLIST_SLOT_DISPATCH] = slot!(cl_dispatch);
    vt[CMDLIST_SLOT_COPY_BUFFER_REGION] = slot!(cl_copy_buffer_region);
    vt[CMDLIST_SLOT_COPY_TEXTURE_REGION] = slot!(cl_copy_texture_region);
    vt[CMDLIST_SLOT_COPY_RESOURCE] = slot!(cl_copy_resource);
    vt[CMDLIST_SLOT_COPY_TILES] = slot!(cl_copy_tiles);
    vt[CMDLIST_SLOT_RESOLVE_SUBRESOURCE] = slot!(cl_resolve_subresource);
    vt[CMDLIST_SLOT_IA_SET_PRIMITIVE_TOPOLOGY] = slot!(cl_ia_set_topology);
    vt[CMDLIST_SLOT_RS_SET_VIEWPORTS] = slot!(cl_rs_set_viewports);
    vt[CMDLIST_SLOT_RS_SET_SCISSOR_RECTS] = slot!(cl_rs_set_scissor_rects);
    vt[CMDLIST_SLOT_OM_SET_BLEND_FACTOR] = slot!(cl_om_set_blend_factor);
    vt[CMDLIST_SLOT_OM_SET_STENCIL_REF] = slot!(cl_om_set_stencil_ref);
    vt[CMDLIST_SLOT_SET_PIPELINE_STATE] = slot!(cl_set_pso);
    vt[CMDLIST_SLOT_RESOURCE_BARRIER] = slot!(cl_resource_barrier);
    vt[CMDLIST_SLOT_EXECUTE_BUNDLE] = slot!(cl_execute_bundle);
    vt[CMDLIST_SLOT_SET_DESCRIPTOR_HEAPS] = slot!(cl_set_descriptor_heaps);
    vt[CMDLIST_SLOT_SET_COMPUTE_ROOT_SIGNATURE] = slot!(cl_set_compute_root_signature);
    vt[CMDLIST_SLOT_SET_GRAPHICS_ROOT_SIGNATURE] = slot!(cl_set_graphics_root_signature);
    vt[CMDLIST_SLOT_SET_COMPUTE_ROOT_DESCRIPTOR_TABLE] = slot!(cl_set_compute_root_table);
    vt[CMDLIST_SLOT_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE] = slot!(cl_set_graphics_root_table);
    vt[CMDLIST_SLOT_SET_COMPUTE_ROOT_32BIT_CONSTANT] = slot!(cl_set_compute_root_constant);
    vt[CMDLIST_SLOT_SET_GRAPHICS_ROOT_32BIT_CONSTANT] = slot!(cl_set_graphics_root_constant);
    vt[CMDLIST_SLOT_SET_COMPUTE_ROOT_32BIT_CONSTANTS] = slot!(cl_set_compute_root_constants);
    vt[CMDLIST_SLOT_SET_GRAPHICS_ROOT_32BIT_CONSTANTS] = slot!(cl_set_graphics_root_constants);
    vt[CMDLIST_SLOT_SET_COMPUTE_ROOT_CBV] = slot!(cl_set_compute_root_view);
    vt[CMDLIST_SLOT_SET_GRAPHICS_ROOT_CBV] = slot!(cl_set_graphics_root_view);
    vt[CMDLIST_SLOT_SET_COMPUTE_ROOT_SRV] = slot!(cl_set_compute_root_view);
    vt[CMDLIST_SLOT_SET_GRAPHICS_ROOT_SRV] = slot!(cl_set_graphics_root_view);
    vt[CMDLIST_SLOT_SET_COMPUTE_ROOT_UAV] = slot!(cl_set_compute_root_view);
    vt[CMDLIST_SLOT_SET_GRAPHICS_ROOT_UAV] = slot!(cl_set_graphics_root_view);
    vt[CMDLIST_SLOT_IA_SET_INDEX_BUFFER] = slot!(cl_ia_set_index_buffer);
    vt[CMDLIST_SLOT_IA_SET_VERTEX_BUFFERS] = slot!(cl_ia_set_vertex_buffers);
    vt[CMDLIST_SLOT_SO_SET_TARGETS] = slot!(cl_so_set_targets);
    vt[CMDLIST_SLOT_OM_SET_RENDER_TARGETS] = slot!(cl_om_set_render_targets);
    vt[CMDLIST_SLOT_CLEAR_DEPTH_STENCIL_VIEW] = slot!(cl_clear_dsv);
    vt[CMDLIST_SLOT_CLEAR_RENDER_TARGET_VIEW] = slot!(cl_clear_rtv);
    vt[CMDLIST_SLOT_CLEAR_UAV_UINT] = slot!(cl_clear_uav_uint);
    vt[CMDLIST_SLOT_CLEAR_UAV_FLOAT] = slot!(cl_clear_uav_float);
    vt[CMDLIST_SLOT_DISCARD_RESOURCE] = slot!(cl_discard_resource);
    vt[CMDLIST_SLOT_BEGIN_QUERY] = slot!(cl_begin_query);
    vt[CMDLIST_SLOT_END_QUERY] = slot!(cl_end_query);
    vt[CMDLIST_SLOT_RESOLVE_QUERY_DATA] = slot!(cl_resolve_query_data);
    vt[CMDLIST_SLOT_SET_PREDICATION] = slot!(cl_set_predication);
    vt[CMDLIST_SLOT_SET_MARKER] = slot!(cl_marker);
    vt[CMDLIST_SLOT_BEGIN_EVENT] = slot!(cl_marker);
    vt[CMDLIST_SLOT_END_EVENT] = slot!(cl_end_event);
    vt[CMDLIST_SLOT_EXECUTE_INDIRECT] = slot!(cl_execute_indirect);
    vt[CMDLIST_SLOT_ATOMIC_COPY_BUFFER_UINT] = slot!(cl_atomic_copy_buffer);
    vt[CMDLIST_SLOT_ATOMIC_COPY_BUFFER_UINT64] = slot!(cl_atomic_copy_buffer);
    vt[CMDLIST_SLOT_OM_SET_DEPTH_BOUNDS] = slot!(cl_om_set_depth_bounds);
    vt[CMDLIST_SLOT_RESOLVE_SUBRESOURCE_REGION] = slot!(cl_resolve_subresource_region);
    vt[CMDLIST_SLOT_WRITE_BUFFER_IMMEDIATE] = slot!(cl_write_buffer_immediate);
    vt[CMDLIST_SLOT_BEGIN_RENDER_PASS] = slot!(cl_begin_render_pass);
    vt[CMDLIST_SLOT_END_RENDER_PASS] = slot!(cl_end_render_pass);
    vt[CMDLIST_SLOT_BUILD_RAYTRACING_AS] = slot!(cl_build_raytracing_as);
    vt[CMDLIST_SLOT_EMIT_RAYTRACING_AS_POSTBUILD_INFO] = slot!(cl_emit_postbuild);
    vt[CMDLIST_SLOT_COPY_RAYTRACING_AS] = slot!(cl_copy_raytracing_as);
    vt[CMDLIST_SLOT_SET_PIPELINE_STATE1] = slot!(cl_set_pipeline_state1);
    vt[CMDLIST_SLOT_DISPATCH_RAYS] = slot!(cl_dispatch_rays);
    vt[CMDLIST_SLOT_RS_SET_SHADING_RATE] = slot!(cl_rs_set_shading_rate);
    vt[CMDLIST_SLOT_RS_SET_SHADING_RATE_IMAGE] = slot!(cl_rs_set_shading_rate_image);
    vt[CMDLIST_SLOT_DISPATCH_MESH] = slot!(cl_dispatch_mesh);
    vt
};

impl Com<CommandListInner> {
    pub fn new_boxed(list_type: u32, device: *mut c_void) -> *mut Self {
        Com::boxed(
            &CMDLIST_VTABLE,
            device,
            CommandListInner {
                list_type,
                closed: AtomicBool::new(false),
                thin: Mutex::new(ThinCommandList::new()),
                state: Mutex::new(ListState::default()),
            },
        )
    }
}

// ============================================================================
// ID3D12CommandQueue (19 slots)
// ============================================================================

pub struct QueueInner {
    pub desc: D3D12CommandQueueDesc,
    /// The engine that executes this queue's lists (`None`: no GPU).
    pub engine: Option<EnginePtr>,
    pub executed_batches: AtomicU64,
    pub executed_lists: AtomicU64,
}

/// An owned engine queue (released with the D3D12 queue).
pub struct EnginePtr(pub *mut D3D12CommandQueueX);

// SAFETY: the engine is internally synchronised.
unsafe impl Send for EnginePtr {}
unsafe impl Sync for EnginePtr {}

impl Drop for EnginePtr {
    fn drop(&mut self) {
        unsafe { release(self.0.cast()) };
    }
}

pub type StandardCommandQueue = Com<QueueInner>;

unsafe extern "system" fn q_execute_command_lists(
    this: *mut c_void,
    n: u32,
    lists: *const *mut c_void,
) {
    let q = unsafe { &Com::<QueueInner>::from(this).inner };
    q.executed_batches.fetch_add(1, Ordering::Relaxed);
    q.executed_lists.fetch_add(u64::from(n), Ordering::Relaxed);
    if lists.is_null() || n == 0 {
        return;
    }
    let Some(e) = &q.engine else {
        return;
    };
    let slice = unsafe { std::slice::from_raw_parts(lists, n as usize) };
    if slice.len() == 1 {
        if let Some(l) = unsafe { inner::<CommandListInner>(slice[0]) } {
            let thin = l.thin.lock().unwrap();
            unsafe { (*e.0).execute_thin(&thin) };
        }
        return;
    }
    let mut batch = Vec::with_capacity(slice.len());
    for &p in slice {
        if let Some(l) = unsafe { inner::<CommandListInner>(p) } {
            let thin = l.thin.lock().unwrap();
            let mut copy = ThinCommandList::new();
            for cmd in thin.commands() {
                copy.push(cmd.clone());
            }
            copy.hoisted_clamps = thin.hoisted_clamps.clone();
            batch.push(copy);
        }
    }
    if !batch.is_empty() {
        unsafe { (*e.0).execute_command_lists(batch) };
    }
}

unsafe extern "system" fn q_present_x(
    _this: *mut c_void,
    plane_count: u32,
    planes: *const c_void,
    _params: *const c_void,
) -> i32 {
    unsafe { crate::thunks::present_frame(plane_count, planes) }
}

unsafe extern "system" fn q_signal(_this: *mut c_void, fence: *mut c_void, value: u64) -> i32 {
    match unsafe { inner::<FenceInner>(fence) } {
        Some(f) => {
            // Submissions finish before ExecuteCommandLists returns: the
            // fence completes now.
            f.signal(value);
            S_OK
        }
        None => E_INVALIDARG,
    }
}

unsafe extern "system" fn q_wait(_this: *mut c_void, fence: *mut c_void, value: u64) -> i32 {
    // A GPU wait for another queue: block until its Signal (bounded, so a
    // title waiting on work it never submits cannot hang forever).
    match unsafe { inner::<FenceInner>(fence) } {
        Some(f) => {
            if !f.wait(value, std::time::Duration::from_secs(2)) {
                log_once("ID3D12CommandQueue::Wait timed out (fence never signalled)");
            }
            S_OK
        }
        None => E_INVALIDARG,
    }
}

unsafe extern "system" fn q_get_timestamp_frequency(_this: *mut c_void, out: *mut u64) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    unsafe { *out = 1_000_000_000 };
    S_OK
}

unsafe extern "system" fn q_get_clock_calibration(
    this: *mut c_void,
    gpu: *mut u64,
    cpu: *mut u64,
) -> i32 {
    let q = unsafe { &Com::<QueueInner>::from(this).inner };
    if !gpu.is_null() {
        // The engine's timestamps count nanoseconds from the queue's start.
        let ns = match &q.engine {
            Some(e) => unsafe { (*e.0).timestamp_ns() },
            None => 0,
        };
        unsafe { *gpu = ns };
    }
    if !cpu.is_null() {
        #[cfg(windows)]
        unsafe {
            let mut c = 0i64;
            QueryPerformanceCounter(&mut c);
            *cpu = c as u64;
        }
    }
    S_OK
}

unsafe extern "system" fn q_get_desc(
    this: *mut c_void,
    out: *mut D3D12CommandQueueDesc,
) -> *mut D3D12CommandQueueDesc {
    if !out.is_null() {
        unsafe { *out = Com::<QueueInner>::from(this).inner.desc };
    }
    out
}

unsafe extern "system" fn q_update_tile_mappings(
    this: *mut c_void,
    res: *mut c_void,
    num_regions: u32,
    _region_coords: *const c_void,
    _region_sizes: *const c_void,
    heap: *mut c_void,
    num_ranges: u32,
    range_flags: *const u32,
    _heap_offsets: *const u32,
    range_tile_counts: *const u32,
    _flags: u32,
) {
    let q = unsafe { &Com::<QueueInner>::from(this).inner };
    let (Some(e), Some(r)) = (&q.engine, unsafe { resource(res) }) else {
        return;
    };
    let Some(uma) = (unsafe { (*e.0).uma.as_ref() }) else {
        return;
    };
    let mut tile_cursor = 0u32;
    let ranges = num_ranges.max(num_regions).max(1) as usize;
    let mut guard = uma.lock().unwrap();
    for i in 0..ranges {
        let flags = if !range_flags.is_null() && i < num_ranges as usize {
            unsafe { *range_flags.add(i) }
        } else {
            0
        };
        let count = if !range_tile_counts.is_null() && i < num_ranges as usize {
            unsafe { *range_tile_counts.add(i) }
        } else {
            1
        };
        // D3D12_TILE_RANGE_FLAG_NULL = 1, D3D12_TILE_RANGE_FLAG_SKIP = 2
        let unbind = (flags & 1) != 0 || heap.is_null();
        if (flags & 2) == 0 {
            guard.update_tile_mappings(r.mem.gpu_va, tile_cursor, count, unbind);
        }
        tile_cursor = tile_cursor.saturating_add(count);
    }
}

unsafe extern "system" fn q_copy_tile_mappings(
    this: *mut c_void,
    dst_res: *mut c_void,
    _dst_coord: *const c_void,
    src_res: *mut c_void,
    _src_coord: *const c_void,
    _region_size: *const c_void,
    _flags: u32,
) {
    let q = unsafe { &Com::<QueueInner>::from(this).inner };
    let (Some(e), Some(dst), Some(src)) = (&q.engine, unsafe { resource(dst_res) }, unsafe {
        resource(src_res)
    }) else {
        return;
    };
    if let Some(uma) = unsafe { (*e.0).uma.as_ref() } {
        let tiles = dst.mem.size.div_ceil(65_536).max(1) as u32;
        uma.lock()
            .unwrap()
            .copy_tile_mappings(dst.mem.gpu_va, src.mem.gpu_va, tiles);
    }
}

static QUEUE_VTABLE: [VTableSlot; QUEUE_VTABLE_LEN] = {
    let mut vt = base_slots::<QUEUE_VTABLE_LEN>();
    vt[QUEUE_SLOT_UPDATE_TILE_MAPPINGS] = slot!(q_update_tile_mappings);
    vt[QUEUE_SLOT_COPY_TILE_MAPPINGS] = slot!(q_copy_tile_mappings);
    vt[QUEUE_SLOT_EXECUTE_COMMAND_LISTS] = slot!(q_execute_command_lists);
    vt[QUEUE_SLOT_SET_MARKER] = slot!(stub_void);
    vt[QUEUE_SLOT_BEGIN_EVENT] = slot!(stub_void);
    vt[QUEUE_SLOT_END_EVENT] = slot!(stub_void);
    vt[QUEUE_SLOT_SIGNAL] = slot!(q_signal);
    vt[QUEUE_SLOT_WAIT] = slot!(q_wait);
    vt[QUEUE_SLOT_GET_TIMESTAMP_FREQUENCY] = slot!(q_get_timestamp_frequency);
    vt[QUEUE_SLOT_GET_CLOCK_CALIBRATION] = slot!(q_get_clock_calibration);
    vt[QUEUE_SLOT_GET_DESC] = slot!(q_get_desc);
    vt[QUEUE_SLOT_PRESENT_X] = slot!(q_present_x);
    vt
};

impl Com<QueueInner> {
    pub fn new_boxed(desc: D3D12CommandQueueDesc, device: *mut c_void) -> *mut Self {
        let engine = (!device.is_null())
            .then(|| device_inner(device).backend.clone())
            .flatten()
            .map(|b| {
                EnginePtr(D3D12CommandQueueX::new_boxed_shared(
                    Some(Arc::clone(&b.runner)),
                    Some(Arc::clone(&b.uma)),
                    Arc::clone(&b.pipelines),
                    Arc::clone(&b.textures),
                ))
            });
        Com::boxed(
            &QUEUE_VTABLE,
            device,
            QueueInner {
                desc,
                engine,
                executed_batches: AtomicU64::new(0),
                executed_lists: AtomicU64::new(0),
            },
        )
    }
}

// ============================================================================
// ID3D12Device (65 slots)
// ============================================================================

unsafe extern "system" fn dev_get_node_count(_this: *mut c_void) -> u32 {
    1
}

unsafe extern "system" fn dev_create_command_queue(
    this: *mut c_void,
    desc: *const D3D12CommandQueueDesc,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    let d = if desc.is_null() {
        D3D12CommandQueueDesc::default()
    } else {
        unsafe { *desc }
    };
    unsafe { hand_out(Com::<QueueInner>::new_boxed(d, this), out) }
}

unsafe extern "system" fn dev_create_command_allocator(
    this: *mut c_void,
    list_type: u32,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    unsafe {
        hand_out(
            Com::boxed(&ALLOCATOR_VTABLE, this, AllocatorInner { list_type }),
            out,
        )
    }
}

/// The first RDNA2 program in shader bytecode (XBCH / FSSM containers, or
/// raw code).
fn program_of(bytes: &[u8]) -> Option<Vec<u32>> {
    if bytes.is_empty() {
        return None;
    }
    strata_shader::coverage::extract_programs("pso.rdna2", bytes)
        .into_iter()
        .next()
        .map(|p| p.words)
}

fn next_pso_key(dev: *mut c_void) -> PipelineKey {
    // Keys above 2^48 never collide with addresses older callers used.
    PipelineKey((1 << 48) | device_inner(dev).next_pso.fetch_add(1, Ordering::Relaxed))
}

/// Fixed-function state of a graphics PSO description.
fn pso_state(d: &GraphicsPipelineStateDesc) -> GraphicsPsoState {
    use ash::vk;
    let mut s = GraphicsPsoState {
        topology: match d.primitive_topology_type {
            1 => vk::PrimitiveTopology::POINT_LIST,
            2 => vk::PrimitiveTopology::LINE_LIST,
            4 => vk::PrimitiveTopology::PATCH_LIST,
            _ => vk::PrimitiveTopology::TRIANGLE_LIST,
        },
        ..Default::default()
    };
    let r = &d.rasterizer;
    s.polygon_mode = if r.fill_mode == 2 {
        vk::PolygonMode::LINE
    } else {
        vk::PolygonMode::FILL
    };
    s.cull_mode = match r.cull_mode {
        2 => vk::CullModeFlags::FRONT,
        3 => vk::CullModeFlags::BACK,
        _ => vk::CullModeFlags::NONE,
    };
    s.front_face = if r.front_counter_clockwise != 0 {
        vk::FrontFace::COUNTER_CLOCKWISE
    } else {
        vk::FrontFace::CLOCKWISE
    };
    s.depth_bias_enable = r.depth_bias != 0 || r.slope_scaled_depth_bias != 0.0;
    s.depth_bias_constant = r.depth_bias as f32;
    s.depth_bias_slope = r.slope_scaled_depth_bias;
    s.depth_clamp = r.depth_clip_enable == 0;
    let ds = &d.depth_stencil;
    s.depth_test_enable = ds.depth_enable != 0;
    s.depth_write_enable = ds.depth_enable != 0 && ds.depth_write_mask != 0;
    s.depth_compare_op = vk::CompareOp::from_raw(vk_compare(ds.depth_func));
    s.stencil_test_enable = ds.stencil_enable != 0;
    let face = |f: &DepthStencilOpDesc| {
        vk::StencilOpState::default()
            .fail_op(vk::StencilOp::from_raw(vk_stencil_op(f.fail_op)))
            .depth_fail_op(vk::StencilOp::from_raw(vk_stencil_op(f.depth_fail_op)))
            .pass_op(vk::StencilOp::from_raw(vk_stencil_op(f.pass_op)))
            .compare_op(vk::CompareOp::from_raw(vk_compare(f.func)))
            .compare_mask(u32::from(ds.stencil_read_mask))
            .write_mask(u32::from(ds.stencil_write_mask))
    };
    s.stencil_front = face(&ds.front);
    s.stencil_back = face(&ds.back);
    s.alpha_to_coverage = d.blend.alpha_to_coverage != 0;
    s.depth_format = dxgi_format(d.dsv_format)
        .and_then(|f| f.depth)
        .map(vk::Format::from_raw);
    s.blend_attachments = (0..d.num_render_targets.min(8) as usize)
        .map(|i| {
            let b = if d.blend.independent_blend != 0 {
                &d.blend.render_target[i]
            } else {
                &d.blend.render_target[0]
            };
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(b.blend_enable != 0)
                .src_color_blend_factor(vk::BlendFactor::from_raw(vk_blend(b.src_blend)))
                .dst_color_blend_factor(vk::BlendFactor::from_raw(vk_blend(b.dest_blend)))
                .color_blend_op(vk::BlendOp::from_raw(vk_blend_op(b.blend_op)))
                .src_alpha_blend_factor(vk::BlendFactor::from_raw(vk_blend(b.src_blend_alpha)))
                .dst_alpha_blend_factor(vk::BlendFactor::from_raw(vk_blend(b.dest_blend_alpha)))
                .alpha_blend_op(vk::BlendOp::from_raw(vk_blend_op(b.blend_op_alpha)))
                .color_write_mask(vk::ColorComponentFlags::from_raw(u32::from(
                    b.write_mask & 0xf,
                )))
        })
        .collect();
    s
}

fn root_of(rs: *mut c_void) -> Option<Arc<RootLayout>> {
    unsafe { inner::<RootSignatureInner>(rs) }.map(|r| Arc::clone(&r.layout))
}

/// Translates and registers a graphics PSO; the PSO object exists even when
/// translation fails (its draws are then skipped and logged), since titles
/// treat creation failure as fatal.
fn make_graphics_pso(
    dev: *mut c_void,
    d: &GraphicsPipelineStateDesc,
    as_bytes: Option<&[u8]>,
    ms: Option<&[u8]>,
) -> *mut Com<PipelineStateInner> {
    let key = next_pso_key(dev);
    if let Some(b) = &device_inner(dev).backend {
        let state = pso_state(d);
        if let Some(ms_raw) = ms {
            let ms_prog = program_of(ms_raw);
            let as_prog = as_bytes.and_then(program_of);
            let ps_prog = program_of(unsafe { d.ps.bytes() });
            if let Some(ms_w) = ms_prog {
                match crate::d3d12x::translate_mesh(
                    as_prog.as_deref(),
                    &ms_w,
                    ps_prog.as_deref(),
                    b.runner.quad_control,
                    Some(state),
                ) {
                    Ok(e) => {
                        b.pipelines.lock().unwrap().insert(key, e);
                    }
                    Err(e) => crate::crash::log_line(&format!("mesh PSO not translated: {e}")),
                }
            } else if let Ok((_, mesh_spv)) =
                strata_shader::dxil::compile_dxil_to_spirv(ms_raw, [32, 1, 1])
            {
                let task_spv = as_bytes
                    .and_then(|a| strata_shader::dxil::compile_dxil_to_spirv(a, [32, 1, 1]).ok())
                    .map(|(_, s)| s);
                let ps_spv =
                    strata_shader::dxil::compile_dxil_to_spirv(unsafe { d.ps.bytes() }, [1, 1, 1])
                        .map(|(_, s)| s)
                        .unwrap_or_default();
                b.pipelines.lock().unwrap().insert(
                    key,
                    crate::d3d12x::PipelineEntry {
                        spirv: ps_spv,
                        vertex: None,
                        hull: None,
                        domain: None,
                        geometry: None,
                        task: task_spv,
                        mesh: Some(mesh_spv),
                        images: Vec::new(),
                        gfx: Some(state),
                    },
                );
            }
        } else {
            let vs_raw = unsafe { d.vs.bytes() };
            let ps_raw = unsafe { d.ps.bytes() };
            let hs_raw = unsafe { d.hs.bytes() };
            let ds_raw = unsafe { d.ds.bytes() };
            let gs_raw = unsafe { d.gs.bytes() };
            let vs = program_of(vs_raw);
            let ps = program_of(ps_raw);
            let hs = program_of(hs_raw);
            let ds = program_of(ds_raw);
            let gs = program_of(gs_raw);
            match (vs, ps) {
                (Some(vs), Some(ps)) => match crate::d3d12x::translate_graphics_full(
                    &vs,
                    &ps,
                    hs.as_deref(),
                    ds.as_deref(),
                    gs.as_deref(),
                    b.runner.quad_control,
                    Some(state),
                ) {
                    Ok(e) => {
                        b.pipelines.lock().unwrap().insert(key, e);
                    }
                    Err(e) => crate::crash::log_line(&format!("graphics PSO not translated: {e}")),
                },
                _ => {
                    // DXIL-only PSO fallback (clean-room DXIL -> SPIR-V 1.6).
                    if let (Ok((_, vs_spv)), Ok((_, ps_spv))) = (
                        strata_shader::dxil::compile_dxil_to_spirv(vs_raw, [1, 1, 1]),
                        strata_shader::dxil::compile_dxil_to_spirv(ps_raw, [1, 1, 1]),
                    ) {
                        let hs_spv = strata_shader::dxil::compile_dxil_to_spirv(hs_raw, [1, 1, 1])
                            .ok()
                            .map(|(_, s)| s);
                        let ds_spv = strata_shader::dxil::compile_dxil_to_spirv(ds_raw, [1, 1, 1])
                            .ok()
                            .map(|(_, s)| s);
                        let gs_spv = strata_shader::dxil::compile_dxil_to_spirv(gs_raw, [1, 1, 1])
                            .ok()
                            .map(|(_, s)| s);
                        let mut st = state;
                        if hs_spv.is_some() && ds_spv.is_some() {
                            st.patch_control_points = 3;
                        }
                        b.pipelines.lock().unwrap().insert(
                            key,
                            crate::d3d12x::PipelineEntry {
                                spirv: ps_spv,
                                vertex: Some(vs_spv),
                                hull: hs_spv,
                                domain: ds_spv,
                                geometry: gs_spv,
                                task: None,
                                mesh: None,
                                images: Vec::new(),
                                gfx: Some(st),
                            },
                        );
                    } else {
                        log_once("graphics PSO has neither RDNA2 nor valid DXIL bytecode");
                    }
                }
            }
        }
    }
    Com::boxed(
        &PIPELINE_STATE_VTABLE,
        dev,
        PipelineStateInner {
            key,
            compute: false,
            root: root_of(d.root_signature),
        },
    )
}

fn workgroup_size_of(bytes: &[u8]) -> [u32; 3] {
    if bytes.starts_with(b"FSSM")
        && let Ok(p) = strata_shader::container::parse_fssm(bytes)
        && let Some(wg) = p.workgroup_size
    {
        return wg;
    }
    if bytes.starts_with(b"XBCH")
        && let Ok(fssm) = strata_shader::container::unpack_xbch(bytes)
        && let Ok(p) = strata_shader::container::parse_fssm(&fssm)
        && let Some(wg) = p.workgroup_size
    {
        return wg;
    }
    [64, 1, 1]
}

fn make_compute_pso(dev: *mut c_void, cs: &[u8], rs: *mut c_void) -> *mut Com<PipelineStateInner> {
    let key = next_pso_key(dev);
    if let Some(b) = &device_inner(dev).backend {
        let wg = workgroup_size_of(cs);
        match program_of(cs) {
            Some(words) => match crate::d3d12x::translate_compute(&words, wg) {
                Ok(e) => {
                    b.pipelines.lock().unwrap().insert(key, e);
                }
                Err(e) => crate::crash::log_line(&format!("compute PSO not translated: {e}")),
            },
            None => {
                // DXIL-only compute PSO fallback.
                if let Ok((_, spirv)) = strata_shader::dxil::compile_dxil_to_spirv(cs, wg) {
                    b.pipelines.lock().unwrap().insert(
                        key,
                        crate::d3d12x::PipelineEntry {
                            spirv,
                            vertex: None,
                            hull: None,
                            domain: None,
                            geometry: None,
                            task: None,
                            mesh: None,
                            images: Vec::new(),
                            gfx: None,
                        },
                    );
                } else {
                    log_once("compute PSO has neither RDNA2 nor valid DXIL bytecode");
                }
            }
        }
    }
    Com::boxed(
        &PIPELINE_STATE_VTABLE,
        dev,
        PipelineStateInner {
            key,
            compute: true,
            root: root_of(rs),
        },
    )
}

unsafe extern "system" fn dev_create_graphics_pso(
    this: *mut c_void,
    desc: *const GraphicsPipelineStateDesc,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if desc.is_null() {
        return E_INVALIDARG;
    }
    unsafe { hand_out(make_graphics_pso(this, &*desc, None, None), out) }
}

unsafe extern "system" fn dev_create_compute_pso(
    this: *mut c_void,
    desc: *const ComputePipelineStateDesc,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if desc.is_null() {
        return E_INVALIDARG;
    }
    let d = unsafe { &*desc };
    unsafe { hand_out(make_compute_pso(this, d.cs.bytes(), d.root_signature), out) }
}

unsafe extern "system" fn dev_create_command_list(
    this: *mut c_void,
    _node: u32,
    list_type: u32,
    _alloc: *mut c_void,
    pso: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    let l = Com::<CommandListInner>::new_boxed(list_type, this);
    if !pso.is_null() {
        unsafe { cl_set_pso(l.cast(), pso) };
    }
    unsafe { hand_out(l, out) }
}

unsafe extern "system" fn dev_create_command_list1(
    this: *mut c_void,
    _node: u32,
    list_type: u32,
    _flags: u32,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    let l = Com::<CommandListInner>::new_boxed(list_type, this);
    unsafe { (*l).inner.closed.store(true, Ordering::Release) };
    unsafe { hand_out(l, out) }
}

/// `CheckFeatureSupport`: what Strata's model provides (resource binding
/// tier 3, UMA, feature level 12_2, shader model 6.6, root signature 1.1).
unsafe extern "system" fn dev_check_feature_support(
    _this: *mut c_void,
    feature: u32,
    data: *mut u8,
    size: u32,
) -> i32 {
    if data.is_null() {
        return E_INVALIDARG;
    }
    let w32 = |o: usize, v: u32| unsafe {
        if o + 4 <= size as usize {
            (data.add(o) as *mut u32).write_unaligned(v)
        }
    };
    let r32 = |o: usize| unsafe {
        if o + 4 <= size as usize {
            (data.add(o) as *const u32).read_unaligned()
        } else {
            0
        }
    };
    match feature {
        0 => {
            // D3D12_OPTIONS: zero everything, then the tiers.
            unsafe { std::ptr::write_bytes(data, 0, size as usize) };
            w32(8, 3); // ResourceBindingTier 3
            w32(16, 3); // TiledResourcesTier 3
            w32(20, 3); // ResourceHeapTier... (CrossNode etc stay 0)
            w32(32, 2); // ResourceHeapTier 2
        }
        1 => {
            // ARCHITECTURE: NodeIndex, TileBasedRenderer, UMA, CacheCoherentUMA.
            w32(4, 0);
            w32(8, 1);
            w32(12, 1);
        }
        2 => {
            // FEATURE_LEVELS: NumFeatureLevels, pFeatureLevelsRequested, MaxSupported.
            let n = r32(0) as usize;
            let req = unsafe { (data.add(8) as *const *const u32).read_unaligned() };
            let mut best = 0;
            if !req.is_null() {
                for i in 0..n {
                    let l = unsafe { *req.add(i) };
                    if l <= 0xc200 && l > best {
                        best = l;
                    }
                }
            }
            w32(16, best);
        }
        3 => {
            // FORMAT_SUPPORT: Format, Support1, Support2.
            let known = dxgi_format(r32(0)).is_some();
            w32(4, if known { 0xffff_ffff } else { 0 });
            w32(8, if known { 0x7fff } else { 0 });
        }
        4 => {
            // MULTISAMPLE_QUALITY_LEVELS: Format, SampleCount, Flags, NumQualityLevels.
            w32(12, u32::from(r32(4) == 1));
        }
        5 => {
            // FORMAT_INFO: Format, PlaneCount (u8).
            let planes = match dxgi_format(r32(0)).and_then(|f| f.depth) {
                Some(VK_D24_UNORM_S8_UINT) | Some(VK_D32_SFLOAT_S8_UINT) => 2,
                _ => 1,
            };
            unsafe {
                if size >= 5 {
                    *data.add(4) = planes;
                }
            }
        }
        6 => {
            // GPU_VIRTUAL_ADDRESS_SUPPORT.
            w32(0, 40);
            w32(4, 47);
        }
        7 => {
            // SHADER_MODEL: HighestShaderModel (in/out).
            w32(0, r32(0).min(0x66));
        }
        12 => {
            // ROOT_SIGNATURE: HighestVersion (in/out).
            w32(0, r32(0).min(2));
        }
        _ => unsafe { std::ptr::write_bytes(data, 0, size as usize) },
    }
    S_OK
}

unsafe extern "system" fn dev_create_descriptor_heap(
    this: *mut c_void,
    desc: *const DescriptorHeapDesc,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if desc.is_null() {
        return E_INVALIDARG;
    }
    let d = unsafe { *desc };
    let size = u64::from(d.num_descriptors.max(1)) * u64::from(descriptor_increment(d.heap_type));
    let Some(mem) = alloc_block(this, size) else {
        return E_OUTOFMEMORY;
    };
    let inner = DescriptorHeapInner {
        desc: d,
        mem,
        _owned: Owned { dev: this, mem },
    };
    unsafe { hand_out(Com::boxed(&DESCRIPTOR_HEAP_VTABLE, this, inner), out) }
}

unsafe extern "system" fn dev_get_descriptor_handle_inc(_this: *mut c_void, heap_type: u32) -> u32 {
    descriptor_increment(heap_type)
}

unsafe extern "system" fn dev_create_root_signature(
    this: *mut c_void,
    _node: u32,
    blob: *const u8,
    len: usize,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if blob.is_null() || len == 0 {
        return E_INVALIDARG;
    }
    let bytes = unsafe { std::slice::from_raw_parts(blob, len) };
    let Some(layout) = RootLayout::parse(bytes) else {
        crate::crash::log_line("CreateRootSignature: blob is not a serialized root signature");
        return E_INVALIDARG;
    };
    if layout.spilled {
        log_once("root signature exceeds 32 user SGPRs: root arguments spill to memory");
    }
    let samplers = if layout.static_samplers.is_empty() {
        None
    } else {
        let b = alloc_block(this, 16 * layout.static_samplers.len() as u64);
        if let Some(b) = &b {
            for (i, s) in layout.static_samplers.iter().enumerate() {
                unsafe { ((b.host + 16 * i as u64) as *mut [u32; 4]).write_unaligned(*s) };
            }
        }
        b
    };
    unsafe {
        hand_out(
            Com::boxed(
                &ROOT_SIGNATURE_VTABLE,
                this,
                RootSignatureInner {
                    layout: Arc::new(layout),
                    samplers,
                    _owned: samplers.map(|mem| Owned { dev: this, mem }),
                },
            ),
            out,
        )
    }
}

/// Writes 8 descriptor dwords at a CPU handle.
fn write_descriptor(handle: u64, w: [u32; 8]) {
    if handle != 0 {
        unsafe { (handle as *mut [u32; 8]).write_unaligned(w) };
    }
}

/// A buffer view as heap dwords: the V#, and Strata's `'SB'` marker with the
/// view's byte length (for UAV clears).
fn buffer_view(base: u64, stride: u32, records: u32, format: u16, bytes: u64) -> [u32; 8] {
    let v = buffer_descriptor(base, stride, records, format);
    [
        v[0],
        v[1],
        v[2],
        v[3],
        0,
        BUFFER_VIEW_MAGIC,
        bytes as u32,
        (bytes >> 32) as u32,
    ]
}

unsafe extern "system" fn dev_create_cbv(
    _this: *mut c_void,
    desc: *const ConstantBufferViewDesc,
    handle: u64,
) {
    if desc.is_null() {
        return;
    }
    let d = unsafe { *desc };
    write_descriptor(
        handle,
        buffer_view(d.location, 0, d.size, 0, u64::from(d.size)),
    );
}

/// The image kind of an SRV/UAV/RTV dimension and its mip/slice selection.
fn view_kind(dim: u32, srv: bool) -> ImageKind {
    match (dim, srv) {
        (2, _) => ImageKind::Tex1D,
        (3, _) => ImageKind::Tex1DArray,
        (4, _) => ImageKind::Tex2D,
        (5, _) => ImageKind::Tex2DArray,
        (6, true) => ImageKind::Tex2DMsaa,
        (7, true) => ImageKind::Tex2DMsaaArray,
        (8, _) => ImageKind::Tex3D,
        (9, true) => ImageKind::Cube,
        (10, true) => ImageKind::Cube,
        _ => ImageKind::Tex2D,
    }
}

/// The default view of a whole resource (a null view desc).
fn default_kind(r: &ResourceInner) -> ImageKind {
    match r.desc.dimension {
        DIMENSION_TEXTURE1D if r.desc.depth_or_array_size > 1 => ImageKind::Tex1DArray,
        DIMENSION_TEXTURE1D => ImageKind::Tex1D,
        DIMENSION_TEXTURE3D => ImageKind::Tex3D,
        _ if r.desc.depth_or_array_size > 1 => ImageKind::Tex2DArray,
        _ => ImageKind::Tex2D,
    }
}

unsafe extern "system" fn dev_create_srv(
    _this: *mut c_void,
    res: *mut c_void,
    desc: *const ShaderResourceViewDesc,
    handle: u64,
) {
    let r = unsafe { resource(res) };
    let d = (!desc.is_null()).then(|| unsafe { *desc });
    match (r, d) {
        // Null resource: a null descriptor (reads return zero).
        (None, _) => write_descriptor(handle, [0; 8]),
        (Some(r), Some(d)) if d.dimension == 1 => {
            // BUFFER: FirstElement, NumElements, StructureByteStride, Flags (bit 0 RAW).
            let raw = d.u[4] & 1 != 0;
            let (stride, fmt) = if d.u[3] != 0 {
                (d.u[3], 0)
            } else if raw {
                (4, 0)
            } else {
                let f = dxgi_format(d.format);
                (f.map_or(4, |f| f.bytes), f.map_or(0, |f| f.gfx10))
            };
            let base = r.mem.gpu_va + d.first_element() * u64::from(stride);
            let records = if d.u[3] != 0 { d.u[2] } else { d.u[2] * stride };
            write_descriptor(
                handle,
                buffer_view(
                    base,
                    if d.u[3] != 0 { stride } else { 0 },
                    records,
                    fmt,
                    u64::from(d.u[2]) * u64::from(stride),
                ),
            );
        }
        (Some(r), Some(d)) if d.dimension == 11 => {
            // RAYTRACING_ACCELERATION_STRUCTURE: its address as a raw buffer.
            let va = d.first_element();
            write_descriptor(handle, buffer_view(va, 0, u32::MAX, 0, 0));
            let _ = r;
        }
        (Some(r), d) => {
            let (kind, mip, levels, first, slices, mapping) = match d {
                None => (default_kind(r), 0, u32::MAX, 0, 0, 0x1688),
                Some(d) => {
                    let u = d.u;
                    let (mip, levels, first, slices) = match d.dimension {
                        2 | 4 | 8 | 9 => (u[0], u[1], 0, 0),
                        3 | 5 => (u[0], u[1], u[2], u[3]),
                        6 => (0, 1, 0, 0),
                        7 => (0, 1, u[0], u[1]),
                        10 => (u[0], u[1], u[2], u[3] * 6),
                        _ => (0, u32::MAX, 0, 0),
                    };
                    (
                        view_kind(d.dimension, true),
                        mip,
                        levels,
                        first,
                        slices,
                        d.component_mapping,
                    )
                }
            };
            let fmt = d.map_or(0, |d| d.format);
            let levels = if levels == u32::MAX { 0 } else { levels };
            match image_descriptor(
                &r.view(),
                fmt,
                kind,
                mip,
                if levels == 0 { u32::MAX } else { levels },
                first,
                slices,
                mapping,
            ) {
                Some(w) => write_descriptor(handle, w),
                None => log_once("CreateShaderResourceView: unsupported format"),
            }
        }
    }
}

unsafe extern "system" fn dev_create_uav(
    _this: *mut c_void,
    res: *mut c_void,
    _counter: *mut c_void,
    desc: *const UnorderedAccessViewDesc,
    handle: u64,
) {
    let Some(r) = (unsafe { resource(res) }) else {
        write_descriptor(handle, [0; 8]);
        return;
    };
    let d = (!desc.is_null()).then(|| unsafe { *desc });
    if r.desc.dimension == DIMENSION_BUFFER || d.is_some_and(|d| d.dimension == 1) {
        let d = d.unwrap_or_default();
        // BUFFER: FirstElement (u64), NumElements, StructureByteStride,
        // CounterOffsetInBytes (u64), Flags (bit 0 RAW).
        let raw = d.u[6] & 1 != 0;
        let (stride, fmt) = if d.u[3] != 0 {
            (d.u[3], 0)
        } else if raw {
            (4, 0)
        } else {
            let f = dxgi_format(d.format);
            (f.map_or(4, |f| f.bytes), f.map_or(0, |f| f.gfx10))
        };
        let n = if d.u[2] == 0 {
            (r.desc.width / u64::from(stride)) as u32
        } else {
            d.u[2]
        };
        let base = r.mem.gpu_va + d.first_element() * u64::from(stride);
        let records = if d.u[3] != 0 { n } else { n * stride };
        write_descriptor(
            handle,
            buffer_view(
                base,
                if d.u[3] != 0 { stride } else { 0 },
                records,
                fmt,
                u64::from(n) * u64::from(stride),
            ),
        );
        return;
    }
    let (kind, mip, first, slices, fmt) = match d {
        None => (default_kind(r), 0, 0, 0, 0),
        Some(d) => {
            let u = d.u;
            let (mip, first, slices) = match d.dimension {
                2 | 4 => (u[0], 0, 0),
                3 | 5 => (u[0], u[1], u[2]),
                8 => (u[0], 0, 0),
                _ => (0, 0, 0),
            };
            (view_kind(d.dimension, false), mip, first, slices, d.format)
        }
    };
    match image_descriptor(&r.view(), fmt, kind, mip, 1, first, slices, 0x1688) {
        Some(w) => write_descriptor(handle, w),
        None => log_once("CreateUnorderedAccessView: unsupported format"),
    }
}

unsafe extern "system" fn dev_create_rtv(
    _this: *mut c_void,
    res: *mut c_void,
    desc: *const RenderTargetViewDesc,
    handle: u64,
) {
    let Some(r) = (unsafe { resource(res) }) else {
        write_descriptor(handle, [0; 8]);
        return;
    };
    let d = (!desc.is_null()).then(|| unsafe { *desc });
    let (mip, first, slices, fmt) = match d {
        None => (0, 0, 1, 0),
        Some(d) => {
            let u = d.u;
            match d.dimension {
                2 | 4 => (u[0], 0, 1, d.format),
                3 | 5 => (u[0], u[1], u[2].max(1), d.format),
                7 => (0, u[0], u[1].max(1), d.format),
                8 => (u[0], u[1], u[2].max(1), d.format),
                _ => (0, 0, 1, d.format),
            }
        }
    };
    // A render target is one subresource (or a slice range) at one level:
    // described as a standalone 2D surface / 2D array there.
    let kind = if slices > 1 {
        ImageKind::Tex2DArray
    } else {
        ImageKind::Tex2D
    };
    match image_descriptor(&r.view(), fmt, kind, mip, 1, first, slices, 0x1688) {
        Some(w) => write_descriptor(handle, w),
        None => log_once("CreateRenderTargetView: unsupported format"),
    }
}

unsafe extern "system" fn dev_create_dsv(
    _this: *mut c_void,
    res: *mut c_void,
    desc: *const DepthStencilViewDesc,
    handle: u64,
) {
    let Some(r) = (unsafe { resource(res) }) else {
        write_descriptor(handle, [0; 8]);
        return;
    };
    let d = (!desc.is_null()).then(|| unsafe { *desc });
    let (mip, first, slices, fmt) = match d {
        None => (0, 0, 1, r.desc.format),
        Some(d) => {
            let u = d.u;
            let f = if d.format == 0 {
                r.desc.format
            } else {
                d.format
            };
            match d.dimension {
                1 | 3 => (u[0], 0, 1, f),
                2 | 4 => (u[0], u[1], u[2].max(1), f),
                6 => (0, u[0], u[1].max(1), f),
                _ => (0, 0, 1, f),
            }
        }
    };
    let kind = if slices > 1 {
        ImageKind::Tex2DArray
    } else {
        ImageKind::Tex2D
    };
    match image_descriptor(&r.view(), fmt, kind, mip, 1, first, slices, 0x1688) {
        Some(w) => write_descriptor(handle, w),
        None => log_once("CreateDepthStencilView: unsupported format"),
    }
}

unsafe extern "system" fn dev_create_sampler(
    _this: *mut c_void,
    desc: *const SamplerDesc,
    handle: u64,
) {
    if desc.is_null() || handle == 0 {
        return;
    }
    let s = sampler_descriptor(unsafe { &*desc });
    unsafe { (handle as *mut [u32; 4]).write_unaligned(s) };
}

unsafe extern "system" fn dev_copy_descriptors(
    _this: *mut c_void,
    n_dst: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    n_src: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    heap_type: u32,
) {
    let inc = u64::from(descriptor_increment(heap_type));
    let range = |starts: *const u64, sizes: *const u32, n: u32| -> Vec<u64> {
        let mut v = Vec::new();
        for i in 0..n as usize {
            let start = unsafe { *starts.add(i) };
            let len = if sizes.is_null() {
                1
            } else {
                unsafe { *sizes.add(i) }
            };
            for k in 0..u64::from(len) {
                v.push(start + k * inc);
            }
        }
        v
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let dst = range(dst_starts, dst_sizes, n_dst);
    let src = range(src_starts, src_sizes, n_src);
    for (d, s) in dst.iter().zip(&src) {
        unsafe { std::ptr::copy(*s as *const u8, *d as *mut u8, inc as usize) };
    }
}

unsafe extern "system" fn dev_copy_descriptors_simple(
    _this: *mut c_void,
    n: u32,
    dst: u64,
    src: u64,
    heap_type: u32,
) {
    if dst == 0 || src == 0 {
        return;
    }
    let bytes = u64::from(n) * u64::from(descriptor_increment(heap_type));
    unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, bytes as usize) };
}

unsafe extern "system" fn dev_get_resource_allocation_info(
    _this: *mut c_void,
    out: *mut ResourceAllocationInfo,
    _mask: u32,
    n: u32,
    descs: *const ResourceDesc,
) -> *mut ResourceAllocationInfo {
    let mut size = 0u64;
    if !descs.is_null() {
        for i in 0..n as usize {
            size = size.next_multiple_of(65536) + resource_size(unsafe { &*descs.add(i) });
        }
    }
    if !out.is_null() {
        unsafe {
            *out = ResourceAllocationInfo {
                size,
                alignment: 65536,
            }
        };
    }
    out
}

unsafe extern "system" fn dev_get_resource_allocation_info1(
    this: *mut c_void,
    out: *mut ResourceAllocationInfo,
    mask: u32,
    n: u32,
    descs: *const ResourceDesc,
    info1: *mut [u64; 3],
) -> *mut ResourceAllocationInfo {
    if !info1.is_null() && !descs.is_null() {
        let mut off = 0u64;
        for i in 0..n as usize {
            let s = resource_size(unsafe { &*descs.add(i) });
            off = off.next_multiple_of(65536);
            unsafe { *info1.add(i) = [off, 65536, s] };
            off += s;
        }
    }
    unsafe { dev_get_resource_allocation_info(this, out, mask, n, descs) }
}

unsafe extern "system" fn dev_get_custom_heap_properties(
    _this: *mut c_void,
    out: *mut HeapProperties,
    _node: u32,
    heap_type: u32,
) -> *mut HeapProperties {
    if !out.is_null() {
        // UMA: every heap is write-back CPU memory in the L0 pool.
        let cpu_page = match heap_type {
            2 => 3, // UPLOAD: WRITE_COMBINE
            3 => 3, // READBACK: WRITE_BACK in D3D terms; one pool either way
            _ => 1, // DEFAULT: NOT_AVAILABLE
        };
        unsafe {
            *out = HeapProperties {
                heap_type: 4, // CUSTOM
                cpu_page_property: cpu_page,
                memory_pool_preference: 1, // L0
                creation_node_mask: 1,
                visible_node_mask: 1,
            }
        };
    }
    out
}

unsafe extern "system" fn dev_create_committed_resource(
    this: *mut c_void,
    heap_props: *const HeapProperties,
    _heap_flags: u32,
    desc: *const ResourceDesc,
    _state: u32,
    _clear: *const c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if desc.is_null() {
        return E_INVALIDARG;
    }
    let hp = if heap_props.is_null() {
        HeapProperties::default()
    } else {
        unsafe { *heap_props }
    };
    let r = Com::<ResourceInner>::new_boxed(unsafe { *desc }, hp, this);
    if r.is_null() {
        return E_OUTOFMEMORY;
    }
    unsafe { hand_out(r, out) }
}

unsafe extern "system" fn dev_create_committed_resource1(
    this: *mut c_void,
    heap_props: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    _session: *mut c_void,
    riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    unsafe {
        dev_create_committed_resource(this, heap_props, heap_flags, desc, state, clear, riid, out)
    }
}

unsafe extern "system" fn dev_create_heap(
    this: *mut c_void,
    desc: *const HeapDesc,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if desc.is_null() {
        return E_INVALIDARG;
    }
    let d = unsafe { *desc };
    let Some(mem) = alloc_block(this, d.size) else {
        return E_OUTOFMEMORY;
    };
    let inner = HeapInner {
        desc: d,
        mem,
        _owned: Owned { dev: this, mem },
    };
    unsafe { hand_out(Com::boxed(&HEAP_VTABLE, this, inner), out) }
}

unsafe extern "system" fn dev_create_heap1(
    this: *mut c_void,
    desc: *const HeapDesc,
    _session: *mut c_void,
    riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    unsafe { dev_create_heap(this, desc, riid, out) }
}

unsafe extern "system" fn dev_create_placed_resource(
    this: *mut c_void,
    heap: *mut c_void,
    offset: u64,
    desc: *const ResourceDesc,
    _state: u32,
    _clear: *const c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    let (Some(h), false) = (unsafe { inner::<HeapInner>(heap) }, desc.is_null()) else {
        return E_INVALIDARG;
    };
    let d = unsafe { *desc };
    let size = resource_size(&d);
    if offset + size > h.mem.size {
        return E_INVALIDARG;
    }
    let mem = Block {
        console_va: h.mem.console_va + offset,
        host: h.mem.host + offset,
        gpu_va: h.mem.gpu_va + offset,
        size,
        owned: false,
        uma: h.mem.uma,
    };
    unsafe { add_ref(heap) };
    let r = Com::<ResourceInner>::with_memory(d, h.desc.properties, this, mem, heap);
    unsafe { hand_out(r, out) }
}

unsafe extern "system" fn dev_create_reserved_resource(
    this: *mut c_void,
    desc: *const ResourceDesc,
    _state: u32,
    _clear: *const c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if desc.is_null() {
        return E_INVALIDARG;
    }
    // Reserved (tiled) resources are backed whole: tile mappings are no-ops.
    let r = Com::<ResourceInner>::new_boxed(unsafe { *desc }, HeapProperties::default(), this);
    if r.is_null() {
        return E_OUTOFMEMORY;
    }
    unsafe { hand_out(r, out) }
}

unsafe extern "system" fn dev_create_reserved_resource1(
    this: *mut c_void,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    _session: *mut c_void,
    riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    unsafe { dev_create_reserved_resource(this, desc, state, clear, riid, out) }
}

unsafe extern "system" fn dev_create_fence(
    this: *mut c_void,
    initial: u64,
    _flags: u32,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    unsafe { hand_out(Com::<FenceInner>::new_boxed_with_device(initial, this), out) }
}

unsafe extern "system" fn dev_get_copyable_footprints(
    _this: *mut c_void,
    desc: *const ResourceDesc,
    first: u32,
    n: u32,
    base: u64,
    layouts: *mut PlacedSubresourceFootprint,
    rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    if desc.is_null() {
        return;
    }
    let mk = |p: *mut PlacedSubresourceFootprint| {
        (!p.is_null()).then(|| unsafe { std::slice::from_raw_parts_mut(p, n as usize) })
    };
    let rows =
        (!rows.is_null()).then(|| unsafe { std::slice::from_raw_parts_mut(rows, n as usize) });
    let row_sizes = (!row_sizes.is_null())
        .then(|| unsafe { std::slice::from_raw_parts_mut(row_sizes, n as usize) });
    let t = copyable_footprints(
        unsafe { &*desc },
        first,
        n,
        base,
        mk(layouts),
        rows,
        row_sizes,
    );
    if !total.is_null() {
        unsafe { *total = t };
    }
}

unsafe extern "system" fn dev_create_query_heap(
    this: *mut c_void,
    desc: *const QueryHeapDesc,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if desc.is_null() {
        return E_INVALIDARG;
    }
    let d = unsafe { *desc };
    // Results are u64s; pipeline statistics are 11 of them, SO statistics 2.
    let stride = match d.query_type {
        3 => 88,
        4..=7 => 16,
        _ => 8,
    };
    let Some(mem) = alloc_block(this, stride * u64::from(d.count.max(1))) else {
        return E_OUTOFMEMORY;
    };
    let inner = QueryHeapInner {
        desc: d,
        mem,
        stride,
        _owned: Owned { dev: this, mem },
    };
    unsafe { hand_out(Com::boxed(&QUERY_HEAP_VTABLE, this, inner), out) }
}

unsafe extern "system" fn dev_create_command_signature(
    this: *mut c_void,
    desc: *const CommandSignatureDesc,
    root: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if desc.is_null() {
        return E_INVALIDARG;
    }
    let d = unsafe { *desc };
    let layout = root_of(root);
    let sgpr_of = |idx: u32| -> u16 {
        layout
            .as_ref()
            .and_then(|l| l.params.get(idx as usize))
            .map_or(0, |p| u16::from(p.sgpr()))
    };
    let mut args = Vec::new();
    for i in 0..d.num_argument_descs as usize {
        let a = unsafe { *d.argument_descs.add(i) };
        args.push(match a.arg_type {
            0 => IndirectArg::Draw,
            1 => IndirectArg::DrawIndexed,
            2 => IndirectArg::Dispatch,
            3 => IndirectArg::VertexBuffer { slot: a.u[0] },
            4 => IndirectArg::IndexBuffer,
            5 => IndirectArg::Constants {
                sgpr: sgpr_of(a.u[0]) + a.u[1] as u16,
                count: a.u[2] as u16,
            },
            6..=8 => IndirectArg::RootAddress {
                sgpr: sgpr_of(a.u[0]),
            },
            9 => IndirectArg::DispatchRays,
            10 => IndirectArg::DispatchMesh,
            _ => {
                return E_INVALIDARG;
            }
        });
    }
    unsafe {
        hand_out(
            Com::boxed(
                &COMMAND_SIGNATURE_VTABLE,
                this,
                CommandSignatureInner {
                    layout: Arc::new(IndirectLayout {
                        stride: d.byte_stride,
                        args,
                    }),
                },
            ),
            out,
        )
    }
}

unsafe extern "system" fn dev_get_resource_tiling(
    _this: *mut c_void,
    _res: *mut c_void,
    num_tiles: *mut u32,
    packed: *mut [u32; 5],
    shape: *mut [u32; 3],
    num_sub: *mut u32,
    _first: u32,
    _tilings: *mut c_void,
) {
    // Strata backs reserved resources whole; report one 64 KiB-tile shape.
    unsafe {
        if !num_tiles.is_null() {
            *num_tiles = 0;
        }
        if !packed.is_null() {
            *packed = [0; 5];
        }
        if !shape.is_null() {
            *shape = [128, 128, 1];
        }
        if !num_sub.is_null() {
            *num_sub = 0;
        }
    }
}

unsafe extern "system" fn dev_get_adapter_luid(
    _this: *mut c_void,
    out: *mut [u32; 2],
) -> *mut [u32; 2] {
    if !out.is_null() {
        unsafe { *out = [0x5354_5241, 0] }; // 'STRA'
    }
    out
}

// ============================================================================
// ID3D12PipelineLibrary (backed by strata_gpu::indirect_upload::PipelineLibrary)
// ============================================================================

struct PipelineLibraryInner {
    lib: Mutex<strata_gpu::indirect_upload::PipelineLibrary>,
}

unsafe fn wide_to_string(p: *const u16) -> String {
    if p.is_null() {
        return String::new();
    }
    let mut len = 0usize;
    while unsafe { *p.add(len) } != 0 && len < 1024 {
        len += 1;
    }
    String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(p, len) })
}

unsafe extern "system" fn plib_store_pipeline(
    this: *mut c_void,
    name: *const u16,
    pso: *mut c_void,
) -> i32 {
    if name.is_null() || pso.is_null() {
        return E_INVALIDARG;
    }
    let key = unsafe { wide_to_string(name) };
    let blob = (pso as usize).to_le_bytes().to_vec();
    unsafe { &Com::<PipelineLibraryInner>::from(this).inner }
        .lib
        .lock()
        .unwrap()
        .store(key, blob);
    S_OK
}

unsafe extern "system" fn plib_load_graphics(
    this: *mut c_void,
    name: *const u16,
    desc: *const c_void,
    riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if name.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    let key = unsafe { wide_to_string(name) };
    let found = unsafe { &Com::<PipelineLibraryInner>::from(this).inner }
        .lib
        .lock()
        .unwrap()
        .load(&key)
        .is_some();
    if !found {
        return E_INVALIDARG;
    }
    let dev = unsafe { Com::<PipelineLibraryInner>::from(this).base.device };
    unsafe { dev_create_graphics_pso(dev, desc.cast(), riid, out) }
}

unsafe extern "system" fn plib_load_compute(
    this: *mut c_void,
    name: *const u16,
    desc: *const c_void,
    riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if name.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    let key = unsafe { wide_to_string(name) };
    let found = unsafe { &Com::<PipelineLibraryInner>::from(this).inner }
        .lib
        .lock()
        .unwrap()
        .load(&key)
        .is_some();
    if !found {
        return E_INVALIDARG;
    }
    let dev = unsafe { Com::<PipelineLibraryInner>::from(this).base.device };
    unsafe { dev_create_compute_pso(dev, desc.cast(), riid, out) }
}

unsafe extern "system" fn plib_get_serialized_size(this: *mut c_void) -> usize {
    unsafe { &Com::<PipelineLibraryInner>::from(this).inner }
        .lib
        .lock()
        .unwrap()
        .serialize()
        .len()
}

unsafe extern "system" fn plib_serialize(
    this: *mut c_void,
    data: *mut c_void,
    size_in_bytes: usize,
) -> i32 {
    if data.is_null() {
        return E_INVALIDARG;
    }
    let bytes = unsafe { &Com::<PipelineLibraryInner>::from(this).inner }
        .lib
        .lock()
        .unwrap()
        .serialize();
    if size_in_bytes < bytes.len() {
        return E_INVALIDARG;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.cast::<u8>(), bytes.len());
    }
    S_OK
}

static PIPELINE_LIBRARY_VTABLE: [VTableSlot; 14] = {
    let mut vt = base_slots::<14>();
    vt[8] = slot!(plib_store_pipeline);
    vt[9] = slot!(plib_load_graphics);
    vt[10] = slot!(plib_load_compute);
    vt[11] = slot!(plib_get_serialized_size);
    vt[12] = slot!(plib_serialize);
    vt[13] = slot!(plib_load_compute); // LoadPipeline (ID3D12PipelineLibrary1)
    vt
};

unsafe extern "system" fn dev_create_pipeline_library(
    this: *mut c_void,
    blob: *const c_void,
    blob_len: usize,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let lib = if !blob.is_null() && blob_len > 0 {
        let slice = unsafe { std::slice::from_raw_parts(blob.cast::<u8>(), blob_len) };
        strata_gpu::indirect_upload::PipelineLibrary::deserialize(slice).unwrap_or_default()
    } else {
        strata_gpu::indirect_upload::PipelineLibrary::default()
    };
    let obj = Com::boxed(
        &PIPELINE_LIBRARY_VTABLE,
        this,
        PipelineLibraryInner {
            lib: Mutex::new(lib),
        },
    );
    unsafe { *out = obj.cast() };
    S_OK
}

unsafe extern "system" fn dev_unsupported(_this: *mut c_void) -> i32 {
    DXGI_ERROR_UNSUPPORTED
}

unsafe extern "system" fn dev_set_event_on_multiple_fence_completion(
    _this: *mut c_void,
    fences: *const *mut c_void,
    values: *const u64,
    n: u32,
    flags: u32,
    event: *mut c_void,
) -> i32 {
    if fences.is_null() || values.is_null() {
        return E_INVALIDARG;
    }
    let pairs: Vec<(usize, u64)> = (0..n as usize)
        .map(|i| unsafe { (*fences.add(i) as usize, *values.add(i)) })
        .collect();
    let any = flags & 1 != 0;
    let done = move || {
        let mut it = pairs.iter().map(|&(f, v)| {
            unsafe { inner::<FenceInner>(f as *mut c_void) }
                .is_some_and(|f| f.value.load(Ordering::Acquire) >= v)
        });
        if any { it.any(|b| b) } else { it.all(|b| b) }
    };
    if done() || event.is_null() {
        if !event.is_null() {
            unsafe { SetEvent(event) };
        }
        return S_OK;
    }
    let ev = event as usize;
    std::thread::spawn(move || {
        while !done() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        unsafe { SetEvent(ev as *mut c_void) };
    });
    S_OK
}

unsafe extern "system" fn dev_create_pipeline_state(
    this: *mut c_void,
    desc: *const [u64; 2],
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if desc.is_null() {
        return E_INVALIDARG;
    }
    let [size, stream] = unsafe { *desc };
    let bytes = unsafe { std::slice::from_raw_parts(stream as *const u8, size as usize) };
    match parse_pso_stream(bytes) {
        Some(PsoStream::Compute { cs, root }) => unsafe {
            hand_out(make_compute_pso(this, &cs, root as *mut c_void), out)
        },
        Some(PsoStream::Graphics(d, as_b, ms)) => unsafe {
            hand_out(
                make_graphics_pso(this, &d, as_b.as_deref(), ms.as_deref()),
                out,
            )
        },
        None => E_INVALIDARG,
    }
}

enum PsoStream {
    Compute {
        cs: Vec<u8>,
        root: usize,
    },
    Graphics(
        Box<GraphicsPipelineStateDesc>,
        Option<Vec<u8>>,
        Option<Vec<u8>>,
    ),
}

/// A `D3D12_PIPELINE_STATE_STREAM_DESC` stream: pointer-aligned subobjects,
/// each a type dword then its desc (at offset 8 when the desc holds
/// pointers, else 4).
fn parse_pso_stream(b: &[u8]) -> Option<PsoStream> {
    let mut d: GraphicsPipelineStateDesc = unsafe { std::mem::zeroed() };
    d.sample_mask = u32::MAX;
    d.rasterizer.depth_clip_enable = 1;
    d.rasterizer.fill_mode = 3;
    d.rasterizer.cull_mode = 3;
    d.depth_stencil.depth_enable = 1;
    d.depth_stencil.depth_write_mask = 1;
    d.depth_stencil.depth_func = 2;
    d.primitive_topology_type = 3;
    for rt in &mut d.blend.render_target {
        rt.write_mask = 0xf;
        rt.src_blend = 2;
        rt.dest_blend = 1;
        rt.blend_op = 1;
        rt.src_blend_alpha = 2;
        rt.dest_blend_alpha = 1;
        rt.blend_op_alpha = 1;
    }
    let mut cs: Option<Vec<u8>> = None;
    let mut as_b: Option<Vec<u8>> = None;
    let mut ms: Option<Vec<u8>> = None;
    let mut o = 0usize;
    let rd = |o: usize, n: usize| b.get(o..o + n);
    let copy_into = |dst: *mut u8, src: &[u8]| unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len())
    };
    while o + 4 <= b.len() {
        let ty = u32::from_le_bytes(rd(o, 4)?.try_into().ok()?);
        let (align, size) = match ty {
            0 => (8, 8),
            1..=6 | 24 | 25 => (8, 16),
            7 => (8, 32),
            8 => (4, 328),
            9 | 13 | 14 | 16 | 18 | 20 => (4, 4),
            10 => (4, 44),
            11 => (4, 52),
            12 | 19 => (8, 16),
            15 => (4, 36),
            17 => (4, 8),
            21 => (4, 56),
            22 => (8, 24),
            _ => return None,
        };
        let at = o + align;
        let v = rd(at, size)?;
        let bytecode = |v: &[u8]| -> Vec<u8> {
            let p = u64::from_le_bytes(v[0..8].try_into().unwrap()) as *const u8;
            let n = u64::from_le_bytes(v[8..16].try_into().unwrap()) as usize;
            if p.is_null() || n == 0 {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
            }
        };
        match ty {
            0 => d.root_signature = u64::from_le_bytes(v.try_into().ok()?) as *mut c_void,
            1 => copy_into(&mut d.vs as *mut ShaderBytecode as *mut u8, v),
            2 => copy_into(&mut d.ps as *mut ShaderBytecode as *mut u8, v),
            3 => copy_into(&mut d.ds as *mut ShaderBytecode as *mut u8, v),
            4 => copy_into(&mut d.hs as *mut ShaderBytecode as *mut u8, v),
            5 => copy_into(&mut d.gs as *mut ShaderBytecode as *mut u8, v),
            6 => cs = Some(bytecode(v)),
            24 => as_b = Some(bytecode(v)),
            25 => ms = Some(bytecode(v)),
            8 => copy_into(&mut d.blend as *mut BlendDesc as *mut u8, v),
            9 => d.sample_mask = u32::from_le_bytes(v.try_into().ok()?),
            10 => copy_into(&mut d.rasterizer as *mut RasterizerDesc as *mut u8, v),
            11 | 21 => copy_into(
                &mut d.depth_stencil as *mut DepthStencilDesc as *mut u8,
                &v[..52],
            ),
            14 => d.primitive_topology_type = u32::from_le_bytes(v.try_into().ok()?),
            15 => {
                for i in 0..8 {
                    d.rtv_formats[i] = u32::from_le_bytes(v[4 * i..4 * i + 4].try_into().ok()?);
                }
                d.num_render_targets = u32::from_le_bytes(v[32..36].try_into().ok()?);
            }
            16 => d.dsv_format = u32::from_le_bytes(v.try_into().ok()?),
            _ => {}
        }
        o = (at + size).next_multiple_of(8);
    }
    Some(match cs {
        Some(cs) => PsoStream::Compute {
            cs,
            root: d.root_signature as usize,
        },
        None => PsoStream::Graphics(Box::new(d), as_b, ms),
    })
}

unsafe extern "system" fn dev_enqueue_make_resident(
    _this: *mut c_void,
    _flags: u32,
    _n: u32,
    _objs: *const *mut c_void,
    fence: *mut c_void,
    value: u64,
) -> i32 {
    // Everything is always resident.
    if let Some(f) = unsafe { inner::<FenceInner>(fence) } {
        f.signal(value);
    }
    S_OK
}

unsafe extern "system" fn dev_enumerate_meta_commands(
    _this: *mut c_void,
    n: *mut u32,
    _descs: *mut c_void,
) -> i32 {
    if !n.is_null() {
        unsafe { *n = 0 };
    }
    S_OK
}

/// `ID3D12StateObject` / `ID3D12StateObjectProperties` for DXR ray tracing pipelines.
///
/// Under Strata's No-API approach, DXR libraries (RDNA2 or DXIL) are compiled
/// into a pipeline entry keyed by `key`, and `GetShaderIdentifier` hands out
/// stable 32-byte shader identifiers (containing the pipeline key + export hash)
/// that titles write into their SBT buffers in title memory.
pub struct StateObjectInner {
    pub key: PipelineKey,
    pub stack_size: std::sync::atomic::AtomicU64,
    pub shader_ids: Mutex<std::collections::HashMap<String, Box<[u8; 32]>>>,
}

unsafe extern "system" fn so_get_shader_identifier(
    this: *mut c_void,
    export_name: *const u16,
) -> *mut c_void {
    let Some(so) = (unsafe { inner::<StateObjectInner>(this) }) else {
        return std::ptr::null_mut();
    };
    let name = if export_name.is_null() {
        String::new()
    } else {
        let mut len = 0usize;
        unsafe {
            while *export_name.add(len) != 0 && len < 256 {
                len += 1;
            }
            String::from_utf16_lossy(std::slice::from_raw_parts(export_name, len))
        }
    };
    let mut map = so.shader_ids.lock().unwrap();
    let entry = map.entry(name.clone()).or_insert_with(|| {
        let mut id = [0u8; 32];
        id[0..8].copy_from_slice(&so.key.0.to_le_bytes());
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in name.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        id[8..16].copy_from_slice(&h.to_le_bytes());
        Box::new(id)
    });
    entry.as_mut_ptr() as *mut c_void
}

unsafe extern "system" fn so_get_shader_stack_size(
    _this: *mut c_void,
    _export_name: *const u16,
) -> u64 {
    256
}

unsafe extern "system" fn so_get_pipeline_stack_size(this: *mut c_void) -> u64 {
    unsafe { inner::<StateObjectInner>(this) }
        .map_or(4096, |s| s.stack_size.load(Ordering::Relaxed))
}

unsafe extern "system" fn so_set_pipeline_stack_size(this: *mut c_void, size: u64) {
    if let Some(s) = unsafe { inner::<StateObjectInner>(this) } {
        s.stack_size.store(size, Ordering::Relaxed);
    }
}

static STATE_OBJECT_VTABLE: [VTableSlot; 12] = {
    let mut vt = base_slots::<12>();
    // ID3D12StateObjectProperties methods sit at slots 3..7 on QI, and we also
    // expose them at slots 8..12 so both ID3D12StateObject and
    // ID3D12StateObjectProperties work without a separate wrapper allocation.
    vt[8] = slot!(so_get_shader_identifier);
    vt[9] = slot!(so_get_shader_stack_size);
    vt[10] = slot!(so_get_pipeline_stack_size);
    vt[11] = slot!(so_set_pipeline_stack_size);
    vt
};

/// `D3D12_STATE_OBJECT_DESC` -> `ID3D12StateObject`.
unsafe extern "system" fn dev_create_state_object(
    this: *mut c_void,
    desc: *const c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if desc.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    // D3D12_STATE_OBJECT_DESC: Type (u32), NumSubobjects (u32), pSubobjects (*const [u64; 2])
    let hdr = unsafe { *(desc as *const [u64; 2]) };
    let num_subobjects = (hdr[0] >> 32) as usize;
    let subobjects_ptr = hdr[1] as *const [u64; 2];
    let key = next_pso_key(this);
    if let Some(b) = &device_inner(this).backend {
        let mut compiled = false;
        if !subobjects_ptr.is_null() {
            for i in 0..num_subobjects {
                let sub = unsafe { *subobjects_ptr.add(i) };
                let sub_ty = (sub[0] & 0xffff_ffff) as u32;
                let sub_desc = sub[1] as *const u8;
                // D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY = 5: first 16 bytes are ShaderBytecode
                if sub_ty == 5 && !sub_desc.is_null() {
                    let bc = unsafe { *(sub_desc as *const ShaderBytecode) };
                    let bytes = unsafe { bc.bytes() };
                    if let Some(words) = program_of(bytes) {
                        if let Ok(entry) = crate::d3d12x::translate_compute(&words, [8, 8, 1]) {
                            b.pipelines.lock().unwrap().insert(key, entry);
                            compiled = true;
                            break;
                        }
                    } else if let Ok((_, spirv)) =
                        strata_shader::dxil::compile_dxil_to_spirv(bytes, [8, 8, 1])
                    {
                        b.pipelines.lock().unwrap().insert(
                            key,
                            crate::d3d12x::PipelineEntry {
                                spirv,
                                vertex: None,
                                hull: None,
                                domain: None,
                                geometry: None,
                                task: None,
                                mesh: None,
                                images: Vec::new(),
                                gfx: None,
                            },
                        );
                        compiled = true;
                        break;
                    }
                }
            }
        }
        if !compiled {
            // Valid empty compute/raygen fallback so DispatchRays can still execute cleanly.
            let (_, spirv) = strata_shader::dxil::compile_dxil_to_spirv(&[], [8, 8, 1])
                .unwrap_or_else(|_| {
                    let mut w = vec![0x0723_0203, 0x0001_0600, 0, 6, 0];
                    w.extend_from_slice(&[(2 << 16) | 17, 1]);
                    w.extend_from_slice(&[(3 << 16) | 14, 0, 1]);
                    w.extend_from_slice(&[(5 << 16) | 15, 5, 1, u32::from_le_bytes(*b"main"), 0]);
                    w.extend_from_slice(&[(6 << 16) | 16, 1, 17, 8, 8, 1]);
                    w.extend_from_slice(&[(2 << 16) | 19, 2]);
                    w.extend_from_slice(&[(3 << 16) | 33, 3, 2]);
                    w.extend_from_slice(&[(5 << 16) | 54, 2, 1, 0, 3]);
                    w.extend_from_slice(&[(2 << 16) | 248, 4]);
                    w.extend_from_slice(&[(1 << 16) | 253]);
                    w.extend_from_slice(&[(1 << 16) | 56]);
                    (strata_shader::dxil::DxilStage::Compute, w)
                });
            b.pipelines.lock().unwrap().insert(
                key,
                crate::d3d12x::PipelineEntry {
                    spirv,
                    vertex: None,
                    hull: None,
                    domain: None,
                    geometry: None,
                    task: None,
                    mesh: None,
                    images: Vec::new(),
                    gfx: None,
                },
            );
        }
    }
    let obj = Com::boxed(
        &STATE_OBJECT_VTABLE,
        this,
        StateObjectInner {
            key,
            stack_size: std::sync::atomic::AtomicU64::new(4096),
            shader_ids: Mutex::new(std::collections::HashMap::new()),
        },
    );
    unsafe { hand_out(obj, out) }
}

// ============================================================================
// Shared handles (in-process registry for CreateSharedHandle / OpenSharedHandle*)
// ============================================================================

static SHARED_HANDLES: Mutex<Option<std::collections::HashMap<usize, (usize, Option<String>)>>> =
    Mutex::new(None);
static NEXT_SHARED_HANDLE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0x57A0_0001);

unsafe extern "system" fn dev_create_shared_handle(
    _this: *mut c_void,
    object: *mut c_void,
    _attributes: *const c_void,
    _access: u32,
    name: *const u16,
    out_handle: *mut *mut c_void,
) -> i32 {
    if object.is_null() || out_handle.is_null() {
        return E_INVALIDARG;
    }
    let name_str = if name.is_null() {
        None
    } else {
        let mut len = 0usize;
        unsafe {
            while *name.add(len) != 0 && len < 260 {
                len += 1;
            }
            Some(String::from_utf16_lossy(std::slice::from_raw_parts(
                name, len,
            )))
        }
    };
    // AddRef the shared object so it stays alive while referenced by the handle.
    unsafe {
        let vt = *(object as *const *const VTableSlot);
        let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 =
            std::mem::transmute((*vt.add(1)).0);
        add_ref(object);
    }
    let h = NEXT_SHARED_HANDLE.fetch_add(1, Ordering::Relaxed);
    let mut g = SHARED_HANDLES.lock().unwrap();
    g.get_or_insert_with(std::collections::HashMap::new)
        .insert(h, (object as usize, name_str));
    unsafe { *out_handle = h as *mut c_void };
    S_OK
}

unsafe extern "system" fn dev_open_shared_handle(
    _this: *mut c_void,
    handle: *mut c_void,
    _riid: *const c_void,
    out_obj: *mut *mut c_void,
) -> i32 {
    if out_obj.is_null() {
        return E_INVALIDARG;
    }
    let g = SHARED_HANDLES.lock().unwrap();
    let Some(&(obj_addr, _)) = g.as_ref().and_then(|m| m.get(&(handle as usize))) else {
        return E_INVALIDARG;
    };
    let obj = obj_addr as *mut c_void;
    unsafe {
        let vt = *(obj as *const *const VTableSlot);
        let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 =
            std::mem::transmute((*vt.add(1)).0);
        add_ref(obj);
        *out_obj = obj;
    }
    S_OK
}

unsafe extern "system" fn dev_open_shared_handle_by_name(
    _this: *mut c_void,
    name: *const u16,
    _access: u32,
    out_handle: *mut *mut c_void,
) -> i32 {
    if name.is_null() || out_handle.is_null() {
        return E_INVALIDARG;
    }
    let mut len = 0usize;
    let needle = unsafe {
        while *name.add(len) != 0 && len < 260 {
            len += 1;
        }
        String::from_utf16_lossy(std::slice::from_raw_parts(name, len))
    };
    let g = SHARED_HANDLES.lock().unwrap();
    if let Some((&h, _)) = g.as_ref().and_then(|m| {
        m.iter()
            .find(|(_, (_, n))| n.as_deref() == Some(needle.as_str()))
    }) {
        unsafe { *out_handle = h as *mut c_void };
        S_OK
    } else {
        E_INVALIDARG
    }
}

unsafe extern "system" fn dev_get_raytracing_prebuild_info(
    _this: *mut c_void,
    inputs: *const BuildRaytracingInputs,
    info: *mut RaytracingPrebuildInfo,
) {
    if inputs.is_null() || info.is_null() {
        return;
    }
    let i = unsafe { *inputs };
    let (prims, leaf) = if i.as_type == 0 {
        (u64::from(i.num_descs), crate::d3d12_rt::INSTANCE_BYTES)
    } else {
        let (geoms, _) = unsafe { build_inputs(&i) };
        let p = geoms
            .iter()
            .map(|g| match g {
                AccelGeometry::Triangles {
                    index_count,
                    vertex_count,
                    index_format,
                    ..
                } => u64::from(
                    if *index_format == 0 {
                        *vertex_count
                    } else {
                        *index_count
                    } / 3,
                ),
                AccelGeometry::Aabbs { count, .. } => *count,
            })
            .sum();
        (p, crate::d3d12_rt::TRIANGLE_BYTES)
    };
    let size = crate::d3d12_rt::max_size(prims, leaf).next_multiple_of(256);
    unsafe {
        *info = RaytracingPrebuildInfo {
            result_max_size: size,
            // Strata builds on the CPU: no scratch is read.
            scratch_size: 256,
            update_scratch_size: 256,
        }
    };
}

unsafe extern "system" fn dev_check_driver_matching_identifier(
    _this: *mut c_void,
    _kind: u32,
    _id: *const c_void,
) -> u32 {
    0 // D3D12_DRIVER_MATCHING_IDENTIFIER_COMPATIBLE_WITH_DEVICE
}

static DEVICE_VTABLE: [VTableSlot; DEVICE_VTABLE_LEN] = {
    let mut vt = base_slots::<DEVICE_VTABLE_LEN>();
    vt[DEVICE_SLOT_GET_NODE_COUNT] = slot!(dev_get_node_count);
    vt[DEVICE_SLOT_CREATE_COMMAND_QUEUE] = slot!(dev_create_command_queue);
    vt[DEVICE_SLOT_CREATE_COMMAND_ALLOCATOR] = slot!(dev_create_command_allocator);
    vt[DEVICE_SLOT_CREATE_GRAPHICS_PSO] = slot!(dev_create_graphics_pso);
    vt[DEVICE_SLOT_CREATE_COMPUTE_PSO] = slot!(dev_create_compute_pso);
    vt[DEVICE_SLOT_CREATE_COMMAND_LIST] = slot!(dev_create_command_list);
    vt[DEVICE_SLOT_CHECK_FEATURE_SUPPORT] = slot!(dev_check_feature_support);
    vt[DEVICE_SLOT_CREATE_DESCRIPTOR_HEAP] = slot!(dev_create_descriptor_heap);
    vt[DEVICE_SLOT_GET_DESCRIPTOR_HANDLE_INC] = slot!(dev_get_descriptor_handle_inc);
    vt[DEVICE_SLOT_CREATE_ROOT_SIGNATURE] = slot!(dev_create_root_signature);
    vt[DEVICE_SLOT_CREATE_CBV] = slot!(dev_create_cbv);
    vt[DEVICE_SLOT_CREATE_SRV] = slot!(dev_create_srv);
    vt[DEVICE_SLOT_CREATE_UAV] = slot!(dev_create_uav);
    vt[DEVICE_SLOT_CREATE_RTV] = slot!(dev_create_rtv);
    vt[DEVICE_SLOT_CREATE_DSV] = slot!(dev_create_dsv);
    vt[DEVICE_SLOT_CREATE_SAMPLER] = slot!(dev_create_sampler);
    vt[DEVICE_SLOT_COPY_DESCRIPTORS] = slot!(dev_copy_descriptors);
    vt[DEVICE_SLOT_COPY_DESCRIPTORS_SIMPLE] = slot!(dev_copy_descriptors_simple);
    vt[DEVICE_SLOT_GET_RESOURCE_ALLOCATION_INFO] = slot!(dev_get_resource_allocation_info);
    vt[DEVICE_SLOT_GET_CUSTOM_HEAP_PROPERTIES] = slot!(dev_get_custom_heap_properties);
    vt[DEVICE_SLOT_CREATE_COMMITTED_RESOURCE] = slot!(dev_create_committed_resource);
    vt[DEVICE_SLOT_CREATE_HEAP] = slot!(dev_create_heap);
    vt[DEVICE_SLOT_CREATE_PLACED_RESOURCE] = slot!(dev_create_placed_resource);
    vt[DEVICE_SLOT_CREATE_RESERVED_RESOURCE] = slot!(dev_create_reserved_resource);
    vt[DEVICE_SLOT_CREATE_SHARED_HANDLE] = slot!(dev_create_shared_handle);
    vt[DEVICE_SLOT_OPEN_SHARED_HANDLE] = slot!(dev_open_shared_handle);
    vt[DEVICE_SLOT_OPEN_SHARED_HANDLE_BY_NAME] = slot!(dev_open_shared_handle_by_name);
    vt[DEVICE_SLOT_MAKE_RESIDENT] = slot!(stub_ok);
    vt[DEVICE_SLOT_EVICT] = slot!(stub_ok);
    vt[DEVICE_SLOT_CREATE_FENCE] = slot!(dev_create_fence);
    vt[DEVICE_SLOT_GET_DEVICE_REMOVED_REASON] = slot!(stub_ok);
    vt[DEVICE_SLOT_GET_COPYABLE_FOOTPRINTS] = slot!(dev_get_copyable_footprints);
    vt[DEVICE_SLOT_CREATE_QUERY_HEAP] = slot!(dev_create_query_heap);
    vt[DEVICE_SLOT_SET_STABLE_POWER_STATE] = slot!(stub_ok);
    vt[DEVICE_SLOT_CREATE_COMMAND_SIGNATURE] = slot!(dev_create_command_signature);
    vt[DEVICE_SLOT_GET_RESOURCE_TILING] = slot!(dev_get_resource_tiling);
    vt[DEVICE_SLOT_GET_ADAPTER_LUID] = slot!(dev_get_adapter_luid);
    vt[DEVICE_SLOT_CREATE_PIPELINE_LIBRARY] = slot!(dev_create_pipeline_library);
    vt[DEVICE_SLOT_SET_EVENT_ON_MULTIPLE_FENCE_COMPLETION] =
        slot!(dev_set_event_on_multiple_fence_completion);
    vt[DEVICE_SLOT_SET_RESIDENCY_PRIORITY] = slot!(stub_ok);
    vt[DEVICE_SLOT_CREATE_PIPELINE_STATE] = slot!(dev_create_pipeline_state);
    vt[DEVICE_SLOT_OPEN_EXISTING_HEAP_FROM_ADDRESS] = slot!(dev_unsupported);
    vt[DEVICE_SLOT_OPEN_EXISTING_HEAP_FROM_FILE_MAPPING] = slot!(dev_unsupported);
    vt[DEVICE_SLOT_ENQUEUE_MAKE_RESIDENT] = slot!(dev_enqueue_make_resident);
    vt[DEVICE_SLOT_CREATE_COMMAND_LIST1] = slot!(dev_create_command_list1);
    vt[DEVICE_SLOT_CREATE_PROTECTED_RESOURCE_SESSION] = slot!(dev_unsupported);
    vt[DEVICE_SLOT_CREATE_COMMITTED_RESOURCE1] = slot!(dev_create_committed_resource1);
    vt[DEVICE_SLOT_CREATE_HEAP1] = slot!(dev_create_heap1);
    vt[DEVICE_SLOT_CREATE_RESERVED_RESOURCE1] = slot!(dev_create_reserved_resource1);
    vt[DEVICE_SLOT_GET_RESOURCE_ALLOCATION_INFO1] = slot!(dev_get_resource_allocation_info1);
    vt[DEVICE_SLOT_CREATE_LIFETIME_TRACKER] = slot!(dev_unsupported);
    vt[DEVICE_SLOT_REMOVE_DEVICE] = slot!(stub_void);
    vt[DEVICE_SLOT_ENUMERATE_META_COMMANDS] = slot!(dev_enumerate_meta_commands);
    vt[DEVICE_SLOT_ENUMERATE_META_COMMAND_PARAMETERS] = slot!(dev_unsupported);
    vt[DEVICE_SLOT_CREATE_META_COMMAND] = slot!(dev_unsupported);
    vt[DEVICE_SLOT_CREATE_STATE_OBJECT] = slot!(dev_create_state_object);
    vt[DEVICE_SLOT_GET_RAYTRACING_AS_PREBUILD_INFO] = slot!(dev_get_raytracing_prebuild_info);
    vt[DEVICE_SLOT_CHECK_DRIVER_MATCHING_IDENTIFIER] = slot!(dev_check_driver_matching_identifier);
    vt
};

impl Com<DeviceInner> {
    /// A device on the first Vulkan 1.3 GPU (or GPU-less, when none).
    // UMA allocations are host pages any thread may use; the manager is only
    // reached through its mutex (Backend is Send + Sync for that reason).
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new_boxed() -> *mut Self {
        let backend = ComputeRunner::new().ok().map(|runner| {
            let backend = Arc::new(Backend {
                runner: Arc::new(runner),
                uma: Arc::new(Mutex::new(UmaManager::new())),
                pipelines: Arc::new(Mutex::new(HashMap::new())),
                textures: Arc::new(Mutex::new(TextureCache::new())),
            });
            crate::pipeline_cache::register(&backend);
            backend
        });
        Self::with_backend(backend)
    }

    /// A device without a GPU: objects work, lists record, nothing executes.
    pub fn new_headless() -> *mut Self {
        Self::with_backend(None)
    }

    pub fn with_backend(backend: Option<Arc<Backend>>) -> *mut Self {
        Com::boxed(
            &DEVICE_VTABLE,
            std::ptr::null_mut(),
            DeviceInner {
                backend,
                next_pso: AtomicU64::new(1),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn f<T: Copy>(p: *mut c_void, slot: usize) -> T {
        unsafe {
            let vt = *(p as *const *const VTableSlot);
            std::mem::transmute_copy(&(*vt.add(slot)).0)
        }
    }

    #[test]
    fn a_headless_device_makes_every_object_and_records_lists() {
        let dev = StandardDevice::new_headless() as *mut c_void;
        unsafe {
            let get_nodes: unsafe extern "system" fn(*mut c_void) -> u32 =
                f(dev, DEVICE_SLOT_GET_NODE_COUNT);
            assert_eq!(get_nodes(dev), 1);
            let inc: unsafe extern "system" fn(*mut c_void, u32) -> u32 =
                f(dev, DEVICE_SLOT_GET_DESCRIPTOR_HANDLE_INC);
            assert_eq!((inc(dev, 0), inc(dev, 1)), (32, 16));

            // Fence: events, signal, private data.
            let create_fence: unsafe extern "system" fn(
                *mut c_void,
                u64,
                u32,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = f(dev, DEVICE_SLOT_CREATE_FENCE);
            let mut fence = std::ptr::null_mut();
            assert_eq!(create_fence(dev, 10, 0, std::ptr::null(), &mut fence), S_OK);
            let completed: unsafe extern "system" fn(*mut c_void) -> u64 =
                f(fence, FENCE_SLOT_GET_COMPLETED_VALUE);
            assert_eq!(completed(fence), 10);
            let guid = [0x11u8; 16];
            let set_pd: unsafe extern "system" fn(
                *mut c_void,
                *const [u8; 16],
                u32,
                *const c_void,
            ) -> i32 = f(fence, SLOT_SET_PRIVATE_DATA);
            let get_pd: unsafe extern "system" fn(
                *mut c_void,
                *const [u8; 16],
                *mut u32,
                *mut c_void,
            ) -> i32 = f(fence, SLOT_GET_PRIVATE_DATA);
            assert_eq!(
                set_pd(fence, &guid, 4, [1u8, 2, 3, 4].as_ptr().cast()),
                S_OK
            );
            let (mut buf, mut sz) = ([0u8; 4], 4u32);
            assert_eq!(get_pd(fence, &guid, &mut sz, buf.as_mut_ptr().cast()), S_OK);
            assert_eq!(buf, [1, 2, 3, 4]);

            // Queue signal.
            let create_q: unsafe extern "system" fn(
                *mut c_void,
                *const D3D12CommandQueueDesc,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = f(dev, DEVICE_SLOT_CREATE_COMMAND_QUEUE);
            let mut q = std::ptr::null_mut();
            assert_eq!(
                create_q(
                    dev,
                    &D3D12CommandQueueDesc::default(),
                    std::ptr::null(),
                    &mut q
                ),
                S_OK
            );
            let signal: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> i32 =
                f(q, QUEUE_SLOT_SIGNAL);
            assert_eq!(signal(q, fence, 42), S_OK);
            assert_eq!(completed(fence), 42);

            // A 2D texture: SRV written into a descriptor heap as a T#.
            let create_heap: unsafe extern "system" fn(
                *mut c_void,
                *const DescriptorHeapDesc,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = f(dev, DEVICE_SLOT_CREATE_DESCRIPTOR_HEAP);
            let mut heap = std::ptr::null_mut();
            let hd = DescriptorHeapDesc {
                heap_type: 0,
                num_descriptors: 4,
                flags: 1,
                node_mask: 0,
            };
            assert_eq!(create_heap(dev, &hd, std::ptr::null(), &mut heap), S_OK);
            let cpu_start: unsafe extern "system" fn(*mut c_void, *mut u64) -> *mut u64 =
                f(heap, 9);
            let mut cpu = 0u64;
            cpu_start(heap, &mut cpu);
            assert_ne!(cpu, 0);

            let create_res: unsafe extern "system" fn(
                *mut c_void,
                *const HeapProperties,
                u32,
                *const ResourceDesc,
                u32,
                *const c_void,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = f(dev, DEVICE_SLOT_CREATE_COMMITTED_RESOURCE);
            let tex_desc = ResourceDesc {
                dimension: DIMENSION_TEXTURE2D,
                width: 64,
                height: 64,
                depth_or_array_size: 1,
                mip_levels: 1,
                format: 28,
                sample_count: 1,
                ..Default::default()
            };
            let mut tex = std::ptr::null_mut();
            assert_eq!(
                create_res(
                    dev,
                    &HeapProperties::default(),
                    0,
                    &tex_desc,
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    &mut tex
                ),
                S_OK
            );
            let create_srv: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                *const ShaderResourceViewDesc,
                u64,
            ) = f(dev, DEVICE_SLOT_CREATE_SRV);
            create_srv(dev, tex, std::ptr::null(), cpu);
            let t = std::ptr::read_unaligned(cpu as *const [u32; 8]);
            let d = strata_shader::descriptor::ImageDescriptor::decode(t);
            assert_eq!((d.width, d.height, d.format), (64, 64, 56));
            let va: unsafe extern "system" fn(*mut c_void) -> u64 =
                f(tex, RESOURCE_SLOT_GET_GPU_VA);
            assert_eq!(d.base_address, va(tex));

            // A list records state, clears and draws; ExecuteCommandLists counts it.
            let create_cl: unsafe extern "system" fn(
                *mut c_void,
                u32,
                u32,
                *mut c_void,
                *mut c_void,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = f(dev, DEVICE_SLOT_CREATE_COMMAND_LIST);
            let mut list = std::ptr::null_mut();
            assert_eq!(
                create_cl(
                    dev,
                    0,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    &mut list
                ),
                S_OK
            );
            let vp = Viewport {
                width: 64.0,
                height: 64.0,
                max_depth: 1.0,
                ..Default::default()
            };
            let set_vp: unsafe extern "system" fn(*mut c_void, u32, *const Viewport) =
                f(list, CMDLIST_SLOT_RS_SET_VIEWPORTS);
            set_vp(list, 1, &vp);
            let clear_rtv: unsafe extern "system" fn(
                *mut c_void,
                u64,
                *const f32,
                u32,
                *const Rect,
            ) = f(list, CMDLIST_SLOT_CLEAR_RENDER_TARGET_VIEW);
            clear_rtv(
                list,
                cpu,
                [1.0f32, 0.0, 0.0, 1.0].as_ptr(),
                0,
                std::ptr::null(),
            );
            let draw: unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32) =
                f(list, CMDLIST_SLOT_DRAW_INSTANCED);
            draw(list, 3, 1, 0, 0);
            let close: unsafe extern "system" fn(*mut c_void) -> i32 = f(list, CMDLIST_SLOT_CLOSE);
            assert_eq!(close(list), S_OK);
            let cmds = (*(list as *const StandardCommandList))
                .inner
                .thin
                .lock()
                .unwrap()
                .commands()
                .to_vec();
            assert!(matches!(cmds[0], ThinCommand::SetViewport(_)));
            assert!(matches!(cmds[1], ThinCommand::ClearImage { .. }));
            assert!(cmds.iter().any(|c| matches!(
                c,
                ThinCommand::DrawInstanced {
                    vertex_count: 3,
                    ..
                }
            )));
            let exec: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) =
                f(q, QUEUE_SLOT_EXECUTE_COMMAND_LISTS);
            exec(q, 1, &list);
            assert_eq!(
                (*(q as *const StandardCommandQueue))
                    .inner
                    .executed_lists
                    .load(Ordering::Relaxed),
                1
            );

            // Feature support answers the feature-level query.
            let check: unsafe extern "system" fn(*mut c_void, u32, *mut u8, u32) -> i32 =
                f(dev, DEVICE_SLOT_CHECK_FEATURE_SUPPORT);
            let levels = [0xb000u32, 0xc000, 0xc100];
            let mut fl = [0u8; 24];
            (fl.as_mut_ptr() as *mut u32).write_unaligned(3);
            (fl.as_mut_ptr().add(8) as *mut *const u32).write_unaligned(levels.as_ptr());
            assert_eq!(check(dev, 2, fl.as_mut_ptr(), 24), S_OK);
            assert_eq!((fl.as_ptr().add(16) as *const u32).read_unaligned(), 0xc100);

            for o in [list, tex, heap, q, fence] {
                release(o);
            }
            let rel: unsafe extern "system" fn(*mut c_void) -> u32 = f(dev, SLOT_RELEASE);
            assert_eq!(rel(dev), 0);
        }
    }

    #[test]
    fn root_parameters_land_in_the_user_sgprs_the_layout_assigns() {
        let dev = StandardDevice::new_headless() as *mut c_void;
        unsafe {
            // Constants(2 dwords) + table: SGPRs 0-1 and 2-3.
            #[repr(C)]
            struct Param {
                ty: u32,
                pad: u32,
                a: u64,
                b: u64,
                vis: u32,
                pad2: u32,
            }
            let range = [0u32, 4, 0, 0, 0];
            let params = [
                Param {
                    ty: 1,
                    pad: 0,
                    a: 0,
                    b: 2,
                    vis: 0,
                    pad2: 0,
                },
                Param {
                    ty: 0,
                    pad: 0,
                    a: 1,
                    b: range.as_ptr() as u64,
                    vis: 0,
                    pad2: 0,
                },
            ];
            let desc: [u64; 5] = [2, params.as_ptr() as u64, 0, 0, 0];
            let blob = serialize_root_signature(desc.as_ptr() as *const u8, 1);
            let create_rs: unsafe extern "system" fn(
                *mut c_void,
                u32,
                *const u8,
                usize,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = f(dev, DEVICE_SLOT_CREATE_ROOT_SIGNATURE);
            let mut rs = std::ptr::null_mut();
            assert_eq!(
                create_rs(dev, 0, blob.as_ptr(), blob.len(), std::ptr::null(), &mut rs),
                S_OK
            );

            let list = StandardCommandList::new_boxed(0, dev) as *mut c_void;
            let set_rs: unsafe extern "system" fn(*mut c_void, *mut c_void) =
                f(list, CMDLIST_SLOT_SET_GRAPHICS_ROOT_SIGNATURE);
            let set_c: unsafe extern "system" fn(*mut c_void, u32, u32, *const u32, u32) =
                f(list, CMDLIST_SLOT_SET_GRAPHICS_ROOT_32BIT_CONSTANTS);
            let set_t: unsafe extern "system" fn(*mut c_void, u32, u64) =
                f(list, CMDLIST_SLOT_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE);
            let draw: unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32) =
                f(list, CMDLIST_SLOT_DRAW_INSTANCED);
            set_rs(list, rs);
            set_c(list, 0, 2, [7u32, 8].as_ptr(), 0);
            set_t(list, 1, 0x1234_5678_9abc);
            draw(list, 3, 1, 0, 0);
            let cmds = (*(list as *const StandardCommandList))
                .inner
                .thin
                .lock()
                .unwrap()
                .commands()
                .to_vec();
            let pushed = cmds
                .iter()
                .find_map(|c| match c {
                    ThinCommand::PushUserSgprs {
                        offset_dw: 0,
                        values,
                    } => Some(values.clone()),
                    _ => None,
                })
                .unwrap();
            assert_eq!(&pushed[..4], &[7, 8, 0x5678_9abc, 0x1234]);
            release(list);
            release(rs);
            release(dev);
        }
    }

    #[test]
    fn a_pso_stream_parses_into_graphics_state() {
        // Topology (type 14) = LINE (2), DSV format (16) = D32_FLOAT (40).
        let mut s = Vec::new();
        for (ty, v) in [(14u32, 2u32), (16, 40)] {
            s.extend_from_slice(&ty.to_le_bytes());
            s.extend_from_slice(&v.to_le_bytes());
        }
        match parse_pso_stream(&s) {
            Some(PsoStream::Graphics(d, None, None)) => {
                assert_eq!(d.primitive_topology_type, 2);
                assert_eq!(d.dsv_format, 40);
                let st = pso_state(&d);
                assert_eq!(st.topology, ash::vk::PrimitiveTopology::LINE_LIST);
                assert_eq!(st.depth_format, Some(ash::vk::Format::D32_SFLOAT));
            }
            _ => panic!("expected a graphics stream"),
        }
    }
}
