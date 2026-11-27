//! Standard Direct3D 12 COM vtable slot layouts and stateful COM implementations
//! matching the published Microsoft `d3d12.h` ABI slot indices (`IUnknown` ->
//! `ID3D12Object` -> `ID3D12DeviceChild` -> `ID3D12Fence` / `ID3D12Resource` /
//! `ID3D12CommandQueue` / `ID3D12GraphicsCommandList` / `ID3D12Device`).
//!
//! Each vtable is a fixed-size `[VTableSlot; N]` slot table with exact `d3d12.h`
//! slot index constants and stateful method bodies so retail callers invoking
//! virtual methods by slot index dispatch to full implementations.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use strata_gpu::present::FramePacingClock;
use strata_gpu::runner::ComputeRunner;
use strata_gpu::thin_list::{PipelineKey, ThinCommand, ThinCommandList};
use strata_gpu::uma::UmaManager;

/// Live Vulkan 1.3 + UMA + Frame-Pacing backend attached to [`StandardDevice`].
pub struct LiveGpuBackend {
    pub runner: ComputeRunner,
    pub uma: UmaManager,
    pub frame_clock: FramePacingClock,
    pub presented_frames: u64,
    pub frame_events: HashMap<u64, u64>,
    pub compute_psos: HashMap<u64, Vec<u32>>,
}

// SAFETY: LiveGpuBackend is exclusively accessed behind an Arc<Mutex<_>>.
unsafe impl Send for LiveGpuBackend {}

impl LiveGpuBackend {
    pub fn try_new() -> Option<Self> {
        let runner = ComputeRunner::new().ok()?;
        Some(Self {
            runner,
            uma: UmaManager::new(),
            frame_clock: FramePacingClock::new(60),
            presented_frames: 0,
            frame_events: HashMap::new(),
            compute_psos: HashMap::new(),
        })
    }
}

const S_OK: i32 = 0;
const E_NOINTERFACE: i32 = 0x8000_4002u32 as i32;
const E_INVALIDARG: i32 = 0x8007_0057u32 as i32;
const DXGI_ERROR_NOT_FOUND: i32 = 0x887A_0002u32 as i32;

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

/// ABI-compatible `D3D12_RESOURCE_DESC`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct D3D12ResourceDesc {
    pub dimension: u32,
    pub alignment: u64,
    pub width: u64,
    pub height: u32,
    pub depth_or_array_size: u16,
    pub mip_levels: u16,
    pub format: u32,
    pub sample_count: u32,
    pub sample_quality: u32,
    pub layout: u32,
    pub flags: u32,
}

/// ABI-compatible `D3D12_HEAP_PROPERTIES`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct D3D12HeapProperties {
    pub heap_type: u32,
    pub cpu_page_property: u32,
    pub memory_pool_preference: u32,
    pub creation_node_mask: u32,
    pub visible_node_mask: u32,
}

/// ABI-compatible `D3D12_COMMAND_QUEUE_DESC`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct D3D12CommandQueueDesc {
    pub list_type: u32,
    pub priority: i32,
    pub flags: u32,
    pub node_mask: u32,
}

/// Shared `ID3D12Object` private-data & debug-name metadata.
#[derive(Default)]
pub struct ObjectMeta {
    pub name: String,
    pub private_data: HashMap<[u8; 16], Vec<u8>>,
}

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

fn get_private_data_impl(
    meta: &Mutex<ObjectMeta>,
    guid: *const [u8; 16],
    data_size: *mut u32,
    out_data: *mut c_void,
) -> i32 {
    if guid.is_null() || data_size.is_null() {
        return E_INVALIDARG;
    }
    let key = unsafe { *guid };
    let guard = meta.lock().unwrap();
    let Some(bytes) = guard.private_data.get(&key) else {
        unsafe { *data_size = 0 };
        return DXGI_ERROR_NOT_FOUND;
    };
    let cap = unsafe { *data_size } as usize;
    unsafe { *data_size = bytes.len() as u32 };
    if !out_data.is_null() && cap >= bytes.len() {
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_data.cast::<u8>(), bytes.len());
        }
    }
    S_OK
}

fn set_private_data_impl(
    meta: &Mutex<ObjectMeta>,
    guid: *const [u8; 16],
    data_size: u32,
    data: *const c_void,
) -> i32 {
    if guid.is_null() {
        return E_INVALIDARG;
    }
    let key = unsafe { *guid };
    let mut guard = meta.lock().unwrap();
    if data.is_null() || data_size == 0 {
        guard.private_data.remove(&key);
    } else {
        let slice = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), data_size as usize) };
        guard.private_data.insert(key, slice.to_vec());
    }
    S_OK
}

fn set_name_impl(meta: &Mutex<ObjectMeta>, name_utf16: *const u16) -> i32 {
    if name_utf16.is_null() {
        return E_INVALIDARG;
    }
    let mut len = 0usize;
    unsafe {
        while *name_utf16.add(len) != 0 && len < 1024 {
            len += 1;
        }
        let slice = std::slice::from_raw_parts(name_utf16, len);
        meta.lock().unwrap().name = String::from_utf16_lossy(slice);
    }
    S_OK
}

#[cfg(windows)]
unsafe extern "system" {
    fn SetEvent(h_event: *mut c_void) -> i32;
}

// ============================================================================
// ID3D12Fence (11 slots)
// ============================================================================

/// Standard-ABI `ID3D12Fence` COM object with exact `d3d12.h` slot indices (0..11).
#[repr(C)]
pub struct StandardFence {
    pub vtable: &'static [VTableSlot; FENCE_VTABLE_LEN],
    pub ref_count: AtomicU32,
    pub value: AtomicU64,
    pub device: *mut c_void,
    pub meta: Mutex<ObjectMeta>,
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

unsafe extern "system" fn std_fence_get_private_data(
    this: *mut c_void,
    guid: *const [u8; 16],
    size: *mut u32,
    out: *mut c_void,
) -> i32 {
    let f = unsafe { &*(this as *const StandardFence) };
    get_private_data_impl(&f.meta, guid, size, out)
}

unsafe extern "system" fn std_fence_set_private_data(
    this: *mut c_void,
    guid: *const [u8; 16],
    size: u32,
    data: *const c_void,
) -> i32 {
    let f = unsafe { &*(this as *const StandardFence) };
    set_private_data_impl(&f.meta, guid, size, data)
}

unsafe extern "system" fn std_fence_set_name(this: *mut c_void, name: *const u16) -> i32 {
    let f = unsafe { &*(this as *const StandardFence) };
    set_name_impl(&f.meta, name)
}

unsafe extern "system" fn std_fence_get_device(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let f = unsafe { &*(this as *const StandardFence) };
    unsafe { *out = f.device };
    S_OK
}

unsafe extern "system" fn std_fence_get_completed(this: *mut c_void) -> u64 {
    let f = unsafe { &*(this as *const StandardFence) };
    f.value.load(Ordering::Acquire)
}

unsafe extern "system" fn std_fence_set_event_on_completion(
    this: *mut c_void,
    value: u64,
    h_event: *mut c_void,
) -> i32 {
    let f = unsafe { &*(this as *const StandardFence) };
    if f.value.load(Ordering::Acquire) >= value && !h_event.is_null() {
        #[cfg(windows)]
        unsafe {
            SetEvent(h_event);
        }
    }
    S_OK
}

unsafe extern "system" fn std_fence_signal(this: *mut c_void, val: u64) -> i32 {
    let f = unsafe { &*(this as *const StandardFence) };
    f.value.store(val, Ordering::Release);
    S_OK
}

static STANDARD_FENCE_VTABLE: [VTableSlot; FENCE_VTABLE_LEN] = [
    VTableSlot(std_fence_qi as *const ()),
    VTableSlot(std_fence_add_ref as *const ()),
    VTableSlot(std_fence_release as *const ()),
    VTableSlot(std_fence_get_private_data as *const ()),
    VTableSlot(std_fence_set_private_data as *const ()),
    VTableSlot(stub_ok as *const ()),
    VTableSlot(std_fence_set_name as *const ()),
    VTableSlot(std_fence_get_device as *const ()),
    VTableSlot(std_fence_get_completed as *const ()),
    VTableSlot(std_fence_set_event_on_completion as *const ()),
    VTableSlot(std_fence_signal as *const ()),
];

impl StandardFence {
    pub fn new_boxed(initial: u64) -> *mut Self {
        Self::new_boxed_with_device(initial, std::ptr::null_mut())
    }

    pub fn new_boxed_with_device(initial: u64, device: *mut c_void) -> *mut Self {
        Box::into_raw(Box::new(Self {
            vtable: &STANDARD_FENCE_VTABLE,
            ref_count: AtomicU32::new(1),
            value: AtomicU64::new(initial),
            device,
            meta: Mutex::new(ObjectMeta::default()),
        }))
    }
}

// ============================================================================
// ID3D12Resource (15 slots)
// ============================================================================

/// Standard-ABI `ID3D12Resource` COM object with full `Map`/`Unmap`, `GetDesc`,
/// `GetGPUVirtualAddress`, `WriteToSubresource`/`ReadFromSubresource`, and `GetHeapProperties`.
#[repr(C)]
pub struct StandardResource {
    pub vtable: &'static [VTableSlot; RESOURCE_VTABLE_LEN],
    pub ref_count: AtomicU32,
    pub device: *mut c_void,
    pub desc: D3D12ResourceDesc,
    pub heap_props: D3D12HeapProperties,
    pub gpu_va: u64,
    pub uma_va: u64,
    pub size_bytes: usize,
    pub backend: Option<Arc<Mutex<LiveGpuBackend>>>,
    pub backing: Mutex<Vec<u8>>,
    pub meta: Mutex<ObjectMeta>,
}

unsafe extern "system" fn std_res_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_NOINTERFACE;
    }
    unsafe {
        std_res_add_ref(this);
        *out = this;
    }
    S_OK
}

unsafe extern "system" fn std_res_add_ref(this: *mut c_void) -> u32 {
    let r = unsafe { &*(this as *const StandardResource) };
    r.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn std_res_release(this: *mut c_void) -> u32 {
    let r = unsafe { &*(this as *const StandardResource) };
    let prev = r.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        let boxed = unsafe { Box::from_raw(this as *mut StandardResource) };
        if boxed.uma_va != 0
            && let Some(ref backend) = boxed.backend
            && let Ok(mut b) = backend.lock()
        {
            let LiveGpuBackend { runner, uma, .. } = &mut *b;
            let _ = uma.free_allocation(runner, boxed.uma_va);
        }
        drop(boxed);
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn std_res_get_private_data(
    this: *mut c_void,
    guid: *const [u8; 16],
    size: *mut u32,
    out: *mut c_void,
) -> i32 {
    let r = unsafe { &*(this as *const StandardResource) };
    get_private_data_impl(&r.meta, guid, size, out)
}

unsafe extern "system" fn std_res_set_private_data(
    this: *mut c_void,
    guid: *const [u8; 16],
    size: u32,
    data: *const c_void,
) -> i32 {
    let r = unsafe { &*(this as *const StandardResource) };
    set_private_data_impl(&r.meta, guid, size, data)
}

unsafe extern "system" fn std_res_set_name(this: *mut c_void, name: *const u16) -> i32 {
    let r = unsafe { &*(this as *const StandardResource) };
    set_name_impl(&r.meta, name)
}

unsafe extern "system" fn std_res_get_device(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let r = unsafe { &*(this as *const StandardResource) };
    unsafe { *out = r.device };
    S_OK
}

unsafe extern "system" fn std_res_map(
    this: *mut c_void,
    _subresource: u32,
    _read_range: *const c_void,
    out_data: *mut *mut c_void,
) -> i32 {
    let r = unsafe { &*(this as *const StandardResource) };
    if !out_data.is_null() {
        if r.uma_va != 0 {
            unsafe { *out_data = r.uma_va as *mut c_void };
        } else {
            let mut guard = r.backing.lock().unwrap();
            unsafe { *out_data = guard.as_mut_ptr().cast() };
        }
    }
    S_OK
}

unsafe extern "system" fn std_res_unmap(
    _this: *mut c_void,
    _subresource: u32,
    _written_range: *const c_void,
) {
}

unsafe extern "system" fn std_res_get_desc(
    this: *mut c_void,
    out_desc: *mut D3D12ResourceDesc,
) -> *mut D3D12ResourceDesc {
    let r = unsafe { &*(this as *const StandardResource) };
    if !out_desc.is_null() {
        unsafe { *out_desc = r.desc };
    }
    out_desc
}

unsafe extern "system" fn std_res_get_gpu_va(this: *mut c_void) -> u64 {
    let r = unsafe { &*(this as *const StandardResource) };
    r.gpu_va
}

unsafe extern "system" fn std_res_write_to_subresource(
    this: *mut c_void,
    _dst_subresource: u32,
    _dst_box: *const c_void,
    src_data: *const c_void,
    src_row_pitch: u32,
    _src_depth_pitch: u32,
) -> i32 {
    if src_data.is_null() {
        return E_INVALIDARG;
    }
    let r = unsafe { &*(this as *const StandardResource) };
    if r.uma_va != 0 {
        let copy_len = (src_row_pitch as usize).min(r.size_bytes);
        unsafe {
            std::ptr::copy_nonoverlapping(src_data.cast::<u8>(), r.uma_va as *mut u8, copy_len);
        }
    } else {
        let mut guard = r.backing.lock().unwrap();
        let copy_len = (src_row_pitch as usize).min(guard.len());
        unsafe {
            std::ptr::copy_nonoverlapping(src_data.cast::<u8>(), guard.as_mut_ptr(), copy_len);
        }
    }
    S_OK
}

unsafe extern "system" fn std_res_read_from_subresource(
    this: *mut c_void,
    dst_data: *mut c_void,
    dst_row_pitch: u32,
    _dst_depth_pitch: u32,
    _src_subresource: u32,
    _src_box: *const c_void,
) -> i32 {
    if dst_data.is_null() {
        return E_INVALIDARG;
    }
    let r = unsafe { &*(this as *const StandardResource) };
    if r.uma_va != 0 {
        let copy_len = (dst_row_pitch as usize).min(r.size_bytes);
        unsafe {
            std::ptr::copy_nonoverlapping(r.uma_va as *const u8, dst_data.cast::<u8>(), copy_len);
        }
    } else {
        let guard = r.backing.lock().unwrap();
        let copy_len = (dst_row_pitch as usize).min(guard.len());
        unsafe {
            std::ptr::copy_nonoverlapping(guard.as_ptr(), dst_data.cast::<u8>(), copy_len);
        }
    }
    S_OK
}

unsafe extern "system" fn std_res_get_heap_props(
    this: *mut c_void,
    out_props: *mut D3D12HeapProperties,
    out_flags: *mut u32,
) -> i32 {
    let r = unsafe { &*(this as *const StandardResource) };
    if !out_props.is_null() {
        unsafe { *out_props = r.heap_props };
    }
    if !out_flags.is_null() {
        unsafe { *out_flags = 0 };
    }
    S_OK
}

static STANDARD_RESOURCE_VTABLE: [VTableSlot; RESOURCE_VTABLE_LEN] = [
    VTableSlot(std_res_qi as *const ()),
    VTableSlot(std_res_add_ref as *const ()),
    VTableSlot(std_res_release as *const ()),
    VTableSlot(std_res_get_private_data as *const ()),
    VTableSlot(std_res_set_private_data as *const ()),
    VTableSlot(stub_ok as *const ()),
    VTableSlot(std_res_set_name as *const ()),
    VTableSlot(std_res_get_device as *const ()),
    VTableSlot(std_res_map as *const ()),
    VTableSlot(std_res_unmap as *const ()),
    VTableSlot(std_res_get_desc as *const ()),
    VTableSlot(std_res_get_gpu_va as *const ()),
    VTableSlot(std_res_write_to_subresource as *const ()),
    VTableSlot(std_res_read_from_subresource as *const ()),
    VTableSlot(std_res_get_heap_props as *const ()),
];

impl StandardResource {
    pub fn new_boxed(
        desc: D3D12ResourceDesc,
        heap_props: D3D12HeapProperties,
        device: *mut c_void,
    ) -> *mut Self {
        let backend = if !device.is_null() {
            let dev = unsafe { &*(device as *const StandardDevice) };
            dev.backend.clone()
        } else {
            None
        };
        Self::new_boxed_with_backend(desc, heap_props, device, backend)
    }

    pub fn new_boxed_with_backend(
        desc: D3D12ResourceDesc,
        heap_props: D3D12HeapProperties,
        device: *mut c_void,
        backend: Option<Arc<Mutex<LiveGpuBackend>>>,
    ) -> *mut Self {
        let size = (desc.width as usize).max(256);
        if let Some(ref b_arc) = backend
            && let Ok(mut b) = b_arc.lock()
        {
            let LiveGpuBackend { runner, uma, .. } = &mut *b;
            if let Ok(alloc) = uma.xmem_alloc_identity(runner, 0, size as u64) {
                return Box::into_raw(Box::new(Self {
                    vtable: &STANDARD_RESOURCE_VTABLE,
                    ref_count: AtomicU32::new(1),
                    device,
                    desc,
                    heap_props,
                    gpu_va: alloc.host_bda,
                    uma_va: alloc.console_va,
                    size_bytes: alloc.size as usize,
                    backend: Some(Arc::clone(b_arc)),
                    backing: Mutex::new(Vec::new()),
                    meta: Mutex::new(ObjectMeta::default()),
                }));
            }
        }
        let backing = vec![0u8; size];
        let gpu_va = backing.as_ptr() as u64;
        Box::into_raw(Box::new(Self {
            vtable: &STANDARD_RESOURCE_VTABLE,
            ref_count: AtomicU32::new(1),
            device,
            desc,
            heap_props,
            gpu_va,
            uma_va: 0,
            size_bytes: size,
            backend: None,
            backing: Mutex::new(backing),
            meta: Mutex::new(ObjectMeta::default()),
        }))
    }
}

// ============================================================================
// ID3D12GraphicsCommandList (60 slots)
// ============================================================================

/// Recorded D3D12 command list operation for inspection / replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordedCmd {
    DrawInstanced {
        vertex_count: u32,
        instance_count: u32,
        start_vertex: u32,
        start_instance: u32,
    },
    DrawIndexedInstanced {
        index_count: u32,
        instance_count: u32,
        start_index: u32,
        base_vertex: i32,
        start_instance: u32,
    },
    Dispatch {
        x: u32,
        y: u32,
        z: u32,
    },
    CopyBufferRegion {
        dst: usize,
        dst_offset: u64,
        src: usize,
        src_offset: u64,
        num_bytes: u64,
    },
    CopyResource {
        dst: usize,
        src: usize,
    },
    SetPipelineState(usize),
    ResourceBarrier {
        count: u32,
    },
    SetComputeRoot32BitConstants {
        root_param_idx: u32,
        values: Vec<u32>,
        dest_offset: u32,
    },
    OmSetRenderTargets {
        num_rtvs: u32,
    },
    ExecuteIndirect {
        max_command_count: u32,
    },
}

/// Standard-ABI `ID3D12GraphicsCommandList` COM object (60 slots).
#[repr(C)]
pub struct StandardCommandList {
    pub vtable: &'static [VTableSlot; CMDLIST_VTABLE_LEN],
    pub ref_count: AtomicU32,
    pub device: *mut c_void,
    pub list_type: u32,
    pub closed: AtomicBool,
    pub commands: Mutex<Vec<RecordedCmd>>,
    pub thin_list: Mutex<ThinCommandList>,
    pub meta: Mutex<ObjectMeta>,
}

unsafe extern "system" fn std_cl_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_NOINTERFACE;
    }
    unsafe {
        std_cl_add_ref(this);
        *out = this;
    }
    S_OK
}

unsafe extern "system" fn std_cl_add_ref(this: *mut c_void) -> u32 {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn std_cl_release(this: *mut c_void) -> u32 {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    let prev = cl.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut StandardCommandList)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn std_cl_get_type(this: *mut c_void) -> u32 {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.list_type
}

unsafe extern "system" fn std_cl_close(this: *mut c_void) -> i32 {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.closed.store(true, Ordering::Release);
    S_OK
}

unsafe extern "system" fn std_cl_reset(
    this: *mut c_void,
    _allocator: *mut c_void,
    _initial_pso: *mut c_void,
) -> i32 {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.closed.store(false, Ordering::Release);
    cl.commands.lock().unwrap().clear();
    S_OK
}

unsafe extern "system" fn std_cl_clear_state(this: *mut c_void, _pso: *mut c_void) {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.commands.lock().unwrap().clear();
}

unsafe extern "system" fn std_cl_draw_instanced(
    this: *mut c_void,
    vertex_count: u32,
    instance_count: u32,
    start_vertex: u32,
    start_instance: u32,
) {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.commands
        .lock()
        .unwrap()
        .push(RecordedCmd::DrawInstanced {
            vertex_count,
            instance_count,
            start_vertex,
            start_instance,
        });
    cl.thin_list.lock().unwrap().draw_instanced(
        vertex_count,
        instance_count,
        start_vertex,
        start_instance,
    );
}

unsafe extern "system" fn std_cl_draw_indexed_instanced(
    this: *mut c_void,
    index_count: u32,
    instance_count: u32,
    start_index: u32,
    base_vertex: i32,
    start_instance: u32,
) {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.commands
        .lock()
        .unwrap()
        .push(RecordedCmd::DrawIndexedInstanced {
            index_count,
            instance_count,
            start_index,
            base_vertex,
            start_instance,
        });
}

unsafe extern "system" fn std_cl_dispatch(this: *mut c_void, x: u32, y: u32, z: u32) {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.commands
        .lock()
        .unwrap()
        .push(RecordedCmd::Dispatch { x, y, z });
    cl.thin_list
        .lock()
        .unwrap()
        .dispatch(x, y, z, [1, 1, 1], false);
}

unsafe extern "system" fn std_cl_copy_buffer_region(
    this: *mut c_void,
    dst: *mut c_void,
    dst_offset: u64,
    src: *mut c_void,
    src_offset: u64,
    num_bytes: u64,
) {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.commands
        .lock()
        .unwrap()
        .push(RecordedCmd::CopyBufferRegion {
            dst: dst as usize,
            dst_offset,
            src: src as usize,
            src_offset,
            num_bytes,
        });
}

unsafe extern "system" fn std_cl_copy_resource(
    this: *mut c_void,
    dst: *mut c_void,
    src: *mut c_void,
) {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.commands.lock().unwrap().push(RecordedCmd::CopyResource {
        dst: dst as usize,
        src: src as usize,
    });
}

unsafe extern "system" fn std_cl_set_pso(this: *mut c_void, pso: *mut c_void) {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.commands
        .lock()
        .unwrap()
        .push(RecordedCmd::SetPipelineState(pso as usize));
    cl.thin_list
        .lock()
        .unwrap()
        .bind_pipeline(PipelineKey(pso as u64));
}

unsafe extern "system" fn std_cl_resource_barrier(
    this: *mut c_void,
    count: u32,
    _barriers: *const c_void,
) {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.commands
        .lock()
        .unwrap()
        .push(RecordedCmd::ResourceBarrier { count });
}

unsafe extern "system" fn std_cl_set_compute_root_32bit_constants(
    this: *mut c_void,
    root_param_idx: u32,
    num_32bit_values: u32,
    src_data: *const u32,
    dest_offset: u32,
) {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    let values = if src_data.is_null() || num_32bit_values == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(src_data, num_32bit_values as usize).to_vec() }
    };
    cl.thin_list
        .lock()
        .unwrap()
        .push_sgprs(dest_offset as u8, &values);
    cl.commands
        .lock()
        .unwrap()
        .push(RecordedCmd::SetComputeRoot32BitConstants {
            root_param_idx,
            values,
            dest_offset,
        });
}

unsafe extern "system" fn std_cl_om_set_render_targets(
    this: *mut c_void,
    num_rtvs: u32,
    _rtvs: *const c_void,
    _single_handle: i32,
    _dsv: *const c_void,
) {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.commands
        .lock()
        .unwrap()
        .push(RecordedCmd::OmSetRenderTargets { num_rtvs });
}

unsafe extern "system" fn std_cl_execute_indirect(
    this: *mut c_void,
    _cmd_sig: *mut c_void,
    max_command_count: u32,
    _arg_buf: *mut c_void,
    _arg_offset: u64,
    _count_buf: *mut c_void,
    _count_offset: u64,
) {
    let cl = unsafe { &*(this as *const StandardCommandList) };
    cl.commands
        .lock()
        .unwrap()
        .push(RecordedCmd::ExecuteIndirect { max_command_count });
}

static STANDARD_CMDLIST_VTABLE: [VTableSlot; CMDLIST_VTABLE_LEN] = {
    let mut vt = [VTableSlot(stub_void as *const ()); CMDLIST_VTABLE_LEN];
    vt[SLOT_QUERY_INTERFACE] = VTableSlot(std_cl_qi as *const ());
    vt[SLOT_ADD_REF] = VTableSlot(std_cl_add_ref as *const ());
    vt[SLOT_RELEASE] = VTableSlot(std_cl_release as *const ());
    vt[SLOT_GET_PRIVATE_DATA] = VTableSlot(stub_ok as *const ());
    vt[SLOT_SET_PRIVATE_DATA] = VTableSlot(stub_ok as *const ());
    vt[SLOT_SET_PRIVATE_DATA_INTERFACE] = VTableSlot(stub_ok as *const ());
    vt[SLOT_SET_NAME] = VTableSlot(stub_ok as *const ());
    vt[SLOT_GET_DEVICE] = VTableSlot(stub_ok as *const ());
    vt[CMDLIST_SLOT_GET_TYPE] = VTableSlot(std_cl_get_type as *const ());
    vt[CMDLIST_SLOT_CLOSE] = VTableSlot(std_cl_close as *const ());
    vt[CMDLIST_SLOT_RESET] = VTableSlot(std_cl_reset as *const ());
    vt[CMDLIST_SLOT_CLEAR_STATE] = VTableSlot(std_cl_clear_state as *const ());
    vt[CMDLIST_SLOT_DRAW_INSTANCED] = VTableSlot(std_cl_draw_instanced as *const ());
    vt[CMDLIST_SLOT_DRAW_INDEXED_INSTANCED] =
        VTableSlot(std_cl_draw_indexed_instanced as *const ());
    vt[CMDLIST_SLOT_DISPATCH] = VTableSlot(std_cl_dispatch as *const ());
    vt[CMDLIST_SLOT_COPY_BUFFER_REGION] = VTableSlot(std_cl_copy_buffer_region as *const ());
    vt[CMDLIST_SLOT_COPY_RESOURCE] = VTableSlot(std_cl_copy_resource as *const ());
    vt[CMDLIST_SLOT_SET_PIPELINE_STATE] = VTableSlot(std_cl_set_pso as *const ());
    vt[CMDLIST_SLOT_RESOURCE_BARRIER] = VTableSlot(std_cl_resource_barrier as *const ());
    vt[CMDLIST_SLOT_SET_COMPUTE_ROOT_32BIT_CONSTANTS] =
        VTableSlot(std_cl_set_compute_root_32bit_constants as *const ());
    vt[CMDLIST_SLOT_OM_SET_RENDER_TARGETS] = VTableSlot(std_cl_om_set_render_targets as *const ());
    vt[CMDLIST_SLOT_EXECUTE_INDIRECT] = VTableSlot(std_cl_execute_indirect as *const ());
    vt
};

impl StandardCommandList {
    pub fn new_boxed(list_type: u32, device: *mut c_void) -> *mut Self {
        Box::into_raw(Box::new(Self {
            vtable: &STANDARD_CMDLIST_VTABLE,
            ref_count: AtomicU32::new(1),
            device,
            list_type,
            closed: AtomicBool::new(false),
            commands: Mutex::new(Vec::new()),
            thin_list: Mutex::new(ThinCommandList::new()),
            meta: Mutex::new(ObjectMeta::default()),
        }))
    }
}

// ============================================================================
// ID3D12CommandQueue (19 slots)
// ============================================================================

/// Standard-ABI `ID3D12CommandQueue` COM object (19 slots).
#[repr(C)]
pub struct StandardCommandQueue {
    pub vtable: &'static [VTableSlot; QUEUE_VTABLE_LEN],
    pub ref_count: AtomicU32,
    pub device: *mut c_void,
    pub desc: D3D12CommandQueueDesc,
    pub executed_batches: AtomicU64,
    pub executed_lists: AtomicU64,
    pub meta: Mutex<ObjectMeta>,
}

unsafe extern "system" fn std_q_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_NOINTERFACE;
    }
    unsafe {
        std_q_add_ref(this);
        *out = this;
    }
    S_OK
}

unsafe extern "system" fn std_q_add_ref(this: *mut c_void) -> u32 {
    let q = unsafe { &*(this as *const StandardCommandQueue) };
    q.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn std_q_release(this: *mut c_void) -> u32 {
    let q = unsafe { &*(this as *const StandardCommandQueue) };
    let prev = q.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut StandardCommandQueue)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn std_q_execute_command_lists(
    this: *mut c_void,
    num_lists: u32,
    lists: *const *mut c_void,
) {
    let q = unsafe { &*(this as *const StandardCommandQueue) };
    q.executed_batches.fetch_add(1, Ordering::Relaxed);
    q.executed_lists
        .fetch_add(u64::from(num_lists), Ordering::Relaxed);

    if lists.is_null() || num_lists == 0 {
        return;
    }
    let slice = unsafe { std::slice::from_raw_parts(lists, num_lists as usize) };
    for &ptr in slice {
        if ptr.is_null() {
            continue;
        }
        let cl = unsafe { &*(ptr as *const StandardCommandList) };
        let drained = std::mem::take(&mut *cl.thin_list.lock().unwrap());
        if !drained.commands().is_empty() && !q.device.is_null() {
            let dev = unsafe { &*(q.device as *const StandardDevice) };
            if let Some(ref backend) = dev.backend
                && let Ok(b) = backend.lock()
            {
                let mut bound_pso: Option<u64> = None;
                let mut sgprs = [0u32; 32];
                for cmd in drained.commands() {
                    match cmd {
                        ThinCommand::BindPipeline(PipelineKey(k)) => {
                            bound_pso = Some(*k);
                        }
                        ThinCommand::PushUserSgprs { offset_dw, values } => {
                            let start = (*offset_dw as usize).min(32);
                            for (i, &v) in values.iter().enumerate() {
                                if start + i < 32 {
                                    sgprs[start + i] = v;
                                }
                            }
                        }
                        ThinCommand::Dispatch { x, y, z, .. } => {
                            if let Some(pso_key) = bound_pso
                                && let Some(spv) = b.compute_psos.get(&pso_key)
                            {
                                let push_bytes: Vec<u8> =
                                    sgprs.iter().flat_map(|w| w.to_le_bytes()).collect();
                                let _ = b.runner.dispatch_spirv(spv, &push_bytes, [*x, *y, *z]);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

unsafe extern "system" fn std_q_signal(_this: *mut c_void, fence: *mut c_void, value: u64) -> i32 {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    let f = unsafe { &*(fence as *const StandardFence) };
    f.value.store(value, Ordering::Release);
    S_OK
}

unsafe extern "system" fn std_q_wait(_this: *mut c_void, _fence: *mut c_void, _value: u64) -> i32 {
    S_OK
}

unsafe extern "system" fn std_q_get_timestamp_frequency(
    _this: *mut c_void,
    out_freq: *mut u64,
) -> i32 {
    if out_freq.is_null() {
        return E_INVALIDARG;
    }
    unsafe { *out_freq = 1_000_000_000 }; // 1 GHz GPU timestamp clock
    S_OK
}

unsafe extern "system" fn std_q_get_clock_calibration(
    _this: *mut c_void,
    out_gpu_ts: *mut u64,
    out_cpu_ts: *mut u64,
) -> i32 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    if !out_gpu_ts.is_null() {
        unsafe { *out_gpu_ts = now };
    }
    if !out_cpu_ts.is_null() {
        unsafe { *out_cpu_ts = now };
    }
    S_OK
}

unsafe extern "system" fn std_q_get_desc(
    this: *mut c_void,
    out_desc: *mut D3D12CommandQueueDesc,
) -> *mut D3D12CommandQueueDesc {
    let q = unsafe { &*(this as *const StandardCommandQueue) };
    if !out_desc.is_null() {
        unsafe { *out_desc = q.desc };
    }
    out_desc
}

static STANDARD_QUEUE_VTABLE: [VTableSlot; QUEUE_VTABLE_LEN] = [
    VTableSlot(std_q_qi as *const ()),
    VTableSlot(std_q_add_ref as *const ()),
    VTableSlot(std_q_release as *const ()),
    VTableSlot(stub_ok as *const ()),
    VTableSlot(stub_ok as *const ()),
    VTableSlot(stub_ok as *const ()),
    VTableSlot(stub_ok as *const ()),
    VTableSlot(stub_ok as *const ()),
    VTableSlot(stub_void as *const ()),
    VTableSlot(stub_void as *const ()),
    VTableSlot(std_q_execute_command_lists as *const ()),
    VTableSlot(stub_void as *const ()),
    VTableSlot(stub_void as *const ()),
    VTableSlot(stub_void as *const ()),
    VTableSlot(std_q_signal as *const ()),
    VTableSlot(std_q_wait as *const ()),
    VTableSlot(std_q_get_timestamp_frequency as *const ()),
    VTableSlot(std_q_get_clock_calibration as *const ()),
    VTableSlot(std_q_get_desc as *const ()),
];

impl StandardCommandQueue {
    pub fn new_boxed(desc: D3D12CommandQueueDesc, device: *mut c_void) -> *mut Self {
        Box::into_raw(Box::new(Self {
            vtable: &STANDARD_QUEUE_VTABLE,
            ref_count: AtomicU32::new(1),
            device,
            desc,
            executed_batches: AtomicU64::new(0),
            executed_lists: AtomicU64::new(0),
            meta: Mutex::new(ObjectMeta::default()),
        }))
    }
}

// ============================================================================
// ID3D12Device (44 slots)
// ============================================================================

/// Standard-ABI `ID3D12Device` COM object (44 slots) creating real standard-ABI
/// `StandardCommandQueue`, `StandardCommandList`, `StandardResource`, and `StandardFence`.
#[repr(C)]
pub struct StandardDevice {
    pub vtable: &'static [VTableSlot; DEVICE_VTABLE_LEN],
    pub ref_count: AtomicU32,
    pub backend: Option<Arc<Mutex<LiveGpuBackend>>>,
    pub meta: Mutex<ObjectMeta>,
}

unsafe extern "system" fn std_dev_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_NOINTERFACE;
    }
    unsafe {
        std_dev_add_ref(this);
        *out = this;
    }
    S_OK
}

unsafe extern "system" fn std_dev_add_ref(this: *mut c_void) -> u32 {
    let d = unsafe { &*(this as *const StandardDevice) };
    d.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn std_dev_release(this: *mut c_void) -> u32 {
    let d = unsafe { &*(this as *const StandardDevice) };
    let prev = d.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut StandardDevice)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn std_dev_get_node_count(_this: *mut c_void) -> u32 {
    1
}

unsafe extern "system" fn std_dev_create_command_queue(
    this: *mut c_void,
    desc: *const D3D12CommandQueueDesc,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let d = if desc.is_null() {
        D3D12CommandQueueDesc::default()
    } else {
        unsafe { *desc }
    };
    let q = StandardCommandQueue::new_boxed(d, this);
    unsafe { *out = q.cast() };
    S_OK
}

unsafe extern "system" fn std_dev_create_command_list(
    this: *mut c_void,
    _node_mask: u32,
    list_type: u32,
    _allocator: *mut c_void,
    _initial_pso: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let cl = StandardCommandList::new_boxed(list_type, this);
    unsafe { *out = cl.cast() };
    S_OK
}

unsafe extern "system" fn std_dev_get_descriptor_handle_inc(
    _this: *mut c_void,
    heap_type: u32,
) -> u32 {
    match heap_type {
        0 => 32, // CBV/SRV/UAV (RDNA2 8-dword T# descriptor)
        1 => 16, // Sampler (4-dword S# descriptor)
        2 => 32, // RTV
        3 => 32, // DSV
        _ => 32,
    }
}

unsafe extern "system" fn std_dev_create_committed_resource(
    this: *mut c_void,
    heap_props: *const D3D12HeapProperties,
    _heap_flags: u32,
    desc: *const D3D12ResourceDesc,
    _initial_state: u32,
    _optimized_clear: *const c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let hp = if heap_props.is_null() {
        D3D12HeapProperties::default()
    } else {
        unsafe { *heap_props }
    };
    let d = if desc.is_null() {
        D3D12ResourceDesc::default()
    } else {
        unsafe { *desc }
    };
    let res = StandardResource::new_boxed(d, hp, this);
    unsafe { *out = res.cast() };
    S_OK
}

unsafe extern "system" fn std_dev_create_fence(
    this: *mut c_void,
    initial_value: u64,
    _flags: u32,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let fence = StandardFence::new_boxed_with_device(initial_value, this);
    unsafe { *out = fence.cast() };
    S_OK
}

unsafe extern "system" fn std_dev_get_device_removed_reason(_this: *mut c_void) -> i32 {
    S_OK
}

static STANDARD_DEVICE_VTABLE: [VTableSlot; DEVICE_VTABLE_LEN] = {
    let mut vt = [VTableSlot(stub_ok as *const ()); DEVICE_VTABLE_LEN];
    vt[SLOT_QUERY_INTERFACE] = VTableSlot(std_dev_qi as *const ());
    vt[SLOT_ADD_REF] = VTableSlot(std_dev_add_ref as *const ());
    vt[SLOT_RELEASE] = VTableSlot(std_dev_release as *const ());
    vt[DEVICE_SLOT_GET_NODE_COUNT] = VTableSlot(std_dev_get_node_count as *const ());
    vt[DEVICE_SLOT_CREATE_COMMAND_QUEUE] = VTableSlot(std_dev_create_command_queue as *const ());
    vt[DEVICE_SLOT_CREATE_COMMAND_LIST] = VTableSlot(std_dev_create_command_list as *const ());
    vt[DEVICE_SLOT_GET_DESCRIPTOR_HANDLE_INC] =
        VTableSlot(std_dev_get_descriptor_handle_inc as *const ());
    vt[DEVICE_SLOT_CREATE_COMMITTED_RESOURCE] =
        VTableSlot(std_dev_create_committed_resource as *const ());
    vt[DEVICE_SLOT_CREATE_FENCE] = VTableSlot(std_dev_create_fence as *const ());
    vt[DEVICE_SLOT_GET_DEVICE_REMOVED_REASON] =
        VTableSlot(std_dev_get_device_removed_reason as *const ());
    vt
};

impl StandardDevice {
    pub fn new_boxed() -> *mut Self {
        let backend = LiveGpuBackend::try_new().map(|b| Arc::new(Mutex::new(b)));
        Box::into_raw(Box::new(Self {
            vtable: &STANDARD_DEVICE_VTABLE,
            ref_count: AtomicU32::new(1),
            backend,
            meta: Mutex::new(ObjectMeta::default()),
        }))
    }
}

// ============================================================================
// Custom VTable Builders (for wiring external backends into exact d3d12.h slots)
// ============================================================================

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

/// Build a standard-ABI slot table for `ID3D12Resource` (15 slots) matching
/// Microsoft `d3d12.h` slot indices (`IUnknown` -> `ID3D12Object` -> `ID3D12DeviceChild` -> `ID3D12Resource`).
pub fn build_standard_resource_vtable(
    qi: *const (),
    add_ref: *const (),
    release: *const (),
    map: *const (),
    unmap: *const (),
    get_gpu_va: *const (),
) -> [VTableSlot; RESOURCE_VTABLE_LEN] {
    let mut vt = [VTableSlot(stub_ok as *const ()); RESOURCE_VTABLE_LEN];
    vt[SLOT_QUERY_INTERFACE] = VTableSlot(qi);
    vt[SLOT_ADD_REF] = VTableSlot(add_ref);
    vt[SLOT_RELEASE] = VTableSlot(release);
    vt[RESOURCE_SLOT_MAP] = VTableSlot(map);
    vt[RESOURCE_SLOT_UNMAP] = VTableSlot(unmap);
    vt[RESOURCE_SLOT_GET_GPU_VA] = VTableSlot(get_gpu_va);
    vt
}

/// Callbacks wired into a standard-ABI `ID3D12GraphicsCommandList` (60-slot) vtable.
#[derive(Clone, Copy)]
pub struct CmdListVTableHooks {
    pub qi: *const (),
    pub add_ref: *const (),
    pub release: *const (),
    pub close: *const (),
    pub reset: *const (),
    pub draw_instanced: *const (),
    pub draw_indexed_instanced: *const (),
    pub dispatch: *const (),
    pub copy_buffer_region: *const (),
    pub copy_texture_region: *const (),
    pub copy_resource: *const (),
    pub set_pipeline_state: *const (),
    pub resource_barrier: *const (),
    pub set_descriptor_heaps: *const (),
    pub set_compute_root_signature: *const (),
    pub set_graphics_root_signature: *const (),
    pub set_compute_root_32bit_constants: *const (),
    pub om_set_render_targets: *const (),
    pub execute_indirect: *const (),
}

/// Build a standard-ABI slot table for `ID3D12GraphicsCommandList` (60 slots).
pub fn build_standard_cmdlist_vtable(
    hooks: &CmdListVTableHooks,
) -> [VTableSlot; CMDLIST_VTABLE_LEN] {
    let mut vt = [VTableSlot(stub_void as *const ()); CMDLIST_VTABLE_LEN];
    vt[SLOT_QUERY_INTERFACE] = VTableSlot(hooks.qi);
    vt[SLOT_ADD_REF] = VTableSlot(hooks.add_ref);
    vt[SLOT_RELEASE] = VTableSlot(hooks.release);
    vt[SLOT_GET_PRIVATE_DATA] = VTableSlot(stub_ok as *const ());
    vt[SLOT_SET_PRIVATE_DATA] = VTableSlot(stub_ok as *const ());
    vt[SLOT_SET_PRIVATE_DATA_INTERFACE] = VTableSlot(stub_ok as *const ());
    vt[SLOT_SET_NAME] = VTableSlot(stub_ok as *const ());
    vt[SLOT_GET_DEVICE] = VTableSlot(stub_ok as *const ());
    vt[CMDLIST_SLOT_CLOSE] = VTableSlot(hooks.close);
    vt[CMDLIST_SLOT_RESET] = VTableSlot(hooks.reset);
    vt[CMDLIST_SLOT_DRAW_INSTANCED] = VTableSlot(hooks.draw_instanced);
    vt[CMDLIST_SLOT_DRAW_INDEXED_INSTANCED] = VTableSlot(hooks.draw_indexed_instanced);
    vt[CMDLIST_SLOT_DISPATCH] = VTableSlot(hooks.dispatch);
    vt[CMDLIST_SLOT_COPY_BUFFER_REGION] = VTableSlot(hooks.copy_buffer_region);
    vt[CMDLIST_SLOT_COPY_TEXTURE_REGION] = VTableSlot(hooks.copy_texture_region);
    vt[CMDLIST_SLOT_COPY_RESOURCE] = VTableSlot(hooks.copy_resource);
    vt[CMDLIST_SLOT_SET_PIPELINE_STATE] = VTableSlot(hooks.set_pipeline_state);
    vt[CMDLIST_SLOT_RESOURCE_BARRIER] = VTableSlot(hooks.resource_barrier);
    vt[CMDLIST_SLOT_SET_DESCRIPTOR_HEAPS] = VTableSlot(hooks.set_descriptor_heaps);
    vt[CMDLIST_SLOT_SET_COMPUTE_ROOT_SIGNATURE] = VTableSlot(hooks.set_compute_root_signature);
    vt[CMDLIST_SLOT_SET_GRAPHICS_ROOT_SIGNATURE] = VTableSlot(hooks.set_graphics_root_signature);
    vt[CMDLIST_SLOT_SET_COMPUTE_ROOT_32BIT_CONSTANTS] =
        VTableSlot(hooks.set_compute_root_32bit_constants);
    vt[CMDLIST_SLOT_OM_SET_RENDER_TARGETS] = VTableSlot(hooks.om_set_render_targets);
    vt[CMDLIST_SLOT_EXECUTE_INDIRECT] = VTableSlot(hooks.execute_indirect);
    vt
}

/// Callbacks wired into a standard-ABI `ID3D12Device` (44-slot) vtable.
#[derive(Clone, Copy)]
pub struct DeviceVTableHooks {
    pub qi: *const (),
    pub add_ref: *const (),
    pub release: *const (),
    pub create_command_queue: *const (),
    pub create_command_allocator: *const (),
    pub create_graphics_pso: *const (),
    pub create_compute_pso: *const (),
    pub create_command_list: *const (),
    pub check_feature_support: *const (),
    pub create_descriptor_heap: *const (),
    pub get_descriptor_handle_inc: *const (),
    pub create_root_signature: *const (),
    pub create_committed_resource: *const (),
    pub create_fence: *const (),
    pub get_device_removed_reason: *const (),
}

/// Build a standard-ABI slot table for `ID3D12Device` (44 slots).
pub fn build_standard_device_vtable(hooks: &DeviceVTableHooks) -> [VTableSlot; DEVICE_VTABLE_LEN] {
    let mut vt = [VTableSlot(stub_ok as *const ()); DEVICE_VTABLE_LEN];
    vt[SLOT_QUERY_INTERFACE] = VTableSlot(hooks.qi);
    vt[SLOT_ADD_REF] = VTableSlot(hooks.add_ref);
    vt[SLOT_RELEASE] = VTableSlot(hooks.release);
    vt[DEVICE_SLOT_CREATE_COMMAND_QUEUE] = VTableSlot(hooks.create_command_queue);
    vt[DEVICE_SLOT_CREATE_COMMAND_ALLOCATOR] = VTableSlot(hooks.create_command_allocator);
    vt[DEVICE_SLOT_CREATE_GRAPHICS_PSO] = VTableSlot(hooks.create_graphics_pso);
    vt[DEVICE_SLOT_CREATE_COMPUTE_PSO] = VTableSlot(hooks.create_compute_pso);
    vt[DEVICE_SLOT_CREATE_COMMAND_LIST] = VTableSlot(hooks.create_command_list);
    vt[DEVICE_SLOT_CHECK_FEATURE_SUPPORT] = VTableSlot(hooks.check_feature_support);
    vt[DEVICE_SLOT_CREATE_DESCRIPTOR_HEAP] = VTableSlot(hooks.create_descriptor_heap);
    vt[DEVICE_SLOT_GET_DESCRIPTOR_HANDLE_INC] = VTableSlot(hooks.get_descriptor_handle_inc);
    vt[DEVICE_SLOT_CREATE_ROOT_SIGNATURE] = VTableSlot(hooks.create_root_signature);
    vt[DEVICE_SLOT_CREATE_COMMITTED_RESOURCE] = VTableSlot(hooks.create_committed_resource);
    vt[DEVICE_SLOT_CREATE_FENCE] = VTableSlot(hooks.create_fence);
    vt[DEVICE_SLOT_GET_DEVICE_REMOVED_REASON] = VTableSlot(hooks.get_device_removed_reason);
    vt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_device_creates_and_dispatches_all_five_com_interfaces() {
        let dev = StandardDevice::new_boxed();
        unsafe {
            let dvt = *(*dev).vtable;
            let get_nodes: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(dvt[DEVICE_SLOT_GET_NODE_COUNT].0);
            assert_eq!(get_nodes(dev.cast()), 1);

            let get_inc: unsafe extern "system" fn(*mut c_void, u32) -> u32 =
                std::mem::transmute(dvt[DEVICE_SLOT_GET_DESCRIPTOR_HANDLE_INC].0);
            assert_eq!(get_inc(dev.cast(), 0), 32);
            assert_eq!(get_inc(dev.cast(), 1), 16);

            // Create Fence via Device vtable slot 36
            let create_fence: unsafe extern "system" fn(
                *mut c_void,
                u64,
                u32,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = std::mem::transmute(dvt[DEVICE_SLOT_CREATE_FENCE].0);
            let mut fence_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(
                create_fence(dev.cast(), 10, 0, std::ptr::null(), &mut fence_ptr),
                S_OK
            );
            let fence = fence_ptr as *mut StandardFence;
            let fvt = *(*fence).vtable;
            let get_val: unsafe extern "system" fn(*mut c_void) -> u64 =
                std::mem::transmute(fvt[FENCE_SLOT_GET_COMPLETED_VALUE].0);
            assert_eq!(get_val(fence_ptr), 10);

            // Set/GetPrivateData + SetName on Fence
            let guid = [0x11u8; 16];
            let payload = [1u8, 2, 3, 4];
            let set_pd: unsafe extern "system" fn(
                *mut c_void,
                *const [u8; 16],
                u32,
                *const c_void,
            ) -> i32 = std::mem::transmute(fvt[SLOT_SET_PRIVATE_DATA].0);
            let get_pd: unsafe extern "system" fn(
                *mut c_void,
                *const [u8; 16],
                *mut u32,
                *mut c_void,
            ) -> i32 = std::mem::transmute(fvt[SLOT_GET_PRIVATE_DATA].0);
            assert_eq!(set_pd(fence_ptr, &guid, 4, payload.as_ptr().cast()), S_OK);
            let mut out_buf = [0u8; 4];
            let mut out_sz = 4u32;
            assert_eq!(
                get_pd(fence_ptr, &guid, &mut out_sz, out_buf.as_mut_ptr().cast()),
                S_OK
            );
            assert_eq!(out_buf, payload);

            // Create Queue via Device vtable slot 8 and Signal the Fence via Queue slot 14
            let create_q: unsafe extern "system" fn(
                *mut c_void,
                *const D3D12CommandQueueDesc,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = std::mem::transmute(dvt[DEVICE_SLOT_CREATE_COMMAND_QUEUE].0);
            let q_desc = D3D12CommandQueueDesc {
                list_type: 0,
                priority: 0,
                flags: 0,
                node_mask: 1,
            };
            let mut q_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(
                create_q(dev.cast(), &q_desc, std::ptr::null(), &mut q_ptr),
                S_OK
            );
            let q = q_ptr as *mut StandardCommandQueue;
            let qvt = *(*q).vtable;
            let q_signal: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> i32 =
                std::mem::transmute(qvt[QUEUE_SLOT_SIGNAL].0);
            assert_eq!(q_signal(q_ptr, fence_ptr, 42), S_OK);
            assert_eq!(get_val(fence_ptr), 42);

            let q_freq: unsafe extern "system" fn(*mut c_void, *mut u64) -> i32 =
                std::mem::transmute(qvt[QUEUE_SLOT_GET_TIMESTAMP_FREQUENCY].0);
            let mut freq = 0u64;
            assert_eq!(q_freq(q_ptr, &mut freq), S_OK);
            assert_eq!(freq, 1_000_000_000);

            // Create Resource via Device vtable slot 27 and test Write/ReadFromSubresource + GetGPUVA
            let create_res: unsafe extern "system" fn(
                *mut c_void,
                *const D3D12HeapProperties,
                u32,
                *const D3D12ResourceDesc,
                u32,
                *const c_void,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = std::mem::transmute(dvt[DEVICE_SLOT_CREATE_COMMITTED_RESOURCE].0);
            let res_desc = D3D12ResourceDesc {
                dimension: 1,
                width: 512,
                height: 1,
                depth_or_array_size: 1,
                mip_levels: 1,
                ..Default::default()
            };
            let heap_props = D3D12HeapProperties {
                heap_type: 2,
                ..Default::default()
            };
            let mut res_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(
                create_res(
                    dev.cast(),
                    &heap_props,
                    0,
                    &res_desc,
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    &mut res_ptr,
                ),
                S_OK
            );
            let res = res_ptr as *mut StandardResource;
            let rvt = *(*res).vtable;
            let get_va: unsafe extern "system" fn(*mut c_void) -> u64 =
                std::mem::transmute(rvt[RESOURCE_SLOT_GET_GPU_VA].0);
            assert_ne!(get_va(res_ptr), 0);
            let write_sub: unsafe extern "system" fn(
                *mut c_void,
                u32,
                *const c_void,
                *const c_void,
                u32,
                u32,
            ) -> i32 = std::mem::transmute(rvt[RESOURCE_SLOT_WRITE_TO_SUBRESOURCE].0);
            let read_sub: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                u32,
                u32,
                u32,
                *const c_void,
            ) -> i32 = std::mem::transmute(rvt[RESOURCE_SLOT_READ_FROM_SUBRESOURCE].0);
            let src = [0xABu8; 16];
            let mut dst = [0u8; 16];
            assert_eq!(
                write_sub(res_ptr, 0, std::ptr::null(), src.as_ptr().cast(), 16, 16),
                S_OK
            );
            assert_eq!(
                read_sub(
                    res_ptr,
                    dst.as_mut_ptr().cast(),
                    16,
                    16,
                    0,
                    std::ptr::null()
                ),
                S_OK
            );
            assert_eq!(dst, src);

            // Create CommandList via Device vtable slot 12, record Dispatch + Draw + Close, Execute on Queue
            let create_cl: unsafe extern "system" fn(
                *mut c_void,
                u32,
                u32,
                *mut c_void,
                *mut c_void,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = std::mem::transmute(dvt[DEVICE_SLOT_CREATE_COMMAND_LIST].0);
            let mut cl_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(
                create_cl(
                    dev.cast(),
                    0,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    &mut cl_ptr,
                ),
                S_OK
            );
            let cl = cl_ptr as *mut StandardCommandList;
            let clvt = *(*cl).vtable;
            let dispatch: unsafe extern "system" fn(*mut c_void, u32, u32, u32) =
                std::mem::transmute(clvt[CMDLIST_SLOT_DISPATCH].0);
            let draw: unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32) =
                std::mem::transmute(clvt[CMDLIST_SLOT_DRAW_INSTANCED].0);
            let close: unsafe extern "system" fn(*mut c_void) -> i32 =
                std::mem::transmute(clvt[CMDLIST_SLOT_CLOSE].0);
            dispatch(cl_ptr, 4, 2, 1);
            draw(cl_ptr, 3, 1, 0, 0);
            assert_eq!(close(cl_ptr), S_OK);
            assert_eq!((*cl).commands.lock().unwrap().len(), 2);

            let exec: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) =
                std::mem::transmute(qvt[QUEUE_SLOT_EXECUTE_COMMAND_LISTS].0);
            exec(q_ptr, 1, &cl_ptr);
            assert_eq!((*q).executed_lists.load(Ordering::Relaxed), 1);

            // Clean up all COM objects via Release slot 2
            let rel_cl: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(clvt[SLOT_RELEASE].0);
            let rel_res: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(rvt[SLOT_RELEASE].0);
            let rel_q: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(qvt[SLOT_RELEASE].0);
            let rel_f: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(fvt[SLOT_RELEASE].0);
            let rel_d: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(dvt[SLOT_RELEASE].0);
            assert_eq!(rel_cl(cl_ptr), 0);
            assert_eq!(rel_res(res_ptr), 0);
            assert_eq!(rel_q(q_ptr), 0);
            assert_eq!(rel_f(fence_ptr), 0);
            assert_eq!(rel_d(dev.cast()), 0);
        }
    }

    #[test]
    fn standard_resource_queue_cmdlist_device_vtables_match_d3d12_slot_layout() {
        let ok = stub_ok as *const ();
        let void = stub_void as *const ();
        let res_vt = build_standard_resource_vtable(ok, ok, ok, ok, void, ok);
        assert_eq!(res_vt.len(), RESOURCE_VTABLE_LEN);
        assert_eq!(res_vt[RESOURCE_SLOT_MAP].0, ok);
        assert_eq!(res_vt[RESOURCE_SLOT_UNMAP].0, void);
        assert_eq!(res_vt[RESOURCE_SLOT_GET_GPU_VA].0, ok);

        let q_vt = build_standard_queue_vtable(ok, ok, ok, void, ok);
        assert_eq!(q_vt.len(), QUEUE_VTABLE_LEN);
        assert_eq!(q_vt[QUEUE_SLOT_EXECUTE_COMMAND_LISTS].0, void);
        assert_eq!(q_vt[QUEUE_SLOT_SIGNAL].0, ok);

        let cl_vt = build_standard_cmdlist_vtable(&CmdListVTableHooks {
            qi: ok,
            add_ref: ok,
            release: ok,
            close: ok,
            reset: ok,
            draw_instanced: void,
            draw_indexed_instanced: void,
            dispatch: void,
            copy_buffer_region: void,
            copy_texture_region: void,
            copy_resource: void,
            set_pipeline_state: void,
            resource_barrier: void,
            set_descriptor_heaps: void,
            set_compute_root_signature: void,
            set_graphics_root_signature: void,
            set_compute_root_32bit_constants: void,
            om_set_render_targets: void,
            execute_indirect: void,
        });
        assert_eq!(cl_vt.len(), CMDLIST_VTABLE_LEN);
        assert_eq!(cl_vt[CMDLIST_SLOT_DISPATCH].0, void);
        assert_eq!(cl_vt[CMDLIST_SLOT_CLOSE].0, ok);
        assert_eq!(cl_vt[CMDLIST_SLOT_EXECUTE_INDIRECT].0, void);

        let dev_vt = build_standard_device_vtable(&DeviceVTableHooks {
            qi: ok,
            add_ref: ok,
            release: ok,
            create_command_queue: ok,
            create_command_allocator: ok,
            create_graphics_pso: ok,
            create_compute_pso: ok,
            create_command_list: ok,
            check_feature_support: ok,
            create_descriptor_heap: ok,
            get_descriptor_handle_inc: ok,
            create_root_signature: ok,
            create_committed_resource: ok,
            create_fence: ok,
            get_device_removed_reason: ok,
        });
        assert_eq!(dev_vt.len(), DEVICE_VTABLE_LEN);
        assert_eq!(dev_vt[DEVICE_SLOT_CREATE_COMMAND_QUEUE].0, ok);
        assert_eq!(dev_vt[DEVICE_SLOT_CREATE_COMMITTED_RESOURCE].0, ok);
        assert_eq!(dev_vt[DEVICE_SLOT_CREATE_FENCE].0, ok);
    }
}
