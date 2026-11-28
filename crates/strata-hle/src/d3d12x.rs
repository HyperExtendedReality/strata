//! Clean-room COM vtable objects for `d3d12_xs.dll` (`ID3D12Device`, `ID3D12CommandQueue`,
//! `ID3D12GraphicsCommandList`) translating D3D12.X calls directly into Strata's
//! No-API `ThinCommandList` + `UmaManager` + Vulkan 1.3 BDA `ComputeRunner`.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use strata_gpu::async_submit::ThinTranslationPool;
use strata_gpu::graphics::{DrawArgs, GraphicsPsoState};
use strata_gpu::images::ImageBindings;
use strata_gpu::indirect_upload::{PipelineLibrary, UploadQueue};
use strata_gpu::present::FramePacingClock;
use strata_gpu::runner::ComputeRunner;
use strata_gpu::texstore::TextureStore;
use strata_gpu::texture::TextureCache;
use strata_gpu::thin_list::{PipelineKey, RenderTargetDesc, ThinCommand, ThinCommandList};
use strata_gpu::timeline::{SpanTrack, TimelineProfiler};
use strata_gpu::uma::UmaManager;
use strata_runtime_shaders::CLAMP_INDIRECT_SPV;
use strata_shader::descriptor::ImageDescriptor;
use strata_shader::resource::{
    DescriptorUse, ImageBinding, Origin, ResourceKind, planned_image_bindings, trace_descriptors,
};
use strata_shader::ssa::{BuildOptions, Stage};

const S_OK: i32 = 0;
const E_NOINTERFACE: i32 = 0x8000_4002u32 as i32;

// ============================================================================
// ID3D12ResourceX, ID3D12PipelineStateX, ID3D12FenceX COM Objects
// ============================================================================

#[repr(C)]
pub struct ResourceVTable {
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    pub map: unsafe extern "system" fn(*mut c_void, u32, *const c_void, *mut *mut c_void) -> i32,
    pub unmap: unsafe extern "system" fn(*mut c_void, u32, *const c_void),
    pub get_gpu_virtual_address: unsafe extern "system" fn(*mut c_void) -> u64,
}

#[repr(C)]
pub struct D3D12ResourceX {
    pub header: ComHeader<ResourceVTable>,
    pub gpu_va: u64,
    pub backing: Mutex<Vec<u8>>,
}

static RESOURCE_VTABLE: ResourceVTable = ResourceVTable {
    query_interface: res_qi,
    add_ref: res_add_ref,
    release: res_release,
    map: res_map,
    unmap: res_unmap,
    get_gpu_virtual_address: res_get_gpu_va,
};

impl D3D12ResourceX {
    pub fn new_boxed(size_bytes: usize, gpu_va: u64) -> *mut Self {
        let backing = vec![0u8; size_bytes.max(256)];
        let va = if gpu_va != 0 {
            gpu_va
        } else {
            backing.as_ptr() as u64
        };
        Box::into_raw(Box::new(Self {
            header: ComHeader {
                vtable: &RESOURCE_VTABLE,
                ref_count: AtomicU32::new(1),
            },
            gpu_va: va,
            backing: Mutex::new(backing),
        }))
    }
}

unsafe extern "system" fn res_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_NOINTERFACE;
    }
    unsafe {
        res_add_ref(this);
        *out = this;
    }
    S_OK
}

unsafe extern "system" fn res_add_ref(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const D3D12ResourceX) };
    obj.header.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn res_release(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const D3D12ResourceX) };
    let prev = obj.header.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut D3D12ResourceX)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn res_map(
    this: *mut c_void,
    _subresource: u32,
    _read_range: *const c_void,
    out_data: *mut *mut c_void,
) -> i32 {
    let obj = unsafe { &*(this as *const D3D12ResourceX) };
    if !out_data.is_null() {
        let mut guard = obj.backing.lock().unwrap();
        unsafe { *out_data = guard.as_mut_ptr() as *mut c_void };
    }
    S_OK
}

unsafe extern "system" fn res_unmap(
    _this: *mut c_void,
    _subresource: u32,
    _written_range: *const c_void,
) {
}

unsafe extern "system" fn res_get_gpu_va(this: *mut c_void) -> u64 {
    let obj = unsafe { &*(this as *const D3D12ResourceX) };
    obj.gpu_va
}

#[repr(C)]
pub struct FenceVTable {
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    pub get_completed_value: unsafe extern "system" fn(*mut c_void) -> u64,
    pub signal: unsafe extern "system" fn(*mut c_void, u64) -> i32,
}

#[repr(C)]
pub struct D3D12FenceX {
    pub header: ComHeader<FenceVTable>,
    pub value: std::sync::atomic::AtomicU64,
}

static FENCE_VTABLE: FenceVTable = FenceVTable {
    query_interface: fence_qi,
    add_ref: fence_add_ref,
    release: fence_release,
    get_completed_value: fence_get_completed,
    signal: fence_signal,
};

impl D3D12FenceX {
    pub fn new_boxed(initial_value: u64) -> *mut Self {
        Box::into_raw(Box::new(Self {
            header: ComHeader {
                vtable: &FENCE_VTABLE,
                ref_count: AtomicU32::new(1),
            },
            value: std::sync::atomic::AtomicU64::new(initial_value),
        }))
    }
}

unsafe extern "system" fn fence_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_NOINTERFACE;
    }
    unsafe {
        fence_add_ref(this);
        *out = this;
    }
    S_OK
}

unsafe extern "system" fn fence_add_ref(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const D3D12FenceX) };
    obj.header.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn fence_release(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const D3D12FenceX) };
    let prev = obj.header.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut D3D12FenceX)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn fence_get_completed(this: *mut c_void) -> u64 {
    let obj = unsafe { &*(this as *const D3D12FenceX) };
    obj.value.load(Ordering::Acquire)
}

unsafe extern "system" fn fence_signal(this: *mut c_void, value: u64) -> i32 {
    let obj = unsafe { &*(this as *const D3D12FenceX) };
    obj.value.store(value, Ordering::Release);
    S_OK
}

/// Shared `IUnknown` header layout for all Strata COM objects.
#[repr(C)]
pub struct ComHeader<V: 'static> {
    pub vtable: &'static V,
    pub ref_count: AtomicU32,
}

// ============================================================================
// ID3D12GraphicsCommandListX COM VTable
// ============================================================================

#[repr(C)]
pub struct CommandListVTable {
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    pub close: unsafe extern "system" fn(*mut c_void) -> i32,
    pub reset: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> i32,
    pub set_pipeline_state: unsafe extern "system" fn(*mut c_void, u64),
    pub set_compute_root_32bit_constants:
        unsafe extern "system" fn(*mut c_void, u32, u32, *const u32, u32),
    pub dispatch: unsafe extern "system" fn(*mut c_void, u32, u32, u32),
    pub draw_instanced: unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32),
    pub draw_indexed_instanced: unsafe extern "system" fn(*mut c_void, u32, u32, u32, i32, u32),
    pub draw_indirect: unsafe extern "system" fn(*mut c_void, u64, u32, u32),
    pub om_set_render_targets: unsafe extern "system" fn(*mut c_void, u32, *const RenderTargetDesc),
    pub resource_barrier: unsafe extern "system" fn(*mut c_void, u64, u32, u32),
    pub copy_buffer_region: unsafe extern "system" fn(*mut c_void, u64, u64, u64, u64, u64),
    pub dispatch_indirect: unsafe extern "system" fn(*mut c_void, u64),
}

#[repr(C)]
pub struct D3D12CommandListX {
    pub header: ComHeader<CommandListVTable>,
    pub list: Mutex<ThinCommandList>,
    pub closed: AtomicU32,
}

static CMD_LIST_VTABLE: CommandListVTable = CommandListVTable {
    query_interface: list_qi,
    add_ref: list_add_ref,
    release: list_release,
    close: list_close,
    reset: list_reset,
    set_pipeline_state: list_set_pipeline_state,
    set_compute_root_32bit_constants: list_set_compute_root_32bit_constants,
    dispatch: list_dispatch,
    draw_instanced: list_draw_instanced,
    draw_indexed_instanced: list_draw_indexed_instanced,
    draw_indirect: list_draw_indirect,
    om_set_render_targets: list_om_set_render_targets,
    resource_barrier: list_resource_barrier,
    copy_buffer_region: list_copy_buffer_region,
    dispatch_indirect: list_dispatch_indirect,
};

impl D3D12CommandListX {
    pub fn new_boxed() -> *mut Self {
        Box::into_raw(Box::new(Self {
            header: ComHeader {
                vtable: &CMD_LIST_VTABLE,
                ref_count: AtomicU32::new(1),
            },
            list: Mutex::new(ThinCommandList::new()),
            closed: AtomicU32::new(0),
        }))
    }
}

unsafe extern "system" fn list_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_NOINTERFACE;
    }
    unsafe {
        list_add_ref(this);
        *out = this;
    }
    S_OK
}

unsafe extern "system" fn list_add_ref(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    obj.header.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn list_release(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    let prev = obj.header.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut D3D12CommandListX)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn list_close(this: *mut c_void) -> i32 {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    obj.closed.store(1, Ordering::Release);
    S_OK
}

unsafe extern "system" fn list_reset(
    this: *mut c_void,
    _alloc: *mut c_void,
    _pso: *mut c_void,
) -> i32 {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    obj.list.lock().unwrap().reset();
    obj.closed.store(0, Ordering::Release);
    S_OK
}

unsafe extern "system" fn list_set_pipeline_state(this: *mut c_void, pso_key: u64) {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    obj.list.lock().unwrap().bind_pipeline(PipelineKey(pso_key));
}

unsafe extern "system" fn list_set_compute_root_32bit_constants(
    this: *mut c_void,
    _root_idx: u32,
    num_32bit_values: u32,
    src_data: *const u32,
    dest_offset_in_32bit_values: u32,
) {
    if src_data.is_null() || num_32bit_values == 0 {
        return;
    }
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    let vals = unsafe { std::slice::from_raw_parts(src_data, num_32bit_values as usize) };
    obj.list
        .lock()
        .unwrap()
        .push_sgprs(dest_offset_in_32bit_values as u8, vals);
}

unsafe extern "system" fn list_dispatch(this: *mut c_void, x: u32, y: u32, z: u32) {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    obj.list
        .lock()
        .unwrap()
        .dispatch(x, y, z, [64, 1, 1], false);
}

unsafe extern "system" fn list_dispatch_indirect(this: *mut c_void, args_va: u64) {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    obj.list.lock().unwrap().dispatch_indirect(args_va);
}

/// Whether `len` bytes at `addr` are committed, readable memory.
pub(crate) fn readable(addr: u64, len: usize) -> bool {
    #[repr(C)]
    struct Mbi {
        base: usize,
        alloc_base: usize,
        alloc_protect: u32,
        partition: u16,
        region_size: usize,
        state: u32,
        protect: u32,
        kind: u32,
    }
    unsafe extern "system" {
        fn VirtualQuery(addr: *const c_void, info: *mut Mbi, len: usize) -> usize;
    }
    const MEM_COMMIT: u32 = 0x1000;
    const NO_READ: u32 = 0x01 | 0x100; // PAGE_NOACCESS | PAGE_GUARD
    const WRITABLE: u32 = 0x04 | 0x08 | 0x40 | 0x80;
    if addr == 0 {
        return false;
    }
    let cached = IN_SUBMISSION.with(std::cell::Cell::get);
    if cached
        && MAPPED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(Default::default)
            .contains(addr, len as u64, false)
    {
        return true;
    }
    let mut info = std::mem::MaybeUninit::<Mbi>::zeroed();
    // SAFETY: a query only.
    let n = unsafe {
        VirtualQuery(
            addr as *const c_void,
            info.as_mut_ptr(),
            std::mem::size_of::<Mbi>(),
        )
    };
    if n == 0 {
        return false;
    }
    // SAFETY: filled by VirtualQuery.
    let info = unsafe { info.assume_init() };
    let ok = info.state == MEM_COMMIT && info.protect & NO_READ == 0;
    if ok
        && cached
        && let Some(c) = MAPPED.lock().unwrap_or_else(|e| e.into_inner()).as_mut()
    {
        c.insert(
            info.base as u64,
            info.region_size as u64,
            info.protect & WRITABLE != 0,
        );
    }
    ok && (addr as usize)
        .checked_add(len)
        .is_some_and(|end| end <= info.base.saturating_add(info.region_size))
}

/// The three group counts of an indirect dispatch, from console memory.
fn indirect_args(
    uma: Option<&Mutex<UmaManager>>,
    runner: &ComputeRunner,
    args_va: u64,
) -> Option<[u32; 3]> {
    let host = match uma {
        Some(uma) => {
            let uma = uma.lock().unwrap();
            // A mirror's VRAM copy may hold what a shader wrote: bring it back first.
            let _ = uma.readback(runner, args_va, 12);
            uma.host_ptr(args_va).map(|p| p as u64).unwrap_or(args_va)
        }
        None => args_va,
    };
    if !readable(host, 12) {
        return None;
    }
    // SAFETY: checked readable above.
    Some(unsafe { std::ptr::read_unaligned(host as *const [u32; 3]) })
}

unsafe extern "system" fn list_draw_instanced(
    this: *mut c_void,
    vertex_count: u32,
    instance_count: u32,
    start_vertex: u32,
    start_instance: u32,
) {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    obj.list.lock().unwrap().draw_instanced(
        vertex_count,
        instance_count,
        start_vertex,
        start_instance,
    );
}

unsafe extern "system" fn list_draw_indexed_instanced(
    this: *mut c_void,
    index_count: u32,
    instance_count: u32,
    start_index: u32,
    base_vertex: i32,
    start_instance: u32,
) {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    obj.list.lock().unwrap().draw_indexed_instanced(
        index_count,
        instance_count,
        start_index,
        base_vertex,
        start_instance,
    );
}

unsafe extern "system" fn list_draw_indirect(
    this: *mut c_void,
    args_bda: u64,
    max_verts: u32,
    max_inst: u32,
) {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    obj.list
        .lock()
        .unwrap()
        .draw_indirect(args_bda, max_verts, max_inst);
}

unsafe extern "system" fn list_om_set_render_targets(
    this: *mut c_void,
    num_rts: u32,
    rts: *const RenderTargetDesc,
) {
    if rts.is_null() || num_rts == 0 {
        return;
    }
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    let slice = unsafe { std::slice::from_raw_parts(rts, num_rts as usize) };
    obj.list.lock().unwrap().set_render_targets(slice);
}

unsafe extern "system" fn list_resource_barrier(
    this: *mut c_void,
    resource_va: u64,
    state_before: u32,
    state_after: u32,
) {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    obj.list
        .lock()
        .unwrap()
        .resource_barrier(resource_va, state_before, state_after);
}

unsafe extern "system" fn list_copy_buffer_region(
    this: *mut c_void,
    dst_va: u64,
    dst_offset: u64,
    src_va: u64,
    src_offset: u64,
    num_bytes: u64,
) {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    obj.list
        .lock()
        .unwrap()
        .copy_buffer_region(dst_va, dst_offset, src_va, src_offset, num_bytes);
}

// ============================================================================
// ID3D12CommandQueueX COM VTable + the execution engine
// ============================================================================

#[repr(C)]
pub struct CommandQueueVTable {
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    pub execute_command_lists: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void),
    pub present_x: unsafe extern "system" fn(*mut c_void, u32, *const c_void, *const c_void) -> i32,
}

/// Graphics pipelines and textures, shared by every queue of a device.
pub type PipelineTable = Arc<Mutex<HashMap<PipelineKey, PipelineEntry>>>;
pub type SharedTextures = Arc<Mutex<TextureCache>>;
/// A device's residency bookkeeping (shared by its queues).
pub type SharedResidency = Arc<Mutex<strata_gpu::residency::ResidencyManager>>;

/// Console memory handed to the prefetch worker thread.
struct SharedUma(Arc<Mutex<UmaManager>>);

// SAFETY: the UMA's raw host pointers are only used behind its mutex, as
// every queue of a device already does from its own thread.
unsafe impl Send for SharedUma {}
unsafe impl Sync for SharedUma {}

impl SharedUma {
    // A method (not a field access) so closures capture the whole wrapper.
    fn get(&self) -> &Mutex<UmaManager> {
        &self.0
    }
}

/// T#s written by `CreateShaderResourceView` since the last submission:
/// read ahead by the texture cache's heap worker before the draws bind them.
static SRV_PREFETCH: Mutex<Vec<[u32; 8]>> = Mutex::new(Vec::new());

/// Queue a new SRV's T# for ahead-of-bind reading (bounded; duplicates of
/// the previous entry are dropped).
pub(crate) fn queue_srv_prefetch(words: [u32; 8]) {
    let mut q = SRV_PREFETCH.lock().unwrap_or_else(|e| e.into_inner());
    if q.len() < 4096 && q.last() != Some(&words) {
        q.push(words);
    }
}

/// `VirtualQuery` results, valid until the next submission or title
/// memory release ([`invalidate_mapped_ranges`]).
static MAPPED: Mutex<Option<strata_gpu::mapped_cache::MappedRangeCache>> = Mutex::new(None);

thread_local! {
    /// Whether this thread is executing a submission ([`MappedScope`]).
    static IN_SUBMISSION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While alive, [`readable`] on this thread trusts and fills [`MAPPED`].
/// Outside a submission every check asks the OS, so a release Strata does
/// not hook (a title's own `VirtualFree`) cannot leave a stale answer.
struct MappedScope(bool);

impl MappedScope {
    fn enter() -> Self {
        let outer = IN_SUBMISSION.with(|f| f.replace(true));
        if !outer {
            invalidate_mapped_ranges();
        }
        Self(outer)
    }
}

impl Drop for MappedScope {
    fn drop(&mut self) {
        IN_SUBMISSION.with(|f| f.set(self.0));
    }
}

/// Forget cached `VirtualQuery` spans (title memory was freed or re-protected,
/// or a new submission starts).
pub(crate) fn invalidate_mapped_ranges() {
    if let Some(c) = MAPPED.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        c.bump_generation();
    }
}

#[repr(C)]
pub struct D3D12CommandQueueX {
    pub header: ComHeader<CommandQueueVTable>,
    pub submitted_lists: AtomicU32,
    pub submitted_commands: AtomicU32,
    pub hoisted_clamps: AtomicU32,
    pub executed_gpu_dispatches: AtomicU32,
    /// Lists run by parallel command-buffer recording.
    pub parallel_lists: AtomicU32,
    pub frame_clock: Mutex<FramePacingClock>,
    /// Optional live Vulkan 1.3 BDA runner + compiled SPIR-V pipeline table.
    pub gpu_runner: Option<Arc<ComputeRunner>>,
    pub pipelines: PipelineTable,
    /// Vulkan images for the T#s / S#s translated shaders use.
    pub textures: SharedTextures,
    /// Render targets by console address, described like textures.
    pub render_targets: Mutex<HashMap<u64, [u32; 8]>>,
    /// `RECTLIST` geometry shaders by the vertex shader's parameter locations.
    rect_gs: Mutex<HashMap<Vec<u32>, std::sync::Arc<Vec<u32>>>>,
    pub executed_draws: AtomicU32,
    /// Draws and dispatches `SetPredication` skipped.
    pub predicated_skips: AtomicU32,
    /// Console memory: mirrors are synced before each list and read back
    /// where a list asks for it.
    pub uma: Option<Arc<Mutex<UmaManager>>>,
    pub synced_bytes: std::sync::atomic::AtomicU64,
    pub readback_bytes: std::sync::atomic::AtomicU64,
    /// GPU-visible scratch memory (spilled root arguments, vertex-buffer
    /// tables): `(console VA, BDA, size)`. Draws are synchronous, so one
    /// block is reused by every draw.
    scratch: Mutex<Option<(u64, u64, u64)>>,
    /// The origin of `WriteTimestamp` values (1 GHz, like
    /// `GetTimestampFrequency` reports).
    epoch: std::time::Instant,
    /// Messages already logged once (unsupported paths).
    logged: Mutex<std::collections::HashSet<&'static str>>,
    /// Parallel thin-list flattener for multi-list `ExecuteCommandLists` batches.
    pub translation_pool: ThinTranslationPool,
    /// Dedicated copy/upload staging queue.
    pub upload_queue: Mutex<UploadQueue>,
    /// Persistent `ID3D12PipelineLibrary` store.
    pub pipeline_library: Mutex<PipelineLibrary>,
    /// Optional content-addressed on-disk texture store.
    pub texture_store: Mutex<Option<TextureStore>>,
    /// CPU/GPU timeline profiler (Chrome Trace JSON export).
    pub timeline: Mutex<TimelineProfiler>,
    /// VRAM residency (`ID3D12Device::MakeResident` / `Evict`), shared by
    /// every queue of a device.
    pub residency: SharedResidency,
    /// Packed 64 KiB slab sub-allocator for small indirect/budget buffers.
    pub budget_slabs: Mutex<strata_gpu::memory_stack::BudgetSlabPacker>,
    /// Hardware acceleration structure tracker (`BuildAccel` / `CopyAccel`).
    pub accel_structs: Mutex<strata_gpu::raytracing::AccelStructManager>,
}

static CMD_QUEUE_VTABLE: CommandQueueVTable = CommandQueueVTable {
    query_interface: queue_qi,
    add_ref: queue_add_ref,
    release: queue_release,
    execute_command_lists: queue_execute_command_lists,
    present_x: queue_present_x,
};

impl D3D12CommandQueueX {
    pub fn new_boxed() -> *mut Self {
        Self::new_boxed_with_runner(None)
    }

    pub fn new_boxed_with_runner(gpu_runner: Option<Arc<ComputeRunner>>) -> *mut Self {
        Self::new_boxed_shared(
            gpu_runner,
            None,
            Arc::new(Mutex::new(HashMap::new())),
            Arc::new(Mutex::new(TextureCache::new())),
        )
    }

    /// As [`Self::new_boxed_with_runner`], with console memory attached.
    pub fn new_boxed_with_memory(
        gpu_runner: Arc<ComputeRunner>,
        uma: Arc<Mutex<UmaManager>>,
    ) -> *mut Self {
        Self::new_boxed_shared(
            Some(gpu_runner),
            Some(uma),
            Arc::new(Mutex::new(HashMap::new())),
            Arc::new(Mutex::new(TextureCache::new())),
        )
    }

    /// An engine sharing a device's pipelines and textures (every queue of
    /// a `ID3D12Device` sees the PSOs and images the others made).
    pub fn new_boxed_shared(
        gpu_runner: Option<Arc<ComputeRunner>>,
        uma: Option<Arc<Mutex<UmaManager>>>,
        pipelines: PipelineTable,
        textures: SharedTextures,
    ) -> *mut Self {
        // With console memory attached, writes to shadowed (write-watched)
        // mirrors are logged: clean textures there bind without a re-hash.
        // `STRATA_TEXTURE_DIRTY_TRACKING=0` always re-hashes.
        if uma.is_some()
            && std::env::var("STRATA_TEXTURE_DIRTY_TRACKING").map_or(true, |v| v.trim() != "0")
        {
            let mut t = textures.lock().unwrap();
            t.set_dirty_tracking(true);
            // Read new SRVs' title memory on a worker before draws bind them.
            if let Some(uma) = &uma
                && !t.prefetch_enabled()
                && std::env::var("STRATA_TEXTURE_PREFETCH").map_or(true, |v| v.trim() != "0")
            {
                let uma = SharedUma(Arc::clone(uma));
                t.enable_prefetch(Arc::new(move |a: u64, n: usize| {
                    let host = uma.get().lock().ok()?.host_ptr_any(a)?;
                    // SAFETY: checked readable for `n` bytes just before.
                    readable(host as u64, n)
                        .then(|| unsafe { std::slice::from_raw_parts(host, n) }.to_vec())
                }));
            }
        }
        let auto_store = std::env::var("STRATA_TEXTURE_STORE")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .and_then(|dir| {
                let path = std::path::PathBuf::from(dir);
                let s1 = TextureStore::open(&path).ok()?;
                if let Ok(s2) = TextureStore::open(&path) {
                    textures.lock().unwrap().attach_store(s2);
                }
                Some(s1)
            });
        Box::into_raw(Box::new(Self {
            header: ComHeader {
                vtable: &CMD_QUEUE_VTABLE,
                ref_count: AtomicU32::new(1),
            },
            submitted_lists: AtomicU32::new(0),
            submitted_commands: AtomicU32::new(0),
            hoisted_clamps: AtomicU32::new(0),
            executed_gpu_dispatches: AtomicU32::new(0),
            parallel_lists: AtomicU32::new(0),
            frame_clock: Mutex::new(FramePacingClock::new(60)),
            gpu_runner,
            pipelines,
            textures,
            render_targets: Mutex::new(HashMap::new()),
            rect_gs: Mutex::new(HashMap::new()),
            executed_draws: AtomicU32::new(0),
            predicated_skips: AtomicU32::new(0),
            uma,
            synced_bytes: std::sync::atomic::AtomicU64::new(0),
            readback_bytes: std::sync::atomic::AtomicU64::new(0),
            scratch: Mutex::new(None),
            epoch: std::time::Instant::now(),
            logged: Mutex::new(std::collections::HashSet::new()),
            translation_pool: ThinTranslationPool::new(
                std::thread::available_parallelism()
                    .map(|n| n.get().clamp(1, 8))
                    .unwrap_or(4),
            ),
            upload_queue: Mutex::new(UploadQueue::default()),
            pipeline_library: Mutex::new(PipelineLibrary::default()),
            texture_store: Mutex::new(auto_store),
            timeline: Mutex::new(TimelineProfiler::new()),
            residency: Arc::new(Mutex::new(strata_gpu::residency::ResidencyManager::new(
                4 << 30,
            ))),
            budget_slabs: Mutex::new(strata_gpu::memory_stack::BudgetSlabPacker::default()),
            accel_structs: Mutex::new(strata_gpu::raytracing::AccelStructManager::new()),
        }))
    }

    /// Opens an on-disk content-addressed texture store for converted textures.
    pub fn open_texture_store(&self, dir: &std::path::Path) -> std::io::Result<()> {
        let store = TextureStore::open(dir)?;
        *self.texture_store.lock().unwrap() = Some(store);
        // Texture binds check the store on a miss and fill it after detiling.
        self.textures
            .lock()
            .unwrap()
            .attach_store(TextureStore::open(dir)?);
        Ok(())
    }

    /// Exports all recorded queue spans as Chrome Trace JSON (`chrome://tracing`).
    pub fn timeline_chrome_trace(&self) -> String {
        self.timeline.lock().unwrap().to_chrome_trace_json()
    }

    pub(crate) fn prepare_submission(&self) {
        let prepare = strata_gpu::perf::start();
        let srvs = std::mem::take(&mut *SRV_PREFETCH.lock().unwrap_or_else(|e| e.into_inner()));
        if !srvs.is_empty() {
            let mut textures = self.textures.lock().unwrap();
            if textures.prefetch_enabled() {
                for words in srvs {
                    textures.prefetch(words);
                }
            }
        }
        // Drain any queued host-to-UMA uploads before running commands.
        {
            let mem = TitleMemory {
                uma: self.uma.as_deref(),
            };
            let t0 = self.timeline.lock().unwrap().now_us();
            let mut uq = self.upload_queue.lock().unwrap();
            let prev_bytes = uq.total_bytes_uploaded;
            uq.flush(|dst_va, data| {
                if mem.write(dst_va, data) {
                    strata_gpu::perf::upload_bytes(data.len() as u64);
                } else {
                    strata_gpu::perf::binding_errors(1);
                    crate::crash::log_line("queued upload: destination is unavailable");
                }
            });
            if uq.total_bytes_uploaded > prev_bytes {
                let mut tl = self.timeline.lock().unwrap();
                let dur = tl.now_us().saturating_sub(t0);
                tl.record_span("UploadQueue::flush", SpanTrack::UploadCopy, t0, dur);
            }
        }
        if let Some(prepare) = prepare {
            strata_gpu::perf::preparation_ns(prepare.elapsed().as_nanos() as u64);
        }
    }

    /// Executes a batch of closed command lists, translating multi-list batches
    /// in parallel via [`ThinTranslationPool`] and draining queued uploads first.
    pub fn execute_command_lists(&self, lists: Vec<ThinCommandList>) {
        if lists.is_empty() {
            return;
        }
        self.prepare_submission();
        if lists.len() == 1 {
            self.execute_thin(&lists[0]);
            return;
        }
        let prepare = strata_gpu::perf::start();
        let t0 = self.timeline.lock().unwrap().now_us();
        let batches = self.translation_pool.translate_parallel(lists);
        if let Some(prepare) = prepare {
            strata_gpu::perf::preparation_ns(prepare.elapsed().as_nanos() as u64);
        }
        {
            let mut tl = self.timeline.lock().unwrap();
            let dur = tl.now_us().saturating_sub(t0);
            tl.record_span(
                format!("ThinTranslationPool({})", batches.len()),
                SpanTrack::Cpu,
                t0,
                dur,
            );
        }
        let flattened: Vec<ThinCommandList> = batches
            .into_iter()
            .map(|batch| {
                let mut list = ThinCommandList::new();
                list.hoisted_clamps = batch.hoisted_clamps;
                for op in batch.ops {
                    list.push(op);
                }
                list
            })
            .collect();
        if self.execute_parallel(&flattened) {
            return;
        }
        for list in &flattened {
            self.execute_thin(list);
        }
    }

    /// Plans one list for [`Self::execute_parallel`]: its dispatches with
    /// their push constants, or `None` when it holds anything the serial
    /// replay does on the CPU between GPU commands (draws, copies, queries,
    /// predication, root/vertex tables, image bindings, indirect clamps).
    fn plan_compute_list(
        &self,
        runner: &ComputeRunner,
        pipes: &HashMap<PipelineKey, PipelineEntry>,
        list: &ThinCommandList,
    ) -> Option<Vec<(PipelineKey, Vec<u8>, [u32; 3])>> {
        if !list.hoisted_clamps.is_empty() {
            return None;
        }
        let mem = TitleMemory {
            uma: self.uma.as_deref(),
        };
        // A fresh list state: no root spill and no vertex table, so the
        // push constants are exactly the first 32 shadowed SGPRs.
        let st = DrawState::default();
        let mut active: Option<PipelineKey> = None;
        let mut shadow = [0u32; 256];
        let mut out = Vec::new();
        for cmd in list.commands() {
            match cmd {
                ThinCommand::BindPipeline(k) => active = Some(*k),
                ThinCommand::PushUserSgprs { offset_dw, values } => {
                    for (i, &v) in values.iter().enumerate() {
                        let idx = (*offset_dw as usize) + i;
                        if idx < shadow.len() {
                            shadow[idx] = v;
                        }
                    }
                }
                ThinCommand::Dispatch {
                    x,
                    y,
                    z,
                    is_thread_count,
                } => {
                    let groups = if *is_thread_count {
                        [(*x).div_ceil(64).max(1), (*y).max(1), (*z).max(1)]
                    } else {
                        [*x, *y, *z]
                    };
                    // Serially a dispatch with no known pipeline is skipped.
                    let Some((k, entry)) = active.and_then(|k| pipes.get(&k).map(|e| (k, e)))
                    else {
                        continue;
                    };
                    // Image bindings write back to title memory after each
                    // dispatch, and a pipeline that fails to build is
                    // skipped serially: both stay on the serial path.
                    if !entry.images.is_empty() || runner.pipeline_for(&entry.spirv, None).is_err()
                    {
                        return None;
                    }
                    let sgprs = self.push_sgprs(runner, &mem, &shadow, &st);
                    let push = sgprs.iter().flat_map(|w| w.to_le_bytes()).collect();
                    out.push((k, push, groups));
                }
                ThinCommand::ResourceBarrier { .. } => {}
                _ => return None,
            }
        }
        Some(out)
    }

    /// Runs a batch of compute-only lists by recording one Vulkan command
    /// buffer per worker in parallel and submitting them in list order.
    /// Returns `false` (having run nothing) when any list needs the serial
    /// replay; `STRATA_SERIAL_LISTS=1` forces that.
    fn execute_parallel(&self, lists: &[ThinCommandList]) -> bool {
        let Some(runner) = &self.gpu_runner else {
            return false;
        };
        if lists.len() < 2 || std::env::var_os("STRATA_SERIAL_LISTS").is_some() {
            return false;
        }
        let _mapped = MappedScope::enter();
        let pipes = self.pipelines.lock().unwrap();
        let mut plans = Vec::with_capacity(lists.len());
        for list in lists {
            match self.plan_compute_list(runner, &pipes, list) {
                Some(plan) => plans.push(plan),
                None => return false,
            }
        }
        // Serially every list syncs the title's CPU writes first; nothing
        // the title does lands between lists of one call, so once suffices.
        if let Some(uma) = &self.uma
            && let Ok(stats) = uma.lock().unwrap().sync_mirrors(runner)
        {
            self.synced_bytes.fetch_add(stats.bytes, Ordering::Relaxed);
        }
        for list in lists {
            crate::frame_capture::record(list);
            self.submitted_lists.fetch_add(1, Ordering::Relaxed);
            self.submitted_commands
                .fetch_add(list.commands().len() as u32, Ordering::Relaxed);
        }
        let jobs: Vec<Vec<strata_gpu::runner::ListDispatch<'_>>> = plans
            .iter()
            .map(|plan| {
                plan.iter()
                    .map(|(k, push, groups)| strata_gpu::runner::ListDispatch {
                        spirv: &pipes[k].spirv,
                        push,
                        groups: *groups,
                    })
                    .collect()
            })
            .collect();
        match runner.dispatch_lists_parallel(&jobs) {
            Ok(()) => {
                let n: usize = jobs.iter().map(Vec::len).sum();
                self.executed_gpu_dispatches
                    .fetch_add(n as u32, Ordering::Relaxed);
                self.parallel_lists
                    .fetch_add(lists.len() as u32, Ordering::Relaxed);
            }
            Err(e) => crate::crash::log_line(&format!("parallel list recording: {e}")),
        }
        drop(pipes);
        for list in lists {
            for cmd in list.commands() {
                if let ThinCommand::ResourceBarrier {
                    state_before,
                    state_after,
                    ..
                } = cmd
                {
                    runner.push_memory_barrier(*state_before, *state_after);
                }
            }
            let _ = runner.flush_barriers();
        }
        true
    }

    pub fn register_pipeline(&self, key: PipelineKey, spirv_words: Vec<u32>) {
        self.pipelines.lock().unwrap().insert(
            key,
            PipelineEntry {
                spirv: spirv_words,
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

    /// Translates a console vertex + pixel shader pair into a graphics
    /// pipeline. The pixel shader's images take the set-0 slots after the
    /// vertex shader's; it runs whole quads when the device has quad control.
    pub fn register_rdna2_graphics_pipeline(
        &self,
        key: PipelineKey,
        vs_words: &[u32],
        ps_words: &[u32],
    ) -> Result<(), strata_shader::CompileError> {
        let quad = self.gpu_runner.as_ref().is_some_and(|r| r.quad_control);
        let entry = translate_graphics(vs_words, ps_words, quad, None)?;
        self.pipelines.lock().unwrap().insert(key, entry);
        Ok(())
    }

    /// Describes the render target at `desc.base_address` (what
    /// `OMSetRenderTargets` names by address) as an image in title memory.
    pub fn register_render_target(&self, desc: ImageDescriptor) {
        self.render_targets
            .lock()
            .unwrap()
            .insert(desc.base_address, desc.encode());
    }

    /// Translates console RDNA2 code and registers it, with the descriptor
    /// trace that tells, at each dispatch, which T#/S# every image
    /// instruction reads.
    pub fn register_rdna2_pipeline(
        &self,
        key: PipelineKey,
        rdna2_words: &[u32],
        workgroup_size: [u32; 3],
    ) -> Result<(), strata_shader::CompileError> {
        let entry = translate_compute(rdna2_words, workgroup_size)?;
        self.pipelines.lock().unwrap().insert(key, entry);
        Ok(())
    }

    /// The GPU clock `WriteTimestamp` writes (ns since the engine started).
    pub fn timestamp_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    /// Logs `what` the first time only.
    fn log_once(&self, what: &'static str) {
        if self.logged.lock().unwrap().insert(what) {
            crate::crash::log_line(what);
        }
    }
}

/// A compute pipeline from console RDNA2 code, with its image trace.
pub fn translate_compute(
    rdna2_words: &[u32],
    workgroup_size: [u32; 3],
) -> Result<PipelineEntry, strata_shader::CompileError> {
    let compiled = strata_shader::compile_rdna2_to_spirv(rdna2_words, workgroup_size)?;
    let decoded = strata_shader::rdna2::decode_program(rdna2_words)?;
    let uses = trace_descriptors(&decoded);
    let images = planned_image_bindings(&decoded, &uses, 0, 0)
        .into_iter()
        .zip(uses)
        .filter_map(|(b, u)| b.map(|b| (u, b)))
        .collect();
    Ok(PipelineEntry {
        spirv: compiled.spirv,
        vertex: None,
        hull: None,
        domain: None,
        geometry: None,
        task: None,
        mesh: None,
        images,
        gfx: None,
    })
}

/// A graphics pipeline from a console vertex + pixel shader pair. The pixel
/// shader's images take the set-0 slots after the vertex shader's.
pub fn translate_graphics(
    vs_words: &[u32],
    ps_words: &[u32],
    quad_control: bool,
    state: Option<GraphicsPsoState>,
) -> Result<PipelineEntry, strata_shader::CompileError> {
    let vs_opts = BuildOptions {
        stage: Stage::Vertex,
        ..Default::default()
    };
    let vs = strata_shader::compile_rdna2_with(vs_words, [1, 1, 1], &vs_opts)?;
    let vs_decoded = strata_shader::rdna2::decode_program(vs_words)?;
    let vs_uses = trace_descriptors(&vs_decoded);
    let vs_bind = planned_image_bindings(&vs_decoded, &vs_uses, 0, 0);
    let next = |pick: fn(&ImageBinding) -> Option<u32>| {
        vs_bind
            .iter()
            .flatten()
            .filter_map(pick)
            .map(|s| s + 1)
            .max()
            .unwrap_or(0)
    };
    let tex_base = next(|b| match b {
        ImageBinding::Sampled(s) | ImageBinding::Fetched(s) => Some(*s),
        ImageBinding::Storage(_) => None,
    });
    let stor_base = next(|b| match b {
        ImageBinding::Storage(s) => Some(*s),
        _ => None,
    });
    let ps_opts = BuildOptions {
        stage: Stage::Pixel,
        texture_slot_base: tex_base,
        storage_slot_base: stor_base,
        require_full_quads: quad_control,
        ps_input_vgprs: None,
        tess_factor_srsrc: None,
    };
    let ps = strata_shader::compile_rdna2_with(ps_words, [1, 1, 1], &ps_opts)?;
    let ps_decoded = strata_shader::rdna2::decode_program(ps_words)?;
    let ps_uses = trace_descriptors(&ps_decoded);
    let ps_bind = planned_image_bindings(&ps_decoded, &ps_uses, tex_base, stor_base);
    let images = vs_bind
        .into_iter()
        .zip(vs_uses)
        .chain(ps_bind.into_iter().zip(ps_uses))
        .filter_map(|(b, u)| b.map(|b| (u, b)))
        .collect();
    Ok(PipelineEntry {
        spirv: ps.spirv,
        vertex: Some(vs.spirv),
        hull: None,
        domain: None,
        geometry: None,
        task: None,
        mesh: None,
        images,
        gfx: state,
    })
}

/// Translates a graphics pipeline with optional Hull, Domain, and Geometry shaders.
#[allow(clippy::too_many_arguments)]
pub fn translate_graphics_full(
    vs_words: &[u32],
    ps_words: &[u32],
    hs_words: Option<&[u32]>,
    ds_words: Option<&[u32]>,
    gs_words: Option<&[u32]>,
    quad_control: bool,
    state: Option<GraphicsPsoState>,
) -> Result<PipelineEntry, strata_shader::CompileError> {
    let mut entry = translate_graphics(vs_words, ps_words, quad_control, state)?;
    if let (Some(h), Some(d)) = (hs_words, ds_words) {
        let hs = strata_shader::tessellation::translate_hull_rdna2(
            h,
            &strata_shader::tessellation::HullShaderDesc::default(),
        )?;
        let ds = strata_shader::tessellation::translate_domain_rdna2(
            d,
            &strata_shader::tessellation::DomainShaderDesc::default(),
        )?;
        entry.hull = Some(hs);
        entry.domain = Some(ds);
        if let Some(ref mut g) = entry.gfx {
            g.patch_control_points = 3;
        }
    }
    if let Some(g) = gs_words {
        let gs = strata_shader::geometry::translate_geometry_rdna2(
            g,
            &strata_shader::geometry::GeometryShaderDesc {
                input: strata_shader::geometry::GsInputPrimitive::Triangles,
                output: strata_shader::geometry::GsOutputPrimitive::TriangleStrip,
                max_vertices: 64,
                invocations: 1,
            },
        )?;
        entry.geometry = Some(gs);
    }
    Ok(entry)
}

/// Translates an Amplification (optional) + Mesh + Pixel pipeline for `VK_EXT_mesh_shader`.
pub fn translate_mesh(
    as_words: Option<&[u32]>,
    ms_words: &[u32],
    ps_words: Option<&[u32]>,
    quad_control: bool,
    state: Option<GraphicsPsoState>,
) -> Result<PipelineEntry, strata_shader::CompileError> {
    let task = match as_words {
        Some(w) => Some(strata_shader::geometry::translate_task_rdna2(
            w,
            &strata_shader::geometry::TaskShaderDesc::default(),
        )?),
        None => None,
    };
    let mesh = strata_shader::geometry::translate_mesh_rdna2(
        ms_words,
        &strata_shader::geometry::MeshShaderDesc {
            workgroup_size: [32, 1, 1],
            max_vertices: 64,
            max_primitives: 64,
            output_topology: strata_shader::geometry::GsOutputPrimitive::TriangleStrip,
        },
    )?;
    let (ps_spirv, images) = if let Some(ps_w) = ps_words {
        let ps_opts = BuildOptions {
            stage: Stage::Pixel,
            require_full_quads: quad_control,
            ..Default::default()
        };
        let ps = strata_shader::compile_rdna2_with(ps_w, [1, 1, 1], &ps_opts)?;
        let ps_decoded = strata_shader::rdna2::decode_program(ps_w)?;
        let ps_uses = trace_descriptors(&ps_decoded);
        let ps_bind = planned_image_bindings(&ps_decoded, &ps_uses, 0, 0);
        let imgs = ps_bind
            .into_iter()
            .zip(ps_uses)
            .filter_map(|(b, u)| b.map(|b| (u, b)))
            .collect();
        (ps.spirv, imgs)
    } else {
        // Depth-only mesh pass: minimal empty fragment shader.
        let (_, frag) =
            strata_shader::dxil::compile_dxil_to_spirv(&[], [1, 1, 1]).unwrap_or_else(|_| {
                let mut w = vec![0x0723_0203, 0x0001_0600, 0, 5, 0];
                w.extend_from_slice(&[(2 << 16) | 17, 1]);
                w.extend_from_slice(&[(3 << 16) | 14, 0, 1]);
                w.extend_from_slice(&[(5 << 16) | 15, 4, 1, u32::from_le_bytes(*b"main"), 0]);
                w.extend_from_slice(&[(3 << 16) | 16, 1, 7]);
                w.extend_from_slice(&[(2 << 16) | 19, 2]);
                w.extend_from_slice(&[(3 << 16) | 33, 3, 2]);
                w.extend_from_slice(&[(5 << 16) | 54, 2, 1, 0, 3]);
                w.extend_from_slice(&[(2 << 16) | 248, 4]);
                w.extend_from_slice(&[(1 << 16) | 253]);
                w.extend_from_slice(&[(1 << 16) | 56]);
                (strata_shader::dxil::DxilStage::Pixel, w)
            });
        (frag, Vec::new())
    };
    Ok(PipelineEntry {
        spirv: ps_spirv,
        vertex: None,
        hull: None,
        domain: None,
        geometry: None,
        task,
        mesh: Some(mesh),
        images,
        gfx: state,
    })
}

/// A registered pipeline: its SPIR-V and the images it binds.
#[derive(Clone)]
pub struct PipelineEntry {
    /// The compute shader, or the pixel shader of a graphics pipeline.
    pub spirv: Vec<u32>,
    /// The vertex shader of a graphics pipeline.
    pub vertex: Option<Vec<u32>>,
    /// Optional Hull (`TessellationControl`) shader.
    pub hull: Option<Vec<u32>>,
    /// Optional Domain (`TessellationEvaluation`) shader.
    pub domain: Option<Vec<u32>>,
    /// Optional Geometry shader.
    pub geometry: Option<Vec<u32>>,
    /// Optional Amplification (`TaskEXT`) shader.
    pub task: Option<Vec<u32>>,
    /// Optional Mesh (`MeshEXT`) shader.
    pub mesh: Option<Vec<u32>>,
    /// Image instructions in program order and the slot each binds.
    pub images: Vec<(DescriptorUse, ImageBinding)>,
    /// The PSO's fixed-function state (`None`: defaults).
    pub gfx: Option<GraphicsPsoState>,
}

/// Reads and writes title memory by console address or GPU address (BDA):
/// through the UMA when an allocation covers it (a mirror's host copy),
/// else directly when readable.
struct TitleMemory<'a> {
    uma: Option<&'a Mutex<UmaManager>>,
}

impl TitleMemory<'_> {
    fn host(&self, addr: u64) -> u64 {
        self.uma
            .and_then(|u| u.lock().unwrap().host_ptr_any(addr))
            .map_or(addr, |p| p as u64)
    }

    fn read(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        let host = self.host(addr);
        if !readable(host, len) {
            return None;
        }
        // SAFETY: checked readable.
        Some(unsafe { std::slice::from_raw_parts(host as *const u8, len) }.to_vec())
    }

    fn write(&self, addr: u64, bytes: &[u8]) -> bool {
        let host = self.host(addr);
        if !readable(host, bytes.len()) {
            return false;
        }
        // SAFETY: title memory the title itself writes; committed (checked).
        let dst = unsafe { std::slice::from_raw_parts_mut(host as *mut u8, bytes.len()) };
        let _ = strata_gpu::readback::land_changed_dwords(bytes, dst);
        // Caches of title memory (textures) must see host-side writes.
        strata_gpu::dirty_log::note(addr, bytes.len() as u64);
        true
    }

    /// Brings `len` bytes at `addr` that shaders may have written in a
    /// VRAM mirror back to the memory the host reads.
    fn sync_for_read(&self, runner: &ComputeRunner, addr: u64, len: u64) {
        if let Some(u) = self.uma {
            let u = u.lock().unwrap();
            if let Some(va) = u.console_va_of(addr) {
                let _ = u.readback(runner, va, len);
            }
        }
    }

    fn dwords(&self, addr: u64, n: usize) -> Option<Vec<u32>> {
        let b = self.read(addr, n * 4)?;
        Some(
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect(),
        )
    }

    fn u64_at(&self, addr: u64) -> Option<u64> {
        let d = self.dwords(addr, 2)?;
        Some(u64::from(d[0]) | (u64::from(d[1]) << 32))
    }

    /// The 64-bit address an SGPR pair holds, as the trace names it.
    fn pair_address(&self, o: &Origin, sgprs: &[u32; 32]) -> Option<u64> {
        match o {
            Origin::User(n) => {
                let n = *n as usize;
                (n + 1 < 32).then(|| u64::from(sgprs[n]) | (u64::from(sgprs[n + 1]) << 32))
            }
            Origin::Load { base, offset } => {
                let at = self
                    .pair_address(base, sgprs)?
                    .wrapping_add(*offset as i64 as u64);
                self.u64_at(at)
            }
            _ => None,
        }
    }

    /// The `n` dwords of a descriptor whose first dword the trace names.
    fn descriptor(&self, o: &Origin, n: usize, sgprs: &[u32; 32]) -> Option<Vec<u32>> {
        match o {
            Origin::User(k) => {
                let k = *k as usize;
                (k + n <= 32).then(|| sgprs[k..k + n].to_vec())
            }
            Origin::Load { base, offset } => {
                let at = self
                    .pair_address(base, sgprs)?
                    .wrapping_add(*offset as i64 as u64);
                self.dwords(at, n)
            }
            _ => None,
        }
    }
}

/// Builds the image bindings of a dispatch. Returns them and the T#s of
/// the storage images to write back afterwards; `Err` names what could not
/// be bound (the dispatch is then skipped rather than run with wrong data).
fn bind_images(
    entry: &PipelineEntry,
    sgprs: &[u32; 32],
    textures: &mut TextureCache,
    runner: &ComputeRunner,
    mem: &TitleMemory<'_>,
) -> Result<(ImageBindings, Vec<[u32; 8]>), String> {
    let mut b = ImageBindings::default();
    let mut writes = Vec::new();
    let read = |a: u64, n: usize| mem.read(a, n);
    for (u, binding) in &entry.images {
        if !u.contiguous {
            return Err(format!(
                "pc {}: descriptor dwords from {}",
                u.pc, u.resource
            ));
        }
        let t = mem
            .descriptor(&u.resource, 8, sgprs)
            .ok_or_else(|| format!("pc {}: cannot read T# at {}", u.pc, u.resource))?;
        let t: [u32; 8] = t.try_into().unwrap();
        match *binding {
            ImageBinding::Sampled(slot) | ImageBinding::Fetched(slot) => {
                let view = textures
                    .texture_dim(runner, t, false, u.dim, &read)
                    .map_err(|e| format!("pc {}: {e}", u.pc))?;
                let sampler = match (&u.sampler, u.kind) {
                    (Some(o), ResourceKind::SampledImage) => {
                        let sw = mem
                            .descriptor(o, 4, sgprs)
                            .ok_or_else(|| format!("pc {}: cannot read S# at {o}", u.pc))?;
                        Some(
                            textures
                                .sampler_for(runner, sw.try_into().unwrap(), u.compare)
                                .map_err(|e| e.to_string())?,
                        )
                    }
                    _ => None,
                };
                b.set_texture_dim(slot, view, sampler, u.dim);
            }
            ImageBinding::Storage(slot) => {
                let view = textures
                    .texture_dim(runner, t, true, u.dim.for_storage(), &read)
                    .map_err(|e| format!("pc {}: {e}", u.pc))?;
                b.set_storage_dim(slot, view, u.dim);
                writes.push(t);
            }
        }
    }
    Ok((b, writes))
}

/// The graphics state a draw applies over its PSO's (`IASetPrimitiveTopology`,
/// `RSSet*`, `OMSet*`).
#[derive(Debug, Clone, Default)]
struct DrawState {
    topology: Option<i32>,
    viewport: Option<[f32; 6]>,
    scissor: Option<[i32; 4]>,
    stencil_ref: u32,
    blend_factor: [f32; 4],
    depth_bounds: Option<(f32, f32)>,
    shading_rate: [u32; 2],
    shading_rate_combiners: [u32; 2],
    shading_rate_image: Option<u64>,
    stream_output: Vec<Option<(u64, u64, u64)>>,
    active_occlusion_query: Option<(u64, bool)>,
    active_stats_query: Option<u64>,
    query_acc: strata_gpu::graphics::DrawQueryResults,
    index: Option<(u64, u32, bool)>,
    vertex_buffers: Vec<Option<(u64, u32, u32)>>,
    colors: Vec<[u32; 8]>,
    depth: Option<[u32; 8]>,
    /// Render targets named by registered address (the D3D12.X test path).
    legacy_rts: Vec<RenderTargetDesc>,
    root_dwords: u32,
    root_spilled: bool,
    vertex_table_sgpr: Option<u8>,
    predicate: Option<(u64, bool)>,
}

impl D3D12CommandQueueX {
    /// GPU-visible scratch memory of at least `len` bytes: `(console VA, BDA)`.
    fn scratch(&self, runner: &ComputeRunner, len: u64) -> Option<(u64, u64)> {
        let uma = self.uma.as_ref()?;
        let mut s = self.scratch.lock().unwrap();
        if let Some((va, bda, size)) = *s
            && size >= len
        {
            return Some((va, bda));
        }
        let mut u = uma.lock().unwrap();
        if let Some((va, _, _)) = s.take() {
            u.free_allocation(runner, va);
        }
        let size = len.max(64 * 1024).next_power_of_two();
        let a = u.xmem_alloc_identity(runner, 0, size).ok()?;
        let r = (a.console_va, a.host_bda, a.size);
        *s = Some(r);
        Some((r.0, r.1))
    }

    /// The user SGPRs a draw or dispatch pushes: the root shadow, with the
    /// vertex-buffer table and (when the root spills) the root's own
    /// address filled in.
    fn push_sgprs(
        &self,
        runner: &ComputeRunner,
        mem: &TitleMemory<'_>,
        shadow: &[u32; 256],
        st: &DrawState,
    ) -> [u32; 32] {
        let mut shadow = *shadow;
        let vb_bytes = st.vertex_buffers.len() * 16;
        let root_bytes = if st.root_spilled {
            st.root_dwords as usize * 4
        } else {
            0
        };
        let need = (vb_bytes + root_bytes) as u64;
        let scratch = if need > 0 {
            self.scratch(runner, need)
        } else {
            None
        };
        if let (Some((va, bda)), Some(sgpr)) = (scratch, st.vertex_table_sgpr)
            && vb_bytes > 0
        {
            // One V# per slot: base, stride, records (bytes / stride).
            let mut table = Vec::with_capacity(vb_bytes);
            for v in &st.vertex_buffers {
                let (base, size, stride) = v.unwrap_or((0, 0, 0));
                let records = size.checked_div(stride).unwrap_or(size);
                let vsharp = crate::d3d12_types::buffer_descriptor(base, stride, records, 0);
                for w in vsharp {
                    table.extend_from_slice(&w.to_le_bytes());
                }
            }
            mem.write(va + root_bytes as u64, &table);
            let at = bda + root_bytes as u64;
            let s = sgpr as usize;
            if s + 1 < shadow.len() {
                shadow[s] = at as u32;
                shadow[s + 1] = (at >> 32) as u32;
            }
        }
        let mut push = [0u32; 32];
        if st.root_spilled {
            if let Some((va, bda)) = scratch {
                let bytes: Vec<u8> = shadow[..st.root_dwords as usize]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect();
                mem.write(va, &bytes);
                push[0] = bda as u32;
                push[1] = (bda >> 32) as u32;
            }
        } else {
            push.copy_from_slice(&shadow[..32]);
        }
        push
    }

    /// Whether `SetPredication` skips the next draw or dispatch.
    fn predicated(&self, mem: &TitleMemory<'_>, runner: &ComputeRunner, st: &DrawState) -> bool {
        let Some((va, skip_if_zero)) = st.predicate else {
            return false;
        };
        mem.sync_for_read(runner, va, 8);
        let v = mem.u64_at(va).unwrap_or(0);
        let skip = (v == 0) == skip_if_zero;
        if skip {
            self.predicated_skips.fetch_add(1, Ordering::Relaxed);
            strata_gpu::perf::predicated_commands(1);
        }
        skip
    }

    /// Runs one recorded dispatch of `entry` with its images bound, then
    /// writes the storage images it used back to title memory.
    fn run_dispatch(
        &self,
        runner: &ComputeRunner,
        entry: &PipelineEntry,
        sgprs: &[u32; 32],
        groups: [u32; 3],
    ) -> bool {
        let push: Vec<u8> = sgprs.iter().flat_map(|w| w.to_le_bytes()).collect();
        if entry.images.is_empty() {
            return runner.dispatch_spirv(&entry.spirv, &push, groups).is_ok();
        }
        let mem = TitleMemory {
            uma: self.uma.as_deref(),
        };
        let mut textures = self.textures.lock().unwrap();
        let (bindings, writes) = match bind_images(entry, sgprs, &mut textures, runner, &mem) {
            Ok(x) => x,
            Err(why) => {
                crate::crash::log_line(&format!("dispatch skipped: {why}"));
                return false;
            }
        };
        if runner
            .dispatch_spirv_bound(&entry.spirv, &push, groups, None, &bindings)
            .is_err()
        {
            return false;
        }
        let read = |a: u64, n: usize| mem.read(a, n);
        let write = |a: u64, b: &[u8]| mem.write(a, b);
        for t in writes {
            if let Err(e) = textures.write_back(runner, t, &write, &read) {
                crate::crash::log_line(&format!("storage image write-back: {e}"));
            }
        }
        true
    }

    /// Runs one draw of a graphics `entry` into the bound targets (or the
    /// registered ones `st.legacy_rts` names), then writes them (and any
    /// storage images) back.
    fn run_draw(
        &self,
        runner: &ComputeRunner,
        entry: &PipelineEntry,
        sgprs: &[u32; 32],
        args: DrawArgs,
        indexed: Option<i32>,
        st: &mut DrawState,
    ) -> bool {
        let Some(vs) = &entry.vertex else {
            return false;
        };
        let push: Vec<u8> = sgprs.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mem = TitleMemory {
            uma: self.uma.as_deref(),
        };
        let read = |a: u64, n: usize| mem.read(a, n);
        let write = |a: u64, b: &[u8]| mem.write(a, b);
        let colors: Vec<[u32; 8]> = if !st.colors.is_empty() || st.depth.is_some() {
            st.colors.clone()
        } else {
            let known = self.render_targets.lock().unwrap();
            match st
                .legacy_rts
                .iter()
                .map(|r| known.get(&r.console_va).copied())
                .collect::<Option<Vec<_>>>()
            {
                Some(w) if !w.is_empty() => w,
                _ => {
                    crate::crash::log_line("draw skipped: render target not registered");
                    return false;
                }
            }
        };
        let depth_fmt = st.depth.map(|d| {
            let raw = crate::d3d12_types::trailer(&d).map_or(0, |t| t.1);
            if raw == 0 {
                ash::vk::Format::D32_SFLOAT
            } else {
                ash::vk::Format::from_raw(raw)
            }
        });
        let mut textures = self.textures.lock().unwrap();
        let (bindings, writes) = match bind_images(entry, sgprs, &mut textures, runner, &mem) {
            Ok(x) => x,
            Err(why) => {
                crate::crash::log_line(&format!("draw skipped: {why}"));
                return false;
            }
        };
        for w in &colors {
            if let Err(e) = textures.color_target(runner, *w, &read) {
                crate::crash::log_line(&format!("draw skipped: render target: {e}"));
                return false;
            }
        }
        if let (Some(d), Some(f)) = (st.depth, depth_fmt)
            && let Err(e) = textures.depth_target(runner, d, f, &read)
        {
            crate::crash::log_line(&format!("draw skipped: depth target: {e}"));
            return false;
        }
        let mut state = entry.gfx.clone().unwrap_or_default();
        // RECTLIST: triangles through a GS that adds each rect's 4th corner.
        let mut rect_gs = None;
        match st.topology {
            Some(strata_shader::rect::RECTLIST_VK_MARKER) => {
                state.topology = ash::vk::PrimitiveTopology::TRIANGLE_LIST;
                if entry.geometry.is_none() && runner.geometry_shader {
                    let locs = strata_shader::rect::vs_param_locations(vs);
                    let mut cache = self.rect_gs.lock().unwrap();
                    rect_gs = Some(std::sync::Arc::clone(cache.entry(locs).or_insert_with_key(
                        |locs| {
                            std::sync::Arc::new(strata_shader::rect::emit_rectlist_gs_spirv(locs))
                        },
                    )));
                } else {
                    static WARNED: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        crate::crash::log_line("rect lists drawn as triangles: no geometry shader");
                    }
                }
            }
            Some(t) => state.topology = ash::vk::PrimitiveTopology::from_raw(t),
            None => {}
        }
        state.viewport = st.viewport;
        state.scissor = st.scissor.map(|[l, t, r, b]| ash::vk::Rect2D {
            offset: ash::vk::Offset2D {
                x: l.max(0),
                y: t.max(0),
            },
            extent: ash::vk::Extent2D {
                width: (r - l.max(0)).max(0) as u32,
                height: (b - t.max(0)).max(0) as u32,
            },
        });
        state.stencil_reference = st.stencil_ref;
        state.blend_constants = st.blend_factor;
        if let Some((min_d, max_d)) = st.depth_bounds {
            state.depth_bounds_test_enable = true;
            state.min_depth_bounds = min_d;
            state.max_depth_bounds = max_d;
        }
        state.shading_rate = st.shading_rate;
        state.shading_rate_combiners = st.shading_rate_combiners;
        let index = match (indexed, st.index) {
            (Some(base_vertex), Some((va, _, index32))) => {
                let uma = self.uma.as_ref().map(|u| u.lock().unwrap());
                match uma.as_ref().and_then(|u| u.buffer_at(va)) {
                    Some((buffer, offset, _)) => Some(strata_gpu::graphics::IndexBufferBinding {
                        buffer,
                        offset,
                        index_type: if index32 {
                            ash::vk::IndexType::UINT32
                        } else {
                            ash::vk::IndexType::UINT16
                        },
                        vertex_offset: base_vertex,
                    }),
                    None => {
                        crate::crash::log_line("indexed draw skipped: index buffer not in UMA");
                        return false;
                    }
                }
            }
            (Some(_), None) => {
                crate::crash::log_line("indexed draw skipped: no index buffer bound");
                return false;
            }
            _ => None,
        };
        st.query_acc.capture_time = strata_gpu::perf::enabled();
        st.query_acc.gpu_ms = None;
        let want_query = st.active_occlusion_query.is_some()
            || st.active_stats_query.is_some()
            || st.query_acc.capture_time;
        if want_query {
            st.query_acc.capture_occlusion = st.active_occlusion_query.is_some();
            st.query_acc.capture_pipeline_stats = st.active_stats_query.is_some();
        }
        let ok = {
            let targets: Vec<&strata_gpu::images::GpuImage> =
                colors.iter().filter_map(|w| textures.image(w)).collect();
            let depth = match (st.depth, depth_fmt) {
                (Some(d), Some(f)) => textures.depth_image(&d, f),
                _ => None,
            };
            runner
                .draw_stages_with_state(
                    strata_gpu::graphics::GraphicsStages::Classic {
                        vs,
                        hs: entry.hull.as_deref(),
                        ds: entry.domain.as_deref(),
                        gs: entry
                            .geometry
                            .as_deref()
                            .or(rect_gs.as_deref().map(Vec::as_slice)),
                        ps: &entry.spirv,
                    },
                    &push,
                    Some(args),
                    None,
                    &targets,
                    depth,
                    index,
                    &bindings,
                    &state,
                    want_query.then_some(&mut st.query_acc),
                )
                .is_ok()
        };
        if let Some(ms) = st.query_acc.gpu_ms {
            strata_gpu::perf::graphics_gpu_ns((ms * 1_000_000.0) as u64);
        }
        for t in colors.iter().chain(&writes) {
            if let Err(e) = textures.write_back(runner, *t, &write, &read) {
                crate::crash::log_line(&format!("render target write-back: {e}"));
            }
        }
        if let (Some(d), Some(f)) = (st.depth, depth_fmt)
            && state.depth_write_enable
            && let Err(e) = textures.write_back_depth(runner, d, f, &write, &read)
        {
            crate::crash::log_line(&format!("depth target write-back: {e}"));
        }
        ok
    }

    /// Runs one mesh/amplification draw (`DispatchMesh`) via `VK_EXT_mesh_shader`.
    fn run_mesh(
        &self,
        runner: &ComputeRunner,
        entry: &PipelineEntry,
        sgprs: &[u32; 32],
        groups: [u32; 3],
        st: &mut DrawState,
    ) -> bool {
        let Some(mesh) = &entry.mesh else {
            return false;
        };
        if !runner.mesh_shader {
            self.log_once("DispatchMesh: host GPU does not expose VK_EXT_mesh_shader");
            return false;
        }
        let push: Vec<u8> = sgprs.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mem = TitleMemory {
            uma: self.uma.as_deref(),
        };
        let read = |a: u64, n: usize| mem.read(a, n);
        let write = |a: u64, b: &[u8]| mem.write(a, b);
        let colors: Vec<[u32; 8]> = st.colors.clone();
        let depth_fmt = st.depth.map(|d| {
            let raw = crate::d3d12_types::trailer(&d).map_or(0, |t| t.1);
            if raw == 0 {
                ash::vk::Format::D32_SFLOAT
            } else {
                ash::vk::Format::from_raw(raw)
            }
        });
        let mut textures = self.textures.lock().unwrap();
        let (bindings, writes) = match bind_images(entry, sgprs, &mut textures, runner, &mem) {
            Ok(x) => x,
            Err(why) => {
                crate::crash::log_line(&format!("mesh draw skipped: {why}"));
                return false;
            }
        };
        for w in &colors {
            if let Err(e) = textures.color_target(runner, *w, &read) {
                crate::crash::log_line(&format!("mesh draw skipped: render target: {e}"));
                return false;
            }
        }
        if let (Some(d), Some(f)) = (st.depth, depth_fmt)
            && let Err(e) = textures.depth_target(runner, d, f, &read)
        {
            crate::crash::log_line(&format!("mesh draw skipped: depth target: {e}"));
            return false;
        }
        let mut state = entry.gfx.clone().unwrap_or_default();
        state.viewport = st.viewport;
        state.scissor = st.scissor.map(|[l, t, r, b]| ash::vk::Rect2D {
            offset: ash::vk::Offset2D {
                x: l.max(0),
                y: t.max(0),
            },
            extent: ash::vk::Extent2D {
                width: (r - l.max(0)).max(0) as u32,
                height: (b - t.max(0)).max(0) as u32,
            },
        });
        state.stencil_reference = st.stencil_ref;
        state.blend_constants = st.blend_factor;
        if let Some((min_d, max_d)) = st.depth_bounds {
            state.depth_bounds_test_enable = true;
            state.min_depth_bounds = min_d;
            state.max_depth_bounds = max_d;
        }
        state.shading_rate = st.shading_rate;
        state.shading_rate_combiners = st.shading_rate_combiners;
        st.query_acc.capture_time = strata_gpu::perf::enabled();
        st.query_acc.gpu_ms = None;
        let want_query = st.active_occlusion_query.is_some()
            || st.active_stats_query.is_some()
            || st.query_acc.capture_time;
        if want_query {
            st.query_acc.capture_occlusion = st.active_occlusion_query.is_some();
            st.query_acc.capture_pipeline_stats = st.active_stats_query.is_some();
        }
        let ok = {
            let targets: Vec<&strata_gpu::images::GpuImage> =
                colors.iter().filter_map(|w| textures.image(w)).collect();
            let depth = match (st.depth, depth_fmt) {
                (Some(d), Some(f)) => textures.depth_image(&d, f),
                _ => None,
            };
            runner
                .draw_mesh_with_state(
                    entry.task.as_deref(),
                    mesh,
                    &entry.spirv,
                    &push,
                    groups,
                    &targets,
                    depth,
                    &bindings,
                    &state,
                    want_query.then_some(&mut st.query_acc),
                )
                .is_ok()
        };
        if let Some(ms) = st.query_acc.gpu_ms {
            strata_gpu::perf::graphics_gpu_ns((ms * 1_000_000.0) as u64);
        }
        for t in colors.iter().chain(&writes) {
            if let Err(e) = textures.write_back(runner, *t, &write, &read) {
                crate::crash::log_line(&format!("mesh render target write-back: {e}"));
            }
        }
        if let (Some(d), Some(f)) = (st.depth, depth_fmt)
            && state.depth_write_enable
            && let Err(e) = textures.write_back_depth(runner, d, f, &write, &read)
        {
            crate::crash::log_line(&format!("mesh depth target write-back: {e}"));
        }
        ok
    }

    /// Executes one closed command list's thin commands, in order.
    pub fn execute_thin(&self, list: &ThinCommandList) {
        let _mapped = MappedScope::enter();
        crate::frame_capture::record(list);
        self.submitted_lists.fetch_add(1, Ordering::Relaxed);
        self.submitted_commands
            .fetch_add(list.commands().len() as u32, Ordering::Relaxed);
        self.hoisted_clamps
            .fetch_add(list.hoisted_clamps.len() as u32, Ordering::Relaxed);
        let Some(runner) = &self.gpu_runner else {
            return;
        };
        // 0. The title's CPU writes since the last list reach the mirrors.
        if let Some(uma) = &self.uma
            && let Ok(stats) = uma.lock().unwrap().sync_mirrors(runner)
        {
            self.synced_bytes.fetch_add(stats.bytes, Ordering::Relaxed);
        }

        // 1. Batched pre-pass for all hoisted indirect draw clamps (single GPU dispatch)
        if !list.hoisted_clamps.is_empty() {
            let count = list.hoisted_clamps.len();
            if let Ok(mut entries_buf) = runner.alloc_bda_buffer((count * 16) as u64) {
                unsafe {
                    let dst = entries_buf.as_mut_slice::<u32>(count * 4);
                    for (i, &(bda, max_c, max_i)) in list.hoisted_clamps.iter().enumerate() {
                        dst[i * 4] = bda as u32;
                        dst[i * 4 + 1] = (bda >> 32) as u32;
                        dst[i * 4 + 2] = max_c;
                        dst[i * 4 + 3] = max_i;
                    }
                }
                let mut pc = Vec::with_capacity(16);
                pc.extend_from_slice(&entries_buf.bda.to_le_bytes());
                pc.extend_from_slice(&(count as u32).to_le_bytes());
                pc.extend_from_slice(&0u32.to_le_bytes());
                let clamp_words: Vec<u32> = CLAMP_INDIRECT_SPV
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let groups = (count as u32).div_ceil(64);
                let _ = runner.dispatch_spirv(&clamp_words, &pc, [groups, 1, 1]);
                runner.free_bda_buffer(entries_buf);
            }
        }

        // 2. Replay the list.
        let mem = TitleMemory {
            uma: self.uma.as_deref(),
        };
        let mut active_pso: Option<PipelineKey> = None;
        let mut shadow = [0u32; 256];
        let mut st = DrawState::default();
        for cmd in list.commands() {
            match cmd {
                ThinCommand::BindPipeline(k) => active_pso = Some(*k),
                ThinCommand::PushUserSgprs { offset_dw, values } => {
                    for (i, &v) in values.iter().enumerate() {
                        let idx = (*offset_dw as usize) + i;
                        if idx < shadow.len() {
                            shadow[idx] = v;
                        }
                    }
                }
                ThinCommand::Dispatch {
                    x,
                    y,
                    z,
                    is_thread_count,
                } => {
                    if self.predicated(&mem, runner, &st) {
                        continue;
                    }
                    let groups = if *is_thread_count {
                        // Console DispatchX with USE_THREAD_DIMENSIONS (0x20):
                        // convert thread counts into whole workgroups.
                        [(*x).div_ceil(64).max(1), (*y).max(1), (*z).max(1)]
                    } else {
                        [*x, *y, *z]
                    };
                    let sgprs = self.push_sgprs(runner, &mem, &shadow, &st);
                    let pipes = self.pipelines.lock().unwrap();
                    if let Some(entry) = active_pso.and_then(|k| pipes.get(&k))
                        && self.run_dispatch(runner, entry, &sgprs, groups)
                    {
                        self.executed_gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                    }
                }
                ThinCommand::SetRenderTargets(r) => {
                    st.legacy_rts = r.clone();
                    st.colors.clear();
                    st.depth = None;
                }
                ThinCommand::BindTargets { colors, depth } => {
                    st.colors = colors.clone();
                    st.depth = *depth;
                    st.legacy_rts.clear();
                }
                ThinCommand::DrawInstanced {
                    vertex_count,
                    instance_count,
                    start_vertex,
                    start_instance,
                } => {
                    let args = DrawArgs {
                        vertex_count: *vertex_count,
                        instance_count: *instance_count,
                        first_vertex: *start_vertex,
                        first_instance: *start_instance,
                    };
                    self.draw(runner, &mem, active_pso, &shadow, &mut st, args, None);
                }
                ThinCommand::DrawIndexedInstanced {
                    index_count,
                    instance_count,
                    start_index,
                    base_vertex,
                    start_instance,
                } => {
                    let args = DrawArgs {
                        vertex_count: *index_count,
                        instance_count: *instance_count,
                        first_vertex: *start_index,
                        first_instance: *start_instance,
                    };
                    self.draw(
                        runner,
                        &mem,
                        active_pso,
                        &shadow,
                        &mut st,
                        args,
                        Some(*base_vertex),
                    );
                }
                ThinCommand::DrawIndirect { args_bda, .. } => {
                    // Clamped in the pre-pass; the arguments are read back
                    // and the draw issued directly.
                    mem.sync_for_read(runner, *args_bda, 16);
                    if let Some(a) = mem.dwords(*args_bda, 4) {
                        let args = DrawArgs {
                            vertex_count: a[0],
                            instance_count: a[1],
                            first_vertex: a[2],
                            first_instance: a[3],
                        };
                        self.draw(runner, &mem, active_pso, &shadow, &mut st, args, None);
                    }
                }
                ThinCommand::DispatchIndirect { args_va } => {
                    if self.predicated(&mem, runner, &st) {
                        continue;
                    }
                    let groups = indirect_args(self.uma.as_deref(), runner, *args_va);
                    let sgprs = self.push_sgprs(runner, &mem, &shadow, &st);
                    let pipes = self.pipelines.lock().unwrap();
                    if let (Some(g), Some(entry)) = (groups, active_pso.and_then(|k| pipes.get(&k)))
                    {
                        // Never clamped: dispatches split past the device limit.
                        if self.run_dispatch(runner, entry, &sgprs, g) {
                            self.executed_gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                // Shader-written data the title's CPU code reads (e.g. ocean
                // height maps its physics samples) goes back to title memory.
                ThinCommand::ReadbackTexture {
                    console_va,
                    byte_len,
                } => {
                    if let Some(uma) = &self.uma
                        && let Ok(n) = uma.lock().unwrap().readback(runner, *console_va, *byte_len)
                    {
                        self.readback_bytes.fetch_add(n, Ordering::Relaxed);
                    }
                }
                ThinCommand::SetViewport(v) => st.viewport = Some(v.map(f32::from_bits)),
                ThinCommand::SetScissor(r) => st.scissor = Some(*r),
                ThinCommand::SetStencilRef(r) => st.stencil_ref = *r,
                ThinCommand::SetBlendFactor(b) => st.blend_factor = b.map(f32::from_bits),
                ThinCommand::SetTopology(t) => st.topology = Some(*t),
                ThinCommand::SetIndexBuffer { va, size, index32 } => {
                    st.index = (*va != 0).then_some((*va, *size, *index32));
                }
                ThinCommand::SetVertexBuffers { start_slot, views } => {
                    let end = *start_slot as usize + views.len();
                    if st.vertex_buffers.len() < end {
                        st.vertex_buffers.resize(end, None);
                    }
                    for (i, v) in views.iter().enumerate() {
                        st.vertex_buffers[*start_slot as usize + i] = (v.0 != 0).then_some(*v);
                    }
                }
                ThinCommand::SetRootLayout {
                    dwords,
                    spilled,
                    vertex_table_sgpr,
                } => {
                    st.root_dwords = *dwords;
                    st.root_spilled = *spilled;
                    st.vertex_table_sgpr = Some(*vertex_table_sgpr);
                }
                ThinCommand::ClearImage {
                    words,
                    aspects,
                    color,
                    depth,
                    stencil,
                } => self.clear(runner, &mem, words, *aspects, *color, *depth, *stencil),
                ThinCommand::FillMemory {
                    va,
                    len,
                    pattern,
                    dwords,
                } => {
                    let n = (*dwords).clamp(1, 4) as usize;
                    let unit: Vec<u8> = pattern[..n].iter().flat_map(|w| w.to_le_bytes()).collect();
                    let bytes: Vec<u8> = unit.iter().copied().cycle().take(*len as usize).collect();
                    if !mem.write(*va, &bytes) {
                        self.log_once("FillMemory: destination is not title memory");
                    }
                }
                ThinCommand::CopyMemory { dst, src, len } => {
                    let gpu_done = self
                        .uma
                        .as_ref()
                        .and_then(|u| {
                            u.lock()
                                .unwrap()
                                .gpu_copy_buffer(runner, *dst, *src, *len)
                                .ok()
                        })
                        .unwrap_or(false);
                    if !gpu_done {
                        mem.sync_for_read(runner, *src, *len);
                        match mem.read(*src, *len as usize) {
                            Some(b) if mem.write(*dst, &b) => {}
                            _ => self.log_once("CopyMemory: a range is not title memory"),
                        }
                    }
                }
                ThinCommand::CopyTexture(c) => self.copy_texture(runner, &mem, c),
                ThinCommand::WriteTimestamp { va } => {
                    let ns = runner
                        .query_gpu_timestamp_ns()
                        .unwrap_or_else(|_| self.epoch.elapsed().as_nanos() as u64);
                    mem.write(*va, &ns.to_le_bytes());
                }
                ThinCommand::WriteU64 { va, value } => {
                    mem.write(*va, &value.to_le_bytes());
                }
                ThinCommand::SetPredication { va, skip_if_zero } => {
                    st.predicate = (*va != 0).then_some((*va, *skip_if_zero));
                }
                ThinCommand::ExecuteIndirect {
                    layout,
                    max_count,
                    args_va,
                    count_va,
                } => self.execute_indirect(
                    runner,
                    &mem,
                    active_pso,
                    &mut shadow,
                    &mut st,
                    layout,
                    *max_count,
                    *args_va,
                    *count_va,
                ),
                ThinCommand::DispatchMesh { x, y, z } => {
                    self.dispatch_mesh(runner, &mem, active_pso, &shadow, &mut st, [*x, *y, *z]);
                }
                ThinCommand::BuildAccel(build) => {
                    match crate::d3d12_rt::build(&mem_reader(&mem, runner), build) {
                        // The latest top-level structure is the one BMA
                        // samples and shortcuts (`STRATA_BVH_BMA`).
                        Ok(written) => {
                            let mut rt = self.accel_structs.lock().unwrap();
                            if build.top_level {
                                let root_va = build.dest + crate::d3d12_rt::HEADER_BYTES;
                                rt.set_tlas_bda(root_va);
                                runner.bma_set_root(root_va);
                            } else {
                                if rt.stats().slab_count == 0 {
                                    rt.add_slab(build.dest, strata_gpu::raytracing::AS_SLAB_BYTES);
                                }
                                let _ = rt.register_blas(
                                    build.dest,
                                    written.max(256),
                                    (written / 128).max(1) as u32,
                                    true,
                                    u64::from(self.submitted_lists.load(Ordering::Relaxed)),
                                );
                            }
                        }
                        Err(e) => {
                            crate::crash::log_line(&format!("acceleration structure build: {e}"));
                        }
                    }
                }
                ThinCommand::CopyAccel { dst, src } => {
                    // Strata's BVHs hold no absolute addresses: a byte copy
                    // of the used size is a valid clone (and compaction).
                    let used = mem.u64_at(*src + 8).unwrap_or(0).min(1 << 30);
                    mem.sync_for_read(runner, *src, used);
                    if let Some(b) = mem.read(*src, used as usize) {
                        mem.write(*dst, &b);
                    }
                }
                ThinCommand::SetDepthBounds { min, max } => {
                    st.depth_bounds = Some((f32::from_bits(*min), f32::from_bits(*max)));
                }
                ThinCommand::SetStreamOutputTargets { start_slot, views } => {
                    let s = *start_slot as usize;
                    if st.stream_output.len() < s + views.len() {
                        st.stream_output.resize(s + views.len(), None);
                    }
                    for (i, &(va, size, filled_va)) in views.iter().enumerate() {
                        st.stream_output[s + i] = (va != 0).then_some((va, size, filled_va));
                    }
                }
                ThinCommand::SetShadingRate {
                    width,
                    height,
                    combiners,
                } => {
                    st.shading_rate = [(*width).max(1), (*height).max(1)];
                    st.shading_rate_combiners = *combiners;
                }
                ThinCommand::SetShadingRateImage { va } => {
                    st.shading_rate_image = (*va != 0).then_some(*va);
                }
                ThinCommand::BeginQuery { va, query_type } => match *query_type {
                    0 | 1 => {
                        st.active_occlusion_query = Some((*va, *query_type == 1));
                        st.query_acc.samples_passed = 0;
                    }
                    3 => {
                        st.active_stats_query = Some(*va);
                        st.query_acc.pipeline_stats = [0; 11];
                    }
                    _ => {}
                },
                ThinCommand::EndQuery { va, query_type } => match *query_type {
                    0 | 1 => {
                        let samples = st.query_acc.samples_passed;
                        let val = if *query_type == 1 {
                            u64::from(samples > 0)
                        } else {
                            samples
                        };
                        mem.write(*va, &val.to_le_bytes());
                        st.active_occlusion_query = None;
                        st.query_acc.capture_occlusion = false;
                    }
                    2 => {
                        let ns = runner
                            .query_gpu_timestamp_ns()
                            .unwrap_or_else(|_| self.epoch.elapsed().as_nanos() as u64);
                        mem.write(*va, &ns.to_le_bytes());
                    }
                    3 => {
                        let bytes: Vec<u8> = st
                            .query_acc
                            .pipeline_stats
                            .iter()
                            .flat_map(|c| c.to_le_bytes())
                            .collect();
                        mem.write(*va, &bytes);
                        st.active_stats_query = None;
                        st.query_acc.capture_pipeline_stats = false;
                    }
                    _ => {
                        mem.write(*va, &0u64.to_le_bytes());
                    }
                },
                ThinCommand::DispatchRays {
                    raygen,
                    miss,
                    hit,
                    callable,
                    width,
                    height,
                    depth,
                } => {
                    self.dispatch_rays(
                        runner,
                        &mem,
                        active_pso,
                        &shadow,
                        &st,
                        *raygen,
                        *miss,
                        *hit,
                        *callable,
                        [*width, *height, *depth],
                    );
                }
                ThinCommand::ResolveSubresourceRegion {
                    dst,
                    dst_origin,
                    src,
                    src_rect,
                    sample_count,
                } => {
                    let w = src_rect[2];
                    let h = src_rect[3];
                    let sc = (*sample_count).max(1);
                    let dst_row_bytes = u64::from(w) * u64::from(dst.bpe);
                    let src_row_bytes = u64::from(w) * u64::from(src.bpe) * u64::from(sc);
                    for y in 0..h {
                        let s_va = src.va
                            + u64::from(src_rect[1] + y) * src.row_pitch
                            + u64::from(src_rect[0]) * u64::from(src.bpe) * u64::from(sc);
                        let d_va = dst.va
                            + u64::from(dst_origin[1] + y) * dst.row_pitch
                            + u64::from(dst_origin[0]) * u64::from(dst.bpe);
                        if y == 0 {
                            mem.sync_for_read(runner, s_va, u64::from(h) * src.row_pitch);
                        }
                        if sc > 1 && src.bpe == 4 && dst.bpe == 4 {
                            if let Some(b) = mem.read(s_va, src_row_bytes as usize) {
                                let resolved = runner
                                    .msaa_resolve_gpu(&b, w, 1, sc, 0)
                                    .unwrap_or_else(|_| {
                                        let mut out = vec![0u8; dst_row_bytes as usize];
                                        strata_gpu::readback::resolve_msaa_rgba8(&b, sc, &mut out);
                                        out
                                    });
                                mem.write(d_va, &resolved);
                            }
                        } else if let Some(b) = mem.read(s_va, dst_row_bytes as usize) {
                            mem.write(d_va, &b);
                        }
                    }
                }
                ThinCommand::CopyTiles {
                    tiled_va,
                    buffer_va,
                    tile_count,
                    tile_bytes,
                    to_tiled,
                } => {
                    let len = u64::from(*tile_count) * u64::from(*tile_bytes);
                    let (src, dst) = if *to_tiled {
                        (*buffer_va, *tiled_va)
                    } else {
                        (*tiled_va, *buffer_va)
                    };
                    mem.sync_for_read(runner, src, len);
                    if let Some(b) = mem.read(src, len as usize) {
                        mem.write(dst, &b);
                    }
                }
                ThinCommand::WriteSamplerFeedback {
                    target_va,
                    mip_level,
                    tile_x,
                    tile_y,
                } => {
                    let off = (u64::from(*tile_y) * 64 + u64::from(*tile_x)) & 0xffff;
                    let va = target_va.saturating_add(off);
                    let cur = mem
                        .read(va, 1)
                        .and_then(|b| b.first().copied())
                        .unwrap_or(0xff);
                    if *mip_level < cur {
                        mem.write(va, &[*mip_level]);
                    }
                }
                ThinCommand::CopyBufferRegion {
                    dst_va,
                    dst_offset,
                    src_va,
                    src_offset,
                    num_bytes,
                } => {
                    let (dst, src) = (dst_va + dst_offset, src_va + src_offset);
                    let gpu_done = self
                        .uma
                        .as_ref()
                        .and_then(|u| {
                            u.lock()
                                .unwrap()
                                .gpu_copy_buffer(runner, dst, src, *num_bytes)
                                .ok()
                        })
                        .unwrap_or(false);
                    if !gpu_done {
                        mem.sync_for_read(runner, src, *num_bytes);
                        match mem.read(src, *num_bytes as usize) {
                            Some(b) if mem.write(dst, &b) => {}
                            _ => self.log_once("CopyBufferRegion: a range is not title memory"),
                        }
                    }
                }
                ThinCommand::UpdateTileMappings {
                    resource_va,
                    start_tile,
                    tile_count,
                    unbind,
                    ..
                } => {
                    if let Some(uma) = &self.uma {
                        uma.lock().unwrap().update_tile_mappings(
                            *resource_va,
                            *start_tile,
                            *tile_count,
                            *unbind,
                        );
                    }
                    let _ =
                        runner.bind_sparse_tiles(*resource_va, *start_tile, *tile_count, *unbind);
                }
                ThinCommand::CopyTileMappings {
                    dst_va,
                    src_va,
                    tile_count,
                } => {
                    if let Some(uma) = &self.uma {
                        uma.lock()
                            .unwrap()
                            .copy_tile_mappings(*dst_va, *src_va, *tile_count);
                    }
                }
                ThinCommand::ResourceBarrier {
                    state_before,
                    state_after,
                    ..
                } => {
                    runner.push_memory_barrier(*state_before, *state_after);
                }
            }
        }
        let _ = runner.flush_barriers();
    }

    /// One draw, honouring predication and the root/vertex tables.
    #[allow(clippy::too_many_arguments)]
    fn draw(
        &self,
        runner: &ComputeRunner,
        mem: &TitleMemory<'_>,
        active_pso: Option<PipelineKey>,
        shadow: &[u32; 256],
        st: &mut DrawState,
        args: DrawArgs,
        indexed: Option<i32>,
    ) {
        if self.predicated(mem, runner, st) {
            return;
        }
        let sgprs = self.push_sgprs(runner, mem, shadow, st);
        let pipes = self.pipelines.lock().unwrap();
        if let Some(entry) = active_pso.and_then(|k| pipes.get(&k))
            && self.run_draw(runner, entry, &sgprs, args, indexed, st)
        {
            self.executed_draws.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn dispatch_mesh(
        &self,
        runner: &ComputeRunner,
        mem: &TitleMemory<'_>,
        active_pso: Option<PipelineKey>,
        shadow: &[u32; 256],
        st: &mut DrawState,
        groups: [u32; 3],
    ) {
        if self.predicated(mem, runner, st) {
            return;
        }
        let sgprs = self.push_sgprs(runner, mem, shadow, st);
        let pipes = self.pipelines.lock().unwrap();
        if let Some(entry) = active_pso.and_then(|k| pipes.get(&k))
            && self.run_mesh(runner, entry, &sgprs, groups, st)
        {
            self.executed_draws.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Executes `DispatchRays` under Strata's No-API model: pushes the raygen,
    /// miss, hit-group, and callable SBT BDA ranges and ray dispatch dimensions
    /// into user SGPRs and dispatches the state object's translated raygen
    /// compute shader over `(ceil(w/8), ceil(h/8), d)` groups (walking Strata's
    /// BDA BVH directly via `image_bvh64_intersect_ray`).
    #[allow(clippy::too_many_arguments)]
    fn dispatch_rays(
        &self,
        runner: &ComputeRunner,
        mem: &TitleMemory<'_>,
        active_pso: Option<PipelineKey>,
        shadow: &[u32; 256],
        st: &DrawState,
        raygen: (u64, u64),
        miss: (u64, u64, u64),
        hit: (u64, u64, u64),
        callable: (u64, u64, u64),
        dims: [u32; 3],
    ) {
        if self.predicated(mem, runner, st) || dims[0] == 0 || dims[1] == 0 || dims[2] == 0 {
            return;
        }
        let mut sgprs = self.push_sgprs(runner, mem, shadow, st);
        // Pack SBT descriptor pointers + dispatch dimensions in the high user SGPRs
        // (s16..s31) so No-API raygen shaders can fetch SBT records directly via BDA.
        sgprs[16] = dims[0];
        sgprs[17] = dims[1];
        sgprs[18] = dims[2];
        sgprs[20] = raygen.0 as u32;
        sgprs[21] = (raygen.0 >> 32) as u32;
        sgprs[22] = miss.0 as u32;
        sgprs[23] = (miss.0 >> 32) as u32;
        sgprs[24] = hit.0 as u32;
        sgprs[25] = (hit.0 >> 32) as u32;
        sgprs[26] = callable.0 as u32;
        sgprs[27] = (callable.0 >> 32) as u32;
        let groups = [dims[0].div_ceil(8), dims[1].div_ceil(8), dims[2].max(1)];
        let pipes = self.pipelines.lock().unwrap();
        if let Some(entry) = active_pso.and_then(|k| pipes.get(&k))
            && self.run_dispatch(runner, entry, &sgprs, groups)
        {
            self.executed_gpu_dispatches.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn clear(
        &self,
        runner: &ComputeRunner,
        mem: &TitleMemory<'_>,
        words: &[u32; 8],
        aspects: strata_gpu::thin_list::ClearAspects,
        color: [u32; 4],
        depth: u32,
        stencil: u32,
    ) {
        let read = |a: u64, n: usize| mem.read(a, n);
        let write = |a: u64, b: &[u8]| mem.write(a, b);
        let depth_fmt = (aspects.depth || aspects.stencil).then(|| {
            let raw = crate::d3d12_types::trailer(words).map_or(0, |t| t.1);
            if raw == 0 {
                ash::vk::Format::D32_SFLOAT
            } else {
                ash::vk::Format::from_raw(raw)
            }
        });
        let mut textures = self.textures.lock().unwrap();
        if let Err(e) = textures.clear(
            runner,
            *words,
            depth_fmt,
            color,
            f32::from_bits(depth),
            stencil,
            &write,
            &read,
        ) {
            crate::crash::log_line(&format!("clear: {e}"));
        }
    }

    fn copy_texture(
        &self,
        runner: &ComputeRunner,
        mem: &TitleMemory<'_>,
        c: &strata_gpu::thin_list::TextureCopy,
    ) {
        let row_bytes = u64::from(c.extent[0]) * u64::from(c.src.bpe);
        for z in 0..c.extent[2] {
            for y in 0..c.extent[1] {
                let s = c.src.va
                    + u64::from(c.src_origin[2] + z) * c.src.slice_pitch
                    + u64::from(c.src_origin[1] + y) * c.src.row_pitch
                    + u64::from(c.src_origin[0]) * u64::from(c.src.bpe);
                let d = c.dst.va
                    + u64::from(c.dst_origin[2] + z) * c.dst.slice_pitch
                    + u64::from(c.dst_origin[1] + y) * c.dst.row_pitch
                    + u64::from(c.dst_origin[0]) * u64::from(c.dst.bpe);
                if z == 0 && y == 0 {
                    let span = u64::from(c.extent[2]) * c.src.slice_pitch.max(c.src.row_pitch);
                    mem.sync_for_read(runner, s, span.max(row_bytes));
                }
                match mem.read(s, row_bytes as usize) {
                    Some(b) if mem.write(d, &b) => {}
                    _ => {
                        self.log_once("CopyTextureRegion: a row is not title memory");
                        return;
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_indirect(
        &self,
        runner: &ComputeRunner,
        mem: &TitleMemory<'_>,
        active_pso: Option<PipelineKey>,
        shadow: &mut [u32; 256],
        st: &mut DrawState,
        layout: &strata_gpu::thin_list::IndirectLayout,
        max_count: u32,
        args_va: u64,
        count_va: u64,
    ) {
        use strata_gpu::thin_list::IndirectArg;
        let count = if count_va != 0 {
            mem.sync_for_read(runner, count_va, 4);
            mem.dwords(count_va, 1).map_or(0, |d| d[0]).min(max_count)
        } else {
            max_count
        };
        let stride = u64::from(layout.stride.max(1));
        mem.sync_for_read(runner, args_va, stride * u64::from(count));
        for i in 0..u64::from(count) {
            let mut at = args_va + i * stride;
            for arg in &layout.args {
                let n = (arg.bytes() / 4) as usize;
                let Some(d) = mem.dwords(at, n) else {
                    self.log_once("ExecuteIndirect: arguments are not title memory");
                    return;
                };
                match *arg {
                    IndirectArg::Draw => {
                        let args = DrawArgs {
                            vertex_count: d[0],
                            instance_count: d[1],
                            first_vertex: d[2],
                            first_instance: d[3],
                        };
                        self.draw(runner, mem, active_pso, shadow, st, args, None);
                    }
                    IndirectArg::DrawIndexed => {
                        let args = DrawArgs {
                            vertex_count: d[0],
                            instance_count: d[1],
                            first_vertex: d[2],
                            first_instance: d[4],
                        };
                        self.draw(runner, mem, active_pso, shadow, st, args, Some(d[3] as i32));
                    }
                    IndirectArg::Dispatch => {
                        if !self.predicated(mem, runner, st) {
                            let sgprs = self.push_sgprs(runner, mem, shadow, st);
                            let pipes = self.pipelines.lock().unwrap();
                            if let Some(entry) = active_pso.and_then(|k| pipes.get(&k))
                                && self.run_dispatch(runner, entry, &sgprs, [d[0], d[1], d[2]])
                            {
                                self.executed_gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    IndirectArg::DispatchMesh => {
                        self.dispatch_mesh(runner, mem, active_pso, shadow, st, [d[0], d[1], d[2]]);
                    }
                    IndirectArg::DispatchRays => {
                        let q = |k: usize| u64::from(d[k]) | (u64::from(d[k + 1]) << 32);
                        self.dispatch_rays(
                            runner,
                            mem,
                            active_pso,
                            shadow,
                            st,
                            (q(0), q(2)),
                            (q(4), q(6), q(8)),
                            (q(10), q(12), q(14)),
                            (q(16), q(18), q(20)),
                            [d[22], d[23], d[24]],
                        );
                    }
                    IndirectArg::VertexBuffer { slot } => {
                        let s = slot as usize;
                        if st.vertex_buffers.len() <= s {
                            st.vertex_buffers.resize(s + 1, None);
                        }
                        let va = u64::from(d[0]) | (u64::from(d[1]) << 32);
                        st.vertex_buffers[s] = (va != 0).then_some((va, d[2], d[3]));
                    }
                    IndirectArg::IndexBuffer => {
                        let va = u64::from(d[0]) | (u64::from(d[1]) << 32);
                        st.index = (va != 0).then_some((va, d[2], d[3] == 42));
                    }
                    IndirectArg::Constants { sgpr, .. } => {
                        for (k, &v) in d.iter().enumerate() {
                            if let Some(s) = shadow.get_mut(sgpr as usize + k) {
                                *s = v;
                            }
                        }
                    }
                    IndirectArg::RootAddress { sgpr } => {
                        for (k, &v) in d.iter().take(2).enumerate() {
                            if let Some(s) = shadow.get_mut(sgpr as usize + k) {
                                *s = v;
                            }
                        }
                    }
                }
                at += u64::from(arg.bytes());
            }
        }
    }
}

/// Title memory as the acceleration-structure builder reads and writes it.
fn mem_reader<'a>(
    mem: &'a TitleMemory<'a>,
    runner: &'a ComputeRunner,
) -> crate::d3d12_rt::Memory<'a> {
    crate::d3d12_rt::Memory {
        read: Box::new(move |a, n| {
            mem.sync_for_read(runner, a, n as u64);
            mem.read(a, n)
        }),
        write: Box::new(move |a, b| mem.write(a, b)),
    }
}

unsafe extern "system" fn queue_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_NOINTERFACE;
    }
    unsafe {
        queue_add_ref(this);
        *out = this;
    }
    S_OK
}

unsafe extern "system" fn queue_add_ref(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const D3D12CommandQueueX) };
    obj.header.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn queue_release(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const D3D12CommandQueueX) };
    let prev = obj.header.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        let q = unsafe { Box::from_raw(this as *mut D3D12CommandQueueX) };
        if let Some(runner) = &q.gpu_runner {
            // Images are shared by a device's queues: the last owner frees them.
            if Arc::strong_count(&q.textures) == 1 {
                q.textures.lock().unwrap().destroy(runner);
            }
            if let (Some(uma), Some((va, _, _))) = (&q.uma, *q.scratch.lock().unwrap()) {
                uma.lock().unwrap().free_allocation(runner, va);
            }
        }
        drop(q);
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn queue_execute_command_lists(
    this: *mut c_void,
    num_lists: u32,
    lists: *const *mut c_void,
) {
    if lists.is_null() || num_lists == 0 {
        return;
    }
    let queue = unsafe { &*(this as *const D3D12CommandQueueX) };
    let slice = unsafe { std::slice::from_raw_parts(lists, num_lists as usize) };
    let live: Vec<&D3D12CommandListX> = slice
        .iter()
        .filter(|p| !p.is_null())
        .map(|&p| unsafe { &*(p as *const D3D12CommandListX) })
        .collect();
    if live.len() > 1 {
        // Multi-list batches go through the translation pool and, when
        // every list is compute-only, parallel command-buffer recording.
        let batch = live
            .iter()
            .map(|cl| {
                let guard = cl.list.lock().unwrap();
                let mut copy = ThinCommandList::new();
                for cmd in guard.commands() {
                    copy.push(cmd.clone());
                }
                copy.hoisted_clamps = guard.hoisted_clamps.clone();
                copy
            })
            .collect();
        queue.execute_command_lists(batch);
        return;
    }
    queue.prepare_submission();
    for cl in live {
        let guard = cl.list.lock().unwrap();
        queue.execute_thin(&guard);
    }
}

unsafe extern "system" fn queue_present_x(
    this: *mut c_void,
    plane_count: u32,
    planes: *const c_void,
    _params: *const c_void,
) -> i32 {
    let queue = unsafe { &*(this as *const D3D12CommandQueueX) };
    queue.frame_clock.lock().unwrap().record_present_x();
    unsafe { crate::thunks::present_frame(plane_count, planes) }
}

// ============================================================================
// ID3D12DeviceX COM VTable
// ============================================================================

#[repr(C)]
pub struct DeviceVTable {
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    pub create_command_queue: unsafe extern "system" fn(
        *mut c_void,
        *const c_void,
        *const c_void,
        *mut *mut c_void,
    ) -> i32,
    pub create_command_list: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        *mut c_void,
        *mut c_void,
        *const c_void,
        *mut *mut c_void,
    ) -> i32,
    pub create_committed_resource:
        unsafe extern "system" fn(*mut c_void, u64, *const c_void, *mut *mut c_void) -> i32,
    pub create_fence:
        unsafe extern "system" fn(*mut c_void, u64, u32, *const c_void, *mut *mut c_void) -> i32,
}

#[repr(C)]
pub struct D3D12DeviceX {
    pub header: ComHeader<DeviceVTable>,
}

static DEVICE_VTABLE: DeviceVTable = DeviceVTable {
    query_interface: dev_qi,
    add_ref: dev_add_ref,
    release: dev_release,
    create_command_queue: dev_create_command_queue,
    create_command_list: dev_create_command_list,
    create_committed_resource: dev_create_committed_resource,
    create_fence: dev_create_fence,
};

impl D3D12DeviceX {
    pub fn new_boxed() -> *mut Self {
        Box::into_raw(Box::new(Self {
            header: ComHeader {
                vtable: &DEVICE_VTABLE,
                ref_count: AtomicU32::new(1),
            },
        }))
    }
}

unsafe extern "system" fn dev_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_NOINTERFACE;
    }
    unsafe {
        dev_add_ref(this);
        *out = this;
    }
    S_OK
}

unsafe extern "system" fn dev_add_ref(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const D3D12DeviceX) };
    obj.header.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn dev_release(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const D3D12DeviceX) };
    let prev = obj.header.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut D3D12DeviceX)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn dev_create_command_queue(
    _this: *mut c_void,
    _desc: *const c_void,
    _riid: *const c_void,
    out_queue: *mut *mut c_void,
) -> i32 {
    if out_queue.is_null() {
        return E_NOINTERFACE;
    }
    unsafe { *out_queue = D3D12CommandQueueX::new_boxed() as *mut c_void };
    S_OK
}

unsafe extern "system" fn dev_create_command_list(
    _this: *mut c_void,
    _node_mask: u32,
    _list_type: u32,
    _alloc: *mut c_void,
    _init_pso: *mut c_void,
    _riid: *const c_void,
    out_list: *mut *mut c_void,
) -> i32 {
    if out_list.is_null() {
        return E_NOINTERFACE;
    }
    unsafe { *out_list = D3D12CommandListX::new_boxed() as *mut c_void };
    S_OK
}

unsafe extern "system" fn dev_create_committed_resource(
    _this: *mut c_void,
    size_bytes: u64,
    _riid: *const c_void,
    out_res: *mut *mut c_void,
) -> i32 {
    if out_res.is_null() {
        return E_NOINTERFACE;
    }
    unsafe { *out_res = D3D12ResourceX::new_boxed(size_bytes as usize, 0) as *mut c_void };
    S_OK
}

unsafe extern "system" fn dev_create_fence(
    _this: *mut c_void,
    initial_value: u64,
    _flags: u32,
    _riid: *const c_void,
    out_fence: *mut *mut c_void,
) -> i32 {
    if out_fence.is_null() {
        return E_NOINTERFACE;
    }
    unsafe { *out_fence = D3D12FenceX::new_boxed(initial_value) as *mut c_void };
    S_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn com_vtable_roundtrip_records_thin_commands_and_executes() {
        unsafe {
            let dev = D3D12DeviceX::new_boxed();
            let dev_vt = (*dev).header.vtable;

            let mut queue_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(
                (dev_vt.create_command_queue)(
                    dev as *mut c_void,
                    std::ptr::null(),
                    std::ptr::null(),
                    &mut queue_ptr
                ),
                S_OK
            );
            let queue = queue_ptr as *mut D3D12CommandQueueX;

            let mut list_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(
                (dev_vt.create_command_list)(
                    dev as *mut c_void,
                    0,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    &mut list_ptr
                ),
                S_OK
            );
            let list = list_ptr as *mut D3D12CommandListX;
            let list_vt = (*list).header.vtable;

            // Record through C-ABI COM vtable
            (list_vt.set_pipeline_state)(list_ptr, 0xCAFE_BABE);
            (list_vt.set_pipeline_state)(list_ptr, 0xCAFE_BABE); // redundant -> deduplicated
            let sgprs = [1u32, 2, 3, 4];
            (list_vt.set_compute_root_32bit_constants)(list_ptr, 0, 4, sgprs.as_ptr(), 0);
            (list_vt.dispatch)(list_ptr, 4, 1, 1);
            (list_vt.draw_indirect)(list_ptr, 0x7F00_0000, 10_000, 16);
            assert_eq!((list_vt.close)(list_ptr), S_OK);

            // Submit via CommandQueue COM vtable
            let lists = [list_ptr];
            ((*queue).header.vtable.execute_command_lists)(queue_ptr, 1, lists.as_ptr());

            assert_eq!((*queue).submitted_lists.load(Ordering::Relaxed), 1);
            assert_eq!((*queue).submitted_commands.load(Ordering::Relaxed), 4); // 1 bind + push + dispatch + indirect
            assert_eq!((*queue).hoisted_clamps.load(Ordering::Relaxed), 1);

            assert_eq!((list_vt.release)(list_ptr), 0);
            assert_eq!(((*queue).header.vtable.release)(queue_ptr), 0);
            assert_eq!((dev_vt.release)(dev as *mut c_void), 0);
        }
    }
}
