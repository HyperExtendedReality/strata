//! Clean-room COM vtable objects for `d3d12_xs.dll` (`ID3D12Device`, `ID3D12CommandQueue`,
//! `ID3D12GraphicsCommandList`) translating D3D12.X calls directly into Strata's
//! No-API `ThinCommandList` + `UmaManager` + Vulkan 1.3 BDA `ComputeRunner`.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use strata_gpu::graphics::DrawArgs;
use strata_gpu::images::ImageBindings;
use strata_gpu::present::FramePacingClock;
use strata_gpu::runner::ComputeRunner;
use strata_gpu::texture::TextureCache;
use strata_gpu::thin_list::{PipelineKey, RenderTargetDesc, ThinCommand, ThinCommandList};
use strata_gpu::uma::UmaManager;
use strata_runtime_shaders::CLAMP_INDIRECT_SPV;
use strata_shader::descriptor::ImageDescriptor;
use strata_shader::resource::{
    DescriptorUse, ImageBinding, Origin, ResourceKind, image_bindings, image_bindings_from,
    trace_descriptors,
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
fn readable(addr: u64, len: usize) -> bool {
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
    if addr == 0 {
        return false;
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
    info.state == MEM_COMMIT
        && info.protect & NO_READ == 0
        && addr as usize + len <= info.base + info.region_size
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
// ID3D12CommandQueueX COM VTable
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

#[repr(C)]
pub struct D3D12CommandQueueX {
    pub header: ComHeader<CommandQueueVTable>,
    pub submitted_lists: AtomicU32,
    pub submitted_commands: AtomicU32,
    pub hoisted_clamps: AtomicU32,
    pub executed_gpu_dispatches: AtomicU32,
    pub frame_clock: Mutex<FramePacingClock>,
    /// Optional live Vulkan 1.3 BDA runner + compiled SPIR-V pipeline table.
    pub gpu_runner: Option<Arc<ComputeRunner>>,
    pub pipelines: Mutex<HashMap<PipelineKey, PipelineEntry>>,
    /// Vulkan images for the T#s / S#s translated shaders use.
    pub textures: Mutex<TextureCache>,
    /// Render targets by console address, described like textures.
    pub render_targets: Mutex<HashMap<u64, [u32; 8]>>,
    pub executed_draws: AtomicU32,
    /// Console memory: mirrors are synced before each list and read back
    /// where a list asks for it.
    pub uma: Option<Arc<Mutex<UmaManager>>>,
    pub synced_bytes: std::sync::atomic::AtomicU64,
    pub readback_bytes: std::sync::atomic::AtomicU64,
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
        Box::into_raw(Box::new(Self {
            header: ComHeader {
                vtable: &CMD_QUEUE_VTABLE,
                ref_count: AtomicU32::new(1),
            },
            submitted_lists: AtomicU32::new(0),
            submitted_commands: AtomicU32::new(0),
            hoisted_clamps: AtomicU32::new(0),
            executed_gpu_dispatches: AtomicU32::new(0),
            frame_clock: Mutex::new(FramePacingClock::new(60)),
            gpu_runner,
            pipelines: Mutex::new(HashMap::new()),
            textures: Mutex::new(TextureCache::new()),
            render_targets: Mutex::new(HashMap::new()),
            executed_draws: AtomicU32::new(0),
            uma: None,
            synced_bytes: std::sync::atomic::AtomicU64::new(0),
            readback_bytes: std::sync::atomic::AtomicU64::new(0),
        }))
    }

    /// As [`Self::new_boxed_with_runner`], with console memory attached.
    pub fn new_boxed_with_memory(
        gpu_runner: Arc<ComputeRunner>,
        uma: Arc<Mutex<UmaManager>>,
    ) -> *mut Self {
        let q = Self::new_boxed_with_runner(Some(gpu_runner));
        // SAFETY: freshly boxed, not shared yet.
        unsafe { (*q).uma = Some(uma) };
        q
    }

    pub fn register_pipeline(&self, key: PipelineKey, spirv_words: Vec<u32>) {
        self.pipelines.lock().unwrap().insert(
            key,
            PipelineEntry {
                spirv: spirv_words,
                vertex: None,
                images: Vec::new(),
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
        let vs_opts = BuildOptions {
            stage: Stage::Vertex,
            ..Default::default()
        };
        let vs = strata_shader::compile_rdna2_with(vs_words, [1, 1, 1], &vs_opts)?;
        let vs_uses = trace_descriptors(&strata_shader::rdna2::decode_program(vs_words)?);
        let vs_bind = image_bindings(&vs_uses);
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
            require_full_quads: self.gpu_runner.as_ref().is_some_and(|r| r.quad_control),
            ps_input_vgprs: None,
        };
        let ps = strata_shader::compile_rdna2_with(ps_words, [1, 1, 1], &ps_opts)?;
        let ps_uses = trace_descriptors(&strata_shader::rdna2::decode_program(ps_words)?);
        let ps_bind = image_bindings_from(&ps_uses, tex_base, stor_base);
        let images = vs_bind
            .into_iter()
            .zip(vs_uses)
            .chain(ps_bind.into_iter().zip(ps_uses))
            .filter_map(|(b, u)| b.map(|b| (u, b)))
            .collect();
        self.pipelines.lock().unwrap().insert(
            key,
            PipelineEntry {
                spirv: ps.spirv,
                vertex: Some(vs.spirv),
                images,
            },
        );
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
        let compiled = strata_shader::compile_rdna2_to_spirv(rdna2_words, workgroup_size)?;
        let decoded = strata_shader::rdna2::decode_program(rdna2_words)?;
        let uses = trace_descriptors(&decoded);
        let images = image_bindings(&uses)
            .into_iter()
            .zip(uses)
            .filter_map(|(b, u)| b.map(|b| (u, b)))
            .collect();
        self.pipelines.lock().unwrap().insert(
            key,
            PipelineEntry {
                spirv: compiled.spirv,
                vertex: None,
                images,
            },
        );
        Ok(())
    }
}

/// A registered pipeline: its SPIR-V and the images it binds.
pub struct PipelineEntry {
    /// The compute shader, or the pixel shader of a graphics pipeline.
    pub spirv: Vec<u32>,
    /// The vertex shader of a graphics pipeline.
    pub vertex: Option<Vec<u32>>,
    /// Image instructions in program order and the slot each binds.
    pub images: Vec<(DescriptorUse, ImageBinding)>,
}

/// Reads and writes title memory by console address: through the UMA when an
/// allocation covers it (a mirror's host copy), else directly when readable.
struct TitleMemory<'a> {
    uma: Option<&'a Mutex<UmaManager>>,
}

impl TitleMemory<'_> {
    fn host(&self, addr: u64) -> u64 {
        self.uma
            .and_then(|u| u.lock().unwrap().host_ptr(addr))
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
        true
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
                let d = self.dwords(at, 2)?;
                Some(u64::from(d[0]) | (u64::from(d[1]) << 32))
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
                    .texture(runner, t, false, &read)
                    .map_err(|e| format!("pc {}: {e}", u.pc))?;
                let sampler = match (&u.sampler, u.kind) {
                    (Some(o), ResourceKind::SampledImage) => {
                        let sw = mem
                            .descriptor(o, 4, sgprs)
                            .ok_or_else(|| format!("pc {}: cannot read S# at {o}", u.pc))?;
                        Some(
                            textures
                                .sampler(runner, sw.try_into().unwrap())
                                .map_err(|e| e.to_string())?,
                        )
                    }
                    _ => None,
                };
                b.set_texture(slot, view, sampler);
            }
            ImageBinding::Storage(slot) => {
                let view = textures
                    .texture(runner, t, true, &read)
                    .map_err(|e| format!("pc {}: {e}", u.pc))?;
                b.set_storage(slot, view);
                writes.push(t);
            }
        }
    }
    Ok((b, writes))
}

impl D3D12CommandQueueX {
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

    /// Runs one recorded draw of a graphics `entry` into the render targets
    /// `rts` name, then writes them (and any storage images) back.
    fn run_draw(
        &self,
        runner: &ComputeRunner,
        entry: &PipelineEntry,
        sgprs: &[u32; 32],
        args: DrawArgs,
        rts: &[RenderTargetDesc],
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
        let rt_words: Vec<[u32; 8]> = {
            let known = self.render_targets.lock().unwrap();
            match rts
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
        let mut textures = self.textures.lock().unwrap();
        let (bindings, writes) = match bind_images(entry, sgprs, &mut textures, runner, &mem) {
            Ok(x) => x,
            Err(why) => {
                crate::crash::log_line(&format!("draw skipped: {why}"));
                return false;
            }
        };
        for w in &rt_words {
            if let Err(e) = textures.texture(runner, *w, false, &read) {
                crate::crash::log_line(&format!("draw skipped: render target: {e}"));
                return false;
            }
        }
        let ok = {
            let targets: Vec<&strata_gpu::images::GpuImage> =
                rt_words.iter().filter_map(|w| textures.image(w)).collect();
            runner
                .draw_spirv(vs, &entry.spirv, &push, args, &targets, &bindings)
                .is_ok()
        };
        for t in rt_words.iter().chain(&writes) {
            if let Err(e) = textures.write_back(runner, *t, &write, &read) {
                crate::crash::log_line(&format!("render target write-back: {e}"));
            }
        }
        ok
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
            q.textures.lock().unwrap().destroy(runner);
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
    for &ptr in slice {
        if ptr.is_null() {
            continue;
        }
        let cl = unsafe { &*(ptr as *const D3D12CommandListX) };
        let guard = cl.list.lock().unwrap();
        queue.submitted_lists.fetch_add(1, Ordering::Relaxed);
        queue
            .submitted_commands
            .fetch_add(guard.commands().len() as u32, Ordering::Relaxed);
        queue
            .hoisted_clamps
            .fetch_add(guard.hoisted_clamps.len() as u32, Ordering::Relaxed);

        // If a live Vulkan 1.3 BDA runner is attached, flush hoisted clamps + compute dispatches!
        if let Some(runner) = &queue.gpu_runner {
            // 0. The title's CPU writes since the last list reach the mirrors.
            if let Some(uma) = &queue.uma
                && let Ok(stats) = uma.lock().unwrap().sync_mirrors(runner)
            {
                queue.synced_bytes.fetch_add(stats.bytes, Ordering::Relaxed);
            }

            // 1. Batched pre-pass for all hoisted indirect draw clamps (single GPU dispatch)
            if !guard.hoisted_clamps.is_empty() {
                let count = guard.hoisted_clamps.len();
                if let Ok(mut entries_buf) = runner.alloc_bda_buffer((count * 16) as u64) {
                    unsafe {
                        let dst = entries_buf.as_mut_slice::<u32>(count * 4);
                        for (i, &(bda, max_c, max_i)) in guard.hoisted_clamps.iter().enumerate() {
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

            // 2. Replay recorded PushUserSgprs + Dispatch commands through registered SPIR-V pipelines
            let pipes = queue.pipelines.lock().unwrap();
            let mut active_pso: Option<PipelineKey> = None;
            let mut sgprs = [0u32; 32];
            let mut rts: Vec<RenderTargetDesc> = Vec::new();
            for cmd in guard.commands() {
                match cmd {
                    ThinCommand::BindPipeline(k) => active_pso = Some(*k),
                    ThinCommand::PushUserSgprs { offset_dw, values } => {
                        for (i, &v) in values.iter().enumerate() {
                            let idx = (*offset_dw as usize) + i;
                            if idx < 32 {
                                sgprs[idx] = v;
                            }
                        }
                    }
                    ThinCommand::Dispatch { x, y, z, .. } => {
                        if let Some(entry) = active_pso.and_then(|k| pipes.get(&k))
                            && queue.run_dispatch(runner, entry, &sgprs, [*x, *y, *z])
                        {
                            queue
                                .executed_gpu_dispatches
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    ThinCommand::SetRenderTargets(r) => rts = r.clone(),
                    ThinCommand::DrawInstanced {
                        vertex_count,
                        instance_count,
                        start_vertex,
                        start_instance,
                    } => {
                        if let Some(entry) = active_pso.and_then(|k| pipes.get(&k)) {
                            let args = DrawArgs {
                                vertex_count: *vertex_count,
                                instance_count: *instance_count,
                                first_vertex: *start_vertex,
                                first_instance: *start_instance,
                            };
                            if queue.run_draw(runner, entry, &sgprs, args, &rts) {
                                queue.executed_draws.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    ThinCommand::DispatchIndirect { args_va } => {
                        let groups = indirect_args(queue.uma.as_deref(), runner, *args_va);
                        if let (Some(g), Some(entry)) =
                            (groups, active_pso.and_then(|k| pipes.get(&k)))
                        {
                            // Never clamped: dispatches split past the device limit.
                            if queue.run_dispatch(runner, entry, &sgprs, g) {
                                queue
                                    .executed_gpu_dispatches
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    // Shader-written data the title's CPU code reads (e.g. ocean
                    // height maps its physics samples) goes back to title memory.
                    ThinCommand::ReadbackTexture {
                        console_va,
                        byte_len,
                    } => {
                        if let Some(uma) = &queue.uma
                            && let Ok(n) =
                                uma.lock().unwrap().readback(runner, *console_va, *byte_len)
                        {
                            queue.readback_bytes.fetch_add(n, Ordering::Relaxed);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

unsafe extern "system" fn queue_present_x(
    this: *mut c_void,
    _plane_count: u32,
    _planes: *const c_void,
    _params: *const c_void,
) -> i32 {
    let queue = unsafe { &*(this as *const D3D12CommandQueueX) };
    queue.frame_clock.lock().unwrap().record_present_x();
    S_OK
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
