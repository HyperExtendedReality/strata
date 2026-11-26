//! Clean-room COM vtable objects for `d3d12_xs.dll` (`ID3D12Device`, `ID3D12CommandQueue`,
//! `ID3D12GraphicsCommandList`) translating D3D12.X calls directly into Strata's
//! No-API `ThinCommandList` + `UmaManager` + Vulkan 1.3 BDA `ComputeRunner`.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU32, Ordering};

use strata_gpu::present::FramePacingClock;
use strata_gpu::runner::ComputeRunner;
use strata_gpu::thin_list::{PipelineKey, RenderTargetDesc, ThinCommand, ThinCommandList};
use strata_runtime_shaders::CLAMP_INDIRECT_SPV;

const S_OK: i32 = 0;
const E_NOINTERFACE: i32 = 0x8000_4002u32 as i32;

// ============================================================================
// ID3D12ResourceX, ID3D12PipelineStateX, ID3D12FenceX COM Objects
// ============================================================================

#[repr(C)]
pub struct ResourceVTable {
    pub query_interface: unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
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
        let va = if gpu_va != 0 { gpu_va } else { backing.as_ptr() as u64 };
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

unsafe extern "system" fn res_qi(this: *mut c_void, _riid: *const c_void, out: *mut *mut c_void) -> i32 {
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

unsafe extern "system" fn res_unmap(_this: *mut c_void, _subresource: u32, _written_range: *const c_void) {}

unsafe extern "system" fn res_get_gpu_va(this: *mut c_void) -> u64 {
    let obj = unsafe { &*(this as *const D3D12ResourceX) };
    obj.gpu_va
}

#[repr(C)]
pub struct FenceVTable {
    pub query_interface: unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
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

unsafe extern "system" fn fence_qi(this: *mut c_void, _riid: *const c_void, out: *mut *mut c_void) -> i32 {
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
    pub query_interface: unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    pub close: unsafe extern "system" fn(*mut c_void) -> i32,
    pub reset: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> i32,
    pub set_pipeline_state: unsafe extern "system" fn(*mut c_void, u64),
    pub set_compute_root_32bit_constants: unsafe extern "system" fn(*mut c_void, u32, u32, *const u32, u32),
    pub dispatch: unsafe extern "system" fn(*mut c_void, u32, u32, u32),
    pub draw_instanced: unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32),
    pub draw_indexed_instanced: unsafe extern "system" fn(*mut c_void, u32, u32, u32, i32, u32),
    pub draw_indirect: unsafe extern "system" fn(*mut c_void, u64, u32, u32),
    pub om_set_render_targets: unsafe extern "system" fn(*mut c_void, u32, *const RenderTargetDesc),
    pub resource_barrier: unsafe extern "system" fn(*mut c_void, u64, u32, u32),
    pub copy_buffer_region: unsafe extern "system" fn(*mut c_void, u64, u64, u64, u64, u64),
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

unsafe extern "system" fn list_qi(this: *mut c_void, _riid: *const c_void, out: *mut *mut c_void) -> i32 {
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

unsafe extern "system" fn list_reset(this: *mut c_void, _alloc: *mut c_void, _pso: *mut c_void) -> i32 {
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
    obj.list.lock().unwrap().dispatch(x, y, z, [64, 1, 1], false);
}

unsafe extern "system" fn list_draw_instanced(
    this: *mut c_void,
    vertex_count: u32,
    instance_count: u32,
    start_vertex: u32,
    start_instance: u32,
) {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    obj.list
        .lock()
        .unwrap()
        .draw_instanced(vertex_count, instance_count, start_vertex, start_instance);
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

unsafe extern "system" fn list_draw_indirect(this: *mut c_void, args_bda: u64, max_verts: u32, max_inst: u32) {
    let obj = unsafe { &*(this as *const D3D12CommandListX) };
    obj.list.lock().unwrap().draw_indirect(args_bda, max_verts, max_inst);
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
    pub query_interface: unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
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
    pub pipelines: Mutex<HashMap<PipelineKey, Vec<u32>>>,
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
        }))
    }

    pub fn register_pipeline(&self, key: PipelineKey, spirv_words: Vec<u32>) {
        self.pipelines.lock().unwrap().insert(key, spirv_words);
    }
}

unsafe extern "system" fn queue_qi(this: *mut c_void, _riid: *const c_void, out: *mut *mut c_void) -> i32 {
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
        unsafe { drop(Box::from_raw(this as *mut D3D12CommandQueueX)) };
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
                        if let Some(spv) = active_pso.and_then(|k| pipes.get(&k)) {
                            let push: Vec<u8> = sgprs.iter().flat_map(|w| w.to_le_bytes()).collect();
                            if runner.dispatch_spirv(spv, &push, [*x, *y, *z]).is_ok() {
                                queue.executed_gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                            }
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
    pub query_interface: unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    pub create_command_queue: unsafe extern "system" fn(*mut c_void, *const c_void, *const c_void, *mut *mut c_void) -> i32,
    pub create_command_list: unsafe extern "system" fn(*mut c_void, u32, u32, *mut c_void, *mut c_void, *const c_void, *mut *mut c_void) -> i32,
    pub create_committed_resource: unsafe extern "system" fn(*mut c_void, u64, *const c_void, *mut *mut c_void) -> i32,
    pub create_fence: unsafe extern "system" fn(*mut c_void, u64, u32, *const c_void, *mut *mut c_void) -> i32,
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

unsafe extern "system" fn dev_qi(this: *mut c_void, _riid: *const c_void, out: *mut *mut c_void) -> i32 {
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
                (dev_vt.create_command_queue)(dev as *mut c_void, std::ptr::null(), std::ptr::null(), &mut queue_ptr),
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
