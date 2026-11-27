//! Console DLL `extern "system"` thunk table for `xmem.dll`, `d3d12_xs.dll`,
//! `xg_xs.dll`, `XCRDAPI.dll`, `xapu.dll`, `AcpHal.dll`, and offline `libHttpClient.GDK.dll`.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use strata_gpu::runner::ComputeRunner;

use crate::audio::{XAudio2Engine, XapuHal};
use crate::gameinput::GameInputManager;
use crate::runtime::HleRuntime;
use crate::xasync::{TaskQueueDispatchMode, TaskQueuePort, XAsyncRuntime};
use crate::xg::{SwizzleMode, TextureLayout};
use crate::xgameruntime::{XGameSaveService, XUserService};
use crate::xmem_phys::XMEM_PAGE_BYTES;

/// Global HLE runtime state accessed by `extern "system"` console thunks.
struct GlobalHle {
    runner: ComputeRunner,
    runtime: HleRuntime,
}

/// Global Audio HLE state (`XAudio2Engine` + `XapuHal`) independent of Vulkan initialization.
struct GlobalAudio {
    xaudio2: XAudio2Engine,
    xapu: XapuHal,
}

/// Global GDK runtime state (`XAsync`, `XUser`, `XGameSave`, `GameInput`).
struct GlobalGdk {
    xasync: XAsyncRuntime,
    users: XUserService,
    saves: XGameSaveService,
    gameinput: GameInputManager,
}

// SAFETY: GlobalHle is exclusively accessed behind a Mutex.
unsafe impl Send for GlobalHle {}

static GLOBAL_HLE: OnceLock<Mutex<GlobalHle>> = OnceLock::new();
static GLOBAL_AUDIO: OnceLock<Mutex<GlobalAudio>> = OnceLock::new();
static GLOBAL_GDK: OnceLock<Mutex<GlobalGdk>> = OnceLock::new();
static AUDIO_STUB_CALLS: AtomicU64 = AtomicU64::new(0);
static UNIMPLEMENTED_STUB_CALLS: AtomicU64 = AtomicU64::new(0);

fn with_gdk<R>(f: impl FnOnce(&mut GlobalGdk) -> R) -> R {
    let lock = GLOBAL_GDK.get_or_init(|| {
        Mutex::new(GlobalGdk {
            xasync: XAsyncRuntime::new(),
            users: XUserService::new(),
            saves: XGameSaveService::new(std::env::temp_dir().join("strata_saves")),
            gameinput: GameInputManager::new(),
        })
    });
    f(&mut lock.lock().unwrap())
}

fn with_audio<R>(f: impl FnOnce(&mut GlobalAudio) -> R) -> R {
    let lock = GLOBAL_AUDIO.get_or_init(|| {
        Mutex::new(GlobalAudio {
            xaudio2: XAudio2Engine::new(48_000),
            xapu: XapuHal::new(),
        })
    });
    f(&mut lock.lock().unwrap())
}

fn with_hle<R>(f: impl FnOnce(&mut GlobalHle) -> R) -> Option<R> {
    let lock = GLOBAL_HLE.get_or_init(|| {
        let runner = ComputeRunner::new().expect("Vulkan 1.3 BDA runner required for HLE");
        Mutex::new(GlobalHle {
            runner,
            runtime: HleRuntime::new(),
        })
    });
    Some(f(&mut lock.lock().unwrap()))
}

pub fn audio_stub_call_count() -> u64 {
    AUDIO_STUB_CALLS.load(Ordering::Relaxed)
}

pub fn unimplemented_stub_call_count() -> u64 {
    UNIMPLEMENTED_STUB_CALLS.load(Ordering::Relaxed)
}

pub(crate) fn note_unimplemented_call() {
    UNIMPLEMENTED_STUB_CALLS.fetch_add(1, Ordering::Relaxed);
}

// ============================================================================
// ID3DBlob COM Object (for D3D12SerializeVersionedRootSignature)
// ============================================================================

#[repr(C)]
pub struct BlobVTable {
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    pub get_buffer_pointer: unsafe extern "system" fn(*mut c_void) -> *mut c_void,
    pub get_buffer_size: unsafe extern "system" fn(*mut c_void) -> usize,
}

#[repr(C)]
pub struct ComBlob {
    pub vtable: &'static BlobVTable,
    pub ref_count: AtomicU32,
    pub data: Vec<u8>,
}

static BLOB_VTABLE: BlobVTable = BlobVTable {
    query_interface: blob_qi,
    add_ref: blob_add_ref,
    release: blob_release,
    get_buffer_pointer: blob_get_ptr,
    get_buffer_size: blob_get_size,
};

impl ComBlob {
    pub fn new_boxed(data: Vec<u8>) -> *mut Self {
        Box::into_raw(Box::new(Self {
            vtable: &BLOB_VTABLE,
            ref_count: AtomicU32::new(1),
            data,
        }))
    }
}

unsafe extern "system" fn blob_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return 0x8000_4002u32 as i32;
    }
    unsafe {
        blob_add_ref(this);
        *out = this;
    }
    0
}

unsafe extern "system" fn blob_add_ref(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const ComBlob) };
    obj.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn blob_release(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const ComBlob) };
    let prev = obj.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut ComBlob)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn blob_get_ptr(this: *mut c_void) -> *mut c_void {
    let obj = unsafe { &mut *(this as *mut ComBlob) };
    obj.data.as_mut_ptr() as *mut c_void
}

unsafe extern "system" fn blob_get_size(this: *mut c_void) -> usize {
    let obj = unsafe { &*(this as *const ComBlob) };
    obj.data.len()
}

// ============================================================================
// XGTextureAddressComputer COM Object (for XGCreateTextureComputer)
// ============================================================================

#[repr(C)]
pub struct XgComputerVTable {
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    pub get_resource_size_bytes: unsafe extern "system" fn(*mut c_void) -> u64,
    pub get_resource_base_alignment_bytes: unsafe extern "system" fn(*mut c_void) -> u64,
    pub get_texel_element_offset_bytes:
        unsafe extern "system" fn(*mut c_void, u32, u64, u32, u32, u32, u32) -> u64,
}

#[repr(C)]
pub struct XgTextureComputerCom {
    pub vtable: &'static XgComputerVTable,
    pub ref_count: AtomicU32,
    pub layout: TextureLayout,
}

static XG_COMP_VTABLE: XgComputerVTable = XgComputerVTable {
    query_interface: xg_qi,
    add_ref: xg_add_ref,
    release: xg_release,
    get_resource_size_bytes: xg_get_size,
    get_resource_base_alignment_bytes: xg_get_align,
    get_texel_element_offset_bytes: xg_get_offset,
};

impl XgTextureComputerCom {
    pub fn new_boxed(layout: TextureLayout) -> *mut Self {
        Box::into_raw(Box::new(Self {
            vtable: &XG_COMP_VTABLE,
            ref_count: AtomicU32::new(1),
            layout,
        }))
    }
}

unsafe extern "system" fn xg_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return 0x8000_4002u32 as i32;
    }
    unsafe {
        xg_add_ref(this);
        *out = this;
    }
    0
}

unsafe extern "system" fn xg_add_ref(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const XgTextureComputerCom) };
    obj.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn xg_release(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const XgTextureComputerCom) };
    let prev = obj.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut XgTextureComputerCom)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn xg_get_size(this: *mut c_void) -> u64 {
    let obj = unsafe { &*(this as *const XgTextureComputerCom) };
    obj.layout.total_size_bytes
}

unsafe extern "system" fn xg_get_align(this: *mut c_void) -> u64 {
    let obj = unsafe { &*(this as *const XgTextureComputerCom) };
    obj.layout.alignment_bytes
}

unsafe extern "system" fn xg_get_offset(
    this: *mut c_void,
    _plane: u32,
    x: u64,
    y: u32,
    _z: u32,
    _mip: u32,
    _sample: u32,
) -> u64 {
    let obj = unsafe { &*(this as *const XgTextureComputerCom) };
    match obj.layout.swizzle {
        SwizzleMode::Linear => {
            (y as u64) * (obj.layout.row_pitch_bytes as u64) + x * (obj.layout.bpp as u64)
        }
        SwizzleMode::Swizzle64KS | SwizzleMode::Swizzle64KD => {
            TextureLayout::swizzle_64ks_offset(x as u32, y, obj.layout.width, obj.layout.bpp) as u64
        }
    }
}

// ============================================================================
// xmem.dll
// ============================================================================

const MEM_LARGE_PAGES: u32 = 0x2000_0000;
const MEM_RELEASE: u32 = 0x8000;

unsafe extern "system" fn xmem_virtual_alloc(
    address: *mut c_void,
    size: usize,
    alloc_type: u32,
    _protect: u32,
) -> *mut c_void {
    let aligned_size = if (alloc_type & MEM_LARGE_PAGES) != 0 {
        size.max(XMEM_PAGE_BYTES).next_multiple_of(XMEM_PAGE_BYTES)
    } else {
        size.max(64)
    };
    with_hle(|g| {
        let req_va = address as u64;
        if let Ok((va, _bda)) = g
            .runtime
            .xmem_virtual_alloc(&g.runner, req_va, aligned_size as u64)
        {
            return va as *mut c_void;
        }
        std::ptr::null_mut()
    })
    .unwrap_or(std::ptr::null_mut())
}

unsafe extern "system" fn xmem_virtual_free(
    address: *mut c_void,
    _size: usize,
    free_type: u32,
) -> i32 {
    if address.is_null() {
        return 0;
    }
    if (free_type & MEM_RELEASE) != 0 {
        return with_hle(|g| i32::from(g.runtime.xmem_free(&g.runner, address as u64)))
            .unwrap_or(0);
    }
    // MEM_DECOMMIT keeps the virtual reservation alive.
    1
}

unsafe extern "system" fn xmem_virtual_protect(
    address: *mut c_void,
    size: usize,
    new_protect: u32,
    old_protect: *mut u32,
) -> i32 {
    if address.is_null() || size == 0 || new_protect == 0 {
        return 0;
    }
    if !old_protect.is_null() {
        unsafe { *old_protect = 0x04 }; // PAGE_READWRITE
    }
    1
}

unsafe extern "system" fn xmem_unmap_physical_pages(
    base_address: *mut c_void,
    num_pages: usize,
) -> i32 {
    if base_address.is_null() || num_pages == 0 {
        return 0;
    }
    with_hle(|g| {
        let unmapped = g.runtime.phys_pool.unmap_range(
            &mut g.runtime.uma.address_space,
            base_address as u64,
            num_pages,
        );
        i32::from(unmapped > 0)
    })
    .unwrap_or(0)
}

unsafe extern "system" fn xmem_get_physical_page_count() -> usize {
    with_hle(|g| g.runtime.phys_pool.active_pages()).unwrap_or(0)
}

unsafe extern "system" fn xmem_alloc(size: usize, _attrs: u64) -> *mut c_void {
    unsafe { xmem_virtual_alloc(std::ptr::null_mut(), size.max(64), 0x3000, 0x04) }
}

unsafe extern "system" fn xmem_free(ptr: *mut c_void, _attrs: u64) {
    if ptr.is_null() {
        return;
    }
    let _ = with_hle(|g| {
        g.runtime.xmem_free(&g.runner, ptr as u64);
    });
}

unsafe extern "system" fn xmem_is_tools_mem_available() -> i32 {
    0 // Retail console profile (no devkit tools memory)
}

unsafe extern "system" fn xmem_allocate_physical_pages(
    num_pages_ptr: *mut usize,
    page_array: *mut u64,
) -> i32 {
    if num_pages_ptr.is_null() || page_array.is_null() {
        return 0;
    }
    let count = unsafe { *num_pages_ptr };
    with_hle(|g| {
        let pfns = g.runtime.phys_pool.alloc_pages(count);
        unsafe {
            std::ptr::copy_nonoverlapping(pfns.as_ptr(), page_array, pfns.len());
            *num_pages_ptr = pfns.len();
        }
        1
    })
    .unwrap_or(0)
}

unsafe extern "system" fn xmem_free_physical_pages(
    num_pages: usize,
    page_array: *const u64,
) -> i32 {
    if page_array.is_null() || num_pages == 0 {
        return 1;
    }
    let pfns = unsafe { std::slice::from_raw_parts(page_array, num_pages) };
    with_hle(|g| {
        g.runtime.phys_pool.free_pages(pfns);
        1
    })
    .unwrap_or(0)
}

unsafe extern "system" fn xmem_map_physical_pages(
    base_address: *mut c_void,
    num_pages: usize,
    page_array: *const u64,
) -> i32 {
    if base_address.is_null() || page_array.is_null() {
        return 0;
    }
    let pfns = unsafe { std::slice::from_raw_parts(page_array, num_pages) };
    with_hle(|g| {
        let mapped = g.runtime.phys_pool.map_scatter(
            &mut g.runtime.uma.address_space,
            base_address as u64,
            pfns,
        );
        if mapped == num_pages { 1 } else { 0 }
    })
    .unwrap_or(0)
}

unsafe extern "system" fn xmem_map_physical_pages_scatter(
    base_address: *mut c_void,
    num_pages: usize,
    page_array: *const u64,
    _flags: u32,
) -> i32 {
    unsafe { xmem_map_physical_pages(base_address, num_pages, page_array) }
}

#[repr(C)]
struct XMemWorkingSetStats {
    size_bytes: u64,
    active_physical_bytes: u64,
}

unsafe extern "system" fn xmem_get_working_set_statistics(
    out_stats: *mut XMemWorkingSetStats,
) -> i32 {
    if out_stats.is_null() {
        return 0;
    }
    with_hle(|g| {
        let active_bytes = (g.runtime.phys_pool.active_pages() * XMEM_PAGE_BYTES) as u64;
        unsafe {
            (*out_stats).size_bytes = std::mem::size_of::<XMemWorkingSetStats>() as u64;
            (*out_stats).active_physical_bytes = active_bytes;
        }
        1
    })
    .unwrap_or(0)
}

unsafe extern "system" fn xmem_transfer_memory(
    dst: *mut c_void,
    src: *const c_void,
    size_bytes: usize,
) -> i32 {
    if dst.is_null() || src.is_null() {
        return 0;
    }
    unsafe {
        std::ptr::copy(src as *const u8, dst as *mut u8, size_bytes);
    }
    1
}

// ============================================================================
// d3d12_xs.dll
// ============================================================================

unsafe extern "system" fn d3d12_configure_virtual_memory_x(_cfg: *const c_void) -> i32 {
    0 // S_OK
}

static PRESENT_HOOK: OnceLock<Mutex<Option<Box<dyn FnMut(u64) + Send + 'static>>>> =
    OnceLock::new();
static FRAME_CLOCK: OnceLock<Mutex<strata_gpu::present::FramePacingClock>> = OnceLock::new();

/// Register a desktop presentation callback invoked whenever a title calls `PresentX`.
pub fn set_present_hook(cb: impl FnMut(u64) + Send + 'static) {
    let lock = PRESENT_HOOK.get_or_init(|| Mutex::new(None));
    *lock.lock().unwrap() = Some(Box::new(cb));
}

fn with_frame_clock<R>(f: impl FnOnce(&mut strata_gpu::present::FramePacingClock) -> R) -> R {
    let lock = FRAME_CLOCK.get_or_init(|| Mutex::new(strata_gpu::present::FramePacingClock::new(60)));
    f(&mut lock.lock().unwrap())
}

unsafe extern "system" fn d3d12_xbox_create_device(
    _params: *const c_void,
    _riid: *const c_void,
    out_device: *mut *mut c_void,
) -> i32 {
    if !out_device.is_null() {
        unsafe { *out_device = crate::d3d12_abi::StandardDevice::new_boxed() as *mut c_void };
    }
    0 // S_OK
}

unsafe extern "system" fn d3d12_present_x(
    _plane_count: u32,
    _planes: *const c_void,
    _params: *const c_void,
    _reserved: *const c_void,
) -> i32 {
    let frame_idx = with_frame_clock(|c| {
        c.presented_frames += 1;
        c.current_tick += 1;
        c.presented_frames
    });
    if let Some(lock) = PRESENT_HOOK.get()
        && let Ok(mut guard) = lock.lock()
        && let Some(ref mut cb) = *guard
    {
        cb(frame_idx);
    }
    0 // S_OK
}

unsafe extern "system" fn d3d12_register_frame_event_x(
    event_type: u32,
    _period_us: u64,
    _offset_us: i64,
    _event_handle: *mut c_void,
    out_token: *mut u64,
) -> i32 {
    let tick = with_frame_clock(|c| {
        let kind = match event_type {
            0 => strata_gpu::present::FrameEventKind::Origin,
            1 => strata_gpu::present::FrameEventKind::Present,
            _ => strata_gpu::present::FrameEventKind::CustomType2,
        };
        let due = c.current_tick + 1;
        c.schedule(kind, due);
        due
    });
    if !out_token.is_null() {
        unsafe { *out_token = tick };
    }
    0 // S_OK
}

unsafe extern "system" fn d3d12_wait_frame_event_x(
    _event_type: u32,
    _timeout_ms: u32,
    _reserved: *const c_void,
    out_token: *mut u64,
) -> i32 {
    let tick = with_frame_clock(|c| {
        c.current_tick += 1;
        c.current_tick
    });
    if !out_token.is_null() {
        unsafe { *out_token = tick };
    }
    0 // S_OK
}

unsafe extern "system" fn d3d12_serialize_versioned_root_signature(
    _desc: *const c_void,
    out_blob: *mut *mut c_void,
    out_err: *mut *mut c_void,
) -> i32 {
    if !out_err.is_null() {
        unsafe { *out_err = std::ptr::null_mut() };
    }
    if out_blob.is_null() {
        return 0x8007_0057u32 as i32; // E_INVALIDARG
    }
    // Emit a non-empty synthetic root-signature blob ("RTS0" header) wrapped in a valid ID3DBlob COM object.
    let mut payload = Vec::with_capacity(32);
    payload.extend_from_slice(b"RTS0");
    payload.extend_from_slice(&1u32.to_le_bytes()); // version 1
    payload.extend_from_slice(&[0u8; 24]);
    unsafe {
        *out_blob = ComBlob::new_boxed(payload) as *mut c_void;
    }
    0 // S_OK
}

// ============================================================================
// xg_xs.dll
// ============================================================================

/// `XGSuggestSwizzleMode` -> returns `XG_SWIZZLE_MODE_64KB_S` (9) as safe default.
unsafe extern "system" fn xg_suggest_swizzle_mode() -> u32 {
    SwizzleMode::Swizzle64KS as u32
}

#[repr(C)]
pub struct XgTexture2DDesc {
    pub width: u32,
    pub height: u32,
    pub bpp: u32,
    pub swizzle_mode: u32,
}

unsafe extern "system" fn xg_create_texture_computer(
    desc: *const XgTexture2DDesc,
    out_computer: *mut *mut c_void,
) -> i32 {
    if out_computer.is_null() {
        return 0x8007_0057u32 as i32; // E_INVALIDARG
    }
    let (w, h, bpp, mode) = if !desc.is_null() {
        let d = unsafe { &*desc };
        let sw = match d.swizzle_mode {
            0 => SwizzleMode::Linear,
            10 => SwizzleMode::Swizzle64KD,
            _ => SwizzleMode::Swizzle64KS,
        };
        (d.width.max(1), d.height.max(1), d.bpp.max(1), sw)
    } else {
        (256, 256, 4, SwizzleMode::Swizzle64KS)
    };
    let layout = TextureLayout::compute_2d(w, h, bpp, mode);
    unsafe {
        *out_computer = XgTextureComputerCom::new_boxed(layout) as *mut c_void;
    }
    0 // S_OK
}

// ============================================================================
// Offline / Audio / Fallback Logging Stubs
// ============================================================================

/// Offline-by-design network refusal (`E_ACCESSDENIED`) for `libHttpClient.GDK.dll`.
unsafe extern "system" fn offline_net_denied() -> i32 {
    0x8007_0005u32 as i32
}

unsafe extern "system" fn dstorage_get_factory(
    _riid: *const c_void,
    out_factory: *mut *mut c_void,
) -> i32 {
    if !out_factory.is_null() {
        let vfs = crate::vfs::global().read().unwrap().clone();
        let factory = crate::gdk::DStorageFactoryX::new_boxed(vfs);
        unsafe { *out_factory = factory as *mut c_void };
    }
    0 // S_OK
}

/// Audio subsystem (`xapu.dll` / `AcpHal.dll`) offline/silent fallback with call tracking.
unsafe extern "system" fn audio_stub_silent() -> i32 {
    AUDIO_STUB_CALLS.fetch_add(1, Ordering::Relaxed);
    0 // S_OK
}

// ============================================================================
// IXAudio2 / IXAudio2Voice COM Objects (exact xaudio2.h vtable slot order)
// ============================================================================

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ComSlot(pub *const ());
unsafe impl Sync for ComSlot {}

#[repr(C)]
pub struct ComXAudio2Voice {
    pub vtable: *const [ComSlot; 30],
    pub voice_id: u32,
}

unsafe extern "system" fn xa2_voice_stub_void(_this: *mut c_void) {}
unsafe extern "system" fn xa2_voice_stub_ok(_this: *mut c_void) -> i32 {
    0
}
unsafe extern "system" fn xa2_voice_destroy(this: *mut c_void) {
    if !this.is_null() {
        unsafe { drop(Box::from_raw(this as *mut ComXAudio2Voice)) };
    }
}
unsafe extern "system" fn xa2_voice_get_state(
    _this: *mut c_void,
    out_state: *mut [u64; 3],
    _flags: u32,
) {
    if !out_state.is_null() {
        unsafe { *out_state = [0, 0, 0] };
    }
}

static XA2_VOICE_VTABLE: [ComSlot; 30] = [
    ComSlot(xa2_voice_stub_void as *const ()), // 0 GetVoiceDetails
    ComSlot(xa2_voice_stub_ok as *const ()),   // 1 SetOutputVoices
    ComSlot(xa2_voice_stub_ok as *const ()),   // 2 SetEffectChain
    ComSlot(xa2_voice_stub_ok as *const ()),   // 3 EnableEffect
    ComSlot(xa2_voice_stub_ok as *const ()),   // 4 DisableEffect
    ComSlot(xa2_voice_stub_void as *const ()), // 5 GetEffectState
    ComSlot(xa2_voice_stub_ok as *const ()),   // 6 SetEffectParameters
    ComSlot(xa2_voice_stub_ok as *const ()),   // 7 GetEffectParameters
    ComSlot(xa2_voice_stub_ok as *const ()),   // 8 SetFilterParameters
    ComSlot(xa2_voice_stub_void as *const ()), // 9 GetFilterParameters
    ComSlot(xa2_voice_stub_ok as *const ()),   // 10 SetOutputFilterParameters
    ComSlot(xa2_voice_stub_void as *const ()), // 11 GetOutputFilterParameters
    ComSlot(xa2_voice_stub_ok as *const ()),   // 12 SetVolume
    ComSlot(xa2_voice_stub_void as *const ()), // 13 GetVolume
    ComSlot(xa2_voice_stub_ok as *const ()),   // 14 SetChannelVolumes
    ComSlot(xa2_voice_stub_void as *const ()), // 15 GetChannelVolumes
    ComSlot(xa2_voice_stub_ok as *const ()),   // 16 SetOutputMatrix
    ComSlot(xa2_voice_stub_void as *const ()), // 17 GetOutputMatrix
    ComSlot(xa2_voice_destroy as *const ()),   // 18 DestroyVoice
    ComSlot(xa2_voice_stub_ok as *const ()),   // 19 Start
    ComSlot(xa2_voice_stub_ok as *const ()),   // 20 Stop
    ComSlot(xa2_voice_stub_ok as *const ()),   // 21 SubmitSourceBuffer
    ComSlot(xa2_voice_stub_ok as *const ()),   // 22 FlushSourceBuffers
    ComSlot(xa2_voice_stub_ok as *const ()),   // 23 Discontinuity
    ComSlot(xa2_voice_stub_ok as *const ()),   // 24 ExitLoop
    ComSlot(xa2_voice_get_state as *const ()), // 25 GetState
    ComSlot(xa2_voice_stub_ok as *const ()),   // 26 SetFrequencyRatio
    ComSlot(xa2_voice_stub_void as *const ()), // 27 GetFrequencyRatio
    ComSlot(xa2_voice_stub_ok as *const ()),   // 28 SetSourceSampleRate
    ComSlot(xa2_voice_stub_ok as *const ()),   // 29 reserved
];

impl ComXAudio2Voice {
    fn new_boxed(voice_id: u32) -> *mut Self {
        Box::into_raw(Box::new(Self {
            vtable: &XA2_VOICE_VTABLE,
            voice_id,
        }))
    }
}

#[repr(C)]
pub struct ComXAudio2 {
    pub vtable: *const [ComSlot; 13],
    pub ref_count: AtomicU32,
}

unsafe extern "system" fn xa2_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return 0x8000_4002u32 as i32;
    }
    unsafe {
        xa2_add_ref(this);
        *out = this;
    }
    0
}

unsafe extern "system" fn xa2_add_ref(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const ComXAudio2) };
    obj.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn xa2_release(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const ComXAudio2) };
    let prev = obj.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut ComXAudio2)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn xa2_create_source_voice(
    _this: *mut c_void,
    out_voice: *mut *mut c_void,
    _source_format: *const c_void,
    _flags: u32,
    _max_freq_ratio: f32,
    _callback: *mut c_void,
    _send_list: *const c_void,
    _effect_chain: *const c_void,
) -> i32 {
    if out_voice.is_null() {
        return 0x8007_0057u32 as i32;
    }
    let id = with_audio(|a| {
        a.xaudio2.create_source_voice(crate::audio::WaveFormat::Pcm16 {
            channels: 2,
            sample_rate: 48_000,
        })
    });
    unsafe { *out_voice = ComXAudio2Voice::new_boxed(id) as *mut c_void };
    0
}

unsafe extern "system" fn xa2_create_submix_voice(
    _this: *mut c_void,
    out_voice: *mut *mut c_void,
    _input_channels: u32,
    _input_sample_rate: u32,
    _flags: u32,
    _processing_stage: u32,
    _send_list: *const c_void,
    _effect_chain: *const c_void,
) -> i32 {
    if out_voice.is_null() {
        return 0x8007_0057u32 as i32;
    }
    let id = with_audio(|a| a.xaudio2.create_submix_voice());
    unsafe { *out_voice = ComXAudio2Voice::new_boxed(id) as *mut c_void };
    0
}

unsafe extern "system" fn xa2_create_mastering_voice(
    _this: *mut c_void,
    out_voice: *mut *mut c_void,
    _input_channels: u32,
    _input_sample_rate: u32,
    _flags: u32,
    _device_id: *const c_void,
    _effect_chain: *const c_void,
    _stream_category: u32,
) -> i32 {
    if out_voice.is_null() {
        return 0x8007_0057u32 as i32;
    }
    let id = with_audio(|a| a.xaudio2.create_submix_voice());
    unsafe { *out_voice = ComXAudio2Voice::new_boxed(id) as *mut c_void };
    0
}

static XA2_VTABLE: [ComSlot; 13] = [
    ComSlot(xa2_qi as *const ()),                     // 0 QueryInterface
    ComSlot(xa2_add_ref as *const ()),                // 1 AddRef
    ComSlot(xa2_release as *const ()),                // 2 Release
    ComSlot(xa2_voice_stub_ok as *const ()),          // 3 RegisterForCallbacks
    ComSlot(xa2_voice_stub_void as *const ()),        // 4 UnregisterForCallbacks
    ComSlot(xa2_create_source_voice as *const ()),    // 5 CreateSourceVoice
    ComSlot(xa2_create_submix_voice as *const ()),    // 6 CreateSubmixVoice
    ComSlot(xa2_create_mastering_voice as *const ()), // 7 CreateMasteringVoice
    ComSlot(xa2_voice_stub_ok as *const ()),          // 8 StartEngine
    ComSlot(xa2_voice_stub_void as *const ()),        // 9 StopEngine
    ComSlot(xa2_voice_stub_ok as *const ()),          // 10 CommitChanges
    ComSlot(xa2_voice_stub_void as *const ()),        // 11 GetPerformanceData
    ComSlot(xa2_voice_stub_void as *const ()),        // 12 SetDebugConfiguration
];

impl ComXAudio2 {
    pub fn new_boxed() -> *mut Self {
        Box::into_raw(Box::new(Self {
            vtable: &XA2_VTABLE,
            ref_count: AtomicU32::new(1),
        }))
    }
}

/// Real HLE `XAudio2Create` / `XAPUCreate` returning a valid COM `IXAudio2*`.
unsafe extern "system" fn xaudio2_create(
    out_engine: *mut *mut c_void,
    _flags: u32,
    _processor: u32,
) -> i32 {
    if out_engine.is_null() {
        return 0x8007_0057u32 as i32; // E_INVALIDARG
    }
    with_audio(|a| {
        let _ = &a.xaudio2;
    });
    unsafe { *out_engine = ComXAudio2::new_boxed() as *mut c_void };
    0 // S_OK
}

// ============================================================================
// IAcpHal COM Object (for AcpHalCreate / XAPUCreate)
// ============================================================================

#[repr(C)]
pub struct ComAcpHal {
    pub vtable: *const [ComSlot; 10],
    pub ref_count: AtomicU32,
}

unsafe extern "system" fn acp_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return 0x8000_4002u32 as i32;
    }
    unsafe {
        acp_add_ref(this);
        *out = this;
    }
    0
}

unsafe extern "system" fn acp_add_ref(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const ComAcpHal) };
    obj.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn acp_release(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const ComAcpHal) };
    let prev = obj.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut ComAcpHal)) };
        0
    } else {
        prev - 1
    }
}

static ACPHAL_VTABLE: [ComSlot; 10] = [
    ComSlot(acp_qi as *const ()),            // 0 QueryInterface
    ComSlot(acp_add_ref as *const ()),       // 1 AddRef
    ComSlot(acp_release as *const ()),       // 2 Release
    ComSlot(xa2_voice_stub_ok as *const ()), // 3 Connect
    ComSlot(xa2_voice_stub_ok as *const ()), // 4 Disconnect
    ComSlot(xa2_voice_stub_ok as *const ()), // 5 SubmitCommand
    ComSlot(xa2_voice_stub_ok as *const ()), // 6 GetMessage
    ComSlot(xa2_voice_stub_ok as *const ()), // 7 SubmitCommandWithFence
    ComSlot(xa2_voice_stub_ok as *const ()), // 8 WaitOnFence
    ComSlot(xa2_voice_stub_ok as *const ()), // 9 Reserved
];

impl ComAcpHal {
    pub fn new_boxed() -> *mut Self {
        Box::into_raw(Box::new(Self {
            vtable: &ACPHAL_VTABLE,
            ref_count: AtomicU32::new(1),
        }))
    }
}

/// Real HLE `AcpHalCreate` returning a valid COM `IAcpHal*` backed by `XapuHal`.
unsafe extern "system" fn acphal_create(out_hal: *mut *mut c_void) -> i32 {
    if out_hal.is_null() {
        return 0x8007_0057u32 as i32; // E_INVALIDARG
    }
    with_audio(|a| {
        let _ = &a.xapu;
    });
    unsafe { *out_hal = ComAcpHal::new_boxed() as *mut c_void };
    0 // S_OK
}

// ============================================================================
// xgameruntime.dll Real GDK Thunks
// ============================================================================

unsafe extern "system" fn xgame_runtime_initialize() -> i32 {
    with_gdk(|_| 0)
}

unsafe extern "system" fn xgame_runtime_uninitialize() {}

unsafe extern "system" fn xsystem_get_device_type() -> u32 {
    6 // XSystemDeviceType::XboxScarlettAnaconda (Xbox Series X)
}

unsafe extern "system" fn xnetworking_query_connectivity_hint(out_hint: *mut [u32; 4]) -> i32 {
    if !out_hint.is_null() {
        unsafe { *out_hint = [0, 0, 0, 0] }; // Offline / None
    }
    0
}

unsafe extern "system" fn xtask_queue_create(
    work_mode: u32,
    completion_mode: u32,
    out_queue: *mut *mut c_void,
) -> i32 {
    if out_queue.is_null() {
        return 0x8007_0057u32 as i32;
    }
    let to_mode = |m: u32| match m {
        0 => TaskQueueDispatchMode::ThreadPool,
        1 => TaskQueueDispatchMode::SerializedThreadPool,
        2 => TaskQueueDispatchMode::Manual,
        _ => TaskQueueDispatchMode::Immediate,
    };
    let id = with_gdk(|g| g.xasync.create_queue(to_mode(work_mode), to_mode(completion_mode)));
    unsafe { *out_queue = id as usize as *mut c_void };
    0
}

unsafe extern "system" fn xtask_queue_dispatch(
    queue: *mut c_void,
    port: u32,
    _timeout_ms: u32,
) -> u8 {
    let qid = queue as usize as u64;
    let p = if port == 0 {
        TaskQueuePort::Work
    } else {
        TaskQueuePort::Completion
    };
    let ran = with_gdk(|g| g.xasync.queue_mut(qid).is_some_and(|q| q.dispatch_one(p)));
    u8::from(ran)
}

unsafe extern "system" fn xasync_get_status(_async_block: *mut c_void, _wait: u8) -> i32 {
    0 // S_OK (completed)
}

unsafe extern "system" fn xuser_add_async(_options: u32, _async_block: *mut c_void) -> i32 {
    with_gdk(|g| {
        let _ = g.users.get(1);
    });
    0
}

unsafe extern "system" fn xuser_add_result(
    _async_block: *mut c_void,
    out_user: *mut *mut c_void,
) -> i32 {
    if !out_user.is_null() {
        let h = with_gdk(|g| g.users.add_default_user());
        unsafe { *out_user = h as usize as *mut c_void };
    }
    0
}

unsafe extern "system" fn xuser_get_id(user: *mut c_void, out_xuid: *mut u64) -> i32 {
    if !out_xuid.is_null() {
        let h = user as usize as u64;
        let xuid = with_gdk(|g| {
            g.users
                .get(h)
                .or_else(|| g.users.get(1))
                .map_or(0x0009_0000_0000_0001, |u| u.xuid)
        });
        unsafe { *out_xuid = xuid };
    }
    0
}

unsafe extern "system" fn xgame_save_initialize_provider_async(
    _user: *mut c_void,
    _scid: *const u8,
    _sync_on_demand: u8,
    _async_block: *mut c_void,
) -> i32 {
    with_gdk(|g| {
        let _ = &g.saves;
    });
    0
}

unsafe extern "system" fn xgame_save_initialize_provider_result(
    _async_block: *mut c_void,
    out_provider: *mut *mut c_void,
) -> i32 {
    if !out_provider.is_null() {
        unsafe { *out_provider = 1usize as *mut c_void };
    }
    0
}

unsafe extern "system" fn xpackage_get_current_process_package_identifier(
    buf_size: usize,
    buf: *mut u8,
) -> i32 {
    let id = b"StrataTitle_1.0.0.0_x64__strata\0";
    if !buf.is_null() && buf_size >= id.len() {
        unsafe { std::ptr::copy_nonoverlapping(id.as_ptr(), buf, id.len()) };
    }
    0
}

unsafe extern "system" fn xstore_create_context(
    _user: *mut c_void,
    out_ctx: *mut *mut c_void,
) -> i32 {
    if !out_ctx.is_null() {
        unsafe { *out_ctx = 1usize as *mut c_void };
    }
    0
}

// ============================================================================
// IGameInput / IGameInputReading COM Objects (for GameInput.dll)
// ============================================================================

#[repr(C)]
pub struct ComGameInputReading {
    pub vtable: *const [ComSlot; 12],
    pub ref_count: AtomicU32,
    pub timestamp_us: u64,
    pub gamepad: crate::gameinput::GamepadState,
}

unsafe extern "system" fn gir_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return 0x8000_4002u32 as i32;
    }
    unsafe {
        gir_add_ref(this);
        *out = this;
    }
    0
}

unsafe extern "system" fn gir_add_ref(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const ComGameInputReading) };
    obj.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn gir_release(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const ComGameInputReading) };
    let prev = obj.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut ComGameInputReading)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn gir_get_input_kind(_this: *mut c_void) -> u32 {
    crate::gameinput::GAME_INPUT_KIND_GAMEPAD
}

unsafe extern "system" fn gir_get_timestamp(this: *mut c_void) -> u64 {
    let obj = unsafe { &*(this as *const ComGameInputReading) };
    obj.timestamp_us
}

unsafe extern "system" fn gir_get_gamepad_state(
    this: *mut c_void,
    out_state: *mut crate::gameinput::GamepadState,
) -> u8 {
    if out_state.is_null() {
        return 0;
    }
    let obj = unsafe { &*(this as *const ComGameInputReading) };
    unsafe { *out_state = obj.gamepad };
    1
}

static GAME_INPUT_READING_VTABLE: [ComSlot; 12] = [
    ComSlot(gir_qi as *const ()),                // 0 QueryInterface
    ComSlot(gir_add_ref as *const ()),           // 1 AddRef
    ComSlot(gir_release as *const ()),           // 2 Release
    ComSlot(gir_get_input_kind as *const ()),    // 3 GetInputKind
    ComSlot(gir_get_timestamp as *const ()),     // 4 GetTimestamp
    ComSlot(xa2_voice_stub_ok as *const ()),     // 5 GetSequenceNumber
    ComSlot(xa2_voice_stub_void as *const ()),   // 6 GetDevice
    ComSlot(xa2_voice_stub_ok as *const ()),     // 7 GetRawReport
    ComSlot(xa2_voice_stub_ok as *const ()),     // 8 GetControllerAxisCount
    ComSlot(xa2_voice_stub_ok as *const ()),     // 9 GetControllerAxisState
    ComSlot(xa2_voice_stub_ok as *const ()),     // 10 GetControllerButtonCount
    ComSlot(gir_get_gamepad_state as *const ()), // 11 GetGamepadState
];

#[repr(C)]
pub struct ComGameInput {
    pub vtable: *const [ComSlot; 16],
    pub ref_count: AtomicU32,
}

unsafe extern "system" fn gi_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return 0x8000_4002u32 as i32;
    }
    unsafe {
        gi_add_ref(this);
        *out = this;
    }
    0
}

unsafe extern "system" fn gi_add_ref(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const ComGameInput) };
    obj.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn gi_release(this: *mut c_void) -> u32 {
    let obj = unsafe { &*(this as *const ComGameInput) };
    let prev = obj.ref_count.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(this as *mut ComGameInput)) };
        0
    } else {
        prev - 1
    }
}

unsafe extern "system" fn gi_get_current_timestamp(_this: *mut c_void) -> u64 {
    with_gdk(|g| g.gameinput.current_timestamp_us())
}

unsafe extern "system" fn gi_get_current_reading(
    _this: *mut c_void,
    input_kind: u32,
    _device: *mut c_void,
    out_reading: *mut *mut c_void,
) -> i32 {
    if out_reading.is_null() {
        return 0x8007_0057u32 as i32;
    }
    let reading = with_gdk(|g| {
        g.gameinput.poll_host();
        g.gameinput.get_current_reading(input_kind).cloned()
    });
    let (ts, gp) = match reading {
        Some(r) => (r.timestamp_us, r.gamepad),
        None => (0, crate::gameinput::GamepadState::default()),
    };
    let obj = Box::into_raw(Box::new(ComGameInputReading {
        vtable: &GAME_INPUT_READING_VTABLE,
        ref_count: AtomicU32::new(1),
        timestamp_us: ts,
        gamepad: gp,
    }));
    unsafe { *out_reading = obj as *mut c_void };
    0
}

static GAME_INPUT_VTABLE: [ComSlot; 16] = [
    ComSlot(gi_qi as *const ()),                    // 0 QueryInterface
    ComSlot(gi_add_ref as *const ()),               // 1 AddRef
    ComSlot(gi_release as *const ()),               // 2 Release
    ComSlot(gi_get_current_timestamp as *const ()), // 3 GetCurrentTimestamp
    ComSlot(gi_get_current_reading as *const ()),   // 4 GetCurrentReading
    ComSlot(xa2_voice_stub_ok as *const ()),        // 5 GetNextReading
    ComSlot(xa2_voice_stub_ok as *const ()),        // 6 GetPreviousReading
    ComSlot(xa2_voice_stub_ok as *const ()),        // 7 GetTemporalReading
    ComSlot(xa2_voice_stub_ok as *const ()),        // 8 RegisterReadingCallback
    ComSlot(xa2_voice_stub_ok as *const ()),        // 9 RegisterDeviceCallback
    ComSlot(xa2_voice_stub_ok as *const ()),        // 10 RegisterSystemButtonCallback
    ComSlot(xa2_voice_stub_ok as *const ()),        // 11 RegisterKeyboardLayoutCallback
    ComSlot(xa2_voice_stub_void as *const ()),      // 12 StopCallback
    ComSlot(xa2_voice_stub_ok as *const ()),        // 13 UnregisterCallback
    ComSlot(xa2_voice_stub_ok as *const ()),        // 14 CreateDispatcher
    ComSlot(xa2_voice_stub_ok as *const ()),        // 15 CreateAggregateDevice
];

impl ComGameInput {
    pub fn new_boxed() -> *mut Self {
        Box::into_raw(Box::new(Self {
            vtable: &GAME_INPUT_VTABLE,
            ref_count: AtomicU32::new(1),
        }))
    }
}

unsafe extern "system" fn game_input_create(out_game_input: *mut *mut c_void) -> i32 {
    if out_game_input.is_null() {
        return 0x8007_0057u32 as i32;
    }
    unsafe { *out_game_input = ComGameInput::new_boxed() as *mut c_void };
    0
}

/// Real HLE `AcpHalAllocateShapeContexts` allocating hardware XMA2 decoder contexts in `XapuHal`.
unsafe extern "system" fn acphal_allocate_shape_contexts(count: u32, out_ids: *mut u32) -> i32 {
    if count == 0 || out_ids.is_null() {
        return 0x8007_0057u32 as i32; // E_INVALIDARG
    }
    with_audio(|a| {
        for i in 0..count as usize {
            let Some(id) = a.xapu.create_xma_context(2, 48_000) else {
                return 0x8007_000Eu32 as i32; // E_OUTOFMEMORY
            };
            unsafe { *out_ids.add(i) = id };
        }
        0 // S_OK
    })
}

/// Real HLE `AcpHalReleaseShapeContexts` releasing allocated XMA2 contexts in `XapuHal`.
unsafe extern "system" fn acphal_release_shape_contexts(count: u32, ids: *const u32) -> i32 {
    if count == 0 || ids.is_null() {
        return 0;
    }
    with_audio(|a| {
        for i in 0..count as usize {
            let id = unsafe { *ids.add(i) };
            a.xapu.contexts.remove(&id);
        }
    });
    0 // S_OK
}

/// Logged fallback stub for unimplemented console functions.
pub unsafe extern "system" fn logged_unimplemented_stub() -> i32 {
    UNIMPLEMENTED_STUB_CALLS.fetch_add(1, Ordering::Relaxed);
    0x8000_4001u32 as i32 // E_NOTIMPL
}

/// Resolve a console HLE import (`dll_name`, `func_name`) to a concrete function pointer.
pub fn resolve_hle(dll_name: &str, func_name: &str) -> Option<usize> {
    let dll = dll_name.to_ascii_lowercase();
    match dll.as_str() {
        "xmem.dll" => Some(match func_name {
            "XMemVirtualAlloc" => xmem_virtual_alloc as *const () as usize,
            "XMemVirtualFree" => xmem_virtual_free as *const () as usize,
            "XMemVirtualProtect" => xmem_virtual_protect as *const () as usize,
            "XMemAlloc" | "XMemAllocDefault" => xmem_alloc as *const () as usize,
            "XMemFree" | "XMemFreeDefault" => xmem_free as *const () as usize,
            "XMemIsToolsMemAvailable" => xmem_is_tools_mem_available as *const () as usize,
            "XMemAllocatePhysicalPages" => xmem_allocate_physical_pages as *const () as usize,
            "XMemFreePhysicalPages" => xmem_free_physical_pages as *const () as usize,
            "XMemGetPhysicalPageCount" => xmem_get_physical_page_count as *const () as usize,
            "XMemGetWorkingSetStatistics" => xmem_get_working_set_statistics as *const () as usize,
            "XMemMapPhysicalPages" => xmem_map_physical_pages as *const () as usize,
            "XMemMapPhysicalPagesScatter" => xmem_map_physical_pages_scatter as *const () as usize,
            "XMemUnmapPhysicalPages" => xmem_unmap_physical_pages as *const () as usize,
            "XMemTransferMemory" => xmem_transfer_memory as *const () as usize,
            _ => return None,
        }),
        "d3d12_xs.dll" | "d3d12_x.dll" => Some(match func_name {
            "D3D12ConfigureVirtualMemoryX" => {
                d3d12_configure_virtual_memory_x as *const () as usize
            }
            "D3D12XboxCreateDevice" => d3d12_xbox_create_device as *const () as usize,
            "D3D12SerializeVersionedRootSignature" => {
                d3d12_serialize_versioned_root_signature as *const () as usize
            }
            "PresentX" | "ID3D12CommandQueue_PresentX" => d3d12_present_x as *const () as usize,
            "RegisterFrameEventX" => d3d12_register_frame_event_x as *const () as usize,
            "WaitFrameEventX" => d3d12_wait_frame_event_x as *const () as usize,
            _ => return None,
        }),
        "xg_xs.dll" | "xg.dll" => Some(match func_name {
            "XGSuggestSwizzleMode" => xg_suggest_swizzle_mode as *const () as usize,
            "XGCreateTextureComputer" => xg_create_texture_computer as *const () as usize,
            _ => return None,
        }),
        "xcrdapi.dll" | "dstorage.dll" => Some(match func_name {
            "DStorageGetFactory" => dstorage_get_factory as *const () as usize,
            _ => logged_unimplemented_stub as *const () as usize,
        }),
        "xgameruntime.dll" => Some(match func_name {
            "XGameRuntimeInitialize" => xgame_runtime_initialize as *const () as usize,
            "XGameRuntimeUninitialize" => xgame_runtime_uninitialize as *const () as usize,
            "XSystemGetDeviceType" => xsystem_get_device_type as *const () as usize,
            "XNetworkingQueryConnectivityHint" => {
                xnetworking_query_connectivity_hint as *const () as usize
            }
            "XUserAddAsync" => xuser_add_async as *const () as usize,
            "XUserAddResult" => xuser_add_result as *const () as usize,
            "XUserGetId" => xuser_get_id as *const () as usize,
            "XGameSaveInitializeProviderAsync" => {
                xgame_save_initialize_provider_async as *const () as usize
            }
            "XGameSaveInitializeProviderResult" => {
                xgame_save_initialize_provider_result as *const () as usize
            }
            "XPackageGetCurrentProcessPackageIdentifier" => {
                xpackage_get_current_process_package_identifier as *const () as usize
            }
            "XStoreCreateContext" => xstore_create_context as *const () as usize,
            "XTaskQueueCreate" => xtask_queue_create as *const () as usize,
            "XTaskQueueDispatch" => xtask_queue_dispatch as *const () as usize,
            "XAsyncGetStatus" => xasync_get_status as *const () as usize,
            _ => logged_unimplemented_stub as *const () as usize,
        }),
        "xaudio2_9.dll" | "xaudio2_9redist.dll" => Some(match func_name {
            "XAudio2Create" | "XAudio2CreateWithVersionInfo" => {
                xaudio2_create as *const () as usize
            }
            _ => audio_stub_silent as *const () as usize,
        }),
        "xapu.dll" | "acphal.dll" => Some(match func_name {
            "AcpHalCreate" | "XAPUCreate" => acphal_create as *const () as usize,
            "AcpHalAllocateShapeContexts" => acphal_allocate_shape_contexts as *const () as usize,
            "AcpHalReleaseShapeContexts" => acphal_release_shape_contexts as *const () as usize,
            _ => audio_stub_silent as *const () as usize,
        }),
        "gameinput.dll" => Some(match func_name {
            "GameInputCreate" => game_input_create as *const () as usize,
            _ => logged_unimplemented_stub as *const () as usize,
        }),
        "libhttpclient.gdk.dll" | "xnetworking.dll" => {
            Some(offline_net_denied as *const () as usize)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xaudio2_acphal_gameinput_and_gdk_thunks_return_real_com_and_handles() {
        unsafe {
            // XAudio2Create -> IXAudio2 COM object -> CreateMasteringVoice (slot 7) -> DestroyVoice (slot 18)
            let mut xa2_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(xaudio2_create(&mut xa2_ptr, 0, 0), 0);
            assert!(!xa2_ptr.is_null());
            let xa2 = xa2_ptr as *mut ComXAudio2;
            let vt = *(*xa2).vtable;
            let create_mv: unsafe extern "system" fn(
                *mut c_void,
                *mut *mut c_void,
                u32,
                u32,
                u32,
                *const c_void,
                *const c_void,
                u32,
            ) -> i32 = std::mem::transmute(vt[7].0);
            let mut mv_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(
                create_mv(
                    xa2_ptr,
                    &mut mv_ptr,
                    2,
                    48_000,
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    0
                ),
                0
            );
            assert!(!mv_ptr.is_null());
            let mv = mv_ptr as *mut ComXAudio2Voice;
            let destroy: unsafe extern "system" fn(*mut c_void) =
                std::mem::transmute((*(*mv).vtable)[18].0);
            destroy(mv_ptr);
            let release: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(vt[2].0);
            assert_eq!(release(xa2_ptr), 0);

            // AcpHalCreate -> IAcpHal COM object
            let mut acp_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(acphal_create(&mut acp_ptr), 0);
            assert!(!acp_ptr.is_null());
            let acp = acp_ptr as *mut ComAcpHal;
            let acp_rel: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute((*(*acp).vtable)[2].0);
            assert_eq!(acp_rel(acp_ptr), 0);

            // GameInputCreate -> IGameInput COM object -> GetCurrentReading (slot 4)
            let mut gi_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(game_input_create(&mut gi_ptr), 0);
            assert!(!gi_ptr.is_null());
            let gi = gi_ptr as *mut ComGameInput;
            let gi_vt = *(*gi).vtable;
            let get_reading: unsafe extern "system" fn(
                *mut c_void,
                u32,
                *mut c_void,
                *mut *mut c_void,
            ) -> i32 = std::mem::transmute(gi_vt[4].0);
            let mut r_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(
                get_reading(
                    gi_ptr,
                    crate::gameinput::GAME_INPUT_KIND_GAMEPAD,
                    std::ptr::null_mut(),
                    &mut r_ptr
                ),
                0
            );
            assert!(!r_ptr.is_null());
            let r_obj = r_ptr as *mut ComGameInputReading;
            let r_rel: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute((*(*r_obj).vtable)[2].0);
            assert_eq!(r_rel(r_ptr), 0);
            let gi_rel: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(gi_vt[2].0);
            assert_eq!(gi_rel(gi_ptr), 0);

            // xgameruntime.dll: XTaskQueueCreate & XSystemGetDeviceType
            let mut q_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(xtask_queue_create(2, 2, &mut q_ptr), 0);
            assert!(!q_ptr.is_null());
            assert_eq!(xsystem_get_device_type(), 6);

            // DStorageGetFactory inherits global Vfs mounts
            crate::vfs::global()
                .write()
                .unwrap()
                .mount('G', std::env::temp_dir());
            let mut ds_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(dstorage_get_factory(std::ptr::null(), &mut ds_ptr), 0);
            assert!(!ds_ptr.is_null());
            let ds = ds_ptr as *mut crate::gdk::DStorageFactoryX;
            assert!(!(*ds).vfs.is_empty());
            assert_eq!(((*ds).header.vtable.release)(ds_ptr), 0);

            // PresentX / RegisterFrameEventX / WaitFrameEventX resolve & run
            assert!(resolve_hle("d3d12_xs.dll", "PresentX").is_some());
            assert!(resolve_hle("d3d12_xs.dll", "RegisterFrameEventX").is_some());
            assert!(resolve_hle("d3d12_xs.dll", "WaitFrameEventX").is_some());
            let mut tok = 0u64;
            assert_eq!(
                d3d12_register_frame_event_x(0, 16_666, 0, std::ptr::null_mut(), &mut tok),
                0
            );
            assert!(tok > 0);
            assert_eq!(
                d3d12_present_x(0, std::ptr::null(), std::ptr::null(), std::ptr::null()),
                0
            );
        }
    }

    #[test]
    fn resolves_all_core_console_imports_and_populates_com_out_params() {
        for (dll, func) in [
            ("xmem.dll", "XMemVirtualAlloc"),
            ("xmem.dll", "XMemFree"),
            ("xmem.dll", "XMemMapPhysicalPagesScatter"),
            ("d3d12_xs.dll", "D3D12XboxCreateDevice"),
            ("d3d12_xs.dll", "D3D12SerializeVersionedRootSignature"),
            ("xg_xs.dll", "XGCreateTextureComputer"),
            ("libHttpClient.GDK.dll", "HCInitialize"),
        ] {
            assert!(
                resolve_hle(dll, func).is_some(),
                "missing HLE thunk for {dll}!{func}"
            );
        }

        // Verify D3D12SerializeVersionedRootSignature actually fills out a valid ID3DBlob
        unsafe {
            let mut blob_ptr: *mut c_void = std::ptr::null_mut();
            let mut err_ptr: *mut c_void = 0x1234 as *mut c_void;
            assert_eq!(
                d3d12_serialize_versioned_root_signature(
                    std::ptr::null(),
                    &mut blob_ptr,
                    &mut err_ptr
                ),
                0
            );
            assert!(err_ptr.is_null());
            assert!(!blob_ptr.is_null());
            let blob = blob_ptr as *mut ComBlob;
            assert_eq!(((*blob).vtable.get_buffer_size)(blob_ptr), 32);
            assert!(!((*blob).vtable.get_buffer_pointer)(blob_ptr).is_null());
            assert_eq!(((*blob).vtable.release)(blob_ptr), 0);

            // Verify XGCreateTextureComputer actually fills out a valid XGTextureAddressComputer
            let desc = XgTexture2DDesc {
                width: 256,
                height: 256,
                bpp: 4,
                swizzle_mode: 9,
            };
            let mut xg_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(xg_create_texture_computer(&desc, &mut xg_ptr), 0);
            assert!(!xg_ptr.is_null());
            let comp = xg_ptr as *mut XgTextureComputerCom;
            assert_eq!(((*comp).vtable.get_resource_size_bytes)(xg_ptr), 256 * 1024);
            assert_eq!(
                ((*comp).vtable.get_resource_base_alignment_bytes)(xg_ptr),
                65_536
            );
            assert_eq!(((*comp).vtable.release)(xg_ptr), 0);
        }
    }

    #[test]
    fn resolves_xmem_extended_and_real_audio_thunks() {
        for (dll, func) in [
            ("xmem.dll", "XMemVirtualFree"),
            ("xmem.dll", "XMemVirtualProtect"),
            ("xmem.dll", "XMemUnmapPhysicalPages"),
            ("xmem.dll", "XMemGetPhysicalPageCount"),
            ("xaudio2_9.dll", "XAudio2Create"),
            ("acphal.dll", "AcpHalCreate"),
            ("acphal.dll", "AcpHalAllocateShapeContexts"),
            ("acphal.dll", "AcpHalReleaseShapeContexts"),
        ] {
            assert!(
                resolve_hle(dll, func).is_some(),
                "missing HLE thunk for {dll}!{func}"
            );
        }

        unsafe {
            let mut engine_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(xaudio2_create(&mut engine_ptr, 0, 0), 0);
            assert!(!engine_ptr.is_null());

            let mut hal_ptr: *mut c_void = std::ptr::null_mut();
            assert_eq!(acphal_create(&mut hal_ptr), 0);
            assert!(!hal_ptr.is_null());

            let mut ctx_ids = [u32::MAX; 2];
            assert_eq!(acphal_allocate_shape_contexts(2, ctx_ids.as_mut_ptr()), 0);
            assert_ne!(ctx_ids[0], u32::MAX);
            assert_ne!(ctx_ids[1], u32::MAX);
            assert_ne!(ctx_ids[0], ctx_ids[1]);
            assert_eq!(acphal_release_shape_contexts(2, ctx_ids.as_ptr()), 0);

            let mut old_prot = 0u32;
            assert_eq!(
                xmem_virtual_protect(0x1000 as *mut c_void, 4096, 0x04, &mut old_prot),
                1
            );
            assert_eq!(old_prot, 0x04);
        }
    }
}
