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
    /// `IXAudio2EngineCallback`s (`RegisterForCallbacks`).
    engine_callbacks: Vec<usize>,
    /// Voice events raised outside a render pass (flushes), delivered by
    /// the audio thread.
    pending: Vec<crate::audio::CallbackEvent>,
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
            engine_callbacks: Vec::new(),
            pending: Vec::new(),
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
    let lock =
        FRAME_CLOCK.get_or_init(|| Mutex::new(strata_gpu::present::FramePacingClock::new(60)));
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

/// Whether `PresentX` shows frames in a desktop window (`strata-cli run --window`).
static WINDOW_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Shows presented frames in a window (created on the first `PresentX`,
/// on the presenting thread, sized to the first back buffer).
pub fn enable_window(on: bool) {
    WINDOW_ENABLED.store(on, Ordering::Release);
}

/// The window, its swapchain and the GPU they belong to.
struct Presenter {
    window: strata_gpu::window::Win32Window,
    display: Option<strata_gpu::display::Display>,
    composed: Option<strata_gpu::images::GpuImage>,
    runner: std::sync::Arc<strata_gpu::runner::ComputeRunner>,
}

// SAFETY: used only by the presenting thread, behind a mutex.
unsafe impl Send for Presenter {}

static FRAMES_SHOWN: AtomicU64 = AtomicU64::new(0);
static PRESENTER: Mutex<Option<Presenter>> = Mutex::new(None);

/// Whether the display can show HDR (for `XDisplayTryEnableHdrMode`).
pub fn display_supports_hdr() -> bool {
    PRESENTER
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|p| p.runner.hdr_colorspaces)
}

/// Present one primary resource per plane. The retail plane is 72 bytes:
/// a token, resource count and resource vector, followed by placement options.
/// Older single-plane callers may use the compact 16-byte prefix.
unsafe fn presented_resources(count: u32, planes: *const c_void) -> Vec<*mut c_void> {
    use crate::d3d12_abi::is_present_resource;
    use crate::d3d12x::readable;

    let find = |base: *const u8, count_offset: usize, list_offset: usize| {
        if !readable(base as u64, list_offset + size_of::<*const c_void>()) {
            return None;
        }
        let count = unsafe { base.add(count_offset).cast::<u32>().read_unaligned() };
        let list = unsafe {
            base.add(list_offset)
                .cast::<*const *mut c_void>()
                .read_unaligned()
        };
        if count == 0 || !readable(list as u64, size_of::<*mut c_void>()) {
            return None;
        }
        let resource = unsafe { list.read_unaligned() };
        unsafe { is_present_resource(resource) }.then_some(resource)
    };

    let mut resources = Vec::new();
    for index in 0..count.min(16) {
        let base = planes.cast::<u8>().wrapping_add(index as usize * 72);
        let resource =
            find(base, 8, 16).or_else(|| (index == 0).then(|| find(base, 4, 8)).flatten());
        if let Some(resource) = resource {
            resources.push(resource);
        } else if index == 0 {
            break;
        }
    }
    resources
}

/// Resolve presented colour surfaces through the device's existing texture cache.
fn show(resources: &[*mut c_void], frame_index: u64) -> Result<bool, String> {
    use crate::d3d12_abi::{Com, DeviceInner, ResourceInner};
    use strata_gpu::display::{Display, compose_planes};

    // SAFETY: presented_resources checked each resource's Strata COM vtable.
    let r = unsafe { &*resources[0].cast::<Com<ResourceInner>>() };
    if r.base.device.is_null() {
        return Ok(false);
    }
    let dev = unsafe { &*r.base.device.cast::<Com<DeviceInner>>() };
    let Some(backend) = &dev.inner.backend else {
        return Ok(false);
    };
    let mut guard = PRESENTER.lock().unwrap();
    if guard
        .as_ref()
        .is_some_and(|p| !std::sync::Arc::ptr_eq(&p.runner, &backend.runner))
    {
        if let Some(mut p) = guard.take() {
            if let Some(display) = p.display.take() {
                display.destroy(&p.runner);
            }
            if let Some(image) = p.composed.take() {
                p.runner.destroy_image(image);
            }
        }
    }
    if guard.is_none() {
        let (w, h) = (r.inner.desc.width as u32, r.inner.desc.height);
        let window = strata_gpu::window::Win32Window::create("Strata", w.max(64), h.max(64))
            .ok_or("cannot create a window")?;
        *guard = Some(Presenter {
            window,
            display: None,
            composed: None,
            runner: std::sync::Arc::clone(&backend.runner),
        });
    }
    let p = guard.as_mut().expect("presenter initialised");
    if !p.window.pump_messages() {
        crate::pipeline_cache::flush();
        std::process::exit(0);
    }
    let (w, h) = p.window.client_size();
    if p.display.is_none() {
        let handle = strata_gpu::swapchain::SurfaceHandle::Win32 {
            hinstance: p.window.hinstance,
            hwnd: p.window.hwnd,
        };
        p.display = Some(Display::new(&p.runner, handle, w, h).map_err(|e| e.to_string())?);
    }
    let read = |a: u64, n: usize| -> Option<Vec<u8>> {
        let uma = backend.uma.lock().unwrap();
        let host = uma.host_ptr_any(a)?;
        crate::d3d12x::readable(host as u64, n)
            .then(|| unsafe { std::slice::from_raw_parts(host, n) }.to_vec())
    };
    let mut textures = backend.textures.lock().unwrap();
    let mut descriptors = Vec::with_capacity(resources.len());
    for &resource in resources {
        let resource = unsafe { &*resource.cast::<Com<ResourceInner>>() };
        if resource.base.device != r.base.device {
            return Err("present planes span multiple devices".into());
        }
        let words = crate::d3d12_types::image_descriptor(
            &resource.inner.view(),
            0,
            strata_shader::descriptor::ImageKind::Tex2D,
            0,
            1,
            0,
            1,
            0x1688,
        )
        .ok_or("invalid present plane descriptor")?;
        textures
            .texture(&p.runner, words, false, &read)
            .map_err(|e| e.to_string())?;
        descriptors.push(words);
    }
    let planes: Vec<_> = descriptors
        .iter()
        .map(|words| textures.image(words).expect("present plane was uploaded"))
        .collect();
    let base = planes[0];
    let frame = if planes.len() == 1 {
        base
    } else {
        let width = planes.iter().map(|p| p.width).max().unwrap();
        let height = planes.iter().map(|p| p.height).max().unwrap();
        if p.composed
            .as_ref()
            .is_none_or(|i| (i.width, i.height, i.format) != (width, height, base.format))
        {
            let new = p
                .runner
                .create_image_2d(base.format, width, height, false, Default::default())
                .map_err(|e| e.to_string())?;
            if let Some(old) = p.composed.replace(new) {
                p.runner.destroy_image(old);
            }
        }
        let target = p.composed.as_ref().unwrap();
        compose_planes(&p.runner, &planes, target).map_err(|e| e.to_string())?;
        target
    };
    crate::frame_capture::finish(&p.runner, &planes, frame, frame_index);
    p.display
        .as_mut()
        .unwrap()
        .present(&p.runner, frame, w, h)
        .map_err(|e| e.to_string())
}

pub(crate) unsafe fn present_frame(plane_count: u32, planes: *const c_void) -> i32 {
    let frame_idx = with_frame_clock(|c| {
        c.presented_frames += 1;
        c.current_tick += 1;
        c.presented_frames
    });
    if WINDOW_ENABLED.load(Ordering::Acquire) {
        let resources = unsafe { presented_resources(plane_count, planes) };
        if !resources.is_empty() {
            match show(&resources, frame_idx) {
                Ok(true) => FRAMES_SHOWN.fetch_add(1, Ordering::Relaxed),
                Ok(false) => 0,
                Err(e) => {
                    crate::crash::log_line(&format!("present: {e}"));
                    0
                }
            };
        }
    }
    if let Some(lock) = PRESENT_HOOK.get()
        && let Ok(mut guard) = lock.lock()
        && let Some(ref mut cb) = *guard
    {
        cb(frame_idx);
    }
    if frame_idx == 1 || frame_idx.is_multiple_of(120) {
        crate::pipeline_cache::flush();
    }
    0
}

pub fn frame_counts() -> (u64, u64) {
    let calls = with_frame_clock(|clock| clock.presented_frames);
    (calls, FRAMES_SHOWN.load(Ordering::Relaxed))
}

unsafe extern "system" fn d3d12_present_x(
    plane_count: u32,
    planes: *const c_void,
    _params: *const c_void,
    _reserved: *const c_void,
) -> i32 {
    unsafe { present_frame(plane_count, planes) }
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

/// `D3D12SerializeVersionedRootSignature(desc, blob, error)`: a
/// `D3D12_VERSIONED_ROOT_SIGNATURE_DESC` (version dword, then the 1.0 or
/// 1.1 desc at offset 8) serialized as D3D12 does (a DXBC container with an
/// `RTS0` part), which `CreateRootSignature` parses back.
unsafe extern "system" fn d3d12_serialize_versioned_root_signature(
    desc: *const c_void,
    out_blob: *mut *mut c_void,
    out_err: *mut *mut c_void,
) -> i32 {
    if !out_err.is_null() {
        unsafe { *out_err = std::ptr::null_mut() };
    }
    if out_blob.is_null() {
        return 0x8007_0057u32 as i32; // E_INVALIDARG
    }
    let payload = if desc.is_null() {
        // No description: an empty root signature.
        let empty = [0u64; 5];
        unsafe { crate::d3d12_types::serialize_root_signature(empty.as_ptr() as *const u8, 1) }
    } else {
        let version = unsafe { *(desc as *const u32) };
        unsafe { crate::d3d12_types::serialize_root_signature((desc as *const u8).add(8), version) }
    };
    unsafe {
        *out_blob = ComBlob::new_boxed(payload) as *mut c_void;
    }
    0 // S_OK
}

/// `D3D12SerializeRootSignature(desc, version, blob, error)`.
unsafe extern "system" fn d3d12_serialize_root_signature(
    desc: *const c_void,
    version: u32,
    out_blob: *mut *mut c_void,
    out_err: *mut *mut c_void,
) -> i32 {
    if !out_err.is_null() {
        unsafe { *out_err = std::ptr::null_mut() };
    }
    if out_blob.is_null() || desc.is_null() {
        return 0x8007_0057u32 as i32; // E_INVALIDARG
    }
    let payload =
        unsafe { crate::d3d12_types::serialize_root_signature(desc as *const u8, version.max(1)) };
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
        unsafe { *out_factory = crate::dstorage::DStorageFactory::new_boxed(vfs) };
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

/// An `IXAudio2Voice` (voices are not COM objects: no `IUnknown`).
#[repr(C)]
pub struct ComXAudio2Voice {
    pub vtable: *const ComSlot,
    pub voice_id: u32,
    /// 0 source, 1 submix, 2 mastering.
    pub kind: u32,
    pub input_channels: u32,
    pub input_rate: u32,
}

fn xa2v<'a>(this: *mut c_void) -> &'a ComXAudio2Voice {
    // SAFETY: called from a voice's own vtable.
    unsafe { &*(this as *const ComXAudio2Voice) }
}

unsafe extern "system" fn xa2_voice_stub_void(_this: *mut c_void) {}
unsafe extern "system" fn xa2_voice_stub_ok(_this: *mut c_void) -> i32 {
    0
}

unsafe extern "system" fn xa2_voice_get_details(this: *mut c_void, out: *mut [u32; 4]) {
    let v = xa2v(this);
    if !out.is_null() {
        unsafe { *out = [0, 0, v.input_channels, v.input_rate] };
    }
}

/// `SetOutputVoices(const XAUDIO2_VOICE_SENDS*)`: `{SendCount, pSends}`,
/// each send `{Flags, pOutputVoice}`. The first send is the voice's target.
unsafe extern "system" fn xa2_voice_set_output_voices(
    this: *mut c_void,
    sends: *const [u64; 2],
) -> i32 {
    let id = xa2v(this).voice_id;
    let target = if sends.is_null() {
        None
    } else {
        let [count, list] = unsafe { *sends };
        if count & 0xffff_ffff == 0 || list == 0 {
            None
        } else {
            let first = unsafe { *(list as *const [u64; 2]) };
            (first[1] != 0).then(|| xa2v(first[1] as *mut c_void).voice_id)
        }
    };
    with_audio(|a| {
        if let Some(v) = a.xaudio2.sources.get_mut(&id) {
            v.submix_target = target;
        }
    });
    0
}

unsafe extern "system" fn xa2_voice_set_volume(this: *mut c_void, volume: f32, _op: u32) -> i32 {
    let id = xa2v(this).voice_id;
    with_audio(|a| {
        if let Some(v) = a.xaudio2.sources.get_mut(&id) {
            v.volume = volume;
        } else if let Some(s) = a.xaudio2.submixes.get_mut(&id) {
            s.volume = volume;
        }
    });
    0
}

unsafe extern "system" fn xa2_voice_get_volume(this: *mut c_void, out: *mut f32) {
    let id = xa2v(this).voice_id;
    let v = with_audio(|a| {
        a.xaudio2
            .sources
            .get(&id)
            .map(|v| v.volume)
            .or_else(|| a.xaudio2.submixes.get(&id).map(|s| s.volume))
            .unwrap_or(1.0)
    });
    if !out.is_null() {
        unsafe { *out = v };
    }
}

unsafe extern "system" fn xa2_voice_set_channel_volumes(
    this: *mut c_void,
    n: u32,
    volumes: *const f32,
    op: u32,
) -> i32 {
    // Strata mixes in stereo: the channels' mean is the voice's volume.
    if volumes.is_null() || n == 0 {
        return 0x8007_0057u32 as i32;
    }
    let v = unsafe { std::slice::from_raw_parts(volumes, n as usize) };
    unsafe { xa2_voice_set_volume(this, v.iter().sum::<f32>() / n as f32, op) }
}

unsafe extern "system" fn xa2_voice_get_channel_volumes(this: *mut c_void, n: u32, out: *mut f32) {
    let mut v = 1.0;
    unsafe { xa2_voice_get_volume(this, &mut v) };
    if !out.is_null() {
        for i in 0..n as usize {
            unsafe { *out.add(i) = v };
        }
    }
}

unsafe extern "system" fn xa2_voice_destroy(this: *mut c_void) {
    if this.is_null() {
        return;
    }
    let id = xa2v(this).voice_id;
    with_audio(|a| {
        a.xaudio2.destroy_voice(id);
        if a.xaudio2.mastering == id {
            a.xaudio2.mastering = 0;
        }
    });
    unsafe { drop(Box::from_raw(this as *mut ComXAudio2Voice)) };
}

unsafe extern "system" fn xa2_voice_start(this: *mut c_void, _flags: u32, _op: u32) -> i32 {
    let id = xa2v(this).voice_id;
    with_audio(|a| {
        if let Some(v) = a.xaudio2.sources.get_mut(&id) {
            v.playing = true;
        }
    });
    0
}

unsafe extern "system" fn xa2_voice_stop(this: *mut c_void, _flags: u32, _op: u32) -> i32 {
    let id = xa2v(this).voice_id;
    with_audio(|a| {
        if let Some(v) = a.xaudio2.sources.get_mut(&id) {
            v.playing = false;
        }
    });
    0
}

/// `XAUDIO2_BUFFER`: Flags, AudioBytes, pAudioData, PlayBegin, PlayLength,
/// LoopBegin, LoopLength, LoopCount, pContext (48 bytes).
#[repr(C)]
struct XAudio2Buffer {
    flags: u32,
    audio_bytes: u32,
    data: *const u8,
    play_begin: u32,
    play_length: u32,
    loop_begin: u32,
    loop_length: u32,
    loop_count: u32,
    context: *mut c_void,
}

unsafe extern "system" fn xa2_voice_submit_buffer(
    this: *mut c_void,
    buf: *const XAudio2Buffer,
    _wma: *const c_void,
) -> i32 {
    if buf.is_null() {
        return 0x8007_0057u32 as i32;
    }
    let b = unsafe { &*buf };
    if b.data.is_null() || b.audio_bytes == 0 {
        return 0x8007_0057u32 as i32;
    }
    // XAudio2 reads the data while it plays; Strata decodes it now.
    let data = unsafe { std::slice::from_raw_parts(b.data, b.audio_bytes as usize) }.to_vec();
    let id = xa2v(this).voice_id;
    with_audio(|a| {
        if let Some(v) = a.xaudio2.sources.get_mut(&id) {
            v.submit_buffer(crate::audio::AudioBuffer {
                data,
                loop_count: b.loop_count,
                end_of_stream: b.flags & 0x40 != 0, // XAUDIO2_END_OF_STREAM
                play_begin: b.play_begin,
                play_length: b.play_length,
                loop_begin: b.loop_begin,
                loop_length: b.loop_length,
                context: b.context as u64,
            });
        }
    });
    0
}

unsafe extern "system" fn xa2_voice_flush(this: *mut c_void) -> i32 {
    let id = xa2v(this).voice_id;
    with_audio(|a| {
        let mut ev = Vec::new();
        if let Some(v) = a.xaudio2.sources.get_mut(&id) {
            v.flush(&mut ev);
        }
        // Delivered by the audio thread, as XAudio2 does.
        a.pending.extend(ev);
    });
    0
}

unsafe extern "system" fn xa2_voice_discontinuity(this: *mut c_void) -> i32 {
    let id = xa2v(this).voice_id;
    with_audio(|a| {
        if let Some(v) = a.xaudio2.sources.get_mut(&id) {
            v.discontinuity();
        }
    });
    0
}

unsafe extern "system" fn xa2_voice_exit_loop(this: *mut c_void, _op: u32) -> i32 {
    let id = xa2v(this).voice_id;
    with_audio(|a| {
        if let Some(v) = a.xaudio2.sources.get_mut(&id) {
            v.exit_loop();
        }
    });
    0
}

/// `GetState(XAUDIO2_VOICE_STATE*, Flags)`: current buffer context, buffers
/// queued, samples played.
unsafe extern "system" fn xa2_voice_get_state(
    this: *mut c_void,
    out_state: *mut [u64; 3],
    _flags: u32,
) {
    if out_state.is_null() {
        return;
    }
    let id = xa2v(this).voice_id;
    let st = with_audio(|a| {
        a.xaudio2.sources.get(&id).map_or([0, 0, 0], |v| {
            [
                v.current_context(),
                u64::from(v.buffers_queued()),
                v.samples_played,
            ]
        })
    });
    unsafe { *out_state = st };
}

unsafe extern "system" fn xa2_voice_set_frequency_ratio(
    this: *mut c_void,
    ratio: f32,
    _op: u32,
) -> i32 {
    let id = xa2v(this).voice_id;
    with_audio(|a| {
        if let Some(v) = a.xaudio2.sources.get_mut(&id) {
            v.pitch_ratio = ratio.clamp(1.0 / 1024.0, 1024.0);
        }
    });
    0
}

unsafe extern "system" fn xa2_voice_get_frequency_ratio(this: *mut c_void, out: *mut f32) {
    let id = xa2v(this).voice_id;
    let r = with_audio(|a| a.xaudio2.sources.get(&id).map_or(1.0, |v| v.pitch_ratio));
    if !out.is_null() {
        unsafe { *out = r };
    }
}

unsafe extern "system" fn xa2_voice_set_source_sample_rate(this: *mut c_void, rate: u32) -> i32 {
    let id = xa2v(this).voice_id;
    with_audio(|a| {
        if let Some(v) = a.xaudio2.sources.get_mut(&id) {
            v.format = match v.format {
                crate::audio::WaveFormat::Pcm16 { channels, .. } => {
                    crate::audio::WaveFormat::Pcm16 {
                        channels,
                        sample_rate: rate,
                    }
                }
                crate::audio::WaveFormat::PcmF32 { channels, .. } => {
                    crate::audio::WaveFormat::PcmF32 {
                        channels,
                        sample_rate: rate,
                    }
                }
                crate::audio::WaveFormat::Xma2 { channels, .. } => crate::audio::WaveFormat::Xma2 {
                    channels,
                    sample_rate: rate,
                },
                crate::audio::WaveFormat::Opus { channels, .. } => crate::audio::WaveFormat::Opus {
                    channels,
                    sample_rate: rate,
                },
            };
        }
    });
    0
}

unsafe extern "system" fn xa2_master_get_channel_mask(_this: *mut c_void, out: *mut u32) -> i32 {
    if !out.is_null() {
        unsafe { *out = 3 }; // SPEAKER_STEREO
    }
    0
}

/// `IXAudio2Voice` slots 0..19, shared by every voice kind.
const fn voice_base<const N: usize>() -> [ComSlot; N] {
    let mut vt = [ComSlot(xa2_voice_stub_ok as *const ()); N];
    vt[0] = ComSlot(xa2_voice_get_details as *const ());
    vt[1] = ComSlot(xa2_voice_set_output_voices as *const ());
    vt[5] = ComSlot(xa2_voice_stub_void as *const ()); // GetEffectState
    vt[9] = ComSlot(xa2_voice_stub_void as *const ()); // GetFilterParameters
    vt[11] = ComSlot(xa2_voice_stub_void as *const ()); // GetOutputFilterParameters
    vt[12] = ComSlot(xa2_voice_set_volume as *const ());
    vt[13] = ComSlot(xa2_voice_get_volume as *const ());
    vt[14] = ComSlot(xa2_voice_set_channel_volumes as *const ());
    vt[15] = ComSlot(xa2_voice_get_channel_volumes as *const ());
    vt[17] = ComSlot(xa2_voice_stub_void as *const ()); // GetOutputMatrix
    vt[18] = ComSlot(xa2_voice_destroy as *const ());
    vt
}

static XA2_SOURCE_VTABLE: [ComSlot; 29] = {
    let mut vt = voice_base::<29>();
    vt[19] = ComSlot(xa2_voice_start as *const ());
    vt[20] = ComSlot(xa2_voice_stop as *const ());
    vt[21] = ComSlot(xa2_voice_submit_buffer as *const ());
    vt[22] = ComSlot(xa2_voice_flush as *const ());
    vt[23] = ComSlot(xa2_voice_discontinuity as *const ());
    vt[24] = ComSlot(xa2_voice_exit_loop as *const ());
    vt[25] = ComSlot(xa2_voice_get_state as *const ());
    vt[26] = ComSlot(xa2_voice_set_frequency_ratio as *const ());
    vt[27] = ComSlot(xa2_voice_get_frequency_ratio as *const ());
    vt[28] = ComSlot(xa2_voice_set_source_sample_rate as *const ());
    vt
};

static XA2_SUBMIX_VTABLE: [ComSlot; 19] = voice_base::<19>();

static XA2_MASTER_VTABLE: [ComSlot; 20] = {
    let mut vt = voice_base::<20>();
    vt[19] = ComSlot(xa2_master_get_channel_mask as *const ());
    vt
};

impl ComXAudio2Voice {
    fn new_boxed(voice_id: u32, kind: u32, input_channels: u32, input_rate: u32) -> *mut Self {
        let vtable = match kind {
            0 => XA2_SOURCE_VTABLE.as_ptr(),
            1 => XA2_SUBMIX_VTABLE.as_ptr(),
            _ => XA2_MASTER_VTABLE.as_ptr(),
        };
        Box::into_raw(Box::new(Self {
            vtable,
            voice_id,
            kind,
            input_channels,
            input_rate,
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

unsafe extern "system" fn xa2_register_for_callbacks(_this: *mut c_void, cb: *mut c_void) -> i32 {
    if cb.is_null() {
        return 0x8007_0057u32 as i32;
    }
    with_audio(|a| a.engine_callbacks.push(cb as usize));
    0
}

unsafe extern "system" fn xa2_unregister_for_callbacks(_this: *mut c_void, cb: *mut c_void) {
    with_audio(|a| a.engine_callbacks.retain(|&c| c != cb as usize));
}

unsafe extern "system" fn xa2_create_source_voice(
    _this: *mut c_void,
    out_voice: *mut *mut c_void,
    source_format: *const u8,
    _flags: u32,
    _max_freq_ratio: f32,
    callback: *mut c_void,
    send_list: *const [u64; 2],
    _effect_chain: *const c_void,
) -> i32 {
    if out_voice.is_null() || source_format.is_null() {
        return 0x8007_0057u32 as i32;
    }
    // WAVEFORMATEX(TENSIBLE): 18 bytes + cbSize.
    let cb_size = unsafe { (source_format.add(16) as *const u16).read_unaligned() } as usize;
    let fmt_bytes = unsafe { std::slice::from_raw_parts(source_format, 18 + cb_size.min(64)) };
    let Some(format) = crate::audio::WaveFormat::from_waveformatex(fmt_bytes) else {
        crate::crash::log_line(
            "CreateSourceVoice: unsupported WAVEFORMATEX (not PCM16 / float / XMA2)",
        );
        return 0x8896_0001u32 as i32; // XAUDIO2_E_INVALID_CALL
    };
    let id = with_audio(|a| {
        let id = a.xaudio2.create_source_voice(format);
        if let Some(v) = a.xaudio2.sources.get_mut(&id) {
            v.callback = callback as usize;
        }
        id
    });
    let voice = ComXAudio2Voice::new_boxed(id, 0, format.channels() as u32, format.sample_rate());
    if !send_list.is_null() {
        unsafe { xa2_voice_set_output_voices(voice.cast(), send_list) };
    }
    unsafe { *out_voice = voice as *mut c_void };
    0
}

unsafe extern "system" fn xa2_create_submix_voice(
    _this: *mut c_void,
    out_voice: *mut *mut c_void,
    input_channels: u32,
    input_sample_rate: u32,
    _flags: u32,
    _processing_stage: u32,
    _send_list: *const c_void,
    _effect_chain: *const c_void,
) -> i32 {
    if out_voice.is_null() {
        return 0x8007_0057u32 as i32;
    }
    let id = with_audio(|a| a.xaudio2.create_submix_voice());
    unsafe {
        *out_voice =
            ComXAudio2Voice::new_boxed(id, 1, input_channels, input_sample_rate) as *mut c_void
    };
    0
}

unsafe extern "system" fn xa2_create_mastering_voice(
    _this: *mut c_void,
    out_voice: *mut *mut c_void,
    input_channels: u32,
    input_sample_rate: u32,
    _flags: u32,
    _device_id: *const c_void,
    _effect_chain: *const c_void,
    _stream_category: u32,
) -> i32 {
    if out_voice.is_null() {
        return 0x8007_0057u32 as i32;
    }
    let id = with_audio(|a| {
        let id = a.xaudio2.create_submix_voice();
        a.xaudio2.mastering = id;
        id
    });
    start_audio_thread();
    let ch = if input_channels == 0 {
        2
    } else {
        input_channels
    };
    let rate = if input_sample_rate == 0 {
        48_000
    } else {
        input_sample_rate
    };
    unsafe { *out_voice = ComXAudio2Voice::new_boxed(id, 2, ch, rate) as *mut c_void };
    0
}

unsafe extern "system" fn xa2_start_engine(_this: *mut c_void) -> i32 {
    with_audio(|a| a.xaudio2.running = true);
    0
}

unsafe extern "system" fn xa2_stop_engine(_this: *mut c_void) {
    with_audio(|a| a.xaudio2.running = false);
}

/// `GetPerformanceData(XAUDIO2_PERFORMANCE_DATA*)`: zeroed but for the
/// active voice count (field 8 of 15 dword/qword fields).
unsafe extern "system" fn xa2_get_performance_data(_this: *mut c_void, out: *mut [u32; 20]) {
    if out.is_null() {
        return;
    }
    let voices = with_audio(|a| a.xaudio2.sources.values().filter(|v| v.playing).count() as u32);
    let mut d = [0u32; 20];
    d[9] = voices; // ActiveSourceVoiceCount
    unsafe { *out = d };
}

static XA2_VTABLE: [ComSlot; 13] = [
    ComSlot(xa2_qi as *const ()),                       // 0 QueryInterface
    ComSlot(xa2_add_ref as *const ()),                  // 1 AddRef
    ComSlot(xa2_release as *const ()),                  // 2 Release
    ComSlot(xa2_register_for_callbacks as *const ()),   // 3 RegisterForCallbacks
    ComSlot(xa2_unregister_for_callbacks as *const ()), // 4 UnregisterForCallbacks
    ComSlot(xa2_create_source_voice as *const ()),      // 5 CreateSourceVoice
    ComSlot(xa2_create_submix_voice as *const ()),      // 6 CreateSubmixVoice
    ComSlot(xa2_create_mastering_voice as *const ()),   // 7 CreateMasteringVoice
    ComSlot(xa2_start_engine as *const ()),             // 8 StartEngine
    ComSlot(xa2_stop_engine as *const ()),              // 9 StopEngine
    ComSlot(xa2_voice_stub_ok as *const ()),            // 10 CommitChanges
    ComSlot(xa2_get_performance_data as *const ()),     // 11 GetPerformanceData
    ComSlot(xa2_voice_stub_void as *const ()),          // 12 SetDebugConfiguration
];

impl ComXAudio2 {
    pub fn new_boxed() -> *mut Self {
        Box::into_raw(Box::new(Self {
            vtable: &XA2_VTABLE,
            ref_count: AtomicU32::new(1),
        }))
    }
}

/// Delivers voice callback events (`IXAudio2VoiceCallback` slots: 0 pass
/// start, 1 pass end, 2 stream end, 3 buffer start, 4 buffer end, 5 loop
/// end). Called with the audio state unlocked: callbacks submit buffers.
fn deliver(events: &[crate::audio::CallbackEvent]) {
    use crate::audio::VoiceEvent;
    for e in events {
        let cb = e.callback as *mut c_void;
        if cb.is_null() {
            continue;
        }
        unsafe {
            let vt = *(cb as *const *const ComSlot);
            let slot = |n: usize| (*vt.add(n)).0;
            match e.event {
                VoiceEvent::PassStart { bytes_required } => {
                    let f: unsafe extern "system" fn(*mut c_void, u32) =
                        std::mem::transmute(slot(0));
                    f(cb, bytes_required);
                }
                VoiceEvent::PassEnd => {
                    let f: unsafe extern "system" fn(*mut c_void) = std::mem::transmute(slot(1));
                    f(cb);
                }
                VoiceEvent::StreamEnd => {
                    let f: unsafe extern "system" fn(*mut c_void) = std::mem::transmute(slot(2));
                    f(cb);
                }
                VoiceEvent::BufferStart(c) | VoiceEvent::BufferEnd(c) | VoiceEvent::LoopEnd(c) => {
                    let n = match e.event {
                        VoiceEvent::BufferStart(_) => 3,
                        VoiceEvent::BufferEnd(_) => 4,
                        _ => 5,
                    };
                    let f: unsafe extern "system" fn(*mut c_void, *mut c_void) =
                        std::mem::transmute(slot(n));
                    f(cb, c as *mut c_void);
                }
            }
        }
    }
}

/// Calls each registered `IXAudio2EngineCallback` slot `n` (0 pass start, 1 pass end).
fn engine_callbacks(cbs: &[usize], n: usize) {
    for &cb in cbs {
        unsafe {
            let vt = *(cb as *const *const ComSlot);
            let f: unsafe extern "system" fn(*mut c_void) = std::mem::transmute((*vt.add(n)).0);
            f(cb as *mut c_void);
        }
    }
}

/// The audio thread: renders the voice graph in 10 ms passes into WASAPI
/// (paced by the device), or by the clock when no device opens, and
/// delivers callbacks between passes.
fn start_audio_thread() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("strata-audio".into())
            .spawn(|| {
                let rate = with_audio(|a| a.xaudio2.master_sample_rate);
                let mut sink = match crate::audio::WasapiSink::open(rate) {
                    Ok(s) => Some(s),
                    Err(hr) => {
                        crate::crash::log_line(&format!(
                            "audio: no WASAPI output (0x{hr:08x}): rendering silently"
                        ));
                        None
                    }
                };
                let pass = (rate / 100) as usize; // 10 ms
                let start = std::time::Instant::now();
                let mut rendered: u64 = 0;
                loop {
                    let frames = match &sink {
                        Some(s) => {
                            let free = s.writable_frames() as usize;
                            if free < pass {
                                std::thread::sleep(std::time::Duration::from_millis(2));
                                continue;
                            }
                            pass
                        }
                        None => {
                            // Keep real time without a device.
                            let due = start.elapsed().as_secs_f64() * f64::from(rate);
                            if (rendered as f64) + pass as f64 > due {
                                std::thread::sleep(std::time::Duration::from_millis(2));
                                continue;
                            }
                            pass
                        }
                    };
                    let (cbs, pending) = with_audio(|a| {
                        (a.engine_callbacks.clone(), std::mem::take(&mut a.pending))
                    });
                    deliver(&pending);
                    engine_callbacks(&cbs, 0);
                    let (pcm, events) = with_audio(|a| a.xaudio2.render(frames));
                    deliver(&events);
                    engine_callbacks(&cbs, 1);
                    if let Some(s) = &mut sink {
                        let _ = s.write_f32(&pcm);
                    }
                    rendered += frames as u64;
                }
            });
    });
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
    let id = with_gdk(|g| {
        g.xasync
            .create_queue(to_mode(work_mode), to_mode(completion_mode))
    });
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

/// Completes an `XAsyncBlock`: a manual queue gets the callback on its
/// completion port (run by `XTaskQueueDispatch`); other queues run it on a
/// worker thread.
pub(crate) fn complete_async(block: *mut c_void, status: i32, result: Vec<u8>) {
    crate::gdk_services::complete(block, status, result, |q, cb| {
        with_gdk(|g| match g.xasync.queue_mut(q) {
            Some(queue) if queue.completion_mode == TaskQueueDispatchMode::Manual => {
                queue.submit(TaskQueuePort::Completion, cb);
                true
            }
            _ => false,
        })
    });
}

/// Runs `work` off the calling thread, then completes `block` with its result.
fn complete_async_later(
    block: *mut c_void,
    work: impl FnOnce() -> (i32, Vec<u8>) + Send + 'static,
) {
    crate::gdk_services::begin(block);
    let addr = block as usize;
    let _ = std::thread::Builder::new()
        .name("strata-gameui".into())
        .spawn(move || {
            let (status, result) = work();
            complete_async(addr as *mut c_void, status, result);
        });
}

unsafe extern "system" fn xasync_get_status(async_block: *mut c_void, wait: u8) -> i32 {
    crate::gdk_services::status(async_block, wait != 0)
}

/// The status of a completed block (its `…Result` call's HRESULT).
fn async_status(block: *mut c_void) -> i32 {
    crate::gdk_services::result(block).map_or(crate::gdk_services::S_OK, |r| r.0)
}

unsafe extern "system" fn xgameui_show_message_dialog_async(
    async_block: *mut c_void,
    title: *const u8,
    content: *const u8,
    first: *const u8,
    second: *const u8,
    third: *const u8,
    _default_button: u32,
    cancel_button: u32,
) -> i32 {
    use crate::gdk_services::c_string;
    let title = c_string(title);
    let content = c_string(content);
    let buttons: Vec<String> = [first, second, third]
        .into_iter()
        .filter(|p| !p.is_null())
        .map(c_string)
        .collect();
    complete_async_later(async_block, move || {
        let b = crate::gdk_services::message_dialog(&title, &content, &buttons, cancel_button);
        (0, b.to_le_bytes().to_vec())
    });
    0
}

unsafe extern "system" fn xgameui_show_message_dialog_result(
    async_block: *mut c_void,
    out: *mut u32,
) -> i32 {
    let Some((status, r)) = crate::gdk_services::result(async_block) else {
        return crate::gdk_services::E_PENDING;
    };
    if !out.is_null() && r.len() >= 4 {
        unsafe { *out = u32::from_le_bytes(r[..4].try_into().unwrap()) };
    }
    crate::gdk_services::finish(async_block);
    status
}

unsafe extern "system" fn xgameui_show_error_dialog_async(
    async_block: *mut c_void,
    hr: i32,
    context: *const u8,
) -> i32 {
    let ctx = crate::gdk_services::c_string(context);
    complete_async_later(async_block, move || {
        let text = format!("Error 0x{:08X}\n{ctx}", hr as u32);
        crate::gdk_services::message_dialog("Error", &text, &["OK".into()], 0);
        (0, Vec::new())
    });
    0
}

/// `…Result(async)` of UIs that report only a status.
unsafe extern "system" fn xgameui_status_result(async_block: *mut c_void) -> i32 {
    let s = async_status(async_block);
    crate::gdk_services::finish(async_block);
    s
}

unsafe extern "system" fn xgameui_show_text_entry_async(
    async_block: *mut c_void,
    _title: *const u8,
    _description: *const u8,
    default_text: *const u8,
    _scope: u32,
    max_len: u32,
) -> i32 {
    // No on-screen keyboard offline: the entry is the default text.
    let mut text = crate::gdk_services::c_string(default_text).into_bytes();
    text.truncate(max_len as usize);
    crate::gdk_services::begin(async_block);
    complete_async(async_block, 0, text);
    0
}

unsafe extern "system" fn xgameui_show_text_entry_result_size(
    async_block: *mut c_void,
    out: *mut u32,
) -> i32 {
    let Some((status, r)) = crate::gdk_services::result(async_block) else {
        return crate::gdk_services::E_PENDING;
    };
    if !out.is_null() {
        unsafe { *out = r.len() as u32 + 1 };
    }
    status
}

unsafe extern "system" fn xgameui_show_text_entry_result(
    async_block: *mut c_void,
    size: u32,
    buf: *mut u8,
    used: *mut u32,
) -> i32 {
    let Some((status, r)) = crate::gdk_services::result(async_block) else {
        return crate::gdk_services::E_PENDING;
    };
    let mut u = 0usize;
    let hr = crate::gdk_services::write_c_string(
        &String::from_utf8_lossy(&r),
        size as usize,
        buf,
        &mut u,
    );
    if !used.is_null() {
        unsafe { *used = u as u32 };
    }
    crate::gdk_services::finish(async_block);
    if hr != 0 { hr } else { status }
}

/// Overlay UIs with nothing to show offline (achievements, profile cards):
/// they open and close at once.
unsafe extern "system" fn xgameui_show_overlay_async(async_block: *mut c_void) -> i32 {
    crate::gdk_services::begin(async_block);
    complete_async(async_block, 0, Vec::new());
    0
}

/// Online-only UIs (invites, web sign-in): cancelled, as a user would.
unsafe extern "system" fn xgameui_show_online_async(async_block: *mut c_void) -> i32 {
    crate::gdk_services::begin(async_block);
    complete_async(async_block, crate::gdk_services::E_ABORT, Vec::new());
    0
}

unsafe extern "system" fn xlaunch_uri(_user: *mut c_void, uri: *const u8) -> i32 {
    // Offline by design: logged, not opened.
    crate::crash::log_line(&format!(
        "XLaunchUri: {}",
        crate::gdk_services::c_string(uri)
    ));
    0
}

unsafe extern "system" fn xlaunch_new_game(exe: *const u8, args: *const u8, _user: *mut c_void) {
    crate::crash::log_line(&format!(
        "XLaunchNewGame: {} {} (the running title ends)",
        crate::gdk_services::c_string(exe),
        crate::gdk_services::c_string(args)
    ));
    std::process::exit(0);
}

unsafe extern "system" fn xlaunch_restart_on_crash(_args: *const u8, _reserved: u32) -> i32 {
    0
}

unsafe extern "system" fn xuser_add_async(_options: u32, async_block: *mut c_void) -> i32 {
    with_gdk(|g| {
        let _ = g.users.get(1);
    });
    // The offline user signs in at once.
    crate::gdk_services::begin(async_block);
    complete_async(async_block, 0, Vec::new());
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
    async_block: *mut c_void,
) -> i32 {
    with_gdk(|g| {
        let _ = &g.saves;
    });
    crate::gdk_services::begin(async_block);
    complete_async(async_block, 0, Vec::new());
    0
}

unsafe extern "system" fn xgame_save_initialize_provider_result(
    _async_block: *mut c_void,
    out_provider: *mut *mut c_void,
) -> i32 {
    if !out_provider.is_null() {
        unsafe { *out_provider = std::ptr::dangling_mut::<c_void>() };
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
        unsafe { *out_ctx = std::ptr::dangling_mut::<c_void>() };
    }
    0
}

// ============================================================================
// IGameInput / IGameInputReading / IGameInputDevice COM Objects (GameInput.dll)
// ============================================================================
//
// Slot orders follow the public `GameInput.h` (v0) interfaces.

const GAMEINPUT_E_READING_NOT_FOUND: i32 = 0x838A_0003u32 as i32;

#[repr(C)]
pub struct ComGameInputReading {
    pub vtable: *const [ComSlot; 25],
    pub ref_count: AtomicU32,
    pub reading: crate::gameinput::InputReading,
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

fn gir<'a>(this: *mut c_void) -> &'a crate::gameinput::InputReading {
    // SAFETY: called from this reading's own vtable.
    unsafe { &(*(this as *const ComGameInputReading)).reading }
}

unsafe extern "system" fn gir_get_input_kind(this: *mut c_void) -> u32 {
    gir(this).kind
}

unsafe extern "system" fn gir_get_sequence_number(this: *mut c_void, _kind: u32) -> u64 {
    gir(this).id
}

unsafe extern "system" fn gir_get_timestamp(this: *mut c_void) -> u64 {
    gir(this).timestamp_us
}

unsafe extern "system" fn gir_get_device(_this: *mut c_void, out: *mut *mut c_void) {
    if !out.is_null() {
        unsafe { *out = game_input_device() };
    }
}

unsafe extern "system" fn gir_zero_u32(_this: *mut c_void) -> u32 {
    0
}

unsafe extern "system" fn gir_false(_this: *mut c_void) -> u8 {
    0
}

unsafe extern "system" fn gir_get_raw_report(_this: *mut c_void, out: *mut *mut c_void) -> u8 {
    if !out.is_null() {
        unsafe { *out = std::ptr::null_mut() };
    }
    0
}

unsafe extern "system" fn gir_get_key_count(this: *mut c_void) -> u32 {
    gir(this).keys_down.len() as u32
}

/// `GetKeyState(count, GameInputKeyState*)`: scan code, code point, virtual
/// key, dead-key flag (12 bytes each).
unsafe extern "system" fn gir_get_key_state(
    this: *mut c_void,
    count: u32,
    out: *mut [u32; 3],
) -> u32 {
    let keys = &gir(this).keys_down;
    let n = keys.len().min(count as usize);
    for (i, &vk) in keys.iter().take(n).enumerate() {
        let scan = unsafe { MapVirtualKeyW(u32::from(vk), 0) };
        unsafe { *out.add(i) = [scan, 0, u32::from(vk)] };
    }
    n as u32
}

unsafe extern "system" fn gir_get_mouse_state(
    this: *mut c_void,
    out: *mut crate::gameinput::MouseState,
) -> u8 {
    let r = gir(this);
    if out.is_null() || r.kind & crate::gameinput::GAME_INPUT_KIND_MOUSE == 0 {
        return 0;
    }
    unsafe { *out = r.mouse };
    1
}

unsafe extern "system" fn gir_get_gamepad_state(
    this: *mut c_void,
    out_state: *mut crate::gameinput::GamepadState,
) -> u8 {
    let r = gir(this);
    if out_state.is_null() || r.kind & crate::gameinput::GAME_INPUT_KIND_GAMEPAD == 0 {
        return 0;
    }
    unsafe { *out_state = r.gamepad };
    1
}

/// `GetUiNavigationState`: the gamepad as UI navigation buttons (same bits
/// for menu/view/accept(A)/cancel(B)/dpad in GameInput's v0 layout order).
unsafe extern "system" fn gir_get_ui_navigation_state(this: *mut c_void, out: *mut u32) -> u8 {
    let r = gir(this);
    if out.is_null() || r.kind & crate::gameinput::GAME_INPUT_KIND_GAMEPAD == 0 {
        return 0;
    }
    unsafe { *out = r.gamepad.buttons };
    1
}

static GAME_INPUT_READING_VTABLE: [ComSlot; 25] = [
    ComSlot(gir_qi as *const ()),                      // 0 QueryInterface
    ComSlot(gir_add_ref as *const ()),                 // 1 AddRef
    ComSlot(gir_release as *const ()),                 // 2 Release
    ComSlot(gir_get_input_kind as *const ()),          // 3 GetInputKind
    ComSlot(gir_get_sequence_number as *const ()),     // 4 GetSequenceNumber
    ComSlot(gir_get_timestamp as *const ()),           // 5 GetTimestamp
    ComSlot(gir_get_device as *const ()),              // 6 GetDevice
    ComSlot(gir_get_raw_report as *const ()),          // 7 GetRawReport
    ComSlot(gir_zero_u32 as *const ()),                // 8 GetControllerAxisCount
    ComSlot(gir_zero_u32 as *const ()),                // 9 GetControllerAxisState
    ComSlot(gir_zero_u32 as *const ()),                // 10 GetControllerButtonCount
    ComSlot(gir_zero_u32 as *const ()),                // 11 GetControllerButtonState
    ComSlot(gir_zero_u32 as *const ()),                // 12 GetControllerSwitchCount
    ComSlot(gir_zero_u32 as *const ()),                // 13 GetControllerSwitchState
    ComSlot(gir_get_key_count as *const ()),           // 14 GetKeyCount
    ComSlot(gir_get_key_state as *const ()),           // 15 GetKeyState
    ComSlot(gir_get_mouse_state as *const ()),         // 16 GetMouseState
    ComSlot(gir_zero_u32 as *const ()),                // 17 GetTouchCount
    ComSlot(gir_zero_u32 as *const ()),                // 18 GetTouchState
    ComSlot(gir_false as *const ()),                   // 19 GetMotionState
    ComSlot(gir_false as *const ()),                   // 20 GetArcadeStickState
    ComSlot(gir_false as *const ()),                   // 21 GetFlightStickState
    ComSlot(gir_get_gamepad_state as *const ()),       // 22 GetGamepadState
    ComSlot(gir_false as *const ()),                   // 23 GetRacingWheelState
    ComSlot(gir_get_ui_navigation_state as *const ()), // 24 GetUiNavigationState
];

fn new_reading(r: crate::gameinput::InputReading) -> *mut c_void {
    Box::into_raw(Box::new(ComGameInputReading {
        vtable: &GAME_INPUT_READING_VTABLE,
        ref_count: AtomicU32::new(1),
        reading: r,
    })) as *mut c_void
}

/// The one gamepad device Strata exposes (a process-lifetime object).
#[repr(C)]
pub struct ComGameInputDevice {
    pub vtable: *const [ComSlot; 22],
    /// `GameInputDeviceInfo` (zero beyond the fields set: no optional tables).
    pub info: [u8; 512],
}

// SAFETY: immutable after creation.
unsafe impl Sync for ComGameInputDevice {}
unsafe impl Send for ComGameInputDevice {}

unsafe extern "system" fn gid_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return 0x8000_4002u32 as i32;
    }
    unsafe { *out = this };
    0
}

unsafe extern "system" fn gid_ref(_this: *mut c_void) -> u32 {
    1
}

unsafe extern "system" fn gid_get_device_info(this: *mut c_void) -> *const u8 {
    unsafe { (*(this as *const ComGameInputDevice)).info.as_ptr() }
}

unsafe extern "system" fn gid_get_device_status(_this: *mut c_void) -> u32 {
    // Connected (0x1) | AnyInput (0x10000000? kept to Connected: what
    // titles test).
    1
}

unsafe extern "system" fn gid_get_battery_state(_this: *mut c_void, out: *mut [u32; 6]) {
    if !out.is_null() {
        unsafe { *out = [0; 6] }; // unknown battery: wired
    }
}

unsafe extern "system" fn gid_set_rumble_state(_this: *mut c_void, params: *const [f32; 4]) {
    let p = if params.is_null() {
        [0.0; 4]
    } else {
        unsafe { *params }
    };
    with_gdk(|g| g.gameinput.set_rumble(p[0], p[1], p[2], p[3]));
}

unsafe extern "system" fn gid_unsupported(_this: *mut c_void) -> i32 {
    0x8000_4001u32 as i32 // E_NOTIMPL
}

unsafe extern "system" fn gid_void(_this: *mut c_void) {}

static GAME_INPUT_DEVICE_VTABLE: [ComSlot; 22] = [
    ComSlot(gid_qi as *const ()),                // 0 QueryInterface
    ComSlot(gid_ref as *const ()),               // 1 AddRef
    ComSlot(gid_ref as *const ()),               // 2 Release
    ComSlot(gid_get_device_info as *const ()),   // 3 GetDeviceInfo
    ComSlot(gid_get_device_status as *const ()), // 4 GetDeviceStatus
    ComSlot(gid_get_battery_state as *const ()), // 5 GetBatteryState
    ComSlot(gid_unsupported as *const ()),       // 6 CreateForceFeedbackEffect
    ComSlot(gir_false as *const ()),             // 7 IsForceFeedbackMotorPoweredOn
    ComSlot(gid_void as *const ()),              // 8 SetForceFeedbackMotorGain
    ComSlot(gid_unsupported as *const ()),       // 9 SetHapticMotorState
    ComSlot(gid_set_rumble_state as *const ()),  // 10 SetRumbleState
    ComSlot(gid_void as *const ()),              // 11 SetInputSynchronizationState
    ComSlot(gid_void as *const ()),              // 12 SendInputSynchronizationHint
    ComSlot(gid_void as *const ()),              // 13 PowerOff
    ComSlot(gid_unsupported as *const ()),       // 14 CreateRawDeviceReport
    ComSlot(gid_unsupported as *const ()),       // 15 GetRawDeviceFeature
    ComSlot(gid_unsupported as *const ()),       // 16 SetRawDeviceFeature
    ComSlot(gid_unsupported as *const ()),       // 17 SendRawDeviceOutput
    ComSlot(gid_unsupported as *const ()),       // 18 SendRawDeviceOutputWithResponse
    ComSlot(gid_unsupported as *const ()),       // 19 ExecuteRawDeviceIoControl
    ComSlot(gir_false as *const ()),             // 20 AcquireExclusiveRawDeviceAccess
    ComSlot(gid_void as *const ()),              // 21 ReleaseExclusiveRawDeviceAccess
];

/// The gamepad device object: `GameInputDeviceInfo` with infoSize, the
/// Xbox controller VID/PID, family (Xbox One = 1 in v0), the gamepad input
/// kind and all four rumble motors.
fn game_input_device() -> *mut c_void {
    static DEV: std::sync::OnceLock<ComGameInputDevice> = std::sync::OnceLock::new();
    let d = DEV.get_or_init(|| {
        let mut info = [0u8; 512];
        info[0..4].copy_from_slice(&512u32.to_le_bytes()); // infoSize
        info[4..6].copy_from_slice(&0x045Eu16.to_le_bytes()); // vendorId
        info[6..8].copy_from_slice(&0x0B12u16.to_le_bytes()); // productId
        info[32] = 0x53; // deviceId: 'S'...
        info[96..100].copy_from_slice(&1u32.to_le_bytes()); // deviceFamily
        info[100..104].copy_from_slice(&crate::gameinput::GAME_INPUT_KIND_GAMEPAD.to_le_bytes());
        info[104..108].copy_from_slice(&0xFu32.to_le_bytes()); // rumble motors
        ComGameInputDevice {
            vtable: &GAME_INPUT_DEVICE_VTABLE,
            info,
        }
    });
    d as *const ComGameInputDevice as *mut c_void
}

#[repr(C)]
pub struct ComGameInput {
    pub vtable: *const [ComSlot; 22],
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

fn hand_out_reading(r: Option<crate::gameinput::InputReading>, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return 0x8007_0057u32 as i32;
    }
    match r {
        Some(r) => {
            unsafe { *out = new_reading(r) };
            0
        }
        None => {
            unsafe { *out = std::ptr::null_mut() };
            GAMEINPUT_E_READING_NOT_FOUND
        }
    }
}

unsafe extern "system" fn gi_get_current_reading(
    _this: *mut c_void,
    input_kind: u32,
    _device: *mut c_void,
    out_reading: *mut *mut c_void,
) -> i32 {
    let reading = with_gdk(|g| {
        g.gameinput.poll_host();
        g.gameinput.get_current_reading(input_kind).cloned()
    });
    hand_out_reading(reading, out_reading)
}

unsafe extern "system" fn gi_get_next_reading(
    _this: *mut c_void,
    reference: *mut c_void,
    input_kind: u32,
    _device: *mut c_void,
    out_reading: *mut *mut c_void,
) -> i32 {
    if reference.is_null() {
        return 0x8007_0057u32 as i32;
    }
    let after = gir(reference).id;
    let reading = with_gdk(|g| {
        g.gameinput.poll_host();
        g.gameinput.get_next_reading(after, input_kind).cloned()
    });
    hand_out_reading(reading, out_reading)
}

unsafe extern "system" fn gi_get_previous_reading(
    _this: *mut c_void,
    reference: *mut c_void,
    input_kind: u32,
    _device: *mut c_void,
    out_reading: *mut *mut c_void,
) -> i32 {
    if reference.is_null() {
        return 0x8007_0057u32 as i32;
    }
    let before = gir(reference).id;
    let reading = with_gdk(|g| {
        g.gameinput
            .get_previous_reading(before, input_kind)
            .cloned()
    });
    hand_out_reading(reading, out_reading)
}

unsafe extern "system" fn gi_get_temporal_reading(
    _this: *mut c_void,
    timestamp: u64,
    _device: *mut c_void,
    out_reading: *mut *mut c_void,
) -> i32 {
    let reading = with_gdk(|g| g.gameinput.get_temporal_reading(timestamp).cloned());
    hand_out_reading(reading, out_reading)
}

static NEXT_CALLBACK_TOKEN: AtomicU32 = AtomicU32::new(1);

fn new_token(out: *mut u64) {
    if !out.is_null() {
        unsafe { *out = u64::from(NEXT_CALLBACK_TOKEN.fetch_add(1, Ordering::Relaxed)) };
    }
}

/// `RegisterReadingCallback(device, kind, threshold, context, callback,
/// token)`: readings are polled, not pushed; the token is valid.
unsafe extern "system" fn gi_register_reading_callback(
    _this: *mut c_void,
    _device: *mut c_void,
    _kind: u32,
    _threshold: f32,
    _context: *mut c_void,
    _callback: *const c_void,
    token: *mut u64,
) -> i32 {
    new_token(token);
    0
}

type DeviceCallback = unsafe extern "system" fn(u64, *mut c_void, *mut c_void, u64, u32, u32);

/// `RegisterDeviceCallback(device, kind, statusFilter, enumerationKind,
/// context, callback, token)`: the gamepad is reported connected at once
/// (blocking and async enumeration alike), so titles that wait for a
/// controller callback see one.
unsafe extern "system" fn gi_register_device_callback(
    _this: *mut c_void,
    _device: *mut c_void,
    kind: u32,
    _status_filter: u32,
    enumeration: u32,
    context: *mut c_void,
    callback: *const c_void,
    token: *mut u64,
) -> i32 {
    new_token(token);
    let t = if token.is_null() {
        0
    } else {
        unsafe { *token }
    };
    if !callback.is_null()
        && enumeration != 0
        && kind & crate::gameinput::GAME_INPUT_KIND_GAMEPAD != 0
    {
        let f: DeviceCallback = unsafe { std::mem::transmute(callback) };
        let ts = with_gdk(|g| g.gameinput.current_timestamp_us());
        unsafe { f(t, context, game_input_device(), ts, 1, 0) };
    }
    0
}

unsafe extern "system" fn gi_register_other_callback(
    _this: *mut c_void,
    _a: *mut c_void,
    _b: u64,
    _context: *mut c_void,
    _callback: *const c_void,
    token: *mut u64,
) -> i32 {
    new_token(token);
    0
}

unsafe extern "system" fn gi_unregister_callback(
    _this: *mut c_void,
    _token: u64,
    _timeout: u64,
) -> u8 {
    1
}

unsafe extern "system" fn gi_find_device(
    _this: *mut c_void,
    _id: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return 0x8007_0057u32 as i32;
    }
    unsafe { *out = game_input_device() };
    0
}

unsafe extern "system" fn gi_set_focus_policy(_this: *mut c_void, _policy: u32) {}

static GAME_INPUT_VTABLE: [ComSlot; 22] = [
    ComSlot(gi_qi as *const ()),                        // 0 QueryInterface
    ComSlot(gi_add_ref as *const ()),                   // 1 AddRef
    ComSlot(gi_release as *const ()),                   // 2 Release
    ComSlot(gi_get_current_timestamp as *const ()),     // 3 GetCurrentTimestamp
    ComSlot(gi_get_current_reading as *const ()),       // 4 GetCurrentReading
    ComSlot(gi_get_next_reading as *const ()),          // 5 GetNextReading
    ComSlot(gi_get_previous_reading as *const ()),      // 6 GetPreviousReading
    ComSlot(gi_get_temporal_reading as *const ()),      // 7 GetTemporalReading
    ComSlot(gi_register_reading_callback as *const ()), // 8 RegisterReadingCallback
    ComSlot(gi_register_device_callback as *const ()),  // 9 RegisterDeviceCallback
    ComSlot(gi_register_other_callback as *const ()),   // 10 RegisterSystemButtonCallback
    ComSlot(gi_register_other_callback as *const ()),   // 11 RegisterKeyboardLayoutCallback
    ComSlot(xa2_voice_stub_void as *const ()),          // 12 StopCallback
    ComSlot(gi_unregister_callback as *const ()),       // 13 UnregisterCallback
    ComSlot(gid_unsupported as *const ()),              // 14 CreateDispatcher
    ComSlot(gid_unsupported as *const ()),              // 15 CreateAggregateDevice
    ComSlot(gi_find_device as *const ()),               // 16 FindDeviceFromId
    ComSlot(gi_find_device as *const ()),               // 17 FindDeviceFromObject
    ComSlot(gi_find_device as *const ()),               // 18 FindDeviceFromPlatformHandle
    ComSlot(gi_find_device as *const ()),               // 19 FindDeviceFromPlatformString
    ComSlot(gid_unsupported as *const ()),              // 20 EnableOemDeviceSupport
    ComSlot(gi_set_focus_policy as *const ()),          // 21 SetFocusPolicy
];

#[cfg(windows)]
unsafe extern "system" {
    fn MapVirtualKeyW(code: u32, map_type: u32) -> u32;
}

#[cfg(not(windows))]
unsafe fn MapVirtualKeyW(_code: u32, _map_type: u32) -> u32 {
    0
}

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
    if let Some(addr) = crate::os_thunks::resolve_os_hle(&dll, func_name) {
        return Some(addr);
    }
    if let Some(addr) = crate::upscaler::resolve_upscaler_hle(&dll, func_name) {
        return Some(addr);
    }
    if let Some(addr) = crate::bvh_builder::resolve_bvh_builder_hle(&dll, func_name) {
        return Some(addr);
    }
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
            "D3D12SerializeRootSignature" => d3d12_serialize_root_signature as *const () as usize,
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
            // XSystem
            "XSystemGetConsoleId" => {
                crate::gdk_services::xsystem_get_console_id as *const () as usize
            }
            "XSystemGetXboxLiveSandboxId" => {
                crate::gdk_services::xsystem_get_sandbox_id as *const () as usize
            }
            "XSystemGetAppSpecificDeviceId" => {
                crate::gdk_services::xsystem_get_app_specific_device_id as *const () as usize
            }
            "XSystemGetRuntimeInfo" => {
                crate::gdk_services::xsystem_get_runtime_info as *const () as usize
            }
            "XSystemGetAnalyticsInfo" => {
                crate::gdk_services::xsystem_get_analytics_info as *const () as usize
            }
            // XThread
            "XThreadIsTimeSensitive" => {
                crate::gdk_services::xthread_is_time_sensitive as *const () as usize
            }
            "XThreadSetTimeSensitive" => {
                crate::gdk_services::xthread_set_time_sensitive as *const () as usize
            }
            "XThreadAssertNotTimeSensitive" => {
                crate::gdk_services::xthread_assert_not_time_sensitive as *const () as usize
            }
            // XDisplay
            "XDisplayTryEnableHdrMode" => {
                crate::gdk_services::xdisplay_try_enable_hdr_mode as *const () as usize
            }
            "XDisplayAcquireTimeoutDeferral" => {
                crate::gdk_services::xdisplay_acquire_timeout_deferral as *const () as usize
            }
            "XDisplayCloseTimeoutDeferralHandle" => {
                crate::gdk_services::xdisplay_close_timeout_deferral_handle as *const () as usize
            }
            // XGameUi
            "XGameUiShowMessageDialogAsync" => {
                xgameui_show_message_dialog_async as *const () as usize
            }
            "XGameUiShowMessageDialogResult" => {
                xgameui_show_message_dialog_result as *const () as usize
            }
            "XGameUiShowErrorDialogAsync" => xgameui_show_error_dialog_async as *const () as usize,
            "XGameUiShowTextEntryAsync" => xgameui_show_text_entry_async as *const () as usize,
            "XGameUiShowTextEntryResultSize" => {
                xgameui_show_text_entry_result_size as *const () as usize
            }
            "XGameUiShowTextEntryResult" => xgameui_show_text_entry_result as *const () as usize,
            "XGameUiShowAchievementsAsync"
            | "XGameUiShowPlayerProfileCardAsync"
            | "XGameUiShowPlayerPickerAsync" => xgameui_show_overlay_async as *const () as usize,
            "XGameUiShowSendGameInviteAsync"
            | "XGameUiShowWebAuthenticationAsync"
            | "XGameUiShowMultiplayerActivityGameInviteAsync" => {
                xgameui_show_online_async as *const () as usize
            }
            "XGameUiShowErrorDialogResult"
            | "XGameUiShowAchievementsResult"
            | "XGameUiShowPlayerProfileCardResult"
            | "XGameUiShowSendGameInviteResult"
            | "XGameUiShowMultiplayerActivityGameInviteResult" => {
                xgameui_status_result as *const () as usize
            }
            // XLauncher
            "XLaunchUri" => xlaunch_uri as *const () as usize,
            "XLaunchNewGame" => xlaunch_new_game as *const () as usize,
            "XLaunchRestartOnCrash" => xlaunch_restart_on_crash as *const () as usize,
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

    /// A title's `IXAudio2VoiceCallback`: counts buffer ends and records the
    /// last context.
    #[repr(C)]
    struct TestCallback {
        vtable: *const ComSlot,
        ends: std::sync::atomic::AtomicU32,
        last_context: std::sync::atomic::AtomicU64,
    }

    unsafe extern "system" fn cb_u32(_this: *mut c_void, _n: u32) {}
    unsafe extern "system" fn cb_void(_this: *mut c_void) {}
    unsafe extern "system" fn cb_ctx(_this: *mut c_void, _c: *mut c_void) {}
    unsafe extern "system" fn cb_buffer_end(this: *mut c_void, c: *mut c_void) {
        let t = unsafe { &*(this as *const TestCallback) };
        t.last_context.store(c as u64, Ordering::SeqCst);
        t.ends.fetch_add(1, Ordering::SeqCst);
    }
    unsafe extern "system" fn cb_error(_this: *mut c_void, _c: *mut c_void, _hr: i32) {}

    static TEST_CALLBACK_VTABLE: [ComSlot; 7] = [
        ComSlot(cb_u32 as *const ()),        // OnVoiceProcessingPassStart
        ComSlot(cb_void as *const ()),       // OnVoiceProcessingPassEnd
        ComSlot(cb_void as *const ()),       // OnStreamEnd
        ComSlot(cb_ctx as *const ()),        // OnBufferStart
        ComSlot(cb_buffer_end as *const ()), // OnBufferEnd
        ComSlot(cb_ctx as *const ()),        // OnLoopEnd
        ComSlot(cb_error as *const ()),      // OnVoiceError
    ];

    #[test]
    fn a_source_voice_plays_a_buffer_and_calls_back_from_the_audio_thread() {
        unsafe {
            let mut xa2 = std::ptr::null_mut();
            assert_eq!(xaudio2_create(&mut xa2, 0, 0), 0);
            let vt = (*(xa2 as *mut ComXAudio2)).vtable;
            let create_mv: unsafe extern "system" fn(
                *mut c_void,
                *mut *mut c_void,
                u32,
                u32,
                u32,
                *const c_void,
                *const c_void,
                u32,
            ) -> i32 = std::mem::transmute((*vt)[7].0);
            let mut master = std::ptr::null_mut();
            assert_eq!(
                create_mv(
                    xa2,
                    &mut master,
                    2,
                    48_000,
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    0
                ),
                0
            );

            // 48 kHz stereo PCM16, 10 ms of a quiet tone.
            let mut wfx = [0u8; 18];
            wfx[0..2].copy_from_slice(&1u16.to_le_bytes());
            wfx[2..4].copy_from_slice(&2u16.to_le_bytes());
            wfx[4..8].copy_from_slice(&48_000u32.to_le_bytes());
            wfx[8..12].copy_from_slice(&(48_000u32 * 4).to_le_bytes());
            wfx[12..14].copy_from_slice(&4u16.to_le_bytes());
            wfx[14..16].copy_from_slice(&16u16.to_le_bytes());
            let callback = Box::new(TestCallback {
                vtable: TEST_CALLBACK_VTABLE.as_ptr(),
                ends: std::sync::atomic::AtomicU32::new(0),
                last_context: std::sync::atomic::AtomicU64::new(0),
            });
            let cb_ptr = &*callback as *const TestCallback as *mut c_void;
            let create_sv: unsafe extern "system" fn(
                *mut c_void,
                *mut *mut c_void,
                *const u8,
                u32,
                f32,
                *mut c_void,
                *const [u64; 2],
                *const c_void,
            ) -> i32 = std::mem::transmute((*vt)[5].0);
            let mut source = std::ptr::null_mut();
            assert_eq!(
                create_sv(
                    xa2,
                    &mut source,
                    wfx.as_ptr(),
                    0,
                    2.0,
                    cb_ptr,
                    std::ptr::null(),
                    std::ptr::null()
                ),
                0
            );

            let pcm: Vec<u8> = (0..480 * 2)
                .flat_map(|i: i32| (((i % 64) - 32) as i16 * 64).to_le_bytes())
                .collect();
            let buf = XAudio2Buffer {
                flags: 0x40,
                audio_bytes: pcm.len() as u32,
                data: pcm.as_ptr(),
                play_begin: 0,
                play_length: 0,
                loop_begin: 0,
                loop_length: 0,
                loop_count: 0,
                context: 0x5151 as *mut c_void,
            };
            let svt = (*(source as *mut ComXAudio2Voice)).vtable;
            let submit: unsafe extern "system" fn(
                *mut c_void,
                *const XAudio2Buffer,
                *const c_void,
            ) -> i32 = std::mem::transmute((*svt.add(21)).0);
            let start: unsafe extern "system" fn(*mut c_void, u32, u32) -> i32 =
                std::mem::transmute((*svt.add(19)).0);
            let get_state: unsafe extern "system" fn(*mut c_void, *mut [u64; 3], u32) =
                std::mem::transmute((*svt.add(25)).0);
            assert_eq!(submit(source, &buf, std::ptr::null()), 0);
            let mut st = [0u64; 3];
            get_state(source, &mut st, 0);
            assert_eq!(st[1], 1, "one buffer queued");
            assert_eq!(start(source, 0, 0), 0);

            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while callback.ends.load(Ordering::SeqCst) == 0 && std::time::Instant::now() < deadline
            {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            assert_eq!(
                callback.ends.load(Ordering::SeqCst),
                1,
                "OnBufferEnd from the audio thread"
            );
            assert_eq!(callback.last_context.load(Ordering::SeqCst), 0x5151);
            get_state(source, &mut st, 0);
            assert_eq!((st[1], st[2]), (0, 480));

            let destroy: unsafe extern "system" fn(*mut c_void) =
                std::mem::transmute((*svt.add(18)).0);
            destroy(source);
            let mvt = (*(master as *mut ComXAudio2Voice)).vtable;
            let destroy_m: unsafe extern "system" fn(*mut c_void) =
                std::mem::transmute((*mvt.add(18)).0);
            destroy_m(master);
            let release: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute((*vt)[2].0);
            release(xa2);
        }
    }

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
                std::mem::transmute((*(*mv).vtable.add(18)).0);
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
            let ds = ds_ptr as *mut crate::dstorage::DStorageFactory;
            assert!(!(*ds).inner.vfs.is_empty());
            let release: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute((*(*ds).vtable.add(2)).0);
            assert_eq!(release(ds_ptr), 0);

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
            let size = ((*blob).vtable.get_buffer_size)(blob_ptr);
            let data = ((*blob).vtable.get_buffer_pointer)(blob_ptr) as *const u8;
            assert!(!data.is_null());
            // A real serialized root signature: it parses back.
            let parsed =
                crate::d3d12_types::RootLayout::parse(std::slice::from_raw_parts(data, size));
            assert_eq!(parsed.map(|l| l.params.len()), Some(0));
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
