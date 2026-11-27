//! Console DLL `extern "system"` thunk table for `xmem.dll`, `d3d12_xs.dll`,
//! `xg_xs.dll`, `XCRDAPI.dll`, `xapu.dll`, `AcpHal.dll`, and offline `libHttpClient.GDK.dll`.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use strata_gpu::runner::ComputeRunner;

use crate::runtime::HleRuntime;
use crate::xg::{SwizzleMode, TextureLayout};
use crate::xmem_phys::XMEM_PAGE_BYTES;

/// Global HLE runtime state accessed by `extern "system"` console thunks.
struct GlobalHle {
    runner: ComputeRunner,
    runtime: HleRuntime,
}

// SAFETY: GlobalHle is exclusively accessed behind a Mutex.
unsafe impl Send for GlobalHle {}

static GLOBAL_HLE: OnceLock<Mutex<GlobalHle>> = OnceLock::new();
static AUDIO_STUB_CALLS: AtomicU64 = AtomicU64::new(0);
static UNIMPLEMENTED_STUB_CALLS: AtomicU64 = AtomicU64::new(0);

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

unsafe extern "system" fn xmem_virtual_alloc(
    address: *mut c_void,
    size: usize,
    _alloc_type: u32,
    _protect: u32,
) -> *mut c_void {
    with_hle(|g| {
        let req_va = address as u64;
        if let Ok((va, _bda)) = g.runtime.xmem_virtual_alloc(&g.runner, req_va, size as u64) {
            return va as *mut c_void;
        }
        std::ptr::null_mut()
    })
    .unwrap_or(std::ptr::null_mut())
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

unsafe extern "system" fn d3d12_xbox_create_device(
    _params: *const c_void,
    _riid: *const c_void,
    out_device: *mut *mut c_void,
) -> i32 {
    if !out_device.is_null() {
        unsafe { *out_device = crate::d3d12x::D3D12DeviceX::new_boxed() as *mut c_void };
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
        let factory = crate::gdk::DStorageFactoryX::new_boxed(crate::vfs::Vfs::new());
        unsafe { *out_factory = factory as *mut c_void };
    }
    0 // S_OK
}

/// Audio subsystem (`xapu.dll` / `AcpHal.dll`) offline/silent stub with call tracking.
unsafe extern "system" fn audio_stub_silent() -> i32 {
    AUDIO_STUB_CALLS.fetch_add(1, Ordering::Relaxed);
    0 // S_OK (silent audio device)
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
            "XMemAlloc" | "XMemAllocDefault" => xmem_alloc as *const () as usize,
            "XMemFree" | "XMemFreeDefault" => xmem_free as *const () as usize,
            "XMemIsToolsMemAvailable" => xmem_is_tools_mem_available as *const () as usize,
            "XMemAllocatePhysicalPages" => xmem_allocate_physical_pages as *const () as usize,
            "XMemFreePhysicalPages" => xmem_free_physical_pages as *const () as usize,
            "XMemGetWorkingSetStatistics" => xmem_get_working_set_statistics as *const () as usize,
            "XMemMapPhysicalPages" => xmem_map_physical_pages as *const () as usize,
            "XMemMapPhysicalPagesScatter" => xmem_map_physical_pages_scatter as *const () as usize,
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
            "XGameRuntimeInitialize"
            | "XGameRuntimeUninitialize"
            | "XSystemGetDeviceType"
            | "XNetworkingQueryConnectivityHint"
            | "XUserAddAsync"
            | "XGameSaveInitializeProviderAsync"
            | "XPackageGetCurrentProcessPackageIdentifier"
            | "XStoreCreateContext"
            | "XTaskQueueCreate"
            | "XTaskQueueDispatch"
            | "XAsyncGetStatus" => audio_stub_silent as *const () as usize,
            _ => logged_unimplemented_stub as *const () as usize,
        }),
        "xaudio2_9.dll" | "xaudio2_9redist.dll" | "xapu.dll" | "acphal.dll" => {
            Some(audio_stub_silent as *const () as usize)
        }
        "gameinput.dll" => Some(audio_stub_silent as *const () as usize),
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
}
