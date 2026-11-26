//! Console DLL `extern "system"` thunk table for `xmem.dll`, `d3d12_xs.dll`,
//! `xg_xs.dll`, `XCRDAPI.dll`, `xapu.dll`, `AcpHal.dll`, and offline `libHttpClient.GDK.dll`.

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use strata_gpu::runner::ComputeRunner;

use crate::runtime::HleRuntime;

/// Global HLE runtime state accessed by `extern "system"` console thunks.
struct GlobalHle {
    runner: ComputeRunner,
    runtime: HleRuntime,
}

// SAFETY: GlobalHle is exclusively accessed behind a Mutex.
unsafe impl Send for GlobalHle {}

static GLOBAL_HLE: OnceLock<Mutex<GlobalHle>> = OnceLock::new();

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

// --- xmem.dll ---

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

unsafe extern "system" fn xmem_free(_ptr: *mut c_void, _attrs: u64) {}

unsafe extern "system" fn xmem_is_tools_mem_available() -> i32 {
    0 // Retail console profile (no devkit tools memory)
}

unsafe extern "system" fn xmem_stub_ok() -> i32 {
    1 // TRUE / S_OK
}

// --- d3d12_xs.dll ---

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
    _blob: *mut *mut c_void,
    _err: *mut *mut c_void,
) -> i32 {
    0 // S_OK (Strata uses a single global No-API bindless + push-constant layout)
}

// --- xg_xs.dll ---

/// `XGSuggestSwizzleMode` -> returns `XG_SWIZZLE_MODE_64KB_S` (9) as safe default.
unsafe extern "system" fn xg_suggest_swizzle_mode() -> u32 {
    9
}

unsafe extern "system" fn xg_create_texture_computer(
    _desc: *const c_void,
    _out: *mut *mut c_void,
) -> i32 {
    0 // S_OK
}

// --- Offline / Audio Stubs ---

/// Offline-by-design network refusal (`E_ACCESSDENIED`) for `libHttpClient.GDK.dll`.
unsafe extern "system" fn offline_net_denied() -> i32 {
    0x8007_0005u32 as i32
}

unsafe extern "system" fn dstorage_get_factory(_riid: *const c_void, out_factory: *mut *mut c_void) -> i32 {
    if !out_factory.is_null() {
        let factory = crate::gdk::DStorageFactoryX::new_boxed(crate::vfs::Vfs::new());
        unsafe { *out_factory = factory as *mut c_void };
    }
    0 // S_OK
}

unsafe extern "system" fn stub_zero() -> usize {
    0
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
            "XMemAllocatePhysicalPages"
            | "XMemFreePhysicalPages"
            | "XMemGetWorkingSetStatistics"
            | "XMemMapPhysicalPages"
            | "XMemMapPhysicalPagesScatter"
            | "XMemTransferMemory" => xmem_stub_ok as *const () as usize,
            _ => return None,
        }),
        "d3d12_xs.dll" | "d3d12_x.dll" => Some(match func_name {
            "D3D12ConfigureVirtualMemoryX" => d3d12_configure_virtual_memory_x as *const () as usize,
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
        "xcrdapi.dll" => Some(match func_name {
            "DStorageGetFactory" => dstorage_get_factory as *const () as usize,
            _ => stub_zero as *const () as usize,
        }),
        "libhttpclient.gdk.dll" => Some(offline_net_denied as *const () as usize),
        "xapu.dll" | "acphal.dll" => Some(stub_zero as *const () as usize),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_all_core_console_imports() {
        for (dll, func) in [
            ("xmem.dll", "XMemVirtualAlloc"),
            ("xmem.dll", "XMemMapPhysicalPagesScatter"),
            ("d3d12_xs.dll", "D3D12XboxCreateDevice"),
            ("d3d12_xs.dll", "D3D12ConfigureVirtualMemoryX"),
            ("xg_xs.dll", "XGSuggestSwizzleMode"),
            ("libHttpClient.GDK.dll", "HCInitialize"),
        ] {
            assert!(resolve_hle(dll, func).is_some(), "missing HLE thunk for {dll}!{func}");
        }
    }
}
