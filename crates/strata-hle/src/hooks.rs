//! Dynamic `LoadLibraryA`/`GetProcAddress` console HLE interceptor & pseudo-handles.
//!
//! When a title dynamically calls `LoadLibrary("d3d12_xs.dll")` or `LoadLibrary("xmem.dll")`
//! followed by `GetProcAddress`, Strata returns a tagged pseudo-handle and resolves
//! the symbol directly through `crate::thunks::resolve_hle`.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::runtime::{DllClass, classify_dll};
use crate::thunks::resolve_hle;

const PSEUDO_HANDLE_BASE: usize = 0x5015_0000;

static PSEUDO_HANDLES: OnceLock<Mutex<HashMap<usize, String>>> = OnceLock::new();

fn table() -> &'static Mutex<HashMap<usize, String>> {
    PSEUDO_HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Intercept `LoadLibrary*` for console-only HLE DLLs, returning a tagged pseudo-handle.
pub fn hook_load_library(dll_name: &str) -> Option<usize> {
    if classify_dll(dll_name) != DllClass::StrataHle {
        return None;
    }
    let mut map = table().lock().unwrap();
    let lower = dll_name.to_ascii_lowercase();
    if let Some((&handle, _)) = map.iter().find(|(_, v)| **v == lower) {
        return Some(handle);
    }
    let handle = PSEUDO_HANDLE_BASE + map.len() + 1;
    map.insert(handle, lower);
    Some(handle)
}

/// Intercept `GetProcAddress` when `module` is a Strata console pseudo-handle.
pub fn hook_get_proc_address(module: usize, func_name: &str) -> Option<usize> {
    let dll = table().lock().unwrap().get(&module)?.clone();
    resolve_hle(&dll, func_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_dynamic_loadlibrary_and_getprocaddress_for_console_dlls() {
        let h = hook_load_library("d3d12_xs.dll").expect("expected pseudo handle");
        assert!(hook_get_proc_address(h, "D3D12XboxCreateDevice").is_some());
        assert_eq!(hook_load_library("kernel32.dll"), None);
    }
}
