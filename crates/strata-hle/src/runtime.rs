//! Console `kernelx` & `d3d12_xs` HLE registry and import classifier.
//!
//! Maps Xbox Series X|S system DLL imports to:
//! - `StrataHle`: implemented directly by Strata (`kernelx`, `d3d12_xs`, `xg_xs`, `xgameruntime`)
//! - `HostPassthrough`: safe Windows OS DLLs (`kernel32`, `ntdll`, `user32`, etc.)
//! - `TitleLocal`: DLLs shipped alongside the title (`XboxBvhBuilder.dll`, `amd_fidelityfx_*`, etc.)
//! - `NeedsStub`: unimplemented console-only imports requiring HLE stubs

use strata_gpu::context::GpuError;
use strata_gpu::runner::ComputeRunner;
use strata_gpu::thin_list::ThinCommandList;
use strata_gpu::uma::UmaManager;

use crate::xmem_phys::PhysicalPagePool;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DllClass {
    StrataHle,
    HostPassthrough,
    TitleLocal,
    NeedsStub,
}

pub fn classify_dll(dll_name: &str) -> DllClass {
    let lower = dll_name.to_ascii_lowercase();
    match lower.as_str() {
        "kernelx.dll"
        | "xmem.dll"
        | "d3d12_xs.dll"
        | "d3d12_x.dll"
        | "xg_xs.dll"
        | "xg.dll"
        | "xgameruntime.dll"
        | "dstorage_xs.dll"
        | "acphal.dll"
        | "xapu.dll"
        | "xcrdapi.dll"
        | "libhttpclient.gdk.dll"
        | "pixevt.dll" => DllClass::StrataHle,

        "kernel32.dll" | "ntdll.dll" | "user32.dll" | "advapi32.dll" | "bcrypt.dll"
        | "crypt32.dll" | "ole32.dll" | "rpcrt4.dll" | "shell32.dll" | "shlwapi.dll"
        | "winhttp.dll" | "ws2_32.dll" | "iphlpapi.dll" | "xaudio2_9.dll" | "mfplat.dll"
        | "mfreadwrite.dll" | "gameinput.dll" => DllClass::HostPassthrough,

        s if s.starts_with("api-ms-win-")
            || s.starts_with("ext-ms-win-")
            || s.starts_with("vcruntime")
            || s.starts_with("msvcp")
            || s.starts_with("ucrtbase") =>
        {
            DllClass::HostPassthrough
        }

        s if s.contains("bvh")
            || s.contains("fidelityfx")
            || s.contains("bink")
            || s.contains("chad") =>
        {
            DllClass::TitleLocal
        }

        _ => DllClass::NeedsStub,
    }
}

/// Runtime HLE state bridging `kernelx` (`XMem*`) and `d3d12_xs` calls onto
/// Strata's 3-tier hybrid UMA + No-API Vulkan thin command lists.
pub struct HleRuntime {
    pub uma: UmaManager,
    pub phys_pool: PhysicalPagePool,
    pub active_list: ThinCommandList,
    next_console_va: u64,
}

impl HleRuntime {
    pub fn new() -> Self {
        Self {
            uma: UmaManager::new(),
            phys_pool: PhysicalPagePool::new(),
            active_list: ThinCommandList::new(),
            next_console_va: 0x0000_0400_0000_0000,
        }
    }

    /// Emulates `kernelx!XMemVirtualAlloc` using Tier 1 Zero-Copy Host UMA
    /// (`VirtualAlloc`/`mmap` + `VK_EXT_external_memory_host` -> 64-bit GPU BDA pointer).
    ///
    /// The title gets a real host pointer (its code dereferences it): at
    /// `requested_va` if that range is free, else wherever Windows puts it.
    pub fn xmem_virtual_alloc(
        &mut self,
        runner: &ComputeRunner,
        requested_va: u64,
        size: u64,
    ) -> Result<(u64, u64), GpuError> {
        let alloc = self.uma.xmem_alloc_identity(runner, requested_va, size)?;
        self.next_console_va = self.next_console_va.max(alloc.console_va + alloc.size);
        Ok((alloc.console_va, alloc.host_bda))
    }

    /// Emulates `xmem!XMemFree` / `XMemFreeDefault`, releasing both the Vulkan BDA
    /// buffer and the underlying host allocation.
    pub fn xmem_free(&mut self, runner: &ComputeRunner, addr: u64) -> bool {
        self.uma.free_allocation(runner, addr)
    }

    pub fn shutdown(&mut self, runner: &ComputeRunner) {
        self.uma.destroy_all(runner);
    }
}

impl Default for HleRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_console_vs_host_dlls() {
        assert_eq!(classify_dll("d3d12_xs.dll"), DllClass::StrataHle);
        assert_eq!(classify_dll("KERNELX.DLL"), DllClass::StrataHle);
        assert_eq!(
            classify_dll("api-ms-win-core-synch-l1-2-0.dll"),
            DllClass::HostPassthrough
        );
        assert_eq!(classify_dll("XboxBvhBuilder.dll"), DllClass::TitleLocal);
    }
}
