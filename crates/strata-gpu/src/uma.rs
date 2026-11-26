//! 3-Tier Hybrid Unified Memory Manager (Console `XMem*` -> No-API Vulkan).
//!
//! Bridges Xbox Series X|S memory allocations onto discrete PC GPUs cleanly:
//! - Tier 1 (`ZeroCopyHost`): `VirtualAlloc` imported directly via
//!   `VK_EXT_external_memory_host` -> 64-bit BDA pointer (zero `memcpy` for
//!   CPU-written dynamic buffers, constants, and streaming data).
//! - Tier 2 (`VramMirror`): `DEVICE_LOCAL` GDDR6 BDA buffer uploaded once via
//!   copy submission (full GPU bandwidth for static meshes, BVHs, textures).
//! - Tier 3 (`SparseTarget`): 64 KiB tiled `VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT`
//!   aliasing via `SparseTilePool` + `vkQueueBindSparse` for RTs/Depth/UAVs.

use std::collections::HashMap;

use crate::context::GpuError;
use crate::memory::{AddressSpace, BackingKind, MemoryRegion};
use crate::runner::{BdaBuffer, ComputeRunner};
use crate::vram::{SparseTilePool, TileId};

#[cfg(windows)]
unsafe extern "system" {
    fn VirtualAlloc(
        addr: *mut core::ffi::c_void,
        size: usize,
        alloc_type: u32,
        protect: u32,
    ) -> *mut core::ffi::c_void;
    fn VirtualFree(addr: *mut core::ffi::c_void, size: usize, free_type: u32) -> i32;
}

const MEM_COMMIT_RESERVE: u32 = 0x3000;
const PAGE_READWRITE: u32 = 0x04;
const MEM_RELEASE: u32 = 0x8000;

/// Console-side `XMemVirtualAlloc` allocation tracked by Strata.
pub struct UmaAllocation {
    pub console_va: u64,
    pub size: u64,
    pub host_bda: u64,
    pub kind: BackingKind,
    raw_host_ptr: Option<*mut u8>,
    buffer: BdaBuffer,
}

impl UmaAllocation {
    /// Direct CPU pointer for Tier 1 (`ZeroCopyHost`) allocations.
    pub fn cpu_ptr(&self) -> Option<*mut u8> {
        self.raw_host_ptr
    }
}

pub struct UmaManager {
    pub address_space: AddressSpace,
    pub sparse_pool: SparseTilePool,
    allocations: HashMap<u64, UmaAllocation>,
}

impl UmaManager {
    pub fn new() -> Self {
        Self {
            address_space: AddressSpace::new(),
            sparse_pool: SparseTilePool::new(),
            allocations: HashMap::new(),
        }
    }

    /// Tier 1 (`XMemVirtualAlloc` -> Zero-Copy Host BDA):
    /// Allocates Windows `VirtualAlloc` RAM, imports it directly into Vulkan via
    /// `VK_EXT_external_memory_host`, and registers the console VA -> host BDA mapping.
    pub fn xmem_virtual_alloc_zero_copy(
        &mut self,
        runner: &ComputeRunner,
        console_va: u64,
        size: u64,
    ) -> Result<&UmaAllocation, GpuError> {
        let aligned_size = size.max(4096).next_multiple_of(runner.min_host_pointer_alignment.max(4096));
        unsafe {
            let raw = VirtualAlloc(
                std::ptr::null_mut(),
                aligned_size as usize,
                MEM_COMMIT_RESERVE,
                PAGE_READWRITE,
            ) as *mut u8;
            if raw.is_null() {
                return Err(GpuError::NoDevice);
            }
            let buf = runner.import_host_bda_buffer(raw, aligned_size)?;
            let host_bda = buf.bda;

            self.address_space.register(MemoryRegion {
                console_base: console_va,
                size: aligned_size,
                host_bda,
                kind: BackingKind::ZeroCopyHost,
                last_gpu_write_epoch: 0,
            });

            self.allocations.insert(
                console_va,
                UmaAllocation {
                    console_va,
                    size: aligned_size,
                    host_bda,
                    kind: BackingKind::ZeroCopyHost,
                    raw_host_ptr: Some(raw),
                    buffer: buf,
                },
            );
        }
        Ok(&self.allocations[&console_va])
    }

    /// Tier 2 (Static / Read-Heavy Data -> `DEVICE_LOCAL` VRAM BDA Mirror):
    /// Allocates fast GDDR6 VRAM and uploads `initial_bytes` via GPU transfer so
    /// shaders read at full VRAM bandwidth instead of over PCIe.
    pub fn alloc_vram_mirror(
        &mut self,
        runner: &ComputeRunner,
        console_va: u64,
        initial_bytes: &[u8],
    ) -> Result<&UmaAllocation, GpuError> {
        let size = (initial_bytes.len() as u64).max(256).next_multiple_of(256);
        let buf = runner.alloc_vram_bda_buffer(size, initial_bytes)?;
        let host_bda = buf.bda;

        self.address_space.register(MemoryRegion {
            console_base: console_va,
            size,
            host_bda,
            kind: BackingKind::VramMirror,
            last_gpu_write_epoch: 0,
        });

        self.allocations.insert(
            console_va,
            UmaAllocation {
                console_va,
                size,
                host_bda,
                kind: BackingKind::VramMirror,
                raw_host_ptr: None,
                buffer: buf,
            },
        );
        Ok(&self.allocations[&console_va])
    }

    /// Tier 3 (Aliased Render Target / UAV -> 64 KiB Sparse Tiles):
    /// Acquires pooled 64 KiB tile IDs for a console target address range.
    pub fn acquire_sparse_target_tiles(&mut self, console_va: u64, size_bytes: u64) -> Vec<(u64, TileId)> {
        self.sparse_pool.acquire_range(console_va, size_bytes)
    }

    /// Translate any console virtual address into its live 64-bit Vulkan BDA pointer.
    pub fn translate_va(&self, console_va: u64) -> Option<u64> {
        self.address_space.translate(console_va)
    }

    /// Free all tracked Vulkan buffers and `VirtualAlloc` blocks cleanly.
    pub fn destroy_all(&mut self, runner: &ComputeRunner) {
        for (_, alloc) in self.allocations.drain() {
            runner.free_bda_buffer(alloc.buffer);
            if let Some(raw) = alloc.raw_host_ptr {
                unsafe {
                    VirtualFree(raw as *mut core::ffi::c_void, 0, MEM_RELEASE);
                }
            }
        }
    }
}

impl Default for UmaManager {
    fn default() -> Self {
        Self::new()
    }
}
