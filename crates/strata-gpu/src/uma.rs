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
use crate::write_watch::WatchedPages;

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
    /// Tier 2 with a title-visible copy: the host pages the title writes and
    /// reads, imported as a transfer buffer.
    shadow: Option<Shadow>,
}

struct Shadow {
    pages: WatchedPages,
    host: BdaBuffer,
}

impl UmaAllocation {
    /// The CPU's view: Tier 1 memory itself, or a Tier 2 mirror's host copy.
    pub fn cpu_ptr(&self) -> Option<*mut u8> {
        self.raw_host_ptr
            .or_else(|| self.shadow.as_ref().map(|s| s.pages.as_ptr()))
    }

    fn free(self, runner: &ComputeRunner) {
        runner.free_bda_buffer(self.buffer);
        if let Some(raw) = self.raw_host_ptr {
            os_free_host_pages(raw, self.size as usize);
        }
        if let Some(shadow) = self.shadow {
            runner.free_bda_buffer(shadow.host);
            drop(shadow.pages);
        }
    }
}

/// What a [`UmaManager::sync_mirrors`] pass moved.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SyncStats {
    pub ranges: usize,
    pub bytes: u64,
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
        let aligned_size = size
            .max(4096)
            .next_multiple_of(runner.min_host_pointer_alignment.max(4096));
        unsafe {
            let raw = os_alloc_host_pages(aligned_size as usize);
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
                    shadow: None,
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
                shadow: None,
            },
        );
        Ok(&self.allocations[&console_va])
    }

    /// Tier 2 with a title-visible host copy: the title reads and writes
    /// host pages (`cpu_ptr`), shaders read and write VRAM. Windows records
    /// the title's writes (`MEM_WRITE_WATCH`); [`Self::sync_mirrors`] uploads
    /// just those pages and [`Self::readback`] brings shader writes back.
    pub fn alloc_shadowed_mirror(
        &mut self,
        runner: &ComputeRunner,
        console_va: u64,
        size: u64,
    ) -> Result<&UmaAllocation, GpuError> {
        let align = runner.min_host_pointer_alignment.max(4096);
        let size = size.max(4096).next_multiple_of(align);
        let pages = WatchedPages::new(size as usize).ok_or(GpuError::HostMemory)?;
        // SAFETY: the pages stay allocated as long as the import (both freed together).
        let host = unsafe { runner.import_host_bda_buffer(pages.as_ptr(), size)? };
        let vram = match runner.alloc_vram_bda_buffer(size, &[]) {
            Ok(v) => v,
            Err(e) => {
                runner.free_bda_buffer(host);
                return Err(e);
            }
        };
        // VRAM starts undefined; give it the (zeroed) host copy once.
        if let Err(e) = runner.copy_buffer_regions(host.buffer, vram.buffer, &[(0, 0, size)]) {
            runner.free_bda_buffer(host);
            runner.free_bda_buffer(vram);
            return Err(e);
        }
        pages.dirty_ranges(true);
        self.address_space.register(MemoryRegion {
            console_base: console_va,
            size,
            host_bda: vram.bda,
            kind: BackingKind::VramMirror,
            last_gpu_write_epoch: 0,
        });
        self.allocations.insert(
            console_va,
            UmaAllocation {
                console_va,
                size,
                host_bda: vram.bda,
                kind: BackingKind::VramMirror,
                raw_host_ptr: None,
                buffer: vram,
                shadow: Some(Shadow { pages, host }),
            },
        );
        Ok(&self.allocations[&console_va])
    }

    /// Uploads every page the title wrote since the last pass into its
    /// mirror's VRAM copy. Run before GPU work that reads mirrors.
    pub fn sync_mirrors(&self, runner: &ComputeRunner) -> Result<SyncStats, GpuError> {
        let mut stats = SyncStats::default();
        for alloc in self.allocations.values() {
            let Some(shadow) = &alloc.shadow else {
                continue;
            };
            // Reset as we read: writes during the copy count for the next pass.
            let ranges = shadow.pages.dirty_ranges(true);
            let regions: Vec<(u64, u64, u64)> = ranges
                .iter()
                .map(|&(o, n)| (o as u64, o as u64, n as u64))
                .collect();
            runner.copy_buffer_regions(shadow.host.buffer, alloc.buffer.buffer, &regions)?;
            stats.ranges += regions.len();
            stats.bytes += regions.iter().map(|r| r.2).sum::<u64>();
        }
        Ok(stats)
    }

    /// Brings `len` bytes at `console_va`, written by shaders, back to the
    /// memory the title reads. Tier 1 needs nothing (it is the same memory);
    /// a shadowed mirror copies VRAM into its host pages. Returns bytes copied.
    pub fn readback(
        &self,
        runner: &ComputeRunner,
        console_va: u64,
        len: u64,
    ) -> Result<u64, GpuError> {
        let Some(alloc) = self
            .allocations
            .values()
            .find(|a| console_va >= a.console_va && console_va < a.console_va + a.size)
        else {
            return Ok(0);
        };
        let Some(shadow) = &alloc.shadow else {
            return Ok(0);
        };
        let off = console_va - alloc.console_va;
        let len = len.min(alloc.size - off);
        runner.copy_buffer_regions(alloc.buffer.buffer, shadow.host.buffer, &[(off, off, len)])?;
        // Our own copy is not a title write.
        shadow.pages.reset_range(off as usize, len as usize);
        Ok(len)
    }

    /// Tier 3 (Aliased Render Target / UAV -> 64 KiB Sparse Tiles):
    /// Acquires pooled 64 KiB tile IDs for a console target address range.
    pub fn acquire_sparse_target_tiles(
        &mut self,
        console_va: u64,
        size_bytes: u64,
    ) -> Vec<(u64, TileId)> {
        self.sparse_pool.acquire_range(console_va, size_bytes)
    }

    /// Tier 1 for `XMemVirtualAlloc`: the title runs natively and dereferences
    /// what it gets, so the console address *is* the host pointer (at
    /// `requested_va` when one is asked for and free; `None` if it is taken).
    pub fn xmem_alloc_identity(
        &mut self,
        runner: &ComputeRunner,
        requested_va: u64,
        size: u64,
    ) -> Result<&UmaAllocation, GpuError> {
        let aligned_size = size
            .max(4096)
            .next_multiple_of(runner.min_host_pointer_alignment.max(4096));
        let raw = os_alloc_host_pages_at(requested_va, aligned_size as usize);
        if raw.is_null() {
            return Err(GpuError::HostMemory);
        }
        // SAFETY: freshly allocated, page aligned, freed with the allocation.
        let buf = match unsafe { runner.import_host_bda_buffer(raw, aligned_size) } {
            Ok(b) => b,
            Err(e) => {
                os_free_host_pages(raw, aligned_size as usize);
                return Err(e);
            }
        };
        let console_va = raw as u64;
        self.address_space.register(MemoryRegion {
            console_base: console_va,
            size: aligned_size,
            host_bda: buf.bda,
            kind: BackingKind::ZeroCopyHost,
            last_gpu_write_epoch: 0,
        });
        self.allocations.insert(
            console_va,
            UmaAllocation {
                console_va,
                size: aligned_size,
                host_bda: buf.bda,
                kind: BackingKind::ZeroCopyHost,
                raw_host_ptr: Some(raw),
                buffer: buf,
                shadow: None,
            },
        );
        Ok(&self.allocations[&console_va])
    }

    /// The CPU address of `console_va` (Tier 1 memory or a mirror's host copy).
    pub fn host_ptr(&self, console_va: u64) -> Option<*mut u8> {
        let a = self
            .allocations
            .values()
            .find(|a| console_va >= a.console_va && console_va < a.console_va + a.size)?;
        // SAFETY: the offset is inside the allocation.
        a.cpu_ptr()
            .map(|p| unsafe { p.add((console_va - a.console_va) as usize) })
    }

    /// Translate any console virtual address into its live 64-bit Vulkan BDA pointer.
    pub fn translate_va(&self, console_va: u64) -> Option<u64> {
        self.address_space.translate(console_va)
    }

    /// Free a single tracked allocation by console VA or host CPU pointer (`XMemFree`).
    pub fn free_allocation(&mut self, runner: &ComputeRunner, addr: u64) -> bool {
        let key = if self.allocations.contains_key(&addr) {
            Some(addr)
        } else {
            self.allocations
                .iter()
                .find(|(_, a)| a.cpu_ptr().map(|p| p as u64) == Some(addr))
                .map(|(&k, _)| k)
        };
        if let Some(k) = key
            && let Some(alloc) = self.allocations.remove(&k)
        {
            alloc.free(runner);
            return true;
        }
        false
    }

    /// Free all tracked Vulkan buffers and host virtual memory blocks cleanly.
    pub fn destroy_all(&mut self, runner: &ComputeRunner) {
        for (_, alloc) in self.allocations.drain() {
            alloc.free(runner);
        }
    }
}

fn os_alloc_host_pages(size: usize) -> *mut u8 {
    #[cfg(windows)]
    unsafe {
        VirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_COMMIT_RESERVE,
            PAGE_READWRITE,
        ) as *mut u8
    }
    #[cfg(unix)]
    unsafe {
        unsafe extern "C" {
            fn mmap(
                addr: *mut core::ffi::c_void,
                length: usize,
                prot: i32,
                flags: i32,
                fd: i32,
                offset: isize,
            ) -> *mut core::ffi::c_void;
        }
        const PROT_READ_WRITE: i32 = 0x1 | 0x2;
        #[cfg(target_os = "linux")]
        const MAP_PRIVATE_ANON: i32 = 0x02 | 0x20;
        #[cfg(not(target_os = "linux"))]
        const MAP_PRIVATE_ANON: i32 = 0x02 | 0x1000;
        let p = mmap(
            std::ptr::null_mut(),
            size,
            PROT_READ_WRITE,
            MAP_PRIVATE_ANON,
            -1,
            0,
        );
        if p as isize == -1 {
            std::ptr::null_mut()
        } else {
            p as *mut u8
        }
    }
}

/// Host pages at `addr` (any address when 0).
fn os_alloc_host_pages_at(addr: u64, size: usize) -> *mut u8 {
    if addr == 0 {
        return os_alloc_host_pages(size);
    }
    #[cfg(windows)]
    // SAFETY: asks for a specific range; null if it is taken.
    unsafe {
        VirtualAlloc(
            addr as *mut core::ffi::c_void,
            size,
            MEM_COMMIT_RESERVE,
            PAGE_READWRITE,
        ) as *mut u8
    }
    #[cfg(not(windows))]
    {
        let _ = size;
        std::ptr::null_mut()
    }
}

fn os_free_host_pages(ptr: *mut u8, _size: usize) {
    #[cfg(windows)]
    unsafe {
        VirtualFree(ptr as *mut core::ffi::c_void, 0, MEM_RELEASE);
    }
    #[cfg(unix)]
    unsafe {
        unsafe extern "C" {
            fn munmap(addr: *mut core::ffi::c_void, length: usize) -> i32;
        }
        munmap(ptr as *mut core::ffi::c_void, _size);
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn shadowed_mirror_uploads_dirty_pages_and_reads_back() {
        let Ok(runner) = ComputeRunner::new() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let mut uma = UmaManager::new();
        let Ok(alloc) = uma.alloc_shadowed_mirror(&runner, 0x9000_0000, 4 * 4096) else {
            eprintln!("no VK_EXT_external_memory_host: skipped");
            return;
        };
        let cpu = alloc.cpu_ptr().unwrap();
        assert_eq!(
            uma.sync_mirrors(&runner).unwrap(),
            SyncStats::default(),
            "nothing written yet"
        );

        unsafe {
            *cpu = 0xAB;
            *cpu.add(2 * 4096 + 7) = 0xCD;
        }
        let stats = uma.sync_mirrors(&runner).unwrap();
        assert_eq!(
            stats,
            SyncStats {
                ranges: 2,
                bytes: 2 * 4096
            }
        );

        // Scribble on the host copy without syncing; a readback restores
        // what VRAM holds (the synced bytes) and is not itself a title write.
        unsafe {
            *cpu = 0x11;
            *cpu.add(2 * 4096 + 7) = 0x22;
        }
        assert_eq!(
            uma.readback(&runner, 0x9000_0000, 4 * 4096).unwrap(),
            4 * 4096
        );
        assert_eq!(unsafe { (*cpu, *cpu.add(2 * 4096 + 7)) }, (0xAB, 0xCD));
        assert_eq!(uma.sync_mirrors(&runner).unwrap().ranges, 0);
        assert_eq!(
            uma.readback(&runner, 0x1234, 16).unwrap(),
            0,
            "unknown address"
        );
        assert_eq!(
            uma.host_ptr(0x9000_0000 + 4096),
            Some(unsafe { cpu.add(4096) })
        );

        let ident = uma.xmem_alloc_identity(&runner, 0, 8192).unwrap();
        let (va, ptr) = (ident.console_va, ident.cpu_ptr().unwrap());
        assert_eq!(va, ptr as u64, "the title's pointer is the host pointer");
        assert_eq!(uma.host_ptr(va + 8), Some(unsafe { ptr.add(8) }));
        uma.destroy_all(&runner);
    }
}

impl Default for UmaManager {
    fn default() -> Self {
        Self::new()
    }
}
