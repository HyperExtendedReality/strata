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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

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
    baseline: Option<BdaBuffer>,
    retry_upload: AtomicBool,
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
            if let Some(baseline) = shadow.baseline {
                runner.free_bda_buffer(baseline);
            }
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
    mirror_diff: bool,
    upload_bytes: AtomicU64,
    readback_requested_bytes: AtomicU64,
    readback_bytes: AtomicU64,
    sync_wait_ns: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
pub struct TransferStats {
    pub upload_bytes: u64,
    pub readback_requested_bytes: u64,
    pub readback_bytes: u64,
    pub sync_wait_ns: u64,
}

impl UmaManager {
    pub fn new() -> Self {
        Self {
            address_space: AddressSpace::new(),
            sparse_pool: SparseTilePool::new(),
            allocations: HashMap::new(),
            mirror_diff: std::env::var("STRATA_GPU_MIRROR_DIFF").is_ok_and(|v| v == "1"),
            upload_bytes: AtomicU64::new(0),
            readback_requested_bytes: AtomicU64::new(0),
            readback_bytes: AtomicU64::new(0),
            sync_wait_ns: AtomicU64::new(0),
        }
    }

    pub fn transfer_stats(&self) -> TransferStats {
        TransferStats {
            upload_bytes: self.upload_bytes.load(Ordering::Relaxed),
            readback_requested_bytes: self.readback_requested_bytes.load(Ordering::Relaxed),
            readback_bytes: self.readback_bytes.load(Ordering::Relaxed),
            sync_wait_ns: self.sync_wait_ns.load(Ordering::Relaxed),
        }
    }

    /// Prefer an opt-in mirror for large resources owned by this queue's UMA.
    pub fn allocate_resource(
        &mut self,
        runner: &ComputeRunner,
        size: u64,
    ) -> Result<&UmaAllocation, GpuError> {
        if self.mirror_diff && size >= 1024 * 1024 {
            let mirrored = self
                .alloc_shadowed_mirror(runner, 0, size)
                .map(|a| a.console_va);
            if let Ok(address) = mirrored {
                return Ok(&self.allocations[&address]);
            }
        }
        self.xmem_alloc_identity(runner, 0, size)
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
        let console_va = if console_va == 0 {
            pages.as_ptr() as u64
        } else {
            console_va
        };
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
        let baseline = if self.mirror_diff && size >= 1024 * 1024 {
            runner
                .alloc_vram_bda_buffer(size, &[])
                .ok()
                .and_then(|baseline| {
                    if runner
                        .copy_buffer_regions(host.buffer, baseline.buffer, &[(0, 0, size)])
                        .is_ok()
                    {
                        Some(baseline)
                    } else {
                        runner.free_bda_buffer(baseline);
                        None
                    }
                })
        } else {
            None
        };
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
                shadow: Some(Shadow {
                    pages,
                    host,
                    baseline,
                    retry_upload: AtomicBool::new(false),
                }),
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
            let mut ranges = shadow.pages.dirty_ranges(true);
            if shadow.retry_upload.swap(false, Ordering::Relaxed) {
                ranges = vec![(0, alloc.size as usize)];
            }
            if ranges.is_empty() {
                continue;
            }
            let start = Instant::now();
            let moved = if let Some(baseline) = &shadow.baseline {
                ranges.iter().try_fold(0, |bytes, &(off, len)| {
                    diff_mirror(
                        runner, alloc, shadow, baseline, off as u64, len as u64, true,
                    )
                    .map(|n| bytes + n)
                })
            } else {
                let regions: Vec<_> = ranges
                    .iter()
                    .map(|&(o, n)| (o as u64, o as u64, n as u64))
                    .collect();
                runner
                    .copy_buffer_regions(shadow.host.buffer, alloc.buffer.buffer, &regions)
                    .map(|()| regions.iter().map(|r| r.2).sum::<u64>())
            };
            self.sync_wait_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if moved.is_err() {
                shadow.retry_upload.store(true, Ordering::Relaxed);
            }
            let moved = moved?;
            self.upload_bytes.fetch_add(moved, Ordering::Relaxed);
            stats.ranges += ranges.len();
            stats.bytes += moved;
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
        if len == 0 {
            return Ok(0);
        }
        let start = Instant::now();
        let moved = if let Some(baseline) = &shadow.baseline {
            diff_mirror(runner, alloc, shadow, baseline, off, len, false)?
        } else {
            runner.copy_buffer_regions(
                alloc.buffer.buffer,
                shadow.host.buffer,
                &[(off, off, len)],
            )?;
            shadow.pages.reset_range(off as usize, len as usize);
            len
        };
        self.sync_wait_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.readback_requested_bytes
            .fetch_add(len, Ordering::Relaxed);
        self.readback_bytes.fetch_add(moved, Ordering::Relaxed);
        Ok(moved)
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

    /// Updates 64 KiB sparse tile mappings for a reserved resource range.
    /// When `unbind` is true, releases the mapped range back to the pool;
    /// otherwise acquires/aliases physical tiles in `SparseTilePool`.
    pub fn update_tile_mappings(
        &mut self,
        resource_va: u64,
        start_tile: u32,
        tile_count: u32,
        unbind: bool,
    ) -> usize {
        let base = resource_va.saturating_add(u64::from(start_tile) * 65_536);
        let bytes = u64::from(tile_count.max(1)) * 65_536;
        if unbind {
            self.sparse_pool.release_range(base, bytes);
            0
        } else {
            self.sparse_pool.acquire_range(base, bytes).len()
        }
    }

    /// Copies sparse tile bindings from `src_va` to `dst_va` for `tile_count` 64 KiB tiles.
    pub fn copy_tile_mappings(&mut self, dst_va: u64, src_va: u64, tile_count: u32) -> usize {
        let bytes = u64::from(tile_count.max(1)) * 65_536;
        let _ = src_va;
        self.sparse_pool.acquire_range(dst_va, bytes).len()
    }

    /// Performs a GPU-side `vkCmdCopyBuffer` when both `src_va` and `dst_va`
    /// fall inside live UMA allocations. Returns `Ok(true)` if executed on the
    /// GPU, or `Ok(false)` if either address is not UMA-backed (caller falls
    /// back to host memory copy).
    pub fn gpu_copy_buffer(
        &self,
        runner: &ComputeRunner,
        dst_va: u64,
        src_va: u64,
        len: u64,
    ) -> Result<bool, GpuError> {
        if len == 0 {
            return Ok(true);
        }
        let find = |va: u64| {
            self.allocations
                .values()
                .find(|a| va >= a.console_va && va.saturating_add(len) <= a.console_va + a.size)
        };
        let (Some(src), Some(dst)) = (find(src_va), find(dst_va)) else {
            return Ok(false);
        };
        let src_off = src_va - src.console_va;
        let dst_off = dst_va - dst.console_va;
        runner.copy_buffer_regions(
            src.buffer.buffer,
            dst.buffer.buffer,
            &[(src_off, dst_off, len)],
        )?;
        Ok(true)
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

    /// The allocation an address falls in, whether it is a console VA (what
    /// the title's CPU code uses) or the 64-bit BDA the GPU sees (what
    /// `GetGPUVirtualAddress` hands out, and so what descriptors hold);
    /// with the address's offset into it.
    fn locate(&self, addr: u64) -> Option<(&UmaAllocation, u64)> {
        self.allocations
            .values()
            .find_map(|a| {
                (addr >= a.console_va && addr < a.console_va + a.size)
                    .then(|| (a, addr - a.console_va))
            })
            .or_else(|| {
                self.allocations.values().find_map(|a| {
                    (addr >= a.host_bda && addr < a.host_bda + a.size)
                        .then(|| (a, addr - a.host_bda))
                })
            })
    }

    /// The CPU address of `addr`, a console VA or a GPU BDA.
    pub fn host_ptr_any(&self, addr: u64) -> Option<*mut u8> {
        let (a, off) = self.locate(addr)?;
        // SAFETY: the offset is inside the allocation.
        a.cpu_ptr().map(|p| unsafe { p.add(off as usize) })
    }

    /// The GPU BDA of `addr`, a console VA or already a BDA.
    pub fn bda_of(&self, addr: u64) -> Option<u64> {
        let (a, off) = self.locate(addr)?;
        Some(a.host_bda + off)
    }

    /// The Vulkan buffer holding `addr` (console VA or BDA) and the offset
    /// into it: what fixed-function reads (index buffers, indirect
    /// arguments) bind directly, without a copy.
    pub fn buffer_at(&self, addr: u64) -> Option<(ash::vk::Buffer, u64, u64)> {
        let (a, off) = self.locate(addr)?;
        Some((a.buffer.buffer, off, a.size - off))
    }

    /// The console VA of `addr` (a console VA or a GPU BDA).
    pub fn console_va_of(&self, addr: u64) -> Option<u64> {
        let (a, off) = self.locate(addr)?;
        Some(a.console_va + off)
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

fn diff_mirror(
    runner: &ComputeRunner,
    allocation: &UmaAllocation,
    shadow: &Shadow,
    baseline: &BdaBuffer,
    offset: u64,
    len: u64,
    upload: bool,
) -> Result<u64, GpuError> {
    static SHADER: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    let shader = SHADER.get_or_init(|| {
        strata_runtime_shaders::MIRROR_DIFF_SPV
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect()
    });
    let mut counter = runner.alloc_bda_buffer(4)?;
    let result = (|| {
        let mut done = 0;
        let mut changed = 0;
        while done < len {
            unsafe {
                counter.as_mut_slice::<u32>(1)[0] = 0;
            }
            let start = offset + done;
            let begin = start & !3;
            let first = (start - begin) as u32;
            let count = (len - done).min(64 * 1024 * 1024 - u64::from(first)) as u32;
            let words = (count + first).div_ceil(4);
            let mut push = [0u8; 48];
            for (dst, value) in push[..32].chunks_exact_mut(8).zip([
                allocation.buffer.bda + begin,
                shadow.host.bda + begin,
                baseline.bda + begin,
                counter.bda,
            ]) {
                dst.copy_from_slice(&value.to_le_bytes());
            }
            for (dst, value) in
                push[32..]
                    .chunks_exact_mut(4)
                    .zip([words, first, count, u32::from(upload)])
            {
                dst.copy_from_slice(&value.to_le_bytes());
            }
            runner.dispatch_spirv(shader, &push, [words.div_ceil(64), 1, 1])?;
            changed += u64::from(unsafe { counter.as_slice::<u32>(1)[0] });
            done += u64::from(count);
        }
        Ok(changed)
    })();
    runner.free_bda_buffer(counter);
    result
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
    fn sparse_gpu_readback_preserves_pending_cpu_bytes_in_the_same_word() {
        let Ok(runner) = ComputeRunner::new() else {
            eprintln!("SKIPPED mirror_diff: no Vulkan device");
            return;
        };
        if runner.min_host_pointer_alignment == 0 {
            eprintln!("SKIPPED mirror_diff: no VK_EXT_external_memory_host");
            return;
        }
        let mut uma = UmaManager::new();
        uma.mirror_diff = true;
        let addr = 0x9000_0000;
        let allocation = uma
            .alloc_shadowed_mirror(&runner, addr, 1024 * 1024)
            .unwrap();
        assert!(allocation.shadow.as_ref().unwrap().baseline.is_some());
        let cpu = allocation.cpu_ptr().unwrap();
        let vram = allocation.buffer.buffer;
        let mut upload = runner.alloc_bda_buffer(4).unwrap();
        unsafe {
            *cpu = 99;
            *cpu.add(3) = 88;
            upload
                .as_mut_slice::<u8>(4)
                .copy_from_slice(&[11, 22, 33, 44]);
        }
        runner
            .copy_buffer_regions(upload.buffer, vram, &[(0, 0, 4)])
            .unwrap();
        assert_eq!(uma.readback(&runner, addr + 1, 2).unwrap(), 2);
        assert_eq!(uma.readback(&runner, addr, 1024 * 1024).unwrap(), 0);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(cpu, 4) },
            [99, 22, 33, 88]
        );
        assert_eq!(uma.sync_mirrors(&runner).unwrap().bytes, 2);
        runner
            .copy_buffer_regions(vram, upload.buffer, &[(0, 0, 4)])
            .unwrap();
        assert_eq!(unsafe { upload.as_slice::<u8>(4) }, [99, 22, 33, 88]);
        assert_eq!(uma.transfer_stats().readback_bytes, 2);
        runner.free_bda_buffer(upload);
        uma.destroy_all(&runner);
    }

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
