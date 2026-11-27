//! Vendor-agnostic `VK_KHR_acceleration_structure` BLAS/TLAS manager.
//!
//! Manages 64 MiB acceleration-structure memory slabs, a 64K-entry open-addressed
//! guest-VA → host-AS-BDA hash table, dynamic BLAS refitting vs rebuild policy,
//! and an async static BLAS compaction queue.

use std::collections::{HashMap, VecDeque};

/// Default acceleration-structure slab size (64 MiB).
pub const AS_SLAB_BYTES: u64 = 64 * 1024 * 1024;
/// Required Vulkan `VK_KHR_acceleration_structure` buffer offset alignment (256 bytes).
pub const AS_ALIGN_BYTES: u64 = 256;
/// Number of buckets in the GPU-readable BLAS-VA hash table (power of 2).
pub const BLAS_HASH_BUCKETS: usize = 65_536;

/// Sub-allocation inside a 64 MiB AS memory slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsSlabSlice {
    pub slab_index: u32,
    pub offset: u64,
    pub size: u64,
    pub device_address: u64,
}

#[derive(Debug)]
struct AsSlab {
    base_bda: u64,
    capacity: u64,
    cursor: u64,
}

/// Tracked BLAS entry mapped from a console guest BVH address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlasEntry {
    pub guest_va: u64,
    pub slice: AsSlabSlice,
    pub primitive_count: u32,
    pub allow_update: bool,
    pub compacted: bool,
    pub last_used_frame: u64,
}

/// GPU-side 16-byte hash table bucket mapping `guest_va` → `host_as_bda`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BlasHashBucket {
    pub guest_va: u64,
    pub host_as_bda: u64,
}

/// Pending static BLAS compaction request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactRequest {
    pub guest_va: u64,
    pub compacted_size: u64,
}

/// Acceleration-structure manager stats.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RtStats {
    pub slab_count: usize,
    pub blas_count: usize,
    pub refits: u64,
    pub rebuilds: u64,
    pub compactions: u64,
    pub total_allocated_bytes: u64,
}

/// Vendor-agnostic `VK_KHR_acceleration_structure` BLAS/TLAS slab & hash-table manager.
pub struct AccelStructManager {
    slabs: Vec<AsSlab>,
    blas_by_guest: HashMap<u64, BlasEntry>,
    hash_table: Vec<BlasHashBucket>,
    compact_queue: VecDeque<CompactRequest>,
    tlas_bda: u64,
    stats: RtStats,
}

impl Default for AccelStructManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AccelStructManager {
    pub fn new() -> Self {
        Self {
            slabs: Vec::new(),
            blas_by_guest: HashMap::new(),
            hash_table: vec![BlasHashBucket::default(); BLAS_HASH_BUCKETS],
            compact_queue: VecDeque::new(),
            tlas_bda: 0,
            stats: RtStats::default(),
        }
    }

    pub fn stats(&self) -> RtStats {
        self.stats
    }

    pub fn tlas_bda(&self) -> u64 {
        self.tlas_bda
    }

    pub fn set_tlas_bda(&mut self, bda: u64) {
        self.tlas_bda = bda;
    }

    /// Register a newly allocated 64 MiB Vulkan AS backing slab with its base BDA.
    pub fn add_slab(&mut self, base_bda: u64, capacity: u64) -> u32 {
        let idx = self.slabs.len() as u32;
        self.slabs.push(AsSlab {
            base_bda,
            capacity,
            cursor: 0,
        });
        self.stats.slab_count = self.slabs.len();
        idx
    }

    /// Sub-allocate a 256-byte-aligned AS slice from existing slabs.
    pub fn alloc_slice(&mut self, size: u64) -> Option<AsSlabSlice> {
        let aligned_size = align_up(size.max(AS_ALIGN_BYTES), AS_ALIGN_BYTES);
        for (i, slab) in self.slabs.iter_mut().enumerate() {
            let off = align_up(slab.cursor, AS_ALIGN_BYTES);
            if off.saturating_add(aligned_size) <= slab.capacity {
                slab.cursor = off + aligned_size;
                self.stats.total_allocated_bytes += aligned_size;
                return Some(AsSlabSlice {
                    slab_index: i as u32,
                    offset: off,
                    size: aligned_size,
                    device_address: slab.base_bda + off,
                });
            }
        }
        None
    }

    /// Register or update a BLAS for `guest_va`.
    /// Returns `true` if an existing dynamic BLAS can be refitted in-place.
    pub fn register_blas(
        &mut self,
        guest_va: u64,
        required_bytes: u64,
        primitive_count: u32,
        allow_update: bool,
        frame: u64,
    ) -> Option<(BlasEntry, bool)> {
        if let Some(existing) = self.blas_by_guest.get_mut(&guest_va) {
            if existing.allow_update
                && allow_update
                && existing.primitive_count == primitive_count
                && required_bytes <= existing.slice.size
            {
                existing.last_used_frame = frame;
                self.stats.refits += 1;
                return Some((*existing, true));
            }
        }
        let slice = self.alloc_slice(required_bytes)?;
        let entry = BlasEntry {
            guest_va,
            slice,
            primitive_count,
            allow_update,
            compacted: false,
            last_used_frame: frame,
        };
        self.blas_by_guest.insert(guest_va, entry);
        self.insert_hash(guest_va, slice.device_address);
        self.stats.blas_count = self.blas_by_guest.len();
        self.stats.rebuilds += 1;
        Some((entry, false))
    }

    /// Queue a static BLAS for asynchronous compaction once its queried compacted size is known.
    pub fn enqueue_compaction(&mut self, guest_va: u64, compacted_size: u64) {
        if let Some(entry) = self.blas_by_guest.get(&guest_va) {
            if !entry.allow_update && !entry.compacted && compacted_size < entry.slice.size {
                self.compact_queue.push_back(CompactRequest {
                    guest_va,
                    compacted_size,
                });
            }
        }
    }

    /// Process up to `budget` pending BLAS compactions, updating the GPU hash table.
    pub fn process_compactions(&mut self, budget: usize) -> usize {
        let mut done = 0usize;
        while done < budget {
            let Some(req) = self.compact_queue.pop_front() else {
                break;
            };
            let Some(new_slice) = self.alloc_slice(req.compacted_size) else {
                self.compact_queue.push_front(req);
                break;
            };
            if let Some(entry) = self.blas_by_guest.get_mut(&req.guest_va) {
                entry.slice = new_slice;
                entry.compacted = true;
                self.insert_hash(req.guest_va, new_slice.device_address);
                self.stats.compactions += 1;
                done += 1;
            }
        }
        done
    }

    /// Lookup a host `VkAccelerationStructureKHR` device address by guest BVH VA.
    pub fn lookup_host_bda(&self, guest_va: u64) -> Option<u64> {
        let mask = BLAS_HASH_BUCKETS - 1;
        let mut idx = hash_va(guest_va) & mask;
        for _ in 0..64 {
            let b = self.hash_table[idx];
            if b.guest_va == guest_va {
                return Some(b.host_as_bda);
            }
            if b.guest_va == 0 {
                return None;
            }
            idx = (idx + 1) & mask;
        }
        self.blas_by_guest
            .get(&guest_va)
            .map(|e| e.slice.device_address)
    }

    /// Raw GPU-uploadable hash table slice (`65536 * 16 B = 1 MiB`).
    pub fn hash_table_slice(&self) -> &[BlasHashBucket] {
        &self.hash_table
    }

    fn insert_hash(&mut self, guest_va: u64, host_as_bda: u64) {
        let mask = BLAS_HASH_BUCKETS - 1;
        let mut idx = hash_va(guest_va) & mask;
        for _ in 0..64 {
            let b = &mut self.hash_table[idx];
            if b.guest_va == 0 || b.guest_va == guest_va {
                b.guest_va = guest_va;
                b.host_as_bda = host_as_bda;
                return;
            }
            idx = (idx + 1) & mask;
        }
    }
}

#[inline]
fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

#[inline]
fn hash_va(va: u64) -> usize {
    let mut x = va ^ (va >> 33);
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_refits_compacts_and_hashes_blas() {
        let mut mgr = AccelStructManager::new();
        mgr.add_slab(0x1000_0000, AS_SLAB_BYTES);

        // Dynamic BLAS: build then refit in-place
        let (b1, refit1) = mgr.register_blas(0xDEAD_0000, 4096, 128, true, 1).unwrap();
        assert!(!refit1);
        assert_eq!(
            mgr.lookup_host_bda(0xDEAD_0000),
            Some(b1.slice.device_address)
        );

        let (b2, refit2) = mgr.register_blas(0xDEAD_0000, 4096, 128, true, 2).unwrap();
        assert!(refit2);
        assert_eq!(b1.slice.device_address, b2.slice.device_address);

        // Static BLAS: build then compact
        let (s1, _) = mgr.register_blas(0xBEEF_0000, 8192, 512, false, 1).unwrap();
        mgr.enqueue_compaction(0xBEEF_0000, 2048);
        assert_eq!(mgr.process_compactions(4), 1);
        let compacted_bda = mgr.lookup_host_bda(0xBEEF_0000).unwrap();
        assert_ne!(compacted_bda, s1.slice.device_address);
        assert_eq!(mgr.stats().refits, 1);
        assert_eq!(mgr.stats().compactions, 1);
    }
}
