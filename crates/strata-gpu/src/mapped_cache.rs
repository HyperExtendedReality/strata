//! Clean-room high-throughput submission & memory-stack optimizations ported
//! from sibling compatibility layer engineering:
//! 1. [`MappedRangeCache`]: Cross-submission range-granular VA validity cache
//!    eliminating repeated per-submit OS `VirtualQuery` syscalls.
//! 2. [`DescriptorDirtyBitmap`]: 128-slot chunked dirty bitmap eliminating
//!    full-heap descriptor scans on submit.
//! 3. [`BindlessSlotParker`]: GPU-feedback-driven slot parking so evicted
//!    bindless slots are not re-uploaded until a shader actually reads them again.

use std::collections::{BTreeMap, HashMap};

/// Cached virtual-memory validity span (`[base, base + len)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedRange {
    pub base: u64,
    pub len: u64,
    pub writable: bool,
    pub generation: u64,
}

/// Cross-submission range-granular VA mapping cache (`MappedCache`).
#[derive(Debug, Default)]
pub struct MappedRangeCache {
    ranges: BTreeMap<u64, MappedRange>,
    generation: u64,
    pub hits: u64,
    pub misses: u64,
}

impl MappedRangeCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment generation on `VirtualFree`/`UnmapViewOfFile` invalidation.
    pub fn bump_generation(&mut self) -> u64 {
        self.generation += 1;
        self.ranges.clear();
        self.generation
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Record a verified readable/writable VA span.
    pub fn insert(&mut self, base: u64, len: u64, writable: bool) {
        if len == 0 {
            return;
        }
        self.ranges.insert(
            base,
            MappedRange {
                base,
                len,
                writable,
                generation: self.generation,
            },
        );
    }

    /// Check whether `[addr, addr + size)` is covered by a cached live range.
    pub fn contains(&mut self, addr: u64, size: u64, need_write: bool) -> bool {
        let Some(end) = addr.checked_add(size) else {
            self.misses += 1;
            return false;
        };
        if let Some((_, r)) = self.ranges.range(..=addr).next_back()
            && r.generation == self.generation
            && r.base.saturating_add(r.len) >= end
            && (!need_write || r.writable)
        {
            self.hits += 1;
            return true;
        }
        self.misses += 1;
        false
    }
}

/// Two-level dirty bitmap over descriptor slots (128 slots per chunk), allowing
/// O(dirty_chunks) descriptor heap sync instead of O(total_slots) page scans.
#[derive(Debug)]
pub struct DescriptorDirtyBitmap {
    slot_count: usize,
    /// Per-chunk (128 slots) dirty flag.
    chunk_dirty: Vec<bool>,
    /// Per-slot dirty bit packed into `u64` words (2 words per 128-slot chunk).
    slot_bits: Vec<u64>,
}

impl DescriptorDirtyBitmap {
    pub const CHUNK_SLOTS: usize = 128;

    pub fn new(slot_count: usize) -> Self {
        let chunks = slot_count.div_ceil(Self::CHUNK_SLOTS);
        Self {
            slot_count,
            chunk_dirty: vec![false; chunks],
            slot_bits: vec![0; chunks * 2],
        }
    }

    #[inline]
    pub fn mark_dirty(&mut self, slot: usize) {
        if slot >= self.slot_count {
            return;
        }
        self.chunk_dirty[slot / Self::CHUNK_SLOTS] = true;
        self.slot_bits[slot / 64] |= 1u64 << (slot % 64);
    }

    /// Drain all dirty slot indices in ascending order and clear their bits.
    pub fn drain_dirty(&mut self) -> Vec<usize> {
        let mut out = Vec::new();
        for (ci, dirty) in self.chunk_dirty.iter_mut().enumerate() {
            if !*dirty {
                continue;
            }
            *dirty = false;
            for half in 0..2 {
                let word_idx = ci * 2 + half;
                let mut bits = self.slot_bits[word_idx];
                self.slot_bits[word_idx] = 0;
                while bits != 0 {
                    let tz = bits.trailing_zeros() as usize;
                    let slot = word_idx * 64 + tz;
                    if slot < self.slot_count {
                        out.push(slot);
                    }
                    bits &= bits - 1;
                }
            }
        }
        out
    }
}

/// Tracks evicted bindless slots that are *parked* with their last descriptor
/// hash so redundant re-writes of an unread descriptor don't trigger VRAM
/// re-uploads until GPU usage feedback reports the slot was actually sampled.
#[derive(Debug, Default)]
pub struct BindlessSlotParker {
    /// `slot -> parked_descriptor_hash`
    parked: HashMap<u32, u64>,
    pub skipped_reuploads: u64,
    pub woken_slots: u64,
}

impl BindlessSlotParker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Park an evicted slot under its descriptor hash.
    pub fn park(&mut self, slot: u32, desc_hash: u64) {
        self.parked.insert(slot, desc_hash);
    }

    /// Returns `true` if a descriptor write to `slot` with `desc_hash` should be
    /// skipped because the slot is parked and the descriptor hasn't changed.
    pub fn should_skip_upload(&mut self, slot: u32, desc_hash: u64) -> bool {
        if self.parked.get(&slot).copied() == Some(desc_hash) {
            self.skipped_reuploads += 1;
            true
        } else {
            self.parked.remove(&slot);
            false
        }
    }

    /// Wake parked slots observed as read in the GPU feedback buffer.
    pub fn apply_gpu_feedback(&mut self, sampled_slots: &[u32]) -> Vec<u32> {
        let mut woken = Vec::new();
        for &slot in sampled_slots {
            if self.parked.remove(&slot).is_some() {
                self.woken_slots += 1;
                woken.push(slot);
            }
        }
        woken
    }

    pub fn parked_count(&self) -> usize {
        self.parked.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_range_cache_tracks_hits_misses_and_generations() {
        let mut c = MappedRangeCache::new();
        c.insert(0x1000_0000, 0x4000, true);
        assert!(c.contains(0x1000_1000, 0x1000, true));
        assert!(!c.contains(0x1000_3000, 0x2000, false));
        assert_eq!(c.hits, 1);
        assert_eq!(c.misses, 1);
        c.bump_generation();
        assert!(!c.contains(0x1000_1000, 0x1000, false));
    }

    #[test]
    fn descriptor_dirty_bitmap_drains_exact_slots() {
        let mut bm = DescriptorDirtyBitmap::new(512);
        bm.mark_dirty(3);
        bm.mark_dirty(129);
        bm.mark_dirty(511);
        assert_eq!(bm.drain_dirty(), vec![3, 129, 511]);
        assert!(bm.drain_dirty().is_empty());
    }

    #[test]
    fn bindless_slot_parker_suppresses_reupload_until_gpu_reads_slot() {
        let mut parker = BindlessSlotParker::new();
        parker.park(42, 0xABCD);
        assert!(parker.should_skip_upload(42, 0xABCD));
        assert_eq!(parker.skipped_reuploads, 1);
        assert_eq!(parker.apply_gpu_feedback(&[10, 42, 99]), vec![42]);
        assert!(!parker.should_skip_upload(42, 0xABCD));
        assert_eq!(parker.woken_slots, 1);
    }
}
