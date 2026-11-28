//! Vendor-agnostic Vulkan 1.3+ Memory Stack:
//! - Bindless feedback eviction & slot parking (evict idle textures unread for `idle_window_s`)
//! - Transient ring trimming (shrink post-burst detile/staging rings back to baseline)
//! - 64 KiB slab packing for 256-byte indirect budgets & command preludes
//! - 1 MiB chunked / 64-byte cache-line dirty comparison for UMA coherence
//! - Per-category VRAM/RAM telemetry (`ramline`) & 14-bucket texture streaming latency (`texlate`)

use std::collections::HashMap;

/// Size of a packed slab for 256-byte indirect budgets / command preludes (64 KiB).
pub const PACKED_SLAB_BYTES: u64 = 64 * 1024;
/// Slot stride inside a packed budget slab (256 bytes).
pub const BUDGET_SLOT_BYTES: u64 = 256;
/// Chunk size for piecewise shadow-buffer diffing (1 MiB).
pub const COMPARE_PIECE_BYTES: usize = 1024 * 1024;
/// Cache-line granularity for CPU-vs-GPU dirty chunk detection (64 bytes).
pub const DIRTY_CHUNK_BYTES: usize = 64;

/// State of a bindless texture slot tracked by [`BindlessFeedbackEvictor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    Resident,
    Parked,
    Evicted,
}

#[derive(Debug, Clone, Copy)]
struct TrackedSlot {
    created_sec: u64,
    last_read_sec: u64,
    bytes: u64,
    view_hash: u64,
    state: SlotState,
}

/// Bindless texture feedback evictor with slot parking.
///
/// Shaders instrumented with `STRATA_BINDLESS_FEEDBACK=1` set a read bit per
/// sampled slot. Slots that lived the full `idle_window_s` without being read
/// are evicted when VRAM usage exceeds `high_water_pct`, down to `low_water_pct`.
/// Re-writing the same view hash to an evicted slot parks it without re-allocating
/// until the GPU actually reads it.
#[derive(Debug)]
pub struct BindlessFeedbackEvictor {
    slots: HashMap<u32, TrackedSlot>,
    idle_window_s: u64,
    high_water_bytes: u64,
    low_water_bytes: u64,
    resident_bytes: u64,
    pub evictions: u64,
    pub parked_hits: u64,
}

impl BindlessFeedbackEvictor {
    pub fn new(idle_window_s: u64, high_water_bytes: u64, low_water_bytes: u64) -> Self {
        Self {
            slots: HashMap::new(),
            idle_window_s,
            high_water_bytes,
            low_water_bytes: low_water_bytes.min(high_water_bytes),
            resident_bytes: 0,
            evictions: 0,
            parked_hits: 0,
        }
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn high_water_bytes(&self) -> u64 {
        self.high_water_bytes
    }

    pub fn set_watermarks(&mut self, high: u64, low: u64) {
        self.high_water_bytes = high;
        self.low_water_bytes = low.min(high);
    }

    /// Stops tracking `slot` (its resource was destroyed).
    pub fn forget(&mut self, slot: u32) {
        if let Some(e) = self.slots.remove(&slot)
            && e.state == SlotState::Resident
        {
            self.resident_bytes = self.resident_bytes.saturating_sub(e.bytes);
        }
    }

    /// Record a descriptor write to `slot`. If the slot was evicted and the title
    /// writes the exact same `view_hash`, park it instead of re-allocating VRAM.
    pub fn on_write(&mut self, slot: u32, view_hash: u64, bytes: u64, now_sec: u64) -> SlotState {
        if let Some(existing) = self.slots.get_mut(&slot) {
            if existing.state == SlotState::Evicted && existing.view_hash == view_hash {
                existing.state = SlotState::Parked;
                self.parked_hits += 1;
                return SlotState::Parked;
            }
            if existing.state == SlotState::Resident {
                self.resident_bytes = self.resident_bytes.saturating_sub(existing.bytes);
            }
        }
        self.slots.insert(
            slot,
            TrackedSlot {
                created_sec: now_sec,
                last_read_sec: now_sec,
                bytes,
                view_hash,
                state: SlotState::Resident,
            },
        );
        self.resident_bytes += bytes;
        SlotState::Resident
    }

    /// Record GPU shader feedback indicating `slot` was sampled at `now_sec`.
    pub fn on_gpu_read(&mut self, slot: u32, now_sec: u64) {
        if let Some(entry) = self.slots.get_mut(&slot) {
            entry.last_read_sec = now_sec;
            if entry.state != SlotState::Resident {
                entry.state = SlotState::Resident;
                self.resident_bytes += entry.bytes;
            }
        }
    }

    /// Evict oldest eligible slots (those that lived the full `idle_window_s` unread)
    /// while `resident_bytes > high_water_bytes`, stopping once `<= low_water_bytes`.
    pub fn sweep(&mut self, now_sec: u64) -> Vec<u32> {
        self.sweep_at_most(now_sec, usize::MAX, |_| true)
    }

    pub(crate) fn sweep_at_most(
        &mut self,
        now_sec: u64,
        limit: usize,
        eligible: impl Fn(u32) -> bool,
    ) -> Vec<u32> {
        if self.resident_bytes <= self.high_water_bytes {
            return Vec::new();
        }
        let mut candidates: Vec<(u64, u32)> = self
            .slots
            .iter()
            .filter(|&(&slot, e)| {
                e.state == SlotState::Resident
                    && now_sec.saturating_sub(e.created_sec) >= self.idle_window_s
                    && now_sec.saturating_sub(e.last_read_sec) >= self.idle_window_s
                    && eligible(slot)
            })
            .map(|(&slot, e)| (e.last_read_sec, slot))
            .collect();
        candidates.sort_unstable();

        let mut evicted = Vec::new();
        for (_, slot) in candidates.into_iter().take(limit) {
            if self.resident_bytes <= self.low_water_bytes {
                break;
            }
            if let Some(entry) = self.slots.get_mut(&slot) {
                entry.state = SlotState::Evicted;
                self.resident_bytes = self.resident_bytes.saturating_sub(entry.bytes);
                self.evictions += 1;
                evicted.push(slot);
            }
        }
        evicted
    }
}

/// Manages shrinking transient GPU rings (e.g. detile/staging buffers) back to
/// a steady-state baseline after loading bursts finish.
#[derive(Debug, Clone)]
pub struct TransientRingTrimmer {
    pub baseline_bytes: u64,
    pub current_bytes: u64,
    pub peak_recent_bytes: u64,
    idle_windows: u32,
    shrink_after_windows: u32,
}

impl TransientRingTrimmer {
    pub fn new(baseline_bytes: u64, shrink_after_windows: u32) -> Self {
        Self {
            baseline_bytes,
            current_bytes: baseline_bytes,
            peak_recent_bytes: 0,
            idle_windows: 0,
            shrink_after_windows: shrink_after_windows.max(1),
        }
    }

    /// Record bytes required during a frame; grows `current_bytes` if needed.
    pub fn record_usage(&mut self, used_bytes: u64) {
        if used_bytes > self.peak_recent_bytes {
            self.peak_recent_bytes = used_bytes;
        }
        if used_bytes > self.current_bytes {
            self.current_bytes = used_bytes.next_power_of_two();
            self.idle_windows = 0;
        }
    }

    /// Periodic tick (e.g. every 30s). Returns `Some(new_bytes)` when the ring should shrink.
    pub fn tick_trim(&mut self) -> Option<u64> {
        let peak = std::mem::take(&mut self.peak_recent_bytes);
        if self.current_bytes > self.baseline_bytes && peak <= self.baseline_bytes {
            self.idle_windows += 1;
            if self.idle_windows >= self.shrink_after_windows {
                self.current_bytes = self.baseline_bytes;
                self.idle_windows = 0;
                return Some(self.baseline_bytes);
            }
        } else {
            self.idle_windows = 0;
        }
        None
    }
}

/// Packs 256-byte indirect-work budgets / preludes into shared 64 KiB BDA slabs.
#[derive(Debug, Default)]
pub struct BudgetSlabPacker {
    slabs: Vec<(u64, u32)>, // (base_bda, used_slots)
}

impl BudgetSlabPacker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_slab(&mut self, base_bda: u64) {
        self.slabs.push((base_bda, 0));
    }

    /// Allocate one 256-byte slot inside an existing 64 KiB slab.
    pub fn alloc_slot(&mut self) -> Option<u64> {
        let max_slots = (PACKED_SLAB_BYTES / BUDGET_SLOT_BYTES) as u32;
        for (base, used) in &mut self.slabs {
            if *used < max_slots {
                let addr = *base + u64::from(*used) * BUDGET_SLOT_BYTES;
                *used += 1;
                return Some(addr);
            }
        }
        None
    }

    pub fn slab_count(&self) -> usize {
        self.slabs.len()
    }
}

/// Compare two shadow buffers in bounded 1 MiB pieces and return changed 64-byte
/// cache-line chunk indices so CPU uploads never clobber adjacent GPU writes.
pub fn diff_dirty_64b_chunks(current: &[u8], last_synced: &[u8]) -> Vec<usize> {
    let len = current.len().min(last_synced.len());
    let mut dirty = Vec::new();
    let mut piece_off = 0usize;
    while piece_off < len {
        let piece_end = (piece_off + COMPARE_PIECE_BYTES).min(len);
        let cur_piece = &current[piece_off..piece_end];
        let old_piece = &last_synced[piece_off..piece_end];
        let mut local = 0usize;
        while local < cur_piece.len() {
            let end = (local + DIRTY_CHUNK_BYTES).min(cur_piece.len());
            if cur_piece[local..end] != old_piece[local..end] {
                dirty.push((piece_off + local) / DIRTY_CHUNK_BYTES);
            }
            local += DIRTY_CHUNK_BYTES;
        }
        piece_off = piece_end;
    }
    dirty
}

/// 14-bucket logarithmic latency histogram (in microseconds) for texture streaming stages.
#[derive(Debug, Default, Clone)]
pub struct LatencyHistogram {
    pub buckets: [u64; 14],
    pub total_samples: u64,
    pub max_us: u64,
}

impl LatencyHistogram {
    /// Record a sample in microseconds (`<100us`, `<200us`, ..., `>=819.2ms`).
    pub fn record_us(&mut self, us: u64) {
        self.total_samples += 1;
        if us > self.max_us {
            self.max_us = us;
        }
        let mut idx = 0usize;
        let mut threshold = 100u64;
        while idx < 13 && us >= threshold {
            threshold *= 2;
            idx += 1;
        }
        self.buckets[idx] += 1;
    }
}

/// Per-category VRAM/RAM memory telemetry (`ramline` equivalent).
#[derive(Debug, Default, Clone)]
pub struct MemoryTelemetry {
    pub vram_textures_bytes: u64,
    pub vram_buffers_bytes: u64,
    pub vram_rt_as_bytes: u64,
    pub vram_staging_bytes: u64,
    pub host_commit_bytes: u64,
    pub detile_latency: LatencyHistogram,
    pub upload_latency: LatencyHistogram,
}

impl MemoryTelemetry {
    pub fn total_vram_bytes(&self) -> u64 {
        self.vram_textures_bytes
            + self.vram_buffers_bytes
            + self.vram_rt_as_bytes
            + self.vram_staging_bytes
    }

    pub fn to_csv_line(&self, elapsed_s: u64) -> String {
        format!(
            "{},{},{},{},{},{},{}",
            elapsed_s,
            self.total_vram_bytes() / (1024 * 1024),
            self.vram_textures_bytes / (1024 * 1024),
            self.vram_buffers_bytes / (1024 * 1024),
            self.vram_rt_as_bytes / (1024 * 1024),
            self.vram_staging_bytes / (1024 * 1024),
            self.host_commit_bytes / (1024 * 1024),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindless_evictor_evicts_idle_and_parks_rewrites() {
        let mut ev = BindlessFeedbackEvictor::new(60, 1000, 500);
        assert_eq!(ev.on_write(1, 0xAAAA, 600, 0), SlotState::Resident);
        assert_eq!(ev.on_write(2, 0xBBBB, 600, 10), SlotState::Resident);
        ev.on_gpu_read(2, 55); // slot 2 recently read

        // At t=65, slot 1 is unread for 65s (>= 60s), slot 2 is unread for 10s
        let evicted = ev.sweep(65);
        assert_eq!(evicted, vec![1]);
        assert_eq!(ev.resident_bytes(), 600);

        // Re-writing same hash to evicted slot 1 parks it without consuming VRAM
        assert_eq!(ev.on_write(1, 0xAAAA, 600, 66), SlotState::Parked);
        assert_eq!(ev.parked_hits, 1);
        assert_eq!(ev.resident_bytes(), 600);
    }

    #[test]
    fn ring_trimmer_shrinks_after_burst() {
        let mut t = TransientRingTrimmer::new(64 * 1024 * 1024, 2);
        t.record_usage(200 * 1024 * 1024);
        assert_eq!(t.current_bytes, 256 * 1024 * 1024);
        assert_eq!(t.tick_trim(), None); // burst window
        t.record_usage(16 * 1024 * 1024);
        assert_eq!(t.tick_trim(), None); // idle window 1
        t.record_usage(16 * 1024 * 1024);
        assert_eq!(t.tick_trim(), Some(64 * 1024 * 1024)); // idle window 2 -> shrink
    }

    #[test]
    fn diff_dirty_64b_chunks_detects_exact_cache_lines() {
        let a = vec![0u8; 4096];
        let mut b = a.clone();
        b[130] = 0xFF; // chunk 2
        b[4095] = 0x01; // chunk 63
        assert_eq!(diff_dirty_64b_chunks(&b, &a), vec![2, 63]);
    }

    #[test]
    fn budget_slab_packer_packs_256_slots_per_64k_slab() {
        let mut p = BudgetSlabPacker::new();
        p.add_slab(0x5000_0000);
        for i in 0..256u64 {
            assert_eq!(p.alloc_slot(), Some(0x5000_0000 + i * 256));
        }
        assert_eq!(p.alloc_slot(), None);
    }
}
