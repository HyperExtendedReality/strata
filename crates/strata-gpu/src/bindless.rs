//! Fence-gated, content-addressed Bindless Descriptor Slot Manager.
//!
//! Solves two major issues identified in VIctory-7900XTX and victory-fork:
//! 1. Deduplicates console 4-dword hardware T#/V#/S# descriptors so we never
//!    rebuild or scan 400,000 mostly-empty descriptor slots per frame.
//! 2. `VICTORY_SHOW_AFTER` pattern: textures uploaded on the dedicated async
//!    Copy Queue remain invisible (bound to fallback slot 0) until their upload
//!    fence completes, eliminating mid-upload page-fault device losses.

use std::collections::HashMap;

/// Raw 128-bit console hardware descriptor (T# / V# / S#).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HardwareDesc(pub [u32; 4]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindlessSlot(pub u32);

/// Slot 0 is reserved for a safe fallback descriptor (null / transparent black).
pub const FALLBACK_SLOT: BindlessSlot = BindlessSlot(0);

#[derive(Debug, Clone, Copy)]
struct SlotRecord {
    slot: BindlessSlot,
    ready_copy_fence: u64,
    last_used_frame: u64,
}

#[derive(Debug)]
pub struct BindlessManager {
    capacity: u32,
    next_slot: u32,
    free_slots: Vec<BindlessSlot>,
    cache: HashMap<HardwareDesc, SlotRecord>,
    /// Slots waiting on an async copy-queue fence before becoming visible.
    pending_uploads: Vec<(HardwareDesc, u64)>,
}

impl BindlessManager {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            next_slot: 1, // slot 0 = FALLBACK_SLOT
            free_slots: Vec::new(),
            cache: HashMap::new(),
            pending_uploads: Vec::new(),
        }
    }

    /// Register or look up a hardware descriptor, gating visibility on `upload_fence`.
    /// Returns `FALLBACK_SLOT` if the descriptor's async copy-queue upload hasn't
    /// reached `completed_copy_fence` yet.
    pub fn resolve(
        &mut self,
        desc: HardwareDesc,
        upload_fence: u64,
        completed_copy_fence: u64,
        frame: u64,
    ) -> BindlessSlot {
        if let Some(rec) = self.cache.get_mut(&desc) {
            rec.last_used_frame = frame;
            return if completed_copy_fence >= rec.ready_copy_fence {
                rec.slot
            } else {
                FALLBACK_SLOT
            };
        }

        let slot = self.alloc_slot(frame);
        self.cache.insert(
            desc,
            SlotRecord {
                slot,
                ready_copy_fence: upload_fence,
                last_used_frame: frame,
            },
        );
        if upload_fence > completed_copy_fence {
            self.pending_uploads.push((desc, upload_fence));
            FALLBACK_SLOT
        } else {
            slot
        }
    }

    /// Advance the completed copy-queue fence and return all descriptors that
    /// just became safe to publish into the live Vulkan bindless heap.
    pub fn publish_ready(&mut self, completed_copy_fence: u64) -> Vec<(HardwareDesc, BindlessSlot)> {
        let mut ready = Vec::new();
        self.pending_uploads.retain(|&(desc, fence)| {
            if completed_copy_fence >= fence {
                if let Some(rec) = self.cache.get(&desc) {
                    ready.push((desc, rec.slot));
                }
                false
            } else {
                true
            }
        });
        ready
    }

    fn alloc_slot(&mut self, current_frame: u64) -> BindlessSlot {
        if let Some(s) = self.free_slots.pop() {
            return s;
        }
        if self.next_slot < self.capacity {
            let s = BindlessSlot(self.next_slot);
            self.next_slot += 1;
            return s;
        }
        // Evict the oldest unused slot (LRU) so the table never overflows.
        let oldest = self
            .cache
            .iter()
            .filter(|(_, r)| r.last_used_frame < current_frame)
            .min_by_key(|(_, r)| r.last_used_frame)
            .map(|(&k, &v)| (k, v.slot));
        if let Some((k, s)) = oldest {
            self.cache.remove(&k);
            return s;
        }
        FALLBACK_SLOT
    }

    pub fn active_count(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hides_texture_slot_until_copy_queue_fence_completes() {
        let mut mgr = BindlessManager::new(1024);
        let desc = HardwareDesc([0x11, 0x22, 0x33, 0x44]);

        // Upload scheduled for copy fence 10, currently at fence 7 -> must return FALLBACK_SLOT.
        assert_eq!(mgr.resolve(desc, 10, 7, 1), FALLBACK_SLOT);
        assert!(mgr.publish_ready(9).is_empty());

        // Copy fence 10 completes -> slot is published and returned on subsequent resolves.
        let published = mgr.publish_ready(10);
        assert_eq!(published.len(), 1);
        assert_ne!(published[0].1, FALLBACK_SLOT);
        assert_eq!(mgr.resolve(desc, 10, 10, 2), published[0].1);
    }
}
