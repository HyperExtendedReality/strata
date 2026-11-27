//! Register spilling (`SCRATCH_LOAD_*` / `SCRATCH_STORE_*`) and per-invocation
//! scratch-frame allocation for high-VGPR RDNA2 shaders.

use std::collections::HashMap;

/// Backing strategy for a spilled shader scratch frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpillBacking {
    /// Per-invocation SPIR-V `Function` storage-class `uint[N]` array (zero VRAM traffic).
    FunctionArray { dwords: u32 },
    /// 64-bit BDA scratch memory window when spill frame exceeds register/private limits.
    BdaScratch { bytes_per_lane: u32 },
}

/// Tracks VGPR/SGPR spill slots and scratch offsets seen in a shader.
#[derive(Debug, Clone, Default)]
pub struct SpillFrame {
    slot_by_offset: HashMap<i32, u32>,
    next_slot: u32,
    pub max_byte_offset: u32,
}

impl SpillFrame {
    pub fn record_access(&mut self, byte_offset: i32, width_dwords: u32) -> u32 {
        let aligned = byte_offset.max(0) as u32;
        let end = aligned.saturating_add(width_dwords * 4);
        if end > self.max_byte_offset {
            self.max_byte_offset = end;
        }
        *self.slot_by_offset.entry(byte_offset).or_insert_with(|| {
            let s = self.next_slot;
            self.next_slot += width_dwords;
            s
        })
    }

    pub fn total_dwords(&self) -> u32 {
        self.max_byte_offset.div_ceil(4).max(self.next_slot)
    }

    /// Choose `FunctionArray` for compact frames (<= 256 dwords = 1 KiB/lane),
    /// otherwise `BdaScratch`.
    pub fn choose_backing(&self) -> SpillBacking {
        let dwords = self.total_dwords();
        if dwords <= 256 {
            SpillBacking::FunctionArray {
                dwords: dwords.max(1),
            }
        } else {
            SpillBacking::BdaScratch {
                bytes_per_lane: dwords * 4,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spill_frame_selects_function_array_or_bda() {
        let mut frame = SpillFrame::default();
        let s0 = frame.record_access(0, 1);
        let s1 = frame.record_access(16, 4);
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(
            frame.choose_backing(),
            SpillBacking::FunctionArray { dwords: 8 }
        );

        frame.record_access(2048, 4);
        assert!(matches!(
            frame.choose_backing(),
            SpillBacking::BdaScratch { .. }
        ));
    }
}
