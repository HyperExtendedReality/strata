//! Wave64 lane permutes (`v_permlane16`, `v_permlanex16`, `v_readlane`,
//! `v_writelane`, `v_readfirstlane`, `ds_swizzle`, `ds_bpermute`, `ds_permute`)
//! and Waterfall Loop synthesis for non-uniform descriptor / SGPR indexing.

/// Cross-lane wave64 permute operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanePermuteOp {
    ReadFirstLane,
    ReadLane { lane: u32 },
    WriteLane { lane: u32 },
    PermLane16 { fi: bool, bound_ctrl: bool },
    PermLaneX16 { fi: bool, bound_ctrl: bool },
    DsSwizzle { offset16: u16 },
    DsBpermute,
    DsPermute,
}

/// Evaluate a Wave64 `v_permlane16_b32` / `v_permlanex16_b32` across 64 lanes.
/// `sel_lo` selects lanes 0..7 (4 bits each), `sel_hi` selects lanes 8..15.
pub fn eval_permlane16(
    src: &[u32; 64],
    sel_lo: u32,
    sel_hi: u32,
    cross_half: bool,
    exec_mask: u64,
) -> [u32; 64] {
    let mut dst = *src;
    for lane in 0..64usize {
        if ((exec_mask >> lane) & 1) == 0 {
            continue;
        }
        let row = lane / 16;
        let in_row = lane % 16;
        let sel_word = if in_row < 8 { sel_lo } else { sel_hi };
        let pick = ((sel_word >> ((in_row & 7) * 4)) & 0xF) as usize;
        let src_row = if cross_half { row ^ 1 } else { row };
        let src_lane = src_row * 16 + pick;
        dst[lane] = src[src_lane];
    }
    dst
}

/// Evaluate `ds_swizzle_b32` in quad-permute (`bit 15 == 1`) or FFT/butterfly mode (`bit 15 == 0`).
pub fn eval_ds_swizzle(src: &[u32; 64], offset16: u16) -> [u32; 64] {
    let mut dst = [0u32; 64];
    let quad_mode = (offset16 & 0x8000) != 0;
    for lane in 0..64usize {
        let src_lane = if quad_mode {
            let quad_base = lane & !3;
            let shift = (lane & 3) * 2;
            quad_base | (((offset16 as usize) >> shift) & 3)
        } else {
            let and_mask = (offset16 & 0x1F) as usize;
            let or_mask = ((offset16 >> 5) & 0x1F) as usize;
            let xor_mask = ((offset16 >> 10) & 0x1F) as usize;
            let half = lane & 32;
            half | ((((lane & 31) & and_mask) | or_mask) ^ xor_mask)
        };
        dst[lane] = src[src_lane];
    }
    dst
}

/// Waterfall loop step: given an active wave mask and per-lane non-uniform values
/// (such as divergent descriptor indices), partitions the wave into uniform
/// iterations where every lane in a group shares the same scalar value.
pub fn run_waterfall<F>(active_mask: u64, lane_values: &[u32; 64], mut body: F) -> usize
where
    F: FnMut(u32, u64),
{
    let mut remaining = active_mask;
    let mut iterations = 0usize;
    while remaining != 0 {
        let first_lane = remaining.trailing_zeros() as usize;
        let uniform_val = lane_values[first_lane];
        let mut group_mask = 0u64;
        for lane in 0..64 {
            if ((remaining >> lane) & 1) != 0 && lane_values[lane] == uniform_val {
                group_mask |= 1u64 << lane;
            }
        }
        body(uniform_val, group_mask);
        remaining &= !group_mask;
        iterations += 1;
    }
    iterations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permlane16_and_ds_swizzle_permute_wave64() {
        let mut lanes = [0u32; 64];
        for i in 0..64 {
            lanes[i] = i as u32;
        }
        // Reverse lanes 0..7 within each 16-lane row, cross-half (row 0 <-> row 1)
        let sel_lo = 0x0123_4567u32;
        let sel_hi = 0x89AB_CDEFu32;
        let out = eval_permlane16(&lanes, sel_lo, sel_hi, true, u64::MAX);
        assert_eq!(out[0], 16 + 7);
        assert_eq!(out[16], 7);

        // Quad-swap pairs: offset16 = 0x8000 | (2<<6 | 3<<4 | 0<<2 | 1) = 0x80B1
        let sw = eval_ds_swizzle(&lanes, 0x80B1);
        assert_eq!(&sw[0..4], &[1, 0, 3, 2]);
    }

    #[test]
    fn waterfall_loop_groups_divergent_indices() {
        let mut vals = [0u32; 64];
        for i in 0..64 {
            vals[i] = (i % 4) as u32;
        }
        let mut seen = Vec::new();
        let iters = run_waterfall(u64::MAX, &vals, |val, mask| {
            assert_eq!(mask.count_ones(), 16);
            seen.push(val);
        });
        assert_eq!(iters, 4);
        assert_eq!(seen, vec![0, 1, 2, 3]);
    }
}
