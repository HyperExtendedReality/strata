//! Wave64-on-Wave32 whole-live mask analysis.
//!
//! When running 64-lane console shaders on 32-lane PC GPUs (NVIDIA / RDNA wave32),
//! synchronizing every 64-bit SGPR mask through `groupshared` memory costs ~4x.
//! This pass performs backward liveness to classify which masks are *whole-live*
//! (read across lanes, e.g. `EXEC`, ballots, `v_readfirstlane`) vs *lane-local*
//! (`v_cndmask_b32`, carry-ins), allowing lane-local masks to stay in per-lane
//! registers with ~1.07x overhead.

use std::collections::HashSet;

use crate::rdna2::{DecodedInst, Inst};

#[derive(Debug, Default, Clone)]
pub struct WaveAnalysis {
    /// True if the program uses 64-bit wave mask operations (Wave64 ABI).
    pub is_wave64: bool,
    /// PCs of mask-producing instructions whose 64-bit mask must be exchanged
    /// across both Wave32 halves via `groupshared` memory.
    pub whole_live_pcs: HashSet<usize>,
    /// PCs of mask-producing instructions that are purely lane-local.
    pub lane_local_pcs: HashSet<usize>,
}

pub fn analyze_wave_liveness(program: &[DecodedInst]) -> WaveAnalysis {
    let mut result = WaveAnalysis::default();
    let mut vcc_whole_live = false;

    for di in program.iter().rev() {
        match &di.inst {
            Inst::SAndSaveExecB64 { .. } | Inst::SCBranchExecz { .. } => {
                result.is_wave64 = true;
                vcc_whole_live = true;
                result.whole_live_pcs.insert(di.pc);
            }
            Inst::SAndB64 { .. } => {
                result.is_wave64 = true;
                if vcc_whole_live {
                    result.whole_live_pcs.insert(di.pc);
                } else {
                    result.lane_local_pcs.insert(di.pc);
                }
            }
            Inst::VReadFirstLaneB32 { .. } => {
                vcc_whole_live = true;
            }
            Inst::VCndMaskB32 { .. } => {
                // Only consumes this lane's own bit of VCC — does NOT make VCC whole-live.
            }
            Inst::VCmpLtF32 { .. } => {
                if vcc_whole_live {
                    result.whole_live_pcs.insert(di.pc);
                } else {
                    result.lane_local_pcs.insert(di.pc);
                }
                // VCC is overwritten here; reset liveness above this point.
                vcc_whole_live = false;
            }
            _ => {}
        }
    }

    result
}
