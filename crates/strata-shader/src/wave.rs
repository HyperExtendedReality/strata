//! Wave64-on-Wave32 whole-live mask analysis.
//!
//! When running 64-lane console shaders on 32-lane or 16-lane PC GPUs,
//! exchanging every 64-bit SGPR mask through `groupshared` memory is expensive.
//! This pass performs backward liveness to classify which mask producers are
//! *whole-live* (their 64-bit bitmask is observed across lanes, e.g. stored as
//! an SGPR/VGPR value, read via `v_readfirstlane`, or tested as a full-wave
//! mask) vs *lane-local* (only each lane's own bit is needed, e.g.
//! `v_cndmask_b32` or per-lane `EXEC` predication).
//!
//! The SSA builder consumes `WaveAnalysis` directly:
//! - `whole_live_pcs` emit `SsaOp::WaveBallot64` (subgroup ballot + workgroup
//!   atomic OR merge across all 64 lanes).
//! - `lane_local_pcs` remain per-lane `Ty::Bool` values in registers.

use std::collections::HashSet;

use crate::rdna2::{DecodedInst, Inst, Operand};

#[derive(Debug, Default, Clone)]
pub struct WaveAnalysis {
    /// True if the program uses 64-bit wave mask operations (Wave64 ABI).
    pub is_wave64: bool,
    /// PCs of mask-producing instructions whose 64-bit mask must be exchanged
    /// across all 64 lanes via `SsaOp::WaveBallot64`.
    pub whole_live_pcs: HashSet<usize>,
    /// PCs of mask-producing instructions that stay purely lane-local.
    pub lane_local_pcs: HashSet<usize>,
}

pub fn analyze_wave_liveness(program: &[DecodedInst]) -> WaveAnalysis {
    let mut result = WaveAnalysis::default();
    let mut vcc_whole_live = false;
    let mut exec_whole_live = false;

    for di in program.iter().rev() {
        match &di.inst {
            // Moving or storing VCC/EXEC as raw 32-bit integers requires the full 64-bit mask.
            Inst::VMovB32 { src0, .. } | Inst::SMovB32 { src0, .. } => {
                if matches!(src0, Operand::VccLo | Operand::VccHi) {
                    result.is_wave64 = true;
                    vcc_whole_live = true;
                }
                if matches!(src0, Operand::ExecLo | Operand::ExecHi) {
                    result.is_wave64 = true;
                    exec_whole_live = true;
                }
            }
            Inst::VReadFirstLaneB32 { .. } => {
                vcc_whole_live = true;
            }
            Inst::SCBranchExecz { .. } | Inst::SCBranchExecnz { .. } => {
                result.is_wave64 = true;
            }
            Inst::SCBranchVccz { .. } | Inst::SCBranchVccnz { .. } => {
                result.is_wave64 = true;
                vcc_whole_live = true;
                result.whole_live_pcs.insert(di.pc);
            }
            Inst::SAndSaveExecB64 { src0, .. } | Inst::SOrSaveExecB64 { src0, .. } => {
                result.is_wave64 = true;
                if exec_whole_live {
                    result.whole_live_pcs.insert(di.pc);
                } else {
                    result.lane_local_pcs.insert(di.pc);
                }
                if matches!(src0, Operand::VccLo | Operand::VccHi) && exec_whole_live {
                    vcc_whole_live = true;
                }
            }
            Inst::SAndB64 { dst, src0, src1 }
            | Inst::SOrB64 { dst, src0, src1 }
            | Inst::SXorB64 { dst, src0, src1 }
            | Inst::SAndN2B64 { dst, src0, src1 }
            | Inst::SOrN2B64 { dst, src0, src1 } => {
                result.is_wave64 = true;
                let writes_exec = *dst == 126;
                let is_whole = if writes_exec {
                    exec_whole_live
                } else {
                    vcc_whole_live
                };
                if is_whole {
                    result.whole_live_pcs.insert(di.pc);
                    if matches!(src0, Operand::VccLo | Operand::VccHi)
                        || matches!(src1, Operand::VccLo | Operand::VccHi)
                    {
                        vcc_whole_live = true;
                    }
                } else {
                    result.lane_local_pcs.insert(di.pc);
                }
            }
            Inst::SMovB64 { dst, src0 } | Inst::SNotB64 { dst, src0 } => {
                result.is_wave64 = true;
                if *dst == 126 && exec_whole_live {
                    result.whole_live_pcs.insert(di.pc);
                    if matches!(src0, Operand::VccLo | Operand::VccHi) {
                        vcc_whole_live = true;
                    }
                } else {
                    result.lane_local_pcs.insert(di.pc);
                }
            }
            Inst::VCndMaskB32 { .. } => {
                // Only consumes this lane's own bit of VCC — stays lane-local.
            }
            Inst::VCmp { .. } => {
                if vcc_whole_live {
                    result.whole_live_pcs.insert(di.pc);
                } else {
                    result.lane_local_pcs.insert(di.pc);
                }
                vcc_whole_live = false;
            }
            _ => {}
        }
    }

    result
}
