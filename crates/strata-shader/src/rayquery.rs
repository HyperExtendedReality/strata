//! Hardware `SPV_KHR_ray_query` loop detection & lifting descriptor for RDNA2
//! inline ray-tracing traversal loops (`image_bvh64_intersect_ray`).
//!
//! Recognises console inline `RayQuery` traversal loops in decoded RDNA2
//! instruction streams and classifies them for lifting to Vulkan
//! `VK_KHR_ray_query` (`OpRayQueryInitializeKHR` / `OpRayQueryProceedKHR`).
//! This scan screens candidates. A replacement also needs proven ray inputs,
//! memory effects, and a value for every register live past the loop.

use crate::rdna2::{DecodedInst, Inst};

/// `SPV_KHR_ray_query` SPIR-V capability and opcode constants.
pub const CAP_RAY_QUERY_KHR: u32 = 4472;
pub const EXT_RAY_QUERY_KHR: &str = "SPV_KHR_ray_query";
pub const OP_TYPE_RAY_QUERY_KHR: u16 = 4472;
pub const OP_RAY_QUERY_INITIALIZE_KHR: u16 = 4473;
pub const OP_RAY_QUERY_TERMINATE_KHR: u16 = 4474;
pub const OP_RAY_QUERY_PROCEED_KHR: u16 = 4477;
pub const OP_RAY_QUERY_GET_INTERSECTION_TYPE_KHR: u16 = 4479;
pub const OP_RAY_QUERY_GET_INTERSECTION_T_KHR: u16 = 6018;

/// Recognized RDNA2 BVH traversal loop candidate for hardware `RayQuery` lifting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RayLoopCandidate {
    /// Dword PC of the first instruction in the traversal loop.
    pub loop_start_pc: usize,
    /// Dword PC of the back-edge branch instruction.
    pub back_edge_pc: usize,
    /// Instruction index of the `image_bvh64_intersect_ray` inside the loop.
    pub bvh_inst_index: usize,
    /// Destination VGPR base of the BVH intersection result.
    pub result_vgpr: u8,
    /// Whether the loop passed the initial store and atomic check.
    pub liftable: bool,
}

/// Scans a decoded RDNA2 program for BVH traversal loops containing
/// `image_bvh64_intersect_ray` enclosed by a backward branch.
pub fn find_ray_loops(program: &[DecodedInst]) -> Vec<RayLoopCandidate> {
    let mut out = Vec::new();
    for (idx, di) in program.iter().enumerate() {
        let Inst::ImageBvh64IntersectRay { dst, .. } = di.inst else {
            continue;
        };
        for (b_idx, b_di) in program.iter().enumerate().skip(idx + 1) {
            let target = match b_di.inst {
                Inst::SBranch { target_dw }
                | Inst::SCBranchScc { target_dw, .. }
                | Inst::SCBranchVccz { target_dw }
                | Inst::SCBranchVccnz { target_dw }
                | Inst::SCBranchExecz { target_dw }
                | Inst::SCBranchExecnz { target_dw } => {
                    if target_dw <= di.pc {
                        Some(target_dw)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(loop_start) = target {
                let has_side_effects = program.iter().take(b_idx).any(|inner| {
                    inner.pc >= loop_start
                        && matches!(
                            inner.inst,
                            Inst::GlobalStore { .. }
                                | Inst::GlobalAtomic { .. }
                                | Inst::BufferAtomic { .. }
                        )
                });
                out.push(RayLoopCandidate {
                    loop_start_pc: loop_start,
                    back_edge_pc: b_di.pc,
                    bvh_inst_index: idx,
                    result_vgpr: dst,
                    liftable: !has_side_effects,
                });
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_liftable_and_side_effecting_bvh_loops() {
        let clean = vec![
            DecodedInst {
                pc: 0,
                len_dw: 2,
                inst: Inst::ImageBvh64IntersectRay {
                    dst: 4,
                    addr_vgpr: 8,
                    rsrc_sgpr: 0,
                },
            },
            DecodedInst {
                pc: 2,
                len_dw: 1,
                inst: Inst::SCBranchExecnz { target_dw: 0 },
            },
        ];
        let loops = find_ray_loops(&clean);
        assert_eq!(loops.len(), 1);
        assert!(loops[0].liftable);
        assert_eq!(loops[0].result_vgpr, 4);

        let dirty = vec![
            clean[0].clone(),
            DecodedInst {
                pc: 2,
                len_dw: 2,
                inst: Inst::GlobalStore {
                    vaddr: 0,
                    vdata: 4,
                    count: 1,
                    saddr: None,
                    offset: 0,
                },
            },
            DecodedInst {
                pc: 4,
                len_dw: 1,
                inst: Inst::SCBranchExecnz { target_dw: 0 },
            },
        ];
        let dirty_loops = find_ray_loops(&dirty);
        assert_eq!(dirty_loops.len(), 1);
        assert!(!dirty_loops[0].liftable);
    }
}
