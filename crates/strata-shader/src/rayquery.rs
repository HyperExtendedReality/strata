//! Hardware `SPV_KHR_ray_query` loop detection & lifting descriptor for RDNA2
//! inline ray-tracing traversal loops (`image_bvh64_intersect_ray`).
//!
//! Recognises console inline `RayQuery` traversal loops in decoded RDNA2
//! instruction streams and classifies them for lifting to Vulkan
//! `VK_KHR_ray_query` (`OpRayQueryInitializeKHR` / `OpRayQueryProceedKHR`).
//! This scan screens candidates. A replacement also needs proven ray inputs,
//! memory effects, and a value for every register live past the loop.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::rdna2::{DecodedInst, Inst};
use crate::ssa::{SsaModule, SsaOp, Terminator, ValueId};

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

/// Why a traversal loop was not lifted (or that it was).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RayLoopVerdict {
    Lifted,
    Refused(&'static str),
}

/// `STRATA_RAY_QUERY=1` turns on [`lift_ray_loops`] (off by default).
pub fn lifting_enabled() -> bool {
    std::env::var("STRATA_RAY_QUERY").is_ok_and(|v| v.trim() == "1")
}

/// Replaces an SSA traversal loop with [`SsaOp::RayQueryClosestHit`] only
/// when everything the replacement changes is proven:
///
/// 1. the loop holds exactly one box test (`image_bvh64_intersect_ray`);
/// 2. memory effects: nothing in it stores, does atomics, touches LDS
///    (another invocation could see it) or synchronizes;
/// 3. ray inputs: origin, `t_max` and inverse direction are loop
///    invariant, and the inverse direction is `rcp(d)` of an invariant `d`
///    (the query needs `d` itself);
/// 4. every value defined in the loop and used after it has a proven
///    value once the loop is gone.
///
/// For (4) the only value Strata can give is a loop-invariant one: the
/// hit distance a traversal computes comes from the title's own ALU leaf
/// test (Strata models box nodes only), and no such computation is
/// provably equal to a ray query's committed `T`, whose intersection
/// precision and watertightness are implementation-defined. A loop whose
/// live-outs are all invariant has no observable result to lift, and
/// removing it would only change termination. So every candidate today is
/// refused with the first obligation it fails, and the module is left
/// untouched; the verdicts say which.
pub fn lift_ray_loops(module: &SsaModule) -> Vec<(usize, RayLoopVerdict)> {
    let mut def_block: HashMap<ValueId, usize> = HashMap::new();
    let mut def_op: HashMap<ValueId, &SsaOp> = HashMap::new();
    for (bi, b) in module.blocks.iter().enumerate() {
        for i in b.insts.iter().filter(|i| !i.dead) {
            def_block.insert(i.id, bi);
            def_op.insert(i.id, &i.op);
        }
    }
    let mut out = Vec::new();
    for (header, block) in module.blocks.iter().enumerate() {
        let Terminator::Loop { body, merge, .. } = block.term else {
            continue;
        };
        let blocks = loop_blocks(module, header, body, merge);
        let insts = || {
            blocks
                .iter()
                .flat_map(|&b| &module.blocks[b].insts)
                .filter(|i| !i.dead)
        };
        let boxes: Vec<&SsaOp> = insts()
            .map(|i| &i.op)
            .filter(|op| matches!(op, SsaOp::Bvh64IntersectBox32 { .. }))
            .collect();
        if boxes.is_empty() {
            continue; // not a traversal loop
        }
        out.push((
            header,
            RayLoopVerdict::Refused(prove(module, &blocks, &boxes, &def_block, &def_op)),
        ));
    }
    out
}

/// The first obligation the loop fails (see [`lift_ray_loops`]).
fn prove(
    module: &SsaModule,
    blocks: &BTreeSet<usize>,
    boxes: &[&SsaOp],
    def_block: &HashMap<ValueId, usize>,
    def_op: &HashMap<ValueId, &SsaOp>,
) -> &'static str {
    let [
        SsaOp::Bvh64IntersectBox32 {
            t_max,
            ox,
            oy,
            oz,
            inv_dx,
            inv_dy,
            inv_dz,
            ..
        },
    ] = boxes
    else {
        return "several box tests in one loop";
    };
    let insts = || {
        blocks
            .iter()
            .flat_map(|&b| &module.blocks[b].insts)
            .filter(|i| !i.dead)
    };
    if insts()
        .any(|i| crate::opt::has_side_effect(&i.op) || matches!(i.op, SsaOp::LdsLoadU32 { .. }))
    {
        return "memory effects in the loop (stores, atomics, LDS or barriers)";
    }
    let invariant = |v: &ValueId| is_invariant(*v, blocks, def_block, def_op, 0);
    if ![t_max, ox, oy, oz, inv_dx, inv_dy, inv_dz]
        .into_iter()
        .all(invariant)
    {
        return "ray inputs change between iterations";
    }
    let direction =
        |v: &ValueId| match strip_bitcasts(*v, blocks, def_op).and_then(|v| def_op.get(&v)) {
            Some(SsaOp::RcpF32(d)) => invariant(d),
            _ => false,
        };
    if ![inv_dx, inv_dy, inv_dz].into_iter().all(direction) {
        return "the ray direction is not proven (inverse direction is not rcp of an invariant)";
    }
    // Values the loop defines that are used after it.
    let mut live_out = HashSet::new();
    for (bi, b) in module.blocks.iter().enumerate() {
        if blocks.contains(&bi) {
            continue;
        }
        let mut note = |v: ValueId| {
            if def_block.get(&v).is_some_and(|d| blocks.contains(d)) {
                live_out.insert(v);
            }
        };
        for i in b.insts.iter().filter(|i| !i.dead) {
            crate::opt::for_each_operand(&i.op, &mut note);
        }
        if let Some(c) = b.term.cond() {
            note(c);
        }
    }
    let proven = |v: &ValueId| match def_op.get(v) {
        // A header phi that only ever feeds itself back keeps its entry value.
        Some(SsaOp::Phi { incoming }) => incoming
            .iter()
            .all(|(pred, x)| x == v || !blocks.contains(pred)),
        _ => false,
    };
    if !live_out.iter().all(proven) {
        return "a value live past the loop has no proven equivalent (the hit distance a \
                traversal computes is not provably a ray query's committed T)";
    }
    "the loop has no observable result to replace"
}

/// Whether `v` has one value for the whole loop: defined before it, a
/// header phi that only feeds itself back, or a pure op of such values.
fn is_invariant(
    v: ValueId,
    blocks: &BTreeSet<usize>,
    def_block: &HashMap<ValueId, usize>,
    def_op: &HashMap<ValueId, &SsaOp>,
    depth: u32,
) -> bool {
    if def_block.get(&v).is_none_or(|b| !blocks.contains(b)) {
        return true;
    }
    if depth > 16 {
        return false;
    }
    match def_op.get(&v) {
        Some(SsaOp::Phi { incoming }) => incoming.iter().all(|(pred, x)| {
            if blocks.contains(pred) {
                *x == v
            } else {
                is_invariant(*x, blocks, def_block, def_op, depth + 1)
            }
        }),
        Some(op) if crate::opt::is_pure(op) => {
            let mut all = true;
            crate::opt::for_each_operand(op, |x| {
                all &= is_invariant(x, blocks, def_block, def_op, depth + 1);
            });
            all
        }
        _ => false,
    }
}

/// `v` with bitcasts removed and self-feeding loop phis replaced by the
/// value they enter with.
fn strip_bitcasts(
    mut v: ValueId,
    blocks: &BTreeSet<usize>,
    def_op: &HashMap<ValueId, &SsaOp>,
) -> Option<ValueId> {
    for _ in 0..16 {
        match def_op.get(&v)? {
            SsaOp::BitcastU32ToF32(x) | SsaOp::BitcastF32ToU32(x) => v = *x,
            SsaOp::Phi { incoming } => {
                let mut entry = incoming
                    .iter()
                    .filter(|(pred, x)| !(blocks.contains(pred) && *x == v));
                match (entry.next(), entry.next()) {
                    (Some((pred, x)), None) if !blocks.contains(pred) => v = *x,
                    _ => return Some(v),
                }
            }
            _ => return Some(v),
        }
    }
    Some(v)
}

/// Blocks of the loop at `header`: the header and everything reachable from
/// `body` without leaving through `merge` or re-entering the header.
fn loop_blocks(module: &SsaModule, header: usize, body: usize, merge: usize) -> BTreeSet<usize> {
    let mut seen = BTreeSet::from([header]);
    let mut stack = vec![body];
    while let Some(b) = stack.pop() {
        if b == merge || !seen.insert(b) {
            continue;
        }
        stack.extend(module.blocks[b].term.successors());
    }
    seen
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

    mod lifting {
        use super::super::*;
        use crate::rdna2::encode::*;

        fn v_rcp_f32(dst: u8, src: u8) -> u32 {
            (0b0111_111 << 25) | ((dst as u32) << 17) | (0x2a << 9) | (256 + src as u32)
        }

        /// A ray in v8..v19 (node, t_max, origin, dir, inv_dir), then a
        /// loop of box tests counted in s20; `inside` runs in the loop,
        /// `after` follows it.
        fn verdicts(rcp_dir: bool, inside: &[u32], after: &[u32]) -> Vec<RayLoopVerdict> {
            let mut w = vec![v_mov_b32_sgpr(8, 0), v_mov_b32_sgpr(9, 1)];
            for (r, f) in [(10, 100.0f32), (11, 0.0), (12, 0.0), (13, 0.0)] {
                w.extend(v_mov_b32_literal(r, f.to_bits()));
            }
            for (r, f) in [(14, 1.0f32), (15, 2.0), (16, 4.0)] {
                // From push constants s2..s4, so nothing folds the rcp.
                w.push(v_mov_b32_sgpr(r, r - 12));
                if rcp_dir {
                    w.push(v_rcp_f32(r + 3, r));
                } else {
                    w.extend(v_mov_b32_literal(r + 3, (1.0 / f).to_bits()));
                }
            }
            w.push(s_movk_i32(20, 0));
            let start = w.len();
            w.extend(image_bvh64_intersect_ray(4, 8, 4));
            w.extend_from_slice(inside);
            w.push(s_addk_i32(20, 1));
            w.push(s_cmp_lt_u32(20, 136)); // s20 < 8
            let rel = start as i32 - (w.len() as i32 + 1);
            w.push(s_cbranch_scc1(rel as i16));
            w.extend_from_slice(after);
            w.push(s_endpgm());
            let program = crate::rdna2::decode_program(&w).unwrap();
            let wave = crate::wave::analyze_wave_liveness(&program);
            let mut m =
                crate::ssa::build_ssa_with(&program, &wave, [64, 1, 1], &Default::default())
                    .unwrap();
            crate::opt::optimize(&mut m);
            lift_ray_loops(&m).into_iter().map(|(_, v)| v).collect()
        }

        fn store_v4() -> Vec<u32> {
            global_store_dword(2, 4, 0, 0).to_vec()
        }

        fn refused(v: &[RayLoopVerdict]) -> &'static str {
            match v {
                [RayLoopVerdict::Refused(why)] => why,
                other => panic!("expected one refusal, got {other:?}"),
            }
        }

        #[test]
        fn a_traversal_result_used_after_the_loop_is_not_lifted() {
            let v = verdicts(true, &[], &store_v4());
            assert!(
                refused(&v).starts_with("a value live past the loop"),
                "{v:?}"
            );
        }

        #[test]
        fn stores_in_the_loop_are_memory_effects() {
            let v = verdicts(true, &store_v4(), &[]);
            assert!(refused(&v).starts_with("memory effects"), "{v:?}");
        }

        #[test]
        fn a_shrinking_t_max_is_a_varying_ray_input() {
            let v = verdicts(true, &[v_add_f32(10, 10, 10)], &store_v4());
            assert!(refused(&v).starts_with("ray inputs change"), "{v:?}");
        }

        #[test]
        fn an_inverse_direction_without_rcp_is_not_a_proven_direction() {
            let v = verdicts(false, &[], &store_v4());
            assert!(
                refused(&v).starts_with("the ray direction is not proven"),
                "{v:?}"
            );
        }

        #[test]
        fn a_loop_with_no_observable_result_is_left_alone() {
            let v = verdicts(true, &[], &[]);
            // Unused box results may be dead already; if the loop survives,
            // it is refused, never lifted.
            assert!(
                v.is_empty() || refused(&v) == "the loop has no observable result to replace",
                "{v:?}"
            );
        }
    }
}
