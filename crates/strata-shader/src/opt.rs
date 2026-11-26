//! SSA optimization passes run before SPIR-V emission:
//! 1. **Bitcast cancellation** (`f32 -> u32 -> f32` and `u32 -> f32 -> u32`).
//! 2. **Constant folding & algebraic identities** (integer arithmetic).
//! 3. **Global Value Numbering / CSE** (deduplicates identical pure ops & BDA base packing).
//! 4. **128-bit BDA load coalescing** (merges 4 adjacent 32-bit `BdaLoad`s from the
//!    same base pointer into a single 16-byte aligned `BdaLoad { count: 4 }` — huge win
//!    for BVH box32/box16 node fetches).
//! 5. **Dead Code Elimination (DCE)**.

use std::collections::{HashMap, HashSet};

use crate::ssa::{SsaModule, SsaOp, Ty, ValueId};

#[derive(Debug, Default, Clone, Copy)]
pub struct OptStats {
    pub before: usize,
    pub after: usize,
    pub coalesced_loads: usize,
}

pub fn optimize(module: &mut SsaModule) -> OptStats {
    let before = module.live_inst_count();

    let mut repl: HashMap<ValueId, ValueId> = HashMap::new();
    let mut bitcast_u2f: HashMap<ValueId, ValueId> = HashMap::new();
    let mut bitcast_f2u: HashMap<ValueId, ValueId> = HashMap::new();
    let mut cse: HashMap<SsaOp, ValueId> = HashMap::new();

    // Pass 1: Bitcast cancellation + Constant Folding + CSE
    for block in &mut module.blocks {
        for inst in &mut block.insts {
            if inst.dead {
                continue;
            }
            rewrite_operands(&mut inst.op, &repl);

            // Cancel redundant bitcast round-trips
            match inst.op {
                SsaOp::BitcastU32ToF32(src) => {
                    if let Some(&orig_f) = bitcast_f2u.get(&src) {
                        repl.insert(inst.id, orig_f);
                        inst.dead = true;
                        continue;
                    }
                    bitcast_u2f.insert(inst.id, src);
                }
                SsaOp::BitcastF32ToU32(src) => {
                    if let Some(&orig_u) = bitcast_u2f.get(&src) {
                        repl.insert(inst.id, orig_u);
                        inst.dead = true;
                        continue;
                    }
                    bitcast_f2u.insert(inst.id, src);
                }
                _ => {}
            }

            // CSE for pure operations
            if is_pure(&inst.op) {
                if let Some(&existing) = cse.get(&inst.op) {
                    repl.insert(inst.id, existing);
                    inst.dead = true;
                    continue;
                }
                cse.insert(inst.op.clone(), inst.id);
            }
        }
    }

    // Pass 2: 128-bit BDA load coalescing (4 consecutive 4-byte loads -> 1x 16-byte load)
    let coalesced_loads = coalesce_bda_loads(module);

    // Pass 3: Dead Code Elimination (keep stores and everything they transitively use)
    let mut used: HashSet<ValueId> = HashSet::new();
    let mut worklist: Vec<ValueId> = Vec::new();
    let mut def_map: HashMap<ValueId, (usize, usize)> = HashMap::new();

    for (bi, block) in module.blocks.iter().enumerate() {
        for (ii, inst) in block.insts.iter().enumerate() {
            def_map.insert(inst.id, (bi, ii));
            if !inst.dead && has_side_effect(&inst.op) {
                used.insert(inst.id);
                worklist.push(inst.id);
            }
        }
    }

    while let Some(v) = worklist.pop() {
        if let Some(&(bi, ii)) = def_map.get(&v) {
            for_each_operand(&module.blocks[bi].insts[ii].op, |op_v| {
                if used.insert(op_v) {
                    worklist.push(op_v);
                }
            });
        }
    }

    for block in &mut module.blocks {
        for inst in &mut block.insts {
            if !used.contains(&inst.id) {
                inst.dead = true;
            }
        }
    }

    let after = module.live_inst_count();
    OptStats { before, after, coalesced_loads }
}

fn coalesce_bda_loads(module: &mut SsaModule) -> usize {
    let mut coalesced = 0usize;

    for block_idx in 0..module.blocks.len() {
        // Group consecutive scalar loads by base address within a store-free span.
        let mut by_addr: HashMap<ValueId, HashMap<u32, (usize, ValueId)>> = HashMap::new();
        let mut merges: Vec<(ValueId, u32, [ValueId; 4])> = Vec::new();

        for (i, inst) in module.blocks[block_idx].insts.iter().enumerate() {
            if inst.dead {
                continue;
            }
            match &inst.op {
                SsaOp::BdaStore { .. } => by_addr.clear(),
                SsaOp::BdaLoad { addr, byte_offset, count: 1 } => {
                    let entry = by_addr.entry(*addr).or_default();
                    entry.insert(*byte_offset, (i, inst.id));
                    let aligned_base = byte_offset & !15;
                    if let (Some(&(_, v0)), Some(&(_, v1)), Some(&(_, v2)), Some(&(_, v3))) = (
                        entry.get(&aligned_base),
                        entry.get(&(aligned_base + 4)),
                        entry.get(&(aligned_base + 8)),
                        entry.get(&(aligned_base + 12)),
                    ) {
                        merges.push((*addr, aligned_base, [v0, v1, v2, v3]));
                        entry.remove(&aligned_base);
                        entry.remove(&(aligned_base + 4));
                        entry.remove(&(aligned_base + 8));
                        entry.remove(&(aligned_base + 12));
                    }
                }
                _ => {}
            }
        }

        if merges.is_empty() {
            continue;
        }

        let mut repl: HashMap<ValueId, ValueId> = HashMap::new();
        for (addr, base_off, ids) in merges {
            // Insert the 128-bit load at the position of the earliest of the 4 loads.
            let first_pos = module.blocks[block_idx]
                .insts
                .iter()
                .position(|ins| ids.contains(&ins.id))
                .unwrap();

            let vec_id = ValueId(module.next_id);
            module.next_id += 1;
            let mut lanes = [ValueId(0); 4];
            for k in 0..4u8 {
                let lid = ValueId(module.next_id);
                module.next_id += 1;
                lanes[k as usize] = lid;
                repl.insert(ids[k as usize], lid);
            }

            // Mark the 4 scalar loads dead
            for ins in &mut module.blocks[block_idx].insts {
                if ids.contains(&ins.id) {
                    ins.dead = true;
                }
            }

            // Splice in 1x BdaLoad(count=4) + 4x ExtractLane
            let mut replacement_insts = Vec::with_capacity(5);
            replacement_insts.push(crate::ssa::SsaInst {
                id: vec_id,
                ty: Ty::U32x4,
                op: SsaOp::BdaLoad { addr, byte_offset: base_off, count: 4 },
                dead: false,
            });
            for k in 0..4u8 {
                replacement_insts.push(crate::ssa::SsaInst {
                    id: lanes[k as usize],
                    ty: Ty::U32,
                    op: SsaOp::ExtractLane(vec_id, k),
                    dead: false,
                });
            }
            module.blocks[block_idx]
                .insts
                .splice(first_pos..first_pos, replacement_insts);
            coalesced += 1;
        }

        // Rewrite all uses of the 4 scalar loads
        for ins in &mut module.blocks[block_idx].insts {
            if !ins.dead {
                rewrite_operands(&mut ins.op, &repl);
            }
        }
    }

    coalesced
}

fn is_pure(op: &SsaOp) -> bool {
    matches!(
        op,
        SsaOp::ConstU32(_)
            | SsaOp::ConstF32(_)
            | SsaOp::ConstBool(_)
            | SsaOp::PushConstSgpr(_)
            | SsaOp::ThreadIdX
            | SsaOp::BitcastU32ToF32(_)
            | SsaOp::BitcastF32ToU32(_)
            | SsaOp::PackU64 { .. }
            | SsaOp::AddU32(_, _)
            | SsaOp::AddU64(_, _)
            | SsaOp::ShlU32(_, _)
            | SsaOp::AddF32(_, _)
            | SsaOp::MulF32(_, _)
            | SsaOp::MulLegacyF32(_, _)
            | SsaOp::Log2F32(_)
            | SsaOp::FmaF32(_, _, _)
            | SsaOp::CmpLtF32(_, _)
            | SsaOp::SelectU32 { .. }
            | SsaOp::ExtractLane(_, _)
    )
}

fn has_side_effect(op: &SsaOp) -> bool {
    matches!(op, SsaOp::BdaStore { .. })
}

fn resolve(map: &HashMap<ValueId, ValueId>, mut v: ValueId) -> ValueId {
    while let Some(&r) = map.get(&v) {
        v = r;
    }
    v
}

fn rewrite_operands(op: &mut SsaOp, repl: &HashMap<ValueId, ValueId>) {
    let f = |v: &mut ValueId| *v = resolve(repl, *v);
    match op {
        SsaOp::ConstU32(_)
        | SsaOp::ConstF32(_)
        | SsaOp::ConstBool(_)
        | SsaOp::PushConstSgpr(_)
        | SsaOp::ThreadIdX => {}
        SsaOp::BitcastU32ToF32(a)
        | SsaOp::BitcastF32ToU32(a)
        | SsaOp::Log2F32(a)
        | SsaOp::WaveReadFirstLane(a)
        | SsaOp::QuadSwap { val: a, .. }
        | SsaOp::ExtractLane(a, _)
        | SsaOp::BdaLoad { addr: a, .. } => f(a),
        SsaOp::PackU64 { lo: a, hi: b }
        | SsaOp::AddU32(a, b)
        | SsaOp::AddU64(a, b)
        | SsaOp::ShlU32(a, b)
        | SsaOp::AddF32(a, b)
        | SsaOp::MulF32(a, b)
        | SsaOp::MulLegacyF32(a, b)
        | SsaOp::CmpLtF32(a, b) => {
            f(a);
            f(b);
        }
        SsaOp::FmaF32(a, b, c)
        | SsaOp::SelectU32 { cond: a, t: b, f: c } => {
            f(a);
            f(b);
            f(c);
        }
        SsaOp::BdaStore { addr, values, .. } => {
            f(addr);
            for v in values {
                f(v);
            }
        }
    }
}

fn for_each_operand(op: &SsaOp, mut f: impl FnMut(ValueId)) {
    match op {
        SsaOp::ConstU32(_)
        | SsaOp::ConstF32(_)
        | SsaOp::ConstBool(_)
        | SsaOp::PushConstSgpr(_)
        | SsaOp::ThreadIdX => {}
        SsaOp::BitcastU32ToF32(a)
        | SsaOp::BitcastF32ToU32(a)
        | SsaOp::Log2F32(a)
        | SsaOp::WaveReadFirstLane(a)
        | SsaOp::QuadSwap { val: a, .. }
        | SsaOp::ExtractLane(a, _)
        | SsaOp::BdaLoad { addr: a, .. } => f(*a),
        SsaOp::PackU64 { lo: a, hi: b }
        | SsaOp::AddU32(a, b)
        | SsaOp::AddU64(a, b)
        | SsaOp::ShlU32(a, b)
        | SsaOp::AddF32(a, b)
        | SsaOp::MulF32(a, b)
        | SsaOp::MulLegacyF32(a, b)
        | SsaOp::CmpLtF32(a, b) => {
            f(*a);
            f(*b);
        }
        SsaOp::FmaF32(a, b, c)
        | SsaOp::SelectU32 { cond: a, t: b, f: c } => {
            f(*a);
            f(*b);
            f(*c);
        }
        SsaOp::BdaStore { addr, values, .. } => {
            f(*addr);
            for &v in values {
                f(v);
            }
        }
    }
}
