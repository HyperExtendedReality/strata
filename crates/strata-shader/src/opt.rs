//! SSA optimization passes run before SPIR-V emission:
//! 1. **Bitcast cancellation** (`f32 -> u32 -> f32` and `u32 -> f32 -> u32`).
//! 2. **Local Value Numbering / CSE** (per basic block, preserving CFG dominance).
//! 3. **128-bit BDA load coalescing** (merges 4 adjacent 32-bit `BdaLoad`s from the
//!    same base pointer into a single 16-byte aligned `BdaLoad { count: 4 }`).
//! 4. **Dead Code Elimination (DCE)** across all blocks, terminators, and `Phi` nodes.

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

    // Pass 1: Per-block bitcast cancellation + local CSE
    for block in &mut module.blocks {
        let mut bitcast_u2f: HashMap<ValueId, ValueId> = HashMap::new();
        let mut bitcast_f2u: HashMap<ValueId, ValueId> = HashMap::new();
        let mut cse: HashMap<SsaOp, ValueId> = HashMap::new();

        for inst in &mut block.insts {
            if inst.dead {
                continue;
            }
            rewrite_operands(&mut inst.op, &repl);

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

            if is_pure(&inst.op) {
                if let Some(&existing) = cse.get(&inst.op) {
                    repl.insert(inst.id, existing);
                    inst.dead = true;
                    continue;
                }
                cse.insert(inst.op.clone(), inst.id);
            }
        }

        if let Some(cond) = block.term.cond_mut() {
            *cond = resolve(&repl, *cond);
        }
    }

    // A loop header's phi reads values its body defines later; rewrite every
    // operand once more now that all replacements are known.
    for block in &mut module.blocks {
        for inst in &mut block.insts {
            if !inst.dead {
                rewrite_operands(&mut inst.op, &repl);
            }
        }
        if let Some(cond) = block.term.cond_mut() {
            *cond = resolve(&repl, *cond);
        }
    }

    // Pass 2: 128-bit BDA load coalescing (4 consecutive 4-byte loads -> 1x 16-byte load)
    let coalesced_loads = coalesce_bda_loads(module);

    // Pass 3: Dead Code Elimination across all blocks, terminators, and Phis
    let mut used: HashSet<ValueId> = HashSet::new();
    let mut worklist: Vec<ValueId> = Vec::new();
    let mut def_map: HashMap<ValueId, (usize, usize)> = HashMap::new();

    for (bi, block) in module.blocks.iter().enumerate() {
        for (ii, inst) in block.insts.iter().enumerate() {
            def_map.insert(inst.id, (bi, ii));
            if !inst.dead && has_side_effect(&inst.op) && used.insert(inst.id) {
                worklist.push(inst.id);
            }
        }
        if let Some(cond) = block.term.cond()
            && used.insert(cond)
        {
            worklist.push(cond);
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
    OptStats {
        before,
        after,
        coalesced_loads,
    }
}

fn coalesce_bda_loads(module: &mut SsaModule) -> usize {
    let mut coalesced = 0usize;

    for block_idx in 0..module.blocks.len() {
        let mut by_addr: HashMap<ValueId, HashMap<u32, (usize, ValueId)>> = HashMap::new();
        let mut merges: Vec<(ValueId, u32, [ValueId; 4])> = Vec::new();

        for (i, inst) in module.blocks[block_idx].insts.iter().enumerate() {
            if inst.dead {
                continue;
            }
            match &inst.op {
                SsaOp::BdaStore { .. } | SsaOp::WorkgroupBarrier => by_addr.clear(),
                SsaOp::BdaLoad {
                    addr,
                    byte_offset,
                    count: 1,
                } => {
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

            for ins in &mut module.blocks[block_idx].insts {
                if ids.contains(&ins.id) {
                    ins.dead = true;
                }
            }

            let mut replacement_insts = Vec::with_capacity(5);
            replacement_insts.push(crate::ssa::SsaInst {
                id: vec_id,
                ty: Ty::U32x4,
                op: SsaOp::BdaLoad {
                    addr,
                    byte_offset: base_off,
                    count: 4,
                },
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

        for blk in &mut module.blocks {
            for ins in &mut blk.insts {
                if !ins.dead {
                    rewrite_operands(&mut ins.op, &repl);
                }
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
            | SsaOp::StageInput(_)
            | SsaOp::BitcastU32ToF32(_)
            | SsaOp::BitcastF32ToU32(_)
            | SsaOp::PackU64 { .. }
            | SsaOp::UnpackLoU64(_)
            | SsaOp::UnpackHiU64(_)
            | SsaOp::AddU32(_, _)
            | SsaOp::SubU32(_, _)
            | SsaOp::MulU32(_, _)
            | SsaOp::AddU64(_, _)
            | SsaOp::AndU64(_, _)
            | SsaOp::BfeI32 { .. }
            | SsaOp::HalfToF32(_)
            | SsaOp::PackHalf2x16(_, _)
            | SsaOp::RoundEvenF32(_)
            | SsaOp::AndU32(_, _)
            | SsaOp::OrU32(_, _)
            | SsaOp::XorU32(_, _)
            | SsaOp::ShlU32(_, _)
            | SsaOp::LshrU32(_, _)
            | SsaOp::AshrI32(_, _)
            | SsaOp::BfeU32 { .. }
            | SsaOp::AddF32(_, _)
            | SsaOp::SubF32(_, _)
            | SsaOp::MulF32(_, _)
            | SsaOp::MulLegacyF32(_, _)
            | SsaOp::MinF32(_, _)
            | SsaOp::MaxF32(_, _)
            | SsaOp::FmaF32(_, _, _)
            | SsaOp::Log2F32(_)
            | SsaOp::Exp2F32(_)
            | SsaOp::RcpF32(_)
            | SsaOp::RsqF32(_)
            | SsaOp::SqrtF32(_)
            | SsaOp::FloorF32(_)
            | SsaOp::FractF32(_)
            | SsaOp::CvtF32U32(_)
            | SsaOp::CvtU32F32(_)
            | SsaOp::CvtF32I32(_)
            | SsaOp::CvtI32F32(_)
            | SsaOp::Cmp { .. }
            | SsaOp::LogicalAnd(_, _)
            | SsaOp::LogicalOr(_, _)
            | SsaOp::LogicalXor(_, _)
            | SsaOp::LogicalNot(_)
            | SsaOp::SelectU32 { .. }
            | SsaOp::ExtractLane(_, _)
    )
}

fn has_side_effect(op: &SsaOp) -> bool {
    matches!(
        op,
        SsaOp::BdaStore { .. }
            | SsaOp::BdaAtomic { .. }
            | SsaOp::LdsStoreU32 { .. }
            | SsaOp::LdsAtomicAddU32 { .. }
            | SsaOp::SpillStore { .. }
            | SsaOp::ImageStore { .. }
            | SsaOp::Export { .. }
            | SsaOp::WorkgroupBarrier
            | SsaOp::WaveBallot64(_)
    )
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
        | SsaOp::ThreadIdX
        | SsaOp::StageInput(_)
        | SsaOp::SpillLoad { .. }
        | SsaOp::WorkgroupBarrier => {}
        SsaOp::Phi { incoming } => {
            for (_, v) in incoming {
                f(v);
            }
        }
        SsaOp::BitcastU32ToF32(a)
        | SsaOp::BitcastF32ToU32(a)
        | SsaOp::UnpackLoU64(a)
        | SsaOp::UnpackHiU64(a)
        | SsaOp::Log2F32(a)
        | SsaOp::Exp2F32(a)
        | SsaOp::RcpF32(a)
        | SsaOp::RsqF32(a)
        | SsaOp::SqrtF32(a)
        | SsaOp::FloorF32(a)
        | SsaOp::FractF32(a)
        | SsaOp::CvtF32U32(a)
        | SsaOp::CvtU32F32(a)
        | SsaOp::CvtF32I32(a)
        | SsaOp::CvtI32F32(a)
        | SsaOp::LogicalNot(a)
        | SsaOp::WaveReadFirstLane(a)
        | SsaOp::WaveBallot64(a)
        | SsaOp::WaveAny(a)
        | SsaOp::QuadSwap { val: a, .. }
        | SsaOp::QuadBroadcast { val: a, .. }
        | SsaOp::LdsLoadU32 { byte_addr: a }
        | SsaOp::SpillStore { val: a, .. }
        | SsaOp::ExtractLane(a, _)
        | SsaOp::HalfToF32(a)
        | SsaOp::RoundEvenF32(a)
        | SsaOp::BdaLoad { addr: a, .. } => f(a),
        SsaOp::ImageSample {
            u, v, lod, bias, ..
        } => {
            f(u);
            f(v);
            lod.iter_mut().for_each(&f);
            bias.iter_mut().for_each(&f);
        }
        SsaOp::ImageSampleDref {
            u, v, dref, lod, ..
        } => {
            f(u);
            f(v);
            f(dref);
            lod.iter_mut().for_each(&f);
        }
        SsaOp::ImageGather { u, v, .. } => {
            f(u);
            f(v);
        }
        SsaOp::ImageQuerySize { lod, .. } => {
            f(lod);
        }
        SsaOp::ImageFetch { x, y, lod, .. } => {
            f(x);
            f(y);
            lod.iter_mut().for_each(&f);
        }
        SsaOp::ImageStore { x, y, values, .. } => {
            f(x);
            f(y);
            values.iter_mut().for_each(&f);
        }
        SsaOp::BdaAtomic {
            addr, val, cmp, ..
        } => {
            f(addr);
            f(val);
            cmp.iter_mut().for_each(&f);
        }
        SsaOp::WaveShuffle { val, lane } => {
            f(val);
            f(lane);
        }
        SsaOp::Export { values, .. } => values.iter_mut().for_each(&f),
        SsaOp::PackU64 { lo: a, hi: b }
        | SsaOp::AddU32(a, b)
        | SsaOp::SubU32(a, b)
        | SsaOp::MulU32(a, b)
        | SsaOp::AddU64(a, b)
        | SsaOp::AndU64(a, b)
        | SsaOp::PackHalf2x16(a, b)
        | SsaOp::AndU32(a, b)
        | SsaOp::OrU32(a, b)
        | SsaOp::XorU32(a, b)
        | SsaOp::ShlU32(a, b)
        | SsaOp::LshrU32(a, b)
        | SsaOp::AshrI32(a, b)
        | SsaOp::AddF32(a, b)
        | SsaOp::SubF32(a, b)
        | SsaOp::MulF32(a, b)
        | SsaOp::MulLegacyF32(a, b)
        | SsaOp::MinF32(a, b)
        | SsaOp::MaxF32(a, b)
        | SsaOp::Cmp { a, b, .. }
        | SsaOp::LogicalAnd(a, b)
        | SsaOp::LogicalOr(a, b)
        | SsaOp::LogicalXor(a, b)
        | SsaOp::LdsStoreU32 {
            byte_addr: a,
            val: b,
        }
        | SsaOp::LdsAtomicAddU32 {
            byte_addr: a,
            val: b,
        } => {
            f(a);
            f(b);
        }
        SsaOp::FmaF32(a, b, c)
        | SsaOp::BfeU32 {
            src: a,
            offset: b,
            width: c,
        }
        | SsaOp::BfeI32 {
            src: a,
            offset: b,
            width: c,
        }
        | SsaOp::SelectU32 {
            cond: a,
            t: b,
            f: c,
        } => {
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
        SsaOp::Bvh64IntersectBox32 {
            node_addr,
            t_max,
            ox,
            oy,
            oz,
            inv_dx,
            inv_dy,
            inv_dz,
        } => {
            f(node_addr);
            f(t_max);
            f(ox);
            f(oy);
            f(oz);
            f(inv_dx);
            f(inv_dy);
            f(inv_dz);
        }
    }
}

fn for_each_operand(op: &SsaOp, mut f: impl FnMut(ValueId)) {
    match op {
        SsaOp::ConstU32(_)
        | SsaOp::ConstF32(_)
        | SsaOp::ConstBool(_)
        | SsaOp::PushConstSgpr(_)
        | SsaOp::ThreadIdX
        | SsaOp::StageInput(_)
        | SsaOp::SpillLoad { .. }
        | SsaOp::WorkgroupBarrier => {}
        SsaOp::Phi { incoming } => {
            for &(_, v) in incoming {
                f(v);
            }
        }
        SsaOp::BitcastU32ToF32(a)
        | SsaOp::BitcastF32ToU32(a)
        | SsaOp::UnpackLoU64(a)
        | SsaOp::UnpackHiU64(a)
        | SsaOp::Log2F32(a)
        | SsaOp::Exp2F32(a)
        | SsaOp::RcpF32(a)
        | SsaOp::RsqF32(a)
        | SsaOp::SqrtF32(a)
        | SsaOp::FloorF32(a)
        | SsaOp::FractF32(a)
        | SsaOp::CvtF32U32(a)
        | SsaOp::CvtU32F32(a)
        | SsaOp::CvtF32I32(a)
        | SsaOp::CvtI32F32(a)
        | SsaOp::LogicalNot(a)
        | SsaOp::WaveReadFirstLane(a)
        | SsaOp::WaveBallot64(a)
        | SsaOp::WaveAny(a)
        | SsaOp::QuadSwap { val: a, .. }
        | SsaOp::QuadBroadcast { val: a, .. }
        | SsaOp::LdsLoadU32 { byte_addr: a }
        | SsaOp::SpillStore { val: a, .. }
        | SsaOp::ExtractLane(a, _)
        | SsaOp::HalfToF32(a)
        | SsaOp::RoundEvenF32(a)
        | SsaOp::BdaLoad { addr: a, .. } => f(*a),
        SsaOp::ImageSample {
            u, v, lod, bias, ..
        } => {
            f(*u);
            f(*v);
            lod.iter().chain(bias.iter()).for_each(|&x| f(x));
        }
        SsaOp::ImageSampleDref {
            u, v, dref, lod, ..
        } => {
            f(*u);
            f(*v);
            f(*dref);
            lod.iter().for_each(|&x| f(x));
        }
        SsaOp::ImageGather { u, v, .. } => {
            f(*u);
            f(*v);
        }
        SsaOp::ImageQuerySize { lod, .. } => {
            f(*lod);
        }
        SsaOp::ImageFetch { x, y, lod, .. } => {
            f(*x);
            f(*y);
            lod.iter().for_each(|&l| f(l));
        }
        SsaOp::ImageStore { x, y, values, .. } => {
            f(*x);
            f(*y);
            values.iter().for_each(|&v| f(v));
        }
        SsaOp::BdaAtomic {
            addr, val, cmp, ..
        } => {
            f(*addr);
            f(*val);
            cmp.iter().for_each(|&c| f(c));
        }
        SsaOp::WaveShuffle { val, lane } => {
            f(*val);
            f(*lane);
        }
        SsaOp::Export { values, .. } => values.iter().for_each(|&v| f(v)),
        SsaOp::PackU64 { lo: a, hi: b }
        | SsaOp::AddU32(a, b)
        | SsaOp::SubU32(a, b)
        | SsaOp::MulU32(a, b)
        | SsaOp::AddU64(a, b)
        | SsaOp::AndU64(a, b)
        | SsaOp::PackHalf2x16(a, b)
        | SsaOp::AndU32(a, b)
        | SsaOp::OrU32(a, b)
        | SsaOp::XorU32(a, b)
        | SsaOp::ShlU32(a, b)
        | SsaOp::LshrU32(a, b)
        | SsaOp::AshrI32(a, b)
        | SsaOp::AddF32(a, b)
        | SsaOp::SubF32(a, b)
        | SsaOp::MulF32(a, b)
        | SsaOp::MulLegacyF32(a, b)
        | SsaOp::MinF32(a, b)
        | SsaOp::MaxF32(a, b)
        | SsaOp::Cmp { a, b, .. }
        | SsaOp::LogicalAnd(a, b)
        | SsaOp::LogicalOr(a, b)
        | SsaOp::LogicalXor(a, b)
        | SsaOp::LdsStoreU32 {
            byte_addr: a,
            val: b,
        }
        | SsaOp::LdsAtomicAddU32 {
            byte_addr: a,
            val: b,
        } => {
            f(*a);
            f(*b);
        }
        SsaOp::FmaF32(a, b, c)
        | SsaOp::BfeU32 {
            src: a,
            offset: b,
            width: c,
        }
        | SsaOp::BfeI32 {
            src: a,
            offset: b,
            width: c,
        }
        | SsaOp::SelectU32 {
            cond: a,
            t: b,
            f: c,
        } => {
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
        SsaOp::Bvh64IntersectBox32 {
            node_addr,
            t_max,
            ox,
            oy,
            oz,
            inv_dx,
            inv_dy,
            inv_dz,
        } => {
            f(*node_addr);
            f(*t_max);
            f(*ox);
            f(*oy);
            f(*oz);
            f(*inv_dx);
            f(*inv_dy);
            f(*inv_dz);
        }
    }
}
