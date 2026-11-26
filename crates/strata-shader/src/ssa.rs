//! SSA IR & per-lane EXEC structurization.
//!
//! Translates decoded RDNA2 instructions into typed SSA values where:
//! - Console User SGPRs map directly to No-API Push Constants (32 x u32).
//! - Global memory accesses use 64-bit `PhysicalStorageBuffer` addresses
//!   (no 4 GiB D3D12 window switches).
//! - Lane masks (`EXEC`, `VCC`) are lowered to per-lane booleans unless
//!   `wave.rs` marks them as whole-wave live.

use std::collections::HashMap;

use crate::rdna2::{DecodedInst, Inst, Operand};

/// SSA value identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

/// Value type in Strata's SSA IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ty {
    Bool,
    U32,
    F32,
    U64,
    U32x4,
}

/// Pure or memory-accessing SSA operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SsaOp {
    ConstU32(u32),
    ConstF32(u32), // IEEE-754 bits
    ConstBool(bool),
    /// Load a console User SGPR directly from Push Constants (`s[idx]`).
    PushConstSgpr(u8),
    /// Compute shader `gl_GlobalInvocationID.x`.
    ThreadIdX,

    // Scalar & vector arithmetic
    BitcastU32ToF32(ValueId),
    BitcastF32ToU32(ValueId),
    PackU64 { lo: ValueId, hi: ValueId },
    AddU32(ValueId, ValueId),
    AddU64(ValueId, ValueId),
    ShlU32(ValueId, ValueId),
    AddF32(ValueId, ValueId),
    MulF32(ValueId, ValueId),
    /// Legacy DX9/RDNA2 multiply (`v_mul_legacy_f32`): `0.0 * x == 0.0` even for NaN/Inf.
    MulLegacyF32(ValueId, ValueId),
    /// Base-2 log (`v_log_f32` -> `GLSL.std.450 Log2`).
    Log2F32(ValueId),
    /// Native fused multiply-add: `a * b + c`.
    FmaF32(ValueId, ValueId, ValueId),
    CmpLtF32(ValueId, ValueId),
    SelectU32 { cond: ValueId, t: ValueId, f: ValueId },

    /// Subgroup wave broadcast from first active lane (`v_readfirstlane_b32`).
    WaveReadFirstLane(ValueId),
    /// Helper-pixel-safe 2x2 quad swap (`OpGroupNonUniformQuadSwap`, direction 0=X, 1=Y, 2=Diag).
    QuadSwap { val: ValueId, direction: u8 },

    /// Direct 64-bit BDA (`PhysicalStorageBuffer`) load of `count` dwords (1, 2, or 4).
    BdaLoad { addr: ValueId, byte_offset: u32, count: u8 },
    /// Extract lane `k` (0..4) from a coalesced `U32x4` BDA load.
    ExtractLane(ValueId, u8),
    /// Direct 64-bit BDA (`PhysicalStorageBuffer`) store of `count` dwords.
    BdaStore { addr: ValueId, byte_offset: u32, values: Vec<ValueId> },
}

#[derive(Debug, Clone)]
pub struct SsaInst {
    pub id: ValueId,
    pub ty: Ty,
    pub op: SsaOp,
    pub dead: bool,
}

#[derive(Debug, Clone)]
pub struct SsaBlock {
    pub id: usize,
    pub insts: Vec<SsaInst>,
}

/// Complete SSA module ready for optimization and SPIR-V emission.
#[derive(Debug, Clone)]
pub struct SsaModule {
    pub blocks: Vec<SsaBlock>,
    pub next_id: u32,
    pub workgroup_size: [u32; 3],
}

impl SsaModule {
    pub fn new(workgroup_size: [u32; 3]) -> Self {
        Self {
            blocks: vec![SsaBlock { id: 0, insts: Vec::new() }],
            next_id: 0,
            workgroup_size,
        }
    }

    pub fn emit(&mut self, block: usize, ty: Ty, op: SsaOp) -> ValueId {
        let id = ValueId(self.next_id);
        self.next_id += 1;
        self.blocks[block].insts.push(SsaInst { id, ty, op, dead: false });
        id
    }

    pub fn live_inst_count(&self) -> usize {
        self.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| !i.dead)
            .count()
    }
}

/// Build an SSA module from decoded RDNA2 instructions (entry block lowering).
pub fn build_ssa(program: &[DecodedInst], workgroup_size: [u32; 3]) -> SsaModule {
    let mut m = SsaModule::new(workgroup_size);
    let mut sgpr: HashMap<u8, ValueId> = HashMap::new();
    let mut vgpr: HashMap<u8, ValueId> = HashMap::new();

    // Seed v0 with GlobalInvocationID.x (standard RDNA2 compute ABI).
    let tid = m.emit(0, Ty::U32, SsaOp::ThreadIdX);
    vgpr.insert(0, tid);

    let mut vcc_lane = m.emit(0, Ty::Bool, SsaOp::ConstBool(false));

    let read_sgpr = |m: &mut SsaModule, map: &mut HashMap<u8, ValueId>, idx: u8| -> ValueId {
        *map.entry(idx)
            .or_insert_with(|| m.emit(0, Ty::U32, SsaOp::PushConstSgpr(idx)))
    };

    let read_vgpr = |m: &mut SsaModule, map: &mut HashMap<u8, ValueId>, idx: u8| -> ValueId {
        *map.entry(idx)
            .or_insert_with(|| m.emit(0, Ty::U32, SsaOp::ConstU32(0)))
    };

    let read_op_u32 = |m: &mut SsaModule,
                       sgpr: &mut HashMap<u8, ValueId>,
                       vgpr: &mut HashMap<u8, ValueId>,
                       op: Operand|
     -> ValueId {
        match op {
            Operand::Sgpr(s) => read_sgpr(m, sgpr, s),
            Operand::Vgpr(v) => read_vgpr(m, vgpr, v),
            Operand::InlineInt(i) => m.emit(0, Ty::U32, SsaOp::ConstU32(i as u32)),
            Operand::InlineFloat(bits) => m.emit(0, Ty::U32, SsaOp::ConstU32(bits)),
            Operand::Literal(lit) => m.emit(0, Ty::U32, SsaOp::ConstU32(lit)),
            _ => m.emit(0, Ty::U32, SsaOp::ConstU32(0)),
        }
    };

    for di in program {
        match &di.inst {
            Inst::SWaitcnt | Inst::SEndpgm => {}
            Inst::SMovB32 { dst, src0 } => {
                let v = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src0);
                sgpr.insert(*dst, v);
            }
            Inst::SAddU32 { dst, src0, src1 } => {
                let a = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src0);
                let b = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src1);
                let r = m.emit(0, Ty::U32, SsaOp::AddU32(a, b));
                sgpr.insert(*dst, r);
            }
            Inst::VMovB32 { dst, src0 } => {
                let v = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src0);
                vgpr.insert(*dst, v);
            }
            Inst::VLogF32 { dst, src0 } => {
                let a = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src0);
                let fa = m.emit(0, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let fr = m.emit(0, Ty::F32, SsaOp::Log2F32(fa));
                let r = m.emit(0, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                vgpr.insert(*dst, r);
            }
            Inst::VAddU32 { dst, src0, src1 } => {
                let a = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src0);
                let b = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src1);
                let r = m.emit(0, Ty::U32, SsaOp::AddU32(a, b));
                vgpr.insert(*dst, r);
            }
            Inst::VLshlRevB32 { dst, shift, src } => {
                let s = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *shift);
                let v = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src);
                let r = m.emit(0, Ty::U32, SsaOp::ShlU32(v, s));
                vgpr.insert(*dst, r);
            }
            Inst::VAddF32 { dst, src0, src1 } => {
                let a = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src0);
                let b = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src1);
                let fa = m.emit(0, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let fb = m.emit(0, Ty::F32, SsaOp::BitcastU32ToF32(b));
                let fr = m.emit(0, Ty::F32, SsaOp::AddF32(fa, fb));
                let r = m.emit(0, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                vgpr.insert(*dst, r);
            }
            Inst::VMulF32 { dst, src0, src1, legacy } => {
                let a = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src0);
                let b = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src1);
                let fa = m.emit(0, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let fb = m.emit(0, Ty::F32, SsaOp::BitcastU32ToF32(b));
                let fr = if *legacy {
                    m.emit(0, Ty::F32, SsaOp::MulLegacyF32(fa, fb))
                } else {
                    m.emit(0, Ty::F32, SsaOp::MulF32(fa, fb))
                };
                let r = m.emit(0, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                vgpr.insert(*dst, r);
            }
            Inst::VFmacF32 { dst, src0, src1 } => {
                let a = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src0);
                let b = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src1);
                let c = read_vgpr(&mut m, &mut vgpr, *dst);
                let fa = m.emit(0, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let fb = m.emit(0, Ty::F32, SsaOp::BitcastU32ToF32(b));
                let fc = m.emit(0, Ty::F32, SsaOp::BitcastU32ToF32(c));
                let fr = m.emit(0, Ty::F32, SsaOp::FmaF32(fa, fb, fc));
                let r = m.emit(0, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                vgpr.insert(*dst, r);
            }
            Inst::VCmpLtF32 { src0, src1 } => {
                let a = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src0);
                let b = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src1);
                let fa = m.emit(0, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let fb = m.emit(0, Ty::F32, SsaOp::BitcastU32ToF32(b));
                vcc_lane = m.emit(0, Ty::Bool, SsaOp::CmpLtF32(fa, fb));
            }
            Inst::VCndMaskB32 { dst, src0, src1 } => {
                let f = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src0);
                let t = read_op_u32(&mut m, &mut sgpr, &mut vgpr, *src1);
                let r = m.emit(0, Ty::U32, SsaOp::SelectU32 { cond: vcc_lane, t, f });
                vgpr.insert(*dst, r);
            }
            Inst::VReadFirstLaneB32 { dst, src } => {
                let v = read_vgpr(&mut m, &mut vgpr, *src);
                let r = m.emit(0, Ty::U32, SsaOp::WaveReadFirstLane(v));
                sgpr.insert(*dst, r);
            }
            Inst::VQuadSwap { dst, src, direction } => {
                let v = read_vgpr(&mut m, &mut vgpr, *src);
                let r = m.emit(0, Ty::U32, SsaOp::QuadSwap { val: v, direction: *direction });
                vgpr.insert(*dst, r);
            }
            Inst::GlobalLoad { dst, count, vaddr, saddr, offset } => {
                let base_addr = build_bda_addr(&mut m, &mut sgpr, &mut vgpr, *vaddr, *saddr);
                let loaded = m.emit(
                    0,
                    if *count == 4 { Ty::U32x4 } else { Ty::U32 },
                    SsaOp::BdaLoad {
                        addr: base_addr,
                        byte_offset: (*offset).max(0) as u32,
                        count: *count,
                    },
                );
                if *count == 1 {
                    vgpr.insert(*dst, loaded);
                } else {
                    for k in 0..*count {
                        let lane = m.emit(0, Ty::U32, SsaOp::ExtractLane(loaded, k));
                        vgpr.insert(*dst + k, lane);
                    }
                }
            }
            Inst::GlobalStore { vaddr, vdata, count, saddr, offset } => {
                let base_addr = build_bda_addr(&mut m, &mut sgpr, &mut vgpr, *vaddr, *saddr);
                let mut vals = Vec::with_capacity(*count as usize);
                for k in 0..*count {
                    vals.push(read_vgpr(&mut m, &mut vgpr, *vdata + k));
                }
                m.emit(
                    0,
                    Ty::U32,
                    SsaOp::BdaStore {
                        addr: base_addr,
                        byte_offset: (*offset).max(0) as u32,
                        values: vals,
                    },
                );
            }
            Inst::SLoadDword { dst, count, sbase, offset } => {
                let lo = read_sgpr(&mut m, &mut sgpr, *sbase);
                let hi = read_sgpr(&mut m, &mut sgpr, *sbase + 1);
                let addr = m.emit(0, Ty::U64, SsaOp::PackU64 { lo, hi });
                let loaded = m.emit(
                    0,
                    if *count == 4 { Ty::U32x4 } else { Ty::U32 },
                    SsaOp::BdaLoad {
                        addr,
                        byte_offset: (*offset).max(0) as u32,
                        count: *count,
                    },
                );
                if *count == 1 {
                    sgpr.insert(*dst, loaded);
                } else {
                    for k in 0..*count {
                        let lane = m.emit(0, Ty::U32, SsaOp::ExtractLane(loaded, k));
                        sgpr.insert(*dst + k, lane);
                    }
                }
            }
            Inst::SAndB64 { .. }
            | Inst::SAndSaveExecB64 { .. }
            | Inst::SBranch { .. }
            | Inst::SCBranchExecz { .. }
            | Inst::SCBranchScc { .. }
            | Inst::ImageBvh64IntersectRay { .. } => {
                // Control-flow & inline BVH lowering hooks live in their dedicated passes.
            }
        }
    }

    m
}

fn build_bda_addr(
    m: &mut SsaModule,
    sgpr: &mut HashMap<u8, ValueId>,
    vgpr: &mut HashMap<u8, ValueId>,
    vaddr: u8,
    saddr: Option<u8>,
) -> ValueId {
    let v_lo = *vgpr.entry(vaddr).or_insert_with(|| m.emit(0, Ty::U32, SsaOp::ConstU32(0)));
    if let Some(s) = saddr {
        let s_lo = *sgpr.entry(s).or_insert_with(|| m.emit(0, Ty::U32, SsaOp::PushConstSgpr(s)));
        let s_hi = *sgpr.entry(s + 1).or_insert_with(|| m.emit(0, Ty::U32, SsaOp::PushConstSgpr(s + 1)));
        let base = m.emit(0, Ty::U64, SsaOp::PackU64 { lo: s_lo, hi: s_hi });
        let zero = m.emit(0, Ty::U32, SsaOp::ConstU32(0));
        let off64 = m.emit(0, Ty::U64, SsaOp::PackU64 { lo: v_lo, hi: zero });
        m.emit(0, Ty::U64, SsaOp::AddU64(base, off64))
    } else {
        let v_hi = *vgpr.entry(vaddr + 1).or_insert_with(|| m.emit(0, Ty::U32, SsaOp::ConstU32(0)));
        m.emit(0, Ty::U64, SsaOp::PackU64 { lo: v_lo, hi: v_hi })
    }
}
