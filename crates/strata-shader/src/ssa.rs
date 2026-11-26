//! SSA IR, per-lane EXEC/VCC lowering & structured CFG construction.
//!
//! Translates decoded RDNA2 instructions into typed SSA basic blocks where:
//! - Console User SGPRs map directly to No-API Push Constants (`uint sgpr[32]`).
//! - Global & MUBUF memory accesses use 64-bit `PhysicalStorageBuffer` addresses.
//! - `DS` instructions lower to Workgroup shared memory (LDS) + `OpControlBarrier`.
//! - Lane masks (`EXEC`, `VCC`) stay per-lane `Ty::Bool` when `wave.rs` classifies
//!   them as lane-local, and lower to `SsaOp::WaveBallot64` when whole-live.
//! - Forward conditional branches (`s_cbranch_execz`, `s_cbranch_scc0/1`) emit
//!   structured multi-block `Terminator::BranchCond` + `SsaOp::Phi` nodes.
//! - Any unsupported instruction returns `Err(LowerError::Unsupported)` instead
//!   of being silently discarded.

use std::collections::HashMap;

use thiserror::Error;

use crate::rdna2::{CmpOp, CmpType, DecodedInst, Inst, Operand};
use crate::wave::WaveAnalysis;

#[derive(Debug, Error)]
pub enum LowerError {
    #[error("unsupported instruction for compute SPIR-V lowering at pc {pc}: {desc}")]
    Unsupported { pc: usize, desc: &'static str },
    #[error("unsupported backward/irreducible branch at pc {pc} -> target {target_dw}")]
    UnsupportedBranch { pc: usize, target_dw: usize },
}

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

/// Structured block terminator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    Return,
    Branch(usize),
    BranchCond {
        cond: ValueId,
        true_block: usize,
        false_block: usize,
        merge_block: usize,
    },
}

/// Pure, memory-accessing, or wave-level SSA operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SsaOp {
    ConstU32(u32),
    ConstF32(u32), // IEEE-754 bits
    ConstBool(bool),
    /// Load a console User SGPR directly from Push Constants (`s[idx]`).
    PushConstSgpr(u8),
    /// Compute shader `gl_GlobalInvocationID.x`.
    ThreadIdX,
    /// SSA merge node at the top of a join block (`OpPhi`).
    Phi { incoming: Vec<(usize, ValueId)> },

    // Scalar & vector arithmetic
    BitcastU32ToF32(ValueId),
    BitcastF32ToU32(ValueId),
    PackU64 { lo: ValueId, hi: ValueId },
    UnpackLoU64(ValueId),
    UnpackHiU64(ValueId),
    AddU32(ValueId, ValueId),
    SubU32(ValueId, ValueId),
    MulU32(ValueId, ValueId),
    AddU64(ValueId, ValueId),
    AndU32(ValueId, ValueId),
    OrU32(ValueId, ValueId),
    XorU32(ValueId, ValueId),
    ShlU32(ValueId, ValueId),
    LshrU32(ValueId, ValueId),
    AshrI32(ValueId, ValueId),
    BfeU32 { src: ValueId, offset: ValueId, width: ValueId },

    AddF32(ValueId, ValueId),
    SubF32(ValueId, ValueId),
    MulF32(ValueId, ValueId),
    /// Legacy DX9/RDNA2 multiply (`v_mul_legacy_f32`): `0.0 * x == 0.0` even for NaN/Inf.
    MulLegacyF32(ValueId, ValueId),
    MinF32(ValueId, ValueId),
    MaxF32(ValueId, ValueId),
    /// Native fused multiply-add: `a * b + c`.
    FmaF32(ValueId, ValueId, ValueId),
    Log2F32(ValueId),
    Exp2F32(ValueId),
    RcpF32(ValueId),
    RsqF32(ValueId),
    SqrtF32(ValueId),
    FloorF32(ValueId),
    FractF32(ValueId),
    CvtF32U32(ValueId),
    CvtU32F32(ValueId),
    CvtF32I32(ValueId),
    CvtI32F32(ValueId),

    // Comparisons & boolean logic
    Cmp { op: CmpOp, ty: CmpType, a: ValueId, b: ValueId },
    LogicalAnd(ValueId, ValueId),
    LogicalOr(ValueId, ValueId),
    LogicalXor(ValueId, ValueId),
    LogicalNot(ValueId),
    SelectU32 { cond: ValueId, t: ValueId, f: ValueId },

    // Subgroup / Wave64 operations
    /// Subgroup wave broadcast from first active lane (`v_readfirstlane_b32`).
    WaveReadFirstLane(ValueId),
    /// Subgroup-size-agnostic 64-lane ballot (`OpGroupNonUniformBallot` + Workgroup atomic merge).
    WaveBallot64(ValueId),
    /// Helper-pixel-safe 2x2 quad swap (`OpGroupNonUniformQuadSwap`, direction 0=X, 1=Y, 2=Diag).
    QuadSwap { val: ValueId, direction: u8 },

    // Workgroup Shared Memory (LDS) & Synchronization
    WorkgroupBarrier,
    LdsLoadU32 { byte_addr: ValueId },
    LdsStoreU32 { byte_addr: ValueId, val: ValueId },

    // Direct 64-bit BDA (`PhysicalStorageBuffer`) memory
    BdaLoad { addr: ValueId, byte_offset: u32, count: u8 },
    ExtractLane(ValueId, u8),
    BdaStore { addr: ValueId, byte_offset: u32, values: Vec<ValueId> },

    /// Inline ray-AABB Box32 node intersection (`image_bvh64_intersect_ray`).
    /// Returns `Ty::U32x4` of hit child node IDs (or `0xFFFF_FFFF` for misses).
    Bvh64IntersectBox32 {
        node_addr: ValueId,
        t_max: ValueId,
        ox: ValueId,
        oy: ValueId,
        oz: ValueId,
        inv_dx: ValueId,
        inv_dy: ValueId,
        inv_dz: ValueId,
    },
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
    pub term: Terminator,
}

/// Complete SSA module ready for optimization and SPIR-V emission.
#[derive(Debug, Clone)]
pub struct SsaModule {
    pub blocks: Vec<SsaBlock>,
    pub next_id: u32,
    pub workgroup_size: [u32; 3],
    pub uses_lds: bool,
    pub uses_wave64_ballot: bool,
}

impl SsaModule {
    pub fn new(workgroup_size: [u32; 3]) -> Self {
        Self {
            blocks: vec![SsaBlock {
                id: 0,
                insts: Vec::new(),
                term: Terminator::Return,
            }],
            next_id: 0,
            workgroup_size,
            uses_lds: false,
            uses_wave64_ballot: false,
        }
    }

    pub fn new_block(&mut self) -> usize {
        let id = self.blocks.len();
        self.blocks.push(SsaBlock {
            id,
            insts: Vec::new(),
            term: Terminator::Return,
        });
        id
    }

    pub fn emit(&mut self, block: usize, ty: Ty, op: SsaOp) -> ValueId {
        if matches!(op, SsaOp::LdsLoadU32 { .. } | SsaOp::LdsStoreU32 { .. }) {
            self.uses_lds = true;
        }
        if matches!(op, SsaOp::WaveBallot64(_)) {
            self.uses_wave64_ballot = true;
        }
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

struct OpenRegion {
    target_dw: usize,
    header_block: usize,
    merge_block: usize,
    pre_vgpr: HashMap<u8, ValueId>,
    pre_sgpr: HashMap<u8, ValueId>,
}

struct Builder {
    m: SsaModule,
    cur_block: usize,
    sgpr: HashMap<u8, ValueId>,
    vgpr: HashMap<u8, ValueId>,
    mask_lanes: HashMap<u8, ValueId>,
    vcc_lane: ValueId,
    vcc_ballot64: Option<ValueId>,
    exec_lane: ValueId,
    exec_is_all_true: bool,
    scc: ValueId,
}

impl Builder {
    fn read_sgpr(&mut self, idx: u8) -> ValueId {
        if let Some(&v) = self.sgpr.get(&idx) {
            return v;
        }
        // Emit PushConstSgpr in entry block 0 so it dominates all blocks.
        let v = self.m.emit(0, Ty::U32, SsaOp::PushConstSgpr(idx));
        self.sgpr.insert(idx, v);
        v
    }

    fn read_vgpr(&mut self, idx: u8) -> ValueId {
        if let Some(&v) = self.vgpr.get(&idx) {
            return v;
        }
        let v = self.m.emit(0, Ty::U32, SsaOp::ConstU32(0));
        self.vgpr.insert(idx, v);
        v
    }

    fn write_vgpr(&mut self, dst: u8, val: ValueId) {
        if self.exec_is_all_true {
            self.vgpr.insert(dst, val);
        } else {
            let old = self.read_vgpr(dst);
            let sel = self.m.emit(
                self.cur_block,
                Ty::U32,
                SsaOp::SelectU32 {
                    cond: self.exec_lane,
                    t: val,
                    f: old,
                },
            );
            self.vgpr.insert(dst, sel);
        }
    }

    fn read_op_u32(&mut self, op: Operand) -> ValueId {
        let b = self.cur_block;
        match op {
            Operand::Sgpr(s) => self.read_sgpr(s),
            Operand::Vgpr(v) => self.read_vgpr(v),
            Operand::InlineInt(i) => self.m.emit(b, Ty::U32, SsaOp::ConstU32(i as u32)),
            Operand::InlineFloat(bits) => self.m.emit(b, Ty::U32, SsaOp::ConstU32(bits)),
            Operand::Literal(lit) => self.m.emit(b, Ty::U32, SsaOp::ConstU32(lit)),
            Operand::VccLo => {
                let mask64 = self.ensure_vcc_ballot64();
                self.m.emit(b, Ty::U32, SsaOp::UnpackLoU64(mask64))
            }
            Operand::VccHi => {
                let mask64 = self.ensure_vcc_ballot64();
                self.m.emit(b, Ty::U32, SsaOp::UnpackHiU64(mask64))
            }
            Operand::ExecLo => {
                let mask64 = self.m.emit(b, Ty::U64, SsaOp::WaveBallot64(self.exec_lane));
                self.m.emit(b, Ty::U32, SsaOp::UnpackLoU64(mask64))
            }
            Operand::ExecHi => {
                let mask64 = self.m.emit(b, Ty::U64, SsaOp::WaveBallot64(self.exec_lane));
                self.m.emit(b, Ty::U32, SsaOp::UnpackHiU64(mask64))
            }
            Operand::Scc => {
                let one = self.m.emit(b, Ty::U32, SsaOp::ConstU32(1));
                let zero = self.m.emit(b, Ty::U32, SsaOp::ConstU32(0));
                self.m.emit(b, Ty::U32, SsaOp::SelectU32 { cond: self.scc, t: one, f: zero })
            }
            Operand::M0 => self.m.emit(b, Ty::U32, SsaOp::ConstU32(0)),
        }
    }

    fn ensure_vcc_ballot64(&mut self) -> ValueId {
        if let Some(id) = self.vcc_ballot64 {
            return id;
        }
        let id = self.m.emit(self.cur_block, Ty::U64, SsaOp::WaveBallot64(self.vcc_lane));
        self.vcc_ballot64 = Some(id);
        id
    }

    fn read_mask_lane(&mut self, op: Operand) -> ValueId {
        let b = self.cur_block;
        match op {
            Operand::VccLo | Operand::VccHi => self.vcc_lane,
            Operand::ExecLo | Operand::ExecHi => self.exec_lane,
            Operand::Sgpr(s) => self
                .mask_lanes
                .get(&s)
                .copied()
                .unwrap_or_else(|| self.m.emit(b, Ty::Bool, SsaOp::ConstBool(true))),
            Operand::InlineInt(i) => self.m.emit(b, Ty::Bool, SsaOp::ConstBool(i != 0)),
            _ => self.m.emit(b, Ty::Bool, SsaOp::ConstBool(true)),
        }
    }

    fn write_mask_lane(&mut self, dst: u8, val: ValueId) {
        match dst {
            106 => {
                self.vcc_lane = val;
                self.vcc_ballot64 = None;
            }
            126 => {
                self.exec_lane = val;
                self.exec_is_all_true = false;
            }
            s => {
                self.mask_lanes.insert(s, val);
            }
        }
    }

    fn build_bda_addr(&mut self, vaddr: u8, saddr: Option<u8>) -> ValueId {
        let b = self.cur_block;
        let v_lo = self.read_vgpr(vaddr);
        if let Some(s) = saddr {
            let s_lo = self.read_sgpr(s);
            let s_hi = self.read_sgpr(s + 1);
            let base = self.m.emit(b, Ty::U64, SsaOp::PackU64 { lo: s_lo, hi: s_hi });
            let zero = self.m.emit(b, Ty::U32, SsaOp::ConstU32(0));
            let off64 = self.m.emit(b, Ty::U64, SsaOp::PackU64 { lo: v_lo, hi: zero });
            self.m.emit(b, Ty::U64, SsaOp::AddU64(base, off64))
        } else {
            let v_hi = self.read_vgpr(vaddr + 1);
            self.m.emit(b, Ty::U64, SsaOp::PackU64 { lo: v_lo, hi: v_hi })
        }
    }

    fn build_mubuf_addr(&mut self, srsrc: u8, vaddr: u8, soffset: Operand, offset: u16, offen: bool) -> ValueId {
        let b = self.cur_block;
        // RDNA2 V# buffer descriptor: dword 0 = base[31:0], dword 1[15:0] = base[47:32]
        let r0 = self.read_sgpr(srsrc);
        let r1 = self.read_sgpr(srsrc + 1);
        let mask16 = self.m.emit(b, Ty::U32, SsaOp::ConstU32(0xffff));
        let r1_hi = self.m.emit(b, Ty::U32, SsaOp::AndU32(r1, mask16));
        let mut addr = self.m.emit(b, Ty::U64, SsaOp::PackU64 { lo: r0, hi: r1_hi });

        let zero = self.m.emit(b, Ty::U32, SsaOp::ConstU32(0));
        if offen {
            let v_off = self.read_vgpr(vaddr);
            let v_off64 = self.m.emit(b, Ty::U64, SsaOp::PackU64 { lo: v_off, hi: zero });
            addr = self.m.emit(b, Ty::U64, SsaOp::AddU64(addr, v_off64));
        }
        let s_off = self.read_op_u32(soffset);
        let s_off64 = self.m.emit(b, Ty::U64, SsaOp::PackU64 { lo: s_off, hi: zero });
        addr = self.m.emit(b, Ty::U64, SsaOp::AddU64(addr, s_off64));
        if offset != 0 {
            let imm = self.m.emit(b, Ty::U32, SsaOp::ConstU32(offset as u32));
            let imm64 = self.m.emit(b, Ty::U64, SsaOp::PackU64 { lo: imm, hi: zero });
            addr = self.m.emit(b, Ty::U64, SsaOp::AddU64(addr, imm64));
        }
        addr
    }
}

/// Build an SSA module from decoded RDNA2 instructions + wave liveness analysis.
pub fn build_ssa(
    program: &[DecodedInst],
    wave: &WaveAnalysis,
    workgroup_size: [u32; 3],
) -> Result<SsaModule, LowerError> {
    let mut m = SsaModule::new(workgroup_size);
    let tid = m.emit(0, Ty::U32, SsaOp::ThreadIdX);
    let vcc_init = m.emit(0, Ty::Bool, SsaOp::ConstBool(false));
    let exec_init = m.emit(0, Ty::Bool, SsaOp::ConstBool(true));
    let scc_init = m.emit(0, Ty::Bool, SsaOp::ConstBool(false));

    let mut b = Builder {
        m,
        cur_block: 0,
        sgpr: HashMap::new(),
        vgpr: HashMap::new(),
        mask_lanes: HashMap::new(),
        vcc_lane: vcc_init,
        vcc_ballot64: None,
        exec_lane: exec_init,
        exec_is_all_true: true,
        scc: scc_init,
    };
    b.vgpr.insert(0, tid);

    let mut open_regions: Vec<OpenRegion> = Vec::new();

    for di in program {
        // Close any structured forward-branch regions whose target PC we just reached.
        while let Some(top) = open_regions.last() {
            if di.pc >= top.target_dw {
                let region = open_regions.pop().unwrap();
                let then_end_block = b.cur_block;
                b.m.blocks[then_end_block].term = Terminator::Branch(region.merge_block);
                b.cur_block = region.merge_block;

                // Emit OpPhi in merge_block for any VGPR modified inside the taken branch.
                let mut modified_vgprs: Vec<u8> = b.vgpr.keys().copied().collect();
                modified_vgprs.sort_unstable();
                for vreg in modified_vgprs {
                    let after_val = b.vgpr[&vreg];
                    let before_val = region.pre_vgpr.get(&vreg).copied().unwrap_or_else(|| {
                        b.m.emit(0, Ty::U32, SsaOp::ConstU32(0))
                    });
                    if after_val != before_val {
                        let phi = b.m.emit(
                            region.merge_block,
                            Ty::U32,
                            SsaOp::Phi {
                                incoming: vec![
                                    (then_end_block, after_val),
                                    (region.header_block, before_val),
                                ],
                            },
                        );
                        b.vgpr.insert(vreg, phi);
                    }
                }
                let mut modified_sgprs: Vec<u8> = b.sgpr.keys().copied().collect();
                modified_sgprs.sort_unstable();
                for sreg in modified_sgprs {
                    let after_val = b.sgpr[&sreg];
                    if let Some(&before_val) = region.pre_sgpr.get(&sreg) {
                        if after_val != before_val {
                            let phi = b.m.emit(
                                region.merge_block,
                                Ty::U32,
                                SsaOp::Phi {
                                    incoming: vec![
                                        (then_end_block, after_val),
                                        (region.header_block, before_val),
                                    ],
                                },
                            );
                            b.sgpr.insert(sreg, phi);
                        }
                    }
                }
            } else {
                break;
            }
        }

        let blk = b.cur_block;
        match &di.inst {
            Inst::SNop | Inst::SWaitcnt | Inst::SEndpgm => {}
            Inst::SBarrier => {
                b.m.emit(blk, Ty::U32, SsaOp::WorkgroupBarrier);
            }
            Inst::SMovB32 { dst, src0 } => {
                let v = b.read_op_u32(*src0);
                b.sgpr.insert(*dst, v);
            }
            Inst::SMovkI32 { dst, simm16 } => {
                let v = b.m.emit(blk, Ty::U32, SsaOp::ConstU32(*simm16 as i32 as u32));
                b.sgpr.insert(*dst, v);
            }
            Inst::SAddkI32 { dst, simm16 } => {
                let a = b.read_sgpr(*dst);
                let imm = b.m.emit(blk, Ty::U32, SsaOp::ConstU32(*simm16 as i32 as u32));
                let r = b.m.emit(blk, Ty::U32, SsaOp::AddU32(a, imm));
                b.sgpr.insert(*dst, r);
            }
            Inst::SMulkI32 { dst, simm16 } => {
                let a = b.read_sgpr(*dst);
                let imm = b.m.emit(blk, Ty::U32, SsaOp::ConstU32(*simm16 as i32 as u32));
                let r = b.m.emit(blk, Ty::U32, SsaOp::MulU32(a, imm));
                b.sgpr.insert(*dst, r);
            }
            Inst::SAddU32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let r = b.m.emit(blk, Ty::U32, SsaOp::AddU32(x, y));
                b.sgpr.insert(*dst, r);
            }
            Inst::SSubU32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let r = b.m.emit(blk, Ty::U32, SsaOp::SubU32(x, y));
                b.sgpr.insert(*dst, r);
            }
            Inst::SMulI32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let r = b.m.emit(blk, Ty::U32, SsaOp::MulU32(x, y));
                b.sgpr.insert(*dst, r);
            }
            Inst::SAndB32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let r = b.m.emit(blk, Ty::U32, SsaOp::AndU32(x, y));
                b.sgpr.insert(*dst, r);
            }
            Inst::SOrB32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let r = b.m.emit(blk, Ty::U32, SsaOp::OrU32(x, y));
                b.sgpr.insert(*dst, r);
            }
            Inst::SXorB32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let r = b.m.emit(blk, Ty::U32, SsaOp::XorU32(x, y));
                b.sgpr.insert(*dst, r);
            }
            Inst::SLshlB32 { dst, src0, shift } => {
                let x = b.read_op_u32(*src0);
                let s = b.read_op_u32(*shift);
                let r = b.m.emit(blk, Ty::U32, SsaOp::ShlU32(x, s));
                b.sgpr.insert(*dst, r);
            }
            Inst::SLshrB32 { dst, src0, shift } => {
                let x = b.read_op_u32(*src0);
                let s = b.read_op_u32(*shift);
                let r = b.m.emit(blk, Ty::U32, SsaOp::LshrU32(x, s));
                b.sgpr.insert(*dst, r);
            }
            Inst::SCmp { op, ty, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                b.scc = b.m.emit(blk, Ty::Bool, SsaOp::Cmp { op: *op, ty: *ty, a: x, b: y });
            }

            // --- 64-bit Wave Mask & EXEC Operations ---
            Inst::SMovB64 { dst, src0 } => {
                let m_val = b.read_mask_lane(*src0);
                b.write_mask_lane(*dst, m_val);
            }
            Inst::SNotB64 { dst, src0 } => {
                let m_val = b.read_mask_lane(*src0);
                let inv = b.m.emit(blk, Ty::Bool, SsaOp::LogicalNot(m_val));
                b.write_mask_lane(*dst, inv);
            }
            Inst::SAndB64 { dst, src0, src1 } => {
                let m0 = b.read_mask_lane(*src0);
                let m1 = b.read_mask_lane(*src1);
                let r = b.m.emit(blk, Ty::Bool, SsaOp::LogicalAnd(m0, m1));
                b.write_mask_lane(*dst, r);
            }
            Inst::SOrB64 { dst, src0, src1 } => {
                let m0 = b.read_mask_lane(*src0);
                let m1 = b.read_mask_lane(*src1);
                let r = b.m.emit(blk, Ty::Bool, SsaOp::LogicalOr(m0, m1));
                b.write_mask_lane(*dst, r);
                if *dst == 126 {
                    // Restoring EXEC from saved mask
                    b.exec_is_all_true = false;
                }
            }
            Inst::SXorB64 { dst, src0, src1 } => {
                let m0 = b.read_mask_lane(*src0);
                let m1 = b.read_mask_lane(*src1);
                let r = b.m.emit(blk, Ty::Bool, SsaOp::LogicalXor(m0, m1));
                b.write_mask_lane(*dst, r);
            }
            Inst::SAndN2B64 { dst, src0, src1 } => {
                let m0 = b.read_mask_lane(*src0);
                let m1 = b.read_mask_lane(*src1);
                let n1 = b.m.emit(blk, Ty::Bool, SsaOp::LogicalNot(m1));
                let r = b.m.emit(blk, Ty::Bool, SsaOp::LogicalAnd(m0, n1));
                b.write_mask_lane(*dst, r);
            }
            Inst::SOrN2B64 { dst, src0, src1 } => {
                let m0 = b.read_mask_lane(*src0);
                let m1 = b.read_mask_lane(*src1);
                let n1 = b.m.emit(blk, Ty::Bool, SsaOp::LogicalNot(m1));
                let r = b.m.emit(blk, Ty::Bool, SsaOp::LogicalOr(m0, n1));
                b.write_mask_lane(*dst, r);
            }
            Inst::SAndSaveExecB64 { dst, src0 } => {
                let old_exec = b.exec_lane;
                b.write_mask_lane(*dst, old_exec);
                let mask = b.read_mask_lane(*src0);
                let new_exec = b.m.emit(blk, Ty::Bool, SsaOp::LogicalAnd(old_exec, mask));
                b.exec_lane = new_exec;
                b.exec_is_all_true = false;
            }
            Inst::SOrSaveExecB64 { dst, src0 } => {
                let old_exec = b.exec_lane;
                b.write_mask_lane(*dst, old_exec);
                let mask = b.read_mask_lane(*src0);
                let new_exec = b.m.emit(blk, Ty::Bool, SsaOp::LogicalOr(old_exec, mask));
                b.exec_lane = new_exec;
                b.exec_is_all_true = false;
            }

            // --- Structured Forward Branches (`OpSelectionMerge` + `OpBranchConditional` + `OpPhi`) ---
            Inst::SCBranchExecz { target_dw }
            | Inst::SCBranchExecnz { target_dw }
            | Inst::SCBranchScc { target_dw, .. }
            | Inst::SCBranchVccz { target_dw }
            | Inst::SCBranchVccnz { target_dw } => {
                if *target_dw <= di.pc {
                    return Err(LowerError::UnsupportedBranch {
                        pc: di.pc,
                        target_dw: *target_dw,
                    });
                }
                // Condition for ENTERING the fallthrough body (not skipping to target_dw):
                let enter_cond = match &di.inst {
                    Inst::SCBranchExecz { .. } => b.exec_lane,
                    Inst::SCBranchExecnz { .. } => b.m.emit(blk, Ty::Bool, SsaOp::LogicalNot(b.exec_lane)),
                    Inst::SCBranchScc { on_one, .. } => {
                        if *on_one {
                            b.m.emit(blk, Ty::Bool, SsaOp::LogicalNot(b.scc))
                        } else {
                            b.scc
                        }
                    }
                    Inst::SCBranchVccz { .. } => b.vcc_lane,
                    Inst::SCBranchVccnz { .. } => b.m.emit(blk, Ty::Bool, SsaOp::LogicalNot(b.vcc_lane)),
                    _ => unreachable!(),
                };

                let body_block = b.m.new_block();
                let merge_block = b.m.new_block();
                b.m.blocks[blk].term = Terminator::BranchCond {
                    cond: enter_cond,
                    true_block: body_block,
                    false_block: merge_block,
                    merge_block,
                };
                open_regions.push(OpenRegion {
                    target_dw: *target_dw,
                    header_block: blk,
                    merge_block,
                    pre_vgpr: b.vgpr.clone(),
                    pre_sgpr: b.sgpr.clone(),
                });
                b.cur_block = body_block;
                // Inside a structured `SCBranchExecz` body, only active lanes reach this block.
                if matches!(di.inst, Inst::SCBranchExecz { .. }) {
                    b.exec_is_all_true = true;
                }
            }
            Inst::SBranch { target_dw } => {
                if *target_dw <= di.pc {
                    return Err(LowerError::UnsupportedBranch {
                        pc: di.pc,
                        target_dw: *target_dw,
                    });
                }
                // Forward unconditional skip: branch directly to a new continuation block.
                let next_blk = b.m.new_block();
                b.m.blocks[blk].term = Terminator::Branch(next_blk);
                b.cur_block = next_blk;
            }

            // --- Vector ALU ---
            Inst::VMovB32 { dst, src0 } => {
                let v = b.read_op_u32(*src0);
                b.write_vgpr(*dst, v);
            }
            Inst::VReadFirstLaneB32 { dst, src } => {
                let v = b.read_vgpr(*src);
                let r = b.m.emit(blk, Ty::U32, SsaOp::WaveReadFirstLane(v));
                b.sgpr.insert(*dst, r);
            }
            Inst::VQuadSwap { dst, src, direction } => {
                let v = b.read_vgpr(*src);
                let r = b.m.emit(blk, Ty::U32, SsaOp::QuadSwap { val: v, direction: *direction });
                b.write_vgpr(*dst, r);
            }
            Inst::VLogF32 { dst, src0 } => {
                let a = b.read_op_u32(*src0);
                let fa = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let fr = b.m.emit(blk, Ty::F32, SsaOp::Log2F32(fa));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VExpF32 { dst, src0 } => {
                let a = b.read_op_u32(*src0);
                let fa = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let fr = b.m.emit(blk, Ty::F32, SsaOp::Exp2F32(fa));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VRcpF32 { dst, src0 } => {
                let a = b.read_op_u32(*src0);
                let fa = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let fr = b.m.emit(blk, Ty::F32, SsaOp::RcpF32(fa));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VRsqF32 { dst, src0 } => {
                let a = b.read_op_u32(*src0);
                let fa = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let fr = b.m.emit(blk, Ty::F32, SsaOp::RsqF32(fa));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VSqrtF32 { dst, src0 } => {
                let a = b.read_op_u32(*src0);
                let fa = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let fr = b.m.emit(blk, Ty::F32, SsaOp::SqrtF32(fa));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VFloorF32 { dst, src0 } => {
                let a = b.read_op_u32(*src0);
                let fa = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let fr = b.m.emit(blk, Ty::F32, SsaOp::FloorF32(fa));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VFractF32 { dst, src0 } => {
                let a = b.read_op_u32(*src0);
                let fa = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let fr = b.m.emit(blk, Ty::F32, SsaOp::FractF32(fa));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VCvtF32U32 { dst, src0 } => {
                let a = b.read_op_u32(*src0);
                let fr = b.m.emit(blk, Ty::F32, SsaOp::CvtF32U32(a));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VCvtU32F32 { dst, src0 } => {
                let a = b.read_op_u32(*src0);
                let fa = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let r = b.m.emit(blk, Ty::U32, SsaOp::CvtU32F32(fa));
                b.write_vgpr(*dst, r);
            }
            Inst::VCvtF32I32 { dst, src0 } => {
                let a = b.read_op_u32(*src0);
                let fr = b.m.emit(blk, Ty::F32, SsaOp::CvtF32I32(a));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VCvtI32F32 { dst, src0 } => {
                let a = b.read_op_u32(*src0);
                let fa = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(a));
                let r = b.m.emit(blk, Ty::U32, SsaOp::CvtI32F32(fa));
                b.write_vgpr(*dst, r);
            }
            Inst::VAddU32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let r = b.m.emit(blk, Ty::U32, SsaOp::AddU32(x, y));
                b.write_vgpr(*dst, r);
            }
            Inst::VSubU32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let r = b.m.emit(blk, Ty::U32, SsaOp::SubU32(x, y));
                b.write_vgpr(*dst, r);
            }
            Inst::VAndB32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let r = b.m.emit(blk, Ty::U32, SsaOp::AndU32(x, y));
                b.write_vgpr(*dst, r);
            }
            Inst::VOrB32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let r = b.m.emit(blk, Ty::U32, SsaOp::OrU32(x, y));
                b.write_vgpr(*dst, r);
            }
            Inst::VXorB32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let r = b.m.emit(blk, Ty::U32, SsaOp::XorU32(x, y));
                b.write_vgpr(*dst, r);
            }
            Inst::VLshlRevB32 { dst, shift, src } => {
                let s = b.read_op_u32(*shift);
                let v = b.read_op_u32(*src);
                let r = b.m.emit(blk, Ty::U32, SsaOp::ShlU32(v, s));
                b.write_vgpr(*dst, r);
            }
            Inst::VLshrRevB32 { dst, shift, src } => {
                let s = b.read_op_u32(*shift);
                let v = b.read_op_u32(*src);
                let r = b.m.emit(blk, Ty::U32, SsaOp::LshrU32(v, s));
                b.write_vgpr(*dst, r);
            }
            Inst::VAshrRevI32 { dst, shift, src } => {
                let s = b.read_op_u32(*shift);
                let v = b.read_op_u32(*src);
                let r = b.m.emit(blk, Ty::U32, SsaOp::AshrI32(v, s));
                b.write_vgpr(*dst, r);
            }
            Inst::VLshlAddU32 { dst, src0, shift, src2 } => {
                let v0 = b.read_op_u32(*src0);
                let sh = b.read_op_u32(*shift);
                let v2 = b.read_op_u32(*src2);
                let shifted = b.m.emit(blk, Ty::U32, SsaOp::ShlU32(v0, sh));
                let r = b.m.emit(blk, Ty::U32, SsaOp::AddU32(shifted, v2));
                b.write_vgpr(*dst, r);
            }
            Inst::VBfeU32 { dst, src0, offset, width } => {
                let v0 = b.read_op_u32(*src0);
                let off = b.read_op_u32(*offset);
                let w = b.read_op_u32(*width);
                let r = b.m.emit(blk, Ty::U32, SsaOp::BfeU32 { src: v0, offset: off, width: w });
                b.write_vgpr(*dst, r);
            }
            Inst::VAddF32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let fx = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(x));
                let fy = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(y));
                let fr = b.m.emit(blk, Ty::F32, SsaOp::AddF32(fx, fy));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VSubF32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let fx = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(x));
                let fy = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(y));
                let fr = b.m.emit(blk, Ty::F32, SsaOp::SubF32(fx, fy));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VMulF32 { dst, src0, src1, legacy } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let fx = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(x));
                let fy = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(y));
                let fr = if *legacy {
                    b.m.emit(blk, Ty::F32, SsaOp::MulLegacyF32(fx, fy))
                } else {
                    b.m.emit(blk, Ty::F32, SsaOp::MulF32(fx, fy))
                };
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VMinF32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let fx = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(x));
                let fy = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(y));
                let fr = b.m.emit(blk, Ty::F32, SsaOp::MinF32(fx, fy));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VMaxF32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let fx = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(x));
                let fy = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(y));
                let fr = b.m.emit(blk, Ty::F32, SsaOp::MaxF32(fx, fy));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VFmacF32 { dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let c = b.read_vgpr(*dst);
                let fx = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(x));
                let fy = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(y));
                let fc = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(c));
                let fr = b.m.emit(blk, Ty::F32, SsaOp::FmaF32(fx, fy, fc));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VFmaF32 { dst, src0, src1, src2, .. } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let z = b.read_op_u32(*src2);
                let fx = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(x));
                let fy = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(y));
                let fz = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(z));
                let fr = b.m.emit(blk, Ty::F32, SsaOp::FmaF32(fx, fy, fz));
                let r = b.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(fr));
                b.write_vgpr(*dst, r);
            }
            Inst::VCmp { op, ty, dst, src0, src1 } => {
                let x = b.read_op_u32(*src0);
                let y = b.read_op_u32(*src1);
                let (lhs, rhs) = if *ty == CmpType::F32 {
                    (
                        b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(x)),
                        b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(y)),
                    )
                } else {
                    (x, y)
                };
                let cmp = b.m.emit(blk, Ty::Bool, SsaOp::Cmp { op: *op, ty: *ty, a: lhs, b: rhs });
                let dst_reg = dst.unwrap_or(106);
                b.write_mask_lane(dst_reg, cmp);
                // Wire `wave.rs` directly into lowering: if whole-live, emit WaveBallot64 now.
                if wave.whole_live_pcs.contains(&di.pc) && dst_reg == 106 {
                    let ballot = b.m.emit(blk, Ty::U64, SsaOp::WaveBallot64(cmp));
                    b.vcc_ballot64 = Some(ballot);
                }
            }
            Inst::VCndMaskB32 { dst, src0, src1, mask_src } => {
                let f_val = b.read_op_u32(*src0);
                let t_val = b.read_op_u32(*src1);
                let cond = mask_src
                    .map(|m_op| b.read_mask_lane(m_op))
                    .unwrap_or(b.vcc_lane);
                let r = b.m.emit(blk, Ty::U32, SsaOp::SelectU32 { cond, t: t_val, f: f_val });
                b.write_vgpr(*dst, r);
            }

            // --- LDS Shared Memory (DS) ---
            Inst::DsWriteB32 { addr, data, offset } => {
                let base = b.read_vgpr(*addr);
                let ea = if *offset == 0 {
                    base
                } else {
                    let off = b.m.emit(blk, Ty::U32, SsaOp::ConstU32(*offset as u32));
                    b.m.emit(blk, Ty::U32, SsaOp::AddU32(base, off))
                };
                let val = b.read_vgpr(*data);
                b.m.emit(blk, Ty::U32, SsaOp::LdsStoreU32 { byte_addr: ea, val });
            }
            Inst::DsReadB32 { dst, addr, offset } => {
                let base = b.read_vgpr(*addr);
                let ea = if *offset == 0 {
                    base
                } else {
                    let off = b.m.emit(blk, Ty::U32, SsaOp::ConstU32(*offset as u32));
                    b.m.emit(blk, Ty::U32, SsaOp::AddU32(base, off))
                };
                let r = b.m.emit(blk, Ty::U32, SsaOp::LdsLoadU32 { byte_addr: ea });
                b.write_vgpr(*dst, r);
            }
            Inst::DsWriteB64 { addr, data, offset } => {
                let base = b.read_vgpr(*addr);
                for k in 0..2u32 {
                    let off = b.m.emit(blk, Ty::U32, SsaOp::ConstU32(*offset as u32 + k * 4));
                    let ea = b.m.emit(blk, Ty::U32, SsaOp::AddU32(base, off));
                    let val = b.read_vgpr(*data + k as u8);
                    b.m.emit(blk, Ty::U32, SsaOp::LdsStoreU32 { byte_addr: ea, val });
                }
            }
            Inst::DsReadB64 { dst, addr, offset } => {
                let base = b.read_vgpr(*addr);
                for k in 0..2u32 {
                    let off = b.m.emit(blk, Ty::U32, SsaOp::ConstU32(*offset as u32 + k * 4));
                    let ea = b.m.emit(blk, Ty::U32, SsaOp::AddU32(base, off));
                    let r = b.m.emit(blk, Ty::U32, SsaOp::LdsLoadU32 { byte_addr: ea });
                    b.write_vgpr(*dst + k as u8, r);
                }
            }
            Inst::DsAddU32 { .. } => {
                return Err(LowerError::Unsupported { pc: di.pc, desc: "ds_add_u32 atomic" });
            }

            // --- Global / Buffer / Scalar Memory ---
            Inst::GlobalLoad { dst, count, vaddr, saddr, offset } => {
                let base_addr = b.build_bda_addr(*vaddr, *saddr);
                let loaded = b.m.emit(
                    blk,
                    if *count == 4 { Ty::U32x4 } else { Ty::U32 },
                    SsaOp::BdaLoad {
                        addr: base_addr,
                        byte_offset: (*offset).max(0) as u32,
                        count: *count,
                    },
                );
                if *count == 1 {
                    b.write_vgpr(*dst, loaded);
                } else {
                    for k in 0..*count {
                        let lane = b.m.emit(blk, Ty::U32, SsaOp::ExtractLane(loaded, k));
                        b.write_vgpr(*dst + k, lane);
                    }
                }
            }
            Inst::GlobalStore { vaddr, vdata, count, saddr, offset } => {
                let base_addr = b.build_bda_addr(*vaddr, *saddr);
                let mut vals = Vec::with_capacity(*count as usize);
                for k in 0..*count {
                    vals.push(b.read_vgpr(*vdata + k));
                }
                b.m.emit(
                    blk,
                    Ty::U32,
                    SsaOp::BdaStore {
                        addr: base_addr,
                        byte_offset: (*offset).max(0) as u32,
                        values: vals,
                    },
                );
            }
            Inst::BufferLoad { dst, count, vaddr, srsrc, soffset, offset, offen, .. } => {
                let addr = b.build_mubuf_addr(*srsrc, *vaddr, *soffset, *offset, *offen);
                let loaded = b.m.emit(
                    blk,
                    if *count == 4 { Ty::U32x4 } else { Ty::U32 },
                    SsaOp::BdaLoad { addr, byte_offset: 0, count: *count },
                );
                if *count == 1 {
                    b.write_vgpr(*dst, loaded);
                } else {
                    for k in 0..*count {
                        let lane = b.m.emit(blk, Ty::U32, SsaOp::ExtractLane(loaded, k));
                        b.write_vgpr(*dst + k, lane);
                    }
                }
            }
            Inst::BufferStore { vdata, count, vaddr, srsrc, soffset, offset, offen, .. } => {
                let addr = b.build_mubuf_addr(*srsrc, *vaddr, *soffset, *offset, *offen);
                let mut vals = Vec::with_capacity(*count as usize);
                for k in 0..*count {
                    vals.push(b.read_vgpr(*vdata + k));
                }
                b.m.emit(blk, Ty::U32, SsaOp::BdaStore { addr, byte_offset: 0, values: vals });
            }
            Inst::SLoadDword { dst, count, sbase, offset } => {
                let lo = b.read_sgpr(*sbase);
                let hi = b.read_sgpr(*sbase + 1);
                let addr = b.m.emit(blk, Ty::U64, SsaOp::PackU64 { lo, hi });
                if *count <= 4 {
                    let loaded = b.m.emit(
                        blk,
                        if *count == 4 { Ty::U32x4 } else { Ty::U32 },
                        SsaOp::BdaLoad {
                            addr,
                            byte_offset: (*offset).max(0) as u32,
                            count: *count,
                        },
                    );
                    if *count == 1 {
                        b.sgpr.insert(*dst, loaded);
                    } else {
                        for k in 0..*count {
                            let lane = b.m.emit(blk, Ty::U32, SsaOp::ExtractLane(loaded, k));
                            b.sgpr.insert(*dst + k, lane);
                        }
                    }
                } else {
                    for chunk in 0..(*count / 4) {
                        let loaded = b.m.emit(
                            blk,
                            Ty::U32x4,
                            SsaOp::BdaLoad {
                                addr,
                                byte_offset: (*offset).max(0) as u32 + (chunk as u32) * 16,
                                count: 4,
                            },
                        );
                        for k in 0..4u8 {
                            let lane = b.m.emit(blk, Ty::U32, SsaOp::ExtractLane(loaded, k));
                            b.sgpr.insert(*dst + chunk * 4 + k, lane);
                        }
                    }
                }
            }

            // --- Inline BVH Ray-Box32 Intersection ---
            Inst::ImageBvh64IntersectRay { dst, addr_vgpr, .. } => {
                // RDNA2 BVH64 VGPR layout:
                // v[a+0..a+1] = node_ptr (u64), v[a+2] = t_max, v[a+3..a+5] = origin,
                // v[a+6..a+8] = dir, v[a+9..a+11] = inv_dir.
                let lo = b.read_vgpr(*addr_vgpr);
                let hi = b.read_vgpr(*addr_vgpr + 1);
                let node_addr = b.m.emit(blk, Ty::U64, SsaOp::PackU64 { lo, hi });
                let t_max_u = b.read_vgpr(*addr_vgpr + 2);
                let ox_u = b.read_vgpr(*addr_vgpr + 3);
                let oy_u = b.read_vgpr(*addr_vgpr + 4);
                let oz_u = b.read_vgpr(*addr_vgpr + 5);
                let idx_u = b.read_vgpr(*addr_vgpr + 9);
                let idy_u = b.read_vgpr(*addr_vgpr + 10);
                let idz_u = b.read_vgpr(*addr_vgpr + 11);

                let t_max = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(t_max_u));
                let ox = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(ox_u));
                let oy = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(oy_u));
                let oz = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(oz_u));
                let inv_dx = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(idx_u));
                let inv_dy = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(idy_u));
                let inv_dz = b.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(idz_u));

                let hits = b.m.emit(
                    blk,
                    Ty::U32x4,
                    SsaOp::Bvh64IntersectBox32 {
                        node_addr,
                        t_max,
                        ox,
                        oy,
                        oz,
                        inv_dx,
                        inv_dy,
                        inv_dz,
                    },
                );
                for k in 0..4u8 {
                    let lane = b.m.emit(blk, Ty::U32, SsaOp::ExtractLane(hits, k));
                    b.write_vgpr(*dst + k, lane);
                }
            }

            // Explicit errors instead of silent drops for graphics-stage / bindless image ops
            Inst::ImageSample { .. } => {
                return Err(LowerError::Unsupported { pc: di.pc, desc: "MIMG image_sample (bindless descriptor lowering pending)" });
            }
            Inst::ImageLoad { .. } => {
                return Err(LowerError::Unsupported { pc: di.pc, desc: "MIMG image_load (bindless descriptor lowering pending)" });
            }
            Inst::ImageStore { .. } => {
                return Err(LowerError::Unsupported { pc: di.pc, desc: "MIMG image_store (bindless descriptor lowering pending)" });
            }
            Inst::Exp { .. } => {
                return Err(LowerError::Unsupported { pc: di.pc, desc: "EXP graphics export (requires VS/PS stage emitter)" });
            }
        }
    }

    // Close any remaining open regions at program end.
    while let Some(region) = open_regions.pop() {
        let then_end_block = b.cur_block;
        b.m.blocks[then_end_block].term = Terminator::Branch(region.merge_block);
        b.cur_block = region.merge_block;
    }

    Ok(b.m)
}
