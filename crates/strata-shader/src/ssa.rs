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
//! - Natural loops (one backward branch per header) become structured loops:
//!   do-while (`s_cbranch_* header` at the bottom) and while (a forward
//!   `s_cbranch_* exit` at the top, `s_branch header` at the bottom). Every
//!   register the loop carries gets a header `OpPhi`; a lane leaves the loop
//!   when its own EXEC bit says so, which is what masking it off for the
//!   remaining iterations amounts to. Breaks out of the middle of a body and
//!   irreducible flow return `UnsupportedBranch`.
//! - Any unsupported instruction returns `Err(LowerError::Unsupported)` instead
//!   of being silently discarded.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use thiserror::Error;

use crate::format::{FormatInfo, NumFormat, PackedFormat, gfx10_format, gfx10_packed_format};
use crate::rdna2::{CmpOp, CmpType, DecodedInst, Inst, InterpOp, Operand, SampleMode};
use crate::resource::MAX_IMAGE_SLOTS;
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
    /// Loop header: `OpLoopMerge merge continue_block`, then into `body`
    /// (while-loops test `cond` first and leave for `merge` when false).
    Loop {
        cond: Option<ValueId>,
        body: usize,
        merge: usize,
        continue_block: usize,
    },
    /// The continue block's back edge: to `header` while `cond` holds (always
    /// when `None`), else to `exit`.
    BackEdge {
        cond: Option<ValueId>,
        header: usize,
        exit: usize,
    },
}

impl Terminator {
    /// The condition value this terminator reads, if any.
    pub fn cond(&self) -> Option<ValueId> {
        match self {
            Terminator::BranchCond { cond, .. } => Some(*cond),
            Terminator::Loop { cond, .. } | Terminator::BackEdge { cond, .. } => *cond,
            _ => None,
        }
    }

    pub fn cond_mut(&mut self) -> Option<&mut ValueId> {
        match self {
            Terminator::BranchCond { cond, .. } => Some(cond),
            Terminator::Loop { cond, .. } | Terminator::BackEdge { cond, .. } => cond.as_mut(),
            _ => None,
        }
    }

    /// Successor blocks.
    pub fn successors(&self) -> Vec<usize> {
        match *self {
            Terminator::Return => vec![],
            Terminator::Branch(t) => vec![t],
            Terminator::BranchCond {
                true_block,
                false_block,
                ..
            } => vec![true_block, false_block],
            Terminator::Loop {
                cond, body, merge, ..
            } => {
                if cond.is_some() {
                    vec![body, merge]
                } else {
                    vec![body]
                }
            }
            Terminator::BackEdge { cond, header, exit } => {
                if cond.is_some() {
                    vec![header, exit]
                } else {
                    vec![header]
                }
            }
        }
    }
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
    Phi {
        incoming: Vec<(usize, ValueId)>,
    },

    // Scalar & vector arithmetic
    BitcastU32ToF32(ValueId),
    BitcastF32ToU32(ValueId),
    PackU64 {
        lo: ValueId,
        hi: ValueId,
    },
    UnpackLoU64(ValueId),
    UnpackHiU64(ValueId),
    AddU32(ValueId, ValueId),
    SubU32(ValueId, ValueId),
    MulU32(ValueId, ValueId),
    AddU64(ValueId, ValueId),
    AndU64(ValueId, ValueId),
    AndU32(ValueId, ValueId),
    OrU32(ValueId, ValueId),
    XorU32(ValueId, ValueId),
    ShlU32(ValueId, ValueId),
    LshrU32(ValueId, ValueId),
    AshrI32(ValueId, ValueId),
    BfeU32 {
        src: ValueId,
        offset: ValueId,
        width: ValueId,
    },
    /// Sign-extending bit-field extract.
    BfeI32 {
        src: ValueId,
        offset: ValueId,
        width: ValueId,
    },
    /// The low 16 bits as an IEEE half, widened (`Ty::F32`).
    HalfToF32(ValueId),
    /// Two floats as halves in one dword (`Ty::U32`): `a` low, `b` high.
    PackHalf2x16(ValueId, ValueId),
    /// Round to nearest even (`Ty::F32`).
    RoundEvenF32(ValueId),

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
    Cmp {
        op: CmpOp,
        ty: CmpType,
        a: ValueId,
        b: ValueId,
    },
    LogicalAnd(ValueId, ValueId),
    LogicalOr(ValueId, ValueId),
    LogicalXor(ValueId, ValueId),
    LogicalNot(ValueId),
    SelectU32 {
        cond: ValueId,
        t: ValueId,
        f: ValueId,
    },

    // Subgroup / Wave64 operations
    /// Subgroup wave broadcast from first active lane (`v_readfirstlane_b32`).
    WaveReadFirstLane(ValueId),
    /// Subgroup-size-agnostic 64-lane ballot (`OpGroupNonUniformBallot` + Workgroup atomic merge).
    WaveBallot64(ValueId),
    /// Whether the predicate holds in any active lane (`OpGroupNonUniformAny`):
    /// the wave-uniform test behind `s_cbranch_vccz/vccnz`.
    WaveAny(ValueId),
    /// Helper-pixel-safe 2x2 quad swap (`OpGroupNonUniformQuadSwap`, direction 0=X, 1=Y, 2=Diag).
    QuadSwap {
        val: ValueId,
        direction: u8,
    },
    /// Every lane of a quad gets lane `lane`'s value (`OpGroupNonUniformQuadBroadcast`).
    QuadBroadcast {
        val: ValueId,
        lane: u8,
    },

    // Workgroup Shared Memory (LDS) & Synchronization
    WorkgroupBarrier,
    LdsLoadU32 {
        byte_addr: ValueId,
    },
    LdsStoreU32 {
        byte_addr: ValueId,
        val: ValueId,
    },
    /// `ds_add_u32` / `ds_add_rtn_u32`: atomic add to LDS, returning the old value.
    LdsAtomicAddU32 {
        byte_addr: ValueId,
        val: ValueId,
    },

    // Direct 64-bit BDA (`PhysicalStorageBuffer`) memory
    BdaLoad {
        addr: ValueId,
        byte_offset: u32,
        count: u8,
    },
    ExtractLane(ValueId, u8),
    BdaStore {
        addr: ValueId,
        byte_offset: u32,
        values: Vec<ValueId>,
    },

    /// `textures[slot]` sampled with `samplers[slot]` at `(u, v)`; the texel's
    /// four components as bits (`Ty::U32x4`). `implicit`: derivatives (pixel
    /// shaders); otherwise `lod` (0 when `None`), plus `bias` if given.
    ImageSample {
        slot: u32,
        u: ValueId,
        v: ValueId,
        lod: Option<ValueId>,
        bias: Option<ValueId>,
        implicit: bool,
    },
    /// `textures[slot]` texel `(x, y)` at level `lod` (0 when `None`), as bits (`Ty::U32x4`).
    ImageFetch {
        slot: u32,
        x: ValueId,
        y: ValueId,
        lod: Option<ValueId>,
    },
    /// Writes four component bit patterns to `storage_images[slot]` at `(x, y)`.
    ImageStore {
        slot: u32,
        x: ValueId,
        y: ValueId,
        values: [ValueId; 4],
    },

    /// A value the stage receives (`Ty::U32` bits).
    StageInput(StageInput),
    /// `exp`: four component bit patterns to an output target (see
    /// [`ExportTarget`]).
    Export {
        target: ExportTarget,
        values: [ValueId; 4],
    },

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
    /// Textures / samplers the module indexes (`textures[0..n]`).
    pub texture_slots: u32,
    /// Storage images the module writes (`storage_images[0..n]`).
    pub storage_slots: u32,
    pub stage: Stage,
    /// Pixel-shader parameters read: location -> flat.
    pub inputs: BTreeMap<u8, bool>,
    pub require_full_quads: bool,
}

/// Values graphics stages receive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageInput {
    /// Vertex shaders' `v0` (`VertexIndex`).
    VertexIndex,
    /// Vertex shaders' `v3` (`InstanceIndex`).
    InstanceIndex,
    /// Pixel shaders: component `chan` (0=x, 1=y, 2=z, 3=w) of `BuiltIn::FragCoord`.
    FragCoord(u8),
    /// Pixel shaders: `BuiltIn::FrontFacing` (1u if front-facing, 0u otherwise).
    FrontFacing,
    /// Pixel shaders: component `chan` of parameter `loc` (`Location loc`),
    /// interpolated unless `flat`.
    Attr { loc: u8, chan: u8, flat: bool },
}

/// Where an `exp` writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExportTarget {
    /// `mrt0..7`: colour output `Location n`.
    Mrt(u8),
    /// `mrtz`: `FragDepth` (first component).
    Depth,
    /// `pos0`: `Position`.
    Position,
    /// `param0..31`: vertex output `Location n`.
    Param(u8),
}

/// Options for building a module beyond its stage.
#[derive(Debug, Clone, Copy, Default)]
pub struct BuildOptions {
    pub stage: Stage,
    /// First texture/storage slot this module's images take (a pixel
    /// shader's come after its vertex shader's: they share set 0).
    pub texture_slot_base: u32,
    pub storage_slot_base: u32,
    /// Pixel shaders with quad operations run whole quads (helper lanes take
    /// part): `RequireFullQuadsKHR`, when the device has quad control.
    pub require_full_quads: bool,
    /// Optional pixel-shader hardware input VGPR layout (`SPI_PS_INPUT_ENA`)
    /// mapping `pos_x`/`pos_y`/`pos_z`/`pos_w` and `front_face` VGPRs to
    /// `BuiltIn::FragCoord` / `BuiltIn::FrontFacing`.
    pub ps_input_vgprs: Option<crate::geometry::PsInputVgprs>,
}

/// The pipeline stage a module is built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Stage {
    #[default]
    Compute,
    Vertex,
    Pixel,
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
            texture_slots: 0,
            storage_slots: 0,
            stage: Stage::Compute,
            inputs: BTreeMap::new(),
            require_full_quads: false,
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
        if matches!(
            op,
            SsaOp::LdsLoadU32 { .. } | SsaOp::LdsStoreU32 { .. } | SsaOp::LdsAtomicAddU32 { .. }
        ) {
            self.uses_lds = true;
        }
        if matches!(op, SsaOp::WaveBallot64(_)) {
            self.uses_wave64_ballot = true;
        }
        let id = ValueId(self.next_id);
        self.next_id += 1;
        self.blocks[block].insts.push(SsaInst {
            id,
            ty,
            op,
            dead: false,
        });
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

/// A register a loop header may need a phi for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoopReg {
    V(u8),
    S(u8),
    Mask(u8),
    Exec,
    Vcc,
    Scc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoopKind {
    /// Conditional back edge at the bottom.
    DoWhile,
    /// Forward exit test at the top (`test_pc`), unconditional back edge.
    While { test_pc: usize },
}

#[derive(Debug, Clone, Copy)]
struct LoopInfo {
    header_pc: usize,
    latch_pc: usize,
    kind: LoopKind,
}

struct OpenLoop {
    info: LoopInfo,
    header_block: usize,
    merge_block: usize,
    /// `(register, phi value)`; the phi's second incoming is patched at the latch.
    phis: Vec<(LoopReg, ValueId)>,
    regions_depth: usize,
}

fn is_branch(inst: &Inst) -> Option<usize> {
    match inst {
        Inst::SCBranchExecz { target_dw }
        | Inst::SCBranchExecnz { target_dw }
        | Inst::SCBranchScc { target_dw, .. }
        | Inst::SCBranchVccz { target_dw }
        | Inst::SCBranchVccnz { target_dw }
        | Inst::SBranch { target_dw } => Some(*target_dw),
        _ => None,
    }
}

/// Natural loops of the program, by header pc. One back edge per header,
/// properly nested; anything else is irreducible here.
fn find_loops(program: &[DecodedInst]) -> Result<HashMap<usize, LoopInfo>, LowerError> {
    let index_of: HashMap<usize, usize> =
        program.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
    let mut loops: HashMap<usize, LoopInfo> = HashMap::new();
    for (i, di) in program.iter().enumerate() {
        let Some(target) = is_branch(&di.inst) else {
            continue;
        };
        if target > di.pc {
            continue;
        }
        let bad = || LowerError::UnsupportedBranch {
            pc: di.pc,
            target_dw: target,
        };
        let &header_idx = index_of.get(&target).ok_or_else(bad)?;
        if loops.contains_key(&target) {
            return Err(bad()); // a second back edge (`continue`)
        }
        let kind = if matches!(di.inst, Inst::SBranch { .. }) {
            // A while-loop: the header must test and leave for just past the back edge.
            let exit_pc = program.get(i + 1).map_or(di.pc + 1, |n| n.pc);
            let test = &program[header_idx];
            match is_branch(&test.inst) {
                Some(t) if t == exit_pc && !matches!(test.inst, Inst::SBranch { .. }) => {
                    LoopKind::While { test_pc: test.pc }
                }
                _ => return Err(bad()),
            }
        } else {
            LoopKind::DoWhile
        };
        loops.insert(
            target,
            LoopInfo {
                header_pc: target,
                latch_pc: di.pc,
                kind,
            },
        );
    }
    // Loops must nest: no two intervals may partially overlap.
    let spans: Vec<(usize, usize)> = loops.values().map(|l| (l.header_pc, l.latch_pc)).collect();
    for &(a0, a1) in &spans {
        for &(b0, b1) in &spans {
            if a0 < b0 && b0 <= a1 && a1 < b1 {
                return Err(LowerError::UnsupportedBranch {
                    pc: a1,
                    target_dw: a0,
                });
            }
        }
    }
    Ok(loops)
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
    loops: HashMap<usize, LoopInfo>,
    open_loops: Vec<OpenLoop>,
    /// Registers each loop header must phi beyond those live at its entry
    /// (from an earlier attempt); `missing` collects the ones found lacking.
    extra_phis: HashMap<usize, BTreeSet<LoopReg>>,
    missing: HashMap<usize, BTreeSet<LoopReg>>,
}

impl Builder {
    fn reg_value(&mut self, r: LoopReg) -> ValueId {
        match r {
            LoopReg::V(v) => self.read_vgpr(v),
            LoopReg::S(s) => self.read_sgpr(s),
            LoopReg::Mask(s) => match self.mask_lanes.get(&s) {
                Some(&v) => v,
                None => {
                    let v = self.m.emit(0, Ty::Bool, SsaOp::ConstBool(true));
                    self.mask_lanes.insert(s, v);
                    v
                }
            },
            LoopReg::Exec => self.exec_lane,
            LoopReg::Vcc => self.vcc_lane,
            LoopReg::Scc => self.scc,
        }
    }

    fn set_reg(&mut self, r: LoopReg, v: ValueId) {
        match r {
            LoopReg::V(x) => {
                self.vgpr.insert(x, v);
            }
            LoopReg::S(x) => {
                self.sgpr.insert(x, v);
            }
            LoopReg::Mask(x) => {
                self.mask_lanes.insert(x, v);
            }
            LoopReg::Exec => self.exec_lane = v,
            LoopReg::Vcc => {
                self.vcc_lane = v;
                self.vcc_ballot64 = None;
            }
            LoopReg::Scc => self.scc = v,
        }
    }

    fn known_regs(&self) -> BTreeSet<LoopReg> {
        let mut regs: BTreeSet<LoopReg> = [LoopReg::Exec, LoopReg::Vcc, LoopReg::Scc].into();
        regs.extend(self.vgpr.keys().map(|&v| LoopReg::V(v)));
        regs.extend(self.sgpr.keys().map(|&s| LoopReg::S(s)));
        regs.extend(self.mask_lanes.keys().map(|&s| LoopReg::Mask(s)));
        regs
    }

    fn reg_ty(r: LoopReg) -> Ty {
        match r {
            LoopReg::V(_) | LoopReg::S(_) => Ty::U32,
            LoopReg::Mask(_) | LoopReg::Exec | LoopReg::Vcc | LoopReg::Scc => Ty::Bool,
        }
    }

    /// Wave-uniform VCC test: whether any active lane has its VCC bit set.
    fn vcc_any(&mut self) -> ValueId {
        let blk = self.cur_block;
        self.m.emit(blk, Ty::Bool, SsaOp::WaveAny(self.vcc_lane))
    }

    /// Condition under which a conditional branch is *not* taken (the lane
    /// falls through). `None` for an unconditional branch.
    fn fallthrough_cond(&mut self, inst: &Inst) -> Option<ValueId> {
        let blk = self.cur_block;
        Some(match inst {
            Inst::SCBranchExecz { .. } => self.exec_lane,
            Inst::SCBranchExecnz { .. } => {
                self.m
                    .emit(blk, Ty::Bool, SsaOp::LogicalNot(self.exec_lane))
            }
            Inst::SCBranchScc { on_one, .. } => {
                if *on_one {
                    self.m.emit(blk, Ty::Bool, SsaOp::LogicalNot(self.scc))
                } else {
                    self.scc
                }
            }
            Inst::SCBranchVccz { .. } => self.vcc_any(),
            Inst::SCBranchVccnz { .. } => {
                let any = self.vcc_any();
                self.m.emit(blk, Ty::Bool, SsaOp::LogicalNot(any))
            }
            _ => return None,
        })
    }

    /// Opens the loop whose header is at the current pc.
    fn start_loop(&mut self, info: LoopInfo, regions_depth: usize, test: Option<&Inst>) {
        let preheader = self.cur_block;
        let header = self.m.new_block();
        let body = self.m.new_block();
        let merge = self.m.new_block();
        self.m.blocks[preheader].term = Terminator::Branch(header);

        let mut regs = self.known_regs();
        if let Some(extra) = self.extra_phis.get(&info.header_pc) {
            regs.extend(extra.iter().copied());
        }
        let mut phis = Vec::with_capacity(regs.len());
        for r in regs {
            let before = self.reg_value(r);
            let phi = self.m.emit(
                header,
                Self::reg_ty(r),
                SsaOp::Phi {
                    incoming: vec![(preheader, before), (preheader, before)],
                },
            );
            self.set_reg(r, phi);
            phis.push((r, phi));
        }
        self.exec_is_all_true = false;
        self.vcc_ballot64 = None;
        self.cur_block = header;
        let cond = test.and_then(|t| self.fallthrough_cond(t));
        self.m.blocks[header].term = Terminator::Loop {
            cond,
            body,
            merge,
            continue_block: usize::MAX,
        };
        self.cur_block = body;
        self.open_loops.push(OpenLoop {
            info,
            header_block: header,
            merge_block: merge,
            phis,
            regions_depth,
        });
    }

    /// Closes the innermost loop at its back edge `inst`.
    fn close_loop(
        &mut self,
        pc: usize,
        inst: &Inst,
        regions_depth: usize,
    ) -> Result<(), LowerError> {
        let target = is_branch(inst).unwrap_or(pc);
        let bad = LowerError::UnsupportedBranch {
            pc,
            target_dw: target,
        };
        let Some(open) = self.open_loops.pop() else {
            return Err(bad);
        };
        if open.info.latch_pc != pc || regions_depth != open.regions_depth {
            // A region opened inside the body is still open: a break.
            return Err(bad);
        }
        let blk = self.cur_block;
        let cond = match open.info.kind {
            LoopKind::DoWhile => {
                if matches!(inst, Inst::SCBranchExecz { .. }) {
                    return Err(bad); // "loop while no lane is active"
                }
                let stay = self.fallthrough_cond(inst).ok_or(bad)?;
                Some(self.m.emit(blk, Ty::Bool, SsaOp::LogicalNot(stay)))
            }
            LoopKind::While { .. } => None,
        };
        let latch = self.m.new_block();
        self.m.blocks[blk].term = Terminator::Branch(latch);
        self.m.blocks[latch].term = Terminator::BackEdge {
            cond,
            header: open.header_block,
            exit: open.merge_block,
        };
        if let Terminator::Loop { continue_block, .. } = &mut self.m.blocks[open.header_block].term
        {
            *continue_block = latch;
        }

        // Patch each phi's back-edge incoming; note registers the body
        // defined that the header did not phi (the caller retries with them).
        let phi_regs: BTreeSet<LoopReg> = open.phis.iter().map(|(r, _)| *r).collect();
        for &(r, phi) in &open.phis {
            let now = self.reg_value(r);
            if let Some(inst) = self.m.blocks[open.header_block]
                .insts
                .iter_mut()
                .find(|i| i.id == phi)
                && let SsaOp::Phi { incoming } = &mut inst.op
            {
                incoming[1] = (latch, now);
            }
        }
        let lacking: BTreeSet<LoopReg> = self.known_regs().difference(&phi_regs).copied().collect();
        if !lacking.is_empty() {
            self.missing
                .entry(open.info.header_pc)
                .or_default()
                .extend(lacking);
        }

        self.cur_block = open.merge_block;
        if matches!(open.info.kind, LoopKind::While { .. }) {
            // The merge is reached from the header: its values are the phis.
            for &(r, phi) in &open.phis {
                self.set_reg(r, phi);
            }
        }
        self.exec_is_all_true = false;
        self.vcc_ballot64 = None;
        Ok(())
    }

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
                self.m.emit(
                    b,
                    Ty::U32,
                    SsaOp::SelectU32 {
                        cond: self.scc,
                        t: one,
                        f: zero,
                    },
                )
            }
            Operand::M0 => self.m.emit(b, Ty::U32, SsaOp::ConstU32(0)),
        }
    }

    fn ensure_vcc_ballot64(&mut self) -> ValueId {
        if let Some(id) = self.vcc_ballot64 {
            return id;
        }
        let id = self
            .m
            .emit(self.cur_block, Ty::U64, SsaOp::WaveBallot64(self.vcc_lane));
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
            let base = self
                .m
                .emit(b, Ty::U64, SsaOp::PackU64 { lo: s_lo, hi: s_hi });
            let zero = self.m.emit(b, Ty::U32, SsaOp::ConstU32(0));
            let off64 = self
                .m
                .emit(b, Ty::U64, SsaOp::PackU64 { lo: v_lo, hi: zero });
            self.m.emit(b, Ty::U64, SsaOp::AddU64(base, off64))
        } else {
            let v_hi = self.read_vgpr(vaddr + 1);
            self.m
                .emit(b, Ty::U64, SsaOp::PackU64 { lo: v_lo, hi: v_hi })
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_mubuf_addr(
        &mut self,
        srsrc: u8,
        vaddr: u8,
        soffset: Operand,
        offset: u16,
        offen: bool,
        idxen: bool,
    ) -> ValueId {
        let b = self.cur_block;
        // RDNA2 V# buffer descriptor: dword 0 = base[31:0], dword 1[15:0] = base[47:32],
        // dword 1[29:16] = stride.
        let r0 = self.read_sgpr(srsrc);
        let r1 = self.read_sgpr(srsrc + 1);
        let mask16 = self.m.emit(b, Ty::U32, SsaOp::ConstU32(0xffff));
        let r1_hi = self.m.emit(b, Ty::U32, SsaOp::AndU32(r1, mask16));
        let mut addr = self
            .m
            .emit(b, Ty::U64, SsaOp::PackU64 { lo: r0, hi: r1_hi });

        let zero = self.m.emit(b, Ty::U32, SsaOp::ConstU32(0));
        if idxen {
            // idxen: v[vaddr] is a record index (and v[vaddr+1] the offset when offen).
            let index = self.read_vgpr(vaddr);
            let sixteen = self.m.emit(b, Ty::U32, SsaOp::ConstU32(16));
            let width = self.m.emit(b, Ty::U32, SsaOp::ConstU32(14));
            let stride = self.m.emit(
                b,
                Ty::U32,
                SsaOp::BfeU32 {
                    src: r1,
                    offset: sixteen,
                    width,
                },
            );
            let rec = self.m.emit(b, Ty::U32, SsaOp::MulU32(index, stride));
            let rec64 = self
                .m
                .emit(b, Ty::U64, SsaOp::PackU64 { lo: rec, hi: zero });
            addr = self.m.emit(b, Ty::U64, SsaOp::AddU64(addr, rec64));
        }
        if offen {
            let v_off = self.read_vgpr(if idxen { vaddr + 1 } else { vaddr });
            let v_off64 = self.m.emit(
                b,
                Ty::U64,
                SsaOp::PackU64 {
                    lo: v_off,
                    hi: zero,
                },
            );
            addr = self.m.emit(b, Ty::U64, SsaOp::AddU64(addr, v_off64));
        }
        let s_off = self.read_op_u32(soffset);
        let s_off64 = self.m.emit(
            b,
            Ty::U64,
            SsaOp::PackU64 {
                lo: s_off,
                hi: zero,
            },
        );
        addr = self.m.emit(b, Ty::U64, SsaOp::AddU64(addr, s_off64));
        if offset != 0 {
            let imm = self.m.emit(b, Ty::U32, SsaOp::ConstU32(offset as u32));
            let imm64 = self
                .m
                .emit(b, Ty::U64, SsaOp::PackU64 { lo: imm, hi: zero });
            addr = self.m.emit(b, Ty::U64, SsaOp::AddU64(addr, imm64));
        }
        addr
    }
}

impl Builder {
    fn konst(&mut self, v: u32) -> ValueId {
        let blk = self.cur_block;
        self.m.emit(blk, Ty::U32, SsaOp::ConstU32(v))
    }

    fn konst_f(&mut self, v: f32) -> ValueId {
        let blk = self.cur_block;
        self.m.emit(blk, Ty::F32, SsaOp::ConstF32(v.to_bits()))
    }

    fn as_f32(&mut self, u: ValueId) -> ValueId {
        let blk = self.cur_block;
        self.m.emit(blk, Ty::F32, SsaOp::BitcastU32ToF32(u))
    }

    fn as_u32(&mut self, f: ValueId) -> ValueId {
        let blk = self.cur_block;
        self.m.emit(blk, Ty::U32, SsaOp::BitcastF32ToU32(f))
    }

    /// Runs `body` only in lanes EXEC holds: a selection around a side effect.
    /// When EXEC is known all-true (or `body` is in a branch only active
    /// lanes reach) it is emitted inline.
    fn guarded(&mut self, body: impl FnOnce(&mut Self)) {
        if self.exec_is_all_true {
            body(self);
            return;
        }
        let head = self.cur_block;
        let then = self.m.new_block();
        let merge = self.m.new_block();
        self.m.blocks[head].term = Terminator::BranchCond {
            cond: self.exec_lane,
            true_block: then,
            false_block: merge,
            merge_block: merge,
        };
        self.cur_block = then;
        body(self);
        let end = self.cur_block;
        self.m.blocks[end].term = Terminator::Branch(merge);
        self.cur_block = merge;
    }

    /// One component of a typed buffer element at `addr` (a byte address),
    /// converted as `fmt` says, as a dword of bits.
    fn typed_component(&mut self, addr: ValueId, fmt: FormatInfo) -> ValueId {
        let blk = self.cur_block;
        let signed = matches!(
            fmt.num,
            NumFormat::Snorm | NumFormat::Sscaled | NumFormat::Sint
        );
        let raw = if fmt.bits == 32 {
            self.m.emit(
                blk,
                Ty::U32,
                SsaOp::BdaLoad {
                    addr,
                    byte_offset: 0,
                    count: 1,
                },
            )
        } else {
            // The aligned dword holding the component, then its bits.
            let lo_mask = self.konst(0xffff_fffc);
            let hi_mask = self.konst(0xffff_ffff);
            let mask = self.m.emit(
                blk,
                Ty::U64,
                SsaOp::PackU64 {
                    lo: lo_mask,
                    hi: hi_mask,
                },
            );
            let aligned = self.m.emit(blk, Ty::U64, SsaOp::AndU64(addr, mask));
            let dword = self.m.emit(
                blk,
                Ty::U32,
                SsaOp::BdaLoad {
                    addr: aligned,
                    byte_offset: 0,
                    count: 1,
                },
            );
            let lo = self.m.emit(blk, Ty::U32, SsaOp::UnpackLoU64(addr));
            let three = self.konst(3);
            let byte = self.m.emit(blk, Ty::U32, SsaOp::AndU32(lo, three));
            let shift = self.m.emit(blk, Ty::U32, SsaOp::ShlU32(byte, three));
            let width = self.konst(u32::from(fmt.bits));
            if signed {
                self.m.emit(
                    blk,
                    Ty::U32,
                    SsaOp::BfeI32 {
                        src: dword,
                        offset: shift,
                        width,
                    },
                )
            } else {
                self.m.emit(
                    blk,
                    Ty::U32,
                    SsaOp::BfeU32 {
                        src: dword,
                        offset: shift,
                        width,
                    },
                )
            }
        };
        let bits = u32::from(fmt.bits);
        match fmt.num {
            NumFormat::Uint | NumFormat::Sint => raw,
            NumFormat::Float if bits == 32 => raw,
            NumFormat::Float => {
                let f = self.m.emit(blk, Ty::F32, SsaOp::HalfToF32(raw));
                self.as_u32(f)
            }
            NumFormat::Unorm | NumFormat::Uscaled => {
                let mut f = self.m.emit(blk, Ty::F32, SsaOp::CvtF32U32(raw));
                if fmt.num == NumFormat::Unorm {
                    let scale = self.konst_f(1.0 / ((1u64 << bits) - 1) as f32);
                    f = self.m.emit(blk, Ty::F32, SsaOp::MulF32(f, scale));
                }
                self.as_u32(f)
            }
            NumFormat::Snorm | NumFormat::Sscaled => {
                let mut f = self.m.emit(blk, Ty::F32, SsaOp::CvtF32I32(raw));
                if fmt.num == NumFormat::Snorm {
                    let scale = self.konst_f(1.0 / ((1u64 << (bits - 1)) - 1) as f32);
                    f = self.m.emit(blk, Ty::F32, SsaOp::MulF32(f, scale));
                    let neg1 = self.konst_f(-1.0);
                    f = self.m.emit(blk, Ty::F32, SsaOp::MaxF32(f, neg1));
                }
                self.as_u32(f)
            }
        }
    }

    /// `count` dwords of a typed buffer load: the format's components, then
    /// (0, 0, 0, 1) for the ones it lacks.
    fn typed_load(&mut self, addr: ValueId, fmt: FormatInfo, count: u8) -> Vec<ValueId> {
        let blk = self.cur_block;
        let mut out = Vec::with_capacity(count as usize);
        for k in 0..count {
            if k < fmt.comps {
                let off = self.konst(u32::from(k) * u32::from(fmt.bits) / 8);
                let zero = self.konst(0);
                let off64 = self
                    .m
                    .emit(blk, Ty::U64, SsaOp::PackU64 { lo: off, hi: zero });
                let a = self.m.emit(blk, Ty::U64, SsaOp::AddU64(addr, off64));
                out.push(self.typed_component(a, fmt));
            } else if k == 3 {
                out.push(self.konst(if fmt.num.is_float() {
                    1.0f32.to_bits()
                } else {
                    1
                }));
            } else {
                out.push(self.konst(0));
            }
        }
        out
    }

    /// Lower a packed 32-bit `2_10_10_10` / `10_10_10_2` typed buffer load into SSA.
    fn typed_load_packed(
        &mut self,
        addr: ValueId,
        fmt: PackedFormat,
        count: u8,
    ) -> Option<Vec<ValueId>> {
        let (shifts, widths, is_signed, is_norm) = match fmt {
            PackedFormat::Unorm2_10_10_10 => ([0u32, 10, 20, 30], [10u32, 10, 10, 2], false, true),
            PackedFormat::Snorm2_10_10_10 => ([0, 10, 20, 30], [10, 10, 10, 2], true, true),
            PackedFormat::Uint2_10_10_10 => ([0, 10, 20, 30], [10, 10, 10, 2], false, false),
            PackedFormat::Sint2_10_10_10 => ([0, 10, 20, 30], [10, 10, 10, 2], true, false),
            PackedFormat::Unorm10_10_10_2 => ([0, 2, 12, 22], [2, 10, 10, 10], false, true),
            PackedFormat::Uint10_10_10_2 => ([0, 2, 12, 22], [2, 10, 10, 10], false, false),
            PackedFormat::F10_11_11 | PackedFormat::F11_11_10 => return None,
        };
        let blk = self.cur_block;
        let word = self.m.emit(
            blk,
            Ty::U32,
            SsaOp::BdaLoad {
                addr,
                byte_offset: 0,
                count: 1,
            },
        );
        let mut out = Vec::with_capacity(count as usize);
        for k in 0..count as usize {
            if k >= 4 {
                out.push(self.konst(0));
                continue;
            }
            let off = self.konst(shifts[k]);
            let width = self.konst(widths[k]);
            let raw = if is_signed {
                self.m.emit(
                    blk,
                    Ty::U32,
                    SsaOp::BfeI32 {
                        src: word,
                        offset: off,
                        width,
                    },
                )
            } else {
                self.m.emit(
                    blk,
                    Ty::U32,
                    SsaOp::BfeU32 {
                        src: word,
                        offset: off,
                        width,
                    },
                )
            };
            let val = if !is_norm {
                raw
            } else if is_signed {
                let mut f = self.m.emit(blk, Ty::F32, SsaOp::CvtF32I32(raw));
                let scale = self.konst_f(1.0 / ((1u32 << (widths[k] - 1)) - 1) as f32);
                f = self.m.emit(blk, Ty::F32, SsaOp::MulF32(f, scale));
                let neg1 = self.konst_f(-1.0);
                f = self.m.emit(blk, Ty::F32, SsaOp::MaxF32(f, neg1));
                self.as_u32(f)
            } else {
                let mut f = self.m.emit(blk, Ty::F32, SsaOp::CvtF32U32(raw));
                let scale = self.konst_f(1.0 / ((1u32 << widths[k]) - 1) as f32);
                f = self.m.emit(blk, Ty::F32, SsaOp::MulF32(f, scale));
                self.as_u32(f)
            };
            out.push(val);
        }
        Some(out)
    }

    /// Lower a packed 32-bit `2_10_10_10` / `10_10_10_2` typed buffer store into a single u32.
    fn typed_store_packed(&mut self, vals: &[ValueId], fmt: PackedFormat) -> Option<ValueId> {
        let (shifts, widths, is_norm) = match fmt {
            PackedFormat::Unorm2_10_10_10 => ([0u32, 10, 20, 30], [10u32, 10, 10, 2], true),
            PackedFormat::Uint2_10_10_10 | PackedFormat::Sint2_10_10_10 => {
                ([0, 10, 20, 30], [10, 10, 10, 2], false)
            }
            PackedFormat::Unorm10_10_10_2 => ([0, 2, 12, 22], [2, 10, 10, 10], true),
            PackedFormat::Uint10_10_10_2 => ([0, 2, 12, 22], [2, 10, 10, 10], false),
            _ => return None,
        };
        let blk = self.cur_block;
        let mut acc = self.konst(0);
        for k in 0..4usize {
            let v = vals.get(k).copied().unwrap_or_else(|| self.konst(0));
            let mask = self.konst((1u32 << widths[k]) - 1);
            let encoded = if is_norm {
                let f = self.as_f32(v);
                let lo = self.konst_f(0.0);
                let hi = self.konst_f(1.0);
                let c = self.m.emit(blk, Ty::F32, SsaOp::MaxF32(f, lo));
                let c = self.m.emit(blk, Ty::F32, SsaOp::MinF32(c, hi));
                let scale = self.konst_f(((1u32 << widths[k]) - 1) as f32);
                let scaled = self.m.emit(blk, Ty::F32, SsaOp::MulF32(c, scale));
                let half = self.konst_f(0.5);
                let rounded = self.m.emit(blk, Ty::F32, SsaOp::AddF32(scaled, half));
                let u = self.m.emit(blk, Ty::U32, SsaOp::CvtU32F32(rounded));
                self.m.emit(blk, Ty::U32, SsaOp::AndU32(u, mask))
            } else {
                self.m.emit(blk, Ty::U32, SsaOp::AndU32(v, mask))
            };
            let sh = self.konst(shifts[k]);
            let placed = self.m.emit(blk, Ty::U32, SsaOp::ShlU32(encoded, sh));
            acc = self.m.emit(blk, Ty::U32, SsaOp::OrU32(acc, placed));
        }
        Some(acc)
    }

    /// Converts a VGPR value to a component of `fmt` (low `bits` bits).
    fn encode_component(&mut self, val: ValueId, fmt: FormatInfo) -> ValueId {
        let blk = self.cur_block;
        let bits = u32::from(fmt.bits);
        let masked = |b: &mut Self, v: ValueId| {
            if bits == 32 {
                v
            } else {
                let m = b.konst(((1u64 << bits) - 1) as u32);
                let blk = b.cur_block;
                b.m.emit(blk, Ty::U32, SsaOp::AndU32(v, m))
            }
        };
        match fmt.num {
            NumFormat::Uint | NumFormat::Sint | NumFormat::Float => masked(self, val),
            NumFormat::Unorm | NumFormat::Snorm => {
                let f = self.as_f32(val);
                let (lo, scale) = if fmt.num == NumFormat::Unorm {
                    (0.0, ((1u64 << bits) - 1) as f32)
                } else {
                    (-1.0, ((1u64 << (bits - 1)) - 1) as f32)
                };
                let lo = self.konst_f(lo);
                let hi = self.konst_f(1.0);
                let s = self.konst_f(scale);
                let c = self.m.emit(blk, Ty::F32, SsaOp::MaxF32(f, lo));
                let c = self.m.emit(blk, Ty::F32, SsaOp::MinF32(c, hi));
                let c = self.m.emit(blk, Ty::F32, SsaOp::MulF32(c, s));
                let r = self.m.emit(blk, Ty::F32, SsaOp::RoundEvenF32(c));
                let i = if fmt.num == NumFormat::Unorm {
                    self.m.emit(blk, Ty::U32, SsaOp::CvtU32F32(r))
                } else {
                    self.m.emit(blk, Ty::U32, SsaOp::CvtI32F32(r))
                };
                masked(self, i)
            }
            NumFormat::Uscaled => {
                let f = self.as_f32(val);
                let i = self.m.emit(blk, Ty::U32, SsaOp::CvtU32F32(f));
                masked(self, i)
            }
            NumFormat::Sscaled => {
                let f = self.as_f32(val);
                let i = self.m.emit(blk, Ty::U32, SsaOp::CvtI32F32(f));
                masked(self, i)
            }
        }
    }

    /// The dwords a typed store writes: components packed little-end first.
    /// Formats narrower than a dword per element would need a non-atomic
    /// read-modify-write of neighbouring bytes, and are refused.
    fn typed_store_dwords(&mut self, vals: &[ValueId], fmt: FormatInfo) -> Option<Vec<ValueId>> {
        let bytes = fmt.bytes_per_element();
        if !bytes.is_multiple_of(4) {
            return None;
        }
        let blk = self.cur_block;
        let comps: Vec<ValueId> = (0..fmt.comps as usize)
            .map(|k| match vals.get(k) {
                Some(&v) => v,
                None => self.konst(0),
            })
            .collect();
        if fmt.num == NumFormat::Float && fmt.bits == 16 {
            let mut out = Vec::new();
            for pair in comps.chunks(2) {
                let a = self.as_f32(pair[0]);
                let b = match pair.get(1) {
                    Some(&v) => self.as_f32(v),
                    None => self.konst_f(0.0),
                };
                out.push(self.m.emit(blk, Ty::U32, SsaOp::PackHalf2x16(a, b)));
            }
            return Some(out);
        }
        let per_dword = (32 / u32::from(fmt.bits)) as usize;
        let mut out = Vec::new();
        for group in comps.chunks(per_dword) {
            let mut acc: Option<ValueId> = None;
            for (i, &c) in group.iter().enumerate() {
                let enc = self.encode_component(c, fmt);
                let part = if i == 0 {
                    enc
                } else {
                    let sh = self.konst(i as u32 * u32::from(fmt.bits));
                    self.m.emit(blk, Ty::U32, SsaOp::ShlU32(enc, sh))
                };
                acc = Some(match acc {
                    None => part,
                    Some(a) => self.m.emit(blk, Ty::U32, SsaOp::OrU32(a, part)),
                });
            }
            out.extend(acc);
        }
        Some(out)
    }

    fn next_slot(counter: &mut u32, pc: usize) -> Result<u32, LowerError> {
        if *counter >= MAX_IMAGE_SLOTS {
            return Err(LowerError::Unsupported {
                pc,
                desc: "more than 16 images of one kind in a shader",
            });
        }
        *counter += 1;
        Ok(*counter - 1)
    }

    /// Writes the components `dmask` selects from a `U32x4` result to
    /// consecutive VGPRs from `dst`.
    fn write_dmask(&mut self, dst: u8, dmask: u8, texel: ValueId) {
        let blk = self.cur_block;
        let mut reg = dst;
        for comp in 0..4u8 {
            if dmask & (1 << comp) != 0 {
                let lane = self.m.emit(blk, Ty::U32, SsaOp::ExtractLane(texel, comp));
                self.write_vgpr(reg, lane);
                reg += 1;
            }
        }
    }
}

/// Build an SSA module from decoded RDNA2 instructions + wave liveness analysis.
///
/// Loops are built again, at most a few times, when a body turns out to
/// define a register its header did not give a phi.
pub fn build_ssa(
    program: &[DecodedInst],
    wave: &WaveAnalysis,
    workgroup_size: [u32; 3],
) -> Result<SsaModule, LowerError> {
    build_ssa_stage(program, wave, workgroup_size, Stage::Compute)
}

/// As [`build_ssa`], for a given pipeline stage.
pub fn build_ssa_stage(
    program: &[DecodedInst],
    wave: &WaveAnalysis,
    workgroup_size: [u32; 3],
    stage: Stage,
) -> Result<SsaModule, LowerError> {
    build_ssa_with(
        program,
        wave,
        workgroup_size,
        &BuildOptions {
            stage,
            ..Default::default()
        },
    )
}

/// As [`build_ssa`], with [`BuildOptions`].
pub fn build_ssa_with(
    program: &[DecodedInst],
    wave: &WaveAnalysis,
    workgroup_size: [u32; 3],
    opts: &BuildOptions,
) -> Result<SsaModule, LowerError> {
    let stage = opts.stage;
    let loops = find_loops(program)?;
    let mut extra: HashMap<usize, BTreeSet<LoopReg>> = HashMap::new();
    for _ in 0..6 {
        let (mut module, missing) =
            build_ssa_once(program, wave, workgroup_size, opts, &loops, &extra)?;
        module.require_full_quads = opts.require_full_quads && stage == Stage::Pixel;
        if stage != Stage::Compute
            && (module.uses_lds
                || module.uses_wave64_ballot
                || module
                    .blocks
                    .iter()
                    .flat_map(|b| &b.insts)
                    .any(|i| i.op == SsaOp::WorkgroupBarrier))
        {
            return Err(LowerError::Unsupported {
                pc: 0,
                desc: "LDS / barriers / 64-lane ballots in a graphics stage",
            });
        }
        let mut grew = false;
        for (pc, regs) in missing {
            let e = extra.entry(pc).or_default();
            let before = e.len();
            e.extend(regs);
            grew |= e.len() != before;
        }
        if !grew {
            return Ok(module);
        }
    }
    Err(LowerError::Unsupported {
        pc: 0,
        desc: "loop-carried registers did not settle",
    })
}

fn build_ssa_once(
    program: &[DecodedInst],
    wave: &WaveAnalysis,
    workgroup_size: [u32; 3],
    opts: &BuildOptions,
    loops: &HashMap<usize, LoopInfo>,
    extra: &HashMap<usize, BTreeSet<LoopReg>>,
) -> Result<(SsaModule, HashMap<usize, BTreeSet<LoopReg>>), LowerError> {
    let stage = opts.stage;
    let mut m = SsaModule::new(workgroup_size);
    m.stage = stage;
    m.texture_slots = opts.texture_slot_base;
    m.storage_slots = opts.storage_slot_base;
    // Initial VGPRs: compute v0 = thread id; vertex (GFX10 legacy VS)
    // v0 = vertex id, v3 = instance id; pixel shaders' barycentric VGPRs
    // are not modelled (v_interp reads interpolated inputs instead).
    let initial: Vec<(u8, SsaOp)> = match stage {
        Stage::Compute => vec![(0, SsaOp::ThreadIdX)],
        Stage::Vertex => vec![
            (0, SsaOp::StageInput(StageInput::VertexIndex)),
            (3, SsaOp::StageInput(StageInput::InstanceIndex)),
        ],
        Stage::Pixel => {
            let mut v = Vec::new();
            if let Some(ps) = opts.ps_input_vgprs {
                if let Some(r) = ps.pos_x {
                    v.push((r, SsaOp::StageInput(StageInput::FragCoord(0))));
                }
                if let Some(r) = ps.pos_y {
                    v.push((r, SsaOp::StageInput(StageInput::FragCoord(1))));
                }
                if let Some(r) = ps.pos_z {
                    v.push((r, SsaOp::StageInput(StageInput::FragCoord(2))));
                }
                if let Some(r) = ps.pos_w {
                    v.push((r, SsaOp::StageInput(StageInput::FragCoord(3))));
                }
                if let Some(r) = ps.front_face {
                    v.push((r, SsaOp::StageInput(StageInput::FrontFacing)));
                }
            }
            v
        }
    };
    let initial: Vec<(u8, ValueId)> = initial
        .into_iter()
        .map(|(r, op)| (r, m.emit(0, Ty::U32, op)))
        .collect();
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
        loops: loops.clone(),
        open_loops: Vec::new(),
        extra_phis: extra.clone(),
        missing: HashMap::new(),
    };
    for (r, v) in initial {
        b.vgpr.insert(r, v);
    }

    let mut open_regions: Vec<OpenRegion> = Vec::new();

    for di in program {
        // Close any structured forward-branch regions whose target PC we just reached.
        while let Some(top) = open_regions.last() {
            if di.pc >= top.target_dw {
                let region = open_regions.pop().unwrap();
                if b.open_loops
                    .last()
                    .is_some_and(|l| open_regions.len() < l.regions_depth)
                {
                    // A branch from before a loop into its body.
                    return Err(LowerError::UnsupportedBranch {
                        pc: di.pc,
                        target_dw: region.target_dw,
                    });
                }
                let then_end_block = b.cur_block;
                b.m.blocks[then_end_block].term = Terminator::Branch(region.merge_block);
                b.cur_block = region.merge_block;

                // Emit OpPhi in merge_block for any VGPR modified inside the taken branch.
                let mut modified_vgprs: Vec<u8> = b.vgpr.keys().copied().collect();
                modified_vgprs.sort_unstable();
                for vreg in modified_vgprs {
                    let after_val = b.vgpr[&vreg];
                    let before_val = region
                        .pre_vgpr
                        .get(&vreg)
                        .copied()
                        .unwrap_or_else(|| b.m.emit(0, Ty::U32, SsaOp::ConstU32(0)));
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

        if let Some(&info) = b.loops.get(&di.pc) {
            let test = match info.kind {
                LoopKind::While { test_pc } if test_pc == di.pc => Some(&di.inst),
                _ => None,
            };
            b.start_loop(info, open_regions.len(), test);
            if test.is_some() {
                continue; // the exit test became the header's branch
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
                let v =
                    b.m.emit(blk, Ty::U32, SsaOp::ConstU32(*simm16 as i32 as u32));
                b.sgpr.insert(*dst, v);
            }
            Inst::SAddkI32 { dst, simm16 } => {
                let a = b.read_sgpr(*dst);
                let imm =
                    b.m.emit(blk, Ty::U32, SsaOp::ConstU32(*simm16 as i32 as u32));
                let r = b.m.emit(blk, Ty::U32, SsaOp::AddU32(a, imm));
                b.sgpr.insert(*dst, r);
            }
            Inst::SMulkI32 { dst, simm16 } => {
                let a = b.read_sgpr(*dst);
                let imm =
                    b.m.emit(blk, Ty::U32, SsaOp::ConstU32(*simm16 as i32 as u32));
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
                b.scc = b.m.emit(
                    blk,
                    Ty::Bool,
                    SsaOp::Cmp {
                        op: *op,
                        ty: *ty,
                        a: x,
                        b: y,
                    },
                );
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
                    b.close_loop(di.pc, &di.inst, open_regions.len())?;
                    continue;
                }
                // Condition for ENTERING the fallthrough body (not skipping to target_dw):
                let enter_cond = b.fallthrough_cond(&di.inst).expect("conditional branch");
                let blk = b.cur_block;

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
                    b.close_loop(di.pc, &di.inst, open_regions.len())?;
                    continue;
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
            Inst::VQuadBroadcast { dst, src, lane } => {
                let v = b.read_vgpr(*src);
                let r = b.m.emit(
                    blk,
                    Ty::U32,
                    SsaOp::QuadBroadcast {
                        val: v,
                        lane: *lane,
                    },
                );
                b.write_vgpr(*dst, r);
            }
            Inst::VQuadSwap {
                dst,
                src,
                direction,
            } => {
                let v = b.read_vgpr(*src);
                let r = b.m.emit(
                    blk,
                    Ty::U32,
                    SsaOp::QuadSwap {
                        val: v,
                        direction: *direction,
                    },
                );
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
            Inst::VLshlAddU32 {
                dst,
                src0,
                shift,
                src2,
            } => {
                let v0 = b.read_op_u32(*src0);
                let sh = b.read_op_u32(*shift);
                let v2 = b.read_op_u32(*src2);
                let shifted = b.m.emit(blk, Ty::U32, SsaOp::ShlU32(v0, sh));
                let r = b.m.emit(blk, Ty::U32, SsaOp::AddU32(shifted, v2));
                b.write_vgpr(*dst, r);
            }
            Inst::VBfeU32 {
                dst,
                src0,
                offset,
                width,
            } => {
                let v0 = b.read_op_u32(*src0);
                let off = b.read_op_u32(*offset);
                let w = b.read_op_u32(*width);
                let r = b.m.emit(
                    blk,
                    Ty::U32,
                    SsaOp::BfeU32 {
                        src: v0,
                        offset: off,
                        width: w,
                    },
                );
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
            Inst::VMulF32 {
                dst,
                src0,
                src1,
                legacy,
            } => {
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
            Inst::VFmaF32 {
                dst,
                src0,
                src1,
                src2,
                ..
            } => {
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
            Inst::VCmp {
                op,
                ty,
                dst,
                src0,
                src1,
            } => {
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
                let cmp = b.m.emit(
                    blk,
                    Ty::Bool,
                    SsaOp::Cmp {
                        op: *op,
                        ty: *ty,
                        a: lhs,
                        b: rhs,
                    },
                );
                let dst_reg = dst.unwrap_or(106);
                b.write_mask_lane(dst_reg, cmp);
                // Wire `wave.rs` directly into lowering: if whole-live, emit WaveBallot64 now.
                if wave.whole_live_pcs.contains(&di.pc) && dst_reg == 106 {
                    let ballot = b.m.emit(blk, Ty::U64, SsaOp::WaveBallot64(cmp));
                    b.vcc_ballot64 = Some(ballot);
                }
            }
            Inst::VCndMaskB32 {
                dst,
                src0,
                src1,
                mask_src,
            } => {
                let f_val = b.read_op_u32(*src0);
                let t_val = b.read_op_u32(*src1);
                let cond = mask_src
                    .map(|m_op| b.read_mask_lane(m_op))
                    .unwrap_or(b.vcc_lane);
                let r = b.m.emit(
                    blk,
                    Ty::U32,
                    SsaOp::SelectU32 {
                        cond,
                        t: t_val,
                        f: f_val,
                    },
                );
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
                b.guarded(|b| {
                    let blk = b.cur_block;
                    b.m.emit(blk, Ty::U32, SsaOp::LdsStoreU32 { byte_addr: ea, val });
                });
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
                    let off =
                        b.m.emit(blk, Ty::U32, SsaOp::ConstU32(*offset as u32 + k * 4));
                    let ea = b.m.emit(blk, Ty::U32, SsaOp::AddU32(base, off));
                    let val = b.read_vgpr(*data + k as u8);
                    b.guarded(|b| {
                        let blk = b.cur_block;
                        b.m.emit(blk, Ty::U32, SsaOp::LdsStoreU32 { byte_addr: ea, val });
                    });
                }
            }
            Inst::DsReadB64 { dst, addr, offset } => {
                let base = b.read_vgpr(*addr);
                for k in 0..2u32 {
                    let off =
                        b.m.emit(blk, Ty::U32, SsaOp::ConstU32(*offset as u32 + k * 4));
                    let ea = b.m.emit(blk, Ty::U32, SsaOp::AddU32(base, off));
                    let r = b.m.emit(blk, Ty::U32, SsaOp::LdsLoadU32 { byte_addr: ea });
                    b.write_vgpr(*dst + k as u8, r);
                }
            }
            Inst::DsAddU32 { addr, data, offset } => {
                let base = b.read_vgpr(*addr);
                let ea = if *offset == 0 {
                    base
                } else {
                    let off = b.m.emit(blk, Ty::U32, SsaOp::ConstU32(*offset as u32));
                    b.m.emit(blk, Ty::U32, SsaOp::AddU32(base, off))
                };
                let val = b.read_vgpr(*data);
                // Lanes EXEC masks off must not add: their operand becomes 0.
                let add = if b.exec_is_all_true {
                    val
                } else {
                    let zero = b.m.emit(blk, Ty::U32, SsaOp::ConstU32(0));
                    b.m.emit(
                        blk,
                        Ty::U32,
                        SsaOp::SelectU32 {
                            cond: b.exec_lane,
                            t: val,
                            f: zero,
                        },
                    )
                };
                b.m.emit(
                    blk,
                    Ty::U32,
                    SsaOp::LdsAtomicAddU32 {
                        byte_addr: ea,
                        val: add,
                    },
                );
            }

            // --- Global / Buffer / Scalar Memory ---
            Inst::GlobalLoad {
                dst,
                count,
                vaddr,
                saddr,
                offset,
            } => {
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
            Inst::GlobalStore {
                vaddr,
                vdata,
                count,
                saddr,
                offset,
            } => {
                let base_addr = b.build_bda_addr(*vaddr, *saddr);
                let mut vals = Vec::with_capacity(*count as usize);
                for k in 0..*count {
                    vals.push(b.read_vgpr(*vdata + k));
                }
                let byte_offset = (*offset).max(0) as u32;
                b.guarded(|b| {
                    let blk = b.cur_block;
                    b.m.emit(
                        blk,
                        Ty::U32,
                        SsaOp::BdaStore {
                            addr: base_addr,
                            byte_offset,
                            values: vals,
                        },
                    );
                });
            }
            Inst::BufferLoad {
                dst,
                count,
                vaddr,
                srsrc,
                soffset,
                offset,
                offen,
                idxen,
                typed_format: Some(fmt),
            } => {
                let addr = b.build_mubuf_addr(*srsrc, *vaddr, *soffset, *offset, *offen, *idxen);
                let vals = if let Some(info) = gfx10_format(u16::from(*fmt)) {
                    b.typed_load(addr, info, *count)
                } else if let Some(packed) = gfx10_packed_format(u16::from(*fmt))
                    && let Some(vals) = b.typed_load_packed(addr, packed, *count)
                {
                    vals
                } else {
                    return Err(LowerError::Unsupported {
                        pc: di.pc,
                        desc: "tbuffer format (packed / unknown)",
                    });
                };
                for (k, v) in vals.into_iter().enumerate() {
                    b.write_vgpr(*dst + k as u8, v);
                }
            }
            Inst::BufferStore {
                vdata,
                count,
                vaddr,
                srsrc,
                soffset,
                offset,
                offen,
                idxen,
                typed_format: Some(fmt),
            } => {
                let addr = b.build_mubuf_addr(*srsrc, *vaddr, *soffset, *offset, *offen, *idxen);
                let vals: Vec<ValueId> = (0..*count).map(|k| b.read_vgpr(*vdata + k)).collect();
                let dwords = if let Some(info) = gfx10_format(u16::from(*fmt)) {
                    b.typed_store_dwords(&vals, info)
                        .ok_or(LowerError::Unsupported {
                            pc: di.pc,
                            desc: "tbuffer store of a sub-dword element format",
                        })?
                } else if let Some(packed) = gfx10_packed_format(u16::from(*fmt))
                    && let Some(word) = b.typed_store_packed(&vals, packed)
                {
                    vec![word]
                } else {
                    return Err(LowerError::Unsupported {
                        pc: di.pc,
                        desc: "tbuffer format (packed / unknown)",
                    });
                };
                b.guarded(|b| {
                    let blk = b.cur_block;
                    for (k, d) in dwords.into_iter().enumerate() {
                        b.m.emit(
                            blk,
                            Ty::U32,
                            SsaOp::BdaStore {
                                addr,
                                byte_offset: k as u32 * 4,
                                values: vec![d],
                            },
                        );
                    }
                });
            }
            Inst::BufferLoad {
                dst,
                count,
                vaddr,
                srsrc,
                soffset,
                offset,
                offen,
                idxen,
                ..
            } => {
                let addr = b.build_mubuf_addr(*srsrc, *vaddr, *soffset, *offset, *offen, *idxen);
                let loaded = b.m.emit(
                    blk,
                    if *count == 4 { Ty::U32x4 } else { Ty::U32 },
                    SsaOp::BdaLoad {
                        addr,
                        byte_offset: 0,
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
            Inst::BufferStore {
                vdata,
                count,
                vaddr,
                srsrc,
                soffset,
                offset,
                offen,
                idxen,
                ..
            } => {
                let addr = b.build_mubuf_addr(*srsrc, *vaddr, *soffset, *offset, *offen, *idxen);
                let mut vals = Vec::with_capacity(*count as usize);
                for k in 0..*count {
                    vals.push(b.read_vgpr(*vdata + k));
                }
                b.guarded(|b| {
                    let blk = b.cur_block;
                    b.m.emit(
                        blk,
                        Ty::U32,
                        SsaOp::BdaStore {
                            addr,
                            byte_offset: 0,
                            values: vals,
                        },
                    );
                });
            }
            Inst::SLoadDword {
                dst,
                count,
                sbase,
                offset,
            } => {
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

            // --- Images: one descriptor-set slot per instruction, in program
            // order (`resource::image_bindings` numbers them the same way). ---
            Inst::ImageSample { dim, .. }
            | Inst::ImageLoad { dim, .. }
            | Inst::ImageStore { dim, .. }
                if *dim != 1 =>
            {
                return Err(LowerError::Unsupported {
                    pc: di.pc,
                    desc: "MIMG dimension other than 2D",
                });
            }
            Inst::ImageSample {
                dst,
                addr_vgpr,
                dmask,
                mode,
                ..
            } => {
                let slot = Builder::next_slot(&mut b.m.texture_slots, di.pc)?;
                let a = *addr_vgpr;
                let (bias, coord) = match mode {
                    SampleMode::Bias => (Some(b.read_vgpr(a)), a + 1),
                    _ => (None, a),
                };
                let u_bits = b.read_vgpr(coord);
                let v_bits = b.read_vgpr(coord + 1);
                let u = b.as_f32(u_bits);
                let v = b.as_f32(v_bits);
                let lod = match mode {
                    SampleMode::Lod => {
                        let l = b.read_vgpr(coord + 2);
                        Some(b.as_f32(l))
                    }
                    _ => None,
                };
                let bias = bias.map(|x| b.as_f32(x));
                let implicit = matches!(mode, SampleMode::Implicit | SampleMode::Bias)
                    && b.m.stage == Stage::Pixel;
                let blk = b.cur_block;
                let texel = b.m.emit(
                    blk,
                    Ty::U32x4,
                    SsaOp::ImageSample {
                        slot,
                        u,
                        v,
                        lod,
                        bias,
                        implicit,
                    },
                );
                b.write_dmask(*dst, *dmask, texel);
            }
            Inst::ImageLoad {
                dst,
                addr_vgpr,
                dmask,
                mip,
                ..
            } => {
                let slot = Builder::next_slot(&mut b.m.texture_slots, di.pc)?;
                let x = b.read_vgpr(*addr_vgpr);
                let y = b.read_vgpr(*addr_vgpr + 1);
                let lod = mip.then(|| b.read_vgpr(*addr_vgpr + 2));
                let blk = b.cur_block;
                let texel =
                    b.m.emit(blk, Ty::U32x4, SsaOp::ImageFetch { slot, x, y, lod });
                b.write_dmask(*dst, *dmask, texel);
            }
            Inst::ImageStore {
                vdata,
                addr_vgpr,
                dmask,
                mip,
                ..
            } => {
                if *mip {
                    return Err(LowerError::Unsupported {
                        pc: di.pc,
                        desc: "image_store_mip",
                    });
                }
                let slot = Builder::next_slot(&mut b.m.storage_slots, di.pc)?;
                let x = b.read_vgpr(*addr_vgpr);
                let y = b.read_vgpr(*addr_vgpr + 1);
                let mut values = [ValueId(0); 4];
                let mut reg = *vdata;
                for (comp, value) in values.iter_mut().enumerate() {
                    *value = if *dmask & (1 << comp) != 0 {
                        reg += 1;
                        b.read_vgpr(reg - 1)
                    } else {
                        b.konst(0)
                    };
                }
                b.guarded(|b| {
                    let blk = b.cur_block;
                    b.m.emit(blk, Ty::U32, SsaOp::ImageStore { slot, x, y, values });
                });
            }
            Inst::VInterp {
                dst,
                op,
                attr,
                chan,
            } => {
                if b.m.stage != Stage::Pixel {
                    return Err(LowerError::Unsupported {
                        pc: di.pc,
                        desc: "v_interp outside a pixel shader",
                    });
                }
                let flat = *op == InterpOp::Mov;
                if *b.m.inputs.entry(*attr).or_insert(flat) != flat {
                    return Err(LowerError::Unsupported {
                        pc: di.pc,
                        desc: "a parameter read both flat and interpolated",
                    });
                }
                // p1 then p2 build the interpolated value; both write it whole.
                let v = b.m.emit(
                    blk,
                    Ty::U32,
                    SsaOp::StageInput(StageInput::Attr {
                        loc: *attr,
                        chan: *chan,
                        flat,
                    }),
                );
                b.write_vgpr(*dst, v);
            }
            Inst::Exp {
                target,
                en,
                vsrc,
                compr,
                ..
            } => {
                let t = match (b.m.stage, *target) {
                    (Stage::Pixel, t @ 0..=7) => Some(ExportTarget::Mrt(t)),
                    (Stage::Pixel, 8) => Some(ExportTarget::Depth),
                    (Stage::Vertex, 12) => Some(ExportTarget::Position),
                    (Stage::Vertex, t @ 32..=63) => Some(ExportTarget::Param(t - 32)),
                    // null, and pos1..3 (misc vector / clip distances), have no effect here.
                    (Stage::Pixel, 9) | (Stage::Vertex, 13..=15) => None,
                    (Stage::Compute, _) => {
                        return Err(LowerError::Unsupported {
                            pc: di.pc,
                            desc: "EXP in a compute shader",
                        });
                    }
                    _ => {
                        return Err(LowerError::Unsupported {
                            pc: di.pc,
                            desc: "EXP target this stage has no output for",
                        });
                    }
                };
                if let Some(target) = t {
                    let mut values = [ValueId(0); 4];
                    for v in values.iter_mut() {
                        *v = b.konst(0);
                    }
                    if *compr {
                        // Two VGPRs of packed halves: (x, y) and (z, w).
                        for pair in 0..2 {
                            if *en & (0b11 << (pair * 2)) == 0 {
                                continue;
                            }
                            let packed = b.read_vgpr(vsrc[pair]);
                            let sixteen = b.konst(16);
                            let blk = b.cur_block;
                            let high = b.m.emit(blk, Ty::U32, SsaOp::LshrU32(packed, sixteen));
                            for (k, half) in [packed, high].into_iter().enumerate() {
                                let blk = b.cur_block;
                                let f = b.m.emit(blk, Ty::F32, SsaOp::HalfToF32(half));
                                values[pair * 2 + k] = b.as_u32(f);
                            }
                        }
                    } else {
                        for k in 0..4 {
                            if *en & (1 << k) != 0 {
                                values[k] = b.read_vgpr(vsrc[k]);
                            }
                        }
                    }
                    let blk = b.cur_block;
                    b.m.emit(blk, Ty::U32, SsaOp::Export { target, values });
                }
            }
        }
    }

    // Close any remaining open regions at program end.
    while let Some(region) = open_regions.pop() {
        let then_end_block = b.cur_block;
        b.m.blocks[then_end_block].term = Terminator::Branch(region.merge_block);
        b.cur_block = region.merge_block;
    }
    if let Some(open) = b.open_loops.last() {
        return Err(LowerError::UnsupportedBranch {
            pc: open.info.latch_pc,
            target_dw: open.info.header_pc,
        });
    }

    Ok((b.m, b.missing))
}
