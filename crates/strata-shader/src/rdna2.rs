//! Clean-room RDNA2 (GFX10.3 / Xbox Series X|S) instruction decoder & IR.
//!
//! Decodes scalar (SOPP, SOPC, SOP1, SOPK, SOP2, SMEM), vector (VOP1, VOPC,
//! VOP2, VOP3), shared-memory (DS/LDS), buffer (MUBUF, MTBUF, FLAT/GLOBAL),
//! image/ray-tracing (MIMG), and graphics export (EXP) encodings.

use thiserror::Error;

pub use crate::alu::{LaneMod, MaskOp, SaluOp, ValuOp};
use crate::opcodes::{Encoding, lookup_opcode};

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("truncated instruction at dword offset {offset}")]
    Truncated { offset: usize },
    #[error("unsupported RDNA2 encoding 0x{word:08x} at dword offset {offset}")]
    Unsupported { word: u32, offset: usize },
}

/// RDNA2 register / operand reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operand {
    Sgpr(u8),
    Vgpr(u8),
    VccLo,
    VccHi,
    ExecLo,
    ExecHi,
    M0,
    Scc,
    InlineInt(i32),
    InlineFloat(u32), // IEEE-754 bits
    Literal(u32),
}

impl Operand {
    /// Decode a 9-bit RDNA2 source operand field.
    pub fn from_src9(code: u16, literal: Option<u32>) -> Self {
        match code {
            0..=105 => Self::Sgpr(code as u8),
            106 => Self::VccLo,
            107 => Self::VccHi,
            124 => Self::M0,
            126 => Self::ExecLo,
            127 => Self::ExecHi,
            128..=192 => Self::InlineInt((code as i32) - 128),
            193..=208 => Self::InlineInt(192 - (code as i32)),
            240 => Self::InlineFloat(0.5f32.to_bits()),
            241 => Self::InlineFloat((-0.5f32).to_bits()),
            242 => Self::InlineFloat(1.0f32.to_bits()),
            243 => Self::InlineFloat((-1.0f32).to_bits()),
            244 => Self::InlineFloat(2.0f32.to_bits()),
            245 => Self::InlineFloat((-2.0f32).to_bits()),
            246 => Self::InlineFloat(4.0f32.to_bits()),
            247 => Self::InlineFloat((-4.0f32).to_bits()),
            248 => Self::InlineFloat(std::f32::consts::FRAC_1_PI.to_bits()),
            253 => Self::Scc,
            255 => Self::Literal(literal.unwrap_or(0)),
            256..=511 => Self::Vgpr((code - 256) as u8),
            _ => Self::InlineInt(0),
        }
    }

    pub fn is_vgpr(self) -> bool {
        matches!(self, Self::Vgpr(_))
    }

    pub fn is_mask_reg(self) -> bool {
        matches!(
            self,
            Self::VccLo | Self::VccHi | Self::ExecLo | Self::ExecHi
        )
    }
}

/// Comparison predicate for SOPC / VOPC instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq,
    /// Integers: not equal. Floats: ordered not equal (`lg`).
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// Always false / always true.
    F,
    T,
    /// Floats: both ordered / either NaN.
    O,
    U,
    /// Floats: unordered negations (`!(a >= b)` etc., true on NaN).
    Nge,
    Nlg,
    Ngt,
    Nle,
    Neq,
    Nlt,
    /// `v_cmp_class_*`: `src1` is a mask of IEEE classes (bit 0 sNaN,
    /// 1 qNaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal,
    /// 8 +normal, 9 +inf).
    Class,
}

/// Scalar/vector comparison operand type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpType {
    F32,
    U32,
    I32,
    /// Lowered before SPIR-V emission into 32-bit compares.
    I64,
    U64,
    F16,
    I16,
    U16,
}

/// VOP3 source modifiers (negate, absolute value, output clamp).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Vop3Mods {
    pub abs: u8,
    pub neg: u8,
    pub clamp: bool,
    pub omod: u8,
}

/// `v_interp_*` (VINTRP) operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpOp {
    /// First half of a barycentric interpolation (`P0 + i * P10`).
    P1,
    /// Second half, finishing it (`+ j * P20`).
    P2,
    /// A vertex parameter as is (flat).
    Mov,
}

/// How `image_sample*` picks its mip level / mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleMode {
    /// `image_sample`: from screen-space derivatives (pixel shaders).
    Implicit,
    /// `image_sample_l`: explicit LOD after the coordinates.
    Lod,
    /// `image_sample_b`: LOD bias before the coordinates.
    Bias,
    /// `image_sample_lz`: level 0.
    LodZero,
    /// `image_sample_c`: depth/shadow comparison with implicit LOD.
    Compare,
    /// `image_sample_c_l`: depth/shadow comparison with explicit LOD.
    CompareLod,
    /// `image_sample_c_lz`: depth/shadow comparison at level 0.
    CompareLodZero,
    /// `image_gather4`: 4-texel gather.
    Gather,
    /// `image_gather4_lz`: 4-texel gather at level 0.
    GatherLodZero,
    /// `image_sample_d`: explicit derivatives before the coordinates.
    Grad,
    /// `image_sample_c_b`: comparison with an LOD bias.
    CompareBias,
    /// `image_sample_c_d`: comparison with explicit derivatives.
    CompareGrad,
    /// `image_gather4_l`: gather with an explicit LOD (read; Vulkan core
    /// gathers sample the base level).
    GatherLod,
    /// `image_gather4_b`: gather with an LOD bias (read; see `GatherLod`).
    GatherBias,
    /// `image_gather4_c`: depth-comparison gather (`OpImageDrefGather`).
    GatherCompare,
    /// `image_gather4_c_l`: comparison gather with an explicit LOD.
    GatherCompareLod,
    /// `image_gather4_c_b`: comparison gather with an LOD bias.
    GatherCompareBias,
    /// `image_gather4_c_lz`: comparison gather at level 0.
    GatherCompareLodZero,
}

impl SampleMode {
    pub fn is_compare(self) -> bool {
        matches!(
            self,
            SampleMode::Compare
                | SampleMode::CompareLod
                | SampleMode::CompareLodZero
                | SampleMode::CompareBias
                | SampleMode::CompareGrad
                | SampleMode::GatherCompare
                | SampleMode::GatherCompareLod
                | SampleMode::GatherCompareBias
                | SampleMode::GatherCompareLodZero
        )
    }

    /// `image_gather4*`.
    pub fn is_gather(self) -> bool {
        matches!(
            self,
            SampleMode::Gather
                | SampleMode::GatherLodZero
                | SampleMode::GatherLod
                | SampleMode::GatherBias
                | SampleMode::GatherCompare
                | SampleMode::GatherCompareLod
                | SampleMode::GatherCompareBias
                | SampleMode::GatherCompareLodZero
        )
    }

    /// Reads an LOD bias before the coordinates.
    pub fn has_bias(self) -> bool {
        matches!(
            self,
            SampleMode::Bias
                | SampleMode::CompareBias
                | SampleMode::GatherBias
                | SampleMode::GatherCompareBias
        )
    }

    /// Reads an explicit LOD after the coordinates.
    pub fn has_lod(self) -> bool {
        matches!(
            self,
            SampleMode::Lod
                | SampleMode::CompareLod
                | SampleMode::GatherLod
                | SampleMode::GatherCompareLod
        )
    }
}

/// The sample mode and `(offset, clamp)` modifiers of a GFX10 MIMG sample or
/// gather opcode (`image_sample*` 0x20..=0x3f, `image_gather4*` 0x40..=0x5f),
/// or `None` for opcodes Strata does not lower. Gathers with an LOD or bias
/// read those operands but sample the base level (Vulkan core gathers have
/// no LOD); compare gathers lower to `OpImageDrefGather`.
pub fn sample_variant(op: u32) -> Option<(SampleMode, bool, bool)> {
    let offset = op & 0x10 != 0;
    let compare = op & 0x08 != 0;
    let kind = op & 0x07;
    match op {
        0x20..=0x3f => {
            // kind: 0 -, 1 _cl, 2 _d, 3 _d_cl, 4 _l, 5 _b, 6 _b_cl, 7 _lz
            let clamp = matches!(kind, 1 | 3 | 6);
            let mode = match (compare, kind) {
                (false, 0 | 1) => SampleMode::Implicit,
                (false, 2 | 3) => SampleMode::Grad,
                (false, 4) => SampleMode::Lod,
                (false, 5 | 6) => SampleMode::Bias,
                (false, _) => SampleMode::LodZero,
                (true, 0 | 1) => SampleMode::Compare,
                (true, 2 | 3) => SampleMode::CompareGrad,
                (true, 4) => SampleMode::CompareLod,
                (true, 5 | 6) => SampleMode::CompareBias,
                (true, _) => SampleMode::CompareLodZero,
            };
            Some((mode, offset, clamp))
        }
        0x40..=0x5f => {
            // kind: 0 -, 1 _cl, 4 _l, 5 _b, 6 _b_cl, 7 _lz (2, 3: none).
            let clamp = matches!(kind, 1 | 6);
            let mode = match (compare, kind) {
                (false, 0 | 1) => SampleMode::Gather,
                (false, 4) => SampleMode::GatherLod,
                (false, 5 | 6) => SampleMode::GatherBias,
                (false, 7) => SampleMode::GatherLodZero,
                (true, 0 | 1) => SampleMode::GatherCompare,
                (true, 4) => SampleMode::GatherCompareLod,
                (true, 5 | 6) => SampleMode::GatherCompareBias,
                (true, 7) => SampleMode::GatherCompareLodZero,
                _ => return None,
            };
            Some((mode, offset, clamp))
        }
        _ => None,
    }
}

/// Atomic memory operation kind for `buffer_atomic_*` and `global_atomic_*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicOp {
    Swap,
    CmpSwap,
    Add,
    Sub,
    SMin,
    UMin,
    SMax,
    UMax,
    And,
    Or,
    Xor,
}

/// VOP3P packed FP16 / dot-product operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vop3pOp {
    PkAddF16,
    PkMulF16,
    PkFmaF16,
    Dot2F32F16,
    Dot4I32I8,
    Dot4U32U8,
}

/// Decoded RDNA2 instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inst {
    // --- Scalar ALU (SOP2 / SOP1 / SOPK / SOPC) ---
    SAddU32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    SSubU32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    SMulI32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    SAndB32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    SOrB32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    SXorB32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    SLshlB32 {
        dst: u8,
        src0: Operand,
        shift: Operand,
    },
    SLshrB32 {
        dst: u8,
        src0: Operand,
        shift: Operand,
    },
    SAndB64 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    SOrB64 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    SXorB64 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    SAndN2B64 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    SOrN2B64 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    SMovB32 {
        dst: u8,
        src0: Operand,
    },
    SMovB64 {
        dst: u8,
        src0: Operand,
    },
    SNotB64 {
        dst: u8,
        src0: Operand,
    },
    SAndSaveExecB64 {
        dst: u8,
        src0: Operand,
    },
    SOrSaveExecB64 {
        dst: u8,
        src0: Operand,
    },
    SMovkI32 {
        dst: u8,
        simm16: i16,
    },
    SAddkI32 {
        dst: u8,
        simm16: i16,
    },
    SMulkI32 {
        dst: u8,
        simm16: i16,
    },
    SCmp {
        op: CmpOp,
        ty: CmpType,
        src0: Operand,
        src1: Operand,
    },

    // --- Scalar Control Flow & Sync (SOPP) ---
    SNop,
    SBarrier,
    SWaitcnt,
    SEndpgm,
    SBranch {
        target_dw: usize,
    },
    SCBranchExecz {
        target_dw: usize,
    },
    SCBranchExecnz {
        target_dw: usize,
    },
    SCBranchVccz {
        target_dw: usize,
    },
    SCBranchVccnz {
        target_dw: usize,
    },
    SCBranchScc {
        on_one: bool,
        target_dw: usize,
    },
    /// `s_sendmsg` (GS / GS_CUT / GS_DONE / etc.).
    SSendMsg {
        msg: u16,
    },

    // --- Scalar Memory (SMEM) ---
    SLoadDword {
        dst: u8,
        count: u8,
        sbase: u8,
        offset: i32,
    },
    /// `s_buffer_load_dword{,x2,x4,x8}` using a 4-SGPR `V#` buffer descriptor.
    SBufferLoadDword {
        dst: u8,
        count: u8,
        srsrc: u8,
        offset: i32,
    },

    // --- Vector ALU (VOP1 / VOP2 / VOPC / VOP3) ---
    VMovB32 {
        dst: u8,
        src0: Operand,
    },
    VReadFirstLaneB32 {
        dst: u8,
        src: u8,
    },
    VLogF32 {
        dst: u8,
        src0: Operand,
    },
    VExpF32 {
        dst: u8,
        src0: Operand,
    },
    VRcpF32 {
        dst: u8,
        src0: Operand,
    },
    VRsqF32 {
        dst: u8,
        src0: Operand,
    },
    VSqrtF32 {
        dst: u8,
        src0: Operand,
    },
    VFloorF32 {
        dst: u8,
        src0: Operand,
    },
    VFractF32 {
        dst: u8,
        src0: Operand,
    },
    VCvtF32U32 {
        dst: u8,
        src0: Operand,
    },
    VCvtU32F32 {
        dst: u8,
        src0: Operand,
    },
    VCvtF32I32 {
        dst: u8,
        src0: Operand,
    },
    VCvtI32F32 {
        dst: u8,
        src0: Operand,
    },
    VAddF32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    VSubF32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    VMulF32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
        legacy: bool,
    },
    VMinF32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    VMaxF32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    VFmacF32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    VFmaF32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
        src2: Operand,
        mods: Vop3Mods,
    },
    VAddU32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    VSubU32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    VAndB32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    VOrB32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    VXorB32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    VLshlRevB32 {
        dst: u8,
        shift: Operand,
        src: Operand,
    },
    VLshrRevB32 {
        dst: u8,
        shift: Operand,
        src: Operand,
    },
    VAshrRevI32 {
        dst: u8,
        shift: Operand,
        src: Operand,
    },
    VLshlAddU32 {
        dst: u8,
        src0: Operand,
        shift: Operand,
        src2: Operand,
    },
    VBfeU32 {
        dst: u8,
        src0: Operand,
        offset: Operand,
        width: Operand,
    },
    VCndMaskB32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
        mask_src: Option<Operand>,
    },
    VCmp {
        op: CmpOp,
        ty: CmpType,
        dst: Option<u8>,
        src0: Operand,
        src1: Operand,
    },
    /// Intra-2x2-quad lane swap (`ds_swizzle` / DPP quad permute).
    VQuadSwap {
        dst: u8,
        src: u8,
        direction: u8,
    },
    /// Every lane of a quad reads lane `lane` of it (`ds_swizzle` quad mode).
    VQuadBroadcast {
        dst: u8,
        src: u8,
        lane: u8,
    },

    // --- Local Data Share / Workgroup Shared Memory (DS) ---
    DsReadB32 {
        dst: u8,
        addr: u8,
        offset: u16,
    },
    DsWriteB32 {
        addr: u8,
        data: u8,
        offset: u16,
    },
    DsReadB64 {
        dst: u8,
        addr: u8,
        offset: u16,
    },
    DsWriteB64 {
        addr: u8,
        data: u8,
        offset: u16,
    },
    DsAddU32 {
        addr: u8,
        data: u8,
        offset: u16,
    },

    // --- Flat / Global Memory (FLAT) ---
    GlobalLoad {
        dst: u8,
        count: u8,
        vaddr: u8,
        saddr: Option<u8>,
        offset: i32,
    },
    GlobalStore {
        vaddr: u8,
        vdata: u8,
        count: u8,
        saddr: Option<u8>,
        offset: i32,
    },
    ScratchLoad {
        dst: u8,
        count: u8,
        offset: i32,
    },
    ScratchStore {
        vdata: u8,
        count: u8,
        offset: i32,
    },

    // --- Buffer Memory (MUBUF / MTBUF) ---
    BufferLoad {
        dst: u8,
        count: u8,
        vaddr: u8,
        srsrc: u8,
        soffset: Operand,
        offset: u16,
        offen: bool,
        idxen: bool,
        typed_format: Option<u8>,
    },
    BufferStore {
        vdata: u8,
        count: u8,
        vaddr: u8,
        srsrc: u8,
        soffset: Operand,
        offset: u16,
        offen: bool,
        idxen: bool,
        typed_format: Option<u8>,
    },

    // --- Image & Ray Tracing (MIMG) ---
    ImageSample {
        dst: u8,
        addr_vgpr: u8,
        rsrc_sgpr: u8,
        samp_sgpr: u8,
        dmask: u8,
        dim: u8,
        mode: SampleMode,
        /// `_o`: a packed texel offset VGPR comes first.
        offset: bool,
        /// `_cl`: an LOD clamp VGPR follows the coordinates.
        clamp: bool,
    },
    /// `image_load` / `image_load_mip` (`mip`: the level follows the coordinates).
    ImageLoad {
        dst: u8,
        addr_vgpr: u8,
        rsrc_sgpr: u8,
        dmask: u8,
        dim: u8,
        mip: bool,
    },
    /// `image_store` / `image_store_mip`.
    ImageStore {
        vdata: u8,
        addr_vgpr: u8,
        rsrc_sgpr: u8,
        dmask: u8,
        dim: u8,
        mip: bool,
    },
    /// `image_bvh64_intersect_ray` — hardware ray/box or ray/triangle node test.
    ImageBvh64IntersectRay {
        dst: u8,
        addr_vgpr: u8,
        rsrc_sgpr: u8,
    },
    /// `image_get_resinfo` — query texture dimensions & mip count.
    ImageGetResInfo {
        dst: u8,
        addr_vgpr: u8,
        rsrc_sgpr: u8,
        dmask: u8,
        dim: u8,
    },
    /// `image_get_lod` — query computed & clamped mip LOD for coordinates.
    ImageGetLod {
        dst: u8,
        addr_vgpr: u8,
        rsrc_sgpr: u8,
        samp_sgpr: u8,
        dmask: u8,
        dim: u8,
    },
    /// `image_atomic_*` — atomic RMW on a storage image texel.
    ImageAtomic {
        op: AtomicOp,
        vdata: u8,
        addr_vgpr: u8,
        rsrc_sgpr: u8,
        dmask: u8,
        dim: u8,
        glc: bool,
    },
    /// `v_mad_i32_i24` / `v_mad_u32_u24` (24-bit multiply-add).
    VMadI24 {
        dst: u8,
        src0: Operand,
        src1: Operand,
        src2: Operand,
        signed: bool,
    },
    /// `v_mad_u64_u32` / `v_mad_i64_i32` (32x32->64 multiply-add).
    VMadU64U32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
        src2: Operand,
        signed: bool,
    },
    /// `v_add_lshl_u32`: `(src0 + src1) << (shift & 31)`.
    VAddLshlU32 {
        dst: u8,
        src0: Operand,
        src1: Operand,
        shift: Operand,
    },

    // --- VOP3P (Packed FP16 & Dot Products) ---
    Vop3p {
        op: Vop3pOp,
        dst: u8,
        src0: Operand,
        src1: Operand,
        src2: Operand,
    },

    // --- Wave64 Cross-Lane Permutes ---
    VPermlane16 {
        dst: u8,
        src0: Operand,
        src1: Operand,
        src2: Operand,
        swap_halves: bool,
    },
    DsBpermuteB32 {
        dst: u8,
        addr: u8,
        data: u8,
    },
    DsPermuteB32 {
        dst: u8,
        addr: u8,
        data: u8,
    },
    DsSwizzleB32 {
        dst: u8,
        src: u8,
        offset: u16,
    },

    // --- Global & Buffer Atomics ---
    GlobalAtomic {
        op: AtomicOp,
        dst: Option<u8>,
        vaddr: u8,
        vdata: u8,
        saddr: Option<u8>,
        offset: i32,
    },
    BufferAtomic {
        op: AtomicOp,
        dst: Option<u8>,
        vdata: u8,
        vaddr: u8,
        srsrc: u8,
        soffset: Operand,
        offset: u16,
        offen: bool,
        idxen: bool,
    },

    // --- Parameter interpolation (VINTRP, pixel shaders) ---
    VInterp {
        dst: u8,
        op: InterpOp,
        attr: u8,
        chan: u8,
    },

    // --- Generic ALU (see `alu.rs`) ---
    /// Scalar ALU op without its own variant (`src1` unused by SOP1 forms).
    Salu {
        op: SaluOp,
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    /// `s_*_saveexec_b{32,64}` / `s_*_wrexec_b{32,64}` beyond `and`/`or`.
    SSaveExec {
        op: MaskOp,
        dst: u8,
        src0: Operand,
        /// `wrexec`: `dst` receives the new EXEC instead of the old.
        wrexec: bool,
    },
    /// 64-bit `s_nand/nor/xnor_b64` on lane masks or SGPR pairs.
    SMask2 {
        op: MaskOp,
        dst: u8,
        src0: Operand,
        src1: Operand,
    },
    /// `s_bitcmp{0,1}_b{32,64}`: SCC = bit `src1` of `src0` is `one`.
    SBitCmp {
        one: bool,
        wide: bool,
        src0: Operand,
        src1: Operand,
    },
    /// `s_cmp_{eq,lg}_u64`.
    SCmp64 {
        ne: bool,
        src0: Operand,
        src1: Operand,
    },
    /// `s_getpc_b64`: the byte address of the next instruction.
    SGetPc {
        dst: u8,
        /// Byte offset of the next instruction from the program start.
        next_byte: u32,
    },
    /// `s_setpc_b64` / `s_swappc_b64` (indirect jump / call).
    SSetPc {
        src: u8,
        /// `s_swappc_b64`: the return address goes here.
        link: Option<u8>,
    },
    /// `s_getreg_b32`: `size` bits of hardware register `id` from `offset`.
    SGetReg {
        dst: u8,
        id: u16,
        offset: u8,
        size: u8,
    },
    /// `s_setreg_b32` / `s_setreg_imm32_b32`.
    SSetReg {
        id: u16,
        offset: u8,
        size: u8,
        src: Operand,
    },
    /// `s_round_mode` / `s_denorm_mode`.
    SSetMode {
        round: Option<u8>,
        denorm: Option<u8>,
    },
    /// `s_memtime` (shader clock) / `s_memrealtime` (device clock).
    SMemTime {
        dst: u8,
        real: bool,
    },
    /// Hints with no effect on results: `s_sleep`, `s_setprio`, `s_trap`,
    /// `s_icache_inv`, `s_inst_prefetch`, `s_clause`, `s_ttracedata`, cache
    /// invalidations, `v_nop`-class ops.
    SHint {
        name: &'static str,
    },
    /// Vector ALU op without its own variant. `sdst` is the VOP3B carry /
    /// scale output (VCC for the 32-bit encodings).
    Valu {
        op: ValuOp,
        dst: u8,
        sdst: Option<u8>,
        src: [Operand; 3],
        mods: Vop3Mods,
        /// VOP3 `OP_SEL` (bits 0..2 select the high half of `src0..2`,
        /// bit 3 writes the high half of the destination).
        opsel: u8,
    },
    /// `v_cmpx_*`: writes EXEC (and no VCC on GFX10).
    VCmpx {
        op: CmpOp,
        ty: CmpType,
        src0: Operand,
        src1: Operand,
    },
    /// A VOP1 / VOP2 / VOPC instruction with DPP16, DPP8 or SDWA operand
    /// modifiers (the extra dword); `inner` reads `Vgpr(src0)` as given.
    LaneMod {
        m: LaneMod,
        inner: Box<Inst>,
    },
    /// A VOP1/VOP2/VOPC op promoted to VOP3 with source/output modifiers
    /// the op itself does not apply.
    Vop3Mod {
        mods: Vop3Mods,
        inner: Box<Inst>,
    },
    /// Sub-dword or multi-dword LDS read (`ds_read_{u8,i8,u16,i16,b96,b128}`).
    DsLoad {
        dst: u8,
        addr: u8,
        offset: u16,
        bytes: u8,
        signed: bool,
    },
    /// Sub-dword or multi-dword LDS write (`ds_write_{b8,b16,b96,b128}`).
    DsStore {
        addr: u8,
        data: u8,
        offset: u16,
        bytes: u8,
    },
    /// `ds_read2{,st64}_b{32,64}`: two reads at `offset{0,1} * (4|8) * (1|64)`.
    DsLoad2 {
        dst: u8,
        addr: u8,
        offset0: u8,
        offset1: u8,
        dwords: u8,
        st64: bool,
    },
    /// `ds_write2{,st64}_b{32,64}`.
    DsStore2 {
        addr: u8,
        data0: u8,
        data1: u8,
        offset0: u8,
        offset1: u8,
        dwords: u8,
        st64: bool,
    },
    /// 32-bit LDS atomic (`ds_{add,sub,min,max,and,or,xor,wrxchg,cmpst}{,_rtn}_*`).
    /// `cmpst`: `data0` is the compare value, `data1` the new value.
    DsAtomic {
        op: AtomicOp,
        dst: Option<u8>,
        addr: u8,
        data0: u8,
        data1: u8,
        offset: u16,
    },
    /// Sub-dword global / FLAT load (`*_load_{ubyte,sbyte,ushort,sshort}`).
    GlobalLoadSub {
        dst: u8,
        vaddr: u8,
        saddr: Option<u8>,
        offset: i32,
        bytes: u8,
        signed: bool,
    },
    /// Sub-dword global / FLAT store (`*_store_{byte,short}`).
    GlobalStoreSub {
        vaddr: u8,
        vdata: u8,
        saddr: Option<u8>,
        offset: i32,
        bytes: u8,
    },
    /// Data after the last reachable `s_endpgm` (constant blocks a program
    /// reads through `s_getpc_b64`).
    ConstData {
        words: Vec<u32>,
    },
    /// An instruction AMD's ISA defines that Strata decodes (so the stream
    /// stays in sync) but does not translate.
    Unlowered {
        mnemonic: &'static str,
    },

    // --- Graphics Shader Output Export (EXP) ---
    Exp {
        target: u8,
        en: u8,
        vsrc: [u8; 4],
        done: bool,
        compr: bool,
    },
}

/// Decoded instruction paired with its starting dword PC.
#[derive(Debug, Clone)]
pub struct DecodedInst {
    pub pc: usize,
    pub len_dw: usize,
    pub inst: Inst,
}

/// Decode a stream of little-endian RDNA2 dwords.
///
/// Encoding precedence follows the AMD RDNA2 ISA specification:
/// - 32-bit scalar: SOPP (`101111111`) -> SOPC (`101111110`) -> SOP1 (`101111101`)
///   -> SOPK (`1011`) -> SOP2 (`10`)
/// - 32-bit vector: VOP1 (`0111111`) -> VOPC (`0111110`) -> VOP2 (`0`)
/// - 64-bit encodings: VOP3 (`110101`), DS (`110110`), FLAT/GLOBAL (`110111`),
///   MUBUF (`111000`), MTBUF (`111010`), MIMG (`111100`), SMEM (`111101`), EXP (`111110`)
fn decode_atomic_op(op: u32) -> Result<AtomicOp, DecodeError> {
    Ok(match op {
        0x30 => AtomicOp::Swap,
        0x31 => AtomicOp::CmpSwap,
        0x32 => AtomicOp::Add,
        0x33 => AtomicOp::Sub,
        // 0x34 is GLOBAL/BUFFER `atomic_csub` (FLAT has none).
        0x35 => AtomicOp::SMin,
        0x36 => AtomicOp::UMin,
        0x37 => AtomicOp::SMax,
        0x38 => AtomicOp::UMax,
        0x39 => AtomicOp::And,
        0x3a => AtomicOp::Or,
        0x3b => AtomicOp::Xor,
        _ => {
            return Err(DecodeError::Unsupported {
                word: op,
                offset: 0,
            });
        }
    })
}

pub fn decode_program(words: &[u32]) -> Result<Vec<DecodedInst>, DecodeError> {
    let mut out = Vec::new();
    let mut pc = 0usize;
    let mut max_target = 0usize;

    while pc < words.len() {
        let w = words[pc];
        let mut len = 1usize;

        let inst = if (w >> 23) == 0b1011_1111_1 {
            // SOPP
            let op = (w >> 16) & 0x7f;
            let simm16 = (w & 0xffff) as i16 as isize;
            let target_dw = ((pc as isize) + 1 + simm16).max(0) as usize;
            match op {
                0x00 => Inst::SNop,
                0x01 | 0x1b | 0x1e => Inst::SEndpgm,
                0x02 => Inst::SBranch { target_dw },
                0x10 | 0x11 => Inst::SSendMsg {
                    msg: (w & 0xffff) as u16,
                },
                0x04 => Inst::SCBranchScc {
                    on_one: false,
                    target_dw,
                },
                0x05 => Inst::SCBranchScc {
                    on_one: true,
                    target_dw,
                },
                0x06 => Inst::SCBranchVccz { target_dw },
                0x07 => Inst::SCBranchVccnz { target_dw },
                0x08 => Inst::SCBranchExecz { target_dw },
                0x09 => Inst::SCBranchExecnz { target_dw },
                0x0a => Inst::SBarrier,
                0x0c | 0x23 => Inst::SWaitcnt,
                0x03 => Inst::SHint { name: "s_wakeup" },
                0x0e => Inst::SHint { name: "s_sleep" },
                0x0f => Inst::SHint { name: "s_setprio" },
                0x12 => Inst::SHint { name: "s_trap" },
                0x13 => Inst::SHint {
                    name: "s_icache_inv",
                },
                0x14 | 0x15 => Inst::SHint {
                    name: "s_perflevel",
                },
                0x16 | 0x28 => Inst::SHint {
                    name: "s_ttracedata",
                },
                0x1f => Inst::SHint { name: "s_code_end" },
                0x20 => Inst::SHint {
                    name: "s_inst_prefetch",
                },
                0x21 => Inst::SHint { name: "s_clause" },
                0x24 => Inst::SSetMode {
                    round: Some((w & 0xf) as u8),
                    denorm: None,
                },
                0x25 => Inst::SSetMode {
                    round: None,
                    denorm: Some((w & 0xf) as u8),
                },
                _ => unlowered(Encoding::Sopp, op, w, pc)?,
            }
        } else if (w >> 23) == 0b1011_1111_0 {
            // SOPC (must be checked BEFORE SOPK and SOP2)
            let op = (w >> 16) & 0x7f;
            let s0 = (w & 0xff) as u16;
            let s1 = ((w >> 8) & 0xff) as u16;
            let lit = if s0 == 255 || s1 == 255 {
                len += 1;
                Some(
                    *words
                        .get(pc + 1)
                        .ok_or(DecodeError::Truncated { offset: pc })?,
                )
            } else {
                None
            };
            let src0 = Operand::from_src9(s0, lit);
            let src1 = Operand::from_src9(s1, lit);
            let cmp = match op {
                0x00 => Some((CmpOp::Eq, CmpType::I32)),
                0x01 => Some((CmpOp::Ne, CmpType::I32)),
                0x02 => Some((CmpOp::Gt, CmpType::I32)),
                0x03 => Some((CmpOp::Ge, CmpType::I32)),
                0x04 => Some((CmpOp::Lt, CmpType::I32)),
                0x05 => Some((CmpOp::Le, CmpType::I32)),
                0x06 => Some((CmpOp::Eq, CmpType::U32)),
                0x07 => Some((CmpOp::Ne, CmpType::U32)),
                0x08 => Some((CmpOp::Gt, CmpType::U32)),
                0x09 => Some((CmpOp::Ge, CmpType::U32)),
                0x0a => Some((CmpOp::Lt, CmpType::U32)),
                0x0b => Some((CmpOp::Le, CmpType::U32)),
                _ => None,
            };
            match (cmp, op) {
                (Some((op, ty)), _) => Inst::SCmp {
                    op,
                    ty,
                    src0,
                    src1,
                },
                (None, 0x0c..=0x0f) => Inst::SBitCmp {
                    one: op & 1 != 0,
                    wide: op >= 0x0e,
                    src0,
                    src1,
                },
                (None, 0x12 | 0x13) => Inst::SCmp64 {
                    ne: op == 0x13,
                    src0,
                    src1,
                },
                _ => unlowered(Encoding::Sopc, op, w, pc)?,
            }
        } else if (w >> 23) == 0b1011_1110_1 {
            // SOP1 (must be checked BEFORE SOPK and SOP2)
            let op = (w >> 8) & 0xff;
            let dst = ((w >> 16) & 0x7f) as u8;
            let ssrc0_code = (w & 0xff) as u16;
            let lit = if ssrc0_code == 255 {
                len += 1;
                Some(
                    *words
                        .get(pc + 1)
                        .ok_or(DecodeError::Truncated { offset: pc })?,
                )
            } else {
                None
            };
            let src0 = Operand::from_src9(ssrc0_code, lit);
            match op {
                0x03 => Inst::SMovB32 { dst, src0 },
                0x04 => Inst::SMovB64 { dst, src0 },
                0x08 => Inst::SNotB64 { dst, src0 },
                // b64 (Wave64) and b32 (Wave32) forms: lane masks either way.
                0x24 | 0x3c => Inst::SAndSaveExecB64 { dst, src0 },
                0x25 | 0x3d => Inst::SOrSaveExecB64 { dst, src0 },
                0x1f => Inst::SGetPc {
                    dst,
                    next_byte: ((pc + len) * 4) as u32,
                },
                0x20 => Inst::SSetPc {
                    src: ssrc0_code as u8,
                    link: None,
                },
                0x21 => Inst::SSetPc {
                    src: ssrc0_code as u8,
                    link: Some(dst),
                },
                _ => {
                    if let Some((mop, wrexec)) = crate::alu::sop1_saveexec(op) {
                        Inst::SSaveExec {
                            op: mop,
                            dst,
                            src0,
                            wrexec,
                        }
                    } else if let Some(sop) = crate::alu::sop1(op) {
                        Inst::Salu {
                            op: sop,
                            dst,
                            src0,
                            src1: Operand::InlineInt(0),
                        }
                    } else {
                        unlowered(Encoding::Sop1, op, w, pc)?
                    }
                }
            }
        } else if (w >> 28) == 0b1011 {
            // SOPK (must be checked AFTER SOPP/SOPC/SOP1 and BEFORE SOP2)
            let op = (w >> 23) & 0x1f;
            let dst = ((w >> 16) & 0x7f) as u8;
            let simm16 = (w & 0xffff) as i16;
            let imm_s = Operand::Literal(simm16 as i32 as u32);
            let imm_u = Operand::Literal(u32::from(simm16 as u16));
            let hw = (w & 0xffff) as u16;
            let (id, offset, size) = (
                hw & 0x3f,
                ((hw >> 6) & 0x1f) as u8,
                ((hw >> 11) & 0x1f) as u8 + 1,
            );
            match op {
                0x00 => Inst::SMovkI32 { dst, simm16 },
                0x01 => Inst::SHint { name: "s_version" },
                0x02 => Inst::Salu {
                    op: SaluOp::CmovkI32,
                    dst,
                    src0: Operand::Sgpr(dst),
                    src1: imm_s,
                },
                0x03..=0x0e => {
                    let (cop, ty) = match op {
                        0x03 => (CmpOp::Eq, CmpType::I32),
                        0x04 => (CmpOp::Ne, CmpType::I32),
                        0x05 => (CmpOp::Gt, CmpType::I32),
                        0x06 => (CmpOp::Ge, CmpType::I32),
                        0x07 => (CmpOp::Lt, CmpType::I32),
                        0x08 => (CmpOp::Le, CmpType::I32),
                        0x09 => (CmpOp::Eq, CmpType::U32),
                        0x0a => (CmpOp::Ne, CmpType::U32),
                        0x0b => (CmpOp::Gt, CmpType::U32),
                        0x0c => (CmpOp::Ge, CmpType::U32),
                        0x0d => (CmpOp::Lt, CmpType::U32),
                        _ => (CmpOp::Le, CmpType::U32),
                    };
                    Inst::SCmp {
                        op: cop,
                        ty,
                        src0: Operand::from_src9(u16::from(dst), None),
                        src1: if ty == CmpType::I32 { imm_s } else { imm_u },
                    }
                }
                0x0f => Inst::SAddkI32 { dst, simm16 },
                0x10 => Inst::SMulkI32 { dst, simm16 },
                0x12 => Inst::SGetReg {
                    dst,
                    id,
                    offset,
                    size,
                },
                0x13 => Inst::SSetReg {
                    id,
                    offset,
                    size,
                    src: Operand::from_src9(u16::from(dst), None),
                },
                0x15 => {
                    len += 1;
                    let lit = *words
                        .get(pc + 1)
                        .ok_or(DecodeError::Truncated { offset: pc })?;
                    Inst::SSetReg {
                        id,
                        offset,
                        size,
                        src: Operand::Literal(lit),
                    }
                }
                0x17..=0x1a => Inst::SWaitcnt,
                _ => unlowered(Encoding::Sopk, op, w, pc)?,
            }
        } else if (w >> 30) == 0b10 {
            // SOP2 (all SOPP/SOPC/SOP1/SOPK prefixes already excluded above)
            let op = (w >> 23) & 0x7f;
            let dst = ((w >> 16) & 0x7f) as u8;
            let s0 = (w & 0xff) as u16;
            let s1 = ((w >> 8) & 0xff) as u16;
            let lit = if s0 == 255 || s1 == 255 {
                len += 1;
                Some(
                    *words
                        .get(pc + 1)
                        .ok_or(DecodeError::Truncated { offset: pc })?,
                )
            } else {
                None
            };
            let src0 = Operand::from_src9(s0, lit);
            let src1 = Operand::from_src9(s1, lit);
            match op {
                0x00 => Inst::SAddU32 { dst, src0, src1 },
                0x01 => Inst::SSubU32 { dst, src0, src1 },
                0x0e => Inst::SAndB32 { dst, src0, src1 },
                0x0f => Inst::SAndB64 { dst, src0, src1 },
                0x10 => Inst::SOrB32 { dst, src0, src1 },
                0x11 => Inst::SOrB64 { dst, src0, src1 },
                0x12 => Inst::SXorB32 { dst, src0, src1 },
                0x13 => Inst::SXorB64 { dst, src0, src1 },
                0x15 => Inst::SAndN2B64 { dst, src0, src1 },
                0x17 => Inst::SOrN2B64 { dst, src0, src1 },
                0x19 => Inst::SMask2 {
                    op: MaskOp::Nand,
                    dst,
                    src0,
                    src1,
                },
                0x1b => Inst::SMask2 {
                    op: MaskOp::Nor,
                    dst,
                    src0,
                    src1,
                },
                0x1d => Inst::SMask2 {
                    op: MaskOp::Xnor,
                    dst,
                    src0,
                    src1,
                },
                0x1e => Inst::SLshlB32 {
                    dst,
                    src0,
                    shift: src1,
                },
                0x20 => Inst::SLshrB32 {
                    dst,
                    src0,
                    shift: src1,
                },
                0x26 => Inst::SMulI32 { dst, src0, src1 },
                _ => match crate::alu::sop2(op) {
                    Some(sop) => Inst::Salu {
                        op: sop,
                        dst,
                        src0,
                        src1,
                    },
                    None => unlowered(Encoding::Sop2, op, w, pc)?,
                },
            }
        } else if (w >> 25) == 0b0111_111 {
            // VOP1
            let op = (w >> 9) & 0xff;
            let dst = ((w >> 17) & 0xff) as u8;
            let s0 = (w & 0x1ff) as u16;
            let (src0, lane_mod) = vop_src0(words, pc, s0, &mut len)?;
            let inner = decode_vop1(op, dst, src0, src0_raw(src0, s0), w, pc)?;
            wrap_lane_mod(inner, lane_mod)
        } else if (w >> 25) == 0b0111_110 {
            // VOPC
            let op = (w >> 17) & 0xff;
            let s0 = (w & 0x1ff) as u16;
            let v1 = ((w >> 9) & 0xff) as u8;
            let (src0, lane_mod) = vop_src0(words, pc, s0, &mut len)?;
            let inner = decode_vopc(op, None, src0, Operand::Vgpr(v1), w, pc)?;
            wrap_lane_mod(inner, lane_mod)
        } else if (w >> 31) == 0 {
            // VOP2
            let op = (w >> 25) & 0x3f;
            let dst = ((w >> 17) & 0xff) as u8;
            let v1 = ((w >> 9) & 0xff) as u8;
            let s0 = (w & 0x1ff) as u16;
            // v_madmk/madak (GFX10.1 0x20/0x21) and v_fmamk/fmaak (0x2c/0x2d,
            // f16 0x37/0x38) always carry a 32-bit literal K.
            if matches!(op, 0x20 | 0x21 | 0x2c | 0x2d | 0x37 | 0x38) {
                len += 1;
                let k = *words
                    .get(pc + 1)
                    .ok_or(DecodeError::Truncated { offset: pc })?;
                let src0 = Operand::from_src9(s0, Some(k));
                match op {
                    // dst = src0 * K + src1
                    0x20 | 0x2c => Inst::VFmaF32 {
                        dst,
                        src0,
                        src1: Operand::Literal(k),
                        src2: Operand::Vgpr(v1),
                        mods: Vop3Mods::default(),
                    },
                    // dst = src0 * src1 + K
                    0x21 | 0x2d => Inst::VFmaF32 {
                        dst,
                        src0,
                        src1: Operand::Vgpr(v1),
                        src2: Operand::Literal(k),
                        mods: Vop3Mods::default(),
                    },
                    0x37 => Inst::Valu {
                        op: ValuOp::FmaF16,
                        dst,
                        sdst: None,
                        src: [src0, Operand::Literal(k), Operand::Vgpr(v1)],
                        mods: Vop3Mods::default(),
                        opsel: 0,
                    },
                    _ => Inst::Valu {
                        op: ValuOp::FmaF16,
                        dst,
                        sdst: None,
                        src: [src0, Operand::Vgpr(v1), Operand::Literal(k)],
                        mods: Vop3Mods::default(),
                        opsel: 0,
                    },
                }
            } else {
                let (src0, lane_mod) = vop_src0(words, pc, s0, &mut len)?;
                let inner = decode_vop2(op, dst, src0, Operand::Vgpr(v1), None, w, pc)?;
                wrap_lane_mod(inner, lane_mod)
            }
        } else if (w >> 26) == 0b1100_10 {
            // VINTRP (32-bit): VSRC[7:0] ATTRCHAN[9:8] ATTR[15:10] OP[17:16] VDST[25:18]
            let op = match (w >> 16) & 3 {
                0 => InterpOp::P1,
                1 => InterpOp::P2,
                2 => InterpOp::Mov,
                _ => {
                    return Err(DecodeError::Unsupported {
                        word: w,
                        offset: pc,
                    });
                }
            };
            Inst::VInterp {
                dst: ((w >> 18) & 0xff) as u8,
                op,
                attr: ((w >> 10) & 0x3f) as u8,
                chan: ((w >> 8) & 3) as u8,
            }
        } else if (w >> 26) == 0b1101_01 {
            // VOP3 (64-bit encoding, optional 32-bit literal)
            len = 2;
            let w1 = *words
                .get(pc + 1)
                .ok_or(DecodeError::Truncated { offset: pc })?;
            let op = (w >> 16) & 0x3ff;
            let dst = (w & 0xff) as u8;
            let sdst_form = crate::alu::vop3_is_sdst(op);
            // VOP3B: SDST[14:8] replaces ABS[10:8] / OP_SEL[14:11].
            let sdst = ((w >> 8) & 0x7f) as u8;
            let abs = if sdst_form { 0 } else { ((w >> 8) & 0x7) as u8 };
            let opsel = if sdst_form { 0 } else { ((w >> 11) & 0xf) as u8 };
            let clamp = ((w >> 15) & 1) != 0;
            let s0 = (w1 & 0x1ff) as u16;
            let s1 = ((w1 >> 9) & 0x1ff) as u16;
            let s2 = ((w1 >> 18) & 0x1ff) as u16;
            let omod = ((w1 >> 27) & 0x3) as u8;
            let neg = ((w1 >> 29) & 0x7) as u8;
            let lit = if s0 == 255 || s1 == 255 || s2 == 255 {
                len += 1;
                Some(
                    *words
                        .get(pc + 2)
                        .ok_or(DecodeError::Truncated { offset: pc })?,
                )
            } else {
                None
            };
            let src0 = Operand::from_src9(s0, lit);
            let src1 = Operand::from_src9(s1, lit);
            let src2 = Operand::from_src9(s2, lit);
            let mods = Vop3Mods {
                abs,
                neg,
                clamp,
                omod,
            };
            let valu = |vop: ValuOp| Inst::Valu {
                op: vop,
                dst,
                sdst: sdst_form.then_some(sdst),
                src: [src0, src1, src2],
                mods,
                opsel,
            };
            let has_mods = abs != 0 || neg != 0 || clamp || omod != 0;
            let promoted = |inner: Inst| match inner {
                // Ops that read the modifiers themselves.
                Inst::Valu { .. } | Inst::VFmaF32 { .. } | Inst::Unlowered { .. } => inner,
                _ if has_mods => Inst::Vop3Mod {
                    mods,
                    inner: Box::new(inner),
                },
                _ => inner,
            };

            if op < 0x100 {
                // VOPC promoted to VOP3 (writes arbitrary SGPR pair `dst`)
                promoted(decode_vopc(op, Some(dst), src0, src1, w, pc)?)
            } else if (0x100..0x140).contains(&op) {
                // VOP2 promoted to VOP3
                match decode_vop2(op - 0x100, dst, src0, src1, Some(src2), w, pc)? {
                    Inst::Valu { op: vop, .. } => valu(vop),
                    inner => promoted(inner),
                }
            } else if (0x180..0x200).contains(&op) {
                // VOP1 promoted to VOP3
                match decode_vop1(op - 0x180, dst, src0, s0, w, pc)? {
                    Inst::Valu { op: vop, .. } => valu(vop),
                    inner => promoted(inner),
                }
            } else {
                match op {
                    // v_fma_legacy_f32 (0x140) / v_mad_f32 (GFX10.1 0x141) / v_fma_f32 (0x14b)
                    0x141 | 0x14b => Inst::VFmaF32 {
                        dst,
                        src0,
                        src1,
                        src2,
                        mods,
                    },
                    0x142 => Inst::VMadI24 {
                        dst,
                        src0,
                        src1,
                        src2,
                        signed: true,
                    },
                    0x143 => Inst::VMadI24 {
                        dst,
                        src0,
                        src1,
                        src2,
                        signed: false,
                    },
                    0x148 => Inst::VBfeU32 {
                        dst,
                        src0,
                        offset: src1,
                        width: src2,
                    }, // v_bfe_u32
                    0x176 => Inst::VMadU64U32 {
                        dst,
                        src0,
                        src1,
                        src2,
                        signed: false,
                    },
                    0x177 => Inst::VMadU64U32 {
                        dst,
                        src0,
                        src1,
                        src2,
                        signed: true,
                    },
                    0x346 => Inst::VLshlAddU32 {
                        dst,
                        src0,
                        shift: src1,
                        src2,
                    }, // v_lshl_add_u32
                    0x347 => Inst::VAddLshlU32 {
                        dst,
                        src0,
                        src1,
                        shift: src2,
                    }, // v_add_lshl_u32
                    0x377 => Inst::VPermlane16 {
                        dst,
                        src0,
                        src1,
                        src2,
                        swap_halves: false,
                    },
                    0x378 => Inst::VPermlane16 {
                        dst,
                        src0,
                        src1,
                        src2,
                        swap_halves: true,
                    },
                    _ => match crate::alu::vop3(op) {
                        Some(vop) => valu(vop),
                        None => unlowered(Encoding::Vop3, op, w, pc)?,
                    },
                }
            }
        } else if (w >> 26) == 0b1100_11 {
            // VOP3P (64-bit packed FP16 / dot-product encoding)
            len = 2;
            let w1 = *words
                .get(pc + 1)
                .ok_or(DecodeError::Truncated { offset: pc })?;
            let op = (w >> 16) & 0x7f;
            let dst = (w & 0xff) as u8;
            let s0 = (w1 & 0x1ff) as u16;
            let s1 = ((w1 >> 9) & 0x1ff) as u16;
            let s2 = ((w1 >> 18) & 0x1ff) as u16;
            let lit = if s0 == 255 || s1 == 255 || s2 == 255 {
                len = 3;
                Some(
                    *words
                        .get(pc + 2)
                        .ok_or(DecodeError::Truncated { offset: pc })?,
                )
            } else {
                None
            };
            let src0 = Operand::from_src9(s0, lit);
            let src1 = Operand::from_src9(s1, lit);
            let src2 = Operand::from_src9(s2, lit);
            let vop3p_op = match op {
                0x0f => Vop3pOp::PkAddF16,
                0x10 => Vop3pOp::PkMulF16,
                0x0e => Vop3pOp::PkFmaF16,
                0x13 => Vop3pOp::Dot2F32F16,
                0x16 => Vop3pOp::Dot4I32I8,
                0x17 => Vop3pOp::Dot4U32U8,
                _ => {
                    out.push(DecodedInst {
                        pc,
                        len_dw: len,
                        inst: unlowered(Encoding::Vop3p, op, w, pc)?,
                    });
                    pc += len;
                    continue;
                }
            };
            Inst::Vop3p {
                op: vop3p_op,
                dst,
                src0,
                src1,
                src2,
            }
        } else if (w >> 26) == 0b1101_10 {
            // DS (LDS / Workgroup shared memory, 64-bit encoding)
            len = 2;
            let w1 = *words
                .get(pc + 1)
                .ok_or(DecodeError::Truncated { offset: pc })?;
            let op = (w >> 18) & 0xff;
            let gds = (w >> 17) & 1 != 0;
            let offset = (w & 0xffff) as u16;
            let offset0 = (w & 0xff) as u8;
            let offset1 = ((w >> 8) & 0xff) as u8;
            let addr = (w1 & 0xff) as u8;
            let data0 = ((w1 >> 8) & 0xff) as u8;
            let data1 = ((w1 >> 16) & 0xff) as u8;
            let vdst = ((w1 >> 24) & 0xff) as u8;
            let atomic = |aop: AtomicOp, rtn: bool| Inst::DsAtomic {
                op: aop,
                dst: rtn.then_some(vdst),
                addr,
                data0,
                data1,
                offset,
            };
            if gds {
                unlowered(Encoding::Ds, op, w, pc)?
            } else {
                match op {
                    0x00 => Inst::DsAddU32 {
                        addr,
                        data: data0,
                        offset,
                    },
                    0x01 => atomic(AtomicOp::Sub, false),
                    0x05 => atomic(AtomicOp::SMin, false),
                    0x06 => atomic(AtomicOp::SMax, false),
                    0x07 => atomic(AtomicOp::UMin, false),
                    0x08 => atomic(AtomicOp::UMax, false),
                    0x09 => atomic(AtomicOp::And, false),
                    0x0a => atomic(AtomicOp::Or, false),
                    0x0b => atomic(AtomicOp::Xor, false),
                    0x10 => atomic(AtomicOp::CmpSwap, false),
                    0x20 => atomic(AtomicOp::Add, true),
                    0x21 => atomic(AtomicOp::Sub, true),
                    0x25 => atomic(AtomicOp::SMin, true),
                    0x26 => atomic(AtomicOp::SMax, true),
                    0x27 => atomic(AtomicOp::UMin, true),
                    0x28 => atomic(AtomicOp::UMax, true),
                    0x29 => atomic(AtomicOp::And, true),
                    0x2a => atomic(AtomicOp::Or, true),
                    0x2b => atomic(AtomicOp::Xor, true),
                    0x2d => atomic(AtomicOp::Swap, true),
                    0x30 => atomic(AtomicOp::CmpSwap, true),
                    0x0d => Inst::DsWriteB32 {
                        addr,
                        data: data0,
                        offset,
                    },
                    0x0e | 0x0f => Inst::DsStore2 {
                        addr,
                        data0,
                        data1,
                        offset0,
                        offset1,
                        dwords: 1,
                        st64: op == 0x0f,
                    },
                    0x4e | 0x4f => Inst::DsStore2 {
                        addr,
                        data0,
                        data1,
                        offset0,
                        offset1,
                        dwords: 2,
                        st64: op == 0x4f,
                    },
                    0x14 => Inst::SNop, // ds_nop
                    0x1e | 0x1f => Inst::DsStore {
                        addr,
                        data: data0,
                        offset,
                        bytes: if op == 0x1e { 1 } else { 2 },
                    },
                    0xde | 0xdf => Inst::DsStore {
                        addr,
                        data: data0,
                        offset,
                        bytes: if op == 0xde { 12 } else { 16 },
                    },
                    0x36 => Inst::DsReadB32 {
                        dst: vdst,
                        addr,
                        offset,
                    },
                    0x37 | 0x38 => Inst::DsLoad2 {
                        dst: vdst,
                        addr,
                        offset0,
                        offset1,
                        dwords: 1,
                        st64: op == 0x38,
                    },
                    0x77 | 0x78 => Inst::DsLoad2 {
                        dst: vdst,
                        addr,
                        offset0,
                        offset1,
                        dwords: 2,
                        st64: op == 0x78,
                    },
                    0x39..=0x3c => Inst::DsLoad {
                        dst: vdst,
                        addr,
                        offset,
                        bytes: if op <= 0x3a { 1 } else { 2 },
                        signed: op == 0x39 || op == 0x3b,
                    },
                    0xfe | 0xff => Inst::DsLoad {
                        dst: vdst,
                        addr,
                        offset,
                        bytes: if op == 0xfe { 12 } else { 16 },
                        signed: false,
                    },
                    0x4d => Inst::DsWriteB64 {
                        addr,
                        data: data0,
                        offset,
                    },
                    0x76 => Inst::DsReadB64 {
                        dst: vdst,
                        addr,
                        offset,
                    },
                    0x35 => {
                        // ds_swizzle_b32: quad-permute mode (offset[15] = 1) or
                        // bit-mask / FFT butterfly mode (offset[15] = 0).
                        if (offset & 0x8000) != 0 {
                            match offset & 0xff {
                                0xB1 => Inst::VQuadSwap {
                                    dst: vdst,
                                    src: addr,
                                    direction: 0,
                                }, // 1,0,3,2
                                0x4E => Inst::VQuadSwap {
                                    dst: vdst,
                                    src: addr,
                                    direction: 1,
                                }, // 2,3,0,1
                                0x1B => Inst::VQuadSwap {
                                    dst: vdst,
                                    src: addr,
                                    direction: 2,
                                }, // 3,2,1,0
                                sel @ (0x00 | 0x55 | 0xAA | 0xFF) => Inst::VQuadBroadcast {
                                    dst: vdst,
                                    src: addr,
                                    lane: (sel & 3) as u8,
                                },
                                _ => Inst::DsSwizzleB32 {
                                    dst: vdst,
                                    src: addr,
                                    offset,
                                },
                            }
                        } else {
                            Inst::DsSwizzleB32 {
                                dst: vdst,
                                src: addr,
                                offset,
                            }
                        }
                    }
                    0xb2 => Inst::DsPermuteB32 {
                        dst: vdst,
                        addr,
                        data: data0,
                    },
                    0xb3 => Inst::DsBpermuteB32 {
                        dst: vdst,
                        addr,
                        data: data0,
                    },
                    _ => unlowered(Encoding::Ds, op, w, pc)?,
                }
            }
        } else if (w >> 26) == 0b1101_11 {
            // FLAT (seg=0) / SCRATCH (seg=1) / GLOBAL (seg=2) (64-bit encoding)
            len = 2;
            let w1 = *words
                .get(pc + 1)
                .ok_or(DecodeError::Truncated { offset: pc })?;
            let seg = (w >> 14) & 0x3;
            let op = (w >> 18) & 0x7f;
            let raw_off = w & 0xfff;
            // GLOBAL / SCRATCH offsets are signed 12-bit; FLAT's are unsigned.
            let offset = if seg == 0 {
                raw_off as i32
            } else {
                ((raw_off << 20) as i32) >> 20
            };
            let vaddr = (w1 & 0xff) as u8;
            let vdata = ((w1 >> 8) & 0xff) as u8;
            let saddr_raw = ((w1 >> 16) & 0x7f) as u8;
            // FLAT has no SADDR; 0x7f (off) and 0x7d (null) mean none.
            let saddr = if seg == 0 || saddr_raw == 0x7f || saddr_raw == 0x7d {
                None
            } else {
                Some(saddr_raw)
            };
            let vdst = ((w1 >> 24) & 0xff) as u8;
            let enc = match seg {
                0 => Encoding::Flat,
                1 => Encoding::FlatScratch,
                _ => Encoding::FlatGlobal,
            };
            let count = |op: u32| match op & 3 {
                0 => 1,
                1 => 2,
                2 => 4,
                _ => 3,
            };
            if seg == 1 {
                match op {
                    0x0c..=0x0e => Inst::ScratchLoad {
                        dst: vdst,
                        count: count(op),
                        offset,
                    },
                    0x1c..=0x1e => Inst::ScratchStore {
                        vdata,
                        count: count(op),
                        offset,
                    },
                    _ => unlowered(enc, op, w, pc)?,
                }
            } else if seg == 3 {
                unlowered(enc, op, w, pc)?
            } else {
                match op {
                    0x08..=0x0b => Inst::GlobalLoadSub {
                        dst: vdst,
                        vaddr,
                        saddr,
                        offset,
                        bytes: if op <= 0x09 { 1 } else { 2 },
                        signed: op == 0x09 || op == 0x0b,
                    },
                    0x0c..=0x0f => Inst::GlobalLoad {
                        dst: vdst,
                        count: count(op),
                        vaddr,
                        saddr,
                        offset,
                    },
                    0x18 | 0x1a => Inst::GlobalStoreSub {
                        vaddr,
                        vdata,
                        saddr,
                        offset,
                        bytes: if op == 0x18 { 1 } else { 2 },
                    },
                    0x1c..=0x1f => Inst::GlobalStore {
                        vaddr,
                        vdata,
                        count: count(op),
                        saddr,
                        offset,
                    },
                    0x30..=0x33 | 0x35..=0x3b => {
                        let aop = decode_atomic_op(op)?;
                        let glc = ((w >> 16) & 1) != 0;
                        Inst::GlobalAtomic {
                            op: aop,
                            dst: glc.then_some(vdst),
                            vaddr,
                            vdata,
                            saddr,
                            offset,
                        }
                    }
                    _ => unlowered(enc, op, w, pc)?,
                }
            }
        } else if (w >> 26) == 0b1110_00 {
            // MUBUF (64-bit encoding); OP[7] is word 0 bit 25.
            len = 2;
            let w1 = *words
                .get(pc + 1)
                .ok_or(DecodeError::Truncated { offset: pc })?;
            let offset = (w & 0xfff) as u16;
            let offen = ((w >> 12) & 1) != 0;
            let idxen = ((w >> 13) & 1) != 0;
            let lds = ((w >> 16) & 1) != 0;
            let op = ((w >> 18) & 0x7f) | (((w >> 25) & 1) << 7);
            let vaddr = (w1 & 0xff) as u8;
            let vdata = ((w1 >> 8) & 0xff) as u8;
            let srsrc = (((w1 >> 16) & 0x1f) << 2) as u8;
            let soff_code = ((w1 >> 24) & 0xff) as u16;
            let soffset = Operand::from_src9(soff_code, None);
            let count = |op: u32| match op & 3 {
                0 => 1,
                1 => 2,
                2 => 4,
                _ => 3,
            };
            match op {
                _ if lds => unlowered(Encoding::Mubuf, op, w, pc)?,
                0x0c..=0x0f => Inst::BufferLoad {
                    dst: vdata,
                    count: count(op),
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: None,
                },
                0x1c..=0x1f => Inst::BufferStore {
                    vdata,
                    count: count(op),
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: None,
                },
                0x30..=0x33 | 0x35..=0x3b => {
                    let aop = decode_atomic_op(op)?;
                    let glc = ((w >> 14) & 1) != 0;
                    Inst::BufferAtomic {
                        op: aop,
                        dst: glc.then_some(vdata),
                        vdata,
                        vaddr,
                        srsrc,
                        soffset,
                        offset,
                        offen,
                        idxen,
                    }
                }
                0x71 => Inst::SHint {
                    name: "buffer_gl0_inv",
                },
                0x72 => Inst::SHint {
                    name: "buffer_gl1_inv",
                },
                _ => unlowered(Encoding::Mubuf, op, w, pc)?,
            }
        } else if (w >> 26) == 0b1110_10 {
            // MTBUF (64-bit encoding)
            len = 2;
            let w1 = *words
                .get(pc + 1)
                .ok_or(DecodeError::Truncated { offset: pc })?;
            let offset = (w & 0xfff) as u16;
            let offen = ((w >> 12) & 1) != 0;
            let idxen = ((w >> 13) & 1) != 0;
            // GFX10: OP[18:16], FORMAT[25:19]; OP[3] (the d16 forms) is word 1 bit 21.
            let op = (w >> 16) & 0x7;
            let format = ((w >> 19) & 0x7f) as u8;
            let d16 = (w1 >> 21) & 1 != 0;
            let vaddr = (w1 & 0xff) as u8;
            let vdata = ((w1 >> 8) & 0xff) as u8;
            let srsrc = (((w1 >> 16) & 0x1f) << 2) as u8;
            let soff_code = ((w1 >> 24) & 0xff) as u16;
            let soffset = Operand::from_src9(soff_code, None);
            match op {
                _ if d16 => unlowered(Encoding::Mtbuf, op | 8, w, pc)?,
                0x0 => Inst::BufferLoad {
                    dst: vdata,
                    count: 1,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: Some(format),
                },
                0x1 => Inst::BufferLoad {
                    dst: vdata,
                    count: 2,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: Some(format),
                },
                0x2 => Inst::BufferLoad {
                    dst: vdata,
                    count: 3,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: Some(format),
                },
                0x3 => Inst::BufferLoad {
                    dst: vdata,
                    count: 4,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: Some(format),
                },
                0x4 => Inst::BufferStore {
                    vdata,
                    count: 1,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: Some(format),
                },
                0x5 => Inst::BufferStore {
                    vdata,
                    count: 2,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: Some(format),
                },
                0x6 => Inst::BufferStore {
                    vdata,
                    count: 3,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: Some(format),
                },
                0x7 => Inst::BufferStore {
                    vdata,
                    count: 4,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: Some(format),
                },
                _ => {
                    return Err(DecodeError::Unsupported {
                        word: w,
                        offset: pc,
                    });
                }
            }
        } else if (w >> 26) == 0b1111_00 {
            // MIMG (image load/store/atomic/sample and image_bvh64_intersect_ray)
            // Bits [2:1] encode `dwords` (NSA extra address words: 0 = contiguous, 1..=3 = NSA).
            let nsa_dwords = ((w >> 1) & 0x3) as usize;
            len = 2 + nsa_dwords;
            if pc + len > words.len() {
                return Err(DecodeError::Truncated { offset: pc });
            }
            let w1 = words[pc + 1];
            let dim = ((w >> 3) & 0x7) as u8;
            let dmask = ((w >> 8) & 0xf) as u8;
            let glc = ((w >> 13) & 1) != 0;
            // OP[6:0] is word 0 bits 24:18, OP[7] is bit 0.
            let op = ((w >> 18) & 0x7f) | ((w & 1) << 7);
            let a16 = (w1 >> 30) & 1 != 0;
            let d16 = (w1 >> 31) & 1 != 0;
            let addr_vgpr = (w1 & 0xff) as u8;
            let dst = ((w1 >> 8) & 0xff) as u8;
            let rsrc_sgpr = (((w1 >> 16) & 0x1f) << 2) as u8;
            let samp_sgpr = (((w1 >> 21) & 0x1f) << 2) as u8;
            match op {
                _ if a16 || d16 => unlowered(Encoding::Mimg, op, w, pc)?,
                0x00 => Inst::ImageLoad {
                    dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    mip: false,
                },
                0x01 => Inst::ImageLoad {
                    dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    mip: true,
                },
                0x08 => Inst::ImageStore {
                    vdata: dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    mip: false,
                },
                0x09 => Inst::ImageStore {
                    vdata: dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    mip: true,
                },
                0x0e => Inst::ImageGetResInfo {
                    dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                },
                0x0f => Inst::ImageAtomic {
                    op: AtomicOp::Swap,
                    vdata: dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    glc,
                },
                0x10 => Inst::ImageAtomic {
                    op: AtomicOp::CmpSwap,
                    vdata: dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    glc,
                },
                0x11 => Inst::ImageAtomic {
                    op: AtomicOp::Add,
                    vdata: dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    glc,
                },
                0x12 => Inst::ImageAtomic {
                    op: AtomicOp::Sub,
                    vdata: dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    glc,
                },
                0x14 => Inst::ImageAtomic {
                    op: AtomicOp::SMin,
                    vdata: dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    glc,
                },
                0x15 => Inst::ImageAtomic {
                    op: AtomicOp::UMin,
                    vdata: dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    glc,
                },
                0x16 => Inst::ImageAtomic {
                    op: AtomicOp::SMax,
                    vdata: dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    glc,
                },
                0x17 => Inst::ImageAtomic {
                    op: AtomicOp::UMax,
                    vdata: dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    glc,
                },
                0x18 => Inst::ImageAtomic {
                    op: AtomicOp::And,
                    vdata: dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    glc,
                },
                0x19 => Inst::ImageAtomic {
                    op: AtomicOp::Or,
                    vdata: dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    glc,
                },
                0x1a => Inst::ImageAtomic {
                    op: AtomicOp::Xor,
                    vdata: dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    dmask,
                    dim,
                    glc,
                },
                op if sample_variant(op).is_some() => {
                    let (mode, offset, clamp) = sample_variant(op).expect("checked");
                    Inst::ImageSample {
                        dst,
                        addr_vgpr,
                        rsrc_sgpr,
                        samp_sgpr,
                        dmask,
                        dim,
                        mode,
                        offset,
                        clamp,
                    }
                }
                0x60 => Inst::ImageGetLod {
                    dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    samp_sgpr,
                    dmask,
                    dim,
                },
                0xe7 => Inst::ImageBvh64IntersectRay {
                    dst,
                    addr_vgpr,
                    rsrc_sgpr,
                },
                _ => unlowered(Encoding::Mimg, op, w, pc)?,
            }
        } else if (w >> 26) == 0b1111_01 {
            // SMEM (64-bit encoding)
            len = 2;
            let w1 = *words
                .get(pc + 1)
                .ok_or(DecodeError::Truncated { offset: pc })?;
            let op = (w >> 18) & 0xff;
            let sbase = ((w & 0x3f) << 1) as u8;
            let dst = ((w >> 6) & 0x7f) as u8;
            // Signed 21-bit immediate; SOFFSET (word 1 bits 31:25) must be null.
            let offset = (((w1 & 0x1f_ffff) << 11) as i32) >> 11;
            let soffset = (w1 >> 25) & 0x7f;
            match op {
                0x00..=0x0c if soffset != 0x7d && soffset != 0x7f => {
                    unlowered(Encoding::Smem, op, w, pc)?
                }
                0x00 => Inst::SLoadDword {
                    dst,
                    count: 1,
                    sbase,
                    offset,
                },
                0x01 => Inst::SLoadDword {
                    dst,
                    count: 2,
                    sbase,
                    offset,
                },
                0x02 => Inst::SLoadDword {
                    dst,
                    count: 4,
                    sbase,
                    offset,
                },
                0x03 => Inst::SLoadDword {
                    dst,
                    count: 8,
                    sbase,
                    offset,
                },
                0x04 => Inst::SLoadDword {
                    dst,
                    count: 16,
                    sbase,
                    offset,
                },
                0x0c => Inst::SBufferLoadDword {
                    dst,
                    count: 16,
                    srsrc: sbase,
                    offset,
                },
                0x24 | 0x25 => Inst::SMemTime {
                    dst,
                    real: op == 0x25,
                },
                0x1f => Inst::SHint { name: "s_gl1_inv" },
                0x20 => Inst::SHint {
                    name: "s_dcache_inv",
                },
                0x08 => Inst::SBufferLoadDword {
                    dst,
                    count: 1,
                    srsrc: sbase,
                    offset,
                },
                0x09 => Inst::SBufferLoadDword {
                    dst,
                    count: 2,
                    srsrc: sbase,
                    offset,
                },
                0x0a => Inst::SBufferLoadDword {
                    dst,
                    count: 4,
                    srsrc: sbase,
                    offset,
                },
                0x0b => Inst::SBufferLoadDword {
                    dst,
                    count: 8,
                    srsrc: sbase,
                    offset,
                },
                _ => unlowered(Encoding::Smem, op, w, pc)?,
            }
        } else if (w >> 26) == 0b1111_10 {
            // EXP (graphics shader export, 64-bit encoding)
            len = 2;
            let w1 = *words
                .get(pc + 1)
                .ok_or(DecodeError::Truncated { offset: pc })?;
            let en = (w & 0xf) as u8;
            let target = ((w >> 4) & 0x3f) as u8;
            let compr = ((w >> 10) & 1) != 0;
            let done = ((w >> 11) & 1) != 0;
            let vsrc = [
                (w1 & 0xff) as u8,
                ((w1 >> 8) & 0xff) as u8,
                ((w1 >> 16) & 0xff) as u8,
                ((w1 >> 24) & 0xff) as u8,
            ];
            Inst::Exp {
                target,
                en,
                vsrc,
                done,
                compr,
            }
        } else {
            return Err(DecodeError::Unsupported {
                word: w,
                offset: pc,
            });
        };

        if let Some(t) = branch_target(&inst) {
            max_target = max_target.max(t);
        }
        let ends = matches!(inst, Inst::SEndpgm | Inst::SHint { name: "s_code_end" });
        out.push(DecodedInst {
            pc,
            len_dw: len,
            inst,
        });
        pc += len;
        // Nothing branches past the last `s_endpgm`: what follows is data
        // (constant blocks read through `s_getpc_b64`) or `s_code_end` padding.
        if ends && pc > max_target && pc < words.len() {
            out.push(DecodedInst {
                pc,
                len_dw: words.len() - pc,
                inst: Inst::ConstData {
                    words: words[pc..].to_vec(),
                },
            });
            break;
        }
    }

    Ok(out)
}

fn branch_target(inst: &Inst) -> Option<usize> {
    match *inst {
        Inst::SBranch { target_dw }
        | Inst::SCBranchExecz { target_dw }
        | Inst::SCBranchExecnz { target_dw }
        | Inst::SCBranchVccz { target_dw }
        | Inst::SCBranchVccnz { target_dw }
        | Inst::SCBranchScc { target_dw, .. } => Some(target_dw),
        _ => None,
    }
}

/// An instruction AMD's ISA defines but Strata does not translate, decoded
/// as [`Inst::Unlowered`] so the stream stays in sync; unknown opcodes are a
/// decode error.
fn unlowered(enc: Encoding, op: u32, word: u32, offset: usize) -> Result<Inst, DecodeError> {
    lookup_opcode(enc, op as u16)
        .map(|mnemonic| Inst::Unlowered { mnemonic })
        .ok_or(DecodeError::Unsupported { word, offset })
}

/// `SRC0` of a 32-bit VOP1 / VOP2 / VOPC word: a literal, a DPP16 / DPP8 /
/// SDWA extra dword (returning the real VGPR and its modifier), or a plain
/// operand.
fn vop_src0(
    words: &[u32],
    pc: usize,
    s0: u16,
    len: &mut usize,
) -> Result<(Operand, Option<LaneMod>), DecodeError> {
    if matches!(s0, 255 | 0xe9 | 0xea | 0xf9 | 0xfa) {
        *len += 1;
        let ext = *words
            .get(pc + 1)
            .ok_or(DecodeError::Truncated { offset: pc })?;
        if s0 == 255 {
            return Ok((Operand::Literal(ext), None));
        }
        let (v, m) = LaneMod::decode(s0, ext).expect("modifier code");
        // SDWA's S0 bit picks an SGPR source instead of a VGPR.
        if let LaneMod::Sdwa { .. } = m
            && (ext >> 23) & 1 != 0
        {
            return Ok((Operand::from_src9(u16::from(v), None), Some(m)));
        }
        return Ok((Operand::Vgpr(v), Some(m)));
    }
    Ok((Operand::from_src9(s0, None), None))
}

/// The raw 9-bit code `decode_vop1` wants for `v_readfirstlane_b32`.
fn src0_raw(src0: Operand, s0: u16) -> u16 {
    match src0 {
        Operand::Vgpr(v) => 256 + u16::from(v),
        _ => s0,
    }
}

fn wrap_lane_mod(inner: Inst, m: Option<LaneMod>) -> Inst {
    match m {
        Some(m) => Inst::LaneMod {
            m,
            inner: Box::new(inner),
        },
        None => inner,
    }
}

fn decode_vop1(
    op: u32,
    dst: u8,
    src0: Operand,
    s0_raw: u16,
    word: u32,
    offset: usize,
) -> Result<Inst, DecodeError> {
    Ok(match op {
        0x01 => Inst::VMovB32 { dst, src0 },
        0x02 => Inst::VReadFirstLaneB32 {
            dst,
            src: (s0_raw.saturating_sub(256)) as u8,
        },
        0x05 => Inst::VCvtF32I32 { dst, src0 },
        0x06 => Inst::VCvtF32U32 { dst, src0 },
        0x07 => Inst::VCvtU32F32 { dst, src0 },
        0x08 => Inst::VCvtI32F32 { dst, src0 },
        0x20 => Inst::VFractF32 { dst, src0 },
        0x24 => Inst::VFloorF32 { dst, src0 },
        0x25 => Inst::VExpF32 { dst, src0 },
        0x27 => Inst::VLogF32 { dst, src0 },
        0x2a | 0x2b => Inst::VRcpF32 { dst, src0 },
        0x2e => Inst::VRsqF32 { dst, src0 },
        0x33 => Inst::VSqrtF32 { dst, src0 },
        _ => match crate::alu::vop1(op) {
            Some(vop) => Inst::Valu {
                op: vop,
                dst,
                sdst: None,
                src: [src0, Operand::InlineInt(0), Operand::InlineInt(0)],
                mods: Vop3Mods::default(),
                opsel: 0,
            },
            None => unlowered(Encoding::Vop1, op, word, offset)?,
        },
    })
}

fn decode_vop2(
    op: u32,
    dst: u8,
    src0: Operand,
    src1: Operand,
    src2: Option<Operand>,
    word: u32,
    offset: usize,
) -> Result<Inst, DecodeError> {
    Ok(match op {
        0x01 => Inst::VCndMaskB32 {
            dst,
            src0,
            src1,
            mask_src: src2,
        },
        0x03 => Inst::VAddF32 { dst, src0, src1 },
        0x04 => Inst::VSubF32 { dst, src0, src1 },
        0x05 => Inst::VSubF32 {
            dst,
            src0: src1,
            src1: src0,
        }, // v_subrev_f32
        0x07 => Inst::VMulF32 {
            dst,
            src0,
            src1,
            legacy: true,
        },
        0x08 => Inst::VMulF32 {
            dst,
            src0,
            src1,
            legacy: false,
        },
        0x0f => Inst::VMinF32 { dst, src0, src1 },
        0x10 => Inst::VMaxF32 { dst, src0, src1 },
        // GFX10 numbering: lshrrev 0x16, ashrrev 0x18, lshlrev 0x1a.
        0x16 => Inst::VLshrRevB32 {
            dst,
            shift: src0,
            src: src1,
        },
        0x18 => Inst::VAshrRevI32 {
            dst,
            shift: src0,
            src: src1,
        },
        0x1a => Inst::VLshlRevB32 {
            dst,
            shift: src0,
            src: src1,
        },
        0x1b => Inst::VAndB32 { dst, src0, src1 },
        0x1c => Inst::VOrB32 { dst, src0, src1 },
        0x1d => Inst::VXorB32 { dst, src0, src1 },
        0x25 => Inst::VAddU32 { dst, src0, src1 },
        0x26 => Inst::VSubU32 { dst, src0, src1 },
        0x27 => Inst::VSubU32 {
            dst,
            src0: src1,
            src1: src0,
        }, // v_subrev_nc_u32
        // GFX10.1 (Scarlett) v_mac_f32: dst = src0 * src1 + dst. Lowered as
        // the fused form; the result differs from the unfused one by at most
        // the intermediate product's rounding.
        0x1f => Inst::VFmacF32 { dst, src0, src1 },
        0x2b => Inst::VFmacF32 { dst, src0, src1 },
        _ => match crate::alu::vop2(op) {
            Some(vop) => Inst::Valu {
                op: vop,
                dst,
                // The 32-bit carry forms read and write VCC.
                sdst: vop.has_sdst().then_some(106),
                src: [src0, src1, src2.unwrap_or(Operand::VccLo)],
                mods: Vop3Mods::default(),
                opsel: 0,
            },
            None => unlowered(Encoding::Vop2, op, word, offset)?,
        },
    })
}

fn decode_vopc(
    op: u32,
    dst: Option<u8>,
    src0: Operand,
    src1: Operand,
    word: u32,
    offset: usize,
) -> Result<Inst, DecodeError> {
    use CmpOp as C;
    // Float predicates by the low 4 bits (f, lt, eq, le, gt, lg, ge, o, u,
    // nge, nlg, ngt, nle, neq, nlt, tru).
    const FLOAT: [CmpOp; 16] = [
        C::F,
        C::Lt,
        C::Eq,
        C::Le,
        C::Gt,
        C::Ne,
        C::Ge,
        C::O,
        C::U,
        C::Nge,
        C::Nlg,
        C::Ngt,
        C::Nle,
        C::Neq,
        C::Nlt,
        C::T,
    ];
    // Integer predicates by the low 3 bits (f, lt, eq, le, gt, ne, ge, t).
    const INT: [CmpOp; 8] = [C::F, C::Lt, C::Eq, C::Le, C::Gt, C::Ne, C::Ge, C::T];
    let lo4 = (op & 0xf) as usize;
    let lo3 = (op & 0x7) as usize;
    let (cmp_op, cmp_ty, x) = match op {
        0x00..=0x0f => (FLOAT[lo4], CmpType::F32, false),
        0x10..=0x1f => (FLOAT[lo4], CmpType::F32, true),
        0x80..=0x87 => (INT[lo3], CmpType::I32, false),
        0x90..=0x97 => (INT[lo3], CmpType::I32, true),
        0x88 => (C::Class, CmpType::F32, false),
        0x98 => (C::Class, CmpType::F32, true),
        0x8f => (C::Class, CmpType::F16, false),
        0x9f => (C::Class, CmpType::F16, true),
        0x89..=0x8e => (INT[lo3], CmpType::I16, false),
        0x99..=0x9e => (INT[lo3], CmpType::I16, true),
        0xa0..=0xa7 => (INT[lo3], CmpType::I64, false),
        0xb0..=0xb7 => (INT[lo3], CmpType::I64, true),
        0xa9..=0xae => (INT[lo3], CmpType::U16, false),
        0xb9..=0xbe => (INT[lo3], CmpType::U16, true),
        0xc0..=0xc7 => (INT[lo3], CmpType::U32, false),
        0xd0..=0xd7 => (INT[lo3], CmpType::U32, true),
        0xe0..=0xe7 => (INT[lo3], CmpType::U64, false),
        0xf0..=0xf7 => (INT[lo3], CmpType::U64, true),
        0xc8..=0xcf | 0xe8..=0xef => (FLOAT[lo4], CmpType::F16, false),
        0xd8..=0xdf | 0xf8..=0xff => (FLOAT[lo4], CmpType::F16, true),
        _ => return unlowered(Encoding::Vopc, op, word, offset),
    };
    Ok(if x {
        Inst::VCmpx {
            op: cmp_op,
            ty: cmp_ty,
            src0,
            src1,
        }
    } else {
        Inst::VCmp {
            op: cmp_op,
            ty: cmp_ty,
            dst,
            src0,
            src1,
        }
    })
}

// Mini-encoder helpers for tests & synthetic RDNA2 programs.
pub mod encode {
    pub fn s_nop() -> u32 {
        0b1011_1111_1 << 23
    }

    pub fn s_endpgm() -> u32 {
        (0b1011_1111_1 << 23) | (0x01 << 16)
    }

    pub fn s_branch(rel16: i16) -> u32 {
        (0b1011_1111_1 << 23) | (0x02 << 16) | (rel16 as u16 as u32)
    }

    pub fn s_cbranch_scc0(rel16: i16) -> u32 {
        (0b1011_1111_1 << 23) | (0x04 << 16) | (rel16 as u16 as u32)
    }

    pub fn s_cbranch_scc1(rel16: i16) -> u32 {
        (0b1011_1111_1 << 23) | (0x05 << 16) | (rel16 as u16 as u32)
    }

    pub fn s_cbranch_execz(rel16: i16) -> u32 {
        (0b1011_1111_1 << 23) | (0x08 << 16) | (rel16 as u16 as u32)
    }

    fn s_load(op: u32, dst: u8, sbase: u8, offset: u32) -> [u32; 2] {
        let w0 =
            (0b1111_01 << 26) | (op << 18) | ((dst as u32) << 6) | ((sbase as u32 >> 1) & 0x3f);
        [w0, (0x7d << 25) | (offset & 0x1f_ffff)]
    }

    /// `s_load_dwordx2 s[dst:dst+1], s[sbase:sbase+1], offset`.
    pub fn s_load_dwordx2(dst: u8, sbase: u8, offset: u32) -> [u32; 2] {
        s_load(0x01, dst, sbase, offset)
    }

    /// `s_load_dwordx4 s[dst:dst+3], s[sbase:sbase+1], offset`.
    pub fn s_load_dwordx4(dst: u8, sbase: u8, offset: u32) -> [u32; 2] {
        s_load(0x02, dst, sbase, offset)
    }

    /// `s_load_dwordx8 s[dst:dst+7], s[sbase:sbase+1], offset`.
    pub fn s_load_dwordx8(dst: u8, sbase: u8, offset: u32) -> [u32; 2] {
        s_load(0x03, dst, sbase, offset)
    }

    pub fn s_add_u32(dst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        (0b10 << 30) | ((dst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }

    pub fn s_cbranch_execnz(rel16: i16) -> u32 {
        (0b1011_1111_1 << 23) | (0x09 << 16) | (rel16 as u16 as u32)
    }

    pub fn s_cbranch_vccnz(rel16: i16) -> u32 {
        (0b1011_1111_1 << 23) | (0x07 << 16) | (rel16 as u16 as u32)
    }

    pub fn s_andn2_b64(dst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        (0b10 << 30) | (0x15 << 23) | ((dst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }

    pub fn s_addk_i32(dst: u8, simm16: i16) -> u32 {
        (0b1011 << 28) | (0x0f << 23) | ((dst as u32) << 16) | (simm16 as u16 as u32)
    }

    /// `s_cmp_lt_u32 ssrc0, ssrc1` (operands are 8-bit SSRC codes: 128+n is inline n).
    pub fn s_cmp_lt_u32(ssrc0: u8, ssrc1: u8) -> u32 {
        (0b1011_1111_0 << 23) | (0x0a << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }

    /// `v_add_nc_u32 dst, v[src0], v[src1]`.
    pub fn v_add_nc_u32(dst: u8, src0_vgpr: u8, src1_vgpr: u8) -> u32 {
        (0x25 << 25) | ((dst as u32) << 17) | ((src1_vgpr as u32) << 9) | (256 + src0_vgpr as u32)
    }

    /// `ds_add_u32 v[addr], v[data] offset:...`.
    pub fn ds_add_u32(addr: u8, data: u8, offset: u16) -> [u32; 2] {
        let w0 = (0b1101_10 << 26) | (offset as u32);
        let w1 = ((data as u32) << 8) | (addr as u32);
        [w0, w1]
    }

    /// `ds_swizzle_b32 dst, v[src] offset:<offset>` (0x8000 | selects: quad mode).
    pub fn ds_swizzle(dst: u8, src: u8, offset: u16) -> [u32; 2] {
        let w0 = (0b1101_10 << 26) | (0x35 << 18) | u32::from(offset);
        let w1 = (u32::from(dst) << 24) | u32::from(src);
        [w0, w1]
    }

    pub fn s_barrier() -> u32 {
        (0b1011_1111_1 << 23) | (0x0a << 16)
    }

    pub fn s_waitcnt() -> u32 {
        (0b1011_1111_1 << 23) | (0x0c << 16)
    }

    pub fn s_mov_b32(dst: u8, ssrc0: u8) -> u32 {
        (0b1011_1110_1 << 23) | ((dst as u32) << 16) | (0x03 << 8) | (ssrc0 as u32)
    }

    pub fn s_mov_b64(dst: u8, ssrc0: u8) -> u32 {
        (0b1011_1110_1 << 23) | ((dst as u32) << 16) | (0x04 << 8) | (ssrc0 as u32)
    }

    pub fn s_and_saveexec_b64(dst: u8, ssrc0: u8) -> u32 {
        (0b1011_1110_1 << 23) | ((dst as u32) << 16) | (0x24 << 8) | (ssrc0 as u32)
    }

    pub fn s_xor_b64(dst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        (0b10 << 30) | (0x13 << 23) | ((dst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }

    pub fn s_or_b64(dst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        (0b10 << 30) | (0x11 << 23) | ((dst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }

    pub fn s_and_b64(dst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        (0b10 << 30) | (0x0f << 23) | ((dst as u32) << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }

    pub fn s_movk_i32(dst: u8, simm16: i16) -> u32 {
        (0b1011 << 28) | ((dst as u32) << 16) | (simm16 as u16 as u32)
    }

    pub fn s_cmp_eq_u32(ssrc0: u8, ssrc1: u8) -> u32 {
        (0b1011_1111_0 << 23) | (0x06 << 16) | ((ssrc1 as u32) << 8) | (ssrc0 as u32)
    }

    pub fn v_mov_b32_vgpr(dst: u8, src_vgpr: u8) -> u32 {
        (0b0111_111 << 25) | ((dst as u32) << 17) | (0x01 << 9) | (256 + src_vgpr as u32)
    }

    pub fn v_mov_b32_sgpr(dst: u8, src_sgpr: u8) -> u32 {
        (0b0111_111 << 25) | ((dst as u32) << 17) | (0x01 << 9) | (src_sgpr as u32)
    }

    pub fn v_add_f32(dst: u8, src0_vgpr: u8, src1_vgpr: u8) -> u32 {
        (0x03 << 25) | ((dst as u32) << 17) | ((src1_vgpr as u32) << 9) | (256 + src0_vgpr as u32)
    }

    pub fn v_mul_f32(dst: u8, src0_vgpr: u8, src1_vgpr: u8) -> u32 {
        (0x08 << 25) | ((dst as u32) << 17) | ((src1_vgpr as u32) << 9) | (256 + src0_vgpr as u32)
    }

    pub fn v_fmac_f32(dst: u8, src0_vgpr: u8, src1_vgpr: u8) -> u32 {
        (0x2b << 25) | ((dst as u32) << 17) | ((src1_vgpr as u32) << 9) | (256 + src0_vgpr as u32)
    }

    pub fn v_add_nc_u32_imm(dst: u8, imm: u8, src_vgpr: u8) -> u32 {
        (0x25 << 25) | ((dst as u32) << 17) | ((src_vgpr as u32) << 9) | (128 + imm as u32)
    }

    pub fn v_and_b32_imm(dst: u8, imm: u8, src_vgpr: u8) -> u32 {
        (0x1b << 25) | ((dst as u32) << 17) | ((src_vgpr as u32) << 9) | (128 + imm as u32)
    }

    pub fn v_lshrrev_b32_imm(dst: u8, shift_imm: u8, src_vgpr: u8) -> u32 {
        (0x16 << 25) | ((dst as u32) << 17) | ((src_vgpr as u32) << 9) | (128 + shift_imm as u32)
    }

    /// `v_mov_b32 dst, <32-bit literal>`.
    pub fn v_mov_b32_literal(dst: u8, value: u32) -> [u32; 2] {
        [
            (0b0111_111 << 25) | ((dst as u32) << 17) | (0x01 << 9) | 255,
            value,
        ]
    }

    pub fn v_lshlrev_b32_imm(dst: u8, shift_imm: u8, src_vgpr: u8) -> u32 {
        (0x1a << 25) | ((dst as u32) << 17) | ((src_vgpr as u32) << 9) | (128 + shift_imm as u32)
    }

    /// `v_cmp_lt_u32 vcc, vsrc0, imm` encoded via `v_cmp_gt_u32 vcc, imm, vsrc0`.
    pub fn v_cmp_lt_u32_imm(vsrc: u8, imm: u8) -> u32 {
        // VOPC: op=0xc4 (gt_u32) with src0=imm, vsrc1=vsrc => imm > vsrc <=> vsrc < imm
        (0b0111_110 << 25) | (0xc4 << 17) | ((vsrc as u32) << 9) | (128 + imm as u32)
    }

    pub fn v_cmp_eq_u32_imm(vsrc: u8, imm: u8) -> u32 {
        (0b0111_110 << 25) | (0xc2 << 17) | ((vsrc as u32) << 9) | (128 + imm as u32)
    }

    pub fn v_fma_f32_vop3(dst: u8, src0_vgpr: u8, src1_vgpr: u8, src2_vgpr: u8) -> [u32; 2] {
        let w0 = (0b1101_01 << 26) | (0x14b << 16) | (dst as u32);
        let w1 = ((256 + src2_vgpr as u32) << 18)
            | ((256 + src1_vgpr as u32) << 9)
            | (256 + src0_vgpr as u32);
        [w0, w1]
    }

    pub fn ds_write_b32(addr: u8, data: u8, offset: u16) -> [u32; 2] {
        let w0 = (0b1101_10 << 26) | (0x0d << 18) | (offset as u32);
        let w1 = ((data as u32) << 8) | (addr as u32);
        [w0, w1]
    }

    pub fn ds_read_b32(dst: u8, addr: u8, offset: u16) -> [u32; 2] {
        let w0 = (0b1101_10 << 26) | (0x36 << 18) | (offset as u32);
        let w1 = ((dst as u32) << 24) | (addr as u32);
        [w0, w1]
    }

    pub fn ds_bpermute_b32(dst: u8, addr: u8, data: u8) -> [u32; 2] {
        [
            (0b1101_10 << 26) | (0xb3 << 18),
            (u32::from(dst) << 24) | (u32::from(data) << 8) | u32::from(addr),
        ]
    }

    pub fn global_load_dword(dst: u8, vaddr: u8, saddr: u8, offset: u16) -> [u32; 2] {
        let w0 = (0b1101_11 << 26) | (0x0c << 18) | (2 << 14) | (offset as u32 & 0xfff);
        let w1 = ((dst as u32) << 24) | ((saddr as u32) << 16) | (vaddr as u32);
        [w0, w1]
    }

    pub fn global_store_dword(vaddr: u8, vdata: u8, saddr: u8, offset: u16) -> [u32; 2] {
        let w0 = (0b1101_11 << 26) | (0x1c << 18) | (2 << 14) | (offset as u32 & 0xfff);
        let w1 = ((saddr as u32) << 16) | ((vdata as u32) << 8) | (vaddr as u32);
        [w0, w1]
    }

    pub fn buffer_load_dword(dst: u8, vaddr: u8, srsrc: u8, offset: u16) -> [u32; 2] {
        let w0 = (0b1110_00 << 26) | (0x0c << 18) | (1 << 12) | (offset as u32 & 0xfff);
        let w1 = (128 << 24)
            | (((srsrc as u32 >> 2) & 0x1f) << 16)
            | ((dst as u32) << 8)
            | (vaddr as u32);
        [w0, w1]
    }

    pub fn buffer_store_dword(vdata: u8, vaddr: u8, srsrc: u8, offset: u16) -> [u32; 2] {
        let w0 = (0b1110_00 << 26) | (0x1c << 18) | (1 << 12) | (offset as u32 & 0xfff);
        let w1 = (128 << 24)
            | (((srsrc as u32 >> 2) & 0x1f) << 16)
            | ((vdata as u32) << 8)
            | (vaddr as u32);
        [w0, w1]
    }

    fn mimg(op: u32, dim: u32, dmask: u8, addr: u8, data: u8, rsrc: u8, samp: u8) -> [u32; 2] {
        let w0 = (0b1111_00 << 26) | (op << 18) | (u32::from(dmask & 0xf) << 8) | ((dim & 7) << 3);
        let w1 = ((u32::from(samp) >> 2) << 21)
            | ((u32::from(rsrc) >> 2) << 16)
            | (u32::from(data) << 8)
            | u32::from(addr);
        [w0, w1]
    }

    /// `image_sample_lz v[dst..], v[addr..addr+1], s[rsrc..+7], s[samp..+3] dmask dim:2D`.
    pub fn image_sample_lz_2d(dst: u8, addr: u8, rsrc: u8, samp: u8, dmask: u8) -> [u32; 2] {
        mimg(0x27, 1, dmask, addr, dst, rsrc, samp)
    }

    /// `image_sample_l` (explicit LOD in `v[addr+2]`), 2D.
    pub fn image_sample_l_2d(dst: u8, addr: u8, rsrc: u8, samp: u8, dmask: u8) -> [u32; 2] {
        mimg(0x24, 1, dmask, addr, dst, rsrc, samp)
    }

    /// `image_load v[dst..], v[addr..addr+1] (x, y), s[rsrc..+7] dmask dim:2D`.
    pub fn image_load_2d(dst: u8, addr: u8, rsrc: u8, dmask: u8) -> [u32; 2] {
        mimg(0x00, 1, dmask, addr, dst, rsrc, 0)
    }

    /// `image_store v[data..], v[addr..addr+1], s[rsrc..+7] dmask dim:2D`.
    pub fn image_store_2d(data: u8, addr: u8, rsrc: u8, dmask: u8) -> [u32; 2] {
        mimg(0x08, 1, dmask, addr, data, rsrc, 0)
    }

    /// Any MIMG sample / gather opcode with any `dim`.
    pub fn image_sample_op(
        op: u32,
        dim: u32,
        dst: u8,
        addr: u8,
        rsrc: u8,
        samp: u8,
        dmask: u8,
    ) -> [u32; 2] {
        mimg(op, dim, dmask, addr, dst, rsrc, samp)
    }

    /// `image_sample_lz` with any MIMG `dim` (0..=7).
    pub fn image_sample_lz_dim(
        dim: u32,
        dst: u8,
        addr: u8,
        rsrc: u8,
        samp: u8,
        dmask: u8,
    ) -> [u32; 2] {
        mimg(0x27, dim, dmask, addr, dst, rsrc, samp)
    }

    /// `image_load` with any MIMG `dim`.
    pub fn image_load_dim(dim: u32, dst: u8, addr: u8, rsrc: u8, dmask: u8) -> [u32; 2] {
        mimg(0x00, dim, dmask, addr, dst, rsrc, 0)
    }

    /// `image_store` with any MIMG `dim`.
    pub fn image_store_dim(dim: u32, data: u8, addr: u8, rsrc: u8, dmask: u8) -> [u32; 2] {
        mimg(0x08, dim, dmask, addr, data, rsrc, 0)
    }

    /// `tbuffer_load_format_{x,xy,xyz,xyzw}` (`count` 1..=4) with a typed
    /// `format`, `idxen` addressing (`v[vaddr]` is the record index).
    pub fn tbuffer_load_format_idx(
        dst: u8,
        count: u8,
        vaddr: u8,
        srsrc: u8,
        format: u8,
        offset: u16,
    ) -> [u32; 2] {
        let w0 = (0b1110_10 << 26)
            | (u32::from(format & 0x7f) << 19)
            | (u32::from(count - 1) << 16)
            | (1 << 13)
            | (u32::from(offset) & 0xfff);
        let w1 = (128 << 24)
            | (((u32::from(srsrc) >> 2) & 0x1f) << 16)
            | (u32::from(dst) << 8)
            | u32::from(vaddr);
        [w0, w1]
    }

    /// `tbuffer_store_format_*` counterpart of [`tbuffer_load_format_idx`].
    pub fn tbuffer_store_format_idx(
        data: u8,
        count: u8,
        vaddr: u8,
        srsrc: u8,
        format: u8,
        offset: u16,
    ) -> [u32; 2] {
        let w0 = (0b1110_10 << 26)
            | (u32::from(format & 0x7f) << 19)
            | ((u32::from(count - 1) + 4) << 16)
            | (1 << 13)
            | (u32::from(offset) & 0xfff);
        let w1 = (128 << 24)
            | (((u32::from(srsrc) >> 2) & 0x1f) << 16)
            | (u32::from(data) << 8)
            | u32::from(vaddr);
        [w0, w1]
    }

    pub fn image_bvh64_intersect_ray(dst: u8, addr_vgpr: u8, rsrc_sgpr: u8) -> [u32; 2] {
        let w0 = (0b1111_00 << 26) | (0x67 << 18) | 1 | (0xf << 8);
        let w1 =
            (((rsrc_sgpr as u32 >> 2) & 0x1f) << 16) | ((dst as u32) << 8) | (addr_vgpr as u32);
        [w0, w1]
    }

    /// `exp <target> v.., v.., v.., v..` with enable mask `en`.
    /// Targets: 0..=7 `mrt`, 8 `mrtz`, 9 `null`, 12..=15 `pos`, 32..=63 `param`.
    pub fn exp(target: u8, en: u8, vsrc: [u8; 4], done: bool, compr: bool) -> [u32; 2] {
        let w0 = (0b1111_10 << 26)
            | (u32::from(done) << 11)
            | (u32::from(compr) << 10)
            | (u32::from(target & 0x3f) << 4)
            | u32::from(en & 0xf);
        let w1 = u32::from_le_bytes(vsrc);
        [w0, w1]
    }

    /// `v_interp_{p1,p2,mov}_f32 dst, src, attr<attr>.<chan>`.
    pub fn v_interp(op: u8, dst: u8, attr: u8, chan: u8, src: u8) -> u32 {
        (0b1100_10 << 26)
            | (u32::from(dst) << 18)
            | (u32::from(op & 3) << 16)
            | (u32::from(attr & 0x3f) << 10)
            | (u32::from(chan & 3) << 8)
            | u32::from(src)
    }

    /// `v_cvt_f32_u32 dst, v[src]`.
    pub fn v_cvt_f32_u32(dst: u8, src_vgpr: u8) -> u32 {
        (0b0111_111 << 25) | ((dst as u32) << 17) | (0x06 << 9) | (256 + src_vgpr as u32)
    }

    /// `v_mul_f32 dst, <src0 code>, v[src1]` (src0: an SSRC code, e.g. 246 = 4.0).
    pub fn v_mul_f32_code(dst: u8, src0_code: u16, src1_vgpr: u8) -> u32 {
        (0x08 << 25) | ((dst as u32) << 17) | ((src1_vgpr as u32) << 9) | u32::from(src0_code)
    }

    /// `v_add_f32 dst, <src0 code>, v[src1]`.
    pub fn v_add_f32_code(dst: u8, src0_code: u16, src1_vgpr: u8) -> u32 {
        (0x03 << 25) | ((dst as u32) << 17) | ((src1_vgpr as u32) << 9) | u32::from(src0_code)
    }

    pub fn exp_mrt0(vsrc: [u8; 4], done: bool) -> [u32; 2] {
        let w0 = (0b1111_10 << 26) | ((done as u32) << 11) | 0xf;
        let w1 = ((vsrc[3] as u32) << 24)
            | ((vsrc[2] as u32) << 16)
            | ((vsrc[1] as u32) << 8)
            | (vsrc[0] as u32);
        [w0, w1]
    }

    /// `scratch_load_dword v[dst], off, off offset:<offset>` (FLAT seg=1, op=0x0c).
    pub fn scratch_load_dword(dst: u8, offset: u16) -> [u32; 2] {
        let w0 = (0b1101_11 << 26) | (0x0c << 18) | (1 << 14) | u32::from(offset & 0xfff);
        let w1 = ((dst as u32) << 24) | (0x7f << 16);
        [w0, w1]
    }

    /// `scratch_store_dword off, v[vdata], off offset:<offset>` (FLAT seg=1, op=0x1c).
    pub fn scratch_store_dword(vdata: u8, offset: u16) -> [u32; 2] {
        let w0 = (0b1101_11 << 26) | (0x1c << 18) | (1 << 14) | u32::from(offset & 0xfff);
        let w1 = (0x7f << 16) | ((vdata as u32) << 8);
        [w0, w1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_gfx10_1_mac_and_mad_legacy() {
        // v_mac_f32 v3, v1, v2 (VOP2 0x1f)
        let mac = (0x1f << 25) | (3 << 17) | (2 << 9) | (256 + 1);
        let p = decode_program(&[mac, encode::s_endpgm()]).unwrap();
        assert!(
            matches!(p[0].inst, Inst::VFmacF32 { dst: 3, .. }),
            "{:?}",
            p[0].inst
        );
        // v_mad_legacy_f32 v4, v1, v2, v3 (VOP3 0x140)
        let lo = (0b110101 << 26) | (0x140 << 16) | 4;
        let hi = (256 + 1) | ((256 + 2) << 9) | ((256 + 3) << 18);
        let p = decode_program(&[lo, hi, encode::s_endpgm()]).unwrap();
        assert!(
            matches!(p[0].inst, Inst::VFmaF32 { dst: 4, .. }),
            "{:?}",
            p[0].inst
        );
    }

    #[test]
    fn decodes_sopc_sopk_vop3_ds_mubuf_exp_without_misclassification() {
        let mut prog = Vec::new();
        prog.push(encode::s_movk_i32(4, -12));
        prog.push(encode::s_cmp_eq_u32(4, 5));
        prog.extend_from_slice(&encode::v_fma_f32_vop3(3, 0, 1, 2));
        prog.extend_from_slice(&encode::ds_write_b32(0, 3, 16));
        prog.extend_from_slice(&encode::ds_read_b32(4, 0, 16));
        prog.extend_from_slice(&encode::buffer_load_dword(5, 0, 8, 64));
        prog.extend_from_slice(&encode::exp_mrt0([0, 1, 2, 3], true));
        prog.push(encode::s_endpgm());

        let decoded = decode_program(&prog).expect("decode failed");
        assert!(matches!(
            decoded[0].inst,
            Inst::SMovkI32 {
                dst: 4,
                simm16: -12
            }
        ));
        assert!(matches!(
            decoded[1].inst,
            Inst::SCmp {
                op: CmpOp::Eq,
                ty: CmpType::U32,
                ..
            }
        ));
        assert!(matches!(decoded[2].inst, Inst::VFmaF32 { dst: 3, .. }));
        assert!(matches!(
            decoded[3].inst,
            Inst::DsWriteB32 {
                addr: 0,
                data: 3,
                offset: 16
            }
        ));
        assert!(matches!(
            decoded[4].inst,
            Inst::DsReadB32 {
                dst: 4,
                addr: 0,
                offset: 16
            }
        ));
        assert!(matches!(
            decoded[5].inst,
            Inst::BufferLoad {
                dst: 5,
                srsrc: 8,
                offset: 64,
                ..
            }
        ));
        assert!(matches!(
            decoded[6].inst,
            Inst::Exp {
                target: 0,
                done: true,
                ..
            }
        ));
        assert!(matches!(decoded[7].inst, Inst::SEndpgm));
    }
}
