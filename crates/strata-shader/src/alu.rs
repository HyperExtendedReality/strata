//! Generic scalar / vector ALU opcodes for the RDNA2 decoder.
//!
//! Opcode numbers follow AMD's machine-readable RDNA2 ISA specification
//! (MIT licence), as tabulated in [`crate::isa_spec`]. Instructions decoded
//! here lower through one generic path in `ssa.rs` instead of one bespoke
//! `Inst` variant each.

/// A scalar ALU operation (`SOP1` / `SOP2` / `SOPK`) without its own `Inst`
/// variant. 64-bit forms operate on SGPR pairs; when an operand is a lane
/// mask (VCC, EXEC, or an SGPR pair a `v_cmp` wrote) its bits come from a
/// wave ballot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SaluOp {
    // SOP2
    AddI32,
    SubI32,
    AddcU32,
    SubbU32,
    MinI32,
    MinU32,
    MaxI32,
    MaxU32,
    CselectB32,
    CselectB64,
    Andn2B32,
    Orn2B32,
    NandB32,
    NorB32,
    XnorB32,
    NandB64,
    NorB64,
    XnorB64,
    LshlB64,
    LshrB64,
    AshrI32,
    AshrI64,
    BfmB32,
    BfmB64,
    BfeU32,
    BfeI32,
    BfeU64,
    BfeI64,
    AbsdiffI32,
    /// `s_lshl{n}_add_u32`.
    LshlAddU32(u8),
    PackLlB32B16,
    PackLhB32B16,
    PackHhB32B16,
    MulHiU32,
    MulHiI32,
    // SOP1
    CmovB32,
    CmovB64,
    NotB32,
    WqmB32,
    WqmB64,
    BrevB32,
    BrevB64,
    Bcnt0B32,
    Bcnt0B64,
    Bcnt1B32,
    Bcnt1B64,
    Ff0B32,
    Ff0B64,
    Ff1B32,
    Ff1B64,
    FlbitB32,
    FlbitB64,
    FlbitI32,
    FlbitI64,
    SextI8,
    SextI16,
    Bitset0B32,
    Bitset0B64,
    Bitset1B32,
    Bitset1B64,
    QuadmaskB32,
    QuadmaskB64,
    AbsI32,
    BitreplicateB64B32,
    // SOPK (immediate in `src1`)
    CmovkI32,
}

impl SaluOp {
    /// Whether the destination is an SGPR pair.
    pub fn writes_64(self) -> bool {
        use SaluOp::*;
        matches!(
            self,
            CselectB64
                | NandB64
                | NorB64
                | XnorB64
                | LshlB64
                | LshrB64
                | AshrI64
                | BfmB64
                | BfeU64
                | BfeI64
                | CmovB64
                | WqmB64
                | BrevB64
                | Bitset0B64
                | Bitset1B64
                | QuadmaskB64
                | BitreplicateB64B32
        )
    }

    /// Whether `src0` is an SGPR pair.
    pub fn src0_64(self) -> bool {
        use SaluOp::*;
        matches!(
            self,
            CselectB64
                | NandB64
                | NorB64
                | XnorB64
                | LshlB64
                | LshrB64
                | AshrI64
                | BfeU64
                | BfeI64
                | CmovB64
                | WqmB64
                | BrevB64
                | Bcnt0B64
                | Bcnt1B64
                | Ff0B64
                | Ff1B64
                | FlbitB64
                | FlbitI64
                | QuadmaskB64
        )
    }

    /// Whether `src1` is an SGPR pair.
    pub fn src1_64(self) -> bool {
        use SaluOp::*;
        matches!(self, CselectB64 | NandB64 | NorB64 | XnorB64)
    }
}

/// `SOP2` opcode → generic op (opcodes with their own `Inst` are `None`).
pub fn sop2(op: u32) -> Option<SaluOp> {
    use SaluOp::*;
    Some(match op {
        2 => AddI32,
        3 => SubI32,
        4 => AddcU32,
        5 => SubbU32,
        6 => MinI32,
        7 => MinU32,
        8 => MaxI32,
        9 => MaxU32,
        10 => CselectB32,
        11 => CselectB64,
        20 => Andn2B32,
        22 => Orn2B32,
        24 => NandB32,
        25 => NandB64,
        26 => NorB32,
        27 => NorB64,
        28 => XnorB32,
        29 => XnorB64,
        31 => LshlB64,
        33 => LshrB64,
        34 => AshrI32,
        35 => AshrI64,
        36 => BfmB32,
        37 => BfmB64,
        39 => BfeU32,
        40 => BfeI32,
        41 => BfeU64,
        42 => BfeI64,
        44 => AbsdiffI32,
        46 => LshlAddU32(1),
        47 => LshlAddU32(2),
        48 => LshlAddU32(3),
        49 => LshlAddU32(4),
        50 => PackLlB32B16,
        51 => PackLhB32B16,
        52 => PackHhB32B16,
        53 => MulHiU32,
        54 => MulHiI32,
        _ => return None,
    })
}

/// `SOP1` opcode → generic op.
pub fn sop1(op: u32) -> Option<SaluOp> {
    use SaluOp::*;
    Some(match op {
        5 => CmovB32,
        6 => CmovB64,
        7 => NotB32,
        9 => WqmB32,
        10 => WqmB64,
        11 => BrevB32,
        12 => BrevB64,
        13 => Bcnt0B32,
        14 => Bcnt0B64,
        15 => Bcnt1B32,
        16 => Bcnt1B64,
        17 => Ff0B32,
        18 => Ff0B64,
        19 => Ff1B32,
        20 => Ff1B64,
        21 => FlbitB32,
        22 => FlbitB64,
        23 => FlbitI32,
        24 => FlbitI64,
        25 => SextI8,
        26 => SextI16,
        27 => Bitset0B32,
        28 => Bitset0B64,
        29 => Bitset1B32,
        30 => Bitset1B64,
        44 => QuadmaskB32,
        45 => QuadmaskB64,
        52 => AbsI32,
        59 => BitreplicateB64B32,
        _ => return None,
    })
}

/// Lane-mask combining op of the `s_*_saveexec` / `s_*_wrexec` family and
/// the 64-bit mask logic ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaskOp {
    And,
    Or,
    Xor,
    /// `src & !exec`.
    AndN2,
    /// `src | !exec`.
    OrN2,
    Nand,
    Nor,
    Xnor,
    /// `!src & exec`.
    AndN1,
    /// `!src | exec`.
    OrN1,
}

/// `SOP1` saveexec / wrexec opcodes (both wave sizes): `(op, wrexec)`.
pub fn sop1_saveexec(op: u32) -> Option<(MaskOp, bool)> {
    use MaskOp::*;
    Some(match op {
        38 | 62 => (Xor, false),
        39 | 63 => (AndN2, false),
        40 | 64 => (OrN2, false),
        41 | 65 => (Nand, false),
        42 | 66 => (Nor, false),
        43 | 67 => (Xnor, false),
        55 | 68 => (AndN1, false),
        56 | 69 => (OrN1, false),
        57 | 70 => (AndN1, true),
        58 | 71 => (AndN2, true),
        _ => return None,
    })
}

/// `s_getreg` / `s_setreg` hardware register ids Strata models.
pub mod hwreg {
    pub const MODE: u16 = 1;
    pub const STATUS: u16 = 2;
    pub const TRAPSTS: u16 = 3;
    pub const HW_ID: u16 = 4;
    pub const GPR_ALLOC: u16 = 5;
    pub const LDS_ALLOC: u16 = 6;
    pub const IB_STS: u16 = 7;
    pub const SH_MEM_BASES: u16 = 15;
    pub const FLAT_SCR_LO: u16 = 20;
    pub const FLAT_SCR_HI: u16 = 21;
    pub const HW_ID1: u16 = 23;
    pub const HW_ID2: u16 = 24;
    pub const POPS_PACKER: u16 = 25;
    pub const SHADER_CYCLES: u16 = 29;

    /// Console default `MODE`: round to nearest even for f32 and f64/f16,
    /// f32 denormals flushed, f64/f16 denormals kept (`FP_DENORM` = 0b1100),
    /// IEEE and DX10 clamp on.
    pub const MODE_DEFAULT: u32 = 0x0000_03c0 | (1 << 9);
}

/// A vector ALU operation without its own `Inst` variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValuOp {
    // --- VOP1 ---
    Nop,
    NotB32,
    BfrevB32,
    FfbhU32,
    FfblB32,
    FfbhI32,
    TruncF32,
    CeilF32,
    RndneF32,
    SinF32,
    CosF32,
    FrexpExpI32F32,
    FrexpMantF32,
    CvtF16F32,
    CvtF32F16,
    CvtRpiI32F32,
    CvtFlrI32F32,
    CvtOffF32I4,
    CvtF32Ubyte(u8),
    CvtF16U16,
    CvtF16I16,
    CvtU16F16,
    CvtI16F16,
    CvtNormI16F16,
    CvtNormU16F16,
    SatPkU8I16,
    RcpIflagF32,
    SwapB32,
    Movrels,
    Movreld,
    Movrelsd,
    /// One-operand f16 op, computed in f32 on the low half.
    F16Unary(F16Un),
    /// An f64 operation (needs `shaderFloat64`; refused).
    F64(&'static str),
    // --- VOP2 ---
    MulI32I24,
    MulHiI32I24,
    MulU32U24,
    MulHiU32U24,
    MinI32,
    MaxI32,
    MinU32,
    MaxU32,
    XnorB32,
    /// `v_add_co_ci_u32`: carry in from VCC / `src2`, carry out to `sdst`.
    AddCoCiU32,
    SubCoCiU32,
    SubrevCoCiU32,
    CvtPkrtzF16F32,
    FmacLegacyF32,
    /// `v_fmamk_f32`: `src0 * K + src1` (K in `src2`).
    FmaMkF32,
    /// `v_fmaak_f32`: `src0 * src1 + K` (K in `src2`).
    FmaAkF32,
    LdexpF16,
    /// Two-operand f16 op on the low halves.
    F16Binary(F16Bin),
    FmacF16,
    FmaF16,
    Dot2cF32F16,
    Dot4cI32I8,
    // --- VOP3 only ---
    FmaLegacyF32,
    CubeIdF32,
    CubeScF32,
    CubeTcF32,
    CubeMaF32,
    BfeI32,
    BfiB32,
    AlignbitB32,
    AlignbyteB32,
    Min3F32,
    Min3I32,
    Min3U32,
    Max3F32,
    Max3I32,
    Max3U32,
    Med3F32,
    Med3I32,
    Med3U32,
    SadU8,
    SadHiU8,
    SadU16,
    SadU32,
    MsadU8,
    LerpU8,
    CvtPkU8F32,
    DivFixupF32,
    DivScaleF32,
    DivFmasF32,
    MulLoU32,
    MulHiU32,
    MulHiI32,
    Xor3B32,
    LshlrevB64,
    LshrrevB64,
    AshrrevI64,
    PermB32,
    XadU32,
    ReadlaneB32,
    WritelaneB32,
    LdexpF32,
    BfmB32,
    BcntU32B32,
    MbcntLoU32B32,
    MbcntHiU32B32,
    CvtPknormI16F32,
    CvtPknormU16F32,
    CvtPkU16U32,
    CvtPkI16I32,
    CvtPknormI16F16,
    CvtPknormU16F16,
    PackB32F16,
    Add3U32,
    LshlOrB32,
    AndOrB32,
    Or3B32,
    MadU32U16,
    MadI32I16,
    MadU16,
    MadI16,
    SubNcI32,
    AddNcI32,
    AddCoU32,
    SubCoU32,
    SubrevCoU32,
    /// 16-bit integer op on the low halves.
    I16(I16Op),
    Min3F16,
    Max3F16,
    Med3F16,
    Min3I16,
    Max3I16,
    Med3I16,
    Min3U16,
    Max3U16,
    Med3U16,
    DivFixupF16,
    MulLitF32,
    TrigPreopF64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum F16Un {
    Rcp,
    Sqrt,
    Rsq,
    Log,
    Exp,
    Floor,
    Ceil,
    Trunc,
    Rndne,
    Fract,
    Sin,
    Cos,
    FrexpMant,
    FrexpExp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum F16Bin {
    Add,
    Sub,
    Subrev,
    Mul,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum I16Op {
    AddU,
    SubU,
    MulLo,
    Lshrrev,
    Ashrrev,
    Lshlrev,
    MaxU,
    MaxI,
    MinU,
    MinI,
    AddI,
    SubI,
}

impl ValuOp {
    /// Writes a lane-mask `sdst` (VOP3B) besides the VGPR.
    pub fn has_sdst(self) -> bool {
        matches!(
            self,
            ValuOp::AddCoCiU32
                | ValuOp::SubCoCiU32
                | ValuOp::SubrevCoCiU32
                | ValuOp::AddCoU32
                | ValuOp::SubCoU32
                | ValuOp::SubrevCoU32
                | ValuOp::DivScaleF32
        )
    }

    /// Writes an SGPR instead of a VGPR (`v_readlane_b32`).
    pub fn writes_sgpr(self) -> bool {
        matches!(self, ValuOp::ReadlaneB32)
    }
}

/// `VOP1` opcode (also VOP3 `0x180 + op`) → generic op.
pub fn vop1(op: u32) -> Option<ValuOp> {
    use ValuOp::*;
    Some(match op {
        0x00 | 0x1b | 0x41 => Nop, // v_nop, v_pipeflush, v_clrexcp
        0x03 => F64("v_cvt_i32_f64"),
        0x04 => F64("v_cvt_f64_i32"),
        0x0a => CvtF16F32,
        0x0b => CvtF32F16,
        0x0c => CvtRpiI32F32,
        0x0d => CvtFlrI32F32,
        0x0e => CvtOffF32I4,
        0x0f => F64("v_cvt_f32_f64"),
        0x10 => F64("v_cvt_f64_f32"),
        0x11..=0x14 => CvtF32Ubyte((op - 0x11) as u8),
        0x15 => F64("v_cvt_u32_f64"),
        0x16 => F64("v_cvt_f64_u32"),
        0x17 => F64("v_trunc_f64"),
        0x18 => F64("v_ceil_f64"),
        0x19 => F64("v_rndne_f64"),
        0x1a => F64("v_floor_f64"),
        0x21 => TruncF32,
        0x22 => CeilF32,
        0x23 => RndneF32,
        0x2b => RcpIflagF32,
        0x2f => F64("v_rcp_f64"),
        0x31 => F64("v_rsq_f64"),
        0x34 => F64("v_sqrt_f64"),
        0x35 => SinF32,
        0x36 => CosF32,
        0x37 => NotB32,
        0x38 => BfrevB32,
        0x39 => FfbhU32,
        0x3a => FfblB32,
        0x3b => FfbhI32,
        0x3c => F64("v_frexp_exp_i32_f64"),
        0x3d => F64("v_frexp_mant_f64"),
        0x3e => F64("v_fract_f64"),
        0x3f => FrexpExpI32F32,
        0x40 => FrexpMantF32,
        0x42 => Movreld,
        0x43 => Movrels,
        0x44 | 0x48 => Movrelsd,
        0x50 => CvtF16U16,
        0x51 => CvtF16I16,
        0x52 => CvtU16F16,
        0x53 => CvtI16F16,
        0x54 => F16Unary(F16Un::Rcp),
        0x55 => F16Unary(F16Un::Sqrt),
        0x56 => F16Unary(F16Un::Rsq),
        0x57 => F16Unary(F16Un::Log),
        0x58 => F16Unary(F16Un::Exp),
        0x59 => F16Unary(F16Un::FrexpMant),
        0x5a => F16Unary(F16Un::FrexpExp),
        0x5b => F16Unary(F16Un::Floor),
        0x5c => F16Unary(F16Un::Ceil),
        0x5d => F16Unary(F16Un::Trunc),
        0x5e => F16Unary(F16Un::Rndne),
        0x5f => F16Unary(F16Un::Fract),
        0x60 => F16Unary(F16Un::Sin),
        0x61 => F16Unary(F16Un::Cos),
        0x62 => SatPkU8I16,
        0x63 => CvtNormI16F16,
        0x64 => CvtNormU16F16,
        0x65 => SwapB32,
        _ => return None,
    })
}

/// `VOP2` opcode (also VOP3 `0x100 + op`) → generic op.
pub fn vop2(op: u32) -> Option<ValuOp> {
    use ValuOp::*;
    Some(match op {
        0x02 => Dot2cF32F16,
        0x06 => FmacLegacyF32,
        0x09 => MulI32I24,
        0x0a => MulHiI32I24,
        0x0b => MulU32U24,
        0x0c => MulHiU32U24,
        0x0d => Dot4cI32I8,
        0x11 => MinI32,
        0x12 => MaxI32,
        0x13 => MinU32,
        0x14 => MaxU32,
        0x1e => XnorB32,
        0x28 => AddCoCiU32,
        0x29 => SubCoCiU32,
        0x2a => SubrevCoCiU32,
        0x2f => CvtPkrtzF16F32,
        0x32 => F16Binary(F16Bin::Add),
        0x33 => F16Binary(F16Bin::Sub),
        0x34 => F16Binary(F16Bin::Subrev),
        0x35 => F16Binary(F16Bin::Mul),
        0x36 => FmacF16,
        0x39 => F16Binary(F16Bin::Max),
        0x3a => F16Binary(F16Bin::Min),
        0x3b => LdexpF16,
        _ => return None,
    })
}

/// VOP3-only opcode (`0x140..` and `0x2ff..`) → generic op.
pub fn vop3(op: u32) -> Option<ValuOp> {
    use ValuOp::*;
    Some(match op {
        0x140 => FmaLegacyF32,
        0x144 => CubeIdF32,
        0x145 => CubeScF32,
        0x146 => CubeTcF32,
        0x147 => CubeMaF32,
        0x149 => BfeI32,
        0x14a => BfiB32,
        0x14c => F64("v_fma_f64"),
        0x14d => LerpU8,
        0x14e => AlignbitB32,
        0x14f => AlignbyteB32,
        0x150 => MulLitF32,
        0x151 => Min3F32,
        0x152 => Min3I32,
        0x153 => Min3U32,
        0x154 => Max3F32,
        0x155 => Max3I32,
        0x156 => Max3U32,
        0x157 => Med3F32,
        0x158 => Med3I32,
        0x159 => Med3U32,
        0x15a => SadU8,
        0x15b => SadHiU8,
        0x15c => SadU16,
        0x15d => SadU32,
        0x15e => CvtPkU8F32,
        0x15f => DivFixupF32,
        0x160 => F64("v_div_fixup_f64"),
        0x164 => F64("v_add_f64"),
        0x165 => F64("v_mul_f64"),
        0x166 => F64("v_min_f64"),
        0x167 => F64("v_max_f64"),
        0x168 => F64("v_ldexp_f64"),
        0x169 => MulLoU32,
        0x16a => MulHiU32,
        0x16c => MulHiI32,
        0x16d => DivScaleF32,
        0x16e => F64("v_div_scale_f64"),
        0x16f => DivFmasF32,
        0x170 => F64("v_div_fmas_f64"),
        0x171 => MsadU8,
        0x174 => TrigPreopF64,
        0x178 => Xor3B32,
        0x2ff => LshlrevB64,
        0x300 => LshrrevB64,
        0x301 => AshrrevI64,
        0x303 => I16(I16Op::AddU),
        0x304 => I16(I16Op::SubU),
        0x305 => I16(I16Op::MulLo),
        0x307 => I16(I16Op::Lshrrev),
        0x308 => I16(I16Op::Ashrrev),
        0x309 => I16(I16Op::MaxU),
        0x30a => I16(I16Op::MaxI),
        0x30b => I16(I16Op::MinU),
        0x30c => I16(I16Op::MinI),
        0x30d => I16(I16Op::AddI),
        0x30e => I16(I16Op::SubI),
        0x30f => AddCoU32,
        0x310 => SubCoU32,
        0x311 => PackB32F16,
        0x312 => CvtPknormI16F16,
        0x313 => CvtPknormU16F16,
        0x314 => I16(I16Op::Lshlrev),
        0x319 => SubrevCoU32,
        0x340 => MadU16,
        0x344 => PermB32,
        0x345 => XadU32,
        0x34b => FmaF16,
        0x351 => Min3F16,
        0x352 => Min3I16,
        0x353 => Min3U16,
        0x354 => Max3F16,
        0x355 => Max3I16,
        0x356 => Max3U16,
        0x357 => Med3F16,
        0x358 => Med3I16,
        0x359 => Med3U16,
        0x35e => MadI16,
        0x35f => DivFixupF16,
        0x360 => ReadlaneB32,
        0x361 => WritelaneB32,
        0x362 => LdexpF32,
        0x363 => BfmB32,
        0x364 => BcntU32B32,
        0x365 => MbcntLoU32B32,
        0x366 => MbcntHiU32B32,
        0x368 => CvtPknormI16F32,
        0x369 => CvtPknormU16F32,
        0x36a => CvtPkU16U32,
        0x36b => CvtPkI16I32,
        0x36d => Add3U32,
        0x36f => LshlOrB32,
        0x371 => AndOrB32,
        0x372 => Or3B32,
        0x373 => MadU32U16,
        0x375 => MadI32I16,
        0x376 => SubNcI32,
        0x37f => AddNcI32,
        _ => return None,
    })
}

/// VOP3 opcodes using the VOP3B layout (`SDST` in bits `[14:8]` instead of
/// `ABS`/`OP_SEL`/`CLAMP`).
pub fn vop3_is_sdst(op: u32) -> bool {
    matches!(
        op,
        0x128 | 0x129 | 0x12a | 0x16d | 0x16e | 0x176 | 0x177 | 0x30f | 0x310 | 0x319
    )
}

/// Data-parallel-primitive (`DPP16` / `DPP8`) and sub-dword (`SDWA`)
/// source/destination modifiers of a VOP1 / VOP2 / VOPC instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LaneMod {
    /// DPP16: `dpp_ctrl[8:0]`, `bound_ctrl`, `row_mask`, `bank_mask`, `fi`.
    Dpp16 {
        ctrl: u16,
        bound_ctrl: bool,
        row_mask: u8,
        bank_mask: u8,
        fetch_inactive: bool,
        src0_neg: bool,
        src0_abs: bool,
        src1_neg: bool,
        src1_abs: bool,
    },
    /// DPP8: 3-bit lane selectors for lanes 0..7 of each group of 8.
    Dpp8 { sel: u32 },
    /// SDWA: byte/word selects (0..3 = byte, 4..5 = word, 6 = dword).
    Sdwa {
        dst_sel: u8,
        /// 0 = pad with zeros, 1 = sign-extend, 2 = preserve.
        dst_unused: u8,
        src0_sel: u8,
        src0_sext: bool,
        src0_neg: bool,
        src0_abs: bool,
        src1_sel: u8,
        src1_sext: bool,
        src1_neg: bool,
        src1_abs: bool,
        clamp: bool,
    },
}

impl LaneMod {
    /// Decode the extra dword that follows a VOP1/VOP2/VOPC word whose
    /// `SRC0` is `0xfa` (DPP16), `0xe9`/`0xea` (DPP8) or `0xf9` (SDWA).
    /// Returns the real `src0` VGPR and the modifier.
    pub fn decode(src0_code: u16, ext: u32) -> Option<(u8, LaneMod)> {
        let vgpr = (ext & 0xff) as u8;
        Some(match src0_code {
            0xfa => (
                vgpr,
                LaneMod::Dpp16 {
                    ctrl: ((ext >> 8) & 0x1ff) as u16,
                    fetch_inactive: (ext >> 18) & 1 != 0,
                    bound_ctrl: (ext >> 19) & 1 != 0,
                    src0_neg: (ext >> 20) & 1 != 0,
                    src0_abs: (ext >> 21) & 1 != 0,
                    src1_neg: (ext >> 22) & 1 != 0,
                    src1_abs: (ext >> 23) & 1 != 0,
                    bank_mask: ((ext >> 24) & 0xf) as u8,
                    row_mask: ((ext >> 28) & 0xf) as u8,
                },
            ),
            0xe9 | 0xea => (vgpr, LaneMod::Dpp8 { sel: ext >> 8 }),
            0xf9 => (
                vgpr,
                LaneMod::Sdwa {
                    dst_sel: ((ext >> 8) & 7) as u8,
                    dst_unused: ((ext >> 11) & 3) as u8,
                    clamp: (ext >> 13) & 1 != 0,
                    src0_sel: ((ext >> 16) & 7) as u8,
                    src0_sext: (ext >> 19) & 1 != 0,
                    src0_neg: (ext >> 20) & 1 != 0,
                    src0_abs: (ext >> 21) & 1 != 0,
                    src1_sel: ((ext >> 24) & 7) as u8,
                    src1_sext: (ext >> 27) & 1 != 0,
                    src1_neg: (ext >> 28) & 1 != 0,
                    src1_abs: (ext >> 29) & 1 != 0,
                },
            ),
            _ => return None,
        })
    }

    /// Source lane (within a row of 16 / group of 8) that `lane` reads, or
    /// `None` when the DPP16 control makes the read out of range.
    pub fn dpp_source_lane(&self, lane: u32) -> Option<u32> {
        match *self {
            LaneMod::Dpp8 { sel } => {
                let i = lane & 7;
                Some((lane & !7) | ((sel >> (3 * i)) & 7))
            }
            LaneMod::Dpp16 { ctrl, .. } => {
                let row = lane & !15;
                let r = lane & 15;
                let c = u32::from(ctrl);
                let pick = match c {
                    0x000..=0x0ff => {
                        let q = lane & 3;
                        return Some((lane & !3) | ((c >> (2 * q)) & 3));
                    }
                    0x101..=0x10f => r + (c & 15), // row_shl: read lane + n
                    0x111..=0x11f => r.wrapping_sub(c & 15), // row_shr
                    0x121..=0x12f => (r + 16 - (c & 15)) & 15, // row_ror
                    0x140 => 15 - r,                          // row_mirror
                    0x141 => (r & 8) | (7 - (r & 7)),         // row_half_mirror
                    0x150..=0x15f => c & 15,                  // row_share
                    0x160..=0x16f => r ^ (c & 15),            // row_xmask
                    _ => return None,
                };
                (pick < 16).then_some(row | pick)
            }
            LaneMod::Sdwa { .. } => Some(lane),
        }
    }
}

/// One generic ALU primitive of the SSA IR (`SsaOp::Alu`). Operand counts
/// and types are fixed per op; constants are passed as operands. Vulkan
/// only allows 32-bit operands for bit counting, reversal, find-bit and
/// bit-field ops, so their 64-bit forms are built from halves in `ssa.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AluOp {
    // u32 -> u32
    UMin,
    UMax,
    SMin,
    SMax,
    UMulHi,
    SMulHi,
    BitCount,
    BitReverse,
    /// Index of the lowest set bit, `0xffff_ffff` for 0.
    FindLsb,
    /// Index (from bit 0) of the highest set bit, `0xffff_ffff` for 0.
    FindUMsb,
    /// Index of the highest bit differing from the sign, `0xffff_ffff` for 0 / -1.
    FindSMsb,
    /// `(base, insert, offset, count)`.
    BitFieldInsert,
    Not,
    // u64
    Shl64,
    Lshr64,
    Ashr64,
    Or64,
    Xor64,
    Not64,
    Sub64,
    And64,
    Add64,
    Mul64,
    /// `(u32) -> u64` sign extension.
    SExt64,
    /// `(u64, u64) -> bool` unsigned / signed less than.
    ULt64,
    SLt64,
    Eq64,
    // f32
    Sin,
    Cos,
    Trunc,
    Ceil,
    Abs,
    Neg,
    /// `(f32, u32 exponent) -> f32`.
    Ldexp,
    FDiv,
    /// `(f32) -> bool`.
    IsNan,
    IsInf,
    // lanes
    /// The invocation's lane in its 64-lane wave (`u32`).
    LaneId64,
    /// The invocation's lane in its hardware subgroup (`u32`).
    SubgroupLane,
    /// `(u64 mask) -> bool`: this lane's bit of a 64-lane mask.
    LaneBit,
    /// Device / subgroup clock (`u64`): `ReadClockKHR`.
    ClockDevice,
    ClockSubgroup,
    /// `(u32) -> u64` zero extension.
    UConvert64,
}

impl AluOp {
    /// Pure (no side effects, freely CSE/DCE-able).
    pub fn is_pure(self) -> bool {
        !matches!(self, AluOp::ClockDevice | AluOp::ClockSubgroup)
    }
}
