//! Clean-room RDNA2 (GFX10.3 / Xbox Series X|S) instruction decoder & IR.
//!
//! Decodes scalar (SOPP, SOPC, SOP1, SOPK, SOP2, SMEM), vector (VOP1, VOPC,
//! VOP2, VOP3), shared-memory (DS/LDS), buffer (MUBUF, MTBUF, FLAT/GLOBAL),
//! image/ray-tracing (MIMG), and graphics export (EXP) encodings.

use thiserror::Error;

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
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Scalar/vector comparison operand type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpType {
    F32,
    U32,
    I32,
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

/// How `image_sample*` picks its mip level.
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

    // --- Scalar Memory (SMEM) ---
    SLoadDword {
        dst: u8,
        count: u8,
        sbase: u8,
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

    // --- Parameter interpolation (VINTRP, pixel shaders) ---
    VInterp {
        dst: u8,
        op: InterpOp,
        attr: u8,
        chan: u8,
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
pub fn decode_program(words: &[u32]) -> Result<Vec<DecodedInst>, DecodeError> {
    let mut out = Vec::new();
    let mut pc = 0usize;

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
                0x01 => Inst::SEndpgm,
                0x02 => Inst::SBranch { target_dw },
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
                0x0c => Inst::SWaitcnt,
                _ => {
                    return Err(DecodeError::Unsupported {
                        word: w,
                        offset: pc,
                    });
                }
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
            let (cmp_op, cmp_ty) = match op {
                0x00 => (CmpOp::Eq, CmpType::I32),
                0x01 => (CmpOp::Ne, CmpType::I32),
                0x02 => (CmpOp::Gt, CmpType::I32),
                0x03 => (CmpOp::Ge, CmpType::I32),
                0x04 => (CmpOp::Lt, CmpType::I32),
                0x05 => (CmpOp::Le, CmpType::I32),
                0x06 => (CmpOp::Eq, CmpType::U32),
                0x07 => (CmpOp::Ne, CmpType::U32),
                0x08 => (CmpOp::Gt, CmpType::U32),
                0x09 => (CmpOp::Ge, CmpType::U32),
                0x0a => (CmpOp::Lt, CmpType::U32),
                0x0b => (CmpOp::Le, CmpType::U32),
                _ => {
                    return Err(DecodeError::Unsupported {
                        word: w,
                        offset: pc,
                    });
                }
            };
            Inst::SCmp {
                op: cmp_op,
                ty: cmp_ty,
                src0,
                src1,
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
                0x1f => Inst::SNotB64 { dst, src0 },
                0x3c => Inst::SAndSaveExecB64 { dst, src0 },
                0x3d => Inst::SOrSaveExecB64 { dst, src0 },
                _ => {
                    return Err(DecodeError::Unsupported {
                        word: w,
                        offset: pc,
                    });
                }
            }
        } else if (w >> 28) == 0b1011 {
            // SOPK (must be checked AFTER SOPP/SOPC/SOP1 and BEFORE SOP2)
            let op = (w >> 23) & 0x1f;
            let dst = ((w >> 16) & 0x7f) as u8;
            let simm16 = (w & 0xffff) as i16;
            match op {
                0x00 => Inst::SMovkI32 { dst, simm16 },
                0x0f => Inst::SAddkI32 { dst, simm16 },
                0x10 => Inst::SMulkI32 { dst, simm16 },
                _ => {
                    return Err(DecodeError::Unsupported {
                        word: w,
                        offset: pc,
                    });
                }
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
                0x2c => Inst::SMulI32 { dst, src0, src1 },
                _ => {
                    return Err(DecodeError::Unsupported {
                        word: w,
                        offset: pc,
                    });
                }
            }
        } else if (w >> 25) == 0b0111_111 {
            // VOP1
            let op = (w >> 9) & 0xff;
            let dst = ((w >> 17) & 0xff) as u8;
            let s0 = (w & 0x1ff) as u16;
            let lit = if s0 == 255 {
                len += 1;
                Some(
                    *words
                        .get(pc + 1)
                        .ok_or(DecodeError::Truncated { offset: pc })?,
                )
            } else {
                None
            };
            decode_vop1(op, dst, Operand::from_src9(s0, lit), s0, w, pc)?
        } else if (w >> 25) == 0b0111_110 {
            // VOPC
            let op = (w >> 17) & 0xff;
            let s0 = (w & 0x1ff) as u16;
            let v1 = ((w >> 9) & 0xff) as u8;
            let lit = if s0 == 255 {
                len += 1;
                Some(
                    *words
                        .get(pc + 1)
                        .ok_or(DecodeError::Truncated { offset: pc })?,
                )
            } else {
                None
            };
            decode_vopc(
                op,
                None,
                Operand::from_src9(s0, lit),
                Operand::Vgpr(v1),
                w,
                pc,
            )?
        } else if (w >> 31) == 0 {
            // VOP2
            let op = (w >> 25) & 0x3f;
            let dst = ((w >> 17) & 0xff) as u8;
            let v1 = ((w >> 9) & 0xff) as u8;
            let s0 = (w & 0x1ff) as u16;
            let lit = if s0 == 255 {
                len += 1;
                Some(
                    *words
                        .get(pc + 1)
                        .ok_or(DecodeError::Truncated { offset: pc })?,
                )
            } else {
                None
            };
            decode_vop2(
                op,
                dst,
                Operand::from_src9(s0, lit),
                Operand::Vgpr(v1),
                None,
                w,
                pc,
            )?
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
            let abs = ((w >> 8) & 0x7) as u8;
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

            if op < 0x100 {
                // VOPC promoted to VOP3 (writes arbitrary SGPR pair `dst`)
                decode_vopc(op, Some(dst), src0, src1, w, pc)?
            } else if (0x100..0x140).contains(&op) {
                // VOP2 promoted to VOP3
                decode_vop2(op - 0x100, dst, src0, src1, Some(src2), w, pc)?
            } else if (0x180..0x200).contains(&op) {
                // VOP1 promoted to VOP3
                decode_vop1(op - 0x180, dst, src0, s0, w, pc)?
            } else {
                match op {
                    0x141 | 0x14b => Inst::VFmaF32 {
                        dst,
                        src0,
                        src1,
                        src2,
                        mods,
                    }, // v_mad_f32 / v_fma_f32
                    0x148 => Inst::VBfeU32 {
                        dst,
                        src0,
                        offset: src1,
                        width: src2,
                    }, // v_bfe_u32
                    0x346 => Inst::VLshlAddU32 {
                        dst,
                        src0,
                        shift: src1,
                        src2,
                    }, // v_lshl_add_u32
                    _ => {
                        return Err(DecodeError::Unsupported {
                            word: w,
                            offset: pc,
                        });
                    }
                }
            }
        } else if (w >> 26) == 0b1101_10 {
            // DS (LDS / Workgroup shared memory, 64-bit encoding)
            len = 2;
            let w1 = *words
                .get(pc + 1)
                .ok_or(DecodeError::Truncated { offset: pc })?;
            let op = (w >> 18) & 0xff;
            let offset = (w & 0xffff) as u16;
            let addr = (w1 & 0xff) as u8;
            let data0 = ((w1 >> 8) & 0xff) as u8;
            let vdst = ((w1 >> 24) & 0xff) as u8;
            match op {
                0x00 => Inst::DsAddU32 {
                    addr,
                    data: data0,
                    offset,
                },
                0x0d => Inst::DsWriteB32 {
                    addr,
                    data: data0,
                    offset,
                },
                0x36 => Inst::DsReadB32 {
                    dst: vdst,
                    addr,
                    offset,
                },
                0x3d => Inst::DsWriteB64 {
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
                    // ds_swizzle_b32 quad-permute mode: offset[15] = 1 and four
                    // 2-bit selects, lane i of each quad reading lane sel_i.
                    if offset & 0x8000 == 0 {
                        return Err(DecodeError::Unsupported {
                            word: w,
                            offset: pc,
                        }); // bit-mask mode
                    }
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
                        _ => {
                            return Err(DecodeError::Unsupported {
                                word: w,
                                offset: pc,
                            });
                        }
                    }
                }
                _ => {
                    return Err(DecodeError::Unsupported {
                        word: w,
                        offset: pc,
                    });
                }
            }
        } else if (w >> 26) == 0b1101_11 {
            // FLAT / GLOBAL (64-bit encoding)
            len = 2;
            let w1 = *words
                .get(pc + 1)
                .ok_or(DecodeError::Truncated { offset: pc })?;
            let op = (w >> 18) & 0x7f;
            let offset = (w & 0xfff) as i32;
            let vaddr = (w1 & 0xff) as u8;
            let vdata = ((w1 >> 8) & 0xff) as u8;
            let saddr_raw = ((w1 >> 16) & 0x7f) as u8;
            let saddr = if saddr_raw == 0x7f {
                None
            } else {
                Some(saddr_raw)
            };
            let vdst = ((w1 >> 24) & 0xff) as u8;
            match op {
                0x0c => Inst::GlobalLoad {
                    dst: vdst,
                    count: 1,
                    vaddr,
                    saddr,
                    offset,
                },
                0x0d => Inst::GlobalLoad {
                    dst: vdst,
                    count: 2,
                    vaddr,
                    saddr,
                    offset,
                },
                0x0e => Inst::GlobalLoad {
                    dst: vdst,
                    count: 4,
                    vaddr,
                    saddr,
                    offset,
                },
                0x1c => Inst::GlobalStore {
                    vaddr,
                    vdata,
                    count: 1,
                    saddr,
                    offset,
                },
                0x1d => Inst::GlobalStore {
                    vaddr,
                    vdata,
                    count: 2,
                    saddr,
                    offset,
                },
                0x1e => Inst::GlobalStore {
                    vaddr,
                    vdata,
                    count: 4,
                    saddr,
                    offset,
                },
                _ => {
                    return Err(DecodeError::Unsupported {
                        word: w,
                        offset: pc,
                    });
                }
            }
        } else if (w >> 26) == 0b1110_00 {
            // MUBUF (64-bit encoding)
            len = 2;
            let w1 = *words
                .get(pc + 1)
                .ok_or(DecodeError::Truncated { offset: pc })?;
            let offset = (w & 0xfff) as u16;
            let offen = ((w >> 12) & 1) != 0;
            let idxen = ((w >> 13) & 1) != 0;
            let op = (w >> 18) & 0x7f;
            let vaddr = (w1 & 0xff) as u8;
            let vdata = ((w1 >> 8) & 0xff) as u8;
            let srsrc = (((w1 >> 16) & 0x1f) << 2) as u8;
            let soff_code = ((w1 >> 24) & 0xff) as u16;
            let soffset = Operand::from_src9(soff_code, None);
            match op {
                0x0c => Inst::BufferLoad {
                    dst: vdata,
                    count: 1,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: None,
                },
                0x0d => Inst::BufferLoad {
                    dst: vdata,
                    count: 2,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: None,
                },
                0x0e => Inst::BufferLoad {
                    dst: vdata,
                    count: 4,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: None,
                },
                0x1c => Inst::BufferStore {
                    vdata,
                    count: 1,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: None,
                },
                0x1d => Inst::BufferStore {
                    vdata,
                    count: 2,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: None,
                },
                0x1e => Inst::BufferStore {
                    vdata,
                    count: 4,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    typed_format: None,
                },
                _ => {
                    return Err(DecodeError::Unsupported {
                        word: w,
                        offset: pc,
                    });
                }
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
            if (w1 >> 21) & 1 != 0 {
                return Err(DecodeError::Unsupported {
                    word: w,
                    offset: pc,
                });
            }
            let vaddr = (w1 & 0xff) as u8;
            let vdata = ((w1 >> 8) & 0xff) as u8;
            let srsrc = (((w1 >> 16) & 0x1f) << 2) as u8;
            let soff_code = ((w1 >> 24) & 0xff) as u16;
            let soffset = Operand::from_src9(soff_code, None);
            match op {
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
            // MIMG (image load/store/sample and image_bvh64_intersect_ray)
            len = 2;
            let w1 = *words
                .get(pc + 1)
                .ok_or(DecodeError::Truncated { offset: pc })?;
            let dim = ((w >> 3) & 0x7) as u8;
            let dmask = ((w >> 8) & 0xf) as u8;
            let op = (w >> 18) & 0xff;
            let addr_vgpr = (w1 & 0xff) as u8;
            let dst = ((w1 >> 8) & 0xff) as u8;
            let rsrc_sgpr = (((w1 >> 16) & 0x1f) << 2) as u8;
            let samp_sgpr = (((w1 >> 21) & 0x1f) << 2) as u8;
            match op {
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
                0x20 => Inst::ImageSample {
                    dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    samp_sgpr,
                    dmask,
                    dim,
                    mode: SampleMode::Implicit,
                },
                0x24 => Inst::ImageSample {
                    dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    samp_sgpr,
                    dmask,
                    dim,
                    mode: SampleMode::Lod,
                },
                0x25 => Inst::ImageSample {
                    dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    samp_sgpr,
                    dmask,
                    dim,
                    mode: SampleMode::Bias,
                },
                0x27 => Inst::ImageSample {
                    dst,
                    addr_vgpr,
                    rsrc_sgpr,
                    samp_sgpr,
                    dmask,
                    dim,
                    mode: SampleMode::LodZero,
                },
                0x66 => Inst::ImageBvh64IntersectRay {
                    dst,
                    addr_vgpr,
                    rsrc_sgpr,
                },
                _ => {
                    return Err(DecodeError::Unsupported {
                        word: w,
                        offset: pc,
                    });
                }
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
            let offset = (w1 & 0x1f_ffff) as i32;
            let count = match op {
                0x00 => 1,
                0x01 => 2,
                0x02 => 4,
                0x03 => 8,
                _ => {
                    return Err(DecodeError::Unsupported {
                        word: w,
                        offset: pc,
                    });
                }
            };
            Inst::SLoadDword {
                dst,
                count,
                sbase,
                offset,
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

        out.push(DecodedInst {
            pc,
            len_dw: len,
            inst,
        });
        pc += len;
    }

    Ok(out)
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
        0x2a => Inst::VRcpF32 { dst, src0 },
        0x2e => Inst::VRsqF32 { dst, src0 },
        0x33 => Inst::VSqrtF32 { dst, src0 },
        _ => return Err(DecodeError::Unsupported { word, offset }),
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
        0x2b => Inst::VFmacF32 { dst, src0, src1 },
        _ => return Err(DecodeError::Unsupported { word, offset }),
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
    let (cmp_op, cmp_ty) = match op {
        0x11 => (CmpOp::Lt, CmpType::F32),
        0x12 => (CmpOp::Eq, CmpType::F32),
        0x13 => (CmpOp::Le, CmpType::F32),
        0x14 => (CmpOp::Gt, CmpType::F32),
        0x15 => (CmpOp::Ne, CmpType::F32),
        0x16 => (CmpOp::Ge, CmpType::F32),
        0x81 => (CmpOp::Lt, CmpType::I32),
        0x82 => (CmpOp::Eq, CmpType::I32),
        0x83 => (CmpOp::Le, CmpType::I32),
        0x84 => (CmpOp::Gt, CmpType::I32),
        0x85 => (CmpOp::Ne, CmpType::I32),
        0x86 => (CmpOp::Ge, CmpType::I32),
        0xc1 => (CmpOp::Lt, CmpType::U32),
        0xc2 => (CmpOp::Eq, CmpType::U32),
        0xc3 => (CmpOp::Le, CmpType::U32),
        0xc4 => (CmpOp::Gt, CmpType::U32),
        0xc5 => (CmpOp::Ne, CmpType::U32),
        0xc6 => (CmpOp::Ge, CmpType::U32),
        _ => return Err(DecodeError::Unsupported { word, offset }),
    };
    Ok(Inst::VCmp {
        op: cmp_op,
        ty: cmp_ty,
        dst,
        src0,
        src1,
    })
}

/// Mini-encoder helpers for tests & synthetic RDNA2 programs.
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
        [w0, offset & 0x1f_ffff]
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
        (0b1011_1110_1 << 23) | ((dst as u32) << 16) | (0x3c << 8) | (ssrc0 as u32)
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

    pub fn global_load_dword(dst: u8, vaddr: u8, saddr: u8, offset: u16) -> [u32; 2] {
        let w0 = (0b1101_11 << 26) | (0x0c << 18) | (offset as u32 & 0xfff);
        let w1 = ((dst as u32) << 24) | ((saddr as u32) << 16) | (vaddr as u32);
        [w0, w1]
    }

    pub fn global_store_dword(vaddr: u8, vdata: u8, saddr: u8, offset: u16) -> [u32; 2] {
        let w0 = (0b1101_11 << 26) | (0x1c << 18) | (offset as u32 & 0xfff);
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
        let w0 = (0b1111_00 << 26) | (0x66 << 18) | (0xf << 8);
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
