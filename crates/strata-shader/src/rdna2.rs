//! Clean-room RDNA2 (GFX10.3 / Xbox Series X|S) instruction decoder & IR.
//!
//! Covers the core encodings needed to decode console compute/graphics shaders
//! into Strata's SSA IR: SOP2, SOP1, SOPK, SOPC, SOPP, VOP2, VOP1, VOPC,
//! SMEM, FLAT/GLOBAL, and `image_bvh64_intersect_ray`.

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
            255 => Self::Literal(literal.unwrap_or(0)),
            256..=511 => Self::Vgpr((code - 256) as u8),
            _ => Self::InlineInt(0),
        }
    }

    pub fn is_vgpr(self) -> bool {
        matches!(self, Self::Vgpr(_))
    }
}

/// Decoded RDNA2 instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inst {
    /// `s_add_u32 sdst, ssrc0, ssrc1`
    SAddU32 { dst: u8, src0: Operand, src1: Operand },
    /// `s_and_b64 sdst[0:1], ssrc0, ssrc1` (often EXEC/VCC mask manipulation)
    SAndB64 { dst: u8, src0: Operand, src1: Operand },
    /// `s_and_saveexec_b64 sdst[0:1], ssrc0`
    SAndSaveExecB64 { dst: u8, src0: Operand },
    /// `s_mov_b32 sdst, ssrc0`
    SMovB32 { dst: u8, src0: Operand },
    /// `s_load_dwordx4 sdst[0:3], sbase[0:1], offset`
    SLoadDword { dst: u8, count: u8, sbase: u8, offset: i32 },
    /// `s_branch rel16`
    SBranch { target_dw: usize },
    /// `s_cbranch_execz rel16`
    SCBranchExecz { target_dw: usize },
    /// `s_cbranch_scc0 / scc1`
    SCBranchScc { on_one: bool, target_dw: usize },
    /// `s_waitcnt` (no-op in SPIR-V structured memory model)
    SWaitcnt,
    /// `s_endpgm`
    SEndpgm,

    /// `v_mov_b32 vdst, src0`
    VMovB32 { dst: u8, src0: Operand },
    /// `v_log_f32 vdst, src0` (exact base-2 log, matching RDNA2 & fixing blurry cube reflections)
    VLogF32 { dst: u8, src0: Operand },
    /// `v_add_f32 vdst, src0, vsrc1`
    VAddF32 { dst: u8, src0: Operand, src1: Operand },
    /// `v_mul_f32` / `v_mul_legacy_f32` (legacy: 0 * x = 0 even for NaN/Inf)
    VMulF32 { dst: u8, src0: Operand, src1: Operand, legacy: bool },
    /// `v_fmac_f32 vdst, src0, vsrc1` (vdst = src0 * src1 + vdst)
    VFmacF32 { dst: u8, src0: Operand, src1: Operand },
    /// `v_add_nc_u32 vdst, src0, vsrc1`
    VAddU32 { dst: u8, src0: Operand, src1: Operand },
    /// `v_lshlrev_b32 vdst, src0, vsrc1`
    VLshlRevB32 { dst: u8, shift: Operand, src: Operand },
    /// `v_cndmask_b32 vdst, src0, vsrc1, vcc`
    VCndMaskB32 { dst: u8, src0: Operand, src1: Operand },
    /// `v_cmp_lt_f32 vcc, src0, vsrc1`
    VCmpLtF32 { src0: Operand, src1: Operand },
    /// `v_readfirstlane_b32 sdst, vsrc`
    VReadFirstLaneB32 { dst: u8, src: u8 },
    /// Intra-2x2-quad lane swap (`ds_swizzle` / DPP quad permute).
    /// Lowered to `OpGroupNonUniformQuadSwap` so helper pixels participate (Belobug fix).
    VQuadSwap { dst: u8, src: u8, direction: u8 },

    /// `global_load_dword{,x2,x4} vdst, vaddr[0:1], saddr, offset`
    GlobalLoad {
        dst: u8,
        count: u8,
        vaddr: u8,
        saddr: Option<u8>,
        offset: i32,
    },
    /// `global_store_dword{,x2,x4} vaddr[0:1], vdata, saddr, offset`
    GlobalStore {
        vaddr: u8,
        vdata: u8,
        count: u8,
        saddr: Option<u8>,
        offset: i32,
    },

    /// `image_bvh64_intersect_ray` — hardware ray/box or ray/triangle node test.
    ImageBvh64IntersectRay {
        dst: u8,
        addr_vgpr: u8,
        rsrc_sgpr: u8,
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
                0x01 => Inst::SEndpgm,
                0x02 => Inst::SBranch { target_dw },
                0x04 => Inst::SCBranchScc { on_one: false, target_dw },
                0x05 => Inst::SCBranchScc { on_one: true, target_dw },
                0x08 => Inst::SCBranchExecz { target_dw },
                0x0c => Inst::SWaitcnt,
                _ => return Err(DecodeError::Unsupported { word: w, offset: pc }),
            }
        } else if (w >> 23) == 0b1011_1110_1 {
            // SOP1
            let op = (w >> 8) & 0xff;
            let dst = ((w >> 16) & 0x7f) as u8;
            let ssrc0_code = (w & 0xff) as u16;
            let lit = if ssrc0_code == 255 {
                len += 1;
                Some(*words.get(pc + 1).ok_or(DecodeError::Truncated { offset: pc })?)
            } else {
                None
            };
            let src0 = Operand::from_src9(ssrc0_code, lit);
            match op {
                0x03 => Inst::SMovB32 { dst, src0 },
                0x3c => Inst::SAndSaveExecB64 { dst, src0 },
                _ => return Err(DecodeError::Unsupported { word: w, offset: pc }),
            }
        } else if (w >> 30) == 0b10 && (w >> 23) != 0b1011_1111_0 {
            // SOP2
            let op = (w >> 23) & 0x7f;
            let dst = ((w >> 16) & 0x7f) as u8;
            let s0 = (w & 0xff) as u16;
            let s1 = ((w >> 8) & 0xff) as u16;
            let lit = if s0 == 255 || s1 == 255 {
                len += 1;
                Some(*words.get(pc + 1).ok_or(DecodeError::Truncated { offset: pc })?)
            } else {
                None
            };
            let src0 = Operand::from_src9(s0, lit);
            let src1 = Operand::from_src9(s1, lit);
            match op {
                0x00 => Inst::SAddU32 { dst, src0, src1 },
                0x0f => Inst::SAndB64 { dst, src0, src1 },
                _ => return Err(DecodeError::Unsupported { word: w, offset: pc }),
            }
        } else if (w >> 25) == 0b0111_111 {
            // VOP1
            let op = (w >> 9) & 0xff;
            let dst = ((w >> 17) & 0xff) as u8;
            let s0 = (w & 0x1ff) as u16;
            let lit = if s0 == 255 {
                len += 1;
                Some(*words.get(pc + 1).ok_or(DecodeError::Truncated { offset: pc })?)
            } else {
                None
            };
            match op {
                0x01 => Inst::VMovB32 { dst, src0: Operand::from_src9(s0, lit) },
                0x02 => Inst::VReadFirstLaneB32 { dst, src: (s0.saturating_sub(256)) as u8 },
                0x27 => Inst::VLogF32 { dst, src0: Operand::from_src9(s0, lit) },
                _ => return Err(DecodeError::Unsupported { word: w, offset: pc }),
            }
        } else if (w >> 25) == 0b0111_110 {
            // VOPC
            let op = (w >> 17) & 0xff;
            let s0 = (w & 0x1ff) as u16;
            let v1 = ((w >> 9) & 0xff) as u8;
            let lit = if s0 == 255 {
                len += 1;
                Some(*words.get(pc + 1).ok_or(DecodeError::Truncated { offset: pc })?)
            } else {
                None
            };
            match op {
                0x11 => Inst::VCmpLtF32 {
                    src0: Operand::from_src9(s0, lit),
                    src1: Operand::Vgpr(v1),
                },
                _ => return Err(DecodeError::Unsupported { word: w, offset: pc }),
            }
        } else if (w >> 31) == 0 {
            // VOP2
            let op = (w >> 25) & 0x3f;
            let dst = ((w >> 17) & 0xff) as u8;
            let v1 = ((w >> 9) & 0xff) as u8;
            let s0 = (w & 0x1ff) as u16;
            let lit = if s0 == 255 {
                len += 1;
                Some(*words.get(pc + 1).ok_or(DecodeError::Truncated { offset: pc })?)
            } else {
                None
            };
            let src0 = Operand::from_src9(s0, lit);
            let src1 = Operand::Vgpr(v1);
            match op {
                0x01 => Inst::VCndMaskB32 { dst, src0, src1 },
                0x03 => Inst::VAddF32 { dst, src0, src1 },
                0x07 => Inst::VMulF32 { dst, src0, src1, legacy: true },
                0x08 => Inst::VMulF32 { dst, src0, src1, legacy: false },
                0x1a => Inst::VLshlRevB32 { dst, shift: src0, src: src1 },
                0x25 => Inst::VAddU32 { dst, src0, src1 },
                0x2b => Inst::VFmacF32 { dst, src0, src1 },
                _ => return Err(DecodeError::Unsupported { word: w, offset: pc }),
            }
        } else if (w >> 26) == 0b1111_01 {
            // SMEM (64-bit encoding)
            len = 2;
            let w1 = *words.get(pc + 1).ok_or(DecodeError::Truncated { offset: pc })?;
            let op = (w >> 18) & 0xff;
            let sbase = ((w & 0x3f) << 1) as u8;
            let dst = ((w >> 6) & 0x7f) as u8;
            let offset = (w1 & 0x1f_ffff) as i32;
            let count = match op {
                0x00 => 1,
                0x01 => 2,
                0x02 => 4,
                _ => return Err(DecodeError::Unsupported { word: w, offset: pc }),
            };
            Inst::SLoadDword { dst, count, sbase, offset }
        } else if (w >> 26) == 0b1101_11 {
            // FLAT / GLOBAL (64-bit encoding)
            len = 2;
            let w1 = *words.get(pc + 1).ok_or(DecodeError::Truncated { offset: pc })?;
            let op = (w >> 18) & 0x7f;
            let offset = (w & 0xfff) as i32;
            let vaddr = (w1 & 0xff) as u8;
            let vdata = ((w1 >> 8) & 0xff) as u8;
            let saddr_raw = ((w1 >> 16) & 0x7f) as u8;
            let saddr = if saddr_raw == 0x7f { None } else { Some(saddr_raw) };
            let vdst = ((w1 >> 24) & 0xff) as u8;
            match op {
                0x0c => Inst::GlobalLoad { dst: vdst, count: 1, vaddr, saddr, offset },
                0x0d => Inst::GlobalLoad { dst: vdst, count: 2, vaddr, saddr, offset },
                0x0e => Inst::GlobalLoad { dst: vdst, count: 4, vaddr, saddr, offset },
                0x1c => Inst::GlobalStore { vaddr, vdata, count: 1, saddr, offset },
                0x1d => Inst::GlobalStore { vaddr, vdata, count: 2, saddr, offset },
                0x1e => Inst::GlobalStore { vaddr, vdata, count: 4, saddr, offset },
                _ => return Err(DecodeError::Unsupported { word: w, offset: pc }),
            }
        } else if (w >> 26) == 0b1111_00 {
            // MIMG (including image_bvh64_intersect_ray)
            len = 2;
            let w1 = *words.get(pc + 1).ok_or(DecodeError::Truncated { offset: pc })?;
            let op = (w >> 18) & 0xff;
            let addr_vgpr = (w1 & 0xff) as u8;
            let dst = ((w1 >> 8) & 0xff) as u8;
            let rsrc_sgpr = (((w1 >> 16) & 0x1f) << 2) as u8;
            match op {
                0x66 => Inst::ImageBvh64IntersectRay { dst, addr_vgpr, rsrc_sgpr },
                _ => return Err(DecodeError::Unsupported { word: w, offset: pc }),
            }
        } else {
            return Err(DecodeError::Unsupported { word: w, offset: pc });
        };

        out.push(DecodedInst { pc, len_dw: len, inst });
        pc += len;
    }

    Ok(out)
}

/// Mini-encoder helpers for tests & synthetic RDNA2 programs.
pub mod encode {
    pub fn s_endpgm() -> u32 {
        (0b1011_1111_1 << 23) | (0x01 << 16)
    }

    pub fn s_waitcnt() -> u32 {
        (0b1011_1111_1 << 23) | (0x0c << 16)
    }

    pub fn s_mov_b32(dst: u8, ssrc0: u8) -> u32 {
        (0b1011_1110_1 << 23) | ((dst as u32) << 16) | (0x03 << 8) | (ssrc0 as u32)
    }

    pub fn v_mov_b32_vgpr(dst: u8, src_vgpr: u8) -> u32 {
        (0b0111_111 << 25) | ((dst as u32) << 17) | (0x01 << 9) | (256 + src_vgpr as u32)
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

    pub fn v_lshlrev_b32_imm(dst: u8, shift_imm: u8, src_vgpr: u8) -> u32 {
        (0x1a << 25) | ((dst as u32) << 17) | ((src_vgpr as u32) << 9) | (128 + shift_imm as u32)
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
}
