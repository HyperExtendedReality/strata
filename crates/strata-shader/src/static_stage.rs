//! Register semantics shared by the straight-line static stage translators.
//! Registers that depend on the title's stage inputs start unknown.

use crate::rdna2::{DecodeError, Inst, Operand};

pub(crate) struct Registers {
    vgpr: [Option<u32>; 256],
    sgpr: [Option<u32>; 128],
}

impl Registers {
    pub(crate) fn new() -> Self {
        Self {
            vgpr: [None; 256],
            sgpr: [None; 128],
        }
    }

    fn source(&self, operand: Operand) -> Option<u32> {
        match operand {
            Operand::InlineFloat(bits) | Operand::Literal(bits) => Some(bits),
            Operand::InlineInt(value) => Some(value as u32),
            Operand::Vgpr(index) => self.vgpr[usize::from(index)],
            Operand::Sgpr(index) => self.sgpr[usize::from(index)],
            _ => None,
        }
    }

    fn float(&self, a: Operand) -> Option<f32> {
        let value = f32::from_bits(self.source(a)?);
        (value.is_normal() || value == 0.0).then_some(value)
    }

    pub(crate) fn execute(&mut self, inst: &Inst) -> bool {
        let (dst, value) = match *inst {
            Inst::SNop | Inst::SWaitcnt => return true,
            Inst::SMovB32 { dst, src0 } => {
                self.sgpr[usize::from(dst)] = self.source(src0);
                return true;
            }
            Inst::VMovB32 { dst, src0 } => (dst, self.source(src0)),
            Inst::VAddF32 { dst, src0, src1 }
            | Inst::VSubF32 { dst, src0, src1 }
            | Inst::VMulF32 {
                dst,
                src0,
                src1,
                legacy: false,
            } => {
                let value = self.float(src0).zip(self.float(src1)).and_then(|(a, b)| {
                    let value = match inst {
                        Inst::VAddF32 { .. } => a + b,
                        Inst::VSubF32 { .. } => a - b,
                        _ => a * b,
                    };
                    (value.is_normal() || value == 0.0).then_some(value.to_bits())
                });
                (dst, value)
            }
            Inst::VCvtF32U32 { dst, src0 } => {
                (dst, self.source(src0).map(|x| (x as f32).to_bits()))
            }
            Inst::VCvtF32I32 { dst, src0 } => (
                dst,
                self.source(src0).map(|x| ((x as i32) as f32).to_bits()),
            ),
            _ => return false,
        };
        self.vgpr[usize::from(dst)] = value;
        true
    }

    pub(crate) fn position(&self, en: u8, src: [u8; 4], pos: &mut [Option<u32>; 4]) {
        for (axis, source) in src.into_iter().enumerate() {
            if en & (1 << axis) != 0 {
                pos[axis] = self.vgpr[usize::from(source)];
            }
        }
    }

    pub(crate) fn vgpr(&self, index: u8) -> Option<u32> {
        self.vgpr[usize::from(index)]
    }
}

pub(crate) fn unsupported(words: &[u32], offset: usize) -> DecodeError {
    DecodeError::Unsupported {
        word: words.get(offset).copied().unwrap_or(0),
        offset,
    }
}

pub(crate) fn full_position(pos: [Option<u32>; 4]) -> Option<[f32; 4]> {
    let [Some(x), Some(y), Some(z), Some(w)] = pos else {
        return None;
    };
    Some([x, y, z, w].map(f32::from_bits))
}
