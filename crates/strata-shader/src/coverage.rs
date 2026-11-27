//! Translation coverage: run every shader found in some files through the
//! compiler and count what it cannot handle yet, by reason. This ranks what
//! to implement next by how many shaders it would unblock.
//!
//! Programs are found by scanning for `XBCH` (zlib-packed) and `FSSM`
//! containers anywhere in a file; files named `*.rdna2` or `*.bin` that hold
//! neither are taken as raw RDNA2 code.

use std::collections::BTreeMap;

use crate::container::{ShaderStage, parse_fssm, unpack_xbch};
use crate::rdna2::{DecodeError, decode_program};
use crate::resource::{DescriptorUse, Origin, trace_descriptors};
use crate::ssa::LowerError;

/// One RDNA2 program and where it was found.
#[derive(Debug, Clone)]
pub struct Program {
    pub source: String,
    pub stage: Option<ShaderStage>,
    pub words: Vec<u32>,
}

fn words_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn find_all(hay: &[u8], needle: &[u8; 4]) -> Vec<usize> {
    hay.windows(4)
        .enumerate()
        .filter(|(_, w)| w == needle)
        .map(|(i, _)| i)
        .collect()
}

/// Every program in a file's bytes (supports `SGD5` archives, `XBCH`, `FSSM`,
/// and raw `.rdna2`/`.bin` dumps).
pub fn extract_programs(name: &str, bytes: &[u8]) -> Vec<Program> {
    let mut out = Vec::new();
    let is_sgd5 = bytes.len() >= 8 && &bytes[0..4] == b"SGD5";
    let tag_prefix = if is_sgd5 { "SGD5/" } else { "" };

    for off in find_all(bytes, b"XBCH") {
        if let Ok(fssm) = unpack_xbch(&bytes[off..])
            && let Ok(p) = parse_fssm(&fssm)
        {
            out.push(Program {
                source: format!("{name}@0x{off:x} ({tag_prefix}XBCH)"),
                stage: Some(p.stage),
                words: words_of(p.code_dwords),
            });
        }
    }
    for off in find_all(bytes, b"FSSM") {
        if let Ok(p) = parse_fssm(&bytes[off..]) {
            out.push(Program {
                source: format!("{name}@0x{off:x} ({tag_prefix}FSSM)"),
                stage: Some(p.stage),
                words: words_of(p.code_dwords),
            });
        }
    }
    let lower = name.to_ascii_lowercase();
    if out.is_empty()
        && (lower.ends_with(".rdna2") || lower.ends_with(".bin"))
        && bytes.len().is_multiple_of(4)
    {
        out.push(Program {
            source: name.to_owned(),
            stage: None,
            words: words_of(bytes),
        });
    }
    out
}

/// Opcode/encoding histogram produced by `strata-cli census`.
#[derive(Debug, Default, Clone)]
pub struct Census {
    pub programs: usize,
    pub total_dwords: usize,
    pub encodings: BTreeMap<String, usize>,
    pub stages: BTreeMap<String, usize>,
}

impl Census {
    pub fn add(&mut self, program: &Program) {
        self.programs += 1;
        self.total_dwords += program.words.len();
        let stage = match program.stage {
            Some(s) => format!("{s:?}"),
            None => "Unknown".to_string(),
        };
        *self.stages.entry(stage).or_insert(0) += 1;
        for &w in &program.words {
            *self.encodings.entry(encoding_of(w)).or_insert(0) += 1;
        }
    }

    pub fn top_encodings(&self, limit: usize) -> Vec<(String, usize)> {
        let mut pairs: Vec<_> = self
            .encodings
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pairs.truncate(limit);
        pairs
    }
}

/// The RDNA2 encoding class and opcode of an instruction word, as the
/// decoder tells them apart.
pub fn encoding_of(w: u32) -> String {
    let (class, op) = if w >> 23 == 0b1_0111_1111 {
        ("SOPP", (w >> 16) & 0x7f)
    } else if w >> 23 == 0b1_0111_1110 {
        ("SOPC", (w >> 16) & 0x7f)
    } else if w >> 23 == 0b1_0111_1101 {
        ("SOP1", (w >> 8) & 0xff)
    } else if w >> 28 == 0b1011 {
        ("SOPK", (w >> 23) & 0x1f)
    } else if w >> 30 == 0b10 {
        ("SOP2", (w >> 23) & 0x7f)
    } else if w >> 25 == 0b011_1111 {
        ("VOP1", (w >> 9) & 0xff)
    } else if w >> 25 == 0b011_1110 {
        ("VOPC", (w >> 17) & 0xff)
    } else if w >> 31 == 0 {
        ("VOP2", (w >> 25) & 0x3f)
    } else {
        match w >> 26 {
            0b11_0101 => ("VOP3", (w >> 16) & 0x3ff),
            0b11_0011 => ("VOP3P", (w >> 16) & 0x7f),
            0b11_0110 => ("DS", (w >> 18) & 0xff),
            0b11_0111 => ("FLAT", (w >> 18) & 0x7f),
            0b11_1000 => ("MUBUF", (w >> 18) & 0x7f),
            0b11_1010 => ("MTBUF", (w >> 16) & 0x7),
            0b11_1100 => ("MIMG", (w >> 18) & 0x7f),
            0b11_1101 => ("SMEM", (w >> 18) & 0xff),
            0b11_1110 => ("EXP", 0),
            _ => ("?", w >> 26),
        }
    };
    format!("{class} op 0x{op:x}")
}

/// What happened to one program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Translated {
        spirv_words: usize,
    },
    /// The reason, worded so equal causes group together.
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Report {
    pub source: String,
    pub stage: Option<ShaderStage>,
    pub dwords: usize,
    pub outcome: Outcome,
    pub descriptors: Vec<DescriptorUse>,
}

/// Decodes, traces and compiles one program.
pub fn translate(program: &Program) -> Report {
    let base = |outcome, descriptors| Report {
        source: program.source.clone(),
        stage: program.stage,
        dwords: program.words.len(),
        outcome,
        descriptors,
    };
    let decoded = match decode_program(&program.words) {
        Ok(d) => d,
        Err(DecodeError::Unsupported { word, .. }) => {
            return base(
                Outcome::Failed(format!("decode: {}", encoding_of(word))),
                Vec::new(),
            );
        }
        Err(DecodeError::Truncated { .. }) => {
            return base(Outcome::Failed("decode: truncated".into()), Vec::new());
        }
    };
    let descriptors = trace_descriptors(&decoded);
    // Vertex and pixel programs build for their stage; the rest as compute.
    let stage = match program.stage {
        Some(ShaderStage::Vertex) => crate::ssa::Stage::Vertex,
        Some(ShaderStage::Pixel) => crate::ssa::Stage::Pixel,
        _ => crate::ssa::Stage::Compute,
    };
    let opts = crate::ssa::BuildOptions {
        stage,
        ..Default::default()
    };
    let outcome = match crate::compile_rdna2_with(&program.words, [64, 1, 1], &opts) {
        Ok(c) => Outcome::Translated {
            spirv_words: c.spirv.len(),
        },
        Err(crate::CompileError::Lower(LowerError::Unsupported { desc, .. })) => {
            Outcome::Failed(format!("lower: {desc}"))
        }
        Err(crate::CompileError::Lower(LowerError::UnsupportedBranch { .. })) => {
            Outcome::Failed("lower: break out of a loop or irreducible branch".into())
        }
        Err(e) => Outcome::Failed(e.to_string()),
    };
    base(outcome, descriptors)
}

/// Totals over many reports.
#[derive(Debug, Default, Clone)]
pub struct Summary {
    pub programs: usize,
    pub translated: usize,
    /// Failure reason -> programs.
    pub reasons: BTreeMap<String, usize>,
    /// Descriptor origin class (`user`, `table`, `nested table`, `waterfall`, `unknown`) -> uses.
    pub descriptor_origins: BTreeMap<&'static str, usize>,
}

impl Summary {
    pub fn add(&mut self, r: &Report) {
        self.programs += 1;
        match &r.outcome {
            Outcome::Translated { .. } => self.translated += 1,
            Outcome::Failed(why) => *self.reasons.entry(why.clone()).or_default() += 1,
        }
        for d in &r.descriptors {
            let class = match (&d.resource, d.table_slot()) {
                (_, Some(_)) => "table",
                (Origin::User(_), _) => "user",
                (Origin::Load { .. }, _) => "nested table",
                (Origin::Waterfall, _) => "waterfall",
                _ => "unknown",
            };
            *self.descriptor_origins.entry(class).or_default() += 1;
        }
    }

    /// Failure reasons, most common first.
    pub fn ranked(&self) -> Vec<(&str, usize)> {
        let mut v: Vec<(&str, usize)> =
            self.reasons.iter().map(|(k, &n)| (k.as_str(), n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna2::encode::*;

    fn raw(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn raw_programs_translate_or_name_their_blocker() {
        let ok = raw(&crate::test_kernels::uniform_do_while_sum());
        let mut bad = vec![v_mov_b32_sgpr(0, 128)];
        bad.extend_from_slice(&exp_mrt0([0, 1, 2, 3], true));
        bad.push(s_endpgm());
        let bad = raw(&bad);

        let mut summary = Summary::default();
        for (name, bytes) in [("a.rdna2", &ok), ("b.bin", &bad)] {
            for p in extract_programs(name, bytes) {
                summary.add(&translate(&p));
            }
        }
        assert_eq!((summary.programs, summary.translated), (2, 1));
        assert_eq!(
            summary.ranked()[0].0,
            "lower: EXP in a compute shader",
            "a raw program has no stage: compute"
        );
        assert!(
            extract_programs("c.txt", &ok).is_empty(),
            "unknown file kinds are not guessed at"
        );
    }

    #[test]
    fn encodings_are_named_like_the_decoder_sees_them() {
        assert_eq!(encoding_of(s_endpgm()), "SOPP op 0x1");
        assert_eq!(encoding_of(exp_mrt0([0; 4], true)[0]), "EXP op 0x0");
        assert_eq!(encoding_of(ds_add_u32(0, 0, 0)[0]), "DS op 0x0");
        assert_eq!(encoding_of(s_load_dwordx4(0, 0, 0)[0]), "SMEM op 0x2");
    }

    #[test]
    fn fssm_containers_are_found_inside_other_bytes() {
        let code = raw(&crate::test_kernels::while_loop_sum());
        let mut fssm = vec![0u8; 0x108];
        fssm[0..4].copy_from_slice(b"FSSM");
        fssm[0x101] = 3; // compute
        let code_rel = (0x108 - 0x10) as u64;
        fssm[0xC0..0xC8].copy_from_slice(&code_rel.to_le_bytes());
        fssm[0xC8..0xD0].copy_from_slice(&(code.len() as u64).to_le_bytes());
        fssm.extend_from_slice(&code);
        let mut file = b"junk before the container".to_vec();
        file.extend_from_slice(&fssm);

        let programs = extract_programs("pack.dat", &file);
        assert_eq!(programs.len(), 1);
        assert_eq!(programs[0].stage, Some(ShaderStage::Compute));
        assert!(matches!(
            translate(&programs[0]).outcome,
            Outcome::Translated { .. }
        ));
    }
}
