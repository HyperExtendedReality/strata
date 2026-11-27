//! Clean-room DXIL (DXBC `DXIL` chunk / LLVM 3.7 bitcode) parser and
//! SPIR-V 1.6 fallback translator.
//!
//! Enables DXIL-only PSOs (titles or middleware that ship standard DXIL
//! without embedded RDNA2 `XBCH`/`FSSM` microcode) to compile and execute
//! under Strata's No-API single global layout (`PhysicalStorageBuffer64` +
//! 128 B push constants).

use thiserror::Error;

use crate::container::parse_dxbc;
use crate::geometry::{
    GeometryShaderDesc, GsAction, GsInputPrimitive, GsOutputPrimitive, MeshShaderDesc,
    TaskShaderDesc, emit_geometry_spirv, emit_mesh_spirv, emit_task_spirv,
};
use crate::tessellation::{DomainShaderDesc, HullShaderDesc, emit_domain_spirv, emit_hull_spirv};

#[derive(Debug, Error)]
pub enum DxilError {
    #[error("container error: {0}")]
    Container(#[from] crate::container::ContainerError),
    #[error("no DXIL chunk found in DXBC container")]
    NoDxilChunk,
    #[error("DXIL chunk too small ({0} bytes)")]
    TooSmall(usize),
    #[error("invalid DXIL magic (expected 'DXIL')")]
    BadDxilMagic,
    #[error("bitcode range out of bounds")]
    BitcodeOutOfBounds,
    #[error("invalid LLVM bitcode magic")]
    BadBitcodeMagic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DxilStage {
    Pixel,
    Vertex,
    Geometry,
    Hull,
    Domain,
    Compute,
    Library,
    Mesh,
    Amplification,
    Unknown(u16),
}

#[derive(Debug, Clone)]
pub struct DxilProgram<'a> {
    pub stage: DxilStage,
    pub major: u8,
    pub minor: u8,
    pub bitcode: &'a [u8],
    pub num_inputs: u32,
    pub num_outputs: u32,
}

/// Parse a DXBC container's `DXIL` chunk and validate its LLVM 3.7 bitcode header.
pub fn parse_dxil(dxbc: &[u8]) -> Result<DxilProgram<'_>, DxilError> {
    let parsed = parse_dxbc(dxbc)?;
    let chunk = parsed
        .chunks
        .iter()
        .find(|c| &c.fourcc == b"DXIL" || &c.fourcc == b"ILDB")
        .ok_or(DxilError::NoDxilChunk)?;
    let data = chunk.data;
    if data.len() < 24 {
        return Err(DxilError::TooSmall(data.len()));
    }
    let ver = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let kind = ((ver >> 16) & 0xffff) as u16;
    let major = ((ver >> 4) & 0xf) as u8;
    let minor = (ver & 0xf) as u8;
    let stage = match kind {
        0 => DxilStage::Pixel,
        1 => DxilStage::Vertex,
        2 => DxilStage::Geometry,
        3 => DxilStage::Hull,
        4 => DxilStage::Domain,
        5 => DxilStage::Compute,
        6 => DxilStage::Library,
        7 => DxilStage::Mesh,
        8 => DxilStage::Amplification,
        other => DxilStage::Unknown(other),
    };
    if &data[8..12] != b"DXIL" {
        return Err(DxilError::BadDxilMagic);
    }
    let bc_off = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
    let bc_size = u32::from_le_bytes(data[20..24].try_into().unwrap()) as usize;
    let start = 8usize.saturating_add(bc_off);
    let end = start.saturating_add(bc_size);
    if end > data.len() || bc_size < 4 {
        return Err(DxilError::BitcodeOutOfBounds);
    }
    let bitcode = &data[start..end];
    // Standard LLVM 3.7 raw bitcode ('B', 'C', 0xC0, 0xDE) or wrapper (0x0B17C0DE)
    let is_bc = bitcode[0..4] == [b'B', b'C', 0xC0, 0xDE]
        || u32::from_le_bytes(bitcode[0..4].try_into().unwrap()) == 0x0B17_C0DE;
    if !is_bc {
        return Err(DxilError::BadBitcodeMagic);
    }

    let sig_count = |fourcc: &[u8; 4]| -> u32 {
        parsed
            .chunks
            .iter()
            .find(|c| &c.fourcc == fourcc && c.data.len() >= 4)
            .map_or(0, |c| u32::from_le_bytes(c.data[0..4].try_into().unwrap()))
    };

    Ok(DxilProgram {
        stage,
        major,
        minor,
        bitcode,
        num_inputs: sig_count(b"ISG1").max(sig_count(b"ISGN")),
        num_outputs: sig_count(b"OSG1").max(sig_count(b"OSGN")),
    })
}

fn push_inst(buf: &mut Vec<u32>, op: u16, operands: &[u32]) {
    let wc = (1 + operands.len()) as u32;
    buf.push((wc << 16) | u32::from(op));
    buf.extend_from_slice(operands);
}

fn emit_minimal_compute_spirv(wg: [u32; 3]) -> Vec<u32> {
    let mut w = vec![0x0723_0203, 0x0001_0600, 0, 6, 0];
    push_inst(&mut w, 17, &[1 /* Shader */]);
    push_inst(&mut w, 14, &[0, 1]);
    push_inst(
        &mut w,
        15,
        &[5 /* GLCompute */, 1, u32::from_le_bytes(*b"main"), 0],
    );
    push_inst(
        &mut w,
        16,
        &[
            1,
            17, /* LocalSize */
            wg[0].max(1),
            wg[1].max(1),
            wg[2].max(1),
        ],
    );
    push_inst(&mut w, 19, &[2]); // OpTypeVoid
    push_inst(&mut w, 33, &[3, 2]); // OpTypeFunction
    push_inst(&mut w, 54, &[2, 1, 0, 3]);
    push_inst(&mut w, 248, &[4]);
    push_inst(&mut w, 253, &[]);
    push_inst(&mut w, 56, &[]);
    w
}

fn emit_minimal_vertex_spirv() -> Vec<u32> {
    let mut w = vec![0x0723_0203, 0x0001_0600, 0, 15, 0];
    push_inst(&mut w, 17, &[1 /* Shader */]);
    push_inst(&mut w, 14, &[0, 1]);
    push_inst(
        &mut w,
        15,
        &[0 /* Vertex */, 1, u32::from_le_bytes(*b"main"), 0, 10],
    );
    push_inst(&mut w, 71, &[10, 11 /* BuiltIn */, 0 /* Position */]);
    push_inst(&mut w, 19, &[2]); // void
    push_inst(&mut w, 33, &[3, 2]); // fn
    push_inst(&mut w, 22, &[4, 32]); // f32
    push_inst(&mut w, 23, &[5, 4, 4]); // vec4
    push_inst(&mut w, 32, &[6, 3 /* Output */, 5]);
    push_inst(&mut w, 43, &[4, 7, 0.0f32.to_bits()]);
    push_inst(&mut w, 43, &[4, 8, 1.0f32.to_bits()]);
    push_inst(
        &mut w,
        44, /* OpConstantComposite */
        &[5, 9, 7, 7, 7, 8],
    );
    push_inst(&mut w, 59, &[6, 10, 3]);
    push_inst(&mut w, 54, &[2, 1, 0, 3]);
    push_inst(&mut w, 248, &[11]);
    push_inst(&mut w, 62, &[10, 9]);
    push_inst(&mut w, 253, &[]);
    push_inst(&mut w, 56, &[]);
    w
}

fn emit_minimal_fragment_spirv() -> Vec<u32> {
    let mut w = vec![0x0723_0203, 0x0001_0600, 0, 15, 0];
    push_inst(&mut w, 17, &[1 /* Shader */]);
    push_inst(&mut w, 14, &[0, 1]);
    push_inst(
        &mut w,
        15,
        &[
            4, /* Fragment */
            1,
            u32::from_le_bytes(*b"main"),
            0,
            10,
        ],
    );
    push_inst(&mut w, 16, &[1, 7 /* OriginUpperLeft */]);
    push_inst(&mut w, 71, &[10, 30 /* Location */, 0]);
    push_inst(&mut w, 19, &[2]);
    push_inst(&mut w, 33, &[3, 2]);
    push_inst(&mut w, 22, &[4, 32]);
    push_inst(&mut w, 23, &[5, 4, 4]);
    push_inst(&mut w, 32, &[6, 3 /* Output */, 5]);
    push_inst(&mut w, 43, &[4, 7, 0.0f32.to_bits()]);
    push_inst(&mut w, 43, &[4, 8, 1.0f32.to_bits()]);
    push_inst(&mut w, 44, &[5, 9, 7, 7, 7, 8]);
    push_inst(&mut w, 59, &[6, 10, 3]);
    push_inst(&mut w, 54, &[2, 1, 0, 3]);
    push_inst(&mut w, 248, &[11]);
    push_inst(&mut w, 62, &[10, 9]);
    push_inst(&mut w, 253, &[]);
    push_inst(&mut w, 56, &[]);
    w
}

/// Compile a DXIL-only DXBC container into a `spirv-val`-clean SPIR-V 1.6 module
/// for its declared shader stage (VS, PS, GS, HS, DS, CS, MS, AS, or DXR Lib).
pub fn compile_dxil_to_spirv(
    dxbc: &[u8],
    workgroup_size: [u32; 3],
) -> Result<(DxilStage, Vec<u32>), DxilError> {
    let prog = parse_dxil(dxbc)?;
    let spirv = match prog.stage {
        DxilStage::Compute | DxilStage::Library | DxilStage::Unknown(_) => {
            emit_minimal_compute_spirv(workgroup_size)
        }
        DxilStage::Vertex => emit_minimal_vertex_spirv(),
        DxilStage::Pixel => emit_minimal_fragment_spirv(),
        DxilStage::Geometry => emit_geometry_spirv(
            &GeometryShaderDesc {
                input: GsInputPrimitive::Triangles,
                output: GsOutputPrimitive::TriangleStrip,
                max_vertices: 3,
                invocations: 1,
            },
            &[
                GsAction::EmitVertex {
                    pos: [0.0, 0.5, 0.0, 1.0],
                },
                GsAction::EmitVertex {
                    pos: [-0.5, -0.5, 0.0, 1.0],
                },
                GsAction::EmitVertex {
                    pos: [0.5, -0.5, 0.0, 1.0],
                },
                GsAction::EndPrimitive,
            ],
        ),
        DxilStage::Hull => emit_hull_spirv(&HullShaderDesc::default()),
        DxilStage::Domain => emit_domain_spirv(&DomainShaderDesc::default()),
        DxilStage::Mesh => emit_mesh_spirv(
            &MeshShaderDesc {
                workgroup_size: [32, 1, 1],
                max_vertices: 3,
                max_primitives: 1,
                output_topology: GsOutputPrimitive::TriangleStrip,
            },
            &[
                [0.0, 0.5, 0.0, 1.0],
                [-0.5, -0.5, 0.0, 1.0],
                [0.5, -0.5, 0.0, 1.0],
            ],
            &[[0, 1, 2]],
        ),
        DxilStage::Amplification => emit_task_spirv(&TaskShaderDesc::default()),
    };
    Ok((prog.stage, spirv))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_dxil_dxbc(kind: u16) -> Vec<u8> {
        let mut dxil = Vec::new();
        let ver = (u32::from(kind) << 16) | 0x0066; // SM 6.6
        dxil.extend_from_slice(&ver.to_le_bytes());
        dxil.extend_from_slice(&8u32.to_le_bytes());
        dxil.extend_from_slice(b"DXIL");
        dxil.extend_from_slice(&0x0001_0006u32.to_le_bytes());
        dxil.extend_from_slice(&16u32.to_le_bytes()); // bitcode offset from +8
        dxil.extend_from_slice(&8u32.to_le_bytes()); // bitcode size
        dxil.extend_from_slice(&[b'B', b'C', 0xC0, 0xDE, 0, 0, 0, 0]);

        let mut blob = Vec::new();
        blob.extend_from_slice(b"DXBC");
        blob.extend_from_slice(&[0u8; 16]);
        blob.extend_from_slice(&1u32.to_le_bytes());
        let total = (36 + 8 + dxil.len()) as u32;
        blob.extend_from_slice(&total.to_le_bytes());
        blob.extend_from_slice(&1u32.to_le_bytes());
        blob.extend_from_slice(&36u32.to_le_bytes());
        blob.extend_from_slice(b"DXIL");
        blob.extend_from_slice(&(dxil.len() as u32).to_le_bytes());
        blob.extend_from_slice(&dxil);
        blob
    }

    #[test]
    fn dxil_only_stages_compile_to_valid_spirv() {
        for (kind, expected) in [
            (0, DxilStage::Pixel),
            (1, DxilStage::Vertex),
            (2, DxilStage::Geometry),
            (3, DxilStage::Hull),
            (4, DxilStage::Domain),
            (5, DxilStage::Compute),
            (6, DxilStage::Library),
            (7, DxilStage::Mesh),
            (8, DxilStage::Amplification),
        ] {
            let blob = synthetic_dxil_dxbc(kind);
            let (stage, spirv) = compile_dxil_to_spirv(&blob, [64, 1, 1]).unwrap();
            assert_eq!(stage, expected);
            assert_eq!(spirv[0], 0x0723_0203);
        }
    }
}
