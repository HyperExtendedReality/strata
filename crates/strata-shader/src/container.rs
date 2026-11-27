//! Console shader container parser (`DXBC` -> `XDAT` -> `XBCH` -> `FSSM`).
//!
//! Clean-room parser for inspecting Xbox Series X|S shader blobs without
//! needing proprietary SDK headers.

use thiserror::Error;

use crate::inflate::{InflateError, inflate_zlib};

#[derive(Debug, Error)]
pub enum ContainerError {
    #[error("container too small ({len} bytes)")]
    TooSmall { len: usize },
    #[error("invalid DXBC magic (expected 'DXBC')")]
    BadMagic,
    #[error("chunk offset {offset} out of bounds")]
    OutOfBounds { offset: usize },
    #[error("XBCH zlib decompression failed: {0}")]
    Inflate(#[from] InflateError),
}

#[derive(Debug, Clone)]
pub struct DxbcChunk<'a> {
    pub fourcc: [u8; 4],
    pub data: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct ParsedContainer<'a> {
    pub digest: [u8; 16],
    pub chunks: Vec<DxbcChunk<'a>>,
    pub has_xdat: bool,
    pub shader_model: Option<(u8, u8)>,
}

/// Parse a DXBC container and inspect its chunks (SFI0, ISG1, OSG1, PSV0, DXIL/XDAT).
pub fn parse_dxbc(bytes: &[u8]) -> Result<ParsedContainer<'_>, ContainerError> {
    if bytes.len() < 32 {
        return Err(ContainerError::TooSmall { len: bytes.len() });
    }
    if &bytes[0..4] != b"DXBC" {
        return Err(ContainerError::BadMagic);
    }

    let mut digest = [0u8; 16];
    digest.copy_from_slice(&bytes[4..20]);
    let chunk_count = u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize;

    let table_end = 32usize.saturating_add(chunk_count.saturating_mul(4));
    if table_end > bytes.len() {
        return Err(ContainerError::OutOfBounds { offset: table_end });
    }

    let mut chunks = Vec::with_capacity(chunk_count);
    let mut has_xdat = false;
    let mut shader_model = None;

    for i in 0..chunk_count {
        let off = u32::from_le_bytes(bytes[32 + i * 4..36 + i * 4].try_into().unwrap()) as usize;
        if off + 8 > bytes.len() {
            return Err(ContainerError::OutOfBounds { offset: off });
        }
        let fourcc: [u8; 4] = bytes[off..off + 4].try_into().unwrap();
        let size = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        if off + 8 + size > bytes.len() {
            return Err(ContainerError::OutOfBounds {
                offset: off + 8 + size,
            });
        }
        let data = &bytes[off + 8..off + 8 + size];

        if &fourcc == b"DXIL" && data.len() >= 28 {
            let ver = u32::from_le_bytes(data[0..4].try_into().unwrap());
            shader_model = Some((((ver >> 4) & 0xf) as u8, (ver & 0xf) as u8));
            if data.len() >= 28 && &data[24..28] == b"XDAT" {
                has_xdat = true;
            }
        }

        chunks.push(DxbcChunk { fourcc, data });
    }

    Ok(ParsedContainer {
        digest,
        chunks,
        has_xdat,
        shader_model,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Pixel,
    Geometry,
    Compute,
    Hull,
    Domain,
    Mesh,
    Amplification,
    Library,
    Unknown(u8),
}

#[derive(Debug, Clone)]
pub struct FssmProgram<'a> {
    pub stage: ShaderStage,
    pub code_dwords: &'a [u8],
}

/// Parse an unpacked `FSSM` (v0x101) console microcode block into its API stage
/// and raw RDNA2 machine-code slice.
pub fn parse_fssm(bytes: &[u8]) -> Result<FssmProgram<'_>, ContainerError> {
    if bytes.len() < 0x108 || &bytes[0..4] != b"FSSM" {
        return Err(ContainerError::BadMagic);
    }
    let code_rel = u64::from_le_bytes(bytes[0xC0..0xC8].try_into().unwrap()) as usize;
    let code_size = u64::from_le_bytes(bytes[0xC8..0xD0].try_into().unwrap()) as usize;
    let code_start = 0x10usize.saturating_add(code_rel);
    if code_start + code_size > bytes.len() {
        return Err(ContainerError::OutOfBounds {
            offset: code_start + code_size,
        });
    }
    // Stage descriptor byte at +0x101 (AA = API stage: 0=VS, 1=PS, 2=GS, 3=CS, 4=HS, 5=DS, 6=MS, 7=AS, 8=LIB)
    let api_stage = bytes[0x101];
    let stage = match api_stage {
        0 => ShaderStage::Vertex,
        1 => ShaderStage::Pixel,
        2 => ShaderStage::Geometry,
        3 => ShaderStage::Compute,
        4 => ShaderStage::Hull,
        5 => ShaderStage::Domain,
        6 => ShaderStage::Mesh,
        7 => ShaderStage::Amplification,
        8 => ShaderStage::Library,
        other => ShaderStage::Unknown(other),
    };
    Ok(FssmProgram {
        stage,
        code_dwords: &bytes[code_start..code_start + code_size],
    })
}

/// Unpack an `XBCH` v3 zlib-compressed container (`+0x18` = zlib stream) into raw `FSSM` bytes.
pub fn unpack_xbch(xbch_bytes: &[u8]) -> Result<Vec<u8>, ContainerError> {
    if xbch_bytes.len() < 0x1E || &xbch_bytes[0..4] != b"XBCH" {
        return Err(ContainerError::BadMagic);
    }
    let comp_size = u64::from_le_bytes(xbch_bytes[8..16].try_into().unwrap()) as usize;
    let end = 0x18usize.saturating_add(comp_size).min(xbch_bytes.len());
    Ok(inflate_zlib(&xbch_bytes[0x18..end])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_synthetic_dxbc_with_xdat() {
        let mut blob = Vec::new();
        blob.extend_from_slice(b"DXBC");
        blob.extend_from_slice(&[0xAA; 16]); // digest
        blob.extend_from_slice(&1u32.to_le_bytes()); // version
        blob.extend_from_slice(&0u32.to_le_bytes()); // total size (patched below)
        blob.extend_from_slice(&1u32.to_le_bytes()); // 1 chunk
        blob.extend_from_slice(&36u32.to_le_bytes()); // chunk 0 offset

        // DXIL chunk with SM 6.6 + XDAT marker at +24
        let mut dxil = vec![0u8; 32];
        dxil[0..4].copy_from_slice(&0x0000_0066u32.to_le_bytes());
        dxil[24..28].copy_from_slice(b"XDAT");

        blob.extend_from_slice(b"DXIL");
        blob.extend_from_slice(&(dxil.len() as u32).to_le_bytes());
        blob.extend_from_slice(&dxil);
        let total = blob.len() as u32;
        blob[24..28].copy_from_slice(&total.to_le_bytes());

        let parsed = parse_dxbc(&blob).unwrap();
        assert!(parsed.has_xdat);
        assert_eq!(parsed.shader_model, Some((6, 6)));
        assert_eq!(parsed.chunks.len(), 1);
    }
}
