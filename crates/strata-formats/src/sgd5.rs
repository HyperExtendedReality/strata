//! Clean-room `SGD5` shader archive scanner and extractor.
//!
//! `SGD5` archives bundle multiple console shader containers (`XBCH`, `FSSM`,
//! `DXBC`) inside a single binary file. This module parses the `SGD5` header
//! and extracts every embedded shader container along with its byte offset
//! and payload slice.

pub const SGD5_MAGIC: &[u8; 4] = b"SGD5";

/// Kind of shader container discovered inside an `SGD5` archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddedContainerKind {
    Xbch,
    Fssm,
    Dxbc,
}

/// One shader container entry extracted from an `SGD5` archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sgd5Entry<'a> {
    pub index: usize,
    pub offset: usize,
    pub kind: EmbeddedContainerKind,
    pub payload: &'a [u8],
}

/// Parsed `SGD5` archive header + discovered shader entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sgd5Archive<'a> {
    pub version: u32,
    pub entries: Vec<Sgd5Entry<'a>>,
}

/// Returns `true` if `bytes` begins with the `SGD5` archive magic.
pub fn is_sgd5(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[0..4] == SGD5_MAGIC
}

/// Parse an `SGD5` archive and return all embedded `XBCH` / `FSSM` / `DXBC`
/// container slices in offset order.
pub fn parse_sgd5(bytes: &[u8]) -> Option<Sgd5Archive<'_>> {
    if !is_sgd5(bytes) {
        return None;
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);

    let mut offsets: Vec<(usize, EmbeddedContainerKind)> = Vec::new();
    let mut i = 8usize;
    while i + 4 <= bytes.len() {
        let tag = &bytes[i..i + 4];
        if tag == b"XBCH" {
            offsets.push((i, EmbeddedContainerKind::Xbch));
            i += 4;
        } else if tag == b"FSSM" {
            offsets.push((i, EmbeddedContainerKind::Fssm));
            i += 4;
        } else if tag == b"DXBC" {
            offsets.push((i, EmbeddedContainerKind::Dxbc));
            i += 4;
        } else {
            i += 1;
        }
    }

    let mut entries = Vec::with_capacity(offsets.len());
    for (idx, &(start, kind)) in offsets.iter().enumerate() {
        let end = offsets
            .get(idx + 1)
            .map(|&(next, _)| next)
            .unwrap_or(bytes.len());
        entries.push(Sgd5Entry {
            index: idx,
            offset: start,
            kind,
            payload: &bytes[start..end],
        });
    }

    Some(Sgd5Archive { version, entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_synthetic_sgd5_archive() {
        let mut blob = Vec::new();
        blob.extend_from_slice(b"SGD5");
        blob.extend_from_slice(&5u32.to_le_bytes());
        blob.extend_from_slice(&[0u8; 8]);
        let off0 = blob.len();
        blob.extend_from_slice(b"XBCH");
        blob.extend_from_slice(&[1, 2, 3, 4]);
        let off1 = blob.len();
        blob.extend_from_slice(b"FSSM");
        blob.extend_from_slice(&[5, 6, 7, 8, 9]);

        assert!(is_sgd5(&blob));
        let arc = parse_sgd5(&blob).unwrap();
        assert_eq!(arc.version, 5);
        assert_eq!(arc.entries.len(), 2);
        assert_eq!(arc.entries[0].offset, off0);
        assert_eq!(arc.entries[0].kind, EmbeddedContainerKind::Xbch);
        assert_eq!(arc.entries[0].payload.len(), off1 - off0);
        assert_eq!(arc.entries[1].offset, off1);
        assert_eq!(arc.entries[1].kind, EmbeddedContainerKind::Fssm);
        assert_eq!(arc.entries[1].payload.len(), 9);
    }

    #[test]
    fn rejects_non_sgd5() {
        assert!(!is_sgd5(b"DXBC\0\0\0\0"));
        assert!(parse_sgd5(b"DXBC\0\0\0\0").is_none());
    }
}
