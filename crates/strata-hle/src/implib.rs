//! Clean-room COFF `.lib` (`!<arch>\n`) short-import archive parser.
//!
//! Lets `strata-cli inspect` list imported DLL names and symbols from Windows
//! / console import libraries without external dependencies.

const ARCH_MAGIC: &[u8; 8] = b"!<arch>\n";

/// One short-import entry extracted from a COFF `.lib` archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplibEntry {
    pub dll: String,
    pub symbol: String,
    pub ordinal_or_hint: u16,
    pub by_ordinal: bool,
}

/// Returns `true` if `bytes` starts with the standard COFF/ar `!<arch>\n` magic.
pub fn is_coff_archive(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[0..8] == ARCH_MAGIC
}

/// Parse short-import members (`Sig1 == 0x0000`, `Sig2 == 0xFFFF`) from a COFF `.lib`.
pub fn parse_implib(bytes: &[u8]) -> Option<Vec<ImplibEntry>> {
    if !is_coff_archive(bytes) {
        return None;
    }
    let mut entries = Vec::new();
    let mut pos = 8usize;
    while pos + 60 <= bytes.len() {
        let hdr = &bytes[pos..pos + 60];
        if &hdr[58..60] != b"`\n" {
            break;
        }
        let size_str = std::str::from_utf8(&hdr[48..58]).ok()?.trim();
        let size = size_str.parse::<usize>().ok()?;
        let data_start = pos + 60;
        let data_end = data_start.checked_add(size)?;
        if data_end > bytes.len() {
            break;
        }
        let member = &bytes[data_start..data_end];
        if member.len() >= 20 {
            let sig1 = u16::from_le_bytes([member[0], member[1]]);
            let sig2 = u16::from_le_bytes([member[2], member[3]]);
            if sig1 == 0x0000 && sig2 == 0xffff {
                let ordinal_or_hint = u16::from_le_bytes([member[16], member[17]]);
                let flags = u16::from_le_bytes([member[18], member[19]]);
                let name_type = (flags >> 2) & 0x7;
                let by_ordinal = name_type == 0;
                let strings = &member[20..];
                if let Some(nul1) = strings.iter().position(|&b| b == 0) {
                    let symbol = String::from_utf8_lossy(&strings[..nul1]).into_owned();
                    let rest = &strings[nul1 + 1..];
                    let nul2 = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
                    let dll = String::from_utf8_lossy(&rest[..nul2]).into_owned();
                    entries.push(ImplibEntry {
                        dll,
                        symbol,
                        ordinal_or_hint,
                        by_ordinal,
                    });
                }
            }
        }
        pos = data_end + (data_end & 1);
    }
    Some(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_synthetic_short_import_lib() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&0x0000u16.to_le_bytes()); // Sig1
        payload.extend_from_slice(&0xffffu16.to_le_bytes()); // Sig2
        payload.extend_from_slice(&0u16.to_le_bytes()); // Version
        payload.extend_from_slice(&0x8664u16.to_le_bytes()); // AMD64
        payload.extend_from_slice(&0u32.to_le_bytes()); // TimeDateStamp
        payload.extend_from_slice(&0u32.to_le_bytes()); // SizeOfData
        payload.extend_from_slice(&42u16.to_le_bytes()); // Hint
        payload.extend_from_slice(&(1u16 << 2).to_le_bytes()); // NameType = IMPORT_NAME
        payload.extend_from_slice(b"XMemVirtualAlloc\0xmem.dll\0");

        let mut lib = Vec::new();
        lib.extend_from_slice(ARCH_MAGIC);
        let mut hdr = [b' '; 60];
        hdr[0..8].copy_from_slice(b"xmem.dll");
        let sz = format!("{:<10}", payload.len());
        hdr[48..58].copy_from_slice(sz.as_bytes());
        hdr[58..60].copy_from_slice(b"`\n");
        lib.extend_from_slice(&hdr);
        lib.extend_from_slice(&payload);

        assert!(is_coff_archive(&lib));
        let entries = parse_implib(&lib).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].dll, "xmem.dll");
        assert_eq!(entries[0].symbol, "XMemVirtualAlloc");
        assert_eq!(entries[0].ordinal_or_hint, 42);
        assert!(!entries[0].by_ordinal);
    }
}
