//! Clean-room PE32+ (x86-64) executable & import-table parser (zero external crates).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PeError {
    #[error("file too small for PE32+ headers ({len} bytes)")]
    TooSmall { len: usize },
    #[error("missing MZ DOS signature")]
    BadDosMagic,
    #[error("missing PE\\0\\0 signature")]
    BadPeMagic,
    #[error("not a PE32+ (64-bit) image (magic=0x{magic:04x})")]
    NotPe64 { magic: u16 },
}

#[derive(Debug, Clone)]
pub struct Section {
    pub name: String,
    pub virtual_addr: u32,
    pub virtual_size: u32,
    pub raw_offset: u32,
    pub raw_size: u32,
    pub characteristics: u32,
}

#[derive(Debug, Clone)]
pub struct ImportDll {
    pub dll_name: String,
    pub first_thunk_rva: u32,
    pub functions: Vec<String>,
    /// Delay-load import (`IMAGE_DELAYLOAD_DESCRIPTOR`): RVA of the slot the
    /// delay helper keeps the module handle in. `None` for ordinary imports.
    pub delay_module_handle_rva: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct ExportFunc {
    pub name: String,
    pub ordinal: u16,
    pub rva: u32,
}

#[derive(Debug, Clone)]
pub struct Pe64Info {
    /// `IMAGE_FILE_DLL` is set: the entry point is `DllMain`.
    pub is_dll: bool,
    pub image_base: u64,
    pub entry_rva: u32,
    pub size_of_image: u32,
    pub size_of_headers: u32,
    pub reloc_dir_rva: u32,
    pub reloc_dir_size: u32,
    pub tls_dir_rva: u32,
    pub tls_dir_size: u32,
    pub sections: Vec<Section>,
    pub imports: Vec<ImportDll>,
    pub exports: Vec<ExportFunc>,
}

impl Pe64Info {
    pub fn total_import_count(&self) -> usize {
        self.imports.iter().map(|d| d.functions.len()).sum()
    }

    /// RVA of an export named `func`, or `ordinal#N`.
    pub fn export_rva(&self, func: &str) -> Option<u32> {
        if let Some(ord) = func.strip_prefix("ordinal#") {
            let ord: u16 = ord.parse().ok()?;
            return self
                .exports
                .iter()
                .find(|e| e.ordinal == ord)
                .map(|e| e.rva);
        }
        self.exports.iter().find(|e| e.name == func).map(|e| e.rva)
    }
}

pub fn parse_pe64(bytes: &[u8]) -> Result<Pe64Info, PeError> {
    if bytes.len() < 0x100 {
        return Err(PeError::TooSmall { len: bytes.len() });
    }
    if &bytes[0..2] != b"MZ" {
        return Err(PeError::BadDosMagic);
    }

    let pe_off = u32_at(bytes, 0x3c) as usize;
    if pe_off + 24 + 0x70 > bytes.len() {
        return Err(PeError::TooSmall { len: bytes.len() });
    }
    if &bytes[pe_off..pe_off + 4] != b"PE\0\0" {
        return Err(PeError::BadPeMagic);
    }

    let num_sections = u16_at(bytes, pe_off + 6) as usize;
    let is_dll = u16_at(bytes, pe_off + 22) & 0x2000 != 0;
    let opt_size = u16_at(bytes, pe_off + 20) as usize;
    let opt_off = pe_off + 24;
    let magic = u16_at(bytes, opt_off);
    if magic != 0x20b {
        return Err(PeError::NotPe64 { magic });
    }

    let entry_rva = u32_at(bytes, opt_off + 16);
    let image_base = u64_at(bytes, opt_off + 24);
    let size_of_image = u32_at(bytes, opt_off + 56);
    let size_of_headers = u32_at(bytes, opt_off + 60);
    let num_rva_and_sizes = u32_at(bytes, opt_off + 108) as usize;

    let export_dir_rva = if num_rva_and_sizes >= 1 {
        u32_at(bytes, opt_off + 112)
    } else {
        0
    };
    let import_dir_rva = if num_rva_and_sizes >= 2 {
        u32_at(bytes, opt_off + 112 + 8)
    } else {
        0
    };
    let (reloc_dir_rva, reloc_dir_size) = if num_rva_and_sizes >= 6 {
        (
            u32_at(bytes, opt_off + 112 + 5 * 8),
            u32_at(bytes, opt_off + 112 + 5 * 8 + 4),
        )
    } else {
        (0, 0)
    };
    let (tls_dir_rva, tls_dir_size) = if num_rva_and_sizes >= 10 {
        (
            u32_at(bytes, opt_off + 112 + 9 * 8),
            u32_at(bytes, opt_off + 112 + 9 * 8 + 4),
        )
    } else {
        (0, 0)
    };

    let delay_dir_rva = if num_rva_and_sizes >= 14 {
        u32_at(bytes, opt_off + 112 + 13 * 8)
    } else {
        0
    };

    let sec_table_off = opt_off + opt_size;
    let mut sections = Vec::with_capacity(num_sections);
    for i in 0..num_sections {
        let off = sec_table_off + i * 40;
        if off + 40 > bytes.len() {
            break;
        }
        let raw_name = &bytes[off..off + 8];
        let name =
            String::from_utf8_lossy(raw_name.split(|&b| b == 0).next().unwrap_or(&[])).into_owned();
        sections.push(Section {
            name,
            virtual_size: u32_at(bytes, off + 8),
            virtual_addr: u32_at(bytes, off + 12),
            raw_size: u32_at(bytes, off + 16),
            raw_offset: u32_at(bytes, off + 20),
            characteristics: u32_at(bytes, off + 36),
        });
    }

    let mut imports = Vec::new();
    if import_dir_rva != 0 {
        if let Some(mut desc_off) = rva_to_offset(&sections, import_dir_rva) {
            while desc_off + 20 <= bytes.len() {
                let orig_first_thunk = u32_at(bytes, desc_off);
                let name_rva = u32_at(bytes, desc_off + 12);
                let first_thunk = u32_at(bytes, desc_off + 16);
                if name_rva == 0 {
                    break;
                }

                let dll_name = rva_to_offset(&sections, name_rva)
                    .map(|o| read_cstr(bytes, o))
                    .unwrap_or_default();

                let thunk_rva = if orig_first_thunk != 0 {
                    orig_first_thunk
                } else {
                    first_thunk
                };
                imports.push(ImportDll {
                    dll_name,
                    first_thunk_rva: first_thunk,
                    functions: read_thunk_names(bytes, &sections, thunk_rva),
                    delay_module_handle_rva: None,
                });
                desc_off += 20;
            }
        }
    }

    // Delay-load imports (#13): IMAGE_DELAYLOAD_DESCRIPTOR, 32 bytes each.
    // Only the RVA-based form (Attributes bit 0) that every modern linker emits.
    if delay_dir_rva != 0 {
        if let Some(mut d_off) = rva_to_offset(&sections, delay_dir_rva) {
            while d_off + 32 <= bytes.len() {
                let attributes = u32_at(bytes, d_off);
                let name_rva = u32_at(bytes, d_off + 4);
                let handle_rva = u32_at(bytes, d_off + 8);
                let iat_rva = u32_at(bytes, d_off + 12);
                let int_rva = u32_at(bytes, d_off + 16);
                if name_rva == 0 {
                    break;
                }
                if attributes & 1 != 0 {
                    let dll_name = rva_to_offset(&sections, name_rva)
                        .map(|o| read_cstr(bytes, o))
                        .unwrap_or_default();
                    imports.push(ImportDll {
                        dll_name,
                        first_thunk_rva: iat_rva,
                        functions: read_thunk_names(bytes, &sections, int_rva),
                        delay_module_handle_rva: Some(handle_rva),
                    });
                }
                d_off += 32;
            }
        }
    }

    // Parse Export Directory (#0) for companion title-local DLLs
    let mut exports = Vec::new();
    if export_dir_rva != 0 {
        if let Some(exp_off) = rva_to_offset(&sections, export_dir_rva) {
            if exp_off + 40 <= bytes.len() {
                let ord_base = u32_at(bytes, exp_off + 16) as u16;
                let num_names = u32_at(bytes, exp_off + 24) as usize;
                let funcs_rva = u32_at(bytes, exp_off + 28);
                let names_rva = u32_at(bytes, exp_off + 32);
                let ords_rva = u32_at(bytes, exp_off + 36);
                if let (Some(f_off), Some(n_off), Some(o_off)) = (
                    rva_to_offset(&sections, funcs_rva),
                    rva_to_offset(&sections, names_rva),
                    rva_to_offset(&sections, ords_rva),
                ) {
                    for i in 0..num_names {
                        if n_off + i * 4 + 4 > bytes.len() || o_off + i * 2 + 2 > bytes.len() {
                            break;
                        }
                        let name_rva = u32_at(bytes, n_off + i * 4);
                        let ord_idx = u16_at(bytes, o_off + i * 2) as usize;
                        if f_off + ord_idx * 4 + 4 > bytes.len() {
                            continue;
                        }
                        let fn_rva = u32_at(bytes, f_off + ord_idx * 4);
                        let name = rva_to_offset(&sections, name_rva)
                            .map(|o| read_cstr(bytes, o))
                            .unwrap_or_default();
                        exports.push(ExportFunc {
                            name,
                            ordinal: ord_base.wrapping_add(ord_idx as u16),
                            rva: fn_rva,
                        });
                    }
                }
            }
        }
    }

    Ok(Pe64Info {
        is_dll,
        image_base,
        entry_rva,
        size_of_image,
        size_of_headers,
        reloc_dir_rva,
        reloc_dir_size,
        tls_dir_rva,
        tls_dir_size,
        sections,
        imports,
        exports,
    })
}

/// Names of a 64-bit import name table: `ordinal#N` or the hint/name string.
fn read_thunk_names(bytes: &[u8], sections: &[Section], thunk_rva: u32) -> Vec<String> {
    let mut functions = Vec::new();
    let Some(mut t_off) = rva_to_offset(sections, thunk_rva) else {
        return functions;
    };
    while t_off + 8 <= bytes.len() {
        let entry = u64_at(bytes, t_off);
        if entry == 0 {
            break;
        }
        if (entry >> 63) != 0 {
            functions.push(format!("ordinal#{}", entry & 0xffff));
        } else if let Some(hint_off) = rva_to_offset(sections, entry as u32) {
            functions.push(read_cstr(bytes, hint_off + 2));
        }
        t_off += 8;
    }
    functions
}

pub(crate) fn rva_to_offset(sections: &[Section], rva: u32) -> Option<usize> {
    for s in sections {
        let span = s.virtual_size.max(s.raw_size);
        if rva >= s.virtual_addr && rva < s.virtual_addr.saturating_add(span) {
            return Some((s.raw_offset + (rva - s.virtual_addr)) as usize);
        }
    }
    None
}

fn read_cstr(bytes: &[u8], off: usize) -> String {
    if off >= bytes.len() {
        return String::new();
    }
    let slice = &bytes[off..];
    let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
    String::from_utf8_lossy(&slice[..end]).into_owned()
}

pub(crate) fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap())
}

pub(crate) fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

pub(crate) fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
