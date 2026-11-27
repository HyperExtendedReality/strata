//! A builder for small, loader-valid PE32+ images: the loader's tests and
//! the probe title's helper DLL are made with it.
//!
//! Layout: `.text` at 0x1000, `.rdata` at 0x2000, `.data` at 0x3000 and
//! `.reloc` at 0x4000, each one 0x200-byte file block after 0x400 bytes of
//! headers (room for one more section header, as linkers leave); ASLR on.

pub const BASE: u64 = 0x1_8000_0000;
pub const TEXT: u32 = 0x1000;
pub const RDATA: u32 = 0x2000;
pub const DATA: u32 = 0x3000;
pub const RELOC: u32 = 0x4000;
pub const RAW: usize = 0x200;
/// `SizeOfHeaders`: the first section's file offset.
pub const HEADERS: usize = 0x400;

#[derive(Default, Clone)]
pub struct TestPe {
    pub text: Vec<u8>,
    pub rdata: Vec<u8>,
    pub data: Vec<u8>,
    /// `(directory index, rva, size)`.
    pub dirs: Vec<(usize, u32, u32)>,
    /// RVAs of 8-byte absolute addresses that need DIR64 relocations.
    pub relocs: Vec<u32>,
    /// Build a DLL (`IMAGE_FILE_DLL`) rather than an executable.
    pub dll: bool,
    /// Entry point RVA; `TEXT` when `None`.
    pub entry: Option<u32>,
    /// `(name, rva)` exports, written after `rdata` with `dll_name`.
    pub exports: Vec<(String, u32)>,
    pub dll_name: String,
}

/// Writes `bytes` at `off`, growing `buf` as needed.
pub fn at(buf: &mut Vec<u8>, off: usize, bytes: &[u8]) {
    if buf.len() < off + bytes.len() {
        buf.resize(off + bytes.len(), 0);
    }
    buf[off..off + bytes.len()].copy_from_slice(bytes);
}

/// `rel32` from the end of an instruction at `insn_rva` of `len` bytes to `target`.
pub fn rel(insn_rva: u32, len: u32, target: u32) -> [u8; 4] {
    (target as i32 - (insn_rva + len) as i32).to_le_bytes()
}

impl TestPe {
    pub fn build(&self) -> Vec<u8> {
        let mut rdata = self.rdata.clone();
        let mut dirs = self.dirs.clone();
        if !self.exports.is_empty() {
            dirs.push(self.write_exports(&mut rdata));
        }

        let mut img = vec![0u8; HEADERS + RAW * 4];
        let w = |img: &mut Vec<u8>, off: usize, bytes: &[u8]| at(img, off, bytes);
        w(&mut img, 0, b"MZ");
        w(&mut img, 0x3c, &0x40u32.to_le_bytes());
        let pe = 0x40;
        w(&mut img, pe, b"PE\0\0");
        w(&mut img, pe + 4, &0x8664u16.to_le_bytes());
        w(&mut img, pe + 6, &4u16.to_le_bytes());
        w(&mut img, pe + 20, &240u16.to_le_bytes());
        // EXECUTABLE | LARGE_ADDRESS_AWARE (| DLL)
        let chars: u16 = if self.dll { 0x2022 } else { 0x0022 };
        w(&mut img, pe + 22, &chars.to_le_bytes());

        let opt = pe + 24;
        w(&mut img, opt, &0x020bu16.to_le_bytes());
        w(&mut img, opt + 2, &[14]);
        w(&mut img, opt + 4, &(RAW as u32).to_le_bytes());
        w(
            &mut img,
            opt + 16,
            &self.entry.unwrap_or(TEXT).to_le_bytes(),
        );
        w(&mut img, opt + 20, &TEXT.to_le_bytes());
        w(&mut img, opt + 24, &BASE.to_le_bytes());
        w(&mut img, opt + 32, &0x1000u32.to_le_bytes());
        w(&mut img, opt + 36, &(RAW as u32).to_le_bytes());
        w(&mut img, opt + 40, &6u16.to_le_bytes());
        w(&mut img, opt + 48, &6u16.to_le_bytes());
        w(&mut img, opt + 56, &0x5000u32.to_le_bytes());
        w(&mut img, opt + 60, &(HEADERS as u32).to_le_bytes());
        w(&mut img, opt + 68, &3u16.to_le_bytes());
        w(&mut img, opt + 70, &0x0160u16.to_le_bytes()); // HIGH_ENTROPY | DYNAMIC_BASE | NX
        w(&mut img, opt + 72, &0x10_0000u64.to_le_bytes());
        w(&mut img, opt + 80, &0x1000u64.to_le_bytes());
        w(&mut img, opt + 88, &0x10_0000u64.to_le_bytes());
        w(&mut img, opt + 96, &0x1000u64.to_le_bytes());
        w(&mut img, opt + 108, &16u32.to_le_bytes());

        let reloc = self.reloc_blocks();
        if !reloc.is_empty() {
            dirs.push((5, RELOC, reloc.len() as u32));
        }
        for (i, rva, size) in dirs {
            w(&mut img, opt + 112 + i * 8, &rva.to_le_bytes());
            w(&mut img, opt + 116 + i * 8, &size.to_le_bytes());
        }

        let sections: [(&[u8; 8], u32, &[u8], u32); 4] = [
            (b".text\0\0\0", TEXT, &self.text, 0x6000_0020),
            (b".rdata\0\0", RDATA, &rdata, 0x4000_0040),
            (b".data\0\0\0", DATA, &self.data, 0xC000_0040),
            (b".reloc\0\0", RELOC, &reloc, 0x4200_0040),
        ];
        for (i, (name, rva, body, chars)) in sections.into_iter().enumerate() {
            assert!(body.len() <= RAW, "test section too large");
            let sec = opt + 240 + i * 40;
            let raw = HEADERS + RAW * i;
            w(&mut img, sec, name);
            w(&mut img, sec + 8, &(RAW as u32).to_le_bytes());
            w(&mut img, sec + 12, &rva.to_le_bytes());
            w(&mut img, sec + 16, &(RAW as u32).to_le_bytes());
            w(&mut img, sec + 20, &(raw as u32).to_le_bytes());
            w(&mut img, sec + 36, &chars.to_le_bytes());
            w(&mut img, raw, body);
        }
        img
    }

    /// Appends `IMAGE_EXPORT_DIRECTORY` and its tables to `rdata`; returns
    /// the directory entry.
    fn write_exports(&self, rdata: &mut Vec<u8>) -> (usize, u32, u32) {
        let mut exports = self.exports.clone();
        exports.sort_by(|a, b| a.0.cmp(&b.0)); // names are binary-searched
        let n = exports.len();
        let dir = rdata.len().next_multiple_of(16);
        let funcs = dir + 40;
        let names = funcs + 4 * n;
        let ords = names + 4 * n;
        let mut strings = ords + 2 * n;
        let rva = |off: usize| RDATA + off as u32;

        let dll_name_at = strings;
        at(rdata, strings, format!("{}\0", self.dll_name).as_bytes());
        strings += self.dll_name.len() + 1;
        for (i, (name, func_rva)) in exports.iter().enumerate() {
            at(rdata, funcs + 4 * i, &func_rva.to_le_bytes());
            at(rdata, names + 4 * i, &rva(strings).to_le_bytes());
            at(rdata, ords + 2 * i, &(i as u16).to_le_bytes());
            at(rdata, strings, format!("{name}\0").as_bytes());
            strings += name.len() + 1;
        }
        // Characteristics, TimeDateStamp, Major+MinorVersion, Name, Base,
        // NumberOfFunctions, NumberOfNames, AddressOf{Functions,Names,NameOrdinals}.
        let n = n as u32;
        let header = [
            0,
            0,
            0,
            rva(dll_name_at),
            1,
            n,
            n,
            rva(funcs),
            rva(names),
            rva(ords),
        ];
        for (i, v) in header.iter().enumerate() {
            at(rdata, dir + 4 * i, &v.to_le_bytes());
        }
        (0, rva(dir), (strings - dir) as u32)
    }

    fn reloc_blocks(&self) -> Vec<u8> {
        let mut pages: Vec<u32> = self.relocs.iter().map(|r| r & !0xfff).collect();
        pages.sort_unstable();
        pages.dedup();
        let mut out = Vec::new();
        for page in pages {
            let mut entries: Vec<u16> = self
                .relocs
                .iter()
                .filter(|r| *r & !0xfff == page)
                .map(|r| 0xA000 | (r & 0xfff) as u16)
                .collect();
            if entries.len() % 2 == 1 {
                entries.push(0);
            }
            out.extend_from_slice(&page.to_le_bytes());
            out.extend_from_slice(&(8 + entries.len() as u32 * 2).to_le_bytes());
            entries
                .iter()
                .for_each(|e| out.extend_from_slice(&e.to_le_bytes()));
        }
        out
    }
}

/// The probe title's helper DLL (`strata_probe_helper.dll`): `DllMain`
/// counts calls per reason in `.data`; `helper_attach_count` returns the
/// process-attach count and `helper_thread_attaches` the thread-attach count.
pub fn probe_helper_dll() -> Vec<u8> {
    let mut pe = TestPe {
        dll: true,
        dll_name: "strata_probe_helper.dll".into(),
        ..Default::default()
    };
    // 0x1000 DllMain: mov edx,edx (the reason is a DWORD: upper half undefined)
    //   ; lea rax,[rip+counts] ; inc dword [rax+rdx*4] ; mov eax,1 ; ret
    at(&mut pe.text, 0x00, &[0x89, 0xD2, 0x48, 0x8D, 0x05]);
    at(&mut pe.text, 0x05, &rel(TEXT + 2, 7, DATA));
    at(
        &mut pe.text,
        0x09,
        &[0xFF, 0x04, 0x90, 0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3],
    );
    // 0x1020 helper_attach_count: mov eax,[rip+counts+4] ; ret
    at(&mut pe.text, 0x20, &[0x8B, 0x05]);
    at(&mut pe.text, 0x22, &rel(TEXT + 0x20, 6, DATA + 4));
    at(&mut pe.text, 0x26, &[0xC3]);
    // 0x1030 helper_thread_attaches: mov eax,[rip+counts+8] ; ret
    at(&mut pe.text, 0x30, &[0x8B, 0x05]);
    at(&mut pe.text, 0x32, &rel(TEXT + 0x30, 6, DATA + 8));
    at(&mut pe.text, 0x36, &[0xC3]);
    at(&mut pe.data, 0, &[0; 16]);
    pe.exports = vec![
        ("helper_attach_count".into(), TEXT + 0x20),
        ("helper_thread_attaches".into(), TEXT + 0x30),
    ];
    pe.build()
}
