// bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com)
// Source: https://github.com/Ahmet-Dedeler/bcpack-rs
// SPDX-License-Identifier: MIT OR Apache-2.0
// If you copy, vendor or port this file, keep this header and credit bcpack-rs in your README.
//! (Vendored into Strata: library modules only; the CLI, `ttpf` and
//! `texture` modules are omitted. Upstream commit in `UPSTREAM_COMMIT`.)
//!
//! Decoder for Xbox Series BCPack texture streams (BC1, BC3, BC4, BC5 and BC7).
//!
//! BCPack is the rANS entropy layer the Xbox Series MDU puts over BCn blocks.
//! This crate decodes BC1 (`0x04`), BC3 (`0x0C`), BC4 (`0x10`), BC5 (`0x14`)
//! and BC7 (`0x1C`) chunks to plain BCn blocks:
//!
//! ```no_run
//! let chunk = std::fs::read("chunk.bin").unwrap();
//! let window = bcpack::bc1::Window::default_table();
//! let quads = bcpack::bc1::QuadTable::default_table();
//! let out = bcpack::bc1::decode(&chunk, 4096, &window, &quads).unwrap();
//! assert_eq!(out.blocks.len(), 4096 * 8);
//! ```
//!
//! For BC3/BC4/BC5 use [`bcn::decode`] with [`bcn::Tables::default_for`], and
//! for BC7 use [`bc7::decode`] with [`bc7::Tables::default_table`].
//!
//! Modules:
//! * [`rans`]: the cycle-level MDU rANS decoder (entropy layer, 12 or 16 lanes);
//! * [`bc1`]: lane symbols -> BC1 endpoints and index words;
//! * [`format`]: mode metadata, lane widths, back-references and dictionary addressing;
//! * [`bcn`]: lane symbols -> BC3, BC4 and BC5 blocks;
//! * [`bc7`]: lane symbols -> BC7 blocks (plus block pack/unpack);
//! * [`pixels`]: BC1/BC3/BC4/BC5/BC7 blocks -> RGBA8 pixels;
//! * [`encode`]: a synthetic encoder used by the tests;
//! * [`tables`]: the exact model tables, generated from Microsoft's
//!   MIT-licensed `silicon-rans` Verilog.
//!
//! See `docs/BCPACK_SPEC.md` and `docs/findings/bc345-bc7.md` for the format
//! and the evidence behind it.

pub mod bc1;
pub mod bc7;
pub mod bcn;
pub mod encode;
pub mod format;
pub mod pixels;
pub mod rans;
pub mod tables;

/// A decoded chunk in stored block order, including entropy and rebuild diagnostics.
pub struct DecodedChunk {
    pub blocks: Vec<u8>,
    pub dictionary_refs: usize,
    pub unfitted_blocks: usize,
    pub consumed: usize,
    pub cycles: u64,
}

/// Decode one chunk using its block count. The supplied window and quads apply to BC1.
pub fn decode_chunk(
    mode: format::Mode,
    chunk: &[u8],
    blocks: usize,
    window: &bc1::Window,
    quads: &bc1::QuadTable,
) -> Result<DecodedChunk, rans::DecodeError> {
    use format::Mode;
    if mode == Mode::Bc6h {
        return Err(rans::DecodeError::Invalid(
            "BC6H block rebuild is unsupported".into(),
        ));
    }
    let lanes = mode.decode_symbols(chunk, blocks)?;
    let (blocks, dictionary_refs, unfitted_blocks) = match mode {
        Mode::Bc1 => {
            let r = bc1::assemble(&lanes, window, quads);
            (r.blocks, r.prestart_refs, r.unfitted_blocks)
        }
        Mode::Bc7 => {
            let r = bc7::assemble(&lanes, bc7::Tables::default_table());
            if r.unknown_controls != 0 {
                return Err(rans::DecodeError::Invalid(format!(
                    "{} unknown BC7 controls",
                    r.unknown_controls
                )));
            }
            (r.blocks, r.dictionary_refs, r.unfitted_blocks)
        }
        mode => {
            let tables = bcn::Tables::default_for(mode).expect("BC3/BC4/BC5 tables");
            let r = bcn::assemble(&lanes, tables);
            (r.blocks, r.dictionary_refs, r.unfitted_blocks)
        }
    };
    Ok(DecodedChunk {
        blocks,
        dictionary_refs,
        unfitted_blocks,
        consumed: lanes.consumed,
        cycles: lanes.cycles,
    })
}
