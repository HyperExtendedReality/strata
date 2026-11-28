// bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com)
// Source: https://github.com/Ahmet-Dedeler/bcpack-rs
// SPDX-License-Identifier: MIT OR Apache-2.0
// If you copy, vendor or port this file, keep this header and credit bcpack-rs in your README.
//! Block rebuild for BC3, BC4 and BC5: lane symbols -> blocks.
//!
//! All three keep, per block, a small state of endpoint fields and per-pixel
//! index *positions*. A position is where the pixel sits on the line from
//! endpoint 0 to endpoint 1: for the 3-bit alpha indices position 0 is
//! endpoint 0, 7 is endpoint 1 and 1..6 are the interpolated values in order
//! (BC4 codes 0, 2, 3, 4, 5, 6, 7, 1); for BC3's 2-bit colour indices the
//! positions 0..3 are BC1 codes 0, 2, 3, 1. Alpha endpoints are held 128
//! below their stored value (`stored = state ^ 0x80`), so the zero state is
//! the flat value 128.
//!
//! A block's state is the sum of two parts, element by element, modulo the
//! element's width:
//!
//! * a reference state: the endpoints of the endpoint reference and the
//!   positions of the index reference ([`crate::format`]: zero, an earlier
//!   block, or a dictionary entry);
//! * the residuals carried by the lanes: zigzag residuals for the endpoints,
//!   and for the indices one symbol per pixel pair (3-bit) or per 2x2
//!   quadrant (2-bit) that a fixed rank table turns into position residuals.
//!
//! Lane layouts (E and O are a lane's first and second symbol of a block):
//!
//! | mode | lanes 0, 1 | endpoints | indices |
//! |---|---|---|---|
//! | BC4 | one symbol each | lane 2 = e0, lane 3 = e1 | lane 4 + k = pixels 2k, 2k + 1 |
//! | BC5 | E/O interleaved | lane 2 = red e0 (E), e1 (O); lane 3 = green | lanes 4-7 red, 8-11 green: lane's E = pixels 4k, 4k + 1, O = 4k + 2, 4k + 3 |
//! | BC3 | E/O interleaved | lane 2 = alpha e0, e1; lanes 7, 8, 9 = R, G, B of colour 0 (E) and 1 (O) | lanes 3-6 alpha pairs as BC5; lane 10 = quadrants TL (E), TR (O), lane 11 = BL, BR |
//!
//! The rank tables and dictionaries are constants of the format. They are not
//! in the stream; the full tables ship in `data/bc{3,4,5}.json` and each value
//! still carries a "known" flag for partial custom tables (see
//! `docs/findings/bc345-bc7.md`).

use crate::format::{self, Mode, Source, DICTIONARY};
use crate::rans::{self, DecodeError, LaneOutput};
use serde_json::Value;

/// Position on the alpha line -> BC4 3-bit code.
pub const ALPHA_CODE: [u8; 8] = [0, 2, 3, 4, 5, 6, 7, 1];
/// BC4 3-bit code -> position on the alpha line.
pub const ALPHA_POSITION: [u8; 8] = [0, 7, 1, 2, 3, 4, 5, 6];
/// Position on the colour line -> BC1 2-bit code.
pub const COLOUR_CODE: [u8; 4] = [0, 2, 3, 1];
/// BC1 2-bit code -> position on the colour line.
pub const COLOUR_POSITION: [u8; 4] = [0, 3, 1, 2];

/// Endpoint fields and index positions of one block (or dictionary entry).
///
/// | mode | `ep` | `ix` |
/// |---|---|---|
/// | BC4 | e0, e1 | 16 positions, raster order |
/// | BC5 | red e0, e1, green e0, e1 | red 0..16, green 16..32 |
/// | BC3 | alpha e0, e1, then R0 G0 B0 R1 G1 B1 | alpha 0..16, colour 16..32 |
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct State {
    pub ep: [u8; 8],
    pub ix: [u8; 32],
}

/// Which values of a [`State`] are *not* known (bit k = element k).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unknown {
    pub ep: u8,
    pub ix: u32,
}

impl Unknown {
    pub fn any(self) -> bool {
        self.ep != 0 || self.ix != 0
    }
}

/// Element widths of a mode's state.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub mode: Mode,
    /// Bits of each endpoint field (0 = unused).
    pub ep_bits: [u8; 8],
    /// Number of 3-bit positions, then of 2-bit positions.
    pub alpha_ix: usize,
    pub colour_ix: usize,
}

pub const BC3: Layout = Layout {
    mode: Mode::Bc3,
    ep_bits: [8, 8, 5, 6, 5, 5, 6, 5],
    alpha_ix: 16,
    colour_ix: 16,
};
pub const BC4: Layout = Layout {
    mode: Mode::Bc4,
    ep_bits: [8, 8, 0, 0, 0, 0, 0, 0],
    alpha_ix: 16,
    colour_ix: 0,
};
pub const BC5: Layout = Layout {
    mode: Mode::Bc5,
    ep_bits: [8, 8, 8, 8, 0, 0, 0, 0],
    alpha_ix: 32,
    colour_ix: 0,
};

impl Layout {
    pub fn of(mode: Mode) -> Option<&'static Layout> {
        match mode {
            Mode::Bc3 => Some(&BC3),
            Mode::Bc4 => Some(&BC4),
            Mode::Bc5 => Some(&BC5),
            _ => None,
        }
    }

    pub fn endpoints(&self) -> usize {
        self.ep_bits.iter().filter(|&&b| b != 0).count()
    }

    pub fn indices(&self) -> usize {
        self.alpha_ix + self.colour_ix
    }

    /// Bits of index element `p`.
    pub fn ix_bits(&self, p: usize) -> u32 {
        if p < self.alpha_ix {
            3
        } else {
            2
        }
    }
}

/// Rank table of the 3-bit index lanes: symbol (6 bits) -> position residuals
/// of the two pixels it covers.
#[derive(Clone, Debug)]
pub struct PairTable {
    pub pairs: [[u8; 2]; 64],
    pub known: [bool; 64],
}

impl PairTable {
    pub fn from_pairs(pairs: [[u8; 2]; 64]) -> PairTable {
        PairTable {
            pairs,
            known: [true; 64],
        }
    }

    /// Symbol for a pair of residuals, if the table has it.
    pub fn symbol(&self, a: u8, b: u8) -> Option<u16> {
        (0..64)
            .find(|&s| self.known[s] && self.pairs[s] == [a, b])
            .map(|s| s as u16)
    }

    fn from_json(v: &Value, what: &str) -> Result<PairTable, String> {
        let arr = v
            .as_array()
            .ok_or_else(|| format!("{what}: not an array"))?;
        if arr.len() != 64 {
            return Err(format!("{what}: {} entries, need 64", arr.len()));
        }
        let mut t = PairTable {
            pairs: [[0; 2]; 64],
            known: [false; 64],
        };
        for (s, e) in arr.iter().enumerate() {
            if e.is_null() {
                continue;
            }
            let p = e
                .as_array()
                .filter(|p| p.len() == 2)
                .ok_or_else(|| format!("{what}: symbol {s} is not a pair"))?;
            for (k, value) in p.iter().enumerate() {
                let x = value.as_u64().filter(|&x| x < 8);
                t.pairs[s][k] = x.ok_or_else(|| format!("{what}: symbol {s} out of range"))? as u8;
            }
            t.known[s] = true;
        }
        Ok(t)
    }
}

/// Rank table of BC3's colour index lanes: symbol (8 bits) -> position
/// residuals of a 2x2 quadrant in the order TL, TR, BL, BR.
#[derive(Clone, Debug)]
pub struct QuadTable {
    pub quads: [[u8; 4]; 256],
    pub known: [bool; 256],
}

impl QuadTable {
    pub fn from_quads(quads: [[u8; 4]; 256]) -> QuadTable {
        QuadTable {
            quads,
            known: [true; 256],
        }
    }

    pub fn symbol(&self, q: [u8; 4]) -> Option<u16> {
        (0..256)
            .find(|&s| self.known[s] && self.quads[s] == q)
            .map(|s| s as u16)
    }

    fn from_json(v: &Value, what: &str) -> Result<QuadTable, String> {
        let arr = v
            .as_array()
            .ok_or_else(|| format!("{what}: not an array"))?;
        if arr.len() != 256 {
            return Err(format!("{what}: {} entries, need 256", arr.len()));
        }
        let mut t = QuadTable {
            quads: [[0; 4]; 256],
            known: [false; 256],
        };
        for (s, e) in arr.iter().enumerate() {
            if e.is_null() {
                continue;
            }
            let p = e
                .as_array()
                .filter(|p| p.len() == 4)
                .ok_or_else(|| format!("{what}: symbol {s} is not a quad"))?;
            for (k, value) in p.iter().enumerate() {
                let x = value.as_u64().filter(|&x| x < 4);
                t.quads[s][k] = x.ok_or_else(|| format!("{what}: symbol {s} out of range"))? as u8;
            }
            t.known[s] = true;
        }
        Ok(t)
    }
}

/// The constant tables of one of the three modes.
#[derive(Clone, Debug)]
pub struct Tables {
    pub layout: &'static Layout,
    /// 3-bit pair ranks (alpha indices).
    pub pairs: PairTable,
    /// 2-bit quadrant ranks (BC3 colour indices; unused otherwise).
    pub quads: QuadTable,
    /// 255 entries and which of their values are unknown.
    pub dictionary: Vec<(State, Unknown)>,
}

/// Default tables, `data/bc3.json`, `data/bc4.json` and `data/bc5.json`.
pub const DEFAULT_BC3_JSON: &str = include_str!("../data/bc3.json");
pub const DEFAULT_BC4_JSON: &str = include_str!("../data/bc4.json");
pub const DEFAULT_BC5_JSON: &str = include_str!("../data/bc5.json");

impl Tables {
    /// Tables with every value unknown (identity-free; decodes structure only).
    pub fn empty(layout: &'static Layout) -> Tables {
        let all = Unknown {
            ep: ((1u32 << layout.endpoints()) - 1) as u8,
            ix: ((1u64 << layout.indices()) - 1) as u32,
        };
        Tables {
            layout,
            pairs: PairTable {
                pairs: [[0; 2]; 64],
                known: [false; 64],
            },
            quads: QuadTable {
                quads: [[0; 4]; 256],
                known: [false; 256],
            },
            dictionary: vec![(State::default(), all); DICTIONARY],
        }
    }

    /// Parse the JSON of `data/bc{3,4,5}.json`:
    ///
    /// ```text
    /// { "pairs": [[a, b] | null, x64],
    ///   "quads": [[tl, tr, bl, br] | null, x256],          (BC3 only)
    ///   "dictionary": [{"ep": [v | null, ...], "ix": [v | null, ...]} | null, x255] }
    /// ```
    ///
    /// `null` marks a value that has not been determined; it decodes as zero
    /// and the decoder counts the blocks that depend on it.
    pub fn from_json(layout: &'static Layout, text: &str) -> Result<Tables, String> {
        let name = layout.mode.name();
        let v: Value = serde_json::from_str(text).map_err(|e| format!("{name}: {e}"))?;
        let mut t = Tables::empty(layout);
        t.pairs = PairTable::from_json(&v["pairs"], &format!("{name} pairs"))?;
        if layout.colour_ix != 0 {
            t.quads = QuadTable::from_json(&v["quads"], &format!("{name} quads"))?;
        }
        let dict = v["dictionary"]
            .as_array()
            .ok_or_else(|| format!("{name}: missing \"dictionary\""))?;
        if dict.len() != DICTIONARY {
            return Err(format!(
                "{name}: {} dictionary entries, need 255",
                dict.len()
            ));
        }
        for (k, e) in dict.iter().enumerate() {
            if e.is_null() {
                continue;
            }
            let (st, unk) = &mut t.dictionary[k];
            let ep = e["ep"].as_array().filter(|a| a.len() == layout.endpoints());
            let ep = ep.ok_or_else(|| {
                format!(
                    "{name}: entry {k} needs {} endpoint values",
                    layout.endpoints()
                )
            })?;
            for (j, x) in ep.iter().enumerate() {
                if x.is_null() {
                    continue;
                }
                let lim = 1u64 << layout.ep_bits[j];
                let x = x.as_u64().filter(|&x| x < lim);
                st.ep[j] =
                    x.ok_or_else(|| format!("{name}: entry {k} endpoint {j} out of range"))? as u8;
                unk.ep &= !(1 << j);
            }
            let ix = e["ix"].as_array().filter(|a| a.len() == layout.indices());
            let ix = ix.ok_or_else(|| {
                format!("{name}: entry {k} needs {} index values", layout.indices())
            })?;
            for (p, x) in ix.iter().enumerate() {
                if x.is_null() {
                    continue;
                }
                let lim = 1u64 << layout.ix_bits(p);
                let x = x.as_u64().filter(|&x| x < lim);
                st.ix[p] =
                    x.ok_or_else(|| format!("{name}: entry {k} index {p} out of range"))? as u8;
                unk.ix &= !(1 << p);
            }
        }
        Ok(t)
    }

    /// The tables shipped in `data/` for `mode` (BC3, BC4 or BC5).
    pub fn default_for(mode: Mode) -> Option<&'static Tables> {
        use std::sync::OnceLock;
        static T: [OnceLock<Tables>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
        let (slot, layout, text) = match mode {
            Mode::Bc3 => (0, &BC3, DEFAULT_BC3_JSON),
            Mode::Bc4 => (1, &BC4, DEFAULT_BC4_JSON),
            Mode::Bc5 => (2, &BC5, DEFAULT_BC5_JSON),
            _ => return None,
        };
        Some(T[slot].get_or_init(|| Tables::from_json(layout, text).expect("embedded tables")))
    }
}

/// Rebuilt blocks and how much of them rests on undetermined table values.
#[derive(Clone, Debug)]
pub struct Rebuilt {
    pub blocks: Vec<u8>,
    /// Blocks with at least one value that depends on an unknown table value.
    pub unfitted_blocks: usize,
    /// References that reached the dictionary (endpoint and index references).
    pub dictionary_refs: usize,
}

/// Lane symbols of one block -> the two references and the residual state.
/// `unk` gets the index elements whose rank symbol is unknown.
#[inline]
pub fn record(
    layout: &Layout,
    lanes: &LaneOutput,
    i: usize,
    t: &Tables,
    unk: &mut Unknown,
) -> (usize, usize, State) {
    let mut st = State::default();
    let pair = |st: &mut State, unk: &mut Unknown, base: usize, k: usize, sym: u16| {
        let s = sym as usize & 63;
        st.ix[base + 2 * k] = t.pairs.pairs[s][0];
        st.ix[base + 2 * k + 1] = t.pairs.pairs[s][1];
        if !t.pairs.known[s] {
            unk.ix |= 3 << (base + 2 * k);
        }
    };
    match layout.mode {
        Mode::Bc4 => {
            let (e, x) = format::references(lanes.lane(0)[i] as u8, lanes.lane(1)[i] as u8);
            st.ep[0] = format::residual(lanes.lane(2)[i], 8);
            st.ep[1] = format::residual(lanes.lane(3)[i], 8);
            for k in 0..8 {
                pair(&mut st, unk, 0, k, lanes.lane(4 + k)[i]);
            }
            (e, x, st)
        }
        _ => {
            let eo = |lane: usize| (lanes.lane(lane)[2 * i], lanes.lane(lane)[2 * i + 1]);
            let (d0, d1) = eo(0);
            let (x0, x1) = eo(1);
            let (e, x) = format::references(format::interleave(d0, d1), format::interleave(x0, x1));
            if layout.mode == Mode::Bc5 {
                for c in 0..2 {
                    let (a, b) = eo(2 + c);
                    st.ep[2 * c] = format::residual(a, 8);
                    st.ep[2 * c + 1] = format::residual(b, 8);
                    for l in 0..4 {
                        let (a, b) = eo(4 + 4 * c + l);
                        pair(&mut st, unk, 16 * c, 2 * l, a);
                        pair(&mut st, unk, 16 * c, 2 * l + 1, b);
                    }
                }
            } else {
                let (a, b) = eo(2);
                st.ep[0] = format::residual(a, 8);
                st.ep[1] = format::residual(b, 8);
                for l in 0..4 {
                    let (a, b) = eo(3 + l);
                    pair(&mut st, unk, 0, 2 * l, a);
                    pair(&mut st, unk, 0, 2 * l + 1, b);
                }
                for c in 0..3 {
                    let (a, b) = eo(7 + c);
                    st.ep[2 + c] = format::residual(a, layout.ep_bits[2 + c] as u32);
                    st.ep[5 + c] = format::residual(b, layout.ep_bits[5 + c] as u32);
                }
                let (tl, tr) = eo(10);
                let (bl, br) = eo(11);
                for (q, sym) in [tl, tr, bl, br].into_iter().enumerate() {
                    let s = sym as usize & 255;
                    let (ox, oy) = ((q & 1) * 2, (q >> 1) * 2);
                    for (k, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                        let p = 16 + 4 * (oy + dy) + ox + dx;
                        st.ix[p] = t.quads.quads[s][k];
                        if !t.quads.known[s] {
                            unk.ix |= 1 << p;
                        }
                    }
                }
            }
            (e, x, st)
        }
    }
}

/// One alpha (BC4) half-block: two endpoints and 16 positions.
#[inline]
pub fn pack_alpha(e0: u8, e1: u8, ix: &[u8], out: &mut [u8]) {
    let mut v = (e0 ^ 0x80) as u64 | ((e1 ^ 0x80) as u64) << 8;
    for (p, &pos) in ix.iter().take(16).enumerate() {
        v |= (ALPHA_CODE[pos as usize & 7] as u64) << (16 + 3 * p);
    }
    out[..8].copy_from_slice(&v.to_le_bytes());
}

/// One colour (BC1 layout) half-block: R0 G0 B0 R1 G1 B1 and 16 positions.
#[inline]
pub fn pack_colour(ep: &[u8], ix: &[u8], out: &mut [u8]) {
    let c0 = (ep[0] as u16) << 11 | (ep[1] as u16) << 5 | ep[2] as u16;
    let c1 = (ep[3] as u16) << 11 | (ep[4] as u16) << 5 | ep[5] as u16;
    let mut w = 0u32;
    for (p, &pos) in ix.iter().take(16).enumerate() {
        w |= (COLOUR_CODE[pos as usize & 3] as u32) << (2 * p);
    }
    out[0..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&w.to_le_bytes());
}

/// A state as the mode's block bytes.
pub fn pack(layout: &Layout, st: &State, out: &mut [u8]) {
    match layout.mode {
        Mode::Bc4 => pack_alpha(st.ep[0], st.ep[1], &st.ix[..16], out),
        Mode::Bc5 => {
            pack_alpha(st.ep[0], st.ep[1], &st.ix[..16], &mut out[..8]);
            pack_alpha(st.ep[2], st.ep[3], &st.ix[16..], &mut out[8..]);
        }
        _ => {
            pack_alpha(st.ep[0], st.ep[1], &st.ix[..16], &mut out[..8]);
            pack_colour(&st.ep[2..8], &st.ix[16..], &mut out[8..]);
        }
    }
}

/// Block bytes back to a state (inverse of [`pack`]).
pub fn unpack(layout: &Layout, block: &[u8]) -> State {
    let mut st = State::default();
    let alpha = |b: &[u8], ep: &mut [u8], ix: &mut [u8]| {
        let v = u64::from_le_bytes(b[..8].try_into().unwrap());
        ep[0] = b[0] ^ 0x80;
        ep[1] = b[1] ^ 0x80;
        for (p, x) in ix.iter_mut().take(16).enumerate() {
            *x = ALPHA_POSITION[(v >> (16 + 3 * p)) as usize & 7];
        }
    };
    match layout.mode {
        Mode::Bc4 => alpha(block, &mut st.ep[0..2], &mut st.ix[..16]),
        Mode::Bc5 => {
            alpha(&block[..8], &mut st.ep[0..2], &mut st.ix[..16]);
            alpha(&block[8..], &mut st.ep[2..4], &mut st.ix[16..]);
        }
        _ => {
            alpha(&block[..8], &mut st.ep[0..2], &mut st.ix[..16]);
            let c0 = u16::from_le_bytes([block[8], block[9]]);
            let c1 = u16::from_le_bytes([block[10], block[11]]);
            let w = u32::from_le_bytes(block[12..16].try_into().unwrap());
            st.ep[2..8].copy_from_slice(&[
                (c0 >> 11) as u8,
                (c0 >> 5 & 63) as u8,
                (c0 & 31) as u8,
                (c1 >> 11) as u8,
                (c1 >> 5 & 63) as u8,
                (c1 & 31) as u8,
            ]);
            for p in 0..16 {
                st.ix[16 + p] = COLOUR_POSITION[(w >> (2 * p)) as usize & 3];
            }
        }
    }
    st
}

/// The rebuild's working set for one chunk: the last 256 states.
pub struct History {
    states: Vec<(State, Unknown)>,
}

impl Default for History {
    fn default() -> Self {
        History {
            states: vec![(State::default(), Unknown::default()); 256],
        }
    }
}

impl History {
    #[inline]
    fn get<'a>(
        &'a self,
        t: &'a Tables,
        src: Source,
        zero: &'a (State, Unknown),
    ) -> &'a (State, Unknown) {
        match src {
            Source::Zero => zero,
            Source::Block(j) => &self.states[j & 255],
            Source::Dictionary(k) => &t.dictionary[k],
        }
    }
}

/// Turn the decoded lanes of a BC3, BC4 or BC5 chunk into blocks.
pub fn assemble(lanes: &LaneOutput, t: &Tables) -> Rebuilt {
    let layout = t.layout;
    let per = layout.mode.symbols_per_block();
    let bb = layout.mode.block_bytes();
    let n = lanes.target / per;
    let mut out = vec![0u8; n * bb];
    let mut hist = History::default();
    let zero = (State::default(), Unknown::default());
    let (mut unfitted, mut dict_refs) = (0usize, 0usize);
    let (ne, ni) = (layout.endpoints(), layout.indices());
    let mut ep_mask = [0u8; 8];
    for (k, mask) in ep_mask.iter_mut().take(ne).enumerate() {
        *mask = ((1u16 << layout.ep_bits[k]) - 1) as u8;
    }
    for i in 0..n {
        let mut unk = Unknown::default();
        let (er, xr, mut st) = record(layout, lanes, i, t, &mut unk);
        let (es, xs) = (format::source(i, er), format::source(i, xr));
        dict_refs += matches!(es, Source::Dictionary(_)) as usize
            + matches!(xs, Source::Dictionary(_)) as usize;
        let (e, eu) = hist.get(t, es, &zero);
        for (k, &mask) in ep_mask.iter().take(ne).enumerate() {
            st.ep[k] = st.ep[k].wrapping_add(e.ep[k]) & mask;
        }
        unk.ep |= eu.ep;
        let (x, xu) = hist.get(t, xs, &zero);
        for p in 0..ni {
            let m = if p < layout.alpha_ix { 7 } else { 3 };
            st.ix[p] = (st.ix[p] + x.ix[p]) & m;
        }
        unk.ix |= xu.ix;
        unfitted += unk.any() as usize;
        pack(layout, &st, &mut out[i * bb..(i + 1) * bb]);
        hist.states[i & 255] = (st, unk);
    }
    Rebuilt {
        blocks: out,
        unfitted_blocks: unfitted,
        dictionary_refs: dict_refs,
    }
}

/// Decode the lane symbols of a BC3, BC4 or BC5 chunk whose unpacked size is
/// `unpacked_len` bytes.
pub fn decode_symbols(
    mode: Mode,
    chunk: &[u8],
    unpacked_len: usize,
) -> Result<LaneOutput, DecodeError> {
    rans::decode_slice(chunk, mode.alphabet_bits(), unpacked_len / 8)
}

/// BCPack BC3, BC4 or BC5 chunk bytes -> blocks in stream order.
pub fn decode(chunk: &[u8], unpacked_len: usize, t: &Tables) -> Result<Rebuilt, DecodeError> {
    let lanes = decode_symbols(t.layout.mode, chunk, unpacked_len)?;
    Ok(assemble(&lanes, t))
}
