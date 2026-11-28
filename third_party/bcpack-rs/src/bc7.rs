// bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com)
// Source: https://github.com/Ahmet-Dedeler/bcpack-rs
// SPDX-License-Identifier: MIT OR Apache-2.0
// If you copy, vendor or port this file, keep this header and credit bcpack-rs in your README.
//! Block rebuild for BC7: lane symbols -> BC7 blocks.
//!
//! 16 lanes, two symbols (E, O) per lane and block:
//!
//! | lane | content |
//! |---|---|
//! | 0, 1 | the two references ([`crate::format`]) |
//! | 2, 3, 4, 5 | subset 0 endpoint residuals R, G, B, A (E = endpoint 0, O = endpoint 1) |
//! | 6, 7, 8, 9 | subset 1 R, G, B, A; in modes 4 and 5 lanes 6, 7, 8 are rows 0-2 of the second index set |
//! | 10 | subset 2 R; in modes 4 and 5 row 3 of the second index set |
//! | 5, 9 | in three-subset modes (0, 2): subset 2 G and B instead of alpha |
//! | 11-14 | rows 0-3 of the first index set |
//! | 15 | control code: mode, partition, rotation, index selection |
//!
//! **State.** A block's state is its number of subsets, its endpoints as
//! 8-bit values aligned to the top bit (per subset R0 R1 G0 G1 B0 B1 A0 A1,
//! in the channel order *after* undoing the rotation), and two sets of 16
//! index values as 4-bit values aligned to the top bit.
//!
//! **Endpoints.** A field of `p` bits is `(reference >> (8 - p)) + residual`
//! modulo `2^p`, stored back shifted. `p` is the mode's width for the channel;
//! modes with p-bits carry the p-bit as the lowest bit of red (so red is one
//! bit wider), and a rotation swaps the widths of alpha and the rotated
//! channel. A subset the reference does not have is predicted from the
//! reference's subset 0. Channels and subsets the mode lacks are zero in the
//! new state.
//!
//! **Indices.** An index lane symbol holds two pixels: its even bits and its
//! odd bits are two zigzag residuals. E covers columns 0-1 of the row, O
//! columns 2-3; the odd-bit residual is the left pixel in rows 0-2 and the
//! right pixel in row 3. An index of `b` bits is `(reference >> (4 - b)) +
//! residual` modulo `2^b`. A set the mode lacks is zero in the new state
//! (every mode but 4 and 5 clears the second set). Anchor indices are written
//! without their top bit.
//!
//! The control codes and the dictionary are constants of the format; the full
//! tables ship in `data/bc7.json` (see `docs/findings/bc345-bc7.md`).

use crate::format::{self, Mode, Source, DICTIONARY};
use crate::rans::{self, DecodeError, LaneOutput};
use serde_json::Value;

/// Subsets per BC7 mode.
pub const SUBSETS: [u8; 8] = [3, 2, 3, 2, 1, 1, 1, 2];
/// Endpoint bits of R, G, B, A per mode, without p-bits.
pub const CHANNEL_BITS: [[u8; 4]; 8] = [
    [4, 4, 4, 0],
    [6, 6, 6, 0],
    [5, 5, 5, 0],
    [7, 7, 7, 0],
    [5, 5, 5, 6],
    [7, 7, 7, 8],
    [7, 7, 7, 7],
    [5, 5, 5, 5],
];
/// Index bits of the first and second set per mode.
pub const INDEX_BITS: [[u8; 2]; 8] = [
    [3, 0],
    [3, 0],
    [2, 0],
    [2, 0],
    [3, 2],
    [2, 2],
    [4, 0],
    [2, 0],
];
/// Partition bits per mode.
pub const PARTITION_BITS: [u8; 8] = [4, 6, 6, 6, 0, 0, 0, 6];

/// Anchor pixel of the second subset of the two-subset partitions (BC7 spec).
pub const ANCHOR_2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
];
/// Anchor pixel of the second subset of the three-subset partitions (BC7 spec).
pub const ANCHOR_3A: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5,
    15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5,
    10, 8, 13, 15, 12, 3, 3,
];
/// Anchor pixel of the third subset of the three-subset partitions (BC7 spec).
pub const ANCHOR_3B: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6,
    10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];

/// What a control code selects.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Control {
    pub mode: u8,
    pub partition: u8,
    pub rotation: u8,
    pub index_selection: u8,
}

impl Control {
    pub fn subsets(self) -> usize {
        SUBSETS[self.mode as usize] as usize
    }

    /// True for the modes whose endpoints end in a p-bit.
    pub fn has_pbits(self) -> bool {
        matches!(self.mode, 0 | 1 | 3 | 6 | 7)
    }

    /// Bits of the state's R, G, B, A fields: the p-bit widens red, a rotation
    /// swaps alpha's width with the rotated channel's.
    pub fn precision(self) -> [u8; 4] {
        let mut p = CHANNEL_BITS[self.mode as usize];
        if self.rotation != 0 {
            p.swap(3, self.rotation as usize - 1);
        } else if self.has_pbits() {
            p[0] += 1;
        }
        p
    }

    /// Index bits of the two sets.
    pub fn index_bits(self) -> [u8; 2] {
        INDEX_BITS[self.mode as usize]
    }

    /// Bit mask of the anchor pixels.
    pub fn anchors(self) -> u16 {
        let part = self.partition as usize & 63;
        match self.subsets() {
            2 => 1 | 1 << ANCHOR_2[part],
            3 => 1 | 1 << ANCHOR_3A[part] | 1 << ANCHOR_3B[part],
            _ => 1,
        }
    }

    /// Every valid control (mode, partition, rotation, index selection): 285.
    pub fn all() -> Vec<Control> {
        let mut v = Vec::new();
        for mode in 0..8u8 {
            let parts = 1u8 << PARTITION_BITS[mode as usize];
            let rots = if mode == 4 || mode == 5 { 4 } else { 1 };
            let sels = if mode == 4 { 2 } else { 1 };
            for partition in 0..parts {
                for rotation in 0..rots {
                    for index_selection in 0..sels {
                        v.push(Control {
                            mode,
                            partition,
                            rotation,
                            index_selection,
                        });
                    }
                }
            }
        }
        v
    }
}

/// Lane-15 symbols (E, O) -> the 9-bit control code.
#[inline]
pub fn control_code(e: u16, o: u16) -> usize {
    (o as usize & 15) << 5 | (e as usize & 31)
}

/// One block's state (or a dictionary entry).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct State {
    pub subsets: u8,
    /// Per subset: R0 R1 G0 G1 B0 B1 A0 A1, 8 bits, aligned to the top bit.
    pub ep: [[u8; 8]; 3],
    /// Two index sets, 4 bits, aligned to the top bit.
    pub ix: [[u8; 16]; 2],
}

/// How much of a dictionary entry is known: the number of leading bits of
/// each value (0 = nothing, 8 / 4 = all), and whether `subsets` is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Known {
    pub subsets: bool,
    pub ep: [[u8; 8]; 3],
    pub ix: [[u8; 16]; 2],
}

/// Unknown values of a block state: one bit per endpoint field and index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unknown {
    pub ep: [u8; 3],
    pub ix: [u16; 2],
}

/// The constant tables of the BC7 mode.
#[derive(Clone, Debug)]
pub struct Tables {
    /// Control code -> control, `None` for the codes the format does not use.
    pub control: Vec<Option<Control>>,
    pub dictionary: Vec<(State, Known)>,
}

/// Default tables, `data/bc7.json`.
pub const DEFAULT_BC7_JSON: &str = include_str!("../data/bc7.json");

impl Tables {
    pub fn empty() -> Tables {
        Tables {
            control: vec![None; 512],
            dictionary: vec![(State::default(), Known::default()); DICTIONARY],
        }
    }

    /// Control code for a control, if the table has it.
    pub fn code_of(&self, c: Control) -> Option<usize> {
        self.control.iter().position(|&x| x == Some(c))
    }

    /// Parse `data/bc7.json`:
    ///
    /// ```text
    /// { "control": [[E, O, mode, partition, rotation, index_selection], ...],
    ///   "dictionary": [null | {"subsets": n | null,
    ///                          "ep": [24 values], "ep_bits": [24 counts],
    ///                          "ix": [32 values], "ix_bits": [32 counts]}, x255] }
    /// ```
    ///
    /// `ep` is three subsets of R0 R1 G0 G1 B0 B1 A0 A1 (8-bit, top-aligned)
    /// and `ix` two sets of 16 (4-bit, top-aligned); `*_bits` say how many
    /// leading bits of each value have been determined.
    pub fn from_json(text: &str) -> Result<Tables, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("bc7: {e}"))?;
        let mut t = Tables::empty();
        let ctl = v["control"].as_array().ok_or("bc7: missing \"control\"")?;
        for e in ctl {
            let f: Vec<u64> = e
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
                .unwrap_or_default();
            if f.len() != 6 || f[0] > 31 || f[1] > 15 || f[2] > 7 {
                return Err("bc7: bad control entry".into());
            }
            let c = Control {
                mode: f[2] as u8,
                partition: f[3] as u8,
                rotation: f[4] as u8,
                index_selection: f[5] as u8,
            };
            if f[3] >> PARTITION_BITS[c.mode as usize] != 0
                || f[4] > if matches!(c.mode, 4 | 5) { 3 } else { 0 }
                || f[5] > (c.mode == 4) as u64
            {
                return Err(format!("bc7: control {f:?} is not a BC7 header"));
            }
            let code = control_code(f[0] as u16, f[1] as u16);
            if t.control[code].is_some() {
                return Err(format!("bc7: control code ({}, {}) twice", f[0], f[1]));
            }
            t.control[code] = Some(c);
        }
        let dict = v["dictionary"]
            .as_array()
            .ok_or("bc7: missing \"dictionary\"")?;
        if dict.len() != DICTIONARY {
            return Err(format!("bc7: {} dictionary entries, need 255", dict.len()));
        }
        let nums = |e: &Value, key: &str, n: usize, lim: u64| -> Result<Vec<u8>, String> {
            let a = e[key].as_array().filter(|a| a.len() == n);
            let a = a.ok_or_else(|| format!("bc7: dictionary \"{key}\" needs {n} values"))?;
            a.iter()
                .map(|x| match x {
                    Value::Null => Ok(0),
                    x => x
                        .as_u64()
                        .filter(|&x| x < lim)
                        .map(|x| x as u8)
                        .ok_or_else(|| format!("bc7: dictionary \"{key}\" value out of range")),
                })
                .collect()
        };
        for (k, e) in dict.iter().enumerate() {
            if e.is_null() {
                continue;
            }
            let (st, kn) = &mut t.dictionary[k];
            if let Some(s) = e["subsets"].as_u64() {
                if !(1..=3).contains(&s) {
                    return Err(format!("bc7: entry {k} subsets {s}"));
                }
                st.subsets = s as u8;
                kn.subsets = true;
            }
            let (ep, eb) = (nums(e, "ep", 24, 256)?, nums(e, "ep_bits", 24, 9)?);
            let (ix, xb) = (nums(e, "ix", 32, 16)?, nums(e, "ix_bits", 32, 5)?);
            for j in 0..24 {
                st.ep[j / 8][j % 8] = ep[j];
                kn.ep[j / 8][j % 8] = eb[j];
            }
            for j in 0..32 {
                st.ix[j / 16][j % 16] = ix[j];
                kn.ix[j / 16][j % 16] = xb[j];
            }
        }
        Ok(t)
    }

    /// The tables shipped in `data/bc7.json`.
    pub fn default_table() -> &'static Tables {
        static T: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
        T.get_or_init(|| Tables::from_json(DEFAULT_BC7_JSON).expect("embedded BC7 tables"))
    }
}

/// Rebuilt blocks and how much of them rests on undetermined table values.
#[derive(Clone, Debug)]
pub struct Rebuilt {
    pub blocks: Vec<u8>,
    /// Blocks with a written value that depends on an unknown table value.
    pub unfitted_blocks: usize,
    /// References that reached the dictionary (endpoint and index references).
    pub dictionary_refs: usize,
    /// Blocks whose control code is not in the table (written as zeros).
    pub unknown_controls: usize,
}

/// The lanes holding the endpoint residuals of subset `row`, channel `ch`
/// (0-3 = R, G, B, A) for a block of `subsets` subsets, if any.
#[inline]
pub fn endpoint_lane(subsets: usize, row: usize, ch: usize) -> Option<usize> {
    match (row, ch) {
        (0, 3) | (1, 3) if subsets == 3 => None,
        (0, c) => Some(2 + c),
        (1, c) => Some(6 + c),
        (2, 0) => Some(10),
        (2, 1) if subsets == 3 => Some(5),
        (2, 2) if subsets == 3 => Some(9),
        _ => None,
    }
}

/// The lanes of the four rows of index set `set`.
pub const INDEX_LANES: [[usize; 4]; 2] = [[11, 12, 13, 14], [6, 7, 8, 10]];

/// The two pixel residuals (left, right) of one index symbol of row `row`.
#[inline]
pub fn index_pair(sym: u16, row: usize) -> (u16, u16) {
    let (even, odd) = format::deinterleave(sym as u8);
    if row < 3 {
        (odd, even)
    } else {
        (even, odd)
    }
}

/// The index symbol holding the residual codes `left` and `right` in row `row`.
#[inline]
pub fn index_symbol(left: u16, right: u16, row: usize) -> u16 {
    if row < 3 {
        format::interleave(right, left) as u16
    } else {
        format::interleave(left, right) as u16
    }
}

/// Little-endian bit writer for one 128-bit block.
struct Bits {
    v: u128,
    at: u32,
}

impl Bits {
    #[inline]
    fn put(&mut self, value: u32, n: u32) {
        if n > 0 && self.at < 128 {
            self.v |= ((value as u128) & ((1u128 << n) - 1)) << self.at;
        }
        self.at += n;
    }
}

/// The order in which the state's channels appear in the block: a rotation
/// swaps alpha with the rotated channel.
#[inline]
fn channel_order(rotation: u8) -> [usize; 4] {
    let mut o = [0, 1, 2, 3];
    if rotation != 0 {
        o.swap(3, rotation as usize - 1);
    }
    o
}

/// A state as BC7 block bytes.
pub fn pack(c: Control, st: &State, out: &mut [u8]) {
    let mode = c.mode as u32;
    let subsets = c.subsets();
    let prec = c.precision();
    let mut w = Bits { v: 0, at: 0 };
    w.put(1 << mode, mode + 1);
    w.put(c.partition as u32, PARTITION_BITS[c.mode as usize] as u32);
    if mode == 4 || mode == 5 {
        w.put(c.rotation as u32, 2);
    }
    if mode == 4 {
        w.put(c.index_selection as u32, 1);
    }
    let field = |row: usize, col: usize, p: u8| (st.ep[row][col] >> (8 - p)) as u32;
    for ch in channel_order(c.rotation) {
        let p = prec[ch];
        if p == 0 {
            continue;
        }
        // red's lowest bit is the p-bit, written after the colours
        let drop = (ch == 0 && c.has_pbits()) as u32;
        for e in 0..2 * subsets {
            w.put(field(e / 2, 2 * ch + e % 2, p) >> drop, p as u32 - drop);
        }
    }
    if mode == 1 {
        // one shared p-bit per subset
        for row in 0..subsets {
            w.put(field(row, 0, prec[0]) & 1, 1);
        }
    } else if c.has_pbits() {
        for e in 0..2 * subsets {
            w.put(field(e / 2, e % 2, prec[0]) & 1, 1);
        }
    }
    let anchors = c.anchors();
    let bits = c.index_bits();
    let mut set = |s: usize| {
        let b = bits[s] as u32;
        for (p, &ix) in st.ix[s].iter().enumerate() {
            let n = b - (anchors >> p & 1) as u32;
            w.put((ix >> (4 - b)) as u32, n);
        }
    };
    match mode {
        4 => {
            set(1);
            set(0);
        }
        5 => {
            set(0);
            set(1);
        }
        _ => set(0),
    }
    out[..16].copy_from_slice(&w.v.to_le_bytes());
}

/// BC7 block bytes -> control and state (inverse of [`pack`]). `None` for the
/// reserved mode (low byte zero). The anchor indices come back with their
/// top bit clear and, in mode 1, both endpoints of a subset get the shared
/// p-bit.
pub fn unpack(block: &[u8]) -> Option<(Control, State)> {
    let v = u128::from_le_bytes(block[..16].try_into().ok()?);
    let mode = (v as u8).trailing_zeros();
    if mode > 7 {
        return None;
    }
    let mut at = mode + 1;
    let mut take = |n: u32| {
        let x = if n == 0 {
            0
        } else {
            ((v >> at) & ((1u128 << n) - 1)) as u32
        };
        at += n;
        x
    };
    let mut c = Control {
        mode: mode as u8,
        ..Default::default()
    };
    c.partition = take(PARTITION_BITS[mode as usize] as u32) as u8;
    if mode == 4 || mode == 5 {
        c.rotation = take(2) as u8;
    }
    if mode == 4 {
        c.index_selection = take(1) as u8;
    }
    let subsets = c.subsets();
    let prec = c.precision();
    let mut st = State {
        subsets: subsets as u8,
        ..Default::default()
    };
    let mut raw = [[0u32; 8]; 3];
    for ch in channel_order(c.rotation) {
        let p = prec[ch];
        if p == 0 {
            continue;
        }
        let drop = (ch == 0 && c.has_pbits()) as u32;
        for e in 0..2 * subsets {
            raw[e / 2][2 * ch + e % 2] = take(p as u32 - drop) << drop;
        }
    }
    if mode == 1 {
        for r in raw.iter_mut().take(subsets) {
            let b = take(1);
            r[0] |= b;
            r[1] |= b;
        }
    } else if c.has_pbits() {
        for e in 0..2 * subsets {
            raw[e / 2][e % 2] |= take(1);
        }
    }
    for (ep, raw_row) in st.ep.iter_mut().zip(raw).take(subsets) {
        for col in 0..8 {
            let p = prec[col / 2];
            if p != 0 {
                ep[col] = (raw_row[col] << (8 - p)) as u8;
            }
        }
    }
    let anchors = c.anchors();
    let bits = c.index_bits();
    let mut set = |st: &mut State, s: usize| {
        let b = bits[s] as u32;
        for (p, ix) in st.ix[s].iter_mut().enumerate() {
            let n = b - (anchors >> p & 1) as u32;
            *ix = (take(n) << (4 - b)) as u8;
        }
    };
    match mode {
        4 => {
            set(&mut st, 1);
            set(&mut st, 0);
        }
        5 => {
            set(&mut st, 0);
            set(&mut st, 1);
        }
        _ => set(&mut st, 0),
    }
    Some((c, st))
}

/// The residual part of block `i`'s record applied to the two reference
/// states: the block's new state. `unk` receives the unknown-value masks.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn apply(
    c: Control,
    lanes: &LaneOutput,
    i: usize,
    er: &State,
    er_known: Option<&Known>,
    er_unk: &Unknown,
    xr: &State,
    xr_known: Option<&Known>,
    xr_unk: &Unknown,
    unk: &mut Unknown,
) -> State {
    let subsets = c.subsets();
    let prec = c.precision();
    let mut st = State {
        subsets: subsets as u8,
        ..Default::default()
    };
    // A dictionary entry whose subset count is unknown: every row it might lend is unknown.
    let common = subsets.min(er.subsets as usize);
    let subsets_unknown = er_known.map(|k| !k.subsets).unwrap_or(false);
    for (ch, &p) in prec.iter().enumerate() {
        if p == 0 {
            continue;
        }
        for row in 0..subsets {
            let from = if row < common { row } else { 0 };
            let lane = endpoint_lane(subsets, row, ch);
            for e in 0..2 {
                let col = 2 * ch + e;
                let res = match lane {
                    Some(l) => format::residual(lanes.lane(l)[2 * i + e], p as u32),
                    None => 0,
                };
                let base = er.ep[from][col] >> (8 - p);
                st.ep[row][col] = (base.wrapping_add(res) & (0xFFu16 >> (8 - p)) as u8) << (8 - p);
                let bad = match er_known {
                    Some(k) => k.ep[from][col] < p || (subsets_unknown && row > 0),
                    None => er_unk.ep[from] >> col & 1 != 0,
                };
                if bad {
                    unk.ep[row] |= 1 << col;
                }
            }
        }
    }
    let bits = c.index_bits();
    for set in 0..2 {
        let b = bits[set];
        if b == 0 {
            continue;
        }
        for (row, &lane) in INDEX_LANES[set].iter().enumerate() {
            for half in 0..2 {
                let (l, r) = index_pair(lanes.lane(lane)[2 * i + half], row);
                for (h, code) in [l, r].into_iter().enumerate() {
                    let p = 4 * row + 2 * half + h;
                    let res = format::residual(code, b as u32);
                    let base = xr.ix[set][p] >> (4 - b);
                    st.ix[set][p] = (base.wrapping_add(res) & (0xF >> (4 - b))) << (4 - b);
                    let bad = match xr_known {
                        Some(k) => k.ix[set][p] < b,
                        None => xr_unk.ix[set] >> p & 1 != 0,
                    };
                    if bad {
                        unk.ix[set] |= 1 << p;
                    }
                }
            }
        }
    }
    st
}

/// Turn the decoded lanes of a BC7 chunk into blocks.
pub fn assemble(lanes: &LaneOutput, t: &Tables) -> Rebuilt {
    let n = lanes.target / 2;
    let mut out = vec![0u8; n * 16];
    let mut hist: Vec<(State, Unknown)> = vec![(State::default(), Unknown::default()); 256];
    let zero = (State::default(), Unknown::default());
    let (mut unfitted, mut dict_refs, mut bad_controls) = (0usize, 0usize, 0usize);
    let (l0, l1, l15) = (lanes.lane(0), lanes.lane(1), lanes.lane(15));
    for i in 0..n {
        let (er, xr) = format::references(
            format::interleave(l0[2 * i], l0[2 * i + 1]),
            format::interleave(l1[2 * i], l1[2 * i + 1]),
        );
        let code = control_code(l15[2 * i], l15[2 * i + 1]);
        let Some(c) = t.control[code] else {
            // Unknown header: nothing can be written, and every later block that
            // refers to this one is unknown too.
            bad_controls += 1;
            unfitted += 1;
            hist[i & 255] = (
                State::default(),
                Unknown {
                    ep: [0xFF; 3],
                    ix: [0xFFFF; 2],
                },
            );
            continue;
        };
        let (es, xs) = (format::source(i, er), format::source(i, xr));
        dict_refs += matches!(es, Source::Dictionary(_)) as usize
            + matches!(xs, Source::Dictionary(_)) as usize;
        let pick = |s: Source| -> (&State, Option<&Known>, &Unknown) {
            match s {
                Source::Zero => (&zero.0, None, &zero.1),
                Source::Block(j) => (&hist[j & 255].0, None, &hist[j & 255].1),
                Source::Dictionary(k) => (&t.dictionary[k].0, Some(&t.dictionary[k].1), &zero.1),
            }
        };
        let (e, ek, eu) = pick(es);
        let (x, xk, xu) = pick(xs);
        let mut unk = Unknown::default();
        let st = apply(c, lanes, i, e, ek, eu, x, xk, xu, &mut unk);
        unfitted += (unk != Unknown::default()) as usize;
        pack(c, &st, &mut out[i * 16..i * 16 + 16]);
        hist[i & 255] = (st, unk);
    }
    Rebuilt {
        blocks: out,
        unfitted_blocks: unfitted,
        dictionary_refs: dict_refs,
        unknown_controls: bad_controls,
    }
}

/// Decode the 16 lane symbol streams of a BCPack BC7 chunk of `unpacked_len` bytes.
pub fn decode_symbols(chunk: &[u8], unpacked_len: usize) -> Result<LaneOutput, DecodeError> {
    rans::decode_slice(chunk, Mode::Bc7.alphabet_bits(), unpacked_len / 8)
}

/// BCPack BC7 chunk bytes -> BC7 blocks in stream order.
pub fn decode(chunk: &[u8], unpacked_len: usize, t: &Tables) -> Result<Rebuilt, DecodeError> {
    let lanes = decode_symbols(chunk, unpacked_len)?;
    Ok(assemble(&lanes, t))
}
