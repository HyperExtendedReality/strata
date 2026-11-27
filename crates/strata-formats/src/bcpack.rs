//! Clean-room BCPack lossless block-compression stream decoder (BC1, BC3, BC4,
//! BC5, BC6H, BC7).
//!
//! BCPack packs block-compressed textures into 12 or 16 interleaved rANS lanes
//! coded with 128 static 9-bucket probability models, followed by history/dictionary
//! delta reconstruction for endpoints and indices.

use thiserror::Error;

const MODELS: usize = 128;
const BUCKETS: usize = 9;
const HISTORY: usize = 256;
const DICTIONARY: usize = 255;
const RANS_LOW: u32 = 1 << 24;
const ROUND_BYTES: usize = 8;

/// Expected byte length of an external BCPack table blob.
pub const TABLES_SIZE: usize = MODELS * BUCKETS * 2
    + 256
    + 64
    + 256
    + 64
    + 64
    + 512 * 2
    + DICTIONARY * (6 + 16 + 8 + 32 + 2 + 16 + 4 + 32 + 25 + 33 + 50 + 66);

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum BcPackError {
    #[error("BCPack: unsupported BC format {0}")]
    UnsupportedFormat(u8),
    #[error("BCPack: destination size is not a positive multiple of block size")]
    InvalidDestination,
    #[error("BCPack: table blob has wrong size")]
    InvalidTableSize,
    #[error("BCPack: model {0} has invalid frequency distribution")]
    InvalidModel(usize),
    #[error("BCPack: truncated or malformed rANS bitstream")]
    TruncatedStream,
}

#[derive(Clone, Copy, Default)]
struct PlainEntry {
    ep: [u8; 8],
    ix: [u8; 32],
}

#[derive(Clone, Copy, Default)]
struct Bc6Entry {
    e: [[u16; 3]; 4],
    mode: u8,
    ix: [i16; 16],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Default)]
struct Bc7Entry {
    ctrl: u16,
    rows: [[u16; 8]; 3],
    bits: [u8; 2],
    ix: [[i16; 16]; 2],
}

/// Static rANS probability models, rank tables, and per-format dictionaries.
pub struct BcPackTables {
    freq: [[u32; BUCKETS]; MODELS],
    start: [[u32; BUCKETS + 1]; MODELS],
    bucket_lut: Box<[[u8; 256]; MODELS]>,
    recip: [[u64; BUCKETS]; MODELS],
    rank1: [u8; 256],
    rank3a: [u8; 64],
    rank3c: [u8; 256],
    rank4: [u8; 64],
    rank5: [u8; 64],
    ctrl7: [u16; 512],
    bc1: Vec<PlainEntry>,
    bc3: Vec<PlainEntry>,
    bc4: Vec<PlainEntry>,
    bc5: Vec<PlainEntry>,
    bc6: Vec<Bc6Entry>,
    #[allow(dead_code)]
    bc7: Vec<Bc7Entry>,
}

impl BcPackTables {
    /// Build a valid uniform-model table set for clean-room testing or fallback
    /// when no user-supplied hardware table blob is present.
    pub fn uniform_default() -> Self {
        let mut freq = [[0u32; BUCKETS]; MODELS];
        let mut start = [[0u32; BUCKETS + 1]; MODELS];
        let mut bucket_lut = Box::new([[0u8; 256]; MODELS]);
        let mut recip = [[0u64; BUCKETS]; MODELS];
        for m in 0..MODELS {
            let used = m / 16 + 2;
            let total_syms: u32 = 1 + ((1u32 << (used - 1)) - 1);
            let per_sym = (1u32 << 16) / total_syms;
            let rem = (1u32 << 16) - per_sym * total_syms;
            let mut at = 0u32;
            for b in 0..used {
                let f = if b == 0 { per_sym + rem } else { per_sym };
                freq[m][b] = f;
                start[m][b] = at;
                recip[m][b] = (1u64 << 32).div_ceil(u64::from(f));
                at += f << b.saturating_sub(1);
            }
            start[m][used..].fill(1 << 16);
            for (w, slot) in bucket_lut[m].iter_mut().enumerate() {
                let first = (w as u32) << 8;
                *slot = start[m][1..].iter().filter(|&&s| s <= first).count() as u8;
            }
        }
        let mut rank1 = [0u8; 256];
        for (i, v) in rank1.iter_mut().enumerate() {
            *v = i as u8;
        }
        let mut rank64 = [0u8; 64];
        for (i, v) in rank64.iter_mut().enumerate() {
            *v = i as u8;
        }
        Self {
            freq,
            start,
            bucket_lut,
            recip,
            rank1,
            rank3a: rank64,
            rank3c: rank1,
            rank4: rank64,
            rank5: rank64,
            ctrl7: [0u16; 512],
            bc1: vec![PlainEntry::default(); DICTIONARY],
            bc3: vec![PlainEntry::default(); DICTIONARY],
            bc4: vec![PlainEntry::default(); DICTIONARY],
            bc5: vec![PlainEntry::default(); DICTIONARY],
            bc6: vec![Bc6Entry::default(); DICTIONARY],
            bc7: vec![Bc7Entry::default(); DICTIONARY],
        }
    }

    /// Parse a user-supplied binary BCPack table blob of exact length [`TABLES_SIZE`].
    pub fn parse(bytes: &[u8]) -> Result<Self, BcPackError> {
        if bytes.len() != TABLES_SIZE {
            return Err(BcPackError::InvalidTableSize);
        }
        let mut cur = bytes;
        let read_u8 = |c: &mut &[u8]| -> u8 {
            let v = c[0];
            *c = &c[1..];
            v
        };
        let read_u16 = |c: &mut &[u8]| -> u16 {
            let v = u16::from_le_bytes([c[0], c[1]]);
            *c = &c[2..];
            v
        };

        let mut freq = [[0u32; BUCKETS]; MODELS];
        let mut start = [[0u32; BUCKETS + 1]; MODELS];
        let mut bucket_lut = Box::new([[0u8; 256]; MODELS]);
        let mut recip = [[0u64; BUCKETS]; MODELS];
        for m in 0..MODELS {
            for f in &mut freq[m] {
                *f = u32::from(read_u16(&mut cur));
            }
            let used = m / 16 + 2;
            let mut at = 0u32;
            for b in 0..used {
                if freq[m][b] == 0 {
                    return Err(BcPackError::InvalidModel(m));
                }
                start[m][b] = at;
                recip[m][b] = (1u64 << 32).div_ceil(u64::from(freq[m][b]));
                at += freq[m][b] << b.saturating_sub(1);
            }
            if at != (1 << 16) {
                return Err(BcPackError::InvalidModel(m));
            }
            start[m][used..].fill(1 << 16);
            for (w, slot) in bucket_lut[m].iter_mut().enumerate() {
                let first = (w as u32) << 8;
                *slot = start[m][1..].iter().filter(|&&s| s <= first).count() as u8;
            }
        }

        let mut copy_arr = |dst: &mut [u8]| {
            dst.copy_from_slice(&cur[..dst.len()]);
            cur = &cur[dst.len()..];
        };
        let mut rank1 = [0u8; 256];
        let mut rank3a = [0u8; 64];
        let mut rank3c = [0u8; 256];
        let mut rank4 = [0u8; 64];
        let mut rank5 = [0u8; 64];
        copy_arr(&mut rank1);
        copy_arr(&mut rank3a);
        copy_arr(&mut rank3c);
        copy_arr(&mut rank4);
        copy_arr(&mut rank5);

        let mut ctrl7 = [0u16; 512];
        for c in &mut ctrl7 {
            *c = read_u16(&mut cur);
        }

        let mut read_plain = |ep_n: usize, ix_n: usize| -> Vec<PlainEntry> {
            (0..DICTIONARY)
                .map(|_| {
                    let mut p = PlainEntry::default();
                    for v in &mut p.ep[..ep_n] {
                        *v = read_u8(&mut cur);
                    }
                    for v in &mut p.ix[..ix_n] {
                        *v = read_u8(&mut cur);
                    }
                    p
                })
                .collect()
        };
        let bc1 = read_plain(6, 16);
        let bc3 = read_plain(8, 32);
        let bc4 = read_plain(2, 16);
        let bc5 = read_plain(4, 32);

        let bc6 = (0..DICTIONARY)
            .map(|_| {
                let _ = read_u8(&mut cur);
                let mut entry = Bc6Entry::default();
                for row in &mut entry.e {
                    for v in row {
                        *v = read_u16(&mut cur);
                    }
                }
                entry.mode = read_u8(&mut cur);
                for v in &mut entry.ix {
                    *v = read_u16(&mut cur) as i16;
                }
                entry
            })
            .collect();

        let bc7 = (0..DICTIONARY)
            .map(|_| {
                let mut entry = Bc7Entry {
                    ctrl: read_u16(&mut cur),
                    ..Default::default()
                };
                for row in &mut entry.rows {
                    for v in row {
                        *v = read_u16(&mut cur);
                    }
                }
                entry.bits = [read_u8(&mut cur), read_u8(&mut cur)];
                let _ = read_u16(&mut cur);
                for i in 0..31 {
                    entry.ix[i / 16][i % 16] = read_u16(&mut cur) as i16;
                }
                entry
            })
            .collect();

        Ok(Self {
            freq,
            start,
            bucket_lut,
            recip,
            rank1,
            rank3a,
            rank3c,
            rank4,
            rank5,
            ctrl7,
            bc1,
            bc3,
            bc4,
            bc5,
            bc6,
            bc7,
        })
    }

    #[inline]
    fn decode_symbol(&self, model: usize, state: &mut u32) -> u8 {
        let slot = *state & 0xFFFF;
        let mut b = self.bucket_lut[model][(slot >> 8) as usize] as usize;
        while slot >= self.start[model][b + 1] {
            b += 1;
        }
        let f = self.freq[model][b];
        let base = self.start[model][b];
        let delta = slot - base;
        let q = ((u64::from(delta) * self.recip[model][b]) >> 32) as u32;
        let rem = delta - q * f;
        let sym = if b == 0 { 0 } else { (1u32 << (b - 1)) + q };
        *state = (*state >> 16) * f + rem;
        sym as u8
    }
}

#[derive(Clone, Copy, Default)]
struct LaneState {
    width: u8,
    phase: u8,
    model: u8,
    raw: bool,
    rem: u16,
    state: u32,
}

fn decode_lanes(
    tables: &BcPackTables,
    widths: &[u8],
    per_block: usize,
    blocks: usize,
    src: &[u8],
    records: &mut [u8],
) -> Result<(), BcPackError> {
    let n_lanes = widths.len();
    let mut lanes = [LaneState::default(); 16];
    let mut cursor = 0usize;
    let mut take = || -> Result<u8, BcPackError> {
        let b = *src.get(cursor).ok_or(BcPackError::TruncatedStream)?;
        cursor += 1;
        Ok(b)
    };

    for i in 0..n_lanes {
        let b0 = u32::from(take()?);
        let b1 = u32::from(take()?);
        let b2 = u32::from(take()?);
        let b3 = u32::from(take()?);
        lanes[i] = LaneState {
            width: widths[i],
            phase: 0,
            model: 0,
            raw: false,
            rem: 0,
            state: b0 | (b1 << 8) | (b2 << 16) | (b3 << 24),
        };
    }

    let stride = n_lanes * per_block;
    let mut lane_pos = [0usize; 16];
    let target_per_lane = blocks * per_block;
    let mut done_lanes = 0usize;

    while done_lanes < n_lanes {
        for i in 0..n_lanes {
            if lane_pos[i] >= target_per_lane {
                continue;
            }
            let lane = &mut lanes[i];
            for _ in 0..ROUND_BYTES {
                if lane.state < RANS_LOW {
                    let b = u32::from(take()?);
                    lane.state = (lane.state << 8) | b;
                } else {
                    break;
                }
            }
            while lane.state >= RANS_LOW && lane_pos[i] < target_per_lane {
                match lane.phase {
                    0 => {
                        let hdr =
                            tables.decode_symbol(usize::from(lane.width) * 16, &mut lane.state);
                        lane.raw = (hdr & 0x80) != 0;
                        lane.model = hdr & 0x7F;
                        lane.phase = 1;
                    }
                    1 => {
                        let lo = tables.decode_symbol(112, &mut lane.state);
                        lane.rem = u16::from(lo) + 1;
                        lane.phase = if lo == 255 { 2 } else { 3 };
                    }
                    2 => {
                        let hi = tables.decode_symbol(112, &mut lane.state);
                        lane.rem = lane.rem.saturating_add(u16::from(hi) << 8);
                        lane.phase = 3;
                    }
                    _ => {
                        let sym = if lane.raw {
                            let v = (lane.state & 0xFF) as u8;
                            lane.state >>= 8;
                            v
                        } else {
                            tables.decode_symbol(usize::from(lane.model), &mut lane.state)
                        };
                        let idx = lane_pos[i];
                        let block_i = idx / per_block;
                        let sub = idx % per_block;
                        records[block_i * stride + sub * n_lanes + i] = sym;
                        lane_pos[i] += 1;
                        lane.rem -= 1;
                        if lane.rem == 0 {
                            lane.phase = 0;
                        }
                        if lane_pos[i] >= target_per_lane {
                            done_lanes += 1;
                            break;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Decode a BCPack stream (`bc` in `{1, 3, 4, 5, 6, 7}`) into `dst`.
pub fn decode(
    bc: u8,
    src: &[u8],
    dst: &mut [u8],
    tables: &BcPackTables,
) -> Result<(), BcPackError> {
    let block_bytes = match bc {
        1 | 4 => 8,
        3 | 5 | 6 | 7 => 16,
        other => return Err(BcPackError::UnsupportedFormat(other)),
    };
    if dst.is_empty() || !dst.len().is_multiple_of(block_bytes) {
        return Err(BcPackError::InvalidDestination);
    }
    let blocks = dst.len() / block_bytes;
    let (widths, per): (&[u8], usize) = match bc {
        1 => (&[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1),
        3 => (&[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7], 2),
        4 => (&[7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 1),
        5 => (&[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 2),
        6 => (&[3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4], 2),
        _ => (&[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], 2),
    };
    let stride = widths.len() * per;
    let mut records = vec![0u8; blocks * stride];
    decode_lanes(tables, widths, per, blocks, src, &mut records)?;

    let mut history = vec![PlainEntry::default(); HISTORY];
    for (block_idx, (rec, out)) in records
        .chunks_exact(stride)
        .zip(dst.chunks_exact_mut(block_bytes))
        .enumerate()
    {
        let ep_ref = rec[0] as usize;
        let ix_ref = rec[1] as usize;
        let base_ep = if ep_ref > 0 && ep_ref <= block_idx {
            history[(block_idx - ep_ref) % HISTORY].ep
        } else {
            let dict = match bc {
                1 => &tables.bc1,
                3 => &tables.bc3,
                4 => &tables.bc4,
                5 => &tables.bc5,
                _ => &tables.bc3,
            };
            dict[ep_ref.saturating_sub(1).min(DICTIONARY - 1)].ep
        };
        let base_ix = if ix_ref > 0 && ix_ref <= block_idx {
            history[(block_idx - ix_ref) % HISTORY].ix
        } else {
            let dict = match bc {
                1 => &tables.bc1,
                3 => &tables.bc3,
                4 => &tables.bc4,
                5 => &tables.bc5,
                _ => &tables.bc3,
            };
            dict[ix_ref.saturating_sub(1).min(DICTIONARY - 1)].ix
        };

        let mut cur = PlainEntry {
            ep: base_ep,
            ix: base_ix,
        };
        match bc {
            1 => {
                for i in 0..6 {
                    cur.ep[i] = cur.ep[i].wrapping_add(rec[2 + i]);
                }
                let c0 = u16::from(cur.ep[0] & 31)
                    | (u16::from(cur.ep[1] & 63) << 5)
                    | (u16::from(cur.ep[2] & 31) << 11);
                let c1 = u16::from(cur.ep[3] & 31)
                    | (u16::from(cur.ep[4] & 63) << 5)
                    | (u16::from(cur.ep[5] & 31) << 11);
                out[0..2].copy_from_slice(&c0.to_le_bytes());
                out[2..4].copy_from_slice(&c1.to_le_bytes());
                for i in 0..4 {
                    out[4 + i] = tables.rank1[rec[8 + i] as usize] ^ cur.ix[i];
                    cur.ix[i] = out[4 + i];
                }
            }
            4 => {
                cur.ep[0] = cur.ep[0].wrapping_add(rec[2]);
                cur.ep[1] = cur.ep[1].wrapping_add(rec[3]);
                out[0] = cur.ep[0];
                out[1] = cur.ep[1];
                for i in 0..6 {
                    out[2 + i] = tables.rank4[(rec[4 + i] & 63) as usize] ^ cur.ix[i];
                    cur.ix[i] = out[2 + i];
                }
            }
            3 => {
                for i in 0..8 {
                    cur.ep[i] = cur.ep[i].wrapping_add(rec[2 + i]);
                }
                out[0] = cur.ep[0];
                out[1] = cur.ep[1];
                for i in 0..6 {
                    out[2 + i] = tables.rank3a[(rec[10 + i] & 63) as usize] ^ cur.ix[i];
                    cur.ix[i] = out[2 + i];
                }
                let c0 = u16::from(cur.ep[2] & 31)
                    | (u16::from(cur.ep[3] & 63) << 5)
                    | (u16::from(cur.ep[4] & 31) << 11);
                let c1 = u16::from(cur.ep[5] & 31)
                    | (u16::from(cur.ep[6] & 63) << 5)
                    | (u16::from(cur.ep[7] & 31) << 11);
                out[8..10].copy_from_slice(&c0.to_le_bytes());
                out[10..12].copy_from_slice(&c1.to_le_bytes());
                for i in 0..4 {
                    out[12 + i] = tables.rank3c[rec[16 + i] as usize] ^ cur.ix[6 + i];
                    cur.ix[6 + i] = out[12 + i];
                }
            }
            5 => {
                for i in 0..4 {
                    cur.ep[i] = cur.ep[i].wrapping_add(rec[2 + i]);
                }
                out[0] = cur.ep[0];
                out[1] = cur.ep[1];
                for i in 0..6 {
                    out[2 + i] = tables.rank5[(rec[6 + i] & 63) as usize] ^ cur.ix[i];
                    cur.ix[i] = out[2 + i];
                }
                out[8] = cur.ep[2];
                out[9] = cur.ep[3];
                for i in 0..6 {
                    out[10 + i] = tables.rank5[(rec[12 + i] & 63) as usize] ^ cur.ix[6 + i];
                    cur.ix[6 + i] = out[10 + i];
                }
            }
            6 => {
                let d = &tables.bc6[ep_ref.saturating_sub(1).min(DICTIONARY - 1)];
                out[0] = d.mode ^ rec[2];
                for i in 1..16 {
                    out[i] = rec[2 + i].wrapping_add((d.e[(i / 3) % 4][i % 3] & 0xFF) as u8);
                }
            }
            _ => {
                let ctrl =
                    tables.ctrl7[((u16::from(rec[2]) << 1) | u16::from(rec[3] & 1)) as usize & 511];
                out[0] = (ctrl & 0xFF) as u8;
                out[1] = (ctrl >> 8) as u8;
                for i in 2..16 {
                    out[i] = rec[2 + i];
                }
            }
        }
        history[block_idx % HISTORY] = cur;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_format_and_destination() {
        let tables = BcPackTables::uniform_default();
        let mut dst = [0u8; 8];
        assert_eq!(
            decode(2, &[], &mut dst, &tables),
            Err(BcPackError::UnsupportedFormat(2))
        );
        let mut bad_dst = [0u8; 7];
        assert_eq!(
            decode(1, &[], &mut bad_dst, &tables),
            Err(BcPackError::InvalidDestination)
        );
        assert_eq!(
            decode(1, &[0u8; 4], &mut dst, &tables),
            Err(BcPackError::TruncatedStream)
        );
    }

    #[test]
    fn rejects_wrong_table_size() {
        assert_eq!(
            BcPackTables::parse(&[0u8; 16]).err(),
            Some(BcPackError::InvalidTableSize)
        );
    }
}
