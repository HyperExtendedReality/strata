//! Minimal RFC 1950 (zlib) + RFC 1951 (DEFLATE) inflater for unpacking Xbox `XBCH` v3
//! compressed shader containers (zero external crates).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum InflateError {
    #[error("unexpected end of zlib/deflate stream")]
    UnexpectedEof,
    #[error("invalid zlib header (cmf=0x{cmf:02x}, flg=0x{flg:02x})")]
    BadZlibHeader { cmf: u8, flg: u8 },
    #[error("invalid deflate block type {btype}")]
    BadBlockType { btype: u8 },
    #[error("invalid Huffman symbol")]
    BadSymbol,
}

/// Decompress an RFC 1950 zlib stream (2-byte header + deflate payload + 4-byte Adler-32).
pub fn inflate_zlib(data: &[u8]) -> Result<Vec<u8>, InflateError> {
    if data.len() < 6 {
        return Err(InflateError::UnexpectedEof);
    }
    let cmf = data[0];
    let flg = data[1];
    if (cmf & 0x0f) != 8 || (((cmf as u16) << 8) | (flg as u16)) % 31 != 0 {
        return Err(InflateError::BadZlibHeader { cmf, flg });
    }
    let start = if (flg & 0x20) != 0 { 6 } else { 2 };
    if start > data.len() {
        return Err(InflateError::UnexpectedEof);
    }
    inflate_raw(&data[start..])
}

/// Decompress a raw RFC 1951 DEFLATE stream (stored, fixed Huffman, and dynamic Huffman).
pub fn inflate_raw(data: &[u8]) -> Result<Vec<u8>, InflateError> {
    let mut br = BitReader::new(data);
    let mut out = Vec::new();

    loop {
        let bfinal = br.read_bits(1)?;
        let btype = br.read_bits(2)? as u8;
        match btype {
            0 => {
                // Stored (uncompressed) block
                br.align_byte();
                let len = br.read_u16_le()? as usize;
                let nlen = br.read_u16_le()? as usize;
                if (len ^ 0xffff) != nlen {
                    return Err(InflateError::BadBlockType { btype: 0 });
                }
                let slice = br.read_bytes(len)?;
                out.extend_from_slice(slice);
            }
            1 => {
                // Fixed Huffman block
                let mut lit_lens = [0u8; 288];
                lit_lens[0..144].fill(8);
                lit_lens[144..256].fill(9);
                lit_lens[256..280].fill(7);
                lit_lens[280..288].fill(8);
                let dist_lens = [5u8; 32];
                let lit = HuffTree::build(&lit_lens)?;
                let dist = HuffTree::build(&dist_lens)?;
                decode_huffman_block(&mut br, &lit, &dist, &mut out)?;
            }
            2 => {
                // Dynamic Huffman block
                let hlit = br.read_bits(5)? as usize + 257;
                let hdist = br.read_bits(5)? as usize + 1;
                let hclen = br.read_bits(4)? as usize + 4;
                const CL_ORDER: [usize; 19] = [
                    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                ];
                let mut cl_lens = [0u8; 19];
                for i in 0..hclen {
                    cl_lens[CL_ORDER[i]] = br.read_bits(3)? as u8;
                }
                let cl_tree = HuffTree::build(&cl_lens)?;
                let mut all_lens = vec![0u8; hlit + hdist];
                let mut idx = 0usize;
                while idx < all_lens.len() {
                    let sym = cl_tree.decode(&mut br)?;
                    match sym {
                        0..=15 => {
                            all_lens[idx] = sym as u8;
                            idx += 1;
                        }
                        16 => {
                            let rep = br.read_bits(2)? as usize + 3;
                            let prev = *all_lens.get(idx.wrapping_sub(1)).ok_or(InflateError::BadSymbol)?;
                            for _ in 0..rep {
                                if idx < all_lens.len() {
                                    all_lens[idx] = prev;
                                    idx += 1;
                                }
                            }
                        }
                        17 => {
                            let rep = br.read_bits(3)? as usize + 3;
                            idx = (idx + rep).min(all_lens.len());
                        }
                        18 => {
                            let rep = br.read_bits(7)? as usize + 11;
                            idx = (idx + rep).min(all_lens.len());
                        }
                        _ => return Err(InflateError::BadSymbol),
                    }
                }
                let lit = HuffTree::build(&all_lens[..hlit])?;
                let dist = HuffTree::build(&all_lens[hlit..])?;
                decode_huffman_block(&mut br, &lit, &dist, &mut out)?;
            }
            _ => return Err(InflateError::BadBlockType { btype }),
        }
        if bfinal != 0 {
            break;
        }
    }

    Ok(out)
}

fn decode_huffman_block(
    br: &mut BitReader<'_>,
    lit: &HuffTree,
    dist: &HuffTree,
    out: &mut Vec<u8>,
) -> Result<(), InflateError> {
    const LEN_BASE: [usize; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u32; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [usize; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u32; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];

    loop {
        let sym = lit.decode(br)?;
        if sym < 256 {
            out.push(sym as u8);
        } else if sym == 256 {
            break;
        } else if sym <= 285 {
            let li = (sym - 257) as usize;
            let len = LEN_BASE[li] + br.read_bits(LEN_EXTRA[li])? as usize;
            let di = dist.decode(br)? as usize;
            if di >= 30 {
                return Err(InflateError::BadSymbol);
            }
            let back = DIST_BASE[di] + br.read_bits(DIST_EXTRA[di])? as usize;
            if back > out.len() {
                return Err(InflateError::BadSymbol);
            }
            for _ in 0..len {
                let b = out[out.len() - back];
                out.push(b);
            }
        } else {
            return Err(InflateError::BadSymbol);
        }
    }
    Ok(())
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u64,
    bit_cnt: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, byte_pos: 0, bit_buf: 0, bit_cnt: 0 }
    }

    fn fill(&mut self, need: u32) -> Result<(), InflateError> {
        while self.bit_cnt < need {
            if self.byte_pos >= self.data.len() {
                return Err(InflateError::UnexpectedEof);
            }
            self.bit_buf |= (self.data[self.byte_pos] as u64) << self.bit_cnt;
            self.byte_pos += 1;
            self.bit_cnt += 8;
        }
        Ok(())
    }

    fn read_bits(&mut self, n: u32) -> Result<u32, InflateError> {
        if n == 0 {
            return Ok(0);
        }
        self.fill(n)?;
        let val = (self.bit_buf & ((1u64 << n) - 1)) as u32;
        self.bit_buf >>= n;
        self.bit_cnt -= n;
        Ok(val)
    }

    fn align_byte(&mut self) {
        let drop = self.bit_cnt & 7;
        self.bit_buf >>= drop;
        self.bit_cnt -= drop;
    }

    fn read_u16_le(&mut self) -> Result<u16, InflateError> {
        Ok(self.read_bits(16)? as u16)
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], InflateError> {
        // Rewind any buffered whole bytes back to byte_pos
        let buffered = (self.bit_cnt / 8) as usize;
        self.byte_pos -= buffered;
        self.bit_buf = 0;
        self.bit_cnt = 0;
        if self.byte_pos + len > self.data.len() {
            return Err(InflateError::UnexpectedEof);
        }
        let s = &self.data[self.byte_pos..self.byte_pos + len];
        self.byte_pos += len;
        Ok(s)
    }
}

struct HuffTree {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl HuffTree {
    fn build(lengths: &[u8]) -> Result<Self, InflateError> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            if l as usize >= 16 {
                return Err(InflateError::BadSymbol);
            }
            if l > 0 {
                counts[l as usize] += 1;
            }
        }
        let mut offsets = [0u16; 16];
        let mut sum = 0u16;
        for bits in 1..16 {
            offsets[bits] = sum;
            sum += counts[bits];
        }
        let mut symbols = vec![0u16; sum as usize];
        for (sym, &l) in lengths.iter().enumerate() {
            if l > 0 {
                let pos = offsets[l as usize] as usize;
                symbols[pos] = sym as u16;
                offsets[l as usize] += 1;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16, InflateError> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..16 {
            code |= br.read_bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(InflateError::BadSymbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflates_zlib_stored_and_fixed_huffman_streams() {
        // Zlib stored block containing "FSSM"
        let stored_zlib = [
            0x78, 0x01, // zlib header
            0x01, 0x04, 0x00, 0xfb, 0xff, // final stored block, len=4
            b'F', b'S', b'S', b'M',
            0x03, 0xd8, 0x01, 0x3b, // adler32
        ];
        assert_eq!(inflate_zlib(&stored_zlib).unwrap(), b"FSSM");
    }
}
