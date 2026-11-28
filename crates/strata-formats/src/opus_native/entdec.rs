//! The Opus range decoder (RFC 6716 §4.1), ported from libopus
//! `celt/entdec.c` / `celt/entcode.c` (BSD-3-Clause, Copyright 2001-2011
//! Xiph.Org, Timothy B. Terriberry).

const SYM_BITS: u32 = 8;
const CODE_BITS: u32 = 32;
const SYM_MAX: u32 = (1 << SYM_BITS) - 1;
const CODE_TOP: u32 = 1 << (CODE_BITS - 1);
const CODE_BOT: u32 = CODE_TOP >> SYM_BITS;
const CODE_EXTRA: u32 = (CODE_BITS - 2) % SYM_BITS + 1;
const UINT_BITS: u32 = 8;
const WINDOW_SIZE: u32 = 32;
/// Fractional bit resolution of [`RangeDecoder::tell_frac`] (1/8 bit).
pub const BITRES: u32 = 3;

/// `EC_ILOG`: the position of the highest set bit, plus one (0 for 0).
pub fn ilog(x: u32) -> u32 {
    32 - x.leading_zeros()
}

/// A range decoder over one Opus frame's bytes, plus the raw-bit reader
/// that consumes the same buffer from its end.
#[derive(Debug, Clone)]
pub struct RangeDecoder<'a> {
    buf: &'a [u8],
    storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: u32,
    nbits_total: i32,
    offs: u32,
    rng: u32,
    val: u32,
    ext: u32,
    rem: u32,
    /// Set when the stream decoded an impossible value.
    pub error: bool,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        let mut d = Self {
            buf,
            storage: buf.len() as u32,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: (CODE_BITS + 1 - ((CODE_BITS - CODE_EXTRA) / SYM_BITS) * SYM_BITS) as i32,
            offs: 0,
            rng: 1 << CODE_EXTRA,
            val: 0,
            ext: 0,
            rem: 0,
            error: false,
        };
        d.rem = d.read_byte();
        d.val = d.rng - 1 - (d.rem >> (SYM_BITS - CODE_EXTRA));
        d.normalize();
        d
    }

    /// Bytes available to the decoder (the frame size).
    pub fn storage(&self) -> u32 {
        self.storage
    }

    fn read_byte(&mut self) -> u32 {
        if self.offs < self.storage {
            let b = self.buf[self.offs as usize];
            self.offs += 1;
            u32::from(b)
        } else {
            0
        }
    }

    fn read_byte_from_end(&mut self) -> u32 {
        if self.end_offs < self.storage {
            self.end_offs += 1;
            u32::from(self.buf[(self.storage - self.end_offs) as usize])
        } else {
            0
        }
    }

    fn normalize(&mut self) {
        while self.rng <= CODE_BOT {
            self.nbits_total += SYM_BITS as i32;
            self.rng <<= SYM_BITS;
            let mut sym = self.rem;
            self.rem = self.read_byte();
            sym = ((sym << SYM_BITS) | self.rem) >> (SYM_BITS - CODE_EXTRA);
            self.val = ((self.val << SYM_BITS).wrapping_add(SYM_MAX & !sym)) & (CODE_TOP - 1);
        }
    }

    /// `ec_decode`: the cumulative frequency of the next symbol in `[0, ft)`.
    pub fn decode(&mut self, ft: u32) -> u32 {
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    /// `ec_decode_bin`: [`Self::decode`] with `ft = 1 << bits`.
    pub fn decode_bin(&mut self, bits: u32) -> u32 {
        self.ext = self.rng >> bits;
        let s = self.val / self.ext;
        (1 << bits) - (s + 1).min(1 << bits)
    }

    /// `ec_dec_update`: consumes the symbol `[fl, fh)` of total `ft`.
    pub fn update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext.wrapping_mul(ft - fh);
        self.val = self.val.wrapping_sub(s);
        self.rng = if fl > 0 {
            self.ext.wrapping_mul(fh - fl)
        } else {
            self.rng - s
        };
        self.normalize();
    }

    /// `ec_dec_bit_logp`: a bit whose probability of 1 is `1 / (1 << logp)`.
    pub fn bit_logp(&mut self, logp: u32) -> bool {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let ret = d < s;
        if !ret {
            self.val = d - s;
        }
        self.rng = if ret { s } else { r - s };
        self.normalize();
        ret
    }

    /// `ec_dec_icdf`: a symbol from an inverse CDF table of `1 << ftb` total.
    pub fn icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
        let mut s = self.rng;
        let d = self.val;
        let r = s >> ftb;
        let mut ret = 0usize;
        let mut t;
        loop {
            t = s;
            s = r.wrapping_mul(u32::from(icdf[ret]));
            if d >= s {
                break;
            }
            ret += 1;
        }
        self.val = d - s;
        self.rng = t - s;
        self.normalize();
        ret
    }

    /// `ec_dec_icdf16`: [`Self::icdf`] with 16-bit table entries.
    pub fn icdf16(&mut self, icdf: &[u16], ftb: u32) -> usize {
        let mut s = self.rng;
        let d = self.val;
        let r = s >> ftb;
        let mut ret = 0usize;
        let mut t;
        loop {
            t = s;
            s = r.wrapping_mul(u32::from(icdf[ret]));
            if d >= s {
                break;
            }
            ret += 1;
        }
        self.val = d - s;
        self.rng = t - s;
        self.normalize();
        ret
    }

    /// `ec_dec_uint`: a uniformly distributed integer in `[0, ft)`, `ft > 1`.
    pub fn uint(&mut self, ft: u32) -> u32 {
        let ftm = ft - 1;
        let mut ftb = ilog(ftm);
        if ftb > UINT_BITS {
            ftb -= UINT_BITS;
            let f = (ftm >> ftb) + 1;
            let s = self.decode(f);
            self.update(s, s + 1, f);
            let t = (s << ftb) | self.bits(ftb);
            if t <= ftm {
                return t;
            }
            self.error = true;
            ftm
        } else {
            let s = self.decode(ft);
            self.update(s, s + 1, ft);
            s
        }
    }

    /// `ec_dec_bits`: `bits` raw bits from the end of the buffer.
    pub fn bits(&mut self, bits: u32) -> u32 {
        let mut window = self.end_window;
        let mut available = self.nend_bits;
        if available < bits {
            loop {
                window |= self.read_byte_from_end() << available;
                available += SYM_BITS;
                if available > WINDOW_SIZE - SYM_BITS {
                    break;
                }
            }
        }
        let ret = if bits == 32 {
            window
        } else {
            window & ((1u32 << bits) - 1)
        };
        window = if bits >= 32 { 0 } else { window >> bits };
        available -= bits;
        self.end_window = window;
        self.nend_bits = available;
        self.nbits_total += bits as i32;
        ret
    }

    /// `ec_tell`: whole bits consumed so far (rounded up).
    pub fn tell(&self) -> i32 {
        self.nbits_total - ilog(self.rng) as i32
    }

    /// `ec_tell_frac`: bits consumed in 1/8-bit units.
    pub fn tell_frac(&self) -> u32 {
        const CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
        let nbits = (self.nbits_total as u32) << BITRES;
        let mut l = ilog(self.rng);
        let r = self.rng >> (l - 16);
        let mut b = (r >> 12) - 8;
        if r > CORRECTION[b as usize] {
            b += 1;
        }
        l = (l << 3) + b;
        nbits.wrapping_sub(l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_decoder_has_used_one_bit() {
        // libopus: "a newly initialized encoder or decoder claims to have
        // used 1 bit".
        let d = RangeDecoder::new(&[0u8; 8]);
        assert_eq!(d.tell(), 1);
        assert_eq!(d.tell_frac(), 8);
    }

    #[test]
    fn raw_bits_come_from_the_end() {
        let buf = [0u8, 0, 0, 0b1010_0101, 0xC3];
        let mut d = RangeDecoder::new(&buf);
        assert_eq!(d.bits(8), 0xC3);
        assert_eq!(d.bits(4), 0b0101);
        assert_eq!(d.bits(4), 0b1010);
    }

    #[test]
    fn ilog_matches_ec_ilog() {
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(255), 8);
        assert_eq!(ilog(256), 9);
        assert_eq!(ilog(u32::MAX), 32);
    }
}
