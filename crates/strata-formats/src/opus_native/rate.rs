//! CELT bit allocation (decoder side), ported from libopus `celt/rate.c`,
//! `celt/rate.h` and `init_caps` in `celt/celt.c` (BSD-3-Clause, Copyright
//! 2007-2009 Xiph.Org, CSIRO, Timothy B. Terriberry).

use super::celt_tables as t;
use super::entdec::{BITRES, RangeDecoder};

pub const NB_EBANDS: usize = 21;
const NB_ALLOC_VECTORS: usize = 11;
const LOG_MAX_PSEUDO: usize = 6;
pub const MAX_FINE_BITS: i32 = 8;
const FINE_OFFSET: i32 = 21;
pub const QTHETA_OFFSET: i32 = 4;
pub const QTHETA_OFFSET_TWOPHASE: i32 = 16;
const ALLOC_STEPS: i32 = 6;

fn eb(i: usize) -> i32 {
    i32::from(t::EBAND5MS[i])
}

/// `get_pulses`: the pulse count of pseudo-pulse index `i`.
pub fn get_pulses(i: i32) -> i32 {
    if i < 8 { i } else { (8 + (i & 7)) << ((i >> 3) - 1) }
}

fn cache_row(band: usize, lm: i32) -> &'static [u8] {
    let idx = t::CACHE_INDEX50[(lm as usize + 1) * NB_EBANDS + band];
    &t::CACHE_BITS50[idx as usize..]
}

/// `bits2pulses`: the pseudo-pulse count closest to `bits` (1/8 bits).
pub fn bits2pulses(band: usize, lm: i32, bits: i32) -> i32 {
    let cache = cache_row(band, lm);
    let mut lo = 0i32;
    let mut hi = i32::from(cache[0]);
    let bits = bits - 1;
    for _ in 0..LOG_MAX_PSEUDO {
        let mid = (lo + hi + 1) >> 1;
        if i32::from(cache[mid as usize]) >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lo_bits = if lo == 0 { -1 } else { i32::from(cache[lo as usize]) };
    if bits - lo_bits <= i32::from(cache[hi as usize]) - bits {
        lo
    } else {
        hi
    }
}

/// `pulses2bits`: the cost (1/8 bits) of `pulses` pseudo-pulses.
pub fn pulses2bits(band: usize, lm: i32, pulses: i32) -> i32 {
    if pulses == 0 {
        0
    } else {
        i32::from(cache_row(band, lm)[pulses as usize]) + 1
    }
}

/// `init_caps`: the per-band allocation caps.
pub fn init_caps(cap: &mut [i32; NB_EBANDS], lm: i32, c: i32) {
    for (i, cp) in cap.iter_mut().enumerate() {
        let n = (eb(i + 1) - eb(i)) << lm;
        let caps = i32::from(t::CACHE_CAPS50[NB_EBANDS * (2 * lm + c - 1) as usize + i]);
        *cp = (caps + 64) * c * n >> 2;
    }
}

/// What [`compute_allocation`] decides for a frame.
#[derive(Debug, Clone, Default)]
pub struct Allocation {
    pub coded_bands: usize,
    pub intensity: usize,
    pub dual_stereo: bool,
    pub balance: i32,
    pub pulses: [i32; NB_EBANDS],
    pub ebits: [i32; NB_EBANDS],
    pub fine_priority: [i32; NB_EBANDS],
}

#[allow(clippy::too_many_arguments)]
fn interp_bits2pulses(
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: &[i32; NB_EBANDS],
    bits2: &[i32; NB_EBANDS],
    thresh: &[i32; NB_EBANDS],
    cap: &[i32; NB_EBANDS],
    mut total: i32,
    skip_rsv: i32,
    mut intensity_rsv: i32,
    mut dual_stereo_rsv: i32,
    c: i32,
    lm: i32,
    dec: &mut RangeDecoder<'_>,
    a: &mut Allocation,
) {
    let alloc_floor = c << BITRES;
    let stereo = i32::from(c > 1);
    let log_m = lm << BITRES;
    let (mut lo, mut hi) = (0i32, 1i32 << ALLOC_STEPS);
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum = 0i32;
        let mut done = false;
        for j in (start..end).rev() {
            let tmp = bits1[j] + (mid * bits2[j] >> ALLOC_STEPS);
            if tmp >= thresh[j] || done {
                done = true;
                psum += tmp.min(cap[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let mut psum = 0i32;
    let mut done = false;
    let bits = &mut a.pulses;
    for j in (start..end).rev() {
        let mut tmp = bits1[j] + (lo * bits2[j] >> ALLOC_STEPS);
        if tmp < thresh[j] && !done {
            tmp = if tmp >= alloc_floor { alloc_floor } else { 0 };
        } else {
            done = true;
        }
        tmp = tmp.min(cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }
    let mut coded_bands = end;
    loop {
        let j = coded_bands - 1;
        if j <= skip_start {
            total += skip_rsv;
            break;
        }
        let mut left = total - psum;
        let percoeff = left / (eb(coded_bands) - eb(start));
        left -= (eb(coded_bands) - eb(start)) * percoeff;
        let rem = (left - (eb(j) - eb(start))).max(0);
        let band_width = eb(coded_bands) - eb(j);
        let mut band_bits = bits[j] + percoeff * band_width + rem;
        if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
            if dec.bit_logp(1) {
                break;
            }
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        psum -= bits[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = i32::from(t::LOG2_FRAC_TABLE[j - start]);
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
        coded_bands -= 1;
    }
    a.intensity = if intensity_rsv > 0 {
        start + dec.uint((coded_bands + 1 - start) as u32) as usize
    } else {
        0
    };
    if a.intensity <= start {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    a.dual_stereo = dual_stereo_rsv > 0 && dec.bit_logp(1);
    let mut left = total - psum;
    let percoeff = left / (eb(coded_bands) - eb(start));
    left -= (eb(coded_bands) - eb(start)) * percoeff;
    for j in start..coded_bands {
        bits[j] += percoeff * (eb(j + 1) - eb(j));
    }
    for j in start..coded_bands {
        let tmp = left.min(eb(j + 1) - eb(j));
        bits[j] += tmp;
        left -= tmp;
    }
    let mut balance = 0i32;
    let ebits = &mut a.ebits;
    let fine = &mut a.fine_priority;
    let mut j = start;
    while j < coded_bands {
        let n0 = eb(j + 1) - eb(j);
        let n = n0 << lm;
        let bit = bits[j] + balance;
        let mut excess;
        if n > 1 {
            excess = (bit - cap[j]).max(0);
            bits[j] = bit - excess;
            let den = c * n + i32::from(c == 2 && n > 2 && !a.dual_stereo && j < a.intensity);
            let nclogn = den * (i32::from(t::LOGN400[j]) + log_m);
            let mut offset = (nclogn >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += den << BITRES >> 2;
            }
            if bits[j] + offset < den * 2 << BITRES {
                offset += nclogn >> 2;
            } else if bits[j] + offset < den * 3 << BITRES {
                offset += nclogn >> 3;
            }
            ebits[j] = (bits[j] + offset + (den << (BITRES - 1))).max(0);
            ebits[j] = (ebits[j] / den) >> BITRES;
            if c * ebits[j] > (bits[j] >> BITRES) {
                ebits[j] = bits[j] >> stereo >> BITRES;
            }
            ebits[j] = ebits[j].min(MAX_FINE_BITS);
            fine[j] = i32::from(ebits[j] * (den << BITRES) >= bits[j] + offset);
            bits[j] -= c * ebits[j] << BITRES;
        } else {
            excess = (bit - (c << BITRES)).max(0);
            bits[j] = bit - excess;
            ebits[j] = 0;
            fine[j] = 1;
        }
        if excess > 0 {
            let extra_fine = (excess >> (stereo + BITRES as i32)).min(MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = extra_fine * c << BITRES;
            fine[j] = i32::from(extra_bits >= excess - balance);
            excess -= extra_bits;
        }
        balance = excess;
        j += 1;
    }
    a.balance = balance;
    while j < end {
        ebits[j] = bits[j] >> stereo >> BITRES;
        bits[j] = 0;
        fine[j] = i32::from(ebits[j] < 1);
        j += 1;
    }
    a.coded_bands = coded_bands;
}

/// `clt_compute_allocation` (decode): per-band pulse bits, fine energy bits
/// and priorities for a frame with `total` 1/8 bits left.
#[allow(clippy::too_many_arguments)]
pub fn compute_allocation(
    start: usize,
    end: usize,
    offsets: &[i32; NB_EBANDS],
    cap: &[i32; NB_EBANDS],
    alloc_trim: i32,
    total: i32,
    c: i32,
    lm: i32,
    dec: &mut RangeDecoder<'_>,
) -> Allocation {
    let mut total = total.max(0);
    let len = NB_EBANDS;
    let mut skip_start = start;
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;
    let (mut intensity_rsv, mut dual_stereo_rsv) = (0, 0);
    if c == 2 {
        intensity_rsv = i32::from(t::LOG2_FRAC_TABLE[end - start]);
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }
    let mut bits1 = [0i32; NB_EBANDS];
    let mut bits2 = [0i32; NB_EBANDS];
    let mut thresh = [0i32; NB_EBANDS];
    let mut trim_offset = [0i32; NB_EBANDS];
    for j in start..end {
        let n = eb(j + 1) - eb(j);
        thresh[j] = (c << BITRES).max((3 * n << lm << BITRES) >> 4);
        trim_offset[j] =
            c * n * (alloc_trim - 5 - lm) * (end - j - 1) as i32 * (1 << (lm + BITRES as i32)) >> 6;
        if n << lm == 1 {
            trim_offset[j] -= c << BITRES;
        }
    }
    let alloc = |v: usize, j: usize| i32::from(t::BAND_ALLOCATION[v * len + j]);
    let (mut lo, mut hi) = (1i32, NB_ALLOC_VECTORS as i32 - 1);
    loop {
        let mut done = false;
        let mut psum = 0i32;
        let mid = (lo + hi) >> 1;
        for j in (start..end).rev() {
            let n = eb(j + 1) - eb(j);
            let mut bitsj = c * n * alloc(mid as usize, j) << lm >> 2;
            if bitsj > 0 {
                bitsj = (bitsj + trim_offset[j]).max(0);
            }
            bitsj += offsets[j];
            if bitsj >= thresh[j] || done {
                done = true;
                psum += bitsj.min(cap[j]);
            } else if bitsj >= c << BITRES {
                psum += c << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
        if lo > hi {
            break;
        }
    }
    hi = lo;
    lo -= 1;
    for j in start..end {
        let n = eb(j + 1) - eb(j);
        let mut b1 = c * n * alloc(lo as usize, j) << lm >> 2;
        let mut b2 = if hi as usize >= NB_ALLOC_VECTORS {
            cap[j]
        } else {
            c * n * alloc(hi as usize, j) << lm >> 2
        };
        if b1 > 0 {
            b1 = (b1 + trim_offset[j]).max(0);
        }
        if b2 > 0 {
            b2 = (b2 + trim_offset[j]).max(0);
        }
        if lo > 0 {
            b1 += offsets[j];
        }
        b2 += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        b2 = (b2 - b1).max(0);
        bits1[j] = b1;
        bits2[j] = b2;
    }
    let mut a = Allocation::default();
    interp_bits2pulses(
        start,
        end,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        skip_rsv,
        intensity_rsv,
        dual_stereo_rsv,
        c,
        lm,
        dec,
        &mut a,
    );
    a
}
