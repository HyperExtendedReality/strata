//! CELT band decoding (float build): PVQ unquantisation and spreading
//! (`celt/vq.c`), and band splitting, stereo, folding, time/frequency
//! changes and anti-collapse (`celt/bands.c`), ported from libopus
//! (BSD-3-Clause, Copyright 2007-2009 Xiph.Org, CSIRO, Jean-Marc Valin,
//! Timothy B. Terriberry).
//!
//! libopus passes pointers into shared buffers; here the same regions are
//! passed as disjoint slices (the regions libopus aliases never overlap).

use super::celt_tables as t;
use super::cwrs::decode_pulses;
use super::entdec::{BITRES, RangeDecoder};
use super::rate::{
    NB_EBANDS, QTHETA_OFFSET, QTHETA_OFFSET_TWOPHASE, bits2pulses, get_pulses, pulses2bits,
};

pub const SPREAD_NONE: i32 = 0;
pub const SPREAD_AGGRESSIVE: i32 = 3;
const EPSILON: f32 = 1e-15;

fn eb(i: usize) -> i32 {
    i32::from(t::EBAND5MS[i])
}

pub fn lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + i32::from(a as i16) * i32::from(b as i16)) >> 15
}

fn bitexact_cos(x: i16) -> i16 {
    let tmp = (4096 + i32::from(x) * i32::from(x)) >> 13;
    let x2 = tmp;
    let x2 = (32767 - x2) + frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2)));
    (1 + x2) as i16
}

fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = super::entdec::ilog(icos as u32) as i32;
    let ls = super::entdec::ilog(isin as u32) as i32;
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

fn isqrt32(v: u32) -> u32 {
    // libopus' isqrt32 is exact: floor(sqrt(v)).
    let mut r = (v as f64).sqrt() as u32;
    while u64::from(r) * u64::from(r) > u64::from(v) {
        r -= 1;
    }
    while u64::from(r + 1) * u64::from(r + 1) <= u64::from(v) {
        r += 1;
    }
    r
}

// --- vq.c -----------------------------------------------------------------

fn exp_rotation1(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    let ms = -s;
    if len < stride {
        return;
    }
    for i in 0..len - stride {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
    }
    if len < 2 * stride + 1 {
        return;
    }
    for i in (0..=len - 2 * stride - 1).rev() {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
    }
}

fn exp_rotation(x: &mut [f32], len: usize, dir: i32, stride: usize, k: i32, spread: i32) {
    const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
    if 2 * k >= len as i32 || spread == SPREAD_NONE {
        return;
    }
    let factor = SPREAD_FACTOR[(spread - 1) as usize];
    let gain = len as f32 / (len as i32 + factor * k) as f32;
    let theta = 0.5 * (gain * gain);
    let c = (0.5 * std::f32::consts::PI * theta).cos();
    let s = (0.5 * std::f32::consts::PI * (1.0 - theta)).cos();
    let mut stride2 = 0usize;
    if len >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }
    let l = len / stride;
    for i in 0..stride {
        let xs = &mut x[i * l..i * l + l];
        if dir < 0 {
            if stride2 != 0 {
                exp_rotation1(xs, l, stride2, s, c);
            }
            exp_rotation1(xs, l, 1, c, s);
        } else {
            exp_rotation1(xs, l, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(xs, l, stride2, s, -c);
            }
        }
    }
}

fn extract_collapse_mask(iy: &[i32], n: usize, b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = n / b;
    let mut mask = 0u32;
    for i in 0..b {
        let tmp = iy[i * n0..i * n0 + n0].iter().fold(0, |a, &v| a | v);
        mask |= u32::from(tmp != 0) << i;
    }
    mask
}

fn alg_unquant(
    x: &mut [f32],
    n: usize,
    k: i32,
    spread: i32,
    b: usize,
    dec: &mut RangeDecoder<'_>,
    gain: f32,
) -> u32 {
    let mut iy = vec![0i32; n];
    let ryy = decode_pulses(&mut iy, n, k as usize, dec);
    let g = gain / ryy.sqrt();
    for (xv, &y) in x[..n].iter_mut().zip(&iy) {
        *xv = g * y as f32;
    }
    exp_rotation(x, n, -1, b, k, spread);
    extract_collapse_mask(&iy, n, b)
}

pub fn renormalise_vector(x: &mut [f32], n: usize, gain: f32) {
    let e = EPSILON + x[..n].iter().map(|v| v * v).sum::<f32>();
    let g = gain / e.sqrt();
    for v in &mut x[..n] {
        *v *= g;
    }
}

// --- bands.c --------------------------------------------------------------

fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, n: usize) {
    let mut xp = 0f32;
    let mut side = 0f32;
    for j in 0..n {
        xp += y[j] * x[j];
        side += y[j] * y[j];
    }
    let xp = mid * xp;
    let mid2 = mid;
    let el = mid2 * mid2 + side - 2.0 * xp;
    let er = mid2 * mid2 + side + 2.0 * xp;
    if er < 6e-4 || el < 6e-4 {
        y[..n].copy_from_slice(&x[..n]);
        return;
    }
    let lgain = 1.0 / el.sqrt();
    let rgain = 1.0 / er.sqrt();
    for j in 0..n {
        let l = mid * x[j];
        let r = y[j];
        x[j] = lgain * (l - r);
        y[j] = rgain * (l + r);
    }
}

fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0f32; n];
    if hadamard {
        let ordery = &t::ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[ordery[i] as usize * n0 + j] = x[j * stride + i];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[i * n0 + j] = x[j * stride + i];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0f32; n];
    if hadamard {
        let ordery = &t::ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[ordery[i] as usize * n0 + j];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[i * n0 + j];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

pub fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    let n0 = n0 >> 1;
    let k = std::f32::consts::FRAC_1_SQRT_2;
    for i in 0..stride {
        for j in 0..n0 {
            let a = stride * 2 * j + i;
            let b = stride * (2 * j + 1) + i;
            let tmp1 = k * x[a];
            let tmp2 = k * x[b];
            x[a] = tmp1 + tmp2;
            x[b] = tmp1 - tmp2;
        }
    }
}

fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut n2 = 2 * n - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let mut qb = (b + n2 * offset) / n2;
    qb = qb.min(b - pulse_cap - (4 << BITRES));
    qb = qb.min(8 << BITRES);
    if qb < (1 << BITRES >> 1) {
        1
    } else {
        let qn = EXP2_TABLE8[(qb & 7) as usize] >> (14 - (qb >> BITRES));
        (qn + 1) >> 1 << 1
    }
}

/// `band_ctx`, decoder side.
pub struct BandCtx<'a, 'b> {
    pub dec: &'b mut RangeDecoder<'a>,
    pub i: usize,
    pub intensity: usize,
    pub spread: i32,
    pub tf_change: i32,
    pub remaining_bits: i32,
    pub seed: u32,
    pub disable_inv: bool,
    pub avoid_split_noise: bool,
}

struct Split {
    inv: bool,
    imid: i32,
    iside: i32,
    delta: i32,
    itheta: i32,
    qalloc: i32,
}

#[allow(clippy::too_many_arguments)]
fn compute_theta(
    ctx: &mut BandCtx<'_, '_>,
    n: usize,
    b: &mut i32,
    bb: usize,
    b0: usize,
    lm: i32,
    stereo: bool,
    fill: &mut u32,
) -> Split {
    let i = ctx.i;
    let pulse_cap = i32::from(t::LOGN400[i]) + lm * (1 << BITRES);
    let offset = (pulse_cap >> 1)
        - if stereo && n == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        };
    let mut qn = compute_qn(n as i32, *b, offset, pulse_cap, stereo);
    if stereo && i >= ctx.intensity {
        qn = 1;
    }
    let tell = ctx.dec.tell_frac() as i32;
    let mut itheta = 0i32;
    let mut inv = false;
    let dec = &mut *ctx.dec;
    if qn != 1 {
        if stereo && n > 2 {
            let p0 = 3i32;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            let fs = dec.decode(ft as u32) as i32;
            let x = if fs < (x0 + 1) * p0 {
                fs / p0
            } else {
                x0 + 1 + (fs - (x0 + 1) * p0)
            };
            let (fl, fh) = if x <= x0 {
                (p0 * x, p0 * (x + 1))
            } else {
                ((x - 1 - x0) + (x0 + 1) * p0, (x - x0) + (x0 + 1) * p0)
            };
            dec.update(fl as u32, fh as u32, ft as u32);
            itheta = x;
        } else if b0 > 1 || stereo {
            itheta = dec.uint((qn + 1) as u32) as i32;
        } else {
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            let fm = dec.decode(ft as u32) as i32;
            let (fl, fs);
            if fm < ((qn >> 1) * ((qn >> 1) + 1) >> 1) {
                itheta = (isqrt32(8 * fm as u32 + 1) as i32 - 1) >> 1;
                fs = itheta + 1;
                fl = itheta * (itheta + 1) >> 1;
            } else {
                itheta = (2 * (qn + 1) - isqrt32(8 * (ft - fm - 1) as u32 + 1) as i32) >> 1;
                fs = qn + 1 - itheta;
                fl = ft - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1);
            }
            dec.update(fl as u32, (fl + fs) as u32, ft as u32);
        }
        itheta = itheta * 16384 / qn;
    } else if stereo {
        if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            inv = dec.bit_logp(2);
        }
        if ctx.disable_inv {
            inv = false;
        }
        itheta = 0;
    }
    let qalloc = ctx.dec.tell_frac() as i32 - tell;
    *b -= qalloc;
    let (imid, iside, delta);
    if itheta == 0 {
        imid = 32767;
        iside = 0;
        *fill &= (1 << bb) - 1;
        delta = -16384;
    } else if itheta == 16384 {
        imid = 0;
        iside = 32767;
        *fill &= ((1 << bb) - 1) << bb;
        delta = 16384;
    } else {
        imid = i32::from(bitexact_cos(itheta as i16));
        iside = i32::from(bitexact_cos((16384 - itheta) as i16));
        delta = frac_mul16(((n - 1) << 7) as i32, bitexact_log2tan(iside, imid));
    }
    Split {
        inv,
        imid,
        iside,
        delta,
        itheta,
        qalloc,
    }
}

fn quant_band_n1(
    ctx: &mut BandCtx<'_, '_>,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    lowband_out: Option<&mut [f32]>,
) -> u32 {
    let mut one = |v: &mut f32, ctx: &mut BandCtx<'_, '_>| {
        let mut sign = 0;
        if ctx.remaining_bits >= 1 << BITRES {
            sign = ctx.dec.bits(1);
            ctx.remaining_bits -= 1 << BITRES;
        }
        *v = if sign != 0 { -1.0 } else { 1.0 };
    };
    one(&mut x[0], ctx);
    if let Some(y) = y {
        one(&mut y[0], ctx);
    }
    if let Some(lo) = lowband_out {
        lo[0] = x[0];
    }
    1
}

#[allow(clippy::too_many_arguments)]
fn quant_partition(
    ctx: &mut BandCtx<'_, '_>,
    x: &mut [f32],
    mut n: usize,
    mut b: i32,
    mut bb: usize,
    lowband: Option<&mut [f32]>,
    mut lm: i32,
    gain: f32,
    mut fill: u32,
) -> u32 {
    let i = ctx.i;
    let b0 = bb;
    let cache_idx = t::CACHE_INDEX50[((lm + 1) as usize) * NB_EBANDS + i];
    let cache = &t::CACHE_BITS50[cache_idx as usize..];
    if lm != -1 && b > i32::from(cache[cache[0] as usize]) + 12 && n > 2 {
        n >>= 1;
        lm -= 1;
        if bb == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        bb = (bb + 1) >> 1;
        let sp = compute_theta(ctx, n, &mut b, bb, b0, lm, false, &mut fill);
        let mid = (1.0 / 32768.0) * sp.imid as f32;
        let side = (1.0 / 32768.0) * sp.iside as f32;
        let mut delta = sp.delta;
        let itheta = sp.itheta;
        if b0 > 1 && (itheta & 0x3fff) != 0 {
            if itheta > 8192 {
                delta -= delta >> (4 - lm);
            } else {
                delta = 0.min(delta + ((n as i32) << BITRES >> (5 - lm)));
            }
        }
        let mut mbits = 0.max(b.min((b - delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= sp.qalloc;
        let (xl, xr) = x.split_at_mut(n);
        let (lb1, lb2) = match lowband {
            Some(l) => {
                let (a, c) = l.split_at_mut(n);
                (Some(a), Some(c))
            }
            None => (None, None),
        };
        let mut rebalance = ctx.remaining_bits;
        let mut cm;
        if mbits >= sbits {
            cm = quant_partition(ctx, xl, n, mbits, bb, lb1, lm, gain * mid, fill);
            rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            cm |= quant_partition(ctx, xr, n, sbits, bb, lb2, lm, gain * side, fill >> bb)
                << (b0 >> 1);
        } else {
            cm = quant_partition(ctx, xr, n, sbits, bb, lb2, lm, gain * side, fill >> bb)
                << (b0 >> 1);
            rebalance = sbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && itheta != 16384 {
                mbits += rebalance - (3 << BITRES);
            }
            cm |= quant_partition(ctx, xl, n, mbits, bb, lb1, lm, gain * mid, fill);
        }
        cm
    } else {
        let mut q = bits2pulses(i, lm, b);
        let mut curr_bits = pulses2bits(i, lm, q);
        ctx.remaining_bits -= curr_bits;
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(i, lm, q);
            ctx.remaining_bits -= curr_bits;
        }
        if q != 0 {
            let k = get_pulses(q);
            alg_unquant(x, n, k, ctx.spread, bb, ctx.dec, gain)
        } else {
            let cm_mask = ((1u64 << bb) - 1) as u32;
            fill &= cm_mask;
            if fill == 0 {
                x[..n].fill(0.0);
                0
            } else {
                let cm = match lowband {
                    None => {
                        for v in &mut x[..n] {
                            ctx.seed = lcg_rand(ctx.seed);
                            *v = ((ctx.seed as i32) >> 20) as f32;
                        }
                        cm_mask
                    }
                    Some(lb) => {
                        for j in 0..n {
                            ctx.seed = lcg_rand(ctx.seed);
                            let tmp = if ctx.seed & 0x8000 != 0 { 1.0 / 256.0 } else { -1.0 / 256.0 };
                            x[j] = lb[j] + tmp;
                        }
                        fill
                    }
                };
                renormalise_vector(x, n, gain);
                cm
            }
        }
    }
}

/// `quant_band` (decode). `scratch`: libopus' `lowband_scratch` is
/// available (the folding source may then be modified on a copy).
#[allow(clippy::too_many_arguments)]
fn quant_band(
    ctx: &mut BandCtx<'_, '_>,
    x: &mut [f32],
    n: usize,
    b: i32,
    mut bb: usize,
    lowband: Option<&mut [f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    gain: f32,
    scratch: bool,
    mut fill: u32,
) -> u32 {
    let n0 = n;
    let mut n_b = n;
    let mut b0 = bb;
    let mut time_divide = 0;
    let mut recombine = 0;
    let long_blocks = b0 == 1;
    let mut tf_change = ctx.tf_change;
    n_b /= bb;
    if n == 1 {
        return quant_band_n1(ctx, x, None, lowband_out);
    }
    if tf_change > 0 {
        recombine = tf_change;
    }
    // The folding source: a private copy when libopus would use the scratch
    // buffer, else the shared `norm` region itself.
    let mut copy;
    let mut lowband: Option<&mut [f32]> = lowband;
    if scratch
        && lowband.is_some()
        && (recombine != 0 || ((n_b & 1) == 0 && tf_change < 0) || b0 > 1)
    {
        copy = lowband.as_deref().map(|l| l[..n].to_vec()).unwrap_or_default();
        lowband = Some(&mut copy[..]);
    }
    for k in 0..recombine {
        if let Some(l) = lowband.as_deref_mut() {
            haar1(l, n >> k, 1 << k);
        }
        fill = u32::from(t::BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize])
            | u32::from(t::BIT_INTERLEAVE_TABLE[(fill >> 4) as usize]) << 2;
    }
    bb >>= recombine;
    n_b <<= recombine;
    while (n_b & 1) == 0 && tf_change < 0 {
        if let Some(l) = lowband.as_deref_mut() {
            haar1(l, n_b, bb);
        }
        fill |= fill << bb;
        bb <<= 1;
        n_b >>= 1;
        time_divide += 1;
        tf_change += 1;
    }
    b0 = bb;
    let n_b0 = n_b;
    if b0 > 1
        && let Some(l) = lowband.as_deref_mut()
    {
        deinterleave_hadamard(l, n_b >> recombine, b0 << recombine, long_blocks);
    }
    let mut cm = quant_partition(ctx, x, n, b, bb, lowband, lm, gain, fill);
    if b0 > 1 {
        interleave_hadamard(x, n_b >> recombine, b0 << recombine, long_blocks);
    }
    n_b = n_b0;
    bb = b0;
    for _ in 0..time_divide {
        bb >>= 1;
        n_b <<= 1;
        cm |= cm >> bb;
        haar1(x, n_b, bb);
    }
    for k in 0..recombine {
        cm = u32::from(t::BIT_DEINTERLEAVE_TABLE[cm as usize]);
        haar1(x, n0 >> k, 1 << k);
    }
    bb <<= recombine;
    if let Some(lo) = lowband_out {
        let nn = ((n0 as f32) * 4_194_304.0).sqrt();
        for j in 0..n0 {
            lo[j] = nn * x[j] * (1.0 / 32768.0);
        }
        // libopus float: MULT16_16_Q15(n, X) == n * X with n = sqrt(N0 << 22) / 32768.
    }
    cm & ((1u32 << bb) - 1)
}

#[allow(clippy::too_many_arguments)]
fn quant_band_stereo(
    ctx: &mut BandCtx<'_, '_>,
    x: &mut [f32],
    y: &mut [f32],
    n: usize,
    mut b: i32,
    bb: usize,
    lowband: Option<&mut [f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    scratch: bool,
    mut fill: u32,
) -> u32 {
    if n == 1 {
        return quant_band_n1(ctx, x, Some(y), lowband_out);
    }
    let orig_fill = fill;
    let sp = compute_theta(ctx, n, &mut b, bb, bb, lm, true, &mut fill);
    let mid = (1.0 / 32768.0) * sp.imid as f32;
    let side = (1.0 / 32768.0) * sp.iside as f32;
    let itheta = sp.itheta;
    let mut cm;
    if n == 2 {
        let mut mbits = b;
        let mut sbits = 0;
        if itheta != 0 && itheta != 16384 {
            sbits = 1 << BITRES;
        }
        mbits -= sbits;
        let c = itheta > 8192;
        ctx.remaining_bits -= sp.qalloc + sbits;
        let mut sign = 0i32;
        if sbits != 0 {
            sign = ctx.dec.bits(1) as i32;
        }
        let sign = (1 - 2 * sign) as f32;
        let (x2, y2): (&mut [f32], &mut [f32]) = if c { (&mut *y, &mut *x) } else { (&mut *x, &mut *y) };
        cm = quant_band(ctx, x2, n, mbits, bb, lowband, lm, lowband_out, 1.0, scratch, orig_fill);
        y2[0] = -sign * x2[1];
        y2[1] = sign * x2[0];
        x[0] *= mid;
        x[1] *= mid;
        y[0] *= side;
        y[1] *= side;
        let tmp = x[0];
        x[0] = tmp - y[0];
        y[0] = tmp + y[0];
        let tmp = x[1];
        x[1] = tmp - y[1];
        y[1] = tmp + y[1];
    } else {
        let mut mbits = 0.max(b.min((b - sp.delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= sp.qalloc;
        let mut rebalance = ctx.remaining_bits;
        if mbits >= sbits {
            cm = quant_band(ctx, x, n, mbits, bb, lowband, lm, lowband_out, 1.0, scratch, fill);
            rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            cm |= quant_band(ctx, y, n, sbits, bb, None, lm, None, side, false, fill >> bb);
        } else {
            cm = quant_band(ctx, y, n, sbits, bb, None, lm, None, side, false, fill >> bb);
            rebalance = sbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && itheta != 16384 {
                mbits += rebalance - (3 << BITRES);
            }
            cm |= quant_band(ctx, x, n, mbits, bb, lowband, lm, lowband_out, 1.0, scratch, fill);
        }
    }
    if n != 2 {
        stereo_merge(x, y, mid, n);
    }
    if sp.inv {
        for v in &mut y[..n] {
            *v = -*v;
        }
    }
    cm
}

/// What [`quant_all_bands`] needs about the frame.
pub struct AllBands<'r> {
    pub start: usize,
    pub end: usize,
    pub pulses: &'r [i32; NB_EBANDS],
    pub short_blocks: bool,
    pub spread: i32,
    pub dual_stereo: bool,
    pub intensity: usize,
    pub tf_res: &'r [i32; NB_EBANDS],
    pub total_bits: i32,
    pub balance: i32,
    pub lm: i32,
    pub coded_bands: usize,
    pub disable_inv: bool,
}

/// `quant_all_bands` (decode): the normalised spectrum of every band into
/// `x` (and `y` for stereo), `collapse_masks[band * C + c]`.
pub fn quant_all_bands(
    p: &AllBands<'_>,
    x_: &mut [f32],
    mut y_: Option<&mut [f32]>,
    collapse_masks: &mut [u8],
    dec: &mut RangeDecoder<'_>,
    seed: &mut u32,
) {
    let c = if y_.is_some() { 2 } else { 1 };
    let m = 1usize << p.lm;
    let bb = if p.short_blocks { m } else { 1 };
    let norm_offset = m * eb(p.start) as usize;
    let norm_len = m * eb(NB_EBANDS - 1) as usize - norm_offset;
    let mut norm_buf = vec![0f32; c * norm_len];
    let (norm, norm2) = norm_buf.split_at_mut(norm_len);
    let mut lowband_offset = 0usize;
    let mut update_lowband = true;
    let mut dual_stereo = p.dual_stereo;
    let mut balance = p.balance;
    let mut ctx = BandCtx {
        dec,
        i: 0,
        intensity: p.intensity,
        spread: p.spread,
        tf_change: 0,
        remaining_bits: 0,
        seed: *seed,
        disable_inv: p.disable_inv,
        avoid_split_noise: bb > 1,
    };
    for i in p.start..p.end {
        ctx.i = i;
        let last = i == p.end - 1;
        let band_lo = m * eb(i) as usize;
        let n = m * eb(i + 1) as usize - band_lo;
        let tell = ctx.dec.tell_frac() as i32;
        if i != p.start {
            balance -= tell;
        }
        let remaining_bits = p.total_bits - tell - 1;
        ctx.remaining_bits = remaining_bits;
        let b = if i < p.coded_bands {
            let curr_balance = balance / 3.min(p.coded_bands as i32 - i as i32);
            0.max(16383.min((remaining_bits + 1).min(p.pulses[i] + curr_balance)))
        } else {
            0
        };
        if (band_lo as i32 - n as i32 >= m as i32 * eb(p.start) || i == p.start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }
        if i == p.start + 1 {
            // special_hybrid_folding (copies nothing for CELT-only frames)
            let n1 = m * (eb(p.start + 1) - eb(p.start)) as usize;
            let n2 = m * (eb(p.start + 2) - eb(p.start + 1)) as usize;
            if n2 > n1 {
                norm.copy_within(2 * n1 - n2..n1, n1);
                if dual_stereo {
                    norm2.copy_within(2 * n1 - n2..n1, n1);
                }
            }
        }
        ctx.tf_change = p.tf_res[i];
        // libopus' lowband_scratch lives in the last band's region of X.
        let scratch = !last;
        let mut effective_lowband: Option<usize> = None;
        let (x_cm0, y_cm0);
        if lowband_offset != 0 && (p.spread != SPREAD_AGGRESSIVE || bb > 1 || ctx.tf_change < 0) {
            let el = 0.max(m as i32 * eb(lowband_offset) - norm_offset as i32 - n as i32) as usize;
            effective_lowband = Some(el);
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if (m as i32 * eb(fold_start)) <= (el + norm_offset) as i32 {
                    break;
                }
            }
            let mut fold_end = lowband_offset - 1;
            loop {
                fold_end += 1;
                if !(fold_end < i && (m as i32 * eb(fold_end)) < (el + norm_offset + n) as i32) {
                    break;
                }
            }
            let (mut xc, mut yc) = (0u32, 0u32);
            let mut fi = fold_start;
            loop {
                xc |= u32::from(collapse_masks[fi * c]);
                yc |= u32::from(collapse_masks[fi * c + c - 1]);
                fi += 1;
                if fi >= fold_end {
                    break;
                }
            }
            x_cm0 = xc;
            y_cm0 = yc;
        } else {
            x_cm0 = (1u32 << bb) - 1;
            y_cm0 = x_cm0;
        }
        if dual_stereo && i == p.intensity {
            dual_stereo = false;
            let upto = band_lo - norm_offset;
            for j in 0..upto {
                norm[j] = 0.5 * (norm[j] + norm2[j]);
            }
        }
        let out_at = band_lo - norm_offset;
        let x = &mut x_[band_lo..band_lo + n];
        let (x_cm, y_cm);
        if dual_stereo {
            let y = &mut y_.as_deref_mut().expect("stereo")[band_lo..band_lo + n];
            let (lo_n, out_n) = norm.split_at_mut(out_at);
            let lb = effective_lowband.map(|e| &mut lo_n[e..e + n]);
            let lo_out = (!last).then(|| &mut out_n[..n]);
            x_cm = quant_band(&mut ctx, x, n, b / 2, bb, lb, p.lm, lo_out, 1.0, scratch, x_cm0);
            let (lo_n2, out_n2) = norm2.split_at_mut(out_at);
            let lb2 = effective_lowband.map(|e| &mut lo_n2[e..e + n]);
            let lo_out2 = (!last).then(|| &mut out_n2[..n]);
            y_cm = quant_band(&mut ctx, y, n, b / 2, bb, lb2, p.lm, lo_out2, 1.0, scratch, y_cm0);
        } else {
            let (lo_n, out_n) = norm.split_at_mut(out_at);
            let lb = effective_lowband.map(|e| &mut lo_n[e..e + n]);
            let lo_out = (!last).then(|| &mut out_n[..n]);
            if let Some(yb) = y_.as_deref_mut() {
                let y = &mut yb[band_lo..band_lo + n];
                x_cm = quant_band_stereo(
                    &mut ctx,
                    x,
                    y,
                    n,
                    b,
                    bb,
                    lb,
                    p.lm,
                    lo_out,
                    scratch,
                    x_cm0 | y_cm0,
                );
            } else {
                x_cm = quant_band(&mut ctx, x, n, b, bb, lb, p.lm, lo_out, 1.0, scratch, x_cm0 | y_cm0);
            }
            y_cm = x_cm;
        }
        collapse_masks[i * c] = x_cm as u8;
        collapse_masks[i * c + c - 1] = y_cm as u8;
        balance += p.pulses[i] + tell;
        update_lowband = b > ((n as i32) << BITRES);
        ctx.avoid_split_noise = false;
    }
    *seed = ctx.seed;
}

/// `anti_collapse`: fills collapsed short blocks with noise.
#[allow(clippy::too_many_arguments)]
pub fn anti_collapse(
    x_: &mut [f32],
    collapse_masks: &[u8],
    lm: i32,
    c: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[f32],
    prev1: &[f32],
    prev2: &[f32],
    pulses: &[i32; NB_EBANDS],
    mut seed: u32,
) {
    for i in start..end {
        let n0 = (eb(i + 1) - eb(i)) as usize;
        let depth = ((1 + pulses[i]) / (eb(i + 1) - eb(i))) >> lm;
        let thresh = 0.5 * (-0.125 * depth as f32).exp2();
        let sqrt_1 = 1.0 / ((n0 << lm) as f32).sqrt();
        for ch in 0..c {
            let mut p1 = prev1[ch * NB_EBANDS + i];
            let mut p2 = prev2[ch * NB_EBANDS + i];
            if c == 1 {
                p1 = p1.max(prev1[NB_EBANDS + i]);
                p2 = p2.max(prev2[NB_EBANDS + i]);
            }
            let ediff = (log_e[ch * NB_EBANDS + i] - p1.min(p2)).max(0.0);
            let mut r = 2.0 * (-ediff).exp2();
            if lm == 3 {
                r *= std::f32::consts::SQRT_2;
            }
            let r = r.min(thresh) * sqrt_1;
            let base = ch * size + ((eb(i) as usize) << lm);
            let xs = &mut x_[base..base + (n0 << lm)];
            let mut renorm = false;
            for k in 0..1usize << lm {
                if collapse_masks[i * c + ch] & (1 << k) == 0 {
                    for j in 0..n0 {
                        seed = lcg_rand(seed);
                        xs[(j << lm) + k] = if seed & 0x8000 != 0 { r } else { -r };
                    }
                    renorm = true;
                }
            }
            if renorm {
                renormalise_vector(xs, n0 << lm, 1.0);
            }
        }
    }
}

/// `denormalise_bands` (float): band shapes times their energies.
pub fn denormalise_bands(
    x: &[f32],
    freq: &mut [f32],
    band_log_e: &[f32],
    start: usize,
    end: usize,
    m: usize,
    silence: bool,
) {
    let n = m * 120;
    let (start, end, bound) = if silence {
        (0, 0, 0)
    } else {
        (start, end, m * eb(end) as usize)
    };
    freq[..m * eb(start) as usize].fill(0.0);
    for i in start..end {
        let lo = m * eb(i) as usize;
        let hi = m * eb(i + 1) as usize;
        let lg = band_log_e[i] + t::E_MEANS[i];
        let g = lg.min(32.0).exp2();
        for j in lo..hi {
            freq[j] = x[j] * g;
        }
    }
    freq[bound..n].fill(0.0);
}
