//! CELT pulse-vector (PVQ) codeword decoding and the Laplace energy decoder,
//! ported from libopus `celt/cwrs.c` and `celt/laplace.c` (BSD-3-Clause,
//! Copyright 2007-2009 Timothy B. Terriberry, 2007-2008 CSIRO, Xiph.Org).
//!
//! Instead of libopus' 1272-entry `CELT_PVQ_U_DATA` table, `U(n, k)` comes
//! from its recurrence `U(n,k) = U(n-1,k) + U(n,k-1) + U(n-1,k-1)`
//! (`U(0,0) = 1`); the values match the table everywhere libopus stores one.

use super::entdec::RangeDecoder;
use std::sync::OnceLock;

const ROWS: usize = 16;
const COLS: usize = 260;

fn u_table() -> &'static [[u32; COLS]; ROWS] {
    static T: OnceLock<Box<[[u32; COLS]; ROWS]>> = OnceLock::new();
    T.get_or_init(|| {
        let mut u = vec![[0u64; COLS]; ROWS];
        u[0][0] = 1;
        for a in 1..ROWS {
            for b in 1..COLS {
                u[a][b] = u[a - 1][b]
                    .wrapping_add(u[a][b - 1])
                    .wrapping_add(u[a - 1][b - 1]);
            }
        }
        let mut t = Box::new([[0u32; COLS]; ROWS]);
        for (dst, src) in t.iter_mut().zip(&u) {
            for (d, s) in dst.iter_mut().zip(src) {
                *d = *s as u32;
            }
        }
        t
    })
}

/// `CELT_PVQ_U(n, k)`.
pub fn pvq_u(n: usize, k: usize) -> u32 {
    u_table()[n.min(k)][n.max(k)]
}

/// `CELT_PVQ_V(n, k)`: the number of PVQ codewords of `n` dimensions and
/// `k` pulses.
pub fn pvq_v(n: usize, k: usize) -> u32 {
    pvq_u(n, k).wrapping_add(pvq_u(n, k + 1))
}

/// `cwrsi`: codeword index `i` to the pulse vector `y` (`n > 1`, `k > 0`);
/// returns the squared norm.
fn cwrsi(mut n: usize, mut k: usize, mut i: u32, y: &mut [i32]) -> f32 {
    let row = |r: usize| &u_table()[r];
    let mut yy = 0.0f32;
    let mut at = 0usize;
    while n > 2 {
        if k >= n {
            let p0 = row(n)[k + 1];
            let s: i32 = -i32::from(i >= p0);
            i -= p0 & s as u32;
            let k0 = k;
            let q = row(n)[n];
            let mut p;
            if q > i {
                k = n;
                loop {
                    k -= 1;
                    p = row(k)[n];
                    if p <= i {
                        break;
                    }
                }
            } else {
                p = row(n)[k];
                while p > i {
                    k -= 1;
                    p = row(n)[k];
                }
            }
            i -= p;
            let val = ((k0 as i32 - k as i32 + s) ^ s) as i16;
            y[at] = i32::from(val);
            at += 1;
            yy += f32::from(val) * f32::from(val);
        } else {
            let p = row(k)[n];
            let q = row(k + 1)[n];
            if p <= i && i < q {
                i -= p;
                y[at] = 0;
                at += 1;
            } else {
                let s: i32 = -i32::from(i >= q);
                i -= q & s as u32;
                let k0 = k;
                let mut p;
                loop {
                    k -= 1;
                    p = row(k)[n];
                    if p <= i {
                        break;
                    }
                }
                i -= p;
                let val = ((k0 as i32 - k as i32 + s) ^ s) as i16;
                y[at] = i32::from(val);
                at += 1;
                yy += f32::from(val) * f32::from(val);
            }
        }
        n -= 1;
    }
    // n == 2
    let p = (2 * k + 1) as u32;
    let s: i32 = -i32::from(i >= p);
    i -= p & s as u32;
    let k0 = k;
    k = ((i + 1) >> 1) as usize;
    if k > 0 {
        i -= (2 * k - 1) as u32;
    }
    let val = ((k0 as i32 - k as i32 + s) ^ s) as i16;
    y[at] = i32::from(val);
    at += 1;
    yy += f32::from(val) * f32::from(val);
    // n == 1
    let s: i32 = -(i as i32);
    let val = ((k as i32 + s) ^ s) as i16;
    y[at] = i32::from(val);
    yy += f32::from(val) * f32::from(val);
    yy
}

/// `decode_pulses`: reads a PVQ codeword for `n` dimensions and `k` pulses
/// into `y`; returns its squared norm.
pub fn decode_pulses(y: &mut [i32], n: usize, k: usize, dec: &mut RangeDecoder<'_>) -> f32 {
    let v = pvq_v(n, k);
    let i = dec.uint(v);
    cwrsi(n, k, i, y)
}

const LAPLACE_LOG_MINP: u32 = 0;
const LAPLACE_MINP: u32 = 1 << LAPLACE_LOG_MINP;
const LAPLACE_NMIN: u32 = 16;

fn laplace_get_freq1(fs0: u32, decay: i32) -> u32 {
    let ft = 32768 - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    ((ft as i32 * (16384 - decay)) >> 15) as u32
}

/// `ec_laplace_decode`: a Laplace-distributed integer (coarse energy).
pub fn laplace_decode(dec: &mut RangeDecoder<'_>, mut fs: u32, decay: i32) -> i32 {
    let mut val = 0i32;
    let fm = dec.decode_bin(15);
    let mut fl = 0u32;
    if fm >= fs {
        val += 1;
        fl = fs;
        fs = laplace_get_freq1(fs, decay) + LAPLACE_MINP;
        while fs > LAPLACE_MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            fs = (((fs - 2 * LAPLACE_MINP) as i32 * decay) >> 15) as u32;
            fs += LAPLACE_MINP;
            val += 1;
        }
        if fs <= LAPLACE_MINP {
            let di = (fm - fl) >> (LAPLACE_LOG_MINP + 1);
            val += di as i32;
            fl += 2 * di * LAPLACE_MINP;
        }
        if fm < fl + fs {
            val = -val;
        } else {
            fl += fs;
        }
    }
    dec.update(fl, (fl + fs).min(32768), 32768);
    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pvq_counts_match_libopus_table_rows() {
        // Spot values of libopus' CELT_PVQ_U_DATA.
        assert_eq!(pvq_u(1, 5), 1);
        assert_eq!(pvq_u(2, 4), 7);
        assert_eq!(pvq_u(3, 3), 13);
        assert_eq!(pvq_u(3, 4), 25);
        // V(2, 1) = 4 codewords: (+1,0) (-1,0) (0,+1) (0,-1).
        assert_eq!(pvq_v(2, 1), 4);
        assert_eq!(pvq_v(3, 1), 6);
    }

    #[test]
    fn every_codeword_decodes_to_a_distinct_vector_with_k_pulses() {
        for (n, k) in [(2usize, 1usize), (2, 3), (3, 2), (4, 3), (5, 2)] {
            let v = pvq_v(n, k);
            let mut seen = std::collections::HashSet::new();
            for i in 0..v {
                let mut y = vec![0i32; n];
                let yy = cwrsi(n, k, i, &mut y);
                assert_eq!(
                    y.iter().map(|x| x.unsigned_abs()).sum::<u32>(),
                    k as u32,
                    "{n},{k},{i}: {y:?}"
                );
                assert_eq!(yy, y.iter().map(|&x| (x * x) as f32).sum::<f32>());
                assert!(seen.insert(y), "{n},{k}: duplicate");
            }
        }
    }
}
