//! CELT band-energy decoding (float build), ported from libopus
//! `celt/quant_bands.c` (BSD-3-Clause, Copyright 2007-2009 Xiph.Org,
//! CSIRO, Jean-Marc Valin).

use super::celt_tables as t;
use super::cwrs::laplace_decode;
use super::entdec::RangeDecoder;
use super::rate::{MAX_FINE_BITS, NB_EBANDS};

const PRED_COEF: [f32; 4] = [29440.0 / 32768.0, 26112.0 / 32768.0, 21248.0 / 32768.0, 16384.0 / 32768.0];
const BETA_COEF: [f32; 4] = [30147.0 / 32768.0, 22282.0 / 32768.0, 12124.0 / 32768.0, 6554.0 / 32768.0];
const BETA_INTRA: f32 = 4915.0 / 32768.0;

/// `unquant_coarse_energy`: `old` (log2 band energies, `[c * 21 + band]`)
/// predicted and corrected in place.
#[allow(clippy::too_many_arguments)]
pub fn unquant_coarse_energy(
    start: usize,
    end: usize,
    old: &mut [f32],
    intra: bool,
    dec: &mut RangeDecoder<'_>,
    c: usize,
    lm: usize,
) {
    let model = &t::E_PROB_MODEL[(lm * 2 + usize::from(intra)) * 42..];
    let mut prev = [0f32; 2];
    let (coef, beta) = if intra {
        (0.0, BETA_INTRA)
    } else {
        (PRED_COEF[lm], BETA_COEF[lm])
    };
    let budget = dec.storage() as i32 * 8;
    for i in start..end {
        for ch in 0..c {
            let tell = dec.tell();
            let qi = if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                laplace_decode(dec, u32::from(model[pi]) << 7, i32::from(model[pi + 1]) << 6)
            } else if budget - tell >= 2 {
                let qi = dec.icdf(&t::SMALL_ENERGY_ICDF, 2) as i32;
                (qi >> 1) ^ -(qi & 1)
            } else if budget - tell >= 1 {
                -i32::from(dec.bit_logp(1))
            } else {
                -1
            };
            let q = qi as f32;
            let e = &mut old[i + ch * NB_EBANDS];
            *e = e.max(-9.0);
            let tmp = (coef * *e + prev[ch] + q).max(-28.0);
            *e = tmp;
            prev[ch] = prev[ch] + q - beta * q;
        }
    }
}

/// `unquant_fine_energy`.
pub fn unquant_fine_energy(
    start: usize,
    end: usize,
    old: &mut [f32],
    fine_quant: &[i32; NB_EBANDS],
    dec: &mut RangeDecoder<'_>,
    c: usize,
) {
    for i in start..end {
        let fq = fine_quant[i];
        if fq <= 0 {
            continue;
        }
        for ch in 0..c {
            let q2 = dec.bits(fq as u32) as f32;
            let offset = (q2 + 0.5) * (1 << (14 - fq)) as f32 * (1.0 / 16384.0) - 0.5;
            old[i + ch * NB_EBANDS] += offset;
        }
    }
}

/// `unquant_energy_finalise`: spends the leftover bits on extra fine energy.
#[allow(clippy::too_many_arguments)]
pub fn unquant_energy_finalise(
    start: usize,
    end: usize,
    old: &mut [f32],
    fine_quant: &[i32; NB_EBANDS],
    fine_priority: &[i32; NB_EBANDS],
    mut bits_left: i32,
    dec: &mut RangeDecoder<'_>,
    c: usize,
) {
    for prio in 0..2 {
        let mut i = start;
        while i < end && bits_left >= c as i32 {
            if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                i += 1;
                continue;
            }
            for ch in 0..c {
                let q2 = dec.bits(1) as f32;
                let offset = (q2 - 0.5) * (1 << (14 - fine_quant[i] - 1)) as f32 * (1.0 / 16384.0);
                old[i + ch * NB_EBANDS] += offset;
                bits_left -= 1;
            }
            i += 1;
        }
    }
}

/// `eMeans`: the mean log2 energy of each band.
pub fn e_mean(i: usize) -> f32 {
    t::E_MEANS[i]
}
