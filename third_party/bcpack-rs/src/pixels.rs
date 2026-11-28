// bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com)
// Source: https://github.com/Ahmet-Dedeler/bcpack-rs
// SPDX-License-Identifier: MIT OR Apache-2.0
// If you copy, vendor or port this file, keep this header and credit bcpack-rs in your README.
//! BCn block -> RGBA8 pixel decoding, pure Rust with no extra dependencies.
//!
//! * BC1: 565 endpoints, 4-colour mode, or 3-colour + transparent black when
//!   `c0 <= c1` (punch-through).
//! * BC3: a BC4 alpha half-block followed by a BC1 colour half-block that
//!   always uses the 4-colour mode.
//! * BC4 / BC5: one or two BC4 half-blocks (8-value palette when `e0 > e1`,
//!   otherwise four interpolants and two fixed limits alongside the endpoints).
//!   SNORM variants interpret -128 as -127 and map [-1, 1] to [0, 255].
//! * BC7: full decode - partition tables, per-mode endpoint precision with
//!   p-bits, bit-replicated endpoint expansion, the 2/3/4-bit index weight
//!   tables, rotation, and the dual index sets of modes 4 and 5.
//!
//! Interpolation uses the spec's integer formulas (`(a*(64-w) + b*w + 32) >> 6`
//! for BC7, `((8-i)*e0 + (i-1)*e1) / 7` and `((6-i)*e0 + (i-1)*e1) / 5` for
//! BC4 UNORM/BC3 alpha, the 8-bit thirds/halves for BC1), which is what the common
//! reference decoders' default paths compute.

use crate::bc7;
use crate::format::Mode;
use std::io::{self, ErrorKind};

fn c565(c: u16) -> [u8; 3] {
    let (r, g, b) = ((c >> 11) & 31, (c >> 5) & 63, c & 31);
    [
        ((r << 3) | (r >> 2)) as u8,
        ((g << 2) | (g >> 4)) as u8,
        ((b << 3) | (b >> 2)) as u8,
    ]
}

/// BC1 colour palette. `opaque` forces the 4-colour mode (BC3 colour blocks).
fn palette(c0: u16, c1: u16, opaque: bool) -> [[u8; 4]; 4] {
    let (a, b) = (c565(c0), c565(c1));
    let mut p = [
        [a[0], a[1], a[2], 255],
        [b[0], b[1], b[2], 255],
        [0, 0, 0, 255],
        [0, 0, 0, 0],
    ];
    let four = c0 > c1 || opaque;
    for k in 0..3 {
        let (x, y) = (a[k] as u32, b[k] as u32);
        if four {
            p[2][k] = ((2 * x + y + 1) / 3) as u8;
            p[3][k] = ((x + 2 * y + 1) / 3) as u8;
        } else {
            p[2][k] = ((x + y) / 2) as u8;
        }
    }
    if four {
        p[3][3] = 255;
    }
    p
}

/// Standard BC1 palette (RGBA): 4-colour when c0 > c1, else 3-colour + transparent black.
pub fn bc1_palette(c0: u16, c1: u16) -> [[u8; 4]; 4] {
    palette(c0, c1, false)
}

/// BC3 colour-block palette: always the 4-colour mode, never punch-through.
pub fn bc1_palette_opaque(c0: u16, c1: u16) -> [[u8; 4]; 4] {
    palette(c0, c1, true)
}

/// Run `block_px` on every block and scatter the 16 pixels into a w x h RGBA image.
fn expand(
    blocks: &[u8],
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
    bb: usize,
    mut block_px: impl FnMut(&[u8], &mut [[u8; 4]; 16]),
) -> io::Result<Vec<u8>> {
    if w.div_ceil(4) > bw || h.div_ceil(4) > bh {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "image exceeds block grid",
        ));
    }
    let block_len = bw.checked_mul(bh).and_then(|n| n.checked_mul(bb));
    if !block_len.is_some_and(|n| n <= blocks.len()) {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "block grid exceeds input length",
        ));
    }
    let len = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "RGBA image size overflow"))?;
    let mut img = Vec::new();
    img.try_reserve_exact(len).map_err(io::Error::other)?;
    img.resize(len, 0);
    if len == 0 {
        return Ok(img);
    }
    for by in 0..bh {
        for bx in 0..bw {
            let o = (by * bw + bx) * bb;
            let mut px = [[0u8; 4]; 16];
            block_px(&blocks[o..o + bb], &mut px);
            for (i, p) in px.iter().enumerate() {
                let (gx, gy) = (bx * 4 + i % 4, by * 4 + i / 4);
                if gx < w && gy < h {
                    img[(gy * w + gx) * 4..(gy * w + gx) * 4 + 4].copy_from_slice(p);
                }
            }
        }
    }
    Ok(img)
}

/// Row-major BC1 blocks (bw x bh) -> RGBA pixels cropped to w x h.
///
/// Panics on invalid input. Use [`blocks_to_rgba`] for fallible decoding.
pub fn bc1_to_rgba(blocks: &[u8], bw: usize, bh: usize, w: usize, h: usize) -> Vec<u8> {
    blocks_to_rgba(Mode::Bc1, false, blocks, bw, bh, w, h).expect("invalid BC1 image")
}

fn bc1_block_rgba(b: &[u8], px: &mut [[u8; 4]; 16]) {
    let c0 = u16::from_le_bytes([b[0], b[1]]);
    let c1 = u16::from_le_bytes([b[2], b[3]]);
    let idx = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let pal = bc1_palette(c0, c1);
    for (p, pixel) in px.iter_mut().enumerate() {
        *pixel = pal[((idx >> (2 * p)) & 3) as usize];
    }
}

/// The 8-entry palette of one BC4 half-block. `signed` selects SNORM semantics.
fn alpha_palette(e0: i32, e1: i32, signed: bool) -> [u8; 8] {
    let (e0, e1) = (e0.max(-127), e1.max(-127));
    let quantize = |num: i32, denom: i32| {
        if signed {
            // Bias the signed interpolant, scale to 255 and round once.
            let bias = 127 * denom;
            (((num + bias) * 255 + bias) / (2 * bias)) as u8
        } else {
            (num / denom) as u8
        }
    };
    let mut p = [0u8; 8];
    p[0] = quantize(e0, 1);
    p[1] = quantize(e1, 1);
    if e0 > e1 {
        for i in 2..8i32 {
            p[i as usize] = quantize((8 - i) * e0 + (i - 1) * e1, 7);
        }
    } else {
        for i in 2..6i32 {
            p[i as usize] = quantize((6 - i) * e0 + (i - 1) * e1, 5);
        }
        p[7] = 255;
    }
    p
}

/// One BC4 half-block -> 16 channel values.
fn bc4_channel(block: &[u8], signed: bool) -> [u8; 16] {
    let (e0, e1) = if signed {
        (block[0] as i8 as i32, block[1] as i8 as i32)
    } else {
        (block[0] as i32, block[1] as i32)
    };
    let pal = alpha_palette(e0, e1, signed);
    let mut v = u64::from_le_bytes(block[..8].try_into().unwrap()) >> 16;
    let mut out = [0u8; 16];
    for p in out.iter_mut() {
        *p = pal[(v & 7) as usize];
        v >>= 3;
    }
    out
}

/// Row-major BC4 blocks -> RGBA pixels (grey replicated to RGB, alpha 255).
pub fn bc4_to_rgba(
    blocks: &[u8],
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
    signed: bool,
) -> io::Result<Vec<u8>> {
    expand(blocks, bw, bh, w, h, 8, |b, px| {
        let c = bc4_channel(b, signed);
        for p in 0..16 {
            px[p] = [c[p], c[p], c[p], 255];
        }
    })
}

/// Row-major BC5 blocks -> RGBA pixels (R, G, B = 0, A = 255).
pub fn bc5_to_rgba(
    blocks: &[u8],
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
    signed: bool,
) -> io::Result<Vec<u8>> {
    expand(blocks, bw, bh, w, h, 16, |b, px| {
        let r = bc4_channel(&b[..8], signed);
        let g = bc4_channel(&b[8..], signed);
        for p in 0..16 {
            px[p] = [r[p], g[p], 0, 255];
        }
    })
}

/// Row-major BC3 blocks -> RGBA pixels (BC4 alpha + always-opaque BC1 colour).
pub fn bc3_to_rgba(blocks: &[u8], bw: usize, bh: usize, w: usize, h: usize) -> io::Result<Vec<u8>> {
    expand(blocks, bw, bh, w, h, 16, |b, px| {
        let a = bc4_channel(&b[..8], false);
        let c0 = u16::from_le_bytes([b[8], b[9]]);
        let c1 = u16::from_le_bytes([b[10], b[11]]);
        let idx = u32::from_le_bytes([b[12], b[13], b[14], b[15]]);
        let pal = bc1_palette_opaque(c0, c1);
        for p in 0..16 {
            let c = pal[((idx >> (2 * p)) & 3) as usize];
            px[p] = [c[0], c[1], c[2], a[p]];
        }
    })
}

/// BC7 index weight tables for 2, 3 and 4 index bits.
pub const WEIGHTS_2: [u32; 4] = [0, 21, 43, 64];
pub const WEIGHTS_3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
pub const WEIGHTS_4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

fn weights(b: u32) -> &'static [u32] {
    match b {
        2 => &WEIGHTS_2,
        3 => &WEIGHTS_3,
        4 => &WEIGHTS_4,
        _ => &[],
    }
}

/// Subset id per pixel (raster order) for each of the 64 two-subset BC7
/// partitions (the anchor pixels are the ones `bc7::Control::anchors` marks;
/// their stored index simply has the top bit clear).
pub const PARTITION_2: [[u8; 16]; 64] = [
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
];

/// Subset id per pixel (raster order) for each of the 64 three-subset BC7
/// partitions.
pub const PARTITION_3: [[u8; 16]; 64] = [
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 1, 2, 2, 2, 2],
    [0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 2, 0, 0, 1, 2, 2, 1, 1, 2, 2, 1, 1],
    [0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2],
    [0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2],
    [0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2],
    [0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2],
    [0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2],
    [0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0, 2, 2, 2, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2],
    [0, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0],
    [0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
    [0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2],
    [0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2],
    [0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 1, 0, 2, 2, 1, 0],
    [0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 2, 0, 0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2],
    [0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1],
    [0, 0, 2, 2, 1, 1, 0, 2, 1, 1, 0, 2, 0, 0, 2, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 2, 2, 2, 2],
    [0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1],
    [0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 2, 2, 2],
    [0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 2],
    [0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
    [0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0],
    [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2],
    [0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1],
    [0, 2, 2, 0, 1, 2, 2, 1, 0, 2, 2, 0, 1, 2, 2, 1],
    [0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1],
    [0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2],
    [0, 2, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1],
    [0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 1, 1, 1, 2],
    [0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
    [0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2],
    [0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 0, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2],
    [0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2],
    [0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1],
    [0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2],
    [0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0],
];

/// Decode one 16-byte BC7 block to 16 RGBA pixels in raster order. A reserved
/// block (no mode bit set) decodes to transparent black.
pub fn bc7_block_rgba(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut out = [[0u8; 4]; 16];
    let Some((c, st)) = bc7::unpack(block) else {
        return out;
    };
    let subsets = c.subsets();
    let bits = c.index_bits();
    // Unquantised 8-bit endpoints e[subset][channel][endpoint]. In the p-bit
    // modes the p-bit extends every component of its endpoint, not just red;
    // `bc7::unpack` stores it in red's lowest bit, so rebuild the other
    // channels' values from it. A channel with zero base precision is alpha in
    // the modes without it and reads as 255.
    let mut e = [[[0u8; 2]; 4]; 3];
    for (row, erow) in e.iter_mut().enumerate().take(subsets) {
        for (ch, ech) in erow.iter_mut().enumerate() {
            for (k, ek) in ech.iter_mut().enumerate() {
                *ek = bc7_ep8(&c, &st, row, ch, k);
            }
        }
    }
    let part = (c.partition as usize) & 63;
    // Modes 4 and 5 carry two index sets. The bitstream stores set 1 first in
    // mode 4 and set 0 first in mode 5; the index-selection bit (mode 4 only)
    // picks which set interpolates the colour channels, the other set feeds
    // the scalar channel named by the rotation (0 = A, 1 = R, 2 = G, 3 = B).
    let (color_set, alpha_set) = if c.mode == 4 {
        if c.index_selection == 1 {
            (0usize, 1usize)
        } else {
            (1, 0)
        }
    } else {
        (0, 1)
    };
    let scalar = [3usize, 0, 1, 2][c.rotation as usize];
    for p in 0..16 {
        let subset = match subsets {
            3 => PARTITION_3[part][p] as usize,
            2 => PARTITION_2[part][p] as usize,
            _ => 0,
        };
        for ch in 0..4 {
            let (set, b) = if bits[1] == 0 {
                (0usize, bits[0] as u32)
            } else if ch == scalar {
                (alpha_set, bits[alpha_set] as u32)
            } else {
                (color_set, bits[color_set] as u32)
            };
            let wtab = weights(b);
            let ix = (st.ix[set][p] >> (4 - b)) as usize;
            let w = wtab[ix & (wtab.len() - 1)];
            let (e0, e1) = (e[subset][ch][0] as u32, e[subset][ch][1] as u32);
            out[p][ch] = ((e0 * (64 - w) + e1 * w + 32) >> 6) as u8;
        }
    }
    out
}

/// The 8-bit value of endpoint `k` (0/1) of `row`, channel `ch`, from the
/// unpacked state. Bit-replicated expansion of the p-bit-precision value.
fn bc7_ep8(c: &bc7::Control, st: &bc7::State, row: usize, ch: usize, k: usize) -> u8 {
    let rep = |top: u8, p: u32| if p >= 8 { top } else { top | (top >> p) };
    let raw = st.ep[row][2 * ch + k];
    if !c.has_pbits() {
        // No p-bits (modes 2, 4, 5): precision is rotation-aware already.
        let p = c.precision()[ch] as u32;
        return if p == 0 { 255 } else { rep(raw, p) };
    }
    // P-bit modes (never rotated): the p-bit sits in red's lowest bit in the
    // state and extends every component of its endpoint.
    if ch == 0 {
        return rep(raw, c.precision()[0] as u32);
    }
    let base = c.precision()[ch] as u32;
    if base == 0 {
        return 255;
    }
    let pbit = (st.ep[row][k] >> (8 - c.precision()[0])) & 1;
    let v = (raw >> (8 - base)) as u32;
    let full = ((v << 1) | pbit as u32) << (8 - base - 1);
    rep(full as u8, base + 1)
}

/// Row-major BC7 blocks -> RGBA pixels cropped to w x h.
pub fn bc7_to_rgba(blocks: &[u8], bw: usize, bh: usize, w: usize, h: usize) -> io::Result<Vec<u8>> {
    expand(blocks, bw, bh, w, h, 16, |b, px| {
        *px = bc7_block_rgba(b.try_into().unwrap())
    })
}

/// Row-major blocks of `mode` (bw x bh blocks, cropped to w x h pixels) -> RGBA8.
/// `signed` selects the SNORM interpretation of BC4/BC5. Returns an error for
/// BC6H, invalid geometry, short input or allocation failure.
pub fn blocks_to_rgba(
    mode: Mode,
    signed: bool,
    blocks: &[u8],
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
) -> io::Result<Vec<u8>> {
    match mode {
        Mode::Bc1 => expand(blocks, bw, bh, w, h, 8, bc1_block_rgba),
        Mode::Bc3 => bc3_to_rgba(blocks, bw, bh, w, h),
        Mode::Bc4 => bc4_to_rgba(blocks, bw, bh, w, h, signed),
        Mode::Bc5 => bc5_to_rgba(blocks, bw, bh, w, h, signed),
        Mode::Bc7 => bc7_to_rgba(blocks, bw, bh, w, h),
        Mode::Bc6h => Err(io::Error::new(
            ErrorKind::InvalidInput,
            "BC6H pixel decoding is unsupported",
        )),
    }
}

/// A DXGI format number -> (block mode, signed). The sRGB variants decode like
/// UNORM during block decode. Their RGB samples remain sRGB-encoded.
pub fn mode_from_dxgi(dxgi: u32) -> Option<(Mode, bool)> {
    Some(match dxgi {
        70..=72 => (Mode::Bc1, false),
        76..=78 => (Mode::Bc3, false),
        79..=80 => (Mode::Bc4, false),
        81 => (Mode::Bc4, true),
        82..=83 => (Mode::Bc5, false),
        84 => (Mode::Bc5, true),
        97..=99 => (Mode::Bc7, false),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_tables_are_well_formed() {
        for p in 0..64 {
            let s2: std::collections::BTreeSet<u8> = PARTITION_2[p].iter().copied().collect();
            let s3: std::collections::BTreeSet<u8> = PARTITION_3[p].iter().copied().collect();
            assert_eq!(s2, [0, 1].into_iter().collect(), "partition {p} (2-subset)");
            assert_eq!(
                s3,
                [0, 1, 2].into_iter().collect(),
                "partition {p} (3-subset)"
            );
            assert_eq!(PARTITION_2[p][0], 0, "pixel 0 is always subset 0");
            assert_eq!(PARTITION_3[p][0], 0, "pixel 0 is always subset 0");
        }
    }

    #[test]
    fn weight_tables_are_ordered() {
        for w in [&WEIGHTS_2[..], &WEIGHTS_3[..], &WEIGHTS_4[..]] {
            assert_eq!(w[0], 0);
            assert_eq!(*w.last().unwrap(), 64);
            assert!(w.windows(2).all(|p| p[0] < p[1]));
        }
    }

    #[test]
    fn alpha_palettes() {
        assert_eq!(
            alpha_palette(255, 0, false),
            [255, 0, 218, 182, 145, 109, 72, 36]
        );
        assert_eq!(
            alpha_palette(0, 255, false),
            [0, 255, 51, 102, 153, 204, 0, 255]
        );
        assert_eq!(
            alpha_palette(-5, 5, true),
            [122, 133, 124, 126, 129, 131, 0, 255]
        );
    }

    #[test]
    fn dxgi_mapping() {
        assert_eq!(mode_from_dxgi(71), Some((Mode::Bc1, false)));
        assert_eq!(mode_from_dxgi(78), Some((Mode::Bc3, false)));
        assert_eq!(mode_from_dxgi(80), Some((Mode::Bc4, false)));
        assert_eq!(mode_from_dxgi(81), Some((Mode::Bc4, true)));
        assert_eq!(mode_from_dxgi(83), Some((Mode::Bc5, false)));
        assert_eq!(mode_from_dxgi(84), Some((Mode::Bc5, true)));
        assert_eq!(mode_from_dxgi(99), Some((Mode::Bc7, false)));
        assert_eq!(mode_from_dxgi(95), None); // BC6H
        assert_eq!(mode_from_dxgi(28), None); // RGBA8
    }
}
