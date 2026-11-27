//! Clean-room RDNA2 compression metadata (DCC, HTILE, CMASK, FMASK) and
//! Sampler Feedback map handling under Strata's No-API memory model.
//!
//! Because Strata keeps render targets and textures in linear / Morton title
//! memory and executes shaders directly over 64-bit BDA / Vulkan images,
//! titles that write RDNA2 compression metadata (fast-clear codes in DCC /
//! HTILE / CMASK, or MSAA fragment-index pointers in FMASK) need those
//! metadata states resolved into plain texel/depth bytes before sampling or
//! presentation.

/// RDNA2 DCC (Delta Color Compression) key values (GFX10.3 public ISA).
pub const DCC_CLEAR_0000: u8 = 0x00;
pub const DCC_CLEAR_0001: u8 = 0x40;
pub const DCC_CLEAR_1110: u8 = 0x80;
pub const DCC_CLEAR_1111: u8 = 0xC0;
pub const DCC_CLEAR_REG: u8 = 0x20;
pub const DCC_UNCOMPRESSED: u8 = 0xFF;

/// RDNA2 CMASK nibble values (4 bits per 8x8 tile).
pub const CMASK_CLEARED: u8 = 0x0;
pub const CMASK_EXPANDED: u8 = 0xF;

/// Decoded 32-bit RDNA2 HTILE word (one per 8x8 depth/stencil tile).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HtileWord {
    /// 14-bit minimum depth in the 8x8 tile (`[0, 16383]` -> `[0.0, 1.0]`).
    pub min_z: u16,
    /// 14-bit maximum depth in the 8x8 tile (`[0, 16383]` -> `[0.0, 1.0]`).
    pub max_z: u16,
    /// 4-bit Z-mask (`0` = tile is fast-cleared to `min_z`, `0xF` = expanded).
    pub zmask: u8,
}

impl HtileWord {
    /// Decode an RDNA2 32-bit HTILE entry (Z-only layout: `min_z[31:18]`,
    /// `max_z[17:4]`, `zmask[3:0]`).
    pub const fn decode(raw: u32) -> Self {
        Self {
            min_z: ((raw >> 18) & 0x3FFF) as u16,
            max_z: ((raw >> 4) & 0x3FFF) as u16,
            zmask: (raw & 0xF) as u8,
        }
    }

    /// Encode back to a 32-bit HTILE word.
    pub const fn encode(self) -> u32 {
        (((self.min_z as u32) & 0x3FFF) << 18)
            | (((self.max_z as u32) & 0x3FFF) << 4)
            | ((self.zmask as u32) & 0xF)
    }

    /// True if this 8x8 tile is in fast-clear state (`zmask == 0`).
    pub const fn is_fast_cleared(self) -> bool {
        self.zmask == 0
    }

    /// Fast-clear depth value as normalized `f32`.
    pub fn clear_depth_f32(self) -> f32 {
        f32::from(self.min_z) / 16383.0
    }
}

/// Resolve DCC fast-clear blocks into a 32-bit RGBA8 linear surface.
///
/// Each byte in `dcc` covers one 256-byte (8x8 for 4 BPP) colour block.
/// Blocks tagged with `DCC_CLEAR_*` are filled with the corresponding clear
/// colour (or `reg_clear_rgba` for `DCC_CLEAR_REG`), while `DCC_UNCOMPRESSED`
/// blocks are left untouched. Returns the number of blocks resolved.
pub fn resolve_dcc_fast_clears(
    surface: &mut [u8],
    width: u32,
    height: u32,
    row_pitch: usize,
    dcc: &[u8],
    reg_clear_rgba: [u8; 4],
) -> usize {
    let tiles_x = (width as usize).div_ceil(8);
    let tiles_y = (height as usize).div_ceil(8);
    let mut resolved = 0usize;
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let idx = ty * tiles_x + tx;
            let Some(&key) = dcc.get(idx) else {
                break;
            };
            let fill = match key {
                DCC_CLEAR_0000 => Some([0, 0, 0, 0]),
                DCC_CLEAR_0001 => Some([0, 0, 0, 255]),
                DCC_CLEAR_1110 => Some([255, 255, 255, 0]),
                DCC_CLEAR_1111 => Some([255, 255, 255, 255]),
                DCC_CLEAR_REG => Some(reg_clear_rgba),
                _ => None,
            };
            if let Some(rgba) = fill {
                for dy in 0..8 {
                    let y = ty * 8 + dy;
                    if y >= height as usize {
                        break;
                    }
                    for dx in 0..8 {
                        let x = tx * 8 + dx;
                        if x >= width as usize {
                            break;
                        }
                        let off = y * row_pitch + x * 4;
                        if off + 4 <= surface.len() {
                            surface[off..off + 4].copy_from_slice(&rgba);
                        }
                    }
                }
                resolved += 1;
            }
        }
    }
    resolved
}

/// Resolve HTILE fast-clear 8x8 tiles into an `R32_FLOAT` depth surface.
pub fn resolve_htile_fast_clears(
    depth_f32: &mut [f32],
    width: u32,
    height: u32,
    htile: &[u32],
) -> usize {
    let tiles_x = (width as usize).div_ceil(8);
    let tiles_y = (height as usize).div_ceil(8);
    let mut resolved = 0usize;
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let idx = ty * tiles_x + tx;
            let Some(&raw) = htile.get(idx) else {
                break;
            };
            let h = HtileWord::decode(raw);
            if h.is_fast_cleared() {
                let z = h.clear_depth_f32();
                for dy in 0..8 {
                    let y = ty * 8 + dy;
                    if y >= height as usize {
                        break;
                    }
                    for dx in 0..8 {
                        let x = tx * 8 + dx;
                        if x >= width as usize {
                            break;
                        }
                        let off = y * width as usize + x;
                        if off < depth_f32.len() {
                            depth_f32[off] = z;
                        }
                    }
                }
                resolved += 1;
            }
        }
    }
    resolved
}

/// Resolve an RDNA2 FMASK + CMASK compressed MSAA surface into single-sample
/// resolved RGBA8 pixels (box average across `sample_count` samples per pixel).
pub fn resolve_fmask_msaa(
    samples: &[[u8; 4]],
    fmask: &[u32],
    cmask: &[u8],
    width: u32,
    height: u32,
    sample_count: u32,
    clear_rgba: [u8; 4],
) -> Vec<[u8; 4]> {
    let sc = sample_count.clamp(1, 8) as usize;
    let bits_per_sample = match sc {
        1..=2 => 1u32,
        3..=4 => 2,
        _ => 4,
    };
    let mask = (1u32 << bits_per_sample) - 1;
    let tiles_x = (width as usize).div_ceil(8);
    let mut out = vec![[0u8; 4]; (width * height) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let tile_idx = (y / 8) * tiles_x + (x / 8);
            let cmask_nibble = cmask.get(tile_idx / 2).map_or(CMASK_EXPANDED, |&b| {
                if tile_idx % 2 == 0 { b & 0xF } else { b >> 4 }
            });
            let px_idx = y * width as usize + x;
            if cmask_nibble == CMASK_CLEARED {
                out[px_idx] = clear_rgba;
                continue;
            }
            let fm = fmask.get(px_idx).copied().unwrap_or(0);
            let mut acc = [0u32; 4];
            for s in 0..sc {
                let frag = ((fm >> (s as u32 * bits_per_sample)) & mask) as usize;
                let sample_px = samples
                    .get(px_idx * sc + frag.min(sc - 1))
                    .copied()
                    .unwrap_or(clear_rgba);
                for c in 0..4 {
                    acc[c] += u32::from(sample_px[c]);
                }
            }
            out[px_idx] = [
                (acc[0] / sc as u32) as u8,
                (acc[1] / sc as u32) as u8,
                (acc[2] / sc as u32) as u8,
                (acc[3] / sc as u32) as u8,
            ];
        }
    }
    out
}

/// D3D12 Sampler Feedback min-mip map tracker (stores minimum sampled mip per
/// feedback tile region, initialized to `0xFF` = unsampled).
#[derive(Debug, Clone)]
pub struct SamplerFeedbackMap {
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub min_mip: Vec<u8>,
}

impl SamplerFeedbackMap {
    pub fn new(width: u32, height: u32, tile_size: u32) -> Self {
        let ts = tile_size.max(1);
        let tiles_x = width.div_ceil(ts);
        let tiles_y = height.div_ceil(ts);
        Self {
            tiles_x,
            tiles_y,
            min_mip: vec![0xFF; (tiles_x * tiles_y) as usize],
        }
    }

    pub fn record_sample(&mut self, tile_x: u32, tile_y: u32, mip_level: u8) {
        if tile_x < self.tiles_x && tile_y < self.tiles_y {
            let idx = (tile_y * self.tiles_x + tile_x) as usize;
            if mip_level < self.min_mip[idx] {
                self.min_mip[idx] = mip_level;
            }
        }
    }

    pub fn clear(&mut self) {
        self.min_mip.fill(0xFF);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dcc_and_htile_fast_clears_resolve() {
        let mut color = vec![0x11u8; 8 * 8 * 4];
        let n = resolve_dcc_fast_clears(&mut color, 8, 8, 32, &[DCC_CLEAR_0001], [0; 4]);
        assert_eq!(n, 1);
        assert_eq!(&color[0..4], &[0, 0, 0, 255]);

        let mut depth = vec![0.0f32; 64];
        let h = HtileWord {
            min_z: 16383,
            max_z: 16383,
            zmask: 0,
        };
        assert_eq!(
            resolve_htile_fast_clears(&mut depth, 8, 8, &[h.encode()]),
            1
        );
        assert!((depth[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fmask_cmask_and_sampler_feedback_work() {
        let samples = vec![[100, 0, 0, 255], [200, 0, 0, 255]];
        let resolved = resolve_fmask_msaa(&samples, &[0b10], &[CMASK_EXPANDED], 1, 1, 2, [0; 4]);
        assert_eq!(resolved[0], [150, 0, 0, 255]);

        let mut fb = SamplerFeedbackMap::new(128, 128, 16);
        fb.record_sample(2, 3, 4);
        fb.record_sample(2, 3, 2);
        assert_eq!(fb.min_mip[(3 * fb.tiles_x + 2) as usize], 2);
    }
}
