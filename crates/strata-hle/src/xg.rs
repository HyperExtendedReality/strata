//! Xbox Series X|S `xg_xs.dll` Texture Layout Computer.
//!
//! Computes aligned pitches, tile dimensions, and 64 KiB-aligned allocation sizes
//! for console tiled (`64KB_S`, `64KB_D`) and linear textures, matching
//! `detile.slang`'s 8x8 micro-tile Morton addressing.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwizzleMode {
    Linear = 0,
    Swizzle64KS = 9,
    Swizzle64KD = 10,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureLayout {
    pub width: u32,
    pub height: u32,
    pub bpp: u32,
    pub swizzle: SwizzleMode,
    pub row_pitch_bytes: u32,
    pub total_size_bytes: u64,
    pub alignment_bytes: u64,
}

impl TextureLayout {
    pub fn compute_2d(width: u32, height: u32, bpp: u32, swizzle: SwizzleMode) -> Self {
        match swizzle {
            SwizzleMode::Linear => {
                let row_pitch = (width * bpp).next_multiple_of(256);
                let total = (row_pitch as u64 * height as u64).next_multiple_of(4096);
                Self {
                    width,
                    height,
                    bpp,
                    swizzle,
                    row_pitch_bytes: row_pitch,
                    total_size_bytes: total,
                    alignment_bytes: 4096,
                }
            }
            SwizzleMode::Swizzle64KS | SwizzleMode::Swizzle64KD => {
                let tiles_x = width.div_ceil(8);
                let tiles_y = height.div_ceil(8);
                let raw = (tiles_x as u64) * (tiles_y as u64) * 64 * (bpp as u64);
                let total = raw.max(65_536).next_multiple_of(65_536);
                Self {
                    width,
                    height,
                    bpp,
                    swizzle,
                    row_pitch_bytes: tiles_x * 8 * bpp,
                    total_size_bytes: total,
                    alignment_bytes: 65_536,
                }
            }
        }
    }

    /// CPU reference for 8x8 Morton micro-tile offset (matches `detile.slang::Swizzle64KS`).
    pub fn swizzle_64ks_offset(x: u32, y: u32, width: u32, bpp: u32) -> usize {
        let tile_x = x >> 3;
        let tile_y = y >> 3;
        let tiles_per_row = (width + 7) >> 3;
        let lx = x & 7;
        let ly = y & 7;
        let morton = ((lx & 1) << 0)
            | ((ly & 1) << 1)
            | ((lx & 2) << 1)
            | ((ly & 2) << 2)
            | ((lx & 4) << 2)
            | ((ly & 4) << 3);
        let elem = ((tile_y * tiles_per_row + tile_x) << 6) | morton;
        (elem * bpp) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_64k_aligned_swizzle_layout() {
        let l = TextureLayout::compute_2d(256, 256, 4, SwizzleMode::Swizzle64KS);
        assert_eq!(l.alignment_bytes, 65_536);
        assert_eq!(l.total_size_bytes, 256 * 1024);
        assert_eq!(TextureLayout::swizzle_64ks_offset(0, 0, 16, 4), 0);
        assert_eq!(TextureLayout::swizzle_64ks_offset(1, 0, 16, 4), 4);
        assert_eq!(TextureLayout::swizzle_64ks_offset(0, 1, 16, 4), 8);
    }
}
