//! Textures made from console image descriptors (T#) and title memory.
//!
//! A T# names an address, format, size and tiling; Strata reads the texel
//! bytes there (title memory is host memory), detiles them, and keeps a
//! Vulkan image per descriptor, uploaded again when the bytes change.
//! Storage images a shader wrote are copied back to title memory with
//! [`TextureCache::write_back`], since the title's CPU code reads them there.
//!
//! Layouts: `SW_MODE` 0 is linear with rows padded to 256 bytes; 9 and 10
//! (`64KB_S`/`64KB_D`) use Strata's 8×8 Morton micro-tile model, the one its
//! own `XGTextureAddressComputer` reports to titles (`strata-hle::xg`). That
//! model is not AMD's full address equation; other modes are refused.
//! Only float-typed (`UNORM`/`SNORM`/`FLOAT`) formats are made, because
//! translated shaders declare float images; integer formats are refused.

use std::collections::HashMap;

use ash::vk;
use strata_shader::descriptor::{ImageDescriptor, ImageKind, SamplerDescriptor, Select};
use strata_shader::format::{FormatInfo, NumFormat, gfx10_format};
use thiserror::Error;

use crate::context::GpuError;
use crate::images::GpuImage;
use crate::runner::ComputeRunner;

#[derive(Debug, Error)]
pub enum TextureError {
    #[error("image format {0} is not one Strata reads")]
    Format(u16),
    #[error("integer image format {0}: translated shaders read float images")]
    IntegerFormat(u16),
    #[error("{0:?} images are not supported (2D only)")]
    Kind(ImageKind),
    #[error("swizzle mode {0} is not modelled")]
    SwizzleMode(u8),
    #[error("cannot read {len} bytes of texels at 0x{addr:x}")]
    Read { addr: u64, len: usize },
    #[error("cannot write texels back to 0x{0:x}")]
    Write(u64),
    #[error("the device cannot {what} {format:?} images")]
    Unsupported {
        format: vk::Format,
        what: &'static str,
    },
    #[error(transparent)]
    Gpu(#[from] GpuError),
}

/// The Vulkan format a translated shader may read a GFX10 format as.
pub fn vk_format(info: &FormatInfo) -> Option<vk::Format> {
    use NumFormat::*;
    use vk::Format as F;
    Some(match (info.comps, info.bits, info.num) {
        (1, 8, Unorm) => F::R8_UNORM,
        (1, 8, Snorm) => F::R8_SNORM,
        (2, 8, Unorm) => F::R8G8_UNORM,
        (2, 8, Snorm) => F::R8G8_SNORM,
        (4, 8, Unorm) => F::R8G8B8A8_UNORM,
        (4, 8, Snorm) => F::R8G8B8A8_SNORM,
        (1, 16, Unorm) => F::R16_UNORM,
        (1, 16, Snorm) => F::R16_SNORM,
        (1, 16, Float) => F::R16_SFLOAT,
        (2, 16, Unorm) => F::R16G16_UNORM,
        (2, 16, Snorm) => F::R16G16_SNORM,
        (2, 16, Float) => F::R16G16_SFLOAT,
        (4, 16, Unorm) => F::R16G16B16A16_UNORM,
        (4, 16, Snorm) => F::R16G16B16A16_SNORM,
        (4, 16, Float) => F::R16G16B16A16_SFLOAT,
        (1, 32, Float) => F::R32_SFLOAT,
        (2, 32, Float) => F::R32G32_SFLOAT,
        (4, 32, Float) => F::R32G32B32A32_SFLOAT,
        _ => return None,
    })
}

fn swizzle(s: Select) -> vk::ComponentSwizzle {
    match s {
        Select::Zero => vk::ComponentSwizzle::ZERO,
        Select::One => vk::ComponentSwizzle::ONE,
        Select::X => vk::ComponentSwizzle::R,
        Select::Y => vk::ComponentSwizzle::G,
        Select::Z => vk::ComponentSwizzle::B,
        Select::W => vk::ComponentSwizzle::A,
    }
}

/// How texels sit in title memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tiling {
    /// Rows of `pitch` bytes.
    Linear { pitch: usize },
    /// Strata's 8×8 Morton micro-tiles (see the module docs).
    Morton,
}

/// The tiling and byte footprint of level 0 of `d`.
pub fn source_layout(d: &ImageDescriptor, bpp: usize) -> Result<(Tiling, usize), TextureError> {
    let (w, h) = (d.width as usize, d.height as usize);
    match d.sw_mode {
        0 => {
            let pitch = (w * bpp).next_multiple_of(256);
            Ok((Tiling::Linear { pitch }, pitch * h))
        }
        5 | 6 | 9 | 10 | 25 | 26 | 27 => {
            Ok((Tiling::Morton, w.div_ceil(8) * h.div_ceil(8) * 64 * bpp))
        }
        m => Err(TextureError::SwizzleMode(m)),
    }
}

/// Dimensions `(width, height)` of mip level `mip` (clamped to `>= 1`).
pub const fn mip_level_dimensions(width: u32, height: u32, mip: u32) -> (u32, u32) {
    let shift = if mip >= 31 { 31 } else { mip };
    let w = width >> shift;
    let h = height >> shift;
    (if w == 0 { 1 } else { w }, if h == 0 { 1 } else { h })
}

/// Total byte footprint across `mip_levels` levels for a 2D texture.
pub fn mip_chain_footprint(
    width: u32,
    height: u32,
    mip_levels: u32,
    bpp: usize,
    sw_mode: u8,
) -> Result<usize, TextureError> {
    let levels = mip_levels.max(1);
    let mut total = 0usize;
    for m in 0..levels {
        let (mw, mh) = mip_level_dimensions(width, height, m);
        let level_bytes = match sw_mode {
            0 => (mw as usize * bpp).next_multiple_of(256) * mh as usize,
            5 | 6 | 9 | 10 | 25 | 26 | 27 => {
                (mw as usize).div_ceil(8) * (mh as usize).div_ceil(8) * 64 * bpp
            }
            other => return Err(TextureError::SwizzleMode(other)),
        };
        total += level_bytes;
    }
    Ok(total)
}

fn morton_offset(x: usize, y: usize, width: usize, bpp: usize) -> usize {
    let tiles_per_row = width.div_ceil(8);
    let (lx, ly) = (x & 7, y & 7);
    let morton = (lx & 1)
        | ((ly & 1) << 1)
        | ((lx & 2) << 1)
        | ((ly & 2) << 2)
        | ((lx & 4) << 2)
        | ((ly & 4) << 3);
    ((((y >> 3) * tiles_per_row + (x >> 3)) << 6) | morton) * bpp
}

/// Title-memory bytes -> tightly packed rows.
pub fn detile(d: &ImageDescriptor, bpp: usize, tiling: Tiling, src: &[u8]) -> Vec<u8> {
    let (w, h) = (d.width as usize, d.height as usize);
    let mut out = vec![0u8; w * h * bpp];
    for y in 0..h {
        match tiling {
            Tiling::Linear { pitch } => {
                out[y * w * bpp..(y + 1) * w * bpp]
                    .copy_from_slice(&src[y * pitch..y * pitch + w * bpp]);
            }
            Tiling::Morton => {
                for x in 0..w {
                    let s = morton_offset(x, y, w, bpp);
                    let o = (y * w + x) * bpp;
                    out[o..o + bpp].copy_from_slice(&src[s..s + bpp]);
                }
            }
        }
    }
    out
}

/// Tightly packed rows -> title-memory bytes (`dst` holds the footprint).
pub fn retile(d: &ImageDescriptor, bpp: usize, tiling: Tiling, tight: &[u8], dst: &mut [u8]) {
    let (w, h) = (d.width as usize, d.height as usize);
    for y in 0..h {
        match tiling {
            Tiling::Linear { pitch } => {
                dst[y * pitch..y * pitch + w * bpp]
                    .copy_from_slice(&tight[y * w * bpp..(y + 1) * w * bpp]);
            }
            Tiling::Morton => {
                for x in 0..w {
                    let s = (y * w + x) * bpp;
                    let o = morton_offset(x, y, w, bpp);
                    dst[o..o + bpp].copy_from_slice(&tight[s..s + bpp]);
                }
            }
        }
    }
}

/// Detile an entire multi-mip 2D chain from title memory into tightly packed
/// per-mip buffers (`out[m]` has `mw * mh * bpp` bytes).
pub fn detile_mip_chain(
    width: u32,
    height: u32,
    mip_levels: u32,
    bpp: usize,
    sw_mode: u8,
    src: &[u8],
) -> Result<Vec<Vec<u8>>, TextureError> {
    let levels = mip_levels.max(1);
    let mut out = Vec::with_capacity(levels as usize);
    let mut cursor = 0usize;
    for m in 0..levels {
        let (mw, mh) = mip_level_dimensions(width, height, m);
        let mut desc = ImageDescriptor::decode([0; 8]);
        desc.width = mw;
        desc.height = mh;
        desc.sw_mode = sw_mode;
        let (tiling, foot) = source_layout(&desc, bpp)?;
        let slice = src.get(cursor..cursor + foot).ok_or(TextureError::Read {
            addr: cursor as u64,
            len: foot,
        })?;
        out.push(detile(&desc, bpp, tiling, slice));
        cursor += foot;
    }
    Ok(out)
}

/// Retile tightly packed per-mip buffers back into a contiguous title-memory
/// mip chain buffer.
pub fn retile_mip_chain(
    width: u32,
    height: u32,
    bpp: usize,
    sw_mode: u8,
    mips: &[Vec<u8>],
) -> Result<Vec<u8>, TextureError> {
    let total = mip_chain_footprint(width, height, mips.len() as u32, bpp, sw_mode)?;
    let mut dst = vec![0u8; total];
    let mut cursor = 0usize;
    for (m, tight) in mips.iter().enumerate() {
        let (mw, mh) = mip_level_dimensions(width, height, m as u32);
        let mut desc = ImageDescriptor::decode([0; 8]);
        desc.width = mw;
        desc.height = mh;
        desc.sw_mode = sw_mode;
        let (tiling, foot) = source_layout(&desc, bpp)?;
        retile(&desc, bpp, tiling, tight, &mut dst[cursor..cursor + foot]);
        cursor += foot;
    }
    Ok(dst)
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

struct Entry {
    image: GpuImage,
    hash: u64,
}

/// Reads `len` bytes of title memory at an address (`None` if unreadable).
pub type ReadFn<'a> = &'a dyn Fn(u64, usize) -> Option<Vec<u8>>;
/// Writes bytes to title memory at an address (`false` if unwritable).
pub type WriteFn<'a> = &'a dyn Fn(u64, &[u8]) -> bool;

/// Vulkan images and samplers for the descriptors titles use.
#[derive(Default)]
pub struct TextureCache {
    /// By the T#'s first five dwords (address, format, size, tiling, selects).
    textures: HashMap<[u32; 5], Entry>,
    samplers: HashMap<[u32; 4], vk::Sampler>,
    pub uploads: u64,
}

fn key(words: &[u32; 8]) -> [u32; 5] {
    [words[0], words[1], words[2], words[3], words[4]]
}

impl TextureCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn check(words: &[u32; 8]) -> Result<(ImageDescriptor, FormatInfo, vk::Format), TextureError> {
        let d = ImageDescriptor::decode(*words);
        if d.kind != ImageKind::Tex2D && !(d.kind == ImageKind::Tex2DArray && d.depth == 1) {
            return Err(TextureError::Kind(d.kind));
        }
        let info = gfx10_format(d.format).ok_or(TextureError::Format(d.format))?;
        if !info.num.is_float() {
            return Err(TextureError::IntegerFormat(d.format));
        }
        let fmt = vk_format(&info).ok_or(TextureError::Format(d.format))?;
        Ok((d, info, fmt))
    }

    /// The image view for a T#, created or refreshed from title memory.
    /// `storage`: the shader writes it (the image gets `STORAGE` usage).
    pub fn texture(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 8],
        storage: bool,
        read: ReadFn<'_>,
    ) -> Result<vk::ImageView, TextureError> {
        let (d, info, fmt) = Self::check(&words)?;
        let bpp = info.bytes_per_element() as usize;
        let (tiling, footprint) = source_layout(&d, bpp)?;
        let src = read(d.base_address, footprint).ok_or(TextureError::Read {
            addr: d.base_address,
            len: footprint,
        })?;
        let hash = fnv(&src);
        let k = key(&words);

        if let Some(e) = self.textures.get_mut(&k) {
            if e.image.storage || !storage {
                if e.hash != hash {
                    runner.upload_image(&e.image, &detile(&d, bpp, tiling, &src))?;
                    e.hash = hash;
                    self.uploads += 1;
                }
                return Ok(e.image.view);
            }
            // Needed as storage now: made again with STORAGE usage below.
            let old = self.textures.remove(&k).expect("present");
            runner.destroy_image(old.image);
        }

        let need = if storage {
            vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::STORAGE_IMAGE
        } else {
            vk::FormatFeatureFlags::SAMPLED_IMAGE
        };
        if !runner.format_supports(fmt, need) {
            return Err(TextureError::Unsupported {
                format: fmt,
                what: if storage { "store to" } else { "sample" },
            });
        }
        let components = vk::ComponentMapping {
            r: swizzle(d.select[0]),
            g: swizzle(d.select[1]),
            b: swizzle(d.select[2]),
            a: swizzle(d.select[3]),
        };
        let image = runner.create_image_2d(fmt, d.width, d.height, storage, components)?;
        if let Err(e) = runner.upload_image(&image, &detile(&d, bpp, tiling, &src)) {
            runner.destroy_image(image);
            return Err(e.into());
        }
        self.uploads += 1;
        let view = image.view;
        self.textures.insert(k, Entry { image, hash });
        Ok(view)
    }

    /// The image made for a T# (after [`Self::texture`]).
    pub fn image(&self, words: &[u32; 8]) -> Option<&GpuImage> {
        self.textures.get(&key(words)).map(|e| &e.image)
    }

    /// Copies a storage image back to the title memory its T# names.
    pub fn write_back(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 8],
        write: WriteFn<'_>,
        read: ReadFn<'_>,
    ) -> Result<(), TextureError> {
        let (d, info, _) = Self::check(&words)?;
        let bpp = info.bytes_per_element() as usize;
        let (tiling, footprint) = source_layout(&d, bpp)?;
        let Some(e) = self.textures.get_mut(&key(&words)) else {
            return Ok(());
        };
        let tight = runner.read_image(&e.image, (d.width * d.height) as usize * bpp)?;
        // Keep the bytes between rows (linear padding) as they were.
        let mut dst = read(d.base_address, footprint).unwrap_or_else(|| vec![0; footprint]);
        retile(&d, bpp, tiling, &tight, &mut dst);
        if !write(d.base_address, &dst) {
            return Err(TextureError::Write(d.base_address));
        }
        e.hash = fnv(&dst); // what title memory now holds: no re-upload needed
        Ok(())
    }

    /// The sampler for an S#.
    pub fn sampler(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 4],
    ) -> Result<vk::Sampler, TextureError> {
        if let Some(&s) = self.samplers.get(&words) {
            return Ok(s);
        }
        let s = runner.create_sampler(&SamplerDescriptor::decode(words))?;
        self.samplers.insert(words, s);
        Ok(s)
    }

    pub fn destroy(&mut self, runner: &ComputeRunner) {
        for (_, e) in self.textures.drain() {
            runner.destroy_image(e.image);
        }
        for (_, s) in self.samplers.drain() {
            runner.destroy_sampler(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(w: u32, h: u32, sw_mode: u8) -> ImageDescriptor {
        ImageDescriptor {
            base_address: 0,
            format: 56,
            width: w,
            height: h,
            depth: 1,
            kind: ImageKind::Tex2D,
            sw_mode,
            select: [Select::X, Select::Y, Select::Z, Select::W],
            base_level: 0,
            last_level: 0,
            base_array: 0,
        }
    }

    #[test]
    fn linear_and_morton_layouts_round_trip() {
        for mode in [0u8, 9] {
            let d = desc(12, 5, mode);
            let (tiling, footprint) = source_layout(&d, 4).unwrap();
            let tight: Vec<u8> = (0..12 * 5 * 4).map(|i| i as u8).collect();
            let mut mem = vec![0xEE; footprint];
            retile(&d, 4, tiling, &tight, &mut mem);
            assert_eq!(detile(&d, 4, tiling, &mem), tight, "mode {mode}");
        }
        assert_eq!(
            source_layout(&desc(8, 8, 0), 4).unwrap(),
            (Tiling::Linear { pitch: 256 }, 2048)
        );
        assert!(matches!(
            source_layout(&desc(8, 8, 24), 4),
            Err(TextureError::SwizzleMode(24))
        ));
    }

    #[test]
    fn morton_matches_strata_xg_model() {
        // Same equation as strata-hle's XGTextureAddressComputer reference.
        assert_eq!(morton_offset(0, 0, 16, 4), 0);
        assert_eq!(morton_offset(1, 0, 16, 4), 4);
        assert_eq!(morton_offset(0, 1, 16, 4), 8);
        assert_eq!(morton_offset(8, 0, 16, 4), 64 * 4);
    }

    #[test]
    fn only_float_formats_map_to_vulkan() {
        assert_eq!(
            vk_format(&gfx10_format(56).unwrap()),
            Some(vk::Format::R8G8B8A8_UNORM)
        );
        assert_eq!(
            vk_format(&gfx10_format(71).unwrap()),
            Some(vk::Format::R16G16B16A16_SFLOAT)
        );
        assert!(matches!(
            TextureCache::check(&{
                let mut d = desc(8, 8, 0);
                d.format = 60; // 8_8_8_8_UINT
                d.encode()
            }),
            Err(TextureError::IntegerFormat(60))
        ));
    }
}
