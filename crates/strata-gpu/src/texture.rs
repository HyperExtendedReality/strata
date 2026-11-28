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
//! Levels, array layers, cube faces and 3D slices follow one another (see
//! [`SurfaceModel`]); BCn formats tile 4×4 blocks as elements.
//! Only float-typed (`UNORM`/`SNORM`/`FLOAT`) and BCn formats are made,
//! because translated shaders declare float images; integer formats are
//! refused.

use std::collections::HashMap;

use ash::vk;
use strata_shader::descriptor::{ImageDescriptor, ImageKind, SamplerDescriptor, Select};
use strata_shader::format::{FormatInfo, NumFormat, gfx10_format};
use strata_shader::resource::ImageDim;
use thiserror::Error;

use crate::context::GpuError;
use crate::images::{GpuImage, ImageShape, ImageSpec, level_extent};
use crate::runner::ComputeRunner;

#[derive(Debug, Error)]
pub enum TextureError {
    #[error("image format {0} is not one Strata reads")]
    Format(u16),
    #[error("integer image format {0}: translated shaders read float images")]
    IntegerFormat(u16),
    #[error("image kind {0:?} is not one Strata reads")]
    Kind(ImageKind),
    #[error("a {shape:?} image cannot be bound as a {dim:?} image")]
    View { shape: ImageShape, dim: ImageDim },
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

/// The BCn format of a GFX10 `IMG_FORMAT` code (public RDNA2 ISA
/// enumeration: `BC1_UNORM` = 109 through `BC7_SRGB` = 122).
pub fn bc_format(code: u16) -> Option<vk::Format> {
    use vk::Format as F;
    Some(match code {
        109 => F::BC1_RGBA_UNORM_BLOCK,
        110 => F::BC1_RGBA_SRGB_BLOCK,
        111 => F::BC2_UNORM_BLOCK,
        112 => F::BC2_SRGB_BLOCK,
        113 => F::BC3_UNORM_BLOCK,
        114 => F::BC3_SRGB_BLOCK,
        115 => F::BC4_UNORM_BLOCK,
        116 => F::BC4_SNORM_BLOCK,
        117 => F::BC5_UNORM_BLOCK,
        118 => F::BC5_SNORM_BLOCK,
        119 => F::BC6H_UFLOAT_BLOCK,
        120 => F::BC6H_SFLOAT_BLOCK,
        121 => F::BC7_UNORM_BLOCK,
        122 => F::BC7_SRGB_BLOCK,
        _ => return None,
    })
}

/// Vulkan format lookup for T# format codes: BCn block formats, packed 10/11-bit
/// and 16-bit formats, and the whole-bit float formats of [`vk_format`].
pub fn vk_format_extended(dfmt: u16) -> Option<vk::Format> {
    use strata_shader::format::{PackedFormat, gfx10_packed_format};
    if let Some(bc) = bc_format(dfmt) {
        return Some(bc);
    }
    if let Some(packed) = gfx10_packed_format(dfmt) {
        return Some(match packed {
            PackedFormat::F10_11_11 | PackedFormat::F11_11_10 => {
                vk::Format::B10G11R11_UFLOAT_PACK32
            }
            PackedFormat::Unorm2_10_10_10 => vk::Format::A2B10G10R10_UNORM_PACK32,
            PackedFormat::Snorm2_10_10_10 => vk::Format::A2B10G10R10_SNORM_PACK32,
            PackedFormat::Uint2_10_10_10 => vk::Format::A2B10G10R10_UINT_PACK32,
            PackedFormat::Sint2_10_10_10 => vk::Format::A2B10G10R10_SINT_PACK32,
            PackedFormat::Unorm10_10_10_2 => vk::Format::A2R10G10B10_UNORM_PACK32,
            PackedFormat::Snorm10_10_10_2 => vk::Format::A2R10G10B10_SNORM_PACK32,
            PackedFormat::Uint10_10_10_2 => vk::Format::A2R10G10B10_UINT_PACK32,
            PackedFormat::Sint10_10_10_2 => vk::Format::A2R10G10B10_SINT_PACK32,
            PackedFormat::F5_9_9_9 => vk::Format::E5B9G9R9_UFLOAT_PACK32,
            PackedFormat::Unorm5_6_5 => vk::Format::R5G6B5_UNORM_PACK16,
            PackedFormat::Unorm1_5_5_5 => vk::Format::A1R5G5B5_UNORM_PACK16,
            PackedFormat::Unorm4_4_4_4 => vk::Format::R4G4B4A4_UNORM_PACK16,
        });
    }
    gfx10_format(dfmt).and_then(|info| vk_format(&info))
}

/// Returns `true` if `fmt` is a 4×4 BCn block-compressed format.
pub fn is_block_compressed(fmt: vk::Format) -> bool {
    (vk::Format::BC1_RGB_UNORM_BLOCK.as_raw()..=vk::Format::BC7_SRGB_BLOCK.as_raw())
        .contains(&fmt.as_raw())
}

/// Byte size of a single 4×4 BCn block (8 for BC1/BC4, 16 for BC2/BC3/BC5/BC6H/BC7).
pub fn bcn_block_bytes(fmt: vk::Format) -> usize {
    match fmt {
        vk::Format::BC1_RGB_UNORM_BLOCK
        | vk::Format::BC1_RGB_SRGB_BLOCK
        | vk::Format::BC1_RGBA_UNORM_BLOCK
        | vk::Format::BC1_RGBA_SRGB_BLOCK
        | vk::Format::BC4_UNORM_BLOCK
        | vk::Format::BC4_SNORM_BLOCK => 8,
        _ => 16,
    }
}

/// Returns `true` if `fmt` is a depth or depth/stencil attachment format.
pub fn is_depth_stencil(fmt: vk::Format) -> bool {
    matches!(
        fmt,
        vk::Format::D16_UNORM
            | vk::Format::D32_SFLOAT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
    )
}

/// How texels sit in title memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tiling {
    /// Rows of `pitch` bytes.
    Linear { pitch: usize },
    /// Strata's 8×8 Morton micro-tiles (`4KB_S`, `4KB_D`, `64KB_S`, `64KB_D`).
    Morton,
    /// GFX10 bank-XOR 64KB swizzle modes (`64KB_S_X` = 25, `64KB_D_X` = 26, `64KB_R_X` = 27).
    MortonXor { mode: u8 },
}

/// The tiling and byte footprint of level 0 of `d`.
pub fn source_layout(d: &ImageDescriptor, bpp: usize) -> Result<(Tiling, usize), TextureError> {
    let (w, h) = (d.width as usize, d.height as usize);
    match d.sw_mode {
        0 => {
            let pitch = (w * bpp).next_multiple_of(256);
            Ok((Tiling::Linear { pitch }, pitch * h))
        }
        5 | 6 | 9 | 10 => Ok((Tiling::Morton, w.div_ceil(8) * h.div_ceil(8) * 64 * bpp)),
        m @ 25..=27 => Ok((
            Tiling::MortonXor { mode: m },
            w.div_ceil(8) * h.div_ceil(8) * 64 * bpp,
        )),
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

fn morton_xor_offset(x: usize, y: usize, width: usize, bpp: usize, mode: u8) -> usize {
    let tiles_per_row = width.div_ceil(8);
    let (tx, ty) = (x >> 3, y >> 3);
    let (lx, ly) = (x & 7, y & 7);
    let bank_xor = match mode {
        26 => (ty & 3) ^ ((tx & 3) << 1),
        27 => (ty & 7) ^ (tx & 3),
        _ => ty & 3, // 64KB_S_X (25)
    };
    let lx_sw = lx ^ (bank_xor & 7);
    let morton = (lx_sw & 1)
        | ((ly & 1) << 1)
        | ((lx_sw & 2) << 1)
        | ((ly & 2) << 2)
        | ((lx_sw & 4) << 2)
        | ((ly & 4) << 3);
    (((ty * tiles_per_row + tx) << 6) | morton) * bpp
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
            Tiling::MortonXor { mode } => {
                for x in 0..w {
                    let s = morton_xor_offset(x, y, w, bpp, mode);
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
            Tiling::MortonXor { mode } => {
                for x in 0..w {
                    let s = (y * w + x) * bpp;
                    let o = morton_xor_offset(x, y, w, bpp, mode);
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
    base_address: u64,
    footprint: usize,
    /// A logged write overlapped the texture since it was last verified.
    dirty: bool,
    /// Its memory is write-watched ([`crate::dirty_log::is_watched`]), so
    /// "not dirty" means unchanged and the bind can skip re-reading.
    watched: bool,
    /// Eviction slot (`0`: never evicted — storage images and depth
    /// targets hold GPU-written contents).
    slot: u32,
    /// The first upload still running on the transfer queue: the texture
    /// binds a placeholder until it finishes ("show after").
    pending: Option<crate::images::PendingUpload>,
}

/// Finishes `e`'s transfer-queue upload, if any (blocking if still running).
fn settle(runner: &ComputeRunner, e: &mut Entry) -> Result<(), GpuError> {
    if let Some(p) = e.pending.take() {
        runner.finish_upload(p)?;
    }
    Ok(())
}

/// Reads `len` bytes of title memory at an address (`None` if unreadable).
pub type ReadFn<'a> = &'a dyn Fn(u64, usize) -> Option<Vec<u8>>;
/// Writes bytes to title memory at an address (`false` if unwritable).
pub type WriteFn<'a> = &'a dyn Fn(u64, &[u8]) -> bool;

/// How a T# lays out in title memory, under Strata's texture model (the
/// one `strata-hle::xg` reports to titles): levels from 0 one after another,
/// each level its slices (array layers, cube faces, 3D slices) one after
/// another, each slice a 2D surface in the descriptor's swizzle mode with
/// 4×4 blocks as elements for BCn formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceModel {
    pub desc: ImageDescriptor,
    pub format: vk::Format,
    /// Bytes per element (texel, or 4×4 block for BCn).
    pub bpe: usize,
    /// 4 for BCn formats, else 1.
    pub block: u32,
    pub shape: ImageShape,
    /// Levels the image holds: `base_level..=last_level` of the resource.
    pub first_level: u32,
    pub levels: u32,
}

impl SurfaceModel {
    /// Slices of resource level `m`.
    fn slices(&self, m: u32) -> u32 {
        match self.shape {
            ImageShape::D3 { depth } => (depth >> m).max(1),
            s => s.layers(),
        }
    }

    /// The tiling and byte footprint of one slice of resource level `m`.
    fn slice_layout(&self, m: u32) -> Result<(ImageDescriptor, Tiling, usize), TextureError> {
        let (w, h) = level_extent(self.desc.width, self.desc.height, m);
        let mut d = self.desc;
        d.width = w.div_ceil(self.block);
        d.height = h.div_ceil(self.block);
        let (t, f) = source_layout(&d, self.bpe)?;
        Ok((d, t, f))
    }

    /// Byte offset of resource level `m` from the base address.
    fn level_offset(&self, m: u32) -> Result<usize, TextureError> {
        let mut off = 0;
        for l in 0..m {
            off += self.slice_layout(l)?.2 * self.slices(l) as usize;
        }
        Ok(off)
    }

    /// Bytes from the base address to the end of the last level held.
    pub fn footprint(&self) -> Result<usize, TextureError> {
        self.level_offset(self.first_level + self.levels)
    }

    /// Title-memory bytes (from the base address, `footprint()` long) →
    /// tightly packed levels in [`ComputeRunner::upload_image_levels`] order.
    fn detile_all(&self, src: &[u8]) -> Result<Vec<u8>, TextureError> {
        let mut out = Vec::new();
        for m in self.first_level..self.first_level + self.levels {
            let (d, tiling, foot) = self.slice_layout(m)?;
            let mut at = self.level_offset(m)?;
            for _ in 0..self.slices(m) {
                let slice = src.get(at..at + foot).ok_or(TextureError::Read {
                    addr: self.desc.base_address + at as u64,
                    len: foot,
                })?;
                out.extend_from_slice(&detile(&d, self.bpe, tiling, slice));
                at += foot;
            }
        }
        Ok(out)
    }

    /// [`Self::detile_all`]: title-memory bytes to tightly packed levels.
    pub fn detile_levels(&self, src: &[u8]) -> Result<Vec<u8>, TextureError> {
        self.detile_all(src)
    }

    /// The `detile.slang` jobs for every level and slice, and the packed
    /// size, when the GPU detiler reproduces [`Self::detile_all`] exactly:
    /// 8×8 Morton surfaces with 1, 2, 4, 8 or 16-byte elements.
    pub fn gpu_detile_jobs(&self) -> Option<(Vec<crate::images::GpuDetileJob>, usize)> {
        if !matches!(self.bpe, 1 | 2 | 4 | 8 | 16) {
            return None;
        }
        let mut jobs = Vec::new();
        let mut cursor = 0usize;
        for m in self.first_level..self.first_level + self.levels {
            let (d, tiling, foot) = self.slice_layout(m).ok()?;
            if !matches!(tiling, Tiling::Morton) {
                return None;
            }
            let mut at = self.level_offset(m).ok()?;
            for _ in 0..self.slices(m) {
                jobs.push(crate::images::GpuDetileJob {
                    src_offset: at as u64,
                    dst_offset: cursor as u64,
                    width: d.width,
                    height: d.height,
                    bpp: self.bpe as u32,
                    mode: 1,
                });
                cursor += (d.width * d.height) as usize * self.bpe;
                at += foot;
            }
        }
        Some((jobs, cursor))
    }

    /// The inverse of [`Self::detile_all`], into `dst` (keeps bytes the
    /// model does not cover, such as linear row padding).
    fn retile_all(&self, tight: &[u8], dst: &mut [u8]) -> Result<(), TextureError> {
        let mut cursor = 0usize;
        for m in self.first_level..self.first_level + self.levels {
            let (d, tiling, foot) = self.slice_layout(m)?;
            let tight_slice = (d.width * d.height) as usize * self.bpe;
            let mut at = self.level_offset(m)?;
            for _ in 0..self.slices(m) {
                let src = tight
                    .get(cursor..cursor + tight_slice)
                    .ok_or(TextureError::Write(self.desc.base_address))?;
                let out = dst
                    .get_mut(at..at + foot)
                    .ok_or(TextureError::Write(self.desc.base_address + at as u64))?;
                retile(&d, self.bpe, tiling, src, out);
                cursor += tight_slice;
                at += foot;
            }
        }
        Ok(())
    }
}

/// `'ST'` in the top half of T# dword 5: a descriptor Strata built itself
/// (D3D12 views), whose low half holds these flags and dword 6 a Vulkan
/// depth format. Console T#s keep metadata there; only the host reads it.
pub const TRAILER_MAGIC: u32 = 0x5354_0000;
/// Bytes are B, G, R, A (the GFX10 format is `8_8_8_8` with swapped selects).
pub const TRAILER_BGRA: u32 = 1;
/// sRGB-encoded.
pub const TRAILER_SRGB: u32 = 2;
/// A depth format (dword 6: its `VkFormat`).
pub const TRAILER_DEPTH: u32 = 4;

/// Strata's trailer flags, when `w` carries them.
pub fn trailer_flags(w: &[u32; 8]) -> Option<u32> {
    (w[5] & 0xffff_0000 == TRAILER_MAGIC).then_some(w[5] & 0xffff)
}

/// Vulkan images and samplers for the descriptors titles use, with integrated
/// dirty-range tracking, persistent [`crate::texstore::TextureStore`] caching,
/// [`crate::memory_stack::BindlessFeedbackEvictor`] VRAM eviction, and
/// [`crate::memory_stack::MemoryTelemetry`] latency histograms.
type TextureKey = ([u32; 5], i32);

pub struct TextureCache {
    /// By the T#'s first five dwords (address, format, size, tiling, selects,
    /// levels, depth) and, for depth targets, the depth format.
    textures: HashMap<([u32; 5], i32), Entry>,
    samplers: HashMap<([u32; 4], Option<u8>), vk::Sampler>,
    key_by_slot: HashMap<u32, ([u32; 5], i32)>,
    next_slot: u32,
    /// Title-memory writes since the last bind ([`crate::dirty_log`]).
    dirty: Option<std::sync::Arc<crate::dirty_log::DirtyQueue>>,
    epoch: std::time::Instant,
    clock_sec: u64,
    /// Detile on the GPU (`detile.slang`) for Morton surfaces of at least
    /// this many bytes (`usize::MAX`: never).
    gpu_detile_min: usize,
    store: Option<crate::texstore::TextureStore>,
    evictor: crate::memory_stack::BindlessFeedbackEvictor,
    pub telemetry: crate::memory_stack::MemoryTelemetry,
    pub uploads: u64,
    pub clean_skips: u64,
    pub store_hits: u64,
    pub evicted_images: u64,
    pub gpu_detiles: u64,
    /// First uploads started on the transfer queue.
    pub async_uploads: u64,
    /// Binds that showed a placeholder while an upload ran.
    pub placeholder_binds: u64,
    /// Sampled textures at least this large upload asynchronously
    /// (`usize::MAX`: never).
    async_min: usize,
    staging_limit: u64,
    pending_bytes: u64,
    pending_order: std::collections::BTreeSet<(std::time::Instant, TextureKey)>,
    configured_budget: u64,
    pressure: bool,
    streaming: StreamingStats,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StreamingStats {
    pub heap_usage: u64,
    pub heap_budget: u64,
    pub commit_available: u64,
    pub resident_bytes: u64,
    pub oldest_upload_ms: u64,
}

/// Default sampled-texture VRAM budget: `STRATA_TEXTURE_BUDGET_MB`, else
/// 4 GiB (fits 8–16 GB cards next to render targets and buffers).
fn texture_budget_bytes() -> u64 {
    std::env::var("STRATA_TEXTURE_BUDGET_MB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&mb| mb >= 64)
        .map_or(4 << 30, |mb| mb << 20)
}

/// `STRATA_GPU_DETILE=0` keeps detiling on the CPU.
fn gpu_detile_min_bytes() -> usize {
    match std::env::var("STRATA_GPU_DETILE")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("0" | "off" | "false") => usize::MAX,
        _ => 256 << 10,
    }
}

impl Default for TextureCache {
    fn default() -> Self {
        let budget = texture_budget_bytes();
        Self {
            textures: HashMap::new(),
            samplers: HashMap::new(),
            key_by_slot: HashMap::new(),
            next_slot: 1,
            dirty: None,
            epoch: std::time::Instant::now(),
            clock_sec: 0,
            gpu_detile_min: gpu_detile_min_bytes(),
            store: None,
            // Textures unread for 30 s go first once the budget is exceeded,
            // down to 75 % of it.
            evictor: {
                crate::memory_stack::BindlessFeedbackEvictor::new(30, budget, budget / 4 * 3)
            },
            telemetry: crate::memory_stack::MemoryTelemetry::default(),
            uploads: 0,
            clean_skips: 0,
            store_hits: 0,
            evicted_images: 0,
            gpu_detiles: 0,
            async_uploads: 0,
            placeholder_binds: 0,
            async_min: 1 << 20,
            staging_limit: 64 << 20,
            pending_bytes: 0,
            pending_order: Default::default(),
            configured_budget: budget,
            pressure: false,
            streaming: StreamingStats::default(),
        }
    }
}

fn key(words: &[u32; 8], depth: Option<vk::Format>) -> ([u32; 5], i32) {
    // The trailer's storage flags make a different Vulkan format.
    let flags = trailer_flags(words).unwrap_or(0) as i32;
    (
        [words[0], words[1], words[2], words[3], words[4]],
        depth.map_or(0, |f| f.as_raw()) ^ (flags << 24),
    )
}

/// The Vulkan format a T#'s format code reads as, its element size and
/// block size (`None`: Strata does not read it).
pub fn rgba8_order(format: vk::Format, bgra: bool, srgb: bool) -> vk::Format {
    match (format, bgra, srgb) {
        (vk::Format::R8G8B8A8_UNORM, true, true) => vk::Format::B8G8R8A8_SRGB,
        (vk::Format::R8G8B8A8_UNORM, true, false) => vk::Format::B8G8R8A8_UNORM,
        (vk::Format::R8G8B8A8_UNORM, false, true) => vk::Format::R8G8B8A8_SRGB,
        _ => format,
    }
}

pub fn texel_format(code: u16) -> Result<(vk::Format, usize, u32), TextureError> {
    if let Some(f) = bc_format(code) {
        return Ok((f, bcn_block_bytes(f), 4));
    }
    // Packed 32-bit formats with a Vulkan equivalent (component order:
    // GFX10 names the high bits first; DXGI/Vulkan agree on the bit layout).
    use strata_shader::format::{PackedFormat, gfx10_packed_format};
    match gfx10_packed_format(code) {
        Some(PackedFormat::F10_11_11 | PackedFormat::F11_11_10) => {
            return Ok((vk::Format::B10G11R11_UFLOAT_PACK32, 4, 1));
        }
        Some(PackedFormat::Unorm2_10_10_10) => {
            return Ok((vk::Format::A2B10G10R10_UNORM_PACK32, 4, 1));
        }
        Some(PackedFormat::Snorm2_10_10_10) => {
            return Ok((vk::Format::A2B10G10R10_SNORM_PACK32, 4, 1));
        }
        Some(PackedFormat::Unorm10_10_10_2) => {
            return Ok((vk::Format::A2R10G10B10_UNORM_PACK32, 4, 1));
        }
        Some(PackedFormat::Snorm10_10_10_2) => {
            return Ok((vk::Format::A2R10G10B10_SNORM_PACK32, 4, 1));
        }
        Some(PackedFormat::F5_9_9_9) => {
            return Ok((vk::Format::E5B9G9R9_UFLOAT_PACK32, 4, 1));
        }
        Some(PackedFormat::Unorm5_6_5) => {
            return Ok((vk::Format::R5G6B5_UNORM_PACK16, 2, 1));
        }
        Some(PackedFormat::Unorm1_5_5_5) => {
            return Ok((vk::Format::A1R5G5B5_UNORM_PACK16, 2, 1));
        }
        Some(PackedFormat::Unorm4_4_4_4) => {
            return Ok((vk::Format::R4G4B4A4_UNORM_PACK16, 2, 1));
        }
        Some(_) => return Err(TextureError::IntegerFormat(code)),
        None => {}
    }
    let info = gfx10_format(code).ok_or(TextureError::Format(code))?;
    if !info.num.is_float() {
        return Err(TextureError::IntegerFormat(code));
    }
    let fmt = vk_format(&info).ok_or(TextureError::Format(code))?;
    Ok((fmt, info.bytes_per_element() as usize, 1))
}

/// The image shape a T# describes (`Cube`: `depth` counts cubes).
pub fn shape_of(d: &ImageDescriptor) -> Result<ImageShape, TextureError> {
    Ok(match d.kind {
        ImageKind::Tex1D | ImageKind::Tex2D | ImageKind::Tex2DMsaa => ImageShape::D2 { layers: 1 },
        ImageKind::Tex1DArray | ImageKind::Tex2DArray | ImageKind::Tex2DMsaaArray => {
            ImageShape::D2 { layers: d.depth }
        }
        ImageKind::Tex3D => ImageShape::D3 { depth: d.depth },
        ImageKind::Cube => ImageShape::Cube { cubes: d.depth },
        k @ ImageKind::Other(_) => return Err(TextureError::Kind(k)),
    })
}

impl TextureCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Trust the title-memory write log ([`crate::dirty_log`]): a cached
    /// texture in write-watched memory that no logged write touched is bound
    /// without re-reading and re-hashing its footprint. Textures outside
    /// watched memory are always verified.
    pub fn set_dirty_tracking(&mut self, enabled: bool) {
        self.dirty = enabled.then(crate::dirty_log::subscribe);
        for e in self.textures.values_mut() {
            e.dirty = true;
        }
    }

    /// Upload new sampled textures of at least `min_bytes` on the transfer
    /// queue (`usize::MAX` disables; needs a device transfer queue).
    pub fn set_async_upload_min(&mut self, min_bytes: usize) {
        self.async_min = min_bytes;
    }

    /// Detile Morton surfaces of at least `min_bytes` on the GPU
    /// (`usize::MAX` disables).
    pub fn set_gpu_detile_min(&mut self, min_bytes: usize) {
        self.gpu_detile_min = min_bytes;
    }

    /// Applies logged writes to the cached entries.
    fn apply_dirty_log(&mut self) {
        let Some(q) = &self.dirty else { return };
        let d = q.drain();
        if d.everything {
            for e in self.textures.values_mut() {
                e.dirty = true;
            }
        }
        for (a, n) in d.ranges {
            self.mark_dirty_range(a, n);
        }
    }

    /// Attach a persistent on-disk [`crate::texstore::TextureStore`] for detiled
    /// texture caching across runs.
    pub fn attach_store(&mut self, store: crate::texstore::TextureStore) {
        self.store = Some(store);
    }

    /// Configure bindless feedback VRAM eviction thresholds.
    pub fn configure_eviction(
        &mut self,
        idle_window_s: u64,
        high_water_bytes: u64,
        low_water_bytes: u64,
    ) {
        self.configured_budget = high_water_bytes;
        self.evictor = crate::memory_stack::BindlessFeedbackEvictor::new(
            idle_window_s,
            high_water_bytes,
            low_water_bytes,
        );
    }

    /// Advance the logical texture-streaming clock (in seconds).
    pub fn set_clock_sec(&mut self, now_sec: u64) {
        self.clock_sec = now_sec;
    }

    /// Mark any cached textures overlapping `[addr, addr + len)` as dirty so
    /// their next bind re-reads and uploads updated mip bytes.
    pub fn mark_dirty_range(&mut self, addr: u64, len: u64) {
        let end = addr.saturating_add(len);
        for entry in self.textures.values_mut() {
            let e_end = entry.base_address.saturating_add(entry.footprint as u64);
            if entry.base_address < end && addr < e_end {
                entry.dirty = true;
            }
        }
    }

    /// Evict idle GPU textures above the VRAM high-water mark, destroying their
    /// Vulkan `VkImage`s to reclaim VRAM immediately. Binds do this on their
    /// own whenever an upload takes the cache over budget.
    pub fn evict_idle(&mut self, runner: &ComputeRunner, now_sec: u64) -> usize {
        self.clock_sec = now_sec;
        let slots = self.evictor.sweep_at_most(now_sec, 8, |slot| {
            self.key_by_slot.get(&slot).and_then(|key| self.textures.get(key))
                .is_some_and(|entry| entry.pending.is_none())
        });
        let mut count = 0usize;
        for slot in slots {
            self.evictor.forget(slot);
            if let Some(k) = self.key_by_slot.remove(&slot)
                && let Some(entry) = self.textures.remove(&k)
            {
                self.telemetry.vram_textures_bytes = self
                    .telemetry
                    .vram_textures_bytes
                    .saturating_sub(entry.footprint as u64);
                let mut entry = entry;
                let _ = settle(runner, &mut entry);
                runner.destroy_image(entry.image);
                self.evicted_images += 1;
                count += 1;
            }
        }
        count
    }

    /// Reclaims completed staging and idle textures between frames.
    pub fn maintain_streaming(&mut self, runner: &ComputeRunner) {
        const GIB: u64 = 1 << 30;
        let heap = runner.texture_heap_budget();
        let commit = crate::residency::available_commit_bytes();
        let (usage, budget) = heap.unwrap_or_default();
        let (heap_pressure, heap_recovery) = (budget / 10 * 9, budget / 10 * 8);
        if self.pressure {
            self.pressure = (budget > 0 && usage >= heap_recovery)
                || commit.is_some_and(|bytes| bytes < 5 * GIB);
        } else {
            self.pressure = (budget > 0 && usage >= heap_pressure)
                || commit.is_some_and(|bytes| bytes < 3 * GIB);
        }
        self.staging_limit = if self.pressure { 16 << 20 } else { 64 << 20 };
        self.streaming = StreamingStats {
            heap_usage: usage,
            heap_budget: budget,
            commit_available: commit.unwrap_or_default(),
            resident_bytes: self.telemetry.vram_textures_bytes,
            oldest_upload_ms: self.pending_order.first().map_or(0, |(start, _)| start.elapsed().as_millis() as u64),
        };
        // Transfers use one FIFO queue. An unfinished oldest transfer also
        // means that newer uploads have yet to signal.
        for _ in 0..8 {
            let Some(&(_, key)) = self.pending_order.first() else { break };
            let upload = self.textures[&key].pending.as_ref().unwrap();
            if matches!(runner.upload_ready(upload), Ok(false)) {
                break;
            }
            if let Err(error) = self.settle_entry(runner, key) {
                crate::perf::binding_errors(1);
                eprintln!("texture upload: {error}");
            }
        }
        let resident = self.evictor.resident_bytes();
        let goal = if self.pressure {
            let reclaim = usage.saturating_sub(heap_recovery);
            self.configured_budget.min(resident.saturating_sub(reclaim.max(resident / 8)))
        } else {
            self.configured_budget
        };
        self.evictor.set_watermarks(goal, goal / 4 * 3);
        self.evict_idle(runner, self.epoch.elapsed().as_secs());
    }

    pub fn streaming_stats(&self) -> StreamingStats {
        self.streaming
    }

    /// The memory model of a T#, with `format` overriding its format (depth
    /// targets).
    pub fn model(
        words: &[u32; 8],
        depth: Option<vk::Format>,
    ) -> Result<SurfaceModel, TextureError> {
        let d = ImageDescriptor::decode(*words);
        let (format, bpe, block) = match depth {
            Some(f) => (f, crate::images::element_bytes(f), 1),
            None => texel_format(d.format)?,
        };
        // Host-built T#s say how the bytes are really stored (BGRA order,
        // sRGB encoding) in Strata's trailer.
        let flags = trailer_flags(words).unwrap_or_default();
        let format = rgba8_order(format, flags & TRAILER_BGRA != 0, flags & TRAILER_SRGB != 0);
        let first_level = u32::from(d.base_level);
        let levels = u32::from(d.last_level).saturating_sub(first_level) + 1;
        Ok(SurfaceModel {
            desc: d,
            format,
            bpe,
            block,
            shape: shape_of(&d)?,
            first_level,
            levels,
        })
    }

    #[cfg(test)]
    fn check(words: &[u32; 8]) -> Result<SurfaceModel, TextureError> {
        Self::model(words, None)
    }

    /// Resolve a color attachment at full fidelity, including its first upload.
    pub fn color_target(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 8],
        read: ReadFn<'_>,
    ) -> Result<&GpuImage, TextureError> {
        self.entry(runner, words, false, None, read)
    }

    /// The image for a T# (`depth`: as a depth/stencil target of that
    /// format), created or refreshed from title memory. `storage`: a
    /// shader writes it (the image gets `STORAGE` usage).
    fn entry(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 8],
        storage: bool,
        depth: Option<vk::Format>,
        read: ReadFn<'_>,
    ) -> Result<&GpuImage, TextureError> {
        self.entry_with(runner, words, storage, depth, read, false)
    }

    /// [`Self::entry`]; `allow_async`: a new sampled texture may start a
    /// transfer-queue upload (the caller shows a placeholder meanwhile).
    fn entry_with(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 8],
        storage: bool,
        depth: Option<vk::Format>,
        read: ReadFn<'_>,
        allow_async: bool,
    ) -> Result<&GpuImage, TextureError> {
        let k = key(&words, depth);
        self.clock_sec = self.epoch.elapsed().as_secs();
        self.apply_dirty_log();
        // Whoever reaches the image now needs the upload done.
        self.settle_entry(runner, k)?;

        // Fast path: a cached entry in write-watched memory that no logged
        // write touched is unchanged: skip reading and hashing title memory.
        if self.dirty.is_some()
            && let Some(e) = self.textures.get(&k)
            && e.watched
            && !e.dirty
            && (e.image.storage || !storage)
        {
            self.clean_skips += 1;
            self.evictor.on_gpu_read(e.slot, self.clock_sec);
            return Ok(&self.textures[&k].image);
        }

        let model = Self::model(&words, depth)?;
        let d = model.desc;
        let footprint = model.footprint()?;
        let src = read(d.base_address, footprint).ok_or(TextureError::Read {
            addr: d.base_address,
            len: footprint,
        })?;
        let hash = fnv(&src);

        let reuse = match self.textures.get(&k) {
            Some(e) => e.image.storage || !storage,
            None => false,
        };
        if !reuse && let Some(old) = self.textures.remove(&k) {
            // Needed as storage now: made again with STORAGE usage below.
            self.destroy_entry(runner, old);
        }
        if reuse {
            let need_upload = self.textures[&k].hash != hash;
            if need_upload {
                let mut e = self.textures.remove(&k).expect("present");
                let filled = self.fill(runner, &e.image, &model, &src);
                if let Err(error) = filled {
                    self.destroy_entry(runner, e);
                    return Err(error);
                }
                e.hash = hash;
                e.dirty = false;
                self.textures.insert(k, e);
            } else {
                let e = self.textures.get_mut(&k).expect("present");
                e.dirty = false;
            }
            let slot = self.textures[&k].slot;
            self.evictor.on_gpu_read(slot, self.clock_sec);
            return Ok(&self.textures[&k].image);
        }

        let need = match (depth, storage) {
            (Some(_), _) => vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            (None, true) => {
                vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::STORAGE_IMAGE
            }
            (None, false) => vk::FormatFeatureFlags::SAMPLED_IMAGE,
        };
        if !runner.format_supports(model.format, need) {
            return Err(TextureError::Unsupported {
                format: model.format,
                what: match (depth, storage) {
                    (Some(_), _) => "render depth to",
                    (None, true) => "store to",
                    (None, false) => "sample",
                },
            });
        }
        let components = if depth.is_some() {
            vk::ComponentMapping::default()
        } else {
            vk::ComponentMapping {
                r: swizzle(d.select[0]),
                g: swizzle(d.select[1]),
                b: swizzle(d.select[2]),
                a: swizzle(d.select[3]),
            }
        };
        let (w, h) = level_extent(d.width, d.height, model.first_level);
        let shape = match model.shape {
            ImageShape::D3 { depth } => ImageShape::D3 {
                depth: (depth >> model.first_level).max(1),
            },
            s => s,
        };
        let image = runner.create_image(&ImageSpec {
            format: model.format,
            width: w,
            height: h,
            shape,
            mips: model.levels,
            storage,
            components,
        })?;
        let mut pending = None;
        if allow_async
            && !storage
            && depth.is_none()
            && src.len() >= self.async_min
            && runner.has_transfer_queue()
            && self.pending_bytes.saturating_add(footprint as u64) <= self.staging_limit
        {
            match self
                .detile_or_store(&model, &src)
                .map(|tight| runner.upload_image_levels_async(&image, &tight, model.block))
            {
                Ok(Ok(Some(p))) => {
                    self.async_uploads += 1;
                    self.uploads += 1;
                    self.pending_bytes += p.bytes;
                    self.pending_order.insert((p.started, k));
                    pending = Some(p);
                }
                Ok(Ok(None)) | Ok(Err(_)) => {}
                Err(e) => {
                    runner.destroy_image(image);
                    return Err(e);
                }
            }
        }
        if pending.is_none()
            && let Err(e) = self.fill(runner, &image, &model, &src)
        {
            runner.destroy_image(image);
            return Err(e);
        }
        // Sampled textures can be evicted and re-read from title memory;
        // storage images and depth targets hold GPU-written contents.
        let slot = if storage || depth.is_some() {
            0
        } else {
            let slot = self.next_slot;
            self.next_slot = self.next_slot.wrapping_add(1).max(1);
            self.key_by_slot.insert(slot, k);
            self.evictor
                .on_write(slot, hash, footprint as u64, self.clock_sec);
            slot
        };
        self.telemetry.vram_textures_bytes += footprint as u64;
        self.textures.insert(
            k,
            Entry {
                image,
                hash,
                base_address: d.base_address,
                footprint,
                dirty: false,
                watched: crate::dirty_log::is_watched(d.base_address, footprint as u64),
                slot,
                pending,
            },
        );
        Ok(&self.textures[&k].image)
    }

    /// Uploads `src` (title-memory bytes) into `image`: from the texture
    /// store when it has the detiled levels, detiled on the GPU for large
    /// Morton surfaces, else detiled on the CPU (and stored).
    fn fill(
        &mut self,
        runner: &ComputeRunner,
        image: &GpuImage,
        model: &SurfaceModel,
        src: &[u8],
    ) -> Result<(), TextureError> {
        let t0 = std::time::Instant::now();
        if self.store.is_none()
            && src.len() >= self.gpu_detile_min
            && let Some((jobs, tight_len)) = model.gpu_detile_jobs()
        {
            if runner
                .upload_image_detiled_gpu(image, src, &jobs, tight_len, model.block)
                .is_ok()
            {
                self.gpu_detiles += 1;
                self.uploads += 1;
                self.telemetry
                    .upload_latency
                    .record_us(t0.elapsed().as_micros() as u64);
                return Ok(());
            }
            // Fall back to the CPU path below.
        }
        let tight = self.detile_or_store(model, src)?;
        self.telemetry
            .detile_latency
            .record_us(t0.elapsed().as_micros() as u64);
        let up_t0 = std::time::Instant::now();
        runner.upload_image_levels(image, &tight, model.block)?;
        self.telemetry
            .upload_latency
            .record_us(up_t0.elapsed().as_micros() as u64);
        self.uploads += 1;
        Ok(())
    }

    fn detile_or_store(
        &mut self,
        model: &SurfaceModel,
        src: &[u8],
    ) -> Result<Vec<u8>, TextureError> {
        let store_key = self.store.as_ref().map(|_| {
            crate::texstore::hash_texture_key(
                model.format.as_raw() as u32,
                model.desc.width,
                model.desc.height,
                src,
            )
        });
        if let (Some(store), Some(sk)) = (self.store.as_mut(), store_key)
            && let Some(cached) = store.get(sk)
        {
            self.store_hits += 1;
            return Ok(cached);
        }
        let tight = model.detile_all(src)?;
        if let (Some(store), Some(sk)) = (self.store.as_ref(), store_key) {
            let _ = store.put(sk, &tight);
        }
        Ok(tight)
    }

    /// The 2D image view for a T#, created or refreshed from title memory.
    /// `storage`: the shader writes it (the image gets `STORAGE` usage).
    pub fn texture(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 8],
        storage: bool,
        read: ReadFn<'_>,
    ) -> Result<vk::ImageView, TextureError> {
        self.texture_dim(runner, words, storage, ImageDim::D2, read)
    }

    /// The view of class `dim` for a T# (an instruction's class may differ
    /// from the T#'s own kind: a 2D read of an array reads layer 0).
    pub fn texture_dim(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 8],
        storage: bool,
        dim: ImageDim,
        read: ReadFn<'_>,
    ) -> Result<vk::ImageView, TextureError> {
        let k = key(&words, None);
        // An upload still on the transfer queue: show the placeholder.
        if !storage
            && let Some(e) = self.textures.get(&k)
            && let Some(p) = &e.pending
            && matches!(runner.upload_ready(p), Ok(false))
            && let Some(v) = runner
                .placeholder_image(e.image.shape)
                .and_then(|i| i.view_for(dim))
        {
            self.placeholder_binds += 1;
            return Ok(v);
        }
        self.entry_with(runner, words, storage, None, read, !storage)?;
        let e = &self.textures[&k];
        if e.pending.is_some() {
            match runner
                .placeholder_image(e.image.shape)
                .and_then(|i| i.view_for(dim))
            {
                Some(v) => {
                    self.placeholder_binds += 1;
                    return Ok(v);
                }
                None => self.settle_entry(runner, k)?,
            }
        }
        let img = &self.textures[&k].image;
        img.view_for(dim).ok_or(TextureError::View {
            shape: img.shape,
            dim,
        })
    }

    /// Finishes every transfer-queue upload still running (before reading
    /// images back or tearing down).
    pub fn settle_uploads(&mut self, runner: &ComputeRunner) -> Result<(), GpuError> {
        let keys: Vec<_> = self
            .textures
            .iter()
            .filter_map(|(key, entry)| entry.pending.is_some().then_some(*key))
            .collect();
        let mut first_error = None;
        for key in keys {
            if let Err(error) = self.settle_entry(runner, key) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn settle_entry(&mut self, runner: &ComputeRunner, k: ([u32; 5], i32)) -> Result<(), GpuError> {
        if let Some(upload) = self.textures.get(&k).and_then(|e| e.pending.as_ref()) {
            self.pending_order.remove(&(upload.started, k));
            self.pending_bytes -= upload.bytes;
        }
        if let Some(entry) = self.textures.get_mut(&k)
            && let Err(error) = settle(runner, entry)
        {
            let entry = self.textures.remove(&k).expect("entry is present");
            self.destroy_entry(runner, entry);
            return Err(error);
        }
        Ok(())
    }

    fn destroy_entry(&mut self, runner: &ComputeRunner, entry: Entry) {
        self.key_by_slot.remove(&entry.slot);
        self.evictor.forget(entry.slot);
        self.telemetry.vram_textures_bytes = self
            .telemetry
            .vram_textures_bytes
            .saturating_sub(entry.footprint as u64);
        runner.destroy_image(entry.image);
    }

    /// Uploads still running on the transfer queue.
    pub fn pending_uploads(&self) -> usize {
        self.pending_order.len()
    }

    /// A depth/stencil target of `format` over the memory a T# names.
    pub fn depth_target(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 8],
        format: vk::Format,
        read: ReadFn<'_>,
    ) -> Result<&GpuImage, TextureError> {
        self.entry(runner, words, false, Some(format), read)
    }

    /// The image made for a T# (after [`Self::texture`]).
    pub fn image(&self, words: &[u32; 8]) -> Option<&GpuImage> {
        self.textures
            .get(&key(words, None))
            .filter(|e| e.pending.is_none())
            .map(|e| &e.image)
    }

    /// The depth target made for a T# (after [`Self::depth_target`]).
    pub fn depth_image(&self, words: &[u32; 8], format: vk::Format) -> Option<&GpuImage> {
        self.textures
            .get(&key(words, Some(format)))
            .map(|e| &e.image)
    }

    /// Copies an image a shader or draw wrote back to the title memory its
    /// T# names (every level and slice it holds).
    pub fn write_back(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 8],
        write: WriteFn<'_>,
        read: ReadFn<'_>,
    ) -> Result<(), TextureError> {
        self.write_back_as(runner, words, None, write, read)
    }

    /// [`Self::write_back`] for a depth target of `format`.
    pub fn write_back_depth(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 8],
        format: vk::Format,
        write: WriteFn<'_>,
        read: ReadFn<'_>,
    ) -> Result<(), TextureError> {
        self.write_back_as(runner, words, Some(format), write, read)
    }

    fn write_back_as(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 8],
        depth: Option<vk::Format>,
        write: WriteFn<'_>,
        read: ReadFn<'_>,
    ) -> Result<(), TextureError> {
        let model = Self::model(&words, depth)?;
        let d = model.desc;
        let footprint = model.footprint()?;
        let k = key(&words, depth);
        self.settle_entry(runner, k)?;
        let Some(e) = self.textures.get_mut(&k) else {
            return Ok(());
        };
        // Depth/stencil images read back their depth aspect only; packed
        // depth-stencil formats are not written back (their stencil plane
        // has no linear layout to copy).
        if matches!(
            model.format,
            vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT
        ) {
            return Ok(());
        }
        let bytes = e.image.tight_bytes(model.bpe, model.block);
        let tight = runner.read_image_levels(&e.image, bytes, model.block)?;
        // Keep the bytes between rows (linear padding) as they were.
        let mut dst = read(d.base_address, footprint).unwrap_or_else(|| vec![0; footprint]);
        model.retile_all(&tight, &mut dst)?;
        if !write(d.base_address, &dst) {
            return Err(TextureError::Write(d.base_address));
        }
        e.hash = fnv(&dst); // what title memory now holds: no re-upload needed
        Ok(())
    }

    /// Fills the image a T# names with a clear value and writes it back to
    /// title memory (`ClearRenderTargetView` / `ClearUnorderedAccessView*`
    /// / `ClearDepthStencilView`).
    #[allow(clippy::too_many_arguments)]
    pub fn clear(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 8],
        depth: Option<vk::Format>,
        color: [u32; 4],
        depth_value: f32,
        stencil: u32,
        write: WriteFn<'_>,
        read: ReadFn<'_>,
    ) -> Result<(), TextureError> {
        let img = self.entry(runner, words, false, depth, read)?;
        runner.clear_image(img, color, depth_value, stencil)?;
        self.write_back_as(runner, words, depth, write, read)
    }

    /// The sampler for an S# (`compare`: for a depth-comparison sample).
    pub fn sampler(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 4],
    ) -> Result<vk::Sampler, TextureError> {
        self.sampler_for(runner, words, false)
    }

    /// [`Self::sampler`], as a comparison sampler when `compare`.
    pub fn sampler_for(
        &mut self,
        runner: &ComputeRunner,
        words: [u32; 4],
        compare: bool,
    ) -> Result<vk::Sampler, TextureError> {
        let mut desc = SamplerDescriptor::decode(words);
        if compare {
            desc.compare = Some(desc.depth_compare_func);
        }
        let k = (words, desc.compare);
        if let Some(&s) = self.samplers.get(&k) {
            return Ok(s);
        }
        let s = runner.create_sampler(&desc)?;
        self.samplers.insert(k, s);
        Ok(s)
    }

    pub fn destroy(&mut self, runner: &ComputeRunner) {
        for (_, mut e) in self.textures.drain() {
            let _ = settle(runner, &mut e);
            runner.destroy_image(e.image);
        }
        for (_, s) in self.samplers.drain() {
            runner.destroy_sampler(s);
        }
        self.pending_bytes = 0;
        self.pending_order.clear();
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
        for mode in [0u8, 9, 25, 26, 27] {
            let d = desc(12, 5, mode);
            let (tiling, footprint) = source_layout(&d, 4).unwrap();
            let tight: Vec<u8> = (0..12 * 5 * 4).map(|i| i as u8).collect();
            let mut mem = vec![0xEE; footprint];
            retile(&d, 4, tiling, &tight, &mut mem);
            assert_eq!(detile(&d, 4, tiling, &mem), tight, "mode {mode}");
        }
        // GFX10 BC codes; 36..=55 are the packed 10/11-bit formats, not BCn.
        assert!(is_block_compressed(vk_format_extended(121).unwrap()));
        assert_eq!(bc_format(109), Some(vk::Format::BC1_RGBA_UNORM_BLOCK));
        assert_eq!(
            vk_format_extended(43),
            Some(vk::Format::B10G11R11_UFLOAT_PACK32)
        );
        assert_eq!(
            vk_format_extended(50),
            Some(vk::Format::A2B10G10R10_UNORM_PACK32)
        );
        assert_eq!(vk_format_extended(200), None);
        assert!(is_depth_stencil(vk::Format::D32_SFLOAT));
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
    fn gpu_detile_jobs_reproduce_the_cpu_detiler() {
        // Morton 2D array: 2 layers, levels 0..=2, RGBA8.
        let mut d = desc(40, 24, 9);
        d.kind = ImageKind::Tex2DArray;
        d.depth = 2;
        d.last_level = 2;
        let m = TextureCache::model(&d.encode(), None).unwrap();
        let src: Vec<u8> = (0..m.footprint().unwrap())
            .map(|i| (i * 13 + 1) as u8)
            .collect();
        let cpu = m.detile_levels(&src).unwrap();
        let (jobs, tight_len) = m.gpu_detile_jobs().unwrap();
        assert_eq!(jobs.len(), 6);
        assert_eq!(tight_len, cpu.len());
        // What detile.slang mode 1 computes, per job.
        let mut gpu = vec![0u8; tight_len];
        for j in &jobs {
            let bpp = j.bpp as usize;
            for y in 0..j.height as usize {
                for x in 0..j.width as usize {
                    let t = j.src_offset as usize + morton_offset(x, y, j.width as usize, bpp);
                    let o = j.dst_offset as usize + (y * j.width as usize + x) * bpp;
                    gpu[o..o + bpp].copy_from_slice(&src[t..t + bpp]);
                }
            }
        }
        assert_eq!(gpu, cpu);
        // Linear and bank-XOR surfaces stay on the CPU path.
        assert!(
            TextureCache::model(&desc(16, 16, 0).encode(), None)
                .unwrap()
                .gpu_detile_jobs()
                .is_none()
        );
        assert!(
            TextureCache::model(&desc(16, 16, 25).encode(), None)
                .unwrap()
                .gpu_detile_jobs()
                .is_none()
        );
    }

    #[test]
    fn surface_model_lays_levels_then_slices() {
        // A 2D array of 3 layers, levels 0..=2, linear RGBA8: each slice's
        // rows pad to 256 bytes; levels follow one another.
        let mut d = desc(64, 16, 0);
        d.kind = ImageKind::Tex2DArray;
        d.depth = 3;
        d.last_level = 2;
        let m = TextureCache::model(&d.encode(), None).unwrap();
        assert_eq!(m.shape, ImageShape::D2 { layers: 3 });
        // Level 0: 256 B pitch x 16 rows x 3; level 1: 256 x 8 x 3; level 2: 256 x 4 x 3.
        assert_eq!(m.footprint().unwrap(), 3 * 256 * (16 + 8 + 4));
        let src: Vec<u8> = (0..m.footprint().unwrap()).map(|i| (i * 7) as u8).collect();
        let tight = m.detile_all(&src).unwrap();
        assert_eq!(tight.len(), 3 * 4 * (64 * 16 + 32 * 8 + 16 * 4));
        let mut back = src.clone();
        m.retile_all(&tight, &mut back).unwrap();
        assert_eq!(back, src);

        // A cube's depth counts cubes: 6 faces each.
        let mut c = desc(8, 8, 9);
        c.kind = ImageKind::Cube;
        c.depth = 1;
        let m = TextureCache::model(&c.encode(), None).unwrap();
        assert_eq!(m.shape, ImageShape::Cube { cubes: 1 });
        assert_eq!(m.footprint().unwrap(), 6 * 64 * 4);

        // BC1: 4x4 blocks of 8 bytes are the elements.
        let mut b = desc(16, 8, 0);
        b.format = 109;
        let m = TextureCache::model(&b.encode(), None).unwrap();
        assert_eq!((m.bpe, m.block), (8, 4));
        assert_eq!(m.footprint().unwrap(), 256 * 2);
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
