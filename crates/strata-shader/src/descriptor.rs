//! RDNA2 (GFX10) resource descriptors, decoded from their dwords.
//!
//! - T# (image, 8 dwords): address, format, size, type, swizzle mode,
//!   component selects, mip range.
//! - S# (sampler, 4 dwords): addressing modes, filters, LOD range and bias.
//!
//! Field positions follow the public RDNA2 ISA documentation
//! (`SQ_IMG_RSRC_WORD0..4`, `SQ_IMG_SAMP_WORD0..2`).

/// What a component of a sampled texel is taken from (`DST_SEL_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Select {
    Zero,
    One,
    X,
    Y,
    Z,
    W,
}

impl Select {
    fn from_bits(v: u32) -> Self {
        match v & 7 {
            1 => Select::One,
            4 => Select::X,
            5 => Select::Y,
            6 => Select::Z,
            7 => Select::W,
            _ => Select::Zero,
        }
    }
    fn bits(self) -> u32 {
        match self {
            Select::Zero => 0,
            Select::One => 1,
            Select::X => 4,
            Select::Y => 5,
            Select::Z => 6,
            Select::W => 7,
        }
    }
}

/// `SQ_RSRC_IMG_*` resource type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageKind {
    Tex1D,
    Tex2D,
    Tex3D,
    Cube,
    Tex1DArray,
    Tex2DArray,
    Tex2DMsaa,
    Tex2DMsaaArray,
    Other(u8),
}

impl ImageKind {
    fn from_bits(v: u32) -> Self {
        match v & 0xf {
            8 => ImageKind::Tex1D,
            9 => ImageKind::Tex2D,
            10 => ImageKind::Tex3D,
            11 => ImageKind::Cube,
            12 => ImageKind::Tex1DArray,
            13 => ImageKind::Tex2DArray,
            14 => ImageKind::Tex2DMsaa,
            15 => ImageKind::Tex2DMsaaArray,
            o => ImageKind::Other(o as u8),
        }
    }
    fn bits(self) -> u32 {
        match self {
            ImageKind::Tex1D => 8,
            ImageKind::Tex2D => 9,
            ImageKind::Tex3D => 10,
            ImageKind::Cube => 11,
            ImageKind::Tex1DArray => 12,
            ImageKind::Tex2DArray => 13,
            ImageKind::Tex2DMsaa => 14,
            ImageKind::Tex2DMsaaArray => 15,
            ImageKind::Other(o) => u32::from(o),
        }
    }
}

/// A decoded T#.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageDescriptor {
    /// Byte address of level 0 (256-byte aligned).
    pub base_address: u64,
    /// GFX10 unified format code (see [`crate::format`]).
    pub format: u16,
    pub width: u32,
    pub height: u32,
    /// Depth (3D) or array layers.
    pub depth: u32,
    pub kind: ImageKind,
    /// `SW_MODE`: 0 is linear.
    pub sw_mode: u8,
    pub select: [Select; 4],
    pub base_level: u8,
    pub last_level: u8,
    pub base_array: u32,
}

impl ImageDescriptor {
    pub fn decode(w: [u32; 8]) -> Self {
        let base = (u64::from(w[0]) | (u64::from(w[1] & 0xff) << 32)) << 8;
        let width = (((w[1] >> 30) & 3) | ((w[2] & 0xfff) << 2)) + 1;
        Self {
            base_address: base,
            format: ((w[1] >> 20) & 0x1ff) as u16,
            width,
            height: ((w[2] >> 14) & 0x3fff) + 1,
            depth: (w[4] & 0x1fff) + 1,
            kind: ImageKind::from_bits(w[3] >> 28),
            sw_mode: ((w[3] >> 20) & 0x1f) as u8,
            select: [
                Select::from_bits(w[3]),
                Select::from_bits(w[3] >> 3),
                Select::from_bits(w[3] >> 6),
                Select::from_bits(w[3] >> 9),
            ],
            base_level: ((w[3] >> 12) & 0xf) as u8,
            last_level: ((w[3] >> 16) & 0xf) as u8,
            base_array: (w[4] >> 16) & 0x1fff,
        }
    }

    /// The dwords [`Self::decode`] reads back as `self` (for tests and for
    /// building descriptors on the host side).
    pub fn encode(&self) -> [u32; 8] {
        let addr = self.base_address >> 8;
        let w = self.width - 1;
        let mut out = [0u32; 8];
        out[0] = addr as u32;
        out[1] =
            ((addr >> 32) as u32 & 0xff) | (u32::from(self.format & 0x1ff) << 20) | ((w & 3) << 30);
        out[2] = ((w >> 2) & 0xfff) | (((self.height - 1) & 0x3fff) << 14);
        out[3] = self.select[0].bits()
            | (self.select[1].bits() << 3)
            | (self.select[2].bits() << 6)
            | (self.select[3].bits() << 9)
            | (u32::from(self.base_level & 0xf) << 12)
            | (u32::from(self.last_level & 0xf) << 16)
            | (u32::from(self.sw_mode & 0x1f) << 20)
            | (self.kind.bits() << 28);
        out[4] = ((self.depth - 1) & 0x1fff) | ((self.base_array & 0x1fff) << 16);
        out
    }
}

/// `SQ_TEX_*` addressing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Clamp {
    Wrap,
    Mirror,
    ClampToEdge,
    MirrorOnceToEdge,
    ClampToBorder,
    MirrorOnceToBorder,
}

impl Clamp {
    fn from_bits(v: u32) -> Self {
        match v & 7 {
            0 => Clamp::Wrap,
            1 => Clamp::Mirror,
            2 => Clamp::ClampToEdge,
            3 => Clamp::MirrorOnceToEdge,
            5 | 7 => Clamp::MirrorOnceToBorder,
            _ => Clamp::ClampToBorder, // 4 (half border) and 6 (border)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Filter {
    Point,
    Linear,
}

/// A decoded S#.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerDescriptor {
    pub clamp: [Clamp; 3],
    /// `MAX_ANISO_RATIO`: 1, 2, 4, 8 or 16 samples.
    pub max_aniso: u8,
    pub mag: Filter,
    pub min: Filter,
    /// `None`: no mipmapping (base level only).
    pub mip: Option<Filter>,
    pub min_lod: f32,
    pub max_lod: f32,
    pub lod_bias: f32,
}

impl SamplerDescriptor {
    pub fn decode(w: [u32; 4]) -> Self {
        let filter = |v: u32| {
            if v & 1 != 0 {
                Filter::Linear
            } else {
                Filter::Point
            }
        };
        let u4_8 = |v: u32| (v & 0xfff) as f32 / 256.0;
        let bias = {
            let raw = (w[2] & 0x3fff) as i32;
            let signed = (raw << 18) >> 18; // s5.8
            signed as f32 / 256.0
        };
        Self {
            clamp: [
                Clamp::from_bits(w[0]),
                Clamp::from_bits(w[0] >> 3),
                Clamp::from_bits(w[0] >> 6),
            ],
            max_aniso: 1 << ((w[0] >> 9) & 7).min(4),
            mag: filter(w[2] >> 20),
            min: filter(w[2] >> 22),
            mip: match (w[2] >> 26) & 3 {
                0 => None,
                1 => Some(Filter::Point),
                _ => Some(Filter::Linear),
            },
            min_lod: u4_8(w[1]),
            max_lod: u4_8(w[1] >> 12),
            lod_bias: bias,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_descriptor_round_trips() {
        let d = ImageDescriptor {
            base_address: 0x7ff6_1234_5600,
            format: 56,
            width: 1920,
            height: 1080,
            depth: 1,
            kind: ImageKind::Tex2D,
            sw_mode: 0,
            select: [Select::X, Select::Y, Select::Z, Select::One],
            base_level: 0,
            last_level: 3,
            base_array: 0,
        };
        assert_eq!(ImageDescriptor::decode(d.encode()), d);
        // Width straddles words 1 and 2.
        let w = d.encode();
        assert_eq!(w[1] >> 30, (1920 - 1) & 3);
        assert_eq!(w[2] & 0xfff, (1920 - 1) >> 2);
    }

    #[test]
    fn sampler_descriptor_fields() {
        // clamp x=wrap(0) y=clamp(2) z=mirror(1); aniso 4x (2);
        // min_lod 0, max_lod 13.0; bias -1.5; mag linear, min point, mip linear.
        let w0 = (2 << 3) | (1 << 6) | (2 << 9);
        let w1 = (13 * 256) << 12;
        let bias = ((-384i32) as u32) & 0x3fff;
        let w2 = bias | (1 << 20) | (2 << 26); // min filter point: 0 at bit 22
        let s = SamplerDescriptor::decode([w0, w1, w2, 0]);
        assert_eq!(s.clamp, [Clamp::Wrap, Clamp::ClampToEdge, Clamp::Mirror]);
        assert_eq!(s.max_aniso, 4);
        assert_eq!(
            (s.mag, s.min, s.mip),
            (Filter::Linear, Filter::Point, Some(Filter::Linear))
        );
        assert_eq!((s.min_lod, s.max_lod, s.lod_bias), (0.0, 13.0, -1.5));
    }
}
