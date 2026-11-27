//! Comprehensive DXGI / GFX10 texture-format table and RDNA2 (GFX10.3) tiled
//! surface addressing (1D/2D/3D, linear, 256B, 4KB, and 64KB swizzle modes,
//! mip chains, mip tails, and CPU detile/retile).

use std::sync::OnceLock;
use thiserror::Error;

/// Numeric interpretation of a texture format's channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelKind {
    Unorm,
    Snorm,
    Srgb,
    Uint,
    Sint,
    Float,
    DepthStencil,
    Typeless,
}

/// Complete metadata for a DXGI / console texture format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureFormatInfo {
    pub dxgi: u32,
    pub name: &'static str,
    /// Bytes per texel (uncompressed) or per 4x4 block (BC formats).
    pub element_bytes: u32,
    /// Block width/height in texels (1 for uncompressed, 4 for BC1..BC7).
    pub block_dim: u32,
    pub channels: u8,
    pub kind: ChannelKind,
    /// Corresponding GFX10 unified `IMG_FORMAT` code, if directly representable.
    pub gfx10_code: Option<u16>,
}

impl TextureFormatInfo {
    pub const fn is_compressed(&self) -> bool {
        self.block_dim > 1
    }

    pub const fn is_srgb(&self) -> bool {
        matches!(self.kind, ChannelKind::Srgb)
    }

    pub const fn is_depth_stencil(&self) -> bool {
        matches!(self.kind, ChannelKind::DepthStencil)
    }

    /// BC format number (1..=7) if this is a block-compressed format.
    pub const fn bc_number(&self) -> Option<u8> {
        match self.dxgi {
            70..=72 => Some(1),
            73..=75 => Some(2),
            76..=78 => Some(3),
            79..=81 => Some(4),
            82..=84 => Some(5),
            94..=96 => Some(6),
            97..=99 => Some(7),
            _ => None,
        }
    }
}

/// Full DXGI texture-format table covering uncompressed, packed, depth/stencil,
/// sRGB, and BC1-BC7 formats used by D3D12.X titles.
pub const FORMAT_TABLE: &[TextureFormatInfo] = &[
    // 128-bit formats
    TextureFormatInfo {
        dxgi: 1,
        name: "R32G32B32A32_TYPELESS",
        element_bytes: 16,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(77),
    },
    TextureFormatInfo {
        dxgi: 2,
        name: "R32G32B32A32_FLOAT",
        element_bytes: 16,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Float,
        gfx10_code: Some(77),
    },
    TextureFormatInfo {
        dxgi: 3,
        name: "R32G32B32A32_UINT",
        element_bytes: 16,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Uint,
        gfx10_code: Some(75),
    },
    TextureFormatInfo {
        dxgi: 4,
        name: "R32G32B32A32_SINT",
        element_bytes: 16,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Sint,
        gfx10_code: Some(76),
    },
    // 96-bit formats
    TextureFormatInfo {
        dxgi: 5,
        name: "R32G32B32_TYPELESS",
        element_bytes: 12,
        block_dim: 1,
        channels: 3,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(74),
    },
    TextureFormatInfo {
        dxgi: 6,
        name: "R32G32B32_FLOAT",
        element_bytes: 12,
        block_dim: 1,
        channels: 3,
        kind: ChannelKind::Float,
        gfx10_code: Some(74),
    },
    TextureFormatInfo {
        dxgi: 7,
        name: "R32G32B32_UINT",
        element_bytes: 12,
        block_dim: 1,
        channels: 3,
        kind: ChannelKind::Uint,
        gfx10_code: Some(72),
    },
    TextureFormatInfo {
        dxgi: 8,
        name: "R32G32B32_SINT",
        element_bytes: 12,
        block_dim: 1,
        channels: 3,
        kind: ChannelKind::Sint,
        gfx10_code: Some(73),
    },
    // 64-bit formats
    TextureFormatInfo {
        dxgi: 9,
        name: "R16G16B16A16_TYPELESS",
        element_bytes: 8,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(71),
    },
    TextureFormatInfo {
        dxgi: 10,
        name: "R16G16B16A16_FLOAT",
        element_bytes: 8,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Float,
        gfx10_code: Some(71),
    },
    TextureFormatInfo {
        dxgi: 11,
        name: "R16G16B16A16_UNORM",
        element_bytes: 8,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(65),
    },
    TextureFormatInfo {
        dxgi: 12,
        name: "R16G16B16A16_UINT",
        element_bytes: 8,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Uint,
        gfx10_code: Some(69),
    },
    TextureFormatInfo {
        dxgi: 13,
        name: "R16G16B16A16_SNORM",
        element_bytes: 8,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Snorm,
        gfx10_code: Some(66),
    },
    TextureFormatInfo {
        dxgi: 14,
        name: "R16G16B16A16_SINT",
        element_bytes: 8,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Sint,
        gfx10_code: Some(70),
    },
    TextureFormatInfo {
        dxgi: 15,
        name: "R32G32_TYPELESS",
        element_bytes: 8,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(64),
    },
    TextureFormatInfo {
        dxgi: 16,
        name: "R32G32_FLOAT",
        element_bytes: 8,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Float,
        gfx10_code: Some(64),
    },
    TextureFormatInfo {
        dxgi: 17,
        name: "R32G32_UINT",
        element_bytes: 8,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Uint,
        gfx10_code: Some(62),
    },
    TextureFormatInfo {
        dxgi: 18,
        name: "R32G32_SINT",
        element_bytes: 8,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Sint,
        gfx10_code: Some(63),
    },
    TextureFormatInfo {
        dxgi: 19,
        name: "R32G8X24_TYPELESS",
        element_bytes: 8,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Typeless,
        gfx10_code: None,
    },
    TextureFormatInfo {
        dxgi: 20,
        name: "D32_FLOAT_S8X24_UINT",
        element_bytes: 8,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::DepthStencil,
        gfx10_code: None,
    },
    // 32-bit packed & RGBA8
    TextureFormatInfo {
        dxgi: 23,
        name: "R10G10B10A2_TYPELESS",
        element_bytes: 4,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(50),
    },
    TextureFormatInfo {
        dxgi: 24,
        name: "R10G10B10A2_UNORM",
        element_bytes: 4,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(50),
    },
    TextureFormatInfo {
        dxgi: 25,
        name: "R10G10B10A2_UINT",
        element_bytes: 4,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Uint,
        gfx10_code: Some(54),
    },
    TextureFormatInfo {
        dxgi: 26,
        name: "R11G11B10_FLOAT",
        element_bytes: 4,
        block_dim: 1,
        channels: 3,
        kind: ChannelKind::Float,
        gfx10_code: Some(36),
    },
    TextureFormatInfo {
        dxgi: 27,
        name: "R8G8B8A8_TYPELESS",
        element_bytes: 4,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(56),
    },
    TextureFormatInfo {
        dxgi: 28,
        name: "R8G8B8A8_UNORM",
        element_bytes: 4,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(56),
    },
    TextureFormatInfo {
        dxgi: 29,
        name: "R8G8B8A8_UNORM_SRGB",
        element_bytes: 4,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Srgb,
        gfx10_code: Some(56),
    },
    TextureFormatInfo {
        dxgi: 30,
        name: "R8G8B8A8_UINT",
        element_bytes: 4,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Uint,
        gfx10_code: Some(60),
    },
    TextureFormatInfo {
        dxgi: 31,
        name: "R8G8B8A8_SNORM",
        element_bytes: 4,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Snorm,
        gfx10_code: Some(57),
    },
    TextureFormatInfo {
        dxgi: 32,
        name: "R8G8B8A8_SINT",
        element_bytes: 4,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Sint,
        gfx10_code: Some(61),
    },
    TextureFormatInfo {
        dxgi: 33,
        name: "R16G16_TYPELESS",
        element_bytes: 4,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(29),
    },
    TextureFormatInfo {
        dxgi: 34,
        name: "R16G16_FLOAT",
        element_bytes: 4,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Float,
        gfx10_code: Some(29),
    },
    TextureFormatInfo {
        dxgi: 35,
        name: "R16G16_UNORM",
        element_bytes: 4,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(23),
    },
    TextureFormatInfo {
        dxgi: 36,
        name: "R16G16_UINT",
        element_bytes: 4,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Uint,
        gfx10_code: Some(27),
    },
    TextureFormatInfo {
        dxgi: 37,
        name: "R16G16_SNORM",
        element_bytes: 4,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Snorm,
        gfx10_code: Some(24),
    },
    TextureFormatInfo {
        dxgi: 38,
        name: "R16G16_SINT",
        element_bytes: 4,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Sint,
        gfx10_code: Some(28),
    },
    TextureFormatInfo {
        dxgi: 39,
        name: "R32_TYPELESS",
        element_bytes: 4,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(22),
    },
    TextureFormatInfo {
        dxgi: 40,
        name: "D32_FLOAT",
        element_bytes: 4,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::DepthStencil,
        gfx10_code: Some(22),
    },
    TextureFormatInfo {
        dxgi: 41,
        name: "R32_FLOAT",
        element_bytes: 4,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Float,
        gfx10_code: Some(22),
    },
    TextureFormatInfo {
        dxgi: 42,
        name: "R32_UINT",
        element_bytes: 4,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Uint,
        gfx10_code: Some(20),
    },
    TextureFormatInfo {
        dxgi: 43,
        name: "R32_SINT",
        element_bytes: 4,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Sint,
        gfx10_code: Some(21),
    },
    TextureFormatInfo {
        dxgi: 44,
        name: "R24G8_TYPELESS",
        element_bytes: 4,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Typeless,
        gfx10_code: None,
    },
    TextureFormatInfo {
        dxgi: 45,
        name: "D24_UNORM_S8_UINT",
        element_bytes: 4,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::DepthStencil,
        gfx10_code: None,
    },
    TextureFormatInfo {
        dxgi: 67,
        name: "R9G9B9E5_SHAREDEXP",
        element_bytes: 4,
        block_dim: 1,
        channels: 3,
        kind: ChannelKind::Float,
        gfx10_code: None,
    },
    // 16-bit formats
    TextureFormatInfo {
        dxgi: 48,
        name: "R8G8_TYPELESS",
        element_bytes: 2,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(14),
    },
    TextureFormatInfo {
        dxgi: 49,
        name: "R8G8_UNORM",
        element_bytes: 2,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(14),
    },
    TextureFormatInfo {
        dxgi: 50,
        name: "R8G8_UINT",
        element_bytes: 2,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Uint,
        gfx10_code: Some(18),
    },
    TextureFormatInfo {
        dxgi: 51,
        name: "R8G8_SNORM",
        element_bytes: 2,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Snorm,
        gfx10_code: Some(15),
    },
    TextureFormatInfo {
        dxgi: 52,
        name: "R8G8_SINT",
        element_bytes: 2,
        block_dim: 1,
        channels: 2,
        kind: ChannelKind::Sint,
        gfx10_code: Some(19),
    },
    TextureFormatInfo {
        dxgi: 53,
        name: "R16_TYPELESS",
        element_bytes: 2,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(13),
    },
    TextureFormatInfo {
        dxgi: 54,
        name: "R16_FLOAT",
        element_bytes: 2,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Float,
        gfx10_code: Some(13),
    },
    TextureFormatInfo {
        dxgi: 55,
        name: "D16_UNORM",
        element_bytes: 2,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::DepthStencil,
        gfx10_code: Some(7),
    },
    TextureFormatInfo {
        dxgi: 56,
        name: "R16_UNORM",
        element_bytes: 2,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(7),
    },
    TextureFormatInfo {
        dxgi: 57,
        name: "R16_UINT",
        element_bytes: 2,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Uint,
        gfx10_code: Some(11),
    },
    TextureFormatInfo {
        dxgi: 58,
        name: "R16_SNORM",
        element_bytes: 2,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Snorm,
        gfx10_code: Some(8),
    },
    TextureFormatInfo {
        dxgi: 59,
        name: "R16_SINT",
        element_bytes: 2,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Sint,
        gfx10_code: Some(12),
    },
    TextureFormatInfo {
        dxgi: 85,
        name: "B5G6R5_UNORM",
        element_bytes: 2,
        block_dim: 1,
        channels: 3,
        kind: ChannelKind::Unorm,
        gfx10_code: None,
    },
    TextureFormatInfo {
        dxgi: 86,
        name: "B5G5R5A1_UNORM",
        element_bytes: 2,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Unorm,
        gfx10_code: None,
    },
    // 8-bit formats
    TextureFormatInfo {
        dxgi: 60,
        name: "R8_TYPELESS",
        element_bytes: 1,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(1),
    },
    TextureFormatInfo {
        dxgi: 61,
        name: "R8_UNORM",
        element_bytes: 1,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(1),
    },
    TextureFormatInfo {
        dxgi: 62,
        name: "R8_UINT",
        element_bytes: 1,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Uint,
        gfx10_code: Some(5),
    },
    TextureFormatInfo {
        dxgi: 63,
        name: "R8_SNORM",
        element_bytes: 1,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Snorm,
        gfx10_code: Some(2),
    },
    TextureFormatInfo {
        dxgi: 64,
        name: "R8_SINT",
        element_bytes: 1,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Sint,
        gfx10_code: Some(6),
    },
    TextureFormatInfo {
        dxgi: 65,
        name: "A8_UNORM",
        element_bytes: 1,
        block_dim: 1,
        channels: 1,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(1),
    },
    // BGRA8
    TextureFormatInfo {
        dxgi: 87,
        name: "B8G8R8A8_UNORM",
        element_bytes: 4,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(56),
    },
    TextureFormatInfo {
        dxgi: 88,
        name: "B8G8R8X8_UNORM",
        element_bytes: 4,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(56),
    },
    TextureFormatInfo {
        dxgi: 90,
        name: "B8G8R8A8_TYPELESS",
        element_bytes: 4,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(56),
    },
    TextureFormatInfo {
        dxgi: 91,
        name: "B8G8R8A8_UNORM_SRGB",
        element_bytes: 4,
        block_dim: 1,
        channels: 4,
        kind: ChannelKind::Srgb,
        gfx10_code: Some(56),
    },
    // Block-compressed BC1..BC7
    TextureFormatInfo {
        dxgi: 70,
        name: "BC1_TYPELESS",
        element_bytes: 8,
        block_dim: 4,
        channels: 4,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(80),
    },
    TextureFormatInfo {
        dxgi: 71,
        name: "BC1_UNORM",
        element_bytes: 8,
        block_dim: 4,
        channels: 4,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(80),
    },
    TextureFormatInfo {
        dxgi: 72,
        name: "BC1_UNORM_SRGB",
        element_bytes: 8,
        block_dim: 4,
        channels: 4,
        kind: ChannelKind::Srgb,
        gfx10_code: Some(81),
    },
    TextureFormatInfo {
        dxgi: 73,
        name: "BC2_TYPELESS",
        element_bytes: 16,
        block_dim: 4,
        channels: 4,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(82),
    },
    TextureFormatInfo {
        dxgi: 74,
        name: "BC2_UNORM",
        element_bytes: 16,
        block_dim: 4,
        channels: 4,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(82),
    },
    TextureFormatInfo {
        dxgi: 75,
        name: "BC2_UNORM_SRGB",
        element_bytes: 16,
        block_dim: 4,
        channels: 4,
        kind: ChannelKind::Srgb,
        gfx10_code: Some(83),
    },
    TextureFormatInfo {
        dxgi: 76,
        name: "BC3_TYPELESS",
        element_bytes: 16,
        block_dim: 4,
        channels: 4,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(84),
    },
    TextureFormatInfo {
        dxgi: 77,
        name: "BC3_UNORM",
        element_bytes: 16,
        block_dim: 4,
        channels: 4,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(84),
    },
    TextureFormatInfo {
        dxgi: 78,
        name: "BC3_UNORM_SRGB",
        element_bytes: 16,
        block_dim: 4,
        channels: 4,
        kind: ChannelKind::Srgb,
        gfx10_code: Some(85),
    },
    TextureFormatInfo {
        dxgi: 79,
        name: "BC4_TYPELESS",
        element_bytes: 8,
        block_dim: 4,
        channels: 1,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(86),
    },
    TextureFormatInfo {
        dxgi: 80,
        name: "BC4_UNORM",
        element_bytes: 8,
        block_dim: 4,
        channels: 1,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(86),
    },
    TextureFormatInfo {
        dxgi: 81,
        name: "BC4_SNORM",
        element_bytes: 8,
        block_dim: 4,
        channels: 1,
        kind: ChannelKind::Snorm,
        gfx10_code: Some(87),
    },
    TextureFormatInfo {
        dxgi: 82,
        name: "BC5_TYPELESS",
        element_bytes: 16,
        block_dim: 4,
        channels: 2,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(88),
    },
    TextureFormatInfo {
        dxgi: 83,
        name: "BC5_UNORM",
        element_bytes: 16,
        block_dim: 4,
        channels: 2,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(88),
    },
    TextureFormatInfo {
        dxgi: 84,
        name: "BC5_SNORM",
        element_bytes: 16,
        block_dim: 4,
        channels: 2,
        kind: ChannelKind::Snorm,
        gfx10_code: Some(89),
    },
    TextureFormatInfo {
        dxgi: 94,
        name: "BC6H_TYPELESS",
        element_bytes: 16,
        block_dim: 4,
        channels: 3,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(90),
    },
    TextureFormatInfo {
        dxgi: 95,
        name: "BC6H_UF16",
        element_bytes: 16,
        block_dim: 4,
        channels: 3,
        kind: ChannelKind::Float,
        gfx10_code: Some(90),
    },
    TextureFormatInfo {
        dxgi: 96,
        name: "BC6H_SF16",
        element_bytes: 16,
        block_dim: 4,
        channels: 3,
        kind: ChannelKind::Float,
        gfx10_code: Some(91),
    },
    TextureFormatInfo {
        dxgi: 97,
        name: "BC7_TYPELESS",
        element_bytes: 16,
        block_dim: 4,
        channels: 4,
        kind: ChannelKind::Typeless,
        gfx10_code: Some(92),
    },
    TextureFormatInfo {
        dxgi: 98,
        name: "BC7_UNORM",
        element_bytes: 16,
        block_dim: 4,
        channels: 4,
        kind: ChannelKind::Unorm,
        gfx10_code: Some(92),
    },
    TextureFormatInfo {
        dxgi: 99,
        name: "BC7_UNORM_SRGB",
        element_bytes: 16,
        block_dim: 4,
        channels: 4,
        kind: ChannelKind::Srgb,
        gfx10_code: Some(93),
    },
];

/// Look up a texture format by DXGI format number.
pub fn lookup_dxgi(dxgi: u32) -> Option<&'static TextureFormatInfo> {
    FORMAT_TABLE.iter().find(|f| f.dxgi == dxgi)
}

/// Look up a texture format by GFX10 unified `IMG_FORMAT` code.
pub fn lookup_gfx10(code: u16) -> Option<&'static TextureFormatInfo> {
    FORMAT_TABLE.iter().find(|f| f.gfx10_code == Some(code))
}

// ============================================================================
// RDNA2 (GFX10.3) Swizzle Modes & Tiled Surface Addressing
// ============================================================================

/// RDNA2 hardware swizzle mode (`ADDR_SW_*` numbering).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Swizzle(pub u32);

impl Swizzle {
    pub const LINEAR: Swizzle = Swizzle(0);
    pub const SW_256B_S: Swizzle = Swizzle(1);
    pub const SW_256B_D: Swizzle = Swizzle(2);
    pub const SW_4KB_S: Swizzle = Swizzle(5);
    pub const SW_4KB_D: Swizzle = Swizzle(6);
    pub const SW_64KB_S: Swizzle = Swizzle(9);
    pub const SW_64KB_D: Swizzle = Swizzle(10);
    pub const SW_64KB_S_T: Swizzle = Swizzle(17);
    pub const SW_64KB_D_T: Swizzle = Swizzle(18);
    pub const SW_4KB_S_X: Swizzle = Swizzle(21);
    pub const SW_4KB_D_X: Swizzle = Swizzle(22);
    pub const SW_4KB_R_X: Swizzle = Swizzle(23);
    pub const SW_64KB_Z_X: Swizzle = Swizzle(24);
    pub const SW_64KB_S_X: Swizzle = Swizzle(25);
    pub const SW_64KB_D_X: Swizzle = Swizzle(26);
    pub const SW_64KB_R_X: Swizzle = Swizzle(27);

    pub const fn block_log2(self) -> Option<u32> {
        Some(match self.0 {
            0 => 0,
            1 | 2 => 8,
            5 | 6 | 21..=23 => 12,
            9 | 10 | 17 | 18 | 24..=27 => 16,
            _ => return None,
        })
    }

    pub const fn is_display(self) -> bool {
        matches!(self.0, 2 | 6 | 10 | 18 | 22 | 26)
    }

    pub const fn is_xor(self) -> bool {
        self.0 >= 16
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    Tex1d,
    Tex2d,
    Tex3d,
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum LayoutError {
    #[error("invalid texture layout description")]
    Invalid,
    #[error("unsupported swizzle mode or dimension")]
    Unsupported,
    #[error("buffer too small for surface copy")]
    TooSmall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceDesc {
    pub dimension: Dimension,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mips: u32,
    pub element_bytes: u32,
    pub block_dim: u32,
    pub samples: u32,
    pub swizzle: Swizzle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MipLayout {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub pitch: u32,
    pub rows: u32,
    pub offset: u64,
    pub size_bytes: u64,
    pub in_tail: bool,
    pub tail_offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceLayout {
    pub desc: SurfaceDesc,
    pub block: [u32; 3],
    pub thick: bool,
    pub slices: u32,
    pub slice_size: u64,
    pub total_size: u64,
    pub alignment: u64,
    pub first_tail_mip: u32,
    pub mips: Vec<MipLayout>,
    pattern: &'static SwizzlePattern,
}

type SwizzlePattern = [Vec<u32>; 3];
static EMPTY_PATTERN: SwizzlePattern = [const { Vec::new() }; 3];

/// 256-byte micro-tile bit patterns (thin S, thin D, thick S/D) for element_log2 in 0..=4.
const MICRO_PATTERNS: [[&[u8; 16]; 5]; 3] = [
    [
        b"x0x1x2x3y0y1y2y3",
        b"--x0x1x2y0y1y2x3",
        b"----x0x1y0y1y2x2",
        b"------x0y0y1x1x2",
        b"--------y0y1x0x1",
    ],
    [
        b"x0x1x2y1y0y2x3y3",
        b"--x0x1x2y0y1y2x3",
        b"----x0x1y0y1x2y2",
        b"------x0y0x1x2y1",
        b"--------x0y0x1y1",
    ],
    [
        b"x0x1z0y0z1y1x2z2",
        b"--x0z0y0z1y1x1z2",
        b"----x0y0z0y1x1z1",
        b"------x0z0y0x1z1",
        b"--------z0y0x0z1",
    ],
];

const BLOCK_256_2D: [[u32; 2]; 5] = [[16, 16], [16, 8], [8, 8], [8, 4], [4, 4]];
const BLOCK_256_3D: [[u32; 3]; 5] = [[8, 4, 8], [4, 4, 8], [4, 4, 4], [4, 2, 4], [2, 2, 4]];

fn build_pattern(kind: usize, element_log2: u32, block_log2: u32) -> SwizzlePattern {
    let mut axes: [Vec<u32>; 3] = Default::default();
    let micro = MICRO_PATTERNS[kind][element_log2 as usize];
    for bit in 0..8u32 {
        let axis = match micro[2 * bit as usize] {
            b'x' => 0,
            b'y' => 1,
            b'z' => 2,
            _ => continue,
        };
        let idx = usize::from(micro[2 * bit as usize + 1] - b'0');
        if axes[axis].len() <= idx {
            axes[axis].resize(idx + 1, 0);
        }
        axes[axis][idx] = bit;
    }
    let order: &[usize] = if kind == 2 { &[1, 0, 2] } else { &[1, 0] };
    for (bit, &axis) in (8..block_log2).zip(order.iter().cycle()) {
        axes[axis].push(bit);
    }
    axes.map(|bits| {
        (0..1u32 << bits.len())
            .map(|c| {
                bits.iter()
                    .enumerate()
                    .map(|(i, &to)| ((c >> i) & 1) << to)
                    .sum()
            })
            .collect()
    })
}

fn shared_pattern(kind: usize, element_log2: u32, block_log2: u32) -> &'static SwizzlePattern {
    static PATTERNS: [[[OnceLock<SwizzlePattern>; 5]; 3]; 3] =
        [const { [const { [const { OnceLock::new() }; 5] }; 3] }; 3];
    let block_idx = ((block_log2 - 8) / 4) as usize;
    PATTERNS[kind][block_idx][element_log2 as usize]
        .get_or_init(|| build_pattern(kind, element_log2, block_log2))
}

impl SurfaceLayout {
    pub fn compute(desc: &SurfaceDesc) -> Result<Self, LayoutError> {
        let d = *desc;
        let block_log2 = d.swizzle.block_log2().ok_or(LayoutError::Invalid)?;
        if d.width == 0
            || d.height == 0
            || d.depth == 0
            || d.mips == 0
            || d.mips > 16
            || d.width > 16384
            || d.height > 16384
            || d.depth > 16384
            || !d.element_bytes.is_power_of_two()
            || d.element_bytes > 16
            || !matches!(d.block_dim, 1 | 4)
            || !d.samples.is_power_of_two()
            || d.samples > 8
        {
            return Err(LayoutError::Invalid);
        }

        let elem_log2 = d.element_bytes.trailing_zeros();
        let thick = d.dimension == Dimension::Tex3d && d.swizzle != Swizzle::LINEAR;

        if d.swizzle == Swizzle::LINEAR {
            let pitch_align = 256u32 / d.element_bytes;
            let mut mips = Vec::with_capacity(d.mips as usize);
            let mut offset = 0u64;
            for mip in 0..d.mips {
                let w = (d.width >> mip).max(1).div_ceil(d.block_dim);
                let h = (d.height >> mip).max(1).div_ceil(d.block_dim);
                let z = if d.dimension == Dimension::Tex3d {
                    (d.depth >> mip).max(1)
                } else {
                    1
                };
                let pitch = w.next_multiple_of(pitch_align.max(1));
                let size_bytes =
                    u64::from(pitch) * u64::from(h) * u64::from(z) * u64::from(d.element_bytes);
                mips.push(MipLayout {
                    width: w,
                    height: h,
                    depth: z,
                    pitch,
                    rows: h,
                    offset,
                    size_bytes,
                    in_tail: false,
                    tail_offset: 0,
                });
                offset += size_bytes.next_multiple_of(256);
            }
            let slices = if d.dimension == Dimension::Tex3d {
                1
            } else {
                d.depth
            };
            let slice_size = offset.max(256);
            return Ok(Self {
                desc: d,
                block: [pitch_align.max(1), 1, 1],
                thick: false,
                slices,
                slice_size,
                total_size: slice_size * u64::from(slices),
                alignment: 256,
                first_tail_mip: d.mips,
                mips,
                pattern: &EMPTY_PATTERN,
            });
        }

        let block_bytes = 1u64 << block_log2;
        let kind = if thick {
            2
        } else if d.swizzle.is_display() {
            1
        } else {
            0
        };
        let pattern = shared_pattern(kind, elem_log2, block_log2);

        let block = if thick {
            let mut b = BLOCK_256_3D[elem_log2 as usize];
            let extra = block_log2 - 8;
            let order = [1usize, 0, 2];
            for i in 0..extra as usize {
                b[order[i % 3]] *= 2;
            }
            b
        } else {
            let [mut bx, mut by] = BLOCK_256_2D[elem_log2 as usize];
            let extra = block_log2 - 8;
            for i in 0..extra {
                if i % 2 == 0 {
                    by *= 2;
                } else {
                    bx *= 2;
                }
            }
            [bx, by, 1]
        };

        let mut mips = Vec::with_capacity(d.mips as usize);
        let mut offset = 0u64;
        let mut first_tail_mip = d.mips;
        let mut tail_cursor = 0u32;

        for mip in 0..d.mips {
            let w = (d.width >> mip).max(1).div_ceil(d.block_dim);
            let h = (d.height >> mip).max(1).div_ceil(d.block_dim);
            let z = if d.dimension == Dimension::Tex3d {
                (d.depth >> mip).max(1)
            } else {
                1
            };
            let in_tail = block_log2 >= 12
                && mip > 0
                && w <= block[0] / 2
                && h <= block[1] / 2
                && z <= block[2];
            if in_tail {
                if first_tail_mip == d.mips {
                    first_tail_mip = mip;
                }
                let sub_bytes =
                    (u64::from(w) * u64::from(h) * u64::from(z) * u64::from(d.element_bytes))
                        .next_multiple_of(256);
                let tail_off = tail_cursor;
                tail_cursor = tail_cursor.saturating_add(sub_bytes as u32);
                mips.push(MipLayout {
                    width: w,
                    height: h,
                    depth: z,
                    pitch: w.next_multiple_of(1),
                    rows: h,
                    offset: 0,
                    size_bytes: sub_bytes,
                    in_tail: true,
                    tail_offset: tail_off,
                });
            } else {
                let bx = w.div_ceil(block[0]);
                let by = h.div_ceil(block[1]);
                let bz = z.div_ceil(block[2]);
                let size_bytes = u64::from(bx) * u64::from(by) * u64::from(bz) * block_bytes;
                mips.push(MipLayout {
                    width: w,
                    height: h,
                    depth: z,
                    pitch: bx * block[0],
                    rows: by * block[1],
                    offset,
                    size_bytes,
                    in_tail: false,
                    tail_offset: 0,
                });
                offset += size_bytes;
            }
        }
        if first_tail_mip < d.mips {
            let tail_block_offset = offset;
            for m in &mut mips[first_tail_mip as usize..] {
                m.offset = tail_block_offset;
            }
            offset += block_bytes;
        }

        let slices = if d.dimension == Dimension::Tex3d {
            1
        } else {
            d.depth
        };
        let slice_size = offset.max(block_bytes);
        Ok(Self {
            desc: d,
            block,
            thick,
            slices,
            slice_size,
            total_size: slice_size * u64::from(slices),
            alignment: block_bytes,
            first_tail_mip,
            mips,
            pattern,
        })
    }

    /// Compute the byte offset of element `(x, y, z)` in `(mip, slice)`.
    pub fn element_offset(&self, mip: u32, slice: u32, x: u32, y: u32, z: u32) -> Option<u64> {
        let m = self.mips.get(mip as usize)?;
        if x >= m.width || y >= m.height || z >= m.depth || slice >= self.slices {
            return None;
        }
        let slice_base = u64::from(slice) * self.slice_size;
        if self.desc.swizzle == Swizzle::LINEAR {
            let row_bytes = u64::from(m.pitch) * u64::from(self.desc.element_bytes);
            let plane_bytes = row_bytes * u64::from(m.rows);
            return Some(
                slice_base
                    + m.offset
                    + u64::from(z) * plane_bytes
                    + u64::from(y) * row_bytes
                    + u64::from(x) * u64::from(self.desc.element_bytes),
            );
        }
        if m.in_tail {
            let linear_elem = (u64::from(z) * u64::from(m.height) + u64::from(y))
                * u64::from(m.width)
                + u64::from(x);
            return Some(
                slice_base
                    + m.offset
                    + u64::from(m.tail_offset)
                    + linear_elem * u64::from(self.desc.element_bytes),
            );
        }
        let bx = x / self.block[0];
        let by = y / self.block[1];
        let bz = z / self.block[2];
        let lx = (x % self.block[0]) as usize;
        let ly = (y % self.block[1]) as usize;
        let lz = (z % self.block[2]) as usize;
        let blocks_x = m.pitch / self.block[0];
        let blocks_y = m.rows / self.block[1];
        let block_index = (u64::from(bz) * u64::from(blocks_y) + u64::from(by))
            * u64::from(blocks_x)
            + u64::from(bx);
        let mut in_block = self.pattern[0].get(lx).copied().unwrap_or(0)
            | self.pattern[1].get(ly).copied().unwrap_or(0)
            | self.pattern[2].get(lz).copied().unwrap_or(0);
        if self.desc.swizzle.is_xor() && self.alignment >= 4096 {
            // Apply RDNA2 pipe-bank XOR across macro-tile coordinates within the block.
            let pipe_bank_xor = ((bx ^ by ^ bz) & 3) << 8;
            if u64::from(in_block ^ pipe_bank_xor) < self.alignment {
                in_block ^= pipe_bank_xor;
            }
        }
        Some(slice_base + m.offset + block_index * self.alignment + u64::from(in_block))
    }

    /// Coalesced byte-copy run mapping tiled source offsets to tightly packed linear offsets,
    /// used by GPU detiling (`detile.slang`) and staging upload planners.
    pub fn copy_table(&self, mip: u32, slice: u32) -> Result<Vec<(u64, u64, u32)>, LayoutError> {
        let m = self.mips.get(mip as usize).ok_or(LayoutError::Invalid)?;
        let bpp = u64::from(self.desc.element_bytes);
        let mut runs: Vec<(u64, u64, u32)> = Vec::new();
        for z in 0..m.depth {
            for y in 0..m.height {
                for x in 0..m.width {
                    let src = self
                        .element_offset(mip, slice, x, y, z)
                        .ok_or(LayoutError::Invalid)?;
                    let dst = u64::from((z * m.height + y) * m.width + x) * bpp;
                    if let Some(last) = runs.last_mut() {
                        if last.0 + u64::from(last.2) == src
                            && last.1 + u64::from(last.2) == dst
                            && last.2.checked_add(bpp as u32).is_some()
                        {
                            last.2 += bpp as u32;
                            continue;
                        }
                    }
                    runs.push((src, dst, bpp as u32));
                }
            }
        }
        Ok(runs)
    }

    /// Detile a single `(mip, slice)` from tiled surface bytes into tightly packed rows.
    pub fn detile_mip(
        &self,
        mip: u32,
        slice: u32,
        tiled: &[u8],
        linear: &mut [u8],
    ) -> Result<(), LayoutError> {
        let m = self.mips.get(mip as usize).ok_or(LayoutError::Invalid)?;
        let bpp = self.desc.element_bytes as usize;
        let need = (m.width as usize) * (m.height as usize) * (m.depth as usize) * bpp;
        if linear.len() < need {
            return Err(LayoutError::TooSmall);
        }
        for z in 0..m.depth {
            for y in 0..m.height {
                for x in 0..m.width {
                    let src_off = self
                        .element_offset(mip, slice, x, y, z)
                        .ok_or(LayoutError::Invalid)? as usize;
                    if src_off + bpp > tiled.len() {
                        return Err(LayoutError::TooSmall);
                    }
                    let dst_off = (((z * m.height + y) * m.width + x) as usize) * bpp;
                    linear[dst_off..dst_off + bpp].copy_from_slice(&tiled[src_off..src_off + bpp]);
                }
            }
        }
        Ok(())
    }

    /// Retile a single `(mip, slice)` from tightly packed rows into tiled surface bytes.
    pub fn retile_mip(
        &self,
        mip: u32,
        slice: u32,
        linear: &[u8],
        tiled: &mut [u8],
    ) -> Result<(), LayoutError> {
        let m = self.mips.get(mip as usize).ok_or(LayoutError::Invalid)?;
        let bpp = self.desc.element_bytes as usize;
        let need = (m.width as usize) * (m.height as usize) * (m.depth as usize) * bpp;
        if linear.len() < need {
            return Err(LayoutError::TooSmall);
        }
        for z in 0..m.depth {
            for y in 0..m.height {
                for x in 0..m.width {
                    let dst_off = self
                        .element_offset(mip, slice, x, y, z)
                        .ok_or(LayoutError::Invalid)? as usize;
                    if dst_off + bpp > tiled.len() {
                        return Err(LayoutError::TooSmall);
                    }
                    let src_off = (((z * m.height + y) * m.width + x) as usize) * bpp;
                    tiled[dst_off..dst_off + bpp].copy_from_slice(&linear[src_off..src_off + bpp]);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_table_covers_bc_and_uncompressed() {
        let bc7 = lookup_dxgi(98).unwrap();
        assert_eq!(bc7.name, "BC7_UNORM");
        assert_eq!(bc7.element_bytes, 16);
        assert_eq!(bc7.block_dim, 4);
        assert_eq!(bc7.bc_number(), Some(7));

        let rgba8 = lookup_dxgi(28).unwrap();
        assert_eq!(rgba8.gfx10_code, Some(56));
        assert_eq!(lookup_gfx10(56).unwrap().element_bytes, 4);
    }

    #[test]
    fn tiled_64kb_s_roundtrips_detile_and_retile() {
        let desc = SurfaceDesc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 64,
            depth: 1,
            mips: 3,
            element_bytes: 4,
            block_dim: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        };
        let layout = SurfaceLayout::compute(&desc).unwrap();
        assert_eq!(layout.alignment, 65536);
        let mut tiled = vec![0u8; layout.total_size as usize];
        let src: Vec<u8> = (0..(64 * 64 * 4)).map(|i| (i * 13 + 7) as u8).collect();
        layout.retile_mip(0, 0, &src, &mut tiled).unwrap();
        let mut dst = vec![0u8; src.len()];
        layout.detile_mip(0, 0, &tiled, &mut dst).unwrap();
        assert_eq!(src, dst);
    }
}
