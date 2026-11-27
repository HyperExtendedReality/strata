//! The GFX10 unified data-format enumeration (RDNA2 `IMG_FORMAT` /
//! `BUF_FMT`), as the public RDNA2 ISA documentation lists it, for the
//! formats whose components are whole bits of 8, 16 or 32.
//!
//! Packed formats (`10_11_11`, `11_11_10`, `10_10_10_2`, `2_10_10_10`),
//! block-compressed and sRGB formats are not in the table: callers get
//! `None` and must refuse them explicitly rather than read garbage.

/// How a component's bits are read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumFormat {
    /// `x / (2^n - 1)`.
    Unorm,
    /// `max(x / (2^(n-1) - 1), -1)`.
    Snorm,
    /// Unsigned integer converted to float, not normalised.
    Uscaled,
    /// Signed integer converted to float, not normalised.
    Sscaled,
    Uint,
    Sint,
    Float,
}

impl NumFormat {
    /// Whether shaders see floats (`true`) or integers.
    pub fn is_float(self) -> bool {
        !matches!(self, NumFormat::Uint | NumFormat::Sint)
    }
}

/// A format of `comps` components of `bits` each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FormatInfo {
    pub code: u16,
    pub comps: u8,
    pub bits: u8,
    pub num: NumFormat,
}

impl FormatInfo {
    pub fn bytes_per_element(&self) -> u32 {
        u32::from(self.comps) * u32::from(self.bits) / 8
    }
}

const NUMS6: [NumFormat; 6] = [
    NumFormat::Unorm,
    NumFormat::Snorm,
    NumFormat::Uscaled,
    NumFormat::Sscaled,
    NumFormat::Uint,
    NumFormat::Sint,
];
const NUMS7: [NumFormat; 7] = [
    NumFormat::Unorm,
    NumFormat::Snorm,
    NumFormat::Uscaled,
    NumFormat::Sscaled,
    NumFormat::Uint,
    NumFormat::Sint,
    NumFormat::Float,
];
const NUMS_INT_FLOAT: [NumFormat; 3] = [NumFormat::Uint, NumFormat::Sint, NumFormat::Float];

/// `(first code, components, bits, number formats in order)`.
const RUNS: &[(u16, u8, u8, &[NumFormat])] = &[
    (1, 1, 8, &NUMS6),            // 8
    (7, 1, 16, &NUMS7),           // 16
    (14, 2, 8, &NUMS6),           // 8_8
    (20, 1, 32, &NUMS_INT_FLOAT), // 32
    (23, 2, 16, &NUMS7),          // 16_16
    // 30..=55: packed 10/11-bit formats, not supported.
    (56, 4, 8, &NUMS6),           // 8_8_8_8
    (62, 2, 32, &NUMS_INT_FLOAT), // 32_32
    (65, 4, 16, &NUMS7),          // 16_16_16_16
    (72, 3, 32, &NUMS_INT_FLOAT), // 32_32_32
    (75, 4, 32, &NUMS_INT_FLOAT), // 32_32_32_32
];

/// The format `code` names, if it is one Strata reads.
pub fn gfx10_format(code: u16) -> Option<FormatInfo> {
    RUNS.iter().find_map(|&(first, comps, bits, nums)| {
        let i = code.checked_sub(first)? as usize;
        nums.get(i).map(|&num| FormatInfo {
            code,
            comps,
            bits,
            num,
        })
    })
}

/// Packed 32-bit GFX10 data formats (`10_11_11`, `11_11_10`, `10_10_10_2`, `2_10_10_10`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedFormat {
    F10_11_11,
    F11_11_10,
    Unorm10_10_10_2,
    Uint10_10_10_2,
    Unorm2_10_10_10,
    Snorm2_10_10_10,
    Uint2_10_10_10,
    Sint2_10_10_10,
}

/// Decode GFX10 packed 10/11-bit format codes (`30..=55`).
pub fn gfx10_packed_format(code: u16) -> Option<PackedFormat> {
    match code {
        36 => Some(PackedFormat::F10_11_11),
        43 => Some(PackedFormat::F11_11_10),
        44 => Some(PackedFormat::Unorm10_10_10_2),
        48 => Some(PackedFormat::Uint10_10_10_2),
        50 => Some(PackedFormat::Unorm2_10_10_10),
        51 => Some(PackedFormat::Snorm2_10_10_10),
        54 => Some(PackedFormat::Uint2_10_10_10),
        55 => Some(PackedFormat::Sint2_10_10_10),
        _ => None,
    }
}

/// Pack 4 normalized floats `[r, g, b, a]` into `R10G10B10A2_UNORM` (`2_10_10_10_UNORM`).
pub fn pack_unorm_2_10_10_10(rgba: [f32; 4]) -> u32 {
    let r = (rgba[0].clamp(0.0, 1.0) * 1023.0).round() as u32 & 0x3ff;
    let g = (rgba[1].clamp(0.0, 1.0) * 1023.0).round() as u32 & 0x3ff;
    let b = (rgba[2].clamp(0.0, 1.0) * 1023.0).round() as u32 & 0x3ff;
    let a = (rgba[3].clamp(0.0, 1.0) * 3.0).round() as u32 & 0x3;
    r | (g << 10) | (b << 20) | (a << 30)
}

/// Unpack `R10G10B10A2_UNORM` (`2_10_10_10_UNORM`) into `[r, g, b, a]`.
pub fn unpack_unorm_2_10_10_10(word: u32) -> [f32; 4] {
    [
        (word & 0x3ff) as f32 / 1023.0,
        ((word >> 10) & 0x3ff) as f32 / 1023.0,
        ((word >> 20) & 0x3ff) as f32 / 1023.0,
        ((word >> 30) & 0x3) as f32 / 3.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_codes_decode() {
        let rgba8 = gfx10_format(56).unwrap();
        assert_eq!(
            (rgba8.comps, rgba8.bits, rgba8.num),
            (4, 8, NumFormat::Unorm)
        );
        assert_eq!(rgba8.bytes_per_element(), 4);
        assert_eq!(gfx10_format(22).unwrap().num, NumFormat::Float); // 32_FLOAT
        assert_eq!(
            gfx10_format(13).unwrap(),
            FormatInfo {
                code: 13,
                comps: 1,
                bits: 16,
                num: NumFormat::Float
            }
        );
        assert_eq!(gfx10_format(71).unwrap().bytes_per_element(), 8); // 16_16_16_16_FLOAT
        assert_eq!(gfx10_format(77).unwrap().bytes_per_element(), 16); // 32_32_32_32_FLOAT
        assert_eq!(gfx10_format(60).unwrap().num, NumFormat::Uint); // 8_8_8_8_UINT
    }

    #[test]
    fn packed_and_unknown_codes_are_refused() {
        assert!(gfx10_format(0).is_none());
        assert!(gfx10_format(36).is_none()); // 10_11_11_FLOAT
        assert!(gfx10_format(50).is_none()); // 2_10_10_10_UNORM
        assert!(gfx10_format(200).is_none());
    }
}
