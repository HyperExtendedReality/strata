//! Clean-room console data format decoders and layout tables.
//!
//! Pure byte-in / byte-out modules with zero external media or SDK dependencies:
//! - [`texture`]: Full DXGI/GFX10 texture-format table and RDNA2 tiled layout math.
//! - [`bcpack`]: Hardware-compatible BCPack (rANS + BC1/BC3/BC4/BC5/BC6H/BC7) decoder.
//! - [`xma`]: XMA / XMA2 packet parser and frame-level audio decoder.

pub mod bcpack;
pub mod opus;
pub mod opus_native;
pub mod sgd5;
pub mod texture;
pub mod wmapro;
mod wmapro_tables;
pub mod xma;
