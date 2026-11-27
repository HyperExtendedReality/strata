//! Precompiled Slang runtime shaders (SPIR-V 1.6 with PhysicalStorageBuffer).

pub const DIRTY_HASH_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dirty_hash.spv"));
pub const DETILE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/detile.spv"));
pub const BVH_ORACLE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bvh_oracle.spv"));
pub const CLAMP_INDIRECT_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/clamp_indirect.spv"));
pub const TEX_CONVERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tex_convert.spv"));
pub const MIP_GEN_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mip_gen.spv"));
pub const PRESENT_BLIT_VS_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/present_blit_vs.spv"));
pub const PRESENT_BLIT_FS_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/present_blit_fs.spv"));

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_valid_spv_header(name: &str, bytes: &[u8]) {
        assert!(
            bytes.len() >= 20 && bytes.len().is_multiple_of(4),
            "{name} invalid length"
        );
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(magic, 0x0723_0203, "{name} bad SPIR-V magic");
    }

    #[test]
    fn all_slang_shaders_compiled_to_spirv() {
        assert_valid_spv_header("dirty_hash", DIRTY_HASH_SPV);
        assert_valid_spv_header("detile", DETILE_SPV);
        assert_valid_spv_header("bvh_oracle", BVH_ORACLE_SPV);
        assert_valid_spv_header("clamp_indirect", CLAMP_INDIRECT_SPV);
        assert_valid_spv_header("tex_convert", TEX_CONVERT_SPV);
        assert_valid_spv_header("mip_gen", MIP_GEN_SPV);
        assert_valid_spv_header("present_blit_vs", PRESENT_BLIT_VS_SPV);
        assert_valid_spv_header("present_blit_fs", PRESENT_BLIT_FS_SPV);
    }
}
