//! Precompiled Slang runtime shaders (SPIR-V 1.6 with PhysicalStorageBuffer).

pub const DIRTY_HASH_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dirty_hash.spv"));
pub const MIRROR_DIFF_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mirror_diff.spv"));
pub const DETILE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/detile.spv"));
pub const BVH_ORACLE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bvh_oracle.spv"));
pub const RAY_PROBE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ray_probe.spv"));
/// BMA Mapping stage (3x3 tile convolution over sampled BVH paths).
pub const BMA_MAPPING_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bma_mapping.spv"));
pub const CLAMP_INDIRECT_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/clamp_indirect.spv"));
pub const TEX_CONVERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tex_convert.spv"));
pub const MIP_GEN_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mip_gen.spv"));
pub const MSAA_RESOLVE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/msaa_resolve.spv"));
pub const CHUNK_DIRTY_HASH_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/chunk_dirty_hash.spv"));
pub const SCATTER_UPLOAD_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/scatter_upload.spv"));
pub const RECT_EXPAND_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rect_expand.spv"));
pub const AMBIENT_CLAMP_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ambient_clamp.spv"));
pub const OVERLAY_HUD_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/overlay_hud.spv"));
pub const BCPACK_DECODE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bcpack_decode.spv"));
pub const COMPOSE_VS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/compose_vs.spv"));
pub const COMPOSE_FS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/compose_fs.spv"));
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
        if std::env::var_os("STRATA_REQUIRE_SPIRV_VAL").is_some() {
            let path = std::env::temp_dir()
                .join(format!("strata-slang-{}-{name}.spv", std::process::id()));
            std::fs::write(&path, bytes).unwrap();
            // ComputeRunner enables scalarBlockLayout for Slang's natural BDA layout.
            let result = std::process::Command::new("spirv-val")
                .args(["--target-env", "vulkan1.3", "--scalar-block-layout"])
                .arg(&path)
                .output();
            let _ = std::fs::remove_file(&path);
            let result = result.expect("spirv-val is required");
            assert!(
                result.status.success(),
                "{name}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }

    #[test]
    fn all_slang_shaders_compiled_to_spirv() {
        assert_valid_spv_header("dirty_hash", DIRTY_HASH_SPV);
        assert_valid_spv_header("mirror_diff", MIRROR_DIFF_SPV);
        assert_valid_spv_header("detile", DETILE_SPV);
        assert_valid_spv_header("bvh_oracle", BVH_ORACLE_SPV);
        assert_valid_spv_header("ray_probe", RAY_PROBE_SPV);
        assert_valid_spv_header("bma_mapping", BMA_MAPPING_SPV);
        assert_valid_spv_header("clamp_indirect", CLAMP_INDIRECT_SPV);
        assert_valid_spv_header("tex_convert", TEX_CONVERT_SPV);
        assert_valid_spv_header("mip_gen", MIP_GEN_SPV);
        assert_valid_spv_header("msaa_resolve", MSAA_RESOLVE_SPV);
        assert_valid_spv_header("chunk_dirty_hash", CHUNK_DIRTY_HASH_SPV);
        assert_valid_spv_header("scatter_upload", SCATTER_UPLOAD_SPV);
        assert_valid_spv_header("rect_expand", RECT_EXPAND_SPV);
        assert_valid_spv_header("ambient_clamp", AMBIENT_CLAMP_SPV);
        assert_valid_spv_header("overlay_hud", OVERLAY_HUD_SPV);
        assert_valid_spv_header("bcpack_decode", BCPACK_DECODE_SPV);
        assert_valid_spv_header("compose_vs", COMPOSE_VS_SPV);
        assert_valid_spv_header("compose_fs", COMPOSE_FS_SPV);
        assert_valid_spv_header("present_blit_vs", PRESENT_BLIT_VS_SPV);
        assert_valid_spv_header("present_blit_fs", PRESENT_BLIT_FS_SPV);
    }
}
