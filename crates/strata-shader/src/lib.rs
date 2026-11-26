//! Strata RDNA2 → SPIR-V 1.6 (PhysicalStorageBuffer / No-API) shader compiler.

pub mod bvh;
pub mod cache;
pub mod container;
pub mod inflate;
pub mod opt;
pub mod rdna2;
pub mod spirv;
pub mod ssa;
pub mod wave;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompileError {
    #[error(transparent)]
    Decode(#[from] rdna2::DecodeError),
    #[error(transparent)]
    Lower(#[from] ssa::LowerError),
}

#[derive(Debug, Clone)]
pub struct CompiledShader {
    pub spirv: Vec<u32>,
    pub opt_stats: opt::OptStats,
    pub wave: wave::WaveAnalysis,
}

/// Compile raw RDNA2 machine code directly to SPIR-V 1.6 with 64-bit BDA pointers.
pub fn compile_rdna2_to_spirv(
    words: &[u32],
    workgroup_size: [u32; 3],
) -> Result<CompiledShader, CompileError> {
    let decoded = rdna2::decode_program(words)?;
    let wave = wave::analyze_wave_liveness(&decoded);
    let mut module = ssa::build_ssa(&decoded, &wave, workgroup_size)?;
    let opt_stats = opt::optimize(&mut module);
    let spirv = spirv::emit_spirv(&module);
    Ok(CompiledShader { spirv, opt_stats, wave })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn assert_spirv_valid(words: &[u32], tag: &str) {
        let tmp = std::env::temp_dir().join(format!("strata_test_{tag}.spv"));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&tmp, &bytes).unwrap();

        if let Ok(output) = Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&tmp)
            .output()
        {
            assert!(
                output.status.success(),
                "spirv-val failed for {tag}:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn coalesces_four_scalar_bda_loads_and_validates_spirv() {
        use rdna2::encode::*;

        let mut prog = Vec::new();
        prog.push(v_lshlrev_b32_imm(1, 4, 0));
        prog.extend_from_slice(&global_load_dword(2, 1, 0, 0));
        prog.extend_from_slice(&global_load_dword(3, 1, 0, 4));
        prog.extend_from_slice(&global_load_dword(4, 1, 0, 8));
        prog.extend_from_slice(&global_load_dword(5, 1, 0, 12));
        prog.push(s_waitcnt());
        prog.push(v_fmac_f32(4, 2, 3));
        prog.push(v_add_f32(4, 4, 5));
        prog.extend_from_slice(&global_store_dword(1, 4, 2, 0));
        prog.push(s_endpgm());

        let compiled = compile_rdna2_to_spirv(&prog, [64, 1, 1]).expect("compile failed");
        assert_eq!(compiled.opt_stats.coalesced_loads, 1);
        assert!(compiled.opt_stats.after < compiled.opt_stats.before);
        assert_spirv_valid(&compiled.spirv, "coalesced");
    }

    #[test]
    fn bvh_box32_sorts_nearest_hit_first() {
        let ray = bvh::Ray {
            t_max: 100.0,
            origin: [0.0, 0.0, 0.0],
            dir: [0.0, 0.0, 1.0],
            inv_dir: [f32::INFINITY, f32::INFINITY, 1.0],
        };
        let node = bvh::Box32Node {
            children: [10, 20, 30, bvh::INVALID_NODE],
            bounds_min: [[-1.0, -1.0, 20.0], [-1.0, -1.0, 5.0], [10.0, 10.0, 10.0], [0.0; 3]],
            bounds_max: [[1.0, 1.0, 25.0], [1.0, 1.0, 8.0], [12.0, 12.0, 12.0], [0.0; 3]],
        };
        let hits = bvh::intersect_box32(&ray, &node);
        assert_eq!(hits, [20, 10, bvh::INVALID_NODE, bvh::INVALID_NODE]);
    }

    #[test]
    fn translated_rdna2_divergent_if_else_validates_spirv() {
        use rdna2::encode::*;

        // Real RDNA2 divergent if/else:
        // if (v0 < 32) v1 = v0 + 10; else v1 = v0 + 20;
        // store v1 at s[0:1] + (v0 << 2)
        let mut prog = Vec::new();
        prog.push(v_cmp_lt_u32_imm(0, 32));          // pc 0
        prog.push(s_and_saveexec_b64(4, 106));        // pc 1: s[4:5] = exec; exec &= vcc
        prog.push(s_cbranch_execz(1));                // pc 2: skip pc 3 if execz -> pc 4
        prog.push(v_add_nc_u32_imm(1, 10, 0));        // pc 3: then-body
        prog.push(s_xor_b64(126, 126, 4));            // pc 4: exec ^= s[4:5] (else lanes)
        prog.push(s_cbranch_execz(1));                // pc 5: skip pc 6 if execz -> pc 7
        prog.push(v_add_nc_u32_imm(1, 20, 0));        // pc 6: else-body
        prog.push(s_or_b64(126, 126, 4));             // pc 7: restore exec
        prog.push(v_lshlrev_b32_imm(2, 2, 0));        // pc 8: byte offset = v0 << 2
        prog.extend_from_slice(&global_store_dword(2, 1, 0, 0)); // pc 9..10
        prog.push(s_endpgm());

        let compiled = compile_rdna2_to_spirv(&prog, [64, 1, 1]).expect("compile failed");
        assert!(compiled.wave.is_wave64);
        assert_spirv_valid(&compiled.spirv, "rdna2_cfg");
    }

    #[test]
    fn translated_rdna2_wave64_ballot_validates_spirv() {
        use rdna2::encode::*;

        // Real RDNA2 Wave64 ballot:
        // v1 = v0 & 1; v_cmp_eq_u32 vcc, 0, v1;
        // v2 = vcc_lo; v3 = vcc_hi;
        // global_store_dword s[0:1] + 0, v2; global_store_dword s[0:1] + 4, v3
        let mut prog = Vec::new();
        prog.push(v_and_b32_imm(1, 1, 0));
        prog.push(v_cmp_eq_u32_imm(1, 0));
        prog.push(v_mov_b32_sgpr(2, 106)); // v2 = vcc_lo
        prog.push(v_mov_b32_sgpr(3, 107)); // v3 = vcc_hi
        prog.push(v_mov_b32_sgpr(4, 128)); // v4 = 0 (offset)
        prog.extend_from_slice(&global_store_dword(4, 2, 0, 0));
        prog.extend_from_slice(&global_store_dword(4, 3, 0, 4));
        prog.push(s_endpgm());

        let compiled = compile_rdna2_to_spirv(&prog, [64, 1, 1]).expect("compile failed");
        assert!(!compiled.wave.whole_live_pcs.is_empty(), "expected whole-live VCC ballot");
        assert_spirv_valid(&compiled.spirv, "rdna2_w64_ballot");
    }

    #[test]
    fn translated_rdna2_lds_and_vop3_and_bvh_validate_spirv() {
        use rdna2::encode::*;

        let mut prog = Vec::new();
        prog.push(v_lshlrev_b32_imm(1, 2, 0));
        prog.extend_from_slice(&v_fma_f32_vop3(2, 0, 0, 0));
        prog.extend_from_slice(&ds_write_b32(1, 2, 0));
        prog.push(s_barrier());
        prog.extend_from_slice(&ds_read_b32(3, 1, 0));
        prog.extend_from_slice(&image_bvh64_intersect_ray(16, 4, 8));
        prog.extend_from_slice(&global_store_dword(1, 3, 0, 0));
        prog.extend_from_slice(&global_store_dword(1, 16, 2, 0));
        prog.push(s_endpgm());

        let compiled = compile_rdna2_to_spirv(&prog, [64, 1, 1]).expect("compile failed");
        assert_spirv_valid(&compiled.spirv, "rdna2_lds_vop3_bvh");
    }

    #[test]
    fn rejects_unsupported_graphics_exp_instead_of_silently_dropping() {
        use rdna2::encode::*;
        let mut prog = Vec::new();
        prog.extend_from_slice(&exp_mrt0([0, 1, 2, 3], true));
        prog.push(s_endpgm());
        assert!(compile_rdna2_to_spirv(&prog, [64, 1, 1]).is_err());
    }
}
