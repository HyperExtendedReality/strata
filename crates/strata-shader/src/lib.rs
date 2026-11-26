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
    let mut module = ssa::build_ssa(&decoded, workgroup_size);
    let opt_stats = opt::optimize(&mut module);
    let spirv = spirv::emit_spirv(&module);
    Ok(CompiledShader { spirv, opt_stats, wave })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn coalesces_four_scalar_bda_loads_and_validates_spirv() {
        use rdna2::encode::*;

        // Synthetic RDNA2 kernel:
        // v1 = v0 << 4 (16-byte stride per thread)
        // load 4 adjacent dwords from BDA pointer in s[0:1] at offsets 0, 4, 8, 12
        // compute v2 = v2 * v3 + v4 + v5 (tests FMA + bitcast cancellation)
        // store result to BDA pointer in s[2:3]
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
        assert_eq!(compiled.opt_stats.coalesced_loads, 1, "expected 4 scalar loads to coalesce into 1x 128-bit load");
        assert!(compiled.opt_stats.after < compiled.opt_stats.before);

        // Validate emitted SPIR-V with Vulkan SDK's `spirv-val` if installed.
        let tmp = std::env::temp_dir().join("strata_test_coalesced.spv");
        let bytes: Vec<u8> = compiled.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&tmp, &bytes).unwrap();

        if let Ok(output) = Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&tmp)
            .output()
        {
            assert!(
                output.status.success(),
                "spirv-val failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let _ = std::fs::remove_file(&tmp);
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
    fn structured_cfg_selection_merge_and_phi_validates_spirv() {
        let spv = spirv::emit_structured_select_kernel([64, 1, 1]);
        let tmp = std::env::temp_dir().join("strata_test_cfg.spv");
        let bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&tmp, &bytes).unwrap();

        if let Ok(output) = Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&tmp)
            .output()
        {
            assert!(
                output.status.success(),
                "spirv-val failed on structured CFG:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn wave64_on_wave32_ballot_validates_spirv() {
        let spv = spirv::emit_wave64_on_wave32_ballot_kernel();
        let tmp = std::env::temp_dir().join("strata_test_w64.spv");
        let bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&tmp, &bytes).unwrap();

        if let Ok(output) = Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&tmp)
            .output()
        {
            assert!(
                output.status.success(),
                "spirv-val failed on Wave64-on-Wave32:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let _ = std::fs::remove_file(&tmp);
    }
}
