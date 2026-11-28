//! Strata RDNA2 → SPIR-V 1.6 (PhysicalStorageBuffer / No-API) shader compiler.

pub mod bma;
pub mod bvh;
pub mod cache;
pub mod container;
pub mod coverage;
pub mod descriptor;
pub mod dxil;
pub mod format;
pub mod geometry;
pub mod inflate;
pub mod opcodes;
pub mod opt;
pub mod rayquery;
pub mod rdna2;
pub mod rect;
pub mod resource;
pub mod sampler;
pub mod spill;
pub mod spirv;
pub mod ssa;
mod static_stage;
pub mod store;
pub mod structurizer;
pub mod tessellation;
#[doc(hidden)]
pub mod test_kernels;
pub mod wave;
pub mod wave_ops;

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
    compile_rdna2_with(words, workgroup_size, &ssa::BuildOptions::default())
}

/// As [`compile_rdna2_to_spirv`], for any stage (vertex and pixel shaders
/// ignore `workgroup_size`).
pub fn compile_rdna2_with(
    words: &[u32],
    workgroup_size: [u32; 3],
    opts: &ssa::BuildOptions,
) -> Result<CompiledShader, CompileError> {
    let decoded = rdna2::decode_program(words)?;
    let wave = wave::analyze_wave_liveness(&decoded);
    let mut module = ssa::build_ssa_with(&decoded, &wave, workgroup_size, opts)?;
    let opt_stats = opt::optimize(&mut module);
    let spirv = spirv::emit_spirv(&module);
    Ok(CompiledShader {
        spirv,
        opt_stats,
        wave,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn assert_spirv_valid(words: &[u32], tag: &str) {
        let tmp = std::env::temp_dir().join(format!("strata_test_{tag}.spv"));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&tmp, &bytes).unwrap();

        let validation = Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&tmp)
            .output();
        let _ = std::fs::remove_file(&tmp);
        match validation {
            Ok(output) => assert!(
                output.status.success(),
                "spirv-val failed for {tag}:\n{}",
                String::from_utf8_lossy(&output.stderr)
            ),
            Err(e)
                if std::env::var_os("STRATA_REQUIRE_SPIRV_VAL").is_none()
                    && e.kind() == std::io::ErrorKind::NotFound =>
            {
                eprintln!("SKIPPED spirv-val: {tag}: {e}");
            }
            Err(e) => panic!("spirv-val could not validate {tag}: {e}"),
        }
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

    /// BMA instrumentation of a box test: spec-constant table address,
    /// Function-scope ray counter, branch-free shortcut — valid SPIR-V.
    #[test]
    fn bma_instrumented_box_test_validates() {
        use rdna2::encode::*;
        let mut prog = Vec::new();
        prog.extend_from_slice(&image_bvh64_intersect_ray(20, 0, 0));
        prog.push(s_waitcnt());
        prog.extend_from_slice(&global_store_dword(30, 20, 2, 0));
        prog.push(s_endpgm());
        let emit = |bma: bool| {
            let decoded = rdna2::decode_program(&prog).unwrap();
            let wave = wave::analyze_wave_liveness(&decoded);
            let mut m =
                ssa::build_ssa_with(&decoded, &wave, [8, 8, 1], &ssa::BuildOptions::default())
                    .unwrap();
            m.bma = bma;
            opt::optimize(&mut m);
            spirv::emit_spirv(&m)
        };
        let plain = emit(false);
        let bma = emit(true);
        let has_spec = |w: &[u32]| {
            // OpDecorate <target> SpecId <id>
            w.windows(4).any(|x| {
                x[0] == (4 << 16) | 71 && x[2] == 1 && x[3] == crate::bma::SPEC_ID_TABLE_LO
            })
        };
        if !bma::shader_instrumentation_enabled() {
            assert!(!has_spec(&plain));
        }
        assert!(has_spec(&bma));
        assert!(bma.len() > plain.len());
        assert_spirv_valid(&plain, "bvh_plain");
        assert_spirv_valid(&bma, "bvh_bma");
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
            bounds_min: [
                [-1.0, -1.0, 20.0],
                [-1.0, -1.0, 5.0],
                [10.0, 10.0, 10.0],
                [0.0; 3],
            ],
            bounds_max: [
                [1.0, 1.0, 25.0],
                [1.0, 1.0, 8.0],
                [12.0, 12.0, 12.0],
                [0.0; 3],
            ],
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
        prog.push(v_cmp_lt_u32_imm(0, 32)); // pc 0
        prog.push(s_and_saveexec_b64(4, 106)); // pc 1: s[4:5] = exec; exec &= vcc
        prog.push(s_cbranch_execz(1)); // pc 2: skip pc 3 if execz -> pc 4
        prog.push(v_add_nc_u32_imm(1, 10, 0)); // pc 3: then-body
        prog.push(s_xor_b64(126, 126, 4)); // pc 4: exec ^= s[4:5] (else lanes)
        prog.push(s_cbranch_execz(1)); // pc 5: skip pc 6 if execz -> pc 7
        prog.push(v_add_nc_u32_imm(1, 20, 0)); // pc 6: else-body
        prog.push(s_or_b64(126, 126, 4)); // pc 7: restore exec
        prog.push(v_lshlrev_b32_imm(2, 2, 0)); // pc 8: byte offset = v0 << 2
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
        assert!(
            !compiled.wave.whole_live_pcs.is_empty(),
            "expected whole-live VCC ballot"
        );
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

    /// Synthetic RDNA2 loop kernels shared with the live GPU test in `strata-cli`.
    pub use crate::test_kernels::*;

    #[test]
    fn uniform_do_while_loop_validates_spirv() {
        let compiled =
            compile_rdna2_to_spirv(&uniform_do_while_sum(), [64, 1, 1]).expect("compile");
        assert_spirv_valid(&compiled.spirv, "loop_do_while");
    }

    #[test]
    fn divergent_exec_loop_validates_spirv() {
        let compiled = compile_rdna2_to_spirv(&divergent_exec_loop(), [64, 1, 1]).expect("compile");
        assert_spirv_valid(&compiled.spirv, "loop_exec");
    }

    #[test]
    fn while_loop_with_top_test_validates_spirv() {
        let compiled = compile_rdna2_to_spirv(&while_loop_sum(), [64, 1, 1]).expect("compile");
        assert_spirv_valid(&compiled.spirv, "loop_while");
    }

    #[test]
    fn nested_loops_validate_spirv() {
        let compiled = compile_rdna2_to_spirv(&nested_loops(), [64, 1, 1]).expect("compile");
        assert_spirv_valid(&compiled.spirv, "loop_nested");
    }

    #[test]
    fn lds_atomic_add_validates_spirv() {
        let compiled = compile_rdna2_to_spirv(&lds_atomic_count(), [64, 1, 1]).expect("compile");
        assert_spirv_valid(&compiled.spirv, "lds_atomic");
    }

    #[test]
    fn typed_buffer_and_image_kernels_validate_spirv() {
        for (name, prog) in [
            ("typed_load_rgba8", typed_load_rgba8_unorm()),
            ("typed_load_rg16_snorm", typed_load_rg16_snorm()),
            ("typed_store_rgba8", typed_store_rgba8_unorm()),
            ("image_fetch", image_fetch_8x8()),
            ("image_sample_lz", image_sample_lz_fixed()),
            ("image_store", image_store_8x8()),
        ] {
            let compiled =
                compile_rdna2_to_spirv(&prog, [64, 1, 1]).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_spirv_valid(&compiled.spirv, name);
        }
    }

    /// Every lowered sample / gather variant (`_d`, `_o`, `_cl`, `_c_b`,
    /// `_c_d`, ...) over 2D, 3D, cube and 2D-array images, in compute and
    /// pixel shaders, produces valid SPIR-V.
    #[test]
    fn sample_variants_validate_spirv() {
        use rdna2::encode::*;
        let mut count = 0;
        for op in 0x20..=0x5fu32 {
            if rdna2::sample_variant(op).is_none() {
                continue;
            }
            for dim in [1u32, 2, 3, 5] {
                // Depth comparison on 3D images does not exist.
                let compare = rdna2::sample_variant(op).is_some_and(|v| v.0.is_compare());
                if compare && dim == 2 {
                    continue;
                }
                for stage in [ssa::Stage::Compute, ssa::Stage::Pixel] {
                    let mut prog = Vec::new();
                    prog.extend_from_slice(&image_sample_op(op, dim, 20, 0, 8, 16, 0xf));
                    prog.push(s_waitcnt());
                    // Keep the result live.
                    prog.extend_from_slice(&global_store_dword(30, 20, 2, 0));
                    prog.push(s_endpgm());
                    let decoded = rdna2::decode_program(&prog).unwrap();
                    let wl = wave::analyze_wave_liveness(&decoded);
                    let opts = ssa::BuildOptions {
                        stage,
                        ..Default::default()
                    };
                    let mut m = ssa::build_ssa_with(&decoded, &wl, [64, 1, 1], &opts)
                        .unwrap_or_else(|e| panic!("op {op:#x} dim {dim}: {e}"));
                    opt::optimize(&mut m);
                    assert!(
                        m.blocks.iter().flat_map(|b| &b.insts).any(|i| !i.dead
                            && matches!(
                                i.op,
                                ssa::SsaOp::ImageSample { .. }
                                    | ssa::SsaOp::ImageSampleDref { .. }
                                    | ssa::SsaOp::ImageGather { .. }
                            )),
                        "op {op:#x}: sample optimized away"
                    );
                    let words = spirv::emit_spirv(&m);
                    assert_spirv_valid(&words, &format!("sample_{op:x}_{dim}_{stage:?}"));
                    count += 1;
                }
            }
        }
        assert!(count >= 250, "{count}");
    }

    #[test]
    fn every_mimg_dim_samples_loads_and_stores_through_its_own_class() {
        use rdna2::encode::*;
        use resource::{ImageDim, trace_descriptors};
        // dim: 0 1D, 1 2D, 2 3D, 3 Cube, 4 1D array, 5 2D array, 6 MSAA, 7 MSAA array.
        for dim in 0..8u32 {
            let mut prog = Vec::new();
            prog.extend_from_slice(&image_sample_lz_dim(dim, 4, 0, 8, 16, 0xf));
            prog.extend_from_slice(&image_load_dim(dim, 8, 0, 8, 0xf));
            prog.extend_from_slice(&image_store_dim(dim, 4, 0, 8, 0xf));
            prog.push(s_endpgm());
            let compiled = compile_rdna2_to_spirv(&prog, [64, 1, 1])
                .unwrap_or_else(|e| panic!("dim {dim}: {e}"));
            assert_spirv_valid(&compiled.spirv, &format!("mimg_dim_{dim}"));

            let uses = trace_descriptors(&rdna2::decode_program(&prog).unwrap());
            let sampled = ImageDim::from_mimg(dim as u8);
            assert_eq!(uses[0].dim, sampled, "dim {dim} sample class");
            assert_eq!(uses[1].dim, sampled.for_storage(), "dim {dim} load class");
            assert_eq!(uses[2].dim, sampled.for_storage(), "dim {dim} store class");
        }
    }

    #[test]
    fn image_slots_match_between_compiler_and_descriptor_trace() {
        use resource::{ImageBinding, planned_image_bindings, trace_descriptors};
        let mut prog = image_fetch_8x8();
        prog.pop(); // s_endpgm
        prog.extend_from_slice(&rdna2::encode::image_store_2d(4, 10, 8, 0xf));
        prog.extend_from_slice(&rdna2::encode::image_sample_lz_2d(4, 10, 8, 16, 0xf));
        prog.push(rdna2::encode::s_endpgm());
        let decoded = rdna2::decode_program(&prog).unwrap();
        let slots: Vec<_> = planned_image_bindings(&decoded, &trace_descriptors(&decoded), 0, 0)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(
            slots,
            [
                ImageBinding::Fetched(0),
                ImageBinding::Storage(0),
                ImageBinding::Sampled(1)
            ]
        );
        let wave = wave::analyze_wave_liveness(&decoded);
        let m = ssa::build_ssa(&decoded, &wave, [64, 1, 1]).unwrap();
        assert_eq!((m.texture_slots, m.storage_slots), (2, 1));
    }

    #[test]
    fn repeated_image_accesses_share_a_descriptor_slot() {
        use rdna2::encode::{image_sample_lz_2d, s_endpgm};
        let mut program = Vec::new();
        for _ in 0..40 {
            program.extend(image_sample_lz_2d(4, 10, 8, 16, 0xf));
        }
        program.push(s_endpgm());
        let shader = compile_rdna2_to_spirv(&program, [64, 1, 1]).unwrap();
        assert_eq!(spirv::required_image_slots(&shader.spirv), Some(1));
        assert_spirv_valid(&shader.spirv, "deduplicated_image_slots");
    }

    #[test]
    fn distinct_table_entries_can_exceed_the_old_sixteen_slot_limit() {
        use rdna2::encode::{image_sample_lz_2d, s_endpgm, s_load_dwordx8};
        let mut program = Vec::new();
        for slot in 0..24 {
            program.extend(s_load_dwordx8(8, 0, slot * 32));
            program.extend(image_sample_lz_2d(4, 10, 8, 16, 0xf));
        }
        program.push(s_endpgm());
        let shader = compile_rdna2_to_spirv(&program, [64, 1, 1]).unwrap();
        assert_eq!(spirv::required_image_slots(&shader.spirv), Some(24));
        assert_spirv_valid(&shader.spirv, "twenty_four_image_slots");
    }

    #[test]
    fn a_break_out_of_a_loop_body_compiles_and_validates_spirv() {
        use rdna2::encode::*;
        // L: s_cbranch_scc1 EXIT(past the back edge) ; s_cbranch_execnz L ; EXIT:
        let prog = vec![
            s_cmp_lt_u32(4, 132),
            v_add_nc_u32_imm(1, 1, 1), // pc 1: header
            s_cbranch_scc1(1),         // pc 2: forward to pc 4, past the latch
            s_cbranch_execnz(-3),      // pc 3: back to pc 1
            s_endpgm(),                // pc 4
        ];
        let compiled = compile_rdna2_to_spirv(&prog, [64, 1, 1])
            .expect("mid-body loop break should lower to structured SPIR-V");
        assert_spirv_valid(&compiled.spirv, "loop_body_break");
    }

    #[test]
    fn exp_in_a_compute_shader_is_rejected() {
        use rdna2::encode::*;
        let mut prog = Vec::new();
        prog.extend_from_slice(&exp_mrt0([0, 1, 2, 3], true));
        prog.push(s_endpgm());
        assert!(compile_rdna2_to_spirv(&prog, [64, 1, 1]).is_err());
    }

    #[test]
    fn packed_2_10_10_10_unorm_typed_buffer_validates_spirv() {
        use rdna2::encode::*;
        // Format 50 = BUF_FMT_2_10_10_10_UNORM
        let mut prog = Vec::new();
        prog.extend_from_slice(&tbuffer_load_format_idx(2, 4, 0, 0, 50, 0));
        prog.push(s_waitcnt());
        prog.extend_from_slice(&tbuffer_store_format_idx(2, 4, 0, 4, 50, 0));
        prog.push(s_endpgm());
        let compiled = compile_rdna2_to_spirv(&prog, [64, 1, 1]).expect("packed format compiles");
        assert_spirv_valid(&compiled.spirv, "packed_2_10_10_10");
    }

    fn stage(stage: ssa::Stage) -> ssa::BuildOptions {
        ssa::BuildOptions {
            stage,
            ..Default::default()
        }
    }

    #[test]
    fn vertex_and_pixel_shaders_validate_spirv() {
        use ssa::Stage::*;
        for (name, prog, opts) in [
            ("vs_fullscreen", fullscreen_triangle_vs(), stage(Vertex)),
            ("ps_interp", interpolated_color_ps(), stage(Pixel)),
            ("ps_flat", flat_color_ps(), stage(Pixel)),
            (
                "ps_sample",
                sampled_color_ps(),
                ssa::BuildOptions {
                    stage: Pixel,
                    texture_slot_base: 3,
                    ..Default::default()
                },
            ),
            (
                "ps_quad_full",
                quad_broadcast_ps(),
                ssa::BuildOptions {
                    stage: Pixel,
                    require_full_quads: true,
                    ..Default::default()
                },
            ),
        ] {
            let compiled = compile_rdna2_with(&prog, [1, 1, 1], &opts)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_spirv_valid(&compiled.spirv, name);
        }
    }

    #[test]
    fn quad_swizzles_decode_exactly() {
        use rdna2::encode::*;
        let d = |off| rdna2::decode_program(&ds_swizzle(1, 0, off)).map(|v| v[0].inst.clone());
        assert!(matches!(
            d(0x80B1),
            Ok(rdna2::Inst::VQuadSwap { direction: 0, .. })
        ));
        assert!(matches!(
            d(0x804E),
            Ok(rdna2::Inst::VQuadSwap { direction: 1, .. })
        ));
        assert!(matches!(
            d(0x80AA),
            Ok(rdna2::Inst::VQuadBroadcast { lane: 2, .. })
        ));
        assert!(matches!(
            d(0x80E4),
            Ok(rdna2::Inst::DsSwizzleB32 { offset: 0x80E4, .. })
        ));
        assert!(matches!(
            d(0x0041),
            Ok(rdna2::Inst::DsSwizzleB32 { offset: 0x0041, .. })
        ));
    }

    #[test]
    fn pixel_shader_frag_coord_and_front_facing_validates_spirv() {
        use rdna2::encode::*;
        // Enable pos_x (bit 8 -> v0), pos_y (bit 9 -> v1), front_face (bit 12 -> v2)
        let ps_vgprs =
            geometry::PsInputVgprs::from_spi_ps_input_ena((1 << 8) | (1 << 9) | (1 << 12));
        assert_eq!(ps_vgprs.pos_x, Some(0));
        assert_eq!(ps_vgprs.pos_y, Some(1));
        assert_eq!(ps_vgprs.front_face, Some(2));

        let mut prog = Vec::new();
        // Export mrt0 = (v0=FragCoord.x, v1=FragCoord.y, v2=FrontFacing, v0)
        prog.extend(exp(0, 0b1111, [0, 1, 2, 0], true, false));
        prog.push(s_endpgm());

        let compiled = compile_rdna2_with(
            &prog,
            [1, 1, 1],
            &ssa::BuildOptions {
                stage: ssa::Stage::Pixel,
                ps_input_vgprs: Some(ps_vgprs),
                ..Default::default()
            },
        )
        .expect("PS with FragCoord + FrontFacing should compile");
        assert_spirv_valid(&compiled.spirv, "ps_frag_coord_front_facing");
    }

    #[test]
    fn scratch_spill_load_store_compiles_and_validates_spirv() {
        use rdna2::encode::*;
        let mut prog = Vec::new();
        prog.push(v_add_nc_u32_imm(1, 7, 0)); // v1 = v0 + 7
        prog.extend_from_slice(&scratch_store_dword(1, 0)); // spill v1 -> slot 0
        prog.extend_from_slice(&scratch_load_dword(2, 0)); // reload slot 0 -> v2
        prog.push(v_lshlrev_b32_imm(3, 2, 0)); // v3 = v0 << 2
        prog.extend_from_slice(&global_store_dword(3, 2, 0, 0));
        prog.push(s_endpgm());

        let compiled = compile_rdna2_to_spirv(&prog, [64, 1, 1]).expect("scratch spill compile");
        assert_spirv_valid(&compiled.spirv, "scratch_spill");
    }

    #[test]
    fn mid_body_loop_break_compiles_and_validates_spirv() {
        use rdna2::encode::*;
        // Do-while loop with a mid-body conditional break past the latch:
        // pc 0: s_movk_i32 s4, 0
        // pc 1: s_addk_i32 s4, 1          <-- loop header
        // pc 2: s_cmp_eq_u32 s4, 3
        // pc 3: s_cbranch_scc1 +2         <-- mid-body break to pc 6 (past latch pc 5)
        // pc 4: s_cmp_lt_u32 s4, 10
        // pc 5: s_cbranch_scc1 -5         <-- back edge to pc 1
        // pc 6: s_endpgm
        let mut prog = Vec::new();
        prog.push(s_movk_i32(4, 0));
        prog.push(s_addk_i32(4, 1));
        prog.push(s_cmp_eq_u32(4, 128 + 3));
        prog.push(s_cbranch_scc1(2));
        prog.push(s_cmp_lt_u32(4, 128 + 10));
        prog.push(s_cbranch_scc1(-5));
        prog.push(s_endpgm());

        let compiled = compile_rdna2_to_spirv(&prog, [64, 1, 1]).expect("mid-body break compile");
        assert_spirv_valid(&compiled.spirv, "mid_body_break");
    }

    #[test]
    fn irreducible_cfg_lowers_via_structurizer_state_machine_and_validates_spirv() {
        use rdna2::encode::*;
        // Two back edges to the same header (pc 1), which `find_loops` rejects and
        // hands off to `structurizer` + `build_ssa_state_machine`:
        // pc 0: s_movk_i32 s4, 0
        // pc 1: s_addk_i32 s4, 1
        // pc 2: s_cmp_eq_u32 s4, 1
        // pc 3: s_cbranch_scc1 -3         <-- back edge 1 -> pc 1
        // pc 4: s_cmp_lt_u32 s4, 4
        // pc 5: s_cbranch_scc1 -5         <-- back edge 2 -> pc 1
        // pc 6: s_endpgm
        let mut prog = Vec::new();
        prog.push(s_movk_i32(4, 0));
        prog.push(s_addk_i32(4, 1));
        prog.push(s_cmp_eq_u32(4, 128 + 1));
        prog.push(s_cbranch_scc1(-3));
        prog.push(s_cmp_lt_u32(4, 128 + 4));
        prog.push(s_cbranch_scc1(-5));
        prog.push(s_endpgm());

        let compiled =
            compile_rdna2_to_spirv(&prog, [64, 1, 1]).expect("irreducible state-machine compile");
        assert_spirv_valid(&compiled.spirv, "irreducible_state_machine");
    }
}
