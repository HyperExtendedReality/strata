use strata_gpu::context::probe_capabilities;
use strata_gpu::runner::ComputeRunner;
use strata_gpu::uma::UmaManager;
use strata_hle::d3d12x::{D3D12CommandListX, D3D12CommandQueueX};
use strata_hle::loader::{MappedImage, check_import_binding};
use strata_hle::pe::parse_pe64;
use strata_hle::runtime::{DllClass, classify_dll};
use strata_hle::xg::TextureLayout;
use strata_runtime_shaders::{
    BVH_ORACLE_SPV, CLAMP_INDIRECT_SPV, DETILE_SPV, DIRTY_HASH_SPV, PRESENT_BLIT_FS_SPV,
    PRESENT_BLIT_VS_SPV,
};
use strata_shader::{bvh, compile_rdna2_to_spirv, rdna2::encode};

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_else(|| "selftest".to_string());
    match cmd.as_str() {
        "probe" => cmd_probe(),
        "selftest" => cmd_selftest(),
        "gpu-test" => cmd_gpu_test(),
        "uma-test" => cmd_uma_test(),
        "inspect" => {
            let path = std::env::args().nth(2).unwrap_or_else(|| {
                eprintln!("Usage: strata-cli inspect <path-to-exe-or-dll>");
                std::process::exit(1);
            });
            cmd_inspect(&path);
        }
        "bind-check" => {
            let path = std::env::args().nth(2).unwrap_or_else(|| {
                eprintln!("Usage: strata-cli bind-check <path-to-exe-or-dll>");
                std::process::exit(1);
            });
            cmd_bind_check(&path);
        }
        "map-test" => {
            let path = std::env::args().nth(2).unwrap_or_else(|| {
                eprintln!("Usage: strata-cli map-test <path-to-exe-or-dll> [--execute]");
                std::process::exit(1);
            });
            let execute = std::env::args().any(|a| a == "--execute");
            cmd_map_test(&path, execute);
        }
        "run" => {
            let path = std::env::args().nth(2).unwrap_or_else(|| {
                eprintln!("Usage: strata-cli run <path-to-exe>");
                std::process::exit(1);
            });
            cmd_run(&path);
        }
        other => {
            eprintln!("Unknown command '{other}'. Usage: strata-cli [probe|selftest|gpu-test|uma-test|inspect|bind-check|map-test|run]");
            std::process::exit(1);
        }
    }
}

fn cmd_probe() {
    match probe_capabilities() {
        Ok(caps) => {
            println!("=== Strata No-API Vulkan Capability Probe ===");
            println!("GPU:                   {}", caps.device_name);
            println!(
                "Vulkan API:            {}.{}.{}",
                caps.api_version.0, caps.api_version.1, caps.api_version.2
            );
            println!("64-bit BDA Pointers:   {}", caps.buffer_device_address);
            println!("Dynamic Rendering:     {}", caps.dynamic_rendering);
            println!("Bindless Indexing:     {}", caps.descriptor_indexing);
            println!(
                "Subgroup Size:         {} (min={}, max={}, native_wave64={})",
                caps.subgroup_size,
                caps.min_subgroup_size,
                caps.max_subgroup_size,
                caps.native_wave64
            );
            println!("Sparse Image 2D:       {}", caps.sparse_image_2d);
            println!("External Memory Host:  {}", caps.external_memory_host);
            println!("Max Push Constants:    {} bytes", caps.max_push_constants_size);
        }
        Err(e) => {
            eprintln!("Vulkan probe failed: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_selftest() {
    println!("=== Strata Self-Test ===");
    println!(
        "Embedded Slang SPIR-V shaders: dirty_hash={} B, detile={} B, bvh_oracle={} B, clamp_indirect={} B",
        DIRTY_HASH_SPV.len(),
        DETILE_SPV.len(),
        BVH_ORACLE_SPV.len(),
        CLAMP_INDIRECT_SPV.len()
    );

    let compiled = build_sample_rdna2_kernel();
    println!(
        "RDNA2 -> SPIR-V 1.6 (BDA): {} words ({} -> {} SSA insts, {}x 128-bit coalesced loads)",
        compiled.spirv.len(),
        compiled.opt_stats.before,
        compiled.opt_stats.after,
        compiled.opt_stats.coalesced_loads
    );

    if let Ok(caps) = probe_capabilities() {
        println!(
            "Host GPU ready: {} (BDA={}, Bindless={}, Wave64={})",
            caps.device_name, caps.buffer_device_address, caps.descriptor_indexing, caps.native_wave64
        );
    }
}

fn cmd_gpu_test() {
    let runner = ComputeRunner::new().expect("failed to initialize Vulkan 1.3 BDA runner");
    println!("=== Strata Live GPU BDA Test on {} ===", runner.device_name);

    // 1. Execute our pure-Rust RDNA2 -> SPIR-V 1.6 (128-bit coalesced BDA) kernel on the GPU!
    let compiled = build_sample_rdna2_kernel();
    let lanes = 64usize;
    let mut in_buf = runner.alloc_bda_buffer((lanes * 16) as u64).unwrap();
    let out_buf = runner.alloc_bda_buffer((lanes * 16) as u64).unwrap();

    unsafe {
        let input = in_buf.as_mut_slice::<[f32; 4]>(lanes);
        for (i, slot) in input.iter_mut().enumerate() {
            let x = (i + 1) as f32;
            *slot = [x, 2.5, 10.0, 0.25 * x];
        }
    }

    // Push console User SGPRs: s[0:1] = in_buf.bda, s[2:3] = out_buf.bda
    let mut sgprs = [0u32; 32];
    sgprs[0] = in_buf.bda as u32;
    sgprs[1] = (in_buf.bda >> 32) as u32;
    sgprs[2] = out_buf.bda as u32;
    sgprs[3] = (out_buf.bda >> 32) as u32;
    let push_bytes: Vec<u8> = sgprs.iter().flat_map(|w| w.to_le_bytes()).collect();

    runner
        .dispatch_spirv(&compiled.spirv, &push_bytes, [1, 1, 1])
        .expect("RDNA2 SPIR-V dispatch failed");

    unsafe {
        let output = out_buf.as_slice::<[f32; 4]>(lanes);
        for (i, slot) in output.iter().enumerate() {
            let x = (i + 1) as f32;
            let expected = x.mul_add(2.5, 10.0) + 0.25 * x;
            assert!(
                (slot[0] - expected).abs() < 1e-5,
                "lane {i} mismatch: got {}, expected {expected}",
                slot[0]
            );
        }
    }
    println!("[PASS] Direct RDNA2 -> SPIR-V (128-bit coalesced BDA): 64/64 lanes bit-exact on GPU");
    runner.free_bda_buffer(in_buf);
    runner.free_bda_buffer(out_buf);

    // 2. Execute Slang `clamp_indirect.spv` (Belobug hoisted indirect clamp) on the GPU!
    let mut args_buf = runner.alloc_bda_buffer(16).unwrap();
    let mut entries_buf = runner.alloc_bda_buffer(16).unwrap();
    unsafe {
        let args = args_buf.as_mut_slice::<u32>(4);
        args[0] = 999_999; // over-limit vertex count
        args[1] = 5_000;   // over-limit instance count
        let entries = entries_buf.as_mut_slice::<u32>(4);
        entries[0] = args_buf.bda as u32;
        entries[1] = (args_buf.bda >> 32) as u32;
        entries[2] = 65_535; // max_count
        entries[3] = 64;     // max_inst
    }
    let mut clamp_pc = Vec::with_capacity(16);
    clamp_pc.extend_from_slice(&entries_buf.bda.to_le_bytes());
    clamp_pc.extend_from_slice(&1u32.to_le_bytes());
    clamp_pc.extend_from_slice(&0u32.to_le_bytes());

    let clamp_words = bytes_to_u32_vec(CLAMP_INDIRECT_SPV);
    runner
        .dispatch_spirv(&clamp_words, &clamp_pc, [1, 1, 1])
        .expect("clamp_indirect.spv dispatch failed");

    unsafe {
        let args = args_buf.as_slice::<u32>(4);
        assert_eq!(args[0], 65_535);
        assert_eq!(args[1], 64);
    }
    println!("[PASS] Slang clamp_indirect.spv (BDA): indirect draw args clamped on GPU");
    runner.free_bda_buffer(args_buf);
    runner.free_bda_buffer(entries_buf);

    // 3. Execute Slang `dirty_hash.spv` (GPU 64 KiB page hashing via WaveActiveBitXor)!
    let page_bytes = 64 * 1024u64;
    let mut pages_buf = runner.alloc_bda_buffer(page_bytes * 2).unwrap();
    let hashes_buf = runner.alloc_bda_buffer(16).unwrap();
    unsafe {
        let dwords = pages_buf.as_mut_slice::<u32>((page_bytes * 2 / 4) as usize);
        for (i, w) in dwords.iter_mut().enumerate() {
            *w = (i as u32).wrapping_mul(0x9E37_79B9);
        }
    }
    let mut hash_pc = Vec::with_capacity(24);
    hash_pc.extend_from_slice(&pages_buf.bda.to_le_bytes());
    hash_pc.extend_from_slice(&hashes_buf.bda.to_le_bytes());
    hash_pc.extend_from_slice(&2u32.to_le_bytes());
    hash_pc.extend_from_slice(&((page_bytes / 4) as u32).to_le_bytes());

    let hash_words = bytes_to_u32_vec(DIRTY_HASH_SPV);
    runner
        .dispatch_spirv(&hash_words, &hash_pc, [2, 1, 1])
        .expect("dirty_hash.spv dispatch 1 failed");
    let h_before = unsafe { [hashes_buf.as_slice::<u32>(2)[0], hashes_buf.as_slice::<u32>(2)[1]] };

    // Mutate 1 dword in page 1 only
    unsafe {
        pages_buf.as_mut_slice::<u32>((page_bytes * 2 / 4) as usize)[(page_bytes / 4) as usize + 77] ^= 0xDEAD_BEEF;
    }
    runner
        .dispatch_spirv(&hash_words, &hash_pc, [2, 1, 1])
        .expect("dirty_hash.spv dispatch 2 failed");
    let h_after = unsafe { [hashes_buf.as_slice::<u32>(2)[0], hashes_buf.as_slice::<u32>(2)[1]] };

    assert_eq!(h_before[0], h_after[0], "clean page 0 hash changed");
    assert_ne!(h_before[1], h_after[1], "dirty page 1 hash did not detect mutation");
    println!(
        "[PASS] Slang dirty_hash.spv (BDA + WaveActiveBitXor): page0=0x{:08x} (clean), page1=0x{:08x}->0x{:08x} (dirty)",
        h_before[0], h_before[1], h_after[1]
    );
    runner.free_bda_buffer(pages_buf);
    runner.free_bda_buffer(hashes_buf);

    // 4. TRUE ZERO-COPY UNIFIED MEMORY TEST (`VK_EXT_external_memory_host` + Windows `VirtualAlloc`):
    test_zero_copy_virtual_alloc(&runner, &compiled);

    // 5. NO-API DYNAMIC RENDERING + SLANG VERTEX-LESS PRESENT BLIT TEST:
    //    ZERO VkRenderPass, ZERO VkFramebuffer, ZERO vertex buffers!
    let mut fb = runner.alloc_bda_buffer(16 * 16 * 4).unwrap();
    unsafe {
        // Fill 16x16 source image with RGBA(64, 100, 50, 255) -> exposure 2.0 => RGBA(128, 200, 100, 255)
        let packed = u32::from_le_bytes([64, 100, 50, 255]);
        fb.as_mut_slice::<u32>(16 * 16).fill(packed);
    }
    let vs_words = bytes_to_u32_vec(PRESENT_BLIT_VS_SPV);
    let fs_words = bytes_to_u32_vec(PRESENT_BLIT_FS_SPV);
    let px = runner
        .render_no_api_present_blit(&vs_words, &fs_words, fb.bda, 16, 16, 2.0)
        .expect("No-API dynamic rendering present_blit failed");
    assert_eq!(px, [128, 200, 100, 255], "unexpected rendered pixel: {px:?}");
    println!(
        "[PASS] No-API VK_KHR_dynamic_rendering + Slang present_blit.slang (0 RenderPasses, 0 VertexBuffers): pixel={px:?}"
    );
    runner.free_bda_buffer(fb);

    // 6. SLANG `detile.slang` (64KB_S Morton -> Linear BDA) LIVE GPU TEST:
    test_gpu_detile_64ks(&runner);

    // 7. SLANG `bvh_oracle.slang` vs CPU `bvh::intersect_box32` DIFFERENTIAL GPU PARITY TEST:
    test_gpu_bvh_oracle_parity(&runner);

    // 8. RDNA2-TRANSLATED STRUCTURED CFG (`v_cmp_lt_u32` + `s_and_saveexec_b64` + `s_cbranch_execz` -> `OpSelectionMerge`/`OpPhi`) GPU TEST:
    let cfg_compiled = build_divergent_cfg_rdna2_kernel();
    let cfg_out = runner.alloc_bda_buffer(64 * 4).unwrap();
    let mut cfg_push = [0u32; 32];
    cfg_push[0] = cfg_out.bda as u32;
    cfg_push[1] = (cfg_out.bda >> 32) as u32;
    let cfg_push_bytes: Vec<u8> = cfg_push.iter().flat_map(|w| w.to_le_bytes()).collect();
    runner
        .dispatch_spirv(&cfg_compiled.spirv, &cfg_push_bytes, [1, 1, 1])
        .expect("RDNA2-translated structured CFG SPIR-V dispatch failed");
    unsafe {
        let vals = cfg_out.as_slice::<u32>(64);
        for (i, &v) in vals.iter().enumerate() {
            let expected = if i < 32 { i as u32 + 10 } else { i as u32 + 20 };
            assert_eq!(v, expected, "CFG lane {i} mismatch: got {v}, expected {expected}");
        }
    }
    println!("[PASS] RDNA2->SPIR-V Structured CFG (s_and_saveexec_b64 + s_cbranch_execz -> OpSelectionMerge + OpPhi): 64/64 divergent lanes exact on GPU");
    runner.free_bda_buffer(cfg_out);

    // 9. RDNA2-TRANSLATED WAVE64 BALLOT (`v_cmp_eq_u32` + `v_mov_b32 vcc_lo/hi` -> Wave64 ballot exchange):
    let w64_compiled = build_wave64_ballot_rdna2_kernel();
    assert!(
        !w64_compiled.wave.whole_live_pcs.is_empty(),
        "wave.rs must classify VCC ballot as whole-live"
    );
    let w64_out = runner.alloc_bda_buffer(16).unwrap();
    let mut w64_push = [0u32; 32];
    w64_push[0] = w64_out.bda as u32;
    w64_push[1] = (w64_out.bda >> 32) as u32;
    let w64_push_bytes: Vec<u8> = w64_push.iter().flat_map(|w| w.to_le_bytes()).collect();
    runner
        .dispatch_spirv(&w64_compiled.spirv, &w64_push_bytes, [1, 1, 1])
        .expect("RDNA2-translated Wave64 ballot SPIR-V dispatch failed");
    let mask = unsafe { w64_out.as_slice::<u64>(1)[0] };
    assert_eq!(mask, 0x5555_5555_5555_5555, "unexpected Wave64 mask: 0x{mask:016x}");
    let tested_sizes: Vec<u32> = if runner.subgroup_size_control {
        [8u32, 16, 32, 64]
            .into_iter()
            .filter(|&sz| sz >= runner.min_subgroup_size && sz <= runner.max_subgroup_size)
            .collect()
    } else {
        Vec::new()
    };
    if tested_sizes.len() > 1 {
        for &req_sg in &tested_sizes {
            runner
                .dispatch_spirv_with_subgroup_size(
                    &w64_compiled.spirv,
                    &w64_push_bytes,
                    [1, 1, 1],
                    Some(req_sg),
                )
                .unwrap_or_else(|e| panic!("Wave64 ballot dispatch failed at Wave{req_sg}: {e}"));
            let req_mask = unsafe { w64_out.as_slice::<u64>(1)[0] };
            assert_eq!(
                req_mask, 0x5555_5555_5555_5555,
                "unexpected Wave64 mask at Wave{req_sg}: 0x{req_mask:016x}"
            );
        }
        println!(
            "[PASS] RDNA2->SPIR-V Wave64 Ballot (verified subgroup sizes {:?} -> 64-bit console VCC): 0x{mask:016x}",
            tested_sizes
        );
    } else if runner.native_wave64 {
        println!("[PASS] RDNA2->SPIR-V Wave64 Ballot (native Wave64 -> 64-bit console VCC): 0x{mask:016x}");
    } else {
        println!(
            "[PASS] RDNA2->SPIR-V Wave64 Ballot (Wave{} + Workgroup atomic merge -> 64-bit console VCC): 0x{mask:016x}",
            runner.subgroup_size
        );
    }
    runner.free_bda_buffer(w64_out);
}

fn test_gpu_detile_64ks(runner: &ComputeRunner) {
    let (w, h, bpp) = (16u32, 16u32, 4u32);
    let mut tiled = runner.alloc_bda_buffer((w * h * bpp) as u64).unwrap();
    let linear = runner.alloc_bda_buffer((w * h * bpp) as u64).unwrap();

    unsafe {
        let dst = tiled.as_mut_slice::<u32>((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let off_dw = TextureLayout::swizzle_64ks_offset(x, y, w, bpp) / 4;
                dst[off_dw] = (y << 16) | x;
            }
        }
    }

    let mut pc = Vec::with_capacity(32);
    pc.extend_from_slice(&tiled.bda.to_le_bytes());
    pc.extend_from_slice(&linear.bda.to_le_bytes());
    pc.extend_from_slice(&w.to_le_bytes());
    pc.extend_from_slice(&h.to_le_bytes());
    pc.extend_from_slice(&bpp.to_le_bytes());
    pc.extend_from_slice(&9u32.to_le_bytes()); // Swizzle64KS

    let detile_words = bytes_to_u32_vec(DETILE_SPV);
    runner
        .dispatch_spirv(&detile_words, &pc, [w.div_ceil(8), h.div_ceil(8), 1])
        .expect("detile.slang GPU dispatch failed");

    unsafe {
        let out = linear.as_slice::<u32>((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let got = out[(y * w + x) as usize];
                let expected = (y << 16) | x;
                assert_eq!(got, expected, "detile mismatch at ({x},{y})");
            }
        }
    }
    println!("[PASS] Slang detile.slang (64KB_S Morton -> Linear via BDA): 256/256 texels exact on GPU");
    runner.free_bda_buffer(tiled);
    runner.free_bda_buffer(linear);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuRayData {
    t_max: f32,
    origin: [f32; 3],
    dir: [f32; 3],
    inv_dir: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuBox32Node {
    children: [u32; 4],
    bmin: [[f32; 3]; 4],
    bmax: [[f32; 3]; 4],
}

fn test_gpu_bvh_oracle_parity(runner: &ComputeRunner) {
    let count = 64usize;
    let mut rays_buf = runner
        .alloc_bda_buffer((count * std::mem::size_of::<GpuRayData>()) as u64)
        .unwrap();
    let mut nodes_buf = runner
        .alloc_bda_buffer((count * std::mem::size_of::<GpuBox32Node>()) as u64)
        .unwrap();
    let res_buf = runner.alloc_bda_buffer((count * 16) as u64).unwrap();

    let mut cpu_expected = Vec::with_capacity(count);
    unsafe {
        let rays = rays_buf.as_mut_slice::<GpuRayData>(count);
        let nodes = nodes_buf.as_mut_slice::<GpuBox32Node>(count);
        for i in 0..count {
            let z_near = 2.0 + (i as f32) * 0.5;
            let z_far = z_near + 10.0;
            let ray = bvh::Ray {
                t_max: 100.0,
                origin: [0.0, 0.0, 0.0],
                dir: [0.0, 0.0, 1.0],
                inv_dir: [f32::INFINITY, f32::INFINITY, 1.0],
            };
            let node = bvh::Box32Node {
                children: [100 + i as u32, 200 + i as u32, 300 + i as u32, bvh::INVALID_NODE],
                bounds_min: [[-1.0, -1.0, z_far], [-1.0, -1.0, z_near], [5.0, 5.0, 5.0], [0.0; 3]],
                bounds_max: [[1.0, 1.0, z_far + 2.0], [1.0, 1.0, z_near + 2.0], [6.0, 6.0, 6.0], [0.0; 3]],
            };
            rays[i] = GpuRayData {
                t_max: ray.t_max,
                origin: ray.origin,
                dir: ray.dir,
                inv_dir: ray.inv_dir,
            };
            nodes[i] = GpuBox32Node {
                children: node.children,
                bmin: node.bounds_min,
                bmax: node.bounds_max,
            };
            cpu_expected.push(bvh::intersect_box32(&ray, &node));
        }
    }

    let mut pc = Vec::with_capacity(32);
    pc.extend_from_slice(&rays_buf.bda.to_le_bytes());
    pc.extend_from_slice(&nodes_buf.bda.to_le_bytes());
    pc.extend_from_slice(&res_buf.bda.to_le_bytes());
    pc.extend_from_slice(&(count as u32).to_le_bytes());

    let bvh_words = bytes_to_u32_vec(BVH_ORACLE_SPV);
    runner
        .dispatch_spirv(&bvh_words, &pc, [1, 1, 1])
        .expect("bvh_oracle.slang GPU dispatch failed");

    unsafe {
        let gpu_out = res_buf.as_slice::<[u32; 4]>(count);
        for i in 0..count {
            assert_eq!(
                gpu_out[i], cpu_expected[i],
                "BVH box32 mismatch at ray {i}: gpu={:?} vs cpu={:?}",
                gpu_out[i], cpu_expected[i]
            );
        }
    }
    println!("[PASS] Slang bvh_oracle.slang vs Rust CPU bvh::intersect_box32: 64/64 rays bit-exact on GPU");
    runner.free_bda_buffer(rays_buf);
    runner.free_bda_buffer(nodes_buf);
    runner.free_bda_buffer(res_buf);
}

#[cfg(windows)]
unsafe extern "system" {
    fn VirtualAlloc(addr: *mut core::ffi::c_void, size: usize, alloc_type: u32, protect: u32) -> *mut core::ffi::c_void;
    fn VirtualFree(addr: *mut core::ffi::c_void, size: usize, free_type: u32) -> i32;
}

fn test_zero_copy_virtual_alloc(runner: &ComputeRunner, compiled: &strata_shader::CompiledShader) {
    const MEM_COMMIT_RESERVE: u32 = 0x3000;
    const PAGE_READWRITE: u32 = 0x04;
    const MEM_RELEASE: u32 = 0x8000;
    const ALLOC_BYTES: usize = 64 * 1024;

    unsafe {
        let raw_ptr = VirtualAlloc(std::ptr::null_mut(), ALLOC_BYTES, MEM_COMMIT_RESERVE, PAGE_READWRITE) as *mut u8;
        assert!(!raw_ptr.is_null(), "VirtualAlloc failed");

        let host_buf = runner
            .import_host_bda_buffer(raw_ptr, ALLOC_BYTES as u64)
            .expect("VK_EXT_external_memory_host import failed");

        // Write 64 lanes of [a, b, c, d] directly into CPU VirtualAlloc memory at offset 0.
        let lanes = 64usize;
        let in_slice = std::slice::from_raw_parts_mut(raw_ptr as *mut [f32; 4], lanes);
        for (i, slot) in in_slice.iter_mut().enumerate() {
            let x = (i + 1) as f32;
            *slot = [x, 3.0, 5.0, 0.5 * x];
        }

        // GPU reads from host_buf.bda + 0 and writes directly to host_buf.bda + 4096 (ZERO copies!).
        let src_bda = host_buf.bda;
        let dst_bda = host_buf.bda + 4096;
        let mut sgprs = [0u32; 32];
        sgprs[0] = src_bda as u32;
        sgprs[1] = (src_bda >> 32) as u32;
        sgprs[2] = dst_bda as u32;
        sgprs[3] = (dst_bda >> 32) as u32;
        let push_bytes: Vec<u8> = sgprs.iter().flat_map(|w| w.to_le_bytes()).collect();

        runner
            .dispatch_spirv(&compiled.spirv, &push_bytes, [1, 1, 1])
            .expect("zero-copy RDNA2 dispatch failed");

        // Read results directly from the CPU VirtualAlloc pointer at offset 4096!
        let out_slice = std::slice::from_raw_parts(raw_ptr.add(4096) as *const [f32; 4], lanes);
        for (i, slot) in out_slice.iter().enumerate() {
            let x = (i + 1) as f32;
            let expected = x.mul_add(3.0, 5.0) + 0.5 * x;
            assert!(
                (slot[0] - expected).abs() < 1e-5,
                "zero-copy lane {i} mismatch: got {}, expected {expected}",
                slot[0]
            );
        }

        println!(
            "[PASS] Zero-Copy UMA (VirtualAlloc {:p} -> BDA 0x{:016x}, align={} B): 64/64 lanes CPU<->GPU with ZERO memcpy",
            raw_ptr, host_buf.bda, runner.min_host_pointer_alignment
        );

        runner.free_bda_buffer(host_buf);
        VirtualFree(raw_ptr as *mut core::ffi::c_void, 0, MEM_RELEASE);
    }
}

fn cmd_uma_test() {
    let runner = ComputeRunner::new().expect("failed to initialize Vulkan 1.3 runner");
    let compiled = build_sample_rdna2_kernel();
    let mut uma = UmaManager::new();

    println!("=== Strata 3-Tier Hybrid UMA Live Test on {} ===", runner.device_name);

    // Tier 1: Console VA -> Zero-Copy Windows VirtualAlloc (VK_EXT_external_memory_host)
    let console_out_va = 0x0000_0400_1000_0000u64;
    let out_alloc = uma
        .xmem_virtual_alloc_zero_copy(&runner, console_out_va, 64 * 1024)
        .expect("Tier 1 XMemVirtualAlloc zero-copy failed");
    let out_cpu_ptr = out_alloc.cpu_ptr().expect("Tier 1 must have CPU pointer");
    println!(
        "[TIER 1] XMemVirtualAlloc Zero-Copy: console_va=0x{:x} -> host_bda=0x{:016x} (cpu={:p})",
        console_out_va, out_alloc.host_bda, out_cpu_ptr
    );

    // Tier 2: Static/read-heavy data -> DEVICE_LOCAL GDDR6 VRAM BDA mirror
    let console_in_va = 0x0000_0400_2000_0000u64;
    let lanes = 64usize;
    let mut init_bytes = Vec::with_capacity(lanes * 16);
    for i in 0..lanes {
        let x = (i + 1) as f32;
        for f in [x, 4.0f32, 7.0f32, 0.125f32 * x] {
            init_bytes.extend_from_slice(&f.to_le_bytes());
        }
    }
    let in_alloc = uma
        .alloc_vram_mirror(&runner, console_in_va, &init_bytes)
        .expect("Tier 2 VRAM mirror failed");
    println!(
        "[TIER 2] DEVICE_LOCAL VRAM Mirror:   console_va=0x{:x} -> host_bda=0x{:016x} ({} B)",
        console_in_va, in_alloc.host_bda, in_alloc.size
    );

    // Cross-tier execution: RDNA2 -> SPIR-V shader reads from Tier 2 GDDR6 VRAM
    // and writes directly into Tier 1 Zero-Copy CPU VirtualAlloc memory!
    let src_bda = uma.translate_va(console_in_va).unwrap();
    let dst_bda = uma.translate_va(console_out_va).unwrap();
    let mut sgprs = [0u32; 32];
    sgprs[0] = src_bda as u32;
    sgprs[1] = (src_bda >> 32) as u32;
    sgprs[2] = dst_bda as u32;
    sgprs[3] = (dst_bda >> 32) as u32;
    let push_bytes: Vec<u8> = sgprs.iter().flat_map(|w| w.to_le_bytes()).collect();

    runner
        .dispatch_spirv(&compiled.spirv, &push_bytes, [1, 1, 1])
        .expect("cross-tier RDNA2 dispatch failed");

    unsafe {
        let out_slice = std::slice::from_raw_parts(out_cpu_ptr as *const [f32; 4], lanes);
        for (i, slot) in out_slice.iter().enumerate() {
            let x = (i + 1) as f32;
            let expected = x.mul_add(4.0, 7.0) + 0.125 * x;
            assert!(
                (slot[0] - expected).abs() < 1e-5,
                "cross-tier lane {i} mismatch: got {}, expected {expected}",
                slot[0]
            );
        }
    }
    println!("[PASS] Cross-Tier GPU Execution: Tier 2 GDDR6 -> RDNA2 SPIR-V -> Tier 1 Zero-Copy CPU RAM (64/64 lanes)");

    // Tier 3: Live vkQueueBindSparse 64 KiB aliased 2D render targets
    let tiles_a = uma.acquire_sparse_target_tiles(0x0000_0400_3000_0000, 256 * 1024);
    let tiles_b = uma.acquire_sparse_target_tiles(0x0000_0400_3002_0000, 256 * 1024);
    assert_eq!(tiles_a[2].1, tiles_b[0].1);
    let (tile_bytes, gran) = runner
        .verify_sparse_tile_aliasing()
        .expect("Tier 3 vkQueueBindSparse failed");
    println!(
        "[TIER 3] Live vkQueueBindSparse: 2 aliased 256x256 VkImages bound to shared {} KiB VRAM tile (granularity={}x{}x{})",
        tile_bytes / 1024,
        gran.width,
        gran.height,
        gran.depth
    );

    uma.destroy_all(&runner);
}

fn cmd_inspect(path: &str) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to read '{path}': {e}");
        std::process::exit(1);
    });
    let info = parse_pe64(&bytes).unwrap_or_else(|e| {
        eprintln!("PE32+ parse failed for '{path}': {e}");
        std::process::exit(1);
    });

    println!("=== Strata PE32+ Console Import Census: {path} ===");
    println!(
        "ImageBase=0x{:016x}, EntryRVA=0x{:08x}, ImageSize={} MiB, Sections={}, DLLs={}, Imports={}",
        info.image_base,
        info.entry_rva,
        info.size_of_image / (1024 * 1024),
        info.sections.len(),
        info.imports.len(),
        info.total_import_count()
    );

    let (mut hle, mut host, mut local, mut stub) = (0usize, 0usize, 0usize, 0usize);
    for dll in &info.imports {
        let class = classify_dll(&dll.dll_name);
        let tag = match class {
            DllClass::StrataHle => {
                hle += dll.functions.len();
                "HLE  "
            }
            DllClass::HostPassthrough => {
                host += dll.functions.len();
                "HOST "
            }
            DllClass::TitleLocal => {
                local += dll.functions.len();
                "LOCAL"
            }
            DllClass::NeedsStub => {
                stub += dll.functions.len();
                "STUB "
            }
        };
        println!("  [{tag}] {:<36} ({} imports)", dll.dll_name, dll.functions.len());
        if class == DllClass::StrataHle {
            for f in &dll.functions {
                println!("          -> {f}");
            }
        }
    }
    println!(
        "Summary: HLE={hle}, HostPassthrough={host}, TitleLocal={local}, NeedsStub={stub}"
    );
}

fn cmd_bind_check(path: &str) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to read '{path}': {e}");
        std::process::exit(1);
    });
    let info = parse_pe64(&bytes).unwrap_or_else(|e| {
        eprintln!("PE32+ parse failed for '{path}': {e}");
        std::process::exit(1);
    });
    let report = check_import_binding(&info);
    println!("=== Strata Live IAT Bind Check: {path} ===");
    println!("Resolved via Strata HLE thunks: {}", report.resolved_hle);
    println!("Resolved via Host Windows DLLs:   {}", report.resolved_host);
    println!("Deferred to Title-Local DLLs:     {}", report.deferred_title_local);
    println!("Unresolved imports:               {}", report.unresolved.len());
    for u in report.unresolved.iter().take(20) {
        println!("  [MISSING] {u}");
    }
}

fn cmd_run(path: &str) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to read '{path}': {e}");
        std::process::exit(1);
    });
    let mapped = MappedImage::load_from_bytes(&bytes).unwrap_or_else(|e| {
        eprintln!("PE32+ map failed for '{path}': {e}");
        std::process::exit(1);
    });
    println!(
        "[INFO] Mapped {} ({} KiB) at {:p}, Entry=0x{:x}, Relocs={}, IAT(HLE={}, Host={}, Local={}, Stubbed={})",
        path,
        mapped.size / 1024,
        mapped.base,
        mapped.entry_point,
        mapped.relocations_applied,
        mapped.bind_report.resolved_hle,
        mapped.bind_report.resolved_host,
        mapped.bind_report.deferred_title_local,
        mapped.bind_report.unresolved.len()
    );
    if !mapped.bind_report.unresolved.is_empty() {
        println!("[WARN] {} unresolved imports patched to logged E_NOTIMPL stubs", mapped.bind_report.unresolved.len());
    }
    println!("[INFO] Invoking mapped PE32+ entry point 0x{:x} on guest thread...", mapped.entry_point);
    match unsafe { mapped.execute_entry_point() } {
        Ok(exit_code) => {
            println!(
                "[PASS] Guest entry point exited cleanly with code {exit_code} (unimplemented_stubs={}, audio_stubs={})",
                strata_hle::thunks::unimplemented_stub_call_count(),
                strata_hle::thunks::audio_stub_call_count(),
            );
        }
        Err(e) => {
            eprintln!("[FAIL] Guest execution failed: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_map_test(path: &str, execute: bool) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to read '{path}': {e}");
        std::process::exit(1);
    });
    let mapped = MappedImage::load_from_bytes(&bytes).unwrap_or_else(|e| {
        eprintln!("PE32+ map failed for '{path}': {e}");
        std::process::exit(1);
    });
    println!("=== Strata In-Memory PE32+ Map Diagnostic ===");
    println!(
        "[PASS] Mapped {} ({} MiB) at {:p}, Entry=0x{:x}, Relocs={}, IAT(HLE={}, Host={}, Local={}, Stubbed={})",
        path,
        mapped.size / (1024 * 1024),
        mapped.base,
        mapped.entry_point,
        mapped.relocations_applied,
        mapped.bind_report.resolved_hle,
        mapped.bind_report.resolved_host,
        mapped.bind_report.deferred_title_local,
        mapped.bind_report.unresolved.len()
    );
    if let Some(tls) = &mapped.tls {
        println!(
            "[PASS] Static TLS Initialized: host_slot={}, template={} B, zero_fill={} B, callbacks={}",
            tls.slot_index, tls.raw_template_bytes, tls.zero_fill_bytes, tls.callback_count
        );
    }

    if execute {
        println!("[INFO] Executing mapped entry point 0x{:x} (--execute)...", mapped.entry_point);
        let code = unsafe { mapped.execute_entry_point() }.unwrap_or_else(|e| {
            eprintln!("[FAIL] Guest entry execution failed: {e}");
            std::process::exit(1);
        });
        println!("[PASS] Guest entry point returned {code}");
    } else {
        println!("[INFO] Entry point was NOT executed (pass `--execute` or use `strata-cli run <path.exe>` to run guest code).");
    }

    // Separate synthetic COM vtable -> ThinCommandList -> Vulkan 1.3 BDA sanity check (independent of mapped PE)
    let runner = std::sync::Arc::new(ComputeRunner::new().expect("Vulkan 1.3 runner failed"));
    let compiled = build_sample_rdna2_kernel();
    let mut uma = UmaManager::new();
    let alloc = uma
        .xmem_virtual_alloc_zero_copy(&runner, 0x0000_0400_5000_0000, 64 * 1024)
        .expect("Zero-copy UMA alloc failed");
    let cpu_ptr = alloc.cpu_ptr().unwrap();
    let bda = alloc.host_bda;

    unsafe {
        let input = std::slice::from_raw_parts_mut(cpu_ptr as *mut [f32; 4], 64);
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = [(i + 1) as f32, 2.0, 3.0, 1.0];
        }

        let queue = D3D12CommandQueueX::new_boxed_with_runner(Some(runner.clone()));
        (*queue).register_pipeline(strata_gpu::thin_list::PipelineKey(1), compiled.spirv);
        let list = D3D12CommandListX::new_boxed();
        let lvt = (*list).header.vtable;

        (lvt.set_pipeline_state)(list as *mut _, 1);
        let sgprs = [bda as u32, (bda >> 32) as u32, (bda + 4096) as u32, ((bda + 4096) >> 32) as u32];
        (lvt.set_compute_root_32bit_constants)(list as *mut _, 0, 4, sgprs.as_ptr(), 0);
        (lvt.dispatch)(list as *mut _, 1, 1, 1);
        (lvt.close)(list as *mut _);

        let lists = [list as *mut _];
        ((*queue).header.vtable.execute_command_lists)(queue as *mut _, 1, lists.as_ptr());

        let out = std::slice::from_raw_parts(cpu_ptr.add(4096) as *const [f32; 4], 64);
        for (i, slot) in out.iter().enumerate() {
            let expected = ((i + 1) as f32).mul_add(2.0, 3.0) + 1.0;
            assert!((slot[0] - expected).abs() < 1e-5);
        }
        println!("[PASS] Synthetic D3D12.X COM VTable -> ThinCommandList -> Vulkan BDA sanity check (64/64 lanes)");

        (lvt.release)(list as *mut _);
        ((*queue).header.vtable.release)(queue as *mut _);
    }
    uma.destroy_all(&runner);
}

fn build_divergent_cfg_rdna2_kernel() -> strata_shader::CompiledShader {
    let mut prog = Vec::new();
    prog.push(encode::v_cmp_lt_u32_imm(0, 32));
    prog.push(encode::s_and_saveexec_b64(4, 106));
    prog.push(encode::s_cbranch_execz(1));
    prog.push(encode::v_add_nc_u32_imm(1, 10, 0));
    prog.push(encode::s_xor_b64(126, 126, 4));
    prog.push(encode::s_cbranch_execz(1));
    prog.push(encode::v_add_nc_u32_imm(1, 20, 0));
    prog.push(encode::s_or_b64(126, 126, 4));
    prog.push(encode::v_lshlrev_b32_imm(2, 2, 0));
    prog.extend_from_slice(&encode::global_store_dword(2, 1, 0, 0));
    prog.push(encode::s_endpgm());
    compile_rdna2_to_spirv(&prog, [64, 1, 1]).expect("RDNA2 divergent CFG -> SPIR-V failed")
}

fn build_wave64_ballot_rdna2_kernel() -> strata_shader::CompiledShader {
    let mut prog = Vec::new();
    prog.push(encode::v_and_b32_imm(1, 1, 0));
    prog.push(encode::v_cmp_eq_u32_imm(1, 0));
    prog.push(encode::v_mov_b32_sgpr(2, 106)); // vcc_lo
    prog.push(encode::v_mov_b32_sgpr(3, 107)); // vcc_hi
    prog.push(encode::v_mov_b32_sgpr(4, 128)); // 0
    prog.extend_from_slice(&encode::global_store_dword(4, 2, 0, 0));
    prog.extend_from_slice(&encode::global_store_dword(4, 3, 0, 4));
    prog.push(encode::s_endpgm());
    compile_rdna2_to_spirv(&prog, [64, 1, 1]).expect("RDNA2 Wave64 ballot -> SPIR-V failed")
}

fn build_sample_rdna2_kernel() -> strata_shader::CompiledShader {
    let mut prog = Vec::new();
    prog.push(encode::v_lshlrev_b32_imm(1, 4, 0));
    prog.extend_from_slice(&encode::global_load_dword(2, 1, 0, 0));
    prog.extend_from_slice(&encode::global_load_dword(3, 1, 0, 4));
    prog.extend_from_slice(&encode::global_load_dword(4, 1, 0, 8));
    prog.extend_from_slice(&encode::global_load_dword(5, 1, 0, 12));
    prog.push(encode::s_waitcnt());
    prog.push(encode::v_fmac_f32(4, 2, 3));
    prog.push(encode::v_add_f32(4, 4, 5));
    prog.extend_from_slice(&encode::global_store_dword(1, 4, 2, 0));
    prog.push(encode::s_endpgm());
    compile_rdna2_to_spirv(&prog, [64, 1, 1]).expect("RDNA2 -> SPIR-V failed")
}

fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
