//! Typed buffer formats and images, on the GPU.
//!
//! - `tbuffer_*` with UNORM/SNORM formats convert like the console does.
//! - Image instructions bind Vulkan images made from T#s in title memory
//!   (through the D3D12.X queue, which reads the descriptors at dispatch),
//!   and storage images are written back to title memory.

// The D3D12.X queue API takes `Arc<ComputeRunner>` (one thread uses it here).
#![allow(clippy::arc_with_non_send_sync)]

use std::sync::{Arc, Mutex};

use strata_gpu::runner::ComputeRunner;
use strata_gpu::texture::{Tiling, retile, source_layout};
use strata_gpu::thin_list::PipelineKey;
use strata_gpu::uma::UmaManager;
use strata_hle::d3d12x::{D3D12CommandListX, D3D12CommandQueueX};
use strata_shader::compile_rdna2_to_spirv;
use strata_shader::descriptor::{ImageDescriptor, ImageKind, Select};
use strata_shader::test_kernels::*;

fn runner() -> Option<ComputeRunner> {
    match ComputeRunner::new() {
        Ok(r) => Some(r),
        Err(_) => {
            eprintln!("no Vulkan device: skipped");
            None
        }
    }
}

fn floats(words: &[u32]) -> Vec<f32> {
    words.iter().map(|w| f32::from_bits(*w)).collect()
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn typed_buffer_loads_and_stores_convert_formats() {
    let Some(runner) = runner() else { return };

    // RGBA8_UNORM: element t = (t, 255 - t, 128, 0).
    let mut input = runner.alloc_bda_buffer(64 * 4).unwrap();
    unsafe {
        let b = input.as_mut_slice::<u8>(256);
        for t in 0..64 {
            b[t * 4..t * 4 + 4].copy_from_slice(&[t as u8, 255 - t as u8, 128, 0]);
        }
    }
    let out = runner.alloc_bda_buffer(64 * 16).unwrap();
    let run = |prog: &[u32], vsharp: [u32; 4]| {
        let spirv = compile_rdna2_to_spirv(prog, [64, 1, 1]).unwrap().spirv;
        let mut push = [0u32; 32];
        push[0] = out.bda as u32;
        push[1] = (out.bda >> 32) as u32;
        push[4..8].copy_from_slice(&vsharp);
        let bytes: Vec<u8> = push.iter().flat_map(|w| w.to_le_bytes()).collect();
        runner.dispatch_spirv(&spirv, &bytes, [1, 1, 1]).unwrap();
        unsafe { out.as_slice::<u32>(256).to_vec() }
    };

    let got = floats(&run(
        &typed_load_rgba8_unorm(),
        buffer_descriptor(input.bda, 4, 64),
    ));
    for t in 0..64usize {
        let want = [
            t as f32 / 255.0,
            (255 - t) as f32 / 255.0,
            128.0 / 255.0,
            0.0,
        ];
        for k in 0..4 {
            assert!(
                close(got[t * 4 + k], want[k]),
                "rgba8 t={t} k={k}: {} vs {}",
                got[t * 4 + k],
                want[k]
            );
        }
    }

    // RG16_SNORM: element t = (-32768 + 1000 t, 32767) -> (max(x/32767, -1), 1, 0, 1).
    unsafe {
        let b = input.as_mut_slice::<u8>(256);
        for t in 0..64 {
            let x = (-32768i32 + 1000 * t as i32) as i16;
            b[t * 4..t * 4 + 2].copy_from_slice(&x.to_le_bytes());
            b[t * 4 + 2..t * 4 + 4].copy_from_slice(&32767i16.to_le_bytes());
        }
    }
    let got = floats(&run(
        &typed_load_rg16_snorm(),
        buffer_descriptor(input.bda, 4, 64),
    ));
    for t in 0..64usize {
        let x = (-32768i32 + 1000 * t as i32) as f32;
        let want = [(x / 32767.0).max(-1.0), 1.0, 0.0, 1.0];
        for k in 0..4 {
            assert!(
                close(got[t * 4 + k], want[k]),
                "rg16 t={t} k={k}: {} vs {}",
                got[t * 4 + k],
                want[k]
            );
        }
    }

    // Typed store of (1.0, 0.5, 0.0, 1.0) as RGBA8_UNORM.
    let target = runner.alloc_bda_buffer(64 * 4).unwrap();
    let spirv = compile_rdna2_to_spirv(&typed_store_rgba8_unorm(), [64, 1, 1])
        .unwrap()
        .spirv;
    let mut push = [0u32; 32];
    push[4..8].copy_from_slice(&buffer_descriptor(target.bda, 4, 64));
    let bytes: Vec<u8> = push.iter().flat_map(|w| w.to_le_bytes()).collect();
    runner.dispatch_spirv(&spirv, &bytes, [1, 1, 1]).unwrap();
    let stored = unsafe { target.as_slice::<u8>(256).to_vec() };
    for t in 0..64 {
        assert_eq!(stored[t * 4..t * 4 + 4], [255, 128, 0, 255], "t={t}");
    }
    runner.free_bda_buffer(input);
    runner.free_bda_buffer(out);
    runner.free_bda_buffer(target);
}

/// R11G11B10 float element: 11-bit (5 exp, 6 mantissa) R and G, 10-bit B.
fn r11g11b10(r: f32, g: f32, b: f32) -> u32 {
    let enc = |v: f32, mbits: u32| -> u32 {
        let bits = v.to_bits();
        let e = (bits >> 23) & 0xff;
        let m = (bits >> (23 - mbits)) & ((1 << mbits) - 1);
        if e < 113 { 0 } else { ((e - 112) << mbits) | m }
    };
    enc(r, 6) | (enc(g, 6) << 11) | (enc(b, 5) << 22)
}

#[test]
fn packed_float_loads_and_sub_dword_stores_run_on_the_gpu() {
    let Some(runner) = runner() else { return };

    // R11G11B10_FLOAT: element t = (t / 8, 2.0, 0.25).
    let mut input = runner.alloc_bda_buffer(64 * 4).unwrap();
    unsafe {
        let w = input.as_mut_slice::<u32>(64);
        for t in 0..64 {
            w[t] = r11g11b10(t as f32 / 8.0, 2.0, 0.25);
        }
    }
    let out = runner.alloc_bda_buffer(64 * 16).unwrap();
    let spirv = compile_rdna2_to_spirv(&typed_load_r11g11b10_float(), [64, 1, 1])
        .unwrap()
        .spirv;
    let mut push = [0u32; 32];
    push[0] = out.bda as u32;
    push[1] = (out.bda >> 32) as u32;
    push[4..8].copy_from_slice(&buffer_descriptor(input.bda, 4, 64));
    let bytes: Vec<u8> = push.iter().flat_map(|w| w.to_le_bytes()).collect();
    runner.dispatch_spirv(&spirv, &bytes, [1, 1, 1]).unwrap();
    let got = floats(unsafe { out.as_slice::<u32>(256) });
    for t in 0..64usize {
        let want = [t as f32 / 8.0, 2.0, 0.25, 1.0];
        for k in 0..4 {
            assert!(
                close(got[t * 4 + k], want[k]),
                "r11g11b10 t={t} k={k}: {} vs {}",
                got[t * 4 + k],
                want[k]
            );
        }
    }

    // 8_UNORM (1 byte) and 16_UNORM (2 bytes) stores: 4 and 2 lanes share each
    // dword; every byte must land.
    for (format, stride, want) in [(1u8, 1usize, vec![128u8]), (7, 2, vec![0x00, 0x80])] {
        let target = runner.alloc_bda_buffer(256).unwrap();
        let spirv = compile_rdna2_to_spirv(&typed_store_half_one(format, 1), [64, 1, 1])
            .unwrap_or_else(|e| panic!("format {format}: {e}"))
            .spirv;
        let mut push = [0u32; 32];
        push[4..8].copy_from_slice(&buffer_descriptor(target.bda, stride as u32, 64));
        let bytes: Vec<u8> = push.iter().flat_map(|w| w.to_le_bytes()).collect();
        runner.dispatch_spirv(&spirv, &bytes, [1, 1, 1]).unwrap();
        let stored = unsafe { target.as_slice::<u8>(256).to_vec() };
        for t in 0..64 {
            let el = &stored[t * stride..t * stride + stride];
            // 0.5 UNORM rounds to 128 (8-bit) / 32768 (16-bit).
            assert!(
                el.iter().zip(&want).all(|(a, b)| a.abs_diff(*b) <= 1),
                "format {format} t={t}: {el:?}"
            );
        }
        runner.free_bda_buffer(target);
    }
    runner.free_bda_buffer(input);
    runner.free_bda_buffer(out);
}

/// Title memory for an 8x8 RGBA8 image: texel (x, y) = (32x, 32y, 7, 255).
fn texel(x: usize, y: usize) -> [u8; 4] {
    [(x * 32) as u8, (y * 32) as u8, 7, 255]
}

fn t_sharp(base: u64, sw_mode: u8, select: [Select; 4]) -> ImageDescriptor {
    ImageDescriptor {
        base_address: base,
        format: 56,
        width: 8,
        height: 8,
        depth: 1,
        kind: ImageKind::Tex2D,
        sw_mode,
        select,
        base_level: 0,
        last_level: 0,
        base_array: 0,
    }
}

const XYZW: [Select; 4] = [Select::X, Select::Y, Select::Z, Select::W];

/// Runs `prog` once (one 64-thread group) through the D3D12.X queue with
/// `sgprs`, returning the 64x4 dwords it wrote to `s[0:1]`.
fn run_through_queue(
    runner: &Arc<ComputeRunner>,
    uma: &Arc<Mutex<UmaManager>>,
    prog: &[u32],
    mut sgprs: [u32; 32],
) -> Vec<u32> {
    let out = runner.alloc_bda_buffer(64 * 16).unwrap();
    sgprs[0] = out.bda as u32;
    sgprs[1] = (out.bda >> 32) as u32;
    unsafe {
        let queue = D3D12CommandQueueX::new_boxed_with_memory(runner.clone(), uma.clone());
        (*queue)
            .register_rdna2_pipeline(PipelineKey(1), prog, [64, 1, 1])
            .expect("translate");
        let list = D3D12CommandListX::new_boxed();
        let vt = (*list).header.vtable;
        let this = list as *mut std::ffi::c_void;
        (vt.set_pipeline_state)(this, 1);
        (vt.set_compute_root_32bit_constants)(this, 0, 32, sgprs.as_ptr(), 0);
        (vt.dispatch)(this, 1, 1, 1);
        (vt.close)(this);
        let lists = [this];
        ((*queue).header.vtable.execute_command_lists)(queue as *mut _, 1, lists.as_ptr());
        assert_eq!(
            (*queue)
                .executed_gpu_dispatches
                .load(std::sync::atomic::Ordering::Relaxed),
            1,
            "dispatch skipped (see stderr)"
        );
        (vt.release)(this);
        ((*queue).header.vtable.release)(queue as *mut _);
    }
    let got = unsafe { out.as_slice::<u32>(256).to_vec() };
    runner.free_bda_buffer(out);
    got
}

#[test]
fn image_instructions_read_and_write_title_memory_textures() {
    let Some(runner) = runner() else { return };
    let runner = Arc::new(runner);
    let uma = Arc::new(Mutex::new(UmaManager::new()));
    let alloc = |size| {
        let mut u = uma.lock().unwrap();
        let a = u.xmem_alloc_identity(&runner, 0, size).ok()?;
        Some((a.console_va, a.cpu_ptr().unwrap()))
    };
    let Some((lin_va, lin)) = alloc(8 * 256) else {
        eprintln!("no VK_EXT_external_memory_host: skipped");
        return;
    };
    let (mort_va, mort) = alloc(8 * 8 * 4).unwrap();
    let (dst_va, dst) = alloc(8 * 256).unwrap();

    // Linear (pitch 256) and Morton copies of the same texels.
    let tight: Vec<u8> = (0..64).flat_map(|i| texel(i % 8, i / 8)).collect();
    for (ptr, mode) in [(lin, 0u8), (mort, 9u8)] {
        let d = t_sharp(0, mode, XYZW);
        let (tiling, footprint) = source_layout(&d, 4).unwrap();
        let mem = unsafe { std::slice::from_raw_parts_mut(ptr, footprint) };
        retile(&d, 4, tiling, &tight, mem);
        assert!(mode != 0 || tiling == Tiling::Linear { pitch: 256 });
    }

    let check_fetch = |got: &[u32], swap_rb: bool, what: &str| {
        let got = floats(got);
        for t in 0..64usize {
            let c = texel(t % 8, t / 8);
            let mut want = [c[0], c[1], c[2], c[3]].map(|b| f32::from(b) / 255.0);
            if swap_rb {
                want.swap(0, 2);
                want[3] = 1.0;
            }
            for k in 0..4 {
                assert!(
                    close(got[t * 4 + k], want[k]),
                    "{what} t={t} k={k}: {} vs {}",
                    got[t * 4 + k],
                    want[k]
                );
            }
        }
    };

    let mut sgprs = [0u32; 32];
    sgprs[8..16].copy_from_slice(&t_sharp(lin_va, 0, XYZW).encode());
    check_fetch(
        &run_through_queue(&runner, &uma, &image_fetch_8x8(), sgprs),
        false,
        "linear fetch",
    );

    sgprs[8..16].copy_from_slice(&t_sharp(mort_va, 9, XYZW).encode());
    check_fetch(
        &run_through_queue(&runner, &uma, &image_fetch_8x8(), sgprs),
        false,
        "morton fetch",
    );

    let zyx1 = [Select::Z, Select::Y, Select::X, Select::One];
    sgprs[8..16].copy_from_slice(&t_sharp(lin_va, 0, zyx1).encode());
    check_fetch(
        &run_through_queue(&runner, &uma, &image_fetch_8x8(), sgprs),
        true,
        "swizzled fetch",
    );

    // Point sampling at (0.3125, 0.3125) of an 8x8 image is texel (2, 2).
    sgprs[8..16].copy_from_slice(&t_sharp(lin_va, 0, XYZW).encode());
    sgprs[16..20].copy_from_slice(&[0; 4]); // wrap, point, no mips
    let got = floats(&run_through_queue(
        &runner,
        &uma,
        &image_sample_lz_fixed(),
        sgprs,
    ));
    let want = texel(2, 2).map(|b| f32::from(b) / 255.0);
    for t in 0..64 {
        for k in 0..4 {
            assert!(close(got[t * 4 + k], want[k]), "sample t={t} k={k}");
        }
    }

    // A storage image written by the shader reaches title memory.
    sgprs[8..16].copy_from_slice(&t_sharp(dst_va, 0, XYZW).encode());
    run_through_queue(&runner, &uma, &image_store_8x8(), sgprs);
    let mem = unsafe { std::slice::from_raw_parts(dst, 8 * 256) };
    for y in 0..8 {
        for x in 0..8 {
            let px = &mem[y * 256 + x * 4..y * 256 + x * 4 + 4];
            assert!(
                px[0] == 255 && (px[1] == 127 || px[1] == 128) && px[2] == 0 && px[3] == 255,
                "({x},{y}) = {px:?}"
            );
        }
    }
    uma.lock().unwrap().destroy_all(&runner);
}
