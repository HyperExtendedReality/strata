//! Draws with translated RDNA2 vertex and pixel shaders through the D3D12.X
//! queue: `EXP` positions and parameters, `v_interp`, colour exports, images
//! in pixel shaders and quad broadcasts, rendered into a render target in
//! title memory and written back there.

// The D3D12.X queue API takes `Arc<ComputeRunner>` (one thread uses it here).
#![allow(clippy::arc_with_non_send_sync)]

use std::sync::{Arc, Mutex};

use strata_gpu::runner::ComputeRunner;
use strata_gpu::texture::{retile, source_layout};
use strata_gpu::thin_list::{PipelineKey, RenderTargetDesc};
use strata_gpu::uma::UmaManager;
use strata_hle::d3d12x::{D3D12CommandListX, D3D12CommandQueueX};
use strata_shader::descriptor::{ImageDescriptor, ImageKind, Select};
use strata_shader::test_kernels::*;

fn image(base: u64) -> ImageDescriptor {
    ImageDescriptor {
        base_address: base,
        format: 56, // 8_8_8_8_UNORM
        width: 8,
        height: 8,
        depth: 1,
        kind: ImageKind::Tex2D,
        sw_mode: 0,
        select: [Select::X, Select::Y, Select::Z, Select::W],
        base_level: 0,
        last_level: 0,
        base_array: 0,
    }
}

/// Draws one fullscreen triangle with `ps` into the 8x8 render target at
/// `rt_va`; returns the queue's draw count.
fn draw(
    runner: &Arc<ComputeRunner>,
    uma: &Arc<Mutex<UmaManager>>,
    ps: &[u32],
    rt_va: u64,
    sgprs: [u32; 32],
) -> u32 {
    unsafe {
        let queue = D3D12CommandQueueX::new_boxed_with_memory(runner.clone(), uma.clone());
        (*queue).register_render_target(image(rt_va));
        (*queue)
            .register_rdna2_graphics_pipeline(PipelineKey(9), &fullscreen_triangle_vs(), ps)
            .expect("translate");
        let list = D3D12CommandListX::new_boxed();
        let vt = (*list).header.vtable;
        let this = list as *mut std::ffi::c_void;
        (vt.set_pipeline_state)(this, 9);
        (vt.set_compute_root_32bit_constants)(this, 0, 32, sgprs.as_ptr(), 0);
        let rt = RenderTargetDesc {
            console_va: rt_va,
            format_is_integer: false,
            blend_enabled: false,
        };
        (vt.om_set_render_targets)(this, 1, &rt);
        (vt.draw_instanced)(this, 3, 1, 0, 0);
        (vt.close)(this);
        let lists = [this];
        ((*queue).header.vtable.execute_command_lists)(queue as *mut _, 1, lists.as_ptr());
        let n = (*queue)
            .executed_draws
            .load(std::sync::atomic::Ordering::Relaxed);
        (vt.release)(this);
        ((*queue).header.vtable.release)(queue as *mut _);
        n
    }
}

fn pixels(ptr: *const u8) -> Vec<[u8; 4]> {
    let mem = unsafe { std::slice::from_raw_parts(ptr, 8 * 256) };
    (0..64)
        .map(|i| {
            let (x, y) = (i % 8, i / 8);
            mem[y * 256 + x * 4..y * 256 + x * 4 + 4]
                .try_into()
                .unwrap()
        })
        .collect()
}

fn near(a: [u8; 4], b: [u8; 4]) -> bool {
    a.iter().zip(b).all(|(&x, y)| x.abs_diff(y) <= 1)
}

#[test]
fn translated_vertex_and_pixel_shaders_draw_into_title_memory() {
    let Ok(runner) = ComputeRunner::new() else {
        eprintln!("no Vulkan device: skipped");
        return;
    };
    let runner = Arc::new(runner);
    let uma = Arc::new(Mutex::new(UmaManager::new()));
    let alloc = |size| {
        let mut u = uma.lock().unwrap();
        let a = u.xmem_alloc_identity(&runner, 0, size).ok()?;
        Some((a.console_va, a.cpu_ptr().unwrap()))
    };
    let Some((rt_va, rt)) = alloc(8 * 256) else {
        eprintln!("no VK_EXT_external_memory_host: skipped");
        return;
    };
    let param0 = [64u8, 128, 191, 255]; // (0.25, 0.5, 0.75, 1.0)

    for (name, ps) in [
        ("interpolated", interpolated_color_ps()),
        ("flat", flat_color_ps()),
    ] {
        unsafe { std::ptr::write_bytes(rt, 0, 8 * 256) };
        assert_eq!(
            draw(&runner, &uma, &ps, rt_va, [0; 32]),
            1,
            "{name}: draw skipped (see stderr)"
        );
        for (i, px) in pixels(rt).into_iter().enumerate() {
            assert!(near(px, param0), "{name} pixel {i}: {px:?}");
        }
    }

    // Sampling in a pixel shader (implicit LOD) at param0.xy = (0.25, 0.5):
    // texel (2, 4) of an 8x8 texture, whatever the pixel.
    let (tex_va, tex) = alloc(8 * 256).unwrap();
    let d = image(tex_va);
    let (tiling, footprint) = source_layout(&d, 4).unwrap();
    let tight: Vec<u8> = (0..64)
        .flat_map(|i| [(i % 8 * 30) as u8, (i / 8 * 30) as u8, 9, 255])
        .collect();
    retile(&d, 4, tiling, &tight, unsafe {
        std::slice::from_raw_parts_mut(tex, footprint)
    });
    let mut sgprs = [0u32; 32];
    sgprs[8..16].copy_from_slice(&d.encode());
    unsafe { std::ptr::write_bytes(rt, 0, 8 * 256) };
    assert_eq!(
        draw(&runner, &uma, &sampled_color_ps(), rt_va, sgprs),
        1,
        "sampled: draw skipped"
    );
    for (i, px) in pixels(rt).into_iter().enumerate() {
        assert_eq!(px, [60, 120, 9, 255], "sampled pixel {i}");
    }

    // A quad broadcast of param0.x: (0.25, 0.25, 0, 1).
    unsafe { std::ptr::write_bytes(rt, 0, 8 * 256) };
    assert_eq!(
        draw(&runner, &uma, &quad_broadcast_ps(), rt_va, [0; 32]),
        1,
        "quad: draw skipped"
    );
    for (i, px) in pixels(rt).into_iter().enumerate() {
        assert!(near(px, [64, 64, 0, 255]), "quad pixel {i}: {px:?}");
    }
    eprintln!("quad control: {}", runner.quad_control);
    uma.lock().unwrap().destroy_all(&runner);
}
