//! The title-facing D3D12 ABI end to end on a live GPU: a device from
//! `StandardDevice::new_boxed`, a root signature from its serialized form, a
//! graphics PSO from raw RDNA2 vertex/pixel code, a render target in an RTV
//! heap, clears, viewport and scissor state, a draw, and the result read
//! back through `Map` — every call through the `d3d12.h` vtable slots.

use std::ffi::c_void;

use strata_hle::d3d12_abi::*;
use strata_hle::d3d12_types::*;
use strata_shader::test_kernels::{fullscreen_triangle_vs, interpolated_color_ps};

unsafe fn f<T: Copy>(p: *mut c_void, slot: usize) -> T {
    unsafe {
        let vt = *(p as *const *const VTableSlot);
        std::mem::transmute_copy(&(*vt.add(slot)).0)
    }
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The 8x8 RGBA8 target's pixels (rows 256 bytes apart).
unsafe fn pixels(base: *const u8) -> Vec<[u8; 4]> {
    let mut v = Vec::new();
    for y in 0..8 {
        for x in 0..8 {
            let p = unsafe { base.add(y * 256 + x * 4) };
            v.push(unsafe { std::ptr::read(p as *const [u8; 4]) });
        }
    }
    v
}

fn near(a: [u8; 4], b: [u8; 4]) -> bool {
    a.iter().zip(&b).all(|(x, y)| x.abs_diff(*y) <= 2)
}

#[test]
fn a_title_draws_through_the_d3d12_vtables() {
    let dev = StandardDevice::new_boxed() as *mut c_void;
    if unsafe { (*(dev as *const StandardDevice)).inner.backend.is_none() } {
        eprintln!("no Vulkan 1.3 GPU: skipped");
        return;
    }
    unsafe {
        // Root signature: no parameters.
        let desc: [u64; 5] = [0, 0, 0, 0, 0];
        let blob = serialize_root_signature(desc.as_ptr() as *const u8, 1);
        let create_rs: unsafe extern "system" fn(
            *mut c_void,
            u32,
            *const u8,
            usize,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = f(dev, DEVICE_SLOT_CREATE_ROOT_SIGNATURE);
        let mut rs = std::ptr::null_mut();
        assert_eq!(
            create_rs(dev, 0, blob.as_ptr(), blob.len(), std::ptr::null(), &mut rs),
            S_OK
        );

        // Graphics PSO from raw RDNA2 code.
        let vs = bytes(&fullscreen_triangle_vs());
        let ps = bytes(&interpolated_color_ps());
        let mut pd: GraphicsPipelineStateDesc = std::mem::zeroed();
        pd.root_signature = rs;
        pd.vs = ShaderBytecode {
            ptr: vs.as_ptr(),
            len: vs.len(),
        };
        pd.ps = ShaderBytecode {
            ptr: ps.as_ptr(),
            len: ps.len(),
        };
        pd.blend.render_target[0].write_mask = 0xf;
        pd.sample_mask = u32::MAX;
        pd.rasterizer.fill_mode = 3;
        pd.rasterizer.cull_mode = 1;
        pd.rasterizer.depth_clip_enable = 1;
        pd.primitive_topology_type = 3;
        pd.num_render_targets = 1;
        pd.rtv_formats[0] = 28; // R8G8B8A8_UNORM
        pd.sample_count = 1;
        let create_pso: unsafe extern "system" fn(
            *mut c_void,
            *const GraphicsPipelineStateDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = f(dev, DEVICE_SLOT_CREATE_GRAPHICS_PSO);
        let mut pso = std::ptr::null_mut();
        assert_eq!(create_pso(dev, &pd, std::ptr::null(), &mut pso), S_OK);

        // An 8x8 render target and its RTV.
        let rd = ResourceDesc {
            dimension: DIMENSION_TEXTURE2D,
            width: 8,
            height: 8,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 28,
            sample_count: 1,
            flags: 1, // ALLOW_RENDER_TARGET
            ..Default::default()
        };
        let create_res: unsafe extern "system" fn(
            *mut c_void,
            *const HeapProperties,
            u32,
            *const ResourceDesc,
            u32,
            *const c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = f(dev, DEVICE_SLOT_CREATE_COMMITTED_RESOURCE);
        let mut rt = std::ptr::null_mut();
        assert_eq!(
            create_res(
                dev,
                &HeapProperties::default(),
                0,
                &rd,
                0,
                std::ptr::null(),
                std::ptr::null(),
                &mut rt
            ),
            S_OK
        );
        let create_heap: unsafe extern "system" fn(
            *mut c_void,
            *const DescriptorHeapDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = f(dev, DEVICE_SLOT_CREATE_DESCRIPTOR_HEAP);
        let mut heap = std::ptr::null_mut();
        let hd = DescriptorHeapDesc {
            heap_type: HEAP_TYPE_RTV,
            num_descriptors: 1,
            ..Default::default()
        };
        assert_eq!(create_heap(dev, &hd, std::ptr::null(), &mut heap), S_OK);
        let cpu_start: unsafe extern "system" fn(*mut c_void, *mut u64) -> *mut u64 = f(heap, 9);
        let mut rtv = 0u64;
        cpu_start(heap, &mut rtv);
        let create_rtv: unsafe extern "system" fn(
            *mut c_void,
            *mut c_void,
            *const RenderTargetViewDesc,
            u64,
        ) = f(dev, DEVICE_SLOT_CREATE_RTV);
        create_rtv(dev, rt, std::ptr::null(), rtv);

        // Queue and list.
        let create_q: unsafe extern "system" fn(
            *mut c_void,
            *const D3D12CommandQueueDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = f(dev, DEVICE_SLOT_CREATE_COMMAND_QUEUE);
        let mut q = std::ptr::null_mut();
        assert_eq!(
            create_q(
                dev,
                &D3D12CommandQueueDesc::default(),
                std::ptr::null(),
                &mut q
            ),
            S_OK
        );
        let create_cl: unsafe extern "system" fn(
            *mut c_void,
            u32,
            u32,
            *mut c_void,
            *mut c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = f(dev, DEVICE_SLOT_CREATE_COMMAND_LIST);
        let mut list = std::ptr::null_mut();
        assert_eq!(
            create_cl(
                dev,
                0,
                0,
                std::ptr::null_mut(),
                pso,
                std::ptr::null(),
                &mut list
            ),
            S_OK
        );

        let set_rs: unsafe extern "system" fn(*mut c_void, *mut c_void) =
            f(list, CMDLIST_SLOT_SET_GRAPHICS_ROOT_SIGNATURE);
        let set_vp: unsafe extern "system" fn(*mut c_void, u32, *const Viewport) =
            f(list, CMDLIST_SLOT_RS_SET_VIEWPORTS);
        let set_sc: unsafe extern "system" fn(*mut c_void, u32, *const Rect) =
            f(list, CMDLIST_SLOT_RS_SET_SCISSOR_RECTS);
        let set_rt: unsafe extern "system" fn(*mut c_void, u32, *const u64, i32, *const u64) =
            f(list, CMDLIST_SLOT_OM_SET_RENDER_TARGETS);
        let clear: unsafe extern "system" fn(*mut c_void, u64, *const f32, u32, *const Rect) =
            f(list, CMDLIST_SLOT_CLEAR_RENDER_TARGET_VIEW);
        let draw: unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32) =
            f(list, CMDLIST_SLOT_DRAW_INSTANCED);
        let close: unsafe extern "system" fn(*mut c_void) -> i32 = f(list, CMDLIST_SLOT_CLOSE);
        let reset: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> i32 =
            f(list, CMDLIST_SLOT_RESET);
        let exec: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) =
            f(q, QUEUE_SLOT_EXECUTE_COMMAND_LISTS);
        let map: unsafe extern "system" fn(
            *mut c_void,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = f(rt, RESOURCE_SLOT_MAP);
        let mut mapped = std::ptr::null_mut();
        assert_eq!(map(rt, 0, std::ptr::null(), &mut mapped), S_OK);

        // 1. A clear alone: every pixel blue.
        set_rt(list, 1, &rtv, 0, std::ptr::null());
        clear(
            list,
            rtv,
            [0.0f32, 0.0, 1.0, 1.0].as_ptr(),
            0,
            std::ptr::null(),
        );
        assert_eq!(close(list), S_OK);
        exec(q, 1, &list);
        for (i, px) in pixels(mapped as *const u8).into_iter().enumerate() {
            assert!(near(px, [0, 0, 255, 255]), "clear pixel {i}: {px:?}");
        }

        // 2. Clear red, then a fullscreen triangle writing param0 everywhere.
        assert_eq!(reset(list, std::ptr::null_mut(), pso), S_OK);
        set_rs(list, rs);
        let vp = Viewport {
            width: 8.0,
            height: 8.0,
            max_depth: 1.0,
            ..Default::default()
        };
        set_vp(list, 1, &vp);
        set_sc(
            list,
            1,
            &Rect {
                left: 0,
                top: 0,
                right: 8,
                bottom: 8,
            },
        );
        set_rt(list, 1, &rtv, 0, std::ptr::null());
        clear(
            list,
            rtv,
            [1.0f32, 0.0, 0.0, 1.0].as_ptr(),
            0,
            std::ptr::null(),
        );
        draw(list, 3, 1, 0, 0);
        assert_eq!(close(list), S_OK);
        exec(q, 1, &list);
        let param0 = [64u8, 128, 191, 255]; // (0.25, 0.5, 0.75, 1.0)
        for (i, px) in pixels(mapped as *const u8).into_iter().enumerate() {
            assert!(near(px, param0), "draw pixel {i}: {px:?}");
        }

        // 3. A scissor of the left half: the right half keeps the clear.
        assert_eq!(reset(list, std::ptr::null_mut(), pso), S_OK);
        set_rs(list, rs);
        set_vp(list, 1, &vp);
        set_sc(
            list,
            1,
            &Rect {
                left: 0,
                top: 0,
                right: 4,
                bottom: 8,
            },
        );
        set_rt(list, 1, &rtv, 0, std::ptr::null());
        clear(
            list,
            rtv,
            [0.0f32, 1.0, 0.0, 1.0].as_ptr(),
            0,
            std::ptr::null(),
        );
        draw(list, 3, 1, 0, 0);
        assert_eq!(close(list), S_OK);
        exec(q, 1, &list);
        for (i, px) in pixels(mapped as *const u8).into_iter().enumerate() {
            let want = if i % 8 < 4 { param0 } else { [0, 255, 0, 255] };
            assert!(near(px, want), "scissored pixel {i}: {px:?}");
        }

        let rel: unsafe extern "system" fn(*mut c_void) -> u32 = f(dev, SLOT_RELEASE);
        for o in [list, q, heap, rt, pso, rs] {
            let r: unsafe extern "system" fn(*mut c_void) -> u32 = f(o, SLOT_RELEASE);
            r(o);
        }
        assert_eq!(rel(dev), 0);
    }
}
