//! Presenting to a desktop window: a cleared image blitted into a swapchain,
//! across a resize (swapchain recreation). Opens a window for a moment, so
//! it is ignored by default: `cargo test -p strata-cli --test display -- --ignored`.

#[test]
#[ignore = "opens a window"]
fn frames_present_and_survive_a_resize() {
    use ash::vk;
    let Ok(runner) = strata_gpu::runner::ComputeRunner::new() else {
        eprintln!("no Vulkan 1.3 GPU: skipped");
        return;
    };
    if !runner.wsi {
        eprintln!("no window-system integration: skipped");
        return;
    }
    let Some(mut window) = strata_gpu::window::Win32Window::create("Strata display test", 160, 90)
    else {
        eprintln!("no window: skipped");
        return;
    };
    let handle = strata_gpu::swapchain::SurfaceHandle::Win32 {
        hinstance: window.hinstance,
        hwnd: window.hwnd,
    };
    let (w, h) = window.client_size();
    let mut display = strata_gpu::display::Display::new(&runner, handle, w, h).expect("swapchain");
    let frame = runner
        .create_image_2d(
            vk::Format::R8G8B8A8_UNORM,
            320,
            180,
            false,
            vk::ComponentMapping::default(),
        )
        .expect("image");
    runner
        .clear_image(
            &frame,
            [
                0.2f32.to_bits(),
                0.4f32.to_bits(),
                0.8f32.to_bits(),
                1.0f32.to_bits(),
            ],
            0.0,
            0,
        )
        .unwrap();
    for _ in 0..3 {
        assert!(window.pump_messages());
        let (w, h) = window.client_size();
        assert!(display.present(&runner, &frame, w, h).expect("present"));
    }
    window.set_client_size(256, 144);
    for _ in 0..3 {
        window.pump_messages();
        let (w, h) = window.client_size();
        assert!(
            display
                .present(&runner, &frame, w, h)
                .expect("present after resize")
        );
    }
    assert_eq!(
        (display.extent.width, display.extent.height),
        window.client_size()
    );
    assert_eq!(display.frames, 6);
    display.destroy(&runner);
    runner.destroy_image(frame);
}
