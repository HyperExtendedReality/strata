use std::path::{Path, PathBuf};
use std::time::Instant;

use ash::vk;
use serde_json::{Value, json};
use strata_gpu::display::compose_planes;
use strata_gpu::present::write_atomic;
use strata_gpu::replay::{CapturedOp, ExecutableFrame, FrameCapture, SurfaceSnapshot};
use strata_gpu::runner::ComputeRunner;

pub fn write_report(path: Option<&Path>, report: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(report).map_err(|e| e.to_string())?;
    println!("{}", String::from_utf8_lossy(&bytes));
    if let Some(path) = path {
        write_atomic(path, &bytes).map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn percentile(samples: &[f64], p: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(sorted[((p * sorted.len() as f64).ceil() as usize).saturating_sub(1)])
}

pub fn command() -> Result<(), String> {
    let mut args = std::env::args().skip(2);
    let mut report_path = None;
    let mut capture = None;
    let mut window = false;
    let mut require_gpu = false;
    let mut cache = None;
    let mut frames = 8;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--window" => window = true,
            "--require-gpu" => require_gpu = true,
            "--frames" => {
                frames = args.next().and_then(|value| value.parse().ok())
                    .filter(|&count| (1..=1_000_000).contains(&count))
                    .ok_or("--frames requires a value between 1 and 1000000")?;
            }
            "--report" | "--capture" | "--pipeline-cache-dir" => {
                let path = PathBuf::from(args.next().ok_or_else(|| format!("{arg} needs a path"))?);
                match arg.as_str() {
                    "--report" => report_path = Some(path),
                    "--capture" => capture = Some(path),
                    _ => cache = Some(path),
                }
            }
            _ => return Err(format!("unknown smoke option {arg}")),
        }
    }
    let start = Instant::now();
    let runner = match ComputeRunner::new() {
        Ok(runner) => runner,
        Err(e) => {
            write_report(
                report_path.as_deref(),
                &json!({"status": "skipped", "reason": e.to_string()}),
            )?;
            return if require_gpu || window || capture.is_some() {
                Err(e.to_string())
            } else {
                Ok(())
            };
        }
    };
    if let Some(cache) = cache {
        runner.enable_pipeline_cache(&cache, true);
    }
    let report = synthetic(&runner, start, window, capture.as_deref(), frames);
    match report {
        Ok(report) => {
            runner.flush_pipeline_cache().map_err(|e| e.to_string())?;
            write_report(report_path.as_deref(), &report)
        }
        Err(error) => {
            write_report(
                report_path.as_deref(),
                &json!({"status": "failed", "reason": error}),
            )?;
            Err(error)
        }
    }
}

fn synthetic(
    runner: &ComputeRunner,
    start: Instant,
    visible: bool,
    capture: Option<&Path>,
    frame_count: usize,
) -> Result<Value, String> {
    let gpu = |e: strata_gpu::context::GpuError| e.to_string();
    let format = vk::Format::R8G8B8A8_UNORM;
    let make = |w, h| {
        runner
            .create_image_2d(format, w, h, false, Default::default())
            .map_err(gpu)
    };
    let scene = make(2, 2)?;
    let overlay = make(8, 4)?;
    let output = make(8, 4)?;
    let mut hud = vec![0; 8 * 4 * 4];
    for y in 0..4 {
        hud[(y * 8 + 3) * 4..(y * 8 + 4) * 4].copy_from_slice(&[0, 128, 0, 128]);
    }
    runner
        .upload_image(
            &scene,
            &[
                [64, 0, 0, 255],
                [64, 0, 0, 255],
                [192, 0, 0, 255],
                [192, 0, 0, 255],
            ]
            .concat(),
        )
        .map_err(gpu)?;
    runner.upload_image(&overlay, &hud).map_err(gpu)?;
    let planes = [&scene, &overlay];
    let mut visible_window = if visible {
        Some(Window::open(runner)?)
    } else {
        None
    };
    let mut frame_times = Vec::new();
    let mut visible_first = None;
    let mut first_output = None;
    for _ in 0..frame_count {
        let frame_start = Instant::now();
        compose_planes(runner, &planes, &output).map_err(gpu)?;
        if first_output.is_none() {
            first_output = Some(start.elapsed().as_secs_f64() * 1000.0);
        }
        if let Some(window) = &mut visible_window
            && window.present(runner, &output)?
            && visible_first.is_none()
        {
            visible_first = Some(start.elapsed().as_secs_f64() * 1000.0);
        }
        frame_times.push(frame_start.elapsed().as_secs_f64() * 1000.0);
    }
    let snap = SurfaceSnapshot::capture(runner, &output).map_err(gpu)?;
    let pixel = |x: usize, y: usize| &snap.pixels[(y * 8 + x) * 4..(y * 8 + x + 1) * 4];
    if pixel(2, 0) != [64, 0, 0, 255]
        || pixel(3, 0) != [32, 128, 0, 255]
        || pixel(4, 0) != [64, 0, 0, 255]
        || pixel(2, 3) != [192, 0, 0, 255]
    {
        return Err(format!(
            "composition pixels differ: {:?}, {:?}, {:?}, {:?}",
            pixel(2, 0),
            pixel(3, 0),
            pixel(4, 0),
            pixel(2, 3)
        ));
    }
    let mut trace = FrameCapture {
        frame_index: 1,
        ..Default::default()
    };
    trace.record(CapturedOp::Present { frame_index: 1 });
    let frame = ExecutableFrame {
        trace,
        planes: planes
            .iter()
            .map(|image| SurfaceSnapshot::capture(runner, image))
            .collect::<Result<_, _>>()
            .map_err(gpu)?,
        output: snap,
    };
    let bytes = frame.encode().map_err(gpu)?;
    let replay = ExecutableFrame::decode(&bytes)
        .ok_or("capture round-trip failed")?
        .replay(runner)
        .map_err(gpu)?;
    if replay.diff.differing_pixels != 0 {
        return Err("GPU replay diverged".into());
    }
    if let Some(path) = capture {
        write_atomic(path, &bytes).map_err(|e| e.to_string())?;
    }
    let mut skipped = Vec::new();
    if !visible {
        skipped.push("desktop presentation requires --window");
    }
    test_wave64_ballot(runner)?;
    if !test_wave_boundary(runner)? {
        skipped.push("native compute Wave64 shuffle unavailable");
    }
    let ray_query = if runner.ray_query {
        Some(ray_query_parity(runner)?)
    } else {
        skipped.push("Vulkan ray query is unavailable");
        None
    };
    let shown = visible_window.map_or(0, |window| window.finish(runner));
    if visible && shown == 0 {
        return Err("the compositor produced frames but the window showed none".into());
    }
    for image in [scene, overlay, output] {
        runner.destroy_image(image);
    }
    let stages = runner.native_wave64_stages;
    let pipeline_stats = runner.pipeline_stats();
    Ok(json!({
        "status": "passed", "workload": "scaled scene and one-pixel premultiplied overlay",
        "adapter": runner.device_name, "driver_version_raw": runner.driver_version,
        "image_slots": runner.image_slots, "subgroup_size": runner.subgroup_size,
        "wave64_ballot_verified": true,
        "native_wave64_shuffle": { "compute": stages.contains(vk::ShaderStageFlags::COMPUTE),
            "vertex": stages.contains(vk::ShaderStageFlags::VERTEX), "fragment": stages.contains(vk::ShaderStageFlags::FRAGMENT),
            "mesh": stages.contains(vk::ShaderStageFlags::MESH_EXT) },
        "frames_composed": frame_times.len(), "present_calls": if visible { frame_times.len() } else { 0 },
        "frames_shown": shown, "first_output_ms": first_output, "first_visible_frame_ms": visible_first,
        "frame_ms_p50": percentile(&frame_times, 0.50), "frame_ms_p95": percentile(&frame_times, 0.95),
        "frame_ms_p99": percentile(&frame_times, 0.99), "frame_ms_p999": percentile(&frame_times, 0.999),
        "ray_query_parity": ray_query,
        "replay_different_pixels": replay.diff.differing_pixels, "replay_compose_and_readback_ms": replay.compose_and_readback_ms,
        "replay_planes": replay.plane_timings.iter().map(|pass| json!({"wall_ms": pass.wall_ms, "gpu_ms": pass.gpu_ms})).collect::<Vec<_>>(),
        "driver_cache_loaded": pipeline_stats.driver_cache_loaded, "prewarmed": pipeline_stats.prewarmed,
        "warmup_ms": pipeline_stats.warmup_ms, "pipeline_stall_ms": pipeline_stats.creation_ms,
        "skipped": skipped,
    }))
}

fn ray_query_parity(runner: &ComputeRunner) -> Result<Value, String> {
    use strata_shader::bvh::{Ray, intersect_triangle};

    let triangle = |z| [[0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]];
    let triangles = [triangle(4.0), triangle(2.0)];
    let rays = [([0.25, 0.125, 0.0], 10.0), ([2.0, 1.0, 0.0], 10.0), ([0.25, 0.125, 0.0], 1.0)];
    let mut timings = Vec::new();
    for (origin, t_max) in rays {
        let ray = Ray { origin, t_max, dir: [0.0, 0.0, 1.0], inv_dir: [f32::INFINITY, f32::INFINITY, 1.0] };
        let start = Instant::now();
        let expected = triangles.iter().enumerate().filter_map(|(index, &[a, b, c])| {
            let [num, den, u, v] = intersect_triangle(&ray, a, b, c);
            num.is_finite().then_some((index as u32, num / den, [u / den, v / den]))
        }).min_by(|a, b| a.1.total_cmp(&b.1));
        let software_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let (actual, timing) = runner.query_opaque_triangles_timed(&triangles, origin, ray.dir, t_max)
            .map_err(|error| error.to_string())?;
        let matches = match (expected, actual) {
            (None, None) => true,
            (Some((primitive, t, uv)), Some(hit)) => primitive == hit.primitive
                && (t - hit.t).abs() < 1e-5
                && uv.into_iter().zip(hit.barycentrics).all(|(a, b)| (a - b).abs() < 1e-5),
            _ => false,
        };
        if !matches {
            return Err(format!("ray query parity for {origin:?}, tMax={t_max}: CPU {expected:?}, GPU {actual:?}"));
        }
        timings.push(json!({
            "hit": actual.is_some(), "software_ms": software_ms,
            "as_build_wall_ms": timing.build.as_secs_f64() * 1_000.0,
            "trace_wall_ms": timing.trace.as_secs_f64() * 1_000.0,
        }));
    }
    Ok(json!(timings))
}

fn test_wave64_ballot(runner: &ComputeRunner) -> Result<(), String> {
    let shader = super::build_wave64_ballot_rdna2_kernel();
    let buffer = runner.alloc_bda_buffer(8).map_err(|e| e.to_string())?;
    let result = runner.dispatch_spirv(&shader.spirv, &buffer.bda.to_le_bytes(), [1, 1, 1]);
    let mask = unsafe { buffer.as_slice::<u64>(1)[0] };
    runner.free_bda_buffer(buffer);
    result.map_err(|e| e.to_string())?;
    if mask != 0x5555_5555_5555_5555 {
        return Err(format!(
            "Wave64 ballot across the subgroup boundary returned {mask:016x}"
        ));
    }
    Ok(())
}

fn test_wave_boundary(runner: &ComputeRunner) -> Result<bool, String> {
    use strata_shader::rdna2::encode::*;
    if !runner
        .native_wave64_stages
        .contains(vk::ShaderStageFlags::COMPUTE)
    {
        return Ok(false);
    }
    let mut program = vec![
        v_add_nc_u32_imm(1, 32, 0),
        v_and_b32_imm(1, 63, 1),
        v_lshlrev_b32_imm(1, 2, 1),
    ];
    program.extend(ds_bpermute_b32(2, 1, 0));
    program.push(v_lshlrev_b32_imm(3, 2, 0));
    program.extend(global_store_dword(3, 2, 0, 0));
    program.push(s_endpgm());
    let shader =
        strata_shader::compile_rdna2_to_spirv(&program, [64, 1, 1]).map_err(|e| e.to_string())?;
    let output = runner.alloc_bda_buffer(256).map_err(|e| e.to_string())?;
    let req = (runner.subgroup_size != 64).then_some(64);
    let dispatch = runner.dispatch_spirv_with_subgroup_size(
        &shader.spirv,
        &output.bda.to_le_bytes(),
        [1, 1, 1],
        req,
    );
    let lanes = unsafe { output.as_slice::<u32>(64) }.to_vec();
    runner.free_bda_buffer(output);
    dispatch.map_err(|e| e.to_string())?;
    if let Some((lane, actual)) = lanes
        .iter()
        .enumerate()
        .find(|&(lane, &actual)| actual != (lane as u32 ^ 32))
    {
        return Err(format!(
            "Wave64 cross-half lane {lane} returned {actual}, expected {}",
            lane ^ 32
        ));
    }
    Ok(true)
}

struct Window {
    window: strata_gpu::window::Win32Window,
    display: strata_gpu::display::Display,
}

impl Window {
    fn open(runner: &ComputeRunner) -> Result<Self, String> {
        let window = strata_gpu::window::Win32Window::create("Strata smoke", 160, 90)
            .ok_or("cannot open the smoke window")?;
        let (width, height) = window.client_size();
        let handle = strata_gpu::swapchain::SurfaceHandle::Win32 {
            hinstance: window.hinstance,
            hwnd: window.hwnd,
        };
        let display = strata_gpu::display::Display::new(runner, handle, width, height)
            .map_err(|e| e.to_string())?;
        Ok(Self { window, display })
    }

    fn present(
        &mut self,
        runner: &ComputeRunner,
        frame: &strata_gpu::images::GpuImage,
    ) -> Result<bool, String> {
        if !self.window.pump_messages() {
            return Err("smoke window closed".into());
        }
        let (width, height) = self.window.client_size();
        self.display
            .present(runner, frame, width, height)
            .map_err(|e| e.to_string())
    }

    fn finish(self, runner: &ComputeRunner) -> u64 {
        let frames = self.display.frames;
        self.display.destroy(runner);
        frames
    }
}
