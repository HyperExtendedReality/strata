//! Loads the probe title (`src/bin/strata_probe.rs`) with its helper DLL
//! through `load_title` and runs it, as a title would be run.

use strata_hle::stubs::UnimplementedPolicy;
use strata_hle::title::{LoadOptions, load_title};

#[test]
fn probe_title_runs_under_strata() {
    run_probe(false);
}

#[test]
#[ignore = "opens a desktop window and requires a Vulkan GPU"]
fn probe_title_presents_two_planes_through_both_entry_points() {
    run_probe(true);
}

fn run_probe(graphics: bool) {
    // The title loader and capture state are process-wide.
    static PROBE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _serial = PROBE.lock().unwrap_or_else(|e| e.into_inner());
    let dir = std::env::temp_dir().join(format!("strata-probe-{}-{graphics}", std::process::id()));
    let t_drive = dir.join("t-drive");
    std::fs::create_dir_all(&t_drive).unwrap();
    let exe = dir.join("strata_probe.exe");
    std::fs::copy(env!("CARGO_BIN_EXE_strata_probe"), &exe).unwrap();
    std::fs::write(
        dir.join("strata_probe_helper.dll"),
        strata_hle::testpe::probe_helper_dll(),
    )
    .unwrap();

    let opts = LoadOptions {
        unimplemented: UnimplementedPolicy::ReturnZero,
        mounts: vec![('T', t_drive.clone())],
        crash_reporter: Some(Some(dir.join("crash.log"))),
        ..Default::default()
    };
    let title = load_title(&exe, &opts).expect("load probe title");
    let names: Vec<String> = title.modules().iter().map(|m| m.name.clone()).collect();
    assert!(
        names.contains(&"strata_probe_helper.dll".to_string()),
        "{names:?}"
    );
    assert_eq!(
        title.main.image.bind_report.resolved_title, 2,
        "both helper exports bound"
    );

    let mut before = (0, 0);
    if graphics {
        std::fs::write(t_drive.join("probe-graphics"), []).unwrap();
        before = strata_hle::thunks::frame_counts();
        strata_hle::frame_capture::arm(dir.join("frame.scap"), before.0 + 5);
        strata_hle::thunks::enable_window(true);
    }
    let code = title.run().expect("run probe");
    if graphics {
        strata_hle::thunks::enable_window(false);
        let counts = strata_hle::thunks::frame_counts();
        assert_eq!((counts.0 - before.0, counts.1 - before.1), (8, 8));
        let capture = strata_gpu::replay::ExecutableFrame::decode(
            &std::fs::read(dir.join("frame.scap")).unwrap(),
        )
        .unwrap();
        assert_eq!(capture.trace.frame_index, before.0 + 5);
        let presented: Vec<_> = capture
            .trace
            .ops
            .iter()
            .filter_map(|op| match op {
                strata_gpu::replay::CapturedOp::Present { frame_index } => Some(*frame_index),
                _ => None,
            })
            .collect();
        assert_eq!(presented, (before.0 + 1..=before.0 + 5).collect::<Vec<_>>());
        assert_eq!(capture.planes.len(), 2);
        assert_eq!(&capture.output.pixels[12..16], &[32, 128, 0, 255]);
    }
    let failures = std::fs::read_to_string(t_drive.join("probe-failures.txt")).unwrap_or_default();
    assert_eq!(code, 0, "probe checks failed:\n{failures}");
    assert_eq!(std::fs::read(t_drive.join("probe.txt")).unwrap(), b"strata");
    let _ = std::fs::remove_dir_all(&dir);
}
