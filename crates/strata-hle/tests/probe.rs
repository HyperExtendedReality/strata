//! Loads the probe title (`src/bin/strata_probe.rs`) with its helper DLL
//! through `load_title` and runs it, as a title would be run.

use strata_hle::stubs::UnimplementedPolicy;
use strata_hle::title::{LoadOptions, load_title};

#[test]
fn probe_title_runs_under_strata() {
    let dir = std::env::temp_dir().join(format!("strata-probe-{}", std::process::id()));
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

    let code = title.run().expect("run probe");
    let failures = std::fs::read_to_string(t_drive.join("probe-failures.txt")).unwrap_or_default();
    assert_eq!(code, 0, "probe checks failed:\n{failures}");
    assert_eq!(std::fs::read(t_drive.join("probe.txt")).unwrap(), b"strata");
    let _ = std::fs::remove_dir_all(&dir);
}
