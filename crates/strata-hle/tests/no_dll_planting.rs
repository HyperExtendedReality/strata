//! Anti-DLL-planting regression test: `host_export` must never load a fake
//! system DLL planted in the current working directory.

use strata_hle::loader::host_export;

#[test]
fn host_export_ignores_planted_dll_in_cwd() {
    let dir = std::env::temp_dir().join("strata_no_dll_planting_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Plant a bogus DLL with a non-existent system name in temp CWD.
    let fake_dll = dir.join("strata_planted_probe.dll");
    std::fs::write(&fake_dll, b"not a valid PE image").unwrap();

    let prev = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();
    let sym = host_export("strata_planted_probe.dll", "FakeExport");
    let _ = std::env::set_current_dir(prev);
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(sym, 0, "planted DLL in CWD must never resolve");
}
