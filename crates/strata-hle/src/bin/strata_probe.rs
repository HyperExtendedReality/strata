//! A stand-in title (our own code) that `tests/probe.rs` loads through
//! Strata's title loader. It is never run directly: its console imports only
//! resolve under Strata.
//!
//! Each check prints `ok`/`FAIL`; the failures are also written to
//! `T:\probe-failures.txt`, and the exit code (through the CRT's `exit`, which
//! Strata turns into the end of the title) is the number of failures.

use std::cell::Cell;
use std::ffi::c_void;

#[link(name = "xmem", kind = "raw-dylib")]
unsafe extern "system" {
    fn XMemIsToolsMemAvailable() -> i32;
}

#[link(name = "kernelx", kind = "raw-dylib")]
unsafe extern "system" {
    /// Not in Strata's HLE: reaches the host's kernel export.
    fn GetTickCount64() -> u64;
    /// Exists nowhere: bound to a named stub.
    fn StrataProbeMissing(a: u64) -> u64;
}

#[link(name = "strata_probe_helper", kind = "raw-dylib")]
unsafe extern "system" {
    fn helper_attach_count() -> u32;
    fn helper_thread_attaches() -> u32;
}

#[link(name = "kernel32", kind = "raw-dylib")]
unsafe extern "system" {
    fn GetModuleHandleW(name: *const u16) -> *mut c_void;
    fn GetModuleHandleExW(flags: u32, name: *const u16, out: *mut *mut c_void) -> i32;
    fn GetModuleFileNameW(module: *mut c_void, buf: *mut u16, size: u32) -> u32;
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
}

unsafe extern "C" {
    static __ImageBase: u8;
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain([0]).collect()
}

thread_local! {
    static SLOT: Cell<u32> = const { Cell::new(7) };
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let mut failures: Vec<&str> = Vec::new();
    let mut check = |ok: bool, what: &'static str| {
        println!("[probe] {} {what}", if ok { "ok  " } else { "FAIL" });
        if !ok {
            failures.push(what);
        }
    };

    check(
        unsafe { XMemIsToolsMemAvailable() } == 0,
        "HLE import (xmem.dll)",
    );
    check(
        unsafe { GetTickCount64() } > 0,
        "kernelx falls back to the host",
    );
    check(
        unsafe { helper_attach_count() } == 1,
        "title DLL attached once, before main",
    );

    SLOT.with(|s| s.set(8));
    let other = std::thread::spawn(|| SLOT.with(Cell::get))
        .join()
        .unwrap_or(0);
    check(
        other == 7 && SLOT.with(Cell::get) == 8,
        "static TLS per thread",
    );
    check(
        unsafe { helper_thread_attaches() } >= 1,
        "DLL_THREAD_ATTACH reaches title DLLs",
    );

    let caught = std::panic::catch_unwind(|| panic!("probe")).is_err();
    check(caught, "unwinding through title frames");

    let base = &raw const __ImageBase as *mut c_void;
    check(
        unsafe { GetModuleHandleW(std::ptr::null()) } == base,
        "GetModuleHandle(NULL) is the title",
    );
    let mut buf = [0u16; 260];
    let n = unsafe { GetModuleFileNameW(std::ptr::null_mut(), buf.as_mut_ptr(), 260) } as usize;
    let me = String::from_utf16_lossy(&buf[..n]).to_ascii_lowercase();
    check(
        me.ends_with("strata_probe.exe"),
        "GetModuleFileName(NULL) is the title",
    );

    let d3d = unsafe { LoadLibraryW(wide("d3d12_xs.dll").as_ptr()) };
    let create = unsafe { GetProcAddress(d3d, c"D3D12XboxCreateDevice".as_ptr().cast()) };
    check(
        !d3d.is_null() && !create.is_null(),
        "LoadLibrary/GetProcAddress of a console DLL",
    );

    let helper = unsafe { LoadLibraryW(wide("strata_probe_helper.dll").as_ptr()) };
    let mut owner = std::ptr::null_mut();
    let from_addr = unsafe {
        GetModuleHandleExW(
            4 | 2,
            helper_attach_count as *const () as *const u16,
            &mut owner,
        )
    };
    check(
        !helper.is_null() && from_addr != 0 && owner == helper,
        "LoadLibrary of a title DLL",
    );

    let wrote = std::fs::write(r"T:\probe.txt", b"strata").is_ok();
    let read = std::fs::read(r"T:\probe.txt")
        .map(|b| b == b"strata")
        .unwrap_or(false);
    check(wrote && read, "console drive T: through the VFS");

    check(
        unsafe { StrataProbeMissing(5) } == 0,
        "named stub returns per policy",
    );

    let _ = std::fs::write(r"T:\probe-failures.txt", failures.join("\n"));
    std::process::exit(failures.len() as i32);
}
