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

/// Create two title resources using the console ABI, then exercise both
/// PresentX entry points. The host test checks which frames reached Vulkan.
unsafe fn graphics_workload(d3d: *mut c_void, create: *mut c_void) -> bool {
    use std::ptr::{null, null_mut};
    use strata_hle::d3d12_abi::*;
    use strata_hle::d3d12_types::*;

    unsafe fn slot<T: Copy>(object: *mut c_void, index: usize) -> T {
        let vtable = *object.cast::<*const VTableSlot>();
        std::mem::transmute_copy(&(*vtable.add(index)).0)
    }

    #[repr(C)]
    struct Plane {
        token: u64,
        resource_count: u32,
        reserved: u32,
        resources: *const *mut c_void,
        options: [u64; 6],
    }
    const _: () = assert!(size_of::<Plane>() == 72);
    type Present =
        unsafe extern "system" fn(u32, *const Plane, *const c_void, *const c_void) -> i32;
    type QueuePresent =
        unsafe extern "system" fn(*mut c_void, u32, *const Plane, *const c_void) -> i32;
    let show = unsafe { GetProcAddress(d3d, c"PresentX".as_ptr().cast()) };
    if create.is_null() || show.is_null() {
        return false;
    }
    let present: Present = unsafe { std::mem::transmute(show) };
    let create_device: unsafe extern "system" fn(
        *const c_void,
        *const c_void,
        *mut *mut c_void,
    ) -> i32 = unsafe { std::mem::transmute(create) };
    let mut device = null_mut();
    if create_device(null(), null(), &mut device) != S_OK || device.is_null() {
        return false;
    }

    let create_texture: unsafe extern "system" fn(
        *mut c_void,
        *const HeapProperties,
        u32,
        *const ResourceDesc,
        u32,
        *const c_void,
        *const c_void,
        *mut *mut c_void,
    ) -> i32 = slot(device, DEVICE_SLOT_CREATE_COMMITTED_RESOURCE);
    let mut textures = [null_mut(); 2];
    for (index, resource) in textures.iter_mut().enumerate() {
        let width = if index == 0 { 2 } else { 8 };
        let desc = ResourceDesc {
            dimension: DIMENSION_TEXTURE2D,
            width,
            height: 4,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 28,
            sample_count: 1,
            ..Default::default()
        };
        if create_texture(
            device,
            &HeapProperties::default(),
            0,
            &desc,
            0,
            null(),
            null(),
            resource,
        ) != S_OK
            || resource.is_null()
        {
            return false;
        }
        let map: unsafe extern "system" fn(
            *mut c_void,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = slot(*resource, RESOURCE_SLOT_MAP);
        let mut mapped = null_mut();
        if map(*resource, 0, null(), &mut mapped) != S_OK || mapped.is_null() {
            return false;
        }
        for y in 0..4 {
            for x in 0..width as usize {
                let pixel = if index == 0 {
                    [64, 0, 0, 255]
                } else if x == 3 {
                    [0, 128, 0, 128]
                } else {
                    [0; 4]
                };
                mapped
                    .cast::<u8>()
                    .add(y * 256 + x * 4)
                    .copy_from_nonoverlapping(pixel.as_ptr(), 4);
            }
        }
    }
    let create_queue: unsafe extern "system" fn(
        *mut c_void,
        *const D3D12CommandQueueDesc,
        *const c_void,
        *mut *mut c_void,
    ) -> i32 = slot(device, DEVICE_SLOT_CREATE_COMMAND_QUEUE);
    let mut queue = null_mut();
    if create_queue(
        device,
        &D3D12CommandQueueDesc::default(),
        null(),
        &mut queue,
    ) != S_OK
        || queue.is_null()
    {
        return false;
    }
    let queue_present: QueuePresent = slot(queue, QUEUE_SLOT_PRESENT_X);
    let planes = textures.each_ref().map(|resource| Plane {
        token: 0,
        resource_count: 1,
        reserved: 0,
        resources: resource,
        options: [0; 6],
    });
    let passes = (0..4).all(|_| present(2, planes.as_ptr(), null(), null()) == S_OK)
        && (0..4).all(|_| queue_present(queue, 2, planes.as_ptr(), null()) == S_OK);
    for object in [textures[0], textures[1], queue, device] {
        let release: unsafe extern "system" fn(*mut c_void) -> u32 = slot(object, SLOT_RELEASE);
        release(object);
    }
    passes
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
    if std::fs::read(r"T:\probe-graphics").is_ok() {
        check(
            unsafe { graphics_workload(d3d, create) },
            "scaled scene and overlay through both PresentX entry points",
        );
    }

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
