//! Path-taking title imports, routed through the [`crate::vfs`].
//!
//! Titles build paths on console drives (`D:\`, `T:\`, `S:\`, ...). Every
//! path-taking import of kernel32/kernelx and the C runtime that the title
//! binds is replaced by a wrapper that rewrites a path on a mounted drive to
//! the host folder behind it, then calls the host's own function. Paths on
//! other drives reach the host unchanged.

use std::ffi::{CStr, CString, c_void};
use std::os::windows::ffi::OsStrExt;

/// # Safety
/// `p` is null or a NUL-terminated UTF-16 string.
unsafe fn wide_arg(p: *const u16) -> Option<Vec<u16>> {
    if p.is_null() {
        return None;
    }
    // SAFETY: forwarded from the caller.
    let len = (0..).take_while(|&i| unsafe { *p.add(i) } != 0).count();
    let text = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(p, len) });
    let host = crate::vfs::global().read().ok()?.rewrite(&text)?;
    Some(
        host.as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect(),
    )
}

/// # Safety
/// `p` is null or a NUL-terminated string.
unsafe fn ansi_arg(p: *const i8) -> Option<CString> {
    if p.is_null() {
        return None;
    }
    // SAFETY: forwarded from the caller.
    let text = unsafe { CStr::from_ptr(p) }.to_string_lossy();
    let host = crate::vfs::global().read().ok()?.rewrite(&text)?;
    CString::new(host.to_string_lossy().into_owned()).ok()
}

/// Wrapper whose first argument is a path (`W` = UTF-16, `A` = narrow).
macro_rules! wrap1 {
    ($name:ident, W, $dll:literal, $sym:literal, ($($a:ident: $t:ty),*) -> $r:ty) => {
        unsafe extern "system" fn $name(path: *const u16, $($a: $t),*) -> $r {
            type F = unsafe extern "system" fn(*const u16, $($t),*) -> $r;
            // SAFETY: the host export has exactly this signature.
            let real: F = unsafe { std::mem::transmute(host_fn!($dll, $sym)) };
            let w = unsafe { wide_arg(path) };
            unsafe { real(w.as_ref().map_or(path, |v| v.as_ptr()), $($a),*) }
        }
    };
    ($name:ident, A, $dll:literal, $sym:literal, ($($a:ident: $t:ty),*) -> $r:ty) => {
        unsafe extern "system" fn $name(path: *const i8, $($a: $t),*) -> $r {
            type F = unsafe extern "system" fn(*const i8, $($t),*) -> $r;
            // SAFETY: the host export has exactly this signature.
            let real: F = unsafe { std::mem::transmute(host_fn!($dll, $sym)) };
            let a = unsafe { ansi_arg(path) };
            unsafe { real(a.as_ref().map_or(path, |c| c.as_ptr()), $($a),*) }
        }
    };
}

/// Wrapper whose first two arguments are paths.
macro_rules! wrap2 {
    ($name:ident, W, $dll:literal, $sym:literal, ($($a:ident: $t:ty),*) -> $r:ty) => {
        unsafe extern "system" fn $name(from: *const u16, to: *const u16, $($a: $t),*) -> $r {
            type F = unsafe extern "system" fn(*const u16, *const u16, $($t),*) -> $r;
            // SAFETY: the host export has exactly this signature.
            let real: F = unsafe { std::mem::transmute(host_fn!($dll, $sym)) };
            let (f, t) = unsafe { (wide_arg(from), wide_arg(to)) };
            unsafe {
                real(f.as_ref().map_or(from, |v| v.as_ptr()), t.as_ref().map_or(to, |v| v.as_ptr()), $($a),*)
            }
        }
    };
    ($name:ident, A, $dll:literal, $sym:literal, ($($a:ident: $t:ty),*) -> $r:ty) => {
        unsafe extern "system" fn $name(from: *const i8, to: *const i8, $($a: $t),*) -> $r {
            type F = unsafe extern "system" fn(*const i8, *const i8, $($t),*) -> $r;
            // SAFETY: the host export has exactly this signature.
            let real: F = unsafe { std::mem::transmute(host_fn!($dll, $sym)) };
            let (f, t) = unsafe { (ansi_arg(from), ansi_arg(to)) };
            unsafe {
                real(f.as_ref().map_or(from, |c| c.as_ptr()), t.as_ref().map_or(to, |c| c.as_ptr()), $($a),*)
            }
        }
    };
}

type Ptr = *mut c_void;

wrap1!(create_file_w, W, "kernel32.dll", "CreateFileW", (access: u32, share: u32, sa: Ptr, disp: u32, flags: u32, template: Ptr) -> Ptr);
wrap1!(create_file_a, A, "kernel32.dll", "CreateFileA", (access: u32, share: u32, sa: Ptr, disp: u32, flags: u32, template: Ptr) -> Ptr);
wrap1!(create_file_2, W, "kernel32.dll", "CreateFile2", (access: u32, share: u32, disp: u32, params: Ptr) -> Ptr);
wrap1!(get_file_attributes_w, W, "kernel32.dll", "GetFileAttributesW", () -> u32);
wrap1!(get_file_attributes_a, A, "kernel32.dll", "GetFileAttributesA", () -> u32);
wrap1!(get_file_attributes_ex_w, W, "kernel32.dll", "GetFileAttributesExW", (level: u32, info: Ptr) -> i32);
wrap1!(get_file_attributes_ex_a, A, "kernel32.dll", "GetFileAttributesExA", (level: u32, info: Ptr) -> i32);
wrap1!(find_first_file_w, W, "kernel32.dll", "FindFirstFileW", (data: Ptr) -> Ptr);
wrap1!(find_first_file_a, A, "kernel32.dll", "FindFirstFileA", (data: Ptr) -> Ptr);
wrap1!(find_first_file_ex_w, W, "kernel32.dll", "FindFirstFileExW", (level: u32, data: Ptr, op: u32, filter: Ptr, flags: u32) -> Ptr);
wrap1!(find_first_file_ex_a, A, "kernel32.dll", "FindFirstFileExA", (level: u32, data: Ptr, op: u32, filter: Ptr, flags: u32) -> Ptr);
wrap1!(create_directory_w, W, "kernel32.dll", "CreateDirectoryW", (sa: Ptr) -> i32);
wrap1!(create_directory_a, A, "kernel32.dll", "CreateDirectoryA", (sa: Ptr) -> i32);
wrap1!(delete_file_w, W, "kernel32.dll", "DeleteFileW", () -> i32);
wrap1!(delete_file_a, A, "kernel32.dll", "DeleteFileA", () -> i32);
wrap1!(remove_directory_w, W, "kernel32.dll", "RemoveDirectoryW", () -> i32);
wrap1!(remove_directory_a, A, "kernel32.dll", "RemoveDirectoryA", () -> i32);
wrap2!(move_file_ex_w, W, "kernel32.dll", "MoveFileExW", (flags: u32) -> i32);
wrap2!(move_file_ex_a, A, "kernel32.dll", "MoveFileExA", (flags: u32) -> i32);
wrap2!(copy_file_w, W, "kernel32.dll", "CopyFileW", (fail: i32) -> i32);
wrap2!(copy_file_a, A, "kernel32.dll", "CopyFileA", (fail: i32) -> i32);
wrap1!(crt_fopen, A, "ucrtbase.dll", "fopen", (mode: *const i8) -> Ptr);
wrap1!(crt_wfopen, W, "ucrtbase.dll", "_wfopen", (mode: *const u16) -> Ptr);

unsafe extern "system" fn crt_fopen_s(out: *mut Ptr, path: *const i8, mode: *const i8) -> i32 {
    type F = unsafe extern "system" fn(*mut Ptr, *const i8, *const i8) -> i32;
    // SAFETY: the host export has exactly this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("ucrtbase.dll", "fopen_s")) };
    let a = unsafe { ansi_arg(path) };
    unsafe { real(out, a.as_ref().map_or(path, |c| c.as_ptr()), mode) }
}

unsafe extern "system" fn crt_wfopen_s(out: *mut Ptr, path: *const u16, mode: *const u16) -> i32 {
    type F = unsafe extern "system" fn(*mut Ptr, *const u16, *const u16) -> i32;
    // SAFETY: the host export has exactly this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("ucrtbase.dll", "_wfopen_s")) };
    let w = unsafe { wide_arg(path) };
    unsafe { real(out, w.as_ref().map_or(path, |v| v.as_ptr()), mode) }
}

/// Whether `dll` is one whose path-taking exports we wrap.
fn wraps_dll(dll: &str) -> bool {
    let d = dll.to_ascii_lowercase();
    d.starts_with("kernel32")
        || d.starts_with("kernelx")
        || d.starts_with("kernelbase")
        || d.starts_with("api-ms-win-core-file-")
        || d.starts_with("api-ms-win-crt-stdio-")
        || d.starts_with("ucrtbase")
        || d.starts_with("msvcrt")
}

/// The wrapper for `dll!func`, if it is a path-taking import we route.
pub(crate) fn file_override(dll: &str, func: &str) -> Option<usize> {
    if !wraps_dll(dll) {
        return None;
    }
    let f: usize = match func {
        "CreateFileW" => create_file_w as *const () as usize,
        "CreateFileA" => create_file_a as *const () as usize,
        "CreateFile2" => create_file_2 as *const () as usize,
        "GetFileAttributesW" => get_file_attributes_w as *const () as usize,
        "GetFileAttributesA" => get_file_attributes_a as *const () as usize,
        "GetFileAttributesExW" => get_file_attributes_ex_w as *const () as usize,
        "GetFileAttributesExA" => get_file_attributes_ex_a as *const () as usize,
        "FindFirstFileW" => find_first_file_w as *const () as usize,
        "FindFirstFileA" => find_first_file_a as *const () as usize,
        "FindFirstFileExW" => find_first_file_ex_w as *const () as usize,
        "FindFirstFileExA" => find_first_file_ex_a as *const () as usize,
        "CreateDirectoryW" => create_directory_w as *const () as usize,
        "CreateDirectoryA" => create_directory_a as *const () as usize,
        "DeleteFileW" => delete_file_w as *const () as usize,
        "DeleteFileA" => delete_file_a as *const () as usize,
        "RemoveDirectoryW" => remove_directory_w as *const () as usize,
        "RemoveDirectoryA" => remove_directory_a as *const () as usize,
        "MoveFileExW" => move_file_ex_w as *const () as usize,
        "MoveFileExA" => move_file_ex_a as *const () as usize,
        "CopyFileW" => copy_file_w as *const () as usize,
        "CopyFileA" => copy_file_a as *const () as usize,
        "fopen" => crt_fopen as *const () as usize,
        "_wfopen" => crt_wfopen as *const () as usize,
        "fopen_s" => crt_fopen_s as *const () as usize,
        "_wfopen_s" => crt_wfopen_s as *const () as usize,
        _ => return None,
    };
    Some(f)
}

#[cfg(all(test, windows, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn wrapped_file_calls_land_in_the_mounted_folder() {
        let root = std::env::temp_dir().join(format!("strata-files-test-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("hello.txt"), b"hi").unwrap();
        // Drive Q is used by this test only.
        crate::vfs::global().write().unwrap().mount('Q', &root);

        type GetAttrW = unsafe extern "system" fn(*const u16) -> u32;
        type GetAttrA = unsafe extern "system" fn(*const i8) -> u32;
        let w: GetAttrW = unsafe {
            std::mem::transmute(file_override("kernelx.dll", "GetFileAttributesW").unwrap())
        };
        let a: GetAttrA = unsafe {
            std::mem::transmute(file_override("KERNEL32.dll", "GetFileAttributesA").unwrap())
        };
        let wide: Vec<u16> = r"Q:\hello.txt".encode_utf16().chain([0]).collect();
        const INVALID: u32 = u32::MAX;
        assert_ne!(unsafe { w(wide.as_ptr()) }, INVALID);
        assert_ne!(unsafe { a(c"q:/hello.txt".as_ptr()) }, INVALID);
        assert_eq!(unsafe { a(c"q:/missing.txt".as_ptr()) }, INVALID);

        type CreateDir = unsafe extern "system" fn(*const i8, Ptr) -> i32;
        let mkdir: CreateDir = unsafe {
            std::mem::transmute(file_override("kernel32.dll", "CreateDirectoryA").unwrap())
        };
        assert_ne!(
            unsafe { mkdir(c"Q:\\made".as_ptr(), std::ptr::null_mut()) },
            0
        );
        assert!(root.join("made").is_dir());

        assert_eq!(file_override("user32.dll", "CreateFileW"), None);
        assert_eq!(file_override("kernel32.dll", "Sleep"), None);
        let _ = std::fs::remove_dir_all(&root);
    }
}
