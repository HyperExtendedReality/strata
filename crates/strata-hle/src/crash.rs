//! Crash reporter: names a fault as `module+rva` and unwinds the faulting
//! thread's stack, writing both to stderr and a crash log.
//!
//! Installed as the unhandled-exception filter, so the title's own SEH and
//! C++ handlers (and Strata's console-instruction VEH) see every exception
//! first; only faults nobody handles reach the report. Unwinding uses the
//! `.pdata` the OS loader registered for each module.

use std::ffi::c_void;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

#[cfg(windows)]
unsafe extern "system" {
    fn SetUnhandledExceptionFilter(
        filter: Option<unsafe extern "system" fn(*mut ExceptionPointers) -> i32>,
    ) -> *mut c_void;
    fn GetModuleHandleExW(flags: u32, addr: *const u16, module: *mut *mut c_void) -> i32;
    fn GetModuleFileNameW(module: *mut c_void, name: *mut u16, size: u32) -> u32;
    fn RtlLookupFunctionEntry(pc: u64, image_base: *mut u64, history: *mut c_void) -> *mut c_void;
    fn RtlVirtualUnwind(
        handler_type: u32,
        image_base: u64,
        pc: u64,
        function_entry: *mut c_void,
        context: *mut Context,
        handler_data: *mut *mut c_void,
        establisher_frame: *mut u64,
        context_pointers: *mut c_void,
    ) -> *mut c_void;
    fn RtlCaptureContext(context: *mut Context);
}

const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x4;
const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x2;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
const MAX_FRAMES: usize = 64;
const CONTEXT_RSP: usize = 0x98;
const CONTEXT_RIP: usize = 0xF8;

#[repr(C)]
struct ExceptionRecord {
    code: u32,
    flags: u32,
    record: *mut ExceptionRecord,
    address: *mut c_void,
    parameter_count: u32,
    information: [usize; 15],
}

#[repr(C)]
struct ExceptionPointers {
    record: *mut ExceptionRecord,
    context: *mut Context,
}

/// Windows x64 `CONTEXT`: 1232 bytes, 16-byte aligned. Only `Rsp` and `Rip`
/// are read here; the rest is passed through to the unwinder.
#[repr(C, align(16))]
#[derive(Clone)]
pub struct Context([u8; 1232]);

impl Context {
    fn reg(&self, off: usize) -> u64 {
        u64::from_le_bytes(self.0[off..off + 8].try_into().unwrap())
    }
    fn set_reg(&mut self, off: usize, v: u64) {
        self.0[off..off + 8].copy_from_slice(&v.to_le_bytes());
    }
    pub fn rip(&self) -> u64 {
        self.reg(CONTEXT_RIP)
    }

    /// The calling thread's context, as `RtlCaptureContext` reports it.
    #[cfg(windows)]
    pub fn capture() -> Box<Self> {
        let mut ctx = Box::new(Self([0; 1232]));
        // SAFETY: a properly sized and aligned CONTEXT.
        unsafe { RtlCaptureContext(&mut *ctx) };
        ctx
    }
}

fn log_path() -> &'static Mutex<PathBuf> {
    static PATH: OnceLock<Mutex<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| Mutex::new(std::env::temp_dir().join("strata-crash.log")))
}

/// Appends a line to the crash log (and stderr).
pub fn log_line(text: &str) {
    eprintln!("[strata] {text}");
    let path = log_path().lock().unwrap_or_else(|e| e.into_inner()).clone();
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{text}");
    }
}

/// Installs the reporter. `log` overrides the default `%TEMP%\strata-crash.log`.
#[cfg(windows)]
pub fn install(log: Option<PathBuf>) {
    if let Some(log) = log {
        *log_path().lock().unwrap_or_else(|e| e.into_inner()) = log;
    }
    // SAFETY: installs a process-wide filter; the previous one is dropped.
    unsafe { SetUnhandledExceptionFilter(Some(unhandled)) };
}

/// `module+0xrva` for an address in a loaded module (title images by their
/// original name), a named stub, or the bare address.
#[cfg(windows)]
pub fn describe_address(addr: usize) -> String {
    if let Some(site) = crate::stubs::site_of(addr) {
        return format!("<stub {}!{}>", site.dll, site.symbol);
    }
    if let Some((name, base)) = crate::title::title_module_at(addr) {
        return format!("{name}+0x{:x}", addr - base);
    }
    let mut module = std::ptr::null_mut();
    // SAFETY: FROM_ADDRESS takes an address in place of a name.
    let ok = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            addr as *const u16,
            &mut module,
        )
    };
    if ok == 0 || module.is_null() {
        return format!("0x{addr:x}");
    }
    let mut buf = [0u16; 260];
    // SAFETY: `buf` holds 260 UTF-16 units.
    let n = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) } as usize;
    let path = String::from_utf16_lossy(&buf[..n]);
    let name = path.rsplit(['\\', '/']).next().unwrap_or(&path).to_owned();
    format!("{name}+0x{:x}", addr - module as usize)
}

/// Return addresses of the stack `ctx` describes, innermost first.
#[cfg(windows)]
pub fn walk_stack(ctx: &Context) -> Vec<usize> {
    let mut ctx = ctx.clone();
    let mut frames = Vec::new();
    while frames.len() < MAX_FRAMES {
        let pc = ctx.rip();
        if pc == 0 {
            break;
        }
        frames.push(pc as usize);
        let mut image_base = 0u64;
        // SAFETY: lookup only.
        let entry = unsafe { RtlLookupFunctionEntry(pc, &mut image_base, std::ptr::null_mut()) };
        if entry.is_null() {
            // A leaf function: its return address is at [rsp].
            let rsp = ctx.reg(CONTEXT_RSP);
            if rsp == 0 || !rsp.is_multiple_of(8) {
                break;
            }
            // SAFETY: the thread's own stack; a bad pointer here would fault
            // inside the reporter, which is acceptable for a dying process.
            let ret = unsafe { (rsp as *const u64).read() };
            ctx.set_reg(CONTEXT_RIP, ret);
            ctx.set_reg(CONTEXT_RSP, rsp + 8);
        } else {
            let (mut data, mut frame) = (std::ptr::null_mut(), 0u64);
            // SAFETY: `entry` belongs to `image_base`; `ctx` is a full CONTEXT.
            unsafe {
                RtlVirtualUnwind(
                    0,
                    image_base,
                    pc,
                    entry,
                    &mut ctx,
                    &mut data,
                    &mut frame,
                    std::ptr::null_mut(),
                );
            }
        }
        if ctx.rip() == pc && frames.len() > 1 {
            break;
        }
    }
    frames
}

fn exception_name(code: u32) -> &'static str {
    match code {
        0xC000_0005 => "access violation",
        0xC000_001D => "illegal instruction",
        0xC000_0094 => "integer divide by zero",
        0xC000_00FD => "stack overflow",
        0xC000_0409 => "stack buffer overrun",
        0x8000_0003 => "breakpoint",
        0xE06D_7363 => "unhandled C++ exception",
        _ => "exception",
    }
}

#[cfg(windows)]
unsafe extern "system" fn unhandled(info: *mut ExceptionPointers) -> i32 {
    if info.is_null() {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // SAFETY: the OS hands us valid pointers for the faulting thread.
    let (rec, ctx) = unsafe { (&*(*info).record, &*(*info).context) };
    let at = describe_address(rec.address as usize);
    let mut detail = String::new();
    if rec.code == 0xC000_0005 && rec.parameter_count >= 2 {
        let op = match rec.information[0] {
            0 => "reading",
            1 => "writing",
            8 => "executing",
            _ => "touching",
        };
        detail = format!(" {op} 0x{:x}", rec.information[1]);
    }
    log_line(&format!(
        "{} (0x{:08X}) at {at}{detail}",
        exception_name(rec.code),
        rec.code
    ));
    for (i, pc) in walk_stack(ctx).into_iter().enumerate() {
        log_line(&format!("  #{i:<2} {}", describe_address(pc)));
    }
    EXCEPTION_CONTINUE_SEARCH
}

#[cfg(all(test, windows, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn describes_addresses_as_module_plus_rva() {
        let text = describe_address(crate::loader::host_export("kernel32.dll", "GetTickCount"));
        assert!(text.to_ascii_lowercase().starts_with("kernel"), "{text}");
        assert!(text.contains("+0x"), "{text}");
        assert_eq!(describe_address(0x10), "0x10");
    }

    #[inline(never)]
    fn capture_here() -> Vec<usize> {
        walk_stack(&Context::capture())
    }

    #[test]
    fn unwinds_through_registered_frames() {
        let frames = capture_here();
        assert!(frames.len() >= 3, "{frames:x?}");
        let me = describe_address(frames[0]);
        let exe = std::env::current_exe().unwrap();
        let exe_name = exe.file_name().unwrap().to_string_lossy();
        assert!(me.starts_with(&*exe_name), "{me} vs {exe_name}");
    }

    #[test]
    fn names_common_exception_codes() {
        assert_eq!(exception_name(0xC000_0005), "access violation");
        assert_eq!(exception_name(0x1234), "exception");
    }
}
