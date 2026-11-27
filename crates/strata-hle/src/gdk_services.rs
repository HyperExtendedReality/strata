//! GDK platform services titles call at start-up and in menus — `XAsync`
//! completion, `XSystem`, `XThread`, `XDisplay`, `XGameUi` and `XLaunch` —
//! implemented offline on the desktop. Signatures follow the public GDK
//! headers (`XAsync.h`, `XSystem.h`, `XThread.h`, `XDisplay.h`, `XGameUI.h`,
//! `XLauncher.h`).
//!
//! Async calls complete for real: the `XAsyncBlock`'s callback runs on its
//! task queue's completion port (a manual queue's `XTaskQueueDispatch`), or
//! on a worker thread for the default (thread-pool) queue, and
//! `XAsyncGetStatus` reports the result.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

pub const S_OK: i32 = 0;
pub const E_PENDING: i32 = 0x8000_000Au32 as i32;
pub const E_ABORT: i32 = 0x8000_4004u32 as i32;
pub const E_INVALIDARG: i32 = 0x8007_0057u32 as i32;
pub const E_NOT_SUFFICIENT_BUFFER: i32 = 0x8007_007Au32 as i32;

/// `XAsyncBlock`: queue, context, completion callback, then the
/// provider's internal space.
#[repr(C)]
pub struct XAsyncBlock {
    pub queue: *mut c_void,
    pub context: *mut c_void,
    pub callback: Option<unsafe extern "system" fn(*mut XAsyncBlock)>,
    pub internal: [u8; 32],
}

/// Status and result bytes of each async block, by address.
static ASYNC: Mutex<Option<HashMap<usize, (i32, Vec<u8>)>>> = Mutex::new(None);

fn with_async<R>(f: impl FnOnce(&mut HashMap<usize, (i32, Vec<u8>)>) -> R) -> R {
    let mut g = ASYNC.lock().unwrap();
    f(g.get_or_insert_with(HashMap::new))
}

/// Marks `block` as begun (`XAsyncGetStatus` reports `E_PENDING`).
pub fn begin(block: *mut c_void) {
    if !block.is_null() {
        with_async(|m| m.insert(block as usize, (E_PENDING, Vec::new())));
    }
}

/// Completes `block` with `status` and `result`, then runs its callback:
/// on its manual queue's completion port (`submit`), or on a worker thread
/// for the default thread-pool queue.
pub fn complete(
    block: *mut c_void,
    status: i32,
    result: Vec<u8>,
    submit: impl FnOnce(u64, Box<dyn FnOnce() + Send>) -> bool,
) {
    if block.is_null() {
        return;
    }
    with_async(|m| m.insert(block as usize, (status, result)));
    let b = unsafe { &*(block as *const XAsyncBlock) };
    let Some(cb) = b.callback else {
        return;
    };
    let addr = block as usize;
    let call: Box<dyn FnOnce() + Send> = Box::new(move || unsafe { cb(addr as *mut XAsyncBlock) });
    let queue = b.queue as usize as u64;
    if queue != 0 && submit(queue, call) {
        return;
    }
    // Default queue (null) or one Strata does not know: the thread pool.
    // (`call` was moved into `submit` only when it accepted it.)
    let call: Box<dyn FnOnce() + Send> = Box::new(move || unsafe { cb(addr as *mut XAsyncBlock) });
    let _ = std::thread::Builder::new()
        .name("strata-xasync".into())
        .spawn(call);
}

/// `XAsyncGetStatus(block, wait)`.
pub fn status(block: *mut c_void, wait: bool) -> i32 {
    let get = || with_async(|m| m.get(&(block as usize)).map(|s| s.0));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match get() {
            // A block Strata never saw completed synchronously.
            None => return S_OK,
            Some(E_PENDING) if wait && std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Some(s) => return s,
        }
    }
}

/// The result bytes a completed `block` holds (for `…Result` calls).
pub fn result(block: *mut c_void) -> Option<(i32, Vec<u8>)> {
    with_async(|m| m.get(&(block as usize)).cloned())
}

/// Forgets `block` once its result has been taken.
pub fn finish(block: *mut c_void) {
    with_async(|m| m.remove(&(block as usize)));
}

/// Writes `s` (plus a terminator) into `buf` of `size` bytes; `used` gets
/// the bytes needed.
pub fn write_c_string(s: &str, size: usize, buf: *mut u8, used: *mut usize) -> i32 {
    let need = s.len() + 1;
    if !used.is_null() {
        unsafe { *used = need };
    }
    if buf.is_null() || size < need {
        return E_NOT_SUFFICIENT_BUFFER;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), buf, s.len());
        *buf.add(s.len()) = 0;
    }
    S_OK
}

pub fn c_string(p: *const u8) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(p.cast()) }
        .to_string_lossy()
        .into_owned()
}

// ============================================================================
// XSystem
// ============================================================================

/// A stable per-install console id: 38 hex-ish characters derived from the
/// machine name (`XSystemConsoleIdBytes` is 39 with the terminator).
pub fn console_id() -> String {
    let name = std::env::var("COMPUTERNAME").unwrap_or_else(|_| "strata".into());
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in name.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    let a = format!("{h:016X}{:016X}", h.rotate_left(17) ^ 0x5354_5241_5441_0000);
    format!("{}.{}", &a[..19], &a[19..])
}

pub unsafe extern "system" fn xsystem_get_console_id(
    size: usize,
    buf: *mut u8,
    used: *mut usize,
) -> i32 {
    write_c_string(&console_id(), size, buf, used)
}

pub unsafe extern "system" fn xsystem_get_sandbox_id(
    size: usize,
    buf: *mut u8,
    used: *mut usize,
) -> i32 {
    write_c_string("RETAIL", size, buf, used)
}

pub unsafe extern "system" fn xsystem_get_app_specific_device_id(
    size: usize,
    buf: *mut u8,
    used: *mut usize,
) -> i32 {
    let id = format!("{}{}", console_id(), "-APP");
    write_c_string(&id[..id.len().min(44)], size, buf, used)
}

/// `XSystemGetRuntimeInfo()` (16-byte struct: returned through a hidden
/// pointer): available and running Game OS versions (`XVersion`: major,
/// minor, build, revision as u16s).
pub unsafe extern "system" fn xsystem_get_runtime_info(out: *mut [u16; 8]) -> *mut [u16; 8] {
    if !out.is_null() {
        unsafe { *out = [10, 0, 25398, 4908, 10, 0, 25398, 4908] };
    }
    out
}

/// `XSystemGetAnalyticsInfo()`: OS and hosting OS versions, then the
/// family ("Gaming") and form ("Xbox Series X") strings, 64 bytes each.
pub unsafe extern "system" fn xsystem_get_analytics_info(out: *mut u8) -> *mut u8 {
    if out.is_null() {
        return out;
    }
    unsafe {
        std::ptr::write_bytes(out, 0, 16 + 128);
        let v: [u16; 8] = [10, 0, 25398, 4908, 10, 0, 25398, 4908];
        std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, out, 16);
        for (i, s) in ["Gaming", "Xbox Series X"].iter().enumerate() {
            std::ptr::copy_nonoverlapping(s.as_ptr(), out.add(16 + 64 * i), s.len());
        }
    }
    out
}

// ============================================================================
// XThread
// ============================================================================

thread_local! {
    static TIME_SENSITIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub unsafe extern "system" fn xthread_is_time_sensitive() -> u8 {
    u8::from(TIME_SENSITIVE.with(|t| t.get()))
}

/// Marks the calling thread time-sensitive (raised priority on the desktop).
pub unsafe extern "system" fn xthread_set_time_sensitive(on: u8) -> i32 {
    TIME_SENSITIVE.with(|t| t.set(on != 0));
    #[cfg(windows)]
    unsafe {
        unsafe extern "system" {
            fn GetCurrentThread() -> *mut c_void;
            fn SetThreadPriority(thread: *mut c_void, priority: i32) -> i32;
        }
        // THREAD_PRIORITY_HIGHEST / NORMAL.
        SetThreadPriority(GetCurrentThread(), if on != 0 { 2 } else { 0 });
    }
    S_OK
}

pub unsafe extern "system" fn xthread_assert_not_time_sensitive() {
    if TIME_SENSITIVE.with(|t| t.get()) {
        crate::crash::log_line("XThreadAssertNotTimeSensitive on a time-sensitive thread");
    }
}

// ============================================================================
// XDisplay
// ============================================================================

/// `XDisplayTryEnableHdrMode(preference, XDisplayHdrModeInfo*)`: enabled
/// when the window's swapchain can show HDR and the title prefers it
/// (preference 0: PreferHdr, 1: PreferRefreshRate, 2: PreferResolution;
/// result 0 Unknown, 1 Enabled, 2 Disabled).
pub unsafe extern "system" fn xdisplay_try_enable_hdr_mode(
    preference: u32,
    info: *mut [f32; 3],
) -> u32 {
    let hdr = preference == 0 && crate::thunks::display_supports_hdr();
    if !info.is_null() {
        unsafe {
            *info = if hdr {
                [0.0, 1000.0, 600.0] // min, max, max full-frame tone-map nits
            } else {
                [0.0, 80.0, 80.0]
            }
        };
    }
    if hdr { 1 } else { 2 }
}

static DEFERRALS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[cfg(windows)]
unsafe extern "system" {
    fn SetThreadExecutionState(flags: u32) -> u32;
}

/// `XDisplayAcquireTimeoutDeferral(handle*)`: keeps the display on.
pub unsafe extern "system" fn xdisplay_acquire_timeout_deferral(handle: *mut usize) -> i32 {
    let n = DEFERRALS.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1;
    #[cfg(windows)]
    unsafe {
        SetThreadExecutionState(0x8000_0002); // ES_CONTINUOUS | ES_DISPLAY_REQUIRED
    }
    if !handle.is_null() {
        unsafe { *handle = n as usize };
    }
    S_OK
}

pub unsafe extern "system" fn xdisplay_close_timeout_deferral_handle(_handle: usize) {
    let left = DEFERRALS
        .fetch_sub(1, std::sync::atomic::Ordering::AcqRel)
        .saturating_sub(1);
    #[cfg(windows)]
    if left == 0 {
        unsafe { SetThreadExecutionState(0x8000_0000) }; // ES_CONTINUOUS only
    }
    let _ = left;
}

// ============================================================================
// XGameUi
// ============================================================================

#[cfg(windows)]
unsafe extern "system" {
    fn MessageBoxW(hwnd: *mut c_void, text: *const u16, caption: *const u16, kind: u32) -> i32;
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain([0]).collect()
}

/// Shows a message dialog as a native message box (offline: the system UI
/// is the desktop's). Returns the `XGameUiMessageDialogButton` chosen
/// (0 first, 1 second, 2 third).
pub fn message_dialog(title: &str, content: &str, buttons: &[String], cancel: u32) -> u32 {
    #[cfg(windows)]
    {
        // MB_OK / MB_OKCANCEL / MB_YESNOCANCEL; labels go in the text.
        let (kind, map): (u32, &[(i32, u32)]) = match buttons.len() {
            0 | 1 => (0, &[(1, 0)]),
            2 => (1, &[(1, 0), (2, 1)]),
            _ => (3, &[(6, 0), (7, 1), (2, 2)]),
        };
        let labels: Vec<String> = buttons
            .iter()
            .enumerate()
            .map(|(i, b)| format!("[{}] {b}", ["OK/Yes", "Cancel/No", "Cancel"][i.min(2)]))
            .collect();
        let text = if buttons.len() > 1 {
            format!("{content}\n\n{}", labels.join("\n"))
        } else {
            content.to_string()
        };
        let r = unsafe {
            MessageBoxW(
                std::ptr::null_mut(),
                wide(&text).as_ptr(),
                wide(title).as_ptr(),
                kind | 0x40,
            )
        };
        map.iter()
            .find(|(id, _)| *id == r)
            .map_or(cancel, |(_, b)| *b)
    }
    #[cfg(not(windows))]
    {
        let _ = (title, content, buttons);
        cancel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "system" fn done(b: *mut XAsyncBlock) {
        let flag = unsafe { (*b).context } as *const std::sync::atomic::AtomicBool;
        unsafe { (*flag).store(true, std::sync::atomic::Ordering::SeqCst) };
    }

    #[test]
    fn async_blocks_complete_and_call_back_on_the_thread_pool() {
        let flag = Box::new(std::sync::atomic::AtomicBool::new(false));
        let mut block = XAsyncBlock {
            queue: std::ptr::null_mut(),
            context: &*flag as *const _ as *mut c_void,
            callback: Some(done),
            internal: [0; 32],
        };
        let p = &mut block as *mut XAsyncBlock as *mut c_void;
        begin(p);
        assert_eq!(status(p, false), E_PENDING);
        complete(p, S_OK, b"r".to_vec(), |_, _| false);
        assert_eq!(status(p, true), S_OK);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !flag.load(std::sync::atomic::Ordering::SeqCst)
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(flag.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(result(p).map(|r| r.1), Some(b"r".to_vec()));
        finish(p);
    }

    #[test]
    fn string_queries_report_sizes() {
        let mut used = 0usize;
        assert_eq!(
            write_c_string("RETAIL", 2, std::ptr::null_mut(), &mut used),
            E_NOT_SUFFICIENT_BUFFER
        );
        assert_eq!(used, 7);
        let mut buf = [0u8; 8];
        assert_eq!(
            write_c_string("RETAIL", 8, buf.as_mut_ptr(), &mut used),
            S_OK
        );
        assert_eq!(&buf[..7], b"RETAIL\0");
        assert_eq!(console_id().len(), 33);
    }
}
