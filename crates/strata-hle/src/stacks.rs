//! All-thread stack dumper (`STRATA_STACKS=<seconds>[,<seconds>...]`) for
//! diagnosing title hangs and deadlocks across every thread in the process.

use std::ffi::c_void;
use std::time::Duration;

/// Snapshot of a single thread's instruction pointer and resolved location.
#[derive(Debug, Clone)]
pub struct ThreadStackSummary {
    pub tid: u32,
    pub rip: u64,
    pub location: String,
}

#[cfg(all(windows, target_arch = "x86_64"))]
#[repr(C, align(16))]
struct ContextAmd64 {
    _pad0: [u8; 0x30],
    context_flags: u32,
    _pad1: [u8; 0xC4],
    rip: u64,
    _pad2: [u8; 0x3D0],
}

#[cfg(all(windows, target_arch = "x86_64"))]
#[repr(C)]
struct ThreadEntry32 {
    dw_size: u32,
    cnt_usage: u32,
    th32_thread_id: u32,
    th32_owner_process_id: u32,
    tp_base_pri: i32,
    tp_delta_pri: i32,
    dw_flags: u32,
}

#[cfg(all(windows, target_arch = "x86_64"))]
unsafe extern "system" {
    fn CreateToolhelp32Snapshot(flags: u32, pid: u32) -> *mut c_void;
    fn Thread32First(snap: *mut c_void, entry: *mut ThreadEntry32) -> i32;
    fn Thread32Next(snap: *mut c_void, entry: *mut ThreadEntry32) -> i32;
    fn GetCurrentProcessId() -> u32;
    fn GetCurrentThreadId() -> u32;
    fn OpenThread(access: u32, inherit: i32, tid: u32) -> *mut c_void;
    fn SuspendThread(thread: *mut c_void) -> u32;
    fn ResumeThread(thread: *mut c_void) -> u32;
    fn GetThreadContext(thread: *mut c_void, ctx: *mut ContextAmd64) -> i32;
    fn CloseHandle(handle: *mut c_void) -> i32;
}

/// Captures the instruction pointer of every other thread in the current
/// process and formats each as `module+rva` when it lands in a mapped title image.
pub fn dump_all_thread_stacks() -> Vec<ThreadStackSummary> {
    let mut out = Vec::new();
    #[cfg(all(windows, target_arch = "x86_64"))]
    unsafe {
        const TH32CS_SNAPTHREAD: u32 = 0x0000_0004;
        const THREAD_GET_CONTEXT: u32 = 0x0008;
        const THREAD_SUSPEND_RESUME: u32 = 0x0002;
        const CONTEXT_CONTROL: u32 = 0x0010_0001;

        let pid = GetCurrentProcessId();
        let self_tid = GetCurrentThreadId();
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap.is_null() || snap as isize == -1 {
            return out;
        }
        let mut entry: ThreadEntry32 = std::mem::zeroed();
        entry.dw_size = std::mem::size_of::<ThreadEntry32>() as u32;
        if Thread32First(snap, &mut entry) != 0 {
            loop {
                if entry.th32_owner_process_id == pid && entry.th32_thread_id != self_tid {
                    let h = OpenThread(
                        THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME,
                        0,
                        entry.th32_thread_id,
                    );
                    if !h.is_null() {
                        if SuspendThread(h) != u32::MAX {
                            let mut ctx: ContextAmd64 = std::mem::zeroed();
                            ctx.context_flags = CONTEXT_CONTROL;
                            if GetThreadContext(h, &mut ctx) != 0 {
                                let loc = crate::title::title_module_at(ctx.rip as usize)
                                    .map(|(m, rva)| format!("{m}+0x{rva:x}"))
                                    .unwrap_or_else(|| format!("0x{:016x}", ctx.rip));
                                out.push(ThreadStackSummary {
                                    tid: entry.th32_thread_id,
                                    rip: ctx.rip,
                                    location: loc,
                                });
                            }
                            ResumeThread(h);
                        }
                        CloseHandle(h);
                    }
                }
                if Thread32Next(snap, &mut entry) == 0 {
                    break;
                }
            }
        }
        CloseHandle(snap);
    }
    out
}

/// Parses `STRATA_STACKS=<sec>[,<sec>...]` and spawns a background watchdog
/// thread that logs all thread locations at each requested offset.
pub fn schedule_from_env() {
    let Ok(raw) = std::env::var("STRATA_STACKS") else {
        return;
    };
    let mut secs: Vec<u64> = raw
        .split(',')
        .filter_map(|s| s.trim().parse::<u64>().ok())
        .collect();
    if secs.is_empty() {
        return;
    }
    secs.sort_unstable();
    let _ = std::thread::Builder::new()
        .name("strata-stacks".into())
        .spawn(move || {
            let mut elapsed = 0u64;
            for target in secs {
                if target > elapsed {
                    std::thread::sleep(Duration::from_secs(target - elapsed));
                    elapsed = target;
                }
                let threads = dump_all_thread_stacks();
                crate::crash::log_line(&format!(
                    "STRATA_STACKS @{target}s ({} threads):",
                    threads.len()
                ));
                for t in threads {
                    crate::crash::log_line(&format!("  tid={} rip={}", t.tid, t.location));
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dumps_other_threads_without_panicking() {
        let (tx, rx) = std::sync::mpsc::channel();
        let h = std::thread::spawn(move || {
            let _ = rx.recv();
        });
        let _stacks = dump_all_thread_stacks();
        let _ = tx.send(());
        let _ = h.join();
    }
}
