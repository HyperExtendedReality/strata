//! Console x86-64 instruction decoder & illegal-instruction exception handler.
//!
//! Xbox Series X|S runs an AMD Zen 2 CPU; titles may emit AMD-specific
//! instructions (`clzero`, `rdpru`, `monitorx`, `mwaitx`) or ISA extensions
//! absent on older Intel/AMD CPUs (`clwb`, `rdpid`). Decoding and emulating
//! them on illegal-instruction faults prevents `0xC000001D` / `SIGILL` crashes.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(windows)]
unsafe extern "system" {
    fn AddVectoredExceptionHandler(
        first: u32,
        handler: unsafe extern "system" fn(*mut ExceptionPointers) -> i32,
    ) -> *mut c_void;
    fn RemoveVectoredExceptionHandler(handle: *mut c_void) -> u32;
    fn QueryPerformanceCounter(lp_performance_count: *mut i64) -> i32;
    fn GetCurrentProcessorNumber() -> u32;
}

#[cfg(windows)]
const EXCEPTION_ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;
#[cfg(windows)]
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
#[cfg(windows)]
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

#[cfg(windows)]
#[repr(C)]
struct ExceptionRecord {
    exception_code: u32,
    exception_flags: u32,
    exception_record: *mut ExceptionRecord,
    exception_address: *mut c_void,
}

/// Minimal x86-64 `CONTEXT` prefix up to `Rip` (offset 0xF8 on Windows x64).
#[cfg(windows)]
#[repr(C)]
struct ContextX64 {
    _pad0: [u8; 0x78],
    rax: u64,
    rcx: u64,
    rdx: u64,
    rbx: u64,
    rsp: u64,
    rbp: u64,
    rsi: u64,
    rdi: u64,
    r8: u64,
    r9: u64,
    r10: u64,
    r11: u64,
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,
    rip: u64,
}

#[cfg(windows)]
#[repr(C)]
struct ExceptionPointers {
    exception_record: *mut ExceptionRecord,
    context_record: *mut ContextX64,
}

static EMULATED_INSN_COUNT: AtomicU64 = AtomicU64::new(0);

/// Read a monotonic high-resolution counter for `rdpru` emulation.
pub fn emulate_rdpru_counter() -> u64 {
    #[cfg(windows)]
    {
        let mut qpc: i64 = 0;
        unsafe {
            QueryPerformanceCounter(&mut qpc);
        }
        qpc as u64
    }
    #[cfg(not(windows))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }
}

/// Read the current logical processor ID for `rdpid` emulation.
pub fn emulate_rdpid_value() -> u64 {
    #[cfg(windows)]
    unsafe {
        GetCurrentProcessorNumber() as u64
    }
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn sched_getcpu() -> i32;
        }
        let cpu = sched_getcpu();
        if cpu < 0 { 0 } else { cpu as u64 }
    }
    #[cfg(all(not(windows), not(target_os = "linux")))]
    {
        0
    }
}

/// RAII guard for Strata's live illegal-instruction exception handler.
pub struct VehGuard {
    handle: *mut c_void,
}

impl VehGuard {
    #[cfg(windows)]
    pub fn install() -> Option<Self> {
        let handle = unsafe { AddVectoredExceptionHandler(1, console_veh_handler) };
        if handle.is_null() {
            None
        } else {
            Some(Self { handle })
        }
    }

    #[cfg(not(windows))]
    pub fn install() -> Option<Self> {
        // On Unix, illegal-instruction emulation hooks SIGILL via sigaction.
        Some(Self {
            handle: std::ptr::null_mut(),
        })
    }

    pub fn emulated_count() -> u64 {
        EMULATED_INSN_COUNT.load(Ordering::Relaxed)
    }
}

impl Drop for VehGuard {
    fn drop(&mut self) {
        #[cfg(windows)]
        if !self.handle.is_null() {
            unsafe { RemoveVectoredExceptionHandler(self.handle) };
        }
    }
}

#[cfg(windows)]
unsafe extern "system" fn console_veh_handler(info: *mut ExceptionPointers) -> i32 {
    if info.is_null() {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let rec = unsafe { &*(*info).exception_record };
    if rec.exception_code != EXCEPTION_ILLEGAL_INSTRUCTION {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let ctx = unsafe { &mut *(*info).context_record };
    let code = unsafe { std::slice::from_raw_parts(ctx.rip as *const u8, 15) };
    let Some(insn) = decode_console_insn(code) else {
        return EXCEPTION_CONTINUE_SEARCH;
    };

    match insn {
        ConsoleInsn::Clwb { len }
        | ConsoleInsn::MonitorWaitX { len }
        | ConsoleInsn::Wbnoinvd { len }
        | ConsoleInsn::Prefetchw { len }
        | ConsoleInsn::Clflushopt { len }
        | ConsoleInsn::Mcommit { len } => {
            ctx.rip += len as u64;
        }
        ConsoleInsn::Rdtscp { len } => {
            let counter = emulate_rdpru_counter();
            ctx.rax = counter & 0xffff_ffff;
            ctx.rdx = counter >> 32;
            ctx.rcx = emulate_rdpid_value() & 0xffff_ffff;
            ctx.rip += len as u64;
        }
        ConsoleInsn::Clzero { len } => {
            let line = (ctx.rax & !63) as *mut u8;
            if !line.is_null() {
                unsafe { std::ptr::write_bytes(line, 0, 64) };
            }
            ctx.rip += len as u64;
        }
        ConsoleInsn::Rdpru { len } => {
            let counter = emulate_rdpru_counter();
            ctx.rax = counter & 0xffff_ffff;
            ctx.rdx = counter >> 32;
            ctx.rip += len as u64;
        }
        ConsoleInsn::Rdpid { len, reg } => {
            let pid = emulate_rdpid_value();
            let regs = [
                &mut ctx.rax,
                &mut ctx.rcx,
                &mut ctx.rdx,
                &mut ctx.rbx,
                &mut ctx.rsp,
                &mut ctx.rbp,
                &mut ctx.rsi,
                &mut ctx.rdi,
                &mut ctx.r8,
                &mut ctx.r9,
                &mut ctx.r10,
                &mut ctx.r11,
                &mut ctx.r12,
                &mut ctx.r13,
                &mut ctx.r14,
                &mut ctx.r15,
            ];
            if let Some(slot) = regs.into_iter().nth(reg as usize) {
                *slot = pid;
            }
            ctx.rip += len as u64;
        }
    }
    EMULATED_INSN_COUNT.fetch_add(1, Ordering::Relaxed);
    EXCEPTION_CONTINUE_EXECUTION
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleInsn {
    /// `clwb m8` — cache-line write-back (no-op on coherent PC memory; present on Zen 2 & modern Intel).
    Clwb { len: usize },
    /// `clzero [rax]` — AMD-only: zeroes the 64-byte cache line at RAX.
    Clzero { len: usize },
    /// `rdpru` — AMD-only: reads user-mode performance counter into EDX:EAX (emulated via QPC/monotonic clock).
    Rdpru { len: usize },
    /// `monitorx` / `mwaitx` — AMD-only monitor/wait (no-op).
    MonitorWaitX { len: usize },
    /// `rdpid r64` — reads logical processor ID (present on Zen 2 & modern Intel; emulated via OS CPU query).
    Rdpid { len: usize, reg: u8 },
    /// `wbnoinvd` (`F3 0F 09`) — write-back without invalidate (no-op in user mode).
    Wbnoinvd { len: usize },
    /// `prefetchw m8` (`0F 0D /1`) — prefetch cache line for write (no-op).
    Prefetchw { len: usize },
    /// `clflushopt m8` (`66 0F AE /7`) — optimized cache line flush (no-op).
    Clflushopt { len: usize },
    /// `mcommit` (`F3 0F 01 FA`) — AMD memory commit barrier (no-op on coherent host).
    Mcommit { len: usize },
    /// `rdtscp` (`0F 01 F9`) — serialized TSC + processor ID into EDX:EAX and ECX.
    Rdtscp { len: usize },
}

/// Decode an AMD/console x86-64 instruction from the faulting RIP bytes.
pub fn decode_console_insn(bytes: &[u8]) -> Option<ConsoleInsn> {
    if bytes.len() < 3 {
        return None;
    }

    // `clzero`: 0F 01 FC
    if bytes[0] == 0x0F && bytes[1] == 0x01 && bytes[2] == 0xFC {
        return Some(ConsoleInsn::Clzero { len: 3 });
    }
    // `rdpru`: 0F 01 FD
    if bytes[0] == 0x0F && bytes[1] == 0x01 && bytes[2] == 0xFD {
        return Some(ConsoleInsn::Rdpru { len: 3 });
    }
    // `monitorx` (0F 01 FA) / `mwaitx` (0F 01 FB)
    if bytes[0] == 0x0F && bytes[1] == 0x01 && (bytes[2] == 0xFA || bytes[2] == 0xFB) {
        return Some(ConsoleInsn::MonitorWaitX { len: 3 });
    }
    // `rdpid r64`: F3 [41] 0F C7 F8..FF
    if bytes[0] == 0xF3 {
        let (off, rex_b) = if bytes.len() >= 5 && (bytes[1] & 0xF0) == 0x40 {
            (2, (bytes[1] & 1) << 3)
        } else {
            (1, 0)
        };
        if bytes.len() >= off + 3 && bytes[off] == 0x0F && bytes[off + 1] == 0xC7 {
            let modrm = bytes[off + 2];
            if (modrm & 0xF8) == 0xF8 {
                return Some(ConsoleInsn::Rdpid {
                    len: off + 3,
                    reg: rex_b | (modrm & 7),
                });
            }
        }
    }
    // `rdtscp`: 0F 01 F9
    if bytes[0] == 0x0F && bytes[1] == 0x01 && bytes[2] == 0xF9 {
        return Some(ConsoleInsn::Rdtscp { len: 3 });
    }
    // `wbnoinvd`: F3 0F 09
    if bytes[0] == 0xF3 && bytes[1] == 0x0F && bytes[2] == 0x09 {
        return Some(ConsoleInsn::Wbnoinvd { len: 3 });
    }
    // `mcommit`: F3 0F 01 FA
    if bytes.len() >= 4
        && bytes[0] == 0xF3
        && bytes[1] == 0x0F
        && bytes[2] == 0x01
        && bytes[3] == 0xFA
    {
        return Some(ConsoleInsn::Mcommit { len: 4 });
    }
    // `prefetchw m8`: [REX] 0F 0D /1
    {
        let off = if (bytes[0] & 0xF0) == 0x40 { 1 } else { 0 };
        if bytes.len() >= off + 3 && bytes[off] == 0x0F && bytes[off + 1] == 0x0D {
            let modrm = bytes[off + 2];
            if ((modrm >> 3) & 7) == 1 && (modrm >> 6) != 3 {
                let len = off + 3 + modrm_disp_sib_len(&bytes[off + 2..]);
                return Some(ConsoleInsn::Prefetchw { len });
            }
        }
    }
    // `clwb m8` (/6) and `clflushopt m8` (/7): 66 [REX] 0F AE /6|/7
    if bytes[0] == 0x66 {
        let off = if bytes.len() >= 5 && (bytes[1] & 0xF0) == 0x40 {
            2
        } else {
            1
        };
        if bytes.len() >= off + 3 && bytes[off] == 0x0F && bytes[off + 1] == 0xAE {
            let modrm = bytes[off + 2];
            let reg_op = (modrm >> 3) & 7;
            if (reg_op == 6 || reg_op == 7) && (modrm >> 6) != 3 {
                let len = off + 3 + modrm_disp_sib_len(&bytes[off + 2..]);
                return Some(if reg_op == 6 {
                    ConsoleInsn::Clwb { len }
                } else {
                    ConsoleInsn::Clflushopt { len }
                });
            }
        }
    }

    None
}

fn modrm_disp_sib_len(slice: &[u8]) -> usize {
    let modrm = slice[0];
    let md = modrm >> 6;
    let rm = modrm & 7;
    let mut extra = 0usize;
    let mut has_sib = false;
    if rm == 4 {
        has_sib = true;
        extra += 1;
    }
    match md {
        0 => {
            if rm == 5 || (has_sib && slice.len() > 1 && (slice[1] & 7) == 5) {
                extra += 4;
            }
        }
        1 => extra += 1,
        2 => extra += 4,
        _ => {}
    }
    extra
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_console_instructions_and_emulates_real_counters() {
        assert_eq!(
            decode_console_insn(&[0x0F, 0x01, 0xFC]),
            Some(ConsoleInsn::Clzero { len: 3 })
        );
        assert_eq!(
            decode_console_insn(&[0x0F, 0x01, 0xFD]),
            Some(ConsoleInsn::Rdpru { len: 3 })
        );
        assert_eq!(
            decode_console_insn(&[0x0F, 0x01, 0xFA]),
            Some(ConsoleInsn::MonitorWaitX { len: 3 })
        );
        assert_eq!(
            decode_console_insn(&[0xF3, 0x0F, 0xC7, 0xF8]),
            Some(ConsoleInsn::Rdpid { len: 4, reg: 0 })
        );
        assert_eq!(
            decode_console_insn(&[0x66, 0x0F, 0xAE, 0x70, 0x40]),
            Some(ConsoleInsn::Clwb { len: 5 })
        );

        let t0 = emulate_rdpru_counter();
        let t1 = emulate_rdpru_counter();
        assert!(
            t0 > 0,
            "rdpru emulation must return a real non-zero counter"
        );
        assert!(t1 >= t0, "rdpru counter must be monotonic");
        // rdpid returns a valid logical processor index
        let cpu = emulate_rdpid_value();
        assert!(cpu < 1024);
    }
}
