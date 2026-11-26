//! Console-only Zen 2 x86-64 instruction decoder & Vectored Exception Handler (VEH).
//!
//! Xbox Series X|S runs an AMD Zen 2 CPU; retail titles may emit instructions
//! absent on older Intel CPUs (`clwb`, `clzero`, `rdpru`, `monitorx`, `mwaitx`, `rdpid`).
//! Decoding and emulating them in a VEH prevents `0xC000001D` illegal-instruction crashes.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(windows)]
unsafe extern "system" {
    fn AddVectoredExceptionHandler(
        first: u32,
        handler: unsafe extern "system" fn(*mut ExceptionPointers) -> i32,
    ) -> *mut c_void;
    fn RemoveVectoredExceptionHandler(handle: *mut c_void) -> u32;
}

const EXCEPTION_ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

#[repr(C)]
struct ExceptionRecord {
    exception_code: u32,
    exception_flags: u32,
    exception_record: *mut ExceptionRecord,
    exception_address: *mut c_void,
}

/// Minimal x86-64 `CONTEXT` prefix up to `Rip` (offset 0xF8 on Windows x64).
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

#[repr(C)]
struct ExceptionPointers {
    exception_record: *mut ExceptionRecord,
    context_record: *mut ContextX64,
}

static EMULATED_INSN_COUNT: AtomicU64 = AtomicU64::new(0);

/// RAII guard for Strata's live Windows Vectored Exception Handler.
pub struct VehGuard {
    handle: *mut c_void,
}

impl VehGuard {
    pub fn install() -> Option<Self> {
        let handle = unsafe { AddVectoredExceptionHandler(1, console_veh_handler) };
        if handle.is_null() { None } else { Some(Self { handle }) }
    }

    pub fn emulated_count() -> u64 {
        EMULATED_INSN_COUNT.load(Ordering::Relaxed)
    }
}

impl Drop for VehGuard {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { RemoveVectoredExceptionHandler(self.handle) };
        }
    }
}

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
        ConsoleInsn::Clwb { len } | ConsoleInsn::MonitorWaitX { len } => {
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
            ctx.rax = 0;
            ctx.rdx = 0;
            ctx.rip += len as u64;
        }
        ConsoleInsn::Rdpid { len, reg } => {
            let regs = [
                &mut ctx.rax, &mut ctx.rcx, &mut ctx.rdx, &mut ctx.rbx,
                &mut ctx.rsp, &mut ctx.rbp, &mut ctx.rsi, &mut ctx.rdi,
                &mut ctx.r8,  &mut ctx.r9,  &mut ctx.r10, &mut ctx.r11,
                &mut ctx.r12, &mut ctx.r13, &mut ctx.r14, &mut ctx.r15,
            ];
            if let Some(slot) = regs.into_iter().nth(reg as usize) {
                *slot = 0;
            }
            ctx.rip += len as u64;
        }
    }
    EMULATED_INSN_COUNT.fetch_add(1, Ordering::Relaxed);
    EXCEPTION_CONTINUE_EXECUTION
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleInsn {
    /// `clwb m8` — cache-line write-back (no-op on coherent PC memory).
    Clwb { len: usize },
    /// `clzero [rax]` — zeroes the 64-byte cache line at RAX.
    Clzero { len: usize },
    /// `rdpru` — read processor register (emulated via QPC / 0).
    Rdpru { len: usize },
    /// `monitorx` / `mwaitx` — AMD-only monitor/wait (no-op).
    MonitorWaitX { len: usize },
    /// `rdpid r64` — read processor ID.
    Rdpid { len: usize, reg: u8 },
}

/// Decode a Zen 2 console-only instruction from the faulting RIP bytes.
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
    // `clwb m8`: 66 [REX] 0F AE /6
    if bytes[0] == 0x66 {
        let off = if bytes.len() >= 5 && (bytes[1] & 0xF0) == 0x40 { 2 } else { 1 };
        if bytes.len() >= off + 3 && bytes[off] == 0x0F && bytes[off + 1] == 0xAE {
            let modrm = bytes[off + 2];
            let reg_op = (modrm >> 3) & 7;
            if reg_op == 6 && (modrm >> 6) != 3 {
                let len = off + 3 + modrm_disp_sib_len(&bytes[off + 2..]);
                return Some(ConsoleInsn::Clwb { len });
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
    fn decodes_zen2_console_instructions() {
        assert_eq!(decode_console_insn(&[0x0F, 0x01, 0xFC]), Some(ConsoleInsn::Clzero { len: 3 }));
        assert_eq!(decode_console_insn(&[0x0F, 0x01, 0xFD]), Some(ConsoleInsn::Rdpru { len: 3 }));
        assert_eq!(decode_console_insn(&[0x0F, 0x01, 0xFA]), Some(ConsoleInsn::MonitorWaitX { len: 3 }));
        assert_eq!(decode_console_insn(&[0xF3, 0x0F, 0xC7, 0xF8]), Some(ConsoleInsn::Rdpid { len: 4, reg: 0 }));
        // `clwb [rax + 0x40]` -> 66 0F AE 70 40
        assert_eq!(decode_console_insn(&[0x66, 0x0F, 0xAE, 0x70, 0x40]), Some(ConsoleInsn::Clwb { len: 5 }));
    }
}
