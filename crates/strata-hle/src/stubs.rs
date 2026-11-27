//! Named stubs for imports nothing implements yet.
//!
//! Every unresolved import gets its own 22 bytes of generated code:
//!
//! ```text
//! mov rcx, <&'static ImportSite>   ; 48 B9 imm64
//! mov rax, <on_unimplemented>      ; 48 B8 imm64
//! jmp rax                          ; FF E0
//! ```
//!
//! The jump leaves the title's return address on the stack, so the handler
//! returns straight to the caller, and a call names the exact `dll!function`
//! that bring-up has to implement next.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

#[cfg(windows)]
unsafe extern "system" {
    fn VirtualAlloc(addr: *mut c_void, size: usize, alloc_type: u32, protect: u32) -> *mut c_void;
    fn VirtualProtect(addr: *mut c_void, size: usize, new_protect: u32, old: *mut u32) -> i32;
    fn FlushInstructionCache(process: *mut c_void, addr: *const c_void, size: usize) -> i32;
    fn GetCurrentProcess() -> *mut c_void;
    fn IsDebuggerPresent() -> i32;
    fn DebugBreak();
}

const MEM_COMMIT_RESERVE: u32 = 0x3000;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_EXECUTE_READ: u32 = 0x20;
const ARENA_BYTES: usize = 0x1_0000;
const STUB_SLOT: usize = 32;
const E_NOTIMPL: u64 = 0x8000_4001;

/// What an unimplemented import does when called.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum UnimplementedPolicy {
    /// Log, break into an attached debugger, then abort. The default:
    /// carrying on usually crashes later, further from the cause.
    #[default]
    Trap = 0,
    /// Log once per import and return 0.
    ReturnZero = 1,
    /// Log once per import and return `E_NOTIMPL` (0x80004001).
    ReturnNotImpl = 2,
}

impl std::str::FromStr for UnimplementedPolicy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "trap" => Ok(Self::Trap),
            "zero" => Ok(Self::ReturnZero),
            "notimpl" => Ok(Self::ReturnNotImpl),
            _ => Err(format!("unknown policy '{s}' (trap|zero|notimpl)")),
        }
    }
}

static POLICY: AtomicU8 = AtomicU8::new(UnimplementedPolicy::Trap as u8);

/// Held by every test that calls a stub, since the policy is process-wide.
#[cfg(test)]
pub(crate) fn policy_lock(policy: UnimplementedPolicy) -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_policy(policy);
    guard
}

pub fn set_policy(policy: UnimplementedPolicy) {
    POLICY.store(policy as u8, Ordering::Relaxed);
}

pub fn policy() -> UnimplementedPolicy {
    match POLICY.load(Ordering::Relaxed) {
        1 => UnimplementedPolicy::ReturnZero,
        2 => UnimplementedPolicy::ReturnNotImpl,
        _ => UnimplementedPolicy::Trap,
    }
}

/// One unresolved import. Leaked: lives as long as the process.
#[derive(Debug)]
pub struct ImportSite {
    pub dll: String,
    pub symbol: String,
    pub calls: AtomicU64,
}

struct Arena {
    page: usize,
    used: usize,
}

struct Registry {
    arena: Option<Arena>,
    /// `(stub address, site)` for every stub made.
    stubs: Vec<(usize, &'static ImportSite)>,
}

fn registry() -> &'static Mutex<Registry> {
    static REG: OnceLock<Mutex<Registry>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(Registry {
            arena: None,
            stubs: Vec::new(),
        })
    })
}

extern "system" fn on_unimplemented(site: &'static ImportSite) -> u64 {
    crate::thunks::note_unimplemented_call();
    let previous = site.calls.fetch_add(1, Ordering::Relaxed);
    let (dll, symbol) = (&site.dll, &site.symbol);
    match policy() {
        UnimplementedPolicy::ReturnZero | UnimplementedPolicy::ReturnNotImpl => {
            if previous == 0 {
                eprintln!("[strata] unimplemented import {dll}!{symbol} called");
            }
            if policy() == UnimplementedPolicy::ReturnZero {
                0
            } else {
                E_NOTIMPL
            }
        }
        UnimplementedPolicy::Trap => {
            let text = format!("unimplemented import {dll}!{symbol} called");
            eprintln!("[strata] {text}");
            crate::crash::log_line(&text);
            // SAFETY: plain Win32 calls without preconditions.
            unsafe {
                if IsDebuggerPresent() != 0 {
                    DebugBreak();
                }
            }
            std::process::abort();
        }
    }
}

fn encode_stub(site: usize, handler: usize) -> [u8; 22] {
    let mut code = [0u8; 22];
    code[0..2].copy_from_slice(&[0x48, 0xB9]);
    code[2..10].copy_from_slice(&(site as u64).to_le_bytes());
    code[10..12].copy_from_slice(&[0x48, 0xB8]);
    code[12..20].copy_from_slice(&(handler as u64).to_le_bytes());
    code[20..22].copy_from_slice(&[0xFF, 0xE0]);
    code
}

/// Makes (or reuses) the stub for `dll!symbol` and returns its address.
#[cfg(windows)]
pub fn stub_for(dll: &str, symbol: &str) -> Option<usize> {
    let mut reg = registry().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(addr, _)) = reg
        .stubs
        .iter()
        .find(|(_, s)| s.dll.eq_ignore_ascii_case(dll) && s.symbol == symbol)
    {
        return Some(addr);
    }
    if reg
        .arena
        .as_ref()
        .is_none_or(|a| a.used + STUB_SLOT > ARENA_BYTES)
    {
        // SAFETY: a fresh allocation; protections are managed below.
        let page = unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                ARENA_BYTES,
                MEM_COMMIT_RESERVE,
                PAGE_EXECUTE_READ,
            )
        };
        if page.is_null() {
            return None;
        }
        reg.arena = Some(Arena {
            page: page as usize,
            used: 0,
        });
    }
    let arena = reg.arena.as_mut()?;
    let addr = arena.page + arena.used;
    arena.used += STUB_SLOT;

    let site: &'static ImportSite = Box::leak(Box::new(ImportSite {
        dll: dll.to_owned(),
        symbol: symbol.to_owned(),
        calls: AtomicU64::new(0),
    }));
    let code = encode_stub(
        site as *const ImportSite as usize,
        on_unimplemented as *const () as usize,
    );
    let mut old = 0;
    // SAFETY: `addr..addr+32` lies inside our arena page; no code runs there yet.
    unsafe {
        VirtualProtect(addr as *mut c_void, STUB_SLOT, PAGE_READWRITE, &mut old);
        std::ptr::copy_nonoverlapping(code.as_ptr(), addr as *mut u8, code.len());
        VirtualProtect(addr as *mut c_void, STUB_SLOT, PAGE_EXECUTE_READ, &mut old);
        FlushInstructionCache(GetCurrentProcess(), addr as *const c_void, STUB_SLOT);
    }
    reg.stubs.push((addr, site));
    Some(addr)
}

/// The site behind a stub address, if it is one.
pub fn site_of(addr: usize) -> Option<&'static ImportSite> {
    let reg = registry().lock().unwrap_or_else(|e| e.into_inner());
    reg.stubs.iter().find(|(a, _)| *a == addr).map(|(_, s)| *s)
}

/// Every stub that was called at least once: `(dll, symbol, calls)`.
pub fn called_stubs() -> Vec<(String, String, u64)> {
    let reg = registry().lock().unwrap_or_else(|e| e.into_inner());
    reg.stubs
        .iter()
        .filter_map(|(_, s)| {
            let n = s.calls.load(Ordering::Relaxed);
            (n > 0).then(|| (s.dll.clone(), s.symbol.clone(), n))
        })
        .collect()
}

#[cfg(all(test, windows, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn stubs_are_per_import_and_name_their_site() {
        let a = stub_for("kernelx.dll", "StrataTestStubA").unwrap();
        let b = stub_for("kernelx.dll", "StrataTestStubB").unwrap();
        assert_ne!(a, b);
        assert_eq!(stub_for("KERNELX.DLL", "StrataTestStubA"), Some(a));
        assert_eq!(site_of(b).unwrap().symbol, "StrataTestStubB");
    }

    #[test]
    fn a_called_stub_counts_and_returns_per_policy() {
        let _policy = policy_lock(UnimplementedPolicy::ReturnNotImpl);
        let addr = stub_for("xstore.dll", "StrataTestStubCall").unwrap();
        let f: extern "system" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(addr) };
        assert_eq!(f(1, 2), E_NOTIMPL);
        set_policy(UnimplementedPolicy::ReturnZero);
        assert_eq!(f(1, 2), 0);
        assert_eq!(site_of(addr).unwrap().calls.load(Ordering::Relaxed), 2);
        assert!(
            called_stubs()
                .iter()
                .any(|(_, s, n)| s == "StrataTestStubCall" && *n == 2)
        );
    }

    #[test]
    fn policy_parses_from_cli_words() {
        assert_eq!("zero".parse(), Ok(UnimplementedPolicy::ReturnZero));
        assert!("loud".parse::<UnimplementedPolicy>().is_err());
    }
}
