//! A title session: the executable, the DLLs it ships beside it, and the
//! hooks that make the process look like the title's own.
//!
//! Import resolution, for every `dll!function`, in order:
//!
//! 1. a Strata override of a host export ([`import_override`]): library
//!    loading, threads, process exit and path-taking file calls;
//! 2. a Strata HLE module (`xmem.dll`, `d3d12_xs.dll`, ...); `kernelx.dll`
//!    falls back to the host's kernel exports;
//! 3. the host's own export, for host DLLs;
//! 4. an export of a DLL the title ships beside its executable (only DLLs
//!    neither of the above serves), mapped like the executable;
//! 5. a named stub ([`crate::stubs`]).
//!
//! `LoadLibrary*` of a console DLL returns a pseudo handle whose
//! `GetProcAddress` finds our functions; of a title DLL, its mapped image
//! (mapping and attaching it first). `GetModuleHandle(NULL)` and
//! `GetModuleFileName(NULL)` answer for the title, not for Strata.
//! Every thread in the process, the title's or the host's, tells every
//! attached title module of its start and end (`DLL_THREAD_ATTACH` /
//! `DETACH`): the Windows loader calls each image's entry thunk, which
//! forwards to its TLS callbacks and `DllMain`. Threads the title starts
//! through `CreateThread`/`_beginthread(ex)` cover images without a thunk.

use std::ffi::{CStr, c_void};
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;

use crate::loader::{
    DLL_THREAD_ATTACH, DLL_THREAD_DETACH, ImportResolver, MapError, MappedImage, Resolution,
};
use crate::pe::{ExportFunc, Pe64Info};
use crate::runtime::{DllClass, classify_dll};
use crate::stubs::UnimplementedPolicy;

type Handle = *mut c_void;
type ThreadProc = unsafe extern "system" fn(*mut c_void) -> u32;
type VoidThreadProc = unsafe extern "C" fn(*mut c_void);

#[cfg(windows)]
unsafe extern "system" {
    fn CreateThread(
        sa: *mut c_void,
        stack: usize,
        start: ThreadProc,
        param: *mut c_void,
        flags: u32,
        tid: *mut u32,
    ) -> Handle;
    fn WaitForSingleObject(handle: Handle, ms: u32) -> u32;
    fn GetExitCodeThread(handle: Handle, code: *mut u32) -> i32;
    fn CloseHandle(handle: Handle) -> i32;
    fn ExitThread(code: u32) -> !;
    fn TerminateThread(handle: Handle, code: u32) -> i32;
    fn GetCurrentThreadId() -> u32;
    fn SetLastError(code: u32);
}

const INFINITE: u32 = u32::MAX;
const ERROR_MOD_NOT_FOUND: u32 = 126;
const ERROR_PROC_NOT_FOUND: u32 = 127;
const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
const LOAD_AS_DATA: u32 = 0x2 | 0x20 | 0x40;
const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x4;

/// One image of the title: the executable or a DLL it ships.
pub struct TitleModule {
    /// File name, lower case (`game.exe`, `xboxbvhbuilder.dll`).
    pub name: String,
    /// Where it was read from.
    pub path: PathBuf,
    pub image: MappedImage,
}

impl TitleModule {
    pub fn base(&self) -> usize {
        self.image.base as usize
    }
    fn contains(&self, addr: usize) -> bool {
        addr >= self.base() && addr < self.base() + self.image.size
    }
}

#[derive(Default)]
struct Registry {
    modules: Vec<&'static TitleModule>,
    /// Mapped, not yet bound: `(name, base, exports)`, so a DLL that imports
    /// back from one being bound still finds its exports.
    pending: Vec<(String, usize, Vec<ExportFunc>)>,
    main: Option<&'static TitleModule>,
    search_dir: Option<PathBuf>,
    /// DLLs bound but not yet attached, dependencies first.
    unattached: Vec<&'static TitleModule>,
}

fn registry() -> MutexGuard<'static, Registry> {
    static REG: OnceLock<Mutex<Registry>> = OnceLock::new();
    REG.get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// A reentrant loader lock: `DllMain` may call `LoadLibrary`.
struct LoaderLock {
    owner: Mutex<(Option<ThreadId>, usize)>,
    free: Condvar,
}

struct LoaderGuard;

fn loader_lock() -> &'static LoaderLock {
    static LOCK: OnceLock<LoaderLock> = OnceLock::new();
    LOCK.get_or_init(|| LoaderLock {
        owner: Mutex::new((None, 0)),
        free: Condvar::new(),
    })
}

fn lock_loader() -> LoaderGuard {
    let lock = loader_lock();
    let me = std::thread::current().id();
    let mut st = lock.owner.lock().unwrap_or_else(|e| e.into_inner());
    while st.0.is_some_and(|t| t != me) {
        st = lock.free.wait(st).unwrap_or_else(|e| e.into_inner());
    }
    st.0 = Some(me);
    st.1 += 1;
    LoaderGuard
}

impl Drop for LoaderGuard {
    fn drop(&mut self) {
        let lock = loader_lock();
        let mut st = lock.owner.lock().unwrap_or_else(|e| e.into_inner());
        st.1 -= 1;
        if st.1 == 0 {
            st.0 = None;
            lock.free.notify_one();
        }
    }
}

/// `C:\x\Foo` -> `foo.dll`; `bar.DLL` -> `bar.dll`.
fn module_file_name(raw: &str) -> String {
    let base = raw
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or(raw)
        .to_ascii_lowercase();
    if base.contains('.') {
        base
    } else {
        format!("{base}.dll")
    }
}

fn find_module(name: &str) -> Option<&'static TitleModule> {
    registry().modules.iter().copied().find(|m| m.name == name)
}

fn module_by_handle(handle: usize) -> Option<&'static TitleModule> {
    registry()
        .modules
        .iter()
        .copied()
        .find(|m| m.base() == handle)
}

/// The title module containing `addr`: `(original name, base)`.
pub fn title_module_at(addr: usize) -> Option<(String, usize)> {
    registry()
        .modules
        .iter()
        .find(|m| m.contains(addr))
        .map(|m| (m.name.clone(), m.base()))
}

/// The file of a DLL the title ships, if `name` is one.
fn title_dll_file(name: &str) -> Option<PathBuf> {
    let dir = registry().search_dir.clone()?;
    std::fs::read_dir(&dir)
        .ok()?
        .flatten()
        .find(|e| e.file_name().to_string_lossy().eq_ignore_ascii_case(name))
        .map(|e| e.path())
}

/// Whether imports of `dll` may come from the title's folder: only DLLs
/// that neither Strata nor the host serves.
fn may_be_title_dll(dll: &str) -> bool {
    matches!(
        classify_dll(dll),
        DllClass::TitleLocal | DllClass::NeedsStub
    )
}

struct TitleResolver {
    name: String,
}

impl ImportResolver for TitleResolver {
    fn mapped(&mut self, image: &MappedImage, _pe: &Pe64Info) {
        registry().pending.push((
            self.name.clone(),
            image.base as usize,
            image.exports.clone(),
        ));
    }

    fn resolve(&mut self, dll: &str, func: &str) -> Resolution {
        match crate::loader::resolve_without_title(dll, func) {
            r @ (Resolution::Hle(_) | Resolution::Host(_) | Resolution::Title(_)) => r,
            other if may_be_title_dll(dll) => match title_export(dll, func) {
                Some(addr) => Resolution::Title(addr),
                None if title_dll_file(&module_file_name(dll)).is_some() => Resolution::Unresolved,
                None => other,
            },
            other => other,
        }
    }

    fn module_handle(&mut self, dll: &str) -> Option<usize> {
        let name = module_file_name(dll);
        if let Some(m) = find_module(&name) {
            return Some(m.base());
        }
        if classify_dll(dll) == DllClass::StrataHle {
            return crate::hooks::hook_load_library(dll);
        }
        crate::loader::HostResolver.module_handle(dll)
    }
}

/// `dll!func` from a title DLL, mapping the DLL first if needed.
fn title_export(dll: &str, func: &str) -> Option<usize> {
    let name = module_file_name(dll);
    if let Some(m) = find_module(&name) {
        return m.image.export(func);
    }
    let pending = registry()
        .pending
        .iter()
        .find(|(n, ..)| *n == name)
        .map(|(_, base, exports)| (*base, exports.clone()));
    if let Some((base, exports)) = pending {
        let rva = match func.strip_prefix("ordinal#") {
            Some(ord) => exports.iter().find(|e| ord.parse() == Ok(e.ordinal))?.rva,
            None => exports.iter().find(|e| e.name == func)?.rva,
        };
        return Some(base + rva as usize);
    }
    let path = title_dll_file(&name)?;
    match load_module(&name, &path) {
        Ok(m) => m.image.export(func),
        Err(e) => {
            crate::crash::log_line(&format!("cannot load title DLL {}: {e}", path.display()));
            None
        }
    }
}

/// Maps and binds one title image (its own title DLLs first) and registers it.
fn load_module(name: &str, path: &Path) -> Result<&'static TitleModule, MapError> {
    let _lock = lock_loader();
    if let Some(m) = find_module(name) {
        return Ok(m);
    }
    let bytes = std::fs::read(path).map_err(|source| MapError::Io {
        path: path.to_owned(),
        source,
    })?;
    let result = MappedImage::load_with(
        &bytes,
        &mut TitleResolver {
            name: name.to_owned(),
        },
    );
    let mut reg = registry();
    reg.pending.retain(|(n, ..)| n != name);
    let image = result?;
    let module: &'static TitleModule = Box::leak(Box::new(TitleModule {
        name: name.to_owned(),
        path: path.to_owned(),
        image,
    }));
    reg.modules.push(module);
    if module.image.is_dll {
        reg.unattached.push(module);
    }
    Ok(module)
}

/// Attaches every DLL bound since the last call, dependencies first.
fn attach_pending() {
    let _lock = lock_loader();
    loop {
        let next = {
            let mut reg = registry();
            if reg.unattached.is_empty() {
                return;
            }
            reg.unattached.remove(0)
        };
        // SAFETY: the DLL's imports are bound; this is its loader attach.
        if !unsafe { next.image.attach() } {
            crate::crash::log_line(&format!("DllMain of {} returned FALSE", next.name));
        }
    }
}

/// Tells every attached title module the Windows loader does not already
/// tell (no entry thunk: see `loader::add_notify_thunk`) of a thread event.
fn notify_all(reason: u32) {
    let modules: Vec<&'static TitleModule> = registry().modules.clone();
    for m in modules
        .into_iter()
        .filter(|m| m.image.is_attached() && !m.image.thread_notices)
    {
        // SAFETY: attached modules are bound.
        unsafe { m.image.notify(reason) };
    }
}

/// How a title is loaded.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    pub unimplemented: UnimplementedPolicy,
    /// Folder searched for the DLLs the title ships; the executable's own
    /// folder when `None`.
    pub search_dir: Option<PathBuf>,
    /// Console drives and the host folders behind them.
    pub mounts: Vec<(char, PathBuf)>,
    /// Install the crash reporter, logging to this file (or the default).
    pub crash_reporter: Option<Option<PathBuf>>,
}

/// A loaded title.
pub struct Title {
    pub main: &'static TitleModule,
}

/// Loads a title executable and every DLL it ships that it imports.
/// Nothing runs until [`Title::run`].
pub fn load_title(path: &Path, opts: &LoadOptions) -> Result<Title, MapError> {
    crate::stubs::set_policy(opts.unimplemented);
    if let Some(log) = &opts.crash_reporter {
        crate::crash::install(log.clone());
    }
    {
        let mut vfs = crate::vfs::global()
            .write()
            .unwrap_or_else(|e| e.into_inner());
        for (drive, root) in &opts.mounts {
            vfs.mount(*drive, root);
        }
    }
    let dir = opts
        .search_dir
        .clone()
        .or_else(|| path.parent().map(Path::to_path_buf));
    registry().search_dir = dir;
    let name = module_file_name(&path.to_string_lossy());
    let main = load_module(&name, path)?;
    registry().main = Some(main);
    Ok(Title { main })
}

static GUEST_MAIN_TID: AtomicU32 = AtomicU32::new(0);
static GUEST_MAIN_HANDLE: AtomicUsize = AtomicUsize::new(0);
static EXIT_REQUESTED: AtomicBool = AtomicBool::new(false);
static EXIT_CODE: AtomicU32 = AtomicU32::new(0);

unsafe extern "system" fn guest_main(param: *mut c_void) -> u32 {
    // SAFETY: `param` is the leaked `&'static TitleModule` from `run`.
    let main = unsafe { &*(param as *const TitleModule) };
    attach_pending();
    // SAFETY: imports bound; this is the title's main thread.
    unsafe { main.image.attach_main_thread() };
    let entry: unsafe extern "system" fn() -> u32 =
        unsafe { std::mem::transmute(main.image.entry_point) };
    unsafe { entry() }
}

impl Title {
    pub fn modules(&self) -> Vec<&'static TitleModule> {
        registry().modules.clone()
    }

    /// Attaches the title's DLLs, then runs the executable's entry point on
    /// a thread of its own and waits for it. Returns the exit code, whether
    /// the entry point returned or the title called `exit`/`ExitProcess`.
    pub fn run(&self) -> Result<u32, MapError> {
        if self.main.image.entry_point == 0 {
            return Err(MapError::ExecFailed("image has no entry point".into()));
        }
        let _veh = crate::veh::VehGuard::install();
        let mut tid = 0u32;
        // SAFETY: `guest_main` takes the leaked module; 8 MiB like the console.
        let handle = unsafe {
            CreateThread(
                std::ptr::null_mut(),
                8 * 1024 * 1024,
                guest_main,
                self.main as *const TitleModule as *mut c_void,
                0,
                &mut tid,
            )
        };
        if handle.is_null() {
            return Err(MapError::ExecFailed(
                std::io::Error::last_os_error().to_string(),
            ));
        }
        GUEST_MAIN_TID.store(tid, Ordering::SeqCst);
        GUEST_MAIN_HANDLE.store(handle as usize, Ordering::SeqCst);
        let mut code = 0u32;
        // SAFETY: our own thread handle.
        unsafe {
            WaitForSingleObject(handle, INFINITE);
            GetExitCodeThread(handle, &mut code);
            CloseHandle(handle);
        }
        GUEST_MAIN_TID.store(0, Ordering::SeqCst);
        GUEST_MAIN_HANDLE.store(0, Ordering::SeqCst);
        if EXIT_REQUESTED.swap(false, Ordering::SeqCst) {
            code = EXIT_CODE.load(Ordering::SeqCst);
        }
        Ok(code)
    }
}

// ---------------------------------------------------------------------------
// Process exit
// ---------------------------------------------------------------------------

/// Ends the title, not the process: the guest's main thread stops and
/// [`Title::run`] returns `code`. Outside a running title, the host's own
/// `ExitProcess`.
fn title_exit(code: u32) -> ! {
    if GUEST_MAIN_TID.load(Ordering::SeqCst) == 0 {
        type ExitProcess = unsafe extern "system" fn(u32) -> !;
        // SAFETY: the host export has this signature.
        let real: ExitProcess =
            unsafe { std::mem::transmute(host_fn!("kernel32.dll", "ExitProcess")) };
        unsafe { real(code) }
    }
    EXIT_CODE.store(code, Ordering::SeqCst);
    EXIT_REQUESTED.store(true, Ordering::SeqCst);
    // SAFETY: plain thread calls; no Rust locals with destructors are live.
    unsafe {
        if GetCurrentThreadId() != GUEST_MAIN_TID.load(Ordering::SeqCst) {
            TerminateThread(GUEST_MAIN_HANDLE.load(Ordering::SeqCst) as Handle, code);
        }
        ExitThread(code)
    }
}

unsafe extern "system" fn hook_exit_process(code: u32) -> ! {
    title_exit(code)
}

unsafe extern "C" fn hook_crt_exit(code: i32) -> ! {
    title_exit(code as u32)
}

unsafe extern "system" fn hook_terminate_process(process: Handle, code: u32) -> i32 {
    // `GetCurrentProcess()` is the pseudo handle -1.
    if process as isize == -1 && GUEST_MAIN_TID.load(Ordering::SeqCst) != 0 {
        title_exit(code)
    }
    type F = unsafe extern "system" fn(Handle, u32) -> i32;
    // SAFETY: the host export has this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("kernel32.dll", "TerminateProcess")) };
    unsafe { real(process, code) }
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

struct Start {
    proc_: usize,
    param: *mut c_void,
    void_ret: bool,
}

fn boxed_start(proc_: usize, param: *mut c_void, void_ret: bool) -> *mut c_void {
    Box::into_raw(Box::new(Start {
        proc_,
        param,
        void_ret,
    })) as *mut c_void
}

unsafe extern "system" fn thread_trampoline(p: *mut c_void) -> u32 {
    // SAFETY: `p` is a `Box<Start>` made by `boxed_start`.
    let start = unsafe { Box::from_raw(p as *mut Start) };
    notify_all(DLL_THREAD_ATTACH);
    let code = if start.void_ret {
        // SAFETY: the title's `_beginthread` routine.
        let f: VoidThreadProc = unsafe { std::mem::transmute(start.proc_) };
        unsafe { f(start.param) };
        0
    } else {
        // SAFETY: the title's thread routine.
        let f: ThreadProc = unsafe { std::mem::transmute(start.proc_) };
        unsafe { f(start.param) }
    };
    notify_all(DLL_THREAD_DETACH);
    code
}

unsafe extern "C" fn void_trampoline(p: *mut c_void) {
    // SAFETY: as `thread_trampoline`.
    unsafe { thread_trampoline(p) };
}

unsafe extern "system" fn hook_create_thread(
    sa: *mut c_void,
    stack: usize,
    start: Option<ThreadProc>,
    param: *mut c_void,
    flags: u32,
    tid: *mut u32,
) -> Handle {
    let Some(start) = start else {
        // SAFETY: no preconditions. ERROR_INVALID_PARAMETER, as the host says.
        unsafe { SetLastError(87) };
        return std::ptr::null_mut();
    };
    let boxed = boxed_start(start as usize, param, false);
    // SAFETY: the trampoline owns `boxed` once the thread starts.
    let h = unsafe { CreateThread(sa, stack, thread_trampoline, boxed, flags, tid) };
    if h.is_null() {
        // SAFETY: the thread never started; we still own it.
        drop(unsafe { Box::from_raw(boxed as *mut Start) });
    }
    h
}

unsafe extern "C" fn hook_begin_thread_ex(
    security: *mut c_void,
    stack: u32,
    start: usize,
    arg: *mut c_void,
    flags: u32,
    thrdaddr: *mut u32,
) -> usize {
    type F =
        unsafe extern "C" fn(*mut c_void, u32, ThreadProc, *mut c_void, u32, *mut u32) -> usize;
    // SAFETY: the host export has this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("ucrtbase.dll", "_beginthreadex")) };
    let boxed = boxed_start(start, arg, false);
    let h = unsafe { real(security, stack, thread_trampoline, boxed, flags, thrdaddr) };
    if h == 0 {
        drop(unsafe { Box::from_raw(boxed as *mut Start) });
    }
    h
}

unsafe extern "C" fn hook_begin_thread(start: usize, stack: u32, arg: *mut c_void) -> usize {
    type F = unsafe extern "C" fn(VoidThreadProc, u32, *mut c_void) -> usize;
    // SAFETY: the host export has this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("ucrtbase.dll", "_beginthread")) };
    let boxed = boxed_start(start, arg, true);
    let h = unsafe { real(void_trampoline, stack, boxed) };
    if h == usize::MAX {
        drop(unsafe { Box::from_raw(boxed as *mut Start) });
    }
    h
}

// ---------------------------------------------------------------------------
// Library loader
// ---------------------------------------------------------------------------

/// # Safety
/// `p` is null or a NUL-terminated UTF-16 string.
unsafe fn wide_str(p: *const u16) -> Option<String> {
    if p.is_null() {
        return None;
    }
    let len = (0..).take_while(|&i| unsafe { *p.add(i) } != 0).count();
    Some(String::from_utf16_lossy(unsafe {
        std::slice::from_raw_parts(p, len)
    }))
}

/// # Safety
/// `p` is null or a NUL-terminated string.
unsafe fn ansi_str(p: *const i8) -> Option<String> {
    (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

/// What `LoadLibrary` of `raw` gives the title, or `None` to ask the host.
fn load_library_for_title(raw: &str, flags: u32) -> Option<Handle> {
    if flags & LOAD_AS_DATA != 0 {
        return None;
    }
    let name = module_file_name(raw);
    if classify_dll(&name) == DllClass::StrataHle {
        return crate::hooks::hook_load_library(&name).map(|h| h as Handle);
    }
    if let Some(m) = find_module(&name) {
        return Some(m.image.module);
    }
    if !may_be_title_dll(&name) {
        return None;
    }
    let path = title_dll_file(&name)?;
    match load_module(&name, &path) {
        Ok(m) => {
            attach_pending();
            Some(m.image.module)
        }
        Err(e) => {
            crate::crash::log_line(&format!("LoadLibrary({raw}) failed: {e}"));
            // SAFETY: no preconditions.
            unsafe { SetLastError(ERROR_MOD_NOT_FOUND) };
            Some(std::ptr::null_mut())
        }
    }
}

unsafe extern "system" fn hook_load_library_w(name: *const u16) -> Handle {
    unsafe { hook_load_library_ex_w(name, std::ptr::null_mut(), 0) }
}

unsafe extern "system" fn hook_load_library_a(name: *const i8) -> Handle {
    unsafe { hook_load_library_ex_a(name, std::ptr::null_mut(), 0) }
}

unsafe extern "system" fn hook_load_library_ex_w(
    name: *const u16,
    file: Handle,
    flags: u32,
) -> Handle {
    if let Some(h) = unsafe { wide_str(name) }.and_then(|n| load_library_for_title(&n, flags)) {
        return h;
    }
    type F = unsafe extern "system" fn(*const u16, Handle, u32) -> Handle;
    // SAFETY: the host export has this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("kernel32.dll", "LoadLibraryExW")) };
    unsafe { real(name, file, flags) }
}

unsafe extern "system" fn hook_load_library_ex_a(
    name: *const i8,
    file: Handle,
    flags: u32,
) -> Handle {
    if let Some(h) = unsafe { ansi_str(name) }.and_then(|n| load_library_for_title(&n, flags)) {
        return h;
    }
    type F = unsafe extern "system" fn(*const i8, Handle, u32) -> Handle;
    // SAFETY: the host export has this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("kernel32.dll", "LoadLibraryExA")) };
    unsafe { real(name, file, flags) }
}

unsafe extern "system" fn hook_load_packaged_library(name: *const u16, _reserved: u32) -> Handle {
    unsafe { hook_load_library_ex_w(name, std::ptr::null_mut(), 0) }
}

unsafe extern "system" fn hook_get_proc_address(module: Handle, name: *const i8) -> usize {
    let func = if (name as usize) >> 16 == 0 {
        format!("ordinal#{}", name as usize)
    } else {
        // SAFETY: a name, not an ordinal.
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned()
    };
    if let Some(dll) = crate::hooks::pseudo_dll(module as usize) {
        return crate::thunks::resolve_hle(&dll, &func).unwrap_or_else(|| {
            // SAFETY: no preconditions.
            unsafe { SetLastError(ERROR_PROC_NOT_FOUND) };
            0
        });
    }
    if let Some(m) = module_by_handle(module as usize) {
        return m.image.export(&func).unwrap_or_else(|| {
            // SAFETY: no preconditions.
            unsafe { SetLastError(ERROR_PROC_NOT_FOUND) };
            0
        });
    }
    // A host module: our override of it where import binding would use one.
    if let Some(dll) = host_module_name(module)
        && let Some(addr) = import_override(&dll, &func)
    {
        return addr;
    }
    type F = unsafe extern "system" fn(Handle, *const i8) -> usize;
    // SAFETY: the host export has this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("kernel32.dll", "GetProcAddress")) };
    unsafe { real(module, name) }
}

fn host_module_name(module: Handle) -> Option<String> {
    type F = unsafe extern "system" fn(Handle, *mut u16, u32) -> u32;
    // SAFETY: the host export has this signature; `buf` holds 260 units.
    let real: F = unsafe { std::mem::transmute(host_fn!("kernel32.dll", "GetModuleFileNameW")) };
    let mut buf = [0u16; 260];
    let n = unsafe { real(module, buf.as_mut_ptr(), buf.len() as u32) } as usize;
    (n > 0).then(|| module_file_name(&String::from_utf16_lossy(&buf[..n])))
}

/// The handle `GetModuleHandle(name)` gives the title, or `None` for the host's answer.
fn module_handle_for_title(name: Option<String>) -> Option<Handle> {
    let Some(name) = name else {
        return registry().main.map(|m| m.image.module);
    };
    let name = module_file_name(&name);
    if let Some(m) = find_module(&name) {
        return Some(m.image.module);
    }
    if classify_dll(&name) == DllClass::StrataHle {
        return crate::hooks::hook_load_library(&name).map(|h| h as Handle);
    }
    None
}

unsafe extern "system" fn hook_get_module_handle_w(name: *const u16) -> Handle {
    if let Some(h) = module_handle_for_title(unsafe { wide_str(name) }) {
        return h;
    }
    type F = unsafe extern "system" fn(*const u16) -> Handle;
    // SAFETY: the host export has this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("kernel32.dll", "GetModuleHandleW")) };
    unsafe { real(name) }
}

unsafe extern "system" fn hook_get_module_handle_a(name: *const i8) -> Handle {
    if let Some(h) = module_handle_for_title(unsafe { ansi_str(name) }) {
        return h;
    }
    type F = unsafe extern "system" fn(*const i8) -> Handle;
    // SAFETY: the host export has this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("kernel32.dll", "GetModuleHandleA")) };
    unsafe { real(name) }
}

/// # Safety
/// `out` is a valid pointer; `name` is what `flags` says it is.
unsafe fn module_handle_ex_for_title(
    flags: u32,
    name: Option<String>,
    addr: usize,
    out: *mut Handle,
) -> bool {
    let found = if flags & GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS != 0 {
        registry()
            .modules
            .iter()
            .find(|m| m.contains(addr))
            .map(|m| m.image.module)
    } else if addr == 0 {
        module_handle_for_title(None)
    } else {
        module_handle_for_title(name)
    };
    match found {
        Some(h) if !out.is_null() => {
            unsafe { *out = h };
            true
        }
        _ => false,
    }
}

unsafe extern "system" fn hook_get_module_handle_ex_w(
    flags: u32,
    name: *const u16,
    out: *mut Handle,
) -> i32 {
    let text = if flags & GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS != 0 {
        None
    } else {
        unsafe { wide_str(name) }
    };
    if unsafe { module_handle_ex_for_title(flags, text, name as usize, out) } {
        return 1;
    }
    type F = unsafe extern "system" fn(u32, *const u16, *mut Handle) -> i32;
    // SAFETY: the host export has this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("kernel32.dll", "GetModuleHandleExW")) };
    unsafe { real(flags, name, out) }
}

unsafe extern "system" fn hook_get_module_handle_ex_a(
    flags: u32,
    name: *const i8,
    out: *mut Handle,
) -> i32 {
    let text = if flags & GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS != 0 {
        None
    } else {
        unsafe { ansi_str(name) }
    };
    if unsafe { module_handle_ex_for_title(flags, text, name as usize, out) } {
        return 1;
    }
    type F = unsafe extern "system" fn(u32, *const i8, *mut Handle) -> i32;
    // SAFETY: the host export has this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("kernel32.dll", "GetModuleHandleExA")) };
    unsafe { real(flags, name, out) }
}

/// The original path of the title module `module` names (NULL: the executable).
fn title_module_path(module: Handle) -> Option<PathBuf> {
    let reg = registry();
    if module.is_null() {
        return reg.main.map(|m| m.path.clone());
    }
    reg.modules
        .iter()
        .find(|m| m.image.module == module)
        .map(|m| m.path.clone())
}

/// Copies `text` + NUL into `buf` as `GetModuleFileName` does.
///
/// # Safety
/// `buf` holds `size` units.
unsafe fn write_name<T: Copy + Default>(text: &[T], buf: *mut T, size: u32) -> u32 {
    if buf.is_null() || size == 0 {
        unsafe { SetLastError(ERROR_INSUFFICIENT_BUFFER) };
        return 0;
    }
    let size = size as usize;
    let n = text.len().min(size - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr(), buf, n);
        *buf.add(n) = T::default();
    }
    if text.len() >= size {
        unsafe { SetLastError(ERROR_INSUFFICIENT_BUFFER) };
        return size as u32;
    }
    n as u32
}

unsafe extern "system" fn hook_get_module_file_name_w(
    module: Handle,
    buf: *mut u16,
    size: u32,
) -> u32 {
    if let Some(path) = title_module_path(module) {
        let wide: Vec<u16> = path.as_os_str().encode_wide().collect();
        return unsafe { write_name(&wide, buf, size) };
    }
    type F = unsafe extern "system" fn(Handle, *mut u16, u32) -> u32;
    // SAFETY: the host export has this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("kernel32.dll", "GetModuleFileNameW")) };
    unsafe { real(module, buf, size) }
}

unsafe extern "system" fn hook_get_module_file_name_a(
    module: Handle,
    buf: *mut i8,
    size: u32,
) -> u32 {
    if let Some(path) = title_module_path(module) {
        let text: Vec<i8> = path.to_string_lossy().bytes().map(|b| b as i8).collect();
        return unsafe { write_name(&text, buf, size) };
    }
    type F = unsafe extern "system" fn(Handle, *mut i8, u32) -> u32;
    // SAFETY: the host export has this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("kernel32.dll", "GetModuleFileNameA")) };
    unsafe { real(module, buf, size) }
}

unsafe extern "system" fn hook_free_library(module: Handle) -> i32 {
    // Title images and pseudo handles stay mapped for the process's life.
    if crate::hooks::pseudo_dll(module as usize).is_some()
        || module_by_handle(module as usize).is_some()
    {
        return 1;
    }
    type F = unsafe extern "system" fn(Handle) -> i32;
    // SAFETY: the host export has this signature.
    let real: F = unsafe { std::mem::transmute(host_fn!("kernel32.dll", "FreeLibrary")) };
    unsafe { real(module) }
}

fn is_kernel_family(dll: &str) -> bool {
    let d = dll.to_ascii_lowercase();
    d.starts_with("kernel32")
        || d.starts_with("kernelx")
        || d.starts_with("kernelbase")
        || d.starts_with("api-ms-win-core-libraryloader-")
        || d.starts_with("api-ms-win-core-processthreads-")
        || d.starts_with("api-ms-win-core-kernel32-legacy-")
}

fn is_crt_family(dll: &str) -> bool {
    let d = dll.to_ascii_lowercase();
    d.starts_with("ucrtbase") || d.starts_with("api-ms-win-crt-runtime-") || d.starts_with("msvcrt")
}

/// Strata's replacement for a host export, if it has one.
pub fn import_override(dll: &str, func: &str) -> Option<usize> {
    if let Some(f) = crate::files::file_override(dll, func) {
        return Some(f);
    }
    let f: usize = if is_kernel_family(dll) {
        match func {
            "LoadLibraryA" => hook_load_library_a as *const () as usize,
            "LoadLibraryW" => hook_load_library_w as *const () as usize,
            "LoadLibraryExA" => hook_load_library_ex_a as *const () as usize,
            "LoadLibraryExW" => hook_load_library_ex_w as *const () as usize,
            "LoadPackagedLibrary" => hook_load_packaged_library as *const () as usize,
            "GetProcAddress" => hook_get_proc_address as *const () as usize,
            "GetModuleHandleA" => hook_get_module_handle_a as *const () as usize,
            "GetModuleHandleW" => hook_get_module_handle_w as *const () as usize,
            "GetModuleHandleExA" => hook_get_module_handle_ex_a as *const () as usize,
            "GetModuleHandleExW" => hook_get_module_handle_ex_w as *const () as usize,
            "GetModuleFileNameA" => hook_get_module_file_name_a as *const () as usize,
            "GetModuleFileNameW" => hook_get_module_file_name_w as *const () as usize,
            "FreeLibrary" => hook_free_library as *const () as usize,
            "ExitProcess" => hook_exit_process as *const () as usize,
            "TerminateProcess" => hook_terminate_process as *const () as usize,
            "CreateThread" => hook_create_thread as *const () as usize,
            _ => return None,
        }
    } else if is_crt_family(dll) {
        match func {
            "exit" | "_exit" | "_Exit" | "quick_exit" => hook_crt_exit as *const () as usize,
            "_beginthreadex" => hook_begin_thread_ex as *const () as usize,
            "_beginthread" => hook_begin_thread as *const () as usize,
            _ => return None,
        }
    } else {
        return None;
    };
    Some(f)
}

#[cfg(all(test, windows, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn module_names_normalise_like_the_windows_loader() {
        assert_eq!(module_file_name(r"C:\Game\Bink2W64"), "bink2w64.dll");
        assert_eq!(module_file_name("XboxBvhBuilder.DLL"), "xboxbvhbuilder.dll");
        assert_eq!(module_file_name("game.exe"), "game.exe");
    }

    #[test]
    fn overrides_cover_loader_thread_exit_and_file_calls() {
        for (dll, func) in [
            ("kernel32.dll", "LoadLibraryExW"),
            ("KERNELX.dll", "GetModuleHandleW"),
            ("api-ms-win-core-libraryloader-l1-2-0.dll", "GetProcAddress"),
            ("api-ms-win-core-processthreads-l1-1-3.dll", "CreateThread"),
            ("api-ms-win-crt-runtime-l1-1-0.dll", "exit"),
            ("ucrtbase.dll", "_beginthreadex"),
            ("kernel32.dll", "CreateFileW"),
        ] {
            assert!(import_override(dll, func).is_some(), "{dll}!{func}");
        }
        assert_eq!(import_override("user32.dll", "LoadLibraryW"), None);
        assert_eq!(import_override("kernel32.dll", "Sleep"), None);
    }

    #[test]
    fn pseudo_handles_answer_get_proc_address() {
        let h = unsafe { hook_load_library_a(c"xmem.dll".as_ptr()) };
        assert!(!h.is_null());
        let p = unsafe { hook_get_proc_address(h, c"XMemIsToolsMemAvailable".as_ptr()) };
        assert_eq!(
            Some(p),
            crate::thunks::resolve_hle("xmem.dll", "XMemIsToolsMemAvailable")
        );
        assert_eq!(
            unsafe { hook_get_proc_address(h, c"NoSuchThing".as_ptr()) },
            0
        );
        assert_eq!(unsafe { hook_free_library(h) }, 1);
    }

    #[test]
    fn host_libraries_still_reach_the_host_with_overrides() {
        let k32 = unsafe { hook_load_library_a(c"kernel32.dll".as_ptr()) };
        assert!(!k32.is_null());
        let create = unsafe { hook_get_proc_address(k32, c"CreateFileW".as_ptr()) };
        assert_eq!(
            Some(create),
            crate::files::file_override("kernel32.dll", "CreateFileW")
        );
        let sleep = unsafe { hook_get_proc_address(k32, c"Sleep".as_ptr()) };
        assert_eq!(sleep, crate::loader::host_export("kernel32.dll", "Sleep"));
    }

    #[test]
    fn thread_hook_runs_the_routine_and_returns_its_code() {
        unsafe extern "system" fn routine(p: *mut c_void) -> u32 {
            p as u32 + 1
        }
        let mut tid = 0;
        let h = unsafe {
            hook_create_thread(
                std::ptr::null_mut(),
                0,
                Some(routine),
                41 as *mut c_void,
                0,
                &mut tid,
            )
        };
        assert!(!h.is_null());
        let mut code = 0;
        unsafe {
            WaitForSingleObject(h, INFINITE);
            GetExitCodeThread(h, &mut code);
            CloseHandle(h);
        }
        assert_eq!(code, 42);
    }
}
