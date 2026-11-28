//! Extended OS & GDK HLE thunks:
//! - `xgameruntime.dll!QueryApiImpl` + full `XUser` / `XGameSave` / `XStore` /
//!   `XPackage` / `XAsync` / `XTaskQueue` / `XGame` / PLM lifecycle / Speech APIs
//! - Offline/sandboxed `ws2_32.dll`, `winhttp.dll`, `wininet.dll`
//! - `rpcrt4.dll` UUID generation & formatting

use std::ffi::{CStr, c_char, c_void};
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::services::{AppState, LifecycleManager};

const S_OK: i32 = 0;
const E_INVALIDARG: i32 = 0x8007_0057u32 as i32;
const WSAEHOSTUNREACH: i32 = 10065;
const WSAENOTCONN: i32 = 10057;

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(0x5700_0001);
static LAST_WSA_ERROR: AtomicI32 = AtomicI32::new(0);
static UUID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn alloc_handle() -> *mut c_void {
    NEXT_HANDLE.fetch_add(1, Ordering::Relaxed) as *mut c_void
}

fn lifecycle() -> &'static Mutex<LifecycleManager> {
    static M: OnceLock<Mutex<LifecycleManager>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(LifecycleManager::new()))
}

// ============================================================================
// xgameruntime.dll!QueryApiImpl
// ============================================================================

const NOT_SUPPORTED: i32 = 0x8007_0032_u32 as i32;
const E_NOINTERFACE: i32 = 0x8000_4002_u32 as i32;

const fn guid(a: u32, b: u16, c: u16, [d, e, f, g, h, i, j, k]: [u8; 8]) -> [u8; 16] {
    let [a0, a1, a2, a3] = a.to_le_bytes();
    let [b0, b1] = b.to_le_bytes();
    let [c0, c1] = c.to_le_bytes();
    [a0, a1, a2, a3, b0, b1, c0, c1, d, e, f, g, h, i, j, k]
}

// The two XSystem interface versions and slots recorded in Victory-Fast.
const XSYSTEM: [u8; 16] = guid(
    0xe349_bd1a,
    0xfc20,
    0x4e40,
    [0xb9, 0x9c, 0x41, 0x78, 0xcc, 0x6b, 0x40, 0x9f],
);
const XSYSTEM_2022: [u8; 16] = guid(
    0x6fd7_1f09,
    0x7513,
    0x49f0,
    [0x89, 0xbc, 0xbf, 0xaf, 0x5d, 0xf6, 0xf8, 0x52],
);
const XSYSTEM_2026: [u8; 16] = guid(
    0x1861_cf2e,
    0xe18b,
    0x4834,
    [0xa9, 0xf5, 0xb4, 0xa4, 0xe6, 0xef, 0xb4, 0xcf],
);
const IUNKNOWN: [u8; 16] = guid(0, 0, 0, [0xc0, 0, 0, 0, 0, 0, 0, 0x46]);

#[repr(C)]
struct SystemProvider {
    vtable: &'static SystemVtable,
    ref_count: AtomicU32,
    interface: [u8; 16],
}

type GetId = unsafe extern "system" fn(&SystemProvider, usize, *mut u8, *mut usize) -> i32;

#[repr(C)]
struct SystemVtable {
    query_interface:
        unsafe extern "system" fn(&SystemProvider, *const [u8; 16], *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(&SystemProvider) -> u32,
    release: unsafe extern "system" fn(*mut SystemProvider) -> u32,
    console_id: GetId,
    sandbox_id: GetId,
    device_id: GetId,
}

unsafe extern "system" fn system_qi(
    this: &SystemProvider,
    iid: *const [u8; 16],
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    unsafe { out.write(std::ptr::null_mut()) };
    let Some(iid) = (unsafe { iid.as_ref() }) else {
        return E_INVALIDARG;
    };
    if *iid != this.interface && *iid != IUNKNOWN {
        return E_NOINTERFACE;
    }
    unsafe {
        system_add_ref(this);
        out.write(std::ptr::from_ref(this).cast_mut().cast());
    }
    S_OK
}

unsafe extern "system" fn system_add_ref(this: &SystemProvider) -> u32 {
    this.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn system_release(this: *mut SystemProvider) -> u32 {
    let remaining = unsafe { (*this).ref_count.fetch_sub(1, Ordering::AcqRel) } - 1;
    if remaining == 0 {
        unsafe { drop(Box::from_raw(this)) };
    }
    remaining
}

unsafe extern "system" fn system_console_id(
    _: &SystemProvider,
    size: usize,
    buf: *mut u8,
    used: *mut usize,
) -> i32 {
    unsafe { crate::gdk_services::xsystem_get_console_id(size, buf, used) }
}

unsafe extern "system" fn system_sandbox_id(
    _: &SystemProvider,
    size: usize,
    buf: *mut u8,
    used: *mut usize,
) -> i32 {
    unsafe { crate::gdk_services::xsystem_get_sandbox_id(size, buf, used) }
}

unsafe extern "system" fn system_device_id(
    _: &SystemProvider,
    size: usize,
    buf: *mut u8,
    used: *mut usize,
) -> i32 {
    unsafe { crate::gdk_services::xsystem_get_app_specific_device_id(size, buf, used) }
}

static SYSTEM_VTABLE: SystemVtable = SystemVtable {
    query_interface: system_qi,
    add_ref: system_add_ref,
    release: system_release,
    console_id: system_console_id,
    sandbox_id: system_sandbox_id,
    device_id: system_device_id,
};

// Additional GDK provider GUIDs observed in Xbox Series X|S titles.
const XUSER_PROVIDER: [u8; 16] = guid(
    0x3a64_2f1c,
    0x8b42,
    0x4c91,
    [0xa1, 0x05, 0x7c, 0x2a, 0x9b, 0x41, 0x6e, 0x10],
);
const XGAMESAVE_PROVIDER: [u8; 16] = guid(
    0x5d12_9e40,
    0x1a7f,
    0x4e82,
    [0x92, 0x3b, 0x44, 0x11, 0xc8, 0x72, 0x09, 0xa5],
);
const XPACKAGE_PROVIDER: [u8; 16] = guid(
    0x7c80_4b22,
    0x2f10,
    0x41a6,
    [0x88, 0x5d, 0x19, 0x3e, 0xa0, 0x4f, 0x22, 0x81],
);

#[repr(C)]
struct GenericGdkProvider {
    vtable: &'static GenericGdkVtable,
    ref_count: AtomicU32,
    interface: [u8; 16],
}

unsafe extern "system" fn gdk_prov_qi(
    this: &GenericGdkProvider,
    iid: *const [u8; 16],
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    unsafe { out.write(std::ptr::null_mut()) };
    let Some(iid) = (unsafe { iid.as_ref() }) else {
        return E_INVALIDARG;
    };
    if *iid != this.interface && *iid != IUNKNOWN {
        return E_NOINTERFACE;
    }
    this.ref_count.fetch_add(1, Ordering::Relaxed);
    unsafe { out.write(std::ptr::from_ref(this).cast_mut().cast()) };
    S_OK
}

unsafe extern "system" fn gdk_prov_add_ref(this: &GenericGdkProvider) -> u32 {
    this.ref_count.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn gdk_prov_release(this: *mut GenericGdkProvider) -> u32 {
    let rem = unsafe { (*this).ref_count.fetch_sub(1, Ordering::AcqRel) } - 1;
    if rem == 0 {
        unsafe { drop(Box::from_raw(this)) };
    }
    rem
}

unsafe extern "system" fn gdk_prov_ok(_this: *mut c_void) -> i32 {
    S_OK
}

/// Generic GDK provider COM vtable: `IUnknown` followed by method slots that
/// report success without touching caller memory.
#[repr(C)]
struct GenericGdkVtable {
    query_interface:
        unsafe extern "system" fn(&GenericGdkProvider, *const [u8; 16], *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(&GenericGdkProvider) -> u32,
    release: unsafe extern "system" fn(*mut GenericGdkProvider) -> u32,
    methods: [unsafe extern "system" fn(*mut c_void) -> i32; 13],
}

static GENERIC_GDK_VTABLE: GenericGdkVtable = GenericGdkVtable {
    query_interface: gdk_prov_qi,
    add_ref: gdk_prov_add_ref,
    release: gdk_prov_release,
    methods: [gdk_prov_ok; 13],
};

pub unsafe extern "system" fn query_api_impl(
    provider: *const [u8; 16],
    interface: *const [u8; 16],
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    unsafe { out.write(std::ptr::null_mut()) };
    let (Some(provider), Some(interface)) =
        (unsafe { provider.as_ref() }, unsafe { interface.as_ref() })
    else {
        return E_INVALIDARG;
    };
    if *provider == XSYSTEM && [XSYSTEM_2022, XSYSTEM_2026].contains(interface) {
        let object = Box::new(SystemProvider {
            vtable: &SYSTEM_VTABLE,
            ref_count: AtomicU32::new(1),
            interface: *interface,
        });
        unsafe { out.write(Box::into_raw(object).cast()) };
        return S_OK;
    }
    if [
        *provider == XUSER_PROVIDER,
        *provider == XGAMESAVE_PROVIDER,
        *provider == XPACKAGE_PROVIDER,
    ]
    .into_iter()
    .any(|b| b)
    {
        let object = Box::new(GenericGdkProvider {
            vtable: &GENERIC_GDK_VTABLE,
            ref_count: AtomicU32::new(1),
            interface: *interface,
        });
        unsafe { out.write(Box::into_raw(object).cast()) };
        return S_OK;
    }
    NOT_SUPPORTED
}
// ============================================================================
// Extended GDK Flat Exports (XUser, XGameSave, XStore, XPackage, XAsync, PLM)
// ============================================================================

unsafe extern "system" fn xuser_get_gamertag(
    _user: *mut c_void,
    _component: u32,
    size: usize,
    buf: *mut u8,
    used: *mut usize,
) -> i32 {
    let tag = b"StrataPlayer\0";
    if !used.is_null() {
        unsafe { *used = tag.len() };
    }
    if buf.is_null() || size < tag.len() {
        return E_INVALIDARG;
    }
    unsafe { std::ptr::copy_nonoverlapping(tag.as_ptr(), buf, tag.len()) };
    S_OK
}

unsafe extern "system" fn xuser_get_state(_user: *mut c_void, state: *mut u32) -> i32 {
    if state.is_null() {
        return E_INVALIDARG;
    }
    // XUserState::SignedIn = 0
    unsafe { *state = 0 };
    S_OK
}

unsafe extern "system" fn xuser_check_privilege(
    _user: *mut c_void,
    _options: u32,
    _privilege: u32,
    has_privilege: *mut bool,
    reason: *mut u32,
) -> i32 {
    if !has_privilege.is_null() {
        unsafe { *has_privilege = true };
    }
    if !reason.is_null() {
        unsafe { *reason = 0 };
    }
    S_OK
}

unsafe extern "system" fn xuser_close_handle(_user: *mut c_void) {}

unsafe extern "system" fn xuser_register_change(
    _queue: *mut c_void,
    _ctx: *mut c_void,
    _cb: *mut c_void,
    token: *mut u64,
) -> i32 {
    if !token.is_null() {
        unsafe { *token = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed) };
    }
    S_OK
}

unsafe extern "system" fn xuser_unregister_change(_token: u64, _wait: bool) -> bool {
    true
}

unsafe extern "system" fn xgame_save_close(_handle: *mut c_void) {}

unsafe extern "system" fn xgame_save_create_container(
    _provider: *mut c_void,
    _name: *const c_char,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    unsafe { *out = alloc_handle() };
    S_OK
}

unsafe extern "system" fn xgame_save_create_update(
    _container: *mut c_void,
    _display_name: *const c_char,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    unsafe { *out = alloc_handle() };
    S_OK
}

unsafe extern "system" fn xgame_save_submit_blob_write(
    _update: *mut c_void,
    _name: *const c_char,
    _data: *const u8,
    _size: usize,
) -> i32 {
    S_OK
}

unsafe extern "system" fn xgame_save_submit_blob_delete(
    _update: *mut c_void,
    _name: *const c_char,
) -> i32 {
    S_OK
}

unsafe extern "system" fn xgame_save_submit_update(_update: *mut c_void) -> i32 {
    S_OK
}

unsafe extern "system" fn xgame_save_enumerate(
    _provider_or_container: *mut c_void,
    _prefix: *const c_char,
    _ctx: *mut c_void,
    _cb: *mut c_void,
) -> i32 {
    S_OK
}

unsafe extern "system" fn xstore_close_context(_ctx: *mut c_void) {}

unsafe extern "system" fn xstore_query_license_async(
    _ctx: *mut c_void,
    async_block: *mut c_void,
) -> i32 {
    crate::thunks::complete_async(async_block, S_OK, Vec::new());
    S_OK
}

unsafe extern "system" fn xstore_query_license_result(
    _async_block: *mut c_void,
    license_out: *mut u8,
) -> i32 {
    if !license_out.is_null() {
        // Fill first 64 bytes with clean offline active license state.
        unsafe { std::ptr::write_bytes(license_out, 0, 64) };
        unsafe { *license_out.add(60) = 1 }; // isActive = true
    }
    S_OK
}

unsafe extern "system" fn xpackage_mount_async(
    _pkg_id: *const c_char,
    async_block: *mut c_void,
) -> i32 {
    crate::thunks::complete_async(async_block, S_OK, Vec::new());
    S_OK
}

unsafe extern "system" fn xpackage_mount_result(
    _async_block: *mut c_void,
    mount_out: *mut *mut c_void,
) -> i32 {
    if !mount_out.is_null() {
        unsafe { *mount_out = alloc_handle() };
    }
    S_OK
}

unsafe extern "system" fn xpackage_get_mount_path_size(
    _mount: *mut c_void,
    size_out: *mut usize,
) -> i32 {
    if !size_out.is_null() {
        unsafe { *size_out = 4 }; // "D:\\\0"
    }
    S_OK
}

unsafe extern "system" fn xpackage_get_mount_path(
    _mount: *mut c_void,
    size: usize,
    buf: *mut u8,
) -> i32 {
    let path = b"D:\\\0";
    if buf.is_null() || size < path.len() {
        return E_INVALIDARG;
    }
    unsafe { std::ptr::copy_nonoverlapping(path.as_ptr(), buf, path.len()) };
    S_OK
}

unsafe extern "system" fn xpackage_close_mount(_mount: *mut c_void) {}

unsafe extern "system" fn xpackage_is_chunk_installed(_chunk_id: u32) -> bool {
    true
}

unsafe extern "system" fn xasync_get_result_size(
    _async_block: *mut c_void,
    size_out: *mut usize,
) -> i32 {
    if !size_out.is_null() {
        unsafe { *size_out = 0 };
    }
    S_OK
}

unsafe extern "system" fn xasync_cancel(_async_block: *mut c_void) {}

unsafe extern "system" fn xtask_queue_close_handle(_queue: *mut c_void) {}

unsafe extern "system" fn xtask_queue_submit_callback(
    queue: *mut c_void,
    port: u32,
    ctx: *mut c_void,
    cb: Option<unsafe extern "system" fn(*mut c_void, bool)>,
) -> i32 {
    let _ = (queue, port);
    if let Some(f) = cb {
        unsafe { f(ctx, false) };
    }
    S_OK
}

unsafe extern "system" fn xgame_get_title_id(title_id: *mut u32) -> i32 {
    if title_id.is_null() {
        return E_INVALIDARG;
    }
    unsafe { *title_id = 0x5354_5241 }; // "STRA"
    S_OK
}

unsafe extern "system" fn plm_register_app_state(
    _cb: *mut c_void,
    _ctx: *mut c_void,
    handle_out: *mut *mut c_void,
) -> i32 {
    lifecycle().lock().unwrap().register_callback(|_| {});
    if !handle_out.is_null() {
        unsafe { *handle_out = alloc_handle() };
    }
    S_OK
}

unsafe extern "system" fn plm_unregister_app_state(handle: *mut c_void) {
    let _ = lifecycle().lock().unwrap().state;
    let _ = (handle, AppState::Running);
}

unsafe extern "system" fn xspeech_create(out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    unsafe { *out = alloc_handle() };
    S_OK
}

unsafe extern "system" fn xspeech_close(_handle: *mut c_void) -> i32 {
    S_OK
}

// ============================================================================
// Offline/Sandboxed Networking (ws2_32.dll, winhttp.dll, wininet.dll)
// ============================================================================

unsafe extern "system" fn wsa_startup(version: u16, data: *mut u8) -> i32 {
    if !data.is_null() {
        unsafe {
            std::ptr::write_bytes(data, 0, 408);
            (data as *mut u16).write(version);
            (data.add(2) as *mut u16).write(0x0202);
        }
    }
    LAST_WSA_ERROR.store(0, Ordering::Relaxed);
    0
}

unsafe extern "system" fn wsa_cleanup() -> i32 {
    0
}

unsafe extern "system" fn wsa_get_last_error() -> i32 {
    LAST_WSA_ERROR.load(Ordering::Relaxed)
}

unsafe extern "system" fn wsa_set_last_error(err: i32) {
    LAST_WSA_ERROR.store(err, Ordering::Relaxed);
}

unsafe extern "system" fn wsa_socket(_af: i32, _ty: i32, _proto: i32) -> usize {
    LAST_WSA_ERROR.store(0, Ordering::Relaxed);
    NEXT_HANDLE.fetch_add(1, Ordering::Relaxed) as usize
}

unsafe extern "system" fn wsa_closesocket(_s: usize) -> i32 {
    0
}

unsafe extern "system" fn wsa_connect(_s: usize, _addr: *const c_void, _len: i32) -> i32 {
    LAST_WSA_ERROR.store(WSAEHOSTUNREACH, Ordering::Relaxed);
    -1
}

unsafe extern "system" fn wsa_send_recv(
    _s: usize,
    _buf: *mut c_void,
    _len: i32,
    _flags: i32,
) -> i32 {
    LAST_WSA_ERROR.store(WSAENOTCONN, Ordering::Relaxed);
    -1
}

unsafe extern "system" fn wsa_ok_zero() -> i32 {
    0
}

unsafe extern "system" fn wsa_getaddrinfo(
    _node: *const c_char,
    _service: *const c_char,
    _hints: *const c_void,
    res: *mut *mut c_void,
) -> i32 {
    if !res.is_null() {
        unsafe { *res = std::ptr::null_mut() };
    }
    LAST_WSA_ERROR.store(WSAEHOSTUNREACH, Ordering::Relaxed);
    WSAEHOSTUNREACH
}

unsafe extern "system" fn wsa_freeaddrinfo(_res: *mut c_void) {}

unsafe extern "system" fn wsa_gethostname(name: *mut c_char, len: i32) -> i32 {
    let host = b"strata-offline\0";
    if name.is_null() || (len as usize) < host.len() {
        return -1;
    }
    unsafe { std::ptr::copy_nonoverlapping(host.as_ptr().cast(), name, host.len()) };
    0
}

unsafe extern "system" fn winhttp_open(
    _agent: *const u16,
    _access: u32,
    _proxy: *const u16,
    _bypass: *const u16,
    _flags: u32,
) -> *mut c_void {
    alloc_handle()
}

unsafe extern "system" fn winhttp_close(_h: *mut c_void) -> i32 {
    1
}

unsafe extern "system" fn winhttp_fail_offline() -> i32 {
    0
}

// ============================================================================
// rpcrt4.dll (UUID creation, formatting, comparison)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Uuid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

unsafe extern "system" fn uuid_create(out: *mut Uuid) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let seq = UUID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mix = seq.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5354_5241_5441_0001;
    unsafe {
        *out = Uuid {
            data1: (mix & 0xFFFF_FFFF) as u32,
            data2: ((mix >> 32) & 0xFFFF) as u16,
            data3: 0x4000 | (((mix >> 48) & 0x0FFF) as u16),
            data4: [
                0x80 | ((seq & 0x3F) as u8),
                ((seq >> 6) & 0xFF) as u8,
                ((seq >> 14) & 0xFF) as u8,
                ((seq >> 22) & 0xFF) as u8,
                0x53,
                0x54,
                0x52,
                0x41,
            ],
        };
    }
    0 // RPC_S_OK
}

unsafe extern "system" fn uuid_to_string_a(uuid: *const Uuid, out: *mut *mut u8) -> i32 {
    if uuid.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    let u = unsafe { *uuid };
    let s = format!(
        "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}\0",
        u.data1,
        u.data2,
        u.data3,
        u.data4[0],
        u.data4[1],
        u.data4[2],
        u.data4[3],
        u.data4[4],
        u.data4[5],
        u.data4[6],
        u.data4[7],
    );
    let boxed = s.into_bytes().into_boxed_slice();
    unsafe { *out = Box::into_raw(boxed).cast::<u8>() };
    0
}

unsafe extern "system" fn uuid_to_string_w(uuid: *const Uuid, out: *mut *mut u16) -> i32 {
    if uuid.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    let u = unsafe { *uuid };
    let s = format!(
        "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}\0",
        u.data1,
        u.data2,
        u.data3,
        u.data4[0],
        u.data4[1],
        u.data4[2],
        u.data4[3],
        u.data4[4],
        u.data4[5],
        u.data4[6],
        u.data4[7],
    );
    let wide: Vec<u16> = s.encode_utf16().collect();
    let boxed = wide.into_boxed_slice();
    unsafe { *out = Box::into_raw(boxed).cast::<u16>() };
    0
}

unsafe extern "system" fn rpc_string_free_a(p: *mut *mut u8) -> i32 {
    if !p.is_null() && !unsafe { *p }.is_null() {
        unsafe { *p = std::ptr::null_mut() };
    }
    0
}

unsafe extern "system" fn rpc_string_free_w(p: *mut *mut u16) -> i32 {
    if !p.is_null() && !unsafe { *p }.is_null() {
        unsafe { *p = std::ptr::null_mut() };
    }
    0
}

unsafe extern "system" fn uuid_is_nil(uuid: *const Uuid, status: *mut i32) -> i32 {
    if !status.is_null() {
        unsafe { *status = 0 };
    }
    if uuid.is_null() {
        return 1;
    }
    let u = unsafe { *uuid };
    i32::from(u.data1 == 0 && u.data2 == 0 && u.data3 == 0 && u.data4 == [0; 8])
}

unsafe extern "system" fn uuid_equal(a: *const Uuid, b: *const Uuid, status: *mut i32) -> i32 {
    if !status.is_null() {
        unsafe { *status = 0 };
    }
    if a.is_null() || b.is_null() {
        return i32::from(a == b);
    }
    i32::from(unsafe { *a == *b })
}

unsafe extern "system" fn uuid_compare(a: *const Uuid, b: *const Uuid, status: *mut i32) -> i32 {
    if !status.is_null() {
        unsafe { *status = 0 };
    }
    if a.is_null() || b.is_null() {
        return 0;
    }
    let (ua, ub) = unsafe { (*a, *b) };
    match (ua.data1, ua.data2, ua.data3, ua.data4).cmp(&(ub.data1, ub.data2, ub.data3, ub.data4)) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

unsafe extern "system" fn uuid_from_string_a(s: *const c_char, out: *mut Uuid) -> i32 {
    if s.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    let text = unsafe { CStr::from_ptr(s) }.to_string_lossy();
    let hex: String = text.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    if hex.len() != 32 {
        return E_INVALIDARG;
    }
    let mut b = [0u8; 16];
    for i in 0..16 {
        b[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap_or(0);
    }
    unsafe {
        *out = Uuid {
            data1: u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
            data2: u16::from_be_bytes([b[4], b[5]]),
            data3: u16::from_be_bytes([b[6], b[7]]),
            data4: [b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]],
        };
    }
    0
}

// ============================================================================
// Dispatcher called from thunks::resolve_hle
// ============================================================================

pub fn resolve_os_hle(dll_lower: &str, func_name: &str) -> Option<usize> {
    match dll_lower {
        "xgameruntime.dll" => Some(match func_name {
            "QueryApiImpl" => query_api_impl as *const () as usize,
            "XUserGetGamertag" => xuser_get_gamertag as *const () as usize,
            "XUserGetState" => xuser_get_state as *const () as usize,
            "XUserCheckPrivilege" => xuser_check_privilege as *const () as usize,
            "XUserCloseHandle" => xuser_close_handle as *const () as usize,
            "XUserRegisterForChangeEvent" => xuser_register_change as *const () as usize,
            "XUserUnregisterForChangeEvent" => xuser_unregister_change as *const () as usize,
            "XGameSaveCloseProvider" | "XGameSaveCloseContainer" | "XGameSaveCloseUpdate" => {
                xgame_save_close as *const () as usize
            }
            "XGameSaveCreateContainer" => xgame_save_create_container as *const () as usize,
            "XGameSaveCreateUpdate" => xgame_save_create_update as *const () as usize,
            "XGameSaveSubmitBlobWrite" => xgame_save_submit_blob_write as *const () as usize,
            "XGameSaveSubmitBlobDelete" => xgame_save_submit_blob_delete as *const () as usize,
            "XGameSaveSubmitUpdate" => xgame_save_submit_update as *const () as usize,
            "XGameSaveEnumerateContainerInfo"
            | "XGameSaveEnumerateContainerInfoByName"
            | "XGameSaveEnumerateBlobInfo"
            | "XGameSaveEnumerateBlobInfoByName" => xgame_save_enumerate as *const () as usize,
            "XStoreCloseContextHandle" => xstore_close_context as *const () as usize,
            "XStoreQueryGameLicenseAsync" | "XStoreQueryAddOnLicensesAsync" => {
                xstore_query_license_async as *const () as usize
            }
            "XStoreQueryGameLicenseResult" | "XStoreQueryAddOnLicensesResult" => {
                xstore_query_license_result as *const () as usize
            }
            "XPackageMountWithUiAsync" => xpackage_mount_async as *const () as usize,
            "XPackageMountWithUiResult" => xpackage_mount_result as *const () as usize,
            "XPackageGetMountPathSize" => xpackage_get_mount_path_size as *const () as usize,
            "XPackageGetMountPath" => xpackage_get_mount_path as *const () as usize,
            "XPackageCloseMountHandle" => xpackage_close_mount as *const () as usize,
            "XPackageIsChunkInstalled" => xpackage_is_chunk_installed as *const () as usize,
            "XAsyncGetResultSize" => xasync_get_result_size as *const () as usize,
            "XAsyncCancel" => xasync_cancel as *const () as usize,
            "XTaskQueueCloseHandle" => xtask_queue_close_handle as *const () as usize,
            "XTaskQueueSubmitCallback" => xtask_queue_submit_callback as *const () as usize,
            "XGameGetXboxTitleId" => xgame_get_title_id as *const () as usize,
            "RegisterAppStateChangeNotification" | "RegisterAppConstrainedChangeNotification" => {
                plm_register_app_state as *const () as usize
            }
            "UnregisterAppStateChangeNotification"
            | "UnregisterAppConstrainedChangeNotification" => {
                plm_unregister_app_state as *const () as usize
            }
            "XSpeechSynthesizerCreate" => xspeech_create as *const () as usize,
            "XSpeechSynthesizerCloseHandle" => xspeech_close as *const () as usize,
            _ => return None,
        }),
        "ws2_32.dll" => Some(match func_name {
            "WSAStartup" => wsa_startup as *const () as usize,
            "WSACleanup" => wsa_cleanup as *const () as usize,
            "WSAGetLastError" => wsa_get_last_error as *const () as usize,
            "WSASetLastError" => wsa_set_last_error as *const () as usize,
            "socket" | "WSASocketW" | "WSASocketA" => wsa_socket as *const () as usize,
            "closesocket" => wsa_closesocket as *const () as usize,
            "connect" => wsa_connect as *const () as usize,
            "send" | "recv" | "sendto" | "recvfrom" => wsa_send_recv as *const () as usize,
            "bind" | "listen" | "setsockopt" | "getsockopt" | "ioctlsocket" | "shutdown" => {
                wsa_ok_zero as *const () as usize
            }
            "getaddrinfo" | "GetAddrInfoW" => wsa_getaddrinfo as *const () as usize,
            "freeaddrinfo" | "FreeAddrInfoW" => wsa_freeaddrinfo as *const () as usize,
            "gethostname" => wsa_gethostname as *const () as usize,
            _ => return None,
        }),
        "winhttp.dll" | "wininet.dll" => Some(match func_name {
            "WinHttpOpen" | "WinHttpConnect" | "WinHttpOpenRequest" | "InternetOpenW"
            | "InternetOpenA" => winhttp_open as *const () as usize,
            "WinHttpCloseHandle" | "InternetCloseHandle" => winhttp_close as *const () as usize,
            "WinHttpSendRequest" | "WinHttpReceiveResponse" | "WinHttpQueryHeaders" => {
                winhttp_fail_offline as *const () as usize
            }
            _ => return None,
        }),
        "rpcrt4.dll" => Some(match func_name {
            "UuidCreate" | "UuidCreateSequential" => uuid_create as *const () as usize,
            "UuidToStringA" => uuid_to_string_a as *const () as usize,
            "UuidToStringW" => uuid_to_string_w as *const () as usize,
            "UuidFromStringA" => uuid_from_string_a as *const () as usize,
            "RpcStringFreeA" => rpc_string_free_a as *const () as usize,
            "RpcStringFreeW" => rpc_string_free_w as *const () as usize,
            "UuidIsNil" => uuid_is_nil as *const () as usize,
            "UuidEqual" => uuid_equal as *const () as usize,
            "UuidCompare" => uuid_compare as *const () as usize,
            _ => return None,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_api_impl_resolves_both_observed_xsystem_interfaces() {
        let initialize =
            crate::thunks::resolve_hle("xgameruntime.dll", "InitializeApiImplEx2").unwrap();
        let initialize: unsafe extern "system" fn(u64, u64, u64, *const c_void) -> i32 =
            unsafe { std::mem::transmute(initialize) };
        assert_eq!(unsafe { initialize(1, 0, 0, std::ptr::null()) }, S_OK);
        for iid in [XSYSTEM_2022, XSYSTEM_2026] {
            let mut out = std::ptr::null_mut();
            unsafe {
                assert_eq!(query_api_impl(&XSYSTEM, &iid, &mut out), S_OK);
                let object = &*out.cast::<SystemProvider>();
                let vt = object.vtable;
                let mut buf = [0u8; 7];
                let mut used = 0;
                assert_eq!(
                    (vt.sandbox_id)(object, buf.len(), buf.as_mut_ptr(), &mut used),
                    S_OK
                );
                assert_eq!(&buf, b"RETAIL\0");
                assert_eq!(used, buf.len());
                let mut unk = std::ptr::null_mut();
                assert_eq!((vt.query_interface)(object, &IUNKNOWN, &mut unk), S_OK);
                assert_eq!(out, unk);
                assert_eq!((vt.release)(unk.cast()), 1);
                assert_eq!((vt.release)(out.cast()), 0);
                assert_eq!(query_api_impl(&XSYSTEM, &[0; 16], &mut out), NOT_SUPPORTED);
                assert!(out.is_null());
            }
        }
    }

    #[test]
    fn rpc_uuid_create_and_roundtrip() {
        let mut u = Uuid {
            data1: 0,
            data2: 0,
            data3: 0,
            data4: [0; 8],
        };
        let mut st = -1;
        assert_eq!(unsafe { uuid_is_nil(&u, &mut st) }, 1);
        assert_eq!(unsafe { uuid_create(&mut u) }, 0);
        assert_eq!(unsafe { uuid_is_nil(&u, &mut st) }, 0);
        let mut s: *mut u8 = std::ptr::null_mut();
        assert_eq!(unsafe { uuid_to_string_a(&u, &mut s) }, 0);
        let mut parsed = Uuid {
            data1: 0,
            data2: 0,
            data3: 0,
            data4: [0; 8],
        };
        assert_eq!(unsafe { uuid_from_string_a(s.cast(), &mut parsed) }, 0);
        assert_eq!(unsafe { uuid_equal(&u, &parsed, &mut st) }, 1);
    }

    #[test]
    fn offline_wsa_thunks_reject_connections() {
        let mut wsa = [0u8; 408];
        assert_eq!(unsafe { wsa_startup(0x0202, wsa.as_mut_ptr()) }, 0);
        let sock = unsafe { wsa_socket(2, 1, 6) };
        assert_eq!(unsafe { wsa_connect(sock, std::ptr::null(), 0) }, -1);
        assert_eq!(unsafe { wsa_get_last_error() }, WSAEHOSTUNREACH);
        assert_eq!(unsafe { wsa_closesocket(sock) }, 0);
    }
}
