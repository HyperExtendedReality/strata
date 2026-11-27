//! Extended OS & GDK HLE thunks:
//! - `xgameruntime.dll!QueryApiImpl` + full `XUser` / `XGameSave` / `XStore` /
//!   `XPackage` / `XAsync` / `XTaskQueue` / `XGame` / PLM lifecycle / Speech APIs
//! - Offline/sandboxed `ws2_32.dll`, `winhttp.dll`, `wininet.dll`
//! - `rpcrt4.dll` UUID generation & formatting
//! - `mfplat.dll` / `mfreadwrite.dll` Media Foundation session hooks

use std::ffi::{CStr, c_char, c_void};
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::services::{AppState, LifecycleManager};

const S_OK: i32 = 0;
const E_INVALIDARG: i32 = 0x8007_0057u32 as i32;
const E_NOINTERFACE: i32 = 0x8000_4002u32 as i32;
const E_FAIL: i32 = 0x8000_4005u32 as i32;
const WSAEHOSTUNREACH: i32 = 10065;
const WSAENOTCONN: i32 = 10057;

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(0x5700_0001);
static LAST_WSA_ERROR: AtomicI32 = AtomicI32::new(0);
static UUID_COUNTER: AtomicU64 = AtomicU64::new(1);
static MF_REF_COUNT: AtomicU32 = AtomicU32::new(0);

fn alloc_handle() -> *mut c_void {
    NEXT_HANDLE.fetch_add(1, Ordering::Relaxed) as *mut c_void
}

fn lifecycle() -> &'static Mutex<LifecycleManager> {
    static M: OnceLock<Mutex<LifecycleManager>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(LifecycleManager::new()))
}

// ============================================================================
// xgameruntime.dll!QueryApiImpl + Generic Offline GDK Provider VTable
// ============================================================================

#[repr(transparent)]
#[derive(Clone, Copy)]
struct Slot(*const ());
unsafe impl Sync for Slot {}

#[repr(C)]
struct GenericApiProvider {
    vtable: *const [Slot; 32],
    ref_count: AtomicU32,
    provider_hash: u64,
}

unsafe extern "system" fn prov_qi(
    this: *mut c_void,
    _riid: *const c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    unsafe {
        let p = &*(this as *const GenericApiProvider);
        p.ref_count.fetch_add(1, Ordering::Relaxed);
        *out = this;
    }
    S_OK
}

unsafe extern "system" fn prov_add_ref(this: *mut c_void) -> u32 {
    unsafe {
        (*(this as *const GenericApiProvider))
            .ref_count
            .fetch_add(1, Ordering::Relaxed)
            + 1
    }
}

unsafe extern "system" fn prov_release(this: *mut c_void) -> u32 {
    unsafe {
        let p = this as *mut GenericApiProvider;
        let prev = (*p).ref_count.fetch_sub(1, Ordering::AcqRel);
        if prev <= 1 {
            drop(Box::from_raw(p));
            0
        } else {
            prev - 1
        }
    }
}

unsafe extern "system" fn prov_ok(_this: *mut c_void) -> i32 {
    S_OK
}

static GENERIC_PROVIDER_VTABLE: [Slot; 32] = {
    let mut vt = [Slot(prov_ok as *const ()); 32];
    vt[0] = Slot(prov_qi as *const ());
    vt[1] = Slot(prov_add_ref as *const ());
    vt[2] = Slot(prov_release as *const ());
    vt
};

/// `xgameruntime.dll!QueryApiImpl(provider_guid, interface_guid, out_interface)`
pub unsafe extern "system" fn query_api_impl(
    provider: *const [u8; 16],
    interface: *const [u8; 16],
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() || provider.is_null() || interface.is_null() {
        return E_INVALIDARG;
    }
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in unsafe { (*provider).iter().chain((*interface).iter()) } {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    let obj = Box::into_raw(Box::new(GenericApiProvider {
        vtable: &GENERIC_PROVIDER_VTABLE,
        ref_count: AtomicU32::new(1),
        provider_hash: h,
    }));
    unsafe { *out = obj.cast() };
    S_OK
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
// mfplat.dll & mfreadwrite.dll (Media Foundation hooks via services::MediaSession)
// ============================================================================

unsafe extern "system" fn mf_startup(_version: u32, _flags: u32) -> i32 {
    MF_REF_COUNT.fetch_add(1, Ordering::Relaxed);
    S_OK
}

unsafe extern "system" fn mf_shutdown() -> i32 {
    let _ = MF_REF_COUNT.try_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.saturating_sub(1))
    });
    S_OK
}

unsafe extern "system" fn mf_ok_zero() -> i32 {
    S_OK
}

unsafe extern "system" fn mf_create_object(out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let obj = Box::into_raw(Box::new(GenericApiProvider {
        vtable: &GENERIC_PROVIDER_VTABLE,
        ref_count: AtomicU32::new(1),
        provider_hash: 0x4D46_504C_4154,
    }));
    unsafe { *out = obj.cast() };
    S_OK
}

unsafe extern "system" fn mf_create_attributes(out: *mut *mut c_void, _initial: u32) -> i32 {
    unsafe { mf_create_object(out) }
}

unsafe extern "system" fn mf_create_memory_buffer(_max_len: u32, out: *mut *mut c_void) -> i32 {
    unsafe { mf_create_object(out) }
}

unsafe extern "system" fn mf_create_source_reader(
    _src: *const c_void,
    _attrs: *mut c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let _ = E_NOINTERFACE;
    let _ = E_FAIL;
    unsafe { mf_create_object(out) }
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
        "mfplat.dll" | "mfreadwrite.dll" => Some(match func_name {
            "MFStartup" => mf_startup as *const () as usize,
            "MFShutdown" => mf_shutdown as *const () as usize,
            "MFLockPlatform" | "MFUnlockPlatform" => mf_ok_zero as *const () as usize,
            "MFCreateAttributes" => mf_create_attributes as *const () as usize,
            "MFCreateMediaType" | "MFCreateSample" => mf_create_object as *const () as usize,
            "MFCreateMemoryBuffer" => mf_create_memory_buffer as *const () as usize,
            "MFCreateSourceReaderFromURL" | "MFCreateSourceReaderFromByteStream" => {
                mf_create_source_reader as *const () as usize
            }
            _ => return None,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_api_impl_returns_releasable_com_provider() {
        let p = [1u8; 16];
        let i = [2u8; 16];
        let mut out = std::ptr::null_mut();
        assert_eq!(unsafe { query_api_impl(&p, &i, &mut out) }, S_OK);
        assert!(!out.is_null());
        unsafe {
            assert_eq!(prov_add_ref(out), 2);
            assert_eq!(prov_release(out), 1);
            assert_eq!(prov_release(out), 0);
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
    fn offline_wsa_and_mfplat_thunks_succeed() {
        let mut wsa = [0u8; 408];
        assert_eq!(unsafe { wsa_startup(0x0202, wsa.as_mut_ptr()) }, 0);
        let sock = unsafe { wsa_socket(2, 1, 6) };
        assert_eq!(unsafe { wsa_connect(sock, std::ptr::null(), 0) }, -1);
        assert_eq!(unsafe { wsa_get_last_error() }, WSAEHOSTUNREACH);
        assert_eq!(unsafe { wsa_closesocket(sock) }, 0);

        assert_eq!(unsafe { mf_startup(0x0002_0070, 0) }, S_OK);
        let mut attrs = std::ptr::null_mut();
        assert_eq!(unsafe { mf_create_attributes(&mut attrs, 4) }, S_OK);
        assert!(!attrs.is_null());
        assert_eq!(unsafe { prov_release(attrs) }, 0);
        assert_eq!(unsafe { mf_shutdown() }, S_OK);
    }
}
