//! Console platform services HLE:
//! - Lifecycle (suspend / resume / constrained state & PLM callbacks)
//! - Display (display modes, HDR10 metadata, VRR, refresh rates)
//! - Media (clean-room Media Foundation / hardware video decode session hooks)
//! - Offline Networking (`XNetworking` connectivity hints)
//! - Speech (offline TTS / STT stubs)
//! - FidelityFX (`FSR2` / `FSR3` / `CAS` No-API dispatch descriptors)
//! - Virtual Registry (in-memory console `HKLM` / `HKCU` hive)

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// Lifecycle (Suspend / Resume / Constrained PLM)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppState {
    Running,
    Constrained,
    Suspended,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleEvent {
    Suspending,
    Resuming,
    ConstrainedChanged(bool),
}

pub struct LifecycleManager {
    pub state: AppState,
    next_token: AtomicU64,
    callbacks: HashMap<u64, Box<dyn Fn(LifecycleEvent) + Send + Sync>>,
    pub suspend_count: u64,
    pub resume_count: u64,
}

impl Default for LifecycleManager {
    fn default() -> Self {
        Self::new()
    }
}

impl LifecycleManager {
    pub fn new() -> Self {
        Self {
            state: AppState::Running,
            next_token: AtomicU64::new(1),
            callbacks: HashMap::new(),
            suspend_count: 0,
            resume_count: 0,
        }
    }

    pub fn register_callback(
        &mut self,
        cb: impl Fn(LifecycleEvent) + Send + Sync + 'static,
    ) -> u64 {
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        self.callbacks.insert(token, Box::new(cb));
        token
    }

    pub fn unregister_callback(&mut self, token: u64) {
        self.callbacks.remove(&token);
    }

    pub fn trigger_suspend(&mut self) {
        self.state = AppState::Suspended;
        self.suspend_count += 1;
        for cb in self.callbacks.values() {
            cb(LifecycleEvent::Suspending);
        }
    }

    pub fn trigger_resume(&mut self) {
        self.state = AppState::Running;
        self.resume_count += 1;
        for cb in self.callbacks.values() {
            cb(LifecycleEvent::Resuming);
        }
    }

    pub fn set_constrained(&mut self, constrained: bool) {
        self.state = if constrained {
            AppState::Constrained
        } else {
            AppState::Running
        };
        for cb in self.callbacks.values() {
            cb(LifecycleEvent::ConstrainedChanged(constrained));
        }
    }
}

// ============================================================================
// Display (Modes, HDR10 Metadata, VRR)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayMode {
    pub width: u32,
    pub height: u32,
    pub refresh_hz: u32,
    pub hdr_capable: bool,
    pub vrr_capable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdrMetadata {
    pub max_luminance_nits: f32,
    pub min_luminance_nits: f32,
    pub max_content_light_level: u16,
    pub max_frame_avg_light_level: u16,
}

#[derive(Debug, Clone)]
pub struct DisplayService {
    pub active_mode: DisplayMode,
    pub supported_modes: Vec<DisplayMode>,
    pub hdr_enabled: bool,
    pub hdr_metadata: HdrMetadata,
}

impl Default for DisplayService {
    fn default() -> Self {
        let modes = vec![
            DisplayMode {
                width: 1920,
                height: 1080,
                refresh_hz: 60,
                hdr_capable: true,
                vrr_capable: true,
            },
            DisplayMode {
                width: 1920,
                height: 1080,
                refresh_hz: 120,
                hdr_capable: true,
                vrr_capable: true,
            },
            DisplayMode {
                width: 2560,
                height: 1440,
                refresh_hz: 60,
                hdr_capable: true,
                vrr_capable: true,
            },
            DisplayMode {
                width: 2560,
                height: 1440,
                refresh_hz: 120,
                hdr_capable: true,
                vrr_capable: true,
            },
            DisplayMode {
                width: 3840,
                height: 2160,
                refresh_hz: 60,
                hdr_capable: true,
                vrr_capable: true,
            },
            DisplayMode {
                width: 3840,
                height: 2160,
                refresh_hz: 120,
                hdr_capable: true,
                vrr_capable: true,
            },
        ];
        Self {
            active_mode: modes[0],
            supported_modes: modes,
            hdr_enabled: false,
            hdr_metadata: HdrMetadata {
                max_luminance_nits: 1000.0,
                min_luminance_nits: 0.001,
                max_content_light_level: 1000,
                max_frame_avg_light_level: 400,
            },
        }
    }
}

// ============================================================================
// Media (Clean-Room Video Session & Frame Queue)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    H264,
    Hevc,
    Vp9,
    Av1,
}

#[derive(Debug, Clone)]
pub struct VideoSession {
    pub id: u32,
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    pub frames_submitted: u64,
}

#[derive(Debug, Default)]
pub struct MediaService {
    next_id: u32,
    pub sessions: HashMap<u32, VideoSession>,
}

impl MediaService {
    pub fn create_session(&mut self, codec: VideoCodec, width: u32, height: u32) -> u32 {
        self.next_id += 1;
        let id = self.next_id;
        self.sessions.insert(
            id,
            VideoSession {
                id,
                codec,
                width,
                height,
                frames_submitted: 0,
            },
        );
        id
    }

    pub fn submit_packet(&mut self, id: u32, _payload: &[u8]) -> Option<Vec<u8>> {
        let s = self.sessions.get_mut(&id)?;
        s.frames_submitted += 1;
        // Return a neutral black RGBA frame placeholder when hardware decode is unavailable.
        Some(vec![0u8; (s.width.min(64) * s.height.min(64) * 4) as usize])
    }
}

// ============================================================================
// Offline Networking (`XNetworking`)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectivityLevel {
    Unknown = 0,
    None = 1,
    LocalAccess = 2,
    InternetAccess = 3,
    ConstrainedInternetAccess = 4,
}

#[derive(Debug, Clone, Copy)]
pub struct ConnectivityHint {
    pub level: ConnectivityLevel,
    pub data_limit_in_megabytes: u32,
    pub network_initialized: bool,
}

#[derive(Debug, Default)]
pub struct OfflineNetworkingService {
    pub allow_lan: bool,
    pub denied_requests: u64,
}

impl OfflineNetworkingService {
    pub fn connectivity_hint(&self) -> ConnectivityHint {
        ConnectivityHint {
            level: if self.allow_lan {
                ConnectivityLevel::LocalAccess
            } else {
                ConnectivityLevel::None
            },
            data_limit_in_megabytes: 0,
            network_initialized: true,
        }
    }
}

// ============================================================================
// Speech (Offline TTS / STT Stub)
// ============================================================================

#[derive(Debug, Default)]
pub struct SpeechService {
    pub synthesis_calls: u64,
    pub recognition_calls: u64,
}

impl SpeechService {
    pub fn synthesize_text(&mut self, _text: &str) -> Vec<i16> {
        self.synthesis_calls += 1;
        Vec::new()
    }
}

// ============================================================================
// FidelityFX (No-API Upscaling / Sharpening Descriptors)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfxEffect {
    Cas,
    Fsr2,
    Fsr3,
    Spd,
}

#[derive(Debug, Clone)]
pub struct FfxContext {
    pub id: u32,
    pub effect: FfxEffect,
    pub max_render_size: [u32; 2],
    pub display_size: [u32; 2],
    pub dispatches: u64,
}

#[derive(Debug, Default)]
pub struct FidelityFxService {
    next_id: u32,
    pub contexts: HashMap<u32, FfxContext>,
}

impl FidelityFxService {
    pub fn create_context(
        &mut self,
        effect: FfxEffect,
        render: [u32; 2],
        display: [u32; 2],
    ) -> u32 {
        self.next_id += 1;
        let id = self.next_id;
        self.contexts.insert(
            id,
            FfxContext {
                id,
                effect,
                max_render_size: render,
                display_size: display,
                dispatches: 0,
            },
        );
        id
    }

    pub fn record_dispatch(&mut self, id: u32) -> bool {
        if let Some(ctx) = self.contexts.get_mut(&id) {
            ctx.dispatches += 1;
            true
        } else {
            false
        }
    }
}

/// Standard 4K FSR upscaling presets (matching retail 4K SDR profiles).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fsr4KPreset {
    Quality,
    Balanced,
    Performance,
}

impl Fsr4KPreset {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "quality" | "q" => Some(Self::Quality),
            "balanced" | "b" => Some(Self::Balanced),
            "performance" | "perf" | "p" => Some(Self::Performance),
            _ => None,
        }
    }

    /// `(input_width, input_height)` for a `3840x2160` output.
    pub const fn input_size(self) -> [u32; 2] {
        match self {
            Self::Quality => [2560, 1440],
            Self::Balanced => [2258, 1270],
            Self::Performance => [1920, 1080],
        }
    }

    pub const fn output_size(self) -> [u32; 2] {
        [3840, 2160]
    }
}

/// Discovered PC FidelityFX runtime library in `STRATA_FIDELITYFX_DIR`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FidelityFxBridge {
    pub dll_path: std::path::PathBuf,
    pub backend: &'static str,
}

impl FidelityFxBridge {
    /// Look for a signed FidelityFX upscaler DLL in `dir`.
    pub fn discover(dir: &std::path::Path) -> Option<Self> {
        const CANDIDATES: &[(&str, &str)] = &[
            ("amd_fidelityfx_vk.dll", "vulkan"),
            ("amd_fidelityfx_upscaler_dx12.dll", "dx12"),
            ("amd_fidelityfx_dx12.dll", "dx12"),
        ];
        for &(name, backend) in CANDIDATES {
            let p = dir.join(name);
            if p.is_file() {
                return Some(Self {
                    dll_path: p,
                    backend,
                });
            }
        }
        None
    }

    /// Dynamically load the discovered FidelityFX DLL via `LoadLibraryW` and
    /// resolve its primary entry points (`ffxCreateContext`, `ffxDispatch`,
    /// `ffxQuery`, `ffxDestroyContext`).
    #[cfg(windows)]
    pub fn load_runtime(&self) -> Result<LoadedFidelityFx, String> {
        use std::ffi::{CString, c_void};
        use std::os::windows::ffi::OsStrExt;
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn LoadLibraryW(lp_lib_file_name: *const u16) -> *mut c_void;
            fn GetProcAddress(module: *mut c_void, name: *const i8) -> *const c_void;
            fn FreeLibrary(module: *mut c_void) -> i32;
        }
        let wide: Vec<u16> = self
            .dll_path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let hmod = unsafe { LoadLibraryW(wide.as_ptr()) };
        if hmod.is_null() {
            return Err(format!(
                "LoadLibraryW failed for {}",
                self.dll_path.display()
            ));
        }
        let resolve = |sym: &str| -> Option<usize> {
            let c = CString::new(sym).ok()?;
            let p = unsafe { GetProcAddress(hmod, c.as_ptr()) } as usize;
            (p != 0).then_some(p)
        };
        let create_context = resolve("ffxCreateContext");
        let dispatch = resolve("ffxDispatch");
        let query = resolve("ffxQuery");
        let destroy_context = resolve("ffxDestroyContext");
        if create_context.is_none() && dispatch.is_none() && query.is_none() {
            unsafe {
                let _ = FreeLibrary(hmod);
            }
            return Err(format!(
                "{} loaded but exports none of ffxCreateContext/ffxDispatch/ffxQuery",
                self.dll_path.display()
            ));
        }
        Ok(LoadedFidelityFx {
            hmod: hmod as usize,
            dll_path: self.dll_path.clone(),
            backend: self.backend,
            create_context,
            dispatch,
            query,
            destroy_context,
        })
    }
}

/// Live dynamically-loaded FidelityFX runtime handle.
#[derive(Debug)]
pub struct LoadedFidelityFx {
    hmod: usize,
    pub dll_path: std::path::PathBuf,
    pub backend: &'static str,
    pub create_context: Option<usize>,
    pub dispatch: Option<usize>,
    pub query: Option<usize>,
    pub destroy_context: Option<usize>,
}

#[cfg(windows)]
impl Drop for LoadedFidelityFx {
    fn drop(&mut self) {
        use std::ffi::c_void;
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn FreeLibrary(module: *mut c_void) -> i32;
        }
        if self.hmod != 0 {
            unsafe {
                let _ = FreeLibrary(self.hmod as *mut c_void);
            }
        }
    }
}

/// Offline `rpcrt4.dll` UUID/GUID generator (deterministic RFC 4122 v4 layout).
pub fn offline_uuid_create(counter: u64) -> [u8; 16] {
    let mut b = [0u8; 16];
    let hi = counter.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x5354_5241_5441_0001;
    let lo = counter.wrapping_add(1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    b[0..8].copy_from_slice(&hi.to_le_bytes());
    b[8..16].copy_from_slice(&lo.to_le_bytes());
    b[6] = (b[6] & 0x0f) | 0x40; // version 4
    b[8] = (b[8] & 0x3f) | 0x80; // variant RFC 4122
    b
}

// ============================================================================
// Virtual Console Registry Hive
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegValue {
    Dword(u32),
    Qword(u64),
    Sz(String),
    Binary(Vec<u8>),
}

#[derive(Debug, Default)]
pub struct VirtualRegistry {
    keys: HashMap<String, HashMap<String, RegValue>>,
}

impl VirtualRegistry {
    pub fn new() -> Self {
        let mut reg = Self::default();
        reg.set_value(
            "HKLM\\SOFTWARE\\Strata\\Console",
            "Platform",
            RegValue::Sz("XboxSeries".into()),
        );
        reg.set_value(
            "HKLM\\SOFTWARE\\Strata\\Console",
            "OfflineMode",
            RegValue::Dword(1),
        );
        reg
    }

    fn normalize(key: &str) -> String {
        key.replace('/', "\\").to_ascii_uppercase()
    }

    pub fn set_value(&mut self, key: &str, name: &str, value: RegValue) {
        self.keys
            .entry(Self::normalize(key))
            .or_default()
            .insert(name.to_ascii_uppercase(), value);
    }

    pub fn get_value(&self, key: &str, name: &str) -> Option<&RegValue> {
        self.keys
            .get(&Self::normalize(key))?
            .get(&name.to_ascii_uppercase())
    }

    pub fn delete_value(&mut self, key: &str, name: &str) -> bool {
        self.keys
            .get_mut(&Self::normalize(key))
            .and_then(|vals| vals.remove(&name.to_ascii_uppercase()))
            .is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn lifecycle_notifies_suspend_and_resume() {
        let mut lm = LifecycleManager::new();
        let hits = Arc::new(AtomicU32::new(0));
        let h = hits.clone();
        let tok = lm.register_callback(move |_| {
            h.fetch_add(1, Ordering::SeqCst);
        });
        lm.trigger_suspend();
        assert_eq!(lm.state, AppState::Suspended);
        lm.trigger_resume();
        assert_eq!(lm.state, AppState::Running);
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        lm.unregister_callback(tok);
    }

    #[test]
    fn virtual_registry_and_fidelityfx_work() {
        let mut reg = VirtualRegistry::new();
        reg.set_value("HKCU\\Software\\Test", "Volume", RegValue::Dword(80));
        assert_eq!(
            reg.get_value("hkcu/software/test", "volume"),
            Some(&RegValue::Dword(80))
        );

        let mut ffx = FidelityFxService::default();
        let ctx = ffx.create_context(FfxEffect::Fsr2, [1920, 1080], [3840, 2160]);
        assert!(ffx.record_dispatch(ctx));
        assert_eq!(ffx.contexts[&ctx].dispatches, 1);
    }
}
