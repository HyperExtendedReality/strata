//! Configuration (`strata.toml` + `STRATA_*` env overrides), external-path
//! policy enforcement, and lightweight level-filtered structured logging.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};

use crate::profile::ConsoleProfile;

/// Structured log verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LogLevel {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
}

impl LogLevel {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "error" | "err" => Self::Error,
            "warn" | "warning" => Self::Warn,
            "debug" | "dbg" => Self::Debug,
            "trace" => Self::Trace,
            _ => Self::Info,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warn => "WARN",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Trace => "TRACE",
        }
    }
}

static ACTIVE_LOG_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Info as u8);

/// Initialize log verbosity from `STRATA_LOG` (defaults to `info`).
pub fn init_logging() -> LogLevel {
    let level = std::env::var("STRATA_LOG")
        .map(|s| LogLevel::from_str_lossy(&s))
        .unwrap_or(LogLevel::Info);
    set_log_level(level);
    level
}

pub fn set_log_level(level: LogLevel) {
    ACTIVE_LOG_LEVEL.store(level as u8, Ordering::Relaxed);
}

pub fn log_level() -> LogLevel {
    match ACTIVE_LOG_LEVEL.load(Ordering::Relaxed) {
        0 => LogLevel::Error,
        1 => LogLevel::Warn,
        2 => LogLevel::Info,
        3 => LogLevel::Debug,
        _ => LogLevel::Trace,
    }
}

/// Emit a formatted log line to stderr if `level <= log_level()`.
pub fn log_msg(level: LogLevel, target: &str, msg: &str) {
    if level <= log_level() {
        eprintln!("[strata:{}:{target}] {msg}", level.as_str());
    }
}

/// Resolved Strata configuration (TOML file + `STRATA_*` env overrides).
#[derive(Debug, Clone, PartialEq)]
pub struct StrataConfig {
    pub game_dir: Option<PathBuf>,
    pub fidelityfx_dir: Option<PathBuf>,
    pub cache_dir: PathBuf,
    pub data_dir: PathBuf,
    pub gpu_adapter: Option<String>,
    pub output_override: Option<(u32, u32)>,
    pub fsr_4k_preset: Option<String>,
    pub console: ConsoleProfile,
    pub frame_interval_ms: Option<f64>,
    pub unimplemented: String,
    pub safe_mode: bool,
    pub capture_frame: Option<u64>,
}

impl Default for StrataConfig {
    fn default() -> Self {
        let base = default_app_data_root();
        Self {
            game_dir: None,
            fidelityfx_dir: None,
            cache_dir: base.join("cache"),
            data_dir: base.join("data"),
            gpu_adapter: None,
            output_override: None,
            fsr_4k_preset: None,
            console: ConsoleProfile::series_x(),
            frame_interval_ms: None,
            unimplemented: "trap".to_string(),
            safe_mode: false,
            capture_frame: None,
        }
    }
}

fn default_app_data_root() -> PathBuf {
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        PathBuf::from(local).join("strata")
    } else {
        std::env::temp_dir().join("strata")
    }
}

/// Parse a `WIDTHxHEIGHT` string such as `"1920x1080"`.
pub fn parse_resolution(spec: &str) -> Option<(u32, u32)> {
    let (w, h) = spec.trim().split_once(['x', 'X'])?;
    let w = w.trim().parse::<u32>().ok()?;
    let h = h.trim().parse::<u32>().ok()?;
    if w == 0 || h == 0 {
        return None;
    }
    Some((w, h))
}

/// Zero-dependency parser for the flat `[section]` / `key = value` TOML subset
/// used by `strata.toml`.
pub fn parse_strata_toml(text: &str) -> StrataConfig {
    let mut cfg = StrataConfig::default();
    let mut section = String::new();
    let mut kv: HashMap<String, String> = HashMap::new();

    for raw_line in text.lines() {
        let line = strip_toml_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].trim().to_ascii_lowercase();
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            let key = if section.is_empty() {
                k.trim().to_ascii_lowercase()
            } else {
                format!("{section}.{}", k.trim().to_ascii_lowercase())
            };
            let val = unquote_toml(v.trim());
            kv.insert(key, val);
        }
    }

    if let Some(v) = kv.get("paths.game_dir") {
        cfg.game_dir = Some(PathBuf::from(v));
    }
    if let Some(v) = kv.get("paths.fidelityfx_dir") {
        cfg.fidelityfx_dir = Some(PathBuf::from(v));
    }
    if let Some(v) = kv.get("paths.cache_dir") {
        cfg.cache_dir = PathBuf::from(v);
    }
    if let Some(v) = kv.get("paths.data_dir") {
        cfg.data_dir = PathBuf::from(v);
    }
    if let Some(v) = kv.get("gpu.adapter") {
        cfg.gpu_adapter = Some(v.clone());
    }
    if let Some(v) = kv.get("gpu.output") {
        cfg.output_override = parse_resolution(v);
    }
    if let Some(v) = kv.get("gpu.fsr_4k") {
        cfg.fsr_4k_preset = Some(v.to_ascii_lowercase());
    }
    if let Some(v) = kv.get("runtime.console")
        && let Some(profile) = ConsoleProfile::from_name(v)
    {
        cfg.console = profile;
    }
    if let Some(v) = kv.get("runtime.frame_interval_ms")
        && let Ok(ms) = v.parse::<f64>()
        && ms > 0.0
    {
        cfg.frame_interval_ms = Some(ms);
    }
    if let Some(v) = kv.get("runtime.unimplemented") {
        cfg.unimplemented = v.to_ascii_lowercase();
    }
    cfg
}

fn strip_toml_comment(line: &str) -> &str {
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    for (i, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_double => escaped = true,
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '#' if !in_single && !in_double => return &line[..i],
            _ => {}
        }
    }
    line
}

fn unquote_toml(v: &str) -> String {
    if v.len() >= 2 && v.starts_with('\'') && v.ends_with('\'') {
        v[1..v.len() - 1].to_string()
    } else if v.len() >= 2 && v.starts_with('"') && v.ends_with('"') {
        v[1..v.len() - 1]
            .replace("\\\\", "\\")
            .replace("\\\"", "\"")
    } else {
        v.to_string()
    }
}

impl StrataConfig {
    /// Load from `STRATA_CONFIG` (or `./strata.toml` if present), then apply
    /// `STRATA_*` environment variable overrides.
    pub fn load() -> Self {
        let path = std::env::var("STRATA_CONFIG")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("strata.toml"));
        let mut cfg = std::fs::read_to_string(&path)
            .map(|s| parse_strata_toml(&s))
            .unwrap_or_default();
        cfg.apply_env_overrides();
        cfg
    }

    pub fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("STRATA_GAME_DIR")
            && !v.trim().is_empty()
        {
            self.game_dir = Some(PathBuf::from(v));
        }
        if let Ok(v) = std::env::var("STRATA_FIDELITYFX_DIR")
            && !v.trim().is_empty()
        {
            self.fidelityfx_dir = Some(PathBuf::from(v));
        }
        if let Ok(v) = std::env::var("STRATA_CACHE_DIR")
            && !v.trim().is_empty()
        {
            self.cache_dir = PathBuf::from(v);
        }
        if let Ok(v) = std::env::var("STRATA_DATA_DIR")
            && !v.trim().is_empty()
        {
            self.data_dir = PathBuf::from(v);
        }
        if let Ok(v) = std::env::var("STRATA_GPU_ADAPTER")
            && !v.trim().is_empty()
        {
            self.gpu_adapter = Some(v);
        }
        if let Ok(v) = std::env::var("STRATA_OUTPUT")
            && let Some(res) = parse_resolution(&v)
        {
            self.output_override = Some(res);
        }
        if let Ok(v) = std::env::var("STRATA_FSR_4K")
            && !v.trim().is_empty()
        {
            self.fsr_4k_preset = Some(v.to_ascii_lowercase());
        }
        if let Ok(v) = std::env::var("STRATA_CONSOLE")
            && let Some(profile) = ConsoleProfile::from_name(&v)
        {
            self.console = profile;
        }
        if let Ok(v) = std::env::var("STRATA_FRAME_INTERVAL_MS")
            && let Ok(ms) = v.parse::<f64>()
            && ms > 0.0
        {
            self.frame_interval_ms = Some(ms);
        }
        if let Ok(v) = std::env::var("STRATA_SAFE") {
            self.safe_mode = matches!(v.trim(), "1" | "true" | "TRUE" | "yes");
        }
        if let Ok(v) = std::env::var("STRATA_CAPTURE_FRAME")
            && let Ok(f) = v.parse::<u64>()
        {
            self.capture_frame = Some(f);
        }
    }

    /// Enforce that none of the configured data/game/cache paths live inside
    /// `repo_root` (matching `docs/asset-policy.md`).
    pub fn validate_paths_outside(&self, repo_root: &Path) -> Result<(), String> {
        let canonical_root =
            std::fs::canonicalize(repo_root).unwrap_or_else(|_| repo_root.to_path_buf());
        let checks: [(&str, Option<&Path>); 4] = [
            ("game_dir", self.game_dir.as_deref()),
            ("fidelityfx_dir", self.fidelityfx_dir.as_deref()),
            ("cache_dir", Some(self.cache_dir.as_path())),
            ("data_dir", Some(self.data_dir.as_path())),
        ];
        for (name, opt_path) in checks {
            let Some(p) = opt_path else { continue };
            if is_inside_dir(p, &canonical_root) {
                return Err(format!(
                    "policy violation: `{name}` ({}) must be outside the repository ({})",
                    p.display(),
                    canonical_root.display()
                ));
            }
        }
        Ok(())
    }
}

/// Returns `true` if `candidate` resolves inside (or equal to) `root`.
pub fn is_inside_dir(candidate: &Path, root: &Path) -> bool {
    let norm_root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let norm_cand = std::fs::canonicalize(candidate).unwrap_or_else(|_| {
        if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            std::env::current_dir()
                .map(|c| c.join(candidate))
                .unwrap_or_else(|_| candidate.to_path_buf())
        }
    });
    let r = normalize_case_path(&norm_root);
    let c = normalize_case_path(&norm_cand);
    c == r || c.starts_with(&format!("{r}{}", std::path::MAIN_SEPARATOR))
}

fn normalize_case_path(p: &Path) -> String {
    let s = p.to_string_lossy().replace('/', "\\");
    let trimmed = s.trim_start_matches(r"\\?\").trim_end_matches('\\');
    trimmed.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_strata_toml_and_resolution() {
        let toml = r#"
# comment
[paths]
game_dir = 'D:\strata-data\title' # trailing comment
cache_dir = "D:\\strata-data\\cache"

[gpu]
adapter = "Radeon"
output = "1920x1080"
fsr_4k = "quality"

[runtime]
console = "series-s"
frame_interval_ms = 16.667
unimplemented = "zero"
"#;
        let cfg = parse_strata_toml(toml);
        assert_eq!(cfg.game_dir, Some(PathBuf::from(r"D:\strata-data\title")));
        assert_eq!(cfg.cache_dir, PathBuf::from(r"D:\strata-data\cache"));
        assert_eq!(cfg.gpu_adapter.as_deref(), Some("Radeon"));
        assert_eq!(cfg.output_override, Some((1920, 1080)));
        assert_eq!(cfg.fsr_4k_preset.as_deref(), Some("quality"));
        assert_eq!(cfg.console, ConsoleProfile::series_s());
        assert_eq!(cfg.frame_interval_ms, Some(16.667));
        assert_eq!(cfg.unimplemented, "zero");
    }

    #[test]
    fn rejects_paths_inside_repo_root() {
        let repo = PathBuf::from(r"C:\dev\strata");
        let mut cfg = StrataConfig {
            game_dir: Some(PathBuf::from(r"C:\dev\strata\game")),
            cache_dir: PathBuf::from(r"D:\external\cache"),
            data_dir: PathBuf::from(r"D:\external\data"),
            ..Default::default()
        };
        assert!(cfg.validate_paths_outside(&repo).is_err());

        cfg.game_dir = Some(PathBuf::from(r"D:\external\game"));
        assert!(cfg.validate_paths_outside(&repo).is_ok());
    }
}
