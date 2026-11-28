//! Opt-in counters sampled at the title's PresentX boundary.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Default, serde::Serialize)]
pub struct FrameCounters {
    pub preparation_ns: u64,
    pub fence_wait_ns: u64,
    pub graphics_gpu_ns: u64,
    pub upload_bytes: u64,
    pub readback_bytes: u64,
    pub pipeline_creation_ns: u64,
    pub pipeline_errors: u64,
    pub binding_errors: u64,
    pub predicated_commands: u64,
    pub present_errors: u64,
}

macro_rules! frame_counters {
    ($($field:ident),+ $(,)?) => {
        #[allow(non_upper_case_globals)]
        mod counters {
            use super::AtomicU64;
            $(pub(super) static $field: AtomicU64 = AtomicU64::new(0);)+
        }

        $(pub fn $field(value: u64) {
            if enabled() {
                counters::$field.fetch_add(value, Relaxed);
            }
        })+

        pub fn sample() -> FrameCounters {
            FrameCounters { $($field: counters::$field.swap(0, Relaxed)),+ }
        }
    };
}

frame_counters!(
    preparation_ns,
    fence_wait_ns,
    graphics_gpu_ns,
    upload_bytes,
    readback_bytes,
    pipeline_creation_ns,
    pipeline_errors,
    binding_errors,
    predicated_commands,
    present_errors,
);

pub fn enable() {
    ENABLED.store(true, Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Relaxed)
}

pub fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}
