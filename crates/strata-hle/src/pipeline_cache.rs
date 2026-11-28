//! Pipeline persistence for devices created during a CLI title run.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};

use crate::d3d12_abi::Backend;

#[derive(Default)]
struct RunCache {
    enabled: bool,
    root: Option<PathBuf>,
    seed: Option<PathBuf>,
    prewarm: bool,
    devices: Vec<Weak<Backend>>,
    finished: Vec<DeviceStats>,
}

static RUN: Mutex<RunCache> = Mutex::new(RunCache {
    enabled: false,
    root: None,
    seed: None,
    prewarm: false,
    devices: Vec::new(),
    finished: Vec::new(),
});

pub fn configure(root: Option<PathBuf>, prewarm: bool, seed: Option<PathBuf>) {
    let mut run = RUN.lock().unwrap();
    run.enabled = true;
    run.root = root;
    run.seed = seed;
    run.prewarm = prewarm;
}

pub(crate) fn background_warmup_enabled() -> bool {
    RUN.lock().unwrap().prewarm
}

pub(crate) fn register(backend: &Arc<Backend>) {
    let mut run = RUN.lock().unwrap();
    if let Some(root) = &run.root {
        let stats =
            backend
                .runner
                .enable_pipeline_cache_with_seed(root, run.prewarm, run.seed.as_deref());
        eprintln!(
            "[PIPELINES] {} driver-cache hit={}, prewarmed={}, warmup={:.2} ms",
            backend.runner.device_name, stats.driver_cache_loaded, stats.prewarmed, stats.warmup_ms
        );
    }
    run.devices.retain(|device| device.strong_count() != 0);
    run.devices.push(Arc::downgrade(backend));
}

/// Per-frame GPU housekeeping for every registered device at `PresentX`:
/// the BMA mapping pass ([`strata_gpu::bma`]; a no-op unless enabled).
pub fn end_frame() {
    let devices: Vec<_> = RUN
        .lock()
        .unwrap()
        .devices
        .iter()
        .filter_map(Weak::upgrade)
        .collect();
    for backend in devices {
        if let Ok(mut textures) = backend.textures.try_lock() {
            textures.maintain_streaming(&backend.runner);
        }
        if let Err(error) = backend.runner.bma_end_frame() {
            crate::crash::log_line(&format!("bma mapping: {error}"));
        }
    }
}

pub fn streaming_stats() -> strata_gpu::texture::StreamingStats {
    let devices: Vec<_> = RUN
        .lock()
        .unwrap()
        .devices
        .iter()
        .filter_map(Weak::upgrade)
        .collect();
    let mut total = strata_gpu::texture::StreamingStats::default();
    for backend in devices {
        if let Ok(textures) = backend.textures.try_lock() {
            let sample = textures.streaming_stats();
            total.heap_usage = total.heap_usage.saturating_add(sample.heap_usage);
            total.heap_budget = total.heap_budget.saturating_add(sample.heap_budget);
            total.commit_available = sample.commit_available;
            total.resident_bytes = total.resident_bytes.saturating_add(sample.resident_bytes);
            total.oldest_upload_ms = total.oldest_upload_ms.max(sample.oldest_upload_ms);
        }
    }
    total
}

pub fn flush() {
    let devices: Vec<_> = RUN
        .lock()
        .unwrap()
        .devices
        .iter()
        .filter_map(Weak::upgrade)
        .collect();
    for backend in devices {
        if let Err(error) = backend.runner.flush_pipeline_cache() {
            crate::crash::log_line(&format!("pipeline cache: {error}"));
        }
    }
}

#[derive(Clone)]
pub struct DeviceStats {
    pub adapter: String,
    pub driver_version: u32,
    pub pipelines: strata_gpu::pipeline_usage::PipelineStats,
    pub transfers: strata_gpu::uma::TransferStats,
}

fn snapshot(backend: &Backend) -> DeviceStats {
    DeviceStats {
        adapter: backend.runner.device_name.clone(),
        driver_version: backend.runner.driver_version,
        pipelines: backend.runner.pipeline_stats(),
        transfers: backend.uma.lock().unwrap().transfer_stats(),
    }
}

pub(crate) fn released(backend: &Backend) {
    let mut run = RUN.lock().unwrap();
    if run.enabled {
        run.finished.push(snapshot(backend));
    }
}

pub fn stats() -> Vec<DeviceStats> {
    let (mut stats, devices) = {
        let run = RUN.lock().unwrap();
        (
            run.finished.clone(),
            run.devices
                .iter()
                .filter_map(Weak::upgrade)
                .collect::<Vec<_>>(),
        )
    };
    stats.extend(devices.iter().map(|backend| snapshot(backend)));
    stats
}
