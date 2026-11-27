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

pub(crate) fn register(backend: &Arc<Backend>) {
    let mut run = RUN.lock().unwrap();
    if !run.enabled {
        return;
    }
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
    run.devices.push(Arc::downgrade(backend));
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
