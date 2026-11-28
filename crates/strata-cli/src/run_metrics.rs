use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use strata_gpu::perf::{self, FrameCounters};

pub struct RunMetrics {
    start: Instant,
    last_present: Option<f64>,
    visible: Vec<f64>,
    shown: u64,
    csv: Option<File>,
    pub csv_error: Option<io::Error>,
}

impl RunMetrics {
    pub fn new(csv: Option<&Path>) -> io::Result<Self> {
        let csv = csv.map(|path| {
            let mut file = File::create_new(path)?;
            writeln!(file, "present,elapsed_ms,interval_ms,shown,preparation_ns,fence_wait_ns,graphics_gpu_ns,upload_bytes,readback_bytes,pipeline_creation_ns,pipeline_errors,binding_errors,predicated_commands,present_errors,texture_heap_usage_bytes,texture_heap_budget_bytes,commit_available_bytes,texture_resident_bytes,oldest_upload_ms")?;
            Ok::<_, io::Error>(file)
        }).transpose()?;
        Ok(Self { start: Instant::now(), last_present: None, visible: Vec::new(), shown: 0, csv, csv_error: None })
    }

    pub fn present(&mut self, frame: u64, shown: u64) {
        let elapsed = self.start.elapsed().as_secs_f64() * 1_000.0;
        let interval = self.last_present.replace(elapsed).map(|last| elapsed - last);
        let visible = shown > self.shown;
        if visible {
            self.visible.push(elapsed);
            self.shown = shown;
        }
        let counters = perf::sample();
        let pressure = strata_hle::pipeline_cache::streaming_stats();
        if let Some(csv) = &mut self.csv {
            let FrameCounters { preparation_ns, fence_wait_ns, graphics_gpu_ns, upload_bytes, readback_bytes, pipeline_creation_ns, pipeline_errors, binding_errors, predicated_commands, present_errors } = counters;
            let interval = interval.map_or_else(String::new, |ms| format!("{ms:.6}"));
            let row = format!(
                "{frame},{elapsed:.6},{interval},{},{preparation_ns},{fence_wait_ns},{graphics_gpu_ns},{upload_bytes},{readback_bytes},{pipeline_creation_ns},{pipeline_errors},{binding_errors},{predicated_commands},{present_errors},{},{},{},{},{}\n",
                u8::from(visible), pressure.heap_usage, pressure.heap_budget,
                pressure.commit_available, pressure.resident_bytes, pressure.oldest_upload_ms,
            );
            if let Err(error) = csv.write_all(row.as_bytes()) {
                self.csv_error = Some(error);
                self.csv = None;
            }
        }
    }

    pub fn first_visible_ms(&self) -> Option<f64> {
        self.visible.first().copied()
    }

    pub fn visible_intervals(&self) -> Vec<f64> {
        self.visible.windows(2).map(|pair| pair[1] - pair[0]).collect()
    }
}
