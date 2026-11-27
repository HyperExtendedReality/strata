//! CPU & GPU timeline profiler with Chrome Trace JSON export and pass summary.

use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanTrack {
    Cpu,
    GpuCompute,
    GpuGraphics,
    UploadCopy,
}

#[derive(Debug, Clone)]
pub struct TimelineSpan {
    pub name: String,
    pub track: SpanTrack,
    pub start_us: u64,
    pub duration_us: u64,
}

#[derive(Debug)]
pub struct TimelineProfiler {
    epoch: Instant,
    spans: Vec<TimelineSpan>,
}

impl Default for TimelineProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl TimelineProfiler {
    pub fn new() -> Self {
        Self {
            epoch: Instant::now(),
            spans: Vec::new(),
        }
    }

    pub fn now_us(&self) -> u64 {
        self.epoch.elapsed().as_micros() as u64
    }

    pub fn record_span(
        &mut self,
        name: impl Into<String>,
        track: SpanTrack,
        start_us: u64,
        duration_us: u64,
    ) {
        self.spans.push(TimelineSpan {
            name: name.into(),
            track,
            start_us,
            duration_us,
        });
    }

    pub fn spans(&self) -> &[TimelineSpan] {
        &self.spans
    }

    pub fn clear(&mut self) {
        self.spans.clear();
    }

    /// Export recorded spans in Chrome Trace Event JSON format (`chrome://tracing` / Perfetto).
    pub fn to_chrome_trace_json(&self) -> String {
        let mut out = String::from("{\"traceEvents\":[");
        for (i, s) in self.spans.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let tid = match s.track {
                SpanTrack::Cpu => 1,
                SpanTrack::GpuGraphics => 2,
                SpanTrack::GpuCompute => 3,
                SpanTrack::UploadCopy => 4,
            };
            out.push_str(&format!(
                "{{\"name\":\"{}\",\"ph\":\"X\",\"ts\":{},\"dur\":{},\"pid\":1,\"tid\":{}}}",
                s.name.replace('"', "\\\""),
                s.start_us,
                s.duration_us,
                tid
            ));
        }
        out.push_str("]}");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeline_exports_chrome_trace_json() {
        let mut prof = TimelineProfiler::new();
        prof.record_span("GBuffer", SpanTrack::GpuGraphics, 100, 1250);
        prof.record_span("Upload", SpanTrack::UploadCopy, 50, 200);
        let json = prof.to_chrome_trace_json();
        assert!(json.contains("\"name\":\"GBuffer\""));
        assert!(json.contains("\"dur\":1250"));
    }
}
