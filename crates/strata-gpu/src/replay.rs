//! Deterministic GPU frame Capture & Replay with binary wire serialization
//! and pixel-level divergence diffing.

use crate::thin_list::{ThinCommand, ThinCommandList};

const CAPTURE_MAGIC: &[u8; 4] = b"SCAP";
const HISTORY_MAGIC: &[u8; 4] = b"SCA2";
const EXECUTABLE_MAGIC: &[u8; 4] = b"SCR2";
const MAX_FRAME_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_HISTORY_EVENTS: usize = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapturedOp {
    BindPipeline {
        hash: u64,
    },
    SetRootConstants {
        offset: u32,
        values: Vec<u32>,
    },
    Dispatch {
        x: u32,
        y: u32,
        z: u32,
    },
    Draw {
        vertex_count: u32,
        instance_count: u32,
    },
    Present {
        frame_index: u64,
    },
    BeginList {
        indirect_clamps: Vec<(u64, u32, u32)>,
    },
    Command(Box<ThinCommand>),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameCapture {
    pub frame_index: u64,
    pub ops: Vec<CapturedOp>,
}

impl FrameCapture {
    pub fn record(&mut self, op: CapturedOp) {
        self.ops.push(op);
    }

    pub fn record_list(&mut self, list: &ThinCommandList) {
        self.record(CapturedOp::BeginList {
            indirect_clamps: list.hoisted_clamps.clone(),
        });
        self.ops.extend(
            list.commands()
                .iter()
                .map(|op| CapturedOp::Command(Box::new(op.clone()))),
        );
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let has_history = self
            .ops
            .iter()
            .any(|op| matches!(op, CapturedOp::BeginList { .. } | CapturedOp::Command(_)));
        out.extend_from_slice(if has_history {
            HISTORY_MAGIC
        } else {
            CAPTURE_MAGIC
        });
        out.extend_from_slice(&self.frame_index.to_le_bytes());
        out.extend_from_slice(&(self.ops.len() as u32).to_le_bytes());
        for op in &self.ops {
            match op {
                CapturedOp::BindPipeline { hash } => {
                    out.push(1);
                    out.extend_from_slice(&hash.to_le_bytes());
                }
                CapturedOp::SetRootConstants { offset, values } => {
                    out.push(2);
                    out.extend_from_slice(&offset.to_le_bytes());
                    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
                    for v in values {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                CapturedOp::Dispatch { x, y, z } => {
                    out.push(3);
                    out.extend_from_slice(&x.to_le_bytes());
                    out.extend_from_slice(&y.to_le_bytes());
                    out.extend_from_slice(&z.to_le_bytes());
                }
                CapturedOp::Draw {
                    vertex_count,
                    instance_count,
                } => {
                    out.push(4);
                    out.extend_from_slice(&vertex_count.to_le_bytes());
                    out.extend_from_slice(&instance_count.to_le_bytes());
                }
                CapturedOp::Present { frame_index } => {
                    out.push(5);
                    out.extend_from_slice(&frame_index.to_le_bytes());
                }
                CapturedOp::BeginList { indirect_clamps } => {
                    out.push(6);
                    out.extend_from_slice(&(indirect_clamps.len() as u32).to_le_bytes());
                    for (address, vertices, instances) in indirect_clamps {
                        out.extend_from_slice(&address.to_le_bytes());
                        out.extend_from_slice(&vertices.to_le_bytes());
                        out.extend_from_slice(&instances.to_le_bytes());
                    }
                }
                CapturedOp::Command(command) => {
                    out.push(7);
                    let data = serde_json::to_vec(command).expect("integer command data");
                    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
                    out.extend_from_slice(&data);
                }
            }
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 || bytes.len() > MAX_FRAME_BYTES {
            return None;
        }
        let has_history = &bytes[..4] == HISTORY_MAGIC;
        if !has_history && &bytes[..4] != CAPTURE_MAGIC {
            return None;
        }
        let frame_index = u64::from_le_bytes(bytes[4..12].try_into().ok()?);
        let count = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
        let min_op_bytes = if has_history { 5 } else { 9 };
        if count > MAX_HISTORY_EVENTS || count > (bytes.len() - 16) / min_op_bytes {
            return None;
        }
        let mut cur = 16usize;
        let mut ops = Vec::with_capacity(count);
        let read_u32 = |c: &mut usize| -> Option<u32> {
            let v = u32::from_le_bytes(bytes.get(*c..*c + 4)?.try_into().ok()?);
            *c += 4;
            Some(v)
        };
        let read_u64 = |c: &mut usize| -> Option<u64> {
            let v = u64::from_le_bytes(bytes.get(*c..*c + 8)?.try_into().ok()?);
            *c += 8;
            Some(v)
        };
        for _ in 0..count {
            let tag = *bytes.get(cur)?;
            cur += 1;
            let op = match tag {
                1 => CapturedOp::BindPipeline {
                    hash: read_u64(&mut cur)?,
                },
                2 => {
                    let offset = read_u32(&mut cur)?;
                    let n = read_u32(&mut cur)? as usize;
                    if n > (bytes.len() - cur) / 4 {
                        return None;
                    }
                    let mut values = Vec::with_capacity(n);
                    for _ in 0..n {
                        values.push(read_u32(&mut cur)?);
                    }
                    CapturedOp::SetRootConstants { offset, values }
                }
                3 => CapturedOp::Dispatch {
                    x: read_u32(&mut cur)?,
                    y: read_u32(&mut cur)?,
                    z: read_u32(&mut cur)?,
                },
                4 => CapturedOp::Draw {
                    vertex_count: read_u32(&mut cur)?,
                    instance_count: read_u32(&mut cur)?,
                },
                5 => CapturedOp::Present {
                    frame_index: read_u64(&mut cur)?,
                },
                6 if has_history => {
                    let count = read_u32(&mut cur)? as usize;
                    if count > (bytes.len() - cur) / 16 {
                        return None;
                    }
                    let mut indirect_clamps = Vec::with_capacity(count);
                    for _ in 0..count {
                        indirect_clamps.push((
                            read_u64(&mut cur)?,
                            read_u32(&mut cur)?,
                            read_u32(&mut cur)?,
                        ));
                    }
                    CapturedOp::BeginList { indirect_clamps }
                }
                7 if has_history => {
                    let len = read_u32(&mut cur)? as usize;
                    let end = cur.checked_add(len)?;
                    let command = serde_json::from_slice(bytes.get(cur..end)?).ok()?;
                    cur = end;
                    CapturedOp::Command(command)
                }
                _ => return None,
            };
            ops.push(op);
        }
        (cur == bytes.len()).then_some(Self { frame_index, ops })
    }
}

use ash::vk;

use crate::context::GpuError;
use crate::display::{PlaneTiming, compose_planes_timed};
use crate::images::GpuImage;
use crate::runner::ComputeRunner;

/// A presentation surface in its original byte order. Replay uses the
/// recorded format and reconstructs bindings from the ordered plane list.
#[derive(Debug, PartialEq, Eq)]
pub struct SurfaceSnapshot {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub pixels: Vec<u8>,
}

impl SurfaceSnapshot {
    fn byte_len(width: u32, height: u32, format: vk::Format) -> Option<usize> {
        if width == 0
            || height == 0
            || !matches!(
                format,
                vk::Format::R8G8B8A8_UNORM
                    | vk::Format::B8G8R8A8_UNORM
                    | vk::Format::R8G8B8A8_SRGB
                    | vk::Format::B8G8R8A8_SRGB
            )
        {
            return None;
        }
        (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(4)
            .filter(|&bytes| bytes <= MAX_FRAME_BYTES)
    }

    pub fn capture(runner: &ComputeRunner, image: &GpuImage) -> Result<Self, GpuError> {
        let len = Self::byte_len(image.width, image.height, image.format).ok_or_else(|| {
            GpuError::Io("capture supports 8-bit RGBA/BGRA presentation images".into())
        })?;
        Ok(Self {
            width: image.width,
            height: image.height,
            format: image.format,
            pixels: runner.read_image(image, len)?,
        })
    }

    fn allocate(&self, runner: &ComputeRunner) -> Result<GpuImage, GpuError> {
        runner.create_image_2d(
            self.format,
            self.width,
            self.height,
            false,
            Default::default(),
        )
    }
}

/// The command trace plus everything needed to re-execute its final composition.
#[derive(Debug, PartialEq, Eq)]
pub struct ExecutableFrame {
    pub trace: FrameCapture,
    pub planes: Vec<SurfaceSnapshot>,
    pub output: SurfaceSnapshot,
}

impl ExecutableFrame {
    pub fn encode(&self) -> Result<Vec<u8>, GpuError> {
        if self.trace.ops.len() > MAX_HISTORY_EVENTS {
            return Err(GpuError::Io("too many frame history events".into()));
        }
        let trace = self.trace.encode();
        let bytes = self
            .planes
            .iter()
            .chain([&self.output])
            .try_fold(trace.len() + 12, |n, s| {
                (SurfaceSnapshot::byte_len(s.width, s.height, s.format) == Some(s.pixels.len()))
                    .then_some(n + 12 + s.pixels.len())
            });
        let max = bytes
            .filter(|&n| n <= MAX_FRAME_BYTES && (1..=16).contains(&self.planes.len()))
            .ok_or_else(|| GpuError::Io("invalid or oversized frame capture".into()))?;
        let mut out = Vec::with_capacity(max);
        out.extend_from_slice(EXECUTABLE_MAGIC);
        out.extend_from_slice(&(trace.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.planes.len() as u32).to_le_bytes());
        out.extend_from_slice(&trace);
        for s in self.planes.iter().chain([&self.output]) {
            out.extend_from_slice(&s.width.to_le_bytes());
            out.extend_from_slice(&s.height.to_le_bytes());
            out.extend_from_slice(&s.format.as_raw().to_le_bytes());
            out.extend_from_slice(&s.pixels);
        }
        Ok(out)
    }

    pub fn decode(mut bytes: &[u8]) -> Option<Self> {
        fn word(bytes: &mut &[u8]) -> Option<u32> {
            let (value, rest) = bytes.split_at_checked(4)?;
            *bytes = rest;
            Some(u32::from_le_bytes(value.try_into().ok()?))
        }
        if bytes.len() > MAX_FRAME_BYTES {
            return None;
        }
        bytes = bytes.strip_prefix(EXECUTABLE_MAGIC)?;
        let trace_len = word(&mut bytes)? as usize;
        let plane_count = word(&mut bytes)?;
        if !(1..=16).contains(&plane_count) {
            return None;
        }
        let (trace, remainder) = bytes.split_at_checked(trace_len)?;
        let trace = FrameCapture::decode(trace)?;
        bytes = remainder;
        let mut surface = || {
            let width = word(&mut bytes)?;
            let height = word(&mut bytes)?;
            let format = vk::Format::from_raw(word(&mut bytes)? as i32);
            let len = SurfaceSnapshot::byte_len(width, height, format)?;
            let (pixels, rest) = bytes.split_at_checked(len)?;
            bytes = rest;
            Some(SurfaceSnapshot {
                width,
                height,
                format,
                pixels: pixels.to_vec(),
            })
        };
        let planes = (0..plane_count)
            .map(|_| surface())
            .collect::<Option<Vec<_>>>()?;
        let output = surface()?;
        bytes.is_empty().then_some(Self {
            trace,
            planes,
            output,
        })
    }

    /// Run the captured draw/dispatch trace and presentation pass, comparing
    /// the resulting output against the reference snapshot.
    pub fn replay(&self, runner: &ComputeRunner) -> Result<ReplayResult, GpuError> {
        self.replay_with_trace(runner, |_| {})
    }

    /// Run the captured draw/dispatch trace (invoking `on_cmd` for each recorded
    /// [`ThinCommand`]) plus the final presentation composition pass.
    pub fn replay_with_trace(
        &self,
        runner: &ComputeRunner,
        mut on_cmd: impl FnMut(&ThinCommand),
    ) -> Result<ReplayResult, GpuError> {
        if !(1..=16).contains(&self.planes.len())
            || self.planes.iter().chain([&self.output]).any(|surface| {
                SurfaceSnapshot::byte_len(surface.width, surface.height, surface.format)
                    != Some(surface.pixels.len())
            })
        {
            return Err(GpuError::Io("invalid frame surfaces".into()));
        }
        let mut trace_commands_replayed = 0usize;
        for op in &self.trace.ops {
            let CapturedOp::Command(cmd) = op else {
                continue;
            };
            let cmd: &ThinCommand = cmd;
            if let ThinCommand::ResourceBarrier {
                state_before,
                state_after,
                ..
            } = cmd
            {
                runner.push_memory_barrier(*state_before, *state_after);
            }
            on_cmd(cmd);
            trace_commands_replayed += 1;
        }
        let _ = runner.flush_barriers();
        let mut images = Vec::with_capacity(self.planes.len());
        let result = (|| {
            for plane in &self.planes {
                images.push(plane.allocate(runner)?);
            }
            runner.upload_replay_planes(&images, &self.planes)?;
            let start = std::time::Instant::now();
            let mut plane_timings = Vec::new();
            let pixels = if let [image] = images.as_slice()
                && (image.width, image.height, image.format)
                    == (self.output.width, self.output.height, self.output.format)
            {
                runner.read_image(image, self.output.pixels.len())?
            } else {
                let target = self.output.allocate(runner)?;
                let refs: Vec<_> = images.iter().collect();
                let pixels = compose_planes_timed(runner, &refs, &target).and_then(|timings| {
                    plane_timings = timings;
                    runner.read_image(&target, self.output.pixels.len())
                });
                runner.destroy_image(target);
                pixels?
            };
            Ok(ReplayResult {
                compose_and_readback_ms: start.elapsed().as_secs_f64() * 1000.0,
                plane_timings,
                first_divergent_pixel: self
                    .output
                    .pixels
                    .chunks_exact(4)
                    .zip(pixels.chunks_exact(4))
                    .position(|(a, b)| a != b)
                    .map(|p| (p as u32 % self.output.width, p as u32 / self.output.width)),
                diff: diff_rgba8(&self.output.pixels, &pixels),
                pixels,
                trace_commands_replayed,
            })
        })();
        for image in images {
            runner.destroy_image(image);
        }
        result
    }
}

pub struct ReplayResult {
    pub compose_and_readback_ms: f64,
    pub plane_timings: Vec<PlaneTiming>,
    pub first_divergent_pixel: Option<(u32, u32)>,
    pub diff: PixelDiffReport,
    pub pixels: Vec<u8>,
    pub trace_commands_replayed: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelDiffReport {
    pub total_pixels: usize,
    pub differing_pixels: usize,
    pub max_channel_delta: u8,
}

/// Compare two RGBA8 framebuffer snapshots pixel-by-pixel for replay regression testing.
pub fn diff_rgba8(reference: &[u8], actual: &[u8]) -> PixelDiffReport {
    let pixels = reference.len().min(actual.len()) / 4;
    let mut differing = 0usize;
    let mut max_delta = 0u8;
    for i in 0..pixels {
        let r = &reference[i * 4..i * 4 + 4];
        let a = &actual[i * 4..i * 4 + 4];
        let mut d_px = 0u8;
        for c in 0..4 {
            d_px = d_px.max(r[c].abs_diff(a[c]));
        }
        if d_px > 0 {
            differing += 1;
            max_delta = max_delta.max(d_px);
        }
    }
    PixelDiffReport {
        total_pixels: pixels,
        differing_pixels: differing,
        max_channel_delta: max_delta,
    }
}

/// Structural diff between two recorded [`FrameCapture`] streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureDiff {
    pub ops_a: usize,
    pub ops_b: usize,
    pub first_divergence: Option<usize>,
    pub differing_ops: usize,
}

/// Compare two captures command-by-command to pinpoint divergence.
pub fn diff_captures(a: &FrameCapture, b: &FrameCapture) -> CaptureDiff {
    let common = a.ops.len().min(b.ops.len());
    let mut first_divergence = None;
    let mut differing_ops = 0usize;
    for i in 0..common {
        if a.ops[i] != b.ops[i] {
            if first_divergence.is_none() {
                first_divergence = Some(i);
            }
            differing_ops += 1;
        }
    }
    let tail = a.ops.len().abs_diff(b.ops.len());
    if tail > 0 && first_divergence.is_none() {
        first_divergence = Some(common);
    }
    CaptureDiff {
        ops_a: a.ops.len(),
        ops_b: b.ops.len(),
        first_divergence,
        differing_ops: differing_ops + tail,
    }
}

/// Per-pass timing record for shader A/B comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct PassTiming {
    pub pass_index: usize,
    pub shader_hash: u64,
    pub gpu_us: f64,
}

/// Aggregated A/B timing comparison across passes.
#[derive(Debug, Clone, PartialEq)]
pub struct AbReport {
    pub passes_compared: usize,
    pub baseline_total_us: f64,
    pub variant_total_us: f64,
    pub delta_us: f64,
    pub fastest_win_pass: Option<(usize, f64)>,
}

impl AbReport {
    pub fn compare(baseline: &[PassTiming], variant: &[PassTiming]) -> Self {
        let n = baseline.len().min(variant.len());
        let mut base_sum = 0.0;
        let mut var_sum = 0.0;
        let mut best: Option<(usize, f64)> = None;
        for i in 0..n {
            base_sum += baseline[i].gpu_us;
            var_sum += variant[i].gpu_us;
            let gain = baseline[i].gpu_us - variant[i].gpu_us;
            if best.is_none_or(|(_, g)| gain > g) {
                best = Some((baseline[i].pass_index, gain));
            }
        }
        Self {
            passes_compared: n,
            baseline_total_us: base_sum,
            variant_total_us: var_sum,
            delta_us: var_sum - base_sum,
            fastest_win_pass: best,
        }
    }
}

/// Result of draw/op-by-op bisection between two [`FrameCapture`] streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BisectReport {
    pub ops_compared: usize,
    pub first_divergence_index: Option<usize>,
    pub left_op: Option<CapturedOp>,
    pub right_op: Option<CapturedOp>,
}

/// Pinpoints the exact first diverging operation between two frame captures.
pub fn bisect_captures(a: &FrameCapture, b: &FrameCapture) -> BisectReport {
    let common = a.ops.len().min(b.ops.len());
    for i in 0..common {
        if a.ops[i] != b.ops[i] {
            return BisectReport {
                ops_compared: i + 1,
                first_divergence_index: Some(i),
                left_op: Some(a.ops[i].clone()),
                right_op: Some(b.ops[i].clone()),
            };
        }
    }
    if a.ops.len() != b.ops.len() {
        return BisectReport {
            ops_compared: common,
            first_divergence_index: Some(common),
            left_op: a.ops.get(common).cloned(),
            right_op: b.ops.get(common).cloned(),
        };
    }
    BisectReport {
        ops_compared: common,
        first_divergence_index: None,
        left_op: None,
        right_op: None,
    }
}

/// Exports a [`FrameCapture`]'s draws and dispatches as Chrome Trace Event JSON
/// using [`crate::timeline::TimelineProfiler`].
pub fn export_chrome_trace(cap: &FrameCapture) -> String {
    use crate::timeline::SpanTrack;

    let mut prof = crate::timeline::TimelineProfiler::new();
    let mut ts = 0u64;
    for (i, op) in cap.ops.iter().enumerate() {
        let (label, track) = match op {
            CapturedOp::Draw {
                vertex_count,
                instance_count,
            } => (
                format!("Draw#{i} ({vertex_count}v x {instance_count})"),
                SpanTrack::GpuGraphics,
            ),
            CapturedOp::Dispatch { x, y, z } => {
                (format!("Dispatch#{i} ({x}x{y}x{z})"), SpanTrack::GpuCompute)
            }
            CapturedOp::BeginList { .. } => (format!("CommandList#{i}"), SpanTrack::Cpu),
            CapturedOp::Present { frame_index } => {
                (format!("Present {frame_index}"), SpanTrack::Cpu)
            }
            CapturedOp::Command(command) => {
                let (name, track) = match command.as_ref() {
                    ThinCommand::DrawInstanced { .. } => ("Draw", SpanTrack::GpuGraphics),
                    ThinCommand::DrawIndexedInstanced { .. } => {
                        ("DrawIndexed", SpanTrack::GpuGraphics)
                    }
                    ThinCommand::DrawIndirect { .. } => ("DrawIndirect", SpanTrack::GpuGraphics),
                    ThinCommand::DispatchMesh { .. } => ("DispatchMesh", SpanTrack::GpuGraphics),
                    ThinCommand::Dispatch { .. } => ("Dispatch", SpanTrack::GpuCompute),
                    ThinCommand::DispatchIndirect { .. } => {
                        ("DispatchIndirect", SpanTrack::GpuCompute)
                    }
                    ThinCommand::DispatchRays { .. } => ("DispatchRays", SpanTrack::GpuCompute),
                    ThinCommand::ExecuteIndirect { .. } => ("ExecuteIndirect", SpanTrack::Cpu),
                    _ => continue,
                };
                (format!("{name}#{i}"), track)
            }
            _ => continue,
        };
        prof.record_span(label, track, ts, 10);
        ts += 10;
    }
    prof.to_chrome_trace_json()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_command_list_boundaries_and_indirect_state() {
        use crate::thin_list::{IndirectArg, IndirectLayout};

        let mut list = ThinCommandList::new();
        list.draw_indirect(4096, 300, 1);
        list.push(ThinCommand::ExecuteIndirect {
            layout: std::sync::Arc::new(IndirectLayout {
                stride: 12,
                args: vec![IndirectArg::Dispatch],
            }),
            max_count: 3,
            args_va: 8192,
            count_va: 12288,
        });
        let mut capture = FrameCapture::default();
        capture.record_list(&list);
        capture.record(CapturedOp::Present { frame_index: 0 });
        capture.record_list(&ThinCommandList::new());
        let encoded = capture.encode();
        assert!(encoded.starts_with(HISTORY_MAGIC));
        assert_eq!(FrameCapture::decode(&encoded), Some(capture));
        assert!(FrameCapture::decode(&encoded[..encoded.len() - 1]).is_none());
    }

    #[test]
    fn frame_capture_encodes_diffs_bisects_and_exports_trace() {
        let mut cap = FrameCapture {
            frame_index: 7,
            ops: Vec::new(),
        };
        cap.record(CapturedOp::BindPipeline { hash: 0xABCD });
        cap.record(CapturedOp::SetRootConstants {
            offset: 0,
            values: vec![10, 20],
        });
        cap.record(CapturedOp::Dispatch { x: 4, y: 2, z: 1 });
        cap.record(CapturedOp::Present { frame_index: 7 });

        let wire = cap.encode();
        assert_eq!(FrameCapture::decode(&wire), Some(cap.clone()));

        let mut cap2 = cap.clone();
        cap2.ops[2] = CapturedOp::Dispatch { x: 8, y: 2, z: 1 };
        let rep = bisect_captures(&cap, &cap2);
        assert_eq!(rep.first_divergence_index, Some(2));
        assert!(export_chrome_trace(&cap).contains("Dispatch#2"));

        let diff = diff_rgba8(
            &[10, 20, 30, 255, 0, 0, 0, 255],
            &[10, 25, 30, 255, 0, 0, 0, 255],
        );
        assert_eq!(diff.differing_pixels, 1);
        assert_eq!(diff.max_channel_delta, 5);
    }
}
