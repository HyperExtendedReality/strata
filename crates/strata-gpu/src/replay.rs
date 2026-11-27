//! Deterministic GPU frame Capture & Replay with binary wire serialization
//! and pixel-level divergence diffing.

const CAPTURE_MAGIC: &[u8; 4] = b"SCAP";

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

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(CAPTURE_MAGIC);
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
            }
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 || &bytes[0..4] != CAPTURE_MAGIC {
            return None;
        }
        let frame_index = u64::from_le_bytes(bytes[4..12].try_into().ok()?);
        let count = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
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
                _ => return None,
            };
            ops.push(op);
        }
        Some(Self { frame_index, ops })
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_capture_encodes_and_diffs_pixels() {
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
        assert_eq!(FrameCapture::decode(&wire), Some(cap));

        let diff = diff_rgba8(
            &[10, 20, 30, 255, 0, 0, 0, 255],
            &[10, 25, 30, 255, 0, 0, 0, 255],
        );
        assert_eq!(diff.differing_pixels, 1);
        assert_eq!(diff.max_channel_delta, 5);
    }
}
