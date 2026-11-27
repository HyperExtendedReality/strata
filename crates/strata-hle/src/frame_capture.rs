//! Optional one-frame capture from the title's queues and presenter.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use strata_gpu::images::GpuImage;
use strata_gpu::replay::{CapturedOp, ExecutableFrame, FrameCapture, SurfaceSnapshot};
use strata_gpu::runner::ComputeRunner;
use strata_gpu::thin_list::{ThinCommand, ThinCommandList};

struct Pending {
    path: PathBuf,
    target_frame: u64,
    trace: FrameCapture,
}

static CAPTURE: Mutex<Option<Pending>> = Mutex::new(None);
static RESULT: Mutex<Option<Result<PathBuf, String>>> = Mutex::new(None);
static ARMED: AtomicBool = AtomicBool::new(false);

pub fn arm(path: PathBuf, target_frame: u64) {
    *RESULT.lock().unwrap() = None;
    *CAPTURE.lock().unwrap() = Some(Pending {
        path,
        target_frame,
        trace: FrameCapture::default(),
    });
    ARMED.store(true, Ordering::Release);
}

pub fn take_result() -> Option<Result<PathBuf, String>> {
    RESULT.lock().unwrap().take()
}

pub(crate) fn record(list: &ThinCommandList) {
    if !ARMED.load(Ordering::Acquire) {
        return;
    }
    let mut pending = CAPTURE.lock().unwrap();
    let Some(pending) = pending.as_mut() else {
        return;
    };
    for command in list.commands() {
        let op = match command {
            ThinCommand::BindPipeline(key) => CapturedOp::BindPipeline { hash: key.0 },
            ThinCommand::PushUserSgprs { offset_dw, values } => CapturedOp::SetRootConstants {
                offset: u32::from(*offset_dw),
                values: values.clone(),
            },
            ThinCommand::Dispatch { x, y, z, .. } => CapturedOp::Dispatch {
                x: *x,
                y: *y,
                z: *z,
            },
            ThinCommand::DrawInstanced {
                vertex_count,
                instance_count,
                ..
            } => CapturedOp::Draw {
                vertex_count: *vertex_count,
                instance_count: *instance_count,
            },
            _ => continue,
        };
        pending.trace.record(op);
    }
}

pub(crate) fn finish(runner: &ComputeRunner, planes: &[&GpuImage], frame: &GpuImage, index: u64) {
    if !ARMED.load(Ordering::Acquire) {
        return;
    }
    let mut slot = CAPTURE.lock().unwrap();
    let Some(pending) = slot.as_mut() else {
        return;
    };
    if index < pending.target_frame {
        pending.trace.ops.clear();
        return;
    }
    let Pending {
        path, mut trace, ..
    } = slot.take().unwrap();
    ARMED.store(false, Ordering::Release);
    drop(slot);
    trace.frame_index = index;
    trace.record(CapturedOp::Present { frame_index: index });
    let result = (|| {
        let planes = planes
            .iter()
            .map(|image| SurfaceSnapshot::capture(runner, image))
            .collect::<Result<Vec<_>, _>>()?;
        let output = SurfaceSnapshot::capture(runner, frame)?;
        ExecutableFrame {
            trace,
            planes,
            output,
        }
        .encode()
    })();
    match result.map_err(|e| e.to_string()).and_then(|bytes| {
        strata_gpu::present::write_atomic(&path, &bytes).map_err(|e| e.to_string())
    }) {
        Ok(()) => {
            eprintln!("[CAPTURE] Frame {index} saved to {}", path.display());
            *RESULT.lock().unwrap() = Some(Ok(path));
        }
        Err(e) => {
            crate::crash::log_line(&format!("frame capture: {e}"));
            *RESULT.lock().unwrap() = Some(Err(e));
        }
    }
}
