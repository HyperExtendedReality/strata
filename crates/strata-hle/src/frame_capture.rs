//! Optional one-frame capture from the title's queues and presenter.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use strata_gpu::images::GpuImage;
use strata_gpu::replay::{
    CapturedOp, ExecutableFrame, FrameCapture, MAX_HISTORY_EVENTS, SurfaceSnapshot,
};
use strata_gpu::runner::ComputeRunner;
use strata_gpu::thin_list::ThinCommandList;

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
    let mut slot = CAPTURE.lock().unwrap();
    let Some(pending) = slot.as_mut() else {
        return;
    };
    if pending
        .trace
        .ops
        .len()
        .saturating_add(list.commands().len())
        >= MAX_HISTORY_EVENTS - 1
    {
        stop_at_history_limit(&mut slot);
        return;
    }
    pending.trace.record_list(list);
}

fn stop_at_history_limit(slot: &mut Option<Pending>) {
    *slot = None;
    ARMED.store(false, Ordering::Release);
    *RESULT.lock().unwrap() = Some(Err("frame capture exceeded its history event limit".into()));
}

pub(crate) fn finish(runner: &ComputeRunner, planes: &[&GpuImage], frame: &GpuImage, index: u64) {
    if !ARMED.load(Ordering::Acquire) {
        return;
    }
    let mut slot = CAPTURE.lock().unwrap();
    let Some(pending) = slot.as_mut() else {
        return;
    };
    if pending.trace.ops.len() >= MAX_HISTORY_EVENTS {
        stop_at_history_limit(&mut slot);
        return;
    }
    pending
        .trace
        .record(CapturedOp::Present { frame_index: index });
    if index < pending.target_frame {
        return;
    }
    let Pending {
        path, mut trace, ..
    } = slot.take().unwrap();
    ARMED.store(false, Ordering::Release);
    drop(slot);
    trace.frame_index = index;
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
