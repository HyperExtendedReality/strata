//! Real `XAsync` / `XTaskQueue` asynchronous task execution and completion
//! framework used by `XUser`, `XGameSave`, `XPackage`, `XStore`, and `XGameRuntime`.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

pub const E_PENDING: i32 = 0x8000_000Au32 as i32;
pub const S_OK: i32 = 0;

/// Dispatch mode for a task queue port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskQueueDispatchMode {
    ThreadPool,
    SerializedThreadPool,
    Manual,
    Immediate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskQueuePort {
    Work,
    Completion,
}

pub type TaskCallback = Box<dyn FnOnce() + Send + 'static>;

/// A task queue with separate work and completion callback ports.
pub struct TaskQueue {
    pub id: u64,
    pub work_mode: TaskQueueDispatchMode,
    pub completion_mode: TaskQueueDispatchMode,
    work_items: VecDeque<TaskCallback>,
    completion_items: VecDeque<TaskCallback>,
}

impl TaskQueue {
    pub fn new(
        id: u64,
        work_mode: TaskQueueDispatchMode,
        completion_mode: TaskQueueDispatchMode,
    ) -> Self {
        Self {
            id,
            work_mode,
            completion_mode,
            work_items: VecDeque::new(),
            completion_items: VecDeque::new(),
        }
    }

    pub fn submit(&mut self, port: TaskQueuePort, cb: TaskCallback) {
        let mode = match port {
            TaskQueuePort::Work => self.work_mode,
            TaskQueuePort::Completion => self.completion_mode,
        };
        if mode == TaskQueueDispatchMode::Immediate {
            cb();
            return;
        }
        match port {
            TaskQueuePort::Work => self.work_items.push_back(cb),
            TaskQueuePort::Completion => self.completion_items.push_back(cb),
        }
    }

    /// Dispatch a single queued callback from `port`. Returns `true` if one ran.
    pub fn dispatch_one(&mut self, port: TaskQueuePort) -> bool {
        let item = match port {
            TaskQueuePort::Work => self.work_items.pop_front(),
            TaskQueuePort::Completion => self.completion_items.pop_front(),
        };
        if let Some(cb) = item {
            cb();
            true
        } else {
            false
        }
    }

    /// Drain all pending callbacks on `port`.
    pub fn drain(&mut self, port: TaskQueuePort) -> usize {
        let mut count = 0;
        while self.dispatch_one(port) {
            count += 1;
        }
        count
    }
}

/// Completed or in-flight `XAsyncBlock` state.
#[derive(Debug, Clone)]
pub struct AsyncOperation {
    pub id: u64,
    pub status: i32,
    pub result_payload: Vec<u8>,
    pub completed: bool,
}

/// Global `XAsync` & `XTaskQueue` registry.
#[derive(Default)]
pub struct XAsyncRuntime {
    next_id: AtomicU64,
    queues: HashMap<u64, TaskQueue>,
    ops: HashMap<u64, AsyncOperation>,
}

impl XAsyncRuntime {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_queue(
        &mut self,
        work: TaskQueueDispatchMode,
        completion: TaskQueueDispatchMode,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        self.queues.insert(id, TaskQueue::new(id, work, completion));
        id
    }

    pub fn queue_mut(&mut self, id: u64) -> Option<&mut TaskQueue> {
        self.queues.get_mut(&id)
    }

    pub fn close_queue(&mut self, id: u64) {
        self.queues.remove(&id);
    }

    pub fn begin_op(&mut self) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        self.ops.insert(
            id,
            AsyncOperation {
                id,
                status: E_PENDING,
                result_payload: Vec::new(),
                completed: false,
            },
        );
        id
    }

    pub fn complete_op(&mut self, id: u64, status: i32, payload: Vec<u8>) {
        if let Some(op) = self.ops.get_mut(&id) {
            op.status = status;
            op.result_payload = payload;
            op.completed = true;
        }
    }

    pub fn get_status(&self, id: u64) -> i32 {
        self.ops
            .get(&id)
            .map_or(0x8007_0006u32 as i32, |op| op.status)
    }

    pub fn get_result_size(&self, id: u64) -> Option<usize> {
        let op = self.ops.get(&id)?;
        op.completed.then_some(op.result_payload.len())
    }

    pub fn take_result(&mut self, id: u64) -> Option<Vec<u8>> {
        let op = self.ops.get(&id)?;
        if !op.completed {
            return None;
        }
        self.ops.remove(&id).map(|o| o.result_payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn task_queue_dispatches_work_and_async_op_completes() {
        let mut rt = XAsyncRuntime::new();
        let q = rt.create_queue(TaskQueueDispatchMode::Manual, TaskQueueDispatchMode::Manual);
        let ran = Arc::new(AtomicBool::new(false));
        let flag = ran.clone();
        rt.queue_mut(q).unwrap().submit(
            TaskQueuePort::Work,
            Box::new(move || {
                flag.store(true, Ordering::SeqCst);
            }),
        );
        assert!(!ran.load(Ordering::SeqCst));
        assert_eq!(rt.queue_mut(q).unwrap().drain(TaskQueuePort::Work), 1);
        assert!(ran.load(Ordering::SeqCst));

        let op = rt.begin_op();
        assert_eq!(rt.get_status(op), E_PENDING);
        rt.complete_op(op, S_OK, vec![1, 2, 3, 4]);
        assert_eq!(rt.get_status(op), S_OK);
        assert_eq!(rt.get_result_size(op), Some(4));
        assert_eq!(rt.take_result(op), Some(vec![1, 2, 3, 4]));
    }
}
