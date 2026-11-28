//! Multi-threaded thin-command-list translation pool, async submission
//! pipeline, and ahead-of-time texture upload queue.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::thin_list::{ThinCommand, ThinCommandList};

/// Result of translating a batch of [`ThinCommandList`]s on background workers.
#[derive(Debug, Clone)]
pub struct TranslatedBatch {
    pub ticket: u64,
    pub op_count: usize,
    pub draw_count: usize,
    pub dispatch_count: usize,
    pub ops: Vec<ThinCommand>,
    pub hoisted_clamps: Vec<(u64, u32, u32)>,
}

/// Multi-threaded worker pool that flattens and prepares thin command lists
/// off the title's rendering threads.
pub struct ThinTranslationPool {
    workers: usize,
}

impl ThinTranslationPool {
    pub fn new(workers: usize) -> Self {
        Self {
            workers: workers.max(1),
        }
    }

    /// Translate multiple recorded thin command lists in parallel, preserving
    /// submission order in the returned batches.
    pub fn translate_parallel(&self, lists: Vec<ThinCommandList>) -> Vec<TranslatedBatch> {
        let count = lists.len();
        if count <= 1 || self.workers == 1 {
            return lists
                .into_iter()
                .enumerate()
                .map(|(i, l)| translate_one(i as u64, l))
                .collect();
        }
        let chunk_size = count.div_ceil(self.workers);
        let mut results: Vec<Option<TranslatedBatch>> = vec![None; count];
        let indexed: Vec<(u64, ThinCommandList)> = lists
            .into_iter()
            .enumerate()
            .map(|(i, l)| (i as u64, l))
            .collect();

        thread::scope(|s| {
            let mut handles = Vec::new();
            let mut remaining = indexed;
            while !remaining.is_empty() {
                let split = remaining.len().min(chunk_size);
                let chunk: Vec<_> = remaining.drain(..split).collect();
                handles.push(s.spawn(move || {
                    chunk
                        .into_iter()
                        .map(|(id, l)| translate_one(id, l))
                        .collect::<Vec<_>>()
                }));
            }
            for h in handles {
                for b in h
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                {
                    let idx = b.ticket as usize;
                    results[idx] = Some(b);
                }
            }
        });

        results.into_iter().flatten().collect()
    }
}

fn translate_one(ticket: u64, list: ThinCommandList) -> TranslatedBatch {
    let (ops, hoisted_clamps) = list.into_recording();
    let mut draw_count = 0usize;
    let mut dispatch_count = 0usize;
    for op in &ops {
        match op {
            ThinCommand::DrawInstanced { .. }
            | ThinCommand::DrawIndexedInstanced { .. }
            | ThinCommand::DrawIndirect { .. } => {
                draw_count += 1;
            }
            ThinCommand::Dispatch { .. } | ThinCommand::DispatchIndirect { .. } => {
                dispatch_count += 1;
            }
            _ => {}
        }
    }
    TranslatedBatch {
        ticket,
        op_count: ops.len(),
        draw_count,
        dispatch_count,
        ops,
        hoisted_clamps,
    }
}

/// Request queued for ahead-of-time texture upload on a dedicated copy queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureUploadTicket {
    pub descriptor_hash: u64,
    pub src_addr: u64,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub ready_fence: u64,
}

/// Asynchronous copy-queue staging list for ahead-of-time texture uploads.
pub struct TextureAheadQueue {
    inner: Arc<(Mutex<VecDeque<TextureUploadTicket>>, Condvar)>,
}

impl Default for TextureAheadQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl TextureAheadQueue {
    pub fn new() -> Self {
        Self {
            inner: Arc::new((Mutex::new(VecDeque::new()), Condvar::new())),
        }
    }

    pub fn enqueue(&self, ticket: TextureUploadTicket) {
        let (lock, cv) = &*self.inner;
        let mut q = lock.lock().unwrap();
        q.push_back(ticket);
        cv.notify_one();
    }

    pub fn drain_ready(&self, max_items: usize) -> Vec<TextureUploadTicket> {
        let (lock, _) = &*self.inner;
        let mut q = lock.lock().unwrap();
        let n = q.len().min(max_items);
        q.drain(..n).collect()
    }

    pub fn pending(&self) -> usize {
        self.inner.0.lock().unwrap().len()
    }
}

/// Dedicated background texture-heap worker (inspired by `VIctory-Fast`'s
/// `heap_worker.rs`): drains [`TextureAheadQueue`] off title recording threads
/// while allowing urgent material/bake uploads to bypass the queue.
pub struct HeapTextureWorker {
    queue: Arc<TextureAheadQueue>,
    processed: Arc<std::sync::atomic::AtomicU64>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl HeapTextureWorker {
    pub fn spawn<F>(queue: Arc<TextureAheadQueue>, on_upload: F) -> Self
    where
        F: Fn(&TextureUploadTicket) + Send + 'static,
    {
        let processed = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let q = Arc::clone(&queue);
        let p = Arc::clone(&processed);
        let s = Arc::clone(&shutdown);
        let handle = thread::Builder::new()
            .name("strata-heap-worker".into())
            .spawn(move || {
                while !s.load(std::sync::atomic::Ordering::Acquire) {
                    let batch = q.drain_ready(16);
                    if batch.is_empty() {
                        thread::sleep(std::time::Duration::from_millis(2));
                        continue;
                    }
                    for ticket in &batch {
                        on_upload(ticket);
                        p.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            })
            .ok();
        Self {
            queue,
            processed,
            shutdown,
            handle,
        }
    }

    pub fn enqueue(&self, ticket: TextureUploadTicket) {
        self.queue.enqueue(ticket);
    }

    pub fn processed_count(&self) -> u64 {
        self.processed.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for HeapTextureWorker {
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::Release);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Asynchronous background PSO compiler queue: lets title threads enqueue
/// SPIR-V shader blobs to be warmed on worker threads without stalling draws.
pub struct AsyncPsoCompiler {
    pending: Arc<(Mutex<VecDeque<(u64, Vec<u32>)>>, Condvar)>,
    compiled_count: Arc<std::sync::atomic::AtomicU64>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl AsyncPsoCompiler {
    /// Spawns `num_workers` background compilation threads invoking `compile_fn`.
    pub fn spawn<F>(num_workers: usize, compile_fn: F) -> Self
    where
        F: Fn(u64, &[u32]) + Send + Sync + 'static,
    {
        let pending = Arc::new((
            Mutex::new(VecDeque::<(u64, Vec<u32>)>::new()),
            Condvar::new(),
        ));
        let compiled_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let compile_fn = Arc::new(compile_fn);
        let mut workers = Vec::new();
        for i in 0..num_workers.max(1) {
            let p = Arc::clone(&pending);
            let c = Arc::clone(&compiled_count);
            let s = Arc::clone(&shutdown);
            let f = Arc::clone(&compile_fn);
            if let Ok(h) = thread::Builder::new()
                .name(format!("strata-pso-{i}"))
                .spawn(move || {
                    loop {
                        let job = {
                            let (lock, cv) = &*p;
                            let mut q = lock.lock().unwrap();
                            while q.is_empty() && !s.load(std::sync::atomic::Ordering::Acquire) {
                                q = cv.wait(q).unwrap();
                            }
                            if q.is_empty() && s.load(std::sync::atomic::Ordering::Acquire) {
                                break;
                            }
                            q.pop_front()
                        };
                        if let Some((key, words)) = job {
                            f(key, &words);
                            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                })
            {
                workers.push(h);
            }
        }
        Self {
            pending,
            compiled_count,
            shutdown,
            workers,
        }
    }

    pub fn enqueue(&self, key: u64, spirv: Vec<u32>) {
        let (lock, cv) = &*self.pending;
        lock.lock().unwrap().push_back((key, spirv));
        cv.notify_one();
    }

    pub fn compiled_count(&self) -> u64 {
        self.compiled_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn pending_count(&self) -> usize {
        self.pending.0.lock().unwrap().len()
    }
}

impl Drop for AsyncPsoCompiler {
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::Release);
        self.pending.1.notify_all();
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_thin_lists_in_parallel_preserving_order() {
        let mut l0 = ThinCommandList::new();
        l0.dispatch(4, 1, 1, [64, 1, 1], false);
        let mut l1 = ThinCommandList::new();
        l1.draw_instanced(3, 1, 0, 0);
        l1.draw_indirect(4096, 6, 1);

        let pool = ThinTranslationPool::new(4);
        let batches = pool.translate_parallel(vec![l0, l1]);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].ticket, 0);
        assert_eq!(batches[0].dispatch_count, 1);
        assert_eq!(batches[1].ticket, 1);
        assert_eq!(batches[1].draw_count, 2);
        assert_eq!(batches[1].hoisted_clamps, [(4096, 6, 1)]);
    }

    #[test]
    fn texture_ahead_queue_fifo() {
        let q = TextureAheadQueue::new();
        q.enqueue(TextureUploadTicket {
            descriptor_hash: 0x11,
            src_addr: 0x1000,
            width: 256,
            height: 256,
            mip_levels: 9,
            ready_fence: 1,
        });
        q.enqueue(TextureUploadTicket {
            descriptor_hash: 0x22,
            src_addr: 0x2000,
            width: 512,
            height: 512,
            mip_levels: 10,
            ready_fence: 2,
        });
        assert_eq!(q.pending(), 2);
        let first = q.drain_ready(1);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].descriptor_hash, 0x11);
        assert_eq!(q.pending(), 1);
    }
}
