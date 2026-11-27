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
                if let Ok(part) = h.join() {
                    for b in part {
                        let idx = b.ticket as usize;
                        results[idx] = Some(b);
                    }
                }
            }
        });

        results.into_iter().flatten().collect()
    }
}

fn translate_one(ticket: u64, list: ThinCommandList) -> TranslatedBatch {
    let ops = list.commands().to_vec();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_thin_lists_in_parallel_preserving_order() {
        let mut l0 = ThinCommandList::new();
        l0.dispatch(4, 1, 1, [64, 1, 1], false);
        let mut l1 = ThinCommandList::new();
        l1.draw_instanced(3, 1, 0, 0);
        l1.draw_instanced(6, 1, 0, 0);

        let pool = ThinTranslationPool::new(4);
        let batches = pool.translate_parallel(vec![l0, l1]);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].ticket, 0);
        assert_eq!(batches[0].dispatch_count, 1);
        assert_eq!(batches[1].ticket, 1);
        assert_eq!(batches[1].draw_count, 2);
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
