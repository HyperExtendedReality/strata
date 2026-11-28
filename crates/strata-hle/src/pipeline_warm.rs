//! Bounded background compilation of registered host pipelines.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use ash::vk;
use strata_gpu::graphics::GraphicsStages;
use strata_gpu::runner::{ComputeRunner, spirv_hash};

use crate::d3d12x::PipelineEntry;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Key {
    Compute(u64),
    Graphics(u64),
}

struct Job {
    key: Key,
    pipeline: PipelineEntry,
    formats: Vec<vk::Format>,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    pending: HashSet<Key>,
    stop: bool,
}

pub(crate) struct Precompiler {
    queue: Arc<(Mutex<Queue>, Condvar)>,
    worker: Option<JoinHandle<()>>,
}

impl Precompiler {
    pub(crate) fn spawn(runner: &Arc<ComputeRunner>) -> Option<Self> {
        let runner = Arc::downgrade(runner);
        let queue = Arc::new((Mutex::new(Queue::default()), Condvar::new()));
        let worker_queue = Arc::clone(&queue);
        let worker = std::thread::Builder::new().name("strata-pso".into()).spawn(move || {
            let (lock, ready) = &*worker_queue;
            loop {
                let job = {
                    let mut queue = ready.wait_while(lock.lock().unwrap(), |queue| {
                        !queue.stop && queue.jobs.is_empty()
                    }).unwrap();
                    if queue.stop {
                        return;
                    }
                    let next = queue.jobs.iter().position(|job| matches!(job.key, Key::Compute(_))).unwrap_or(0);
                    queue.jobs.remove(next).unwrap()
                };
                let Some(runner) = runner.upgrade() else { return };
                let pipeline = &job.pipeline;
                let result = if matches!(job.key, Key::Compute(_)) {
                    runner.pipeline_for(&pipeline.spirv, None).map(|_| ())
                } else {
                    let stages = if let Some(mesh) = &pipeline.mesh {
                        GraphicsStages::Mesh { task: pipeline.task.as_deref(), mesh, ps: &pipeline.spirv }
                    } else if let Some(vs) = &pipeline.vertex {
                        GraphicsStages::Classic {
                            vs, hs: pipeline.hull.as_deref(), ds: pipeline.domain.as_deref(),
                            gs: pipeline.geometry.as_deref(), ps: &pipeline.spirv,
                        }
                    } else {
                        lock.lock().unwrap().pending.remove(&job.key);
                        continue;
                    };
                    runner.precompile_graphics(stages, &job.formats, &pipeline.gfx.clone().unwrap_or_default())
                };
                if let Err(error) = result {
                    crate::crash::log_line(&format!("background pipeline compilation: {error}"));
                }
                lock.lock().unwrap().pending.remove(&job.key);
            }
        }).ok()?;
        Some(Self { queue, worker: Some(worker) })
    }

    pub(crate) fn enqueue(&self, graphics_key: Option<u64>, pipeline: &PipelineEntry, formats: Vec<vk::Format>) {
        let words = [Some(pipeline.spirv.as_slice()), pipeline.vertex.as_deref(), pipeline.hull.as_deref(),
            pipeline.domain.as_deref(), pipeline.geometry.as_deref(), pipeline.task.as_deref(), pipeline.mesh.as_deref()]
            .into_iter().flatten().map(<[u32]>::len).sum::<usize>();
        if words > 256 * 1024 || pipeline.spirv.is_empty() {
            return;
        }
        let (lock, ready) = &*self.queue;
        let key = graphics_key.map_or_else(|| Key::Compute(spirv_hash(&pipeline.spirv)), Key::Graphics);
        let mut queue = lock.lock().unwrap();
        if queue.pending.len() < 16 && queue.pending.insert(key) {
            queue.jobs.push_back(Job { key, pipeline: pipeline.clone(), formats });
            ready.notify_one();
        }
    }
}

impl Drop for Precompiler {
    fn drop(&mut self) {
        let (lock, ready) = &*self.queue;
        lock.lock().unwrap().stop = true;
        ready.notify_one();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
