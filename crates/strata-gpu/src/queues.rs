//! Multi-queue `VK_KHR_timeline_semaphore` scheduler across Graphics,
//! Async Compute, and dedicated Transfer/Copy queues.
//!
//! Enables overlapping PCIe buffer/texture uploads on a Transfer queue with
//! Graphics and Async Compute execution without CPU stalls.

use ash::vk;

/// Logical hardware queue role in Strata's multi-queue engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueKind {
    Graphics = 0,
    AsyncCompute = 1,
    TransferCopy = 2,
}

impl QueueKind {
    pub const COUNT: usize = 3;

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Monotonic timeline point on a specific hardware queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimelinePoint {
    pub queue: QueueKind,
    pub value: u64,
    pub stage_mask: vk::PipelineStageFlags2,
}

/// Planned queue submission with explicit timeline semaphore waits and signal.
#[derive(Debug, Clone)]
pub struct QueueSubmitPlan {
    pub queue: QueueKind,
    pub waits: Vec<TimelinePoint>,
    pub signal: TimelinePoint,
}

/// Tracks per-queue `VK_KHR_timeline_semaphore` counters and cross-queue edges.
#[derive(Debug)]
pub struct TimelineScheduler {
    next_value: [u64; QueueKind::COUNT],
    completed_value: [u64; QueueKind::COUNT],
    pending_waits: [Vec<TimelinePoint>; QueueKind::COUNT],
}

impl Default for TimelineScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl TimelineScheduler {
    pub fn new() -> Self {
        Self {
            next_value: [0; QueueKind::COUNT],
            completed_value: [0; QueueKind::COUNT],
            pending_waits: std::array::from_fn(|_| Vec::new()),
        }
    }

    /// Last submitted timeline value on `queue`.
    pub fn submitted(&self, queue: QueueKind) -> u64 {
        self.next_value[queue.index()]
    }

    /// Last known GPU-completed timeline value on `queue`.
    pub fn completed(&self, queue: QueueKind) -> u64 {
        self.completed_value[queue.index()]
    }

    /// Update observed GPU completion for `queue` (e.g. from `vkGetSemaphoreCounterValue`).
    pub fn observe_completed(&mut self, queue: QueueKind, value: u64) {
        let slot = &mut self.completed_value[queue.index()];
        if value > *slot {
            *slot = value;
        }
    }

    /// Record that the next submission on `consumer` must wait for `producer_point`.
    /// Redundant waits (already completed or dominated by a newer wait on the same queue)
    /// are automatically elided.
    pub fn add_wait(&mut self, consumer: QueueKind, point: TimelinePoint) {
        if point.value == 0 || point.value <= self.completed_value[point.queue.index()] {
            return;
        }
        let waits = &mut self.pending_waits[consumer.index()];
        if let Some(existing) = waits.iter_mut().find(|w| w.queue == point.queue) {
            if point.value > existing.value {
                existing.value = point.value;
            }
            existing.stage_mask |= point.stage_mask;
        } else {
            waits.push(point);
        }
    }

    /// Plan a submission on `queue`, advancing its timeline counter and draining pending waits.
    pub fn plan_submit(
        &mut self,
        queue: QueueKind,
        signal_stage: vk::PipelineStageFlags2,
    ) -> QueueSubmitPlan {
        let idx = queue.index();
        self.next_value[idx] += 1;
        let signal = TimelinePoint {
            queue,
            value: self.next_value[idx],
            stage_mask: signal_stage,
        };
        let waits = std::mem::take(&mut self.pending_waits[idx]);
        QueueSubmitPlan {
            queue,
            waits,
            signal,
        }
    }

    /// Convert a [`QueueSubmitPlan`] into Vulkan `SemaphoreSubmitInfo` wait/signal descriptors.
    pub fn build_semaphore_infos(
        plan: &QueueSubmitPlan,
        semaphores: &[vk::Semaphore; QueueKind::COUNT],
    ) -> (
        Vec<vk::SemaphoreSubmitInfo<'static>>,
        vk::SemaphoreSubmitInfo<'static>,
    ) {
        let waits = plan
            .waits
            .iter()
            .map(|w| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphores[w.queue.index()])
                    .value(w.value)
                    .stage_mask(w.stage_mask)
            })
            .collect();
        let signal = vk::SemaphoreSubmitInfo::default()
            .semaphore(semaphores[plan.signal.queue.index()])
            .value(plan.signal.value)
            .stage_mask(plan.signal.stage_mask);
        (waits, signal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_cross_queue_transfer_to_graphics_and_elides_completed() {
        let mut sched = TimelineScheduler::new();
        let upload1 = sched.plan_submit(QueueKind::TransferCopy, vk::PipelineStageFlags2::COPY);
        let upload2 = sched.plan_submit(QueueKind::TransferCopy, vk::PipelineStageFlags2::COPY);
        // Queue both waits on Graphics; scheduler should keep only the dominating value (2)
        sched.add_wait(QueueKind::Graphics, upload1.signal);
        sched.add_wait(QueueKind::Graphics, upload2.signal);
        let gfx = sched.plan_submit(QueueKind::Graphics, vk::PipelineStageFlags2::ALL_GRAPHICS);
        assert_eq!(gfx.waits.len(), 1);
        assert_eq!(gfx.waits[0].queue, QueueKind::TransferCopy);
        assert_eq!(gfx.waits[0].value, 2);
        assert_eq!(gfx.signal.value, 1);

        // Once TransferCopy reaches 2, waiting on 2 is elided
        sched.observe_completed(QueueKind::TransferCopy, 2);
        sched.add_wait(QueueKind::AsyncCompute, upload2.signal);
        let cmp = sched.plan_submit(
            QueueKind::AsyncCompute,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );
        assert!(cmp.waits.is_empty());
    }
}
