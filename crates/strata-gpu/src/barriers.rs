//! Deferred Vulkan Synchronization2 (`vkCmdPipelineBarrier2`) barrier coalescer.
//!
//! Batches consecutive UAV/memory, buffer, and image layout barriers into minimal
//! `vkCmdPipelineBarrier2` calls, merging overlapping stage/access masks and
//! dropping redundant no-op transitions.

use ash::vk;

/// Telemetry counters for barrier coalescing efficiency.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BarrierStats {
    pub requests: u64,
    pub coalesced_uav: u64,
    pub dropped_noop: u64,
    pub flushes: u64,
}

/// Deferred Synchronization2 barrier recorder for a single command buffer.
#[derive(Debug, Default)]
pub struct BarrierCoalescer {
    pending_uav: Option<vk::MemoryBarrier2<'static>>,
    buffer_barriers: Vec<vk::BufferMemoryBarrier2<'static>>,
    image_barriers: Vec<vk::ImageMemoryBarrier2<'static>>,
    stats: BarrierStats,
}

impl BarrierCoalescer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats(&self) -> BarrierStats {
        self.stats
    }

    pub fn is_empty(&self) -> bool {
        self.pending_uav.is_none()
            && self.buffer_barriers.is_empty()
            && self.image_barriers.is_empty()
    }

    /// Record a global memory / UAV barrier, coalescing with any pending UAV barrier.
    pub fn push_memory(
        &mut self,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
    ) {
        self.stats.requests += 1;
        if let Some(existing) = &mut self.pending_uav {
            existing.src_stage_mask |= src_stage;
            existing.src_access_mask |= src_access;
            existing.dst_stage_mask |= dst_stage;
            existing.dst_access_mask |= dst_access;
            self.stats.coalesced_uav += 1;
        } else {
            self.pending_uav = Some(
                vk::MemoryBarrier2::default()
                    .src_stage_mask(src_stage)
                    .src_access_mask(src_access)
                    .dst_stage_mask(dst_stage)
                    .dst_access_mask(dst_access),
            );
        }
    }

    /// Convenience helper for a compute/shader UAV barrier.
    pub fn push_uav(&mut self) {
        self.push_memory(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
    }

    /// Record a scoped buffer memory barrier, merging adjacent barriers on the same buffer range.
    pub fn push_buffer(&mut self, barrier: vk::BufferMemoryBarrier2<'static>) {
        self.stats.requests += 1;
        if let Some(prev) = self.buffer_barriers.iter_mut().find(|b| {
            b.buffer == barrier.buffer
                && b.offset == barrier.offset
                && b.size == barrier.size
                && b.src_queue_family_index == barrier.src_queue_family_index
                && b.dst_queue_family_index == barrier.dst_queue_family_index
        }) {
            prev.src_stage_mask |= barrier.src_stage_mask;
            prev.src_access_mask |= barrier.src_access_mask;
            prev.dst_stage_mask |= barrier.dst_stage_mask;
            prev.dst_access_mask |= barrier.dst_access_mask;
            self.stats.coalesced_uav += 1;
            return;
        }
        self.buffer_barriers.push(barrier);
    }

    /// Record an image layout transition barrier, dropping redundant read-only no-ops
    /// and folding chained transitions on the same image/subresource.
    pub fn push_image(&mut self, barrier: vk::ImageMemoryBarrier2<'static>) {
        self.stats.requests += 1;
        let has_write = barrier.src_access_mask.intersects(
            vk::AccessFlags2::SHADER_WRITE
                | vk::AccessFlags2::SHADER_STORAGE_WRITE
                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags2::TRANSFER_WRITE
                | vk::AccessFlags2::HOST_WRITE
                | vk::AccessFlags2::MEMORY_WRITE,
        );
        if barrier.old_layout == barrier.new_layout
            && !has_write
            && barrier.src_queue_family_index == barrier.dst_queue_family_index
        {
            self.stats.dropped_noop += 1;
            return;
        }
        if let Some(prev) = self.image_barriers.iter_mut().find(|b| {
            b.image == barrier.image
                && same_subresource_range(b.subresource_range, barrier.subresource_range)
                && b.new_layout == barrier.old_layout
                && b.src_queue_family_index == barrier.src_queue_family_index
                && b.dst_queue_family_index == barrier.dst_queue_family_index
        }) {
            prev.new_layout = barrier.new_layout;
            prev.dst_stage_mask |= barrier.dst_stage_mask;
            prev.dst_access_mask |= barrier.dst_access_mask;
            self.stats.coalesced_uav += 1;
            return;
        }
        self.image_barriers.push(barrier);
    }

    /// Flush all queued barriers into `cmd` via a single `vkCmdPipelineBarrier2`.
    pub unsafe fn flush(&mut self, device: &ash::Device, cmd: vk::CommandBuffer) {
        if self.is_empty() {
            return;
        }
        let mem_slice = self.pending_uav.as_slice();
        let dep = vk::DependencyInfo::default()
            .memory_barriers(mem_slice)
            .buffer_memory_barriers(&self.buffer_barriers)
            .image_memory_barriers(&self.image_barriers);
        device.cmd_pipeline_barrier2(cmd, &dep);
        self.pending_uav = None;
        self.buffer_barriers.clear();
        self.image_barriers.clear();
        self.stats.flushes += 1;
    }

    /// Clear pending barriers without issuing Vulkan commands (for unit tests or reset).
    pub fn take_counts(&mut self) -> (usize, usize, usize) {
        let counts = (
            usize::from(self.pending_uav.is_some()),
            self.buffer_barriers.len(),
            self.image_barriers.len(),
        );
        self.pending_uav = None;
        self.buffer_barriers.clear();
        self.image_barriers.clear();
        if counts != (0, 0, 0) {
            self.stats.flushes += 1;
        }
        counts
    }
}

fn same_subresource_range(a: vk::ImageSubresourceRange, b: vk::ImageSubresourceRange) -> bool {
    a.aspect_mask == b.aspect_mask
        && a.base_mip_level == b.base_mip_level
        && a.level_count == b.level_count
        && a.base_array_layer == b.base_array_layer
        && a.layer_count == b.layer_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesces_uav_and_drops_noop_image_barriers() {
        let mut c = BarrierCoalescer::new();
        c.push_uav();
        c.push_uav();
        c.push_uav();

        // No-op read-only transition should be dropped
        c.push_image(
            vk::ImageMemoryBarrier2::default()
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags2::SHADER_READ)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ),
        );

        // Chained layout transition A -> B -> C should fold to A -> C
        c.push_image(
            vk::ImageMemoryBarrier2::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE),
        );
        c.push_image(
            vk::ImageMemoryBarrier2::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ),
        );

        let (m, b, i) = c.take_counts();
        assert_eq!((m, b, i), (1, 0, 1));
        let st = c.stats();
        assert_eq!(st.requests, 6);
        assert_eq!(st.coalesced_uav, 3);
        assert_eq!(st.dropped_noop, 1);
        assert_eq!(st.flushes, 1);
    }
}
