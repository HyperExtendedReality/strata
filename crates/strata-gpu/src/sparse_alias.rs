//! Live sparse image aliasing for identically laid out RGBA8 color tiles.

use ash::vk;

use crate::{context::GpuError, runner::ComputeRunner, vram::TILE_BYTES};

struct AliasResources<'a> {
    runner: &'a ComputeRunner,
    images: [vk::Image; 2],
    target: vk::ImageView,
    tile: vk::DeviceMemory,
}

impl Drop for AliasResources<'_> {
    fn drop(&mut self) {
        unsafe {
            self.runner.device.destroy_image_view(self.target, None);
            for image in self.images {
                self.runner.device.destroy_image(image, None);
            }
            self.runner.device.free_memory(self.tile, None);
        }
    }
}

impl ComputeRunner {
    /// Bind one shared 64 KiB tile in two sparse color-attachment images,
    /// clear the first alias as a render target, and read through the second.
    pub fn verify_sparse_tile_aliasing(&self) -> Result<(u64, vk::Extent3D), GpuError> {
        let families = unsafe {
            self.instance
                .get_physical_device_queue_family_properties(self.pdev)
        };
        if !self.sparse_aliased_images
            || !families[self.queue_family as usize]
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS)
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT.into());
        }
        let extent = vk::Extent3D {
            width: 256,
            height: 256,
            depth: 1,
        };
        let flags = vk::ImageCreateFlags::SPARSE_BINDING
            | vk::ImageCreateFlags::SPARSE_RESIDENCY
            | vk::ImageCreateFlags::SPARSE_ALIASED;
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC;
        let ci = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage);
        let mut resources = AliasResources {
            runner: self,
            images: [vk::Image::null(); 2],
            target: vk::ImageView::null(),
            tile: vk::DeviceMemory::null(),
        };
        let dev = &self.device;
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let granularity = unsafe {
            let properties = self.instance.get_physical_device_image_format_properties(
                self.pdev,
                ci.format,
                ci.image_type,
                ci.tiling,
                ci.usage,
                ci.flags,
            )?;
            if properties.max_extent.width < extent.width
                || properties.max_extent.height < extent.height
            {
                return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED.into());
            }
            for image in &mut resources.images {
                *image = dev.create_image(&ci, None)?;
            }
            let [first, second] = resources.images;
            let req = dev.get_image_memory_requirements(first);
            let second_req = dev.get_image_memory_requirements(second);
            let sparse = dev.get_image_sparse_memory_requirements(first);
            let other = dev.get_image_sparse_memory_requirements(second);
            let get_granularity = |requirements: &[vk::SparseImageMemoryRequirements]| {
                let [req] = requirements else { return None };
                (req.format_properties.aspect_mask == vk::ImageAspectFlags::COLOR
                    && req.image_mip_tail_first_lod > 0)
                    .then_some(req.format_properties.image_granularity)
            };
            let gran = get_granularity(&sparse).ok_or(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)?;
            if req.alignment != TILE_BYTES
                || second_req.alignment != TILE_BYTES
                || get_granularity(&other) != Some(gran)
                || gran.depth != 1
                || gran.width > extent.width
                || gran.height > extent.height
                || u64::from(gran.width) * u64::from(gran.height) * 4 != TILE_BYTES
            {
                return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED.into());
            }
            let memory_type_index =
                self.device_local_type(req.memory_type_bits & second_req.memory_type_bits)?;
            resources.tile = dev.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(TILE_BYTES)
                    .memory_type_index(memory_type_index),
                None,
            )?;
            let bind = [vk::SparseImageMemoryBind::default()
                .subresource(
                    vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::COLOR),
                )
                .extent(gran)
                .memory(resources.tile)];
            let images = resources.images.map(|image| {
                vk::SparseImageMemoryBindInfo::default()
                    .image(image)
                    .binds(&bind)
            });
            let info = [vk::BindSparseInfo::default().image_binds(&images)];
            let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let result = {
                let _queue = self.queue_lock.lock().unwrap_or_else(|e| e.into_inner());
                dev.queue_bind_sparse(self.queue, &info, fence)
                    .and_then(|()| self.wait_submitted_fence(self.queue, fence))
            };
            dev.destroy_fence(fence, None);
            result?;
            resources.target = dev.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(first)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(ci.format)
                    .subresource_range(range),
                None,
            )?;
            gran
        };

        let staging = self.alloc_bda_buffer(TILE_BYTES)?;
        let tile_bytes = TILE_BYTES as usize;
        let result = self
            .one_shot(|dev, cmd| unsafe {
                for image in resources.images {
                    let barrier = [vk::ImageMemoryBarrier2::default()
                        .image(image)
                        .subresource_range(range)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .dst_access_mask(
                            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                        )];
                    dev.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&barrier),
                    );
                }
                let copy = vk::BufferImageCopy::default()
                    .image_extent(granularity)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    );
                let attachments = [vk::RenderingAttachmentInfo::default()
                    .image_view(resources.target)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [1.0, 0.0, 1.0, 1.0],
                        },
                    })];
                dev.cmd_begin_rendering(
                    cmd,
                    &vk::RenderingInfo::default()
                        .render_area(vk::Rect2D::default().extent(vk::Extent2D {
                            width: granularity.width,
                            height: granularity.height,
                        }))
                        .layer_count(1)
                        .color_attachments(&attachments),
                );
                dev.cmd_end_rendering(cmd);
                let visibility = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
                dev.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&visibility),
                );
                dev.cmd_copy_image_to_buffer(
                    cmd,
                    resources.images[1],
                    vk::ImageLayout::GENERAL,
                    staging.buffer,
                    &[copy],
                );
                let host = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)];
                dev.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&host),
                );
            })
            .and_then(|()| {
                let bytes = unsafe { staging.as_slice::<u8>(tile_bytes) };
                if !bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|pixel| *pixel == [255, 0, 255, 255])
                {
                    return Err(GpuError::Io(
                        "sparse aliases returned different texels".into(),
                    ));
                }
                Ok((TILE_BYTES, granularity))
            });
        self.free_bda_buffer(staging);
        result
    }
}
