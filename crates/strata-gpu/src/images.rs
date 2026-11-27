//! 2D images, samplers and the descriptor set compute and graphics shaders
//! index (set 0: `textures[16]`, `samplers[16]`, `storage_images[16]`).
//!
//! Images live in `GENERAL` layout once created, valid for sampling, storage
//! writes and copies alike; every use is a whole synchronous submission, so
//! barriers are full-image and coarse.

use ash::vk;
use strata_shader::descriptor::{Clamp, Filter, SamplerDescriptor};

use crate::context::GpuError;
use crate::runner::ComputeRunner;

/// Slots per binding of the global descriptor set.
pub const IMAGE_SLOTS: usize = strata_shader::spirv::IMAGE_ARRAY_LEN as usize;

/// A device-local 2D image with one mip level and a view of it.
pub struct GpuImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    /// Created with `STORAGE` usage (and an identity view).
    pub storage: bool,
}

/// What a dispatch or draw binds, by slot; `None` slots get a 1x1 default.
#[derive(Default, Clone)]
pub struct ImageBindings {
    pub textures: Vec<Option<(vk::ImageView, Option<vk::Sampler>)>>,
    pub storage: Vec<Option<vk::ImageView>>,
}

impl ImageBindings {
    pub fn set_texture(&mut self, slot: u32, view: vk::ImageView, sampler: Option<vk::Sampler>) {
        let i = slot as usize;
        if self.textures.len() <= i {
            self.textures.resize(i + 1, None);
        }
        self.textures[i] = Some((view, sampler));
    }

    pub fn set_storage(&mut self, slot: u32, view: vk::ImageView) {
        let i = slot as usize;
        if self.storage.len() <= i {
            self.storage.resize(i + 1, None);
        }
        self.storage[i] = Some(view);
    }
}

fn clamp_mode(c: Clamp, mirror_once: bool) -> vk::SamplerAddressMode {
    match c {
        Clamp::Wrap => vk::SamplerAddressMode::REPEAT,
        Clamp::Mirror => vk::SamplerAddressMode::MIRRORED_REPEAT,
        Clamp::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        Clamp::MirrorOnceToEdge if mirror_once => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
        Clamp::MirrorOnceToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        Clamp::ClampToBorder | Clamp::MirrorOnceToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
    }
}

fn filter(f: Filter) -> vk::Filter {
    match f {
        Filter::Point => vk::Filter::NEAREST,
        Filter::Linear => vk::Filter::LINEAR,
    }
}

impl ComputeRunner {
    /// Whether an optimal-tiling image of `format` supports `features`.
    pub fn format_supports(&self, format: vk::Format, features: vk::FormatFeatureFlags) -> bool {
        // SAFETY: a property query.
        let props = unsafe {
            self.instance
                .get_physical_device_format_properties(self.pdev, format)
        };
        props.optimal_tiling_features.contains(features)
    }

    fn device_local_type(&self, bits: u32) -> Result<u32, GpuError> {
        // SAFETY: a property query.
        let mem = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.pdev)
        };
        (0..mem.memory_type_count)
            .find(|&i| {
                bits & (1 << i) != 0
                    && mem.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or(GpuError::NoDevice)
    }

    /// Creates a `width`×`height` image of `format`, seen through `components`
    /// (ignored for storage images, whose views must be identity).
    pub fn create_image_2d(
        &self,
        format: vk::Format,
        width: u32,
        height: u32,
        storage: bool,
        components: vk::ComponentMapping,
    ) -> Result<GpuImage, GpuError> {
        let mut usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC;
        if storage {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if self.format_supports(format, vk::FormatFeatureFlags::COLOR_ATTACHMENT) {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT; // render targets are textures too
        }
        // SAFETY: plain object creation on our device.
        unsafe {
            let ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = self.device.create_image(&ci, None)?;
            let req = self.device.get_image_memory_requirements(image);
            let memory = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(self.device_local_type(req.memory_type_bits)?),
                None,
            )?;
            self.device.bind_image_memory(image, memory, 0)?;
            let view = self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(if storage {
                        vk::ComponentMapping::default()
                    } else {
                        components
                    })
                    .subresource_range(color_range()),
                None,
            )?;
            Ok(GpuImage {
                image,
                memory,
                view,
                format,
                width,
                height,
                storage,
            })
        }
    }

    /// Replaces the image's contents with tightly packed rows and leaves it in
    /// `GENERAL` layout.
    pub fn upload_image(&self, img: &GpuImage, tight: &[u8]) -> Result<(), GpuError> {
        let mut staging = self.alloc_bda_buffer(tight.len().max(4) as u64)?;
        // SAFETY: the staging buffer holds `tight.len()` bytes.
        unsafe {
            staging
                .as_mut_slice::<u8>(tight.len())
                .copy_from_slice(tight)
        };
        let buffer = staging.buffer;
        let result = self.one_shot(|dev, cmd| unsafe {
            barrier(
                dev,
                cmd,
                img.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            let region = [buffer_image_copy(img)];
            dev.cmd_copy_buffer_to_image(
                cmd,
                buffer,
                img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
            );
            barrier(
                dev,
                cmd,
                img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
        });
        self.free_bda_buffer(staging);
        result
    }

    /// The image's contents as tightly packed rows of `bytes` total.
    pub fn read_image(&self, img: &GpuImage, bytes: usize) -> Result<Vec<u8>, GpuError> {
        let staging = self.alloc_bda_buffer(bytes.max(4) as u64)?;
        let buffer = staging.buffer;
        let result = self.one_shot(|dev, cmd| unsafe {
            barrier(
                dev,
                cmd,
                img.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
            );
            let region = [buffer_image_copy(img)];
            dev.cmd_copy_image_to_buffer(cmd, img.image, vk::ImageLayout::GENERAL, buffer, &region);
        });
        // SAFETY: the copy above finished (one_shot waits).
        let out = result.map(|()| unsafe { staging.as_slice::<u8>(bytes).to_vec() });
        self.free_bda_buffer(staging);
        out
    }

    pub fn destroy_image(&self, img: GpuImage) {
        // SAFETY: no submission uses it (all submissions are waited on).
        unsafe {
            self.device.destroy_image_view(img.view, None);
            self.device.destroy_image(img.image, None);
            self.device.free_memory(img.memory, None);
        }
    }

    /// A sampler for an S#. Anisotropy is not enabled on this device, so
    /// anisotropic S#s filter as their min/mag/mip filters say.
    pub fn create_sampler(&self, s: &SamplerDescriptor) -> Result<vk::Sampler, GpuError> {
        let mirror_once = self.sampler_mirror_clamp_to_edge;
        let (mip_mode, max_lod) = match s.mip {
            None => (vk::SamplerMipmapMode::NEAREST, 0.25), // base level only
            Some(Filter::Point) => (vk::SamplerMipmapMode::NEAREST, s.max_lod),
            Some(Filter::Linear) => (vk::SamplerMipmapMode::LINEAR, s.max_lod),
        };
        let ci = vk::SamplerCreateInfo::default()
            .mag_filter(filter(s.mag))
            .min_filter(filter(s.min))
            .mipmap_mode(mip_mode)
            .address_mode_u(clamp_mode(s.clamp[0], mirror_once))
            .address_mode_v(clamp_mode(s.clamp[1], mirror_once))
            .address_mode_w(clamp_mode(s.clamp[2], mirror_once))
            .mip_lod_bias(s.lod_bias)
            .min_lod(s.min_lod)
            .max_lod(max_lod.max(s.min_lod))
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK);
        // SAFETY: plain object creation.
        Ok(unsafe { self.device.create_sampler(&ci, None)? })
    }

    pub fn destroy_sampler(&self, s: vk::Sampler) {
        // SAFETY: no submission uses it.
        unsafe { self.device.destroy_sampler(s, None) };
    }

    /// Records `f` into a one-time command buffer, submits it and waits.
    pub(crate) fn one_shot(
        &self,
        f: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), GpuError> {
        // SAFETY: a command buffer from our pool, submitted and waited on.
        unsafe {
            let cmd = self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            f(&self.device, cmd);
            self.device.end_command_buffer(cmd)?;
            self.submit_and_wait(cmd)
        }
    }
}

pub(crate) fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

fn buffer_image_copy(img: &GpuImage) -> vk::BufferImageCopy {
    vk::BufferImageCopy::default()
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1),
        )
        .image_extent(vk::Extent3D {
            width: img.width,
            height: img.height,
            depth: 1,
        })
}

/// A full-image barrier between any earlier and any later work.
///
/// # Safety
/// `cmd` is recording.
pub(crate) unsafe fn barrier(
    dev: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    let b = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::HOST)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::HOST_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::HOST)
        .dst_access_mask(
            vk::AccessFlags2::MEMORY_READ
                | vk::AccessFlags2::MEMORY_WRITE
                | vk::AccessFlags2::HOST_READ,
        )
        .old_layout(from)
        .new_layout(to)
        .image(image)
        .subresource_range(color_range())];
    // SAFETY: forwarded from the caller.
    unsafe {
        dev.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&b),
        )
    };
}
