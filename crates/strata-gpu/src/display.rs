//! Showing title frames in a window: the back buffer the title presents (an
//! image the texture cache made from its T#) is blitted, scaled to the
//! window's client area, into a swapchain image and presented.
//!
//! No-API: one `vkCmdBlitImage` per frame — no render pass, framebuffer or
//! pipeline. The swapchain is recreated when the window's size changes or
//! the surface reports it out of date. Back buffers in 10-bit or FP16
//! formats get an HDR swapchain (`HDR10_ST2084` / `EXTENDED_SRGB_LINEAR`)
//! when the display offers one; otherwise SDR `B8G8R8A8`.

use ash::vk;

use crate::context::GpuError;
use crate::graphics::{DrawArgs, DrawQueryResults, GraphicsPsoState, GraphicsStages};
use crate::images::{GpuImage, ImageBindings, aspect_of, barrier_image};
use crate::runner::ComputeRunner;
use crate::swapchain::SurfaceHandle;

/// Renders display planes in submission order into a distinct colour image.
/// The base is opaque and subsequent planes use premultiplied alpha.
pub fn compose_planes(
    runner: &ComputeRunner,
    planes: &[&GpuImage],
    target: &GpuImage,
) -> Result<(), GpuError> {
    compose(runner, planes, target, None)
}

#[derive(Debug, Clone, Copy)]
pub struct PlaneTiming {
    pub wall_ms: f64,
    pub gpu_ms: Option<f64>,
}

pub fn compose_planes_timed(
    runner: &ComputeRunner,
    planes: &[&GpuImage],
    target: &GpuImage,
) -> Result<Vec<PlaneTiming>, GpuError> {
    let mut timings = Vec::with_capacity(planes.len());
    compose(runner, planes, target, Some(&mut timings))?;
    Ok(timings)
}

fn compose(
    runner: &ComputeRunner,
    planes: &[&GpuImage],
    target: &GpuImage,
    mut timings: Option<&mut Vec<PlaneTiming>>,
) -> Result<(), GpuError> {
    if planes.is_empty() || planes.iter().any(|p| p.image == target.image) {
        return Err(GpuError::Vk(vk::Result::ERROR_INITIALIZATION_FAILED));
    }
    static SHADERS: std::sync::OnceLock<(Vec<u32>, Vec<u32>)> = std::sync::OnceLock::new();
    let (vs, fs) = SHADERS.get_or_init(|| {
        let words = |bytes: &[u8]| {
            bytes
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                .collect()
        };
        (
            words(strata_runtime_shaders::COMPOSE_VS_SPV),
            words(strata_runtime_shaders::COMPOSE_FS_SPV),
        )
    });
    runner.one_shot(|dev, cmd| unsafe {
        for plane in planes {
            barrier_image(
                dev,
                cmd,
                plane,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
            );
        }
    })?;
    for (index, plane) in planes.iter().enumerate() {
        let mut images = ImageBindings::default();
        images.set_texture(0, plane.view, None);
        let blend = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(index > 0)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let state = GraphicsPsoState {
            blend_attachments: vec![blend],
            ..Default::default()
        };
        let mut query = DrawQueryResults {
            capture_time: timings.is_some(),
            ..Default::default()
        };
        let start = std::time::Instant::now();
        runner.draw_stages_with_state(
            GraphicsStages::Classic {
                vs,
                hs: None,
                ds: None,
                gs: None,
                ps: fs,
            },
            &[],
            Some(DrawArgs {
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            }),
            None,
            &[target],
            None,
            None,
            &images,
            &state,
            timings.is_some().then_some(&mut query),
        )?;
        if let Some(timings) = &mut timings {
            timings.push(PlaneTiming {
                wall_ms: start.elapsed().as_secs_f64() * 1000.0,
                gpu_ms: query.gpu_ms,
            });
        }
    }
    Ok(())
}

/// The colour space a frame is shown in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayMode {
    Sdr,
    /// PQ (ST 2084), BT.2020 primaries: for 10-bit back buffers.
    Hdr10,
    /// scRGB (linear BT.709 beyond 1.0): for FP16 back buffers.
    ScRgb,
}

/// A window's swapchain and what presenting to it needs.
pub struct Display {
    handle: SurfaceHandle,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub extent: vk::Extent2D,
    pub mode: DisplayMode,
    acquired: vk::Semaphore,
    blitted: vk::Semaphore,
    fence: vk::Fence,
    cmd: vk::CommandBuffer,
    pub frames: u64,
}

impl Display {
    /// A swapchain for `handle`, sized `width`×`height` (the window's client
    /// area). Fails without window-system integration.
    pub fn new(
        runner: &ComputeRunner,
        handle: SurfaceHandle,
        width: u32,
        height: u32,
    ) -> Result<Self, GpuError> {
        if !runner.wsi {
            return Err(GpuError::NoDevice);
        }
        let surface_loader = ash::khr::surface::Instance::new(&runner._entry, &runner.instance);
        let swapchain_loader = ash::khr::swapchain::Device::new(&runner.instance, &runner.device);
        // SAFETY: plain object creation on our instance/device.
        unsafe {
            let surface = match handle {
                #[cfg(windows)]
                SurfaceHandle::Win32 { hinstance, hwnd } => {
                    let w =
                        ash::khr::win32_surface::Instance::new(&runner._entry, &runner.instance);
                    w.create_win32_surface(
                        &vk::Win32SurfaceCreateInfoKHR::default()
                            .hinstance(hinstance as vk::HINSTANCE)
                            .hwnd(hwnd as vk::HWND),
                        None,
                    )?
                }
                _ => return Err(GpuError::NoDevice),
            };
            let supported = surface_loader.get_physical_device_surface_support(
                runner.pdev,
                runner.queue_family,
                surface,
            )?;
            if !supported {
                surface_loader.destroy_surface(surface, None);
                return Err(GpuError::NoDevice);
            }
            let sem = vk::SemaphoreCreateInfo::default();
            let acquired = runner.device.create_semaphore(&sem, None)?;
            let blitted = runner.device.create_semaphore(&sem, None)?;
            let fence = runner.device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?;
            let cmd = runner.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(runner.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            let mut d = Self {
                handle,
                surface_loader,
                swapchain_loader,
                surface,
                swapchain: vk::SwapchainKHR::null(),
                images: Vec::new(),
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                extent: vk::Extent2D { width, height },
                mode: DisplayMode::Sdr,
                acquired,
                blitted,
                fence,
                cmd,
                frames: 0,
            };
            d.recreate(runner, width, height, DisplayMode::Sdr)?;
            Ok(d)
        }
    }

    /// The surface format for `mode`, falling back to SDR.
    fn pick_format(
        &self,
        runner: &ComputeRunner,
        mode: DisplayMode,
    ) -> (vk::SurfaceFormatKHR, DisplayMode) {
        // SAFETY: a query on our surface.
        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(runner.pdev, self.surface)
                .unwrap_or_default()
        };
        let want = match mode {
            DisplayMode::Hdr10 => Some((
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::ColorSpaceKHR::HDR10_ST2084_EXT,
            )),
            DisplayMode::ScRgb => Some((
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
            )),
            DisplayMode::Sdr => None,
        };
        if let Some((f, cs)) = want
            && runner.hdr_colorspaces
            && let Some(sf) = formats
                .iter()
                .find(|s| s.format == f && s.color_space == cs)
        {
            return (*sf, mode);
        }
        let sdr = formats
            .iter()
            .find(|s| {
                matches!(
                    s.format,
                    vk::Format::B8G8R8A8_UNORM | vk::Format::R8G8B8A8_UNORM
                ) && s.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .or_else(|| formats.first().copied())
            .unwrap_or(vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            });
        (sdr, DisplayMode::Sdr)
    }

    /// (Re)creates the swapchain at `width`×`height` in `mode`.
    fn recreate(
        &mut self,
        runner: &ComputeRunner,
        width: u32,
        height: u32,
        mode: DisplayMode,
    ) -> Result<(), GpuError> {
        // SAFETY: nothing uses the old swapchain (the last present's fence was waited on).
        unsafe {
            let _ = runner.device.device_wait_idle();
            let caps = self
                .surface_loader
                .get_physical_device_surface_capabilities(runner.pdev, self.surface)?;
            let extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: width.clamp(
                        caps.min_image_extent.width,
                        caps.max_image_extent.width.max(1),
                    ),
                    height: height.clamp(
                        caps.min_image_extent.height,
                        caps.max_image_extent.height.max(1),
                    ),
                }
            };
            if extent.width == 0 || extent.height == 0 {
                // Minimised: keep the old swapchain until there is an area.
                return Ok(());
            }
            let (sf, mode) = self.pick_format(runner, mode);
            let count = (caps.min_image_count + 1).min(if caps.max_image_count > 0 {
                caps.max_image_count
            } else {
                3
            });
            let old = self.swapchain;
            let ci = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(count)
                .image_format(sf.format)
                .image_color_space(sf.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true)
                .old_swapchain(old);
            self.swapchain = self.swapchain_loader.create_swapchain(&ci, None)?;
            if old != vk::SwapchainKHR::null() {
                self.swapchain_loader.destroy_swapchain(old, None);
            }
            self.images = self.swapchain_loader.get_swapchain_images(self.swapchain)?;
            self.format = sf.format;
            self.color_space = sf.color_space;
            self.extent = extent;
            self.mode = mode;
        }
        Ok(())
    }

    /// The display mode a back buffer of `format` asks for.
    pub fn mode_for(format: vk::Format) -> DisplayMode {
        match format {
            vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32 => {
                DisplayMode::Hdr10
            }
            vk::Format::R16G16B16A16_SFLOAT => DisplayMode::ScRgb,
            _ => DisplayMode::Sdr,
        }
    }

    /// Shows `frame` scaled to the window's `width`×`height` client area.
    /// `Ok(false)`: the window has no area (minimised); nothing shown.
    pub fn present(
        &mut self,
        runner: &ComputeRunner,
        frame: &GpuImage,
        width: u32,
        height: u32,
    ) -> Result<bool, GpuError> {
        let mode = Self::mode_for(frame.format);
        if width == 0 || height == 0 {
            return Ok(false);
        }
        if width != self.extent.width
            || height != self.extent.height
            || (mode != self.mode && runner.hdr_colorspaces)
        {
            self.recreate(runner, width, height, mode)?;
        }
        for attempt in 0..2 {
            match unsafe { self.present_once(runner, frame) } {
                Ok(()) => {
                    self.frames += 1;
                    return Ok(true);
                }
                Err(GpuError::Vk(vk::Result::ERROR_OUT_OF_DATE_KHR)) if attempt == 0 => {
                    self.recreate(runner, width, height, mode)?;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(false)
    }

    unsafe fn present_once(
        &mut self,
        runner: &ComputeRunner,
        frame: &GpuImage,
    ) -> Result<(), GpuError> {
        let dev = &runner.device;
        unsafe {
            dev.wait_for_fences(&[self.fence], true, u64::MAX)?;
            let (index, _) = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.acquired,
                vk::Fence::null(),
            )?;
            dev.reset_fences(&[self.fence])?;
            let target = self.images[index as usize];
            dev.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
            dev.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let range = |aspect| {
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect)
                    .level_count(1)
                    .layer_count(1)
            };
            let to = |img, from, to, aspect, src: vk::AccessFlags2, dst: vk::AccessFlags2| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(src)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(dst)
                    .old_layout(from)
                    .new_layout(to)
                    .image(img)
                    .subresource_range(range(aspect))
            };
            let src_aspect = aspect_of(frame.format);
            let pre = [
                to(
                    target,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageAspectFlags::COLOR,
                    vk::AccessFlags2::NONE,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
                to(
                    frame.image,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::GENERAL,
                    src_aspect,
                    vk::AccessFlags2::MEMORY_WRITE,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
            ];
            dev.cmd_pipeline_barrier2(
                self.cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&pre),
            );
            let layers = |aspect| {
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(aspect)
                    .layer_count(1)
            };
            let blit = vk::ImageBlit::default()
                .src_subresource(layers(src_aspect))
                .src_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: frame.width as i32,
                        y: frame.height as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(layers(vk::ImageAspectFlags::COLOR))
                .dst_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: self.extent.width as i32,
                        y: self.extent.height as i32,
                        z: 1,
                    },
                ]);
            dev.cmd_blit_image(
                self.cmd,
                frame.image,
                vk::ImageLayout::GENERAL,
                target,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );
            let post = [to(
                target,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::ImageAspectFlags::COLOR,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::AccessFlags2::NONE,
            )];
            dev.cmd_pipeline_barrier2(
                self.cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&post),
            );
            dev.end_command_buffer(self.cmd)?;
            let waits = [self.acquired];
            let stages = [vk::PipelineStageFlags::TRANSFER];
            let cmds = [self.cmd];
            let signals = [self.blitted];
            dev.queue_submit(
                runner.queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&waits)
                    .wait_dst_stage_mask(&stages)
                    .command_buffers(&cmds)
                    .signal_semaphores(&signals)],
                self.fence,
            )?;
            let swapchains = [self.swapchain];
            let indices = [index];
            self.swapchain_loader.queue_present(
                runner.queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&signals)
                    .swapchains(&swapchains)
                    .image_indices(&indices),
            )?;
        }
        Ok(())
    }

    /// Releases the swapchain and surface.
    pub fn destroy(self, runner: &ComputeRunner) {
        // SAFETY: waits for the device before destroying.
        unsafe {
            let _ = runner.device.device_wait_idle();
            runner.device.destroy_semaphore(self.acquired, None);
            runner.device.destroy_semaphore(self.blitted, None);
            runner.device.destroy_fence(self.fence, None);
            runner
                .device
                .free_command_buffers(runner.cmd_pool, &[self.cmd]);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
        let _ = self.handle;
    }
}
