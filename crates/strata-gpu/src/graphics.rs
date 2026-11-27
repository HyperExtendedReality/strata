//! Draws with translated vertex and pixel shaders: dynamic rendering into
//! colour images, no vertex buffers (vertex shaders fetch their own data
//! through BDA / typed buffers), the global set 0 and push constants.
//!
//! The viewport is flipped (negative height) so clip space follows the
//! console's D3D convention, +y up.

use ash::vk;

use crate::context::GpuError;
use crate::images::{GpuImage, ImageBindings, barrier};
use crate::runner::{ComputeRunner, PUSH_STAGES, spirv_hash};

/// `vkCmdDraw` arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Graphics pipelines by (vertex SPIR-V, pixel SPIR-V, colour formats).
pub(crate) type GfxKey = (u64, u64, Vec<i32>);

impl ComputeRunner {
    fn gfx_pipeline(
        &self,
        vs: &[u32],
        ps: &[u32],
        formats: &[vk::Format],
    ) -> Result<vk::Pipeline, GpuError> {
        let key: GfxKey = (
            spirv_hash(vs),
            spirv_hash(ps),
            formats.iter().map(|f| f.as_raw()).collect(),
        );
        let mut table = self.gfx_pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&(_, _, p)) = table.get(&key) {
            return Ok(p);
        }
        // SAFETY: plain object creation; modules are kept with the pipeline.
        unsafe {
            let vsm = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(vs), None)?;
            let psm = match self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(ps), None)
            {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_shader_module(vsm, None);
                    return Err(e.into());
                }
            };
            let name = c"main";
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vsm)
                    .name(name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(psm)
                    .name(name),
            ];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::CLOCKWISE)
                .line_width(1.0);
            let ms = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = formats
                .iter()
                .map(|_| {
                    vk::PipelineColorBlendAttachmentState::default()
                        .color_write_mask(vk::ColorComponentFlags::RGBA)
                })
                .collect();
            let blend =
                vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
            let mut rendering =
                vk::PipelineRenderingCreateInfo::default().color_attachment_formats(formats);
            let ci = [vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&ms)
                .color_blend_state(&blend)
                .dynamic_state(&dynamic)
                .layout(self.pipe_layout)
                .push_next(&mut rendering)];
            let pipeline =
                match self
                    .device
                    .create_graphics_pipelines(self.pipeline_cache(), &ci, None)
                {
                    Ok(p) => p[0],
                    Err((_, e)) => {
                        self.device.destroy_shader_module(vsm, None);
                        self.device.destroy_shader_module(psm, None);
                        return Err(e.into());
                    }
                };
            table.insert(key, (vsm, psm, pipeline));
            Ok(pipeline)
        }
    }

    /// Draws `args` with `vs`/`ps` into `targets` (colour attachments 0..n,
    /// loaded and stored so earlier contents stay), with `images` bound.
    pub fn draw_spirv(
        &self,
        vs: &[u32],
        ps: &[u32],
        push: &[u8],
        args: DrawArgs,
        targets: &[&GpuImage],
        images: &ImageBindings,
    ) -> Result<(), GpuError> {
        let Some(first) = targets.first() else {
            return Ok(());
        };
        let formats: Vec<vk::Format> = targets.iter().map(|t| t.format).collect();
        let pipeline = self.gfx_pipeline(vs, ps, &formats)?;
        // SAFETY: freed after the submission is waited on.
        let set = unsafe { self.image_set(images)? };
        let (w, h) = (first.width, first.height);
        let result = self.one_shot(|dev, cmd| unsafe {
            for t in targets {
                barrier(
                    dev,
                    cmd,
                    t.image,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::GENERAL,
                );
            }
            let attachments: Vec<vk::RenderingAttachmentInfo> = targets
                .iter()
                .map(|t| {
                    vk::RenderingAttachmentInfo::default()
                        .image_view(t.view)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .load_op(vk::AttachmentLoadOp::LOAD)
                        .store_op(vk::AttachmentStoreOp::STORE)
                })
                .collect();
            let area = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: w,
                    height: h,
                },
            };
            dev.cmd_begin_rendering(
                cmd,
                &vk::RenderingInfo::default()
                    .render_area(area)
                    .layer_count(1)
                    .color_attachments(&attachments),
            );
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipe_layout,
                0,
                &[set],
                &[],
            );
            if !push.is_empty() {
                dev.cmd_push_constants(cmd, self.pipe_layout, PUSH_STAGES, 0, push);
            }
            let vp = vk::Viewport {
                x: 0.0,
                y: h as f32,
                width: w as f32,
                height: -(h as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            };
            dev.cmd_set_viewport(cmd, 0, &[vp]);
            dev.cmd_set_scissor(cmd, 0, &[area]);
            dev.cmd_draw(
                cmd,
                args.vertex_count,
                args.instance_count,
                args.first_vertex,
                args.first_instance,
            );
            dev.cmd_end_rendering(cmd);
            for t in targets {
                barrier(
                    dev,
                    cmd,
                    t.image,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::GENERAL,
                );
            }
        });
        // SAFETY: the submission finished (one_shot waits).
        unsafe { self.free_image_set(set) };
        result
    }
}
