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

/// `vkCmdDraw` / `vkCmdDrawIndexed` arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Optional index buffer binding for indexed draws (`vkCmdBindIndexBuffer` + `vkCmdDrawIndexed`).
#[derive(Debug, Clone, Copy)]
pub struct IndexBufferBinding {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub index_type: vk::IndexType,
    pub vertex_offset: i32,
}

/// Full graphics pipeline state (rasterizer, depth/stencil, per-MRT blend, scissor).
#[derive(Debug, Clone)]
pub struct GraphicsPsoState {
    pub topology: vk::PrimitiveTopology,
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_bias_enable: bool,
    pub depth_bias_constant: f32,
    pub depth_bias_slope: f32,
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
    pub stencil_test_enable: bool,
    pub depth_format: Option<vk::Format>,
    pub blend_attachments: Vec<vk::PipelineColorBlendAttachmentState>,
    pub scissor: Option<vk::Rect2D>,
}

impl Default for GraphicsPsoState {
    fn default() -> Self {
        Self {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::CLOCKWISE,
            depth_bias_enable: false,
            depth_bias_constant: 0.0,
            depth_bias_slope: 0.0,
            depth_test_enable: false,
            depth_write_enable: false,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            stencil_test_enable: false,
            depth_format: None,
            blend_attachments: Vec::new(),
            scissor: None,
        }
    }
}

impl GraphicsPsoState {
    fn state_hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mix = |h: &mut u64, v: u64| {
            *h ^= v;
            *h = h.wrapping_mul(0x100_0000_01b3);
        };
        mix(&mut h, self.topology.as_raw() as u64);
        mix(&mut h, self.polygon_mode.as_raw() as u64);
        mix(&mut h, self.cull_mode.as_raw() as u64);
        mix(&mut h, self.front_face.as_raw() as u64);
        mix(&mut h, u64::from(self.depth_bias_enable));
        mix(&mut h, u64::from(self.depth_bias_constant.to_bits()));
        mix(&mut h, u64::from(self.depth_bias_slope.to_bits()));
        mix(&mut h, u64::from(self.depth_test_enable));
        mix(&mut h, u64::from(self.depth_write_enable));
        mix(&mut h, self.depth_compare_op.as_raw() as u64);
        mix(&mut h, u64::from(self.stencil_test_enable));
        mix(
            &mut h,
            self.depth_format.map_or(0, |f| f.as_raw() as u64 + 1),
        );
        for b in &self.blend_attachments {
            mix(&mut h, u64::from(b.blend_enable));
            mix(&mut h, b.src_color_blend_factor.as_raw() as u64);
            mix(&mut h, b.dst_color_blend_factor.as_raw() as u64);
            mix(&mut h, b.color_blend_op.as_raw() as u64);
            mix(&mut h, b.color_write_mask.as_raw() as u64);
        }
        h
    }
}

/// Graphics pipelines by (vertex SPIR-V, pixel SPIR-V, colour formats + state hash).
pub(crate) type GfxKey = (u64, u64, Vec<i32>);

impl ComputeRunner {
    fn gfx_pipeline_with_state(
        &self,
        vs: &[u32],
        ps: &[u32],
        formats: &[vk::Format],
        state: &GraphicsPsoState,
    ) -> Result<vk::Pipeline, GpuError> {
        let mut fmt_key: Vec<i32> = formats.iter().map(|f| f.as_raw()).collect();
        let sh = state.state_hash();
        fmt_key.push(sh as i32);
        fmt_key.push((sh >> 32) as i32);
        let key: GfxKey = (spirv_hash(vs), spirv_hash(ps), fmt_key);
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
            let assembly =
                vk::PipelineInputAssemblyStateCreateInfo::default().topology(state.topology);
            let viewport = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(state.polygon_mode)
                .cull_mode(state.cull_mode)
                .front_face(state.front_face)
                .depth_bias_enable(state.depth_bias_enable)
                .depth_bias_constant_factor(state.depth_bias_constant)
                .depth_bias_slope_factor(state.depth_bias_slope)
                .line_width(1.0);
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(state.depth_test_enable)
                .depth_write_enable(state.depth_write_enable)
                .depth_compare_op(state.depth_compare_op)
                .stencil_test_enable(state.stencil_test_enable);
            let ms = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> =
                if state.blend_attachments.len() == formats.len() {
                    state.blend_attachments.clone()
                } else {
                    formats
                        .iter()
                        .map(|_| {
                            vk::PipelineColorBlendAttachmentState::default()
                                .color_write_mask(vk::ColorComponentFlags::RGBA)
                        })
                        .collect()
                };
            let blend =
                vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
            let mut rendering =
                vk::PipelineRenderingCreateInfo::default().color_attachment_formats(formats);
            if let Some(dfmt) = state.depth_format {
                rendering = rendering.depth_attachment_format(dfmt);
            }
            let ci = [vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .depth_stencil_state(&depth_stencil)
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

    /// Draws `args` with `vs`/`ps` into `targets` using default PSO state.
    pub fn draw_spirv(
        &self,
        vs: &[u32],
        ps: &[u32],
        push: &[u8],
        args: DrawArgs,
        targets: &[&GpuImage],
        images: &ImageBindings,
    ) -> Result<(), GpuError> {
        self.draw_spirv_with_state(
            vs,
            ps,
            push,
            args,
            targets,
            None,
            None,
            images,
            &GraphicsPsoState::default(),
        )
    }

    /// Draws `args` with full [`GraphicsPsoState`], optional depth attachment, and
    /// optional index buffer binding.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_spirv_with_state(
        &self,
        vs: &[u32],
        ps: &[u32],
        push: &[u8],
        args: DrawArgs,
        targets: &[&GpuImage],
        depth_target: Option<&GpuImage>,
        index_buffer: Option<IndexBufferBinding>,
        images: &ImageBindings,
        state: &GraphicsPsoState,
    ) -> Result<(), GpuError> {
        let Some(first) = targets.first() else {
            return Ok(());
        };
        let formats: Vec<vk::Format> = targets.iter().map(|t| t.format).collect();
        let pipeline = self.gfx_pipeline_with_state(vs, ps, &formats, state)?;
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
            let depth_att = depth_target.map(|d| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(d.view)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
            });
            let area = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: w,
                    height: h,
                },
            };
            let mut render_info = vk::RenderingInfo::default()
                .render_area(area)
                .layer_count(1)
                .color_attachments(&attachments);
            if let Some(ref da) = depth_att {
                render_info = render_info.depth_attachment(da);
            }
            dev.cmd_begin_rendering(cmd, &render_info);
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
            dev.cmd_set_scissor(cmd, 0, &[state.scissor.unwrap_or(area)]);
            if let Some(ib) = index_buffer {
                dev.cmd_bind_index_buffer(cmd, ib.buffer, ib.offset, ib.index_type);
                dev.cmd_draw_indexed(
                    cmd,
                    args.vertex_count,
                    args.instance_count,
                    args.first_vertex,
                    ib.vertex_offset,
                    args.first_instance,
                );
            } else {
                dev.cmd_draw(
                    cmd,
                    args.vertex_count,
                    args.instance_count,
                    args.first_vertex,
                    args.first_instance,
                );
            }
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
