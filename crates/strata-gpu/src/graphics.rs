//! Draws with translated vertex and pixel shaders: dynamic rendering into
//! colour images, no vertex buffers (vertex shaders fetch their own data
//! through BDA / typed buffers), the global set 0 and push constants.
//!
//! The viewport is flipped (negative height) so clip space follows the
//! console's D3D convention, +y up.

use ash::vk;

use crate::context::GpuError;
use crate::images::{GpuImage, ImageBindings, barrier_image};
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
    pub stencil_front: vk::StencilOpState,
    pub stencil_back: vk::StencilOpState,
    /// `DepthClipEnable` off: depth is clamped instead of clipped.
    pub depth_clamp: bool,
    pub alpha_to_coverage: bool,
    pub depth_format: Option<vk::Format>,
    pub blend_attachments: Vec<vk::PipelineColorBlendAttachmentState>,
    /// Dynamic state (not part of the pipeline): the title's scissor.
    pub scissor: Option<vk::Rect2D>,
    /// Dynamic: the title's D3D viewport `[x, y, width, height, min_z,
    /// max_z]` (`None`: the whole target). Flipped to Vulkan's convention
    /// when recorded (D3D's +y is up in clip space).
    pub viewport: Option<[f32; 6]>,
    /// Dynamic: `OMSetStencilRef`.
    pub stencil_reference: u32,
    /// Dynamic: `OMSetBlendFactor`.
    pub blend_constants: [f32; 4],
    /// `OMSetDepthBounds`: enable depth bounds test when supported by hardware.
    pub depth_bounds_test_enable: bool,
    /// Dynamic: `OMSetDepthBounds` min/max.
    pub min_depth_bounds: f32,
    pub max_depth_bounds: f32,
    /// Tessellation patch control points (0 = no tessellation stage).
    pub patch_control_points: u32,
    /// Dynamic: `RSSetShadingRate` fragment size (`[1, 1]`, `[2, 2]`, etc.).
    pub shading_rate: [u32; 2],
    /// Dynamic: `RSSetShadingRate` combiner ops (`D3D12_SHADING_RATE_COMBINER`).
    pub shading_rate_combiners: [u32; 2],
}

/// Optional hardware query capture around a draw call (`VkQueryPool`).
#[derive(Debug, Default, Clone)]
pub struct DrawQueryResults {
    /// Set `true` to capture `VK_QUERY_TYPE_OCCLUSION` samples passed.
    pub capture_occlusion: bool,
    /// Set `true` to capture `VK_QUERY_TYPE_PIPELINE_STATISTICS`.
    pub capture_pipeline_stats: bool,
    /// Measure the submitted draw with Vulkan timestamps when available.
    pub capture_time: bool,
    pub gpu_ms: Option<f64>,
    /// Samples passing depth/stencil tests (populated after draw).
    pub samples_passed: u64,
    /// `D3D12_QUERY_DATA_PIPELINE_STATISTICS` (11 u64 counters in D3D12 order).
    pub pipeline_stats: [u64; 11],
}

/// Shader stages for a No-API graphics draw (classic vertex or mesh pipeline).
#[derive(Debug, Clone, Copy)]
pub enum GraphicsStages<'a> {
    Classic {
        vs: &'a [u32],
        hs: Option<&'a [u32]>,
        ds: Option<&'a [u32]>,
        gs: Option<&'a [u32]>,
        ps: &'a [u32],
    },
    Mesh {
        task: Option<&'a [u32]>,
        mesh: &'a [u32],
        ps: &'a [u32],
    },
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
            stencil_front: vk::StencilOpState::default()
                .compare_op(vk::CompareOp::ALWAYS)
                .compare_mask(0xff)
                .write_mask(0xff),
            stencil_back: vk::StencilOpState::default()
                .compare_op(vk::CompareOp::ALWAYS)
                .compare_mask(0xff)
                .write_mask(0xff),
            depth_clamp: false,
            alpha_to_coverage: false,
            depth_format: None,
            blend_attachments: Vec::new(),
            scissor: None,
            viewport: None,
            stencil_reference: 0,
            blend_constants: [0.0; 4],
            depth_bounds_test_enable: false,
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
            patch_control_points: 0,
            shading_rate: [1, 1],
            shading_rate_combiners: [0, 0],
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
        for st in [&self.stencil_front, &self.stencil_back] {
            mix(&mut h, st.fail_op.as_raw() as u64);
            mix(&mut h, st.pass_op.as_raw() as u64);
            mix(&mut h, st.depth_fail_op.as_raw() as u64);
            mix(&mut h, st.compare_op.as_raw() as u64);
            mix(&mut h, u64::from(st.compare_mask));
            mix(&mut h, u64::from(st.write_mask));
        }
        mix(&mut h, u64::from(self.depth_clamp));
        mix(&mut h, u64::from(self.alpha_to_coverage));
        mix(&mut h, u64::from(self.depth_bounds_test_enable));
        mix(&mut h, u64::from(self.patch_control_points));
        mix(
            &mut h,
            self.depth_format.map_or(0, |f| f.as_raw() as u64 + 1),
        );
        for b in &self.blend_attachments {
            mix(&mut h, u64::from(b.blend_enable));
            mix(&mut h, b.src_color_blend_factor.as_raw() as u64);
            mix(&mut h, b.dst_color_blend_factor.as_raw() as u64);
            mix(&mut h, b.color_blend_op.as_raw() as u64);
            mix(&mut h, b.src_alpha_blend_factor.as_raw() as u64);
            mix(&mut h, b.dst_alpha_blend_factor.as_raw() as u64);
            mix(&mut h, b.alpha_blend_op.as_raw() as u64);
            mix(&mut h, b.color_write_mask.as_raw() as u64);
        }
        h
    }
}

/// Graphics pipelines by (primary stage hash, fragment hash, colour formats + extra stage & state hash).
pub(crate) type GfxKey = (u64, u64, Vec<i32>);

fn vrs_combiner(op: u32) -> vk::FragmentShadingRateCombinerOpKHR {
    match op {
        1 => vk::FragmentShadingRateCombinerOpKHR::REPLACE,
        2 => vk::FragmentShadingRateCombinerOpKHR::MIN,
        3 => vk::FragmentShadingRateCombinerOpKHR::MAX,
        4 => vk::FragmentShadingRateCombinerOpKHR::MUL,
        _ => vk::FragmentShadingRateCombinerOpKHR::KEEP,
    }
}

impl ComputeRunner {
    pub(crate) fn gfx_pipeline_for_stages(
        &self,
        stages_in: GraphicsStages<'_>,
        formats: &[vk::Format],
        state: &GraphicsPsoState,
    ) -> Result<vk::Pipeline, GpuError> {
        let mut fmt_key: Vec<i32> = formats.iter().map(|f| f.as_raw()).collect();
        let sh = state.state_hash();
        fmt_key.push(sh as i32);
        fmt_key.push((sh >> 32) as i32);
        let (h0, h1, is_mesh, has_tess) = match stages_in {
            GraphicsStages::Classic { vs, hs, ds, gs, ps } => {
                let mut extra = 0u64;
                if let Some(s) = hs {
                    extra ^= spirv_hash(s).rotate_left(7);
                }
                if let Some(s) = ds {
                    extra ^= spirv_hash(s).rotate_left(19);
                }
                if let Some(s) = gs {
                    extra ^= spirv_hash(s).rotate_left(31);
                }
                fmt_key.push(extra as i32);
                fmt_key.push((extra >> 32) as i32);
                (
                    spirv_hash(vs),
                    spirv_hash(ps),
                    false,
                    hs.is_some() && ds.is_some(),
                )
            }
            GraphicsStages::Mesh { task, mesh, ps } => {
                let th = task.map_or(0u64, spirv_hash);
                fmt_key.push(1); // mesh tag
                fmt_key.push(th as i32);
                fmt_key.push((th >> 32) as i32);
                (spirv_hash(mesh), spirv_hash(ps), true, false)
            }
        };
        let key: GfxKey = (h0, h1, fmt_key);
        let mut table = self.gfx_pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&(_, p)) = table.get(&key) {
            return Ok(p);
        }
        match stages_in {
            GraphicsStages::Classic { vs, hs, ds, gs, ps } => {
                for words in [Some(vs), hs, ds, gs, Some(ps)].into_iter().flatten() {
                    self.check_shader_images(words)?;
                }
            }
            GraphicsStages::Mesh { task, mesh, ps } => {
                for words in [task, Some(mesh), Some(ps)].into_iter().flatten() {
                    self.check_shader_images(words)?;
                }
            }
        }
        let start = std::time::Instant::now();
        // SAFETY: plain object creation; modules are kept with the pipeline.
        unsafe {
            let mut modules: Vec<vk::ShaderModule> = Vec::new();
            let mut stage_infos: Vec<vk::PipelineShaderStageCreateInfo> = Vec::new();
            let name = c"main";
            let add_stage = |flag: vk::ShaderStageFlags,
                             code: &[u32],
                             modules: &mut Vec<vk::ShaderModule>,
                             stage_infos: &mut Vec<vk::PipelineShaderStageCreateInfo>|
             -> Result<(), GpuError> {
                let m = self.device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(code),
                    None,
                )?;
                modules.push(m);
                stage_infos.push(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(flag)
                        .module(m)
                        .name(name),
                );
                Ok(())
            };
            let init_res: Result<(), GpuError> = match stages_in {
                GraphicsStages::Classic { vs, hs, ds, gs, ps } => add_stage(
                    vk::ShaderStageFlags::VERTEX,
                    vs,
                    &mut modules,
                    &mut stage_infos,
                )
                .and_then(|_| {
                    if let (Some(h), Some(d)) = (hs, ds) {
                        add_stage(
                            vk::ShaderStageFlags::TESSELLATION_CONTROL,
                            h,
                            &mut modules,
                            &mut stage_infos,
                        )?;
                        add_stage(
                            vk::ShaderStageFlags::TESSELLATION_EVALUATION,
                            d,
                            &mut modules,
                            &mut stage_infos,
                        )?;
                    }
                    if let Some(g) = gs {
                        add_stage(
                            vk::ShaderStageFlags::GEOMETRY,
                            g,
                            &mut modules,
                            &mut stage_infos,
                        )?;
                    }
                    add_stage(
                        vk::ShaderStageFlags::FRAGMENT,
                        ps,
                        &mut modules,
                        &mut stage_infos,
                    )
                }),
                GraphicsStages::Mesh { task, mesh, ps } => {
                    let r = if let Some(t) = task {
                        add_stage(
                            vk::ShaderStageFlags::TASK_EXT,
                            t,
                            &mut modules,
                            &mut stage_infos,
                        )
                    } else {
                        Ok(())
                    };
                    r.and_then(|_| {
                        add_stage(
                            vk::ShaderStageFlags::MESH_EXT,
                            mesh,
                            &mut modules,
                            &mut stage_infos,
                        )
                    })
                    .and_then(|_| {
                        add_stage(
                            vk::ShaderStageFlags::FRAGMENT,
                            ps,
                            &mut modules,
                            &mut stage_infos,
                        )
                    })
                }
            };
            if let Err(e) = init_res {
                for m in modules {
                    self.device.destroy_shader_module(m, None);
                }
                return Err(e);
            }
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let topo = if has_tess {
                vk::PrimitiveTopology::PATCH_LIST
            } else {
                state.topology
            };
            let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topo);
            let tess_state = vk::PipelineTessellationStateCreateInfo::default()
                .patch_control_points(state.patch_control_points.max(1));
            let viewport = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(state.depth_clamp && self.depth_clamp)
                .polygon_mode(state.polygon_mode)
                .cull_mode(state.cull_mode)
                .front_face(state.front_face)
                .depth_bias_enable(state.depth_bias_enable)
                .depth_bias_constant_factor(state.depth_bias_constant)
                .depth_bias_slope_factor(state.depth_bias_slope)
                .line_width(1.0);
            let use_depth_bounds = state.depth_bounds_test_enable && self.depth_bounds;
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(state.depth_test_enable)
                .depth_write_enable(state.depth_write_enable)
                .depth_compare_op(state.depth_compare_op)
                .depth_bounds_test_enable(use_depth_bounds)
                .stencil_test_enable(state.stencil_test_enable)
                .front(state.stencil_front)
                .back(state.stencil_back);
            let ms = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                .alpha_to_coverage_enable(state.alpha_to_coverage);
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
            let mut dynamic_states = vec![
                vk::DynamicState::VIEWPORT,
                vk::DynamicState::SCISSOR,
                vk::DynamicState::STENCIL_REFERENCE,
                vk::DynamicState::BLEND_CONSTANTS,
            ];
            if use_depth_bounds {
                dynamic_states.push(vk::DynamicState::DEPTH_BOUNDS);
            }
            if self.fragment_shading_rate {
                dynamic_states.push(vk::DynamicState::FRAGMENT_SHADING_RATE_KHR);
            }
            let dynamic =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
            let mut rendering =
                vk::PipelineRenderingCreateInfo::default().color_attachment_formats(formats);
            if let Some(dfmt) = state.depth_format {
                rendering = rendering.depth_attachment_format(dfmt);
                if crate::images::aspect_of(dfmt).contains(vk::ImageAspectFlags::STENCIL) {
                    rendering = rendering.stencil_attachment_format(dfmt);
                }
            }
            let mut gpci = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stage_infos)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .depth_stencil_state(&depth_stencil)
                .multisample_state(&ms)
                .color_blend_state(&blend)
                .dynamic_state(&dynamic)
                .layout(self.pipe_layout)
                .push_next(&mut rendering);
            if !is_mesh {
                gpci = gpci
                    .vertex_input_state(&vertex_input)
                    .input_assembly_state(&assembly);
                if has_tess {
                    gpci = gpci.tessellation_state(&tess_state);
                }
            }
            let ci = [gpci];
            let _cache = self.pipeline_cache_lock.lock().unwrap();
            let pipeline =
                match self
                    .device
                    .create_graphics_pipelines(self.pipeline_cache(), &ci, None)
                {
                    Ok(p) => p[0],
                    Err((_, e)) => {
                        for m in modules {
                            self.device.destroy_shader_module(m, None);
                        }
                        return Err(e.into());
                    }
                };
            let key_words: Vec<_> = key.2.iter().map(|&word| word as u32).collect();
            self.record_pipeline(
                format!(
                    "gs-{:016x}-{:016x}-{:016x}",
                    key.0,
                    key.1,
                    spirv_hash(&key_words)
                ),
                true,
                start.elapsed(),
                || crate::pipeline_usage::Recipe::graphics(stages_in, formats, state),
            );
            table.insert(key, (modules, pipeline));
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
        self.draw_stages_with_state(
            GraphicsStages::Classic {
                vs,
                hs: None,
                ds: None,
                gs: None,
                ps,
            },
            push,
            Some(args),
            None,
            targets,
            depth_target,
            index_buffer,
            images,
            state,
            None,
        )
    }

    /// Draws a mesh/amplification pipeline (`VK_EXT_mesh_shader`) with No-API dynamic rendering.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_mesh_with_state(
        &self,
        task: Option<&[u32]>,
        mesh: &[u32],
        ps: &[u32],
        push: &[u8],
        groups: [u32; 3],
        targets: &[&GpuImage],
        depth_target: Option<&GpuImage>,
        images: &ImageBindings,
        state: &GraphicsPsoState,
        query: Option<&mut DrawQueryResults>,
    ) -> Result<(), GpuError> {
        if !self.mesh_shader || self.ext_mesh_shader.is_none() {
            return Err(GpuError::Vk(vk::Result::ERROR_EXTENSION_NOT_PRESENT));
        }
        self.draw_stages_with_state(
            GraphicsStages::Mesh { task, mesh, ps },
            push,
            None,
            Some(groups),
            targets,
            depth_target,
            None,
            images,
            state,
            query,
        )
    }

    /// Unified No-API graphics draw supporting classic (VS + optional HS/DS/GS + PS)
    /// and mesh (optional AS/Task + MS + PS) pipelines, dynamic depth bounds, VRS,
    /// and hardware `VkQueryPool` occlusion + pipeline-statistics capture.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_stages_with_state(
        &self,
        stages: GraphicsStages<'_>,
        push: &[u8],
        args: Option<DrawArgs>,
        mesh_groups: Option<[u32; 3]>,
        targets: &[&GpuImage],
        depth_target: Option<&GpuImage>,
        index_buffer: Option<IndexBufferBinding>,
        images: &ImageBindings,
        state: &GraphicsPsoState,
        query: Option<&mut DrawQueryResults>,
    ) -> Result<(), GpuError> {
        // Depth-only passes (shadow maps, depth pre-passes) have no colour target.
        let Some(first) = targets.first().copied().or(depth_target) else {
            return Ok(());
        };
        let formats: Vec<vk::Format> = targets.iter().map(|t| t.format).collect();
        let mut state_with_depth;
        let state = match depth_target {
            Some(d) if state.depth_format != Some(d.format) => {
                state_with_depth = state.clone();
                state_with_depth.depth_format = Some(d.format);
                &state_with_depth
            }
            None if state.depth_format.is_some() => {
                state_with_depth = state.clone();
                state_with_depth.depth_format = None;
                state_with_depth.depth_test_enable = false;
                state_with_depth.depth_write_enable = false;
                state_with_depth.stencil_test_enable = false;
                &state_with_depth
            }
            _ => state,
        };
        let pipeline = self.gfx_pipeline_for_stages(stages, &formats, state)?;
        // SAFETY: freed after the submission is waited on.
        let set = unsafe { self.image_set(images)? };
        let (w, h) = (first.width, first.height);

        let want_occ = query.as_ref().is_some_and(|q| q.capture_occlusion);
        let want_stats = query
            .as_ref()
            .is_some_and(|q| q.capture_pipeline_stats && self.pipeline_statistics_query);
        let occ_pool = if want_occ {
            Some(unsafe {
                self.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::OCCLUSION)
                        .query_count(1),
                    None,
                )?
            })
        } else {
            None
        };
        let stats_flags = vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_VERTICES
            | vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_PRIMITIVES
            | vk::QueryPipelineStatisticFlags::VERTEX_SHADER_INVOCATIONS
            | vk::QueryPipelineStatisticFlags::GEOMETRY_SHADER_INVOCATIONS
            | vk::QueryPipelineStatisticFlags::GEOMETRY_SHADER_PRIMITIVES
            | vk::QueryPipelineStatisticFlags::CLIPPING_INVOCATIONS
            | vk::QueryPipelineStatisticFlags::CLIPPING_PRIMITIVES
            | vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS
            | vk::QueryPipelineStatisticFlags::TESSELLATION_CONTROL_SHADER_PATCHES
            | vk::QueryPipelineStatisticFlags::TESSELLATION_EVALUATION_SHADER_INVOCATIONS
            | vk::QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS;
        let stats_pool = if want_stats {
            Some(unsafe {
                self.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::PIPELINE_STATISTICS)
                        .pipeline_statistics(stats_flags)
                        .query_count(1),
                    None,
                )?
            })
        } else {
            None
        };
        let time_pool =
            if self.timestamp_bits != 0 && query.as_ref().is_some_and(|q| q.capture_time) {
                unsafe {
                    self.device.create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                }
                .ok()
            } else {
                None
            };

        let result = self.one_shot(|dev, cmd| unsafe {
            if let Some(p) = time_pool {
                dev.cmd_reset_query_pool(cmd, p, 0, 2);
                dev.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, p, 0);
            }
            if let Some(p) = occ_pool {
                dev.cmd_reset_query_pool(cmd, p, 0, 1);
            }
            if let Some(p) = stats_pool {
                dev.cmd_reset_query_pool(cmd, p, 0, 1);
            }
            for t in targets.iter().copied().chain(depth_target) {
                barrier_image(
                    dev,
                    cmd,
                    t,
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
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
            });
            let has_stencil = depth_target.is_some_and(|d| {
                crate::images::aspect_of(d.format).contains(vk::ImageAspectFlags::STENCIL)
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
                if has_stencil {
                    render_info = render_info.stencil_attachment(da);
                }
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
            let [vx, vy, vw, vh, zmin, zmax] = state
                .viewport
                .unwrap_or([0.0, 0.0, w as f32, h as f32, 0.0, 1.0]);
            let vp = vk::Viewport {
                x: vx,
                y: vy + vh,
                width: vw,
                height: -vh,
                min_depth: zmin,
                max_depth: zmax,
            };
            dev.cmd_set_viewport(cmd, 0, &[vp]);
            dev.cmd_set_scissor(cmd, 0, &[state.scissor.unwrap_or(area)]);
            dev.cmd_set_stencil_reference(
                cmd,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                state.stencil_reference,
            );
            dev.cmd_set_blend_constants(cmd, &state.blend_constants);
            if state.depth_bounds_test_enable && self.depth_bounds {
                dev.cmd_set_depth_bounds(cmd, state.min_depth_bounds, state.max_depth_bounds);
            }
            if let Some(ref vrs) = self.khr_fragment_shading_rate {
                let frag_size = vk::Extent2D {
                    width: state.shading_rate[0].max(1),
                    height: state.shading_rate[1].max(1),
                };
                let combiners = [
                    vrs_combiner(state.shading_rate_combiners[0]),
                    vrs_combiner(state.shading_rate_combiners[1]),
                ];
                (vrs.fp().cmd_set_fragment_shading_rate_khr)(cmd, &frag_size, &combiners);
            }
            if let Some(p) = occ_pool {
                dev.cmd_begin_query(cmd, p, 0, vk::QueryControlFlags::PRECISE);
            }
            if let Some(p) = stats_pool {
                dev.cmd_begin_query(cmd, p, 0, vk::QueryControlFlags::empty());
            }
            if let (Some(groups), Some(mesh_ext)) = (mesh_groups, &self.ext_mesh_shader) {
                mesh_ext.cmd_draw_mesh_tasks(cmd, groups[0], groups[1], groups[2]);
            } else if let Some(args) = args {
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
            }
            if let Some(p) = stats_pool {
                dev.cmd_end_query(cmd, p, 0);
            }
            if let Some(p) = occ_pool {
                dev.cmd_end_query(cmd, p, 0);
            }
            dev.cmd_end_rendering(cmd);
            if let Some(p) = time_pool {
                dev.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::BOTTOM_OF_PIPE, p, 1);
            }
            for t in targets.iter().copied().chain(depth_target) {
                barrier_image(
                    dev,
                    cmd,
                    t,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::GENERAL,
                );
            }
        });
        if result.is_ok()
            && let Some(q) = query
        {
            if let Some(p) = time_pool {
                let mut ticks = [0u64; 2];
                if unsafe {
                    self.device.get_query_pool_results(
                        p,
                        0,
                        &mut ticks,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                }
                .is_ok()
                {
                    let mask = 1u64
                        .checked_shl(self.timestamp_bits)
                        .map_or(u64::MAX, |n| n - 1);
                    q.gpu_ms = Some(
                        (ticks[1].wrapping_sub(ticks[0]) & mask) as f64
                            * f64::from(self.timestamp_period)
                            / 1_000_000.0,
                    );
                }
            }
            if let Some(p) = occ_pool {
                let mut samples = [0u64; 1];
                let _ = unsafe {
                    self.device.get_query_pool_results(
                        p,
                        0,
                        &mut samples,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                };
                q.samples_passed = q.samples_passed.saturating_add(samples[0]);
            }
            if let Some(p) = stats_pool {
                let mut stats = [0u64; 11];
                let _ = unsafe {
                    self.device.get_query_pool_results(
                        p,
                        0,
                        &mut stats,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                };
                for (dst, src) in q.pipeline_stats.iter_mut().zip(stats.iter()) {
                    *dst = dst.saturating_add(*src);
                }
            }
        }
        unsafe {
            if let Some(p) = time_pool {
                self.device.destroy_query_pool(p, None);
            }
            if let Some(p) = occ_pool {
                self.device.destroy_query_pool(p, None);
            }
            if let Some(p) = stats_pool {
                self.device.destroy_query_pool(p, None);
            }
            self.free_image_set(set);
        }
        result
    }
}
