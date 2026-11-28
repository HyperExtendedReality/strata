//! Live No-API Vulkan 1.3+ BDA Compute Runner (`ash`).
//!
//! Creates a minimal Vulkan 1.3 compute device with `bufferDeviceAddress` enabled,
//! allocates host-visible BDA buffers, pushes 64-bit GPU pointers via Push Constants,
//! dispatches arbitrary SPIR-V 1.6 compute shaders (both direct-emitted RDNA2->SPIR-V
//! and build-time Slang shaders), and reads back results on real GPU hardware.

use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;
use std::sync::Mutex;

use ash::vk;

use crate::context::GpuError;

pub struct BdaBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub bda: u64,
    pub size: u64,
    ptr: *mut u8,
    /// True if backed by an imported CPU host pointer (`VK_EXT_external_memory_host`)
    /// rather than `vkMapMemory` (must NOT call `vkUnmapMemory` on drop).
    imported_host: bool,
}

impl BdaBuffer {
    pub unsafe fn as_slice<T: Copy>(&self, count: usize) -> &[T] {
        debug_assert!(count * std::mem::size_of::<T>() <= self.size as usize);
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, count) }
    }

    pub unsafe fn as_mut_slice<T: Copy>(&mut self, count: usize) -> &mut [T] {
        debug_assert!(count * std::mem::size_of::<T>() <= self.size as usize);
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, count) }
    }
}

/// Stages that see the 128 bytes of push constants (user SGPRs).
pub use crate::layout::PUSH_STAGES;

pub struct ComputeRunner {
    pub(crate) _entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) pdev: vk::PhysicalDevice,
    memory_budget: bool,
    pub(crate) device: ash::Device,
    ext_host_mem: Option<ash::ext::external_memory_host::Device>,
    pub min_host_pointer_alignment: u64,
    pub(crate) queue: vk::Queue,
    pub(crate) queue_lock: Mutex<()>,
    pub(crate) cmd_pool: vk::CommandPool,
    pub(crate) cmd_pool_lock: Mutex<()>,
    idle_fences: Mutex<Vec<vk::Fence>>,
    idle_recording: Mutex<Vec<(vk::CommandPool, vk::CommandBuffer)>>,
    /// Push constants + set 0 (images); shared by every translated pipeline.
    pub(crate) pipe_layout: vk::PipelineLayout,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    desc_pool: Mutex<vk::DescriptorPool>,
    /// 1x1 image and sampler bound to every slot a dispatch leaves empty.
    pub(crate) default_image: Option<crate::images::GpuImage>,
    /// 1-texel 3D and 1-cube (storage-capable) defaults for empty slots of
    /// the 3D, cube and array classes.
    pub(crate) default_3d: Option<crate::images::GpuImage>,
    pub(crate) default_cube: Option<crate::images::GpuImage>,
    default_cube_view: vk::ImageView,
    default_sampler: vk::Sampler,
    pub(crate) sampler_mirror_clamp_to_edge: bool,
    /// `depthClamp` is enabled (D3D12 `DepthClipEnable = FALSE`).
    pub(crate) depth_clamp: bool,
    /// `depthBounds` is enabled (D3D12 `OMSetDepthBounds`).
    pub(crate) depth_bounds: bool,
    /// `tessellationShader` is enabled (Hull / Domain shaders).
    pub tessellation_shader: bool,
    /// `geometryShader` is enabled (Geometry shaders).
    pub geometry_shader: bool,
    /// `pipelineStatisticsQuery` is enabled (`VkQueryPool` pipeline stats).
    pub pipeline_statistics_query: bool,
    /// `VK_EXT_mesh_shader` device loader + feature flags.
    pub(crate) ext_mesh_shader: Option<ash::ext::mesh_shader::Device>,
    pub mesh_shader: bool,
    pub task_shader: bool,
    /// `VK_KHR_fragment_shading_rate` device loader + feature flag.
    pub(crate) khr_fragment_shading_rate: Option<ash::khr::fragment_shading_rate::Device>,
    pub fragment_shading_rate: bool,
    /// `VK_EXT_transform_feedback` device loader + feature flag.
    #[allow(dead_code)]
    pub(crate) ext_transform_feedback: Option<ash::ext::transform_feedback::Device>,
    pub transform_feedback: bool,
    /// `VK_KHR_ray_tracing_pipeline` device loader + feature flag.
    #[allow(dead_code)]
    pub(crate) khr_ray_tracing_pipeline: Option<ash::khr::ray_tracing_pipeline::Device>,
    pub ray_tracing_pipeline: bool,
    pub(crate) acceleration_structure: Option<ash::khr::acceleration_structure::Device>,
    pub(crate) acceleration_scratch_alignment: u64,
    pub ray_query: bool,
    pub sparse_aliased_images: bool,
    /// Surfaces and swapchains are available (presenting to a window).
    pub wsi: bool,
    /// HDR swapchain colour spaces (`VK_EXT_swapchain_colorspace`).
    pub hdr_colorspaces: bool,
    pub(crate) queue_family: u32,
    /// `VK_KHR_shader_quad_control` is enabled: pixel shaders may require full quads.
    pub quad_control: bool,
    pub device_name: String,
    pub driver_version: u32,
    pub image_slots: u32,
    pub(crate) timestamp_period: f32,
    pub(crate) timestamp_bits: u32,
    pub subgroup_size: u32,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub native_wave64: bool,
    pub native_wave64_stages: vk::ShaderStageFlags,
    pub subgroup_size_control: bool,
    /// `maxComputeWorkGroupCount`: dispatches beyond it are split.
    pub max_workgroup_count: [u32; 3],
    pipeline_cache: vk::PipelineCache,
    pub(crate) pipeline_cache_lock: Mutex<()>,
    pub(crate) pipeline_usage: Mutex<crate::pipeline_usage::Usage>,
    /// `(FNV-1a of the SPIR-V, required subgroup size or 0)` -> pipeline.
    pipelines: Mutex<HashMap<(u64, u32), (vk::ShaderModule, vk::Pipeline)>>,
    pub(crate) gfx_pipelines:
        Mutex<HashMap<crate::graphics::GfxKey, (Vec<vk::ShaderModule>, vk::Pipeline)>>,
    /// BMA traversal-shortcut table (created on first instrumented pipeline).
    pub(crate) bma: Mutex<Option<crate::bma::BmaTable>>,
    /// Dedicated transfer queue for asynchronous uploads, if the device has one.
    pub(crate) transfer: Option<crate::images::TransferQueue>,
}

/// FNV-1a over SPIR-V words: the pipeline table's key.
pub fn spirv_hash(words: &[u32]) -> u64 {
    words
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        })
}

impl ComputeRunner {
    pub fn new() -> Result<Self, GpuError> {
        unsafe {
            let entry = ash::Entry::load()?;
            let app_name = CString::new("StrataRunner").unwrap();
            let app_info = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .api_version(vk::API_VERSION_1_3);
            // Window-system integration when the loader offers it (presenting
            // is optional: compute-only use works without).
            let avail_inst = entry.enumerate_instance_extension_properties(None)?;
            let has_inst = |n: &std::ffi::CStr| {
                avail_inst
                    .iter()
                    .any(|e| std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) == n)
            };
            let mut inst_exts: Vec<*const i8> = Vec::new();
            #[cfg(windows)]
            let wsi_inst =
                has_inst(ash::khr::surface::NAME) && has_inst(ash::khr::win32_surface::NAME);
            #[cfg(not(windows))]
            let wsi_inst = false;
            if wsi_inst {
                inst_exts.push(ash::khr::surface::NAME.as_ptr());
                #[cfg(windows)]
                inst_exts.push(ash::khr::win32_surface::NAME.as_ptr());
            }
            let hdr_colorspaces = wsi_inst && has_inst(ash::ext::swapchain_colorspace::NAME);
            if hdr_colorspaces {
                inst_exts.push(ash::ext::swapchain_colorspace::NAME.as_ptr());
            }
            let inst_ci = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&inst_exts);
            let instance = entry.create_instance(&inst_ci, None)?;

            let pdev = instance
                .enumerate_physical_devices()?
                .into_iter()
                .max_by_key(
                    |&d| match instance.get_physical_device_properties(d).device_type {
                        vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                        vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                        _ => 1,
                    },
                )
                .ok_or(GpuError::NoDevice)?;

            let props = instance.get_physical_device_properties(pdev);
            let device_name = std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned();

            let qprops = instance.get_physical_device_queue_family_properties(pdev);
            let qfamily = qprops
                .iter()
                .position(|q| {
                    q.queue_flags.contains(
                        vk::QueueFlags::GRAPHICS
                            | vk::QueueFlags::COMPUTE
                            | vk::QueueFlags::SPARSE_BINDING,
                    )
                })
                .or_else(|| {
                    qprops.iter().position(|q| {
                        q.queue_flags
                            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                    })
                })
                .or_else(|| {
                    qprops.iter().position(|q| {
                        q.queue_flags
                            .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::SPARSE_BINDING)
                    })
                })
                .or_else(|| {
                    qprops
                        .iter()
                        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                })
                .ok_or(GpuError::NoDevice)? as u32;

            let prio = [1.0f32];
            // A dedicated transfer family (copy engine) for asynchronous
            // texture uploads, when its copy granularity is one texel.
            // `STRATA_ASYNC_TEXTURE_UPLOAD=0` keeps everything on one queue.
            let async_off =
                std::env::var("STRATA_ASYNC_TEXTURE_UPLOAD").is_ok_and(|v| v.trim() == "0");
            let tfamily = (!async_off)
                .then(|| {
                    qprops.iter().position(|q| {
                        q.queue_flags.contains(vk::QueueFlags::TRANSFER)
                            && !q
                                .queue_flags
                                .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                            && q.min_image_transfer_granularity.width == 1
                            && q.min_image_transfer_granularity.height == 1
                            && q.min_image_transfer_granularity.depth == 1
                    })
                })
                .flatten()
                .map(|f| f as u32)
                .filter(|&f| f != qfamily);
            let mut q_ci = vec![
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(qfamily)
                    .queue_priorities(&prio),
            ];
            if let Some(tf) = tfamily {
                q_ci.push(
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(tf)
                        .queue_priorities(&prio),
                );
            }

            let exts = instance.enumerate_device_extension_properties(pdev)?;
            let has_ext = |name: &std::ffi::CStr| {
                exts.iter()
                    .any(|e| std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) == name)
            };
            let has_ext_host = has_ext(ash::ext::external_memory_host::NAME);
            let memory_budget = has_ext(ash::ext::memory_budget::NAME);
            let has_mesh_ext = has_ext(ash::ext::mesh_shader::NAME);
            let has_vrs_ext = has_ext(ash::khr::fragment_shading_rate::NAME);
            let has_xfb_ext = has_ext(ash::ext::transform_feedback::NAME);
            let has_as_ext = has_ext(ash::khr::acceleration_structure::NAME)
                && has_ext(ash::khr::deferred_host_operations::NAME);
            let has_rt_ext = has_as_ext && has_ext(ash::khr::ray_tracing_pipeline::NAME);
            let has_ray_query_ext = has_as_ext && has_ext(ash::khr::ray_query::NAME);

            let mut host_mem_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
            let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
            let mut subgroup_size_props =
                vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut subgroup_props)
                .push_next(&mut subgroup_size_props);
            let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
            if has_as_ext {
                props2 = props2.push_next(&mut as_props);
            }
            if has_ext_host {
                props2 = props2.push_next(&mut host_mem_props);
            }
            instance.get_physical_device_properties2(pdev, &mut props2);
            let min_host_pointer_alignment = host_mem_props.min_imported_host_pointer_alignment;

            let mut avail_f12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut avail_f13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut avail_quad = vk::PhysicalDeviceShaderQuadControlFeaturesKHR::default();
            let mut avail_mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
            let mut avail_vrs = vk::PhysicalDeviceFragmentShadingRateFeaturesKHR::default();
            let mut avail_xfb = vk::PhysicalDeviceTransformFeedbackFeaturesEXT::default();
            let mut avail_rt = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
            let mut avail_as = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
            let mut avail_ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
            let has_quad_ext = has_ext(ash::khr::shader_quad_control::NAME);
            let mut avail_f11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut avail_feats2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut avail_f11)
                .push_next(&mut avail_f12)
                .push_next(&mut avail_f13);
            if has_quad_ext {
                avail_feats2 = avail_feats2.push_next(&mut avail_quad);
            }
            if has_mesh_ext {
                avail_feats2 = avail_feats2.push_next(&mut avail_mesh);
            }
            if has_vrs_ext {
                avail_feats2 = avail_feats2.push_next(&mut avail_vrs);
            }
            if has_xfb_ext {
                avail_feats2 = avail_feats2.push_next(&mut avail_xfb);
            }
            if has_rt_ext {
                avail_feats2 = avail_feats2.push_next(&mut avail_rt);
            }
            if has_as_ext {
                avail_feats2 = avail_feats2.push_next(&mut avail_as);
            }
            if has_ray_query_ext {
                avail_feats2 = avail_feats2.push_next(&mut avail_ray_query);
            }
            instance.get_physical_device_features2(pdev, &mut avail_feats2);
            let sampler_mirror_clamp_to_edge = avail_f12.sampler_mirror_clamp_to_edge == vk::TRUE;
            // Quad control rides on maximal reconvergence in the spec.
            let has_reconv_ext = has_ext(ash::khr::shader_maximal_reconvergence::NAME);
            let quad_control =
                has_quad_ext && has_reconv_ext && avail_quad.shader_quad_control == vk::TRUE;
            let mesh_shader = has_mesh_ext && avail_mesh.mesh_shader == vk::TRUE;
            let task_shader = mesh_shader && avail_mesh.task_shader == vk::TRUE;
            let fragment_shading_rate =
                has_vrs_ext && avail_vrs.pipeline_fragment_shading_rate == vk::TRUE;
            let transform_feedback = has_xfb_ext && avail_xfb.transform_feedback == vk::TRUE;
            let can_build_as = avail_as.acceleration_structure == vk::TRUE;
            let ray_tracing_pipeline = can_build_as && avail_rt.ray_tracing_pipeline == vk::TRUE;
            let ray_query = can_build_as && avail_ray_query.ray_query == vk::TRUE;
            let use_as = ray_tracing_pipeline || ray_query;
            let subgroup_size_control_enabled = avail_f13.subgroup_size_control == vk::TRUE;
            let subgroup_size_control = subgroup_size_control_enabled
                && subgroup_size_props
                    .required_subgroup_size_stages
                    .contains(vk::ShaderStageFlags::COMPUTE);

            let mut enabled_exts: Vec<*const i8> = Vec::new();
            if memory_budget {
                enabled_exts.push(ash::ext::memory_budget::NAME.as_ptr());
            }
            if has_ext_host {
                enabled_exts.push(ash::ext::external_memory_host::NAME.as_ptr());
            }
            let wsi = wsi_inst && has_ext(ash::khr::swapchain::NAME);
            if wsi {
                enabled_exts.push(ash::khr::swapchain::NAME.as_ptr());
            }
            let mut quad_feats =
                vk::PhysicalDeviceShaderQuadControlFeaturesKHR::default().shader_quad_control(true);
            let mut reconv_feats =
                vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default()
                    .shader_maximal_reconvergence(true);
            if quad_control {
                enabled_exts.push(ash::khr::shader_quad_control::NAME.as_ptr());
                enabled_exts.push(ash::khr::shader_maximal_reconvergence::NAME.as_ptr());
            }
            let mut mesh_feats = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
                .mesh_shader(mesh_shader)
                .task_shader(task_shader);
            if mesh_shader {
                enabled_exts.push(ash::ext::mesh_shader::NAME.as_ptr());
            }
            let mut vrs_feats = vk::PhysicalDeviceFragmentShadingRateFeaturesKHR::default()
                .pipeline_fragment_shading_rate(true)
                .attachment_fragment_shading_rate(
                    avail_vrs.attachment_fragment_shading_rate == vk::TRUE,
                );
            if fragment_shading_rate {
                enabled_exts.push(ash::khr::fragment_shading_rate::NAME.as_ptr());
            }
            let mut xfb_feats =
                vk::PhysicalDeviceTransformFeedbackFeaturesEXT::default().transform_feedback(true);
            if transform_feedback {
                enabled_exts.push(ash::ext::transform_feedback::NAME.as_ptr());
            }
            let mut rt_feats = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
                .ray_tracing_pipeline(true);
            let mut as_feats = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);
            let mut ray_query_feats =
                vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
            if ray_tracing_pipeline {
                enabled_exts.push(ash::khr::ray_tracing_pipeline::NAME.as_ptr());
            }
            if ray_query {
                enabled_exts.push(ash::khr::ray_query::NAME.as_ptr());
            }
            if use_as {
                enabled_exts.push(ash::khr::acceleration_structure::NAME.as_ptr());
                enabled_exts.push(ash::khr::deferred_host_operations::NAME.as_ptr());
            }

            // Slang's `SV_VertexID` lowering declares the DrawParameters capability.
            // Slang's 8/16-bit integer and half-float code (detile, BVH
            // oracle) declares Int8 / Int16 / Float16 and 8/16-bit storage.
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default()
                .shader_draw_parameters(avail_f11.shader_draw_parameters == vk::TRUE)
                .storage_buffer16_bit_access(avail_f11.storage_buffer16_bit_access == vk::TRUE);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .scalar_block_layout(true)
                .sampler_mirror_clamp_to_edge(sampler_mirror_clamp_to_edge)
                .shader_int8(avail_f12.shader_int8 == vk::TRUE)
                .shader_float16(avail_f12.shader_float16 == vk::TRUE)
                .storage_buffer8_bit_access(avail_f12.storage_buffer8_bit_access == vk::TRUE);
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .dynamic_rendering(true)
                .subgroup_size_control(subgroup_size_control_enabled);
            // Fixed-function state D3D12 titles use: per-MRT blend, wireframe,
            // depth clamp (DepthClipEnable = FALSE), depth bounds, tessellation,
            // geometry shaders, pipeline statistics, cube arrays, and format-less
            // storage writes (translated shaders declare images without a format).
            let av = instance.get_physical_device_features(pdev);
            let on = |b: vk::Bool32| b == vk::TRUE;
            let depth_clamp = on(av.depth_clamp);
            let depth_bounds = on(av.depth_bounds);
            let tessellation_shader = on(av.tessellation_shader);
            let geometry_shader = on(av.geometry_shader);
            let pipeline_statistics_query = on(av.pipeline_statistics_query);
            let sparse_binding = on(av.sparse_binding)
                && qprops[qfamily as usize]
                    .queue_flags
                    .contains(vk::QueueFlags::SPARSE_BINDING);
            let sparse_image_2d = sparse_binding && on(av.sparse_residency_image2_d);
            let sparse_aliased_images = sparse_image_2d && on(av.sparse_residency_aliased);
            let core_feats = vk::PhysicalDeviceFeatures::default()
                .shader_int64(true)
                .shader_int16(on(av.shader_int16))
                .sparse_binding(sparse_binding)
                .sparse_residency_image2_d(sparse_image_2d)
                .sparse_residency_aliased(sparse_aliased_images)
                .independent_blend(on(av.independent_blend))
                .fill_mode_non_solid(on(av.fill_mode_non_solid))
                .depth_clamp(depth_clamp)
                .depth_bounds(depth_bounds)
                .depth_bias_clamp(on(av.depth_bias_clamp))
                .tessellation_shader(tessellation_shader)
                .geometry_shader(geometry_shader)
                .pipeline_statistics_query(pipeline_statistics_query)
                .occlusion_query_precise(on(av.occlusion_query_precise))
                .image_cube_array(on(av.image_cube_array))
                .shader_storage_image_write_without_format(on(
                    av.shader_storage_image_write_without_format
                ))
                .shader_storage_image_read_without_format(on(
                    av.shader_storage_image_read_without_format
                ));

            let mut dev_ci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&q_ci)
                .enabled_extension_names(&enabled_exts)
                .enabled_features(&core_feats)
                .push_next(&mut f11)
                .push_next(&mut f12)
                .push_next(&mut f13);
            if quad_control {
                dev_ci = dev_ci
                    .push_next(&mut quad_feats)
                    .push_next(&mut reconv_feats);
            }
            if mesh_shader {
                dev_ci = dev_ci.push_next(&mut mesh_feats);
            }
            if fragment_shading_rate {
                dev_ci = dev_ci.push_next(&mut vrs_feats);
            }
            if transform_feedback {
                dev_ci = dev_ci.push_next(&mut xfb_feats);
            }
            if ray_tracing_pipeline {
                dev_ci = dev_ci.push_next(&mut rt_feats);
            }
            if ray_query {
                dev_ci = dev_ci.push_next(&mut ray_query_feats);
            }
            if use_as {
                dev_ci = dev_ci.push_next(&mut as_feats);
            }

            let device = instance.create_device(pdev, &dev_ci, None)?;
            let ext_host_mem = has_ext_host
                .then(|| ash::ext::external_memory_host::Device::new(&instance, &device));
            let ext_mesh_shader =
                mesh_shader.then(|| ash::ext::mesh_shader::Device::new(&instance, &device));
            let khr_fragment_shading_rate = fragment_shading_rate
                .then(|| ash::khr::fragment_shading_rate::Device::new(&instance, &device));
            let ext_transform_feedback = transform_feedback
                .then(|| ash::ext::transform_feedback::Device::new(&instance, &device));
            let khr_ray_tracing_pipeline = ray_tracing_pipeline
                .then(|| ash::khr::ray_tracing_pipeline::Device::new(&instance, &device));
            let acceleration_structure =
                use_as.then(|| ash::khr::acceleration_structure::Device::new(&instance, &device));
            let queue = device.get_device_queue(qfamily, 0);

            let pool_ci = vk::CommandPoolCreateInfo::default()
                .queue_family_index(qfamily)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let cmd_pool = device.create_command_pool(&pool_ci, None)?;
            let transfer = match tfamily {
                Some(tf) => {
                    let pool = device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(tf)
                            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None,
                    )?;
                    Some(crate::images::TransferQueue {
                        family: tf,
                        queue: Mutex::new(device.get_device_queue(tf, 0)),
                        pool: Mutex::new(pool),
                    })
                }
                None => None,
            };

            let n = crate::layout::image_capacity(&props.limits);
            if n == 0 {
                return Err(GpuError::ImageSlots {
                    required: 1,
                    available: 0,
                });
            }
            let crate::layout::GlobalLayout {
                set_layout,
                pipeline_layout: pipe_layout,
            } = crate::layout::GlobalLayout::create(&device, n)?;
            let pool_sizes = crate::layout::SET0_BINDINGS.map(|(_, ty)| vk::DescriptorPoolSize {
                ty,
                descriptor_count: n * 8,
            });
            let desc_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(8)
                    .pool_sizes(&pool_sizes),
                None,
            )?;

            let subgroup_size = subgroup_props.subgroup_size;
            let min_subgroup_size = subgroup_size_props.min_subgroup_size;
            let max_subgroup_size = subgroup_size_props.max_subgroup_size;
            let native_wave64 = max_subgroup_size >= 64;
            let native_wave64_stages = if native_wave64
                && subgroup_props
                    .supported_operations
                    .contains(vk::SubgroupFeatureFlags::SHUFFLE)
            {
                if subgroup_size == 64 {
                    subgroup_props.supported_stages
                } else if subgroup_size_control_enabled {
                    subgroup_props.supported_stages
                        & subgroup_size_props.required_subgroup_size_stages
                } else {
                    vk::ShaderStageFlags::empty()
                }
            } else {
                vk::ShaderStageFlags::empty()
            };
            let max_workgroup_count = props.limits.max_compute_work_group_count;
            let pipeline_cache =
                device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)?;

            let mut runner = Self {
                _entry: entry,
                instance,
                pdev,
                device,
                ext_host_mem,
                memory_budget,
                min_host_pointer_alignment,
                queue,
                cmd_pool,
                cmd_pool_lock: Mutex::new(()),
                idle_fences: Mutex::default(),
                idle_recording: Mutex::default(),
                queue_lock: Mutex::new(()),
                transfer,
                pipe_layout,
                set_layout,
                desc_pool: Mutex::new(desc_pool),
                default_image: None,
                default_3d: None,
                default_cube: None,
                default_cube_view: vk::ImageView::null(),
                default_sampler: vk::Sampler::null(),
                sampler_mirror_clamp_to_edge,
                depth_clamp,
                depth_bounds,
                tessellation_shader,
                geometry_shader,
                pipeline_statistics_query,
                ext_mesh_shader,
                mesh_shader,
                task_shader,
                khr_fragment_shading_rate,
                fragment_shading_rate,
                ext_transform_feedback,
                transform_feedback,
                khr_ray_tracing_pipeline,
                ray_tracing_pipeline,
                acceleration_structure,
                acceleration_scratch_alignment: u64::from(
                    as_props.min_acceleration_structure_scratch_offset_alignment,
                ),
                ray_query,
                sparse_aliased_images,
                wsi,
                hdr_colorspaces,
                queue_family: qfamily,
                quad_control,
                device_name,
                driver_version: props.driver_version,
                image_slots: n,
                timestamp_period: props.limits.timestamp_period,
                timestamp_bits: qprops[qfamily as usize].timestamp_valid_bits,
                subgroup_size,
                min_subgroup_size,
                max_subgroup_size,
                native_wave64,
                native_wave64_stages,
                subgroup_size_control,
                max_workgroup_count,
                pipeline_cache,
                pipeline_cache_lock: Mutex::new(()),
                pipeline_usage: Mutex::default(),
                pipelines: Mutex::new(HashMap::new()),
                gfx_pipelines: Mutex::new(HashMap::new()),
                bma: Mutex::new(None),
            };
            runner.init_defaults()?;
            Ok(runner)
        }
    }

    /// Usage and budget of the device-local heap selected for textures.
    pub fn texture_heap_budget(&self) -> Option<(u64, u64)> {
        if !self.memory_budget {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe { self.instance.get_physical_device_memory_properties2(self.pdev, &mut properties) };
        let memory = properties.memory_properties;
        let heap = memory.memory_types[..memory.memory_type_count as usize]
            .iter()
            .find(|ty| ty.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))?
            .heap_index as usize;
        (budget.heap_budget[heap] != 0).then_some((budget.heap_usage[heap], budget.heap_budget[heap]))
    }

    /// The 1x1 image (storage-capable) and sampler empty slots are bound to.
    fn init_defaults(&mut self) -> Result<(), GpuError> {
        let img = self.create_image_2d(
            vk::Format::R8G8B8A8_UNORM,
            1,
            1,
            true,
            vk::ComponentMapping::default(),
        )?;
        self.upload_image(&img, &[0, 0, 0, 0])?;
        self.default_image = Some(img);
        use crate::images::{ImageShape, ImageSpec};
        for (shape, is_3d) in [
            (ImageShape::D3 { depth: 1 }, true),
            (ImageShape::Cube { cubes: 1 }, false),
        ] {
            let img = self.create_image(&ImageSpec {
                format: vk::Format::R8G8B8A8_UNORM,
                width: 1,
                height: 1,
                shape,
                mips: 1,
                storage: true,
                components: vk::ComponentMapping::default(),
            })?;
            let zeros = vec![0u8; 4 * shape.slices() as usize];
            self.upload_image_levels(&img, &zeros, 1)?;
            if is_3d {
                self.default_3d = Some(img);
            } else {
                // SAFETY: a view of an image we own, destroyed with the runner.
                self.default_cube_view = unsafe {
                    self.device.create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(img.image)
                            .view_type(vk::ImageViewType::CUBE)
                            .format(img.format)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .level_count(1)
                                    .layer_count(6),
                            ),
                        None,
                    )?
                };
                self.default_cube = Some(img);
            }
        }
        self.default_sampler = self.create_sampler(
            &strata_shader::descriptor::SamplerDescriptor::decode([0; 4]),
        )?;
        Ok(())
    }

    /// Allocates and fills a set-0 descriptor set for `b` (defaults elsewhere).
    ///
    /// # Safety
    /// The caller frees the set (after the submission using it completes).
    pub(crate) unsafe fn image_set(
        &self,
        b: &crate::images::ImageBindings,
    ) -> Result<vk::DescriptorSet, GpuError> {
        use strata_shader::resource::ImageDim;
        use strata_shader::spirv::{storage_binding, texture_binding};
        let d2 = self.default_image.as_ref().expect("defaults initialised");
        let d3 = self.default_3d.as_ref().expect("defaults initialised");
        let cube = self.default_cube.as_ref().expect("defaults initialised");
        let cube_array = cube.alt_views[1].expect("a cube has an array view");
        // Storage-capable images have identity views; the cube's main view
        // is its array view (a cube cannot be a storage view), so sampled
        // cube slots use a dedicated cube view made at init.
        let default_view = |dim: ImageDim, storage: bool| match dim {
            ImageDim::D2 => d2.view,
            ImageDim::D3 => d3.view,
            ImageDim::Cube if !storage => self.default_cube_view,
            ImageDim::Cube | ImageDim::D2Array => cube_array,
        };
        let n = self.image_slots as usize;
        let info = |view: vk::ImageView| {
            vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)
        };
        let tex: Vec<(u32, Vec<vk::DescriptorImageInfo>)> = strata_shader::spirv::IMAGE_DIMS
            .iter()
            .map(|&dim| {
                let infos = (0..n)
                    .map(|i| {
                        let view = b
                            .textures
                            .get(i)
                            .copied()
                            .flatten()
                            .filter(|t| t.2 == dim)
                            .map_or(default_view(dim, false), |t| t.0);
                        info(view)
                    })
                    .collect();
                (texture_binding(dim), infos)
            })
            .collect();
        let samp: Vec<vk::DescriptorImageInfo> = (0..n)
            .map(|i| {
                let s = b
                    .textures
                    .get(i)
                    .copied()
                    .flatten()
                    .and_then(|t| t.1)
                    .unwrap_or(self.default_sampler);
                vk::DescriptorImageInfo::default().sampler(s)
            })
            .collect();
        let stor: Vec<(u32, Vec<vk::DescriptorImageInfo>)> =
            [ImageDim::D2, ImageDim::D3, ImageDim::D2Array]
                .iter()
                .map(|&dim| {
                    let infos = (0..n)
                        .map(|i| {
                            let view = b
                                .storage
                                .get(i)
                                .copied()
                                .flatten()
                                .filter(|t| t.1 == dim)
                                .map_or(default_view(dim, true), |t| t.0);
                            info(view)
                        })
                        .collect();
                    (storage_binding(dim), infos)
                })
                .collect();
        let pool = self.desc_pool.lock().unwrap_or_else(|e| e.into_inner());
        let layouts = [self.set_layout];
        // SAFETY: forwarded from the caller; the pool lock serialises its use.
        unsafe {
            let set = self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*pool)
                    .set_layouts(&layouts),
            )?[0];
            let mut writes = vec![
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(strata_shader::spirv::BINDING_SAMPLERS)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&samp),
            ];
            for (binding, infos) in &tex {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(*binding)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(infos),
                );
            }
            for (binding, infos) in &stor {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(*binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(infos),
                );
            }
            self.device.update_descriptor_sets(&writes, &[]);
            Ok(set)
        }
    }

    /// # Safety
    /// No submission still uses `set`.
    pub(crate) unsafe fn free_image_set(&self, set: vk::DescriptorSet) {
        let pool = self.desc_pool.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: forwarded from the caller.
        unsafe {
            let _ = self.device.free_descriptor_sets(*pool, &[set]);
        }
    }

    pub fn alloc_bda_buffer(&self, size: u64) -> Result<BdaBuffer, GpuError> {
        self.alloc_bda_buffer_with_usage(size, vk::BufferUsageFlags::empty())
    }

    pub(crate) fn alloc_bda_buffer_with_usage(
        &self,
        size: u64,
        extra_usage: vk::BufferUsageFlags,
    ) -> Result<BdaBuffer, GpuError> {
        unsafe {
            let buf_ci = vk::BufferCreateInfo::default()
                .size(size)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::INDEX_BUFFER
                        | vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::INDIRECT_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | extra_usage,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&buf_ci, None)?;
            let req = self.device.get_buffer_memory_requirements(buffer);

            let mem_props = self
                .instance
                .get_physical_device_memory_properties(self.pdev);
            let want =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let Some(mem_type) = (0..mem_props.memory_type_count).find(|&i| {
                (req.memory_type_bits & (1 << i)) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(want)
            }) else {
                self.device.destroy_buffer(buffer, None);
                return Err(GpuError::NoDevice);
            };

            let mut alloc_flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let alloc_ci = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type)
                .push_next(&mut alloc_flags);
            let memory = match self.device.allocate_memory(&alloc_ci, None) {
                Ok(memory) => memory,
                Err(error) => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(error.into());
                }
            };
            let mapped = self
                .device
                .bind_buffer_memory(buffer, memory, 0)
                .and_then(|()| {
                    self.device
                        .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                });
            let ptr = match mapped {
                Ok(ptr) => ptr.cast(),
                Err(error) => {
                    self.device.destroy_buffer(buffer, None);
                    self.device.free_memory(memory, None);
                    return Err(error.into());
                }
            };
            let bda = self
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));

            Ok(BdaBuffer {
                buffer,
                memory,
                bda,
                size,
                ptr,
                imported_host: false,
            })
        }
    }

    /// Zero-copy import of an existing CPU allocation (e.g. Windows `VirtualAlloc` /
    /// console `XMemVirtualAlloc`) directly into a 64-bit GPU BDA pointer via
    /// `VK_EXT_external_memory_host`.
    ///
    /// # Safety
    /// `host_ptr` must be valid for `size` bytes, aligned to `min_host_pointer_alignment`,
    /// and remain allocated until `free_bda_buffer` is called.
    pub unsafe fn import_host_bda_buffer(
        &self,
        host_ptr: *mut u8,
        size: u64,
    ) -> Result<BdaBuffer, GpuError> {
        let ext = self.ext_host_mem.as_ref().ok_or(GpuError::NoDevice)?;
        unsafe {
            let mut ext_buf_ci = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
            let buf_ci = vk::BufferCreateInfo::default()
                .size(size)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::INDEX_BUFFER
                        | vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::INDIRECT_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut ext_buf_ci);
            let buffer = self.device.create_buffer(&buf_ci, None)?;
            let req = self.device.get_buffer_memory_requirements(buffer);

            let mut ptr_props = vk::MemoryHostPointerPropertiesEXT::default();
            (ext.fp().get_memory_host_pointer_properties_ext)(
                self.device.handle(),
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                host_ptr as *const std::ffi::c_void,
                &mut ptr_props,
            )
            .result()?;

            let valid_bits = req.memory_type_bits & ptr_props.memory_type_bits;
            let mem_type = (0..32u32)
                .find(|&i| (valid_bits & (1 << i)) != 0)
                .ok_or(GpuError::NoDevice)?;

            let align = self.min_host_pointer_alignment.max(1);
            let alloc_size = req.size.max(size).next_multiple_of(align);

            let mut alloc_flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut import_info = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
                .host_pointer(host_ptr as *mut std::ffi::c_void);
            let alloc_ci = vk::MemoryAllocateInfo::default()
                .allocation_size(alloc_size)
                .memory_type_index(mem_type)
                .push_next(&mut alloc_flags)
                .push_next(&mut import_info);

            let memory = self.device.allocate_memory(&alloc_ci, None)?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;

            let bda = self
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));

            Ok(BdaBuffer {
                buffer,
                memory,
                bda,
                size,
                ptr: host_ptr,
                imported_host: true,
            })
        }
    }

    /// Tier 2: Allocate a `DEVICE_LOCAL` GDDR6 BDA buffer and upload `initial_bytes` via GPU copy.
    pub fn alloc_vram_bda_buffer(
        &self,
        size: u64,
        initial_bytes: &[u8],
    ) -> Result<BdaBuffer, GpuError> {
        unsafe {
            let buf_ci = vk::BufferCreateInfo::default()
                .size(size)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::INDEX_BUFFER
                        | vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::INDIRECT_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&buf_ci, None)?;
            let req = self.device.get_buffer_memory_requirements(buffer);

            let mem_props = self
                .instance
                .get_physical_device_memory_properties(self.pdev);
            let mem_type = (0..mem_props.memory_type_count)
                .find(|&i| {
                    (req.memory_type_bits & (1 << i)) != 0
                        && mem_props.memory_types[i as usize]
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .ok_or(GpuError::NoDevice)?;

            let mut alloc_flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let alloc_ci = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type)
                .push_next(&mut alloc_flags);
            let memory = self.device.allocate_memory(&alloc_ci, None)?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;

            // Upload initial bytes through a short-lived staging buffer
            if !initial_bytes.is_empty() {
                let stage_ci = vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                let stage_buf = self.device.create_buffer(&stage_ci, None)?;
                let stage_req = self.device.get_buffer_memory_requirements(stage_buf);
                let want =
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                let stage_mem_type = (0..mem_props.memory_type_count)
                    .find(|&i| {
                        (stage_req.memory_type_bits & (1 << i)) != 0
                            && mem_props.memory_types[i as usize]
                                .property_flags
                                .contains(want)
                    })
                    .ok_or(GpuError::NoDevice)?;
                let stage_mem = self.device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(stage_req.size)
                        .memory_type_index(stage_mem_type),
                    None,
                )?;
                self.device.bind_buffer_memory(stage_buf, stage_mem, 0)?;
                let stage_ptr =
                    self.device
                        .map_memory(stage_mem, 0, size, vk::MemoryMapFlags::empty())?
                        as *mut u8;
                std::ptr::copy_nonoverlapping(
                    initial_bytes.as_ptr(),
                    stage_ptr,
                    initial_bytes.len(),
                );
                self.device.unmap_memory(stage_mem);

                let _pool = self.cmd_pool_lock.lock().unwrap_or_else(|e| e.into_inner());
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
                let region = [vk::BufferCopy::default().size(initial_bytes.len() as u64)];
                self.device.cmd_copy_buffer(cmd, stage_buf, buffer, &region);
                self.device.end_command_buffer(cmd)?;

                let result = self.submit_and_wait(cmd);
                self.device.destroy_buffer(stage_buf, None);
                self.device.free_memory(stage_mem, None);
                result?;
            }

            let bda = self
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            Ok(BdaBuffer {
                buffer,
                memory,
                bda,
                size,
                ptr: std::ptr::null_mut(),
                imported_host: true, // not mapped -> skip vkUnmapMemory
            })
        }
    }

    /// No-API `VK_KHR_dynamic_rendering` live test: renders a vertex-less Slang
    /// fullscreen triangle (`present_blit.slang`) with ZERO `VkRenderPass`, ZERO
    /// `VkFramebuffer`, and ZERO vertex buffers, reading RGBA8 pixels from a 64-bit
    /// BDA pointer and returning the rendered top-left pixel RGBA8.
    pub fn render_no_api_present_blit(
        &self,
        vs_words: &[u32],
        fs_words: &[u32],
        src_rgba8_bda: u64,
        width: u32,
        height: u32,
        exposure: f32,
    ) -> Result<[u8; 4], GpuError> {
        unsafe {
            // 1. Create RGBA8 color target image + view
            let img_ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let img = self.device.create_image(&img_ci, None)?;
            let req = self.device.get_image_memory_requirements(img);
            let mem_props = self
                .instance
                .get_physical_device_memory_properties(self.pdev);
            let mem_type = (0..mem_props.memory_type_count)
                .find(|&i| {
                    (req.memory_type_bits & (1 << i)) != 0
                        && mem_props.memory_types[i as usize]
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .ok_or(GpuError::NoDevice)?;
            let img_mem = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )?;
            self.device.bind_image_memory(img, img_mem, 0)?;

            let view = self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(img)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )?;

            // 2. Build No-API graphics pipeline using VK_KHR_dynamic_rendering (NO RenderPass!)
            let vs_mod = self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(vs_words),
                None,
            )?;
            let fs_mod = self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(fs_words),
                None,
            )?;
            let vs_name = CString::new("main_vs").unwrap();
            let fs_name = CString::new("main_fs").unwrap();
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vs_mod)
                    .name(&vs_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fs_mod)
                    .name(&fs_name),
            ];

            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default(); // ZERO vertex inputs!
            let input_asm = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissor = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            }];
            let vp_state = vk::PipelineViewportStateCreateInfo::default()
                .viewports(&viewport)
                .scissors(&scissor);
            let rast = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .line_width(1.0);
            let ms = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let blend_att = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_att);

            let color_fmts = [vk::Format::R8G8B8A8_UNORM];
            let mut dyn_info =
                vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_fmts);

            let pipe_ci = [vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_asm)
                .viewport_state(&vp_state)
                .rasterization_state(&rast)
                .multisample_state(&ms)
                .color_blend_state(&blend)
                .layout(self.pipe_layout)
                .push_next(&mut dyn_info)];
            let pipeline = self
                .device
                .create_graphics_pipelines(self.pipeline_cache, &pipe_ci, None)
                .map_err(|(_, e)| e)?[0];

            // 3. Readback buffer (1 pixel = 4 bytes, padded to 256)
            let readback = self.alloc_bda_buffer(256)?;

            // 4. Record dynamic rendering pass + copy pixel (0,0) to readback
            let _pool = self.cmd_pool_lock.lock().unwrap_or_else(|e| e.into_inner());
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

            let to_att = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_att),
            );

            let att = [vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                })];
            let render_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width, height },
                })
                .layer_count(1)
                .color_attachments(&att);

            self.device.cmd_begin_rendering(cmd, &render_info);
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);

            let mut pc = Vec::with_capacity(20);
            pc.extend_from_slice(&src_rgba8_bda.to_le_bytes());
            pc.extend_from_slice(&width.to_le_bytes());
            pc.extend_from_slice(&height.to_le_bytes());
            pc.extend_from_slice(&exposure.to_le_bytes());
            self.device
                .cmd_push_constants(cmd, self.pipe_layout, PUSH_STAGES, 0, &pc);
            self.device.cmd_draw(cmd, 3, 1, 0, 0); // Fullscreen triangle (ZERO vertex buffers!)
            self.device.cmd_end_rendering(cmd);

            let to_src = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_src),
            );

            let copy = [vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                })];
            self.device.cmd_copy_image_to_buffer(
                cmd,
                img,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback.buffer,
                &copy,
            );
            self.device.end_command_buffer(cmd)?;

            let pixel = self
                .submit_and_wait(cmd)
                .map(|()| readback.as_slice::<u8>(4).try_into().unwrap());
            self.free_bda_buffer(readback);
            self.device.destroy_pipeline(pipeline, None);
            self.device.destroy_shader_module(vs_mod, None);
            self.device.destroy_shader_module(fs_mod, None);
            self.device.destroy_image_view(view, None);
            self.device.destroy_image(img, None);
            self.device.free_memory(img_mem, None);

            pixel
        }
    }

    pub fn free_bda_buffer(&self, buf: BdaBuffer) {
        unsafe {
            if !buf.imported_host {
                self.device.unmap_memory(buf.memory);
            }
            self.device.destroy_buffer(buf.buffer, None);
            self.device.free_memory(buf.memory, None);
        }
    }

    /// Dispatch a SPIR-V 1.6 compute shader with raw push constants and wait for completion.
    pub fn dispatch_spirv(
        &self,
        spirv_words: &[u32],
        push_bytes: &[u8],
        groups: [u32; 3],
    ) -> Result<(), GpuError> {
        self.dispatch_spirv_with_subgroup_size(spirv_words, push_bytes, groups, None)
    }

    /// Dispatch a SPIR-V 1.6 compute shader, optionally requesting a specific hardware
    /// subgroup size (e.g. 32 or 64) when `subgroupSizeControl` is supported.
    ///
    /// Group counts beyond `maxComputeWorkGroupCount` are split into several
    /// `vkCmdDispatchBase` calls in one submission, so every group runs with
    /// its own `WorkgroupId` (a clamp would silently drop the rest).
    pub fn dispatch_spirv_with_subgroup_size(
        &self,
        spirv_words: &[u32],
        push_bytes: &[u8],
        groups: [u32; 3],
        required_subgroup_size: Option<u32>,
    ) -> Result<(), GpuError> {
        self.dispatch_spirv_bound(
            spirv_words,
            push_bytes,
            groups,
            required_subgroup_size,
            &Default::default(),
        )
    }

    /// As [`Self::dispatch_spirv_with_subgroup_size`], with images bound in set 0.
    pub fn dispatch_spirv_bound(
        &self,
        spirv_words: &[u32],
        push_bytes: &[u8],
        groups: [u32; 3],
        required_subgroup_size: Option<u32>,
        images: &crate::images::ImageBindings,
    ) -> Result<(), GpuError> {
        let pipeline = self.pipeline_for(spirv_words, required_subgroup_size)?;
        // SAFETY: freed below, after the submission is waited on.
        let set = unsafe { self.image_set(images)? };
        let result = unsafe { self.dispatch_recorded(pipeline, set, [(push_bytes, groups)]) };
        unsafe { self.free_image_set(set) };
        result
    }

    /// Runs independent dispatches of an internal BDA shader in one submission.
    pub(crate) fn dispatch_bda_batch<'a>(
        &self,
        spirv_words: &[u32],
        dispatches: impl IntoIterator<Item = (&'a [u8], [u32; 3])>,
    ) -> Result<(), GpuError> {
        let pipeline = self.pipeline_for(spirv_words, None)?;
        let set = unsafe { self.image_set(&Default::default())? };
        let result = unsafe { self.dispatch_recorded(pipeline, set, dispatches) };
        unsafe { self.free_image_set(set) };
        result
    }

    /// # Safety
    /// `set` stays valid until this returns.
    unsafe fn dispatch_recorded<'a>(
        &self,
        pipeline: vk::Pipeline,
        set: vk::DescriptorSet,
        dispatches: impl IntoIterator<Item = (&'a [u8], [u32; 3])>,
    ) -> Result<(), GpuError> {
        let _pool = self.cmd_pool_lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let cmd_ai = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&cmd_ai)?[0];

            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let prior_writes = [vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::HOST,
                )
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::HOST_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)];
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&prior_writes),
            );
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipe_layout,
                0,
                &[set],
                &[],
            );
            for (push_bytes, groups) in dispatches {
                if !push_bytes.is_empty() {
                    self.device.cmd_push_constants(cmd, self.pipe_layout, PUSH_STAGES, 0, push_bytes);
                }
                for (base, count) in crate::thin_list::split_dispatch(groups, self.max_workgroup_count) {
                    self.device.cmd_dispatch_base(
                        cmd, base[0], base[1], base[2], count[0], count[1], count[2],
                    );
                }
            }
            let host_read = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::HOST | vk::PipelineStageFlags2::ALL_COMMANDS,
                )
                .dst_access_mask(vk::AccessFlags2::HOST_READ | vk::AccessFlags2::MEMORY_READ)];
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&host_read),
            );
            self.device.end_command_buffer(cmd)?;
            self.submit_and_wait(cmd)
        }
    }

    /// The compute pipeline for `spirv_words`, created once and kept.
    pub fn pipeline_for(
        &self,
        spirv_words: &[u32],
        required_subgroup_size: Option<u32>,
    ) -> Result<vk::Pipeline, GpuError> {
        let size = required_subgroup_size.filter(|&s| {
            self.subgroup_size_control && s >= self.min_subgroup_size && s <= self.max_subgroup_size
        });
        let key = (spirv_hash(spirv_words), size.unwrap_or(0));
        if let Some(&(_, p)) = self
            .pipelines
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return Ok(p);
        }
        // BMA-instrumented shaders read the table through spec constants
        // (resolved before locking: creating the table dispatches).
        let bma_table = if crate::bma::uses_bma_table(spirv_words) {
            Some(self.bma_table_address()?)
        } else {
            None
        };
        let _cache = self.pipeline_cache_lock.lock().unwrap();
        if let Some(&(_, p)) = self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Ok(p);
        }
        self.check_shader_images(spirv_words)?;
        let start = std::time::Instant::now();
        unsafe {
            let sm_ci = vk::ShaderModuleCreateInfo::default().code(spirv_words);
            let module = self.device.create_shader_module(&sm_ci, None)?;
            let entry_name = CString::new("main").unwrap();
            let mut req_sg_info = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default();
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&entry_name);
            if let Some(req_size) = size {
                req_sg_info = req_sg_info.required_subgroup_size(req_size);
                stage = stage.push_next(&mut req_sg_info);
            }
            let spec_data: [u32; 2] = bma_table.map_or([0, 0], |a| [a as u32, (a >> 32) as u32]);
            let spec_bytes: Vec<u8> = spec_data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let spec_entries = [
                vk::SpecializationMapEntry::default()
                    .constant_id(strata_shader::bma::SPEC_ID_TABLE_LO)
                    .offset(0)
                    .size(4),
                vk::SpecializationMapEntry::default()
                    .constant_id(strata_shader::bma::SPEC_ID_TABLE_HI)
                    .offset(4)
                    .size(4),
            ];
            let spec_info = vk::SpecializationInfo::default()
                .map_entries(&spec_entries)
                .data(&spec_bytes);
            if bma_table.is_some() {
                stage = stage.specialization_info(&spec_info);
            }
            let pipe_ci = [vk::ComputePipelineCreateInfo::default()
                .flags(vk::PipelineCreateFlags::DISPATCH_BASE)
                .stage(stage)
                .layout(self.pipe_layout)];
            let created = self
                .device
                .create_compute_pipelines(self.pipeline_cache, &pipe_ci, None)
                .map_err(|(_, e)| e);
            let pipeline = match created {
                Ok(p) => p[0],
                Err(e) => {
                    crate::perf::pipeline_errors(1);
                    self.device.destroy_shader_module(module, None);
                    return Err(e.into());
                }
            };
            self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).insert(key, (module, pipeline));
            crate::perf::pipeline_creation_ns(start.elapsed().as_nanos() as u64);
            self.record_pipeline(
                format!("cs-{:016x}-{}", key.0, key.1),
                false,
                start.elapsed(),
                || crate::pipeline_usage::Recipe::Compute {
                    code: spirv_words.to_vec(),
                    subgroup: size,
                },
            );
            Ok(pipeline)
        }
    }

    /// Creates the pipelines of `shaders` ahead of use (a warm-up before
    /// play, so the first frames do not stall on pipeline creation).
    /// Returns how many could be created.
    pub fn warm<'a>(&self, shaders: impl IntoIterator<Item = &'a [u32]>) -> usize {
        shaders
            .into_iter()
            .filter(|s| self.pipeline_for(s, None).is_ok())
            .count()
    }

    pub(crate) fn pipeline_cache(&self) -> vk::PipelineCache {
        self.pipeline_cache
    }

    pub(crate) fn check_shader_images(&self, words: &[u32]) -> Result<(), GpuError> {
        if let Some(required) = strata_shader::spirv::required_image_slots(words)
            && required > self.image_slots
        {
            return Err(GpuError::ImageSlots {
                required,
                available: self.image_slots,
            });
        }
        Ok(())
    }

    pub(crate) fn adapter_cache_key(&self) -> String {
        let props = unsafe { self.instance.get_physical_device_properties(self.pdev) };
        let uuid: String = props
            .pipeline_cache_uuid
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        format!(
            "{:04x}-{:04x}-{:08x}-{}-{}-{}",
            props.vendor_id,
            props.device_id,
            props.driver_version,
            uuid,
            self.image_slots,
            strata_shader::cache::COMPILER_VERSION
        )
    }

    pub fn cached_pipeline_count(&self) -> usize {
        self.pipelines
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Merges a `VkPipelineCache` blob saved by [`Self::save_pipeline_cache`].
    /// A blob from another driver or device is rejected and ignored.
    pub fn load_pipeline_cache(&self, path: &Path) -> Result<bool, GpuError> {
        if !path
            .metadata()
            .is_ok_and(|m| (32..=64 * 1024 * 1024).contains(&m.len()))
        {
            return Ok(false);
        }
        let Ok(data) = std::fs::read(path) else {
            return Ok(false);
        };
        let props = unsafe { self.instance.get_physical_device_properties(self.pdev) };
        let Some(header) = data.get(..32) else {
            return Ok(false);
        };
        let word = |i: usize| u32::from_le_bytes(header[i..i + 4].try_into().unwrap());
        if !(32..=data.len()).contains(&(word(0) as usize))
            || word(4) != 1
            || word(8) != props.vendor_id
            || word(12) != props.device_id
            || header[16..32] != props.pipeline_cache_uuid
        {
            return Ok(false);
        }
        let _cache = self.pipeline_cache_lock.lock().unwrap();
        unsafe {
            let ci = vk::PipelineCacheCreateInfo::default().initial_data(&data);
            let Ok(loaded) = self.device.create_pipeline_cache(&ci, None) else {
                return Ok(false);
            };
            let merged = self
                .device
                .merge_pipeline_caches(self.pipeline_cache, &[loaded]);
            self.device.destroy_pipeline_cache(loaded, None);
            merged?;
        }
        Ok(true)
    }

    /// Writes the driver's pipeline cache atomically (`.tmp`, then rename).
    pub fn save_pipeline_cache(&self, path: &Path) -> Result<usize, GpuError> {
        let data = {
            let _cache = self.pipeline_cache_lock.lock().unwrap();
            unsafe { self.device.get_pipeline_cache_data(self.pipeline_cache)? }
        };
        let io = |e: std::io::Error| GpuError::Io(e.to_string());
        crate::present::write_atomic(path, &data).map_err(io)?;
        Ok(data.len())
    }

    /// Copies `(src_offset, dst_offset, len)` regions between two buffers
    /// and waits; the host sees the result afterwards.
    pub fn copy_buffer_regions(
        &self,
        src: vk::Buffer,
        dst: vk::Buffer,
        regions: &[(u64, u64, u64)],
    ) -> Result<(), GpuError> {
        if regions.is_empty() {
            return Ok(());
        }
        let copies: Vec<vk::BufferCopy> = regions
            .iter()
            .map(|&(s, d, n)| {
                vk::BufferCopy::default()
                    .src_offset(s)
                    .dst_offset(d)
                    .size(n)
            })
            .collect();
        self.one_shot(|_, cmd| unsafe {
            // Earlier shader and host writes before the copy; host reads after.
            let before = [vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::HOST,
                )
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::HOST_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(
                    vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
                )];
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&before),
            );
            self.device.cmd_copy_buffer(cmd, src, dst, &copies);
            let after = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::HOST | vk::PipelineStageFlags2::ALL_COMMANDS,
                )
                .dst_access_mask(vk::AccessFlags2::HOST_READ | vk::AccessFlags2::MEMORY_READ)];
            self.device
                .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&after));
        })
    }

    /// Reads a real calibrated GPU timestamp in nanoseconds using a Vulkan
    /// `QueryType::TIMESTAMP` query pool and `vkCmdWriteTimestamp`.
    pub fn query_gpu_timestamp_ns(&self) -> Result<u64, GpuError> {
        unsafe {
            let pool = self.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(1),
                None,
            )?;
            let submit_res = self.one_shot(|dev, cmd| {
                dev.cmd_reset_query_pool(cmd, pool, 0, 1);
                dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::ALL_COMMANDS, pool, 0);
            });
            let mut ticks = [0u64; 1];
            let query_res = submit_res.and_then(|()| {
                self.device
                    .get_query_pool_results(
                        pool,
                        0,
                        &mut ticks,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .map_err(Into::into)
            });
            self.device.destroy_query_pool(pool, None);
            query_res?;
            let props = self.instance.get_physical_device_properties(self.pdev);
            let period_ns = f64::from(props.limits.timestamp_period.max(1.0));
            Ok((ticks[0] as f64 * period_ns) as u64)
        }
    }

    /// # Safety
    /// `cmd` is a recorded primary command buffer and the caller owns the pool lock.
    pub(crate) unsafe fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<(), GpuError> {
        unsafe { self.submit_command_buffer_and_wait(cmd, None) }
    }

    /// Takes ownership of a recorded primary command buffer from `cmd_pool`.
    /// The caller holds the pool lock.
    pub(crate) unsafe fn submit_command_buffer_and_wait(
        &self,
        cmd: vk::CommandBuffer,
        wait: Option<vk::Semaphore>,
    ) -> Result<(), GpuError> {
        unsafe {
            let cmds = [cmd];
            let result = self.submit_buffers_and_wait(&cmds, wait);
            self.device.free_command_buffers(self.cmd_pool, &cmds);
            result
        }
    }

    /// Borrows executable primary command buffers until the GPU is done with them.
    pub(crate) unsafe fn submit_buffers_and_wait(
        &self,
        cmds: &[vk::CommandBuffer],
        wait: Option<vk::Semaphore>,
    ) -> Result<(), GpuError> {
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let waits = wait.as_slice();
        let submit = [vk::SubmitInfo::default()
            .command_buffers(cmds)
            .wait_semaphores(waits)
            .wait_dst_stage_mask(&stages[..waits.len()])];
        unsafe {
            let cached = self.idle_fences.lock().unwrap().pop();
            let fence = match cached {
                Some(fence) => fence,
                None => self.device.create_fence(&vk::FenceCreateInfo::default(), None)?,
            };
            let submitted = {
                let _queue = self.queue_lock.lock().unwrap_or_else(|e| e.into_inner());
                self.device.queue_submit(self.queue, &submit, fence)
            };
            let result = submitted.and_then(|()| {
                let start = crate::perf::start();
                let wait = self.device.wait_for_fences(&[fence], true, u64::MAX);
                if let Some(start) = start {
                    crate::perf::fence_wait_ns(start.elapsed().as_nanos() as u64);
                }
                if matches!(wait, Err(error) if error != vk::Result::ERROR_DEVICE_LOST) {
                    // Keep the command buffers live until the submission lands.
                    let _queue = self.queue_lock.lock().unwrap_or_else(|e| e.into_inner());
                    let _ = self.wait_submitted_fence(self.queue, fence);
                }
                wait
            });
            let mut idle = self.idle_fences.lock().unwrap();
            if result.is_ok() && idle.len() < 8 && self.device.reset_fences(&[fence]).is_ok() {
                idle.push(fence);
            } else {
                self.device.destroy_fence(fence, None);
            }
            result.map_err(Into::into)
        }
    }

    pub(crate) fn wait_idle(&self) -> Result<(), vk::Result> {
        let _queue = self.queue_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _transfer = self
            .transfer
            .as_ref()
            .map(|queue| queue.queue.lock().unwrap_or_else(|e| e.into_inner()));
        unsafe { self.device.device_wait_idle() }
    }

    /// Records independent jobs from disjoint pools, then submits the
    /// resulting command buffers together in input order.
    pub(crate) fn record_and_submit_independent<T: Sync>(
        &self,
        jobs: &[T],
        record: impl Fn(&ash::Device, vk::CommandBuffer, &T) + Sync,
    ) -> Result<(), GpuError> {
        if jobs.is_empty() {
            return Ok(());
        }
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let per_worker = jobs.len().div_ceil(threads.min(4));
        let mut buffers = Vec::new();
        let result = (|| {
            for _ in jobs.chunks(per_worker) {
                buffers.push(self.take_recording_buffer()?);
            }
            let recorded = std::thread::scope(|scope| {
                let record = &record;
                let handles: Vec<_> = jobs
                    .chunks(per_worker)
                    .zip(&buffers)
                    .map(|(group, &(_, cmd))| {
                        scope.spawn(move || unsafe {
                            self.device.begin_command_buffer(
                                cmd,
                                &vk::CommandBufferBeginInfo::default()
                                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                            )?;
                            for job in group {
                                record(&self.device, cmd, job);
                            }
                            self.device.end_command_buffer(cmd)?;
                            Ok::<_, vk::Result>(cmd)
                        })
                    })
                    .collect();
                let mut buffers = Vec::with_capacity(handles.len());
                let mut error = None;
                for handle in handles {
                    match handle.join() {
                        Ok(Ok(cmd)) => buffers.push(cmd),
                        Ok(Err(err)) => {
                            error.get_or_insert(err.into());
                        }
                        Err(_) => {
                            error.get_or_insert_with(|| {
                                GpuError::Io("parallel Vulkan command recording failed".into())
                            });
                        }
                    }
                }
                error.map_or(Ok(buffers), Err)
            })?;
            unsafe { self.submit_buffers_and_wait(&recorded, None) }
        })();
        let mut idle = self.idle_recording.lock().unwrap();
        for (pool, cmd) in buffers {
            unsafe {
                if idle.len() < 8 && self.device.reset_command_pool(pool, vk::CommandPoolResetFlags::empty()).is_ok() {
                    idle.push((pool, cmd));
                } else {
                    self.device.destroy_command_pool(pool, None);
                }
            }
        }
        result
    }

    fn take_recording_buffer(&self) -> Result<(vk::CommandPool, vk::CommandBuffer), vk::Result> {
        if let Some(recording) = self.idle_recording.lock().unwrap().pop() {
            return Ok(recording);
        }
        unsafe {
            let pool = self.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.queue_family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT), None,
            )?;
            let result = self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
            );
            match result {
                Ok(commands) => Ok((pool, commands[0])),
                Err(error) => {
                    self.device.destroy_command_pool(pool, None);
                    Err(error)
                }
            }
        }
    }

    /// A host allocation failure in the wait does not make pending objects safe
    /// to free. Keep waiting for completion or device loss in that case.
    pub(crate) unsafe fn wait_submitted_fence(
        &self,
        queue: vk::Queue,
        fence: vk::Fence,
    ) -> Result<(), vk::Result> {
        unsafe {
            let start = crate::perf::start();
            let result = self.device.wait_for_fences(&[fence], true, u64::MAX);
            if matches!(result, Err(err) if err != vk::Result::ERROR_DEVICE_LOST) {
                loop {
                    match self.device.queue_wait_idle(queue) {
                        Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => break,
                        Err(_) => std::thread::yield_now(),
                    }
                }
            }
            if let Some(start) = start {
                crate::perf::fence_wait_ns(start.elapsed().as_nanos() as u64);
            }
            result
        }
    }
}

impl Drop for ComputeRunner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for fence in self.idle_fences.get_mut().unwrap().drain(..) {
                self.device.destroy_fence(fence, None);
            }
            for (pool, _) in self.idle_recording.get_mut().unwrap().drain(..) {
                self.device.destroy_command_pool(pool, None);
            }
            self.bma_destroy();
            let table = self.pipelines.get_mut().unwrap_or_else(|e| e.into_inner());
            for (_, (module, pipeline)) in table.drain() {
                self.device.destroy_pipeline(pipeline, None);
                self.device.destroy_shader_module(module, None);
            }
            let gfx = self
                .gfx_pipelines
                .get_mut()
                .unwrap_or_else(|e| e.into_inner());
            for (_, (modules, pipeline)) in gfx.drain() {
                self.device.destroy_pipeline(pipeline, None);
                for m in modules {
                    self.device.destroy_shader_module(m, None);
                }
            }
            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);
            if self.default_cube_view != vk::ImageView::null() {
                self.device.destroy_image_view(self.default_cube_view, None);
            }
            for img in [
                self.default_image.take(),
                self.default_3d.take(),
                self.default_cube.take(),
            ]
            .into_iter()
            .flatten()
            {
                for v in img.alt_views.into_iter().flatten() {
                    self.device.destroy_image_view(v, None);
                }
                self.device.destroy_image_view(img.view, None);
                self.device.destroy_image(img.image, None);
                self.device.free_memory(img.memory, None);
            }
            self.device.destroy_sampler(self.default_sampler, None);
            let pool = *self.desc_pool.get_mut().unwrap_or_else(|e| e.into_inner());
            self.device.destroy_descriptor_pool(pool, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_pipeline_layout(self.pipe_layout, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            if let Some(t) = &self.transfer {
                let pool = *t.pool.lock().unwrap_or_else(|e| e.into_inner());
                self.device.destroy_command_pool(pool, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
