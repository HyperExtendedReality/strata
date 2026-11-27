//! The one pipeline layout.
//!
//! Every pipeline Strata creates (translated console shaders, Slang utility
//! compute shaders, the vertex-less present pass) uses this single
//! `VkPipelineLayout`:
//! - 128 bytes of push constants (32 x `u32` console user SGPRs), seen by the
//!   compute, vertex and fragment stages
//! - Set 0, binding 0: `SAMPLED_IMAGE[IMAGE_ARRAY_LEN]` (2D)
//! - Set 0, binding 1: `SAMPLER[IMAGE_ARRAY_LEN]`
//! - Set 0, binding 2: `STORAGE_IMAGE[IMAGE_ARRAY_LEN]` (2D)
//! - Set 0, bindings 3/4/5: `SAMPLED_IMAGE[IMAGE_ARRAY_LEN]` (3D, cube, 2D array)
//! - Set 0, bindings 6/7: `STORAGE_IMAGE[IMAGE_ARRAY_LEN]` (3D, 2D array)
//!
//! The binding numbers and array length come from the shader translator, so
//! the layout and the SPIR-V it emits cannot drift apart. Buffers, constant
//! tables and BVH nodes are reached through 64-bit BDA pointers in the push
//! constants; pipelines that use no images simply leave set 0 unbound.

use ash::vk;
use strata_shader::spirv::{
    BINDING_SAMPLERS, BINDING_STORAGE_3D, BINDING_STORAGE_ARRAY, BINDING_STORAGE_IMAGES,
    BINDING_TEXTURES, BINDING_TEXTURES_3D, BINDING_TEXTURES_ARRAY, BINDING_TEXTURES_CUBE,
    IMAGE_ARRAY_LEN,
};

pub const PUSH_CONSTANT_BYTES: u32 = 128;

/// Stages that see the push constants and set 0. Every `vkCmdPushConstants`
/// call passes exactly these flags. `ALL` covers compute, vertex, fragment,
/// tessellation, geometry, task, mesh, and ray-tracing stages under Strata's
/// single No-API global layout.
pub const PUSH_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

/// Set 0's bindings, in binding order.
pub const SET0_BINDINGS: [(u32, vk::DescriptorType); 8] = [
    (BINDING_TEXTURES, vk::DescriptorType::SAMPLED_IMAGE),
    (BINDING_SAMPLERS, vk::DescriptorType::SAMPLER),
    (BINDING_STORAGE_IMAGES, vk::DescriptorType::STORAGE_IMAGE),
    (BINDING_TEXTURES_3D, vk::DescriptorType::SAMPLED_IMAGE),
    (BINDING_TEXTURES_CUBE, vk::DescriptorType::SAMPLED_IMAGE),
    (BINDING_TEXTURES_ARRAY, vk::DescriptorType::SAMPLED_IMAGE),
    (BINDING_STORAGE_3D, vk::DescriptorType::STORAGE_IMAGE),
    (BINDING_STORAGE_ARRAY, vk::DescriptorType::STORAGE_IMAGE),
];

pub struct GlobalLayout {
    pub set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
}

/// The single layout exposes four sampled and three storage image arrays,
/// plus one sampler array, to each stage.
pub fn image_capacity(limits: &vk::PhysicalDeviceLimits) -> u32 {
    [
        IMAGE_ARRAY_LEN,
        limits.max_per_stage_descriptor_sampled_images / 4,
        limits.max_per_stage_descriptor_storage_images / 3,
        limits.max_per_stage_descriptor_samplers,
        limits.max_descriptor_set_sampled_images / 4,
        limits.max_descriptor_set_storage_images / 3,
        limits.max_descriptor_set_samplers,
        limits.max_per_stage_resources / 8,
    ]
    .into_iter()
    .min()
    .unwrap_or(0)
}

impl GlobalLayout {
    /// Creates the set-0 layout and the pipeline layout.
    ///
    /// # Safety
    /// `device` is a live device; the caller destroys both handles.
    pub unsafe fn create(device: &ash::Device, capacity: u32) -> Result<Self, vk::Result> {
        let bindings = SET0_BINDINGS.map(|(b, ty)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(ty)
                .descriptor_count(capacity)
                .stage_flags(PUSH_STAGES)
        });
        // SAFETY: forwarded from the caller.
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };

        let push_range = [vk::PushConstantRange::default()
            .stage_flags(PUSH_STAGES)
            .offset(0)
            .size(PUSH_CONSTANT_BYTES)];
        let set_layouts = [set_layout];
        let pipe_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_range);
        // SAFETY: forwarded from the caller.
        let pipeline_layout = match unsafe { device.create_pipeline_layout(&pipe_ci, None) } {
            Ok(l) => l,
            Err(e) => {
                // SAFETY: created above and not used by anything yet.
                unsafe { device.destroy_descriptor_set_layout(set_layout, None) };
                return Err(e);
            }
        };

        Ok(Self {
            set_layout,
            pipeline_layout,
        })
    }
}
