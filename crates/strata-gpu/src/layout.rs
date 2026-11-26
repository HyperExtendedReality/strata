//! Single Global No-API Pipeline Layout.
//!
//! Every translated console shader and Slang utility shader shares this exact
//! `VkPipelineLayout`:
//! - 128 bytes of Push Constants (32 x `u32` console User SGPRs)
//! - Set 0, Binding 0: Unbounded bindless `SAMPLED_IMAGE` array
//! - Set 0, Binding 1: Unbounded bindless `STORAGE_IMAGE` array
//! - Set 0, Binding 2: Unbounded bindless `SAMPLER` array
//!
//! All buffers, constant tables, and BVH nodes are accessed via 64-bit BDA
//! pointers passed in Push Constants—eliminating root-signature translation
//! and per-draw descriptor-table allocation.

use ash::vk;

pub const MAX_BINDLESS_IMAGES: u32 = 65_536;
pub const MAX_BINDLESS_STORAGE_IMAGES: u32 = 16_384;
pub const MAX_BINDLESS_SAMPLERS: u32 = 2_048;
pub const PUSH_CONSTANT_BYTES: u32 = 128;

pub struct GlobalLayout {
    pub set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
}

impl GlobalLayout {
    /// Create the single global bindless descriptor set layout + pipeline layout.
    pub unsafe fn create(device: &ash::Device) -> Result<Self, vk::Result> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(MAX_BINDLESS_IMAGES)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(MAX_BINDLESS_STORAGE_IMAGES)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(MAX_BINDLESS_SAMPLERS)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let flags = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
        ];

        let mut ext_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
            .binding_flags(&flags);

        let set_ci = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut ext_flags);

        let set_layout = unsafe { device.create_descriptor_set_layout(&set_ci, None)? };

        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .offset(0)
            .size(PUSH_CONSTANT_BYTES)];

        let set_layouts = [set_layout];
        let pipe_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_range);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipe_ci, None)? };

        Ok(Self { set_layout, pipeline_layout })
    }
}
