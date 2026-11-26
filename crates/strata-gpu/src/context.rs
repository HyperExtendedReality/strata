//! No-API Vulkan 1.3+ device & capability probe (`ash`).
//!
//! Queries and enables the exact hardware features that eliminate legacy API
//! abstractions for Xbox Series X|S workloads:
//! - `bufferDeviceAddress` (64-bit raw GPU pointers)
//! - `descriptorIndexing` (single global bindless texture/sampler heap)
//! - `dynamicRendering` (zero RenderPass / Framebuffer object churn)
//! - `subgroupSizeControl` (Wave32 / Wave64 awareness)
//! - `VK_EXT_external_memory_host` (zero-copy CPU-written title memory)
//! - `sparseResidencyImage2D` (64 KiB tiled aliased render targets)

use std::ffi::{CStr, CString};

use ash::vk;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("failed to load Vulkan loader: {0}")]
    Load(#[from] ash::LoadingError),
    #[error("Vulkan error: {0}")]
    Vk(#[from] vk::Result),
    #[error("no suitable Vulkan 1.3+ GPU found")]
    NoDevice,
}

#[derive(Debug, Clone)]
pub struct NoApiCapabilities {
    pub device_name: String,
    pub api_version: (u32, u32, u32),
    pub buffer_device_address: bool,
    pub dynamic_rendering: bool,
    pub descriptor_indexing: bool,
    pub subgroup_size: u32,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub native_wave64: bool,
    pub sparse_image_2d: bool,
    pub external_memory_host: bool,
    pub max_push_constants_size: u32,
}

/// Probe the primary discrete/integrated GPU for Strata's No-API requirements.
pub fn probe_capabilities() -> Result<NoApiCapabilities, GpuError> {
    unsafe {
        let entry = ash::Entry::load()?;
        let app_name = CString::new("Strata").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .api_version(vk::API_VERSION_1_3);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = entry.create_instance(&create_info, None)?;

        let pdevs = instance.enumerate_physical_devices()?;
        let pdev = pdevs
            .into_iter()
            .max_by_key(|&d| {
                let props = instance.get_physical_device_properties(d);
                match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                    _ => 1,
                }
            })
            .ok_or(GpuError::NoDevice)?;

        let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
        let mut subgroup_size_props = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut subgroup_props)
            .push_next(&mut subgroup_size_props);
        instance.get_physical_device_properties2(pdev, &mut props2);

        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut feats2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f12)
            .push_next(&mut f13);
        instance.get_physical_device_features2(pdev, &mut feats2);

        let sparse_image_2d = feats2.features.sparse_residency_image2_d == vk::TRUE;

        let exts = instance.enumerate_device_extension_properties(pdev)?;
        let has_ext = |name: &CStr| {
            exts.iter().any(|e| {
                CStr::from_ptr(e.extension_name.as_ptr()) == name
            })
        };

        let p = &props2.properties;
        let name = CStr::from_ptr(p.device_name.as_ptr()).to_string_lossy().into_owned();
        let v = p.api_version;
        let max_push_constants_size = p.limits.max_push_constants_size;

        let caps = NoApiCapabilities {
            device_name: name,
            api_version: (vk::api_version_major(v), vk::api_version_minor(v), vk::api_version_patch(v)),
            buffer_device_address: f12.buffer_device_address == vk::TRUE,
            dynamic_rendering: f13.dynamic_rendering == vk::TRUE,
            descriptor_indexing: f12.descriptor_indexing == vk::TRUE
                && f12.runtime_descriptor_array == vk::TRUE,
            subgroup_size: subgroup_props.subgroup_size,
            min_subgroup_size: subgroup_size_props.min_subgroup_size,
            max_subgroup_size: subgroup_size_props.max_subgroup_size,
            native_wave64: subgroup_size_props.max_subgroup_size >= 64,
            sparse_image_2d,
            external_memory_host: has_ext(ash::ext::external_memory_host::NAME),
            max_push_constants_size,
        };

        instance.destroy_instance(None);
        Ok(caps)
    }
}
