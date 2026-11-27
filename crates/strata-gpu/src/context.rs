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
    #[error("I/O: {0}")]
    Io(String),
    #[error("host memory allocation failed")]
    HostMemory,
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

#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub index: usize,
    pub device_name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: vk::PhysicalDeviceType,
    pub dedicated_vram_bytes: u64,
    pub api_version: (u32, u32, u32),
}

/// Select a physical device using an optional `selector` (zero-based index or
/// case-insensitive name substring), falling back to `STRATA_GPU_ADAPTER` env
/// and then discrete > integrated > other.
///
/// # Safety
/// `instance` must be a valid Vulkan instance.
pub unsafe fn pick_physical_device(
    instance: &ash::Instance,
    selector: Option<&str>,
) -> Result<vk::PhysicalDevice, GpuError> {
    let pdevs = unsafe { instance.enumerate_physical_devices()? };
    if pdevs.is_empty() {
        return Err(GpuError::NoDevice);
    }
    let env_sel = std::env::var("STRATA_GPU_ADAPTER").ok();
    let eff = selector
        .or(env_sel.as_deref())
        .map(str::trim)
        .filter(|s| !s.is_empty());

    if let Some(sel) = eff {
        if let Ok(idx) = sel.parse::<usize>()
            && let Some(&dev) = pdevs.get(idx)
        {
            return Ok(dev);
        }
        let needle = sel.to_ascii_lowercase();
        for &d in &pdevs {
            let props = unsafe { instance.get_physical_device_properties(d) };
            let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                .to_string_lossy()
                .to_ascii_lowercase();
            if name.contains(&needle) {
                return Ok(d);
            }
        }
    }

    pdevs
        .into_iter()
        .max_by_key(|&d| {
            let props = unsafe { instance.get_physical_device_properties(d) };
            match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                _ => 1,
            }
        })
        .ok_or(GpuError::NoDevice)
}

/// Enumerate all Vulkan physical devices with vendor/device IDs and dedicated VRAM.
pub fn enumerate_adapters() -> Result<Vec<AdapterInfo>, GpuError> {
    unsafe {
        let entry = ash::Entry::load()?;
        let app_name = CString::new("Strata").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .api_version(vk::API_VERSION_1_3);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = entry.create_instance(&create_info, None)?;
        let pdevs = instance.enumerate_physical_devices()?;
        let mut out = Vec::with_capacity(pdevs.len());
        for (index, pdev) in pdevs.into_iter().enumerate() {
            let props = instance.get_physical_device_properties(pdev);
            let mem = instance.get_physical_device_memory_properties(pdev);
            let mut vram = 0u64;
            for i in 0..mem.memory_heap_count as usize {
                let heap = mem.memory_heaps[i];
                if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                    vram = vram.saturating_add(heap.size);
                }
            }
            let name = CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned();
            let v = props.api_version;
            out.push(AdapterInfo {
                index,
                device_name: name,
                vendor_id: props.vendor_id,
                device_id: props.device_id,
                device_type: props.device_type,
                dedicated_vram_bytes: vram,
                api_version: (
                    vk::api_version_major(v),
                    vk::api_version_minor(v),
                    vk::api_version_patch(v),
                ),
            });
        }
        instance.destroy_instance(None);
        Ok(out)
    }
}

/// Probe the primary discrete/integrated GPU (or `STRATA_GPU_ADAPTER`) for Strata's No-API requirements.
pub fn probe_capabilities() -> Result<NoApiCapabilities, GpuError> {
    probe_capabilities_for(None)
}

/// Probe a specific GPU adapter selected by index or name substring.
pub fn probe_capabilities_for(selector: Option<&str>) -> Result<NoApiCapabilities, GpuError> {
    unsafe {
        let entry = ash::Entry::load()?;
        let app_name = CString::new("Strata").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .api_version(vk::API_VERSION_1_3);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = entry.create_instance(&create_info, None)?;

        let pdev = pick_physical_device(&instance, selector)?;

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
            exts.iter()
                .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == name)
        };

        let p = &props2.properties;
        let name = CStr::from_ptr(p.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();
        let v = p.api_version;
        let max_push_constants_size = p.limits.max_push_constants_size;

        let caps = NoApiCapabilities {
            device_name: name,
            api_version: (
                vk::api_version_major(v),
                vk::api_version_minor(v),
                vk::api_version_patch(v),
            ),
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
