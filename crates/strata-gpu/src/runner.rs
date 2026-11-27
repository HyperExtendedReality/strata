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
    _entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) pdev: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    ext_host_mem: Option<ash::ext::external_memory_host::Device>,
    pub min_host_pointer_alignment: u64,
    pub(crate) queue: vk::Queue,
    pub(crate) cmd_pool: vk::CommandPool,
    /// Push constants + set 0 (images); shared by every translated pipeline.
    pub(crate) pipe_layout: vk::PipelineLayout,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    desc_pool: Mutex<vk::DescriptorPool>,
    /// 1x1 image and sampler bound to every slot a dispatch leaves empty.
    default_image: Option<crate::images::GpuImage>,
    default_sampler: vk::Sampler,
    pub(crate) sampler_mirror_clamp_to_edge: bool,
    /// `VK_KHR_shader_quad_control` is enabled: pixel shaders may require full quads.
    pub quad_control: bool,
    pub device_name: String,
    pub subgroup_size: u32,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub native_wave64: bool,
    pub subgroup_size_control: bool,
    /// `maxComputeWorkGroupCount`: dispatches beyond it are split.
    pub max_workgroup_count: [u32; 3],
    pipeline_cache: vk::PipelineCache,
    /// `(FNV-1a of the SPIR-V, required subgroup size or 0)` -> pipeline.
    pipelines: Mutex<HashMap<(u64, u32), (vk::ShaderModule, vk::Pipeline)>>,
    pub(crate) gfx_pipelines:
        Mutex<HashMap<crate::graphics::GfxKey, (vk::ShaderModule, vk::ShaderModule, vk::Pipeline)>>,
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
            let inst_ci = vk::InstanceCreateInfo::default().application_info(&app_info);
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
            let q_ci = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(qfamily)
                .queue_priorities(&prio)];

            let exts = instance.enumerate_device_extension_properties(pdev)?;
            let has_ext_host = exts.iter().any(|e| {
                std::ffi::CStr::from_ptr(e.extension_name.as_ptr())
                    == ash::ext::external_memory_host::NAME
            });

            let mut host_mem_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
            let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
            let mut subgroup_size_props =
                vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut subgroup_props)
                .push_next(&mut subgroup_size_props);
            if has_ext_host {
                props2 = props2.push_next(&mut host_mem_props);
            }
            instance.get_physical_device_properties2(pdev, &mut props2);
            let min_host_pointer_alignment = host_mem_props.min_imported_host_pointer_alignment;

            let mut avail_f12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut avail_f13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut avail_quad = vk::PhysicalDeviceShaderQuadControlFeaturesKHR::default();
            let has_quad_ext = exts.iter().any(|e| {
                std::ffi::CStr::from_ptr(e.extension_name.as_ptr())
                    == ash::khr::shader_quad_control::NAME
            });
            let mut avail_f11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut avail_feats2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut avail_f11)
                .push_next(&mut avail_f12)
                .push_next(&mut avail_f13);
            if has_quad_ext {
                avail_feats2 = avail_feats2.push_next(&mut avail_quad);
            }
            instance.get_physical_device_features2(pdev, &mut avail_feats2);
            let sampler_mirror_clamp_to_edge = avail_f12.sampler_mirror_clamp_to_edge == vk::TRUE;
            // Quad control rides on maximal reconvergence in the spec.
            let has_reconv_ext = exts.iter().any(|e| {
                std::ffi::CStr::from_ptr(e.extension_name.as_ptr())
                    == ash::khr::shader_maximal_reconvergence::NAME
            });
            let quad_control =
                has_quad_ext && has_reconv_ext && avail_quad.shader_quad_control == vk::TRUE;
            let subgroup_size_control = avail_f13.subgroup_size_control == vk::TRUE
                && subgroup_size_props
                    .required_subgroup_size_stages
                    .contains(vk::ShaderStageFlags::COMPUTE);

            let mut enabled_exts: Vec<*const i8> = Vec::new();
            if has_ext_host {
                enabled_exts.push(ash::ext::external_memory_host::NAME.as_ptr());
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

            // Slang's `SV_VertexID` lowering declares the DrawParameters capability.
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default()
                .shader_draw_parameters(avail_f11.shader_draw_parameters == vk::TRUE);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .scalar_block_layout(true)
                .sampler_mirror_clamp_to_edge(sampler_mirror_clamp_to_edge);
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .dynamic_rendering(true)
                .subgroup_size_control(subgroup_size_control);
            let core_feats = vk::PhysicalDeviceFeatures::default()
                .shader_int64(true)
                .sparse_binding(true)
                .sparse_residency_image2_d(true);

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

            let device = instance.create_device(pdev, &dev_ci, None)?;
            let ext_host_mem = has_ext_host
                .then(|| ash::ext::external_memory_host::Device::new(&instance, &device));
            let queue = device.get_device_queue(qfamily, 0);

            let pool_ci = vk::CommandPoolCreateInfo::default()
                .queue_family_index(qfamily)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let cmd_pool = device.create_command_pool(&pool_ci, None)?;

            let crate::layout::GlobalLayout {
                set_layout,
                pipeline_layout: pipe_layout,
            } = crate::layout::GlobalLayout::create(&device)?;
            let n = strata_shader::spirv::IMAGE_ARRAY_LEN;
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
            let max_workgroup_count = props.limits.max_compute_work_group_count;
            let pipeline_cache =
                device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)?;

            let mut runner = Self {
                _entry: entry,
                instance,
                pdev,
                device,
                ext_host_mem,
                min_host_pointer_alignment,
                queue,
                cmd_pool,
                pipe_layout,
                set_layout,
                desc_pool: Mutex::new(desc_pool),
                default_image: None,
                default_sampler: vk::Sampler::null(),
                sampler_mirror_clamp_to_edge,
                quad_control,
                device_name,
                subgroup_size,
                min_subgroup_size,
                max_subgroup_size,
                native_wave64,
                subgroup_size_control,
                max_workgroup_count,
                pipeline_cache,
                pipelines: Mutex::new(HashMap::new()),
                gfx_pipelines: Mutex::new(HashMap::new()),
            };
            runner.init_defaults()?;
            Ok(runner)
        }
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
        let default = self.default_image.as_ref().expect("defaults initialised");
        let n = crate::images::IMAGE_SLOTS;
        let tex: Vec<vk::DescriptorImageInfo> = (0..n)
            .map(|i| {
                let view = b
                    .textures
                    .get(i)
                    .copied()
                    .flatten()
                    .map_or(default.view, |(v, _)| v);
                vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect();
        let samp: Vec<vk::DescriptorImageInfo> = (0..n)
            .map(|i| {
                let s = b
                    .textures
                    .get(i)
                    .copied()
                    .flatten()
                    .and_then(|(_, s)| s)
                    .unwrap_or(self.default_sampler);
                vk::DescriptorImageInfo::default().sampler(s)
            })
            .collect();
        let stor: Vec<vk::DescriptorImageInfo> = (0..n)
            .map(|i| {
                let view = b.storage.get(i).copied().flatten().unwrap_or(default.view);
                vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)
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
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&tex),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&samp),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&stor),
            ];
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
        unsafe {
            let buf_ci = vk::BufferCreateInfo::default()
                .size(size)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
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
            let want =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let mem_type = (0..mem_props.memory_type_count)
                .find(|&i| {
                    (req.memory_type_bits & (1 << i)) != 0
                        && mem_props.memory_types[i as usize]
                            .property_flags
                            .contains(want)
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

            let ptr = self
                .device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?
                as *mut u8;
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

                let cmds = [cmd];
                let fence = self
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)?;
                self.device.queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&cmds)],
                    fence,
                )?;
                self.device.wait_for_fences(&[fence], true, u64::MAX)?;

                self.device.destroy_fence(fence, None);
                self.device.free_command_buffers(self.cmd_pool, &cmds);
                self.device.destroy_buffer(stage_buf, None);
                self.device.free_memory(stage_mem, None);
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

    /// Tier 3: Verify live `vkQueueBindSparse` 64 KiB tile aliasing between two
    /// sparse 2D `VkImage`s sharing the exact same `DEVICE_LOCAL` VRAM tile.
    pub fn verify_sparse_tile_aliasing(&self) -> Result<(u64, vk::Extent3D), GpuError> {
        unsafe {
            let img_ci = vk::ImageCreateInfo::default()
                .flags(
                    vk::ImageCreateFlags::SPARSE_BINDING
                        | vk::ImageCreateFlags::SPARSE_RESIDENCY
                        | vk::ImageCreateFlags::SPARSE_ALIASED,
                )
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D {
                    width: 256,
                    height: 256,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            let img_a = self.device.create_image(&img_ci, None)?;
            let img_b = self.device.create_image(&img_ci, None)?;

            let req = self.device.get_image_memory_requirements(img_a);
            let sparse_reqs = self.device.get_image_sparse_memory_requirements(img_a);
            let tile_granularity = sparse_reqs
                .first()
                .map(|r| r.format_properties.image_granularity)
                .unwrap_or(vk::Extent3D {
                    width: 128,
                    height: 128,
                    depth: 1,
                });

            // Allocate a single 64 KiB (or req.alignment) DEVICE_LOCAL physical tile
            let tile_bytes = req.alignment.max(64 * 1024);
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
            let shared_tile_mem = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(tile_bytes)
                    .memory_type_index(mem_type),
                None,
            )?;

            // Bind the EXACT SAME physical 64 KiB tile to (0,0,0) of both img_a and img_b!
            let bind_a = [vk::SparseImageMemoryBind::default()
                .subresource(vk::ImageSubresource {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    array_layer: 0,
                })
                .offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .extent(tile_granularity)
                .memory(shared_tile_mem)
                .memory_offset(0)];
            let bind_b = bind_a;

            let img_binds = [
                vk::SparseImageMemoryBindInfo::default()
                    .image(img_a)
                    .binds(&bind_a),
                vk::SparseImageMemoryBindInfo::default()
                    .image(img_b)
                    .binds(&bind_b),
            ];
            let sparse_info = [vk::BindSparseInfo::default().image_binds(&img_binds)];

            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
            self.device
                .queue_bind_sparse(self.queue, &sparse_info, fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;

            self.device.destroy_fence(fence, None);
            self.device.destroy_image(img_a, None);
            self.device.destroy_image(img_b, None);
            self.device.free_memory(shared_tile_mem, None);

            Ok((tile_bytes, tile_granularity))
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
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipe_ci, None)
                .map_err(|(_, e)| e)?[0];

            // 3. Readback buffer (1 pixel = 4 bytes, padded to 256)
            let readback = self.alloc_bda_buffer(256)?;

            // 4. Record dynamic rendering pass + copy pixel (0,0) to readback
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

            let cmds = [cmd];
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                fence,
            )?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;

            let px_slice = readback.as_slice::<u8>(4);
            let pixel = [px_slice[0], px_slice[1], px_slice[2], px_slice[3]];

            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.cmd_pool, &cmds);
            self.free_bda_buffer(readback);
            self.device.destroy_pipeline(pipeline, None);
            self.device.destroy_shader_module(vs_mod, None);
            self.device.destroy_shader_module(fs_mod, None);
            self.device.destroy_image_view(view, None);
            self.device.destroy_image(img, None);
            self.device.free_memory(img_mem, None);

            Ok(pixel)
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
        let chunks = crate::thin_list::split_dispatch(groups, self.max_workgroup_count);
        // SAFETY: freed below, after the submission is waited on.
        let set = unsafe { self.image_set(images)? };
        let result = unsafe { self.dispatch_recorded(pipeline, set, push_bytes, &chunks) };
        unsafe { self.free_image_set(set) };
        result
    }

    /// # Safety
    /// `set` stays valid until this returns.
    unsafe fn dispatch_recorded(
        &self,
        pipeline: vk::Pipeline,
        set: vk::DescriptorSet,
        push_bytes: &[u8],
        chunks: &[([u32; 3], [u32; 3])],
    ) -> Result<(), GpuError> {
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
            if !push_bytes.is_empty() {
                self.device
                    .cmd_push_constants(cmd, self.pipe_layout, PUSH_STAGES, 0, push_bytes);
            }
            for &(base, count) in chunks {
                self.device.cmd_dispatch_base(
                    cmd, base[0], base[1], base[2], count[0], count[1], count[2],
                );
            }
            let host_read = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
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
        let mut table = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&(_, p)) = table.get(&key) {
            return Ok(p);
        }
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
                    self.device.destroy_shader_module(module, None);
                    return Err(e.into());
                }
            };
            table.insert(key, (module, pipeline));
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

    pub fn cached_pipeline_count(&self) -> usize {
        self.pipelines
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Merges a `VkPipelineCache` blob saved by [`Self::save_pipeline_cache`].
    /// A blob from another driver or device is rejected and ignored.
    pub fn load_pipeline_cache(&self, path: &Path) -> Result<bool, GpuError> {
        let Ok(data) = std::fs::read(path) else {
            return Ok(false);
        };
        let _table = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
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
            let _table = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
            unsafe { self.device.get_pipeline_cache_data(self.pipeline_cache)? }
        };
        let io = |e: std::io::Error| GpuError::Io(e.to_string());
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(io)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &data).map_err(io)?;
        std::fs::rename(&tmp, path).map_err(io)?;
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
            self.device.end_command_buffer(cmd)?;
            self.submit_and_wait(cmd)
        }
    }

    /// # Safety
    /// `cmd` is a recorded primary command buffer from `cmd_pool`.
    pub(crate) unsafe fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<(), GpuError> {
        unsafe {
            let cmds = [cmd];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
            let result = self
                .device
                .queue_submit(self.queue, &submit, fence)
                .and_then(|()| self.device.wait_for_fences(&[fence], true, u64::MAX));
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.cmd_pool, &cmds);
            result.map_err(Into::into)
        }
    }
}

impl Drop for ComputeRunner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            let table = self.pipelines.get_mut().unwrap_or_else(|e| e.into_inner());
            for (_, (module, pipeline)) in table.drain() {
                self.device.destroy_pipeline(pipeline, None);
                self.device.destroy_shader_module(module, None);
            }
            let gfx = self
                .gfx_pipelines
                .get_mut()
                .unwrap_or_else(|e| e.into_inner());
            for (_, (vsm, psm, pipeline)) in gfx.drain() {
                self.device.destroy_pipeline(pipeline, None);
                self.device.destroy_shader_module(vsm, None);
                self.device.destroy_shader_module(psm, None);
            }
            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);
            if let Some(img) = self.default_image.take() {
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
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
