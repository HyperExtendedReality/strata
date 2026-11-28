//! A hardware closest-hit query for opaque triangle geometry. The returned
//! primitive index is the ordinal in the supplied triangle array.

use ash::vk;

use crate::context::GpuError;
use crate::runner::{BdaBuffer, ComputeRunner};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleHit {
    pub primitive: u32,
    pub t: f32,
    pub barycentrics: [f32; 2],
}

#[derive(Default, Debug, Clone, Copy)]
pub struct RayQueryTiming {
    pub build: std::time::Duration,
    pub trace: std::time::Duration,
}

struct QueryScene<'a> {
    runner: &'a ComputeRunner,
    api: &'a ash::khr::acceleration_structure::Device,
    structures: Vec<vk::AccelerationStructureKHR>,
    buffers: Vec<BdaBuffer>,
}

impl Drop for QueryScene<'_> {
    fn drop(&mut self) {
        for handle in self.structures.drain(..).rev() {
            unsafe { self.api.destroy_acceleration_structure(handle, None) };
        }
        for buffer in self.buffers.drain(..) {
            self.runner.free_bda_buffer(buffer);
        }
    }
}

impl QueryScene<'_> {
    fn buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<&mut BdaBuffer, GpuError> {
        self.buffers
            .push(self.runner.alloc_bda_buffer_with_usage(size, usage)?);
        Ok(self.buffers.last_mut().expect("just allocated"))
    }

    fn build(
        &mut self,
        kind: vk::AccelerationStructureTypeKHR,
        geometry: vk::AccelerationStructureGeometryKHR<'_>,
        count: u32,
    ) -> Result<u64, GpuError> {
        let geometries = [geometry];
        let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(kind)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.api.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info,
                &[count],
                &mut sizes,
            );
        }
        let storage = self
            .buffer(
                sizes.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            )?
            .buffer;
        let handle = unsafe {
            self.api.create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default()
                    .ty(kind)
                    .size(sizes.acceleration_structure_size)
                    .buffer(storage),
                None,
            )?
        };
        self.structures.push(handle);
        let alignment = self.runner.acceleration_scratch_alignment.max(1);
        let scratch = self
            .buffer(
                sizes.build_scratch_size.max(1) + alignment - 1,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?
            .bda;
        info = info
            .dst_acceleration_structure(handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch.next_multiple_of(alignment),
            });
        let range = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count)];
        self.runner.one_shot(|dev, cmd| unsafe {
            self.api
                .cmd_build_acceleration_structures(cmd, &[info], &[&range]);
            let visibility = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
                        | vk::PipelineStageFlags2::COMPUTE_SHADER,
                )
                .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)];
            dev.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&visibility),
            );
        })?;
        Ok(unsafe {
            self.api.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(handle),
            )
        })
    }
}

impl ComputeRunner {
    pub fn query_opaque_triangles(
        &self,
        triangles: &[[[f32; 3]; 3]],
        origin: [f32; 3],
        direction: [f32; 3],
        t_max: f32,
    ) -> Result<Option<TriangleHit>, GpuError> {
        self.query_opaque_triangles_timed(triangles, origin, direction, t_max)
            .map(|(hit, _)| hit)
    }

    pub fn query_opaque_triangles_timed(
        &self,
        triangles: &[[[f32; 3]; 3]],
        origin: [f32; 3],
        direction: [f32; 3],
        t_max: f32,
    ) -> Result<(Option<TriangleHit>, RayQueryTiming), GpuError> {
        let api = self
            .acceleration_structure
            .as_ref()
            .filter(|_| self.ray_query)
            .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
        if triangles.is_empty() {
            return Ok((None, RayQueryTiming::default()));
        }
        let vertex_count = u32::try_from(triangles.len().saturating_mul(3))
            .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let primitive_count = vertex_count / 3;
        let mut limits = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        unsafe {
            self.instance.get_physical_device_properties2(
                self.pdev,
                &mut vk::PhysicalDeviceProperties2::default().push_next(&mut limits),
            );
        }
        if u64::from(primitive_count) > limits.max_primitive_count
            || limits.max_instance_count == 0
            || !origin.iter().chain(&direction).all(|n| n.is_finite())
            || direction == [0.0; 3]
            || !t_max.is_finite()
            || t_max < 0.0
            || !triangles.iter().flatten().flatten().all(|n| n.is_finite())
        {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED.into());
        }
        let build = std::time::Instant::now();
        let mut scene = QueryScene {
            runner: self,
            api,
            structures: Vec::new(),
            buffers: Vec::new(),
        };
        let input_usage = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        let vertices = scene.buffer(std::mem::size_of_val(triangles) as u64, input_usage)?;
        unsafe { vertices.as_mut_slice(triangles.len()) }.copy_from_slice(triangles);
        let mesh = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vertices.bda,
            })
            .vertex_stride(12)
            .max_vertex(vertex_count - 1)
            .index_type(vk::IndexType::NONE_KHR);
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: mesh });
        let blas = scene.build(
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            geometry,
            primitive_count,
        )?;

        let instance = scene.buffer(
            std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64,
            input_usage,
        )?;
        (unsafe { instance.as_mut_slice(1) })[0] = vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas,
            },
        };
        let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default().data(
            vk::DeviceOrHostAddressConstKHR {
                device_address: instance.bda,
            },
        );
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { instances });
        let tlas = scene.build(vk::AccelerationStructureTypeKHR::TOP_LEVEL, geometry, 1)?;
        let build = build.elapsed();
        let trace = std::time::Instant::now();

        let output = scene.buffer(16, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let mut push = Vec::with_capacity(48);
        push.extend_from_slice(&tlas.to_le_bytes());
        push.extend_from_slice(&output.bda.to_le_bytes());
        for value in origin
            .into_iter()
            .chain([0.0])
            .chain(direction)
            .chain([t_max])
        {
            push.extend_from_slice(&value.to_le_bytes());
        }
        static SHADER: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
        let words = SHADER.get_or_init(|| {
            strata_runtime_shaders::RAY_PROBE_SPV
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(u32::from_le_bytes)
                .collect()
        });
        self.dispatch_spirv(words, &push, [1, 1, 1])?;
        let result = unsafe { output.as_slice::<u32>(4) };
        let hit = (result[0] != u32::MAX).then(|| TriangleHit {
            primitive: result[0],
            t: f32::from_bits(result[1]),
            barycentrics: [f32::from_bits(result[2]), f32::from_bits(result[3])],
        });
        Ok((
            hit,
            RayQueryTiming {
                build,
                trace: trace.elapsed(),
            },
        ))
    }
}
