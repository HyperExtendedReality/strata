//! Images, samplers and the descriptor set compute and graphics shaders
//! index. Set 0 has one array per image class (a SPIR-V image type fixes its
//! dimensionality), all `IMAGE_ARRAY_LEN` long and indexed by the same slot:
//!
//! | binding | contents |
//! | --- | --- |
//! | 0 | sampled 2D |
//! | 1 | samplers |
//! | 2 | storage 2D |
//! | 3 | sampled 3D |
//! | 4 | sampled cube |
//! | 5 | sampled 2D array |
//! | 6 | storage 3D |
//! | 7 | storage 2D array |
//!
//! Images live in `GENERAL` layout once created, valid for sampling, storage
//! writes, attachments and copies alike; every use is a whole synchronous
//! submission, so barriers are full-image and coarse.

use ash::vk;
use strata_shader::descriptor::{Clamp, Filter, SamplerDescriptor};
use strata_shader::resource::ImageDim;

use crate::context::GpuError;
use crate::runner::ComputeRunner;

/// Slots per binding of the global descriptor set.
pub const IMAGE_SLOTS: usize = strata_shader::spirv::IMAGE_ARRAY_LEN as usize;

/// What kind of Vulkan image to make.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageShape {
    /// 2D with `layers` array layers (1 for a plain 2D image).
    D2 { layers: u32 },
    /// 3D with `depth` slices.
    D3 { depth: u32 },
    /// `cubes` cubes (6 layers each), cube-compatible.
    Cube { cubes: u32 },
}

impl ImageShape {
    /// Array layers of the Vulkan image.
    pub fn layers(self) -> u32 {
        match self {
            ImageShape::D2 { layers } => layers.max(1),
            ImageShape::D3 { .. } => 1,
            ImageShape::Cube { cubes } => cubes.max(1) * 6,
        }
    }

    /// Depth of the Vulkan image's extent.
    pub fn depth(self) -> u32 {
        match self {
            ImageShape::D3 { depth } => depth.max(1),
            _ => 1,
        }
    }

    /// Slices a level holds (layers, or 3D slices): what upload data repeats.
    pub fn slices(self) -> u32 {
        self.layers() * self.depth()
    }
}

/// Everything [`ComputeRunner::create_image`] needs.
#[derive(Debug, Clone, Copy)]
pub struct ImageSpec {
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub shape: ImageShape,
    pub mips: u32,
    /// Created with `STORAGE` usage (views are then identity-swizzled).
    pub storage: bool,
    pub components: vk::ComponentMapping,
}

/// A device-local image and views of it for every class it can be read as.
pub struct GpuImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    /// The view of the image's own class (2D, 2D array, 3D or cube).
    pub view: vk::ImageView,
    /// Additional views: `[2D (layer 0), 2D array (all layers)]`, where the
    /// image allows them (not 3D).
    pub alt_views: [Option<vk::ImageView>; 2],
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub shape: ImageShape,
    pub mips: u32,
    /// Created with `STORAGE` usage (and identity views).
    pub storage: bool,
}

impl GpuImage {
    /// The view to bind for an instruction of class `dim`, if this image
    /// can be seen that way.
    pub fn view_for(&self, dim: ImageDim) -> Option<vk::ImageView> {
        match (self.shape, dim) {
            (ImageShape::D3 { .. }, ImageDim::D3) => Some(self.view),
            (ImageShape::D3 { .. }, _) => None,
            (ImageShape::Cube { .. }, ImageDim::Cube) => Some(self.view),
            (_, ImageDim::Cube) => None,
            (_, ImageDim::D3) => None,
            (ImageShape::D2 { layers }, ImageDim::D2) if layers <= 1 => Some(self.view),
            (ImageShape::D2 { layers }, ImageDim::D2Array) if layers > 1 => Some(self.view),
            (_, ImageDim::D2) => self.alt_views[0],
            (_, ImageDim::D2Array) => self.alt_views[1],
        }
    }

    /// Bytes of tightly packed texels (or blocks) of every level and slice.
    pub fn tight_bytes(&self, bytes_per_element: usize, block: u32) -> usize {
        (0..self.mips)
            .map(|m| {
                let (w, h) = level_extent(self.width, self.height, m);
                w.div_ceil(block) as usize
                    * h.div_ceil(block) as usize
                    * bytes_per_element
                    * self.level_slices(m) as usize
            })
            .sum()
    }

    /// Slices of level `m` (3D images halve their depth per level).
    pub fn level_slices(&self, m: u32) -> u32 {
        match self.shape {
            ImageShape::D3 { depth } => (depth >> m).max(1),
            s => s.layers(),
        }
    }
}

/// `(width, height)` of level `m`, at least 1×1.
pub fn level_extent(width: u32, height: u32, m: u32) -> (u32, u32) {
    ((width >> m).max(1), (height >> m).max(1))
}

/// What a dispatch or draw binds, by slot; `None` slots get a default image
/// of the right class.
#[derive(Default, Clone)]
pub struct ImageBindings {
    pub textures: Vec<Option<(vk::ImageView, Option<vk::Sampler>, ImageDim)>>,
    pub storage: Vec<Option<(vk::ImageView, ImageDim)>>,
}

impl ImageBindings {
    /// A 2D texture at `slot`.
    pub fn set_texture(&mut self, slot: u32, view: vk::ImageView, sampler: Option<vk::Sampler>) {
        self.set_texture_dim(slot, view, sampler, ImageDim::D2);
    }

    /// A texture of class `dim` at `slot`.
    pub fn set_texture_dim(
        &mut self,
        slot: u32,
        view: vk::ImageView,
        sampler: Option<vk::Sampler>,
        dim: ImageDim,
    ) {
        let i = slot as usize;
        if self.textures.len() <= i {
            self.textures.resize(i + 1, None);
        }
        self.textures[i] = Some((view, sampler, dim));
    }

    /// A 2D storage image at `slot`.
    pub fn set_storage(&mut self, slot: u32, view: vk::ImageView) {
        self.set_storage_dim(slot, view, ImageDim::D2);
    }

    /// A storage image of class `dim` at `slot`.
    pub fn set_storage_dim(&mut self, slot: u32, view: vk::ImageView, dim: ImageDim) {
        let i = slot as usize;
        if self.storage.len() <= i {
            self.storage.resize(i + 1, None);
        }
        self.storage[i] = Some((view, dim.for_storage()));
    }
}

fn clamp_mode(c: Clamp, mirror_once: bool) -> vk::SamplerAddressMode {
    match c {
        Clamp::Wrap => vk::SamplerAddressMode::REPEAT,
        Clamp::Mirror => vk::SamplerAddressMode::MIRRORED_REPEAT,
        Clamp::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        Clamp::MirrorOnceToEdge if mirror_once => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
        Clamp::MirrorOnceToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        Clamp::ClampToBorder | Clamp::MirrorOnceToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
    }
}

fn filter(f: Filter) -> vk::Filter {
    match f {
        Filter::Point => vk::Filter::NEAREST,
        Filter::Linear => vk::Filter::LINEAR,
    }
}

/// The aspect of a format's image (depth/stencil formats are not colour).
pub fn aspect_of(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT
        | vk::Format::D16_UNORM_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

/// The aspect sampled / copied as texel data (depth for depth/stencil).
fn data_aspect(format: vk::Format) -> vk::ImageAspectFlags {
    let a = aspect_of(format);
    if a.contains(vk::ImageAspectFlags::DEPTH) {
        vk::ImageAspectFlags::DEPTH
    } else {
        a
    }
}

impl ComputeRunner {
    /// Whether an optimal-tiling image of `format` supports `features`.
    pub fn format_supports(&self, format: vk::Format, features: vk::FormatFeatureFlags) -> bool {
        // SAFETY: a property query.
        let props = unsafe {
            self.instance
                .get_physical_device_format_properties(self.pdev, format)
        };
        props.optimal_tiling_features.contains(features)
    }

    fn device_local_type(&self, bits: u32) -> Result<u32, GpuError> {
        // SAFETY: a property query.
        let mem = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.pdev)
        };
        (0..mem.memory_type_count)
            .find(|&i| {
                bits & (1 << i) != 0
                    && mem.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or(GpuError::NoDevice)
    }

    /// Creates a `width`×`height` image of `format`, seen through `components`
    /// (ignored for storage images, whose views must be identity).
    pub fn create_image_2d(
        &self,
        format: vk::Format,
        width: u32,
        height: u32,
        storage: bool,
        components: vk::ComponentMapping,
    ) -> Result<GpuImage, GpuError> {
        self.create_image(&ImageSpec {
            format,
            width,
            height,
            shape: ImageShape::D2 { layers: 1 },
            mips: 1,
            storage,
            components,
        })
    }

    /// Creates an image of any [`ImageShape`] with `spec.mips` levels, and
    /// the views every class it can be bound as needs.
    pub fn create_image(&self, spec: &ImageSpec) -> Result<GpuImage, GpuError> {
        let format = spec.format;
        let aspect = aspect_of(format);
        let is_depth = aspect.contains(vk::ImageAspectFlags::DEPTH)
            || aspect.contains(vk::ImageAspectFlags::STENCIL);
        let mut usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC;
        if spec.storage {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if is_depth {
            if self.format_supports(format, vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
                usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
            }
        } else if self.format_supports(format, vk::FormatFeatureFlags::COLOR_ATTACHMENT) {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT; // render targets are textures too
        }
        let (image_type, flags) = match spec.shape {
            ImageShape::D3 { .. } => (vk::ImageType::TYPE_3D, vk::ImageCreateFlags::empty()),
            ImageShape::Cube { .. } => (
                vk::ImageType::TYPE_2D,
                vk::ImageCreateFlags::CUBE_COMPATIBLE,
            ),
            ImageShape::D2 { .. } => (vk::ImageType::TYPE_2D, vk::ImageCreateFlags::empty()),
        };
        let max_mips = 32
            - spec
                .width
                .max(spec.height)
                .max(spec.shape.depth())
                .leading_zeros();
        let mips = spec.mips.clamp(1, max_mips.max(1));
        let layers = spec.shape.layers();
        let components = if spec.storage {
            vk::ComponentMapping::default()
        } else {
            spec.components
        };
        // SAFETY: plain object creation on our device.
        unsafe {
            let ci = vk::ImageCreateInfo::default()
                .flags(flags)
                .image_type(image_type)
                .format(format)
                .extent(vk::Extent3D {
                    width: spec.width,
                    height: spec.height,
                    depth: spec.shape.depth(),
                })
                .mip_levels(mips)
                .array_layers(layers)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = self.device.create_image(&ci, None)?;
            let req = self.device.get_image_memory_requirements(image);
            let memory = match self.device_local_type(req.memory_type_bits).and_then(|t| {
                self.device
                    .allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(req.size)
                            .memory_type_index(t),
                        None,
                    )
                    .map_err(GpuError::from)
            }) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_image(image, None);
                    return Err(e);
                }
            };
            self.device.bind_image_memory(image, memory, 0)?;
            let view_aspect = data_aspect(format);
            let make_view = |ty: vk::ImageViewType, layer_count: u32| {
                self.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(ty)
                        .format(format)
                        .components(components)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(view_aspect)
                                .level_count(mips)
                                .layer_count(layer_count),
                        ),
                    None,
                )
            };
            let (view, alt_views) = match spec.shape {
                ImageShape::D3 { .. } => (make_view(vk::ImageViewType::TYPE_3D, 1)?, [None, None]),
                ImageShape::Cube { cubes } => {
                    let ty = if cubes > 1 {
                        vk::ImageViewType::CUBE_ARRAY
                    } else {
                        vk::ImageViewType::CUBE
                    };
                    // A cube view cannot be a storage view: storage binds the array.
                    let main = if spec.storage {
                        make_view(vk::ImageViewType::TYPE_2D_ARRAY, layers)?
                    } else {
                        make_view(ty, layers)?
                    };
                    (
                        main,
                        [
                            Some(make_view(vk::ImageViewType::TYPE_2D, 1)?),
                            Some(make_view(vk::ImageViewType::TYPE_2D_ARRAY, layers)?),
                        ],
                    )
                }
                ImageShape::D2 { layers } if layers > 1 => (
                    make_view(vk::ImageViewType::TYPE_2D_ARRAY, layers)?,
                    [Some(make_view(vk::ImageViewType::TYPE_2D, 1)?), None],
                ),
                ImageShape::D2 { .. } => (
                    make_view(vk::ImageViewType::TYPE_2D, 1)?,
                    [None, Some(make_view(vk::ImageViewType::TYPE_2D_ARRAY, 1)?)],
                ),
            };
            Ok(GpuImage {
                image,
                memory,
                view,
                alt_views,
                format,
                width: spec.width,
                height: spec.height,
                shape: spec.shape,
                mips,
                storage: spec.storage,
            })
        }
    }

    /// Replaces level 0 of a single-slice image with tightly packed rows
    /// and leaves it in `GENERAL` layout.
    pub fn upload_image(&self, img: &GpuImage, tight: &[u8]) -> Result<(), GpuError> {
        self.upload_image_levels(img, tight, 1)
    }

    /// Replaces the image's contents with `tight`: for each level from 0,
    /// each slice (layer or 3D slice) of tightly packed rows (or BCn block
    /// rows). Only the levels `tight` covers are written.
    pub fn upload_image_levels(
        &self,
        img: &GpuImage,
        tight: &[u8],
        block: u32,
    ) -> Result<(), GpuError> {
        let regions = copy_regions(img, tight.len(), block);
        let mut staging = self.alloc_bda_buffer(tight.len().max(4) as u64)?;
        // SAFETY: the staging buffer holds `tight.len()` bytes.
        unsafe {
            staging
                .as_mut_slice::<u8>(tight.len())
                .copy_from_slice(tight)
        };
        let buffer = staging.buffer;
        let result = self.one_shot(|dev, cmd| unsafe {
            barrier_image(
                dev,
                cmd,
                img,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            dev.cmd_copy_buffer_to_image(
                cmd,
                buffer,
                img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
            barrier_image(
                dev,
                cmd,
                img,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
        });
        self.free_bda_buffer(staging);
        result
    }

    /// Level 0 of a single-slice image as tightly packed rows of `bytes` total.
    pub fn read_image(&self, img: &GpuImage, bytes: usize) -> Result<Vec<u8>, GpuError> {
        self.read_image_levels(img, bytes, 1)
    }

    /// The image's levels and slices in [`Self::upload_image_levels`]'s
    /// layout, `bytes` total.
    pub fn read_image_levels(
        &self,
        img: &GpuImage,
        bytes: usize,
        block: u32,
    ) -> Result<Vec<u8>, GpuError> {
        let regions = copy_regions(img, bytes, block);
        let staging = self.alloc_bda_buffer(bytes.max(4) as u64)?;
        let buffer = staging.buffer;
        let result = self.one_shot(|dev, cmd| unsafe {
            barrier_image(
                dev,
                cmd,
                img,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
            );
            dev.cmd_copy_image_to_buffer(
                cmd,
                img.image,
                vk::ImageLayout::GENERAL,
                buffer,
                &regions,
            );
        });
        // SAFETY: the copy above finished (one_shot waits).
        let out = result.map(|()| unsafe { staging.as_slice::<u8>(bytes).to_vec() });
        self.free_bda_buffer(staging);
        out
    }

    /// Fills every level and slice of `img` with one clear value
    /// (`vkCmdClearColorImage` / `vkCmdClearDepthStencilImage`).
    pub fn clear_image(
        &self,
        img: &GpuImage,
        color: [u32; 4],
        depth: f32,
        stencil: u32,
    ) -> Result<(), GpuError> {
        let aspect = aspect_of(img.format);
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect)
            .level_count(img.mips)
            .layer_count(img.shape.layers());
        self.one_shot(|dev, cmd| unsafe {
            barrier_image(
                dev,
                cmd,
                img,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
            );
            if aspect == vk::ImageAspectFlags::COLOR {
                dev.cmd_clear_color_image(
                    cmd,
                    img.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue { uint32: color },
                    &[range],
                );
            } else {
                dev.cmd_clear_depth_stencil_image(
                    cmd,
                    img.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearDepthStencilValue { depth, stencil },
                    &[range],
                );
            }
            barrier_image(
                dev,
                cmd,
                img,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
            );
        })
    }

    pub fn destroy_image(&self, img: GpuImage) {
        // SAFETY: no submission uses it (all submissions are waited on).
        unsafe {
            for v in img.alt_views.into_iter().flatten() {
                self.device.destroy_image_view(v, None);
            }
            self.device.destroy_image_view(img.view, None);
            self.device.destroy_image(img.image, None);
            self.device.free_memory(img.memory, None);
        }
    }

    /// A sampler for an S#. Anisotropy is not enabled on this device, so
    /// anisotropic S#s filter as their min/mag/mip filters say. A depth
    /// comparison function makes a comparison sampler.
    pub fn create_sampler(&self, s: &SamplerDescriptor) -> Result<vk::Sampler, GpuError> {
        let mirror_once = self.sampler_mirror_clamp_to_edge;
        let (mip_mode, max_lod) = match s.mip {
            None => (vk::SamplerMipmapMode::NEAREST, 0.25), // base level only
            Some(Filter::Point) => (vk::SamplerMipmapMode::NEAREST, s.max_lod),
            Some(Filter::Linear) => (vk::SamplerMipmapMode::LINEAR, s.max_lod),
        };
        let mut ci = vk::SamplerCreateInfo::default()
            .mag_filter(filter(s.mag))
            .min_filter(filter(s.min))
            .mipmap_mode(mip_mode)
            .address_mode_u(clamp_mode(s.clamp[0], mirror_once))
            .address_mode_v(clamp_mode(s.clamp[1], mirror_once))
            .address_mode_w(clamp_mode(s.clamp[2], mirror_once))
            .mip_lod_bias(s.lod_bias)
            .min_lod(s.min_lod)
            .max_lod(max_lod.max(s.min_lod))
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK);
        if let Some(func) = s.compare {
            ci = ci
                .compare_enable(true)
                .compare_op(vk::CompareOp::from_raw(i32::from(func)));
        }
        // SAFETY: plain object creation.
        Ok(unsafe { self.device.create_sampler(&ci, None)? })
    }

    pub fn destroy_sampler(&self, s: vk::Sampler) {
        // SAFETY: no submission uses it.
        unsafe { self.device.destroy_sampler(s, None) };
    }

    /// Records `f` into a one-time command buffer, submits it and waits.
    pub(crate) fn one_shot(
        &self,
        f: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), GpuError> {
        // SAFETY: a command buffer from our pool, submitted and waited on.
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
            f(&self.device, cmd);
            self.device.end_command_buffer(cmd)?;
            self.submit_and_wait(cmd)
        }
    }
}

/// Buffer ↔ image regions for the levels `bytes` covers, each level's slices
/// following one another (`block`: 4 for BCn data, else 1).
fn copy_regions(img: &GpuImage, bytes: usize, block: u32) -> Vec<vk::BufferImageCopy> {
    let aspect = data_aspect(img.format);
    let bpe = element_bytes(img.format);
    let mut offset = 0usize;
    let mut out = Vec::new();
    for m in 0..img.mips {
        let (w, h) = level_extent(img.width, img.height, m);
        let level = w.div_ceil(block) as usize
            * h.div_ceil(block) as usize
            * bpe
            * img.level_slices(m) as usize;
        if offset + level > bytes {
            break;
        }
        let (layers, depth) = match img.shape {
            ImageShape::D3 { .. } => (1, img.level_slices(m)),
            s => (s.layers(), 1),
        };
        out.push(
            vk::BufferImageCopy::default()
                .buffer_offset(offset as u64)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(aspect)
                        .mip_level(m)
                        .layer_count(layers),
                )
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth,
                }),
        );
        offset += level;
    }
    out
}

/// Bytes per texel (or per 4×4 block for BCn) of the formats Strata makes.
pub fn element_bytes(format: vk::Format) -> usize {
    use vk::Format as F;
    match format {
        F::R8_UNORM | F::R8_SNORM | F::R8_UINT | F::R8_SINT | F::S8_UINT => 1,
        F::R8G8_UNORM
        | F::R8G8_SNORM
        | F::R8G8_UINT
        | F::R8G8_SINT
        | F::R16_UNORM
        | F::R16_SNORM
        | F::R16_SFLOAT
        | F::R16_UINT
        | F::R16_SINT
        | F::D16_UNORM
        | F::B5G6R5_UNORM_PACK16
        | F::A1R5G5B5_UNORM_PACK16 => 2,
        F::R16G16B16A16_UNORM
        | F::R16G16B16A16_SNORM
        | F::R16G16B16A16_SFLOAT
        | F::R16G16B16A16_UINT
        | F::R16G16B16A16_SINT
        | F::R32G32_SFLOAT
        | F::R32G32_UINT
        | F::R32G32_SINT
        | F::BC1_RGBA_UNORM_BLOCK
        | F::BC1_RGBA_SRGB_BLOCK
        | F::BC4_UNORM_BLOCK
        | F::BC4_SNORM_BLOCK
        | F::D32_SFLOAT_S8_UINT => 8,
        F::R32G32B32_SFLOAT | F::R32G32B32_UINT | F::R32G32B32_SINT => 12,
        F::R32G32B32A32_SFLOAT
        | F::R32G32B32A32_UINT
        | F::R32G32B32A32_SINT
        | F::BC2_UNORM_BLOCK
        | F::BC2_SRGB_BLOCK
        | F::BC3_UNORM_BLOCK
        | F::BC3_SRGB_BLOCK
        | F::BC5_UNORM_BLOCK
        | F::BC5_SNORM_BLOCK
        | F::BC6H_UFLOAT_BLOCK
        | F::BC6H_SFLOAT_BLOCK
        | F::BC7_UNORM_BLOCK
        | F::BC7_SRGB_BLOCK => 16,
        _ => 4,
    }
}

/// A full-image barrier (every level, layer and aspect of `img`) between
/// any earlier and any later work.
///
/// # Safety
/// `cmd` is recording.
pub(crate) unsafe fn barrier_image(
    dev: &ash::Device,
    cmd: vk::CommandBuffer,
    img: &GpuImage,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    unsafe {
        barrier_range(
            dev,
            cmd,
            img.image,
            from,
            to,
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect_of(img.format))
                .level_count(vk::REMAINING_MIP_LEVELS)
                .layer_count(vk::REMAINING_ARRAY_LAYERS),
        )
    }
}

unsafe fn barrier_range(
    dev: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
    range: vk::ImageSubresourceRange,
) {
    let b = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::HOST)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::HOST_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::HOST)
        .dst_access_mask(
            vk::AccessFlags2::MEMORY_READ
                | vk::AccessFlags2::MEMORY_WRITE
                | vk::AccessFlags2::HOST_READ,
        )
        .old_layout(from)
        .new_layout(to)
        .image(image)
        .subresource_range(range)];
    // SAFETY: forwarded from the caller.
    unsafe {
        dev.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&b),
        )
    };
}
