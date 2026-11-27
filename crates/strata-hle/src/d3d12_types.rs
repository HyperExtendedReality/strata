//! `d3d12.h` ABI structures (x64 layout, checked by size assertions), the
//! DXGI → GFX10 format mapping, and the builders that turn D3D12 view
//! descriptions into the RDNA2 descriptors translated shaders read.
//!
//! No-API model: a D3D12 descriptor heap is plain title-visible memory and a
//! CPU descriptor handle is a pointer into it. `CreateShaderResourceView`
//! and friends write the console's own descriptor dwords there — T# (8
//! dwords) for images, V# (4 dwords) for buffers, S# (4 dwords) for
//! samplers — exactly what an RDNA2 shader loads with `s_load_dwordx8` from
//! a table. Nothing is created on the Vulkan side until a draw or dispatch
//! reads a descriptor (see `d3d12x::bind_images`).
//!
//! Render-target and depth-stencil heaps are CPU-only in D3D12; Strata
//! stores a T# there too, plus a private trailer in dwords 5–7 (`'ST'`
//! magic, flags, Vulkan depth format) that only the host reads.

use strata_shader::descriptor::{
    Clamp, Filter, ImageDescriptor, ImageKind, SamplerDescriptor, Select,
};

// ============================================================================
// Plain ABI structures
// ============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShaderBytecode {
    pub ptr: *const u8,
    pub len: usize,
}

impl ShaderBytecode {
    /// The bytecode bytes (empty when absent).
    ///
    /// # Safety
    /// `ptr..ptr+len` is readable when non-null.
    pub unsafe fn bytes<'a>(&self) -> &'a [u8] {
        if self.ptr.is_null() || self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StreamOutputDesc {
    pub decl: *const u8,
    pub num_entries: u32,
    pub strides: *const u32,
    pub num_strides: u32,
    pub rasterized_stream: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderTargetBlendDesc {
    pub blend_enable: i32,
    pub logic_op_enable: i32,
    pub src_blend: u32,
    pub dest_blend: u32,
    pub blend_op: u32,
    pub src_blend_alpha: u32,
    pub dest_blend_alpha: u32,
    pub blend_op_alpha: u32,
    pub logic_op: u32,
    pub write_mask: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct BlendDesc {
    pub alpha_to_coverage: i32,
    pub independent_blend: i32,
    pub render_target: [RenderTargetBlendDesc; 8],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RasterizerDesc {
    pub fill_mode: u32,
    pub cull_mode: u32,
    pub front_counter_clockwise: i32,
    pub depth_bias: i32,
    pub depth_bias_clamp: f32,
    pub slope_scaled_depth_bias: f32,
    pub depth_clip_enable: i32,
    pub multisample_enable: i32,
    pub antialiased_line_enable: i32,
    pub forced_sample_count: u32,
    pub conservative_raster: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DepthStencilOpDesc {
    pub fail_op: u32,
    pub depth_fail_op: u32,
    pub pass_op: u32,
    pub func: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DepthStencilDesc {
    pub depth_enable: i32,
    pub depth_write_mask: u32,
    pub depth_func: u32,
    pub stencil_enable: i32,
    pub stencil_read_mask: u8,
    pub stencil_write_mask: u8,
    pub front: DepthStencilOpDesc,
    pub back: DepthStencilOpDesc,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InputLayoutDesc {
    pub elements: *const u8,
    pub num_elements: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CachedPipelineState {
    pub blob: *const u8,
    pub size: usize,
}

/// `D3D12_GRAPHICS_PIPELINE_STATE_DESC`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GraphicsPipelineStateDesc {
    pub root_signature: *mut core::ffi::c_void,
    pub vs: ShaderBytecode,
    pub ps: ShaderBytecode,
    pub ds: ShaderBytecode,
    pub hs: ShaderBytecode,
    pub gs: ShaderBytecode,
    pub stream_output: StreamOutputDesc,
    pub blend: BlendDesc,
    pub sample_mask: u32,
    pub rasterizer: RasterizerDesc,
    pub depth_stencil: DepthStencilDesc,
    pub input_layout: InputLayoutDesc,
    pub ib_strip_cut_value: u32,
    pub primitive_topology_type: u32,
    pub num_render_targets: u32,
    pub rtv_formats: [u32; 8],
    pub dsv_format: u32,
    pub sample_count: u32,
    pub sample_quality: u32,
    pub node_mask: u32,
    pub cached_pso: CachedPipelineState,
    pub flags: u32,
}

/// `D3D12_COMPUTE_PIPELINE_STATE_DESC`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ComputePipelineStateDesc {
    pub root_signature: *mut core::ffi::c_void,
    pub cs: ShaderBytecode,
    pub node_mask: u32,
    pub cached_pso: CachedPipelineState,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DescriptorHeapDesc {
    pub heap_type: u32,
    pub num_descriptors: u32,
    pub flags: u32,
    pub node_mask: u32,
}

pub const HEAP_TYPE_CBV_SRV_UAV: u32 = 0;
pub const HEAP_TYPE_SAMPLER: u32 = 1;
pub const HEAP_TYPE_RTV: u32 = 2;
pub const HEAP_TYPE_DSV: u32 = 3;

/// Descriptor sizes: a T# is 8 dwords (V#s and S#s fit in the same slot).
pub const fn descriptor_increment(heap_type: u32) -> u32 {
    match heap_type {
        HEAP_TYPE_SAMPLER => 16,
        _ => 32,
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ConstantBufferViewDesc {
    pub location: u64,
    pub size: u32,
}

/// `D3D12_SHADER_RESOURCE_VIEW_DESC`: the union is read as dwords.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ShaderResourceViewDesc {
    pub format: u32,
    pub dimension: u32,
    pub component_mapping: u32,
    pub _pad: u32,
    pub u: [u32; 6],
}

impl ShaderResourceViewDesc {
    pub fn first_element(&self) -> u64 {
        u64::from(self.u[0]) | (u64::from(self.u[1]) << 32)
    }
}

/// `D3D12_UNORDERED_ACCESS_VIEW_DESC` (union at offset 8, read as dwords).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct UnorderedAccessViewDesc {
    pub format: u32,
    pub dimension: u32,
    pub u: [u32; 8],
}

impl UnorderedAccessViewDesc {
    pub fn first_element(&self) -> u64 {
        u64::from(self.u[0]) | (u64::from(self.u[1]) << 32)
    }
}

/// `D3D12_RENDER_TARGET_VIEW_DESC` (union at offset 8).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderTargetViewDesc {
    pub format: u32,
    pub dimension: u32,
    pub u: [u32; 4],
}

/// `D3D12_DEPTH_STENCIL_VIEW_DESC` (union at offset 12).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DepthStencilViewDesc {
    pub format: u32,
    pub dimension: u32,
    pub flags: u32,
    pub u: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SamplerDesc {
    pub filter: u32,
    pub address: [u32; 3],
    pub mip_lod_bias: f32,
    pub max_anisotropy: u32,
    pub comparison_func: u32,
    pub border_color: [f32; 4],
    pub min_lod: f32,
    pub max_lod: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexBufferView {
    pub location: u64,
    pub size: u32,
    pub format: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VertexBufferView {
    pub location: u64,
    pub size: u32,
    pub stride: u32,
}

/// `D3D12_RESOURCE_BARRIER` (union at offset 8).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ResourceBarrier {
    pub barrier_type: u32,
    pub flags: u32,
    pub resource: *mut core::ffi::c_void,
    pub u: [u32; 4],
}

/// `D3D12_TEXTURE_COPY_LOCATION`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TextureCopyLocation {
    pub resource: *mut core::ffi::c_void,
    pub copy_type: u32,
    /// `PlacedFootprint` (offset u64, format, width, height, depth,
    /// row pitch) or `SubresourceIndex` (dword 0).
    pub u: [u32; 8],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Box3 {
    pub left: u32,
    pub top: u32,
    pub front: u32,
    pub right: u32,
    pub bottom: u32,
    pub back: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct QueryHeapDesc {
    pub query_type: u32,
    pub count: u32,
    pub node_mask: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CommandSignatureDesc {
    pub byte_stride: u32,
    pub num_argument_descs: u32,
    pub argument_descs: *const IndirectArgumentDesc,
    pub node_mask: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IndirectArgumentDesc {
    pub arg_type: u32,
    pub u: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeapProperties {
    pub heap_type: u32,
    pub cpu_page_property: u32,
    pub memory_pool_preference: u32,
    pub creation_node_mask: u32,
    pub visible_node_mask: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HeapDesc {
    pub size: u64,
    pub properties: HeapProperties,
    pub alignment: u64,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceAllocationInfo {
    pub size: u64,
    pub alignment: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SubresourceFootprint {
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub row_pitch: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlacedSubresourceFootprint {
    pub offset: u64,
    pub footprint: SubresourceFootprint,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceDesc {
    pub dimension: u32,
    pub alignment: u64,
    pub width: u64,
    pub height: u32,
    pub depth_or_array_size: u16,
    pub mip_levels: u16,
    pub format: u32,
    pub sample_count: u32,
    pub sample_quality: u32,
    pub layout: u32,
    pub flags: u32,
}

pub const DIMENSION_BUFFER: u32 = 1;
pub const DIMENSION_TEXTURE1D: u32 = 2;
pub const DIMENSION_TEXTURE2D: u32 = 3;
pub const DIMENSION_TEXTURE3D: u32 = 4;

/// `D3D12_RAYTRACING_GEOMETRY_DESC` (triangles or AABBs, union at 8).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaytracingGeometryDesc {
    pub geometry_type: u32,
    pub flags: u32,
    /// Triangles: transform (u64), index format, vertex format, index
    /// count, vertex count, index buffer (u64), vertex buffer (u64 start,
    /// u64 stride). AABBs: count (u64), buffer (u64 start, u64 stride).
    pub u: [u32; 12],
}

/// `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BuildRaytracingInputs {
    pub as_type: u32,
    pub flags: u32,
    pub num_descs: u32,
    pub descs_layout: u32,
    /// Instance descs address (top level), or a pointer to geometry descs
    /// (array or array of pointers) for bottom level.
    pub descs: u64,
}

/// `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BuildRaytracingDesc {
    pub dest: u64,
    pub inputs: BuildRaytracingInputs,
    pub source: u64,
    pub scratch: u64,
}

/// `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RaytracingPrebuildInfo {
    pub result_max_size: u64,
    pub scratch_size: u64,
    pub update_scratch_size: u64,
}

#[allow(clippy::assertions_on_constants)]
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<ShaderBytecode>() == 16);
    assert!(size_of::<RenderTargetBlendDesc>() == 40);
    assert!(size_of::<BlendDesc>() == 328);
    assert!(size_of::<RasterizerDesc>() == 44);
    assert!(size_of::<DepthStencilDesc>() == 52);
    assert!(offset_of!(GraphicsPipelineStateDesc, stream_output) == 88);
    assert!(offset_of!(GraphicsPipelineStateDesc, blend) == 120);
    assert!(offset_of!(GraphicsPipelineStateDesc, sample_mask) == 448);
    assert!(offset_of!(GraphicsPipelineStateDesc, rasterizer) == 452);
    assert!(offset_of!(GraphicsPipelineStateDesc, depth_stencil) == 496);
    assert!(offset_of!(GraphicsPipelineStateDesc, input_layout) == 552);
    assert!(offset_of!(GraphicsPipelineStateDesc, rtv_formats) == 580);
    assert!(offset_of!(GraphicsPipelineStateDesc, flags) == 648);
    assert!(size_of::<GraphicsPipelineStateDesc>() == 656);
    assert!(size_of::<ComputePipelineStateDesc>() == 56);
    assert!(size_of::<ShaderResourceViewDesc>() == 40);
    assert!(size_of::<UnorderedAccessViewDesc>() == 40);
    assert!(size_of::<RenderTargetViewDesc>() == 24);
    assert!(size_of::<DepthStencilViewDesc>() == 24);
    assert!(size_of::<SamplerDesc>() == 52);
    assert!(size_of::<ResourceBarrier>() == 32);
    assert!(size_of::<TextureCopyLocation>() == 48);
    assert!(size_of::<CommandSignatureDesc>() == 24);
    assert!(size_of::<IndirectArgumentDesc>() == 16);
    assert!(size_of::<HeapDesc>() == 48);
    assert!(size_of::<ResourceDesc>() == 56);
    assert!(size_of::<PlacedSubresourceFootprint>() == 32);
    assert!(size_of::<RaytracingGeometryDesc>() == 56);
    assert!(size_of::<BuildRaytracingDesc>() == 48);
};

// ============================================================================
// DXGI formats
// ============================================================================

/// How a DXGI format is stored and read under Strata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatMap {
    /// GFX10 `IMG_FORMAT` / `BUF_FMT` code (0: none).
    pub gfx10: u16,
    /// Bytes per element (texel, or 4×4 block).
    pub bytes: u32,
    /// 4 for BCn, else 1.
    pub block: u32,
    /// Component selects for a view (BGRA formats read swapped, X8 reads 1).
    pub select: [Select; 4],
    /// Stored as BGRA (the Vulkan image is `B8G8R8A8`).
    pub bgra: bool,
    pub srgb: bool,
    /// The Vulkan depth/stencil format of a depth format.
    pub depth: Option<i32>,
}

const XYZW: [Select; 4] = [Select::X, Select::Y, Select::Z, Select::W];
const XYZ1: [Select; 4] = [Select::X, Select::Y, Select::Z, Select::One];
const X001: [Select; 4] = [Select::X, Select::Zero, Select::Zero, Select::One];
const XY01: [Select; 4] = [Select::X, Select::Y, Select::Zero, Select::One];
const ZYXW: [Select; 4] = [Select::Z, Select::Y, Select::X, Select::W];
const ZYX1: [Select; 4] = [Select::Z, Select::Y, Select::X, Select::One];
const A8: [Select; 4] = [Select::Zero, Select::Zero, Select::Zero, Select::X];

/// Vulkan raw values of the depth formats Strata renders.
pub const VK_D16_UNORM: i32 = 124;
pub const VK_D32_SFLOAT: i32 = 126;
pub const VK_D24_UNORM_S8_UINT: i32 = 129;
pub const VK_D32_SFLOAT_S8_UINT: i32 = 130;

/// The Strata storage of DXGI format `dxgi` (`None`: unknown / unsupported).
/// Typeless formats take their UNORM/FLOAT member.
pub fn dxgi_format(dxgi: u32) -> Option<FormatMap> {
    let m = |gfx10: u16, bytes: u32, select: [Select; 4]| FormatMap {
        gfx10,
        bytes,
        block: 1,
        select,
        bgra: false,
        srgb: false,
        depth: None,
    };
    let bc = |gfx10: u16, bytes: u32, srgb: bool| FormatMap {
        gfx10,
        bytes,
        block: 4,
        select: XYZW,
        bgra: false,
        srgb,
        depth: None,
    };
    let depth = |gfx10: u16, bytes: u32, vk: i32| FormatMap {
        gfx10,
        bytes,
        block: 1,
        select: X001,
        bgra: false,
        srgb: false,
        depth: Some(vk),
    };
    Some(match dxgi {
        1 | 2 => m(77, 16, XYZW),
        3 => m(75, 16, XYZW),
        4 => m(76, 16, XYZW),
        5 | 6 => m(74, 12, XYZ1),
        7 => m(72, 12, XYZ1),
        8 => m(73, 12, XYZ1),
        9 | 10 => m(71, 8, XYZW),
        11 => m(65, 8, XYZW),
        12 => m(69, 8, XYZW),
        13 => m(66, 8, XYZW),
        14 => m(70, 8, XYZW),
        15 | 16 => m(64, 8, XY01),
        17 => m(62, 8, XY01),
        18 => m(63, 8, XY01),
        19 | 20 => depth(22, 8, VK_D32_SFLOAT_S8_UINT),
        21 => m(22, 8, X001),
        22 => m(5, 8, X001), // X32_TYPELESS_G8X24_UINT
        23 | 24 => m(50, 4, XYZW),
        25 => m(54, 4, XYZW),
        26 => m(36, 4, XYZ1),
        27 | 28 => m(56, 4, XYZW),
        29 => FormatMap {
            srgb: true,
            ..m(56, 4, XYZW)
        },
        30 => m(60, 4, XYZW),
        31 => m(57, 4, XYZW),
        32 => m(61, 4, XYZW),
        33 | 34 => m(29, 4, XY01),
        35 => m(23, 4, XY01),
        36 => m(27, 4, XY01),
        37 => m(24, 4, XY01),
        38 => m(28, 4, XY01),
        39 | 41 => m(22, 4, X001),
        40 => depth(22, 4, VK_D32_SFLOAT),
        42 => m(20, 4, X001),
        43 => m(21, 4, X001),
        44..=47 => depth(22, 4, VK_D24_UNORM_S8_UINT),
        48 | 49 => m(14, 2, XY01),
        50 => m(18, 2, XY01),
        51 => m(15, 2, XY01),
        52 => m(19, 2, XY01),
        53 | 56 => m(7, 2, X001),
        54 => m(13, 2, X001),
        55 => depth(7, 2, VK_D16_UNORM),
        57 => m(11, 2, X001),
        58 => m(8, 2, X001),
        59 => m(12, 2, X001),
        60 | 61 => m(1, 1, X001),
        62 => m(5, 1, X001),
        63 => m(2, 1, X001),
        64 => m(6, 1, X001),
        65 => m(1, 1, A8),
        67 => m(81, 4, XYZ1),      // R9G9B9E5_SHAREDEXP
        68 | 69 => m(56, 4, XYZW), // R8G8_B8G8 / G8R8_G8B8
        70 | 71 => bc(109, 8, false),
        72 => bc(110, 8, true),
        73 | 74 => bc(111, 16, false),
        75 => bc(112, 16, true),
        76 | 77 => bc(113, 16, false),
        78 => bc(114, 16, true),
        79 | 80 => bc(115, 8, false),
        81 => bc(116, 8, false),
        82 | 83 => bc(117, 16, false),
        84 => bc(118, 16, false),
        85 => m(79, 2, XYZ1), // B5G6R5_UNORM
        86 => m(80, 2, XYZW), // B5G5R5A1_UNORM
        87 | 90 => FormatMap {
            bgra: true,
            ..m(56, 4, ZYXW)
        },
        88 | 92 => FormatMap {
            bgra: true,
            ..m(56, 4, ZYX1)
        },
        89 => m(50, 4, XYZW), // R10G10B10_XR_BIAS_A2_UNORM
        91 => FormatMap {
            bgra: true,
            srgb: true,
            ..m(56, 4, ZYXW)
        },
        93 => FormatMap {
            bgra: true,
            srgb: true,
            ..m(56, 4, ZYX1)
        },
        94 | 95 => bc(119, 16, false),
        96 => bc(120, 16, false),
        97 | 98 => bc(121, 16, false),
        99 => bc(122, 16, true),
        103 | 107 | 130..=132 => m(1, 1, X001), // YUV planar/packed (NV12, YUY2, P208, V208, V408)
        115 => m(82, 2, XYZW),                  // B4G4R4A4_UNORM
        _ => return None,
    })
}

/// Bytes per element of a DXGI format (1 when unknown, so sizes stay sane).
pub fn dxgi_bytes(dxgi: u32) -> (u32, u32) {
    dxgi_format(dxgi).map_or((1, 1), |f| (f.bytes, f.block))
}

// ============================================================================
// Strata's private T# trailer (RTV / DSV heaps and host-built SRVs)
// ============================================================================

pub use strata_gpu::texture::{TRAILER_BGRA, TRAILER_DEPTH, TRAILER_MAGIC, TRAILER_SRGB};

/// Strata's flags and Vulkan depth format, when `w` is a T# it built.
pub fn trailer(w: &[u32; 8]) -> Option<(u32, i32)> {
    strata_gpu::texture::trailer_flags(w).map(|f| (f, w[6] as i32))
}

// ============================================================================
// Resources under Strata's memory model
// ============================================================================

/// A resource's memory as the texture model lays it out: linear, rows
/// padded to 256 bytes, levels then slices (see `strata_gpu::texture`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureLayout {
    pub width: u32,
    pub height: u32,
    /// Array layers (2D) or depth (3D).
    pub depth_or_layers: u32,
    pub is_3d: bool,
    pub mips: u32,
    pub bytes: u32,
    pub block: u32,
}

impl TextureLayout {
    pub fn of(desc: &ResourceDesc) -> Self {
        let (bytes, block) = dxgi_bytes(desc.format);
        let width = desc.width.max(1) as u32;
        let height = if desc.dimension == DIMENSION_TEXTURE1D {
            1
        } else {
            desc.height.max(1)
        };
        let depth_or_layers = u32::from(desc.depth_or_array_size).max(1);
        let is_3d = desc.dimension == DIMENSION_TEXTURE3D;
        let max_mips = 32
            - width
                .max(height)
                .max(if is_3d { depth_or_layers } else { 1 })
                .leading_zeros();
        let mips = if desc.mip_levels == 0 {
            max_mips
        } else {
            u32::from(desc.mip_levels).min(max_mips)
        };
        Self {
            width,
            height,
            depth_or_layers,
            is_3d,
            mips,
            bytes,
            block,
        }
    }

    /// `(width, height)` of level `m`.
    pub fn extent(&self, m: u32) -> (u32, u32) {
        ((self.width >> m).max(1), (self.height >> m).max(1))
    }

    /// Slices of level `m`.
    pub fn slices(&self, m: u32) -> u32 {
        if self.is_3d {
            (self.depth_or_layers >> m).max(1)
        } else {
            self.depth_or_layers
        }
    }

    /// Row pitch of level `m` in bytes (rows of blocks for BCn).
    pub fn row_pitch(&self, m: u32) -> u64 {
        let (w, _) = self.extent(m);
        (u64::from(w.div_ceil(self.block)) * u64::from(self.bytes)).next_multiple_of(256)
    }

    /// Rows (of blocks) of level `m`.
    pub fn rows(&self, m: u32) -> u32 {
        self.extent(m).1.div_ceil(self.block)
    }

    pub fn slice_bytes(&self, m: u32) -> u64 {
        self.row_pitch(m) * u64::from(self.rows(m))
    }

    /// Byte offset of level `m`, slice `s`.
    pub fn offset(&self, m: u32, s: u32) -> u64 {
        let mut off = 0;
        for l in 0..m {
            off += self.slice_bytes(l) * u64::from(self.slices(l));
        }
        off + self.slice_bytes(m) * u64::from(s)
    }

    pub fn total_bytes(&self) -> u64 {
        self.offset(self.mips, 0)
    }

    /// `(mip, slice)` of subresource index `sub` (D3D12: mips vary fastest).
    pub fn subresource(&self, sub: u32) -> (u32, u32) {
        (sub % self.mips.max(1), sub / self.mips.max(1))
    }
}

/// Bytes a resource of `desc` occupies, 64 KiB aligned.
pub fn resource_size(desc: &ResourceDesc) -> u64 {
    let size = if desc.dimension == DIMENSION_BUFFER {
        desc.width
    } else {
        TextureLayout::of(desc).total_bytes()
    };
    size.max(256).next_multiple_of(65536)
}

/// `GetCopyableFootprints`: placed footprints for `num` subresources from
/// `first`, packed from `base_offset` with D3D12's 512-byte placement and
/// 256-byte row-pitch alignment. Returns the total bytes.
pub fn copyable_footprints(
    desc: &ResourceDesc,
    first: u32,
    num: u32,
    base_offset: u64,
    mut layouts: Option<&mut [PlacedSubresourceFootprint]>,
    mut num_rows: Option<&mut [u32]>,
    mut row_sizes: Option<&mut [u64]>,
) -> u64 {
    if desc.dimension == DIMENSION_BUFFER {
        if let Some(l) = layouts.as_deref_mut().and_then(|l| l.first_mut()) {
            *l = PlacedSubresourceFootprint {
                offset: base_offset,
                footprint: SubresourceFootprint {
                    format: 0,
                    width: desc.width as u32,
                    height: 1,
                    depth: 1,
                    row_pitch: (desc.width as u32).next_multiple_of(256),
                },
            };
        }
        if let Some(r) = num_rows.as_deref_mut().and_then(|r| r.first_mut()) {
            *r = 1;
        }
        if let Some(r) = row_sizes.as_deref_mut().and_then(|r| r.first_mut()) {
            *r = desc.width;
        }
        return desc.width;
    }
    let t = TextureLayout::of(desc);
    let mut offset = base_offset;
    let mut total = 0;
    for i in 0..num {
        let (m, _) = t.subresource(first + i);
        let (w, h) = t.extent(m);
        let rows = h.div_ceil(t.block);
        let row_bytes = u64::from(w.div_ceil(t.block)) * u64::from(t.bytes);
        let pitch = row_bytes.next_multiple_of(256);
        let depth = if t.is_3d { t.slices(m) } else { 1 };
        offset = offset.next_multiple_of(512);
        if let Some(l) = layouts.as_deref_mut().and_then(|l| l.get_mut(i as usize)) {
            *l = PlacedSubresourceFootprint {
                offset,
                footprint: SubresourceFootprint {
                    format: desc.format,
                    width: w,
                    height: h,
                    depth,
                    row_pitch: pitch as u32,
                },
            };
        }
        if let Some(r) = num_rows.as_deref_mut().and_then(|r| r.get_mut(i as usize)) {
            *r = rows;
        }
        if let Some(r) = row_sizes.as_deref_mut().and_then(|r| r.get_mut(i as usize)) {
            *r = row_bytes;
        }
        let size = pitch * u64::from(rows) * u64::from(depth);
        total = offset + size - base_offset;
        offset += size;
    }
    total
}

// ============================================================================
// Descriptor builders
// ============================================================================

/// A resource as a view builder needs it.
#[derive(Debug, Clone, Copy)]
pub struct ViewResource {
    /// The GPU VA of the resource's first byte (its BDA).
    pub gpu_va: u64,
    pub desc: ResourceDesc,
}

/// Selects from a `D3D12_SHADER_COMPONENT_MAPPING` (`D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING`
/// is `0x1688`: identity), composed over the format's own selects.
fn component_mapping(mapping: u32, fmt: [Select; 4]) -> [Select; 4] {
    if mapping == 0 || mapping == 0x1688 {
        return fmt;
    }
    std::array::from_fn(|i| match (mapping >> (3 * i)) & 7 {
        0..=3 => fmt[((mapping >> (3 * i)) & 7) as usize],
        4 => Select::Zero,
        _ => Select::One,
    })
}

/// The T# of an image view: `mip`/`levels` and `first_slice`/`slices` pick
/// the subresources, `kind` the view's dimensionality.
#[allow(clippy::too_many_arguments)]
pub fn image_descriptor(
    res: &ViewResource,
    format: u32,
    kind: ImageKind,
    mip: u32,
    levels: u32,
    first_slice: u32,
    slices: u32,
    mapping: u32,
) -> Option<[u32; 8]> {
    let fmt_dxgi = if format == 0 { res.desc.format } else { format };
    let f = dxgi_format(fmt_dxgi)?;
    let t = TextureLayout::of(&res.desc);
    let mip = mip.min(t.mips - 1);
    let levels = levels.clamp(1, t.mips - mip);
    let whole_slices = slices == 0 || (first_slice == 0 && slices >= t.slices(0));
    // A single level, or every slice: describable exactly. A sub-range of
    // slices over several levels is read as the whole array (the model
    // strides levels by the resource's slice count).
    let (base, width, height, base_level, last_level, depth) =
        if !t.is_3d && (!whole_slices || levels == 1 && mip > 0) {
            let (w, h) = t.extent(mip);
            let n = if whole_slices {
                t.slices(mip)
            } else {
                slices.max(1)
            };
            (res.gpu_va + t.offset(mip, first_slice), w, h, 0, 0, n)
        } else {
            (
                res.gpu_va,
                t.width,
                t.height,
                mip,
                mip + levels - 1,
                if t.is_3d {
                    t.depth_or_layers
                } else {
                    t.slices(0)
                },
            )
        };
    let depth = match kind {
        ImageKind::Cube => (depth / 6).max(1),
        ImageKind::Tex1D | ImageKind::Tex2D | ImageKind::Tex2DMsaa => 1,
        _ => depth.max(1),
    };
    let select = component_mapping(mapping, f.select);
    let d = ImageDescriptor {
        base_address: base,
        format: f.gfx10,
        width,
        height,
        depth,
        kind,
        sw_mode: 0,
        select: if f.bgra { XYZW } else { select },
        base_level: base_level as u8,
        last_level: last_level as u8,
        base_array: 0,
    };
    let mut w = d.encode();
    let mut flags = 0;
    if f.bgra {
        flags |= TRAILER_BGRA;
    }
    if f.srgb {
        flags |= TRAILER_SRGB;
    }
    if f.depth.is_some() {
        flags |= TRAILER_DEPTH;
    }
    w[5] = TRAILER_MAGIC | flags;
    w[6] = f.depth.unwrap_or(0) as u32;
    // The view's own selects for BGRA images (the Vulkan image is BGRA, so
    // identity reads RGBA): kept in dword 7 for completeness.
    w[7] = select
        .iter()
        .enumerate()
        .map(|(i, s)| (*s as u32) << (4 * i))
        .sum();
    Some(w)
}

/// A V# (buffer descriptor), per the public RDNA2 ISA `SQ_BUF_RSRC_WORD0..3`:
/// base address (48 bits), stride, record count, component selects and a
/// GFX10 buffer format (`format` 0: raw dwords, read as `32_FLOAT`).
pub fn buffer_descriptor(base: u64, stride: u32, num_records: u32, format: u16) -> [u32; 4] {
    let fmt = if format == 0 { 22 } else { format };
    let sel = 4 | (5 << 3) | (6 << 6) | (7 << 9);
    [
        base as u32,
        ((base >> 32) as u32 & 0xffff) | ((stride & 0x3fff) << 16),
        num_records,
        sel | (u32::from(fmt & 0x7f) << 12) | (1 << 24) | (3 << 28),
    ]
}

/// The S# of a `D3D12_SAMPLER_DESC`.
pub fn sampler_descriptor(s: &SamplerDesc) -> [u32; 4] {
    let clamp = |a: u32| match a {
        2 => Clamp::Mirror,
        3 => Clamp::ClampToEdge,
        4 => Clamp::ClampToBorder,
        5 => Clamp::MirrorOnceToEdge,
        _ => Clamp::Wrap,
    };
    let lin = |b: bool| if b { Filter::Linear } else { Filter::Point };
    let aniso = s.filter & 0x40 != 0;
    SamplerDescriptor {
        clamp: s.address.map(clamp),
        max_aniso: if aniso {
            s.max_anisotropy.clamp(1, 16) as u8
        } else {
            1
        },
        mag: lin(aniso || s.filter & 0x04 != 0),
        min: lin(aniso || s.filter & 0x10 != 0),
        mip: Some(lin(aniso || s.filter & 0x01 != 0)),
        min_lod: s.min_lod.max(0.0),
        max_lod: s.max_lod.clamp(0.0, 15.99),
        lod_bias: s.mip_lod_bias,
        depth_compare_func: s.comparison_func.saturating_sub(1).min(7) as u8,
        compare: None,
    }
    .encode()
}

// ============================================================================
// Fixed-function state translations (D3D12 enum → Vulkan raw value)
// ============================================================================

/// `D3D12_BLEND` → `VkBlendFactor`.
pub fn vk_blend(b: u32) -> i32 {
    match b {
        1 => 0,   // ZERO
        2 => 1,   // ONE
        3 => 2,   // SRC_COLOR
        4 => 3,   // ONE_MINUS_SRC_COLOR
        5 => 6,   // SRC_ALPHA
        6 => 7,   // ONE_MINUS_SRC_ALPHA
        7 => 8,   // DST_ALPHA
        8 => 9,   // ONE_MINUS_DST_ALPHA
        9 => 4,   // DST_COLOR
        10 => 5,  // ONE_MINUS_DST_COLOR
        11 => 14, // SRC_ALPHA_SATURATE
        14 => 10, // CONSTANT_COLOR
        15 => 11, // ONE_MINUS_CONSTANT_COLOR
        16 => 15, // SRC1_COLOR
        17 => 16, // ONE_MINUS_SRC1_COLOR
        18 => 17, // SRC1_ALPHA
        19 => 18, // ONE_MINUS_SRC1_ALPHA
        20 => 12, // CONSTANT_ALPHA
        21 => 13, // ONE_MINUS_CONSTANT_ALPHA
        _ => 1,
    }
}

/// `D3D12_BLEND_OP` → `VkBlendOp`.
pub fn vk_blend_op(op: u32) -> i32 {
    match op {
        2 => 1, // SUBTRACT
        3 => 2, // REVERSE_SUBTRACT
        4 => 3, // MIN
        5 => 4, // MAX
        _ => 0, // ADD
    }
}

/// `D3D12_COMPARISON_FUNC` → `VkCompareOp` (both orders match, D3D12 from 1).
pub fn vk_compare(f: u32) -> i32 {
    f.saturating_sub(1).min(7) as i32
}

/// `D3D12_STENCIL_OP` → `VkStencilOp` (same order, D3D12 from 1).
pub fn vk_stencil_op(op: u32) -> i32 {
    op.saturating_sub(1).min(7) as i32
}

/// `D3D_PRIMITIVE_TOPOLOGY` → `VkPrimitiveTopology`.
pub fn vk_topology(t: u32) -> i32 {
    match t {
        1 => 0,        // POINT_LIST
        2 => 1,        // LINE_LIST
        3 => 2,        // LINE_STRIP
        5 => 4,        // TRIANGLE_STRIP
        10 => 6,       // LINE_LIST_WITH_ADJACENCY
        11 => 7,       // LINE_STRIP_WITH_ADJACENCY
        12 => 8,       // TRIANGLE_LIST_WITH_ADJACENCY
        13 => 9,       // TRIANGLE_STRIP_WITH_ADJACENCY
        33..=64 => 10, // PATCH_LIST
        _ => 3,        // TRIANGLE_LIST
    }
}

// ============================================================================
// Root signatures
// ============================================================================

/// One root parameter and where its values go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RootParam {
    /// A descriptor table: its GPU address (2 user SGPRs).
    Table {
        sgpr: u8,
        ranges: Vec<DescriptorRange>,
    },
    /// Root constants: `count` dwords.
    Constants { sgpr: u8, count: u8 },
    /// A root CBV/SRV/UAV: its GPU address (2 user SGPRs).
    Descriptor { sgpr: u8, kind: u32 },
}

impl RootParam {
    pub fn sgpr(&self) -> u8 {
        match self {
            RootParam::Table { sgpr, .. }
            | RootParam::Constants { sgpr, .. }
            | RootParam::Descriptor { sgpr, .. } => *sgpr,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorRange {
    pub range_type: u32,
    pub count: u32,
    pub base_register: u32,
    pub space: u32,
    pub offset: u32,
}

/// A root signature as user-SGPR assignments.
///
/// Strata's convention (no console metadata is read yet): parameters take
/// user SGPRs in order, tables and root descriptors as a 64-bit address
/// (two SGPRs), constants as their dwords; then the static-sampler table
/// address and the vertex-buffer table address, two SGPRs each. When that
/// exceeds the 32 user SGPRs Strata pushes, `spilled` is set: SGPRs 0–1 hold
/// the address of the whole layout in memory instead.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RootLayout {
    pub params: Vec<RootParam>,
    pub static_samplers: Vec<[u32; 4]>,
    pub sampler_table_sgpr: Option<u8>,
    pub vertex_table_sgpr: u8,
    pub dwords: u32,
    pub spilled: bool,
}

fn rd(b: &[u8], at: usize) -> Option<u32> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
}

/// Finds the `RTS0` part of a DXBC container, or takes `blob` as a bare
/// serialized root signature.
fn rts0(blob: &[u8]) -> &[u8] {
    if blob.len() >= 32 && &blob[0..4] == b"DXBC" {
        let n = rd(blob, 28).unwrap_or(0) as usize;
        for i in 0..n.min(64) {
            let Some(off) = rd(blob, 32 + 4 * i) else {
                break;
            };
            let off = off as usize;
            if blob.get(off..off + 4) == Some(b"RTS0") {
                let len = rd(blob, off + 4).unwrap_or(0) as usize;
                return blob.get(off + 8..off + 8 + len).unwrap_or(&[]);
            }
        }
        return &[];
    }
    blob
}

impl RootLayout {
    /// Parses a serialized root signature (version 1.0 or 1.1), bare or in
    /// a DXBC container's `RTS0` part.
    pub fn parse(blob: &[u8]) -> Option<Self> {
        let b = rts0(blob);
        let version = rd(b, 0)?;
        let num_params = rd(b, 4)? as usize;
        let params_off = rd(b, 8)? as usize;
        let num_samplers = rd(b, 12)? as usize;
        let samplers_off = rd(b, 16)? as usize;
        let v11 = version >= 2;
        let mut layout = RootLayout::default();
        let mut sgpr = 0u32;
        for i in 0..num_params.min(64) {
            let at = params_off + 12 * i;
            let ty = rd(b, at)?;
            let payload = rd(b, at + 8)? as usize;
            let p = match ty {
                0 => {
                    let n = rd(b, payload)? as usize;
                    let ranges_off = rd(b, payload + 4)? as usize;
                    let stride = if v11 { 24 } else { 20 };
                    let mut ranges = Vec::new();
                    for r in 0..n.min(256) {
                        let ra = ranges_off + stride * r;
                        ranges.push(DescriptorRange {
                            range_type: rd(b, ra)?,
                            count: rd(b, ra + 4)?,
                            base_register: rd(b, ra + 8)?,
                            space: rd(b, ra + 12)?,
                            offset: rd(b, ra + if v11 { 20 } else { 16 })?,
                        });
                    }
                    let p = RootParam::Table {
                        sgpr: sgpr as u8,
                        ranges,
                    };
                    sgpr += 2;
                    p
                }
                1 => {
                    let count = rd(b, payload + 8)?.min(64);
                    let p = RootParam::Constants {
                        sgpr: sgpr as u8,
                        count: count as u8,
                    };
                    sgpr += count;
                    p
                }
                kind => {
                    let p = RootParam::Descriptor {
                        sgpr: sgpr as u8,
                        kind,
                    };
                    sgpr += 2;
                    p
                }
            };
            layout.params.push(p);
        }
        for i in 0..num_samplers.min(2048) {
            let at = samplers_off + 52 * i;
            let f = |k: usize| rd(b, at + 4 * k);
            let desc = SamplerDesc {
                filter: f(0)?,
                address: [f(1)?, f(2)?, f(3)?],
                mip_lod_bias: f32::from_bits(f(4)?),
                max_anisotropy: f(5)?,
                comparison_func: f(6)?,
                border_color: [0.0; 4],
                min_lod: f32::from_bits(f(8)?),
                max_lod: f32::from_bits(f(9)?),
            };
            layout.static_samplers.push(sampler_descriptor(&desc));
        }
        if !layout.static_samplers.is_empty() {
            layout.sampler_table_sgpr = Some(sgpr as u8);
            sgpr += 2;
        }
        layout.vertex_table_sgpr = sgpr as u8;
        sgpr += 2;
        layout.dwords = sgpr;
        layout.spilled = sgpr > 32;
        Some(layout)
    }
}

/// `D3D12SerializeRootSignature` / `D3D12SerializeVersionedRootSignature`:
/// a `D3D12_ROOT_SIGNATURE_DESC` (1.0) or 1.1 desc → the serialized form,
/// wrapped in a one-part DXBC container (`RTS0`), as D3D12 produces.
///
/// # Safety
/// `desc` points at a valid root signature description of `version`.
pub unsafe fn serialize_root_signature(desc: *const u8, version: u32) -> Vec<u8> {
    let rd_ptr = |p: *const u8, off: usize| unsafe { (p.add(off) as *const u32).read_unaligned() };
    let rd_ptr64 =
        |p: *const u8, off: usize| unsafe { (p.add(off) as *const u64).read_unaligned() };
    // Both versions: NumParameters, pParameters (8-aligned), NumStaticSamplers,
    // pStaticSamplers, Flags.
    let num_params = rd_ptr(desc, 0) as usize;
    let params = rd_ptr64(desc, 8) as *const u8;
    let num_samplers = rd_ptr(desc, 16) as usize;
    let samplers = rd_ptr64(desc, 24) as *const u8;
    let flags = rd_ptr(desc, 32);
    let v11 = version >= 2;

    let mut body: Vec<u8> = Vec::new();
    let put = |v: &mut Vec<u8>, x: u32| v.extend_from_slice(&x.to_le_bytes());
    // Header (24 bytes), parameters (12 each), then payloads, then samplers.
    let header = 24usize;
    let params_off = header;
    let mut payloads: Vec<u8> = Vec::new();
    let payload_base = params_off + 12 * num_params;
    let mut param_entries: Vec<[u32; 3]> = Vec::new();
    // D3D12_ROOT_PARAMETER(1): ParameterType u32, union at 8 (16 bytes:
    // table {NumDescriptorRanges u32, pDescriptorRanges ptr} / constants
    // {ShaderRegister, RegisterSpace, Num32BitValues} / descriptor
    // {ShaderRegister, RegisterSpace[, Flags]}), ShaderVisibility at 24; 32 bytes.
    for i in 0..num_params {
        let p = unsafe { params.add(32 * i) };
        let ty = rd_ptr(p, 0);
        let vis = rd_ptr(p, 24);
        let at = (payload_base + payloads.len()) as u32;
        match ty {
            0 => {
                let n = rd_ptr(p, 8) as usize;
                let ranges = rd_ptr64(p, 16) as *const u8;
                let ranges_at = at + 8;
                put(&mut payloads, n as u32);
                put(&mut payloads, ranges_at);
                for r in 0..n {
                    let rp = unsafe { ranges.add(if v11 { 24 } else { 20 } * r) };
                    let words = if v11 { 6 } else { 5 };
                    for k in 0..words {
                        put(&mut payloads, rd_ptr(rp, 4 * k));
                    }
                }
            }
            1 => {
                for k in 0..3 {
                    put(&mut payloads, rd_ptr(p, 8 + 4 * k));
                }
            }
            _ => {
                let words = if v11 { 3 } else { 2 };
                for k in 0..words {
                    put(&mut payloads, rd_ptr(p, 8 + 4 * k));
                }
            }
        }
        param_entries.push([ty, vis, at]);
    }
    let samplers_off = payload_base + payloads.len();
    put(&mut body, version);
    put(&mut body, num_params as u32);
    put(&mut body, params_off as u32);
    put(&mut body, num_samplers as u32);
    put(&mut body, samplers_off as u32);
    put(&mut body, flags);
    for e in &param_entries {
        for &x in e {
            put(&mut body, x);
        }
    }
    body.extend_from_slice(&payloads);
    for i in 0..num_samplers {
        let sp = unsafe { samplers.add(52 * i) };
        for k in 0..13 {
            put(&mut body, rd_ptr(sp, 4 * k));
        }
    }
    dxbc_wrap(b"RTS0", &body)
}

/// A one-part DXBC container (the checksum is left zero: D3D12 runtimes
/// Strata hands it to do not verify it).
pub fn dxbc_wrap(fourcc: &[u8; 4], part: &[u8]) -> Vec<u8> {
    let total = 32 + 4 + 8 + part.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0u8; 16]);
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&36u32.to_le_bytes());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(part.len() as u32).to_le_bytes());
    out.extend_from_slice(part);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_layout_pads_rows_and_stacks_levels() {
        let desc = ResourceDesc {
            dimension: DIMENSION_TEXTURE2D,
            width: 100,
            height: 10,
            depth_or_array_size: 2,
            mip_levels: 2,
            format: 28, // R8G8B8A8_UNORM
            sample_count: 1,
            ..Default::default()
        };
        let t = TextureLayout::of(&desc);
        assert_eq!(t.row_pitch(0), 512); // 400 -> 512
        assert_eq!(t.offset(0, 1), 512 * 10);
        assert_eq!(t.offset(1, 0), 2 * 512 * 10);
        assert_eq!(t.subresource(3), (1, 1));
        let mut fp = [PlacedSubresourceFootprint::default(); 4];
        let total = copyable_footprints(&desc, 0, 4, 0, Some(&mut fp), None, None);
        assert_eq!(fp[1].offset % 512, 0);
        assert_eq!(fp[1].footprint.row_pitch, 256); // subresource 1 is mip 1 (50 texels)
        assert!(total >= 4 * 256);
    }

    #[test]
    fn srv_descriptor_is_a_t_sharp_the_texture_cache_reads() {
        let res = ViewResource {
            gpu_va: 0x1_0000_0000,
            desc: ResourceDesc {
                dimension: DIMENSION_TEXTURE2D,
                width: 64,
                height: 32,
                depth_or_array_size: 6,
                mip_levels: 1,
                format: 87, // B8G8R8A8_UNORM
                sample_count: 1,
                ..Default::default()
            },
        };
        let w = image_descriptor(&res, 0, ImageKind::Cube, 0, 1, 0, 0, 0x1688).unwrap();
        let d = ImageDescriptor::decode(w);
        assert_eq!((d.width, d.height, d.depth), (64, 32, 1));
        assert_eq!(d.kind, ImageKind::Cube);
        assert_eq!(d.format, 56);
        assert_eq!(trailer(&w), Some((TRAILER_BGRA, 0)));
    }

    #[test]
    fn root_signature_round_trips_through_serialization() {
        // Two params: a 4-dword constant block and a table of one SRV range.
        #[repr(C)]
        struct Range {
            ty: u32,
            n: u32,
            base: u32,
            space: u32,
            off: u32,
        }
        #[repr(C)]
        struct Param {
            ty: u32,
            pad: u32,
            a: u64,
            b: u64,
            vis: u32,
            pad2: u32,
        }
        #[repr(C)]
        struct Desc {
            n: u32,
            pad: u32,
            params: *const Param,
            ns: u32,
            pad2: u32,
            samplers: *const u8,
            flags: u32,
        }
        let range = Range {
            ty: 0,
            n: 8,
            base: 0,
            space: 0,
            off: 0,
        };
        let params = [
            Param {
                ty: 1,
                pad: 0,
                a: 0, // register 0, space 0
                b: 4, // Num32BitValues
                vis: 0,
                pad2: 0,
            },
            Param {
                ty: 0,
                pad: 0,
                a: 1,
                b: &range as *const Range as u64,
                vis: 0,
                pad2: 0,
            },
        ];
        let desc = Desc {
            n: 2,
            pad: 0,
            params: params.as_ptr(),
            ns: 0,
            pad2: 0,
            samplers: std::ptr::null(),
            flags: 0,
        };
        let blob = unsafe { serialize_root_signature(&desc as *const Desc as *const u8, 1) };
        let l = RootLayout::parse(&blob).unwrap();
        assert_eq!(l.params[0], RootParam::Constants { sgpr: 0, count: 4 });
        match &l.params[1] {
            RootParam::Table { sgpr, ranges } => {
                assert_eq!(*sgpr, 4);
                assert_eq!(ranges[0].count, 8);
            }
            p => panic!("{p:?}"),
        }
        assert_eq!(l.vertex_table_sgpr, 6);
        assert!(!l.spilled);
    }

    #[test]
    fn sampler_desc_maps_filters_and_compare() {
        let s = SamplerDesc {
            filter: 0x95, // COMPARISON_MIN_MAG_MIP_LINEAR
            address: [3, 3, 3],
            comparison_func: 4, // LESS_EQUAL
            max_lod: 15.0,
            ..Default::default()
        };
        let d = SamplerDescriptor::decode(sampler_descriptor(&s));
        assert_eq!(
            (d.mag, d.min, d.mip),
            (Filter::Linear, Filter::Linear, Some(Filter::Linear))
        );
        assert_eq!(d.clamp[0], Clamp::ClampToEdge);
        assert_eq!(d.depth_compare_func, 3);
    }
}
