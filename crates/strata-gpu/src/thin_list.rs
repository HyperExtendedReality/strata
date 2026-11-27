//! Thin Command-List Recorder, Bind Deduplication & Frame Replay.
//!
//! Combines high-impact optimizations from VIctory-Belobug-AMD and VIctory-7900XTX:
//! - `VICTORY_THIN`: title threads record lightweight commands with zero Vulkan calls;
//!   worker threads translate them in parallel.
//! - `VICTORY_BIND_CACHE`: skips redundant render-target & pipeline rebinds.
//! - Integer render-target blend sanitization (console ignores blend on integer RTs,
//!   Vulkan/D3D12 reject the pipeline).
//! - Dispatches over the device's group-count limit split, not clamped (a clamp
//!   drops the groups past 65,535: blocky patches).
//! - Compute-written texture readback scheduling (fixes water/boat CPU physics).

/// The smallest `maxComputeWorkGroupCount` Vulkan guarantees per dimension.
pub const MAX_INDIRECT_DISPATCH_GROUPS: u32 = 65_535;

/// Splits a dispatch of `groups` into `(base, count)` pieces no larger than
/// `limit` in any dimension, covering every group exactly once (for
/// `vkCmdDispatchBase`, which keeps each group's `WorkgroupId`).
pub fn split_dispatch(groups: [u32; 3], limit: [u32; 3]) -> Vec<([u32; 3], [u32; 3])> {
    let limit = limit.map(|l| l.max(1));
    let mut out = Vec::new();
    let mut bz = 0;
    while bz < groups[2] {
        let cz = limit[2].min(groups[2] - bz);
        let mut by = 0;
        while by < groups[1] {
            let cy = limit[1].min(groups[1] - by);
            let mut bx = 0;
            while bx < groups[0] {
                let cx = limit[0].min(groups[0] - bx);
                out.push(([bx, by, bz], [cx, cy, cz]));
                bx += cx;
            }
            by += cy;
        }
        bz += cz;
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineKey(pub u64);

/// A linear surface in title memory: element rows `row_pitch` apart,
/// slices `slice_pitch` apart (elements are texels, or 4×4 blocks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Surface {
    pub va: u64,
    pub row_pitch: u64,
    pub slice_pitch: u64,
    /// Bytes per element.
    pub bpe: u32,
}

/// `CopyTextureRegion`: `extent` elements from `src` at `src_origin` to
/// `dst` at `dst_origin` (x, y in elements, z in slices).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureCopy {
    pub dst: Surface,
    pub dst_origin: [u32; 3],
    pub src: Surface,
    pub src_origin: [u32; 3],
    pub extent: [u32; 3],
}

/// One argument of an `ExecuteIndirect` command signature, in the order
/// the argument buffer holds them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndirectArg {
    /// `D3D12_DRAW_ARGUMENTS` (4 dwords).
    Draw,
    /// `D3D12_DRAW_INDEXED_ARGUMENTS` (5 dwords).
    DrawIndexed,
    /// `D3D12_DISPATCH_ARGUMENTS` (3 dwords).
    Dispatch,
    /// `D3D12_DISPATCH_MESH_ARGUMENTS` (3 dwords).
    DispatchMesh,
    /// `D3D12_DISPATCH_RAYS_DESC` (104 bytes = 26 dwords).
    DispatchRays,
    /// A `D3D12_VERTEX_BUFFER_VIEW` (4 dwords) for `slot`.
    VertexBuffer { slot: u32 },
    /// A `D3D12_INDEX_BUFFER_VIEW` (4 dwords).
    IndexBuffer,
    /// `count` root constants to user SGPR `sgpr` onward.
    Constants { sgpr: u16, count: u16 },
    /// A root CBV/SRV/UAV address (2 dwords) to user SGPRs `sgpr..sgpr+2`.
    RootAddress { sgpr: u16 },
}

impl IndirectArg {
    /// Bytes the argument occupies in the argument buffer.
    pub fn bytes(self) -> u32 {
        match self {
            IndirectArg::Draw => 16,
            IndirectArg::DrawIndexed => 20,
            IndirectArg::Dispatch | IndirectArg::DispatchMesh => 12,
            IndirectArg::DispatchRays => 104,
            IndirectArg::VertexBuffer { .. } | IndirectArg::IndexBuffer => 16,
            IndirectArg::Constants { count, .. } => 4 * u32::from(count),
            IndirectArg::RootAddress { .. } => 8,
        }
    }
}

/// An `ExecuteIndirect` command signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndirectLayout {
    pub stride: u32,
    pub args: Vec<IndirectArg>,
}

/// One bottom-level geometry of a `BuildRaytracingAccelerationStructure`,
/// read from title memory when the build executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelGeometry {
    Triangles {
        /// 3x4 row-major float transform address (0: identity).
        transform: u64,
        /// `DXGI_FORMAT` of indices (0: none, 42 `R32_UINT`, 57 `R16_UINT`).
        index_format: u32,
        /// `DXGI_FORMAT` of positions (6 `R32G32B32_FLOAT`, 16 `R32G32_FLOAT`,
        /// 10 `R16G16B16A16_FLOAT`, 34 `R16G16_FLOAT`).
        vertex_format: u32,
        index_count: u32,
        vertex_count: u32,
        index_va: u64,
        vertex_va: u64,
        vertex_stride: u64,
        opaque: bool,
    },
    Aabbs {
        count: u64,
        va: u64,
        stride: u64,
        opaque: bool,
    },
}

/// `BuildRaytracingAccelerationStructure`: what to build, and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccelBuild {
    pub dest: u64,
    pub dest_size: u64,
    pub top_level: bool,
    /// Bottom level: the geometries.
    pub geometries: Vec<AccelGeometry>,
    /// Top level: `D3D12_RAYTRACING_INSTANCE_DESC` array address and count.
    pub instances: (u64, u32),
}

/// Which depth/stencil aspects a clear writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClearAspects {
    pub color: bool,
    pub depth: bool,
    pub stencil: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderTargetDesc {
    pub console_va: u64,
    pub format_is_integer: bool,
    pub blend_enabled: bool,
}

impl RenderTargetDesc {
    /// Sanitize console RT state for Vulkan/D3D12 (Belobug PR #45 fix).
    pub fn sanitized(mut self) -> Self {
        if self.format_is_integer {
            self.blend_enabled = false;
        }
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinCommand {
    BindPipeline(PipelineKey),
    SetRenderTargets(Vec<RenderTargetDesc>),
    PushUserSgprs {
        offset_dw: u8,
        values: Vec<u32>,
    },
    Dispatch {
        x: u32,
        y: u32,
        z: u32,
        is_thread_count: bool,
    },
    /// Group counts read at execution from three dwords at `args_va` (a
    /// console address the GPU may have written), then split like `Dispatch`.
    DispatchIndirect {
        args_va: u64,
    },
    DrawInstanced {
        vertex_count: u32,
        instance_count: u32,
        start_vertex: u32,
        start_instance: u32,
    },
    DrawIndexedInstanced {
        index_count: u32,
        instance_count: u32,
        start_index: u32,
        base_vertex: i32,
        start_instance: u32,
    },
    DrawIndirect {
        args_bda: u64,
        max_verts: u32,
        max_inst: u32,
    },
    ResourceBarrier {
        resource_va: u64,
        state_before: u32,
        state_after: u32,
    },
    CopyBufferRegion {
        dst_va: u64,
        dst_offset: u64,
        src_va: u64,
        src_offset: u64,
        num_bytes: u64,
    },
    /// `UpdateTileMappings`: binds or unbinds 64 KiB sparse tiles for a reserved resource.
    UpdateTileMappings {
        resource_va: u64,
        start_tile: u32,
        tile_count: u32,
        heap_va: u64,
        heap_start_tile: u32,
        unbind: bool,
    },
    /// `CopyTileMappings`: copies sparse tile bindings between two reserved resources.
    CopyTileMappings {
        dst_va: u64,
        src_va: u64,
        tile_count: u32,
    },
    /// Schedule GPU -> CPU readback of a compute-written texture (e.g. ocean heightmap).
    ReadbackTexture {
        console_va: u64,
        byte_len: u64,
    },

    // --- D3D12 fixed-function state (recorded as data, applied per draw) ---
    /// `RSSetViewports` (first viewport), as f32 bits: x, y, w, h, min z, max z.
    SetViewport([u32; 6]),
    /// `RSSetScissorRects` (first rect): left, top, right, bottom.
    SetScissor([i32; 4]),
    SetStencilRef(u32),
    /// `OMSetBlendFactor`, as f32 bits.
    SetBlendFactor([u32; 4]),
    /// `IASetPrimitiveTopology`, as a `VkPrimitiveTopology` value.
    SetTopology(i32),
    /// `IASetIndexBuffer` (`va` 0: unbound).
    SetIndexBuffer {
        va: u64,
        size: u32,
        index32: bool,
    },
    /// `IASetVertexBuffers`: `(address, size, stride)` from `start_slot`.
    SetVertexBuffers {
        start_slot: u32,
        views: Vec<(u64, u32, u32)>,
    },
    /// The root signature's user-SGPR budget: `dwords` used, whether they
    /// spill to memory, and where the vertex-buffer table address goes.
    SetRootLayout {
        dwords: u32,
        spilled: bool,
        vertex_table_sgpr: u8,
    },
    /// `OMSetRenderTargets`: colour targets and an optional depth target,
    /// as T#s over title memory (with Strata's trailer).
    BindTargets {
        colors: Vec<[u32; 8]>,
        depth: Option<[u32; 8]>,
    },
    /// `ClearRenderTargetView` / `ClearDepthStencilView` /
    /// `ClearUnorderedAccessView*` on an image: `color` as bits, `depth`
    /// as f32 bits.
    ClearImage {
        words: [u32; 8],
        aspects: ClearAspects,
        color: [u32; 4],
        depth: u32,
        stencil: u32,
    },
    /// Fills `len` bytes at `va` with a repeating `pattern` of `dwords`
    /// dwords (`ClearUnorderedAccessView*` on a buffer).
    FillMemory {
        va: u64,
        len: u64,
        pattern: [u32; 4],
        dwords: u8,
    },
    /// `CopyBufferRegion` / `CopyResource` of buffers / `ResolveQueryData`.
    CopyMemory {
        dst: u64,
        src: u64,
        len: u64,
    },
    /// `CopyTextureRegion` / `CopyResource` of textures.
    CopyTexture(Box<TextureCopy>),
    /// `EndQuery(TIMESTAMP)`: the GPU clock (ns) to `va` when executed.
    WriteTimestamp {
        va: u64,
    },
    /// A query result written when executed (occlusion queries report
    /// "visible": Strata does not count samples).
    WriteU64 {
        va: u64,
        value: u64,
    },
    /// `SetPredication` (`va` 0: off): later draws and dispatches are
    /// skipped when the u64 at `va` is zero (`skip_if_zero`) or non-zero.
    SetPredication {
        va: u64,
        skip_if_zero: bool,
    },
    /// `ExecuteIndirect`.
    ExecuteIndirect {
        layout: std::sync::Arc<IndirectLayout>,
        max_count: u32,
        args_va: u64,
        count_va: u64,
    },
    /// `DispatchMesh` (amplification/mesh pipelines).
    DispatchMesh {
        x: u32,
        y: u32,
        z: u32,
    },
    /// `BuildRaytracingAccelerationStructure`.
    BuildAccel(Box<AccelBuild>),
    /// `CopyRaytracingAccelerationStructure` (clone / compact).
    CopyAccel {
        dst: u64,
        src: u64,
    },
    /// `OMSetDepthBounds` (`min`, `max` as f32 bits).
    SetDepthBounds {
        min: u32,
        max: u32,
    },
    /// `SOSetTargets`: `(buffer_va, size_bytes, filled_size_va)` from `start_slot`.
    SetStreamOutputTargets {
        start_slot: u32,
        views: Vec<(u64, u64, u64)>,
    },
    /// `RSSetShadingRate`: fragment shading rate (width, height in pixels) and
    /// two combiner ops (`D3D12_SHADING_RATE_COMBINER`).
    SetShadingRate {
        width: u32,
        height: u32,
        combiners: [u32; 2],
    },
    /// `RSSetShadingRateImage`: GPU VA of the screen-space shading-rate image (0: none).
    SetShadingRateImage {
        va: u64,
    },
    /// `BeginQuery`: starts hardware occlusion / pipeline-statistics tracking into `va`.
    BeginQuery {
        va: u64,
        query_type: u32,
    },
    /// `EndQuery`: finalizes hardware query and writes result into `va`.
    EndQuery {
        va: u64,
        query_type: u32,
    },
    /// `DispatchRays`: DXR shader binding table ranges + dispatch dimensions.
    DispatchRays {
        /// `(start_address, size_in_bytes)` of the raygen shader record.
        raygen: (u64, u64),
        /// `(start_address, size_in_bytes, stride_in_bytes)` of the miss table.
        miss: (u64, u64, u64),
        /// `(start_address, size_in_bytes, stride_in_bytes)` of the hit-group table.
        hit: (u64, u64, u64),
        /// `(start_address, size_in_bytes, stride_in_bytes)` of the callable table.
        callable: (u64, u64, u64),
        width: u32,
        height: u32,
        depth: u32,
    },
    /// `ResolveSubresourceRegion`: resolves a sub-rectangle of an MSAA or
    /// compressed source surface into a destination surface.
    ResolveSubresourceRegion {
        dst: Surface,
        dst_origin: [u32; 2],
        src: Surface,
        /// `[left, top, width, height]` in texels.
        src_rect: [u32; 4],
        sample_count: u32,
    },
    /// `CopyTiles`: copies `tile_count` 64 KiB tiles between a tiled resource and a linear buffer.
    CopyTiles {
        tiled_va: u64,
        buffer_va: u64,
        tile_count: u32,
        tile_bytes: u32,
        to_tiled: bool,
    },
    /// `WriteSamplerFeedback`: records a sampled mip level into a sampler feedback map.
    WriteSamplerFeedback {
        target_va: u64,
        mip_level: u8,
        tile_x: u32,
        tile_y: u32,
    },
}

#[derive(Debug, Default)]
pub struct ThinCommandList {
    commands: Vec<ThinCommand>,
    bound_pipeline: Option<PipelineKey>,
    bound_rts: Vec<RenderTargetDesc>,
    /// Collected indirect draw clamps to execute in a single batched `clamp_indirect.slang` dispatch.
    pub hoisted_clamps: Vec<(u64, u32, u32)>,
    pub skipped_redundant_binds: usize,
}

impl ThinCommandList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind_pipeline(&mut self, key: PipelineKey) {
        if self.bound_pipeline == Some(key) {
            self.skipped_redundant_binds += 1;
            return;
        }
        self.bound_pipeline = Some(key);
        self.commands.push(ThinCommand::BindPipeline(key));
    }

    pub fn set_render_targets(&mut self, rts: &[RenderTargetDesc]) {
        let sanitized: Vec<RenderTargetDesc> = rts
            .iter()
            .copied()
            .map(RenderTargetDesc::sanitized)
            .collect();
        if self.bound_rts == sanitized {
            self.skipped_redundant_binds += 1;
            return;
        }
        self.bound_rts = sanitized.clone();
        self.commands.push(ThinCommand::SetRenderTargets(sanitized));
    }

    pub fn push_sgprs(&mut self, offset_dw: u8, values: &[u32]) {
        self.commands.push(ThinCommand::PushUserSgprs {
            offset_dw,
            values: values.to_vec(),
        });
    }

    /// Record a compute dispatch, handling the console thread-count-vs-group-count
    /// flag. Counts are kept whole; execution splits them to the device limit.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32, wg_size: [u32; 3], is_thread_count: bool) {
        let (gx, gy, gz) = if is_thread_count {
            (
                x.div_ceil(wg_size[0].max(1)),
                y.div_ceil(wg_size[1].max(1)),
                z.div_ceil(wg_size[2].max(1)),
            )
        } else {
            (x, y, z)
        };
        self.commands.push(ThinCommand::Dispatch {
            x: gx,
            y: gy,
            z: gz,
            is_thread_count: false,
        });
    }

    /// Record an indirect dispatch: `[x, y, z]` group counts at `args_va`.
    pub fn dispatch_indirect(&mut self, args_va: u64) {
        self.commands
            .push(ThinCommand::DispatchIndirect { args_va });
    }

    /// Record an indirect draw and hoist its clamp entry into `hoisted_clamps`
    /// so all indirect draws are clamped in a single pre-pass compute dispatch.
    pub fn draw_indirect(&mut self, args_bda: u64, max_verts: u32, max_inst: u32) {
        self.hoisted_clamps.push((args_bda, max_verts, max_inst));
        self.commands.push(ThinCommand::DrawIndirect {
            args_bda,
            max_verts,
            max_inst,
        });
    }

    pub fn draw_instanced(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        start_vertex: u32,
        start_instance: u32,
    ) {
        self.commands.push(ThinCommand::DrawInstanced {
            vertex_count,
            instance_count,
            start_vertex,
            start_instance,
        });
    }

    pub fn draw_indexed_instanced(
        &mut self,
        index_count: u32,
        instance_count: u32,
        start_index: u32,
        base_vertex: i32,
        start_instance: u32,
    ) {
        self.commands.push(ThinCommand::DrawIndexedInstanced {
            index_count,
            instance_count,
            start_index,
            base_vertex,
            start_instance,
        });
    }

    pub fn resource_barrier(&mut self, resource_va: u64, state_before: u32, state_after: u32) {
        self.commands.push(ThinCommand::ResourceBarrier {
            resource_va,
            state_before,
            state_after,
        });
    }

    pub fn copy_buffer_region(
        &mut self,
        dst_va: u64,
        dst_offset: u64,
        src_va: u64,
        src_offset: u64,
        num_bytes: u64,
    ) {
        self.commands.push(ThinCommand::CopyBufferRegion {
            dst_va,
            dst_offset,
            src_va,
            src_offset,
            num_bytes,
        });
    }

    pub fn reset(&mut self) {
        self.commands.clear();
        self.bound_pipeline = None;
        self.bound_rts.clear();
        self.hoisted_clamps.clear();
    }

    /// Appends any command as is (state and copy commands need no
    /// recording-time work).
    pub fn push(&mut self, cmd: ThinCommand) {
        self.commands.push(cmd);
    }

    pub fn schedule_texture_readback(&mut self, console_va: u64, byte_len: u64) {
        self.commands.push(ThinCommand::ReadbackTexture {
            console_va,
            byte_len,
        });
    }

    pub fn commands(&self) -> &[ThinCommand] {
        &self.commands
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deduplicates_binds_sanitizes_int_rt_and_hoists_clamps() {
        let mut list = ThinCommandList::new();
        list.bind_pipeline(PipelineKey(7));
        list.bind_pipeline(PipelineKey(7)); // redundant

        let int_rt = RenderTargetDesc {
            console_va: 0x2000_0000,
            format_is_integer: true,
            blend_enabled: true, // invalid in Vulkan/D3D12
        };
        list.set_render_targets(&[int_rt]);
        list.set_render_targets(&[int_rt]); // redundant

        list.dispatch(128, 1, 1, [64, 1, 1], true); // 128 threads -> 2 groups
        list.draw_indirect(0x7F00_1000, 10_000, 64);

        assert_eq!(list.skipped_redundant_binds, 2);
        assert_eq!(list.hoisted_clamps.len(), 1);
        match &list.commands()[1] {
            ThinCommand::SetRenderTargets(rts) => assert!(!rts[0].blend_enabled),
            _ => panic!("expected sanitized RT"),
        }
        match &list.commands()[2] {
            ThinCommand::Dispatch { x, .. } => assert_eq!(*x, 2),
            _ => panic!("expected thread-count converted dispatch"),
        }
    }

    #[test]
    fn large_dispatches_are_split_not_clamped() {
        let mut list = ThinCommandList::new();
        list.dispatch(100_000, 1, 1, [1, 1, 1], false);
        assert_eq!(
            list.commands()[0],
            ThinCommand::Dispatch {
                x: 100_000,
                y: 1,
                z: 1,
                is_thread_count: false
            }
        );

        let pieces = split_dispatch([100_000, 70_000, 1], [65_535; 3]);
        assert_eq!(pieces.len(), 4);
        assert_eq!(pieces[1], ([65_535, 0, 0], [34_465, 65_535, 1]));
        assert_eq!(pieces[3], ([65_535, 65_535, 0], [34_465, 4_465, 1]));
        let covered: u64 = pieces
            .iter()
            .map(|(_, c)| c.iter().map(|&n| u64::from(n)).product::<u64>())
            .sum();
        assert_eq!(covered, 100_000 * 70_000);
        assert_eq!(
            split_dispatch([4, 4, 4], [65_535; 3]),
            [([0, 0, 0], [4, 4, 4])]
        );
        assert!(split_dispatch([0, 1, 1], [65_535; 3]).is_empty());
    }
}
