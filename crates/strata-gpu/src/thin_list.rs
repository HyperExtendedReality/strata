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
    /// Schedule GPU -> CPU readback of a compute-written texture (e.g. ocean heightmap).
    ReadbackTexture {
        console_va: u64,
        byte_len: u64,
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
