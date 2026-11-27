//! Persist the pipeline descriptions a title actually used, alongside the driver cache.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ash::vk;
use serde::{Deserialize, Serialize};

use crate::graphics::{GraphicsPsoState, GraphicsStages};
use crate::runner::ComputeRunner;

const VERSION: &str = "usage-v1";
const MAX_RECIPES: usize = 256;
const MAX_RECIPE_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Default, Clone, Copy)]
pub struct PipelineStats {
    pub driver_cache_loaded: bool,
    pub prewarmed: u32,
    pub warmup_ms: f64,
    pub created_compute: u32,
    pub created_graphics: u32,
    pub creation_ms: f64,
}

#[derive(Serialize, Deserialize)]
pub(crate) enum Recipe {
    Compute {
        code: Vec<u32>,
        subgroup: Option<u32>,
    },
    Graphics {
        stages: Stages,
        formats: Vec<i32>,
        state: Box<FixedState>,
    },
}

#[derive(Serialize, Deserialize)]
pub(crate) enum Stages {
    Classic {
        vs: Vec<u32>,
        hs: Option<Vec<u32>>,
        ds: Option<Vec<u32>>,
        gs: Option<Vec<u32>>,
        ps: Vec<u32>,
    },
    Mesh {
        task: Option<Vec<u32>>,
        mesh: Vec<u32>,
        ps: Vec<u32>,
    },
}

#[derive(Serialize, Deserialize)]
pub(crate) struct FixedState {
    topology: i32,
    polygon: i32,
    cull: u32,
    front_face: i32,
    depth_bias_enable: bool,
    depth_bias: [f32; 2],
    depth_test: bool,
    depth_write: bool,
    depth_compare: i32,
    stencil_test: bool,
    stencil_faces: [[u32; 6]; 2],
    depth_clamp: bool,
    alpha_to_coverage: bool,
    depth_format: Option<i32>,
    blends: Vec<[u32; 8]>,
    depth_bounds: bool,
    patch_points: u32,
}

impl From<&GraphicsPsoState> for FixedState {
    fn from(s: &GraphicsPsoState) -> Self {
        let face = |s: vk::StencilOpState| {
            [
                s.fail_op.as_raw() as u32,
                s.pass_op.as_raw() as u32,
                s.depth_fail_op.as_raw() as u32,
                s.compare_op.as_raw() as u32,
                s.compare_mask,
                s.write_mask,
            ]
        };
        Self {
            topology: s.topology.as_raw(),
            polygon: s.polygon_mode.as_raw(),
            cull: s.cull_mode.as_raw(),
            front_face: s.front_face.as_raw(),
            depth_bias_enable: s.depth_bias_enable,
            depth_bias: [s.depth_bias_constant, s.depth_bias_slope],
            depth_test: s.depth_test_enable,
            depth_write: s.depth_write_enable,
            depth_compare: s.depth_compare_op.as_raw(),
            stencil_test: s.stencil_test_enable,
            stencil_faces: [face(s.stencil_front), face(s.stencil_back)],
            depth_clamp: s.depth_clamp,
            alpha_to_coverage: s.alpha_to_coverage,
            depth_format: s.depth_format.map(vk::Format::as_raw),
            blends: s
                .blend_attachments
                .iter()
                .map(|b| {
                    [
                        b.blend_enable,
                        b.src_color_blend_factor.as_raw() as u32,
                        b.dst_color_blend_factor.as_raw() as u32,
                        b.color_blend_op.as_raw() as u32,
                        b.src_alpha_blend_factor.as_raw() as u32,
                        b.dst_alpha_blend_factor.as_raw() as u32,
                        b.alpha_blend_op.as_raw() as u32,
                        b.color_write_mask.as_raw(),
                    ]
                })
                .collect(),
            depth_bounds: s.depth_bounds_test_enable,
            patch_points: s.patch_control_points,
        }
    }
}

impl FixedState {
    fn for_pipeline(&self) -> GraphicsPsoState {
        let face = |[fail, pass, depth_fail, compare, compare_mask, write_mask]: [u32; 6]| {
            vk::StencilOpState::default()
                .fail_op(vk::StencilOp::from_raw(fail as i32))
                .pass_op(vk::StencilOp::from_raw(pass as i32))
                .depth_fail_op(vk::StencilOp::from_raw(depth_fail as i32))
                .compare_op(vk::CompareOp::from_raw(compare as i32))
                .compare_mask(compare_mask)
                .write_mask(write_mask)
        };
        let [depth_bias_constant, depth_bias_slope] = self.depth_bias;
        GraphicsPsoState {
            topology: vk::PrimitiveTopology::from_raw(self.topology),
            polygon_mode: vk::PolygonMode::from_raw(self.polygon),
            cull_mode: vk::CullModeFlags::from_raw(self.cull),
            front_face: vk::FrontFace::from_raw(self.front_face),
            depth_bias_enable: self.depth_bias_enable,
            depth_bias_constant,
            depth_bias_slope,
            depth_test_enable: self.depth_test,
            depth_write_enable: self.depth_write,
            depth_compare_op: vk::CompareOp::from_raw(self.depth_compare),
            stencil_test_enable: self.stencil_test,
            stencil_front: face(self.stencil_faces[0]),
            stencil_back: face(self.stencil_faces[1]),
            depth_clamp: self.depth_clamp,
            alpha_to_coverage: self.alpha_to_coverage,
            depth_format: self.depth_format.map(vk::Format::from_raw),
            blend_attachments: self
                .blends
                .iter()
                .map(|&[enable, sc, dc, op, sa, da, aop, mask]| {
                    vk::PipelineColorBlendAttachmentState::default()
                        .blend_enable(enable != 0)
                        .src_color_blend_factor(vk::BlendFactor::from_raw(sc as i32))
                        .dst_color_blend_factor(vk::BlendFactor::from_raw(dc as i32))
                        .color_blend_op(vk::BlendOp::from_raw(op as i32))
                        .src_alpha_blend_factor(vk::BlendFactor::from_raw(sa as i32))
                        .dst_alpha_blend_factor(vk::BlendFactor::from_raw(da as i32))
                        .alpha_blend_op(vk::BlendOp::from_raw(aop as i32))
                        .color_write_mask(vk::ColorComponentFlags::from_raw(mask))
                })
                .collect(),
            depth_bounds_test_enable: self.depth_bounds,
            patch_control_points: self.patch_points,
            ..Default::default()
        }
    }
}

impl Recipe {
    pub(crate) fn graphics(
        stages: GraphicsStages<'_>,
        formats: &[vk::Format],
        state: &GraphicsPsoState,
    ) -> Self {
        let words = |s: &[u32]| s.to_vec();
        let stages = match stages {
            GraphicsStages::Classic { vs, hs, ds, gs, ps } => Stages::Classic {
                vs: words(vs),
                hs: hs.map(words),
                ds: ds.map(words),
                gs: gs.map(words),
                ps: words(ps),
            },
            GraphicsStages::Mesh { task, mesh, ps } => Stages::Mesh {
                task: task.map(words),
                mesh: words(mesh),
                ps: words(ps),
            },
        };
        Self::Graphics {
            stages,
            formats: formats.iter().map(|f| f.as_raw()).collect(),
            state: Box::new(state.into()),
        }
    }

    fn warm(&self, runner: &ComputeRunner) -> bool {
        match self {
            Self::Compute { code, subgroup } => runner.pipeline_for(code, *subgroup).is_ok(),
            Self::Graphics {
                stages,
                formats,
                state,
            } => {
                let stages = match stages {
                    Stages::Classic { vs, hs, ds, gs, ps } => GraphicsStages::Classic {
                        vs,
                        hs: hs.as_deref(),
                        ds: ds.as_deref(),
                        gs: gs.as_deref(),
                        ps,
                    },
                    Stages::Mesh { task, mesh, ps } => GraphicsStages::Mesh {
                        task: task.as_deref(),
                        mesh,
                        ps,
                    },
                };
                let formats: Vec<_> = formats.iter().copied().map(vk::Format::from_raw).collect();
                runner
                    .gfx_pipeline_for_stages(stages, &formats, &state.for_pipeline())
                    .is_ok()
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct Usage {
    directory: Option<PathBuf>,
    pending: BTreeMap<String, Recipe>,
    stats: PipelineStats,
}

impl ComputeRunner {
    /// Load the device-specific driver cache and a bounded set of earlier PSOs.
    pub fn enable_pipeline_cache(&self, root: &Path, prewarm: bool) -> PipelineStats {
        self.enable_pipeline_cache_with_seed(root, prewarm, None)
    }

    pub fn enable_pipeline_cache_with_seed(
        &self,
        root: &Path,
        prewarm: bool,
        seed: Option<&Path>,
    ) -> PipelineStats {
        let directory = root.join(VERSION).join(self.adapter_cache_key());
        let mut driver_cache_loaded = self
            .load_pipeline_cache(&directory.join("driver.bin"))
            .unwrap_or(false);
        if let Some(seed) = seed {
            driver_cache_loaded |= self.load_pipeline_cache(seed).unwrap_or(false);
        }
        let start = Instant::now();
        let mut prewarmed = 0;
        if prewarm && let Ok(entries) = std::fs::read_dir(&directory) {
            let mut entries: Vec<_> = entries
                .flatten()
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
                .collect();
            entries.sort_by_cached_key(|e| {
                std::cmp::Reverse(e.metadata().and_then(|m| m.modified()).ok())
            });
            for entry in entries.into_iter().take(32) {
                if start.elapsed() > Duration::from_millis(250) {
                    break;
                }
                if entry.metadata().is_ok_and(|m| m.len() <= MAX_RECIPE_BYTES)
                    && let Ok(file) = std::fs::File::open(entry.path())
                    && let Ok(recipe) =
                        serde_json::from_reader::<_, Recipe>(std::io::BufReader::new(file))
                    && recipe.warm(self)
                {
                    prewarmed += 1;
                }
            }
        }
        let mut usage = self.pipeline_usage.lock().unwrap();
        usage.directory = Some(directory);
        usage.stats = PipelineStats {
            driver_cache_loaded,
            prewarmed,
            warmup_ms: start.elapsed().as_secs_f64() * 1000.0,
            ..PipelineStats::default()
        };
        usage.stats
    }

    pub fn pipeline_stats(&self) -> PipelineStats {
        self.pipeline_usage.lock().unwrap().stats
    }

    pub(crate) fn record_pipeline(
        &self,
        key: String,
        graphics: bool,
        elapsed: Duration,
        recipe: impl FnOnce() -> Recipe,
    ) {
        let mut usage = self.pipeline_usage.lock().unwrap();
        let stats = &mut usage.stats;
        if graphics {
            stats.created_graphics += 1;
        } else {
            stats.created_compute += 1;
        }
        stats.creation_ms += elapsed.as_secs_f64() * 1000.0;
        if usage.directory.is_some() && usage.pending.len() < MAX_RECIPES {
            usage.pending.entry(key).or_insert_with(recipe);
        }
    }

    pub fn flush_pipeline_cache(&self) -> Result<(), crate::context::GpuError> {
        let (directory, pending) = {
            let mut usage = self.pipeline_usage.lock().unwrap();
            let Some(directory) = usage.directory.clone() else {
                return Ok(());
            };
            (directory, std::mem::take(&mut usage.pending))
        };
        let io = |e: std::io::Error| crate::context::GpuError::Io(e.to_string());
        let result = (|| {
            self.save_pipeline_cache(&directory.join("driver.bin"))?;
            for (key, recipe) in &pending {
                let Ok(bytes) = serde_json::to_vec(recipe) else {
                    continue;
                };
                if bytes.len() as u64 <= MAX_RECIPE_BYTES {
                    crate::present::write_atomic(&directory.join(format!("{key}.json")), &bytes)
                        .map_err(io)?;
                }
            }
            let mut old: Vec<_> = std::fs::read_dir(&directory)
                .map_err(io)?
                .flatten()
                .filter(|e| e.path().extension().is_some_and(|s| s == "json"))
                .collect();
            old.sort_by_cached_key(|e| {
                std::cmp::Reverse(e.metadata().and_then(|m| m.modified()).ok())
            });
            for e in old.into_iter().skip(MAX_RECIPES) {
                let _ = std::fs::remove_file(e.path());
            }
            Ok(())
        })();
        if result.is_err() {
            self.pipeline_usage.lock().unwrap().pending.extend(pending);
        }
        result
    }
}
