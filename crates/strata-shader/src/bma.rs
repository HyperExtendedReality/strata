//! BMA — BVH Mapping Algorithm with the Bounding-Volume Mapping Heuristic
//! (Chen, Xie, Yuan, *Computers* 2026, 15(6), 363, doi:10.3390/computers15060363).
//!
//! Rays from nearby pixels tend to share BVH ancestors, even after several
//! bounces. BMA exploits that in three stages:
//!
//! 1. **Sampling** — the centre pixel of every small screen tile traces its
//!    rays from the root as usual; the node it visits at each depth is
//!    recorded, per ray (primary, first bounce, …).
//! 2. **Mapping** — for each tile, ray index and depth, a 3×3 convolution
//!    over the neighbouring tiles' sampled nodes accumulates kernel weight per
//!    node. While the best node's weight is at least the threshold, the walk
//!    goes one level deeper; the deepest node that passed is the tile's
//!    *shortcut*.
//! 3. **Traversal** — every other pixel of the tile starts that ray's
//!    traversal at the shortcut node instead of the root.
//!
//! The paper interleaves the stages row by row on the CPU. On the GPU Strata
//! runs them temporally: shaders sample during frame *N*, the mapping pass
//! (`bma_mapping.slang`) runs at the frame boundary, and frame *N + 1* uses the
//! shortcuts. A shortcut is only taken while the BVH has the same build
//! generation it was sampled from (see [`BUILD_ID_BEFORE_ROOT`]), so a rebuilt
//! structure always falls back to exact traversal.
//!
//! Nothing here is title specific: it applies to any BVH Strata builds.
//!
//! Like the paper, this is approximate: a ray whose true closest hit lies
//! outside its tile's shortcut subtree misses it. Shortcuts are therefore
//! off by default (`STRATA_BVH_BMA` enables them), and primary rays are
//! excluded unless `STRATA_BVH_BMA_PRIMARY=1`, which keeps primary visibility
//! exact while secondary / GI / AO rays take the shortcuts.

use crate::bvh::{Box32Meta, Box32Node, INVALID_NODE, Ray, intersect_box32};

/// The build generation (`u32`) sits this many bytes before a Strata BVH's
/// root box node (in the structure header).
pub const BUILD_ID_BEFORE_ROOT: u64 = 8;
/// The structure size (`u64`, header included) sits this many bytes before
/// the root box node.
pub const SIZE_BEFORE_ROOT: u64 = 56;
/// Ray slots per tile (the paper's `r1`: primary plus bounces).
pub const MAX_RAYS: u32 = 4;
/// Depth levels sampled per ray (the paper's `d2`).
pub const MAX_DEPTH: u32 = 24;

/// `"BMA1"`.
pub const TABLE_MAGIC: u32 = u32::from_le_bytes(*b"BMA1");
/// Byte size of the GPU table header.
pub const TABLE_HEADER_BYTES: u64 = 64;

/// Byte offsets of the GPU table header fields, shared by the SPIR-V
/// emitter, the runtime and `bma_mapping.slang`.
pub mod hdr {
    pub const MAGIC: u64 = 0;
    /// Build generation the current shortcuts were mapped from.
    pub const MAP_BUILD: u64 = 4;
    pub const TILES_X: u64 = 8;
    pub const TILES_Y: u64 = 12;
    pub const TILE_SHIFT: u64 = 16;
    pub const RAYS: u64 = 20;
    pub const DEPTH: u64 = 24;
    /// [`super::FLAG_ENABLED`] | [`super::FLAG_PRIMARY`].
    pub const FLAGS: u64 = 28;
    /// Root box node address (`u64`, low then high dword).
    pub const ROOT_LO: u64 = 32;
    pub const ROOT_HI: u64 = 36;
    /// Build generation the shaders saw while sampling this frame.
    pub const SAMPLE_BUILD: u64 = 40;
    /// Threshold `v` (`f32`, informational; the mapping pass takes it as a
    /// push constant).
    pub const THRESHOLD: u64 = 44;
    /// Byte offset of the sample-path array from the table start.
    pub const PATHS_OFFSET: u64 = 48;
    /// A scratch dword masked-off writes go to.
    pub const SINK: u64 = 52;
}

/// Sampling and shortcuts are active.
pub const FLAG_ENABLED: u32 = 1;
/// Primary rays (ray slot 0) take shortcuts too.
pub const FLAG_PRIMARY: u32 = 2;

/// SPIR-V `SpecId`s of the table address (low, high dword) in shaders the
/// emitter instrumented for BMA. A runtime creating such a pipeline must
/// specialize them to a valid table (one with `FLAGS = 0` disables BMA).
pub const SPEC_ID_TABLE_LO: u32 = 0x5342;
pub const SPEC_ID_TABLE_HI: u32 = 0x5343;

/// Whether compiled compute shaders get the BMA instrumentation
/// (`STRATA_BVH_BMA` set to an enabling value; read once).
pub fn shader_instrumentation_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| BmaConfig::from_env().is_some())
}

/// BMA parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BmaConfig {
    /// Probability threshold `v` a node needs to become a shortcut.
    pub threshold: f32,
    /// Small-tile edge in pixels (a power of two).
    pub tile: u32,
    /// Convolution kernel `c[j][k]` over the 3×3 neighbouring tiles
    /// (`[dy][dx]`, centre at `[1][1]`); weights sum to 1.
    pub kernel: [[f32; 3]; 3],
    /// Ray slots (`r1`), at most [`MAX_RAYS`].
    pub rays: u32,
    /// Sampled depths (`d2`), at most [`MAX_DEPTH`].
    pub depth: u32,
    /// Let primary rays shortcut too.
    pub primary: bool,
}

impl Default for BmaConfig {
    fn default() -> Self {
        Self {
            // The paper's worked example uses 0.9.
            threshold: 0.9,
            tile: 8,
            kernel: DEFAULT_KERNEL,
            rays: MAX_RAYS,
            depth: MAX_DEPTH,
            primary: false,
        }
    }
}

/// Centre-weighted 3×3 kernel (the paper's kernel gives the centre tile the
/// most influence); sums to 1.
pub const DEFAULT_KERNEL: [[f32; 3]; 3] = [
    [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
    [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
    [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
];

impl BmaConfig {
    /// From the environment: `STRATA_BVH_BMA=<threshold>` (`1`/`on` for the
    /// default 0.9; unset, `0` or `off` disables), `STRATA_BVH_BMA_TILE=<px>`,
    /// `STRATA_BVH_BMA_PRIMARY=1`.
    pub fn from_env() -> Option<Self> {
        Self::from_vars(|k| std::env::var(k).ok())
    }

    /// [`Self::from_env`] over any variable lookup.
    pub fn from_vars(var: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let mut c = Self::default();
        let v = var("STRATA_BVH_BMA")?;
        match v.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" | "false" | "no" => return None,
            "1" | "on" | "true" | "yes" => {}
            t => {
                let th: f32 = t.parse().ok()?;
                if !(th > 0.0 && th <= 1.0) {
                    return None;
                }
                c.threshold = th;
            }
        }
        if let Some(t) = var("STRATA_BVH_BMA_TILE").and_then(|t| t.trim().parse::<u32>().ok()) {
            c.tile = t.clamp(2, 64).next_power_of_two();
        }
        c.primary = var("STRATA_BVH_BMA_PRIMARY").is_some_and(|p| p.trim() == "1");
        Some(c)
    }

    pub fn tile_shift(&self) -> u32 {
        self.tile.max(2).next_power_of_two().trailing_zeros()
    }

    /// Kernel flattened row-major (for the mapping pass push constants).
    pub fn kernel_flat(&self) -> [f32; 9] {
        std::array::from_fn(|i| self.kernel[i / 3][i % 3])
    }
}

/// Sampled paths: for each tile and ray slot, the byte offset (from the
/// root) + 1 of the first node visited at each depth; 0 = none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplePaths {
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub rays: u32,
    pub depth: u32,
    pub data: Vec<u32>,
}

impl SamplePaths {
    pub fn new(tiles_x: u32, tiles_y: u32, rays: u32, depth: u32) -> Self {
        Self {
            tiles_x,
            tiles_y,
            rays,
            depth,
            data: vec![0; (tiles_x * tiles_y * rays * depth) as usize],
        }
    }

    fn index(&self, tx: u32, ty: u32, r: u32, d: u32) -> Option<usize> {
        (tx < self.tiles_x && ty < self.tiles_y && r < self.rays && d < self.depth)
            .then(|| (((ty * self.tiles_x + tx) * self.rays + r) * self.depth + d) as usize)
    }

    /// Records that ray `r` of tile `(tx, ty)` visited the node at byte
    /// offset `node_off` (from the root) at depth `d`; the first visit wins
    /// (nearest-first traversal descends along the path to the nearest hit).
    pub fn record(&mut self, tx: u32, ty: u32, r: u32, d: u32, node_off: u32) {
        if let Some(i) = self.index(tx, ty, r, d)
            && self.data[i] == 0
        {
            self.data[i] = node_off.wrapping_add(1);
        }
    }

    pub fn get(&self, tx: u32, ty: u32, r: u32, d: u32) -> u32 {
        self.index(tx, ty, r, d).map_or(0, |i| self.data[i])
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }
}

/// Per tile and ray slot, the shortcut node's byte offset (from the root)
/// + 1; 0 = start at the root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortcutMap {
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub rays: u32,
    pub entries: Vec<u32>,
}

impl ShortcutMap {
    /// Where ray `r` of tile `(tx, ty)` starts: `Some(offset)` of the
    /// shortcut node from the root, `None` for the root itself.
    pub fn start(&self, tx: u32, ty: u32, r: u32) -> Option<u32> {
        if tx >= self.tiles_x || ty >= self.tiles_y || r >= self.rays {
            return None;
        }
        let e = self.entries[((ty * self.tiles_x + tx) * self.rays + r) as usize];
        (e != 0).then(|| e - 1)
    }

    /// Tiles × rays that got a shortcut.
    pub fn mapped(&self) -> usize {
        self.entries.iter().filter(|&&e| e != 0).count()
    }
}

/// The Mapping stage (the paper's Algorithm 3), generalised from rows of
/// large tiles to a full 2D 3×3 neighbourhood. Border tiles keep the root,
/// as the paper does for its first and last rows.
pub fn map(paths: &SamplePaths, cfg: &BmaConfig) -> ShortcutMap {
    let (tx_n, ty_n, rays) = (paths.tiles_x, paths.tiles_y, paths.rays);
    let mut m = ShortcutMap {
        tiles_x: tx_n,
        tiles_y: ty_n,
        rays,
        entries: vec![0; (tx_n * ty_n * rays) as usize],
    };
    if tx_n < 3 || ty_n < 3 {
        return m;
    }
    let mut acc: Vec<(u32, f32)> = Vec::with_capacity(9);
    for ty in 1..ty_n - 1 {
        for tx in 1..tx_n - 1 {
            for r in 0..rays {
                let mut best_node = 0u32;
                for d in 0..paths.depth {
                    acc.clear();
                    for j in 0..3 {
                        for k in 0..3 {
                            let n = paths.get(tx + k - 1, ty + j - 1, r, d);
                            if n == 0 {
                                continue;
                            }
                            let w = cfg.kernel[j as usize][k as usize];
                            match acc.iter_mut().find(|(key, _)| *key == n) {
                                Some((_, v)) => *v += w,
                                None => acc.push((n, w)),
                            }
                        }
                    }
                    let Some(&(node, p)) = acc
                        .iter()
                        .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
                    else {
                        break;
                    };
                    if p + 1e-6 < cfg.threshold {
                        break;
                    }
                    best_node = node;
                }
                // Depth 0 is the root itself: no shortcut to take.
                let root_entry = 1;
                if best_node != 0 && best_node != root_entry {
                    m.entries[((ty * tx_n + tx) * rays + r) as usize] = best_node;
                }
            }
        }
    }
    m
}

/// A Strata SBVH viewed as bytes, with the root box node at `root`.
pub struct Box32Bvh<'a> {
    pub bytes: &'a [u8],
    pub root: u64,
}

/// Node pointer → byte offset (`(ptr & !7) * 8`) and type (`ptr & 7`).
pub fn decode_ptr(p: u32) -> (u64, u32) {
    (u64::from(p & !7) * 8, p & 7)
}

/// Box node type in a node pointer.
pub const TYPE_BOX: u32 = 5;

impl Box32Bvh<'_> {
    pub fn node(&self, off: u64) -> Option<(Box32Node, Box32Meta)> {
        let o = off as usize;
        let b = self.bytes.get(o..o + 128)?;
        let u = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        let f = |i: usize| f32::from_bits(u(i));
        let node = Box32Node {
            children: std::array::from_fn(|c| u(4 * c)),
            bounds_min: std::array::from_fn(|c| std::array::from_fn(|a| f(16 + 12 * c + 4 * a))),
            bounds_max: std::array::from_fn(|c| std::array::from_fn(|a| f(64 + 12 * c + 4 * a))),
        };
        let meta = Box32Meta::from_bytes(b[112..128].try_into().ok()?);
        Some((node, meta))
    }

    /// Closest-hit traversal from the box node at byte offset `start`
    /// (the root, or a shortcut). `leaf` tests a leaf pointer and returns
    /// its hit distance. `visit(offset, depth)` sees every box node tested.
    /// Returns the closest leaf hit `(t, pointer)` and the box tests made.
    pub fn closest_hit(
        &self,
        start: u64,
        ray: &Ray,
        mut leaf: impl FnMut(u32) -> Option<f32>,
        mut visit: impl FnMut(u64, u16),
    ) -> (Option<(f32, u32)>, u32) {
        let mut best: Option<(f32, u32)> = None;
        let mut stack = vec![start];
        let mut tests = 0u32;
        let mut ray = *ray;
        while let Some(off) = stack.pop() {
            let Some((node, meta)) = self.node(off) else {
                continue;
            };
            tests += 1;
            visit(off, meta.depth);
            let hits = intersect_box32(&ray, &node);
            // Push far to near so the nearest child is popped first.
            for &h in hits.iter().rev() {
                if h == INVALID_NODE {
                    continue;
                }
                let (child_off, kind) = decode_ptr(h);
                if kind == TYPE_BOX {
                    stack.push(child_off);
                } else if let Some(t) = leaf(h)
                    && best.is_none_or(|(bt, _)| t < bt)
                {
                    best = Some((t, h));
                    ray.t_max = t;
                }
            }
        }
        (best, tests)
    }
}

/// The whole BMA pipeline on the CPU for one ray slot over a
/// `width × height` image (a reference for the GPU path and for tests):
/// samples the tile centres from the root, maps, then traces every other
/// pixel from its shortcut. `ray_at(x, y)` builds a pixel's ray and `leaf`
/// tests leaves. Returns each pixel's hit and the total box tests.
pub fn render_reference(
    bvh: &Box32Bvh<'_>,
    cfg: &BmaConfig,
    width: u32,
    height: u32,
    ray_at: impl Fn(u32, u32) -> Ray,
    leaf: impl Fn(&Ray, u32) -> Option<f32>,
) -> (Vec<Option<(f32, u32)>>, u64) {
    let shift = cfg.tile_shift();
    let tile = 1u32 << shift;
    let (tiles_x, tiles_y) = (width.div_ceil(tile), height.div_ceil(tile));
    let mut paths = SamplePaths::new(tiles_x, tiles_y, 1, cfg.depth);
    let mut out = vec![None; (width * height) as usize];
    let mut tests = 0u64;
    let half = tile / 2;
    // Sampling.
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let (x, y) = (tx * tile + half, ty * tile + half);
            if x >= width || y >= height {
                continue;
            }
            let ray = ray_at(x, y);
            let (hit, n) = bvh.closest_hit(
                bvh.root,
                &ray,
                |p| leaf(&ray, p),
                |off, d| paths.record(tx, ty, 0, u32::from(d), (off - bvh.root) as u32),
            );
            tests += u64::from(n);
            out[(y * width + x) as usize] = hit;
        }
    }
    // Mapping.
    let m = map(&paths, cfg);
    // Traversal.
    for y in 0..height {
        for x in 0..width {
            let (tx, ty) = (x >> shift, y >> shift);
            if x % tile == half && y % tile == half {
                continue;
            }
            let start = m
                .start(tx, ty, 0)
                .map_or(bvh.root, |o| bvh.root + u64::from(o));
            let ray = ray_at(x, y);
            let (hit, n) = bvh.closest_hit(start, &ray, |p| leaf(&ray, p), |_, _| {});
            tests += u64::from(n);
            out[(y * width + x) as usize] = hit;
        }
    }
    (out, tests)
}

/// GPU table layout for a `tiles_x × tiles_y` grid: total bytes and the
/// sample-path array's offset.
pub fn table_layout(tiles_x: u32, tiles_y: u32, rays: u32, depth: u32) -> (u64, u64) {
    let tiles = u64::from(tiles_x) * u64::from(tiles_y);
    let paths = TABLE_HEADER_BYTES + tiles * u64::from(rays) * 4;
    (
        paths + tiles * u64::from(rays) * u64::from(depth) * 4,
        paths,
    )
}

/// The table header bytes for `cfg` (`enabled = false` gives a table that
/// instrumented shaders read but never act on).
pub fn table_header(
    cfg: &BmaConfig,
    tiles_x: u32,
    tiles_y: u32,
    root: u64,
    enabled: bool,
) -> [u8; TABLE_HEADER_BYTES as usize] {
    let (_, paths_off) = table_layout(tiles_x, tiles_y, cfg.rays, cfg.depth);
    let mut h = [0u8; TABLE_HEADER_BYTES as usize];
    let mut put =
        |at: u64, v: u32| h[at as usize..at as usize + 4].copy_from_slice(&v.to_le_bytes());
    let flags = if enabled {
        FLAG_ENABLED | if cfg.primary { FLAG_PRIMARY } else { 0 }
    } else {
        0
    };
    put(hdr::MAGIC, TABLE_MAGIC);
    put(hdr::TILES_X, tiles_x);
    put(hdr::TILES_Y, tiles_y);
    put(hdr::TILE_SHIFT, cfg.tile_shift());
    put(hdr::RAYS, cfg.rays);
    put(hdr::DEPTH, cfg.depth);
    put(hdr::FLAGS, flags);
    put(hdr::ROOT_LO, root as u32);
    put(hdr::ROOT_HI, (root >> 32) as u32);
    put(hdr::THRESHOLD, cfg.threshold.to_bits());
    put(hdr::PATHS_OFFSET, paths_off as u32);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_parsing() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| pairs.iter().find(|p| p.0 == k).map(|p| p.1.to_string())
        };
        assert_eq!(BmaConfig::from_vars(env(&[])), None);
        assert_eq!(
            BmaConfig::from_vars(env(&[("STRATA_BVH_BMA", "off")])),
            None
        );
        assert_eq!(
            BmaConfig::from_vars(env(&[("STRATA_BVH_BMA", "1.5")])),
            None
        );
        let c = BmaConfig::from_vars(env(&[
            ("STRATA_BVH_BMA", "0.75"),
            ("STRATA_BVH_BMA_TILE", "6"),
            ("STRATA_BVH_BMA_PRIMARY", "1"),
        ]))
        .unwrap();
        assert_eq!(c.threshold, 0.75);
        assert_eq!(c.tile, 8);
        assert!(c.primary);
        assert_eq!(
            BmaConfig::from_vars(env(&[("STRATA_BVH_BMA", "on")]))
                .unwrap()
                .threshold,
            0.9
        );
    }

    #[test]
    fn kernel_sums_to_one() {
        let s: f32 = DEFAULT_KERNEL.iter().flatten().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }

    /// The paper's worked example: a 3×3 tile neighbourhood that agrees on
    /// the path down to depth 2, then splits.
    #[test]
    fn mapping_takes_deepest_node_above_threshold() {
        let cfg = BmaConfig {
            threshold: 0.9,
            ..BmaConfig::default()
        };
        let mut p = SamplePaths::new(3, 3, 1, 4);
        for ty in 0..3 {
            for tx in 0..3 {
                p.record(tx, ty, 0, 0, 0); // root
                p.record(tx, ty, 0, 1, 128);
                p.record(tx, ty, 0, 2, 256);
                // Depth 3 diverges: only the centre and one edge agree.
                let d3 = if (tx, ty) == (1, 1) || (tx, ty) == (0, 1) {
                    384
                } else {
                    512 + 128 * (tx + 3 * ty)
                };
                p.record(tx, ty, 0, 3, d3);
            }
        }
        let m = map(&p, &cfg);
        assert_eq!(m.start(1, 1, 0), Some(256));
        // Borders keep the root.
        assert_eq!(m.start(0, 0, 0), None);
        assert_eq!(m.mapped(), 1);
        // Depth 3's best node only carries 4/16 + 2/16 = 0.375.
        let m = map(
            &p,
            &BmaConfig {
                threshold: 0.5,
                ..cfg
            },
        );
        assert_eq!(m.start(1, 1, 0), Some(256));
        let m = map(
            &p,
            &BmaConfig {
                threshold: 0.3,
                ..cfg
            },
        );
        assert_eq!(m.start(1, 1, 0), Some(384));
    }

    #[test]
    fn root_only_agreement_maps_nothing() {
        let mut p = SamplePaths::new(3, 3, 1, 2);
        for ty in 0..3 {
            for tx in 0..3 {
                p.record(tx, ty, 0, 0, 0);
                p.record(tx, ty, 0, 1, 128 * (1 + tx + 3 * ty));
            }
        }
        assert_eq!(map(&p, &BmaConfig::default()).mapped(), 0);
    }

    #[test]
    fn first_visit_per_depth_wins() {
        let mut p = SamplePaths::new(1, 1, 1, 2);
        p.record(0, 0, 0, 1, 128);
        p.record(0, 0, 0, 1, 256);
        assert_eq!(p.get(0, 0, 0, 1), 129);
        p.record(5, 0, 0, 0, 1); // out of range: ignored
        p.clear();
        assert_eq!(p.get(0, 0, 0, 1), 0);
    }

    #[test]
    fn table_layout_and_header() {
        let cfg = BmaConfig::default();
        let (size, paths) = table_layout(10, 5, cfg.rays, cfg.depth);
        assert_eq!(paths, TABLE_HEADER_BYTES + 50 * 4 * 4);
        assert_eq!(size, paths + 50 * 4 * 24 * 4);
        let h = table_header(&cfg, 10, 5, 0x1_0000_0040, true);
        let at = |o: u64| u32::from_le_bytes(h[o as usize..o as usize + 4].try_into().unwrap());
        assert_eq!(at(hdr::MAGIC), TABLE_MAGIC);
        assert_eq!(at(hdr::FLAGS), FLAG_ENABLED);
        assert_eq!(at(hdr::ROOT_HI), 1);
        assert_eq!(at(hdr::TILE_SHIFT), 3);
        assert_eq!(u64::from(at(hdr::PATHS_OFFSET)), paths);
        assert_eq!(at(hdr::FLAGS) & FLAG_PRIMARY, 0);
        let off = table_header(&cfg, 10, 5, 0, false);
        assert_eq!(off[hdr::FLAGS as usize], 0);
    }
}
