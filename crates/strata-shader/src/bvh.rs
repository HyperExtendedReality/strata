//! Clean-room CPU reference for RDNA2 `image_bvh64_intersect_ray` (box32 & triangle).
//!
//! Used as the bit-exact verification oracle against both our direct SPIR-V
//! lowering and the Slang `bvh_oracle.slang` GPU shader.

pub const INVALID_NODE: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub t_max: f32,
    pub origin: [f32; 3],
    pub dir: [f32; 3],
    pub inv_dir: [f32; 3],
}

#[derive(Debug, Clone, Copy)]
pub struct Box32Node {
    pub children: [u32; 4],
    pub bounds_min: [[f32; 3]; 4],
    pub bounds_max: [[f32; 3]; 4],
}

/// Four-way AABB slab test + 5-comparator sorting network (nearest hit first).
pub fn intersect_box32(ray: &Ray, node: &Box32Node) -> [u32; 4] {
    let mut hits = [INVALID_NODE; 4];
    let mut dists = [f32::INFINITY; 4];

    for i in 0..4 {
        if node.children[i] == INVALID_NODE {
            continue;
        }
        let mut tmin = 0.0f32;
        let mut tmax = ray.t_max;
        for axis in 0..3 {
            let t0 = (node.bounds_min[i][axis] - ray.origin[axis]) * ray.inv_dir[axis];
            let t1 = (node.bounds_max[i][axis] - ray.origin[axis]) * ray.inv_dir[axis];
            let near = t0.min(t1);
            let far = t0.max(t1);
            tmin = tmin.max(near);
            tmax = tmax.min(far);
        }
        if tmin <= tmax {
            hits[i] = node.children[i];
            dists[i] = tmin;
        }
    }

    // 4-element sorting network: (0,2), (1,3), (0,1), (2,3), (1,2)
    for (a, b) in [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)] {
        if dists[b] < dists[a] {
            dists.swap(a, b);
            hits.swap(a, b);
        }
    }

    hits
}

/// Strata metadata in the Box32 node trailer (bytes `112..128`), which the
/// RDNA2 box test never reads. Written by the SBVH builder; read by the BMA
/// traversal shortcut (`depth`) and the optional child-prior pruning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Box32Meta {
    /// Parent node pointer (`offset >> 3 | type`), [`INVALID_NODE`] at the root.
    pub parent: u32,
    /// Depth below the root (root = 0).
    pub depth: u16,
    /// Leaves under the node (saturating).
    pub leaf_count: u16,
    /// Per-child surface-area ratio `SA(child) / SA(node)` as binary16
    /// (0 for empty slots).
    pub child_priors: [u16; 4],
}

impl Box32Meta {
    /// Byte offset of the trailer in a Box32 node.
    pub const OFFSET: usize = 112;
    /// Byte offset of `depth` (low half of the dword) in a Box32 node.
    pub const DEPTH_OFFSET: usize = 116;

    pub fn to_bytes(&self) -> [u8; 16] {
        let mut o = [0u8; 16];
        o[0..4].copy_from_slice(&self.parent.to_le_bytes());
        o[4..6].copy_from_slice(&self.depth.to_le_bytes());
        o[6..8].copy_from_slice(&self.leaf_count.to_le_bytes());
        for (i, p) in self.child_priors.iter().enumerate() {
            o[8 + 2 * i..10 + 2 * i].copy_from_slice(&p.to_le_bytes());
        }
        o
    }

    pub fn from_bytes(b: &[u8; 16]) -> Self {
        let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
        Self {
            parent: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            depth: u16_at(4),
            leaf_count: u16_at(6),
            child_priors: std::array::from_fn(|i| u16_at(8 + 2 * i)),
        }
    }

    /// Child priors as `f32`.
    pub fn priors(&self) -> [f32; 4] {
        self.child_priors.map(f16_to_f32)
    }
}

/// Box test with optional surface-area-prior pruning of the non-nearest
/// children (a Strata heuristic built on [`Box32Meta::child_priors`]; the
/// paper's BMA algorithm itself is [`crate::bma`], which never drops
/// children and instead shortcuts the traversal start node).
///
/// Children after the nearest whose `prior / (1 + max(t_near, 0))` is below
/// `threshold` are returned as [`INVALID_NODE`]. The nearest hit is always
/// kept; `threshold <= 0` matches [`intersect_box32`] bit-for-bit.
pub fn intersect_box32_prior_pruned(
    ray: &Ray,
    node: &Box32Node,
    priors: [f32; 4],
    bma_threshold: f32,
) -> [u32; 4] {
    let mut hits = [INVALID_NODE; 4];
    let mut dists = [f32::INFINITY; 4];
    let mut p = priors;

    for i in 0..4 {
        if node.children[i] == INVALID_NODE {
            continue;
        }
        let mut tmin = 0.0f32;
        let mut tmax = ray.t_max;
        for axis in 0..3 {
            let t0 = (node.bounds_min[i][axis] - ray.origin[axis]) * ray.inv_dir[axis];
            let t1 = (node.bounds_max[i][axis] - ray.origin[axis]) * ray.inv_dir[axis];
            let near = t0.min(t1);
            let far = t0.max(t1);
            tmin = tmin.max(near);
            tmax = tmax.min(far);
        }
        if tmin <= tmax {
            hits[i] = node.children[i];
            dists[i] = tmin;
        }
    }

    for (a, b) in [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)] {
        if dists[b] < dists[a] {
            dists.swap(a, b);
            hits.swap(a, b);
            p.swap(a, b);
        }
    }

    if bma_threshold > 0.0 {
        for k in 1..4 {
            if hits[k] != INVALID_NODE {
                let score = p[k] / (1.0 + dists[k].max(0.0));
                if score < bma_threshold {
                    hits[k] = INVALID_NODE;
                }
            }
        }
    }

    hits
}

/// Watertight ray-triangle test returning `(t_num, t_den, u_num, v_num)`.
/// Miss returns `(+inf, 1.0, 0.0, 0.0)` matching RDNA2 hardware convention.
pub fn intersect_triangle(ray: &Ray, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> [f32; 4] {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let p = cross(ray.dir, e2);
    let det = dot(e1, p);
    if det == 0.0 || det.is_nan() {
        return [f32::INFINITY, 1.0, 0.0, 0.0];
    }

    let tvec = [
        ray.origin[0] - v0[0],
        ray.origin[1] - v0[1],
        ray.origin[2] - v0[2],
    ];
    let u = dot(tvec, p);
    let q = cross(tvec, e1);
    let v = dot(ray.dir, q);
    let t = dot(e2, q);

    let (inside, t_ok) = if det > 0.0 {
        (
            u >= 0.0 && v >= 0.0 && u + v <= det,
            t >= 0.0 && t <= ray.t_max * det,
        )
    } else {
        (
            u <= 0.0 && v <= 0.0 && u + v >= det,
            t <= 0.0 && t >= ray.t_max * det,
        )
    };

    if inside && t_ok {
        [t, det, u, v]
    } else {
        [f32::INFINITY, 1.0, 0.0, 0.0]
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// RDNA2 Box16 BVH node storing four IEEE-754 binary16 AABBs.
#[derive(Debug, Clone, Copy)]
pub struct Box16Node {
    pub children: [u32; 4],
    pub bounds_min_f16: [[u16; 3]; 4],
    pub bounds_max_f16: [[u16; 3]; 4],
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        sign | (0xFF << 23) | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Four-way Box16 AABB intersection (expands fp16 bounds to fp32 and runs [`intersect_box32`]).
pub fn intersect_box16(ray: &Ray, node: &Box16Node) -> [u32; 4] {
    let mut b32 = Box32Node {
        children: node.children,
        bounds_min: [[0.0; 3]; 4],
        bounds_max: [[0.0; 3]; 4],
    };
    for i in 0..4 {
        for a in 0..3 {
            b32.bounds_min[i][a] = f16_to_f32(node.bounds_min_f16[i][a]);
            b32.bounds_max[i][a] = f16_to_f32(node.bounds_max_f16[i][a]);
        }
    }
    intersect_box32(ray, &b32)
}

/// RDNA2 BVH node pointer kind encoded in the low 3 bits (`node_ptr & 0x7`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvhNodeKind {
    Triangle(u8),
    Box16,
    Box32,
    Instance,
    Procedural,
}

impl BvhNodeKind {
    pub fn from_node_ptr(node_ptr: u32) -> Self {
        match node_ptr & 0x7 {
            t @ 0..=3 => Self::Triangle(t as u8),
            4 => Self::Box16,
            5 => Self::Box32,
            6 => Self::Instance,
            _ => Self::Procedural,
        }
    }
}

/// RDNA2 128-byte TLAS Instance node: 3x4 row-major inverse transform + BLAS root pointer.
#[derive(Debug, Clone, Copy)]
pub struct InstanceNode {
    /// World-to-object 3x4 affine matrix (rows: x, y, z).
    pub inv_transform: [[f32; 4]; 3],
    pub blas_root_ptr: u32,
    pub instance_id: u32,
    pub instance_mask: u8,
}

/// Transforms `ray` into object space using `node.inv_transform` and returns
/// `(transformed_ray, blas_root_ptr)` if `ray_mask & node.instance_mask != 0`.
pub fn intersect_instance(ray: &Ray, node: &InstanceNode, ray_mask: u8) -> Option<(Ray, u32)> {
    if (ray_mask & node.instance_mask) == 0 || node.blas_root_ptr == INVALID_NODE {
        return None;
    }
    let m = &node.inv_transform;
    let o = ray.origin;
    let d = ray.dir;
    let new_origin = [
        m[0][0] * o[0] + m[0][1] * o[1] + m[0][2] * o[2] + m[0][3],
        m[1][0] * o[0] + m[1][1] * o[1] + m[1][2] * o[2] + m[1][3],
        m[2][0] * o[0] + m[2][1] * o[1] + m[2][2] * o[2] + m[2][3],
    ];
    let new_dir = [
        m[0][0] * d[0] + m[0][1] * d[1] + m[0][2] * d[2],
        m[1][0] * d[0] + m[1][1] * d[1] + m[1][2] * d[2],
        m[2][0] * d[0] + m[2][1] * d[1] + m[2][2] * d[2],
    ];
    let inv = |v: f32| if v == 0.0 { f32::INFINITY } else { 1.0 / v };
    Some((
        Ray {
            t_max: ray.t_max,
            origin: new_origin,
            dir: new_dir,
            inv_dir: [inv(new_dir[0]), inv(new_dir[1]), inv(new_dir[2])],
        },
        node.blas_root_ptr,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box32_sorts_hits_nearest_first_and_instance_transforms_ray() {
        let ray = Ray {
            t_max: 100.0,
            origin: [0.0, 0.0, 0.0],
            dir: [0.0, 0.0, 1.0],
            inv_dir: [f32::INFINITY, f32::INFINITY, 1.0],
        };
        let node = Box32Node {
            children: [10, 20, INVALID_NODE, 30],
            bounds_min: [
                [-1.0, -1.0, 10.0],
                [-1.0, -1.0, 2.0],
                [0.0, 0.0, 0.0],
                [-1.0, -1.0, 5.0],
            ],
            bounds_max: [
                [1.0, 1.0, 12.0],
                [1.0, 1.0, 4.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 6.0],
            ],
        };
        assert_eq!(intersect_box32(&ray, &node), [20, 30, 10, INVALID_NODE]);

        let inst = InstanceNode {
            inv_transform: [
                [1.0, 0.0, 0.0, -5.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            blas_root_ptr: 0x45,
            instance_id: 7,
            instance_mask: 0xFF,
        };
        let (tr, root) = intersect_instance(&ray, &inst, 0x01).unwrap();
        assert_eq!(root, 0x45);
        assert_eq!(tr.origin, [-5.0, 0.0, 0.0]);
        assert_eq!(BvhNodeKind::from_node_ptr(0x45), BvhNodeKind::Box32);
    }
}
