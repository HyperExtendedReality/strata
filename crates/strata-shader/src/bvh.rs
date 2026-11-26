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

    let tvec = [ray.origin[0] - v0[0], ray.origin[1] - v0[1], ray.origin[2] - v0[2]];
    let u = dot(tvec, p);
    let q = cross(tvec, e1);
    let v = dot(ray.dir, q);
    let t = dot(e2, q);

    let (inside, t_ok) = if det > 0.0 {
        (u >= 0.0 && v >= 0.0 && u + v <= det, t >= 0.0 && t <= ray.t_max * det)
    } else {
        (u <= 0.0 && v <= 0.0 && u + v >= det, t <= 0.0 && t >= ray.t_max * det)
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
