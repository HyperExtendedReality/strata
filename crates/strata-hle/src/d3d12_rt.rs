//! `BuildRaytracingAccelerationStructure`: bounding volume hierarchies built
//! into title memory, in the node layout Strata's translated
//! `image_bvh64_intersect_ray` reads (No-API: an acceleration structure is
//! plain GPU memory the title's own traversal code walks; no Vulkan
//! acceleration-structure object exists).
//!
//! Layout (Strata's model, offsets from the structure's base):
//!
//! | at | what |
//! | --- | --- |
//! | 0 | header: `'SBVH'`, version, used bytes (u64), node count, root pointer, kind, primitive count, root bounds (min xyz, max xyz) — 64 bytes |
//! | 64.. | 4-wide box nodes, 128 bytes: child pointers `[u32; 4]`, child `i`'s min at `16 + 12i`, max at `64 + 12i` (the offsets the SPIR-V lowering loads), 16 bytes padding |
//! | .. | leaves: triangles (64 bytes: three vertices, geometry index, primitive index, flags), AABBs (64 bytes: min, max, geometry, primitive), instances (128 bytes: bottom level address, world-to-object and object-to-world 3x4, ID/mask, contribution/flags) |
//!
//! Child pointers follow the RDNA2 convention of a node pointer: the byte
//! offset from the base divided by 8, with the node type in the low three
//! bits (0 triangle, 5 box, 6 instance, 7 procedural); `u32::MAX` is empty.
//! The translated intersection instruction takes absolute node addresses,
//! so traversal code adds the base back (`base + (ptr & !7) * 8`).

use strata_gpu::thin_list::{AccelBuild, AccelGeometry};

pub const MAGIC: u32 = u32::from_le_bytes(*b"SBVH");
pub const HEADER_BYTES: u64 = 64;
/// Box nodes use 112 bytes and pad to 128: node pointers (`address >> 3`
/// with the type in the low 3 bits) need 64-byte-aligned nodes.
pub const BOX_BYTES: u64 = 128;
pub const TRIANGLE_BYTES: u64 = 64;
pub const AABB_BYTES: u64 = 64;
pub const INSTANCE_BYTES: u64 = 128;
pub const TYPE_TRIANGLE: u32 = 0;
pub const TYPE_BOX: u32 = 5;
pub const TYPE_INSTANCE: u32 = 6;
pub const TYPE_PROCEDURAL: u32 = 7;
pub const INVALID: u32 = u32::MAX;

/// Title memory access for the builder (by GPU address).
pub struct Memory<'a> {
    pub read: Box<dyn Fn(u64, usize) -> Option<Vec<u8>> + 'a>,
    pub write: Box<dyn Fn(u64, &[u8]) -> bool + 'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Aabb {
    min: [f32; 3],
    max: [f32; 3],
}

impl Aabb {
    const EMPTY: Aabb = Aabb {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };
    fn grow(&mut self, p: [f32; 3]) {
        for a in 0..3 {
            self.min[a] = self.min[a].min(p[a]);
            self.max[a] = self.max[a].max(p[a]);
        }
    }
    fn union(mut self, o: &Aabb) -> Aabb {
        self.grow(o.min);
        self.grow(o.max);
        self
    }
    fn centroid(&self) -> [f32; 3] {
        std::array::from_fn(|a| 0.5 * (self.min[a] + self.max[a]))
    }
}

/// A leaf to place: its bounds and the bytes of its node.
struct Leaf {
    bounds: Aabb,
    kind: u32,
    bytes: Vec<u8>,
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn half(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = (h >> 10) & 0x1f;
    let m = f32::from(h & 0x3ff);
    match e {
        0 => s * m * 2f32.powi(-24),
        31 => s * f32::INFINITY,
        _ => s * (1.0 + m / 1024.0) * 2f32.powi(i32::from(e) - 15),
    }
}

/// Reads vertex `i` in `format` at `va + i * stride`.
fn vertex(mem: &Memory<'_>, va: u64, stride: u64, format: u32, i: u64) -> Option<[f32; 3]> {
    let at = va + i * stride;
    Some(match format {
        16 => {
            let f = f32s(&(mem.read)(at, 8)?);
            [f[0], f[1], 0.0]
        }
        10 | 34 => {
            let b = (mem.read)(at, if format == 10 { 6 } else { 4 })?;
            let h: Vec<f32> = b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half(u16::from_le_bytes(*c)))
                .collect();
            [h[0], h[1], h.get(2).copied().unwrap_or(0.0)]
        }
        _ => {
            let f = f32s(&(mem.read)(at, 12)?);
            [f[0], f[1], f[2]]
        }
    })
}

fn transform(m: &[f32], p: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| {
        m[4 * r] * p[0] + m[4 * r + 1] * p[1] + m[4 * r + 2] * p[2] + m[4 * r + 3]
    })
}

/// The inverse of a 3x4 affine transform (row-major).
fn invert(m: &[f32]) -> [f32; 12] {
    let a = |r: usize, c: usize| m[4 * r + c];
    let det = a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1))
        - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0))
        + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
    let inv_det = if det.abs() > f32::MIN_POSITIVE {
        1.0 / det
    } else {
        0.0
    };
    let mut r = [[0f32; 3]; 3];
    r[0][0] = (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) * inv_det;
    r[0][1] = (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) * inv_det;
    r[0][2] = (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) * inv_det;
    r[1][0] = (a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) * inv_det;
    r[1][1] = (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) * inv_det;
    r[1][2] = (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) * inv_det;
    r[2][0] = (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)) * inv_det;
    r[2][1] = (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) * inv_det;
    r[2][2] = (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) * inv_det;
    let t = [a(0, 3), a(1, 3), a(2, 3)];
    let mut out = [0f32; 12];
    for row in 0..3 {
        for c in 0..3 {
            out[4 * row + c] = r[row][c];
        }
        out[4 * row + 3] = -(r[row][0] * t[0] + r[row][1] * t[1] + r[row][2] * t[2]);
    }
    out
}

fn leaves_of_geometry(
    mem: &Memory<'_>,
    g: &AccelGeometry,
    geom_index: u32,
    out: &mut Vec<Leaf>,
) -> Result<(), String> {
    match *g {
        AccelGeometry::Triangles {
            transform: xf,
            index_format,
            vertex_format,
            index_count,
            vertex_count,
            index_va,
            vertex_va,
            vertex_stride,
            opaque,
        } => {
            let m = if xf != 0 {
                Some(f32s(&(mem.read)(xf, 48).ok_or("transform unreadable")?))
            } else {
                None
            };
            let tris = if index_format == 0 {
                vertex_count / 3
            } else {
                index_count / 3
            };
            let index_bytes = match index_format {
                0 => 0,
                57 => 2,
                _ => 4,
            };
            let indices = if index_bytes > 0 {
                Some(
                    (mem.read)(index_va, (tris * 3 * index_bytes) as usize)
                        .ok_or("index buffer unreadable")?,
                )
            } else {
                None
            };
            let idx = |k: usize| -> u64 {
                match &indices {
                    None => k as u64,
                    Some(b) if index_bytes == 2 => {
                        u64::from(u16::from_le_bytes([b[2 * k], b[2 * k + 1]]))
                    }
                    Some(b) => {
                        u64::from(u32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap()))
                    }
                }
            };
            for t in 0..tris as usize {
                let mut v = [[0f32; 3]; 3];
                for (c, vc) in v.iter_mut().enumerate() {
                    let p = vertex(mem, vertex_va, vertex_stride, vertex_format, idx(3 * t + c))
                        .ok_or("vertex buffer unreadable")?;
                    *vc = m.as_ref().map_or(p, |m| transform(m, p));
                }
                let mut bounds = Aabb::EMPTY;
                let mut bytes = Vec::with_capacity(TRIANGLE_BYTES as usize);
                for p in &v {
                    bounds.grow(*p);
                    for x in p {
                        bytes.extend_from_slice(&x.to_le_bytes());
                    }
                }
                bytes.extend_from_slice(&geom_index.to_le_bytes());
                bytes.extend_from_slice(&(t as u32).to_le_bytes());
                bytes.extend_from_slice(&u32::from(opaque).to_le_bytes());
                bytes.resize(TRIANGLE_BYTES as usize, 0);
                out.push(Leaf {
                    bounds,
                    kind: TYPE_TRIANGLE,
                    bytes,
                });
            }
        }
        AccelGeometry::Aabbs {
            count,
            va,
            stride,
            opaque,
        } => {
            for i in 0..count {
                let f = f32s(&(mem.read)(va + i * stride.max(24), 24).ok_or("AABB unreadable")?);
                let bounds = Aabb {
                    min: [f[0], f[1], f[2]],
                    max: [f[3], f[4], f[5]],
                };
                let mut bytes = Vec::with_capacity(AABB_BYTES as usize);
                for x in &f[..6] {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                bytes.extend_from_slice(&geom_index.to_le_bytes());
                bytes.extend_from_slice(&(i as u32).to_le_bytes());
                bytes.extend_from_slice(&u32::from(opaque).to_le_bytes());
                bytes.resize(AABB_BYTES as usize, 0);
                out.push(Leaf {
                    bounds,
                    kind: TYPE_PROCEDURAL,
                    bytes,
                });
            }
        }
    }
    Ok(())
}

/// `D3D12_RAYTRACING_INSTANCE_DESC`s → instance leaves, bounded by their
/// bottom levels' root bounds (from each bottom level's header).
fn leaves_of_instances(
    mem: &Memory<'_>,
    va: u64,
    count: u32,
    out: &mut Vec<Leaf>,
) -> Result<(), String> {
    for i in 0..u64::from(count) {
        let d = (mem.read)(va + 64 * i, 64).ok_or("instance descs unreadable")?;
        let xf = f32s(&d[..48]);
        let id_mask = u32::from_le_bytes(d[48..52].try_into().unwrap());
        let contrib_flags = u32::from_le_bytes(d[52..56].try_into().unwrap());
        let blas = u64::from_le_bytes(d[56..64].try_into().unwrap());
        if blas == 0 || id_mask >> 24 == 0 {
            continue; // null or fully masked: never hit
        }
        let h = (mem.read)(blas, HEADER_BYTES as usize).ok_or("bottom level unreadable")?;
        if u32::from_le_bytes(h[0..4].try_into().unwrap()) != MAGIC {
            return Err(format!(
                "instance {i}: bottom level at 0x{blas:x} is not a Strata BVH"
            ));
        }
        let rb = f32s(&h[32..56]);
        let mut bounds = Aabb::EMPTY;
        for c in 0..8 {
            let p = [
                if c & 1 == 0 { rb[0] } else { rb[3] },
                if c & 2 == 0 { rb[1] } else { rb[4] },
                if c & 4 == 0 { rb[2] } else { rb[5] },
            ];
            bounds.grow(transform(&xf, p));
        }
        let mut bytes = Vec::with_capacity(INSTANCE_BYTES as usize);
        bytes.extend_from_slice(&blas.to_le_bytes());
        for x in invert(&xf) {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        for x in &xf {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        bytes.extend_from_slice(&id_mask.to_le_bytes());
        bytes.extend_from_slice(&contrib_flags.to_le_bytes());
        bytes.resize(INSTANCE_BYTES as usize, 0);
        out.push(Leaf {
            bounds,
            kind: TYPE_INSTANCE,
            bytes,
        });
    }
    Ok(())
}

/// A node of the tree being built.
enum Node {
    Leaf(usize),
    Inner(Vec<Node>, Aabb),
}

impl Node {
    fn bounds(&self, leaves: &[Leaf]) -> Aabb {
        match self {
            Node::Leaf(i) => leaves[*i].bounds,
            Node::Inner(_, b) => *b,
        }
    }
}

/// Splits `items` at the median of their centroids along the widest axis.
fn split(leaves: &[Leaf], items: &mut [usize]) -> usize {
    let mut cb = Aabb::EMPTY;
    for &i in items.iter() {
        cb.grow(leaves[i].bounds.centroid());
    }
    let axis = (0..3)
        .max_by(|&a, &b| {
            (cb.max[a] - cb.min[a])
                .partial_cmp(&(cb.max[b] - cb.min[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    items.sort_by(|&a, &b| {
        leaves[a].bounds.centroid()[axis]
            .partial_cmp(&leaves[b].bounds.centroid()[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    items.len() / 2
}

/// A 4-wide tree over `items`: up to four groups, each split recursively.
fn build_node(leaves: &[Leaf], items: &mut [usize]) -> Node {
    if items.len() == 1 {
        return Node::Leaf(items[0]);
    }
    let mut groups: Vec<&mut [usize]> = vec![items];
    while groups.len() < 4 {
        let Some((gi, _)) = groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.len() > 1)
            .max_by_key(|(_, g)| g.len())
        else {
            break;
        };
        let g = groups.swap_remove(gi);
        let mid = split(leaves, g);
        let (a, b) = g.split_at_mut(mid);
        groups.push(a);
        groups.push(b);
    }
    let children: Vec<Node> = groups.into_iter().map(|g| build_node(leaves, g)).collect();
    let bounds = children
        .iter()
        .fold(Aabb::EMPTY, |acc, c| acc.union(&c.bounds(leaves)));
    Node::Inner(children, bounds)
}

fn count_inner(n: &Node) -> u64 {
    match n {
        Node::Leaf(_) => 0,
        Node::Inner(c, _) => 1 + c.iter().map(count_inner).sum::<u64>(),
    }
}

/// Upper bound on the bytes a structure over `prims` primitives takes.
pub fn max_size(prims: u64, leaf_bytes: u64) -> u64 {
    HEADER_BYTES + prims.max(1) * (BOX_BYTES + leaf_bytes)
}

/// Serializes the tree: box nodes first (root at `HEADER_BYTES`), then leaves.
fn serialize(root: &Node, leaves: &[Leaf]) -> (Vec<u8>, u32) {
    let inner = count_inner(root);
    let leaf_base = HEADER_BYTES + inner * BOX_BYTES;
    let mut leaf_off = Vec::with_capacity(leaves.len());
    let mut at = leaf_base;
    for l in leaves {
        leaf_off.push(at);
        at += l.bytes.len() as u64;
    }
    let total = at;
    let mut out = vec![0u8; total as usize];
    // Every node is 64-byte aligned, so `off / 8` has its low 3 bits free.
    let ptr = |off: u64, kind: u32| (off / 8) as u32 | kind;
    // Breadth-first placement of box nodes.
    let mut next_box = HEADER_BYTES;
    let mut queue: std::collections::VecDeque<(&Node, u64)> = std::collections::VecDeque::new();
    let root_ptr = match root {
        Node::Leaf(i) => ptr(leaf_off[*i], leaves[*i].kind),
        Node::Inner(..) => {
            queue.push_back((root, next_box));
            next_box += BOX_BYTES;
            ptr(HEADER_BYTES, TYPE_BOX)
        }
    };
    while let Some((node, off)) = queue.pop_front() {
        let Node::Inner(children, _) = node else {
            continue;
        };
        let o = off as usize;
        for slot in 0..4 {
            let (p, b) = match children.get(slot) {
                Some(Node::Leaf(i)) => (ptr(leaf_off[*i], leaves[*i].kind), leaves[*i].bounds),
                Some(c @ Node::Inner(_, b)) => {
                    let child_off = next_box;
                    next_box += BOX_BYTES;
                    queue.push_back((c, child_off));
                    (ptr(child_off, TYPE_BOX), *b)
                }
                None => (
                    INVALID,
                    Aabb {
                        min: [0.0; 3],
                        max: [0.0; 3],
                    },
                ),
            };
            out[o + 4 * slot..o + 4 * slot + 4].copy_from_slice(&p.to_le_bytes());
            for a in 0..3 {
                let mn = 16 + 12 * slot + 4 * a;
                let mx = 64 + 12 * slot + 4 * a;
                out[o + mn..o + mn + 4].copy_from_slice(&b.min[a].to_le_bytes());
                out[o + mx..o + mx + 4].copy_from_slice(&b.max[a].to_le_bytes());
            }
        }
    }
    for (l, &off) in leaves.iter().zip(&leaf_off) {
        out[off as usize..off as usize + l.bytes.len()].copy_from_slice(&l.bytes);
    }
    (out, root_ptr)
}

/// Builds `b` into title memory.
pub fn build(mem: &Memory<'_>, b: &AccelBuild) -> Result<u64, String> {
    let mut leaves = Vec::new();
    if b.top_level {
        leaves_of_instances(mem, b.instances.0, b.instances.1, &mut leaves)?;
    } else {
        for (gi, g) in b.geometries.iter().enumerate() {
            leaves_of_geometry(mem, g, gi as u32, &mut leaves)?;
        }
    }
    let (mut bytes, root_ptr, bounds, inner) = if leaves.is_empty() {
        (
            vec![0u8; HEADER_BYTES as usize],
            INVALID,
            Aabb {
                min: [0.0; 3],
                max: [0.0; 3],
            },
            0,
        )
    } else {
        let mut items: Vec<usize> = (0..leaves.len()).collect();
        let root = build_node(&leaves, &mut items);
        let bounds = root.bounds(&leaves);
        let inner = count_inner(&root);
        let (bytes, root_ptr) = serialize(&root, &leaves);
        (bytes, root_ptr, bounds, inner)
    };
    let used = bytes.len() as u64;
    if b.dest_size != 0 && used > b.dest_size {
        return Err(format!(
            "needs {used} bytes, the destination has {}",
            b.dest_size
        ));
    }
    let mut header = Vec::with_capacity(HEADER_BYTES as usize);
    header.extend_from_slice(&MAGIC.to_le_bytes());
    header.extend_from_slice(&1u32.to_le_bytes());
    header.extend_from_slice(&used.to_le_bytes());
    header.extend_from_slice(&(inner as u32).to_le_bytes());
    header.extend_from_slice(&root_ptr.to_le_bytes());
    header.extend_from_slice(&u32::from(b.top_level).to_le_bytes());
    header.extend_from_slice(&(leaves.len() as u32).to_le_bytes());
    for x in bounds.min.iter().chain(&bounds.max) {
        header.extend_from_slice(&x.to_le_bytes());
    }
    header.resize(HEADER_BYTES as usize, 0);
    bytes[..HEADER_BYTES as usize].copy_from_slice(&header);
    if !(mem.write)(b.dest, &bytes) {
        return Err(format!("destination 0x{:x} is not title memory", b.dest));
    }
    Ok(used)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use strata_shader::bvh::{Box32Node, Ray, intersect_box32};

    /// A sparse byte memory for the builder.
    fn memory(m: &RefCell<HashMap<u64, u8>>) -> Memory<'_> {
        Memory {
            read: Box::new(move |a, n| {
                let m = m.borrow();
                (0..n as u64).map(|i| m.get(&(a + i)).copied()).collect()
            }),
            write: Box::new(move |a, b| {
                let mut m = m.borrow_mut();
                for (i, &x) in b.iter().enumerate() {
                    m.insert(a + i as u64, x);
                }
                true
            }),
        }
    }

    fn put(m: &RefCell<HashMap<u64, u8>>, at: u64, b: &[u8]) {
        let mut m = m.borrow_mut();
        for (i, &x) in b.iter().enumerate() {
            m.insert(at + i as u64, x);
        }
    }

    fn box_node(m: &RefCell<HashMap<u64, u8>>, at: u64) -> Box32Node {
        let r = |o: u64| {
            let m = m.borrow();
            let b: Vec<u8> = (0..4).map(|i| m[&(at + o + i)]).collect();
            b.try_into().unwrap()
        };
        Box32Node {
            children: std::array::from_fn(|i| u32::from_le_bytes(r(4 * i as u64))),
            bounds_min: std::array::from_fn(|i| {
                std::array::from_fn(|a| f32::from_le_bytes(r(16 + 12 * i as u64 + 4 * a as u64)))
            }),
            bounds_max: std::array::from_fn(|i| {
                std::array::from_fn(|a| f32::from_le_bytes(r(64 + 12 * i as u64 + 4 * a as u64)))
            }),
        }
    }

    #[test]
    fn a_built_bvh_is_traversed_by_the_intersection_reference() {
        // Ten unit triangles along +x at z = 5; a ray down -z at x = 7.25 must
        // reach exactly the leaf of triangle 7.
        let mem_cells = RefCell::new(HashMap::new());
        let mut verts = Vec::new();
        for t in 0..10 {
            let x = t as f32;
            for p in [[x, 0.0, 5.0], [x + 1.0, 0.0, 5.0], [x, 1.0, 5.0]] {
                for c in p {
                    verts.extend_from_slice(&f32::to_le_bytes(c));
                }
            }
        }
        put(&mem_cells, 0x1000, &verts);
        let mem = memory(&mem_cells);
        let job = AccelBuild {
            dest: 0x10_0000,
            dest_size: 0,
            top_level: false,
            geometries: vec![AccelGeometry::Triangles {
                transform: 0,
                index_format: 0,
                vertex_format: 6,
                index_count: 0,
                vertex_count: 30,
                index_va: 0,
                vertex_va: 0x1000,
                vertex_stride: 12,
                opaque: true,
            }],
            instances: (0, 0),
        };
        let used = build(&mem, &job).unwrap();
        assert!(used <= max_size(10, TRIANGLE_BYTES));

        let dir = [0.0f32, 0.0, -1.0];
        let ray = Ray {
            t_max: 100.0,
            origin: [7.25, 0.25, 10.0],
            dir,
            inv_dir: dir.map(|d| if d == 0.0 { f32::INFINITY } else { 1.0 / d }),
        };
        let header_root = {
            let m = mem_cells.borrow();
            let b: Vec<u8> = (0..4).map(|i| m[&(0x10_0000 + 20 + i)]).collect();
            u32::from_le_bytes(b.try_into().unwrap())
        };
        assert_eq!(header_root & 7, TYPE_BOX);
        let mut stack = vec![header_root];
        let mut hit_prims = Vec::new();
        while let Some(p) = stack.pop() {
            let off = u64::from(p & !7) * 8;
            if p & 7 == TYPE_BOX {
                let node = box_node(&mem_cells, 0x10_0000 + off);
                stack.extend(
                    intersect_box32(&ray, &node)
                        .into_iter()
                        .filter(|&c| c != INVALID),
                );
            } else {
                let m = mem_cells.borrow();
                let b: Vec<u8> = (0..4).map(|i| m[&(0x10_0000 + off + 40 + i)]).collect();
                hit_prims.push(u32::from_le_bytes(b.try_into().unwrap()));
            }
        }
        assert_eq!(hit_prims, [7]);
    }
}
