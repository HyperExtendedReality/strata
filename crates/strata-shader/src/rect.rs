//! Console `D3D_PRIMITIVE_TOPOLOGY_RECTLIST` (topology 13) emulation.
//!
//! Xbox Series titles use 3-vertex rectangles `(v0, v1, v2)` for fullscreen
//! passes, decals, and UI quads; Vulkan has no `RECT_LIST` topology. Each
//! 3-vertex rect expands into 2 triangles (6 vertices) with the implicit 4th
//! vertex computed as `v3 = v1 + v2 - v0`.

/// `D3D_PRIMITIVE_TOPOLOGY_RECTLIST` constant value.
pub const RECTLIST_TOPOLOGY_ID: u32 = 13;

/// Expands 3-vertex screen-space rectangles into 6-vertex triangle lists,
/// synthesizing the 4th corner `v3 = v1 + v2 - v0` for each rectangle.
pub fn expand_rect_list_positions(rects: &[[f32; 4]]) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity((rects.len() / 3) * 6);
    for tri in rects.chunks_exact(3) {
        let [v0, v1, v2] = [tri[0], tri[1], tri[2]];
        let v3 = [
            v1[0] + v2[0] - v0[0],
            v1[1] + v2[1] - v0[1],
            v1[2] + v2[2] - v0[2],
            v1[3] + v2[3] - v0[3],
        ];
        out.extend_from_slice(&[v0, v1, v2, v2, v1, v3]);
    }
    out
}

/// Expands a `RectList` draw's vertex count into the corresponding
/// `TriangleList` vertex count (`(verts / 3) * 6`).
pub fn expanded_vertex_count(rect_vertex_count: u32) -> u32 {
    (rect_vertex_count / 3) * 6
}

/// Generates triangle-list indices mapping each 3-vertex rect `(i, i+1, i+2)`
/// into two triangles when vertex attributes are fetched via BDA with an
/// expanded 4th corner slot at `base + 3`.
pub fn expand_rect_list_indices(rect_count: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(rect_count as usize * 6);
    for r in 0..rect_count {
        let b = r * 4;
        out.extend_from_slice(&[b, b + 1, b + 2, b + 2, b + 1, b + 3]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_three_vertex_rect_into_two_triangles() {
        let v0 = [-1.0, 1.0, 0.0, 1.0];
        let v1 = [1.0, 1.0, 0.0, 1.0];
        let v2 = [-1.0, -1.0, 0.0, 1.0];
        let tris = expand_rect_list_positions(&[v0, v1, v2]);
        assert_eq!(tris.len(), 6);
        assert_eq!(tris[5], [1.0, -1.0, 0.0, 1.0]);
        assert_eq!(expanded_vertex_count(3), 6);
        assert_eq!(expand_rect_list_indices(1), vec![0, 1, 2, 2, 1, 3]);
    }
}
