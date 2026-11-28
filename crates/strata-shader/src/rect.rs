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

/// `ThinCommand::SetTopology` value marking a `RECTLIST` draw (Vulkan has no
/// rect topology; the draw runs as a triangle list through
/// [`emit_rectlist_gs_spirv`]).
pub const RECTLIST_VK_MARKER: i32 = -(RECTLIST_TOPOLOGY_ID as i32);

/// `Location`s of a vertex shader's `vec4` output variables (its parameter
/// exports), ascending.
pub fn vs_param_locations(spirv: &[u32]) -> Vec<u32> {
    let mut locations = std::collections::HashMap::new();
    let mut outputs = Vec::new();
    let mut at = 5;
    while at < spirv.len() {
        let word = spirv[at];
        let (count, op) = ((word >> 16) as usize, word & 0xFFFF);
        if count == 0 || at + count > spirv.len() {
            break;
        }
        let ops = &spirv[at + 1..at + count];
        match op {
            // OpDecorate %id Location n
            71 if ops.len() >= 3 && ops[1] == 30 => {
                locations.insert(ops[0], ops[2]);
            }
            // OpVariable %ty %id Output
            59 if ops.len() >= 3 && ops[2] == 3 => outputs.push(ops[1]),
            _ => {}
        }
        at += count;
    }
    let mut out: Vec<u32> = outputs
        .iter()
        .filter_map(|id| locations.get(id).copied())
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// A SPIR-V 1.6 geometry shader turning each `RECTLIST` rectangle (a
/// triangle `v0, v1, v2` from the vertex shader) into a 4-vertex triangle
/// strip with `v3 = v1 + v2 - v0` for the position and every `vec4`
/// parameter at `locations` — the fourth corner the console rasterizer
/// synthesizes.
pub fn emit_rectlist_gs_spirv(locations: &[u32]) -> Vec<u32> {
    let mut next = 1u32;
    let mut id = || {
        let v = next;
        next += 1;
        v
    };
    fn inst(buf: &mut Vec<u32>, op: u32, operands: &[u32]) {
        buf.push(((1 + operands.len() as u32) << 16) | op);
        buf.extend_from_slice(operands);
    }
    let (main, void, fn_ty, f32_ty, vec4, u32_ty) = (id(), id(), id(), id(), id(), id());
    let (c0, c1, c2, c3) = (id(), id(), id(), id());
    let (arr3, ptr_in_arr, ptr_in_v4, ptr_out_v4) = (id(), id(), id(), id());
    // Attribute 0 is the position; then one per parameter location.
    let attrs: Vec<(Option<u32>, u32, u32)> = std::iter::once(None)
        .chain(locations.iter().copied().map(Some))
        .map(|loc| (loc, id(), id()))
        .collect();
    let label = id();

    let mut body = Vec::new();
    let mut loaded = Vec::new();
    for &(_, input, _) in &attrs {
        let mut corners = [0u32; 4];
        for (k, c) in [c0, c1, c2].into_iter().enumerate() {
            let (ptr, val) = (id(), id());
            inst(&mut body, 65, &[ptr_in_v4, ptr, input, c]); // OpAccessChain
            inst(&mut body, 61, &[vec4, val, ptr]); // OpLoad
            corners[k] = val;
        }
        let (sum, v3) = (id(), id());
        inst(&mut body, 129, &[vec4, sum, corners[1], corners[2]]); // OpFAdd
        inst(&mut body, 131, &[vec4, v3, sum, corners[0]]); // OpFSub
        corners[3] = v3;
        loaded.push(corners);
    }
    for k in 0..4 {
        for (&(_, _, output), corners) in attrs.iter().zip(&loaded) {
            inst(&mut body, 62, &[output, corners[k]]); // OpStore
        }
        inst(&mut body, 218, &[]); // OpEmitVertex
    }
    inst(&mut body, 219, &[]); // OpEndPrimitive

    let mut w = vec![0x0723_0203, 0x0001_0600, 0x0053_5452, 0, 0];
    inst(&mut w, 17, &[1]); // Capability Shader
    inst(&mut w, 17, &[2]); // Capability Geometry
    inst(&mut w, 14, &[0, 1]); // MemoryModel Logical GLSL450
    let mut entry = vec![3, main, u32::from_le_bytes(*b"main"), 0];
    for &(_, i, o) in &attrs {
        entry.extend_from_slice(&[i, o]);
    }
    inst(&mut w, 15, &entry);
    inst(&mut w, 16, &[main, 22]); // Triangles
    inst(&mut w, 16, &[main, 0, 1]); // Invocations 1
    inst(&mut w, 16, &[main, 29]); // OutputTriangleStrip
    inst(&mut w, 16, &[main, 26, 4]); // OutputVertices 4
    for &(loc, i, o) in &attrs {
        match loc {
            None => {
                inst(&mut w, 71, &[i, 11, 0]); // BuiltIn Position
                inst(&mut w, 71, &[o, 11, 0]);
            }
            Some(n) => {
                inst(&mut w, 71, &[i, 30, n]); // Location
                inst(&mut w, 71, &[o, 30, n]);
            }
        }
    }
    inst(&mut w, 19, &[void]);
    inst(&mut w, 33, &[fn_ty, void]);
    inst(&mut w, 22, &[f32_ty, 32]);
    inst(&mut w, 23, &[vec4, f32_ty, 4]);
    inst(&mut w, 21, &[u32_ty, 32, 0]);
    for (c, v) in [(c0, 0), (c1, 1), (c2, 2), (c3, 3)] {
        inst(&mut w, 43, &[u32_ty, c, v]);
    }
    inst(&mut w, 28, &[arr3, vec4, c3]); // OpTypeArray vec4[3]
    inst(&mut w, 32, &[ptr_in_arr, 1, arr3]);
    inst(&mut w, 32, &[ptr_in_v4, 1, vec4]);
    inst(&mut w, 32, &[ptr_out_v4, 3, vec4]);
    for &(_, i, o) in &attrs {
        inst(&mut w, 59, &[ptr_in_arr, i, 1]);
        inst(&mut w, 59, &[ptr_out_v4, o, 3]);
    }
    inst(&mut w, 54, &[void, main, 0, fn_ty]);
    inst(&mut w, 248, &[label]);
    w.extend_from_slice(&body);
    inst(&mut w, 253, &[]); // OpReturn
    inst(&mut w, 56, &[]); // OpFunctionEnd
    w[3] = next;
    w
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

    #[test]
    fn rectlist_gs_validates_and_reads_vs_locations() {
        // A VS output interface: Position (builtin) + params at 0 and 3.
        let mut vs = vec![0x0723_0203, 0x0001_0600, 0, 10, 0];
        for (var, loc) in [(5u32, 0u32), (6, 3)] {
            vs.extend_from_slice(&[(4 << 16) | 71, var, 30, loc]);
        }
        for var in [4u32, 5, 6] {
            vs.extend_from_slice(&[(4 << 16) | 59, 2, var, 3]);
        }
        let locs = vs_param_locations(&vs);
        assert_eq!(locs, [0, 3]);
        let spv = emit_rectlist_gs_spirv(&locs);
        let tmp = std::env::temp_dir().join("strata_test_rect_gs.spv");
        let bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&tmp, &bytes).unwrap();
        if let Ok(out) = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&tmp)
            .output()
        {
            assert!(
                out.status.success(),
                "spirv-val failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let _ = std::fs::remove_file(&tmp);
    }
}
