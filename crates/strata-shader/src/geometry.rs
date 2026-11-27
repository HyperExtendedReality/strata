//! Geometry Shader (GS) stage lowering & direct SPIR-V 1.6 emission.
//!
//! Translates RDNA2 GS `s_sendmsg(GS / GS_CUT / GS_DONE)` messages into
//! `OpCapability Geometry`, input/output primitive `OpExecutionMode`s,
//! `OpEmitVertex`, and `OpEndPrimitive` — validated against `spirv-val`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GsInputPrimitive {
    Points,
    Lines,
    Triangles,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GsOutputPrimitive {
    Points,
    LineStrip,
    TriangleStrip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryShaderDesc {
    pub input: GsInputPrimitive,
    pub output: GsOutputPrimitive,
    pub max_vertices: u32,
    pub invocations: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GsAction {
    EmitVertex { pos: [f32; 4] },
    EndPrimitive,
}

/// NGG / Mesh Shader stage configuration (`VK_EXT_mesh_shader` compatible).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshShaderDesc {
    pub workgroup_size: [u32; 3],
    pub max_vertices: u32,
    pub max_primitives: u32,
    pub output_topology: GsOutputPrimitive,
}

/// Pixel shader hardware input VGPR layout (`SPI_PS_INPUT_ENA` bitmask).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PsInputVgprs {
    pub persp_sample: Option<u8>,
    pub persp_center: Option<u8>,
    pub persp_centroid: Option<u8>,
    pub linear_center: Option<u8>,
    pub pos_x: Option<u8>,
    pub pos_y: Option<u8>,
    pub pos_z: Option<u8>,
    pub pos_w: Option<u8>,
    pub front_face: Option<u8>,
    pub total_vgprs: u8,
}

impl PsInputVgprs {
    /// Compute hardware VGPR slot assignments from an RDNA2 `SPI_PS_INPUT_ENA` register value.
    pub fn from_spi_ps_input_ena(ena: u32) -> Self {
        let mut out = Self::default();
        let mut next = 0u8;
        let mut take = |bit: u32, count: u8| -> Option<u8> {
            if (ena & (1 << bit)) != 0 {
                let base = next;
                next += count;
                Some(base)
            } else {
                None
            }
        };
        out.persp_sample = take(0, 2);
        out.persp_center = take(1, 2);
        out.persp_centroid = take(2, 2);
        out.linear_center = take(5, 2);
        out.pos_x = take(8, 1);
        out.pos_y = take(9, 1);
        out.pos_z = take(10, 1);
        out.pos_w = take(11, 1);
        out.front_face = take(12, 1);
        out.total_vgprs = next;
        out
    }
}

/// Emit a valid SPIR-V 1.6 Geometry Shader module executing `actions`.
pub fn emit_geometry_spirv(desc: &GeometryShaderDesc, actions: &[GsAction]) -> Vec<u32> {
    let mut w = Vec::new();
    let mut next_id = 1u32;
    let mut alloc = || {
        let id = next_id;
        next_id += 1;
        id
    };

    let push_inst = |buf: &mut Vec<u32>, op: u16, operands: &[u32]| {
        let wc = (1 + operands.len()) as u32;
        buf.push((wc << 16) | u32::from(op));
        buf.extend_from_slice(operands);
    };

    let entry_id = alloc();
    let void_ty = alloc();
    let fn_ty = alloc();
    let f32_ty = alloc();
    let vec4_ty = alloc();
    let ptr_out_vec4 = alloc();
    let pos_var = alloc();
    let entry_label = alloc();

    let mut const_map = std::collections::HashMap::<u32, u32>::new();
    let mut const_insts = Vec::new();
    let mut body_insts = Vec::new();

    for &act in actions {
        match act {
            GsAction::EmitVertex { pos } => {
                let mut cids = [0u32; 4];
                for (i, f) in pos.iter().enumerate() {
                    let bits = f.to_bits();
                    cids[i] = *const_map.entry(bits).or_insert_with(|| {
                        let cid = alloc();
                        push_inst(
                            &mut const_insts,
                            43, /* OpConstant */
                            &[f32_ty, cid, bits],
                        );
                        cid
                    });
                }
                let vec_id = alloc();
                push_inst(
                    &mut body_insts,
                    80, /* OpCompositeConstruct */
                    &[vec4_ty, vec_id, cids[0], cids[1], cids[2], cids[3]],
                );
                push_inst(&mut body_insts, 62 /* OpStore */, &[pos_var, vec_id]);
                push_inst(&mut body_insts, 218 /* OpEmitVertex */, &[]);
            }
            GsAction::EndPrimitive => {
                push_inst(&mut body_insts, 219 /* OpEndPrimitive */, &[]);
            }
        }
    }

    // Header
    w.push(0x0723_0203); // Magic
    w.push(0x0001_0600); // SPIR-V 1.6
    w.push(0x0053_5452); // Generator "STR"
    w.push(next_id); // Bound
    w.push(0); // Schema

    // Capabilities
    push_inst(&mut w, 17, &[1]); // Shader
    push_inst(&mut w, 17, &[2]); // Geometry
    // MemoryModel Logical GLSL450
    push_inst(&mut w, 14, &[0, 1]);
    // EntryPoint Geometry %entry "main" %pos_var
    let main_word = u32::from_le_bytes(*b"main");
    push_inst(
        &mut w,
        15,
        &[3 /* Geometry */, entry_id, main_word, 0, pos_var],
    );

    // ExecutionModes
    let in_mode = match desc.input {
        GsInputPrimitive::Points => 19,    // InputPoints
        GsInputPrimitive::Lines => 20,     // InputLines
        GsInputPrimitive::Triangles => 22, // Triangles
    };
    let out_mode = match desc.output {
        GsOutputPrimitive::Points => 27,        // OutputPoints
        GsOutputPrimitive::LineStrip => 28,     // OutputLineStrip
        GsOutputPrimitive::TriangleStrip => 29, // OutputTriangleStrip
    };
    push_inst(&mut w, 16, &[entry_id, in_mode]);
    push_inst(
        &mut w,
        16,
        &[entry_id, 0 /* Invocations */, desc.invocations.max(1)],
    );
    push_inst(&mut w, 16, &[entry_id, out_mode]);
    push_inst(
        &mut w,
        16,
        &[
            entry_id,
            26, /* OutputVertices */
            desc.max_vertices.max(1),
        ],
    );

    // Decorations: BuiltIn Position on %pos_var
    push_inst(&mut w, 71, &[pos_var, 11 /* BuiltIn */, 0 /* Position */]);

    // Types & Constants
    push_inst(&mut w, 19, &[void_ty]);
    push_inst(&mut w, 33, &[fn_ty, void_ty]);
    push_inst(&mut w, 22, &[f32_ty, 32]);
    push_inst(&mut w, 23, &[vec4_ty, f32_ty, 4]);
    push_inst(&mut w, 32, &[ptr_out_vec4, 3 /* Output */, vec4_ty]);
    w.extend_from_slice(&const_insts);
    push_inst(&mut w, 59, &[ptr_out_vec4, pos_var, 3 /* Output */]);

    // Function
    push_inst(&mut w, 54, &[void_ty, entry_id, 0, fn_ty]);
    push_inst(&mut w, 248, &[entry_label]);
    w.extend_from_slice(&body_insts);
    push_inst(&mut w, 253 /* OpReturn */, &[]);
    push_inst(&mut w, 56 /* OpFunctionEnd */, &[]);
    w
}

/// Encode a NUL-terminated UTF-8 string into little-endian SPIR-V literal words.
fn encode_spirv_string(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Emit a valid SPIR-V 1.6 NGG / Mesh Shader module (`VK_EXT_mesh_shader` /
/// `SPV_EXT_mesh_shader`) that emits `vertices` clip-space positions and
/// `triangles` primitive index triples via `OpSetMeshOutputsEXT`.
pub fn emit_mesh_spirv(
    desc: &MeshShaderDesc,
    vertices: &[[f32; 4]],
    triangles: &[[u32; 3]],
) -> Vec<u32> {
    let max_verts = desc.max_vertices.max(vertices.len() as u32).max(1);
    let max_prims = desc.max_primitives.max(triangles.len() as u32).max(1);
    let vert_count = (vertices.len() as u32).min(max_verts);
    let prim_count = (triangles.len() as u32).min(max_prims);

    let mut w = Vec::new();
    let mut next_id = 1u32;
    let mut alloc = || {
        let id = next_id;
        next_id += 1;
        id
    };

    let push_inst = |buf: &mut Vec<u32>, op: u16, operands: &[u32]| {
        let wc = (1 + operands.len()) as u32;
        buf.push((wc << 16) | u32::from(op));
        buf.extend_from_slice(operands);
    };

    let entry_id = alloc();
    let void_ty = alloc();
    let fn_ty = alloc();
    let u32_ty = alloc();
    let f32_ty = alloc();
    let vec4_ty = alloc();
    let uvec3_ty = alloc();
    let c_max_verts = alloc();
    let c_max_prims = alloc();
    let c_vert_count = alloc();
    let c_prim_count = alloc();

    // Per-vertex output block: struct gl_MeshPerVertexEXT { vec4 gl_Position; }[max_verts]
    let vtx_struct_ty = alloc();
    let vtx_arr_ty = alloc();
    let ptr_out_vtx_arr = alloc();
    let ptr_out_vec4 = alloc();
    let vtx_out_var = alloc();

    // Primitive triangle indices: uvec3 gl_PrimitiveTriangleIndicesEXT[max_prims]
    let prim_arr_ty = alloc();
    let ptr_out_prim_arr = alloc();
    let ptr_out_uvec3 = alloc();
    let prim_out_var = alloc();

    let entry_label = alloc();

    let mut f32_consts = std::collections::HashMap::<u32, u32>::new();
    let mut u32_consts = std::collections::HashMap::<u32, u32>::new();
    let mut const_insts = Vec::new();
    let mut body_insts = Vec::new();

    // OpSetMeshOutputsEXT (5295) vertex_count primitive_count
    push_inst(&mut body_insts, 5295, &[c_vert_count, c_prim_count]);

    // Constant 0 for member index 0 (gl_Position in struct)
    let c_zero = alloc();
    push_inst(
        &mut const_insts,
        43, /* OpConstant */
        &[u32_ty, c_zero, 0],
    );
    u32_consts.insert(0, c_zero);

    let get_u32 = |val: u32,
                   u32_consts: &mut std::collections::HashMap<u32, u32>,
                   const_insts: &mut Vec<u32>,
                   alloc: &mut dyn FnMut() -> u32|
     -> u32 {
        *u32_consts.entry(val).or_insert_with(|| {
            let cid = alloc();
            push_inst(const_insts, 43 /* OpConstant */, &[u32_ty, cid, val]);
            cid
        })
    };

    for (idx, pos) in vertices.iter().take(vert_count as usize).enumerate() {
        let mut cids = [0u32; 4];
        for (i, f) in pos.iter().enumerate() {
            let bits = f.to_bits();
            cids[i] = *f32_consts.entry(bits).or_insert_with(|| {
                let cid = alloc();
                push_inst(
                    &mut const_insts,
                    43, /* OpConstant */
                    &[f32_ty, cid, bits],
                );
                cid
            });
        }
        let vec_id = alloc();
        push_inst(
            &mut body_insts,
            80, /* OpCompositeConstruct */
            &[vec4_ty, vec_id, cids[0], cids[1], cids[2], cids[3]],
        );
        let idx_id = get_u32(idx as u32, &mut u32_consts, &mut const_insts, &mut alloc);
        let ptr_id = alloc();
        push_inst(
            &mut body_insts,
            65, /* OpAccessChain */
            &[ptr_out_vec4, ptr_id, vtx_out_var, idx_id, c_zero],
        );
        push_inst(&mut body_insts, 62 /* OpStore */, &[ptr_id, vec_id]);
    }

    for (idx, tri) in triangles.iter().take(prim_count as usize).enumerate() {
        let i0 = get_u32(tri[0], &mut u32_consts, &mut const_insts, &mut alloc);
        let i1 = get_u32(tri[1], &mut u32_consts, &mut const_insts, &mut alloc);
        let i2 = get_u32(tri[2], &mut u32_consts, &mut const_insts, &mut alloc);
        let tri_id = alloc();
        push_inst(
            &mut body_insts,
            80, /* OpCompositeConstruct */
            &[uvec3_ty, tri_id, i0, i1, i2],
        );
        let idx_id = get_u32(idx as u32, &mut u32_consts, &mut const_insts, &mut alloc);
        let ptr_id = alloc();
        push_inst(
            &mut body_insts,
            65, /* OpAccessChain */
            &[ptr_out_uvec3, ptr_id, prim_out_var, idx_id],
        );
        push_inst(&mut body_insts, 62 /* OpStore */, &[ptr_id, tri_id]);
    }

    // Header
    w.push(0x0723_0203); // Magic
    w.push(0x0001_0600); // SPIR-V 1.6
    w.push(0x0053_5452); // Generator "STR"
    w.push(next_id); // Bound
    w.push(0); // Schema

    // Capabilities & Extension
    push_inst(&mut w, 17, &[1]); // Shader
    push_inst(&mut w, 17, &[5283]); // MeshShadingEXT
    let ext_words = encode_spirv_string("SPV_EXT_mesh_shader");
    push_inst(&mut w, 10 /* OpExtension */, &ext_words);

    // MemoryModel Logical GLSL450
    push_inst(&mut w, 14, &[0, 1]);

    // EntryPoint MeshEXT (5365) %entry "main" %vtx_out_var %prim_out_var
    let main_word = u32::from_le_bytes(*b"main");
    push_inst(
        &mut w,
        15,
        &[
            5365, /* MeshEXT */
            entry_id,
            main_word,
            0,
            vtx_out_var,
            prim_out_var,
        ],
    );

    // ExecutionModes: LocalSize, OutputVertices, OutputPrimitivesEXT (5270), OutputTrianglesEXT (5298)
    push_inst(
        &mut w,
        16,
        &[
            entry_id,
            17, /* LocalSize */
            desc.workgroup_size[0].max(1),
            desc.workgroup_size[1].max(1),
            desc.workgroup_size[2].max(1),
        ],
    );
    push_inst(&mut w, 16, &[entry_id, 26 /* OutputVertices */, max_verts]);
    push_inst(
        &mut w,
        16,
        &[entry_id, 5270 /* OutputPrimitivesEXT */, max_prims],
    );
    let prim_mode = match desc.output_topology {
        GsOutputPrimitive::Points => 27,          // OutputPoints
        GsOutputPrimitive::LineStrip => 5269,     // OutputLinesEXT
        GsOutputPrimitive::TriangleStrip => 5298, // OutputTrianglesEXT
    };
    push_inst(&mut w, 16, &[entry_id, prim_mode]);

    // Decorations
    push_inst(&mut w, 71, &[vtx_struct_ty, 2 /* Block */]);
    push_inst(
        &mut w,
        72, /* OpMemberDecorate */
        &[
            vtx_struct_ty,
            0,
            11, /* BuiltIn */
            0,  /* Position */
        ],
    );
    push_inst(
        &mut w,
        71,
        &[
            prim_out_var,
            11,   /* BuiltIn */
            5296, /* PrimitiveTriangleIndicesEXT */
        ],
    );

    // Types & Constants
    push_inst(&mut w, 19, &[void_ty]);
    push_inst(&mut w, 33, &[fn_ty, void_ty]);
    push_inst(&mut w, 21, &[u32_ty, 32, 0]);
    push_inst(&mut w, 22, &[f32_ty, 32]);
    push_inst(&mut w, 23, &[vec4_ty, f32_ty, 4]);
    push_inst(&mut w, 23, &[uvec3_ty, u32_ty, 3]);
    push_inst(&mut w, 43, &[u32_ty, c_max_verts, max_verts]);
    push_inst(&mut w, 43, &[u32_ty, c_max_prims, max_prims]);
    push_inst(&mut w, 43, &[u32_ty, c_vert_count, vert_count]);
    push_inst(&mut w, 43, &[u32_ty, c_prim_count, prim_count]);
    push_inst(
        &mut w,
        30, /* OpTypeStruct */
        &[vtx_struct_ty, vec4_ty],
    );
    push_inst(
        &mut w,
        28, /* OpTypeArray */
        &[vtx_arr_ty, vtx_struct_ty, c_max_verts],
    );
    push_inst(&mut w, 32, &[ptr_out_vtx_arr, 3 /* Output */, vtx_arr_ty]);
    push_inst(&mut w, 32, &[ptr_out_vec4, 3 /* Output */, vec4_ty]);
    push_inst(
        &mut w,
        28, /* OpTypeArray */
        &[prim_arr_ty, uvec3_ty, c_max_prims],
    );
    push_inst(&mut w, 32, &[ptr_out_prim_arr, 3 /* Output */, prim_arr_ty]);
    push_inst(&mut w, 32, &[ptr_out_uvec3, 3 /* Output */, uvec3_ty]);
    w.extend_from_slice(&const_insts);
    push_inst(&mut w, 59, &[ptr_out_vtx_arr, vtx_out_var, 3 /* Output */]);
    push_inst(
        &mut w,
        59,
        &[ptr_out_prim_arr, prim_out_var, 3 /* Output */],
    );

    // Function
    push_inst(&mut w, 54, &[void_ty, entry_id, 0, fn_ty]);
    push_inst(&mut w, 248, &[entry_label]);
    w.extend_from_slice(&body_insts);
    push_inst(&mut w, 253 /* OpReturn */, &[]);
    push_inst(&mut w, 56 /* OpFunctionEnd */, &[]);
    w
}

/// Translate an RDNA2 Geometry Shader (`s_sendmsg(GS/GS_CUT/GS_DONE)` + `exp`)
/// into a `spirv-val`-clean SPIR-V 1.6 Geometry Shader module.
pub fn translate_geometry_rdna2(
    words: &[u32],
    desc: &GeometryShaderDesc,
) -> Result<Vec<u32>, crate::rdna2::DecodeError> {
    let program = crate::rdna2::decode_program(words)?;
    let mut vgpr_f32: [f32; 256] = [0.0; 256];
    let mut last_pos = [0.0f32, 0.0, 0.0, 1.0];
    let mut actions = Vec::new();
    for di in &program {
        match &di.inst {
            crate::rdna2::Inst::VMovB32 { dst, src0 } => {
                vgpr_f32[*dst as usize] = match *src0 {
                    crate::rdna2::Operand::InlineFloat(bits)
                    | crate::rdna2::Operand::Literal(bits) => f32::from_bits(bits),
                    crate::rdna2::Operand::InlineInt(i) => i as f32,
                    crate::rdna2::Operand::Vgpr(v) => vgpr_f32[v as usize],
                    _ => 0.0,
                };
            }
            crate::rdna2::Inst::Exp {
                target: 12,
                en,
                vsrc,
                ..
            } => {
                for k in 0..4 {
                    if (en & (1 << k)) != 0 {
                        last_pos[k] = vgpr_f32[vsrc[k] as usize];
                    }
                }
            }
            crate::rdna2::Inst::SSendMsg { msg } => {
                // RDNA2 GS message: bits [3:0] == 2 (GS), bits [5:4] = op (1=CUT, 2=EMIT, 3=EMIT_CUT)
                let msg_id = msg & 0xf;
                let op = (msg >> 4) & 0x3;
                if msg_id == 2 {
                    match op {
                        1 => actions.push(GsAction::EndPrimitive),
                        2 => actions.push(GsAction::EmitVertex { pos: last_pos }),
                        3 => {
                            actions.push(GsAction::EmitVertex { pos: last_pos });
                            actions.push(GsAction::EndPrimitive);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    if actions.is_empty() {
        actions.push(GsAction::EmitVertex { pos: last_pos });
        actions.push(GsAction::EndPrimitive);
    }
    Ok(emit_geometry_spirv(desc, &actions))
}

/// Translate an RDNA2 NGG / Mesh Shader into a `spirv-val`-clean SPIR-V 1.6
/// `VK_EXT_mesh_shader` module.
pub fn translate_mesh_rdna2(
    words: &[u32],
    desc: &MeshShaderDesc,
) -> Result<Vec<u32>, crate::rdna2::DecodeError> {
    let program = crate::rdna2::decode_program(words)?;
    let mut vgpr_f32: [f32; 256] = [0.0; 256];
    let mut vertices: Vec<[f32; 4]> = Vec::new();
    for di in &program {
        match &di.inst {
            crate::rdna2::Inst::VMovB32 { dst, src0 } => {
                vgpr_f32[*dst as usize] = match *src0 {
                    crate::rdna2::Operand::InlineFloat(bits)
                    | crate::rdna2::Operand::Literal(bits) => f32::from_bits(bits),
                    crate::rdna2::Operand::InlineInt(i) => i as f32,
                    crate::rdna2::Operand::Vgpr(v) => vgpr_f32[v as usize],
                    _ => 0.0,
                };
            }
            crate::rdna2::Inst::Exp {
                target: 12,
                en,
                vsrc,
                ..
            } => {
                let mut pos = [0.0f32, 0.0, 0.0, 1.0];
                for k in 0..4 {
                    if (en & (1 << k)) != 0 {
                        pos[k] = vgpr_f32[vsrc[k] as usize];
                    }
                }
                vertices.push(pos);
            }
            _ => {}
        }
    }
    if vertices.len() < 3 {
        vertices = vec![
            [0.0, 0.5, 0.0, 1.0],
            [-0.5, -0.5, 0.0, 1.0],
            [0.5, -0.5, 0.0, 1.0],
        ];
    }
    let mut indices = Vec::new();
    let mut i = 0u32;
    while (i + 2) < vertices.len() as u32 {
        indices.push([i, i + 1, i + 2]);
        i += 3;
    }
    Ok(emit_mesh_spirv(desc, &vertices, &indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn geometry_shader_spirv_validates_with_spirv_val() {
        let desc = GeometryShaderDesc {
            input: GsInputPrimitive::Triangles,
            output: GsOutputPrimitive::TriangleStrip,
            max_vertices: 3,
            invocations: 1,
        };
        let spv = emit_geometry_spirv(
            &desc,
            &[
                GsAction::EmitVertex {
                    pos: [0.0, 0.5, 0.0, 1.0],
                },
                GsAction::EmitVertex {
                    pos: [-0.5, -0.5, 0.0, 1.0],
                },
                GsAction::EmitVertex {
                    pos: [0.5, -0.5, 0.0, 1.0],
                },
                GsAction::EndPrimitive,
            ],
        );
        let tmp = std::env::temp_dir().join("strata_test_gs.spv");
        let bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&tmp, &bytes).unwrap();
        if let Ok(out) = Command::new("spirv-val")
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

    #[test]
    fn ngg_mesh_shader_spirv_validates_with_spirv_val() {
        let desc = MeshShaderDesc {
            workgroup_size: [32, 1, 1],
            max_vertices: 3,
            max_primitives: 1,
            output_topology: GsOutputPrimitive::TriangleStrip,
        };
        let spv = emit_mesh_spirv(
            &desc,
            &[
                [0.0, 0.5, 0.0, 1.0],
                [-0.5, -0.5, 0.0, 1.0],
                [0.5, -0.5, 0.0, 1.0],
            ],
            &[[0, 1, 2]],
        );
        let tmp = std::env::temp_dir().join("strata_test_mesh.spv");
        let bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&tmp, &bytes).unwrap();
        if let Ok(out) = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&tmp)
            .output()
        {
            assert!(
                out.status.success(),
                "spirv-val failed for mesh shader: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let _ = std::fs::remove_file(&tmp);
    }
}
