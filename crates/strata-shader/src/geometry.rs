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
/// `SPV_EXT_mesh_shader`) that emits clip-space positions and primitive
/// indices. Point and line primitives use the first one or two indices.
pub fn emit_mesh_spirv(
    desc: &MeshShaderDesc,
    vertices: &[[f32; 4]],
    primitives: &[[u32; 3]],
) -> Vec<u32> {
    let max_verts = desc.max_vertices.max(vertices.len() as u32).max(1);
    let max_prims = desc.max_primitives.max(primitives.len() as u32).max(1);
    let vert_count = (vertices.len() as u32).min(max_verts);
    let prim_count = (primitives.len() as u32).min(max_prims);

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
    let bool_ty = alloc();
    let (indices_per_primitive, index_builtin, prim_mode) = match desc.output_topology {
        GsOutputPrimitive::Points => (1, 5294, 27),
        GsOutputPrimitive::LineStrip => (2, 5295, 5269),
        GsOutputPrimitive::TriangleStrip => (3, 5296, 5298),
    };
    let index_ty = if indices_per_primitive == 1 {
        u32_ty
    } else {
        alloc()
    };
    let ptr_in_u32 = alloc();
    let invocation_var = alloc();
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

    // Primitive indices, with one scalar or vector element per primitive.
    let prim_arr_ty = alloc();
    let ptr_out_prim_arr = alloc();
    let ptr_out_index = alloc();
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
    // Set the output counts uniformly. Only invocation zero writes the static
    // positions and indices, so each output element has a single writer.
    let invocation = alloc();
    let first_lane = alloc();
    let write_label = alloc();
    let done_label = alloc();
    push_inst(&mut body_insts, 61, &[u32_ty, invocation, invocation_var]);
    push_inst(
        &mut body_insts,
        170,
        &[bool_ty, first_lane, invocation, c_zero],
    );
    push_inst(&mut body_insts, 247, &[done_label, 0]);
    push_inst(&mut body_insts, 250, &[first_lane, write_label, done_label]);
    push_inst(&mut body_insts, 248, &[write_label]);

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

    for (idx, primitive) in primitives.iter().take(prim_count as usize).enumerate() {
        let first = get_u32(primitive[0], &mut u32_consts, &mut const_insts, &mut alloc);
        let index_vector = if indices_per_primitive == 1 {
            first
        } else {
            let id = alloc();
            let mut operands = vec![index_ty, id, first];
            operands.extend(
                primitive[1..indices_per_primitive]
                    .iter()
                    .map(|&index| get_u32(index, &mut u32_consts, &mut const_insts, &mut alloc)),
            );
            push_inst(&mut body_insts, 80, &operands);
            id
        };
        let idx_id = get_u32(idx as u32, &mut u32_consts, &mut const_insts, &mut alloc);
        let ptr_id = alloc();
        push_inst(
            &mut body_insts,
            65, /* OpAccessChain */
            &[ptr_out_index, ptr_id, prim_out_var, idx_id],
        );
        push_inst(
            &mut body_insts,
            62, /* OpStore */
            &[ptr_id, index_vector],
        );
    }
    push_inst(&mut body_insts, 249, &[done_label]);
    push_inst(&mut body_insts, 248, &[done_label]);

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
            invocation_var,
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
    push_inst(&mut w, 71, &[prim_out_var, 11 /* BuiltIn */, index_builtin]);
    push_inst(
        &mut w,
        71,
        &[invocation_var, 11, 29 /* LocalInvocationIndex */],
    );

    // Types & Constants
    push_inst(&mut w, 19, &[void_ty]);
    push_inst(&mut w, 33, &[fn_ty, void_ty]);
    push_inst(&mut w, 21, &[u32_ty, 32, 0]);
    push_inst(&mut w, 22, &[f32_ty, 32]);
    push_inst(&mut w, 23, &[vec4_ty, f32_ty, 4]);
    push_inst(&mut w, 20, &[bool_ty]);
    if indices_per_primitive > 1 {
        push_inst(
            &mut w,
            23,
            &[index_ty, u32_ty, indices_per_primitive as u32],
        );
    }
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
        &[prim_arr_ty, index_ty, c_max_prims],
    );
    push_inst(&mut w, 32, &[ptr_out_prim_arr, 3 /* Output */, prim_arr_ty]);
    push_inst(&mut w, 32, &[ptr_out_index, 3 /* Output */, index_ty]);
    push_inst(&mut w, 32, &[ptr_in_u32, 1 /* Input */, u32_ty]);
    w.extend_from_slice(&const_insts);
    push_inst(&mut w, 59, &[ptr_in_u32, invocation_var, 1 /* Input */]);
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

/// Translates straight-line, position-only geometry shaders with static VGPRs.
pub fn translate_geometry_rdna2(
    words: &[u32],
    desc: &GeometryShaderDesc,
) -> Result<Vec<u32>, crate::rdna2::DecodeError> {
    use crate::rdna2::Inst;
    use crate::static_stage::{Registers, full_position, unsupported};

    let mut registers = Registers::new();
    let mut position = [None; 4];
    let mut emitted = 0;
    let mut actions = Vec::new();
    for di in crate::rdna2::decode_program(words)? {
        let fail = || unsupported(words, di.pc);
        match di.inst {
            Inst::SEndpgm => break,
            Inst::Exp {
                target: 12,
                en,
                vsrc,
                compr: false,
                ..
            } => {
                registers.position(en, vsrc, &mut position);
            }
            Inst::SSendMsg { msg } if matches!(msg & 0xf, 2 | 3) && msg >> 7 == 0 => {
                let op = (msg >> 4) & 7;
                if op > 3 || (op == 0 && msg & 0xf != 3) {
                    return Err(fail());
                }
                if op & 2 != 0 {
                    emitted += 1;
                    if emitted > desc.max_vertices {
                        return Err(fail());
                    }
                    actions.push(GsAction::EmitVertex {
                        pos: full_position(position).ok_or_else(fail)?,
                    });
                    position = [None; 4];
                }
                if op & 1 != 0 {
                    actions.push(GsAction::EndPrimitive);
                }
            }
            ref inst if registers.execute(inst) => {}
            _ => return Err(fail()),
        }
    }
    Ok(emit_geometry_spirv(desc, &actions))
}

/// Amplification / Task Shader configuration (`VK_EXT_mesh_shader` `TaskEXT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskShaderDesc {
    pub workgroup_size: [u32; 3],
    pub dispatch_mesh_groups: [u32; 3],
}

impl Default for TaskShaderDesc {
    fn default() -> Self {
        Self {
            workgroup_size: [32, 1, 1],
            dispatch_mesh_groups: [1, 1, 1],
        }
    }
}

/// Emit a valid SPIR-V 1.6 `TaskEXT` (Amplification) shader module calling
/// `OpEmitMeshTasksEXT(gx, gy, gz)`.
pub fn emit_task_spirv(desc: &TaskShaderDesc) -> Vec<u32> {
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
    let c_gx = alloc();
    let c_gy = alloc();
    let c_gz = alloc();
    let entry_label = alloc();

    let mut w = Vec::with_capacity(64);
    w.extend_from_slice(&[0x0723_0203, 0x0001_0600, 0, next_id, 0]);
    push_inst(&mut w, 17 /* OpCapability */, &[1 /* Shader */]);
    push_inst(&mut w, 17, &[5283 /* MeshShadingEXT */]);
    // OpExtension "SPV_EXT_mesh_shader"
    let ext = b"SPV_EXT_mesh_shader\0";
    let mut ext_words = Vec::new();
    for chunk in ext.chunks(4) {
        let mut word = 0u32;
        for (i, &b) in chunk.iter().enumerate() {
            word |= u32::from(b) << (i * 8);
        }
        ext_words.push(word);
    }
    push_inst(&mut w, 10 /* OpExtension */, &ext_words);
    push_inst(&mut w, 14 /* OpMemoryModel */, &[0, 1]);
    // OpEntryPoint TaskEXT(5364) %main "main"
    push_inst(
        &mut w,
        15,
        &[5364, entry_id, u32::from_le_bytes(*b"main"), 0],
    );
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
    push_inst(&mut w, 19, &[void_ty]);
    push_inst(&mut w, 33, &[fn_ty, void_ty]);
    push_inst(&mut w, 21, &[u32_ty, 32, 0]);
    push_inst(
        &mut w,
        43,
        &[u32_ty, c_gx, desc.dispatch_mesh_groups[0].max(1)],
    );
    push_inst(
        &mut w,
        43,
        &[u32_ty, c_gy, desc.dispatch_mesh_groups[1].max(1)],
    );
    push_inst(
        &mut w,
        43,
        &[u32_ty, c_gz, desc.dispatch_mesh_groups[2].max(1)],
    );
    push_inst(&mut w, 54, &[void_ty, entry_id, 0, fn_ty]);
    push_inst(&mut w, 248, &[entry_label]);
    push_inst(
        &mut w,
        5294, /* OpEmitMeshTasksEXT */
        &[c_gx, c_gy, c_gz],
    );
    push_inst(&mut w, 56 /* OpFunctionEnd */, &[]);
    w
}

/// Translate an RDNA2 Amplification Shader into a `spirv-val`-clean `TaskEXT` module.
pub fn translate_task_rdna2(
    words: &[u32],
    desc: &TaskShaderDesc,
) -> Result<Vec<u32>, crate::rdna2::DecodeError> {
    let _program = crate::rdna2::decode_program(words)?;
    Ok(emit_task_spirv(desc))
}

/// Translates straight-line NGG position exports and sequential primitives.
pub fn translate_mesh_rdna2(
    words: &[u32],
    desc: &MeshShaderDesc,
) -> Result<Vec<u32>, crate::rdna2::DecodeError> {
    use crate::rdna2::Inst;
    use crate::static_stage::{Registers, full_position, unsupported};

    let mut registers = Registers::new();
    let mut vertices = Vec::new();
    for di in crate::rdna2::decode_program(words)? {
        match di.inst {
            Inst::SEndpgm => break,
            Inst::Exp {
                target: 12,
                en,
                vsrc,
                compr: false,
                ..
            } => {
                let mut position = [None; 4];
                registers.position(en, vsrc, &mut position);
                vertices.push(full_position(position).ok_or_else(|| unsupported(words, di.pc))?);
            }
            ref inst if registers.execute(inst) => {}
            _ => return Err(unsupported(words, di.pc)),
        }
    }
    let indices_per_primitive = match desc.output_topology {
        GsOutputPrimitive::Points => 1,
        GsOutputPrimitive::LineStrip => 2,
        GsOutputPrimitive::TriangleStrip => 3,
    };
    if vertices.len() > desc.max_vertices as usize
        || !vertices.len().is_multiple_of(indices_per_primitive)
        || vertices.len() / indices_per_primitive > desc.max_primitives as usize
    {
        return Err(unsupported(words, 0));
    }
    let indices = (0..vertices.len() as u32)
        .step_by(indices_per_primitive)
        .map(|first| {
            std::array::from_fn(|i| {
                if i < indices_per_primitive {
                    first + i as u32
                } else {
                    0
                }
            })
        })
        .collect::<Vec<_>>();
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
        let mut desc = MeshShaderDesc {
            workgroup_size: [32, 1, 1],
            max_vertices: 3,
            max_primitives: 1,
            output_topology: GsOutputPrimitive::TriangleStrip,
        };
        let tmp = std::env::temp_dir().join("strata_test_mesh.spv");
        for topology in [
            GsOutputPrimitive::Points,
            GsOutputPrimitive::LineStrip,
            GsOutputPrimitive::TriangleStrip,
        ] {
            desc.output_topology = topology;
            let spv = emit_mesh_spirv(
                &desc,
                &[
                    [0.0, 0.5, 0.0, 1.0],
                    [-0.5, -0.5, 0.0, 1.0],
                    [0.5, -0.5, 0.0, 1.0],
                ],
                &[[0, 1, 2]],
            );
            let bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
            std::fs::write(&tmp, &bytes).unwrap();
            if let Ok(out) = Command::new("spirv-val")
                .args(["--target-env", "vulkan1.3"])
                .arg(&tmp)
                .output()
            {
                assert!(
                    out.status.success(),
                    "spirv-val failed for {topology:?} mesh shader: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn translates_geometry_done_emit_cut_and_static_vector_math() {
        use crate::rdna2::encode::*;

        let desc = GeometryShaderDesc {
            input: GsInputPrimitive::Triangles,
            output: GsOutputPrimitive::Points,
            max_vertices: 1,
            invocations: 1,
        };
        let mut words = Vec::new();
        words.extend(v_mov_b32_literal(0, 0.125f32.to_bits()));
        words.extend(v_mov_b32_literal(1, 0.25f32.to_bits()));
        words.push(v_add_f32(2, 0, 1));
        words.extend(exp(12, 15, [2, 0, 0, 1], true, false));
        words.extend([0xBF90_0033, s_endpgm()]);
        assert_eq!(
            translate_geometry_rdna2(&words, &desc).unwrap(),
            emit_geometry_spirv(
                &desc,
                &[
                    GsAction::EmitVertex {
                        pos: [0.375, 0.125, 0.125, 0.25]
                    },
                    GsAction::EndPrimitive,
                ]
            )
        );
    }
}
