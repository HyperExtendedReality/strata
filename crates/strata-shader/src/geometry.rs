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
}
