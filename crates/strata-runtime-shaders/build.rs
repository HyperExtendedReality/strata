// Build-time Slang -> SPIR-V 1.6 compiler for Strata internal runtime shaders.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const SHADERS: &[(&str, &str, &str, &str)] = &[
    ("dirty_hash.slang", "dirty_hash.spv", "main", "compute"),
    ("mirror_diff.slang", "mirror_diff.spv", "main", "compute"),
    ("detile.slang", "detile.spv", "main", "compute"),
    ("bvh_oracle.slang", "bvh_oracle.spv", "main", "compute"),
    ("ray_probe.slang", "ray_probe.spv", "main", "compute"),
    ("bma_mapping.slang", "bma_mapping.spv", "main", "compute"),
    (
        "clamp_indirect.slang",
        "clamp_indirect.spv",
        "main",
        "compute",
    ),
    ("tex_convert.slang", "tex_convert.spv", "main", "compute"),
    ("mip_gen.slang", "mip_gen.spv", "main", "compute"),
    ("msaa_resolve.slang", "msaa_resolve.spv", "main", "compute"),
    (
        "chunk_dirty_hash.slang",
        "chunk_dirty_hash.spv",
        "main",
        "compute",
    ),
    (
        "scatter_upload.slang",
        "scatter_upload.spv",
        "main",
        "compute",
    ),
    ("rect_expand.slang", "rect_expand.spv", "main", "compute"),
    (
        "ambient_clamp.slang",
        "ambient_clamp.spv",
        "main",
        "compute",
    ),
    ("overlay_hud.slang", "overlay_hud.spv", "main", "compute"),
    (
        "bcpack_decode.slang",
        "bcpack_decode.spv",
        "main",
        "compute",
    ),
    ("compose_vs.slang", "compose_vs.spv", "main", "vertex"),
    ("compose_fs.slang", "compose_fs.spv", "main", "fragment"),
    (
        "present_blit.slang",
        "present_blit_vs.spv",
        "main_vs",
        "vertex",
    ),
    (
        "present_blit.slang",
        "present_blit_fs.spv",
        "main_fs",
        "fragment",
    ),
];

fn find_slangc() -> PathBuf {
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let candidate = Path::new(&sdk).join("Bin").join("slangc.exe");
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from("slangc")
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest_dir.join("shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let slangc = find_slangc();

    println!("cargo:rerun-if-changed=shaders/common.slang");

    for &(src_name, out_name, entry, stage) in SHADERS {
        let src = shader_dir.join(src_name);
        let dst = out_dir.join(out_name);
        println!("cargo:rerun-if-changed=shaders/{src_name}");

        let output = Command::new(&slangc)
            .arg(&src)
            .arg("-I")
            .arg(&shader_dir)
            .arg("-target")
            .arg("spirv")
            .arg("-profile")
            .arg("spirv_1_6")
            .arg("-entry")
            .arg(entry)
            .arg("-stage")
            .arg(stage)
            .arg("-fvk-use-entrypoint-name")
            .arg("-O2")
            .arg("-o")
            .arg(&dst)
            .output()
            .unwrap_or_else(|e| panic!("failed to invoke slangc ({}): {e}", slangc.display()));

        if !output.status.success() {
            panic!(
                "slangc failed for {src_name}:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }
}
