//! Repository chores.
//!
//! - `cargo xtask asset-check`: fails if any file git tracks or is about to
//!   track looks like game or console content (executables, archives, shader
//!   containers, media) or is suspiciously large. Strata ships source only;
//!   this keeps it that way.
//! - `cargo xtask install-hook`: makes `asset-check` a git pre-commit hook.

use std::path::Path;
use std::process::{Command, ExitCode};

/// Extensions of content that never belongs in this repository.
const DENIED_EXTENSIONS: &[&str] = &[
    // executables and their debug data
    "exe", "dll", "sys", "pdb", "xex", // console packages and archives
    "xvc", "msixvc", "xvd", "xsp", "rpf", "pak", "arc", "big", "wad", "xbox", "appx", "msix",
    // shader binaries
    "sgd5", "xbch", "fssm", "cso", "dxbc", "dxil", // media
    "bk2", "bik", "xma", "xwm", "wem", "bnk", "fsb", "dds", "ktx", "ktx2", "tga", "mp4", "wav",
    "ogg", // saves and keys
    "sav", "key", "pfx",
];

/// Leading bytes of formats that are content whatever the file is called.
const DENIED_MAGIC: &[(&[u8], &str)] = &[
    (b"MZ", "Windows executable (PE)"),
    (b"DXBC", "shader container (DXBC)"),
    (b"XBCH", "console shader container (XBCH)"),
    (b"FSSM", "console shader microcode (FSSM)"),
    (b"msix", "console package"),
    (b"RIFF", "RIFF media"),
    (b"BIKi", "Bink video"),
    (b"KB2", "Bink 2 video"),
];

/// Larger files need a reason to be here.
const MAX_BYTES: u64 = 2 * 1024 * 1024;

/// Why `path` (with leading `head` bytes and `size`) must not be committed.
fn violation(path: &str, head: &[u8], size: u64) -> Option<String> {
    let lower = path.to_ascii_lowercase();
    let ext = Path::new(&lower)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if DENIED_EXTENSIONS.contains(&ext) {
        return Some(format!("*.{ext} is game or console content"));
    }
    if let Some((_, what)) = DENIED_MAGIC
        .iter()
        .find(|(magic, _)| head.starts_with(magic))
    {
        return Some(format!("contents are a {what}"));
    }
    if size > MAX_BYTES {
        return Some(format!(
            "{} KiB is over the {} KiB limit",
            size / 1024,
            MAX_BYTES / 1024
        ));
    }
    None
}

fn git_files(args: &[&str]) -> Result<Vec<String>, String> {
    let out = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .split('\0')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect())
}

fn asset_check() -> Result<usize, String> {
    let mut files = git_files(&["ls-files", "-z", "--cached"])?;
    files.extend(git_files(&[
        "ls-files",
        "-z",
        "--others",
        "--exclude-standard",
    ])?);
    files.sort();
    files.dedup();
    let mut bad = 0;
    for f in &files {
        let Ok(meta) = std::fs::metadata(f) else {
            continue;
        }; // deleted in the worktree
        let mut head = [0u8; 8];
        let n = std::fs::File::open(f)
            .and_then(|mut h| std::io::Read::read(&mut h, &mut head))
            .unwrap_or(0);
        if let Some(why) = violation(f, &head[..n], meta.len()) {
            eprintln!("asset-check: {f}: {why}");
            bad += 1;
        }
    }
    println!("asset-check: {} files checked, {bad} refused", files.len());
    Ok(bad)
}

fn install_hook() -> Result<(), String> {
    let dir = git_files(&["rev-parse", "--git-path", "hooks"])
        .ok()
        .and_then(|v| v.first().map(|s| s.trim().to_owned()))
        .unwrap_or_else(|| ".git/hooks".into());
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let hook = Path::new(&dir).join("pre-commit");
    std::fs::write(&hook, "#!/bin/sh\nexec cargo xtask asset-check\n")
        .map_err(|e| e.to_string())?;
    println!("installed {}", hook.display());
    Ok(())
}

fn extract_strata_switches(text: &str) -> std::collections::BTreeSet<String> {
    let mut found = std::collections::BTreeSet::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i + 7 <= bytes.len() {
        if &bytes[i..i + 7] == b"STRATA_" {
            let mut end = i + 7;
            while end < bytes.len()
                && (bytes[end].is_ascii_uppercase()
                    || bytes[end].is_ascii_digit()
                    || bytes[end] == b'_')
            {
                end += 1;
            }
            if end > i + 7 {
                found.insert(text[i..end].to_string());
            }
            i = end;
        } else {
            i += 1;
        }
    }
    found
}

fn switches_check() -> Result<usize, String> {
    let files = git_files(&["ls-files", "-z", "--cached"])?;
    let mut documented = std::collections::BTreeSet::new();
    let mut code_refs = std::collections::BTreeSet::new();
    for f in &files {
        let Ok(content) = std::fs::read_to_string(f) else {
            continue;
        };
        if f.ends_with(".md") {
            documented.extend(extract_strata_switches(&content));
        } else if f.ends_with(".rs") {
            code_refs.extend(extract_strata_switches(&content));
        }
    }
    let mut dead = 0usize;
    for sw in &documented {
        if !code_refs.contains(sw) {
            eprintln!("switches: documented switch `{sw}` is not referenced in Rust code");
            dead += 1;
        }
    }
    println!(
        "switches: {} documented, {} referenced in Rust, {dead} dead",
        documented.len(),
        code_refs.len()
    );
    Ok(dead)
}

fn main() -> ExitCode {
    let result = match std::env::args().nth(1).as_deref() {
        Some("asset-check") => asset_check().map(|bad| bad == 0),
        Some("install-hook") => install_hook().map(|()| true),
        Some("switches") => switches_check().map(|dead| dead == 0),
        _ => {
            eprintln!("usage: cargo xtask <asset-check|install-hook|switches>");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("xtask: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_and_small_assets_pass() {
        assert_eq!(
            violation("crates/strata-hle/src/loader.rs", b"//! PE32", 40_000),
            None
        );
        assert_eq!(
            violation("assets/readme/banner.svg", b"<svg xml", 30_000),
            None
        );
    }

    #[test]
    fn content_is_refused_by_name_magic_or_size() {
        assert!(violation("game/GTA.exe", b"MZ\x90\0", 1_000).is_some());
        assert!(violation("data/shaders.rpf", b"RPF8", 10).is_some());
        assert!(
            violation("notes/renamed.txt", b"MZ\x90\0", 10)
                .unwrap()
                .contains("PE")
        );
        assert!(
            violation("x/blob.dat", b"XBCH\x03\0", 10)
                .unwrap()
                .contains("XBCH")
        );
        assert!(
            violation("x/big.json", b"{", 3 * 1024 * 1024)
                .unwrap()
                .contains("limit")
        );
    }

    #[test]
    fn repository_passes_policy() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root");
        let prev = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(root).expect("chdir workspace root");
        let res = asset_check();
        let _ = std::env::set_current_dir(prev);
        assert_eq!(
            res.expect("asset_check runs"),
            0,
            "repository has asset policy violations"
        );
    }
}
