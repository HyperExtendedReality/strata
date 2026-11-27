//! Virtual File System (VFS) mapping Xbox Series X|S virtual drives
//! (`G:\` / `D:\` title storage, `T:\` temp, `S:\` persistent local storage)
//! to host Windows directories.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

/// The VFS the title's file imports go through (see [`crate::files`]).
pub fn global() -> &'static RwLock<Vfs> {
    static VFS: OnceLock<RwLock<Vfs>> = OnceLock::new();
    VFS.get_or_init(|| RwLock::new(Vfs::new()))
}

#[derive(Debug, Default, Clone)]
pub struct Vfs {
    /// Uppercase drive letter (e.g. `'D'`, `'S'`) -> host root folder.
    mounts: HashMap<char, PathBuf>,
}

impl Vfs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mount(&mut self, drive: char, host_root: impl Into<PathBuf>) {
        self.mounts
            .insert(drive.to_ascii_uppercase(), host_root.into());
    }

    pub fn is_empty(&self) -> bool {
        self.mounts.is_empty()
    }

    /// Rewrite a console path like `D:\data\shaders.rpf` or `S:/saves/slot0.sav`
    /// to its mounted host path. Paths on unmounted drives pass through unchanged.
    pub fn resolve(&self, console_path: &str) -> PathBuf {
        self.rewrite(console_path)
            .unwrap_or_else(|| Path::new(console_path).to_path_buf())
    }

    /// The host path for a path on a mounted drive; `None` for anything else.
    /// Accepts the `\\?\` prefix.
    pub fn rewrite(&self, console_path: &str) -> Option<PathBuf> {
        let path = console_path.strip_prefix(r"\\?\").unwrap_or(console_path);
        let bytes = path.as_bytes();
        if bytes.len() < 2 || bytes[1] != b':' || !bytes[0].is_ascii_alphabetic() {
            return None;
        }
        let root = self.mounts.get(&(bytes[0] as char).to_ascii_uppercase())?;
        let rel = path[2..].trim_start_matches(['\\', '/']);
        if rel.is_empty() {
            return Some(root.clone());
        }
        Some(root.join(rel.replace('/', "\\")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_mounted_console_drives_and_passes_others_through() {
        let data = std::env::temp_dir().join("strata-vfs-test").join("data");
        let saves = std::env::temp_dir().join("strata-vfs-test").join("saves");
        let mut vfs = Vfs::new();
        vfs.mount('d', &data);
        vfs.mount('S', &saves);

        assert_eq!(vfs.resolve(r"D:\shaders.rpf"), data.join("shaders.rpf"));
        assert_eq!(vfs.resolve("s:/slot0.sav"), saves.join("slot0.sav"));
        assert_eq!(vfs.rewrite(r"\\?\D:\a/b.bin"), Some(data.join(r"a\b.bin")));
        assert_eq!(vfs.rewrite("D:"), Some(data.clone()));
        assert_eq!(
            vfs.resolve(r"C:\Windows\System32\kernel32.dll"),
            PathBuf::from(r"C:\Windows\System32\kernel32.dll")
        );
        assert_eq!(vfs.rewrite(r"C:\Windows"), None);
    }
}
