//! Virtual File System (VFS) mapping Xbox Series X|S virtual drives
//! (`G:\` / `D:\` title storage, `T:\` temp, `S:\` persistent local storage)
//! to host Windows directories.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
        self.mounts.insert(drive.to_ascii_uppercase(), host_root.into());
    }

    /// Rewrite a console path like `D:\data\shaders.rpf` or `S:/saves/slot0.sav`
    /// to its mounted host path. Paths on unmounted drives pass through unchanged.
    pub fn resolve(&self, console_path: &str) -> PathBuf {
        let bytes = console_path.as_bytes();
        if bytes.len() >= 2 && bytes[1] == b':' {
            let drive = (bytes[0] as char).to_ascii_uppercase();
            if let Some(root) = self.mounts.get(&drive) {
                let rel = console_path[2..].trim_start_matches(['\\', '/']);
                return root.join(rel.replace('/', "\\"));
            }
        }
        Path::new(console_path).to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_mounted_console_drives_and_passes_others_through() {
        let mut vfs = Vfs::new();
        vfs.mount('d', r"C:\Games\GTA VI\VICE\data");
        vfs.mount('S', r"C:\Users\hyper\AppData\Local\strata\saves");

        assert_eq!(
            vfs.resolve(r"D:\shaders.rpf"),
            PathBuf::from(r"C:\Games\GTA VI\VICE\data\shaders.rpf")
        );
        assert_eq!(
            vfs.resolve("s:/slot0.sav"),
            PathBuf::from(r"C:\Users\hyper\AppData\Local\strata\saves\slot0.sav")
        );
        assert_eq!(
            vfs.resolve(r"C:\Windows\System32\kernel32.dll"),
            PathBuf::from(r"C:\Windows\System32\kernel32.dll")
        );
    }
}
