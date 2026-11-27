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

    /// Rewrite a console path like `D:\data\shaders.pak` or `S:/saves/slot0.sav`
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
    /// Mount a GDK package root directory (`G:\` and `D:\`), parsing
    /// `MicrosoftGame.config` if present.
    pub fn mount_package_dir(&mut self, dir: &Path) -> Option<MicrosoftGameConfig> {
        self.mount('G', dir);
        self.mount('D', dir);
        let cfg_path = dir.join("MicrosoftGame.config");
        let xml = std::fs::read_to_string(cfg_path).ok()?;
        Some(parse_microsoft_game_config(&xml))
    }
}

/// Parsed fields from an Xbox GDK `MicrosoftGame.config` manifest.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MicrosoftGameConfig {
    pub identity_name: Option<String>,
    pub title_id: Option<String>,
    pub executable_name: Option<String>,
}

fn extract_xml_attr(xml: &str, tag: &str, attr: &str) -> Option<String> {
    let prefix = format!("<{tag} ");
    let tag_pos = xml.find(&prefix)?;
    let rest = &xml[tag_pos..];
    let end = rest.find('>')?;
    let elem = &rest[..end];
    let needle = format!("{attr}=\"");
    let attr_pos = elem.find(&needle)? + needle.len();
    let val_rest = &elem[attr_pos..];
    let val_end = val_rest.find('"')?;
    Some(val_rest[..val_end].to_string())
}

fn extract_xml_text(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)?;
    Some(xml[start..start + end].trim().to_string())
}

/// Parse core identity, TitleId, and primary executable from `MicrosoftGame.config`.
pub fn parse_microsoft_game_config(xml: &str) -> MicrosoftGameConfig {
    MicrosoftGameConfig {
        identity_name: extract_xml_attr(xml, "Identity", "Name"),
        title_id: extract_xml_text(xml, "TitleId"),
        executable_name: extract_xml_attr(xml, "Executable", "Name"),
    }
}

/// Minimal header metadata parsed from an Xbox `.xvc` package container (`"msft-xvd"` magic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XvcHeader {
    pub version: u32,
    pub content_id: [u8; 16],
    pub encrypted: bool,
}

/// Parse the `"msft-xvd"` / XVC container header (at offset `0x200` in standard XVD/XVC images).
pub fn parse_xvc_header(bytes: &[u8]) -> Option<XvcHeader> {
    let base = if bytes.len() >= 0x240 && &bytes[0x200..0x208] == b"msft-xvd" {
        0x200
    } else if bytes.len() >= 0x40 && &bytes[0..8] == b"msft-xvd" {
        0
    } else {
        return None;
    };
    let version = u32::from_le_bytes(bytes[base + 8..base + 12].try_into().ok()?);
    let flags = u32::from_le_bytes(bytes[base + 12..base + 16].try_into().ok()?);
    let mut content_id = [0u8; 16];
    content_id.copy_from_slice(&bytes[base + 0x10..base + 0x20]);
    Some(XvcHeader {
        version,
        content_id,
        encrypted: (flags & 1) == 0,
    })
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

        assert_eq!(vfs.resolve(r"D:\shaders.pak"), data.join("shaders.pak"));
        assert_eq!(vfs.resolve("s:/slot0.sav"), saves.join("slot0.sav"));
        assert_eq!(vfs.rewrite(r"\\?\D:\a/b.bin"), Some(data.join(r"a\b.bin")));
        assert_eq!(vfs.rewrite("D:"), Some(data.clone()));
        assert_eq!(
            vfs.resolve(r"C:\Windows\System32\kernel32.dll"),
            PathBuf::from(r"C:\Windows\System32\kernel32.dll")
        );
        assert_eq!(vfs.rewrite(r"C:\Windows"), None);
    }

    #[test]
    fn parses_microsoft_game_config_and_xvc_header() {
        let xml = r#"
            <Game configVersion="1">
                <Identity Name="Strata.SampleTitle" Publisher="CN=Strata" Version="1.0.0.0"/>
                <TitleId>1234ABCD</TitleId>
                <ExecutableList>
                    <Executable Name="GameXS.exe" Id="Game"/>
                </ExecutableList>
            </Game>
        "#;
        let cfg = parse_microsoft_game_config(xml);
        assert_eq!(cfg.identity_name.as_deref(), Some("Strata.SampleTitle"));
        assert_eq!(cfg.title_id.as_deref(), Some("1234ABCD"));
        assert_eq!(cfg.executable_name.as_deref(), Some("GameXS.exe"));

        let mut xvc = vec![0u8; 0x240];
        xvc[0x200..0x208].copy_from_slice(b"msft-xvd");
        xvc[0x208..0x20c].copy_from_slice(&3u32.to_le_bytes());
        xvc[0x20c..0x210].copy_from_slice(&1u32.to_le_bytes()); // unencrypted flag
        xvc[0x210..0x220].copy_from_slice(&[0xAB; 16]);
        let hdr = parse_xvc_header(&xvc).expect("valid XVC header");
        assert_eq!(hdr.version, 3);
        assert!(!hdr.encrypted);
        assert_eq!(hdr.content_id, [0xAB; 16]);
    }
}
