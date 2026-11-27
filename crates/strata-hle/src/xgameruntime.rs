//! Full offline GDK runtime services: `XGameRuntime`, `XUser`, `XGameSave`,
//! `XPackage`, and `XStore`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::vfs::Vfs;

// ============================================================================
// XUser (Offline Local Profile & Privileges)
// ============================================================================

#[derive(Debug, Clone)]
pub struct LocalUser {
    pub handle: u64,
    pub xuid: u64,
    pub gamertag: String,
    pub local_id: u64,
    pub signed_in: bool,
}

#[derive(Debug)]
pub struct XUserService {
    next_handle: AtomicU64,
    users: HashMap<u64, LocalUser>,
}

impl Default for XUserService {
    fn default() -> Self {
        Self::new()
    }
}

impl XUserService {
    pub fn new() -> Self {
        let mut users = HashMap::new();
        users.insert(
            1,
            LocalUser {
                handle: 1,
                xuid: 0x0009_0000_0000_0001,
                gamertag: "PLAYER1".into(),
                local_id: 1,
                signed_in: true,
            },
        );
        Self {
            next_handle: AtomicU64::new(2),
            users,
        }
    }

    pub fn add_default_user(&mut self) -> u64 {
        let h = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.users.insert(
            h,
            LocalUser {
                handle: h,
                xuid: 0x0009_0000_0000_0001,
                gamertag: "PLAYER1".into(),
                local_id: 1,
                signed_in: true,
            },
        );
        h
    }

    pub fn get(&self, handle: u64) -> Option<&LocalUser> {
        self.users.get(&handle)
    }

    pub fn close(&mut self, handle: u64) {
        if handle != 1 {
            self.users.remove(&handle);
        }
    }

    /// Offline privilege check: single-player privileges allowed, multiplayer denied offline.
    pub fn check_privilege(&self, _handle: u64, privilege_id: u32) -> (bool, u32) {
        const MULTIPLAYER: u32 = 254;
        if privilege_id == MULTIPLAYER {
            (false, 1) // Denied: offline mode
        } else {
            (true, 0)
        }
    }
}

// ============================================================================
// XGameSave (Local VFS-Backed Save Containers & Blobs)
// ============================================================================

#[derive(Debug, Clone)]
pub struct SaveContainer {
    pub name: String,
    pub blobs: HashMap<String, Vec<u8>>,
}

#[derive(Debug)]
pub struct XGameSaveService {
    root_dir: PathBuf,
    containers: HashMap<String, SaveContainer>,
    pub remaining_quota: i64,
}

impl XGameSaveService {
    pub fn new(root_dir: PathBuf) -> Self {
        Self {
            root_dir,
            containers: HashMap::new(),
            remaining_quota: 256 * 1024 * 1024, // 256 MiB offline quota
        }
    }

    pub fn open_container(&mut self, name: &str) -> &mut SaveContainer {
        self.containers
            .entry(name.to_string())
            .or_insert_with(|| SaveContainer {
                name: name.to_string(),
                blobs: HashMap::new(),
            })
    }

    pub fn write_blob(
        &mut self,
        container: &str,
        blob_name: &str,
        data: &[u8],
    ) -> Result<(), std::io::Error> {
        let dir = self.root_dir.join(container);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join(blob_name), data)?;
        self.open_container(container)
            .blobs
            .insert(blob_name.to_string(), data.to_vec());
        Ok(())
    }

    pub fn read_blob(&mut self, container: &str, blob_name: &str) -> Option<Vec<u8>> {
        if let Some(c) = self.containers.get(container)
            && let Some(b) = c.blobs.get(blob_name)
        {
            return Some(b.clone());
        }
        let path = self.root_dir.join(container).join(blob_name);
        let bytes = std::fs::read(path).ok()?;
        self.open_container(container)
            .blobs
            .insert(blob_name.to_string(), bytes.clone());
        Some(bytes)
    }

    pub fn delete_blob(&mut self, container: &str, blob_name: &str) {
        if let Some(c) = self.containers.get_mut(container) {
            c.blobs.remove(blob_name);
        }
        let _ = std::fs::remove_file(self.root_dir.join(container).join(blob_name));
    }

    pub fn list_containers(&self) -> Vec<String> {
        self.containers.keys().cloned().collect()
    }
}

// ============================================================================
// XPackage & XStore (Offline Package Mounting & License Queries)
// ============================================================================

#[derive(Debug, Clone)]
pub struct PackageMount {
    pub package_id: String,
    pub mount_path: PathBuf,
    pub fully_installed: bool,
}

#[derive(Debug, Default)]
pub struct XPackageService {
    pub current_package_id: String,
    pub mounts: HashMap<String, PackageMount>,
    pub installed_chunks: Vec<u32>,
}

impl XPackageService {
    pub fn new(package_id: impl Into<String>) -> Self {
        Self {
            current_package_id: package_id.into(),
            mounts: HashMap::new(),
            installed_chunks: vec![0, 1],
        }
    }

    pub fn mount(&mut self, package_id: &str, path: PathBuf) -> &PackageMount {
        self.mounts
            .entry(package_id.to_string())
            .or_insert_with(|| PackageMount {
                package_id: package_id.to_string(),
                mount_path: path,
                fully_installed: true,
            })
    }

    pub fn is_chunk_installed(&self, _chunk_id: u32) -> bool {
        true // Unpacked local titles have all chunks available on disk
    }
}

#[derive(Debug, Clone)]
pub struct StoreLicense {
    pub sku_store_id: String,
    pub is_active: bool,
    pub is_trial: bool,
}

#[derive(Debug, Default)]
pub struct XStoreService {
    pub offline_mode: bool,
    pub licenses: HashMap<String, StoreLicense>,
}

impl XStoreService {
    pub fn new() -> Self {
        Self {
            offline_mode: true,
            licenses: HashMap::new(),
        }
    }

    pub fn query_game_license(&self) -> StoreLicense {
        StoreLicense {
            sku_store_id: "OFFLINE_LOCAL_TITLE".into(),
            is_active: true,
            is_trial: false,
        }
    }
}

/// Top-level `XGameRuntime` aggregator owning all offline GDK services.
pub struct XGameRuntime {
    pub initialized: bool,
    pub user: XUserService,
    pub game_save: XGameSaveService,
    pub package: XPackageService,
    pub store: XStoreService,
    pub vfs: Vfs,
}

impl XGameRuntime {
    pub fn new(save_dir: PathBuf, vfs: Vfs) -> Self {
        Self {
            initialized: false,
            user: XUserService::new(),
            game_save: XGameSaveService::new(save_dir),
            package: XPackageService::new("STRATA.LOCAL.PACKAGE_1.0.0.0_x64"),
            store: XStoreService::new(),
            vfs,
        }
    }

    pub fn initialize(&mut self) -> i32 {
        self.initialized = true;
        0 // S_OK
    }

    pub fn uninitialize(&mut self) {
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xgamesave_roundtrips_blobs_on_disk() {
        let dir = std::env::temp_dir().join("strata_xgs_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut rt = XGameRuntime::new(dir.clone(), Vfs::new());
        assert_eq!(rt.initialize(), 0);

        rt.game_save
            .write_blob("slot0", "profile.dat", b"HELLO_SAVE")
            .unwrap();
        assert_eq!(
            rt.game_save.read_blob("slot0", "profile.dat").as_deref(),
            Some(b"HELLO_SAVE".as_slice())
        );
        assert!(rt.store.query_game_license().is_active);
        assert_eq!(rt.user.get(1).unwrap().gamertag, "PLAYER1");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
