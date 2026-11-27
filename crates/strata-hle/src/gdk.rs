//! Offline GDK Services (`XUser`, `XGameSave`, `XPackage`; DirectStorage lives in
//! [`crate::dstorage`]).
//!
//! Provides clean-room offline implementations so console titles can mount local
//! storage, stream archives via DirectStorage, and read/write saves without any
//! network access.

/// Offline console user profile returned to `XUser` / `XGameSave` queries.
#[derive(Debug, Clone)]
pub struct OfflineProfile {
    pub gamertag: String,
    pub xuid: u64,
}

impl Default for OfflineProfile {
    fn default() -> Self {
        Self {
            gamertag: "PLAYER1".to_string(),
            xuid: 0x0009_0000_0000_0001,
        }
    }
}
