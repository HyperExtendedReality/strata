# Vendoring policy

This directory is an **unmodified** copy of the library modules of
https://github.com/Ahmet-Dedeler/bcpack-rs at the commit in `UPSTREAM_COMMIT`
(MIT OR Apache-2.0). The only differences from upstream are:

- `Cargo.toml`: library only (no CLI binary, no `ttpf` / `cli` features).
- `src/lib.rs`: the `ttpf` and `texture` module declarations are removed and
  a one-paragraph vendoring note is added to the crate docs.
- `src/main.rs`, `src/ttpf.rs`, `src/texture.rs`, `python/`, `tests/`,
  `tools/` and `docs/` are not copied.

Rules:

1. Do not edit the copied sources here. Keeping them byte-identical to
   upstream makes license compliance trivial (the original headers and
   `LICENSE-*` files stay with the code) and keeps updates a plain re-copy.
2. Strata-specific behaviour lives in `crates/strata-formats/src/bcpack.rs`
   (format mapping, error mapping, BC2/BC6H fallback) or, for GPU decoding,
   in Strata's own shaders.
3. Decoder fixes and new formats (BC6H block rebuild) go upstream as pull
   requests, then come back here by re-vendoring.

Updating:

```powershell
git clone --depth 1 https://github.com/Ahmet-Dedeler/bcpack-rs $env:TEMP\bcpack-rs
foreach ($f in 'bc1','bc7','bcn','encode','format','pixels','rans','tables') {
  Copy-Item "$env:TEMP\bcpack-rs\src\$f.rs" third_party\bcpack-rs\src\ }
Copy-Item -Recurse -Force "$env:TEMP\bcpack-rs\data" third_party\bcpack-rs\
Copy-Item "$env:TEMP\bcpack-rs\LICENSE-*","$env:TEMP\bcpack-rs\THIRD_PARTY_NOTICES.md","$env:TEMP\bcpack-rs\README.md" third_party\bcpack-rs\
git -C $env:TEMP\bcpack-rs rev-parse HEAD > third_party\bcpack-rs\UPSTREAM_COMMIT
cargo test -p strata-formats bcpack
```

Re-apply the `src/lib.rs` edits above if upstream changed that file.
