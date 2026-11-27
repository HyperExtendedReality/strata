# Repository Asset Policy

Strata is a **source-code-only** repository.

1. **No game files, binaries, or copyrighted assets:** Executables (`.exe`, `.dll`, `.sys`, `.xex`), archives (`.xvc`, `.msixvc`, `.rpf`, `.pak`, `.sgd5`), shader containers (`.xbch`, `.fssm`, `.dxbc`, `.dxil`), media (`.bk2`, `.xma`, `.dds`, `.wav`), saves, and keys are strictly forbidden.
2. **External data paths only:** User-supplied title files, caches, and save data configured via `strata.toml` or `STRATA_*` environment variables must live **outside** the repository tree. Strata refuses to run if a configured data path resolves inside the repository.
3. **Automated enforcement:** `cargo xtask asset-check` scans tracked and untracked files by extension, magic bytes, and size limit (2 MiB). Install it as a git pre-commit hook via `cargo xtask install-hook`; CI also runs it on every push and pull request.
4. **License:** All source files in this repository are licensed under **GPL-3.0-or-later** (see [LICENSE](../LICENSE)).
