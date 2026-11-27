# Deterministic Frame Capture & Replay

Strata's GPU backend includes a binary frame capture format (`SCAP`) in `strata-gpu::replay` for diagnosing rendering bugs, regressions, and shader variants without re-running a full title.

## Capabilities

- **Recording:** Captures pipeline binds, root constants, compute dispatches, graphics draws, pass markers with per-pass GPU/CPU timings, and presents into a compact binary `.scap` stream.
- **Capture Diffing (`strata-cli replay diff <a.scap> <b.scap>`):** Compares two captures command-by-command and reports the first point of divergence plus summary deltas.
- **Shader A/B Comparison:** Evaluates two shader/capture timing profiles pass-by-pass (`AbReport`) to measure frame-time improvements or regressions.
- **Pixel Diffing:** Compares RGBA8 reference and actual framebuffer snapshots (`diff_rgba8`), reporting differing pixel counts and maximum per-channel delta.
