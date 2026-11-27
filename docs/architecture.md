# Strata Architecture

Strata is a clean-room, vendor-agnostic console compatibility layer built in Rust on top of a **No-API Vulkan 1.3+ (`ash`)** backend and a **direct RDNA2 → SPIR-V 1.6** shader compiler.

## Workspace Crates

- `strata-shader`: RDNA2 decoder, SSA IR, CFG structurizer (selections, loops, mid-body breaks), Wave64 liveness & subgroup-size-aware lowering, optimizer (bitcast cancellation, CSE, DCE, 128-bit BDA load coalescing), direct SPIR-V 1.6 (`PhysicalStorageBuffer64`) emitter, CPU BVH reference, and versioned persistent shader store.
- `strata-runtime-shaders`: Internal GPU passes written in Slang (`dirty_hash`, `detile`, `clamp_indirect`, `bvh_oracle`, `present_blit`, `mip_gen`, `tex_convert`) compiled at build time into embedded SPIR-V 1.6.
- `strata-formats`: Pure-Rust decoders for console texture layouts (`64KB_S`/`64KB_D`/`64KB_R_X`/`4KB_S`/linear), BCPack, XMA audio frames, and `SGD5` shader archives.
- `strata-gpu`: Raw `ash` Vulkan 1.3+ engine with 3-Tier Hybrid UMA (`VK_EXT_external_memory_host`, `DEVICE_LOCAL` mirrors, 64 KiB sparse tile aliasing), single global push-constant + bindless layout, async thin-list translation workers, dedicated transfer queue, background dword-diff readback relay, mip-chain texture manager, and deterministic frame capture/replay.
- `strata-hle`: Native x86-64 PE32+ & COFF `.lib` parser, Windows-loader-backed title mapper, static TLS redirection, Zen 2 VEH instruction emulation, VFS drive remapping, software audio mixer/XMA bridge, FidelityFX upscaler bridge, offline GDK/OS platform services, and `d3d12_xs.dll` COM vtables.
- `strata-cli`: CLI for GPU probing, self-tests, live GPU/UMA tests, PE32+/`.lib` inspection, shader census & batch pre-translation, pipeline warming, frame capture replay/diffing, and title execution.
- `xtask`: Repository hygiene checks (`asset-check`, `install-hook`).
