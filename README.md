<p align="center">
  <img src="assets/readme/banner.svg" width="100%" alt="Strata – No-API Vulkan 1.3+ and Slang console compatibility layer: native x86-64, direct RDNA2 to SPIR-V 1.6, 64-bit BDA, 3-Tier Hybrid UMA">
</p>

<p align="center">
  <img alt="License: GPL-3.0-or-later" src="https://img.shields.io/badge/license-GPL--3.0--or--later-7c5cff?style=for-the-badge">
  <img alt="Windows 10 / 11 x64" src="https://img.shields.io/badge/Windows-10%20%2F%2011%20x64-2f80ed?style=for-the-badge">
  <img alt="NVIDIA · AMD · Intel Arc" src="https://img.shields.io/badge/GPUs-NVIDIA%20%C2%B7%20AMD%20%C2%B7%20Intel%20Arc-00b894?style=for-the-badge">
  <img alt="Vulkan 1.3+ No-API" src="https://img.shields.io/badge/Vulkan-1.3%2B%20No--API-e0405a?style=for-the-badge">
  <img alt="Slang + SPIR-V 1.6" src="https://img.shields.io/badge/Shaders-Slang%20%2B%20SPIR--V%201.6-1fb9d8?style=for-the-badge">
  <img alt="Rust 2024" src="https://img.shields.io/badge/Rust-Edition%202024-f74c00?style=for-the-badge">
</p>

<p align="center">
  <b><a href="#legal--clean-room-notice">Legal Notice</a></b> ·
  <b><a href="#highlights">Highlights</a></b> ·
  <b><a href="#how-it-works">How It Works</a></b> ·
  <b><a href="#workspace-layout">Workspace</a></b> ·
  <b><a href="#what-works-today">What Works</a></b> ·
  <b><a href="#quick-start">Quick Start</a></b> ·
  <b><a href="#cli-reference">CLI</a></b> ·
  <b><a href="#testing--verification">Tests</a></b>
</p>

# Strata — No-API Vulkan & Slang Console Compatibility Layer

**Strata** is an experimental, clean-room, open-source compatibility layer written in **Rust** for running unpacked x86-64 console software natively on Windows PCs across **NVIDIA, AMD, and Intel Arc** GPUs—in the spirit of Wine and Proton—without CPU emulation.

Instead of rebuilding legacy Direct3D 12 abstractions on top of console hardware primitives, Strata takes a vendor-agnostic **"No-API" Vulkan 1.3+ (`ash`)** approach paired with a **pure-Rust Direct RDNA2 → SPIR-V 1.6** shader compiler, **subgroup-size-aware Wave64 support** (SIMD8 / SIMD16 / Wave32 / Wave64), and **build-time Slang** runtime shaders.

---

## Legal & Clean-Room Notice

> [!IMPORTANT]
> **Strata is an independent, general-purpose software compatibility research project.**
>
> - **No Target Game:** Strata is **not** tailored to, affiliated with, or endorsed by any specific game, publisher, or console manufacturer. It is a general-purpose runtime and shader-translation architecture.
> - **No Game Files or Copyrighted Data:** This repository contains **source code only**. It does **not** distribute, link to, or provide any game executables, assets, archives, firmware, or copyrighted media.
> - **No Proprietary SDKs or Keys:** This repository contains **zero** proprietary console SDK headers, binaries, import libraries, or encryption keys. All OS/graphics interfaces are independently authored clean-room implementations based on public specifications (AMD RDNA2 ISA, Khronos Vulkan 1.3+ & SPIR-V 1.6, Slang, and Microsoft Win32/PE32+ documentation).
> - **No DRM Circumvention:** Strata operates strictly on **user-supplied, already-unpacked** binaries that the user legally owns. It does **not** download, decrypt, unpack, or bypass DRM or package encryption (XVC / MSIXVC / etc.).
> - **Offline by Design:** Network/telemetry console APIs are stubbed offline by default.

---

## Highlights

<table>
  <tr>
    <td width="33%" valign="top">
      <b>No-API Vulkan 1.3+ (Raw <code>ash</code>)</b><br><br>
      No <code>wgpu</code>/<code>vulkano</code> wrappers, no <code>VkRenderPass</code>, no <code>VkFramebuffer</code>, and no vertex input layouts. Every shader shares <b>one global pipeline layout</b> (128 B push constants for User SGPRs + one image descriptor set sized to device limits, up to 32 slots per image kind) with <code>VK_KHR_dynamic_rendering</code>.
    </td>
    <td width="33%" valign="top">
      <b>Real 64-bit GPU Pointers (BDA)</b><br><br>
      Uses <code>VK_KHR_buffer_device_address</code> (<code>PhysicalStorageBuffer64</code>) so raw memory accesses, constant tables, and BVH node fetches compile to direct pointer loads—eliminating 4 GiB raw-buffer limits and multi-window <code>switch</code> overhead.
    </td>
    <td width="33%" valign="top">
      <b>3-Tier Hybrid Unified Memory</b><br><br>
      Combines <b>Tier 1</b> zero-copy CPU↔GPU memory (<code>VirtualAlloc</code> → <code>VK_EXT_external_memory_host</code>), <b>Tier 2</b> <code>DEVICE_LOCAL</code> GDDR6 mirrors, and <b>Tier 3</b> 64 KiB sparse tiled image aliasing (<code>vkQueueBindSparse</code>).
    </td>
  </tr>
  <tr>
    <td width="33%" valign="top">
      <b>Direct RDNA2 → SPIR-V 1.6 (Pure Rust)</b><br><br>
      Zero runtime HLSL text generation, zero DXC compile stalls, and zero <code>dxil.dll</code> signing. Includes SSA optimization (bitcast cancellation, CSE, DCE, <b>128-bit BDA load coalescing</b>), structured CFG (<code>OpSelectionMerge</code>/<code>OpPhi</code>), and atomic disk caching.
    </td>
    <td width="33%" valign="top">
      <b>Multi-Vendor Wave64 Support (NVIDIA · AMD · Intel Arc)</b><br><br>
      Backward whole-live mask liveness keeps lane-local masks in registers while subgroup-size-aware ballots (<code>OpGroupNonUniformBallot</code>, <code>OpAtomicOr</code>, <code>subgroupSizeControl</code>) handle <b>Intel Arc SIMD8/16</b>, <b>NVIDIA &amp; AMD RDNA Wave32</b>, and <b>AMD native Wave64</b> seamlessly.
    </td>
    <td width="33%" valign="top">
      <b>Build-Time Slang Runtime Shaders</b><br><br>
      Internal GPU compute and presentation passes (64 KiB dirty-page hashing, Morton texture detiling, batched indirect-draw clamping, BVH differential oracle, vertex-less fullscreen present) are written in <b>Slang</b> using <code>Ptr&lt;T&gt;</code> and generics, compiled offline in <code>build.rs</code>.
    </td>
  </tr>
</table>

---

## How It Works

```text
  User-supplied unpacked x86-64 PE32+ executable
                       │
                       ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  strata-hle (PE32+ Mapper, Relocations, IAT, Static TLS, VEH)     │
  │  - Native x86-64 CPU execution (no CPU emulation)                   │
  │  - Zen 2 instruction VEH (clwb, clzero, rdpru, monitorx, rdpid)     │
  │  - Clean-room thunks (xmem, d3d12_xs, xg_xs, DStorage, offline GDK) │
  └─────────────────────────┬───────────────────────────────────────────┘
                            │ D3D12.X COM vtables (ThinCommandList)
                            ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  strata-gpu (No-API Vulkan 1.3+ via raw `ash`)                    │
  │  - 3-Tier UMA: ExternalMemoryHost + GDDR6 Mirrors + 64 KiB Sparse   │
  │  - Fence-gated Bindless Slot Manager + Batched Indirect Clamping    │
  │  - VK_KHR_dynamic_rendering + Win32 VK_KHR_swapchain Presenter      │
  └─────────────────────────┬─────────────────────────┬─────────────────┘
                            │                         │
                            ▼                         ▼
  ┌──────────────────────────────────────┐  ┌───────────────────────────┐
  │  strata-shader (Pure Rust)         │  │  strata-runtime-shaders │
  │  XBCH (zlib) -> FSSM -> RDNA2 Decode │  │  Build-time Slang -> SPV  │
  │  -> SSA -> Opt (128b Coalesce)       │  │  - dirty_hash.slang       │
  │  -> Wave64-on-32 -> Direct SPIR-V 1.6│  │  - detile.slang           │
  │  -> Atomic Disk Cache                │  │  - clamp_indirect.slang   │
  └──────────────────────────────────────┘  │  - bvh_oracle.slang       │
                                            │  - present_blit.slang     │
                                            └───────────────────────────┘
```

---

## Workspace Layout

| Crate | Purpose |
| :--- | :--- |
| [`crates/strata-shader`](crates/strata-shader) | Pure-Rust RDNA2 decoder, SSA IR, optimizer (bitcast cancellation, CSE, DCE, 128-bit BDA load coalescing), Wave64 liveness & subgroup permutes (`wave_ops.rs`), `DXBC`/`XBCH` (zlib)/`FSSM` container parser, CPU BVH reference (Box16/Box32/Triangle/Instance), `SPV_KHR_ray_query` loop detector (`rayquery.rs`), `RectList` topology emulation (`rect.rs`), atomic SPIR-V disk cache, versioned shader store, and direct SPIR-V 1.6 (`PhysicalStorageBuffer64`) emitter. |
| [`crates/strata-formats`](crates/strata-formats) | Clean-room console asset format decoders: `SGD5` shader archives, `BCPack`, `XMA2` audio frame decoder, RFC 6716 `Opus` TOC/frame decoder, and console texture format/swizzle tables. |
| [`crates/strata-runtime-shaders`](crates/strata-runtime-shaders) | Emulator-authored internal shaders written in **Slang** (`Ptr<T>`, generics, wave intrinsics) compiled offline via `slangc` in `build.rs` and embedded as SPIR-V 1.6 byte slices. |
| [`crates/strata-gpu`](crates/strata-gpu) | Raw `ash` No-API Vulkan 1.3+ engine: capability probe, single global bindless + push-constant layout, 3-tier `UmaManager` (`MEM_WRITE_WATCH` + `PAGE_GUARD` + 3-way dword-diff readback), 64 KiB `SparseTilePool`, fence-gated `BindlessManager`, `ThinCommandList` + `ThinTranslationPool`, `AsyncPsoCompiler` + `HeapTextureWorker`, OOM residency pressure tiers, F1/F3 stats `overlay`, SCAP replay/bisect/trace, `FramePacingClock`, `SwapchainPresenter`, and live BDA `ComputeRunner` with Vulkan timestamp queries. |
| [`crates/strata-hle`](crates/strata-hle) | Zero-dependency PE32+ parser/mapper (sections, DIR64 relocations, IAT, exports, static TLS), Zen 2 + `PAGE_GUARD` VEH handler, `STRATA_STACKS` all-thread stack dumper, verified code patcher & write-site table, `PhysicalPagePool` scatter mapper, expanded VFS drive rewriter, `LoadLibrary`/`GetProcAddress` hooks, `DStorage`, `xgameruntime` (`QueryApiImpl` + full offline GDK/OS thunks: `ws2_32`, `winhttp`, `rpcrt4`, `mfplat`), `XboxBvhBuilder.dll` HLE, host upscaler bridge (`STRATA_UPSCALE`/`STRATA_AMBIENT`), `xg_xs` swizzle layout computer, `GameInput`, `XAudio2` (PCM/XMA2/Opus), and `d3d12_xs` COM vtables (incl. `ID3D12PipelineLibrary`). |
| [`crates/strata-cli`](crates/strata-cli) | Command-line tool for GPU capability probing, self-tests, live GPU & UMA hardware verification, PE32+ import/mapping diagnostics, shader translation/warming, and SCAP frame replay/diff/bisect/trace. |

---

## What Works Today (Experimental Prototype)

Strata is an early-stage research prototype. Subsystems below are verified on synthetic unit tests, `spirv-val --target-env vulkan1.3`, and live Vulkan 1.3 GPU hardware tests—**not** yet full retail console games.

<details>
<summary><b>Shader Compiler & Slang Runtime Shaders</b></summary>

| Component | Status | Verified By / Current Scope |
| :--- | :---: | :--- |
| RDNA2 decoder (`SOPP`/`SOPC`/`SOP1`/`SOPK`/`SOP2`, `VOP1`/`VOPC`/`VOP2`/`VOP3`, `SMEM`, `FLAT`/`GLOBAL`/`SCRATCH`, `DS`, `MUBUF`/`MTBUF`, `MIMG`, `EXP`) | ✅ | Unit tests; 400+ opcode table across all 18 GFX10.3 encodings (`opcodes.rs`, incl. `VOP3P`/`VINTRP`/`SCRATCH`) + IR decoding for compute/ALU/memory/scratch/ray-tracing |
| Direct RDNA2 → SSA → SPIR-V 1.6 (BDA, LDS incl. `ds_add_u32` atomics, VOP3, nearest-first sorted BVH64 Box32 + Box16/Triangle/Instance reference, `SPV_KHR_ray_query` loop detector, `RectList` expansion, `SCRATCH` spill frames), stores predicated on EXEC | ✅ | `spirv-val` + live GPU execution on synthetic RDNA2 kernels; `SpillFrame` (`spill.rs`) wired to `Function` scratch arrays + Wave64 permutes (`wave_ops.rs` → `OpGroupNonUniformBroadcastFirst`/`Shuffle`/`ShuffleXor`) |
| Typed buffers (`tbuffer_*`): GFX10 formats converted (UNORM `x/(2^n-1)`, SNORM clamped to -1, scaled, half, 32-bit), `idxen` record addressing | ✅ | `spirv-val` + live GPU results (RGBA8_UNORM, RG16_SNORM loads; RGBA8_UNORM store). Packed 10/11-bit formats and sub-dword typed stores return explicit errors |
| Image & atomic instructions (`image_sample`/`_l`/`_b`/`_lz`/`_c*`, `image_gather4*`, `image_get_resinfo`, `image_load(_mip)`, `image_store`, `buffer_atomic_*`, `global_atomic_*`, `s_buffer_load_dword*`, `VOP3P` packed FP16/dot, `v_permlane16`/`ds_bpermute`) → set-0 image arrays and BDA atomics | ✅ | `spirv-val` + live GPU: all MIMG dims (1D/2D/3D/Cube/Array/MSAA), depth compare (`OpImageSampleDref*`), gather (`OpImageGather`), resinfo (`OpImageQuerySizeLod`), cross-lane shuffles (`OpGroupNonUniformShuffle`), and 64-bit BDA atomics |
| Vertex / pixel / geometry / NGG mesh stages: `EXP` (`pos0`, `param0..31`, `mrt0..7`, `mrtz`, compressed halves), `v_interp_p1/p2/mov`, `SPI_PS_INPUT_ENA` (`FragCoord`/`FrontFacing` VGPRs), RDNA2-to-SPIR-V GS (`translate_geometry_rdna2`: `s_sendmsg` → `OpEmitVertex`/`OpEndPrimitive`) & NGG/Mesh (`translate_mesh_rdna2`: `SPV_EXT_mesh_shader` / `OpSetMeshOutputsEXT`), implicit-LOD sampling, quad swizzles (`RequireFullQuadsKHR`) | ✅ | `spirv-val` + live GPU draws through the D3D12.X queue (GFX10 legacy VS layout + GS + `MeshEXT` SPIR-V emission validated by `spirv-val`) |
| 128-bit BDA load coalescing & bitcast/CSE/DCE opt | ✅ | Unit test + live GPU execution |
| RDNA2-translated Structured CFG (`s_and_saveexec_b64` + `s_cbranch_execz` → `OpSelectionMerge` + `OpPhi`) | ✅ | `spirv-val` + live GPU divergent-lane execution (forward branches) |
| Structured loops (do-while, while, nested, mid-body breaks with merge-block `OpPhi`s, per-lane EXEC exit → `OpLoopMerge` + header `OpPhi`) & CFG structurizer (`OpSwitch` state-machine fallback with full block-body & predicate lowering) | ✅ | `spirv-val` + live GPU results for uniform, while, nested, divergent EXEC loops, mid-body breaks, and `structurizer.rs` state-machine lowering for irreducible/multi-back-edge CFGs |
| Descriptor tracing (V#/T#/S#/BVH descriptor → user SGPR, root table + offset, nested table, waterfall) | ✅ | Unit tests; reported by `strata-cli translate` |
| Translation coverage tool (`strata-cli translate`: finds `XBCH`/`FSSM` in any file, ranks blockers by encoding/reason) | ✅ | Unit tests on raw and embedded containers |
| RDNA2-translated Wave64 ballot (`wave.rs` whole-live analysis → `OpGroupNonUniformBallot` + Workgroup atomic merge) | ✅ | `spirv-val` + live GPU test across hardware subgroup sizes (`0x5555555555555555`) |
| `DXBC` / `XDAT` / `XBCH` (pure-Rust RFC 1950/1951 zlib) / `FSSM` parser | ✅ | Unit tests on synthetic containers & streams |
| Atomic content-addressed SPIR-V disk cache | ✅ | Roundtrip unit test |
| Build-time Slang shaders (`dirty_hash`, `detile`, `clamp_indirect`, `bvh_oracle`, `present_blit`) | ✅ | Compiled by `slangc` in `build.rs` + live GPU execution |

</details>

<details>
<summary><b>No-API Vulkan 1.3+ GPU Backend & 3-Tier Hybrid UMA</b></summary>

| Component | Status | Verified By / Current Scope |
| :--- | :---: | :--- |
| Tier 1 Zero-Copy Host UMA (`VirtualAlloc`/`mmap` → `VK_EXT_external_memory_host` → 64-bit BDA) | ✅ | Live GPU read/write test (0 `memcpy`, 64/64 lanes) |
| Tier 2 `DEVICE_LOCAL` GDDR6 VRAM Mirror + Cross-Tier GPU execution | ✅ | Live GPU test reading Tier 2 VRAM → writing Tier 1 CPU RAM |
| Tier 3 `vkQueueBindSparse` 64 KiB Aliased 2D `VkImage`s | ✅ | Live GPU sparse bind test (2 `SPARSE_ALIASED` images sharing 1 tile) |
| No-API `VK_KHR_dynamic_rendering` + Slang vertex-less `present_blit.slang` | ✅ | Live GPU render test (0 RenderPasses, 0 Framebuffers, 0 VertexBuffers) |
| Fence-gated Bindless Slot Manager (`SHOW_AFTER` pattern) | ✅ | Unit test |
| `ThinCommandList` + `ThinTranslationPool` + `AsyncPsoCompiler` + `HeapTextureWorker` (parallel multi-list flattening, GPU-side `vkCmdCopyBuffer`, `DispatchX` thread-count conversion, `UpdateTileMappings`/`CopyTileMappings` → `SparseTilePool`, MSAA resolve, Vulkan `vk::QueryPool` GPU timestamps, `PressureTier` + 500 ms `OomBackoff`, F1/F3 stats overlay) | ✅ | Unit tests + live COM → Vulkan compute/graphics execution |
| Dispatches over `maxComputeWorkGroupCount` split into `vkCmdDispatchBase` pieces (never clamped), direct and indirect (`DispatchIndirect` reads its group counts from title memory, after mirror read-back) | ✅ | Unit test + live GPU tests (every group of split direct and indirect dispatches runs) |
| Textures from T#s in title memory (GFX10 formats → Vulkan incl. BC1–BC7 & Depth/Stencil, `DST_SEL` swizzles, linear, Morton `4KB`/`64KB_S`/`64KB_D`, and bank-XOR `64KB_S_X`/`D_X`/`R_X`, multi-mip detile/retile), re-uploaded when bytes change; samplers from S#s | ✅ | Unit + live GPU tests (`texture.rs`) |
| Graphics draws: No-API `VK_KHR_dynamic_rendering` with full `GraphicsPsoState` (per-MRT blend, depth/stencil test & write, rasterizer cull/fill/depth bias/scissor, optional index buffers), D3D clip convention (flipped viewport), written back to title memory | ✅ | Live GPU draw test (`tests/draws.rs`) |
| Shadowed Tier 2 mirrors: `MEM_WRITE_WATCH` + `PAGE_GUARD` VEH tracker (`PageGuardTracker`), page-exact upload of CPU writes, 3-way dword-diff GPU→title read-back (`ReadbackTexture`) | ✅ | Live GPU & unit tests (dirty pages only, read-back restores VRAM bytes without clobbering concurrent CPU writes) |
| Pipeline table + persistent atomic `VkPipelineCache` (compute & graphics), `PipelineLibrary` serialization, versioned shader store, `strata-cli warm` | ✅ | Shared `VkPipelineCache` used for compute and graphics pipelines, saved atomically (`.tmp` + rename); unit tests + `strata-cli warm` / `shaders` |
| `FramePacingClock` + cross-platform `SurfaceHandle` / `SwapchainPresenter` (`Win32`, Linux `Xlib` / `Xcb` / `Wayland`; `acquire_next_image` / `present_image`) | ✅ | Windows (`VK_KHR_win32_surface`) + Linux X11/Wayland (`VK_KHR_xlib_surface`, `VK_KHR_xcb_surface`, `VK_KHR_wayland_surface`) surface creation + unit tests |

</details>

<details>
<summary><b>PE32+ Loader & Console HLE</b></summary>

| Component | Status | Verified By / Current Scope |
| :--- | :---: | :--- |
| PE32+ mapping through the Windows loader (patched copy mapped as a DLL: `.pdata` registered, relocations, protections) + Strata IAT binding | ✅ | Unit tests: entry execution, `RtlLookupFunctionEntry`/`RtlPcToFileHeader` on title code |
| Static TLS on every thread (OS-allocated; index-0 `gs:[0x58]` loads redirected), TLS callbacks after binding | ✅ | Unit test on guest, pre-existing and new threads; probe title |
| Title DLLs beside the executable (mapped, bound, `DllMain` in dependency order), delay-load imports bound eagerly | ✅ | Probe title (`tests/probe.rs`) |
| Loader hooks: `LoadLibrary*`/`GetProcAddress` (console pseudo handles, title DLLs), `GetModuleHandle(NULL)`/`GetModuleFileName(NULL)` answer for the title, `exit`/`ExitProcess` end the title not Strata | ✅ | Unit tests + probe title |
| `DLL_THREAD_ATTACH/DETACH` for every thread in the process, host-created ones included (an entry thunk in a `.strata` section the Windows loader calls) | ✅ | Unit test (a host `std::thread` reaches a title DLL's `DllMain`) + probe title |
| `XMemVirtualAlloc` hands the title real host pointers (console address = host address) | ✅ | Live GPU UMA test |
| `kernelx.dll` imports Strata does not implement fall back to the host's kernel exports | ✅ | Probe title |
| Console drives through the VFS for path-taking kernel32/kernelx/CRT imports (`CreateFile*`, attributes, find, directories, `fopen`, ...) + `MicrosoftGame.config` / `.xvc` container header parsing & package mounting (`Vfs::mount_package_dir`) | ✅ | Unit test + probe title |
| Crash reporter (unhandled faults as `module+rva`, `.pdata` stack unwind, crash log) | ✅ | Unit tests |
| Live Windows VEH (`clzero`, `rdpru` via QPC, `rdpid` via `GetCurrentProcessorNumber`, `monitorx`/`mwaitx`, `clwb`) | ✅ | Unit tests (note: `clwb`/`rdpid` are also present on modern Intel CPUs; VEH catches faults on older/non-AMD CPUs) |
| `xmem.dll` (`XMemVirtualAlloc`/`Free`/`Protect`, `XMemAlloc`/`Free`, physical-page pool, scatter map/unmap, page-count & working-set stats, `XMemTransferMemory`) | ✅ | Wired to `UmaManager` + `PhysicalPagePool` (incl. `MEM_LARGE_PAGES` alignment, `MEM_RELEASE`/`MEM_DECOMMIT`, `XMemUnmapPhysicalPages`); unit + live UMA tests |
| `d3d12_xs.dll` COM vtables (`Device`, `Queue`, `CommandList`, `Resource`, `Fence`, `ID3DBlob`), frame-pacing (`PresentX`, `RegisterFrameEventX`, `WaitFrameEventX`) & `xg_xs.dll` (`XGTextureAddressComputer`) | ✅ | `D3D12XboxCreateDevice` returns exact `d3d12.h` 44-slot `StandardDevice` backed by `UmaManager` (`VK_EXT_external_memory_host`), `ThinCommandList`, `ComputeRunner`, and `FramePacingClock`; `PresentX` wired to `Win32Window` via `strata-cli run --window` |
| Audio (`xaudio2_9.dll` / `xapu.dll` / `AcpHal.dll` with PCM/XMA2/Opus), `DStorageGetFactory`, `xgameruntime.dll` (`QueryApiImpl` + full offline GDK), `XboxBvhBuilder.dll`, host upscaler bridge (`amd_fidelityfx_*` / `STRATA_UPSCALE` / `STRATA_AMBIENT`), `GameInput.dll`, `ws2_32`/`winhttp`/`rpcrt4`/`mfplat`, `STRATA_STACKS`, and safe code patching (`patch.rs`) | ✅ | Real COM vtables (`ComXAudio2`/`ComXAudio2Voice`, `ComAcpHal`, `ComGameInput`/`ComGameInputReading`, `ID3D12PipelineLibrary`), shared global `Vfs` in `DStorageGetFactory`, real `QueryApiImpl`/`XAsync`/`XUser`/`XGameSave`/`XPackage`/`XStore`/OS thunks; unresolved console imports get per-import named stubs (`--unimplemented trap` / `zero` / `notimpl`) |

</details>

---

## Quick Start

### Requirements
- **OS:** Windows 10 / 11 x64
- **Toolchain:** Rust (Edition 2024, MSVC target) + Visual Studio C++ Build Tools
- **SDK:** [Vulkan SDK](https://vulkan.lunarg.com/) 1.3+ (provides `slangc.exe` for build-time Slang compilation and `spirv-val.exe` for shader validation tests)
- **GPU:** **NVIDIA**, **AMD**, or **Intel Arc** Vulkan 1.3+ GPU supporting `bufferDeviceAddress`, `descriptorIndexing`, `dynamicRendering`, `sparseResidencyImage2D`, and `VK_EXT_external_memory_host` (SIMD8 / SIMD16 / Wave32 / Wave64 supported)

### Build & Run Tests

```powershell
# Run all unit tests (includes slangc compilation and spirv-val validation)
cargo test --workspace

# Require spirv-val and list reported hardware skips on Windows
.\scripts\test.ps1

# Verify scaled presentation planes and executable replay on the selected GPU
cargo run -p strata-cli -- smoke --require-gpu

# Inspect your GPU's No-API Vulkan 1.3+ capabilities
cargo run -p strata-cli -- probe

# Run all 9 live GPU tests (BDA, Slang shaders, Zero-Copy UMA, Dynamic Rendering, Wave64-on-32)
cargo run -p strata-cli -- gpu-test

# Run live 3-Tier Hybrid UMA test (Zero-Copy Host + GDDR6 Mirror + vkQueueBindSparse)
cargo run -p strata-cli -- uma-test
```

---

## CLI Reference

```powershell
strata-cli probe                   # Query GPU No-API Vulkan 1.3+ features & subgroup size
strata-cli selftest                # Verify embedded Slang SPIR-V & direct RDNA2 -> SPIR-V emitter
strata-cli gpu-test                # Run 9 live GPU tests (RDNA2 BDA, Slang, Zero-Copy UMA, Wave64-on-32)
strata-cli uma-test                # Run live 3-tier hybrid UMA + vkQueueBindSparse test
strata-cli inspect <path.exe|lib>      # Parse PE32+ headers or COFF .lib import libraries & classify imports
strata-cli bind-check <path.exe>       # Dry-run IAT resolution against Strata HLE + Host Windows
strata-cli map-test <path.exe> [--execute] # Map PE32+, patch IAT, init TLS (+ optional entry exec & synthetic COM check)
strata-cli run <path.exe> [--window] [--unimplemented trap|zero|notimpl] [--mount D=C:/folder]... [--search-dir DIR] [--crash-log FILE]
                                       # Load the title and its DLLs, bind, attach, run (+ optional desktop PresentX window); exit code of the title
strata-cli warm <shader-cache-dir> [pipeline-cache-file]   # Create every cached shader's pipeline ahead of play
strata-cli translate <file-or-dir>     # Compile every shader found (SGD5/XBCH/FSSM); rank what blocks translation
strata-cli census <file-or-dir>        # Print stage distribution and top RDNA2 instruction encodings
strata-cli shaders <file-or-dir> [store-dir]               # Batch pre-translate shaders into persistent versioned store
strata-cli replay <inspect <a.scap> | diff <a> <b> | bisect <a> <b> | trace <a.scap> [out.json] | run <a.scap>> # Inspect, diff, bisect, trace, or replay deterministic SCAP frame captures
strata-cli config                      # Print resolved strata.toml + STRATA_* env config and verify asset-path policy
scripts\play.ps1 -Exe <path.exe> [-Console series-x|series-s] [-Adapter <idx|name>] [-Fsr4K quality|balanced|performance]
cargo xtask asset-check                # Refuse game/console content in the repository (install-hook: pre-commit)
```

---

## Testing & Verification

`strata-cli smoke --window --require-gpu --report <file>` reports first visible frame, presents, frames shown, p50/p95 frame times, replay pixel differences, adapter capabilities, and skipped checks. Without `--window`, it exercises the compositor offscreen. For a test that starts at the PE loader and exercises both title-facing `PresentX` calls, run `cargo test -p strata-hle --test probe probe_title_presents_two_planes -- --ignored` on a Windows desktop with Vulkan.

`strata-cli run <title> --window --report <file>` reports frame pacing, pipeline creation stalls, and mirrored-memory transfers. Normal launches use an adapter-specific Vulkan pipeline cache and warm a bounded set of previously seen compute and graphics pipelines. Use `--pipeline-cache <file>` to merge a cache produced by the standalone `warm` command. Use `--no-prewarm` to measure the driver cache by itself or `--cold-pipelines` for a launch without Strata's persistent pipeline cache. `smoke --pipeline-cache-dir <dir>` lets you compare two runs of the same synthetic workload.

Set `STRATA_GPU_MIRROR_DIFF=1` to try GPU-side byte merging on queue-owned allocations of at least 1 MiB. This uses additional VRAM for the last synchronized state, and the run report distinguishes requested readback bytes from bytes changed. See [capture and replay](docs/replay.md) for the executable capture format and timing commands.

Strata uses a **differential, hardware-in-the-loop verification strategy**:
1. **Static SPIR-V Validation:** Every SPIR-V binary emitted by `strata-shader` is validated against Khronos `spirv-val --target-env vulkan1.3`.
2. **Differential GPU Oracles:** Complex algorithms (such as watertight RDNA2 `image_bvh64_intersect_ray` box32 sorting and `64KB_S` Morton detiling) are implemented in both pure Rust CPU reference code and Slang GPU shaders, then executed live on the GPU and compared bit-for-bit.
3. **A stand-in title:** `crates/strata-hle/src/bin/strata_probe.rs` is our own Rust program, built with console imports, that `tests/probe.rs` loads and runs through the title loader to check HLE imports, title DLLs, TLS on threads, unwinding, module-handle hooks, the VFS and exit handling end to end.
4. **Zero-Copy & Sparse Hardware Tests:** `gpu-test` and `uma-test` exercise `VK_EXT_external_memory_host`, `DEVICE_LOCAL` BDA mirrors, `VK_KHR_dynamic_rendering`, `vkQueueBindSparse`, and `Wave64-on-Wave32` directly on real Vulkan 1.3+ silicon.

---

## License & Provenance

Licensed under **GPL-3.0-or-later**. See [LICENSE](LICENSE).

Strata is an independent clean-room implementation. No code was copied from proprietary SDKs or other compatibility layers. All architectural references are strictly public specifications (Khronos Vulkan / SPIR-V, AMD RDNA2 ISA, Slang, and Microsoft PE32+/Win32 documentation).
