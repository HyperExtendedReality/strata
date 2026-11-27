# Deterministic Frame Capture & Replay

Strata can save a title's command trace and final presentation planes for inspection and Vulkan replay. Save captures outside the source tree when running a title with external assets.

```powershell
strata-cli run C:\fixtures\title.exe --capture C:\captures\frame.scap --capture-frame 5 --report C:\captures\run.json
strata-cli replay inspect C:\captures\frame.scap
strata-cli replay run C:\captures\frame.scap
strata-cli replay diff C:\captures\before.scap C:\captures\after.scap
```

`--capture` enables desktop presentation and defaults to the first frame. The executable capture includes the ordered RGBA8/BGRA8 input planes and the composited reference image. Replay reconstructs the Vulkan images and reruns the Slang presentation pipeline. The report gives the first differing pixel, maximum byte delta, and wall time and GPU timestamps for each composition pass when supported. `replay run` exits with an error if pixels differ.

Queue submissions record pipeline binds, root constants, direct dispatches, and direct draws for structural comparison. The original command-only `SCAP` format still opens in `inspect` and `diff`. Executable captures replay the final composition pass. Re-executing the title's earlier draws and dispatches would require capturing their resource histories and state transitions.

To reproduce with generated input, run `strata-cli smoke --require-gpu --capture C:\captures\synthetic.scap`. Add `--window` to measure desktop presentation.
