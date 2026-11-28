# Third-party notices

Strata is GPL-3.0-or-later. It includes the following third-party code, each
under a license compatible with it.

## bcpack-rs

`third_party/bcpack-rs/` is a vendored copy (library modules only) of
https://github.com/Ahmet-Dedeler/bcpack-rs (upstream commit in
`third_party/bcpack-rs/UPSTREAM_COMMIT`), by Ahmet Dedeler, licensed under
MIT OR Apache-2.0 (`third_party/bcpack-rs/LICENSE-MIT`,
`third_party/bcpack-rs/LICENSE-APACHE`). Its rANS model tables come from
Microsoft's MIT-licensed silicon-rans; see
`third_party/bcpack-rs/THIRD_PARTY_NOTICES.md`.

## FFmpeg (WMA Pro / XMA decoder)

`crates/strata-formats/src/wmapro.rs` is a Rust port of the XMA decoding path
of FFmpeg's `libavcodec/wmaprodec.c` and parts of `libavcodec/wma.c`;
`crates/strata-formats/src/wmapro_tables.rs` is generated from
`libavcodec/wmaprodata.h`.

    Copyright (c) 2007 Baptiste Coudurier, Benjamin Larsson, Ulion
    Copyright (c) 2008 - 2009 Sascha Sommer, Benjamin Larsson

These files are licensed under the GNU Lesser General Public License,
version 2.1 or (at your option) any later version. FFmpeg's source is at
https://ffmpeg.org/.

## Runtime-loaded system and optional libraries (not distributed)

`crates/strata-hle/src/opus_host.rs` decodes Opus through decoders it loads
at run time and does not ship:

- libopus (`opus.dll`, BSD-3-Clause, https://opus-codec.org/), searched in
  `STRATA_OPUS_DIR`, next to the executable, then the DLL search path.
- Windows' "Microsoft Opus Audio Decoder" Media Foundation transform
  (`MSOpusDecoder.dll`, via `mfplat.dll`), a system component.

## opus-pure (native decoder)

The native playback backend uses `opus-pure` 0.2.4
(https://github.com/stephenberry/opus-pure), a BSD-3-Clause Rust port of
libopus derived from `rusty-opus` and `opus-rs`. The standalone entropy,
PVQ, and allocation primitives in `crates/strata-formats/src/opus_native/`
were ported from libopus 1.5.2 (commit ddbe48383984d56acd9e1ab6a090c54ca6b735a6).

```
Copyright 2026 Stephen Berry
Copyright 2026 restsend.com

Copyright 2001-2023 Xiph.Org, Skype Limited, Octasic,
                    Jean-Marc Valin, Timothy B. Terriberry,
                    CSIRO, Gregory Maxwell, Mark Borgerding,
                    Erik de Castro Lopo, Mozilla, Amazon

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Opus is subject to the royalty-free patent licenses which are
specified at:

Xiph.Org Foundation:
https://datatracker.ietf.org/ipr/1524/

Microsoft Corporation:
https://datatracker.ietf.org/ipr/1914/

Broadcom Corporation:
https://datatracker.ietf.org/ipr/1526/
```
