//! WMA Pro frame decoder in its XMA1/XMA2 configuration.
//!
//! A Rust port of the XMA path of FFmpeg's `libavcodec/wmaprodec.c` and
//! `ff_wma_run_level_decode` / `ff_wma_get_large_val` from `wma.c`
//! (Copyright (c) 2007 Baptiste Coudurier, Benjamin Larsson, Ulion;
//! Copyright (c) 2008 - 2009 Sascha Sommer, Benjamin Larsson), used under
//! the GNU Lesser General Public License 2.1 or later. The tables in
//! [`crate::wmapro_tables`] are generated from `wmaprodata.h`.
//!
//! SPDX-License-Identifier: LGPL-2.1-or-later
//!
//! XMA streams use fixed WMA Pro parameters: decode flags `0x10d6` (length
//! prefix, dynamic range compression data, up to 4 subframes), 512 samples
//! per frame, 16-bit scaling and at most 2 channels per stream. The frame
//! packing into 2048-byte packets is [`crate::xma`]'s job; this module
//! decodes one frame's bits (starting at its 15-bit length) at a time.
//!
//! Differences from FFmpeg: no multichannel (> 2) channel groups (an XMA
//! stream has 1 or 2 channels), and the inverse MDCT is computed directly
//! (same definition and scale as FFmpeg's half-length inverse MDCT).

use crate::wmapro_tables as t;
use std::sync::OnceLock;

/// Samples per channel each frame produces.
pub const SAMPLES_PER_FRAME: usize = 512;
const MAX_BANDS: usize = 29;
const MAX_SUBFRAMES: usize = 32;
const MAX_CHANNELS: usize = 2;
const BLOCK_SIZES: usize = 3; // 512, 256, 128
const MAX_SUBFRAMES_LOG2: u32 = 2; // decode flags 0x10d6
const MIN_SAMPLES_PER_SUBFRAME: usize = SAMPLES_PER_FRAME >> MAX_SUBFRAMES_LOG2;
const SUBFRAME_LEN_BITS: u32 = 2;
const LOG2_FRAME_SIZE: u32 = 15;
const BITS_PER_SAMPLE: i32 = 16;
const HUFF_VEC1_SIZE: u32 = 101;

/// Why a frame could not be decoded (the decoder then outputs silence).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WmaProError {
    InvalidData,
    Unsupported,
}

type R<T> = Result<T, WmaProError>;

fn log2(v: usize) -> u32 {
    usize::BITS - 1 - v.max(1).leading_zeros()
}

/// MSB-first bit reader; reads past the end return zeros (FFmpeg pads its
/// buffers), and `left()` tells callers how much real data remains.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8], bit_len: usize) -> Self {
        Self {
            data,
            pos: 0,
            end: bit_len.min(data.len() * 8),
        }
    }

    fn bit(&mut self) -> u32 {
        let p = self.pos;
        self.pos += 1;
        if p >= self.end {
            return 0;
        }
        u32::from(self.data[p / 8] >> (7 - (p % 8))) & 1
    }

    fn get(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.bit();
        }
        v
    }

    fn sbits(&mut self, n: u32) -> i32 {
        let v = self.get(n) as i32;
        (v << (32 - n)) >> (32 - n)
    }

    fn skip(&mut self, n: usize) {
        self.pos += n;
    }

    fn count(&self) -> usize {
        self.pos
    }

    fn left(&self) -> isize {
        self.end as isize - self.pos as isize
    }
}

/// A prefix code built like FFmpeg's `ff_vlc_init_from_lengths`: codes are
/// assigned in table order, each `len` bits long.
struct Vlc {
    by_len: Vec<Vec<(u32, i32)>>,
}

impl Vlc {
    fn from_lengths(entries: impl Iterator<Item = (u8, i32)>) -> Self {
        let mut by_len: Vec<Vec<(u32, i32)>> = vec![Vec::new(); 33];
        let mut code: u64 = 0;
        for (len, sym) in entries {
            if len == 0 {
                continue;
            }
            let len = u32::from(len);
            by_len[len as usize].push(((code >> (32 - len)) as u32, sym));
            code += 1u64 << (32 - len);
        }
        for l in &mut by_len {
            l.sort_unstable();
        }
        Self { by_len }
    }

    fn pairs(table: &[[u8; 2]], offset: i32) -> Self {
        Self::from_lengths(table.iter().map(|p| (p[1], i32::from(p[0]) + offset)))
    }

    fn read(&self, b: &mut Bits<'_>) -> R<i32> {
        let mut code = 0u32;
        for len in 1..self.by_len.len() {
            code = (code << 1) | b.bit();
            let l = &self.by_len[len];
            if let Ok(i) = l.binary_search_by_key(&code, |e| e.0) {
                return Ok(l[i].1);
            }
        }
        Err(WmaProError::InvalidData)
    }
}

struct Tables {
    sf: Vlc,
    sf_rl: Vlc,
    coef: [Vlc; 2],
    vec4: Vlc,
    vec2: Vlc,
    vec1: Vlc,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| Tables {
        sf: Vlc::pairs(&t::SCALE_TABLE, -60),
        sf_rl: Vlc::pairs(&t::SCALE_RL_TABLE, 0),
        coef: [
            Vlc::from_lengths(
                t::COEF0_LENS
                    .iter()
                    .zip(t::COEF0_SYMS.iter())
                    .map(|(&l, &s)| (l, i32::from(s))),
            ),
            Vlc::pairs(&t::COEF1_TABLE, 0),
        ],
        vec4: Vlc::from_lengths(
            t::VEC4_LENS
                .iter()
                .zip(t::VEC4_SYMS.iter())
                .map(|(&l, &s)| (l, i32::from(s) - 1)),
        ),
        vec2: Vlc::pairs(&t::VEC2_TABLE, -1),
        vec1: Vlc::pairs(&t::VEC1_TABLE, 0),
    })
}

/// Sine window of `n` taps (`ff_sine_window_init`).
fn sine_window(n: usize) -> &'static [f32] {
    static W: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    let w = W.get_or_init(|| {
        (0..=9)
            .map(|b| {
                let n = 1usize << b;
                (0..n)
                    .map(|i| {
                        ((i as f64 + 0.5) * (std::f64::consts::PI / (2.0 * n as f64))).sin() as f32
                    })
                    .collect()
            })
            .collect()
    });
    &w[log2(n) as usize]
}

/// Half-length inverse MDCT of `n` coefficients into `n` samples, as
/// FFmpeg's `av_tx` float MDCT computes it: the negated middle half of the
/// full `2n` output,
/// `y[i] = -s * sum_k X[k] cos(pi/n (i + n + 1/2)(k + 1/2))`,
/// `s = 1 / (n/2) / 2^(bits_per_sample - 1)`.
fn imdct_half(input: &[f32], out: &mut [f32]) {
    static COS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    let n = input.len();
    let cos = COS.get_or_init(|| {
        (0..BLOCK_SIZES)
            .map(|i| {
                let n = SAMPLES_PER_FRAME >> i;
                let mut c = Vec::with_capacity(n * n);
                for o in 0..n {
                    for k in 0..n {
                        let a = std::f64::consts::PI / n as f64
                            * (o as f64 + n as f64 + 0.5)
                            * (k as f64 + 0.5);
                        c.push(a.cos() as f32);
                    }
                }
                c
            })
            .collect()
    });
    let table = &cos[log2(SAMPLES_PER_FRAME / n) as usize];
    let scale = 1.0 / (n as f32 / 2.0) / (1u32 << (BITS_PER_SAMPLE - 1)) as f32;
    for (o, y) in out[..n].iter_mut().enumerate() {
        let row = &table[o * n..o * n + n];
        let s: f32 = row.iter().zip(input).map(|(c, x)| c * x).sum();
        *y = -s * scale;
    }
}

#[derive(Clone)]
struct Channel {
    prev_block_len: usize,
    transmit_coefs: bool,
    num_subframes: usize,
    subframe_len: [usize; MAX_SUBFRAMES],
    cur_subframe: usize,
    decoded_samples: usize,
    grouped: bool,
    quant_step: i32,
    reuse_sf: bool,
    scale_factor_step: i32,
    max_scale_factor: i32,
    saved_scale_factors: [[i32; MAX_BANDS]; 2],
    scale_factor_idx: usize,
    /// Which `saved_scale_factors` row the current subframe uses.
    sf_cur: usize,
    table_idx: usize,
    num_vec_coeffs: usize,
    /// Offset of the current subframe's coefficients in `out`.
    coeffs: usize,
    out: Vec<f32>,
}

impl Channel {
    fn new() -> Self {
        Self {
            prev_block_len: SAMPLES_PER_FRAME,
            transmit_coefs: false,
            num_subframes: 0,
            subframe_len: [0; MAX_SUBFRAMES],
            cur_subframe: 0,
            decoded_samples: 0,
            grouped: false,
            quant_step: 0,
            reuse_sf: false,
            scale_factor_step: 1,
            max_scale_factor: 0,
            saved_scale_factors: [[0; MAX_BANDS]; 2],
            scale_factor_idx: 0,
            sf_cur: 0,
            table_idx: 0,
            num_vec_coeffs: 0,
            coeffs: 0,
            out: vec![0.0; SAMPLES_PER_FRAME + SAMPLES_PER_FRAME / 2],
        }
    }
}

#[derive(Clone, Default)]
struct Group {
    channels: Vec<usize>,
    transform: bool,
    transform_band: [bool; MAX_BANDS],
    matrix: [f32; MAX_CHANNELS * MAX_CHANNELS],
}

/// One XMA stream's WMA Pro decoder (1 or 2 channels).
pub struct WmaProDecoder {
    nb_channels: usize,
    num_sfb: [usize; BLOCK_SIZES],
    sfb_offsets: [[usize; MAX_BANDS]; BLOCK_SIZES],
    sf_offsets: [[[usize; MAX_BANDS]; BLOCK_SIZES]; BLOCK_SIZES],
    ch: Vec<Channel>,
    groups: Vec<Group>,
    skip_frame: bool,
    // Per-subframe state.
    subframe_len: usize,
    table_idx: usize,
    num_bands: usize,
    esc_len: u32,
    transmit_num_vec_coeffs: bool,
    cur_channels: Vec<usize>,
    parsed_all_subframes: bool,
    num_saved_bits: usize,
    tmp: Vec<f32>,
    /// The last frame's dynamic range compression gain byte.
    pub drc_gain: u8,
    pub frames: u64,
}

impl std::fmt::Debug for WmaProDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WmaProDecoder")
            .field("channels", &self.nb_channels)
            .field("frames", &self.frames)
            .finish()
    }
}

/// The rate band tables are laid out for (`get_rate` for XMA).
fn table_rate(sample_rate: u32) -> u32 {
    match sample_rate {
        r if r > 44100 => 48000,
        r if r > 32000 => 44100,
        r if r > 24000 => 32000,
        _ => 24000,
    }
}

impl WmaProDecoder {
    /// A decoder for an XMA stream of `channels` (1 or 2) at `sample_rate`.
    pub fn new(channels: usize, sample_rate: u32) -> Option<Self> {
        if !(1..=MAX_CHANNELS).contains(&channels) {
            return None;
        }
        let rate = table_rate(sample_rate) as usize;
        let mut num_sfb = [0usize; BLOCK_SIZES];
        let mut sfb = [[0usize; MAX_BANDS]; BLOCK_SIZES];
        for i in 0..BLOCK_SIZES {
            let subframe_len = SAMPLES_PER_FRAME >> i;
            let mut band = 1usize;
            for x in 0..MAX_BANDS - 1 {
                if sfb[i][band - 1] >= subframe_len {
                    break;
                }
                let mut offset = (subframe_len * 2 * t::CRITICAL_FREQ[x] as usize) / rate + 2;
                offset &= !3;
                if offset > sfb[i][band - 1] {
                    sfb[i][band] = offset;
                    band += 1;
                }
                if offset >= subframe_len {
                    break;
                }
            }
            sfb[i][band - 1] = subframe_len;
            num_sfb[i] = band - 1;
            if num_sfb[i] == 0 {
                return None;
            }
        }
        let mut sf_offsets = [[[0usize; MAX_BANDS]; BLOCK_SIZES]; BLOCK_SIZES];
        for i in 0..BLOCK_SIZES {
            for b in 0..num_sfb[i] {
                let offset = ((sfb[i][b] + sfb[i][b + 1] - 1) << i) >> 1;
                for x in 0..BLOCK_SIZES {
                    let mut v = 0usize;
                    while v + 1 < MAX_BANDS && sfb[x][v + 1] << x < offset {
                        v += 1;
                    }
                    sf_offsets[i][x][b] = v;
                }
            }
        }
        Some(Self {
            nb_channels: channels,
            num_sfb,
            sfb_offsets: sfb,
            sf_offsets,
            ch: vec![Channel::new(); channels],
            groups: Vec::new(),
            skip_frame: true,
            subframe_len: 0,
            table_idx: 0,
            num_bands: 0,
            esc_len: 0,
            transmit_num_vec_coeffs: false,
            cur_channels: Vec::new(),
            parsed_all_subframes: false,
            num_saved_bits: 0,
            tmp: vec![0.0; SAMPLES_PER_FRAME],
            drc_gain: 0,
            frames: 0,
        })
    }

    pub fn channels(&self) -> usize {
        self.nb_channels
    }

    /// Forgets the overlap and prediction state (a seek or stream restart).
    pub fn reset(&mut self) {
        self.ch = vec![Channel::new(); self.nb_channels];
        self.skip_frame = true;
    }

    /// Decodes one frame (`bit_len` bits of `data`, starting at the 15-bit
    /// frame length) into `out`: `SAMPLES_PER_FRAME` samples per channel,
    /// interleaved. The first frame after a reset only primes the overlap
    /// and outputs silence (FFmpeg drops it). On a bitstream error the frame
    /// is silence and the state resets. Returns the frame's "more frames
    /// follow" bit.
    pub fn decode_frame(&mut self, data: &[u8], bit_len: usize, out: &mut Vec<f32>) -> R<bool> {
        let base = out.len();
        out.resize(base + SAMPLES_PER_FRAME * self.nb_channels, 0.0);
        let mut b = Bits::new(data, bit_len);
        self.num_saved_bits = b.end;
        match self.frame(&mut b) {
            Ok(more) => {
                let skip = std::mem::replace(&mut self.skip_frame, false);
                if !skip {
                    for (c, ch) in self.ch.iter().enumerate() {
                        for (i, &s) in ch.out[..SAMPLES_PER_FRAME].iter().enumerate() {
                            out[base + i * self.nb_channels + c] = s;
                        }
                    }
                }
                for ch in &mut self.ch {
                    ch.out.copy_within(
                        SAMPLES_PER_FRAME..SAMPLES_PER_FRAME + SAMPLES_PER_FRAME / 2,
                        0,
                    );
                }
                self.frames += 1;
                Ok(more)
            }
            Err(e) => {
                self.reset();
                Err(e)
            }
        }
    }

    fn frame(&mut self, b: &mut Bits<'_>) -> R<bool> {
        let len = b.get(LOG2_FRAME_SIZE) as usize;
        self.decode_tilehdr(b)?;
        if self.nb_channels > 1 && b.bit() == 1 && b.bit() == 1 {
            b.skip(4 * self.nb_channels * self.nb_channels);
        }
        self.drc_gain = b.get(8) as u8;
        if b.bit() == 1 {
            let bits = log2(SAMPLES_PER_FRAME * 2);
            if b.bit() == 1 {
                b.get(bits); // trim start
            }
            if b.bit() == 1 {
                b.get(bits); // trim end
            }
        }
        self.parsed_all_subframes = false;
        for ch in &mut self.ch {
            ch.decoded_samples = 0;
            ch.cur_subframe = 0;
            ch.reuse_sf = false;
        }
        while !self.parsed_all_subframes {
            self.decode_subframe(b)?;
        }
        // The length covers everything up to the "more frames" bit, which
        // follows one padding bit.
        if len != b.count() + 2 {
            return Err(WmaProError::InvalidData);
        }
        b.skip(len - b.count() - 1);
        Ok(b.bit() == 1)
    }

    fn decode_subframe_length(&self, b: &mut Bits<'_>, offset: usize) -> R<usize> {
        if offset == SAMPLES_PER_FRAME - MIN_SAMPLES_PER_SUBFRAME {
            return Ok(MIN_SAMPLES_PER_SUBFRAME);
        }
        if b.left() < 1 {
            return Err(WmaProError::InvalidData);
        }
        // max_subframe_len_bit (4 subframes)
        let shift = if b.bit() == 1 {
            1 + b.get(SUBFRAME_LEN_BITS - 1)
        } else {
            0
        };
        let len = SAMPLES_PER_FRAME >> shift;
        if !(MIN_SAMPLES_PER_SUBFRAME..=SAMPLES_PER_FRAME).contains(&len) {
            return Err(WmaProError::InvalidData);
        }
        Ok(len)
    }

    fn decode_tilehdr(&mut self, b: &mut Bits<'_>) -> R<()> {
        let n = self.nb_channels;
        let mut num_samples = [0usize; MAX_CHANNELS];
        let mut contains = [false; MAX_CHANNELS];
        let mut channels_for_cur = n;
        let mut min_channel_len = 0usize;
        for ch in &mut self.ch {
            ch.num_subframes = 0;
        }
        let fixed = b.bit() == 1;
        loop {
            for c in 0..n {
                contains[c] = if num_samples[c] == min_channel_len {
                    fixed
                        || channels_for_cur == 1
                        || min_channel_len == SAMPLES_PER_FRAME - MIN_SAMPLES_PER_SUBFRAME
                        || b.bit() == 1
                } else {
                    false
                };
            }
            let len = self.decode_subframe_length(b, min_channel_len)?;
            min_channel_len += len;
            for c in 0..n {
                let ch = &mut self.ch[c];
                if contains[c] {
                    if ch.num_subframes >= MAX_SUBFRAMES {
                        return Err(WmaProError::InvalidData);
                    }
                    ch.subframe_len[ch.num_subframes] = len;
                    num_samples[c] += len;
                    ch.num_subframes += 1;
                    if num_samples[c] > SAMPLES_PER_FRAME {
                        return Err(WmaProError::InvalidData);
                    }
                } else if num_samples[c] <= min_channel_len {
                    if num_samples[c] < min_channel_len {
                        channels_for_cur = 0;
                        min_channel_len = num_samples[c];
                    }
                    channels_for_cur += 1;
                }
            }
            if min_channel_len >= SAMPLES_PER_FRAME {
                break;
            }
        }
        Ok(())
    }

    fn decode_channel_transform(&mut self, b: &mut Bits<'_>) -> R<()> {
        self.groups.clear();
        if self.nb_channels < 2 {
            return Ok(());
        }
        if b.bit() == 1 {
            return Err(WmaProError::Unsupported); // channel transform bit
        }
        let mut remaining = self.cur_channels.len();
        while remaining > 0 && self.groups.len() < self.cur_channels.len() {
            let mut g = Group::default();
            // remaining <= 2: the group takes every ungrouped channel.
            for &c in &self.cur_channels {
                if !self.ch[c].grouped {
                    g.channels.push(c);
                }
                self.ch[c].grouped = true;
            }
            let num = remaining;
            if num == 2 {
                if b.bit() == 1 {
                    if b.bit() == 1 {
                        return Err(WmaProError::Unsupported);
                    }
                } else {
                    g.transform = true;
                    // Mid/side (a 2-channel group in a 2-channel stream).
                    g.matrix = [1.0, -1.0, 1.0, 1.0];
                }
            }
            if g.transform {
                if b.bit() == 0 {
                    for i in 0..self.num_bands {
                        g.transform_band[i] = b.bit() == 1;
                    }
                } else {
                    g.transform_band[..self.num_bands].fill(true);
                }
            }
            remaining -= num;
            self.groups.push(g);
        }
        Ok(())
    }

    fn decode_coeffs(&mut self, b: &mut Bits<'_>, c: usize) -> R<()> {
        let tb = tables();
        let vlctable = b.bit() as usize;
        let (run, level): (&[u16], &[f32]) = if vlctable == 1 {
            (&t::COEF1_RUN, &t::COEF1_LEVEL)
        } else {
            (&t::COEF0_RUN, &t::COEF0_LEVEL)
        };
        let subframe_len = self.subframe_len;
        let transmit_nvc = self.transmit_num_vec_coeffs;
        let esc_len = self.esc_len;
        let ch = &mut self.ch[c];
        let num_vec_coeffs = ch.num_vec_coeffs;
        let base = ch.coeffs;
        let coeffs = &mut ch.out[base..base + subframe_len];
        let mut rl_mode = false;
        let mut cur = 0usize;
        let mut num_zeros = 0usize;
        while (transmit_nvc || !rl_mode) && cur + 3 < num_vec_coeffs {
            let mut vals = [0u32; 4];
            let idx = tb.vec4.read(b)?;
            if idx < 0 {
                for i in (0..4).step_by(2) {
                    let idx = tb.vec2.read(b)?;
                    if idx < 0 {
                        let mut v0 = tb.vec1.read(b)? as u32;
                        if v0 == HUFF_VEC1_SIZE - 1 {
                            v0 += wma_get_large_val(b);
                        }
                        let mut v1 = tb.vec1.read(b)? as u32;
                        if v1 == HUFF_VEC1_SIZE - 1 {
                            v1 += wma_get_large_val(b);
                        }
                        vals[i] = v0;
                        vals[i + 1] = v1;
                    } else {
                        let idx = idx as u32;
                        vals[i] = idx >> 4;
                        vals[i + 1] = idx & 0xF;
                    }
                }
            } else {
                let idx = idx as u32;
                vals = [idx >> 12, (idx >> 8) & 0xF, (idx >> 4) & 0xF, idx & 0xF];
            }
            for v in vals {
                if v != 0 {
                    let f = v as f32;
                    coeffs[cur] = if b.bit() == 1 { f } else { -f };
                    num_zeros = 0;
                } else {
                    coeffs[cur] = 0.0;
                    num_zeros += 1;
                    rl_mode |= num_zeros > subframe_len >> 8;
                }
                cur += 1;
            }
        }
        if cur < subframe_len {
            coeffs[cur..].fill(0.0);
            wma_run_level_decode(
                b,
                &tb.coef[vlctable],
                level,
                run,
                coeffs,
                cur,
                subframe_len,
                esc_len,
            )?;
        }
        Ok(())
    }

    fn decode_scale_factors(&mut self, b: &mut Bits<'_>) -> R<()> {
        let tb = tables();
        let num_bands = self.num_bands;
        for &c in &self.cur_channels.clone() {
            let table_idx = self.table_idx;
            let ch = &mut self.ch[c];
            ch.sf_cur = 1 - ch.scale_factor_idx;
            if ch.reuse_sf {
                let offs = &self.sf_offsets[table_idx][ch.table_idx];
                for bnd in 0..num_bands {
                    ch.saved_scale_factors[ch.sf_cur][bnd] =
                        ch.saved_scale_factors[ch.scale_factor_idx][offs[bnd]];
                }
            }
            if ch.cur_subframe == 0 || b.bit() == 1 {
                let cur = ch.sf_cur;
                if !ch.reuse_sf {
                    ch.scale_factor_step = b.get(2) as i32 + 1;
                    let mut val = 45 / ch.scale_factor_step;
                    for bnd in 0..num_bands {
                        val += tb.sf.read(b)?;
                        ch.saved_scale_factors[cur][bnd] = val;
                    }
                } else {
                    let mut i = 0usize;
                    while i < num_bands {
                        let idx = tb.sf_rl.read(b)?;
                        let (skip, val, sign);
                        if idx == 0 {
                            let code = b.get(14);
                            val = (code >> 6) as i32;
                            sign = (code & 1) as i32 - 1;
                            skip = ((code & 0x3f) >> 1) as usize;
                        } else if idx == 1 {
                            break;
                        } else {
                            skip = t::SCALE_RL_RUN[idx as usize] as usize;
                            val = i32::from(t::SCALE_RL_LEVEL[idx as usize]);
                            sign = b.bit() as i32 - 1;
                        }
                        i += skip;
                        if i >= num_bands {
                            return Err(WmaProError::InvalidData);
                        }
                        ch.saved_scale_factors[cur][i] += (val ^ sign) - sign;
                        i += 1;
                    }
                }
                ch.scale_factor_idx = 1 - ch.scale_factor_idx;
                ch.table_idx = table_idx;
                ch.reuse_sf = true;
            }
            let sf = &ch.saved_scale_factors[ch.sf_cur][..num_bands];
            ch.max_scale_factor = sf.iter().copied().max().unwrap_or(0);
        }
        Ok(())
    }

    fn inverse_channel_transform(&mut self) {
        let sfb = self.sfb_offsets[self.table_idx];
        for g in &self.groups {
            if !g.transform {
                continue;
            }
            let n = g.channels.len();
            for bnd in 0..self.num_bands {
                let (lo, hi) = (sfb[bnd], sfb[bnd + 1].min(self.subframe_len));
                if g.transform_band[bnd] {
                    for y in lo..hi {
                        let mut data = [0.0f32; MAX_CHANNELS];
                        for (j, &c) in g.channels.iter().enumerate() {
                            data[j] = self.ch[c].out[self.ch[c].coeffs + y];
                        }
                        for (k, &c) in g.channels.iter().enumerate() {
                            let sum: f32 = (0..n).map(|j| data[j] * g.matrix[k * n + j]).sum();
                            let at = self.ch[c].coeffs + y;
                            self.ch[c].out[at] = sum;
                        }
                    }
                } else if self.nb_channels == 2 {
                    for &c in &g.channels {
                        let base = self.ch[c].coeffs;
                        for v in &mut self.ch[c].out[base + lo..base + hi] {
                            *v *= 181.0 / 128.0;
                        }
                    }
                }
            }
        }
    }

    fn window(&mut self) {
        for &c in &self.cur_channels {
            let ch = &mut self.ch[c];
            let mut winlen = ch.prev_block_len;
            let mut start = ch.coeffs - (winlen >> 1);
            if self.subframe_len < winlen {
                start += (winlen - self.subframe_len) >> 1;
                winlen = self.subframe_len;
            }
            let win = sine_window(winlen);
            let half = winlen >> 1;
            // ff_vector_fmul_window(dst = src0 = start, src1 = start + half)
            // With FFmpeg's i = m - half and j = half - 1 - m:
            for m in 0..half {
                let ii = m;
                let jj = half - 1 - m;
                let s0 = ch.out[start + ii];
                let s1 = ch.out[start + half + jj];
                let wi = win[ii];
                let wj = win[half + jj];
                ch.out[start + ii] = s0 * wj - s1 * wi;
                ch.out[start + half + jj] = s0 * wi + s1 * wj;
            }
            ch.prev_block_len = self.subframe_len;
        }
    }

    fn decode_subframe(&mut self, b: &mut Bits<'_>) -> R<()> {
        let n = self.nb_channels;
        let mut offset = SAMPLES_PER_FRAME;
        let mut subframe_len = SAMPLES_PER_FRAME;
        let mut total_samples = SAMPLES_PER_FRAME * n;
        for ch in &mut self.ch {
            ch.grouped = false;
            if offset > ch.decoded_samples {
                offset = ch.decoded_samples;
                subframe_len = ch.subframe_len[ch.cur_subframe.min(MAX_SUBFRAMES - 1)];
            }
        }
        self.cur_channels.clear();
        for (i, ch) in self.ch.iter_mut().enumerate() {
            total_samples -= ch.decoded_samples;
            let cs = ch.cur_subframe.min(MAX_SUBFRAMES - 1);
            if offset == ch.decoded_samples && subframe_len == ch.subframe_len[cs] {
                total_samples -= ch.subframe_len[cs];
                ch.decoded_samples += ch.subframe_len[cs];
                self.cur_channels.push(i);
            }
        }
        if total_samples == 0 {
            self.parsed_all_subframes = true;
        }
        if self.cur_channels.is_empty() || subframe_len == 0 {
            return Err(WmaProError::InvalidData);
        }
        self.table_idx = log2(SAMPLES_PER_FRAME / subframe_len) as usize;
        if self.table_idx >= BLOCK_SIZES {
            return Err(WmaProError::InvalidData);
        }
        self.num_bands = self.num_sfb[self.table_idx];
        let offset = offset + (SAMPLES_PER_FRAME >> 1);
        for &c in &self.cur_channels {
            self.ch[c].coeffs = offset;
        }
        self.subframe_len = subframe_len;
        self.esc_len = log2(subframe_len - 1) + 1;

        if b.bit() == 1 {
            let mut num_fill_bits = b.get(2) as usize;
            if num_fill_bits == 0 {
                let len = b.get(4);
                num_fill_bits = b.get(len) as usize + 1;
            }
            if b.count() + num_fill_bits > self.num_saved_bits {
                return Err(WmaProError::InvalidData);
            }
            b.skip(num_fill_bits);
        }
        if b.bit() == 1 {
            return Err(WmaProError::Unsupported); // reserved bit
        }
        self.decode_channel_transform(b)?;

        let mut transmit_coeffs = false;
        for &c in &self.cur_channels {
            self.ch[c].transmit_coefs = b.bit() == 1;
            transmit_coeffs |= self.ch[c].transmit_coefs;
        }
        if transmit_coeffs {
            let mut quant_step = (90 * BITS_PER_SAMPLE) >> 4;
            self.transmit_num_vec_coeffs = b.bit() == 1;
            if self.transmit_num_vec_coeffs {
                let num_bits = log2(subframe_len.div_ceil(4)) + 1;
                for &c in &self.cur_channels {
                    let nvc = (b.get(num_bits) as usize) << 2;
                    if nvc > subframe_len {
                        return Err(WmaProError::InvalidData);
                    }
                    self.ch[c].num_vec_coeffs = nvc;
                }
            } else {
                for &c in &self.cur_channels {
                    self.ch[c].num_vec_coeffs = subframe_len;
                }
            }
            let mut step = b.sbits(6);
            quant_step += step;
            if step == -32 || step == 31 {
                let sign = i32::from(step == 31) - 1;
                let mut quant = 0;
                while b.count() + 5 < self.num_saved_bits {
                    step = b.get(5) as i32;
                    if step != 31 {
                        break;
                    }
                    quant += 31;
                }
                quant_step += ((quant + step) ^ sign) - sign;
            }
            if self.cur_channels.len() == 1 {
                self.ch[self.cur_channels[0]].quant_step = quant_step;
            } else {
                let modifier_len = b.get(3);
                for &c in &self.cur_channels {
                    self.ch[c].quant_step = quant_step;
                    if b.bit() == 1 {
                        self.ch[c].quant_step += if modifier_len != 0 {
                            b.get(modifier_len) as i32 + 1
                        } else {
                            1
                        };
                    }
                }
            }
            self.decode_scale_factors(b)?;
        }

        for &c in &self.cur_channels.clone() {
            if self.ch[c].transmit_coefs && b.count() < self.num_saved_bits {
                self.decode_coeffs(b, c)?;
            } else {
                let base = self.ch[c].coeffs;
                self.ch[c].out[base..base + subframe_len].fill(0.0);
            }
        }

        if transmit_coeffs {
            self.inverse_channel_transform();
            let sfb = self.sfb_offsets[self.table_idx];
            for &c in &self.cur_channels {
                let ch = &mut self.ch[c];
                let sf = ch.saved_scale_factors[ch.sf_cur];
                for bnd in 0..self.num_bands {
                    let end = sfb[bnd + 1].min(subframe_len);
                    let exp =
                        ch.quant_step - (ch.max_scale_factor - sf[bnd]) * ch.scale_factor_step;
                    let quant = 10f64.powf(f64::from(exp) / 20.0) as f32;
                    for k in sfb[bnd]..end {
                        self.tmp[k] = ch.out[ch.coeffs + k] * quant;
                    }
                }
                let base = ch.coeffs;
                imdct_half(
                    &self.tmp[..subframe_len],
                    &mut ch.out[base..base + subframe_len],
                );
            }
        }

        self.window();
        for &c in &self.cur_channels {
            let ch = &mut self.ch[c];
            if ch.cur_subframe >= ch.num_subframes {
                return Err(WmaProError::InvalidData);
            }
            ch.cur_subframe += 1;
        }
        Ok(())
    }
}

/// `ff_wma_get_large_val`.
fn wma_get_large_val(b: &mut Bits<'_>) -> u32 {
    let mut n = 8;
    if b.bit() == 1 {
        n += 8;
        if b.bit() == 1 {
            n += 8;
            if b.bit() == 1 {
                n += 7;
            }
        }
    }
    b.get(n)
}

/// `ff_wma_run_level_decode` (version 1, float coefficients).
#[allow(clippy::too_many_arguments)]
fn wma_run_level_decode(
    b: &mut Bits<'_>,
    vlc: &Vlc,
    level_table: &[f32],
    run_table: &[u16],
    coeffs: &mut [f32],
    mut offset: usize,
    num_coefs: usize,
    frame_len_bits: u32,
) -> R<()> {
    let mask = coeffs.len() - 1;
    while offset < num_coefs {
        let code = vlc.read(b)?;
        if code > 1 {
            let code = code as usize;
            offset += run_table[code] as usize;
            let l = level_table[code];
            coeffs[offset & mask] = if b.bit() == 1 { l } else { -l };
        } else if code == 1 {
            break;
        } else {
            let level = wma_get_large_val(b) as f32;
            if b.bit() == 1 {
                if b.bit() == 1 {
                    if b.bit() == 1 {
                        return Err(WmaProError::InvalidData);
                    }
                    offset += b.get(frame_len_bits) as usize + 4;
                } else {
                    offset += b.get(2) as usize + 1;
                }
            }
            coeffs[offset & mask] = if b.bit() == 1 { level } else { -level };
        }
        offset += 1;
    }
    if offset > num_coefs {
        return Err(WmaProError::InvalidData);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_codes_are_complete_and_decodable() {
        let tb = tables();
        // Kraft sums of complete prefix codes are exactly 1.
        for (name, v) in [
            ("sf", &tb.sf),
            ("sf_rl", &tb.sf_rl),
            ("coef0", &tb.coef[0]),
            ("coef1", &tb.coef[1]),
            ("vec4", &tb.vec4),
            ("vec2", &tb.vec2),
            ("vec1", &tb.vec1),
        ] {
            let kraft: f64 = v
                .by_len
                .iter()
                .enumerate()
                .map(|(l, e)| e.len() as f64 / 2f64.powi(l as i32))
                .sum();
            assert!((kraft - 1.0).abs() < 1e-9, "{name}: {kraft}");
        }
        // The first entry of each table is the all-zero code of its length.
        let len = t::VEC1_TABLE[0][1];
        let data = [0u8; 4];
        let mut b = Bits::new(&data, 32);
        assert_eq!(
            tb.vec1.read(&mut b).unwrap(),
            i32::from(t::VEC1_TABLE[0][0])
        );
        assert_eq!(b.count(), len as usize);
    }

    #[test]
    fn band_layout_matches_ffmpeg_rules() {
        let d = WmaProDecoder::new(2, 48000).unwrap();
        for i in 0..BLOCK_SIZES {
            let n = d.num_sfb[i];
            assert!(n > 0 && n < MAX_BANDS);
            assert_eq!(d.sfb_offsets[i][n], SAMPLES_PER_FRAME >> i);
            for b in 0..n {
                assert!(d.sfb_offsets[i][b] < d.sfb_offsets[i][b + 1]);
                assert_eq!(d.sfb_offsets[i][b] % 4, 0);
            }
        }
    }

    #[test]
    fn imdct_matches_ffmpeg_naive_inverse_mdct() {
        let n = 128;
        let mut x = vec![0.0f32; n];
        x[3] = 32768.0 * (n as f32 / 2.0);
        let mut y = vec![0.0f32; n];
        imdct_half(&x, &mut y);
        for (o, v) in y.iter().enumerate() {
            // FFmpeg's naive av_tx inverse MDCT (`ff_tx_mdct_naive_inv`):
            // first half cos((2j+1) pi (2n - 2i - 1) / 4n), second half
            // -cos((2j+1) pi (3n + 2i + 1) / 4n), here for j = 3.
            let (j, nn) = (3.0f64, n as f64);
            let ph = std::f64::consts::PI / (4.0 * nn);
            let want = if o < n / 2 {
                ((2.0 * j + 1.0) * ph * (2.0 * nn - 2.0 * o as f64 - 1.0)).cos()
            } else {
                let i = (o - n / 2) as f64;
                -((2.0 * j + 1.0) * ph * (3.0 * nn + 2.0 * i + 1.0)).cos()
            };
            assert!((f64::from(*v) - want).abs() < 1e-4, "{o}: {v} vs {want}");
        }
    }

    #[test]
    fn garbage_frames_fail_cleanly_and_output_silence() {
        let mut d = WmaProDecoder::new(2, 48000).unwrap();
        let mut seed = 0x1234_5678u32;
        let mut errors = 0;
        for _ in 0..200 {
            let data: Vec<u8> = (0..256)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    seed as u8
                })
                .collect();
            let mut out = Vec::new();
            if d.decode_frame(&data, 2048, &mut out).is_err() {
                errors += 1;
            }
            assert_eq!(out.len(), SAMPLES_PER_FRAME * 2);
            assert!(out.iter().all(|s| s.is_finite()));
        }
        assert!(errors > 0);
    }
}
