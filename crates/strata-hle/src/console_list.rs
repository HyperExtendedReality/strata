//! Clean-room Xbox Series X|S (`D3D12.X`) inline command-list buffer.
//!
//! Console titles' inlined `ID3D12GraphicsCommandList` fast paths write
//! command dwords straight into a linear buffer the object tracks at fixed
//! offsets: `+0x18` write cursor, `+0x28` buffer start, `+0x30` buffer end.
//! Strata owns that buffer ([`InlineBuffer`]) and its pointers
//! ([`InlineRecordSlots`], placed at `+0x18` of every Strata COM object so
//! no other field sits where a title writes), and drains what the title
//! wrote into the list's [`ThinCommandList`] in order with its vtable calls.
//!
//! What is decoded is limited to what can be justified from public
//! documentation: the stream is read as AMD PM4 (type-2 filler, type-3
//! packets framed by their header's count), and only these type-3 opcodes
//! (AMD's public GFX10 packet definitions) are given meaning:
//!
//! * `NOP` (0x10): skipped.
//! * `NUM_INSTANCES` (0x2F): instance count for following auto draws.
//! * `DRAW_INDEX_AUTO` (0x2D): a non-indexed draw of `count` vertices.
//! * `DISPATCH_DIRECT` (0x15): a dispatch of `x, y, z` groups, or threads
//!   when the initiator's `USE_THREAD_DIMENSIONS` (bit 5) is set.
//!
//! Every other packet is counted and reported by opcode, never guessed at,
//! and a dword that is not PM4 framing stops the drain (the rest of the
//! buffer is reported as undecoded).

use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering};

/// Size of the inline buffer Strata hands each command list (256 KiB).
pub const INLINE_BUFFER_DWORDS: usize = 64 * 1024;

pub const PKT3_NOP: u8 = 0x10;
pub const PKT3_DISPATCH_DIRECT: u8 = 0x15;
pub const PKT3_DRAW_INDEX_AUTO: u8 = 0x2D;
pub const PKT3_NUM_INSTANCES: u8 = 0x2F;

/// The inline-writer fields at object offsets `+0x18..+0x38`.
#[repr(C)]
#[derive(Debug, Default)]
pub struct InlineRecordSlots {
    /// `+0x18`: next dword the title writes.
    pub write_ptr: AtomicPtr<u32>,
    /// `+0x20`: not given a meaning; kept so `+0x28` stays aligned.
    pub reserved: AtomicUsize,
    /// `+0x28`: start of the inline buffer.
    pub buf_start: AtomicPtr<u32>,
    /// `+0x30`: exclusive end of the inline buffer.
    pub buf_end: AtomicPtr<u32>,
}

impl InlineRecordSlots {
    /// Points the slots at `buf` with an empty stream.
    pub fn attach(&self, buf: &InlineBuffer) {
        let start = buf.start();
        self.buf_start.store(start, Ordering::Release);
        self.buf_end
            .store(start.wrapping_add(buf.len()), Ordering::Release);
        self.write_ptr.store(start, Ordering::Release);
    }

    /// Dwords the title has written, clamped to the buffer.
    pub fn written(&self) -> usize {
        let start = self.buf_start.load(Ordering::Acquire);
        let end = self.buf_end.load(Ordering::Acquire);
        let cur = self.write_ptr.load(Ordering::Acquire);
        if start.is_null() || cur < start {
            return 0;
        }
        (cur.min(end) as usize - start as usize) / 4
    }
}

/// The inline buffer a command list owns (interior-mutable: the title's
/// code writes it through the `+0x18` cursor).
pub struct InlineBuffer {
    words: Box<[AtomicU32]>,
    /// Dwords already drained into the thin list.
    pub read: AtomicUsize,
    /// `NUM_INSTANCES` state.
    pub instances: AtomicU32,
    /// Type-3 packets with no decoding here, and non-PM4 stops.
    pub undecoded: AtomicU32,
}

impl Default for InlineBuffer {
    fn default() -> Self {
        Self::new(INLINE_BUFFER_DWORDS)
    }
}

impl InlineBuffer {
    pub fn new(dwords: usize) -> Self {
        Self {
            words: (0..dwords).map(|_| AtomicU32::new(0)).collect(),
            read: AtomicUsize::new(0),
            instances: AtomicU32::new(1),
            undecoded: AtomicU32::new(0),
        }
    }

    pub fn start(&self) -> *mut u32 {
        self.words.as_ptr() as *mut u32
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Forgets everything written (`Reset`).
    pub fn reset(&self, slots: &InlineRecordSlots) {
        slots.attach(self);
        self.read.store(0, Ordering::Release);
        self.instances.store(1, Ordering::Release);
    }

    /// Decodes the dwords written since the last drain.
    pub fn drain(&self, slots: &InlineRecordSlots) -> Vec<InlineOp> {
        let end = slots.written().min(self.words.len());
        let begin = self.read.load(Ordering::Acquire).min(end);
        let words: Vec<u32> = self.words[begin..end]
            .iter()
            .map(|w| w.load(Ordering::Relaxed))
            .collect();
        let mut instances = self.instances.load(Ordering::Relaxed);
        let decoded = decode_pm4(&words, &mut instances);
        self.instances.store(instances, Ordering::Relaxed);
        self.undecoded
            .fetch_add(decoded.undecoded, Ordering::Relaxed);
        // A stop leaves the cursor where it is: nothing past it is trusted.
        self.read.store(begin + decoded.consumed, Ordering::Release);
        decoded.ops
    }
}

/// A command decoded from the inline stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InlineOp {
    Draw {
        vertex_count: u32,
        instance_count: u32,
    },
    Dispatch {
        x: u32,
        y: u32,
        z: u32,
        is_thread_count: bool,
    },
}

/// Result of [`decode_pm4`].
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Decoded {
    pub ops: Vec<InlineOp>,
    /// Dwords consumed (whole packets only).
    pub consumed: usize,
    /// Packets with no meaning here, plus one for a non-PM4 stop.
    pub undecoded: u32,
    /// Opcodes of the undecoded type-3 packets.
    pub unknown_opcodes: Vec<u8>,
}

/// Decodes whole PM4 packets from `words`. A truncated final packet is
/// left unconsumed (the title may still be writing it).
pub fn decode_pm4(words: &[u32], instances: &mut u32) -> Decoded {
    let mut out = Decoded::default();
    let mut pc = 0usize;
    while pc < words.len() {
        let header = words[pc];
        match header >> 30 {
            // Type-2: a single filler dword.
            2 => {
                pc += 1;
            }
            3 => {
                let body = ((header >> 16) & 0x3fff) as usize + 1;
                if pc + 1 + body > words.len() {
                    break;
                }
                let opcode = ((header >> 8) & 0xff) as u8;
                let b = &words[pc + 1..pc + 1 + body];
                match opcode {
                    PKT3_NOP => {}
                    PKT3_NUM_INSTANCES => *instances = b[0].max(1),
                    PKT3_DRAW_INDEX_AUTO if body >= 2 => out.ops.push(InlineOp::Draw {
                        vertex_count: b[0],
                        instance_count: *instances,
                    }),
                    PKT3_DISPATCH_DIRECT if body >= 4 => out.ops.push(InlineOp::Dispatch {
                        x: b[0],
                        y: b[1],
                        z: b[2],
                        is_thread_count: b[3] & 0x20 != 0,
                    }),
                    other => {
                        out.undecoded += 1;
                        out.unknown_opcodes.push(other);
                    }
                }
                pc += 1 + body;
            }
            // Type-0/1 register writes, or not PM4: stop rather than guess.
            _ => {
                out.undecoded += 1;
                break;
            }
        }
        out.consumed = pc;
    }
    out
}

/// A PM4 type-3 header for `opcode` with `body` dwords.
pub fn pkt3(opcode: u8, body: usize) -> u32 {
    (3 << 30) | (((body as u32 - 1) & 0x3fff) << 16) | (u32::from(opcode) << 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_justified_pm4_packets_and_reports_the_rest() {
        let words = [
            pkt3(PKT3_NOP, 2),
            0,
            0,
            0x8000_0000, // type-2 filler
            pkt3(PKT3_NUM_INSTANCES, 1),
            4,
            pkt3(PKT3_DRAW_INDEX_AUTO, 2),
            3,
            2,
            pkt3(0x76, 3), // SET_SH_REG: no meaning given here
            0,
            0,
            0,
            pkt3(PKT3_DISPATCH_DIRECT, 4),
            64,
            1,
            1,
            0x21,
            pkt3(PKT3_DRAW_INDEX_AUTO, 2), // truncated: the title is mid-write
            6,
        ];
        let mut instances = 1;
        let d = decode_pm4(&words, &mut instances);
        assert_eq!(
            d.ops,
            [
                InlineOp::Draw {
                    vertex_count: 3,
                    instance_count: 4
                },
                InlineOp::Dispatch {
                    x: 64,
                    y: 1,
                    z: 1,
                    is_thread_count: true
                },
            ]
        );
        assert_eq!(d.unknown_opcodes, [0x76]);
        assert_eq!(d.consumed, words.len() - 2);
    }

    #[test]
    fn a_non_pm4_dword_stops_the_drain() {
        let mut instances = 1;
        let d = decode_pm4(&[0x1234_5678, pkt3(PKT3_NOP, 1), 0], &mut instances);
        assert!(d.ops.is_empty());
        assert_eq!((d.consumed, d.undecoded), (0, 1));
    }

    #[test]
    fn drains_incrementally_through_the_write_cursor() {
        let buf = InlineBuffer::new(64);
        let slots = InlineRecordSlots::default();
        buf.reset(&slots);
        let write = |words: &[u32]| unsafe {
            let p = slots.write_ptr.load(Ordering::Acquire);
            p.copy_from_nonoverlapping(words.as_ptr(), words.len());
            slots.write_ptr.store(p.add(words.len()), Ordering::Release);
        };
        write(&[pkt3(PKT3_DRAW_INDEX_AUTO, 2), 3, 2]);
        assert_eq!(buf.drain(&slots).len(), 1);
        assert!(buf.drain(&slots).is_empty(), "nothing new");
        write(&[pkt3(PKT3_DISPATCH_DIRECT, 4), 1, 1, 1, 1]);
        assert_eq!(buf.drain(&slots).len(), 1);
        buf.reset(&slots);
        assert_eq!(slots.written(), 0);
    }
}
