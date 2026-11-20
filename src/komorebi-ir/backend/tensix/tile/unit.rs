//! The unit program (today's `komorebi.dsl.tile.unit`): for each compute CPU (unpack,
//! math, pack) a loop nest of coprocessor instructions with named registers, the Dst
//! rows they walk and the sync points between the CPUs, with a cost a tile. Rules are
//! passes over it (`Pass`): kind "algebra", "layout" or "schedule"; checks of the
//! hazard rules run after them. Ported as it is: the operation table, the words'
//! encoding, the listing (`UnitProgram::dump`) and the cost model with its probes.
//! Wormhole's run of the words (`wormhole_run`, `chip_words`) comes with the forms.

use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use super::py;

/// The compute CPUs, in the listing's order.
pub const CPUS: [&str; 3] = ["unpack", "math", "pack"];
/// The math CPU's index in `CPUS`.
pub const MATH: usize = 1;
/// The kinds of a pass.
pub const KINDS: [&str; 3] = ["algebra", "layout", "schedule"];
/// Fixed vector-unit registers: 0.0, 1.0, -1.0; 12 to 14 are set by SFPCONFIG.
pub const FIXED: [(&str, u32); 3] = [("zero", 9), ("one", 10), ("neg_one", 11)];
/// The registers SFPCONFIG sets.
pub const PROGRAMMABLE: [u32; 3] = [12, 13, 14];
/// LReg 11 as a fourth constant register: set at the start of a unit's body and set
/// back to -1.0 at its end (rule constant_registers, the step form).
pub const SAVED: u32 = 11;
/// LReg 0 .. 7 hold values.
pub const WORK: u32 = 8;
/// Entries of one thread's replay buffer.
pub const REPLAY_ENTRIES: u32 = 32;
/// The running clock, a card's busy AICLK (read with the session open).
pub const AICLK_MHZ: u32 = 1350;
/// SFPSHFT2 modes that take 2 cycles and stall the unit one (VectorUnit table).
pub const SHFT2_SLOW: [i64; 3] = [2, 3, 4];
/// The words that add into their Dst block (the moves overwrite it: no wait).
pub const DST_READERS: [&str; 3] = ["MVMUL", "ELWMUL", "ELWADD"];
/// SFPIADD's immediate form reads only lreg_c.
pub const IADD_IMM: i64 = 1;
/// SFPIADD mod1 bit 1: lreg_c - lreg_dest.
pub const IADD_SUB: i64 = 2;
/// SFPIADD mod1 bit 2: no flags.
pub const IADD_NO_CC: i64 = 4;
/// SFPCONFIG mod1: MOD1_IMM16_IS_VALUE.
pub const CONFIG_IMM: i64 = 1;
/// SFPSHFT mod1 5: lreg_c shifted by the immediate.
pub const SHIFT_IMM_VC: i64 = 5;

/// Cycles (AICLK) of the math CPU's vector code: one-core Blackhole measurements
/// (experiments/core/tile_probes/README.md, section 8) or, where marked, the ISA's
/// word. Each a name, its cycles and the source the listing prints.
pub const PRICES: &[(&str, f64, &str)] = &[
    (
        "word",
        1.0,
        "a ready word pushed: 1 cycle (probe 8: 1.128 in a loop of 16)",
    ),
    (
        "loop",
        2.0,
        "a pass of a loop the core runs: N + 2 cycles (probe 12)",
    ),
    // the band's own loop and tests: hidden behind the replayed or pushed words (the
    // add band at 32 rows 93.5 on the card, its whole tile 93.0; the norm's 682, 689.5)
    (
        "band_loop",
        0.0,
        "a pass of a band's loop: hidden (tile_rules: the band)",
    ),
    (
        "guard",
        0.0,
        "a band's test of the live rows: hidden (tile_rules: the band)",
    ),
    ("replayed", 1.004, "a replayed word: 1.004 cycles (probe 8)"),
    (
        "mad_stall",
        1.0,
        "a MAD result read next: 1 stall (probe 9: 4.03 against 3.04)",
    ),
    // read by a store: no stall (experiments/language/tile_rules: add, scale, SiLU, on
    // the card)
    (
        "shft2_stall",
        1.0,
        "SFPSHFT2 lane modes stall the unit 1 cycle (stated)",
    ),
    (
        "built_word",
        4.14,
        "a word built at run time: 4.14 cycles (probe 8)",
    ),
    (
        "table_word",
        9.37,
        "a word loaded from an L1 table: 9.37 cycles (probe 8)",
    ),
    (
        "macro_row",
        1.046,
        "one SFPLOADMACRO a row: 1.046 cycles (probe 9)",
    ),
    (
        "swap_stall",
        1.0,
        "SFPSWAP: the unit takes only SFPNOP the next cycle (stated)",
    ),
    // a wide row's tile copied into Dst a pass (125.8 measured 2026-11-18, card 1)
    (
        "dst_copy",
        126.0,
        "a float32 tile unpacked into Dst: 126 cycles (experiments/language/tile_cost)",
    ),
];

/// The matrix unit (probe 11: Dst fp32, bf16 sources; one word a phase of fidelity).
pub const MATRIX_PRICES: &[(&str, f64, &str)] = &[
    (
        "matrix_word",
        1.0,
        "MVMUL, ELWMUL, ELWADD, MOVA2D, MOVB2D: 1 cycle (probe 11)",
    ),
    (
        "dst_raw",
        6.0,
        "a word adding into a Dst block waits 6 cycles after its last write",
    ),
    (
        "matrix_sync",
        8.0,
        "one MVMUL pushed alone: 8 cycles push to sync (probe 11)",
    ),
];

/// A price by name: its cycles and its source.
fn price(table: &[(&'static str, f64, &'static str)], name: &str) -> (f64, &'static str) {
    table
        .iter()
        .find(|(n, _, _)| *n == name)
        .map_or((0.0, ""), |(_, c, s)| (*c, *s))
}

/// The class of an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The 2-cycle multiply-add unit.
    Mad,
    /// A simple vector operation.
    Simple,
    /// A load into a register.
    Load,
    /// A store from a register.
    Store,
    /// A change of the lanes' flags.
    Flags,
    /// A step of the register-window counters.
    Rwc,
    /// The replay buffer.
    Replay,
    /// A wait.
    Stall,
    /// A matrix-unit word.
    Matrix,
    /// A load macro.
    Macro,
}

impl Kind {
    /// Its name in today's table (`mad`, `simple`, ...).
    pub fn name(self) -> &'static str {
        match self {
            Kind::Mad => "mad",
            Kind::Simple => "simple",
            Kind::Load => "load",
            Kind::Store => "store",
            Kind::Flags => "flags",
            Kind::Rwc => "rwc",
            Kind::Replay => "replay",
            Kind::Stall => "stall",
            Kind::Matrix => "matrix",
            Kind::Macro => "macro",
        }
    }
}

/// One coprocessor instruction: its opcode, its macro's arguments (name, width in
/// bits) from the high field down to bit `low`, the register arguments it reads and
/// writes, and its class; `implied`: LRegs it reads that no field names.
#[derive(Debug)]
pub struct Op {
    /// Its name.
    pub name: &'static str,
    /// Its opcode, the word's top byte.
    pub code: u32,
    /// Its fields, high to low.
    pub args: &'static [(&'static str, u32)],
    /// The register fields it reads.
    pub reads: &'static [&'static str],
    /// The register fields it writes.
    pub writes: &'static [&'static str],
    /// Its class.
    pub kind: Kind,
    /// The lowest field's bit.
    pub low: u32,
    /// LRegs it reads that no field names.
    pub implied: &'static [u32],
}

const R: &[(&str, u32)] = &[
    ("imm12_math", 12),
    ("lreg_c", 4),
    ("lreg_dest", 4),
    ("instr_mod1", 4),
];
const MAD: &[(&str, u32)] = &[
    ("lreg_src_a", 4),
    ("lreg_src_b", 4),
    ("lreg_src_c", 4),
    ("lreg_dest", 4),
    ("instr_mod1", 4),
];
const LS: &[(&str, u32)] = &[
    ("lreg_ind", 4),
    ("instr_mod0", 4),
    ("sfpu_addr_mode", 3),
    ("dest_reg_addr", 13),
];
const IMM16: &[(&str, u32)] = &[("imm16_math", 16), ("lreg_dest", 4), ("instr_mod1", 4)];
const SWAP: &[(&str, u32)] = &[
    ("imm12_math", 12),
    ("lreg_src_c", 4),
    ("lreg_dest", 4),
    ("instr_mod1", 4),
];
const MAD_READS: &[&str] = &["lreg_src_a", "lreg_src_b", "lreg_src_c"];

/// An Op with the defaults: no registers, simple, from bit 0, nothing implied.
const fn op(name: &'static str, code: u32, args: &'static [(&'static str, u32)]) -> Op {
    Op {
        name,
        code,
        args,
        reads: &[],
        writes: &[],
        kind: Kind::Simple,
        low: 0,
        implied: &[],
    }
}

/// `op` reading and writing register fields.
const fn rw(
    name: &'static str,
    code: u32,
    args: &'static [(&'static str, u32)],
    reads: &'static [&'static str],
    writes: &'static [&'static str],
    kind: Kind,
) -> Op {
    Op {
        name,
        code,
        args,
        reads,
        writes,
        kind,
        low: 0,
        implied: &[],
    }
}

/// The operation table, in today's order.
pub const OPS: &[Op] = &[
    rw("SFPLOAD", 0x70, LS, &[], &["lreg_ind"], Kind::Load),
    rw("SFPSTORE", 0x72, LS, &["lreg_ind"], &[], Kind::Store),
    rw(
        "SFPLOADI",
        0x71,
        &[("lreg_ind", 4), ("instr_mod0", 4), ("imm16", 16)],
        &[],
        &["lreg_ind"],
        Kind::Load,
    ),
    rw("SFPMAD", 0x84, MAD, MAD_READS, &["lreg_dest"], Kind::Mad),
    rw("SFPADD", 0x85, MAD, MAD_READS, &["lreg_dest"], Kind::Mad),
    rw("SFPMUL", 0x86, MAD, MAD_READS, &["lreg_dest"], Kind::Mad),
    rw(
        "SFPMULI",
        0x74,
        IMM16,
        &["lreg_dest"],
        &["lreg_dest"],
        Kind::Mad,
    ),
    rw(
        "SFPADDI",
        0x75,
        IMM16,
        &["lreg_dest"],
        &["lreg_dest"],
        Kind::Mad,
    ),
    rw("SFPMOV", 0x7C, R, &["lreg_c"], &["lreg_dest"], Kind::Simple),
    rw("SFPNOT", 0x80, R, &["lreg_c"], &["lreg_dest"], Kind::Simple),
    rw(
        "SFPAND",
        0x7E,
        R,
        &["lreg_c", "lreg_dest"],
        &["lreg_dest"],
        Kind::Simple,
    ),
    rw(
        "SFPXOR",
        0x8D,
        R,
        &["lreg_c", "lreg_dest"],
        &["lreg_dest"],
        Kind::Simple,
    ),
    rw(
        "SFPSETMAN",
        0x83,
        R,
        &["lreg_c", "lreg_dest"],
        &["lreg_dest"],
        Kind::Simple,
    ),
    rw("SFPABS", 0x7D, R, &["lreg_c"], &["lreg_dest"], Kind::Simple),
    rw(
        "SFPEXEXP",
        0x77,
        R,
        &["lreg_c"],
        &["lreg_dest"],
        Kind::Simple,
    ),
    rw(
        "SFPARECIP",
        0x99,
        R,
        &["lreg_c"],
        &["lreg_dest"],
        Kind::Simple,
    ),
    rw(
        "SFPSETSGN",
        0x89,
        R,
        &["lreg_c", "lreg_dest"],
        &["lreg_dest"],
        Kind::Simple,
    ),
    rw(
        "SFPSETEXP",
        0x82,
        R,
        &["lreg_c", "lreg_dest"],
        &["lreg_dest"],
        Kind::Simple,
    ),
    rw(
        "SFPIADD",
        0x79,
        R,
        &["lreg_c", "lreg_dest"],
        &["lreg_dest"],
        Kind::Simple,
    ),
    rw("SFPGT", 0x97, R, &["lreg_c", "lreg_dest"], &[], Kind::Flags),
    rw("SFPSETCC", 0x7B, R, &["lreg_c"], &[], Kind::Flags),
    // mod1 1: VC gets the larger, VD the smaller (a VD of 8 and up is not written)
    rw(
        "SFPSWAP",
        0x92,
        SWAP,
        &["lreg_src_c", "lreg_dest"],
        &["lreg_src_c", "lreg_dest"],
        Kind::Simple,
    ),
    // LReg 0-3 (and 4-7): per column, the 4 lane rows across the 4 registers
    op("SFPTRANSP", 0x8C, R),
    rw("SFPENCC", 0x8A, R, &[], &[], Kind::Flags),
    // the lanes' flags pushed onto their stack, and back (Wormhole's compare form)
    rw("SFPPUSHC", 0x87, R, &[], &[], Kind::Flags),
    rw("SFPPOPC", 0x88, R, &[], &[], Kind::Flags),
    rw(
        "SFPCAST",
        0x90,
        &[("lreg_src_c", 4), ("lreg_dest", 4), ("instr_mod1", 4)],
        &["lreg_src_c"],
        &["lreg_dest"],
        Kind::Simple,
    ),
    rw(
        "SFP_STOCH_RND",
        0x8E,
        &[
            ("rnd_mode", 3),
            ("imm8_math", 5),
            ("lreg_src_b", 4),
            ("lreg_src_c", 4),
            ("lreg_dest", 4),
            ("instr_mod1", 4),
        ],
        &["lreg_src_b", "lreg_src_c"],
        &["lreg_dest"],
        Kind::Simple,
    ),
    rw(
        "SFPSHFT",
        0x7A,
        R,
        &["lreg_c"],
        &["lreg_dest"],
        Kind::Simple,
    ),
    rw(
        "SFPSHFT2",
        0x94,
        SWAP,
        &["lreg_src_c"],
        &["lreg_dest"],
        Kind::Simple,
    ),
    rw(
        "SFPDIVP2",
        0x76,
        R,
        &["lreg_c"],
        &["lreg_dest"],
        Kind::Simple,
    ),
    rw("SFPLOADMACRO", 0x93, LS, &[], &[], Kind::Macro),
    op("SFPNOP", 0x8F, &[]),
    // the value it moves is LReg 0's (Imm16's under MOD1_IMM16_IS_VALUE, VD 4 and up)
    Op {
        implied: &[0],
        ..op(
            "SFPCONFIG",
            0x91,
            &[("imm16_math", 16), ("config_dest", 4), ("instr_mod1", 4)],
        )
    },
    Op {
        kind: Kind::Rwc,
        low: 6,
        ..op(
            "INCRWC",
            0x38,
            &[("rwc_cr", 6), ("rwc_d", 4), ("rwc_b", 4), ("rwc_a", 4)],
        )
    },
    Op {
        kind: Kind::Rwc,
        ..op(
            "SETRWC",
            0x37,
            &[
                ("clear_ab_vld", 2),
                ("rwc_cr", 4),
                ("rwc_d", 4),
                ("rwc_b", 4),
                ("rwc_a", 4),
                ("BitMask", 6),
            ],
        )
    },
    Op {
        kind: Kind::Replay,
        ..op(
            "REPLAY",
            0x04,
            &[
                ("start_idx", 10),
                ("len", 10),
                ("execute_while_loading", 3),
                ("load_mode", 1),
            ],
        )
    },
    Op {
        kind: Kind::Stall,
        ..op("STALLWAIT", 0xA2, &[("stall_res", 9), ("wait_res", 15)])
    },
    // Dst[8 rows] += SrcB[8 rows] @ SrcA[16 x 16]; the address mode steps the counters
    Op {
        kind: Kind::Matrix,
        ..op(
            "MVMUL",
            0x26,
            &[
                ("clear_dvalid", 2),
                ("instr_mod19", 3),
                ("addr_mode", 5),
                ("dst", 14),
            ],
        )
    },
];

/// The operation named `name`, if the table has it.
pub fn op_named(name: &str) -> Option<&'static Op> {
    OPS.iter().find(|o| o.name == name)
}

/// A unit program that breaks a hazard rule the checks cannot repair, or a word that
/// cannot be made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitError(pub String);

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The 32-bit instruction word of `name` with `args` (by field name); a value wider
/// than its field is refused (an immediate is never let spill into the opcode).
pub fn encode(name: &str, args: &[(&str, i64)]) -> Result<u32, UnitError> {
    let op = op_named(name).ok_or_else(|| UnitError(format!("no operation {name}")))?;
    let (mut word, mut shift) = (0u64, op.low);
    for (arg, width) in op.args.iter().rev() {
        let value = args
            .iter()
            .find(|(n, _)| n == arg)
            .map(|(_, v)| *v)
            .ok_or_else(|| UnitError(format!("{name}: no {arg}")))?;
        if value < 0 || value >= 1i64 << width {
            return Err(UnitError(format!(
                "{name} {arg}={value} does not fit {width} bits"
            )));
        }
        word |= (value as u64) << shift;
        shift += width;
    }
    Ok((u64::from(op.code) << 24 | word) as u32)
}

/// (name, fields) of a 32-bit instruction word, or none for an opcode not in OPS.
pub fn decode(word: u32) -> Option<(&'static str, Vec<(&'static str, i64)>)> {
    let op = OPS.iter().find(|o| o.code == word >> 24)?;
    let (mut out, mut shift) = (Vec::new(), op.low);
    for (arg, width) in op.args.iter().rev() {
        out.push((*arg, i64::from(word >> shift & ((1u32 << width) - 1))));
        shift += width;
    }
    out.reverse();
    Some((op.name, out))
}

/// A register argument: an LReg number, or a value's name until the layout pass
/// (`L<n>` names LReg n).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Arg {
    /// A number.
    Num(i64),
    /// A name.
    Name(String),
}

impl fmt::Display for Arg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arg::Num(n) => write!(f, "{n}"),
            Arg::Name(s) => f.write_str(s),
        }
    }
}

impl Arg {
    /// Its number; a name is none.
    pub fn num(&self) -> Option<i64> {
        match self {
            Arg::Num(n) => Some(*n),
            Arg::Name(_) => None,
        }
    }
}

/// Instruction `op` with its macro's arguments (a register argument is a value's name
/// until the layout pass, then an LReg number) and what it computes.
#[derive(Clone, Debug)]
pub struct Insn {
    /// Its operation.
    pub op: &'static Op,
    /// Its fields, in the operation's order.
    pub args: Vec<Arg>,
    /// What it computes.
    pub note: String,
    /// Registers it reads and writes beside its fields (a load macro's loaded register
    /// and the register its template names), LReg numbers.
    pub uses: Vec<Arg>,
    /// The C++ of its immediate when the body's caller gives it (a run-time scalar's
    /// half): emitted as TT_<op>, built on the core, its `imm16` field left 0.
    pub runtime: String,
    /// Wormhole's words in its place where it runs this one otherwise.
    pub wormhole: Vec<Insn>,
}

impl Insn {
    /// `name` with exactly its operation's fields; a field that is not a register's
    /// (`lreg_*`) is a number, but SFPCONFIG's `config_dest`, which names a constant
    /// register until the layout pass.
    pub fn new(name: &str, args: &[(&str, Arg)], note: &str) -> Result<Insn, UnitError> {
        let op = op_named(name).ok_or_else(|| UnitError(format!("no operation {name}")))?;
        let mut out = Vec::new();
        for (field, _) in op.args {
            let value = args
                .iter()
                .find(|(n, _)| n == field)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| UnitError(format!("{name}: no {field}")))?;
            let named = field.starts_with("lreg") || *field == "config_dest";
            if !named && value.num().is_none() {
                return Err(UnitError(format!("{name} {field}={value}: not a number")));
            }
            out.push(value);
        }
        if let Some((extra, _)) = args
            .iter()
            .find(|(n, _)| !op.args.iter().any(|(f, _)| f == n))
        {
            return Err(UnitError(format!("{name}: no field {extra}")));
        }
        Ok(Insn {
            op,
            args: out,
            note: note.to_string(),
            uses: Vec::new(),
            runtime: String::new(),
            wormhole: Vec::new(),
        })
    }

    /// Field `name`'s value.
    pub fn arg(&self, name: &str) -> Option<&Arg> {
        self.op
            .args
            .iter()
            .position(|(n, _)| *n == name)
            .and_then(|i| self.args.get(i))
    }

    /// Field `name` set to `value`; false when the operation has no such field.
    pub fn set(&mut self, name: &str, value: Arg) -> bool {
        match self.op.args.iter().position(|(n, _)| *n == name) {
            Some(i) if i < self.args.len() => {
                self.args[i] = value;
                true
            }
            _ => false,
        }
    }

    /// Field `name` as a number (0 when the operation has no such numeric field: the
    /// constructor holds every non-register field as a number).
    fn n(&self, name: &str) -> i64 {
        self.arg(name).and_then(Arg::num).unwrap_or(0)
    }

    /// The register arguments read (value names or LReg numbers), and the LRegs it
    /// reads that no field names, by both names (`L0` before the layout pass, 0).
    pub fn reads(&self) -> Vec<Arg> {
        let mut out: Vec<Arg> = self
            .read_fields()
            .iter()
            .filter_map(|n| self.arg(n).cloned())
            .collect();
        let imm = self.op.name == "SFPCONFIG" && self.n("instr_mod1") & CONFIG_IMM != 0;
        if !(imm && self.n("config_dest") >= 4) {
            for r in self.op.implied {
                out.push(Arg::Name(format!("L{r}")));
                out.push(Arg::Num(i64::from(*r)));
            }
        }
        out.extend(self.uses.iter().cloned());
        out
    }

    /// The names of the register fields read.
    pub fn read_fields(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.op.reads.to_vec();
        let remove = |names: &mut Vec<&'static str>, field: &str| {
            if let Some(i) = names.iter().position(|n| *n == field) {
                names.remove(i);
            }
        };
        let op = self.op.name;
        if op == "SFPIADD" && self.n("instr_mod1") & IADD_IMM != 0 {
            remove(&mut names, "lreg_dest");
        }
        if op == "SFPLOADI" && matches!(self.n("instr_mod0"), 8 | 10) {
            names.push("lreg_ind");
        }
        if matches!(op, "SFPSETSGN" | "SFPSETEXP" | "SFPSETMAN") && self.n("instr_mod1") & 1 != 0 {
            remove(&mut names, "lreg_dest");
        }
        if op == "SFPSHFT" && self.n("instr_mod1") & 5 != 5 {
            let mut shifted = vec!["lreg_dest"];
            if self.n("instr_mod1") & 1 == 0 {
                shifted.extend(names);
            }
            names = shifted;
        }
        names
    }

    /// The register arguments written (SFPSWAP writes only work registers), or a
    /// register's name today cannot read.
    pub fn writes(&self) -> Result<Vec<Arg>, UnitError> {
        let mut out = Vec::new();
        for value in self.op.writes.iter().filter_map(|n| self.arg(n)) {
            if self.op.name != "SFPSWAP" || work_register(value)? {
                out.push(value.clone());
            }
        }
        out.extend(self.uses.iter().cloned());
        Ok(out)
    }

    /// True when it changes the lanes' condition flags.
    pub fn sets_flags(&self) -> bool {
        match self.op.name {
            "SFPIADD" => self.n("instr_mod1") & IADD_NO_CC == 0,
            "SFPGT" => self.n("instr_mod1") & 1 != 0,
            name => matches!(name, "SFPENCC" | "SFPSETCC"),
        }
    }
}

/// True for LReg 0-7, by number, by `L<n>` or by a value's name (not a fixed or
/// programmable constant's); refused where today raises: an empty name, or `L` and
/// no number.
pub fn work_register(value: &Arg) -> Result<bool, UnitError> {
    let refused =
        |name: &str| UnitError(format!("work register: {} names none", py::str_repr(name)));
    match value {
        Arg::Num(n) => Ok(*n < i64::from(WORK)),
        Arg::Name(name) if name.is_empty() => Err(refused(name)),
        Arg::Name(name) => match name.strip_prefix('L').map(str::parse::<i64>) {
            Some(Ok(n)) => Ok(n < i64::from(WORK)),
            Some(Err(_)) => Err(refused(name)),
            None => Ok(!FIXED.iter().any(|(f, _)| f == name) && !name.starts_with('p')),
        },
    }
}

/// A line of the library that is not one coprocessor instruction (a RISC-V write of a
/// configuration register, a library unit call): its C++ text and what it does. Not
/// priced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Call {
    /// Its C++.
    pub text: String,
    /// What it does.
    pub note: String,
}

/// Matrix-unit words issued by a library call (`text`): their (op, Dst block) in issue
/// order, priced by `matrix_cost`; `synced`: a sync follows the last. With `insns` the
/// words are emitted (their instructions) in place of the call; `wormhole` Wormhole's
/// own words in their place there.
#[derive(Clone, Debug)]
pub struct Product {
    /// The library call.
    pub text: String,
    /// The words: (op, Dst block; none for a word on no block, a counter's step).
    pub words: Vec<(String, Option<i64>)>,
    /// What it does.
    pub note: String,
    /// A sync follows the last word.
    pub synced: bool,
    /// The words as instructions, when emitted.
    pub insns: Vec<Insn>,
    /// Wormhole's own words.
    pub wormhole: Vec<Insn>,
}

/// A count the C++ computes from the body's live rows (the band): `text` its
/// expression, `of(rows)` its value for the cost model; `guard`: an `if`, 0 or 1.
#[derive(Clone)]
pub struct Runtime {
    /// Its C++.
    pub text: String,
    /// Its value at a number of live rows.
    pub of: Rc<dyn Fn(i64) -> i64>,
    /// An `if`.
    pub guard: bool,
}

impl fmt::Debug for Runtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Runtime({:?}, guard {})", self.text, self.guard)
    }
}

/// A loop's count.
#[derive(Clone, Debug)]
pub enum Count {
    /// Known at the build.
    Times(i64),
    /// From the live rows.
    Runtime(Runtime),
}

/// The passes of a loop of `count` at `rows` live rows.
pub fn passes(count: &Count, rows: i64) -> i64 {
    match count {
        Count::Times(n) => *n,
        Count::Runtime(r) => (r.of)(rows).max(0),
    }
}

/// `count` passes over `body`; `unroll`: the C++ loop is unrolled.
#[derive(Clone, Debug)]
pub struct Loop {
    /// Its name.
    pub name: String,
    /// Its passes.
    pub count: Count,
    /// Its body.
    pub body: Vec<Item>,
    /// Unrolled in the C++.
    pub unroll: bool,
}

/// A sync point between CPUs: `cpu` waits for (or hands over to) `other`, through the
/// library call `text`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sync {
    /// The CPU that waits or hands over.
    pub cpu: String,
    /// The other.
    pub other: String,
    /// The library call.
    pub text: String,
    /// What it does.
    pub note: String,
}

/// An item of a loop nest.
#[derive(Clone, Debug)]
pub enum Item {
    /// An instruction.
    Insn(Insn),
    /// A library line.
    Call(Call),
    /// Matrix-unit words.
    Product(Product),
    /// A loop.
    Loop(Loop),
    /// A sync point.
    Sync(Sync),
}

/// A rule applied: its kind (algebra, layout, schedule), name and what it did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pass {
    /// Its kind.
    pub kind: String,
    /// Its name.
    pub name: String,
    /// What it did.
    pub done: Vec<String>,
}

/// A hazard rule checked: its id, the rule, and the verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Check {
    /// Its id.
    pub rule: String,
    /// The rule.
    pub text: String,
    /// The verdict: "ok", "repaired: ..." or "refused: ...".
    pub verdict: String,
}

/// A tile product of a unit: its result in Dst tile `dst` (of the caller's), of `a`
/// (unpacked into SrcB) by `w` (into SrcA), one 32 x 32 tile each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitProduct {
    /// Its Dst tile.
    pub dst: i64,
    /// Into SrcB.
    pub a: String,
    /// Into SrcA.
    pub w: String,
}

/// A constant register's value as the listing prints it (its Python `repr`).
#[derive(Clone, Debug, PartialEq)]
pub enum Constant {
    /// A float.
    Float(f64),
    /// An integer.
    Int(i64),
    /// A text (a note or a word's bits).
    Text(String),
}

impl Constant {
    /// Its `repr`.
    pub fn repr(&self) -> String {
        match self {
            Constant::Float(x) => py::float_repr(*x),
            Constant::Int(n) => n.to_string(),
            Constant::Text(s) => py::str_repr(s),
        }
    }
}

/// One term of a cost: what, its count, its cycles (none: not priced) and its source.
#[derive(Clone, Debug, PartialEq)]
pub struct Term {
    /// What is priced.
    pub what: String,
    /// How many.
    pub n: i64,
    /// Its cycles, if priced.
    pub cycles: Option<f64>,
    /// Where its price comes from.
    pub source: String,
}

/// One tile's cost on a CPU: its coprocessor instructions, its cycles (none when no
/// term is priced: today's sum of nothing, printed `0`), and its terms.
#[derive(Clone, Debug, PartialEq)]
pub struct Cost {
    /// Coprocessor instructions.
    pub instructions: i64,
    /// Cycles, rounded to a tenth.
    pub cycles: Option<f64>,
    /// The terms.
    pub terms: Vec<Term>,
}

/// One tile function on one core: per CPU its setup (`init`, once per stream) and its
/// program for one tile (`tile`), loop nests of items.
#[derive(Clone, Debug, Default)]
pub struct UnitProgram {
    /// Its name.
    pub name: String,
    /// Its Dst mode.
    pub dest: String,
    /// Each operand's tensor (none: a product's result).
    pub operands: Vec<Option<String>>,
    /// The setup of each CPU, in `CPUS` order.
    pub init: [Vec<Item>; 3],
    /// The program for one tile of each CPU, in `CPUS` order.
    pub tile: [Vec<Item>; 3],
    /// The math CPU's later parts (`<name>_<part>(dst)`), in order.
    pub parts: Vec<(String, Vec<Item>)>,
    /// The constant registers the setup sets, in order.
    pub constants: Vec<(u32, Constant)>,
    /// The registers it writes and the caller keeps.
    pub clobbers: Vec<String>,
    /// The passes, in order.
    pub passes: Vec<Pass>,
    /// The checks, in order.
    pub checks: Vec<Check>,
    /// The rules not checked.
    pub not_checked: Vec<String>,
    /// The band: the body takes its live rows.
    pub band: bool,
    /// Its run-time scalars' parameters, in argument order.
    pub scalars: Vec<String>,
    /// The tile products the body issues first, in issue order.
    pub products: Vec<UnitProduct>,
    /// Their fidelity.
    pub fidelity: Option<String>,
    /// Its results: result k in Dst tile k of the call.
    pub results: i64,
    /// The operands unpacked with row 0 broadcast to every row.
    pub row_operands: Vec<usize>,
    /// The operands unpacked with column 0 broadcast to every column.
    pub col_operands: Vec<usize>,
    /// A wide row's results: result k's first Dst tile of each chunk.
    pub chunk_results: Vec<i64>,
    /// A wide row's reductions read its resident rows.
    pub chunk_fields_in_sums: bool,
}

impl UnitProgram {
    /// An empty unit program of `name` on a `dest` Dst, as today's constructor gives.
    pub fn new(name: &str, dest: &str) -> UnitProgram {
        UnitProgram {
            name: name.to_string(),
            dest: dest.to_string(),
            results: 1,
            chunk_results: vec![0],
            ..UnitProgram::default()
        }
    }

    /// The cost of one tile on CPU `cpu` (an index of `CPUS`) at `rows` live rows.
    pub fn cost(&self, cpu: usize, rows: i64) -> Result<Cost, UnitError> {
        let mut terms = Vec::new();
        let mut state = State {
            previous: None,
            recording: None,
            buffer: HashMap::new(),
            rows,
        };
        if let Some(items) = self.tile.get(cpu) {
            price_items(items, 1, &mut terms, &mut state)?;
        }
        if CPUS.get(cpu) == Some(&"math") {
            for (_, items) in &self.parts {
                state.previous = None;
                price_items(items, 1, &mut terms, &mut state)?;
            }
        }
        let priced: Vec<&Term> = terms.iter().filter(|t| t.cycles.is_some()).collect();
        let instructions = priced
            .iter()
            .filter(|t| {
                t.what
                    .split_whitespace()
                    .next()
                    .is_some_and(|w| op_named(w).is_some())
            })
            .map(|t| t.n)
            .sum();
        let cycles = (!priced.is_empty())
            .then(|| py::round1(py::fsum(priced.iter().filter_map(|t| t.cycles))));
        Ok(Cost {
            instructions,
            cycles,
            terms,
        })
    }

    /// The unit program as text: setup, per-tile loop nests, passes, checks and the
    /// cost a tile (today's `dump`).
    pub fn dump(&self) -> Result<String, UnitError> {
        let mut lines = vec![format!("unit program {} ({} Dst)", self.name, self.dest)];
        for (r, v) in &self.constants {
            lines.push(format!(
                "  constant LReg{r} = {} (set by the setup)",
                v.repr()
            ));
        }
        if !self.clobbers.is_empty() {
            lines.push(format!("  clobbers {}", self.clobbers.join(", ")));
        }
        for (i, cpu) in CPUS.iter().enumerate() {
            if !self.init[i].is_empty() {
                lines.push(format!("  {cpu} setup:"));
                dump_items(&self.init[i], 4, &mut lines);
            }
            lines.push(format!("  {cpu} per tile:"));
            dump_items(&self.tile[i], 4, &mut lines);
            if *cpu == "math" {
                for (part, items) in &self.parts {
                    lines.push(format!("  math per tile, part {part}:"));
                    dump_items(items, 4, &mut lines);
                }
            }
        }
        for p in &self.passes {
            let done = if p.done.is_empty() {
                "nothing".to_string()
            } else {
                p.done.join("; ")
            };
            lines.push(format!("  pass {} {}: {done}", p.kind, p.name));
        }
        for c in &self.checks {
            lines.push(format!("  check {} ({}): {}", c.rule, c.text, c.verdict));
        }
        for rule in &self.not_checked {
            lines.push(format!("  not checked {rule}"));
        }
        for (i, cpu) in CPUS.iter().enumerate() {
            let cost = self.cost(i, 32)?;
            let cycles = cost.cycles.map_or("0".to_string(), py::float_repr);
            lines.push(format!(
                "  cost {cpu}: {} instructions, {cycles} cycles a tile, and the calls not priced",
                cost.instructions
            ));
            for t in merge(&cost.terms) {
                let c = t
                    .cycles
                    .map_or("-".to_string(), |c| py::float_repr(py::round1(c)));
                lines.push(format!(
                    "    {}: {} x, {c} cycles ({})",
                    t.what, t.n, t.source
                ));
            }
        }
        Ok(lines.join("\n"))
    }
}

/// Each item of a loop nest as the listing's lines, at `depth` spaces.
fn dump_items(items: &[Item], depth: usize, out: &mut Vec<String>) {
    let pad = " ".repeat(depth);
    for item in items {
        match item {
            Item::Loop(l) => {
                let how = if l.unroll { ", unrolled" } else { "" };
                let count = match &l.count {
                    Count::Times(n) => n.to_string(),
                    Count::Runtime(r) => {
                        format!("{} ({})", if r.guard { "if" } else { "band" }, r.text)
                    }
                };
                out.push(format!("{pad}loop {} < {count}{how}:", l.name));
                dump_items(&l.body, depth + 2, out);
            }
            Item::Insn(insn) => {
                let args: Vec<String> = insn
                    .op
                    .args
                    .iter()
                    .zip(&insn.args)
                    .map(|((n, _), v)| format!("{n}={v}"))
                    .collect();
                let note = if insn.note.is_empty() {
                    String::new()
                } else {
                    format!("  ; {}", insn.note)
                };
                out.push(format!("{pad}{} {}{note}", insn.op.name, args.join(" ")));
            }
            Item::Sync(s) => out.push(format!(
                "{pad}sync {} <- {}: {}  ; {}",
                s.cpu, s.other, s.text, s.note
            )),
            Item::Product(p) => out.push(format!(
                "{pad}product {}: {} words  ; {}",
                p.text,
                p.words.len(),
                p.note
            )),
            Item::Call(c) => out.push(format!("{pad}call {}  ; {}", c.text, c.note)),
        }
    }
}

/// The pricing's state: the last instruction issued (a stall follows a MAD result
/// read next), the words being recorded into the replay buffer, and the recorded
/// sequences by start index.
struct State<'u> {
    previous: Option<&'u Insn>,
    recording: Option<(i64, i64)>,
    buffer: HashMap<i64, Vec<&'u Insn>>,
    rows: i64,
}

/// Appends a term per item, `times` passes (today's `_price`).
fn price_items<'u>(
    items: &'u [Item],
    times: i64,
    terms: &mut Vec<Term>,
    state: &mut State<'u>,
) -> Result<(), UnitError> {
    for item in items {
        match item {
            Item::Loop(l) => {
                let count = passes(&l.count, state.rows);
                let runtime = matches!(l.count, Count::Runtime(_));
                if let Count::Runtime(Runtime { guard: true, .. }) = l.count {
                    let (cycles, source) = price(PRICES, "guard");
                    terms.push(term(
                        "band guard",
                        times,
                        Some(times as f64 * cycles),
                        source,
                    ));
                } else if !l.unroll {
                    // a runtime face loop's pass is a loop's; a band's is hidden
                    let band = runtime && l.name == "band";
                    let (cycles, source) = price(PRICES, if band { "band_loop" } else { "loop" });
                    let what = if band { "band loop pass" } else { "loop pass" };
                    terms.push(term(
                        what,
                        times * count,
                        Some((times * count) as f64 * cycles),
                        source,
                    ));
                }
                price_items(&l.body, times * count, terms, state)?;
            }
            Item::Insn(insn) if insn.op.name == "REPLAY" => {
                let start = insn.n("start_idx");
                let len = insn.n("len");
                if insn.n("load_mode") != 0 {
                    // Recording with execution: what follows runs as pushed words.
                    state.recording = Some((start, len));
                    state.buffer.insert(start, Vec::new());
                    issue(insn, times, terms, state, "word")?;
                    continue;
                }
                let played: Vec<&Insn> = state
                    .buffer
                    .get(&start)
                    .map(|b| b.iter().take(len.max(0) as usize).copied().collect())
                    .unwrap_or_default();
                if (played.len() as i64) < len {
                    return Err(UnitError(format!(
                        "REPLAY of {len} plays words never recorded"
                    )));
                }
                for insn in played {
                    issue(insn, times, terms, state, "replayed")?;
                }
            }
            Item::Insn(insn) => {
                issue(insn, times, terms, state, "word")?;
                if let Some((start, count)) = state.recording {
                    let recorded = state.buffer.entry(start).or_default();
                    recorded.push(insn);
                    if recorded.len() as i64 == count {
                        state.recording = None;
                    }
                }
            }
            Item::Call(c) => terms.push(term(&c.text, times, None, "library call, not priced")),
            Item::Product(p) => {
                let cycles = matrix_cost(&p.words, p.synced);
                let source = "matrix_cost of the expanded words (probe 11)";
                terms.push(term(
                    "matrix words",
                    times * p.words.len() as i64,
                    Some(times as f64 * cycles),
                    source,
                ));
                state.previous = None;
            }
            Item::Sync(s) => terms.push(term(
                &format!("sync {}", s.text),
                times,
                None,
                "sync, not priced",
            )),
        }
    }
    Ok(())
}

fn term(what: &str, n: i64, cycles: Option<f64>, source: &str) -> Term {
    Term {
        what: what.to_string(),
        n,
        cycles,
        source: source.to_string(),
    }
}

/// The price of `insn` issued `times` (pushed: "word", or "replayed") and of the stall
/// in front of it, if any (today's `_issue`).
fn issue<'u>(
    insn: &'u Insn,
    times: i64,
    terms: &mut Vec<Term>,
    state: &mut State<'u>,
    how: &str,
) -> Result<(), UnitError> {
    let mut stall = None;
    if let Some(previous) = state.previous {
        let reads = insn.reads();
        if previous.op.kind == Kind::Mad
            && insn.op.name != "SFPSTORE"
            && previous.writes()?.iter().any(|w| reads.contains(w))
        {
            stall = Some("mad_stall");
        }
        if previous.op.name == "SFPSHFT2" && SHFT2_SLOW.contains(&previous.n("instr_mod1")) {
            stall = Some("shft2_stall");
        }
        if previous.op.name == "SFPSWAP" && insn.op.name != "SFPNOP" {
            stall = Some("swap_stall");
        }
    }
    if let Some(stall) = stall {
        let (cycles, source) = price(PRICES, stall);
        terms.push(term(
            &format!("stall ({stall})"),
            times,
            Some(times as f64 * cycles),
            source,
        ));
    }
    let how = if insn.op.name == "SFPLOADMACRO" {
        "macro_row"
    } else {
        how
    };
    let (cycles, source) = price(PRICES, how);
    let name = if how == "word" {
        insn.op.name.to_string()
    } else {
        format!("{} {how}", insn.op.name)
    };
    terms.push(term(&name, times, Some(times as f64 * cycles), source));
    state.previous = Some(insn);
    Ok(())
}

/// Cycles of matrix-unit `words` ((op, Dst block) in issue order): one a cycle, a word
/// adding into a block waits 6 cycles after that block's last write (K words apart:
/// max(1, 6 / K) a word), the last result 8 cycles to a sync.
pub fn matrix_cost(words: &[(String, Option<i64>)], synced: bool) -> f64 {
    let (gap, _) = price(MATRIX_PRICES, "dst_raw");
    let (one, _) = price(MATRIX_PRICES, "matrix_word");
    let mut t = -1.0;
    let mut written: HashMap<Option<i64>, f64> = HashMap::new();
    for (op, block) in words {
        t += one;
        if DST_READERS.contains(&op.as_str()) {
            if let Some(last) = written.get(block) {
                t = f64::max(t, last + gap);
            }
        }
        written.insert(*block, t);
    }
    if words.is_empty() {
        return 0.0;
    }
    t + if synced {
        price(MATRIX_PRICES, "matrix_sync").0
    } else {
        1.0
    }
}

/// Schedule rule: `words` reordered so a word adding into a Dst block comes as late
/// after that block's last write as the order allows (each block's own words keep
/// their order: its sum rounds the same). Blocks are kept in the order they first
/// appear, as today's dict keeps them; the next block is the one written longest ago,
/// then the lowest. Words of no block among others are refused (today's comparison of
/// none with a number raises).
pub fn rotate_blocks(
    words: &[(String, Option<i64>)],
) -> Result<Vec<(String, Option<i64>)>, UnitError> {
    type Queue = std::collections::VecDeque<(String, Option<i64>)>;
    let mut queues: Vec<(Option<i64>, Queue)> = Vec::new();
    for w in words {
        match queues.iter_mut().find(|(b, _)| *b == w.1) {
            Some((_, q)) => q.push_back(w.clone()),
            None => queues.push((w.1, Queue::from([w.clone()]))),
        }
    }
    if queues.len() > 1 && queues.iter().any(|(b, _)| b.is_none()) {
        return Err(UnitError(
            "rotate_blocks: a word of no Dst block among others".to_string(),
        ));
    }
    let mut last: Vec<i64> = vec![-(words.len() as i64); queues.len()];
    let mut order = Vec::new();
    for i in 0..words.len() {
        let chosen = queues
            .iter()
            .enumerate()
            .filter(|(_, (_, q))| !q.is_empty())
            .min_by_key(|(k, (b, _))| (last[*k], *b))
            .map(|(k, _)| k);
        let Some(k) = chosen else { break };
        if let Some(w) = queues[k].1.pop_front() {
            order.push(w);
        }
        last[k] = i as i64;
    }
    Ok(order)
}

/// Terms of the same name summed, in first-seen order.
pub fn merge(terms: &[Term]) -> Vec<Term> {
    let mut out: Vec<Term> = Vec::new();
    for t in terms {
        match out.iter_mut().find(|o| o.what == t.what) {
            Some(o) => {
                o.n += t.n;
                o.cycles = match (o.cycles, t.cycles) {
                    (Some(a), Some(b)) => Some(a + b),
                    _ => None,
                };
            }
            None => out.push(t.clone()),
        }
    }
    out
}

/// Every Insn of a loop nest, loop bodies once, in order.
pub fn insns(items: &[Item]) -> Vec<&Insn> {
    let mut out = Vec::new();
    for item in items {
        match item {
            Item::Loop(l) => out.extend(insns(&l.body)),
            Item::Insn(i) => out.push(i),
            _ => {}
        }
    }
    out
}

/// The straight-line instruction lists of a loop nest (each loop body's own).
pub fn blocks(items: &[Item]) -> Vec<Vec<&Insn>> {
    let mut out = Vec::new();
    let mut run = Vec::new();
    for item in items {
        match item {
            Item::Loop(l) => {
                if !run.is_empty() {
                    out.push(std::mem::take(&mut run));
                }
                out.extend(blocks(&l.body));
            }
            Item::Insn(i) => run.push(i),
            _ => {}
        }
    }
    if !run.is_empty() {
        out.push(run);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(blocks: impl IntoIterator<Item = i64>) -> Vec<(String, Option<i64>)> {
        blocks
            .into_iter()
            .map(|b| ("MVMUL".to_string(), Some(b)))
            .collect()
    }

    #[test]
    fn the_matrix_unit_prices_and_rotation_are_todays() {
        // today's test_matrix_unit_rotation_prices_and_rule, the same numbers
        let n = 24 * 64;
        for (k, want) in [(1, 6.0), (2, 3.0), (3, 2.0), (4, 1.5), (5, 1.2), (6, 1.0)] {
            let cost = matrix_cost(&mv((0..n).map(|i| i % k)), false) / n as f64;
            assert!((cost - want).abs() <= 0.01, "{k}: {cost}");
        }
        assert_eq!(matrix_cost(&mv([0]), true), 8.0);
        let moves = vec![("MOVA2D".to_string(), Some(0)); 4];
        assert_eq!(matrix_cost(&moves, false), 4.0);
        let inner = mv((0..8).flat_map(|b| [b, b]));
        let rotated = rotate_blocks(&inner).expect("rotated");
        let order: Vec<Option<i64>> = rotated.iter().map(|w| w.1).collect();
        assert_eq!(order, (0..8).chain(0..8).map(Some).collect::<Vec<_>>());
        assert_eq!(
            (matrix_cost(&inner, true), matrix_cost(&rotated, true)),
            (63.0, 23.0)
        );
    }

    #[test]
    fn a_word_too_wide_is_refused_and_a_word_round_trips() {
        let args = [("lreg_ind", 0), ("instr_mod0", 0), ("imm16", 1 << 16)];
        let refused = encode("SFPLOADI", &args).expect_err("too wide");
        assert_eq!(refused.0, "SFPLOADI imm16=65536 does not fit 16 bits");
        let word = encode(
            "SFPMUL",
            &[
                ("lreg_src_a", 0),
                ("lreg_src_b", 1),
                ("lreg_src_c", 9),
                ("lreg_dest", 1),
                ("instr_mod1", 0),
            ],
        );
        let (name, fields) = word.ok().and_then(decode).expect("decoded");
        assert_eq!(name, "SFPMUL");
        assert_eq!(fields[2], ("lreg_src_c", 9));
        assert!(decode(0xFF00_0000).is_none());
    }

    #[test]
    fn a_field_the_operation_has_not_is_refused() {
        let a = |n: i64| Arg::Num(n);
        assert!(Insn::new("SFPNOP", &[("lreg_c", a(0))], "").is_err());
        assert!(Insn::new("SETRWC", &[("rwc_cr", a(0))], "").is_err());
        let named = Insn::new(
            "SFPMOV",
            &[
                ("imm12_math", a(0)),
                ("lreg_c", Arg::Name("x".into())),
                ("lreg_dest", Arg::Name("y".into())),
                ("instr_mod1", a(0)),
            ],
            "a copy",
        );
        let named = named.expect("registers may be names");
        assert_eq!(named.reads(), [Arg::Name("x".into())]);
        assert_eq!(named.writes().expect("written"), [Arg::Name("y".into())]);
        // where today's int() raises, the port refuses
        let swap = |lreg: &str| {
            let args = [
                ("imm12_math", a(0)),
                ("lreg_src_c", Arg::Name(lreg.into())),
                ("lreg_dest", a(1)),
                ("instr_mod1", a(0)),
            ];
            Insn::new("SFPSWAP", &args, "").expect("a swap").writes()
        };
        assert_eq!(swap("L3").expect("written"), [Arg::Name("L3".into()), a(1)]);
        let refused = swap("Lx").expect_err("no number");
        assert_eq!(refused.0, "work register: 'Lx' names none");
        assert!(swap("").is_err());
    }
}
