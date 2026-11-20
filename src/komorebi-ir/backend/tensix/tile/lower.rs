//! A traced tile expression over a stream of tiles to its unit program (today's
//! `komorebi.dsl.tile.lower`): the chain's vector function is written as vector-unit
//! instructions on named values, the rules run as passes (algebra, layout, schedule),
//! then the hazard checks (`checks`); and the stream form's entry to the chain
//! compiler (`stream_chain`).
//!
//! Ported: the whole-tile stream form of one result on `+`, `-`, `*`, negation,
//! `abs`, multiply-add, `min` / `max`, selects, constants, run-time scalars and every
//! vector function (`FUNCTIONS`), with every pass of that path and the load macro; tile
//! products, alone or before the vector code, at the fidelity given; operands spread
//! from a tile's row 0 (`.row()`) and per-row operands (`.col()`: a 4-row block's even
//! and odd vectors, with `drop_copies`, `common_values`, the per-row pre-pass's
//! estimate and the schedule's `interleave`). Refused by name (K964) where it would
//! act: `flash_exp2`, several results, a per-row pre-pass cheaper than the rows, and
//! the band. Not on this path, so not ported: `allocate`'s high registers.

use std::collections::HashMap;

use super::chain::{shape, Chain, ChainError, Compiler, Source};
use super::checks::{
    check_dst_window, check_mad_readers, hazard_checks, named_register, named_spans, NEG_ONE_BF16,
    STALL_SFPU, TILE, WAIT_MATH,
};
use super::ir::{Attr, KernelIR, NodeId};
use super::matrix::{tile_products, wormhole_products};
use super::py;
use super::r#macro::{self, NoFit, Plan};
use super::schedule::interleave;
use super::unit::{
    insns, op_named, Arg, Call, Check, Constant, Count, Insn, Item, Kind, Loop, Pass, Product,
    Sync, UnitError, UnitProduct, UnitProgram, FIXED, IADD_IMM, IADD_NO_CC, IADD_SUB, MATH, PACK,
    PROGRAMMABLE, REPLAY_ENTRIES, SAVED, SHIFT_IMM_VC, UNPACK, WORK,
};
use super::vector::{FnResult, VId, VValue, Vs};

/// Dst access through ADDR_MOD_7 (no increments), the format the registers hold.
pub const ADDR_MOD: i64 = 7;
/// ADDR_MOD_6 steps Dst by 2 (one row group) after a load macro.
pub const MACRO_ADDR_MOD: i64 = 6;
/// Row groups of a face.
pub const ROWS: i64 = 8;
/// Faces of a tile.
pub const FACES: i64 = 4;
/// Bands of 4 rows a face (rule band: a body runs the bands of live rows).
pub const BANDS: i64 = 4;
/// SFPLOAD / SFPSTORE's instr_mod0 by Dst: its own format (fp32), FP16A (a 16-bit Dst).
pub const DEST_MODES: [(&str, i64); 2] = [("fp32", 0), ("fp16a", 1)];
/// Operand tiles a whole-tile stream body reads: Dst tiles 0 .. 3 (half the fp32 Dst).
pub const OPERANDS: usize = 4;
/// 4-row blocks of a tile (the per-row form's rows).
pub const BLOCKS: i64 = FACES * ROWS / 2;
/// SETRWC's mask and counter-reset bit of Dst.
pub const SET_D: i64 = 0x4;
/// SETRWC's counter-reset bit of Dst.
pub const CR_D: i64 = 0x4;
/// The constant registers in the order the layout pass fills them: 12 to 14, then 11
/// (set and restored by the body itself, rule constant_registers' fourth register).
pub const CONSTANT_REGISTERS: [u32; 4] = [PROGRAMMABLE[0], PROGRAMMABLE[1], PROGRAMMABLE[2], SAVED];
/// The name of the fourth constant register until the layout pass.
pub const SAVED_NAME: &str = "p3";
/// Address mode the row's last store steps Dst with (rule store_step), when no load
/// macro takes ADDR_MOD_6.
pub const STEP_ADDR_MOD: i64 = MACRO_ADDR_MOD;
/// The format of a stream body's tensors on a fp32 Dst: fp32 or the B family.
pub const TILE_DTYPES: [&str; 5] = ["float32", "bfloat16", "bfp8_b", "bfp4_b", "bfp2_b"];
/// The A family's formats, read into a 16-bit Dst.
pub const A_DTYPES: [&str; 4] = ["float16", "bfp8", "bfp4", "bfp2"];
/// SFPENCC's (imm, mod1): the condition on, every lane's flag set.
pub const CC_RESET: (i64, i64) = (3, 10);
/// +inf's high half: SFPMULI's immediate (a bf16).
pub const INF_BF16: i64 = 0x7F80;
/// The vector functions today's unit program builds.
pub const FUNCTIONS: [&str; 8] = [
    "exp", "sigmoid", "rsqrt", "recip", "sqrt", "tanh", "erf", "log",
];
/// The rules a compute body must obey that no check here covers yet.
pub const NOT_CHECKED: [&str; 8] = [
    "interleave: two independent MAD chains interleaved (a speed rule)",
    "load-macro: SFPLOADMACRO delays counted in instructions, drained by SFPNOPs",
    "exp-range: the biased exp argument kept inside its range",
    "indirect: indirect register forms",
    "stop-word: the stop word polled in every wait and unit loop (the caller's)",
    "dst-tiles: Dst tiles per acquire (the caller's index)",
    "placement: code placement and the kernel buffer's size",
    "src-banks: an unpack of more faces than Src has banks (2) waits for a release",
];

fn plan_error(message: String) -> ChainError {
    ChainError::Plan(message, None)
}

fn name(s: &str) -> Arg {
    Arg::Name(s.to_string())
}

/// Field `field` of `insn` (0 when it has none: every caller knows the operation).
fn at(insn: &Insn, field: &str) -> Arg {
    insn.arg(field).cloned().unwrap_or(Arg::Num(0))
}

fn is_name(v: &Arg, s: &str) -> bool {
    matches!(v, Arg::Name(n) if n == s)
}

/// Python's `repr` of a list of LReg numbers.
fn listed(values: &[i64]) -> String {
    let items: Vec<String> = values.iter().map(i64::to_string).collect();
    format!("[{}]", items.join(", "))
}

fn insn(op: &str, args: &[(&str, i64)], note: &str) -> Result<Insn, UnitError> {
    let args: Vec<(&str, Arg)> = args.iter().map(|(n, v)| (*n, Arg::Num(*v))).collect();
    Insn::new(op, &args, note)
}

/// The instructions of one row group (4 rows x 8 columns of each face, the vector
/// unit's 32 lanes) on named values, `code` in issue order.
#[derive(Clone, Debug, Default)]
pub struct Body {
    /// The code.
    pub code: Vec<Insn>,
    count: usize,
    /// The programmable constant registers by name (`p<k>`), in order.
    pub prog: Vec<(String, Constant)>,
    /// SFPLOAD / SFPSTORE's format: Dst's own (FP16A on a 16-bit Dst).
    pub mode: i64,
    /// Each run-time scalar's value by its index, held across the rows.
    pub scalars: Vec<(usize, String)>,
    /// The programmable constant registers by the bits they hold.
    bits: Vec<(i64, String)>,
    /// The per-row form's operand reads, when it is the per-row form.
    rows: Option<Rows>,
}

/// The per-row form's operand reads: a per-row operand loaded once and held, the
/// others at the half of the 4-row block the value is of (0 even, 1 odd).
#[derive(Clone, Debug, Default)]
struct Rows {
    per_row: Vec<usize>,
    half: i64,
    shared: Vec<(usize, Arg)>,
}

impl Body {
    fn fresh(&mut self) -> String {
        self.count += 1;
        format!("v{}", self.count)
    }

    /// Appends `name`; a written register argument left out gets a new value, any
    /// other field 0 (a new value is counted for every written field, given or not,
    /// as today's `setdefault` counts it). Returns the value written.
    fn op(&mut self, op: &str, note: &str, args: &[(&str, Arg)]) -> Result<Arg, ChainError> {
        self.word(op, note, args)?
            .ok_or_else(|| ChainError::Unit(format!("{op} writes no value")))
    }

    /// `op` as `op` appends it, a word that may write no value (none returned).
    fn word(
        &mut self,
        op: &str,
        note: &str,
        args: &[(&str, Arg)],
    ) -> Result<Option<Arg>, ChainError> {
        let o = op_named(op).ok_or_else(|| UnitError(format!("no operation {op}")))?;
        let mut all: Vec<(&str, Arg)> = args.to_vec();
        for w in o.writes {
            let v = self.fresh();
            if !all.iter().any(|(n, _)| n == w) {
                all.push((w, Arg::Name(v)));
            }
        }
        for (field, _) in o.args {
            if !all.iter().any(|(n, _)| n == field) {
                all.push((field, Arg::Num(0)));
            }
        }
        let made = Insn::new(op, &all, note)?;
        let written = made.writes()?.into_iter().next();
        self.code.push(made);
        Ok(written)
    }

    /// A programmable constant register holding fp32 `value`, set once by the setup
    /// after the schedule pass hoists it: its name until the layout pass.
    fn register_const(&mut self, value: f64) -> Result<Arg, ChainError> {
        let bits = i64::from((value as f32).to_bits());
        self.register(bits, Constant::Float(value))
    }

    /// A programmable constant register holding the integer `bits`, as
    /// `register_const`; the setup's note is the bits in hex.
    fn register_bits(&mut self, bits: i64) -> Result<Arg, ChainError> {
        self.register(bits, Constant::Text(format!("{bits:#x}")))
    }

    /// The register holding `bits` (written `constant` in the listing): an earlier one
    /// of the same bits, else the next programmable register with its setup.
    fn register(&mut self, bits: i64, constant: Constant) -> Result<Arg, ChainError> {
        if let Some((_, reg)) = self.bits.iter().find(|(b, _)| *b == bits) {
            return Ok(name(reg));
        }
        if self.prog.len() == PROGRAMMABLE.len() {
            return Err(ChainError::Unit(
                "more constants than the 3 programmable registers".into(),
            ));
        }
        let reg = format!("p{}", self.prog.len());
        self.bits.push((bits, reg.clone()));
        let note = match &constant {
            Constant::Text(text) => text.clone(),
            other => other.repr(),
        };
        self.prog.push((reg.clone(), constant));
        self.code.extend(config(&reg, bits, &note)?);
        Ok(name(&reg))
    }

    /// ±(a b) ± c, rounded once, into `into` or a new value.
    #[allow(clippy::too_many_arguments)]
    fn mad_of(
        &mut self,
        a: Arg,
        b: Arg,
        c: Arg,
        note: &str,
        neg_a: bool,
        neg_c: bool,
        into: Option<Arg>,
    ) -> Result<Arg, ChainError> {
        let mode = i64::from(neg_a) | i64::from(neg_c) << 1;
        let mut args = vec![
            ("lreg_src_a", a),
            ("lreg_src_b", b),
            ("lreg_src_c", c),
            ("instr_mod1", Arg::Num(mode)),
        ];
        if let Some(into) = into {
            args.push(("lreg_dest", into));
        }
        self.op("SFPMAD", note, &args)
    }

    /// A flag-setting word: the lanes where it holds run what follows until `end`.
    fn when(&mut self, op: &str, note: &str, args: &[(&str, Arg)]) -> Result<(), ChainError> {
        self.word(op, note, args).map(|_| ())
    }

    /// Every lane on again.
    fn end(&mut self) -> Result<(), ChainError> {
        let (imm, mode) = CC_RESET;
        let args = [
            ("imm12_math", Arg::Num(imm)),
            ("instr_mod1", Arg::Num(mode)),
        ];
        self.word("SFPENCC", "every lane on", &args).map(|_| ())
    }

    fn load(&mut self, address: i64, note: &str) -> Result<Arg, ChainError> {
        let note = if note.is_empty() {
            format!("Dst + {address}")
        } else {
            note.to_string()
        };
        let mode = self.mode;
        self.op(
            "SFPLOAD",
            &note,
            &[
                ("instr_mod0", Arg::Num(mode)),
                ("sfpu_addr_mode", Arg::Num(ADDR_MOD)),
                ("dest_reg_addr", Arg::Num(address)),
            ],
        )
    }

    /// Operand `k` as the per-row form reads it, or none outside that form.
    fn row_operand(&mut self, k: usize) -> Result<Option<Arg>, ChainError> {
        let (per_row, half) = match &self.rows {
            None => return Ok(None),
            Some(r) => (r.per_row.contains(&k), r.half),
        };
        let address = TILE * k as i64;
        if !per_row {
            let note = format!("operand {k}, {}", if half == 0 { "even" } else { "odd" });
            return self.load(address + 2 * half, &note).map(Some);
        }
        let held = self.rows.as_ref().and_then(|r| {
            r.shared
                .iter()
                .find(|(i, _)| *i == k)
                .map(|(_, v)| v.clone())
        });
        if let Some(v) = held {
            return Ok(Some(v));
        }
        let v = self.load(address, &format!("per-row operand {k}"))?;
        if let Some(r) = &mut self.rows {
            r.shared.push((k, v.clone()));
        }
        Ok(Some(v))
    }

    fn store(&mut self, value: Arg, address: i64, note: &str) -> Result<(), ChainError> {
        let args = [
            ("lreg_ind", value),
            ("instr_mod0", Arg::Num(self.mode)),
            ("sfpu_addr_mode", Arg::Num(ADDR_MOD)),
            ("dest_reg_addr", Arg::Num(address)),
        ];
        self.code.push(Insn::new("SFPSTORE", &args, note)?);
        Ok(())
    }

    /// `value` (fp32) into a new value: one SFPLOADI when bf16 or fp16 holds it
    /// exactly, else the low then the high half.
    fn constant(&mut self, value: f64) -> Result<Arg, ChainError> {
        let bits = i64::from((value as f32).to_bits());
        let out = name(&self.fresh());
        let note = py::float_repr(value);
        let word = |b: &mut Body, mode: i64, imm: i64| {
            let args = [
                ("lreg_ind", out.clone()),
                ("instr_mod0", Arg::Num(mode)),
                ("imm16", Arg::Num(imm)),
            ];
            // a new value counted for its written field, as `op` counts it
            b.fresh();
            Insn::new("SFPLOADI", &args, &note).map(|w| b.code.push(w))
        };
        if bits & 0xFFFF == 0 {
            word(self, 0, bits >> 16)?;
            return Ok(out);
        }
        if value.abs() < 65504.0 {
            if let Some(half) = py::half_bits(value as f32) {
                word(self, 1, i64::from(half))?;
                return Ok(out);
            }
        }
        word(self, 2, bits & 0xFFFF)?;
        word(self, 8, bits >> 16)?;
        Ok(out)
    }

    /// Run-time scalar `index`'s value: loaded once before the rows by its two
    /// run-time SFPLOADI (`scalar_loads`) and held.
    fn scalar(&mut self, index: usize) -> String {
        if let Some((_, v)) = self.scalars.iter().find(|(i, _)| *i == index) {
            return v.clone();
        }
        let v = self.fresh();
        self.scalars.push((index, v.clone()));
        v
    }

    fn mad(&mut self, a: Arg, b: Arg, c: Arg) -> Result<Arg, ChainError> {
        let args = [
            ("lreg_src_a", a),
            ("lreg_src_b", b),
            ("lreg_src_c", c),
            ("instr_mod1", Arg::Num(0)),
        ];
        self.op("SFPMAD", "", &args)
    }

    fn mul(&mut self, a: Arg, b: Arg, note: &str) -> Result<Arg, ChainError> {
        let args = [
            ("lreg_src_a", a),
            ("lreg_src_b", b),
            ("lreg_src_c", name("zero")),
        ];
        self.op("SFPMUL", note, &args)
    }

    fn add(&mut self, a: Arg, b: Arg, note: &str) -> Result<Arg, ChainError> {
        let args = [
            ("lreg_src_a", name("one")),
            ("lreg_src_b", a),
            ("lreg_src_c", b),
        ];
        self.op("SFPADD", note, &args)
    }

    fn neg(&mut self, a: Arg) -> Result<Arg, ChainError> {
        let note = format!("-{a}");
        self.op(
            "SFPMOV",
            &note,
            &[("lreg_c", a), ("instr_mod1", Arg::Num(1))],
        )
    }
}

/// exp's constants as today writes them, by their bits: 1 / ln 2 (`1.442695`), the
/// polynomial's start (`0.00137805939`), ln 2's high and low parts (`-0.693145752`,
/// `-1.42860677e-06`), and the polynomial's other coefficients, highest first.
const EXP_INV_LN2: f64 = f64::from_bits(0x3FF7_1547_5A31_A4BE);
const EXP_START: f64 = f64::from_bits(0x3F56_9400_00C4_89A7);
const EXP_LN2_HIGH: f64 = f64::from_bits(0xBFE6_2E40_0006_7144);
const EXP_LN2_LOW: f64 = f64::from_bits(0xBEB7_F7D1_C150_7ED8);
const EXP_POLYNOMIAL: [f64; 4] = [
    f64::from_bits(0x3F81_25ED_BFD4_7362),
    f64::from_bits(0x3FA5_55B5_A027_ABAA),
    f64::from_bits(0x3FC5_5545_00E6_0400),
    f64::from_bits(0x3FDF_FFFF_6003_311E),
];

/// rsqrt's initial guess (its bits less x's halved) and its degree-2 correction's
/// coefficients (`2.2825186`, `2.2533049`), by their bits.
const RSQRT_MAGIC: i64 = 0x5F11_10A0;
const RSQRT_C1: f64 = f64::from_bits(0x4002_4299_1C9C_17F9);
const RSQRT_C2: f64 = f64::from_bits(0x4002_06C4_B82B_5518);
/// tanh below |x| = `TANH_SMALL` (0.625): x + x t p(t), t = x^2, p's coefficients
/// lowest first (`-0.33333284` .. `-0.0057170433`).
const TANH_SMALL: f64 = f64::from_bits(0x3FE4_0000_0000_0000);
const TANH_P: [f64; 5] = [
    f64::from_bits(0xBFD5_5553_439E_E635),
    f64::from_bits(0x3FC1_1076_8542_749A),
    f64::from_bits(0xBFAB_8444_83F9_B06C),
    f64::from_bits(0x3F95_256D_5E0D_8D98),
    f64::from_bits(0xBF77_6AC1_200D_94FE),
];
/// erf: x N(x^2) / D(x^2) on x clamped to [-10, 10] (tt-metal's fp32 erf rational),
/// N's and D's coefficients lowest first (`1.1283791065` .., `1.0` ..).
const ERF_N: [f64; 8] = [
    f64::from_bits(0x3FF2_0DD7_3FFE_83E5),
    f64::from_bits(0x3FCB_7DB9_A000_004D),
    f64::from_bits(0x3FAF_CFF4_2000_2C1C),
    f64::from_bits(0x3F77_0830_4000_355D),
    f64::from_bits(0x3F44_16CE_BFFF_D250),
    f64::from_bits(0x3EF2_897D_0001_C707),
    f64::from_bits(0x3E5D_676F_4001_FFEB),
    f64::from_bits(0xBDF3_AD01_8000_5ADE),
];
const ERF_D: [f64; 9] = [
    f64::from_bits(0x3FF0_0000_0000_0000),
    f64::from_bits(0x3FE0_C1ED_5FFF_98F7),
    f64::from_bits(0x3FC0_974A_BFFF_4D6A),
    f64::from_bits(0x3F94_1D2E_9FFE_7CEE),
    f64::from_bits(0x3F60_0300_0000_E4B3),
    f64::from_bits(0x3F21_4628_6001_AF46),
    f64::from_bits(0x3EB6_129B_6001_2D94),
    f64::from_bits(0xBE2E_326C_0000_54C9),
    f64::from_bits(0xBD9D_9F04_8000_0CE5),
];
/// log(1 + f) = f + f^2 r(f) for f in [-0.25, 0.5] (Norbert Juffa's logf, as
/// tt-metal), r's coefficients lowest first (`-0x1p-1` .. `-0x1.92cp-5`), and ln 2 in
/// fp32 (`0.693147182`).
const LOG_P: [f64; 9] = [
    f64::from_bits(0xBFE0_0000_0000_0000),
    f64::from_bits(0x3FD5_5557_2000_0000),
    f64::from_bits(0xBFD0_0001_A000_0000),
    f64::from_bits(0x3FC9_9800_0000_0000),
    f64::from_bits(0xBFC5_5000_0000_0000),
    f64::from_bits(0x3FC2_7400_0000_0000),
    f64::from_bits(0xBFC0_C400_0000_0000),
    f64::from_bits(0x3FBB_8400_0000_0000),
    f64::from_bits(0xBFA9_2C00_0000_0000),
];
const LN2_FP32: f64 = f64::from_bits(0x3FE6_2E42_FFC0_255B);

/// A field set to a number.
fn n(field: &'static str, value: i64) -> (&'static str, Arg) {
    (field, Arg::Num(value))
}

/// e^a in fp32 (tt-metal's `_sfpu_exp_fp32_accurate_`, guarded form): round a / ln 2 to
/// an integer, a two-part Cody-Waite reduction, a degree-6 polynomial, then the exponent
/// added back with overflow to infinity and underflow to 0.
fn exp(b: &mut Body, a: Arg) -> Result<Arg, ChainError> {
    let inv_ln2 = b.constant(EXP_INV_LN2)?;
    let j = b.mul(a.clone(), inv_ln2, "a / ln 2")?;
    let mut r = b.constant(EXP_START)?;
    let sm = b.op(
        "SFP_STOCH_RND",
        "round to sign-magnitude 16",
        &[
            ("lreg_src_b", name("zero")),
            ("lreg_src_c", j),
            n("instr_mod1", 7),
        ],
    )?;
    let j = b.op("SFPCAST", "back to fp32", &[("lreg_src_c", sm.clone())])?;
    let high = b.constant(EXP_LN2_HIGH)?;
    let f = b.mad_of(
        j.clone(),
        high,
        a,
        "f = a - j ln 2 (high)",
        false,
        false,
        None,
    )?;
    let low = b.constant(EXP_LN2_LOW)?;
    let f = b.mad_of(j.clone(), low, f, "f -= j ln 2 (low)", false, false, None)?;
    for c in EXP_POLYNOMIAL {
        let c = b.constant(c)?;
        r = b.mad_of(r, f.clone(), c, "polynomial", false, false, None)?;
    }
    let i = b.op("SFPABS", "|j| as an integer", &[("lreg_c", sm)])?;
    let y = b.mad_of(r, f.clone(), name("one"), "polynomial", false, false, None)?;
    let i = b.op(
        "SFPSETSGN",
        "j's sign on |j|",
        &[("lreg_c", i), ("lreg_dest", j)],
    )?;
    let r = b.mad_of(y.clone(), f, name("one"), "e^f", false, false, None)?;
    let y = b.op(
        "SFPMULI",
        "overflow: inf",
        &[n("imm16_math", INF_BF16), ("lreg_dest", y)],
    )?;
    let e = b.op(
        "SFPEXEXP",
        "biased exponent of e^f",
        &[("lreg_c", r.clone()), n("instr_mod1", 1)],
    )?;
    let e = b.op(
        "SFPIADD",
        "+ j",
        &[("lreg_c", i), ("lreg_dest", e), n("instr_mod1", IADD_NO_CC)],
    )?;
    b.when(
        "SFPIADD",
        "exponent < 255",
        &[
            n("imm12_math", 0xF01), // -255
            ("lreg_c", e.clone()),
            n("instr_mod1", IADD_IMM),
        ],
    )?;
    b.op(
        "SFPMOV",
        "",
        &[("lreg_c", e.clone()), ("lreg_dest", y.clone())],
    )?;
    b.op(
        "SFPSETEXP",
        "2^j e^f",
        &[("lreg_c", r), ("lreg_dest", y.clone())],
    )?;
    b.when(
        "SFPIADD",
        "exponent < 1",
        &[
            n("imm12_math", 0xFFF), // -1
            ("lreg_c", e.clone()),
            ("lreg_dest", e),
            n("instr_mod1", IADD_IMM),
        ],
    )?;
    b.op(
        "SFPMOV",
        "underflow: 0",
        &[("lreg_c", name("zero")), ("lreg_dest", y.clone())],
    )?;
    b.end()?;
    Ok(y)
}

/// 1 / x (tt-metal's `sfpu_reciprocal_iter<2>`): the unit's estimate and two Newton
/// steps, the second only where the first is not NaN (t < 0).
fn recip(b: &mut Body, x: Arg) -> Result<Arg, ChainError> {
    let two = b.register_const(2.0)?;
    let y = b.op("SFPARECIP", "estimate of 1 / x", &[("lreg_c", x.clone())])?;
    let t = b.mad_of(
        x.clone(),
        y.clone(),
        two.clone(),
        "t = x y - 2",
        false,
        true,
        None,
    )?;
    let minus_t = b.neg(t.clone())?;
    let zero = b.constant(-0.0)?;
    let y1 = b.mad_of(
        y.clone(),
        minus_t,
        zero,
        "y1 = y (-t) - 0",
        false,
        false,
        None,
    )?;
    b.when(
        "SFPGT",
        "t < 0",
        &[
            ("lreg_c", t),
            ("lreg_dest", name("zero")),
            n("instr_mod1", 1),
        ],
    )?;
    let t = b.mad_of(x, y1.clone(), two, "t = x y1 - 2", false, true, None)?;
    let minus_t = b.neg(t)?;
    let zero = b.constant(-0.0)?;
    b.mad_of(
        y1,
        minus_t,
        zero,
        "y = y1 (-t) - 0",
        false,
        false,
        Some(y.clone()),
    )?;
    b.end()?;
    Ok(y)
}

/// 1 / sqrt(x) for a positive finite x (tt-metal's `_calculate_sqrt_body_`, its 23-bit
/// reciprocal form, without the guards for 0, infinity and x < 0): the initial guess
/// from x's bits, a degree-2 correction, then one Newton step.
fn rsqrt(b: &mut Body, x: Arg) -> Result<Arg, ChainError> {
    let magic = b.register_bits(RSQRT_MAGIC)?;
    let c1 = b.register_const(RSQRT_C1)?;
    let c2 = b.register_const(RSQRT_C2)?;
    let i = b.op(
        "SFPSHFT",
        "x's bits >> 1",
        &[
            n("imm12_math", 0xFFF), // -1
            ("lreg_c", x.clone()),
            n("instr_mod1", SHIFT_IMM_VC),
        ],
    )?;
    let y = b.op(
        "SFPIADD",
        "y = magic - (x >> 1)",
        &[
            ("lreg_c", magic),
            ("lreg_dest", i),
            n("instr_mod1", IADD_SUB | IADD_NO_CC),
        ],
    )?;
    let xy = b.mul(x.clone(), y.clone(), "x y")?;
    let c = b.mad_of(y.clone(), xy, name("zero"), "c = -y x y", true, false, None)?;
    let sum = b.add(c.clone(), c2, "c2 + c")?;
    let factor = b.mad_of(c, sum, c1, "c1 + c (c2 + c)", false, false, None)?;
    let y = b.mul(y, factor, "y")?;
    let xy = b.mul(x, y.clone(), "x y")?;
    let e = b.mad_of(y.clone(), xy, name("one"), "1 - y x y", true, false, None)?;
    let half = b.op(
        "SFPDIVP2",
        "y / 2",
        &[
            n("imm12_math", 0xFF),
            ("lreg_c", y.clone()),
            n("instr_mod1", 1),
        ],
    )?;
    b.mad_of(e, half, y, "y + (1 - y x y) y / 2", false, false, None)
}

/// The square root of x >= 0: x rsqrt(x) (0 at 0: rsqrt's guess from 0's bits is
/// finite).
fn sqrt(b: &mut Body, x: Arg) -> Result<Arg, ChainError> {
    let r = rsqrt(b, x.clone())?;
    b.mul(x, r, "x rsqrt(x)")
}

/// c[0] + x (c[1] + x (...)), one multiply-add a coefficient.
fn horner(b: &mut Body, x: &Arg, coefficients: &[f64], note: &str) -> Result<Arg, ChainError> {
    let Some((&last, rest)) = coefficients.split_last() else {
        return Err(ChainError::Unit("a polynomial of no coefficients".into()));
    };
    let mut p = b.constant(last)?;
    for &c in rest.iter().rev() {
        let c = b.constant(c)?;
        p = b.mad_of(p, x.clone(), c, note, false, false, None)?;
    }
    Ok(p)
}

/// tanh(x): below |x| = 0.625 the odd polynomial x + x t p(t), above it
/// (1 - e^-2|x|) / (1 + e^-2|x|) with the accurate exp and x's sign.
fn tanh(b: &mut Body, x: Arg) -> Result<Arg, ChainError> {
    let t = b.mul(x.clone(), x.clone(), "t = x^2")?;
    let p = horner(b, &t, &TANH_P, "tanh polynomial")?;
    let xt = b.mul(x.clone(), t, "x t")?;
    let small = b.mad_of(xt, p, x.clone(), "x + x t p(t)", false, false, None)?;
    let a = b.op(
        "SFPABS",
        "|x|",
        &[("lreg_c", x.clone()), n("instr_mod1", 1)],
    )?;
    let out = b.op("SFPMOV", "x (its sign)", &[("lreg_c", x)])?;
    let minus_two = b.constant(-2.0)?;
    let scaled = b.mul(a.clone(), minus_two, "-2 |x|")?;
    let e = exp(b, scaled)?;
    let num = b.mad_of(
        e.clone(),
        name("one"),
        name("one"),
        "1 - e",
        true,
        false,
        None,
    )?;
    let den = b.add(e, name("one"), "1 + e")?;
    let inverse = recip(b, den)?;
    let big = b.mul(num, inverse, "(1 - e) / (1 + e)")?;
    let out = b.op(
        "SFPSETSGN",
        "the quotient with x's sign",
        &[("lreg_c", big), ("lreg_dest", out)],
    )?;
    let small_x = b.constant(TANH_SMALL)?;
    b.when(
        "SFPGT",
        "|x| < 0.625",
        &[("lreg_c", a), ("lreg_dest", small_x), n("instr_mod1", 1)],
    )?;
    let args = [("lreg_c", small), ("lreg_dest", out.clone())];
    b.word("SFPMOV", "tanh: the polynomial", &args)?;
    b.end()?;
    Ok(out)
}

/// erf(x): x N(x^2) / D(x^2) of x clamped to [-10, 10] (tt-metal's fp32 rational), the
/// quotient by `recip`, the result clamped to [-1, 1].
fn erf(b: &mut Body, x: Arg) -> Result<Arg, ChainError> {
    let low = b.constant(-10.0)?;
    let c = maximum(b, x, low)?;
    let high = b.constant(10.0)?;
    let c = minimum(b, c, high)?;
    let s = b.mul(c.clone(), c.clone(), "s = x^2")?;
    let numerator = horner(b, &s, &ERF_N, "erf numerator")?;
    let n_of = b.mul(c, numerator, "x N(s)")?;
    let denominator = horner(b, &s, &ERF_D, "erf denominator")?;
    let inverse = recip(b, denominator)?;
    let y = b.mul(n_of, inverse, "N / D")?;
    let low = b.constant(-1.0)?;
    let y = maximum(b, y, low)?;
    minimum(b, y, name("one"))
}

/// The natural log of a normal x > 0: x = 2^k m, m in [0.75, 1.5) (m above 1.5 halved,
/// k + 1), log(m) = f + f^2 r(f) with f = m - 1, then k ln 2 + log(m).
fn log(b: &mut Body, x: Arg) -> Result<Arg, ChainError> {
    let e = b.op("SFPEXEXP", "x's exponent - 127", &[("lreg_c", x.clone())])?;
    let m = b.op(
        "SFPSETEXP",
        "m in [1, 2)",
        &[n("imm12_math", 127), ("lreg_c", x), n("instr_mod1", 1)],
    )?;
    let e = b.op(
        "SFPCAST",
        "k as sign-magnitude",
        &[("lreg_src_c", e), n("instr_mod1", 3)],
    )?;
    let k = b.op("SFPCAST", "k as fp32", &[("lreg_src_c", e)])?;
    let limit = b.constant(1.5)?;
    b.when(
        "SFPGT",
        "m > 1.5",
        &[
            ("lreg_c", limit),
            ("lreg_dest", m.clone()),
            n("instr_mod1", 1),
        ],
    )?;
    let half = b.constant(0.5)?;
    let into = Some(m.clone());
    b.mad_of(m.clone(), half, name("zero"), "m / 2", false, false, into)?;
    let into = Some(k.clone());
    b.mad_of(
        k.clone(),
        name("one"),
        name("one"),
        "k + 1",
        false,
        false,
        into,
    )?;
    b.end()?;
    let minus_one = b.constant(-1.0)?;
    let f = b.add(m, minus_one, "f = m - 1 (exact)")?;
    let r = horner(b, &f, &LOG_P, "log polynomial")?;
    let square = b.mul(f.clone(), f.clone(), "f^2")?;
    let r = b.mad_of(r, square, f, "f + f^2 r", false, false, None)?;
    let ln2 = b.constant(LN2_FP32)?;
    b.mad_of(k, ln2, r, "k ln 2 + log(m)", false, false, None)
}

/// 1 / (1 + e^-x) on fp32 registers, as tt-metal's `_sfpu_sigmoid_<true>`.
fn sigmoid(b: &mut Body, x: Arg) -> Result<Arg, ChainError> {
    let minus_x = b.neg(x)?;
    let e = exp(b, minus_x)?;
    let sum = b.add(e, name("one"), "1 + e^-x")?;
    recip(b, sum)
}

/// The larger of x and y in every lane: one SFPSWAP on copies.
fn maximum(b: &mut Body, x: Arg, y: Arg) -> Result<Arg, ChainError> {
    Ok(swapped(b, x, y, "max")?.0)
}

/// The smaller of x and y in every lane: the other result of maximum's swap.
fn minimum(b: &mut Body, x: Arg, y: Arg) -> Result<Arg, ChainError> {
    Ok(swapped(b, x, y, "min")?.1)
}

/// Copies of x and y swapped where y is the larger (`what` names the copies' use):
/// (the larger, the smaller).
fn swapped(b: &mut Body, x: Arg, y: Arg, what: &str) -> Result<(Arg, Arg), ChainError> {
    let big = b.op("SFPMOV", &format!("{what}: first"), &[("lreg_c", x)])?;
    let small = b.op("SFPMOV", &format!("{what}: second"), &[("lreg_c", y)])?;
    b.word(
        "SFPSWAP",
        "max, min",
        &[
            ("lreg_src_c", big.clone()),
            ("lreg_dest", small.clone()),
            n("instr_mod1", 1),
        ],
    )?;
    Ok((big, small))
}

/// Lane tests of a condition: (instruction, mod1, the operands swapped).
const CONDITIONS: [(&str, (&str, i64, bool)); 2] =
    [("gt", ("SFPGT", 1, false)), ("lt", ("SFPGT", 1, true))];
/// SFPSETCC's mod1 of the tests against 0 (on the integer view: -0.0 is not 0).
const ZERO_TESTS: [(&str, i64); 2] = [("eq", 6), ("ne", 2)];
/// The tests made of SFPGT's: the tests whose lanes take the other value over the
/// value where the test holds (ne: the reverse), one SFPGT each.
const INVERSES: [(&str, &[&str]); 4] = [
    ("ge", &["lt"]),
    ("le", &["gt"]),
    ("eq", &["gt", "lt"]),
    ("ne", &["gt", "lt"]),
];

/// Whether test `kind` against node `y` is == or != against the constant +0.0 (an
/// SFPSETCC).
fn zero_test(vs: &Vs, kind: &str, y: VId) -> bool {
    let zero = vs.get(y).is_some_and(
        |n| matches!(n.value, VValue::Const(v) if n.kind == "const" && (v as f32).to_bits() == 0),
    );
    ZERO_TESTS.iter().any(|(k, _)| *k == kind) && zero
}

/// A select whose test is the inverse of SFPGT's: the value taken where the test
/// holds first, the other written over it in the lanes of each test of `INVERSES`.
#[allow(clippy::too_many_arguments)]
fn select_inverse(
    b: &mut Body,
    vs: &Vs,
    (kind, x, y): (&'static str, VId, VId),
    tests: &[&'static str],
    then: Arg,
    other: Arg,
    done: &mut Vec<(VId, Arg)>,
) -> Result<Arg, ChainError> {
    let (first, over) = if kind == "ne" {
        (other, then)
    } else {
        (then, other)
    };
    let out = b.op(
        "SFPMOV",
        &format!("select: x {kind} y"),
        &[("lreg_c", first)],
    )?;
    for &test in tests {
        when(b, vs, (test, x, y), done)?;
        let args = [("lreg_c", over.clone()), ("lreg_dest", out.clone())];
        b.word("SFPMOV", &format!("select: where x {test} y"), &args)?;
        b.end()?;
    }
    Ok(out)
}

/// Lane flags from comparison `kind` of x and y: the lanes where it holds run what
/// follows.
fn when(
    b: &mut Body,
    vs: &Vs,
    (kind, x, y): (&str, VId, VId),
    done: &mut Vec<(VId, Arg)>,
) -> Result<(), ChainError> {
    let x = value(b, vs, x, done)?;
    let y = value(b, vs, y, done)?;
    if let Some((_, mode)) = ZERO_TESTS.iter().find(|(k, _)| *k == kind) {
        if y == name("zero") {
            let args = [("lreg_c", x), n("instr_mod1", *mode)];
            return b.when("SFPSETCC", &format!("x {kind} 0"), &args);
        }
    }
    let Some((_, (op, mode, swap))) = CONDITIONS.iter().find(|(k, _)| *k == kind) else {
        return Err(plan_error(format!(
            "the comparison {kind} is not built in the unit program"
        )));
    };
    let (x, y) = if *swap { (y, x) } else { (x, y) };
    let args = [("lreg_c", y), ("lreg_dest", x), n("instr_mod1", *mode)];
    b.when(op, &format!("x {kind} y"), &args)
}

/// The value of vector node `id` (an operand, a constant, an operation), each node
/// once (`done`); operand k is a load of Dst tile k.
fn value(b: &mut Body, vs: &Vs, id: VId, done: &mut Vec<(VId, Arg)>) -> Result<Arg, ChainError> {
    if let Some((_, v)) = done.iter().find(|(k, _)| *k == id) {
        return Ok(v.clone());
    }
    let node = vs
        .get(id)
        .ok_or_else(|| ChainError::NotPorted("a vector value of no function".into()))?;
    let args = |b: &mut Body, done: &mut Vec<(VId, Arg)>| -> Result<Vec<Arg>, ChainError> {
        node.args.iter().map(|&a| value(b, vs, a, done)).collect()
    };
    let out = match (node.kind, &node.value) {
        ("operand", VValue::Operand(index, part)) => {
            if let Some(v) = b.row_operand(*index)? {
                done.push((id, v.clone()));
                return Ok(v);
            }
            if part.is_some() {
                return Err(plan_error(
                    "the unit program reads whole-tile operands".into(),
                ));
            }
            if *index >= OPERANDS {
                return Err(plan_error(format!(
                    "the unit program reads at most {OPERANDS} operand tiles"
                )));
            }
            b.load(TILE * *index as i64, &format!("operand {index}"))?
        }
        ("const", VValue::Const(x)) => match (*x as f32).to_bits() {
            0 => name("zero"),
            0x3F80_0000 => name("one"),
            _ => b.constant(*x)?,
        },
        ("abs", _) => {
            let Some(x) = args(b, done)?.into_iter().next() else {
                return Err(ChainError::NotPorted("an abs of no value".into()));
            };
            b.op(
                "SFPABS",
                "|x|",
                &[("lreg_c", x), ("instr_mod1", Arg::Num(1))],
            )?
        }
        ("mad", _) => {
            let mut a = args(b, done)?.into_iter();
            match (a.next(), a.next(), a.next()) {
                (Some(x), Some(y), Some(z)) => b.mad(x, y, z)?,
                _ => {
                    return Err(ChainError::NotPorted(
                        "a multiply-add of no 3 values".into(),
                    ))
                }
            }
        }
        ("scalar", VValue::Scalar(index)) => name(&b.scalar(*index)),
        ("mul" | "add" | "sub" | "neg", _) => {
            let mut a = args(b, done)?.into_iter();
            let (x, y) = (a.next(), a.next());
            match (node.kind, x, y) {
                ("mul", Some(x), Some(y)) => {
                    let called = node
                        .args
                        .get(1)
                        .and_then(|&n| vs.get(n))
                        .is_some_and(|n| n.kind == "call");
                    b.mul(x, y, if called { "x sigmoid(x)" } else { "" })?
                }
                ("add", Some(x), Some(y)) => b.add(x, y, "")?,
                ("sub", Some(x), Some(y)) => {
                    let negated = b.neg(y)?;
                    b.add(x, negated, "")?
                }
                ("neg", Some(x), _) => b.neg(x)?,
                _ => {
                    return Err(ChainError::NotPorted(format!(
                        "a {} of no operands",
                        node.kind
                    )))
                }
            }
        }
        ("call", VValue::Call(f)) if FUNCTIONS.contains(f) => {
            let Some(x) = args(b, done)?.into_iter().next() else {
                return Err(ChainError::NotPorted(format!("a call of {f} on no value")));
            };
            match *f {
                "exp" => exp(b, x)?,
                "sigmoid" => sigmoid(b, x)?,
                "rsqrt" => rsqrt(b, x)?,
                "recip" => recip(b, x)?,
                "sqrt" => sqrt(b, x)?,
                "tanh" => tanh(b, x)?,
                "erf" => erf(b, x)?,
                _ => log(b, x)?,
            }
        }
        ("call", VValue::Call(f)) => {
            return Err(plan_error(format!(
                "vector function {f} is not built in the unit program"
            )))
        }
        ("max" | "min", _) => {
            let mut a = args(b, done)?.into_iter();
            let (Some(x), Some(y)) = (a.next(), a.next()) else {
                return Err(ChainError::NotPorted(format!(
                    "a {} of no two values",
                    node.kind
                )));
            };
            if node.kind == "max" {
                maximum(b, x, y)?
            } else {
                minimum(b, x, y)?
            }
        }
        ("select", VValue::Cond(kind, x, y)) => {
            let mut a = args(b, done)?.into_iter();
            let (Some(then), Some(other)) = (a.next(), a.next()) else {
                return Err(ChainError::NotPorted("a select of no two values".into()));
            };
            let inverse = INVERSES.iter().find(|(k, _)| k == kind);
            match inverse {
                Some((_, tests)) if !zero_test(vs, kind, *y) => {
                    select_inverse(b, vs, (kind, *x, *y), tests, then, other, done)?
                }
                _ => {
                    let out = b.op("SFPMOV", "select: the other value", &[("lreg_c", other)])?;
                    when(b, vs, (kind, *x, *y), done)?;
                    let args = [("lreg_c", then), ("lreg_dest", out.clone())];
                    b.word("SFPMOV", "select: where it holds", &args)?;
                    b.end()?;
                    out
                }
            }
        }
        ("flash_exp2", _) => {
            return Err(ChainError::NotPorted(format!(
                "the vector operation {}",
                node.kind
            )))
        }
        (kind, _) => {
            return Err(plan_error(format!(
                "vector operation {kind} is not built in the unit program"
            )))
        }
    };
    done.push((id, out.clone()));
    Ok(out)
}

/// The instructions of `code` that read `v`, in order.
fn readers(code: &[Insn], v: &Arg) -> Vec<usize> {
    (0..code.len())
        .filter(|&i| code[i].reads().contains(v))
        .collect()
}

/// The instructions of `code` that write `v`, in order.
fn writers(code: &[Insn], v: &Arg) -> Result<Vec<usize>, ChainError> {
    let mut out = Vec::new();
    for (i, x) in code.iter().enumerate() {
        if x.writes()?.contains(v) {
            out.push(i);
        }
    }
    Ok(out)
}

fn flip(insn: &mut Insn, field: &str, bits: i64) {
    let v = at(insn, field).num().unwrap_or(0) ^ bits;
    insn.set(field, Arg::Num(v));
}

/// Algebra: a negation (SFPMOV mod1 1) disappears into a multiply-add, into its
/// producer (one reader: both signs flipped) or its consumer (one reader, as a product
/// operand: the product's sign flipped). Returns what it did.
pub fn fold_negation(code: &mut Vec<Insn>) -> Result<Vec<String>, ChainError> {
    let mut done = Vec::new();
    let mut i = 0;
    while i < code.len() {
        if code[i].op.name != "SFPMOV" || at(&code[i], "instr_mod1") != Arg::Num(1) {
            i += 1;
            continue;
        }
        let (source, out) = (at(&code[i], "lreg_c"), at(&code[i], "lreg_dest"));
        let made = writers(code, &source)?;
        let read = readers(code, &source);
        let users = readers(code, &out);
        if let [m] = made[..] {
            if code[m].op.name == "SFPMAD" && read == [i] {
                let producer = &mut code[m];
                flip(producer, "instr_mod1", 3);
                producer.set("lreg_dest", out);
                producer.note = format!("-({})", producer.note);
                done.push(format!("a negation into its producer ({})", producer.note));
                code.remove(i);
                continue;
            }
        }
        if let [u] = users[..] {
            if code[u].op.name == "SFPMAD" {
                let user = &mut code[u];
                let slots: Vec<&str> = ["lreg_src_a", "lreg_src_b"]
                    .into_iter()
                    .filter(|s| at(user, s) == out)
                    .collect();
                if slots.len() == 1 && at(user, "lreg_src_c") != out {
                    if slots == ["lreg_src_b"] {
                        let a = at(user, "lreg_src_a");
                        user.set("lreg_src_b", a);
                    }
                    user.set("lreg_src_a", source);
                    flip(user, "instr_mod1", 1);
                    done.push(format!("a negation into its consumer ({})", user.note));
                    code.remove(i);
                    continue;
                }
            }
        }
        if !users.is_empty() && users.iter().all(|&u| addend_only(&code[u], &out)) {
            for &u in &users {
                code[u].set("lreg_src_c", source.clone());
                flip(&mut code[u], "instr_mod1", 2);
            }
            done.push(format!(
                "a negation into the addend of {} multiply-add(s)",
                users.len()
            ));
            code.remove(i);
            continue;
        }
        i += 1;
    }
    Ok(done)
}

/// True when multiply-add `insn` reads `value` once, as its addend (VC).
fn addend_only(insn: &Insn, value: &Arg) -> bool {
    if !matches!(insn.op.name, "SFPMAD" | "SFPADD") {
        return false;
    }
    let reads = ["lreg_src_a", "lreg_src_b", "lreg_src_c"].map(|f| at(insn, f));
    reads.iter().filter(|r| *r == value).count() == 1 && at(insn, "lreg_src_c") == *value
}

/// Algebra: a product read only by an add becomes one multiply-add (one rounding in
/// place of two). Returns what it did.
pub fn fuse_mad(code: &mut Vec<Insn>) -> Result<Vec<String>, ChainError> {
    let mut done = Vec::new();
    let mut i = 0;
    while i < code.len() {
        if code[i].op.name != "SFPMUL" || !is_name(&at(&code[i], "lreg_src_c"), "zero") {
            i += 1;
            continue;
        }
        let out = at(&code[i], "lreg_dest");
        let users = readers(code, &out);
        let made = writers(code, &out)?;
        let add = match (&users[..], made.len()) {
            ([u], 1) => *u,
            _ => {
                i += 1;
                continue;
            }
        };
        if code[add].op.name != "SFPADD" || !is_name(&at(&code[add], "lreg_src_a"), "one") {
            i += 1;
            continue;
        }
        let (b, c) = (at(&code[add], "lreg_src_b"), at(&code[add], "lreg_src_c"));
        if (b == out) == (c == out) {
            i += 1;
            continue;
        }
        let product = code[i].clone();
        let mad = op_named("SFPMAD").ok_or_else(|| UnitError("no operation SFPMAD".into()))?;
        let fused = &mut code[add];
        fused.op = mad;
        fused.set("lreg_src_a", at(&product, "lreg_src_a"));
        fused.set("lreg_src_b", at(&product, "lreg_src_b"));
        fused.set("lreg_src_c", if b == out { c.clone() } else { b });
        if c == out && at(fused, "instr_mod1").num().unwrap_or(0) & 2 != 0 {
            // the add negated the product: the product's sign flips, not the addend's
            flip(fused, "instr_mod1", 3);
        }
        flip(
            fused,
            "instr_mod1",
            at(&product, "instr_mod1").num().unwrap_or(0) & 1,
        );
        let first = if product.note.is_empty() {
            "a b"
        } else {
            &product.note
        };
        fused.note = format!("{first} + c, one rounding").trim().to_string();
        done.push(format!("a product into its add ({})", fused.note));
        code.remove(i);
    }
    Ok(done)
}

/// The depth of the lane conditions open after `insn`, from `depth` before it.
fn depth_after(insn: &Insn, depth: usize) -> usize {
    match (insn.sets_flags(), insn.op.name) {
        (false, _) => depth,
        (true, "SFPENCC") => 0,
        (true, _) => depth + 1,
    }
}

/// `insn`'s register fields that name a value of `alias`, renamed to its value.
fn renamed(insn: &mut Insn, alias: &HashMap<Arg, Arg>) {
    for a in insn.args.iter_mut() {
        if let Some(to) = alias.get(a).filter(|_| matches!(a, Arg::Name(_))) {
            *a = to.clone();
        }
    }
}

/// True for a value's name (`v<n>`): not a fixed, constant or numbered register.
fn is_value(v: &Arg) -> bool {
    matches!(v, Arg::Name(n) if n.starts_with('v'))
}

/// Algebra: a plain copy (SFPMOV) of a value read no more after it is no copy: its
/// readers take the value itself. Returns what it did.
pub fn drop_copies(code: &mut Vec<Insn>) -> Result<Vec<String>, ChainError> {
    let mut first: HashMap<Arg, usize> = HashMap::new();
    let mut last: HashMap<Arg, usize> = HashMap::new();
    for (i, insn) in code.iter().enumerate() {
        for v in insn.reads().into_iter().chain(insn.writes()?) {
            first.entry(v.clone()).or_insert(i);
            last.insert(v, i);
        }
    }
    let (mut alias, mut keep, mut depth) = (HashMap::new(), Vec::new(), 0);
    for (i, mut insn) in std::mem::take(code).into_iter().enumerate() {
        renamed(&mut insn, &alias);
        depth = depth_after(&insn, depth);
        let source = insn.arg("lreg_c").cloned();
        let out = insn.arg("lreg_dest").cloned();
        if let (Some(source), Some(out)) = (source, out) {
            let copy = insn.op.name == "SFPMOV"
                && at(&insn, "instr_mod1") == Arg::Num(0)
                && depth == 0
                && is_value(&source)
                && last.get(&source) == Some(&i)
                && first.get(&out) == Some(&i);
            if copy {
                let end = last.get(&out).copied().unwrap_or(i);
                last.insert(source.clone(), end);
                alias.insert(out, source);
                continue;
            }
        }
        keep.push(insn);
    }
    let done = alias.len();
    *code = keep;
    Ok(if done > 0 {
        vec![format!("{done} copies of values read no more")]
    } else {
        Vec::new()
    })
}

/// A field's value as `common_values` compares it: a constant by its bits.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Known {
    Bits(i64),
    Value(Arg),
}

/// What an instruction computes, as `common_values` compares it: its operation and
/// each field it does not write.
type Computed = (&'static str, Vec<(&'static str, Known)>);

/// Algebra: an instruction computing what an earlier one computed from the same values
/// is dropped, its readers take the earlier result (a per-row value computed once a
/// block, not once a vector). Values written once only. Returns what it did.
pub fn common_values(code: &mut Vec<Insn>) -> Result<Vec<String>, ChainError> {
    let mut writes: HashMap<Arg, usize> = HashMap::new();
    let mut loaders: Vec<(Arg, Vec<usize>)> = Vec::new();
    for (i, insn) in code.iter().enumerate() {
        for v in insn.writes()? {
            *writes.entry(v.clone()).or_insert(0) += 1;
            match loaders.iter_mut().find(|(l, _)| *l == v) {
                Some((_, at)) => at.push(i),
                None => loaders.push((v, vec![i])),
            }
        }
    }
    // a constant is known by its bits, whichever value loads it (bits 0 are not known,
    // as today's test of them)
    let mut bits: HashMap<Arg, i64> = HashMap::new();
    for (v, at) in &loaders {
        if at.iter().all(|&i| code[i].op.name == "SFPLOADI") {
            let words: Vec<&Insn> = at.iter().map(|&i| &code[i]).collect();
            if let Some(b) = bits_of(&words).filter(|b| *b != 0) {
                bits.insert(v.clone(), b);
            }
        }
    }
    let mut seen: HashMap<Computed, Arg> = HashMap::new();
    let (mut alias, mut keep, mut depth) = (HashMap::new(), Vec::new(), 0);
    let before = code.len();
    for mut insn in std::mem::take(code) {
        renamed(&mut insn, &alias);
        depth = depth_after(&insn, depth);
        // constants stay loaded where read (the constant passes place them)
        let out = insn.writes()?;
        let kind = insn.op.kind;
        let reads_known = insn
            .reads()
            .iter()
            .all(|v| writes.get(v).copied().unwrap_or(0) <= 1 || bits.contains_key(v));
        let candidate = depth == 0
            && !insn.sets_flags()
            && matches!(kind, Kind::Mad | Kind::Simple)
            && insn.op.name != "SFPSWAP"
            && out.len() == 1
            && is_value(&out[0])
            && writes.get(&out[0]) == Some(&1)
            && reads_known;
        if candidate {
            let fields: Vec<(&'static str, Known)> = insn
                .op
                .args
                .iter()
                .filter(|(f, _)| !insn.op.writes.contains(f))
                .map(|(f, _)| {
                    let v = at(&insn, f);
                    let known = match bits.get(&v) {
                        Some(b) => Known::Bits(*b),
                        None => Known::Value(v),
                    };
                    (*f, known)
                })
                .collect();
            let key = (insn.op.name, fields);
            if let Some(earlier) = seen.get(&key) {
                alias.insert(out[0].clone(), earlier.clone());
                continue;
            }
            seen.insert(key, out[0].clone());
        }
        keep.push(insn);
    }
    let read: Vec<Arg> = keep
        .iter()
        .filter(|x| x.op.name != "SFPLOADI")
        .flat_map(Insn::reads)
        .collect();
    let mut kept = Vec::new();
    for x in keep {
        let loads_unread =
            x.op.name == "SFPLOADI" && x.writes()?.first().is_none_or(|w| !read.contains(w));
        if !loads_unread {
            kept.push(x);
        }
    }
    let done = before - kept.len();
    *code = kept;
    Ok(if done > 0 {
        vec![format!(
            "{done} instructions computed once for their readers"
        )]
    } else {
        Vec::new()
    })
}

/// Rule per_row_prepass: the values of row code `code` made from the per-row operands'
/// loads (at `addresses`) and constants alone, computed once a block in a pass over the
/// operand tile first, stored over it (in all its lanes). Returns the pass's code (none
/// when it is not made) and its report; `code` then loads the value.
pub fn per_row_prepass(
    code: &mut Vec<Insn>,
    addresses: &[i64],
) -> Result<(Option<Vec<Insn>>, Vec<String>), ChainError> {
    let address_of = |x: &Insn| at(x, "dest_reg_addr").num();
    let loads: Vec<usize> = (0..code.len())
        .filter(|&i| {
            code[i].op.name == "SFPLOAD"
                && address_of(&code[i]).is_some_and(|a| addresses.contains(&a))
        })
        .collect();
    let mut consts: Vec<Arg> = Vec::new();
    for x in code.iter().filter(|x| x.op.name == "SFPLOADI") {
        for v in x.writes()? {
            let mut only_loadi = true;
            for w in code.iter() {
                if w.writes()?.contains(&v) && w.op.name != "SFPLOADI" {
                    only_loadi = false;
                }
            }
            if only_loadi && !consts.contains(&v) {
                consts.push(v);
            }
        }
    }
    let fixed: Vec<Arg> = FIXED.iter().map(|(n, _)| name(n)).collect();
    let mut only: Vec<Arg> = Vec::new();
    for &i in &loads {
        if let Some(v) = code[i].writes()?.into_iter().next() {
            only.push(v);
        }
    }
    let mut part: Vec<usize> = Vec::new();
    for (i, x) in code.iter().enumerate() {
        let skipped = loads.contains(&i)
            || matches!(x.op.name, "SFPLOAD" | "SFPSTORE" | "SFPLOADI")
            || x.sets_flags();
        if skipped {
            continue;
        }
        let reads = x.reads();
        let fits = reads.iter().any(|v| only.contains(v))
            && reads
                .iter()
                .all(|v| only.contains(v) || consts.contains(v) || fixed.contains(v));
        let written = x.writes()?;
        if fits && written.len() == 1 {
            only.push(written[0].clone());
            part.push(i);
        }
    }
    if part.is_empty() {
        let done = "nothing: no value of the per-row operands alone".to_string();
        return Ok((None, vec![done]));
    }
    let mut computed: Vec<Arg> = Vec::new();
    for &i in &part {
        computed.extend(code[i].writes()?);
    }
    let rest: Vec<usize> = (0..code.len())
        .filter(|i| !part.contains(i) && !loads.contains(i))
        .collect();
    let read_by_rest = |v: &Arg| rest.iter().any(|&i| code[i].reads().contains(v));
    let mut frontier: Vec<Arg> = Vec::new();
    for v in &computed {
        if read_by_rest(v) && !frontier.contains(v) {
            frontier.push(v.clone());
        }
    }
    let (&[load], [value]) = (loads.as_slice(), frontier.as_slice()) else {
        let done = format!(
            "refused: one per-row operand and one value of it read by the rows ({} operand(s), {} value(s))",
            loads.len(),
            frontier.len()
        );
        return Ok((None, vec![done]));
    };
    let value = value.clone();
    let operand_value = code[load].writes()?.into_iter().next();
    if operand_value.is_some_and(|v| read_by_rest(&v)) {
        return Ok((
            None,
            vec!["refused: the rows read the operand itself too".into()],
        ));
    }
    let needed: Vec<Arg> = part
        .iter()
        .flat_map(|&i| code[i].reads())
        .filter(|v| consts.contains(v))
        .collect();
    let mut setup: Vec<usize> = Vec::new();
    for (i, x) in code.iter().enumerate() {
        if x.op.name == "SFPLOADI" && x.writes()?.iter().any(|w| needed.contains(w)) {
            setup.push(i);
        }
    }
    let setup_words: Vec<Insn> = setup.iter().map(|&i| code[i].clone()).collect();
    let part_words: Vec<Insn> = part.iter().map(|&i| code[i].clone()).collect();
    let mut dropped: Vec<usize> = part.clone();
    for &s in &setup {
        let written = code[s].writes()?;
        let read = (0..code.len())
            .filter(|i| !part.contains(i) && !setup.contains(i))
            .any(|i| code[i].reads().iter().any(|v| written.contains(v)));
        if !read {
            dropped.push(s);
        }
    }
    let operand = code[load].clone();
    let address = at(&operand, "dest_reg_addr");
    code[load].set("lreg_ind", value.clone());
    code[load].note = format!("{}: the per-row pass's value", code[load].note);
    let mut kept = Vec::new();
    for (i, x) in std::mem::take(code).into_iter().enumerate() {
        if !dropped.contains(&i) {
            kept.push(x);
        }
    }
    *code = kept;
    let mut pre = setup_words;
    let mut reload = operand.clone();
    reload.note = "per-row operand".into();
    pre.push(reload);
    pre.extend(part_words);
    let store = [
        ("lreg_ind", value),
        ("instr_mod0", at(&operand, "instr_mod0")),
        ("sfpu_addr_mode", Arg::Num(ADDR_MOD)),
        ("dest_reg_addr", address),
    ];
    pre.push(Insn::new(
        "SFPSTORE",
        &store,
        "the per-row value over the operand",
    )?);
    let done = format!(
        "{} instruction(s) and {} load(s) a block, once",
        part.len(),
        setup.len()
    );
    Ok((Some(pre), vec![done]))
}

/// The per-row pass over the operand tile's 16 blocks (allocated, its constants loaded
/// once before the walk, the step in its store), then Dst back to 0: what its cost is
/// estimated on.
pub fn prepass_loop(pre: &[Insn], prog: &[(String, Constant)]) -> Result<Vec<Item>, ChainError> {
    let mut code = pre.to_vec();
    let mut pinned = Vec::new();
    for x in code.iter().filter(|x| x.op.name == "SFPLOADI") {
        pinned.extend(x.writes()?);
    }
    allocate(&mut code, prog, &pinned)?;
    let (consts, mut walk): (Vec<Insn>, Vec<Insn>) =
        code.into_iter().partition(|x| x.op.name == "SFPLOADI");
    walk.push(insn(
        "INCRWC",
        &[("rwc_cr", 0), ("rwc_d", 4), ("rwc_b", 0), ("rwc_a", 0)],
        "next 4-row block",
    )?);
    store_step(&mut walk, 4);
    let mut walk = items(walk);
    check_mad_readers(&mut walk)?;
    let step = setrwc(CR_D, 8, SET_D, "Dst + 8 rows")?;
    let mut out = items(consts);
    out.push(Item::Loop(Loop {
        name: "face".into(),
        count: Count::Times(FACES),
        body: vec![
            Item::Loop(Loop {
                name: "row".into(),
                count: Count::Times(ROWS / 2),
                body: walk,
                unroll: true,
            }),
            Item::Insn(step.clone()),
            Item::Insn(step),
        ],
        unroll: false,
    }));
    out.push(Item::Insn(setrwc(0, 0, SET_D, "Dst counter back to 0")?));
    Ok(out)
}

/// The 32-bit value SFPLOADI words build (FLOATB, FLOATA, USHORT, then UPPER), or
/// none for another form.
fn bits_of(words: &[&Insn]) -> Option<i64> {
    let mut bits: Option<i64> = None;
    for w in words {
        let (mode, imm) = (at(w, "instr_mod0").num()?, at(w, "imm16").num()?);
        bits = match (mode, bits) {
            (0, _) => Some(imm << 16),
            (1, _) => Some(i64::from(
                py::half_to_f32(u16::try_from(imm).ok()?).to_bits(),
            )),
            (2, _) => Some(imm),
            (8, Some(b)) => Some((b & 0xFFFF) | imm << 16),
            _ => return None,
        };
    }
    bits
}

/// The SFPLOADI of the constant-register setup: `bits` into LReg 0 (one word when the
/// low half is 0) and the SFPCONFIG that copies it into constant register `reg`.
fn config(reg: &str, bits: i64, note: &str) -> Result<Vec<Insn>, UnitError> {
    let load = |mode: i64, imm: i64| {
        let args = [
            ("lreg_ind", name("L0")),
            ("instr_mod0", Arg::Num(mode)),
            ("imm16", Arg::Num(imm)),
        ];
        Insn::new("SFPLOADI", &args, note)
    };
    let mut out = Vec::new();
    if bits & 0xFFFF != 0 {
        out.push(load(2, bits & 0xFFFF)?);
        out.push(load(8, bits >> 16)?);
    } else {
        out.push(load(0, bits >> 16)?);
    }
    let args = [
        ("imm16_math", Arg::Num(0)),
        ("config_dest", name(reg)),
        ("instr_mod1", Arg::Num(0)),
    ];
    out.push(Insn::new(
        "SFPCONFIG",
        &args,
        &format!("constant register {reg}"),
    )?);
    Ok(out)
}

/// Layout: the constants the row code loads (SFPLOADI) move into the free programmable
/// registers, most loader words saved first; equal constants share one. `registers`
/// 4: LReg 11 too, when no code reads its -1.0 (the body sets it and restores it).
/// Returns what it did.
pub fn constant_registers(
    code: &mut Vec<Insn>,
    prog: &mut Vec<(String, Constant)>,
    registers: usize,
) -> Result<Vec<String>, ChainError> {
    let mut registers = registers;
    if code.iter().any(|x| x.reads().contains(&name("neg_one"))) {
        registers = registers.min(PROGRAMMABLE.len());
    }
    let mut found: Vec<(Arg, Vec<usize>)> = Vec::new();
    for (i, x) in code.iter().enumerate() {
        let held = at(x, "lreg_ind");
        if x.op.name == "SFPLOADI" && matches!(&held, Arg::Name(s) if s.starts_with('v')) {
            match found.iter_mut().find(|(v, _)| *v == held) {
                Some((_, words)) => words.push(i),
                None => found.push((held, vec![i])),
            }
        }
    }
    let as_vd: Vec<Arg> = code
        .iter()
        .filter_map(|x| x.arg("lreg_dest").cloned())
        .collect();
    let mut by_bits: Vec<(i64, Vec<Arg>)> = Vec::new();
    for (v, words) in &found {
        let made = writers(code, v)?;
        if made.len() != words.len() || made.iter().any(|&w| code[w].op.name != "SFPLOADI") {
            continue;
        }
        if as_vd.contains(v) {
            continue; // read as VD (SFPGT's left side): VD 12 .. 14 stores a template
        }
        let loads: Vec<&Insn> = words.iter().map(|&w| &code[w]).collect();
        let Some(bits) = bits_of(&loads) else {
            continue;
        };
        match by_bits.iter_mut().find(|(b, _)| *b == bits) {
            Some((_, values)) => values.push(v.clone()),
            None => by_bits.push((bits, vec![v.clone()])),
        }
    }
    let words_of = |v: &Arg| {
        found
            .iter()
            .find(|(f, _)| f == v)
            .map_or(0, |(_, w)| w.len())
    };
    let mut order: Vec<(i64, Vec<Arg>, usize)> = by_bits
        .into_iter()
        .map(|(b, vs)| {
            let saved = vs.iter().map(words_of).sum();
            (b, vs, saved)
        })
        .collect();
    order.sort_by_key(|(_, _, saved)| std::cmp::Reverse(*saved));
    let notes: Vec<(Arg, String)> = found
        .iter()
        .map(|(v, w)| {
            (
                v.clone(),
                w.first().map_or(String::new(), |&i| code[i].note.clone()),
            )
        })
        .collect();
    let mut done = Vec::new();
    for (bits, values, saved) in order {
        if prog.len() >= registers {
            break;
        }
        let reg = format!("p{}", prog.len());
        let note = values
            .first()
            .and_then(|v| notes.iter().find(|(n, _)| n == v))
            .map_or(String::new(), |(_, n)| n.clone());
        prog.push((reg.clone(), Constant::Text(note.clone())));
        for v in &values {
            code.retain(|x| !(x.op.name == "SFPLOADI" && at(x, "lreg_ind") == *v));
            for x in code.iter_mut() {
                for (field, _) in x.op.args {
                    if x.arg(field) == Some(v) {
                        x.set(field, name(&reg));
                    }
                }
            }
        }
        let setup = config(&reg, bits, &note)?;
        code.splice(0..0, setup);
        done.push(format!("{note} in {reg} ({saved} words a row)"));
    }
    Ok(done)
}

/// Algebra: a -0.0 loaded only as the addend of multiply-adds becomes the fixed zero
/// with the addend's sign flipped (one SFPLOADI less). Returns what it did.
pub fn negated_zero(code: &mut Vec<Insn>) -> Result<Vec<String>, ChainError> {
    let mut done = Vec::new();
    let mut i = 0;
    while i < code.len() {
        let x = &code[i];
        if x.op.name != "SFPLOADI"
            || (at(x, "instr_mod0"), at(x, "imm16")) != (Arg::Num(0), Arg::Num(0x8000))
        {
            i += 1;
            continue;
        }
        let v = at(x, "lreg_ind");
        let read = readers(code, &v);
        let made = writers(code, &v)?;
        let addends = read.iter().all(|&r| {
            let y = &code[r];
            let fields = y.read_fields();
            y.op.name == "SFPMAD"
                && fields.iter().filter(|f| **f == "lreg_src_c").count() == 1
                && fields.iter().filter(|f| at(y, f) == v).count() == 1
                && at(y, "lreg_src_c") == v
        });
        if made != [i] || !addends {
            i += 1;
            continue;
        }
        for &r in &read {
            code[r].set("lreg_src_c", name("zero"));
            flip(&mut code[r], "instr_mod1", 2);
        }
        code.remove(i);
        done.push(format!("-0.0 as -zero ({} reader(s))", read.len()));
    }
    Ok(done)
}

/// The run-time scalars' loads (each a float handler word, `<name>_bits` of the body's
/// call): two SFPLOADI a scalar whose immediates the core builds from the argument,
/// before the rows; refused for a reciprocal or an integer size. Returns the loads and
/// the words' names.
pub fn scalar_loads(
    ir: &KernelIR,
    chain: &Chain,
    body: &Body,
) -> Result<(Vec<Insn>, Vec<String>), ChainError> {
    let (mut loads, mut names) = (Vec::new(), Vec::new());
    for (index, scalar) in chain.values.iter().enumerate() {
        let float = ir
            .values
            .iter()
            .any(|v| v.name == scalar.name && v.kind == "float");
        if !float {
            return Err(plan_error(format!(
                "{}: a run-time scalar of the unit program is a float word",
                ir.name
            )));
        }
        if scalar.recip {
            return Err(plan_error(format!(
                "{}: a division by the run-time scalar {}: the unit program multiplies; pass its reciprocal word and multiply by it",
                ir.name, scalar.name
            )));
        }
        let word = format!("{}_bits", scalar.name);
        names.push(word.clone());
        let Some((_, held)) = body.scalars.iter().find(|(i, _)| *i == index) else {
            continue;
        };
        for (mode, text) in [
            (2, format!("{word} & 0xFFFFu")),
            (8, format!("{word} >> 16")),
        ] {
            let args = [
                ("lreg_ind", name(held)),
                ("instr_mod0", Arg::Num(mode)),
                ("imm16", Arg::Num(0)),
            ];
            let mut load = Insn::new(
                "SFPLOADI",
                &args,
                &format!("{}, the call's word", scalar.name),
            )?;
            load.runtime = text;
            loads.push(load);
        }
    }
    Ok((loads, names))
}

/// Every value of `values` read as `to`; their loads but `to`'s dropped.
fn share(code: &mut Vec<Insn>, values: &[Arg], to: &Arg) {
    code.retain(|x| {
        let held = at(x, "lreg_ind");
        !(x.op.name == "SFPLOADI" && values.contains(&held) && held != *to)
    });
    for x in code.iter_mut() {
        for (field, _) in x.op.args {
            if x.arg(field).is_some_and(|v| values.contains(v)) {
                x.set(field, to.clone());
            }
        }
    }
}

/// Layout: constants the row still loads move into work registers that stay set
/// across the rows, most words saved first, while the row's values still fit; equal
/// constants share one; `loads` and `pinned` are held already (run-time scalars).
/// Returns what it did, the loads and the pinned values.
pub fn resident_constants(
    code: &mut Vec<Insn>,
    prog: &[(String, Constant)],
    loads: Vec<Insn>,
    pinned: Vec<Arg>,
) -> Result<Held, ChainError> {
    let mut found: Vec<(Arg, Vec<usize>)> = Vec::new();
    for (i, x) in code.iter().enumerate() {
        let held = at(x, "lreg_ind");
        if x.op.name == "SFPLOADI" && matches!(held, Arg::Name(_)) {
            match found.iter_mut().find(|(v, _)| *v == held) {
                Some((_, words)) => words.push(i),
                None => found.push((held, vec![i])),
            }
        }
    }
    let mut by_bits: Vec<(i64, Vec<Arg>, usize)> = Vec::new();
    for (v, words) in &found {
        let made = writers(code, v)?;
        let loads: Vec<&Insn> = words.iter().map(|&w| &code[w]).collect();
        let bits = bits_of(&loads);
        let named = matches!(v, Arg::Name(s) if s.starts_with('v'));
        if let (true, Some(bits), true) = (made == *words, bits, named) {
            match by_bits.iter_mut().find(|(b, _, _)| *b == bits) {
                Some((_, values, saved)) => {
                    values.push(v.clone());
                    *saved += words.len();
                }
                None => by_bits.push((bits, vec![v.clone()], words.len())),
            }
        }
    }
    by_bits.sort_by_key(|(_, _, saved)| std::cmp::Reverse(*saved));
    let (mut done, mut loads, mut pinned) = (Vec::new(), loads, pinned);
    for (_, values, saved) in by_bits {
        let Some(first) = values.first().cloned() else {
            continue;
        };
        let mut trial = code.clone();
        share(&mut trial, &values, &first);
        let is_word = |x: &Insn| x.op.name == "SFPLOADI" && at(x, "lreg_ind") == first;
        let words: Vec<Insn> = trial.iter().filter(|x| is_word(x)).cloned().collect();
        trial.retain(|x| !is_word(x));
        let mut whole: Vec<Insn> = loads.iter().chain(&words).chain(&trial).cloned().collect();
        let mut held = pinned.clone();
        held.push(first.clone());
        match allocate(&mut whole, prog, &held) {
            Ok(_) => {}
            Err(ChainError::Unit(_)) => continue,
            Err(e) => return Err(e),
        }
        *code = trial;
        let note = words.first().map_or(String::new(), |w| w.note.clone());
        loads.extend(words);
        pinned.push(first);
        done.push(format!("{note} held across rows ({saved} words a row)"));
    }
    Ok(Held {
        done,
        loads,
        pinned,
    })
}

/// What `resident_constants` did, and the values it holds across the rows.
#[derive(Clone, Debug)]
pub struct Held {
    /// What it did.
    pub done: Vec<String>,
    /// The loads before the rows.
    pub loads: Vec<Insn>,
    /// The values held to the end.
    pub pinned: Vec<Arg>,
}

/// Schedule: the constant registers' setup (SFPLOADI into L0, SFPCONFIG) moves out of
/// the rows into `setup`, once per stream; LReg 11's to `head`, its restore to `tail`.
pub fn hoist(
    code: &mut Vec<Insn>,
    setup: &mut Vec<Insn>,
    head: &mut Vec<Insn>,
    tail: &mut Vec<Insn>,
) -> Result<Vec<String>, ChainError> {
    let mut done = Vec::new();
    let mut front: Vec<Insn> = Vec::new();
    let mut i = code.len() as isize - 1;
    while i >= 0 {
        let k = i as usize;
        if code[k].op.name != "SFPCONFIG" {
            i -= 1;
            continue;
        }
        let mut j = k;
        while j > 0 && code[j - 1].op.name == "SFPLOADI" {
            if !is_name(&at(&code[j - 1], "lreg_ind"), "L0") {
                break;
            }
            j -= 1;
        }
        let dest = at(&code[k], "config_dest");
        let moved: Vec<Insn> = code.drain(j..=k).collect();
        if is_name(&dest, SAVED_NAME) {
            front.splice(0..0, moved);
        } else {
            setup.splice(0..0, moved);
            done.push(format!("constant register {dest}"));
        }
        i = j as isize - 1;
    }
    if !front.is_empty() {
        *head = front;
        *tail = restore_saved()?;
        done.push("LReg 11: set before the rows, -1.0 again after them".into());
    }
    done.reverse();
    Ok(done)
}

/// LReg 11 back to -1.0 (tt-metal's `_restore_lconst_neg1_`), then the SFPNOP an
/// SFPCONFIG of a constant register takes.
pub fn restore_saved() -> Result<Vec<Insn>, UnitError> {
    let load = [
        ("lreg_ind", name("L0")),
        ("instr_mod0", Arg::Num(0)),
        ("imm16", Arg::Num(NEG_ONE_BF16)),
    ];
    Ok(vec![
        Insn::new("SFPLOADI", &load, "-1.0 into LReg 11 again")?,
        insn(
            "SFPCONFIG",
            &[
                ("imm16_math", 0x5555),
                ("config_dest", i64::from(SAVED)),
                ("instr_mod1", 8),
            ],
            "LReg 11 restored",
        )?,
        insn("SFPNOP", &[], "hazard config-nop: after SFPCONFIG")?,
    ])
}

/// Each value's last instruction in `code` (`pinned` ones: past the end), the values
/// not yet placed in `regs`.
fn last_uses(
    code: &[Insn],
    regs: &HashMap<String, i64>,
    pinned: &[Arg],
) -> Result<HashMap<String, usize>, ChainError> {
    let mut last = HashMap::new();
    for (i, x) in code.iter().enumerate() {
        for v in x.reads().into_iter().chain(x.writes()?) {
            if let Arg::Name(s) = &v {
                if !regs.contains_key(s) && !s.starts_with('L') {
                    let end = if pinned.contains(&v) { code.len() } else { i };
                    last.insert(s.clone(), end);
                }
            }
        }
    }
    Ok(last)
}

fn unique(values: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for v in values {
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

/// Layout: each value an LReg 0 .. 7 for its live range (an operand's that dies there,
/// else the lowest free); fixed constants 9 / 10 / 11, the programmable 12 up, then
/// 11. `pinned` values live to the end of `code`; a register the code names (`L<n>`)
/// is not given to a value live while it is. Returns what it did.
pub fn allocate(
    code: &mut [Insn],
    prog: &[(String, Constant)],
    pinned: &[Arg],
) -> Result<Vec<String>, ChainError> {
    let work = i64::from(WORK);
    let mut regs: HashMap<String, i64> = FIXED
        .iter()
        .map(|(n, r)| ((*n).to_string(), i64::from(*r)))
        .collect();
    for (i, (p, _)) in prog.iter().enumerate() {
        let r = CONSTANT_REGISTERS
            .get(i)
            .ok_or_else(|| plan_error(format!("no constant register for {p}")))?;
        regs.insert(p.clone(), i64::from(*r));
    }
    let last = last_uses(code, &regs, pinned)?;
    let named = named_spans(code)?;
    let fits = |r: i64, i: usize, v: &str| {
        let end = last.get(v).copied().unwrap_or(i);
        named
            .get(&r)
            .is_none_or(|spans| spans.iter().all(|&(s, e)| e < i || s > end))
    };
    let mut free: Vec<i64> = (0..work).collect();
    for (i, x) in code.iter().enumerate() {
        let writes = x.writes()?;
        let dying: Vec<String> = x
            .reads()
            .into_iter()
            .filter_map(|v| match v {
                Arg::Name(s)
                    if last.get(&s) == Some(&i) && !writes.contains(&Arg::Name(s.clone())) =>
                {
                    Some(s)
                }
                _ => None,
            })
            .collect();
        let reg_of = |regs: &HashMap<String, i64>, v: &str| regs.get(v).copied().unwrap_or(work);
        for d in unique(dying.clone()) {
            if reg_of(&regs, &d) < work {
                free.push(reg_of(&regs, &d));
            }
        }
        let written: Vec<String> = unique(
            writes
                .iter()
                .filter_map(|v| match v {
                    Arg::Name(s) => Some(s.clone()),
                    Arg::Num(_) => None,
                })
                .collect(),
        );
        for v in written {
            if v.starts_with('L') {
                continue;
            }
            if !regs.contains_key(&v) {
                let taken: Vec<i64> = dying
                    .iter()
                    .map(|d| reg_of(&regs, d))
                    .filter(|&r| r < work && free.contains(&r) && fits(r, i, &v))
                    .collect();
                let mut fit: Vec<i64> = free.iter().copied().filter(|&r| fits(r, i, &v)).collect();
                fit.sort_unstable();
                let Some(&r) = taken.first().or(fit.first()) else {
                    return Err(ChainError::Unit(
                        "komorebi.dsl.tile: more live values than the 8 work registers".into(),
                    ));
                };
                regs.insert(v.clone(), r);
                if let Some(k) = free.iter().position(|&f| f == r) {
                    free.remove(k);
                }
                if last.get(&v).copied().unwrap_or(i) == i {
                    free.push(r);
                }
            } else if last.get(&v) == Some(&i) {
                let r = reg_of(&regs, &v);
                if r < work && !free.contains(&r) {
                    // rewritten in place (SFPSWAP's smaller value) and read no more
                    free.push(r);
                }
            }
        }
    }
    for x in code.iter_mut() {
        for (field, _) in x.op.args {
            if let Some(Arg::Name(s)) = x.arg(field).cloned() {
                let r = if s.starts_with('L') {
                    named_register(&s)?
                } else {
                    // today's KeyError: no refusal of its own, so not caught as one
                    *regs
                        .get(&s)
                        .ok_or_else(|| plan_error(format!("{s} is read and never written")))?
                };
                x.set(field, Arg::Num(r));
            }
        }
    }
    let mut used: Vec<i64> = regs
        .iter()
        .filter(|(v, r)| !FIXED.iter().any(|(f, _)| f == v) && **r < work)
        .map(|(_, r)| *r)
        .collect();
    used.sort_unstable();
    used.dedup();
    Ok(vec![format!("{} of {WORK} work registers", used.len())])
}

/// Schedule: an unrolled loop's first pass records the first entries of its body into
/// the replay buffer as it runs; the other passes replay them and issue only the rest
/// (rule replay). Returns the loop nest that replaces `face`, and what it did.
pub fn replay(face: Loop) -> Result<(Vec<Item>, Vec<String>), UnitError> {
    let count = match face.count {
        Count::Times(n) => n,
        Count::Runtime(_) => return Ok((vec![Item::Loop(face)], Vec::new())),
    };
    let replayed = insns(&face.body).iter().any(|x| x.op.name == "REPLAY");
    if !face.unroll || count < 2 || replayed {
        return Ok((vec![Item::Loop(face)], Vec::new()));
    }
    let n = face.body.len().min(REPLAY_ENTRIES as usize);
    let record = insn(
        "REPLAY",
        &replay_args(n, 1),
        &format!("record {n} while they run"),
    )?;
    let play = insn("REPLAY", &replay_args(n, 0), &format!("replay {n}"))?;
    let done = vec![format!(
        "{n} of {} instructions of each {} recorded once",
        face.body.len(),
        face.name
    )];
    let mut rest = vec![Item::Insn(play)];
    rest.extend(face.body.iter().skip(n).cloned());
    let mut out = vec![Item::Insn(record)];
    out.extend(face.body);
    out.push(Item::Loop(Loop {
        name: face.name,
        count: Count::Times(count - 1),
        body: rest,
        unroll: true,
    }));
    Ok((out, done))
}

fn replay_args(count: usize, load: i64) -> [(&'static str, i64); 4] {
    [
        ("start_idx", 0),
        ("len", count as i64),
        ("execute_while_loading", load),
        ("load_mode", load),
    ]
}

/// SETRWC's fields, `rwc_cr`, `rwc_d` and `BitMask` given.
fn setrwc(cr: i64, d: i64, mask: i64, note: &str) -> Result<Insn, UnitError> {
    insn(
        "SETRWC",
        &[
            ("clear_ab_vld", 0),
            ("rwc_cr", cr),
            ("rwc_d", d),
            ("rwc_b", 0),
            ("rwc_a", 0),
            ("BitMask", mask),
        ],
        note,
    )
}

/// Rule store_step: the row's closing INCRWC (Dst + `incr`) folded into its last Dst
/// access when that is a store: the store takes STEP_ADDR_MOD (Dst + `incr`). Returns
/// whether it did.
pub fn store_step(row: &mut Vec<Insn>, incr: i64) -> bool {
    match row.last() {
        Some(x) if x.op.name == "INCRWC" && at(x, "rwc_d") == Arg::Num(incr) => {}
        _ => return false,
    }
    let Some(k) = row.iter().rposition(|x| x.arg("dest_reg_addr").is_some()) else {
        return false;
    };
    if row[k].op.name != "SFPSTORE" {
        return false;
    }
    row[k].set("sfpu_addr_mode", Arg::Num(STEP_ADDR_MOD));
    row[k].note = format!("{}, then Dst + {incr}", row[k].note);
    row.pop();
    true
}

fn step_mode(incr: i64) -> Call {
    Call {
        text: format!(
            "ckernel::addr_mod_t{{.srca = {{.incr = 0}}, .srcb = {{.incr = 0}}, .dest = {{.incr = {incr}}}}}.set(ADDR_MOD_{STEP_ADDR_MOD})"
        ),
        note: format!(
            "address mode {STEP_ADDR_MOD}: Dst + {incr} after the row's store (3 SETC16)"
        ),
    }
}

/// The RISC-V's write of the Dst base: tile `index` (C++ text) of the caller's.
fn dst_base(index: &str) -> Call {
    Call {
        text: format!(
            "ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>({index})"
        ),
        note: if index == "dst" {
            "Dst base of tile dst (RISC-V writes)".into()
        } else {
            format!("Dst base of tile {index} (RISC-V writes)")
        },
    }
}

fn call(text: &str, note: &str) -> Item {
    Item::Call(Call {
        text: text.into(),
        note: note.into(),
    })
}

fn sync(cpu: &str, other: &str, text: &str, note: &str) -> Item {
    Item::Sync(Sync {
        cpu: cpu.into(),
        other: other.into(),
        text: text.into(),
        note: note.into(),
    })
}

fn pass(kind: &str, name: &str, done: Vec<String>) -> Pass {
    Pass {
        kind: kind.into(),
        name: name.into(),
        done,
    }
}

fn items(code: Vec<Insn>) -> Vec<Item> {
    code.into_iter().map(Item::Insn).collect()
}

/// The fidelity of a tile function's products with no `tensix.fidelity` hint: today's
/// default (`HiFi4`).
pub const FIDELITY: &str = "HiFi4";

/// The tensor names of matmul node `node`'s two operands (a, w).
fn product_operands(ir: &KernelIR, node: NodeId) -> Result<(String, String), ChainError> {
    let tensor = |n: NodeId| match ir
        .arena
        .get(n.0)
        .map(|n| n.attrs.iter().find(|(k, _)| k == "view"))
    {
        Some(Some((_, Attr::View(v)))) => Ok(v.tensor.clone()),
        _ => Err(ChainError::NotPorted(
            "a product of an operand that is no load".into(),
        )),
    };
    let args = ir
        .arena
        .get(node.0)
        .map(|n| n.args.clone())
        .unwrap_or_default();
    match args[..] {
        [a, w] => Ok((tensor(a)?, tensor(w)?)),
        _ => Err(ChainError::NotPorted("a product of no two operands".into())),
    }
}

/// The math CPU's lines of the unit's tile products (`leaves`: (Dst tile, the matmul
/// node)), in operand order, each the no-MOP product's words after the Dst base of its
/// tile; records the unit's products and the unpack CPU's calls.
fn products(
    ir: &KernelIR,
    unit: &mut UnitProgram,
    leaves: &[(i64, NodeId)],
    fidelity: &str,
) -> Result<Vec<Item>, ChainError> {
    let mut lines = Vec::new();
    let mut first = None;
    for (i, &(k, node)) in leaves.iter().enumerate() {
        let (a, w) = product_operands(ir, node)?;
        first.get_or_insert_with(|| (a.clone(), w.clone()));
        let pairs = tile_products(fidelity, 1, 1, k)?;
        let wormhole = wormhole_products(fidelity, k)?;
        unit.products.push(UnitProduct {
            dst: k,
            a: a.clone(),
            w: w.clone(),
        });
        let index = if k == 0 {
            "dst".to_string()
        } else {
            format!("dst + {k}")
        };
        lines.push(Item::Call(dst_base(&index)));
        let (words, insns): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        lines.push(Item::Product(Product {
            text: format!("{a} @ {w} into Dst tile dst + {k} at {fidelity}"),
            words,
            note: "1 cycle a word; a block's two words 8 apart".into(),
            synced: i == leaves.len() - 1, // the vector unit's stall waits for it
            insns,
            wormhole: wormhole.into_iter().map(|(_, x)| x).collect(),
        }));
        unit.tile[UNPACK].push(call(
            &format!("llk_unpack_AB_matmul({a}, {w}, 0, 0, 1, 1, 1)"),
            &format!("{a} into SrcB, {w} into SrcA (the product of Dst tile {k})"),
        ));
    }
    unit.fidelity = Some(fidelity.to_string());
    let Some((a, w)) = first else {
        return Err(ChainError::NotPorted("a unit of no products".into()));
    };
    unit.init[UNPACK].push(call(
        &format!("mm_no_mop_init_short({a}, {w}, false, 1, 1, 1)"),
        &format!("the products' unpack and address modes, at math_fidelity {fidelity}"),
    ));
    Ok(lines)
}

/// The unit program of a tile function that is one product (`a @ w` stored): the
/// no-MOP product's words alone, as `matrix.product_program` emits them.
fn product_alone(
    ir: &KernelIR,
    node: NodeId,
    fidelity: &str,
    band: bool,
) -> Result<UnitProgram, ChainError> {
    if band {
        return Err(plan_error(format!(
            "{}: the band runs vector code; a product alone runs whole tiles",
            ir.name
        )));
    }
    let mut unit = UnitProgram::new(&ir.name, "fp32");
    unit.operands = vec![None];
    unit.tile[UNPACK] = vec![sync(
        "unpack",
        "reader",
        "cb_wait_front(in, 1)",
        "the operand tiles",
    )];
    let lines = products(ir, &mut unit, &[(0, node)], fidelity)?;
    let words = lines
        .iter()
        .find_map(|l| match l {
            Item::Product(p) => Some(p.words.len()),
            _ => None,
        })
        .unwrap_or(0);
    let mut math = vec![sync(
        "math",
        "pack",
        "tile_regs_acquire()",
        "a free Dst half",
    )];
    math.extend(lines);
    math.push(sync(
        "math",
        "pack",
        "tile_regs_commit()",
        "hands Dst to the pack CPU",
    ));
    unit.tile[MATH] = math;
    unit.tile[PACK] = vec![
        sync("pack", "math", "tile_regs_wait()", "Dst from the math CPU"),
        call("pack_tile(dst, out)", "Dst tile dst to the output buffer"),
        sync(
            "pack",
            "math",
            "tile_regs_release()",
            "frees (and clears) the Dst half",
        ),
        sync("pack", "writer", "cb_push_back(out, 1)", "an output tile"),
    ];
    unit.passes.push(pass(
        "schedule",
        "products",
        vec![format!("1 product at {fidelity}: {words} words")],
    ));
    Ok(unit)
}

/// The tensor a register operand of `chain` loads (none: a tile product's result).
fn operand_tensor(ir: &KernelIR, source: Source) -> Result<Option<String>, ChainError> {
    let Source::Node(n) = source else {
        return Err(ChainError::NotPorted("a stage's operand".into()));
    };
    let node = ir
        .arena
        .get(n.0)
        .ok_or_else(|| ChainError::NotPorted("a node of no kernel".into()))?;
    if node.kind == "matmul" {
        return Ok(None);
    }
    match node.attrs.iter().find(|(k, _)| k == "view") {
        Some((_, Attr::View(v))) => Ok(Some(v.tensor.clone())),
        _ => Err(ChainError::NotPorted(format!(
            "an operand that is a {}",
            node.kind
        ))),
    }
}

/// The unit program of stream kernel `ir` on `dest` Dst (`DEST_MODES`), its hazard
/// checks run: operand k (a load) in Dst tile k, the result in tile 0.
/// `load_macro`: try the load-macro rule (a refusal is a pass's record); `band`: live
/// rows a run-time word.
pub fn unit_program(
    ir: &KernelIR,
    dest: &str,
    load_macro: bool,
    band: bool,
    fidelity: &str,
) -> Result<UnitProgram, ChainError> {
    let Some(&(_, mode)) = DEST_MODES.iter().find(|(d, _)| *d == dest) else {
        return Err(plan_error(format!("no unit program for a {dest} Dst")));
    };
    if ir.stores.len() > 1 {
        return Err(ChainError::NotPorted(
            "a function of several results".into(),
        ));
    }
    let mut chain = stream_chain(ir)?;
    let root = ir.stores.first().map(|s| s.value);
    if let Some(r) = root.filter(|r| ir.arena.get(r.0).is_some_and(|n| n.kind == "matmul")) {
        return product_alone(ir, r, fidelity, band);
    }
    if chain.prev.is_some() {
        return Err(plan_error(format!(
            "{}: the unit program reads at most {OPERANDS} operand tiles",
            ir.name
        )));
    }
    let mut tensors = Vec::new();
    for leaf in &chain.operands {
        tensors.push(operand_tensor(ir, leaf.source)?);
    }
    let mut products_at = Vec::new();
    for (k, (leaf, tensor)) in chain.operands.iter().zip(&tensors).enumerate() {
        if let (None, Source::Node(n)) = (tensor, leaf.source) {
            products_at.push((k as i64, n));
        }
    }
    let leaves = products_at.len();
    if leaves > OPERANDS {
        return Err(plan_error(format!(
            "{}: {leaves} products: Dst holds {OPERANDS} operand tiles of a body (a product's result stays in Dst for the vector code)",
            ir.name
        )));
    }
    if chain.operands.len() > OPERANDS {
        return Err(plan_error(format!(
            "{}: the unit program reads at most {OPERANDS} operand tiles",
            ir.name
        )));
    }
    let of = |t: &super::ir::TensorParam, family: &[&str]| {
        t.dtype.as_deref().is_some_and(|d| family.contains(&d))
    };
    if dest == "fp16a"
        && !ir
            .tensors
            .iter()
            .filter(|t| !t.output)
            .all(|t| of(t, &A_DTYPES))
    {
        return Err(plan_error(format!(
            "{}: a 16-bit Dst's tensors are of the A family",
            ir.name
        )));
    }
    if dest == "fp32" && !ir.tensors.iter().all(|t| of(t, &TILE_DTYPES)) {
        return Err(plan_error(format!(
            "{}: tensors are fp32 or of the B family ({})",
            ir.name,
            TILE_DTYPES[1..].join(", ")
        )));
    }
    let per_row: Vec<usize> = (0..chain.operands.len())
        .filter(|&k| chain.operands[k].kind == "bcast_col")
        .collect();
    let mut body = Body {
        mode,
        ..Body::default()
    };
    let FnResult::One(result) = chain.func.result else {
        return Err(ChainError::NotPorted(
            "a function of even and odd results".into(),
        ));
    };
    let result = chain.vs.wrap(result);
    if per_row.is_empty() {
        let stored = value(&mut body, &chain.vs, result, &mut Vec::new())?;
        body.store(stored, 0, "the result to Dst")?;
    } else {
        // A per-row operand (a state tile, each row's value in all its lanes): a row
        // step takes a 4-row block's even and odd vectors, the state loaded once.
        body.rows = Some(Rows {
            per_row: per_row.clone(),
            ..Rows::default()
        });
        let mut results = Vec::new();
        for half in [0, 1] {
            if let Some(r) = &mut body.rows {
                r.half = half;
            }
            results.push(value(&mut body, &chain.vs, result, &mut Vec::new())?);
        }
        body.rows = None;
        for (half, stored) in results.into_iter().enumerate() {
            let note = if half == 0 { "even" } else { "odd" };
            let note = format!("the result to Dst ({note})");
            body.store(stored, 2 * half as i64, &note)?;
        }
    }
    let mut unit = UnitProgram::new(&ir.name, dest);
    unit.operands = tensors;
    unit.row_operands = (0..chain.operands.len())
        .filter(|&k| chain.operands[k].kind == "bcast_row")
        .collect();
    unit.col_operands = per_row.clone();
    unit.results = ir.stores.len() as i64;
    let templates = chain
        .templates
        .iter()
        .map(|(k, v)| format!("{k}: {v}"))
        .collect();
    unit.passes.push(pass("algebra", "templates", templates));
    let done = fold_negation(&mut body.code)?;
    unit.passes.push(pass("algebra", "fold_negation", done));
    let done = fuse_mad(&mut body.code)?;
    unit.passes.push(pass("algebra", "fuse_mad", done));
    if !per_row.is_empty() {
        // The even and odd vectors of a block are two independent chains.
        let done = drop_copies(&mut body.code)?;
        unit.passes.push(pass("algebra", "drop_copies", done));
        let done = common_values(&mut body.code)?;
        unit.passes.push(pass("algebra", "common_values", done));
        let mut trial = body.code.clone();
        let addresses: Vec<i64> = per_row.iter().map(|&k| TILE * k as i64).collect();
        let (pre, mut done) = per_row_prepass(&mut trial, &addresses)?;
        if let Some(pre) = pre {
            let mut estimate = UnitProgram::new("estimate", dest);
            estimate.tile[MATH] = prepass_loop(&pre, &[])?;
            let spent = estimate.cost(MATH, 32)?.cycles.unwrap_or(0.0);
            let saved = BLOCKS * (body.code.len() as i64 - trial.len() as i64);
            done.push(format!(
                "{saved} words a tile saved against {spent:.1} cycles of the pass"
            ));
            if spent < saved as f64 {
                return Err(ChainError::NotPorted(
                    "a per-row pre-pass cheaper than the rows".into(),
                ));
            }
            done.push("kept in the rows (cheaper)".into());
        }
        unit.passes.push(pass("algebra", "per_row_prepass", done));
        let end = body.code.len();
        let moved = interleave(&mut body.code, 0, end)?;
        let done = vec![format!(
            "{moved} instructions moved in the even and odd chains"
        )];
        unit.passes.push(pass("schedule", "interleave", done));
    }
    let (scalars, names) = scalar_loads(ir, &chain, &body)?;
    unit.scalars = names;
    let done = negated_zero(&mut body.code)?;
    unit.passes.push(pass("algebra", "negated_zero", done));
    let done = constant_registers(&mut body.code, &mut body.prog, CONSTANT_REGISTERS.len())?;
    unit.passes.push(pass("layout", "constant_registers", done));
    let (mut setup, mut head, mut tail) = (Vec::new(), Vec::new(), Vec::new());
    let done = hoist(&mut body.code, &mut setup, &mut head, &mut tail)?;
    unit.passes.push(pass("schedule", "hoist", done));
    let head_len = head.len();
    let mut saved: Vec<Insn> = head.into_iter().chain(tail).collect();
    allocate(&mut saved, &body.prog, &[])?;
    let tail = saved.split_off(head_len);
    let head = saved;
    let pins: Vec<Arg> = scalars.iter().map(|x| at(x, "lreg_ind")).collect();
    let Held {
        done,
        loads: held,
        pinned,
    } = resident_constants(&mut body.code, &body.prog, scalars, pins)?;
    unit.passes.push(pass("layout", "resident_constants", done));
    let mut whole: Vec<Insn> = held.iter().chain(&body.code).cloned().collect();
    let done = allocate(&mut whole, &body.prog, &pinned)?;
    unit.passes.push(pass("layout", "allocate", done));
    body.code = whole.split_off(held.len());
    let held = whole;
    let mut held_regs: Vec<i64> = held
        .iter()
        .filter_map(|x| at(x, "lreg_ind").num())
        .collect();
    held_regs.sort_unstable();
    held_regs.dedup();
    let mut clobbered = Vec::new();
    for x in &body.code {
        if x.writes()?
            .iter()
            .any(|w| w.num().is_some_and(|r| held_regs.contains(&r)))
        {
            clobbered.push(py::str_repr(x.op.name));
        }
    }
    unit.checks.push(Check {
        rule: "resident".into(),
        text: "a register held across rows is written by no row instruction".into(),
        verdict: if clobbered.is_empty() {
            format!("ok: LReg {}", listed(&held_regs))
        } else {
            format!("refused: [{}]", clobbered.join(", "))
        },
    });
    if !clobbered.is_empty() {
        return Err(ChainError::Unit(format!(
            "{}: a held constant register is overwritten",
            ir.name
        )));
    }
    allocate(&mut setup, &body.prog, &[])?;
    unit.constants = body
        .prog
        .iter()
        .zip(CONSTANT_REGISTERS)
        .map(|((_, v), r)| (r, v.clone()))
        .collect();
    let mut plan: Option<Plan> = None;
    if load_macro && dest != "fp32" {
        unit.passes.push(pass(
            "schedule",
            "load_macro",
            vec![format!("refused: {dest} Dst")],
        ));
    } else if load_macro && per_row.is_empty() {
        let fitted = r#macro::fit(&mut body.code).and_then(|p| {
            let rows = p.rows();
            if rows == 0 || ROWS % rows != 0 {
                return Err(NoFit::Refused(format!(
                    "{rows} rows a macro turn do not divide a face"
                )));
            }
            if band && (ROWS / rows) % BANDS != 0 {
                return Err(NoFit::Refused(format!(
                    "a macro turn of {rows} row steps; a band is 2"
                )));
            }
            Ok(p)
        });
        match fitted {
            Ok(p) => plan = Some(p),
            Err(NoFit::Refused(why)) => {
                unit.passes.push(pass(
                    "schedule",
                    "load_macro",
                    vec![format!("refused: {why}")],
                ));
            }
            Err(NoFit::Error(e)) => return Err(e.into()),
        }
    }
    let mut init = vec![
        Item::Insn(insn(
            "SFPCONFIG",
            &[("imm16_math", 0), ("config_dest", 15), ("instr_mod1", 1)],
            "lane configuration",
        )?),
        call(
            "ckernel::addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7)",
            "address mode 7: no increments (3 SETC16)",
        ),
        Item::Insn(setrwc(0, 0, 0xF, "reset the counters")?),
    ];
    init.extend(items(setup));
    let (face, drained) = match &plan {
        None => {
            let (incr, note, rows) = if per_row.is_empty() {
                (2, "next 4 rows", ROWS)
            } else {
                (4, "next 4-row block", ROWS / 2)
            };
            let mut row = std::mem::take(&mut body.code);
            row.push(insn(
                "INCRWC",
                &[("rwc_cr", 0), ("rwc_d", incr), ("rwc_b", 0), ("rwc_a", 0)],
                note,
            )?);
            let stepped = store_step(&mut row, incr);
            if stepped {
                init.push(Item::Call(step_mode(incr)));
            }
            let done = if stepped {
                vec![format!("the INCRWC into the last store (Dst + {incr})")]
            } else {
                Vec::new()
            };
            unit.passes.push(pass("schedule", "store_step", done));
            let face = Loop {
                name: "row".into(),
                count: Count::Times(rows),
                body: items(row),
                unroll: true,
            };
            (face, Vec::new())
        }
        Some(p) => {
            let face = Loop {
                name: "row".into(),
                count: Count::Times(ROWS / p.rows()),
                body: items(r#macro::rows(p, MACRO_ADDR_MOD)?),
                unroll: true,
            };
            init.push(call(
                "ckernel::addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6)",
                "address mode 6: Dst + 2 after a macro (3 SETC16)",
            ));
            init.extend(items(r#macro::setup(p)?));
            unit.passes.push(pass(
                "schedule",
                "load_macro",
                vec![format!(
                    "{} operation(s) and the store in one SFPLOADMACRO a row (store delay {}); {} plain load(s) a row; registers in turns of {} (loaded) and {} (plain)",
                    p.chain.len(),
                    p.store_delay,
                    p.others.len(),
                    p.loaded_turns,
                    p.plain_turns
                )],
            ));
            (face, r#macro::drain(p)?)
        }
    };
    unit.init[1] = init;
    let step = setrwc(CR_D, 8, SET_D, "Dst + 8 rows")?;
    let copies: Vec<String> = unit
        .operands
        .iter()
        .enumerate()
        .filter(|(_, t)| t.is_some())
        .map(|(k, _)| format!("copy_tile(in{k}, 0, dst + {k})"))
        .collect();
    unit.tile[UNPACK] = vec![sync(
        "unpack",
        "reader",
        "cb_wait_front(in, 1)",
        "the input tiles",
    )];
    let multiplied = if leaves > 0 {
        products(ir, &mut unit, &products_at, fidelity)?
    } else {
        Vec::new()
    };
    unit.tile[UNPACK].extend(
        copies
            .iter()
            .map(|c| call(c, "the tile into Dst (fp32, unpacked to Dst)")),
    );
    unit.tile[UNPACK].push(call("cb_pop_front(in, 1)", "frees the input tiles"));
    if leaves > 0 {
        let words: usize = multiplied
            .iter()
            .map(|x| match x {
                Item::Product(p) => p.words.len(),
                _ => 0,
            })
            .sum();
        unit.passes.push(pass(
            "schedule",
            "products",
            vec![format!(
                "{leaves} product(s) at {fidelity} before the vector code ({words} matrix words), the vector unit's stall on the matrix unit after them"
            )],
        ));
    }
    let mut math = vec![sync(
        "math",
        "pack",
        "tile_regs_acquire()",
        "a free Dst half",
    )];
    math.extend(multiplied);
    math.extend(
        copies
            .iter()
            .map(|c| call(c, "the math part of the copy (none to Dst)")),
    );
    math.push(Item::Call(dst_base("dst")));
    math.push(Item::Insn(insn(
        "STALLWAIT",
        &[("stall_res", STALL_SFPU), ("wait_res", WAIT_MATH)],
        if leaves > 0 {
            "SFPU waits for the matrix unit (the products)"
        } else {
            "SFPU waits for the matrix unit"
        },
    )?));
    math.extend(items(head));
    math.extend(items(held));
    math.push(Item::Loop(Loop {
        name: "face".into(),
        count: Count::Times(FACES),
        body: vec![Item::Loop(face), Item::Insn(step.clone()), Item::Insn(step)],
        unroll: false,
    }));
    math.extend(items(drained));
    math.extend(items(tail));
    math.push(Item::Insn(setrwc(0, 0, SET_D, "Dst counter back to 0")?));
    math.push(sync(
        "math",
        "pack",
        "tile_regs_commit()",
        "hands Dst to the pack CPU",
    ));
    unit.tile[1] = math;
    unit.tile[2] = vec![
        sync("pack", "math", "tile_regs_wait()", "Dst from the math CPU"),
        call("pack_tile(dst, out)", "Dst tile dst to the output buffer"),
        sync("pack", "math", "tile_regs_release()", "frees the Dst half"),
        sync("pack", "writer", "cb_push_back(out, 1)", "an output tile"),
    ];
    hazard_checks(&mut unit)?;
    let Some(faces) = unit.tile[1].iter_mut().find_map(|x| match x {
        Item::Loop(l) => Some(l),
        _ => None,
    }) else {
        return Err(ChainError::Unit("a unit program of no face loop".into()));
    };
    let Some(Item::Loop(face)) = faces.body.first().cloned() else {
        return Err(ChainError::Unit("a face loop of no row loop".into()));
    };
    let (rows, done) = replay(face)?;
    unit.passes.push(pass("schedule", "replay", done));
    if band {
        return Err(ChainError::NotPorted(
            "the band (a body on the live rows)".into(),
        ));
    }
    faces.body.splice(0..1, rows);
    unit.not_checked = NOT_CHECKED
        .iter()
        .filter(|r| plan.is_none() || !r.contains("load-macro"))
        .map(|r| (*r).to_string())
        .collect();
    // a call's tiles: its operands' and results'
    let tiles = (unit.operands.len() as i64).max(unit.results);
    check_dst_window(&mut unit, tiles)?;
    Ok(unit)
}

/// The tile products of a stream kernel (today's `lower.products_of`), each refused
/// unless the unpack can feed it: both operands a 32 x 32 tile read from a buffer.
pub fn products_of(
    ir: &KernelIR,
    nodes: &[NodeId],
    shapes: &mut Vec<Option<&'static str>>,
) -> Result<Vec<NodeId>, ChainError> {
    let mut out = Vec::new();
    for &n in nodes {
        let Some(node) = ir.arena.get(n.0) else {
            continue;
        };
        if node.kind != "matmul" {
            continue;
        }
        for (side, &arg) in ["left", "right"].iter().zip(&node.args) {
            let kind = ir.arena.get(arg.0).map_or("", |a| a.kind.as_str());
            if kind != "load" && kind != "resident" {
                let made = if kind == "matmul" {
                    "a product, its result in Dst".to_string()
                } else {
                    format!("a {kind}, computed on the vector unit")
                };
                return Err(ChainError::Plan(
                    format!(
                        "{}: a product after vector code: the {side} operand of `@` is {made}, and the matrix unit reads its operands unpacked from a buffer into SrcA and SrcB; give that value to a second body (pack, then unpack)",
                        ir.name
                    ),
                    node.line.clone(),
                ));
            }
            if shape(ir, arg, shapes)? != "tile" {
                return Err(ChainError::Plan(
                    format!(
                        "{}: the unpack feeds a product one 32 x 32 tile a side (x.tiles[t]); the {side} operand of `@` is a row block",
                        ir.name
                    ),
                    node.line.clone(),
                ));
            }
        }
        out.push(n);
    }
    Ok(out)
}

/// The Chain of each store of a stream kernel (today's `_stream_chains`): one compiler,
/// so one record of the templates, which every chain carries whole.
pub fn stream_chains(ir: &KernelIR) -> Result<Vec<Chain>, ChainError> {
    let mut shapes = vec![None; ir.arena.len()];
    let nodes = ir.nodes();
    let products = products_of(ir, &nodes, &mut shapes)?;
    for &n in &nodes {
        shape(ir, n, &mut shapes)?;
    }
    let mut compiler = Compiler::new(ir, shapes, products);
    let mut chains = Vec::new();
    for store in &ir.stores {
        chains.push(compiler.chain(store.value)?);
    }
    for chain in &mut chains {
        let mut stage = Some(chain);
        while let Some(c) = stage {
            c.templates = compiler.templates().to_vec();
            stage = c.prev.as_deref_mut();
        }
    }
    Ok(chains)
}

/// The Chain of a stream kernel's one store (today's `lower.stream_chain`).
pub fn stream_chain(ir: &KernelIR) -> Result<Chain, ChainError> {
    if ir.stores.len() != 1 {
        return Err(ChainError::Plan(
            format!("{}: the unit program takes one store", ir.name),
            None,
        ));
    }
    stream_chains(ir)?
        .pop()
        .ok_or_else(|| ChainError::NotPorted("a kernel of no store".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exps_constants_are_todays_bits() {
        // Python: (1.442695).hex(), and so on, of lower.exp's literals
        let hex: Vec<String> = [EXP_INV_LN2, EXP_START, EXP_LN2_HIGH, EXP_LN2_LOW]
            .into_iter()
            .chain(EXP_POLYNOMIAL)
            .map(py::float_hex)
            .collect();
        assert_eq!(
            hex,
            [
                "0x1.715475a31a4bep+0",
                "0x1.6940000c489a7p-10",
                "-0x1.62e4000067144p-1",
                "-0x1.7f7d1c1507ed8p-20",
                "0x1.125edbfd47362p-7",
                "0x1.555b5a027abaap-5",
                "0x1.5554500e60400p-3",
                "0x1.fffff6003311ep-2",
            ]
        );
    }

    #[test]
    fn the_vector_functions_constants_are_todays_bits() {
        // Python: hex(0x5F1110A0), and (2.2825186).hex() and so on of lower.py's
        // rsqrt literals, TANH_SMALL, TANH_P, ERF_N, ERF_D, LOG_P and LN2_FP32
        assert_eq!(format!("{RSQRT_MAGIC:#x}"), "0x5f1110a0");
        let hex: Vec<String> = [RSQRT_C1, RSQRT_C2, TANH_SMALL]
            .into_iter()
            .chain(TANH_P)
            .chain(ERF_N)
            .chain(ERF_D)
            .chain(LOG_P)
            .chain([LN2_FP32])
            .map(py::float_hex)
            .collect();
        assert_eq!(
            hex,
            [
                "0x1.242991c9c17f9p+1",
                "0x1.206c4b82b5518p+1",
                "0x1.4000000000000p-1",
                "-0x1.55553439ee635p-2",
                "0x1.110768542749ap-3",
                "-0x1.b844483f9b06cp-5",
                "0x1.5256d5e0d8d98p-6",
                "-0x1.76ac1200d94fep-8",
                "0x1.20dd73ffe83e5p+0",
                "0x1.b7db9a000004dp-3",
                "0x1.fcff420002c1cp-5",
                "0x1.708304000355dp-8",
                "0x1.416cebfffd250p-11",
                "0x1.2897d0001c707p-16",
                "0x1.d676f4001ffebp-26",
                "-0x1.3ad0180005adep-32",
                "0x1.0000000000000p+0",
                "0x1.0c1ed5fff98f7p-1",
                "0x1.0974abfff4d6ap-3",
                "0x1.41d2e9ffe7ceep-6",
                "0x1.003000000e4b3p-9",
                "0x1.146286001af46p-13",
                "0x1.6129b60012d94p-20",
                "-0x1.e326c000054c9p-29",
                "-0x1.d9f0480000ce5p-38",
                "-0x1.0000000000000p-1",
                "0x1.5555720000000p-2",
                "-0x1.00001a0000000p-2",
                "0x1.9980000000000p-3",
                "-0x1.5500000000000p-3",
                "0x1.2740000000000p-3",
                "-0x1.0c40000000000p-3",
                "0x1.b840000000000p-4",
                "-0x1.92c0000000000p-5",
                "0x1.62e42ffc0255bp-1",
            ]
        );
    }
}
