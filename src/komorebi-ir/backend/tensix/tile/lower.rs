//! A traced tile expression over a stream of tiles to its unit program (today's
//! `komorebi.dsl.tile.lower`): the chain's vector function is written as vector-unit
//! instructions on named values, the rules run as passes (algebra, layout, schedule),
//! then the hazard checks (`checks`); and the stream form's entry to the chain
//! compiler (`stream_chain`).
//!
//! Ported: the whole-tile stream form of one result on `+`, `-`, `*`, negation,
//! `abs`, multiply-add, `min` / `max`, selects, constants, run-time scalars and the
//! vector functions `exp`, `recip` and `sigmoid`, with every pass of that path and the
//! load macro. Refused by name (K964) where it would act: the other vector functions,
//! `flash_exp2`, several results, per-row operands, tile products and the band. Not on this path, so not ported: `drop_copies`, `common_values`, the per-row
//! pre-pass and `allocate`'s high registers.

use std::collections::HashMap;

use super::chain::{shape, Chain, ChainError, Compiler, Source};
use super::checks::{
    check_dst_window, hazard_checks, named_register, named_spans, NEG_ONE_BF16, STALL_SFPU, TILE,
    WAIT_MATH,
};
use super::ir::{Attr, KernelIR, NodeId};
use super::py;
use super::r#macro::{self, NoFit, Plan};
use super::unit::{
    insns, op_named, Arg, Call, Check, Constant, Count, Insn, Item, Loop, Pass, Sync, UnitError,
    UnitProgram, FIXED, IADD_IMM, IADD_NO_CC, PROGRAMMABLE, REPLAY_ENTRIES, SAVED, WORK,
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
/// The vector functions today's unit program builds (`PORTED` of them so far).
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
}

impl Body {
    fn fresh(&mut self) -> String {
        self.count += 1;
        format!("v{}", self.count)
    }

    /// Appends `name`; a written register argument left out gets a new value, any
    /// other field 0. Returns the value written.
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
            if !all.iter().any(|(n, _)| n == w) {
                let v = self.fresh();
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
        let constant = Constant::Float(value);
        let note = constant.repr();
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

/// The vector functions lowered so far (of `FUNCTIONS`).
const PORTED: [&str; 3] = ["exp", "recip", "sigmoid"];

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
        ("call", VValue::Call(f)) if PORTED.contains(f) => {
            let Some(x) = args(b, done)?.into_iter().next() else {
                return Err(ChainError::NotPorted(format!("a call of {f} on no value")));
            };
            match *f {
                "exp" => exp(b, x)?,
                "recip" => recip(b, x)?,
                _ => sigmoid(b, x)?,
            }
        }
        ("call", VValue::Call(f)) if FUNCTIONS.contains(f) => {
            return Err(ChainError::NotPorted(format!("the vector function {f}")))
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
    if root
        .and_then(|r| ir.arena.get(r.0))
        .is_some_and(|n| n.kind == "matmul")
    {
        return Err(ChainError::NotPorted(
            "a function that is one tile product".into(),
        ));
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
    let leaves = tensors.iter().filter(|t| t.is_none()).count();
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
    if chain.operands.iter().any(|l| l.kind != "copy") {
        return Err(ChainError::NotPorted(
            "per-row and per-column operands".into(),
        ));
    }
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
    let stored = value(&mut body, &chain.vs, result, &mut Vec::new())?;
    body.store(stored, 0, "the result to Dst")?;
    let mut unit = UnitProgram::new(&ir.name, dest);
    unit.operands = tensors;
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
    } else if load_macro {
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
            let incr = 2;
            let mut row = std::mem::take(&mut body.code);
            row.push(insn(
                "INCRWC",
                &[("rwc_cr", 0), ("rwc_d", incr), ("rwc_b", 0), ("rwc_a", 0)],
                "next 4 rows",
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
                count: Count::Times(ROWS),
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
    if leaves > 0 {
        return Err(ChainError::NotPorted(
            "tile products before the vector code".into(),
        ));
    }
    let mut unpack = vec![sync(
        "unpack",
        "reader",
        "cb_wait_front(in, 1)",
        "the input tiles",
    )];
    unpack.extend(
        copies
            .iter()
            .map(|c| call(c, "the tile into Dst (fp32, unpacked to Dst)")),
    );
    unpack.push(call("cb_pop_front(in, 1)", "frees the input tiles"));
    unit.tile[0] = unpack;
    let mut math = vec![sync(
        "math",
        "pack",
        "tile_regs_acquire()",
        "a free Dst half",
    )];
    math.extend(
        copies
            .iter()
            .map(|c| call(c, "the math part of the copy (none to Dst)")),
    );
    math.push(Item::Call(dst_base("dst")));
    math.push(Item::Insn(insn(
        "STALLWAIT",
        &[("stall_res", STALL_SFPU), ("wait_res", WAIT_MATH)],
        "SFPU waits for the matrix unit",
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
}
