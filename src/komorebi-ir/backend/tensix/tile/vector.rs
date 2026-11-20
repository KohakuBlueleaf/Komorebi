//! The vector unit's values (today's `komorebi.dsl.tile.sfpu`): expressions of 32 lanes
//! of fp32 (`V`), the library functions of one vector, and `sfpu_fn`, which writes one
//! inline SFPI function computing an expression and checks it as today does. Every
//! `+`, `-` and `*` rounds on its own; `mad` is the fused multiply-add.
//!
//! Not ported: the row form's functions (`_banded`, `tiles`, `join_cols`, `put_rows`,
//! `row_reduce`), operands read along a row or column, and Kohaku-TT's flash exp; the
//! chain compiler makes none of them in the stream form.

use std::collections::BTreeSet;

use super::py;

/// Most register operands of one function: two broadcasts may share a slot.
pub const OPERANDS: usize = 6;
/// Most fp32 scalar arguments of one function.
pub const SCALARS: usize = 4;
/// The parameters naming a function's slots, in order (`out` third).
pub const ARGS: &[&str] = &["a", "b", "out", "c", "d", "e", "f"];
/// The library's vector-unit headers' prefix.
pub const LLK: &str = "ckernel_sfpu_";
/// 1.0 as a vector.
pub const ONE: &str = "sfpi::vFloat(1.0f)";
/// A scalar argument's bits read as a float.
pub const CONVERTER: &str = "ckernel::sfpu::Converter::as_float";
/// SFPI's comparison operators by today's names.
pub const COMPARES: &[(&str, &str)] = &[
    ("lt", "<"),
    ("le", "<="),
    ("gt", ">"),
    ("ge", ">="),
    ("eq", "=="),
    ("ne", "!="),
];

/// A library function of one vector: its C++ call on `{x}`, its header, the table op
/// whose setup it runs under (none: any), and whether it is for fp32 registers only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Function {
    /// The call.
    pub call: &'static str,
    /// The header, after `LLK`.
    pub header: &'static str,
    /// The setup's table op.
    pub init: Option<&'static str>,
    /// fp32 registers only.
    pub fp32: bool,
}

const fn function(
    call: &'static str,
    header: &'static str,
    init: Option<&'static str>,
    fp32: bool,
) -> Function {
    Function {
        call,
        header,
        init,
        fp32,
    }
}

/// The library functions by name, in today's order.
pub const FUNCTIONS: &[(&str, Function)] = &[
    (
        "sigmoid",
        function(
            "ckernel::sfpu::_sfpu_sigmoid_<DST_ACCUM_MODE>({x})",
            "sigmoid.h",
            Some("silu"),
            false,
        ),
    ),
    (
        "recip",
        function(
            "ckernel::sfpu::sfpu_reciprocal<false>({x})",
            "recip.h",
            Some("silu"),
            false,
        ),
    ),
    (
        "exp",
        function(
            "ckernel::sfpu::_sfpu_exp_accurate_<DST_ACCUM_MODE>({x})",
            "exp.h",
            None,
            false,
        ),
    ),
    (
        "tanh",
        function(
            "ckernel::sfpu::_sfpu_tanh_fp32_accurate_({x})",
            "tanh.h",
            None,
            true,
        ),
    ),
    (
        "erf",
        function(
            "ckernel::sfpu::piecewise_rational_eval<ERF_NUM_DEGREE, ERF_DEN_DEGREE, ERF_NUM_SEGMENTS, ERF_LUT_SIZE, true, false>(ERF_LUT, sfpi::symmetric_clamp({x}, 10.0f))",
            "erf.h",
            None,
            true,
        ),
    ),
    (
        "log",
        function(
            "ckernel::sfpu::calculate_log_body<false, false, true>({x}, 0)",
            "log.h",
            None,
            false,
        ),
    ),
    (
        "sqrt",
        function(
            "ckernel::sfpu::_calculate_sqrt_body_<APPROX, false>({x})",
            "sqrt.h",
            Some("sqrt"),
            false,
        ),
    ),
    (
        "rsqrt",
        function(
            "ckernel::sfpu::_calculate_sqrt_body_<APPROX, true>({x})",
            "sqrt.h",
            Some("rsqrt"),
            false,
        ),
    ),
];

/// The library function named `name`.
pub fn function_named(name: &str) -> Option<&'static Function> {
    FUNCTIONS.iter().find(|(n, _)| *n == name).map(|(_, f)| f)
}

/// A vector value's number in its expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VId(pub usize);

/// What a vector value holds besides its operands.
#[derive(Clone, Debug, PartialEq)]
pub enum VValue {
    /// Nothing.
    None,
    /// Register operand `index`, whole or one part (0 even, 1 odd columns).
    Operand(usize, Option<u8>),
    /// A constant.
    Const(f64),
    /// Scalar argument `index`.
    Scalar(usize),
    /// A library function, by name.
    Call(&'static str),
    /// A select's comparison: its kind and its two sides.
    Cond(&'static str, VId, VId),
}

/// A vector value: an operand, a constant, a scalar, or an operation on others.
#[derive(Clone, Debug, PartialEq)]
pub struct VNode {
    /// `operand`, `const`, `scalar`, `add`, `sub`, `mul`, `mad`, `neg`, `abs`,
    /// `min`, `max`, `select`, `call`.
    pub kind: &'static str,
    /// The operands.
    pub args: Vec<VId>,
    /// The rest.
    pub value: VValue,
}

/// A value of the chain compiler: a vector value, or a Python number not yet made one
/// (today's compiler computes with numbers until an operation meets a vector).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Val {
    /// A vector value.
    V(VId),
    /// A number.
    F(f64),
}

/// The vector values of one function, by number; each made once (today's objects).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Vs {
    /// The values.
    pub nodes: Vec<VNode>,
}

impl Vs {
    fn push(&mut self, kind: &'static str, args: Vec<VId>, value: VValue) -> VId {
        self.nodes.push(VNode { kind, args, value });
        VId(self.nodes.len() - 1)
    }

    /// `v` as a vector value: a number becomes a constant.
    pub fn wrap(&mut self, v: Val) -> VId {
        match v {
            Val::V(id) => id,
            Val::F(x) => self.push("const", vec![], VValue::Const(x)),
        }
    }

    /// The value numbered `id`.
    pub fn get(&self, id: VId) -> Option<&VNode> {
        self.nodes.get(id.0)
    }

    fn binary(&mut self, kind: &'static str, a: Val, b: Val, fold: fn(f64, f64) -> f64) -> Val {
        if let (Val::F(x), Val::F(y)) = (a, b) {
            return Val::F(fold(x, y));
        }
        let (a, b) = (self.wrap(a), self.wrap(b));
        Val::V(self.push(kind, vec![a, b], VValue::None))
    }

    /// `a + b` (two numbers: Python's sum).
    pub fn add(&mut self, a: Val, b: Val) -> Val {
        self.binary("add", a, b, |x, y| x + y)
    }

    /// `a - b`.
    pub fn sub(&mut self, a: Val, b: Val) -> Val {
        self.binary("sub", a, b, |x, y| x - y)
    }

    /// `a * b`.
    pub fn mul(&mut self, a: Val, b: Val) -> Val {
        self.binary("mul", a, b, |x, y| x * y)
    }

    /// `-a`.
    pub fn neg(&mut self, a: Val) -> Val {
        match a {
            Val::F(x) => Val::F(-x),
            Val::V(id) => Val::V(self.push("neg", vec![id], VValue::None)),
        }
    }

    /// `abs(a)`.
    pub fn abs(&mut self, a: Val) -> Val {
        match a {
            Val::F(x) => Val::F(x.abs()),
            Val::V(id) => Val::V(self.push("abs", vec![id], VValue::None)),
        }
    }

    /// The lanewise smaller of two values.
    pub fn minimum(&mut self, a: Val, b: Val) -> Val {
        let (a, b) = (self.wrap(a), self.wrap(b));
        Val::V(self.push("min", vec![a, b], VValue::None))
    }

    /// The lanewise larger of two values.
    pub fn maximum(&mut self, a: Val, b: Val) -> Val {
        let (a, b) = (self.wrap(a), self.wrap(b));
        Val::V(self.push("max", vec![a, b], VValue::None))
    }

    /// `then` where `a kind b` holds, `other` elsewhere (`a` a vector value).
    pub fn select(&mut self, kind: &'static str, a: VId, b: Val, then: Val, other: Val) -> Val {
        let b = self.wrap(b);
        let (then, other) = (self.wrap(then), self.wrap(other));
        Val::V(self.push("select", vec![then, other], VValue::Cond(kind, a, b)))
    }

    /// Library function `name` of `x`.
    pub fn call(&mut self, name: &'static str, x: Val) -> Val {
        let x = self.wrap(x);
        Val::V(self.push("call", vec![x], VValue::Call(name)))
    }

    /// x * sigmoid(x), as the silu tile op computes it.
    pub fn silu(&mut self, x: Val) -> Val {
        let x = Val::V(self.wrap(x));
        let s = self.call("sigmoid", x);
        self.mul(x, s)
    }

    /// max(x, 0).
    pub fn relu(&mut self, x: Val) -> Val {
        self.maximum(x, Val::F(0.0))
    }

    /// x * x.
    pub fn square(&mut self, x: Val) -> Val {
        let x = Val::V(self.wrap(x));
        self.mul(x, x)
    }

    /// Register operand `index` of a function (0 .. 5), or why not.
    pub fn operand(&mut self, index: usize) -> Result<VId, String> {
        if index >= OPERANDS {
            return Err(format!("a function has operands 0 .. {}", OPERANDS - 1));
        }
        Ok(self.push("operand", vec![], VValue::Operand(index, None)))
    }

    /// Scalar argument `index` of a function (0 .. 3), or why not.
    pub fn scalar(&mut self, index: usize) -> Result<VId, String> {
        if index >= SCALARS {
            return Err(format!("a function has scalars 0 .. {}", SCALARS - 1));
        }
        Ok(self.push("scalar", vec![], VValue::Scalar(index)))
    }
}

/// The C++ fp32 literal with exactly the bits of `value` rounded to fp32, or why not.
pub fn literal(value: f64) -> Result<String, String> {
    let exact = value as f32;
    if !exact.is_finite() {
        return Err("a vector constant is a finite number".to_string());
    }
    Ok(format!("{}f", py::float_hex(f64::from(exact))))
}

/// A lowered value's key: an operand, a constant (by its literal: 0.0 and -0.0 apart)
/// and a scalar are one variable each; any other value is one by its number.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Key {
    Operand(usize, Option<u8>),
    Const(String),
    Scalar(usize),
    Node(VId),
}

/// The statements computing a set of values, each value once.
struct Lowering<'v> {
    vs: &'v Vs,
    lines: Vec<String>,
    names: Vec<(Key, String)>,
    used: BTreeSet<usize>,
    calls: BTreeSet<&'static str>,
    scalars: BTreeSet<usize>,
    /// The operands in the order first read (all whole: no row or column layout).
    alongs: Vec<usize>,
}

impl Lowering<'_> {
    fn new_name(&mut self, text: String) -> String {
        let name = format!("t{}", self.names.len());
        self.lines.push(format!("sfpi::vFloat {name} = {text};"));
        name
    }

    fn node(&self, id: VId) -> Result<&VNode, String> {
        self.vs
            .get(id)
            .ok_or_else(|| "a vector value of no function".to_string())
    }

    fn text(&mut self, id: VId) -> Result<String, String> {
        let node = self.node(id)?.clone();
        let mut args = Vec::new();
        for &a in &node.args {
            args.push(self.value(a)?);
        }
        let arg = |k: usize| args.get(k).cloned().unwrap_or_default();
        Ok(match (node.kind, &node.value) {
            ("operand", VValue::Operand(index, part)) => {
                self.used.insert(*index);
                if !self.alongs.contains(index) {
                    self.alongs.push(*index);
                }
                let slot = if *index < 2 {
                    ARGS[*index]
                } else {
                    ARGS.get(index + 1).copied().unwrap_or("?")
                };
                let odd = if part.is_some_and(|p| p != 0) {
                    " + 1"
                } else {
                    ""
                };
                format!("sfpi::dst_reg[{slot} * 32{odd}]")
            }
            ("const", VValue::Const(x)) => literal(*x)?,
            ("scalar", VValue::Scalar(i)) => {
                self.scalars.insert(*i);
                format!("sfpi::vFloat({CONVERTER}(s{i}))")
            }
            ("add" | "sub", _) => {
                let (a, b) = (arg(0), arg(1));
                let mut product = false;
                for &n in &node.args {
                    product |= self.product(n)?;
                }
                if product {
                    // written as 1 * a + b so that the product is rounded before the sum
                    let b = if node.kind == "add" {
                        b
                    } else {
                        format!("(-{b})")
                    };
                    format!("__builtin_rvtt_sfpmad({ONE}.get(), {a}.get(), {b}.get(), 0)")
                } else {
                    let op = if node.kind == "add" { '+' } else { '-' };
                    format!("{a} {op} {b}")
                }
            }
            ("mul", _) => format!("{} * {}", arg(0), arg(1)),
            ("mad", _) => format!(
                "__builtin_rvtt_sfpmad({}.get(), {}.get(), {}.get(), 0)",
                arg(0),
                arg(1),
                arg(2)
            ),
            ("neg", _) => format!("-{}", arg(0)),
            ("abs", _) => format!("sfpi::abs({})", arg(0)),
            ("min" | "max", _) => format!("sfpi::{}({}, {})", node.kind, arg(0), arg(1)),
            ("call", VValue::Call(name)) => {
                let Some(function) = function_named(name) else {
                    return Err(format!("no vector function {}", py::str_repr(name)));
                };
                self.calls.insert(name);
                function.call.replace("{x}", &arg(0))
            }
            (kind, _) => return Err(format!("the vector value {kind} is not ported")),
        })
    }

    /// Whether the value is a product the compiler would fuse into a following add.
    fn product(&self, id: VId) -> Result<bool, String> {
        let mut id = id;
        loop {
            let node = self.node(id)?;
            match node.kind {
                "mul" => return Ok(true),
                "neg" => match node.args.first() {
                    Some(&a) => id = a,
                    None => return Ok(false),
                },
                _ => return Ok(false),
            }
        }
    }

    fn value(&mut self, id: VId) -> Result<String, String> {
        let node = self.node(id)?.clone();
        let key = match &node.value {
            VValue::Operand(index, part) if node.kind == "operand" => Key::Operand(*index, *part),
            VValue::Const(x) if node.kind == "const" => Key::Const(literal(*x)?),
            VValue::Scalar(i) if node.kind == "scalar" => Key::Scalar(*i),
            _ => Key::Node(id),
        };
        if let Some((_, name)) = self.names.iter().find(|(k, _)| *k == key) {
            return Ok(name.clone());
        }
        if node.kind == "select" {
            let VValue::Cond(kind, a, b) = node.value else {
                return Err("a select of no comparison".to_string());
            };
            let (a, b) = (self.value(a)?, self.value(b)?);
            let mut arms = Vec::new();
            for &arg in &node.args {
                arms.push(self.value(arg)?);
            }
            let (then, other) = match arms.as_slice() {
                [then, other] => (then.clone(), other.clone()),
                _ => return Err("a select of no two values".to_string()),
            };
            let name = self.new_name(other);
            self.names.push((key, name.clone()));
            let op = COMPARES
                .iter()
                .find(|(k, _)| *k == kind)
                .map_or("?", |(_, op)| *op);
            self.lines
                .push(format!("v_if({a} {op} {b}) {{ {name} = {then}; }}"));
            self.lines.push("v_endif;".to_string());
            return Ok(name);
        }
        let text = self.text(id)?;
        let name = self.new_name(text);
        self.names.push((key, name.clone()));
        Ok(name)
    }
}

/// What a function stores: one value, or the even and odd columns' values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FnResult {
    /// One value.
    One(Val),
    /// The even and the odd columns.
    Pair(Val, Val),
}

/// A vector-unit function written from expressions: `operands` register operands and
/// `scalars` fp32 scalar arguments, run under the setup of table op `init`.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedFn {
    /// Its name.
    pub name: String,
    /// Its whole C++ definition.
    pub code: String,
    /// Its register operands.
    pub operands: usize,
    /// The table op whose setup it runs under.
    pub init: String,
    /// The headers it includes.
    pub headers: Vec<String>,
    /// It needs fp32 registers.
    pub fp32: bool,
    /// Its scalar arguments.
    pub scalars: usize,
    /// Its operands in the order first read (each whole).
    pub along: Vec<usize>,
    /// What it stores.
    pub result: FnResult,
}

/// The function `name` storing `result` into its output slot (today's `sfpu_fn`), or
/// today's refusal of it.
pub fn sfpu_fn(vs: &mut Vs, name: &str, result: FnResult) -> Result<GeneratedFn, String> {
    let pairs = matches!(result, FnResult::Pair(..));
    let results = match result {
        FnResult::One(v) => vec![vs.wrap(v)],
        FnResult::Pair(even, odd) => vec![vs.wrap(even), vs.wrap(odd)],
    };
    let mut low = Lowering {
        vs,
        lines: Vec::new(),
        names: Vec::new(),
        used: BTreeSet::new(),
        calls: BTreeSet::new(),
        scalars: BTreeSet::new(),
        alongs: Vec::new(),
    };
    let mut outs = Vec::new();
    for r in &results {
        outs.push(low.value(*r)?);
    }
    let whole = low.names.iter().all(|(k, _)| match k {
        Key::Operand(_, part) => {
            if pairs {
                part.is_some()
            } else {
                part.is_none()
            }
        }
        _ => true,
    });
    if !whole {
        return Err(
            "a pair function reads operands by .even / .odd, any other reads them whole"
                .to_string(),
        );
    }
    let contiguous = |set: &BTreeSet<usize>| set.last().is_some_and(|m| m + 1 == set.len());
    if !contiguous(&low.used) {
        return Err("a function uses operands 0 .. n - 1".to_string());
    }
    let inits: BTreeSet<&str> = low
        .calls
        .iter()
        .filter_map(|c| function_named(c).and_then(|f| f.init))
        .collect();
    if inits.len() > 1 {
        let list = |items: Vec<&str>| {
            let quoted: Vec<String> = items.into_iter().map(py::str_repr).collect();
            format!("[{}]", quoted.join(", "))
        };
        return Err(format!(
            "functions {} need different setups ({}); one vector-unit function runs under one",
            list(low.calls.iter().copied().collect()),
            list(inits.iter().copied().collect())
        ));
    }
    let (count, scalars) = (low.used.len(), low.scalars.len());
    if !low.scalars.is_empty() && !contiguous(&low.scalars) {
        return Err("a function uses scalars 0 .. n - 1".to_string());
    }
    let mut names: Vec<String> = ARGS[..(count + 1).max(3).min(ARGS.len())]
        .iter()
        .map(|a| a.to_string())
        .collect();
    names.extend((0..scalars).map(|i| format!("s{i}")));
    let params: Vec<String> = names.iter().map(|a| format!("uint32_t {a}")).collect();
    let mut body = low.lines.clone();
    for (i, out) in outs.iter().enumerate() {
        body.push("if constexpr (!DST_ACCUM_MODE) {".to_string());
        body.push(format!("    {out} = float32_to_bf16_rne({out});"));
        body.push("}".to_string());
        let odd = if i > 0 { " + 1" } else { "" };
        body.push(format!("sfpi::dst_reg[out * 32{odd}] = {out};"));
    }
    body.push(
        if pairs {
            "sfpi::dst_reg += 2;"
        } else {
            "sfpi::dst_reg++;"
        }
        .to_string(),
    );
    let mut headers = vec!["sfpi.h".to_string(), format!("{LLK}conversions.h")];
    let called: BTreeSet<String> = low
        .calls
        .iter()
        .filter_map(|c| function_named(c).map(|f| format!("{LLK}{}", f.header)))
        .collect();
    headers.extend(called);
    if scalars > 0 {
        headers.push("sfpu/ckernel_sfpu_converter.h".to_string());
    }
    let mut lines = vec![
        "#ifdef TRISC_MATH".to_string(),
        format!("inline void {name}({}) {{", params.join(", ")),
        format!(
            "    for (int rows = 0; rows < {}; rows++) {{",
            if pairs { 4 } else { 8 }
        ),
    ];
    lines.extend(body.iter().map(|l| format!("        {l}")));
    lines.extend(["    }", "}", "#endif"].map(String::from));
    let fp32 = low
        .calls
        .iter()
        .any(|c| function_named(c).is_some_and(|f| f.fp32));
    Ok(GeneratedFn {
        name: name.to_string(),
        code: lines.join("\n"),
        operands: count,
        init: inits.first().copied().unwrap_or("mul").to_string(),
        headers,
        fp32,
        scalars,
        along: low.alongs.clone(),
        result,
    })
}
