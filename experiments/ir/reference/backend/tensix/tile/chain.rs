//! The chain compiler (today's `komorebi.dsl.tile.chain`): the vector-unit function
//! (`Chain`) computing a stored value over at most four register slots, with each
//! transcendental's template. The stream form's entry to it is in `lower`.
//!
//! A stream form's operand is read copied, or spread from a tile (`.row()`: its row 0
//! to every row, `bcast_row`; `.col()`: its column 0 to every column, `bcast_col`).
//! Not ported: the row form (`banded`, `grouped`, reduced values and fields read per
//! row, slices, `cat`, `.even` / `.odd`, casts), and the templates chosen by a stated
//! error (Kohaku-TT's flash exp): the IR states no error, so each transcendental takes
//! its default, the accurate template. What would take them is refused by name.

use super::ir::{Attr, KernelIR, NodeId};
use super::py;
use super::unit::UnitError;
use super::vector::{self, FnResult, GeneratedFn, Val, Vs};
use crate::backend::{Refusal, NOT_BUILT, OVER_LIMIT};
use crate::program::notes::Anchor;

/// Destination register slots of one half with fp32 registers.
pub const SLOTS: usize = 4;

/// ln(2) as today writes it (`0.6931471805599453`), by its bits: `exp2`'s factor.
pub const LN2: f64 = f64::from_bits(0x3FE6_2E42_FEFA_39EF);

/// Why a chain or a unit program is not made: today's refusal (its text, and the
/// line it names), or a part of today's compiler not ported.
#[derive(Clone, Debug, PartialEq)]
pub enum ChainError {
    /// Today's `PlanError`: the message, and the source line when the node has one.
    Plan(String, Option<String>),
    /// Today's `UnitError`: its message whole.
    Unit(String),
    /// A part not ported, by name.
    NotPorted(String),
}

impl From<UnitError> for ChainError {
    fn from(e: UnitError) -> ChainError {
        ChainError::Unit(e.0)
    }
}

impl ChainError {
    /// Today's text of a refusal: `komorebi.dsl.tile: <message> (at <line>)`.
    pub fn text(&self) -> String {
        match self {
            ChainError::Plan(message, Some(line)) => {
                format!("komorebi.dsl.tile: {message} (at {line})")
            }
            ChainError::Plan(message, None) => format!("komorebi.dsl.tile: {message}"),
            ChainError::Unit(message) => message.clone(),
            ChainError::NotPorted(what) => format!("not ported: {what}"),
        }
    }

    /// The backend's refusal of the function at `at`: what today refuses too is over
    /// the compute unit's limits (K962, today's text whole), the rest not built (K964).
    pub fn refusal(&self, at: Option<Anchor>) -> Refusal {
        match self {
            ChainError::NotPorted(what) => Refusal {
                code: NOT_BUILT,
                what: format!("the Tensix backend's tile compiler does not take {what} yet"),
                at,
            },
            _ => Refusal {
                code: OVER_LIMIT,
                what: self.text(),
                at,
            },
        }
    }

    fn slots(&self) -> bool {
        matches!(self, ChainError::Plan(m, _) if m.contains("register slots"))
    }
}

/// What a register operand reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// A node of the kernel: a load, or a tile product (its result in Dst).
    Node(NodeId),
    /// The previous stage's result.
    Prev,
}

/// A register operand: `source` unpacked as `kind` (`copy`; `prev` for a stage's).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Leaf {
    /// What it reads.
    pub source: Source,
    /// How.
    pub kind: &'static str,
}

/// A runtime scalar operand: a float value, by name, maybe its reciprocal; the launch
/// gives its fp32 bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scalar {
    /// The runtime value's name.
    pub name: String,
    /// Its reciprocal.
    pub recip: bool,
}

impl Scalar {
    /// `name`, or `1/name`.
    pub fn key(&self) -> String {
        format!("{}{}", if self.recip { "1/" } else { "" }, self.name)
    }
}

/// One vector-unit function: its register operands in slots 0 .., its runtime scalars
/// (passed as fp32 bits), the operations it runs, and the stage run before it.
#[derive(Clone, Debug, PartialEq)]
pub struct Chain {
    /// The register operands.
    pub operands: Vec<Leaf>,
    /// The function.
    pub func: GeneratedFn,
    /// Its vector values.
    pub vs: Vs,
    /// The output slot.
    pub out: usize,
    /// A function of column pairs.
    pub pairs: bool,
    /// Each node kind it computes and how many, in the order first met.
    pub ops: Vec<(String, usize)>,
    /// The runtime scalars.
    pub values: Vec<Scalar>,
    /// The stage run before this one (its result is operand 0).
    pub prev: Option<Box<Chain>>,
    /// The templates chosen, by `<kind> (line <line>)`: the compiler's whole record.
    pub templates: Vec<(String, String)>,
}

/// One way to compute a transcendental: its relative error bound over inputs in `over`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Template {
    /// Its name.
    pub name: &'static str,
    /// Its relative error bound.
    pub error: f64,
    /// The inputs it holds over.
    pub over: (f64, f64),
}

impl Template {
    /// Today's record of the template chosen for a node that states no error.
    pub fn describe(&self) -> String {
        format!(
            "{}, relative error {} over [{}, {}] (asked: default over [-inf, inf])",
            self.name,
            py::format_g(self.error, 3),
            py::format_g(self.over.0, 6),
            py::format_g(self.over.1, 6)
        )
    }
}

/// Each transcendental's default (accurate) template: the one taken when no error is
/// stated (today's `TEMPLATES`, last of each).
pub const TEMPLATES: &[(&str, Template)] = &[
    ("exp", accurate(3e-7, f64::NEG_INFINITY)),
    ("exp2", accurate(6e-6, f64::NEG_INFINITY)),
    ("tanh", accurate(3e-7, f64::NEG_INFINITY)),
    ("erf", accurate(4e-7, f64::NEG_INFINITY)),
    ("log", accurate(1e-7, 0.0)),
];

const fn accurate(error: f64, low: f64) -> Template {
    Template {
        name: "accurate",
        error,
        over: (low, f64::INFINITY),
    }
}

/// The form of a node in the stream form: `tile` or `scalar`, or why it is not taken.
pub(crate) fn shape(
    ir: &KernelIR,
    node: NodeId,
    of: &mut Vec<Option<&'static str>>,
) -> Result<&'static str, ChainError> {
    if let Some(Some(known)) = of.get(node.0) {
        return Ok(known);
    }
    let n = ir
        .arena
        .get(node.0)
        .ok_or_else(|| ChainError::NotPorted("a node of no kernel".into()))?;
    let made = match n.kind.as_str() {
        "load" => match n.attrs.first() {
            Some((_, Attr::View(v))) if v.kind == "row" => "row",
            _ => "tile",
        },
        "const" | "param" | "size" => "scalar",
        "col" => {
            let Some(&a) = n.args.first() else {
                return Err(ChainError::NotPorted("a .col() of no operand".into()));
            };
            match shape(ir, a, of)? {
                "tile" => "tile",
                "scalar" => {
                    return Err(ChainError::Plan(
                        ".col() broadcasts a reduced value or a tile".into(),
                        n.line.clone(),
                    ))
                }
                other => {
                    return Err(ChainError::NotPorted(format!(
                        "the row form's .col() of a {other}"
                    )))
                }
            }
        }
        "reduce" => {
            let Some(&a) = n.args.first() else {
                return Err(ChainError::NotPorted("a reduction of no operand".into()));
            };
            if shape(ir, a, of)? != "row" {
                return Err(ChainError::Plan(
                    "tile.reduce_sum reduces a row: load x.tiles[r, :]".into(),
                    n.line.clone(),
                ));
            }
            return Err(ChainError::NotPorted("the row form's reduce".into()));
        }
        "slot" | "word" | "state" | "slice" | "cat" | "resident" => {
            return Err(ChainError::NotPorted(format!("the row form's {}", n.kind)))
        }
        _ => {
            let mut tile = false;
            for &a in &n.args.clone() {
                match shape(ir, a, of)? {
                    "tile" => tile = true,
                    "scalar" => {}
                    other => return Err(ChainError::NotPorted(format!("a {other} operand"))),
                }
            }
            if tile {
                "tile"
            } else {
                "scalar"
            }
        }
    };
    if made == "row" {
        return Err(ChainError::NotPorted("the row form's row loads".into()));
    }
    if let Some(slot) = of.get_mut(node.0) {
        *slot = Some(made);
    }
    Ok(made)
}

/// Builds the vector-unit functions of a kernel.
pub(crate) struct Compiler<'k> {
    ir: &'k KernelIR,
    shapes: Vec<Option<&'static str>>,
    scalars: Vec<Scalar>,
    count: usize,
    templates: Vec<(String, String)>,
    /// The tile products: register operands as loads are.
    products: Vec<NodeId>,
    /// The row form's nodes read as register operands as its row loads and resident
    /// rows are (today's `leaf_of` of `row_program` and `level_program`).
    row: Option<Vec<NodeId>>,
    /// The register slots a function takes (today's `limit`).
    limit: usize,
}

impl<'k> Compiler<'k> {
    /// A compiler of `ir`'s chains, the forms of its nodes `shapes` and its tile
    /// products `products` known.
    pub(crate) fn new(
        ir: &'k KernelIR,
        shapes: Vec<Option<&'static str>>,
        products: Vec<NodeId>,
    ) -> Compiler<'k> {
        Compiler {
            ir,
            shapes,
            scalars: Vec::new(),
            count: 0,
            templates: Vec::new(),
            products,
            row: None,
            limit: SLOTS,
        }
    }

    /// A compiler of the row form's chains of `ir`: the forms of its nodes `shapes`
    /// (the row form's), the nodes `leaves` read as operands (its reductions and its
    /// per-row values), at most `limit` register operands a function.
    pub(crate) fn rows(
        ir: &'k KernelIR,
        shapes: Vec<Option<&'static str>>,
        leaves: Vec<NodeId>,
        limit: usize,
    ) -> Compiler<'k> {
        let mut made = Compiler::new(ir, shapes, Vec::new());
        made.row = Some(leaves);
        made.limit = limit;
        made
    }

    /// The templates chosen so far, in order.
    pub(crate) fn templates(&self) -> &[(String, String)] {
        &self.templates
    }

    fn plan(&self, message: String, node: NodeId) -> ChainError {
        let line = self.ir.arena.get(node.0).and_then(|n| n.line.clone());
        ChainError::Plan(message, line)
    }

    /// The Scalar of runtime value `name`, registered once.
    fn scalar(&mut self, name: &str, recip: bool) -> Scalar {
        let s = Scalar {
            name: name.to_string(),
            recip,
        };
        if let Some(known) = self.scalars.iter().find(|k| k.key() == s.key()) {
            return known.clone();
        }
        self.scalars.push(s.clone());
        s
    }

    /// The Chain computing `root`; a chain reading more blocks than the register slots
    /// runs as stages (`Chain.prev`), its first operand of `root` computed first.
    pub(crate) fn chain(&mut self, root: NodeId) -> Result<Chain, ChainError> {
        let error = match self.single(root, None) {
            Ok(chain) => return Ok(chain),
            Err(e) if e.slots() => e,
            Err(e) => return Err(e),
        };
        let top = self
            .ir
            .arena
            .get(root.0)
            .ok_or_else(|| ChainError::NotPorted("a node of no kernel".into()))?;
        let staged = ["add", "sub", "mul", "max", "min"].contains(&top.kind.as_str());
        if top.args.len() != 2 || !staged {
            return Err(error);
        }
        for first in top.args.clone() {
            let tried = self
                .chain(first)
                .and_then(|prev| self.single(root, Some((first, prev))));
            match tried {
                Ok(out) => return Ok(out),
                Err(e) if e.slots() => continue,
                Err(e) => return Err(e),
            }
        }
        Err(error)
    }

    /// One generated function computing `root`; `prev` (node, Chain): that node is the
    /// previous stage's result, operand 0.
    fn single(&mut self, root: NodeId, prev: Option<(NodeId, Chain)>) -> Result<Chain, ChainError> {
        let mut vs = Vs::default();
        let mut operands = Vec::new();
        let mut prev_operand = None;
        if prev.is_some() {
            prev_operand = Some(vs.operand(0).map_err(ChainError::NotPorted)?);
            operands.push(Leaf {
                source: Source::Prev,
                kind: "prev",
            });
        }
        let mut s = Build {
            prev: prev.as_ref().map(|p| p.0),
            prev_operand,
            vs,
            operands,
            keys: Vec::new(),
            ops: Vec::new(),
            values: Vec::new(),
            places: Vec::new(),
            built: Vec::new(),
            line: self.ir.arena.get(root.0).and_then(|n| n.line.clone()),
        };
        let result = s.build(self, root, "copy")?;
        self.count += 1;
        let name = format!("{}_fn{}", self.ir.name, self.count);
        let func = vector::sfpu_fn(&mut s.vs, &name, FnResult::One(result))
            .map_err(|m| self.plan(m, root))?;
        let mut chain = Chain {
            operands: s.operands,
            func,
            vs: s.vs,
            out: 0,
            pairs: false,
            ops: s.ops,
            values: s.values,
            prev: None,
            templates: Vec::new(),
        };
        if let Some((_, before)) = prev {
            for (k, n) in &before.ops {
                match chain.ops.iter_mut().find(|(o, _)| o == k) {
                    Some((_, m)) => *m += n,
                    None => chain.ops.push((k.clone(), *n)),
                }
            }
            chain.prev = Some(Box::new(before));
        }
        Ok(chain)
    }
}

/// The state of one `single`: the function's values and operands as they are made.
struct Build {
    prev: Option<NodeId>,
    prev_operand: Option<vector::VId>,
    vs: Vs,
    operands: Vec<Leaf>,
    keys: Vec<((Source, &'static str), vector::VId)>,
    ops: Vec<(String, usize)>,
    values: Vec<Scalar>,
    places: Vec<(String, vector::VId)>,
    built: Vec<((NodeId, &'static str), Val)>,
    line: Option<String>,
}

impl Build {
    fn plan(&self, message: String) -> ChainError {
        ChainError::Plan(message, self.line.clone())
    }

    /// The scalar argument of `s`, placed once.
    fn value(&mut self, s: Scalar) -> Result<Val, ChainError> {
        if let Some((_, id)) = self.places.iter().find(|(k, _)| *k == s.key()) {
            return Ok(Val::V(*id));
        }
        if self.values.len() == 4 {
            return Err(self.plan("the chain needs more than 4 runtime scalars".into()));
        }
        let id = self
            .vs
            .scalar(self.values.len())
            .map_err(ChainError::NotPorted)?;
        self.places.push((s.key(), id));
        self.values.push(s);
        Ok(Val::V(id))
    }

    /// The register operand reading `node` whole as `how` (today's stream form's
    /// `leaf_of`): a product copied, a load copied or spread (its column 0 to every
    /// column, `bcast_col`; its row 0 to every row, `bcast_row`).
    fn leaf(
        &mut self,
        compiler: &Compiler,
        node: NodeId,
        how: &'static str,
    ) -> Result<Val, ChainError> {
        let kind = compiler
            .ir
            .arena
            .get(node.0)
            .map_or("", |n| n.kind.as_str());
        let row_view = || {
            compiler.ir.arena.get(node.0).is_some_and(|n| {
                n.attrs
                    .iter()
                    .any(|(k, v)| k == "view" && matches!(v, Attr::View(v) if v.kind == "row"))
            })
        };
        // today's leaf rule; its row-view test is never reached from a kernel the shapes
        // take (a tile meeting a row is refused there first), kept as today's line
        let read = if let Some(leaves) = &compiler.row {
            ["load", "resident"].contains(&kind) && row_view() || leaves.contains(&node)
        } else if compiler.products.contains(&node) {
            how == "copy"
        } else {
            kind == "load" && ["copy", "bcast_col", "bcast_row"].contains(&how)
        };
        if !read {
            return Err(compiler.plan(format!("{kind} cannot be read here"), node));
        }
        let key = (Source::Node(node), how);
        if let Some((_, id)) = self.keys.iter().find(|(k, _)| *k == key) {
            return Ok(Val::V(*id));
        }
        if self.operands.len() == compiler.limit {
            let limit = compiler.limit;
            return Err(self.plan(format!("the chain needs more than {limit} register slots")));
        }
        let id = self
            .vs
            .operand(self.operands.len())
            .map_err(ChainError::NotPorted)?;
        self.keys.push((key, id));
        self.operands.push(Leaf {
            source: Source::Node(node),
            kind: how,
        });
        Ok(Val::V(id))
    }

    /// The value of `node`, its loads read as `how`, made once.
    fn build(
        &mut self,
        compiler: &mut Compiler,
        node: NodeId,
        how: &'static str,
    ) -> Result<Val, ChainError> {
        if let Some((_, v)) = self.built.iter().find(|(k, _)| *k == (node, how)) {
            return Ok(*v);
        }
        let made = self.make(compiler, node, how)?;
        self.built.push(((node, how), made));
        Ok(made)
    }

    fn make(
        &mut self,
        compiler: &mut Compiler,
        node: NodeId,
        how: &'static str,
    ) -> Result<Val, ChainError> {
        if self.prev == Some(node) {
            if let Some(id) = self.prev_operand {
                return Ok(Val::V(id));
            }
        }
        let n = compiler
            .ir
            .arena
            .get(node.0)
            .ok_or_else(|| ChainError::NotPorted("a node of no kernel".into()))?
            .clone();
        let k = n.kind.as_str();
        match self.ops.iter_mut().find(|(o, _)| o == k) {
            Some((_, count)) => *count += 1,
            None => self.ops.push((k.to_string(), 1)),
        }
        let row_leaf = compiler.row.is_some() && ["resident", "reduce"].contains(&k);
        if k == "load" || row_leaf || compiler.products.contains(&node) {
            return self.leaf(compiler, node, how);
        }
        let attr = |name: &str| {
            n.attrs
                .iter()
                .find(|(a, _)| a == name)
                .map(|(_, v)| v.clone())
        };
        match k {
            "const" => {
                return match attr("value") {
                    Some(Attr::Float(x)) => Ok(Val::F(x)),
                    _ => Err(ChainError::NotPorted(
                        "a constant that is not a float".into(),
                    )),
                }
            }
            "param" => {
                return match attr("value") {
                    Some(Attr::Value(name)) => {
                        let s = compiler.scalar(&name, false);
                        self.value(s)
                    }
                    _ => Err(ChainError::NotPorted("a runtime value of no name".into())),
                }
            }
            "row" | "col" => {
                let Some(&a) = n.args.first() else {
                    return Err(ChainError::NotPorted(format!("the node {k} of no operand")));
                };
                let spread = if k == "row" { "bcast_row" } else { "bcast_col" };
                // a per-row value (the row form's) is read as an operand
                let ir = compiler.ir;
                if k == "col"
                    && compiler.row.is_some()
                    && shape(ir, a, &mut compiler.shapes)? == "col"
                {
                    return self.leaf(compiler, a, spread);
                }
                return self.build(compiler, a, spread);
            }
            "resident" | "reduce" | "acc" | "size" | "cast" | "all" | "slice" | "cat" | "even"
            | "odd" | "lagmax" | "pairs" => {
                return Err(ChainError::NotPorted(format!("the node {k}")));
            }
            _ => {}
        }
        let shape_of = |c: &mut Compiler, a: NodeId| {
            let ir = c.ir;
            shape(ir, a, &mut c.shapes)
        };
        if k == "div" {
            if let [a, b] = n.args[..] {
                if shape_of(compiler, b)? == "scalar" {
                    let a = self.build(compiler, a, how)?;
                    let bn = compiler.ir.arena.get(b.0).cloned();
                    let bn =
                        bn.ok_or_else(|| ChainError::NotPorted("a node of no kernel".into()))?;
                    let value = bn
                        .attrs
                        .iter()
                        .find(|(x, _)| x == "value")
                        .map(|(_, v)| v.clone());
                    return match (bn.kind.as_str(), value) {
                        ("const", Some(Attr::Float(x))) => {
                            if x == 0.0 {
                                return Err(ChainError::NotPorted(
                                    "a division by the constant 0 (today's compiler raises ZeroDivisionError)".into(),
                                ));
                            }
                            Ok(self.vs.mul(a, Val::F(1.0 / x)))
                        }
                        (_, Some(Attr::Value(name))) => {
                            let s = compiler.scalar(&name, true);
                            let r = self.value(s)?;
                            Ok(self.vs.mul(a, r))
                        }
                        _ => Err(ChainError::NotPorted("a division by this scalar".into())),
                    };
                }
            }
        }
        let mut args = Vec::new();
        for &a in &n.args {
            args.push(self.build(compiler, a, how)?);
        }
        let arg = |i: usize| args.get(i).copied().unwrap_or(Val::F(0.0));
        if n.args.len() != expected_args(k) {
            return Err(ChainError::NotPorted(format!(
                "the node {k} of {} operands",
                n.args.len()
            )));
        }
        let vs = &mut self.vs;
        Ok(match k {
            "div" => {
                let r = vs.call("recip", arg(1));
                vs.mul(arg(0), r)
            }
            "where" => {
                let compare = match attr("compare") {
                    Some(Attr::Text(t)) => compare_kind(&t)?,
                    _ => return Err(ChainError::NotPorted("a where of no comparison".into())),
                };
                let x = vs.wrap(arg(0));
                let y = Val::V(vs.wrap(arg(1)));
                vs.select(compare, x, y, arg(2), arg(3))
            }
            "exp" | "exp2" | "tanh" | "erf" | "log" => {
                let t = TEMPLATES
                    .iter()
                    .find(|(name, _)| *name == k)
                    .map(|(_, t)| *t)
                    .ok_or_else(|| ChainError::NotPorted(format!("the template of {k}")))?;
                let key = format!(
                    "{k} (line {})",
                    n.line.clone().unwrap_or_else(|| "None".to_string())
                );
                let described = t.describe();
                match compiler.templates.iter_mut().find(|(o, _)| *o == key) {
                    Some((_, d)) => *d = described,
                    None => compiler.templates.push((key, described)),
                }
                match k {
                    "exp" => vs.call("exp", arg(0)),
                    "exp2" => {
                        let scaled = vs.mul(arg(0), Val::F(LN2));
                        vs.call("exp", scaled)
                    }
                    "tanh" => vs.call("tanh", arg(0)),
                    "erf" => vs.call("erf", arg(0)),
                    _ => vs.call("log", arg(0)),
                }
            }
            "neg" => vs.neg(arg(0)),
            "abs" => vs.abs(arg(0)),
            "sigmoid" => vs.call("sigmoid", arg(0)),
            "silu" => vs.silu(arg(0)),
            "sqrt" => vs.call("sqrt", arg(0)),
            "rsqrt" => vs.call("rsqrt", arg(0)),
            "recip" => vs.call("recip", arg(0)),
            "relu" => vs.relu(arg(0)),
            "square" => vs.square(arg(0)),
            "add" => vs.add(arg(0), arg(1)),
            "sub" => vs.sub(arg(0), arg(1)),
            "mul" => vs.mul(arg(0), arg(1)),
            "max" => vs.maximum(arg(0), arg(1)),
            "min" => vs.minimum(arg(0), arg(1)),
            other => return Err(ChainError::NotPorted(format!("the node {other}"))),
        })
    }
}

/// The operands a node kind of the stream form takes.
fn expected_args(kind: &str) -> usize {
    match kind {
        "where" => 4,
        "div" | "add" | "sub" | "mul" | "max" | "min" | "matmul" => 2,
        _ => 1,
    }
}

/// A comparison's kind as the vector unit names it.
fn compare_kind(text: &str) -> Result<&'static str, ChainError> {
    ["lt", "le", "gt", "ge", "eq", "ne"]
        .into_iter()
        .find(|k| *k == text)
        .ok_or_else(|| ChainError::NotPorted(format!("the comparison {text}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_constants_are_todays_bits() {
        // Python: (0.6931471805599453).hex()
        assert_eq!(py::float_hex(LN2), "0x1.62e42fefa39efp-1");
        let described: Vec<String> = TEMPLATES.iter().map(|(_, t)| t.describe()).collect();
        assert_eq!(
            described[4],
            "accurate, relative error 1e-07 over [0, inf] (asked: default over [-inf, inf])"
        );
    }
}
