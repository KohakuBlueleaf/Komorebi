//! The chain compiler (today's `komorebi.dsl.tile.chain`): the vector-unit function
//! (`Chain`) computing a stored value over at most four register slots, with each
//! transcendental's template; and the stream form's entry to it (today's
//! `lower.stream_chain`, `_stream_chains`, `products_of`): each load is a register
//! operand, and so is each tile product.
//!
//! Not ported: the row form (`banded`, `grouped`, per-row and broadcast operands,
//! slices, `cat`, `.even` / `.odd`, casts), and the templates chosen by a stated error
//! (Kohaku-TT's flash exp): the IR states no error, so each transcendental takes its
//! default, the accurate template. What would take them is refused by name.

use super::ir::{Attr, KernelIR, NodeId};
use super::py;
use super::vector::{self, FnResult, GeneratedFn, Val, Vs};

/// Destination register slots of one half with fp32 registers.
pub const SLOTS: usize = 4;

/// ln(2) as today writes it (`0.6931471805599453`), by its bits: `exp2`'s factor.
pub const LN2: f64 = f64::from_bits(0x3FE6_2E42_FEFA_39EF);

/// Why a chain is not made: today's refusal (its text, and the line it names), or a
/// part of today's compiler not ported.
#[derive(Clone, Debug, PartialEq)]
pub enum ChainError {
    /// Today's `PlanError`: the message, and the source line when the node has one.
    Plan(String, Option<String>),
    /// A part not ported, by name.
    NotPorted(String),
}

impl ChainError {
    /// Today's text of a refusal: `komorebi.dsl.tile: <message> (at <line>)`.
    pub fn text(&self) -> String {
        match self {
            ChainError::Plan(message, Some(line)) => {
                format!("komorebi.dsl.tile: {message} (at {line})")
            }
            ChainError::Plan(message, None) => format!("komorebi.dsl.tile: {message}"),
            ChainError::NotPorted(what) => format!("not ported: {what}"),
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
fn shape(
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
        "slot" | "word" | "state" | "reduce" | "col" | "slice" | "cat" | "resident" => {
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
struct Compiler<'k> {
    ir: &'k KernelIR,
    shapes: Vec<Option<&'static str>>,
    scalars: Vec<Scalar>,
    count: usize,
    templates: Vec<(String, String)>,
    /// The tile products: register operands as loads are.
    products: Vec<NodeId>,
}

impl Compiler<'_> {
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
    fn chain(&mut self, root: NodeId) -> Result<Chain, ChainError> {
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
        let result = s.build(self, root)?;
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
    built: Vec<(NodeId, Val)>,
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

    /// The register operand reading `node` whole.
    fn leaf(&mut self, compiler: &Compiler, node: NodeId) -> Result<Val, ChainError> {
        let kind = compiler
            .ir
            .arena
            .get(node.0)
            .map_or("", |n| n.kind.as_str());
        if !(kind == "load" || compiler.products.contains(&node)) {
            return Err(compiler.plan(format!("{kind} cannot be read here"), node));
        }
        let key = (Source::Node(node), "copy");
        if let Some((_, id)) = self.keys.iter().find(|(k, _)| *k == key) {
            return Ok(Val::V(*id));
        }
        if self.operands.len() == SLOTS {
            return Err(self.plan(format!("the chain needs more than {SLOTS} register slots")));
        }
        let id = self
            .vs
            .operand(self.operands.len())
            .map_err(ChainError::NotPorted)?;
        self.keys.push((key, id));
        self.operands.push(Leaf {
            source: Source::Node(node),
            kind: "copy",
        });
        Ok(Val::V(id))
    }

    fn build(&mut self, compiler: &mut Compiler, node: NodeId) -> Result<Val, ChainError> {
        if let Some((_, v)) = self.built.iter().find(|(n, _)| *n == node) {
            return Ok(*v);
        }
        let made = self.make(compiler, node)?;
        self.built.push((node, made));
        Ok(made)
    }

    fn make(&mut self, compiler: &mut Compiler, node: NodeId) -> Result<Val, ChainError> {
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
        if k == "load" || compiler.products.contains(&node) {
            return self.leaf(compiler, node);
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
            "resident" | "reduce" | "acc" | "size" | "cast" | "row" | "all" | "col" | "slice"
            | "cat" | "even" | "odd" | "lagmax" | "pairs" => {
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
                    let a = self.build(compiler, a)?;
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
            args.push(self.build(compiler, a)?);
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
    let mut compiler = Compiler {
        ir,
        shapes,
        scalars: Vec::new(),
        count: 0,
        templates: Vec::new(),
        products,
    };
    let mut chains = Vec::new();
    for store in &ir.stores {
        chains.push(compiler.chain(store.value)?);
    }
    for chain in &mut chains {
        let mut stage = Some(chain);
        while let Some(c) = stage {
            c.templates = compiler.templates.clone();
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
