//! The tile IR of one kernel (today's `komorebi.dsl.tile.ir`): its tensors and values,
//! the grid, the block expressions and the stores; `KernelIR::dump` prints it as today
//! does. `kernel_ir` builds it from a `func` body as today's `tile_unit` traces a tile
//! function at the 1-tile width: a load of each tile parameter's tile `unit`, the body
//! on them, its result stored into `result`. What the trace would not make is refused
//! by name (K964).

use crate::backend::tensix::plan::param_name;
use crate::backend::{Refusal, NOT_BUILT};
use crate::program::body::{Body, InstId, Op, ValueId};
use crate::program::notes::{Anchor, Owner};
use crate::program::{BodyRef, FuncKind, Module};
use crate::types::{BinOp, CmpOp, ConstValue, Format, FuncId, ScalarType, Type, UnaryOp};

use super::py;

/// A tensor of the kernel; `dtype` is fixed per kernel class.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorParam {
    /// Its name.
    pub name: String,
    /// Written by the kernel.
    pub output: bool,
    /// Its format by today's name (`float32`, `bfloat16`, ...).
    pub dtype: Option<String>,
}

/// A runtime value of the kernel (`float` or `int`): one argument, no recompile.
#[derive(Clone, Debug, PartialEq)]
pub struct ValueParam {
    /// Its name.
    pub name: String,
    /// `float` or `int`.
    pub kind: String,
}

/// An integer of the kernel: a constant, the grid's unit, a loop's variable, a runtime
/// int, a size of a tensor, or arithmetic on them.
#[derive(Clone, Debug, PartialEq)]
pub enum Int {
    /// A constant.
    Const(i64),
    /// The grid's unit, by its name when it has one.
    Unit(Option<String>),
    /// A device loop's variable.
    Loop(String),
    /// A runtime int, by name.
    Param(String),
    /// `what(name)` or `what(name, dim)`: `size`, `tiles`, `tile_rows`, `row_tiles`.
    Size {
        /// Which size.
        what: String,
        /// The tensor.
        name: String,
        /// The dim, for `size`.
        dim: Option<i64>,
    },
    /// `a op b` (`+ - * // %`, or `min`).
    Op(String, Box<Int>, Box<Int>),
}

impl Int {
    /// The value written out.
    pub fn text(&self) -> String {
        match self {
            Int::Const(v) => v.to_string(),
            Int::Unit(name) => name.clone().unwrap_or_else(|| "unit".to_string()),
            Int::Loop(name) | Int::Param(name) => name.clone(),
            Int::Size { what, name, dim } => match dim {
                None => format!("{what}({name})"),
                Some(d) => format!("{what}({name}, {d})"),
            },
            Int::Op(op, a, b) if op == "min" => format!("min({}, {})", a.text(), b.text()),
            Int::Op(op, a, b) => format!("({} {op} {})", a.text(), b.text()),
        }
    }
}

/// Tiles of a tensor: one tile at flat tile index `index` (`tile`), the whole tile row
/// `index` (`row`), a run `cols` of flat tiles (`flat`), rows `cols` (`rows`), or tiles
/// `cols` of row `index` (`at`).
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    /// The tensor's name.
    pub tensor: String,
    /// `tile`, `row`, `flat`, `rows` or `at`.
    pub kind: String,
    /// The tile or the row.
    pub index: Int,
    /// (lo, hi) of a range.
    pub cols: Option<(Int, Int)>,
}

impl View {
    /// The view written out.
    pub fn text(&self) -> String {
        let (at, name) = (self.index.text(), &self.tensor);
        let range = || match &self.cols {
            Some((lo, hi)) => (lo.text(), hi.text()),
            None => ("None".to_string(), "None".to_string()),
        };
        match self.kind.as_str() {
            "tile" => format!("{name}.tiles[{at}]"),
            "row" => format!("{name}.tiles[{at}, :]"),
            "flat" => {
                let (lo, hi) = range();
                format!("{name}.tiles[{lo}:{hi}]")
            }
            "rows" => {
                let (lo, hi) = range();
                format!("{name}.tiles[{lo}:{hi}, :]")
            }
            _ => {
                let (lo, hi) = range();
                format!("{name}.tiles[{at}, {lo}:{hi}]")
            }
        }
    }
}

/// A node's field, printed as today's `f"{k}={v}"` prints it.
#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    /// `None`.
    None,
    /// A text, printed as it is (`compare=gt`).
    Text(String),
    /// A float, printed as Python's `repr`.
    Float(f64),
    /// An integer of the kernel, printed as its text.
    Int(Int),
    /// A runtime value, printed as its name.
    Value(String),
    /// A view, printed as its text.
    View(View),
}

impl Attr {
    fn text(&self) -> String {
        match self {
            Attr::None => "None".to_string(),
            Attr::Text(t) | Attr::Value(t) => t.clone(),
            Attr::Float(x) => py::float_repr(*x),
            Attr::Int(i) => i.text(),
            Attr::View(v) => v.text(),
        }
    }
}

/// A node's number in its kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// A block value: a load, a scalar, or an operation on blocks; `attrs` holds the
/// kind's fields in today's order.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// The kind: `load`, `const`, `param`, `add`, `where`, ...
    pub kind: String,
    /// The operands.
    pub args: Vec<NodeId>,
    /// The fields.
    pub attrs: Vec<(String, Attr)>,
}

/// `view` = `value`.
#[derive(Clone, Debug, PartialEq)]
pub struct Store {
    /// Where.
    pub view: View,
    /// What.
    pub value: NodeId,
}

/// Units 0 .. count - 1 split over the cores; `block` units per NoC barrier.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    /// The units.
    pub count: Int,
    /// Units per barrier.
    pub block: i64,
    /// A 2-D grid's columns.
    pub cols: Option<Int>,
}

/// A device loop `var < count` inside the grid.
#[derive(Clone, Debug, PartialEq)]
pub struct Range {
    /// The count.
    pub count: Int,
    /// The variable.
    pub var: Int,
}

/// One kernel class: its tensors, runtime values, static values, the grid, the stores
/// inside it, and a device loop with its accumulation (`acc += a @ b`).
#[derive(Clone, Debug, PartialEq)]
pub struct KernelIR {
    /// Its name.
    pub name: String,
    /// Its tensors, inputs and outputs.
    pub tensors: Vec<TensorParam>,
    /// Its runtime values.
    pub values: Vec<ValueParam>,
    /// Its static values: (name, Python's `repr` of the value).
    pub statics: Vec<(String, String)>,
    /// The grid.
    pub grid: Option<Grid>,
    /// The stores.
    pub stores: Vec<Store>,
    /// The device loop.
    pub range: Option<Range>,
    /// (acc, its product).
    pub accumulate: Option<(NodeId, NodeId)>,
    /// (carry node, its next value) of a loop.
    pub carries: Vec<(NodeId, NodeId)>,
    /// Every node made, by number (as today's objects; `nodes` lists the read ones).
    pub arena: Vec<Node>,
}

impl KernelIR {
    /// A kernel with no grid, store or node.
    pub fn new(name: &str, tensors: Vec<TensorParam>, values: Vec<ValueParam>) -> Self {
        KernelIR {
            name: name.to_string(),
            tensors,
            values,
            statics: Vec::new(),
            grid: None,
            stores: Vec::new(),
            range: None,
            accumulate: None,
            carries: Vec::new(),
            arena: Vec::new(),
        }
    }

    /// A new node.
    pub fn node(&mut self, kind: &str, args: Vec<NodeId>, attrs: Vec<(&str, Attr)>) -> NodeId {
        self.arena.push(Node {
            kind: kind.to_string(),
            args,
            attrs: attrs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        });
        NodeId(self.arena.len() - 1)
    }

    /// Every node the stores read, operands before their users: from the accumulation,
    /// the carries, then the stores, each operand first, each node once.
    pub fn nodes(&self) -> Vec<NodeId> {
        let mut order = Vec::new();
        let mut seen = vec![false; self.arena.len()];
        let roots = self
            .accumulate
            .iter()
            .map(|a| a.1)
            .chain(self.carries.iter().map(|c| c.1))
            .chain(self.stores.iter().map(|s| s.value));
        for root in roots {
            // (node, its next operand): a walk without recursion
            let mut stack = vec![(root, 0usize)];
            while let Some((n, k)) = stack.pop() {
                if k == 0 {
                    if seen.get(n.0).copied().unwrap_or(true) {
                        continue;
                    }
                    seen[n.0] = true;
                }
                match self.arena[n.0].args.get(k) {
                    Some(&a) => {
                        stack.push((n, k + 1));
                        if !seen.get(a.0).copied().unwrap_or(true) {
                            stack.push((a, 0));
                        }
                    }
                    None => order.push(n),
                }
            }
        }
        order
    }

    /// The IR as text.
    pub fn dump(&self) -> String {
        let mut lines = vec![format!("kernel {}", self.name)];
        for t in &self.tensors {
            let side = if t.output { "out" } else { "in " };
            let dtype = t.dtype.clone().unwrap_or_else(|| "None".to_string());
            lines.push(format!("  {side} {}: {dtype}", t.name));
        }
        for v in &self.values {
            lines.push(format!("  runtime {}: {}", v.name, v.kind));
        }
        for (n, v) in &self.statics {
            lines.push(format!("  static {n} = {v}"));
        }
        let nodes = self.nodes();
        let mut names: Vec<Option<String>> = vec![None; self.arena.len()];
        for (i, n) in nodes.iter().enumerate() {
            names[n.0] = Some(format!("%{i}"));
        }
        let name = |names: &[Option<String>], n: NodeId| {
            names
                .get(n.0)
                .cloned()
                .flatten()
                .unwrap_or_else(|| "?".to_string())
        };
        if let Some(g) = &self.grid {
            lines.push(match &g.cols {
                Some(cols) => format!("  grid i < {}, j < {}", g.count.text(), cols.text()),
                None => format!("  grid unit < {}, block {}", g.count.text(), g.block),
            });
        }
        if let Some(r) = &self.range {
            lines.push(format!("    range {} < {}", r.var.text(), r.count.text()));
        }
        for &n in &nodes {
            let node = &self.arena[n.0];
            let args: Vec<String> = node.args.iter().map(|&a| name(&names, a)).collect();
            let args = args.join(", ");
            // the view first, then the others in order
            let shown = node
                .attrs
                .iter()
                .filter(|(k, _)| k == "view")
                .chain(node.attrs.iter().filter(|(k, _)| k != "view"));
            let extra: Vec<String> = shown.map(|(k, v)| format!("{k}={}", v.text())).collect();
            let extra = extra.join(", ");
            let inner: Vec<&str> = [args.as_str(), extra.as_str()]
                .into_iter()
                .filter(|x| !x.is_empty())
                .collect();
            lines.push(format!(
                "    {} = {}({})",
                name(&names, n),
                node.kind,
                inner.join(", ")
            ));
        }
        if let Some((acc, product)) = self.accumulate {
            if names.get(acc.0).is_some_and(Option::is_none) {
                names[acc.0] = Some("acc".to_string());
            }
            lines.push(format!("    acc += {}", name(&names, product)));
        }
        for &(c, value) in &self.carries {
            lines.push(format!(
                "    next {} = {}",
                name(&names, c),
                name(&names, value)
            ));
        }
        for s in &self.stores {
            lines.push(format!(
                "    store {} = {}",
                s.view.text(),
                name(&names, s.value)
            ));
        }
        lines.join("\n")
    }
}

/// |x| from which every float32 is an integer (today's `trace.WHOLE`, 2^23).
const WHOLE: f64 = 8388608.0;

/// A value of the body while it is traced: a block's node, or a scalar, which today's
/// tracer makes a node of afresh at each use (a number or a runtime value).
#[derive(Clone, Debug)]
enum Traced {
    Tile(NodeId),
    Const(f64),
    Value(String),
}

/// Today's name of the format of a tile the tile level reads, or why not.
fn dtype(format: Format) -> Result<&'static str, String> {
    match format {
        Format::F32 => Ok("float32"),
        Format::Bf16 => Ok("bfloat16"),
        Format::F16 => Ok("float16"),
        Format::Bfp8 | Format::Bfp4 | Format::Bfp2 => Err(format!(
            "a tile of {format:?}: today's tile level has two families of it (`_b`, read into an fp32 Dst, and the A family, into a 16-bit one) and the module names neither"
        )),
        other => Err(format!("a tile of {other:?}")),
    }
}

/// The tracer over one kernel: its IR, the module, and the functions being traced
/// (a call of one of them again is refused).
struct Tracer<'m> {
    module: &'m Module,
    ir: KernelIR,
    calls: Vec<FuncId>,
}

impl Tracer<'_> {
    /// A node of `value`: a block's own node, or a new node of a scalar.
    fn node(&mut self, value: &Traced) -> NodeId {
        match value {
            Traced::Tile(n) => *n,
            Traced::Const(x) => self
                .ir
                .node("const", vec![], vec![("value", Attr::Float(*x))]),
            Traced::Value(name) => {
                self.ir
                    .node("param", vec![], vec![("value", Attr::Value(name.clone()))])
            }
        }
    }

    /// `kind` of `args`, each made a node in order.
    fn op(&mut self, kind: &str, args: &[Traced], attrs: Vec<(&str, Attr)>) -> Traced {
        let nodes = args.iter().map(|a| self.node(a)).collect();
        Traced::Tile(self.ir.node(kind, nodes, attrs))
    }

    /// `where(a cmp b, x, y)`, as today's comparison operators make it: the block on
    /// the left (a number on the left is Python's reflected comparison, flipped).
    fn select(&mut self, cmp: &str, a: Traced, b: Traced, x: Traced, y: Traced) -> Traced {
        self.op(
            "where",
            &[a, b, x, y],
            vec![("compare", Attr::Text(cmp.into()))],
        )
    }

    /// Today's `floor(x)`: x rounded to nearest by the 2^23 sum, less 1 where that is
    /// above x; x itself from 2^23 up.
    fn floor(&mut self, x: Traced) -> Traced {
        let (w, z) = (Traced::Const(WHOLE), Traced::Const(0.0));
        let m = self.select("lt", x.clone(), z.clone(), Traced::Const(-WHOLE), w.clone());
        let sum = self.op("add", &[x.clone(), m.clone()], vec![]);
        let r = self.op("sub", &[sum, m], vec![]);
        let over = self.op("sub", &[r.clone(), x.clone()], vec![]);
        let less = self.op("sub", &[r.clone(), Traced::Const(1.0)], vec![]);
        let inner = self.select("gt", over, z, less, r);
        let size = self.op("abs", std::slice::from_ref(&x), vec![]);
        self.select("lt", size, w, inner, x)
    }

    /// The value `body` returns given `args`, or why it is not traced.
    fn body(&mut self, f: FuncId, body: &Body, args: &[Traced]) -> Result<Traced, Refusal> {
        let refused = |i: usize, what: String| Refusal {
            code: NOT_BUILT,
            what: format!("the Tensix backend's tile compiler does not take {what} yet"),
            at: Some(Anchor::Inst(BodyRef::Func(f), InstId(i as u32))),
        };
        let mut values: Vec<Option<Traced>> = vec![None; body.values.len()];
        let get = |values: &[Option<Traced>], i: usize, v: ValueId| {
            values
                .get(v.0 as usize)
                .cloned()
                .flatten()
                .ok_or_else(|| refused(i, "a value made by an operation it refused".into()))
        };
        let tile = |values: &[Option<Traced>], i: usize, v: ValueId| match get(values, i, v)? {
            t @ Traced::Tile(_) => Ok(t),
            _ => Err(refused(i, "a function of one tile on a scalar".into())),
        };
        for (i, inst) in body.insts.iter().enumerate() {
            let made = match &inst.op {
                Op::Arg(k) => args
                    .get(*k as usize)
                    .cloned()
                    .ok_or_else(|| refused(i, "an argument the call does not pass".into()))?,
                Op::Const(ConstValue::Float(ScalarType::F32, x)) => Traced::Const(*x),
                Op::Const(c) => return Err(refused(i, format!("a constant {c:?}"))),
                Op::Binary(op, a, b) => {
                    let kind = match op {
                        BinOp::Add => "add",
                        BinOp::Sub => "sub",
                        BinOp::Mul => "mul",
                        BinOp::Div => "div",
                        BinOp::Min => "min",
                        BinOp::Max => "max",
                        other => return Err(refused(i, format!("the operation {other:?}"))),
                    };
                    let (a, b) = (get(&values, i, *a)?, get(&values, i, *b)?);
                    if !matches!((&a, &b), (Traced::Tile(_), _) | (_, Traced::Tile(_))) {
                        return Err(refused(i, "arithmetic on two scalars".into()));
                    }
                    self.op(kind, &[a, b], vec![])
                }
                Op::Unary(op, a) => {
                    let a = tile(&values, i, *a)?;
                    let kind = match op {
                        UnaryOp::Floor => {
                            values_set(&mut values, &inst.defs, self.floor(a));
                            continue;
                        }
                        UnaryOp::Neg => "neg",
                        UnaryOp::Abs => "abs",
                        UnaryOp::Relu => "relu",
                        UnaryOp::Square => "square",
                        UnaryOp::Exp => "exp",
                        UnaryOp::Exp2 => "exp2",
                        UnaryOp::Tanh => "tanh",
                        UnaryOp::Erf => "erf",
                        UnaryOp::Log => "log",
                        UnaryOp::Sigmoid => "sigmoid",
                        UnaryOp::Silu => "silu",
                        UnaryOp::Sqrt => "sqrt",
                        UnaryOp::Rsqrt => "rsqrt",
                        UnaryOp::Reciprocal => "recip",
                    };
                    self.op(kind, &[a], vec![])
                }
                Op::Select { cmp, a, b, x, y } => {
                    let (a, b) = (get(&values, i, *a)?, get(&values, i, *b)?);
                    let (x, y) = (get(&values, i, *x)?, get(&values, i, *y)?);
                    let cmp = match cmp {
                        CmpOp::Gt => "gt",
                        CmpOp::Lt => "lt",
                        CmpOp::Ge => "ge",
                        CmpOp::Le => "le",
                        CmpOp::Eq => "eq",
                        CmpOp::Ne => "ne",
                    };
                    match (&a, &b) {
                        (Traced::Tile(_), _) => self.select(cmp, a, b, x, y),
                        (_, Traced::Tile(_)) => {
                            let flipped = match cmp {
                                "gt" => "lt",
                                "lt" => "gt",
                                "ge" => "le",
                                "le" => "ge",
                                same => same,
                            };
                            self.select(flipped, b, a, x, y)
                        }
                        _ => return Err(refused(i, "a comparison of two scalars".into())),
                    }
                }
                Op::Matmul(a, b) => {
                    let (a, b) = (tile(&values, i, *a)?, tile(&values, i, *b)?);
                    self.op("matmul", &[a, b], vec![])
                }
                Op::CallFunc(g, operands) => {
                    let callee = self.module.funcs.get(g.0 as usize);
                    let Some(callee) = callee.filter(|c| matches!(c.kind, FuncKind::Plain)) else {
                        return Err(refused(
                            i,
                            "a call of a function that is not a `func`".into(),
                        ));
                    };
                    if self.calls.contains(g) {
                        return Err(refused(i, "a function that calls itself".into()));
                    }
                    let Some(inner) = callee.body.as_ref() else {
                        return Err(refused(i, "a call of a function with no body".into()));
                    };
                    let given = operands
                        .iter()
                        .map(|v| get(&values, i, *v))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.calls.push(*g);
                    let result = self.body(*g, inner, &given)?;
                    self.calls.pop();
                    result
                }
                Op::Return(Some(v)) => return tile(&values, i, *v),
                other => {
                    let name = format!("{other:?}");
                    let name = name.split([' ', '(', '{']).next().unwrap_or("").to_string();
                    return Err(refused(i, format!("the operation {name}")));
                }
            };
            values_set(&mut values, &inst.defs, made);
        }
        Err(refused(
            body.insts.len().saturating_sub(1),
            "a body that returns no tile".into(),
        ))
    }
}

/// Each value `defs` defines set to `made`.
fn values_set(values: &mut [Option<Traced>], defs: &[ValueId], made: Traced) {
    for d in defs {
        if let Some(slot) = values.get_mut(d.0 as usize) {
            *slot = Some(made.clone());
        }
    }
}

/// The KernelIR of function `f` traced as today's `tile_unit` traces a tile function
/// at the 1-tile width, under the name `name` (today's C++ name of the unit): its tile
/// parameters are tensors, its `f32` ones runtime floats, the grid walks the first
/// tensor's tiles, and the result is stored into the `float32` tensor `result`.
pub fn kernel_ir(module: &Module, f: FuncId, name: &str) -> Result<KernelIR, Refusal> {
    let refused = |what: String, at: Anchor| Refusal {
        code: NOT_BUILT,
        what: format!("the Tensix backend's tile compiler does not take {what} yet"),
        at: Some(at),
    };
    let Some(func) = module.funcs.get(f.0 as usize) else {
        return Err(refused("a call of no function".into(), Anchor::Func(f)));
    };
    if !matches!(func.kind, FuncKind::Plain) {
        return Err(refused(
            "a function that is not a `func`".into(),
            Anchor::Func(f),
        ));
    }
    if !matches!(func.ret, Type::Tile(_)) {
        return Err(refused(
            "a function whose result is not one tile".into(),
            Anchor::Func(f),
        ));
    }
    let Some(body) = func.body.as_ref() else {
        return Err(refused("a function with no body".into(), Anchor::Func(f)));
    };
    let (mut tensors, mut values, mut args) = (Vec::new(), Vec::new(), Vec::new());
    for (i, ty) in func.params.iter().enumerate() {
        let at = Anchor::ParamOf(Owner::Func(f), i as u32);
        let Some(p) = param_name(module, Owner::Func(f), i as u32) else {
            return Err(refused("a parameter with no name".into(), at));
        };
        match ty {
            Type::Tile(format) => {
                let dtype = dtype(*format).map_err(|what| refused(what, at))?;
                tensors.push(TensorParam {
                    name: p.to_string(),
                    output: false,
                    dtype: Some(dtype.to_string()),
                });
                args.push(None);
            }
            Type::Scalar(ScalarType::F32) => {
                values.push(ValueParam {
                    name: p.to_string(),
                    kind: "float".to_string(),
                });
                args.push(Some(Traced::Value(p.to_string())));
            }
            other => return Err(refused(format!("a parameter of {other:?}"), at)),
        }
    }
    let Some(first) = tensors.first().map(|t| t.name.clone()) else {
        return Err(refused("a function of no tile".into(), Anchor::Func(f)));
    };
    let loads: Vec<String> = tensors.iter().map(|t| t.name.clone()).collect();
    tensors.push(TensorParam {
        name: "result".to_string(),
        output: true,
        dtype: Some("float32".to_string()),
    });
    let mut tracer = Tracer {
        module,
        ir: KernelIR::new(name, tensors, values),
        calls: vec![f],
    };
    tracer.ir.grid = Some(Grid {
        count: Int::Size {
            what: "tiles".into(),
            name: first,
            dim: None,
        },
        block: 1,
        cols: None,
    });
    let tile = |tensor: &str| View {
        tensor: tensor.to_string(),
        kind: "tile".into(),
        index: Int::Unit(None),
        cols: None,
    };
    // the loads first, in the order of the tensors, as today's call site makes them
    let mut loaded = Vec::new();
    for t in &loads {
        let view = Attr::View(tile(t));
        let attrs = vec![("view", view), ("pad", Attr::None), ("share", Attr::None)];
        loaded.push(Traced::Tile(tracer.ir.node("load", vec![], attrs)));
    }
    let mut loaded = loaded.into_iter();
    let args: Vec<Traced> = args
        .into_iter()
        .filter_map(|a| a.or_else(|| loaded.next()))
        .collect();
    let result = tracer.body(f, body, &args)?;
    let value = tracer.node(&result);
    tracer.ir.stores.push(Store {
        view: tile("result"),
        value,
    });
    Ok(tracer.ir)
}
