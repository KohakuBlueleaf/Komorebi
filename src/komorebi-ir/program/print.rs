//! The program structure as text: one line for each declaration and each instruction,
//! for reading and for golden tests. Not a format anything parses; the byte code is the
//! encoded form.
//!
//! Ids are written by their place: `f0` a function, `p0` a program, `c0` a control
//! program, `P0` a module parameter, `R0` a record type, `m0` a message type; inside a
//! program `s0` a state member, `h0` a handler, `r0` a run, `k0` a chain; inside a body
//! `i0` an instruction, `v0` a value, `l0` a local slot. A body is named by its owner:
//! `f0`, `p0.h0`, `p0.r0.seed`, `p0.r0.on0` (the run's first `on`), `c0`. An instruction
//! line is its place, the values it defines with their types, then the operation with
//! its operands in the order the operation declares them.

use super::body::{Body, Inst, Op, Range, ScalarsFrom, ValueId};
use super::derive::{Cause, Effects};
use super::notes::{Anchor, HintArg, Named, Owner};
use super::{BodyRef, CallArg, FuncKind, MemberKind, Module};
use crate::types::{
    Ability, Axis, BinOp, Binding, CmpOp, Combiner, ConstValue, Count, Entry, On, Peers, SplitBy,
    Type,
};

/// The module as text, in the module's order, the notes last.
pub fn print(module: &Module) -> String {
    let mut p = Printer { out: String::new() };
    p.module(module);
    p.out
}

struct Printer {
    out: String,
}

fn list<T>(items: &[T], f: impl Fn(&T) -> String) -> String {
    let parts: Vec<String> = items.iter().map(f).collect();
    format!("[{}]", parts.join(", "))
}

/// A declaration's tile shape as the text writes it: nothing for the default (place 0),
/// else `  tile [R, C]` (`  tile #i` for a place past the module's shapes).
fn shape(m: &Module, i: u32) -> String {
    match (i, m.shape(i)) {
        (0, _) => String::new(),
        (_, Some(t)) => format!("  tile [{}, {}]", t.rows, t.cols),
        (_, None) => format!("  tile #{i}"),
    }
}

fn yes(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

fn v(x: &ValueId) -> String {
    format!("v{}", x.0)
}

fn opt(x: &Option<ValueId>) -> String {
    x.as_ref().map_or("none".to_string(), v)
}

fn values(xs: &[ValueId]) -> String {
    list(xs, v)
}

/// Where a body's words come from: a value, or a state member `s0`.
fn from(b: &ScalarsFrom) -> String {
    match b {
        ScalarsFrom::Value(x) => v(x),
        ScalarsFrom::Member(m) => format!("s{}", m.0),
    }
}

fn count(c: &Count) -> String {
    match c {
        Count::Const(n) => n.to_string(),
        Count::Param(p) => format!("P{}", p.0),
        Count::GroupSize => "group.size".into(),
    }
}

fn axis(a: Axis) -> &'static str {
    match a {
        Axis::Row => "row",
        Axis::Col => "col",
    }
}

/// A type in short: `tile<bf16>`, `view<bf16>`, `stream<bf16, 1>`, `message<m0>`.
pub fn type_text(t: &Type) -> String {
    let ret = |r: &Option<crate::types::ScalarType>| {
        r.map_or("none".to_string(), |s| s.name().to_string())
    };
    match t {
        Type::Scalar(s) => s.name().into(),
        Type::Bool => "bool".into(),
        Type::Tile(f) => format!("tile<{f}>"),
        Type::Reduced(f, a, Count::Const(1)) => format!("reduced<{f}, {}>", axis(*a)),
        Type::Reduced(f, a, n) => format!("reduced<{f}, {}, {}>", axis(*a), count(n)),
        Type::Block(f, n) => format!("tile<{f}>[{}]", count(n)),
        Type::Tensor(f, false) => format!("tensor<{f}>"),
        Type::Tensor(f, true) => format!("tensor<{f}> replicated"),
        Type::View(Some(f)) => format!("view<{f}>"),
        Type::View(None) => "view".into(),
        Type::Items(c) => format!("items<{}>", count(c)),
        Type::Stream(f, n) => format!("stream<{f}, {}>", count(n)),
        Type::Zip(a, b) => format!("zip<{}, {}>", type_text(a), type_text(b)),
        Type::Sink(f, n) => format!("sink<{f}, {}>", count(n)),
        Type::Words(s) => format!("words<{s}>"),
        Type::Records(r) => format!("records<R{}>", r.0),
        Type::RecordValue(r) => format!("record<R{}>", r.0),
        Type::Message(m) => format!("message<m{}>", m.0),
        Type::Body => "body".into(),
        Type::SubGroups => "subgroups".into(),
        Type::Leaders => "leaders".into(),
        Type::Broadcast(f) => format!("broadcast<{f}>"),
        Type::Group => "group".into(),
        Type::Handle(r) => format!("handle<{}>", ret(r)),
        Type::Outcome(r) => format!("outcome<{}>", ret(r)),
        Type::Reason => "reason".into(),
    }
}

fn types(ts: &[Type]) -> String {
    list(ts, type_text)
}

/// A constant: its type, then its value (an integer in decimal, signed where its type
/// is; a float as Rust writes an `f64` for reading back).
pub fn const_text(c: &ConstValue) -> String {
    match c {
        ConstValue::Int(s, bits) => {
            let w = s.bits();
            let value = if s.is_signed() && w < 64 && (bits >> (w - 1)) & 1 == 1 {
                (*bits as i128) - (1i128 << w)
            } else if s.is_signed() {
                i128::from(*bits as i64)
            } else {
                i128::from(*bits)
            };
            format!("{s} {value}")
        }
        ConstValue::Float(s, x) => format!("{s} {x:?}"),
        ConstValue::Bool(b) => b.to_string(),
    }
}

/// A body's place as the printed text writes it: `f0`, `p0.h1`, `p0.r0.seed`,
/// `p0.r0.on1`, `c0`.
pub fn body_ref(b: &BodyRef) -> String {
    match b {
        BodyRef::Func(f) => format!("f{}", f.0),
        BodyRef::Handler(p, h) => format!("p{}.h{}", p.0, h.0),
        BodyRef::Seed(p, r) => format!("p{}.r{}.seed", p.0, r.0),
        BodyRef::On(p, r, k) => format!("p{}.r{}.on{k}", p.0, r.0),
        BodyRef::Control(c) => format!("c{}", c.0),
    }
}

fn entry(e: &Entry) -> String {
    match e {
        Entry::Handler(p, h) => format!("p{}.h{}", p.0, h.0),
        Entry::Run(p, r) => format!("p{}.r{}", p.0, r.0),
        Entry::Chain(p, c) => format!("p{}.k{}", p.0, c.0),
    }
}

fn owner(o: &Owner) -> String {
    match o {
        Owner::Func(f) => format!("f{}", f.0),
        Owner::Handler(p, h) => format!("p{}.h{}", p.0, h.0),
        Owner::Run(p, r) => format!("p{}.r{}", p.0, r.0),
        Owner::Chain(p, c) => format!("p{}.k{}", p.0, c.0),
        Owner::Control(c) => format!("c{}", c.0),
    }
}

fn anchor(a: &Anchor) -> String {
    match a {
        Anchor::Module => "module".into(),
        Anchor::Func(f) => format!("f{}", f.0),
        Anchor::Program(p) => format!("p{}", p.0),
        Anchor::Handler(p, h) => format!("p{}.h{}", p.0, h.0),
        Anchor::Run(p, r) => format!("p{}.r{}", p.0, r.0),
        Anchor::Member(p, s) => format!("p{}.s{}", p.0, s.0),
        Anchor::Control(c) => format!("c{}", c.0),
        Anchor::Chain(p, c) => format!("p{}.k{}", p.0, c.0),
        Anchor::Call(p, c, k) => format!("p{}.k{}.call{k}", p.0, c.0),
        Anchor::Param(x) => format!("P{}", x.0),
        Anchor::Message(m) => format!("m{}", m.0),
        Anchor::Record(r) => format!("R{}", r.0),
        Anchor::ParamOf(o, i) => format!("{} param {i}", owner(o)),
        Anchor::Inst(b, i) => format!("{} i{}", body_ref(b), i.0),
    }
}

fn combiner(c: &Combiner) -> String {
    match c {
        Combiner::Func(f) => format!("f{}", f.0),
        other => format!("{other:?}"),
    }
}

fn range(r: &Range) -> String {
    match r.step {
        Some(s) => format!("{}:{}:{}", v(&r.lo), v(&r.hi), v(&s)),
        None => format!("{}:{}", v(&r.lo), v(&r.hi)),
    }
}

fn hint_arg(a: &HintArg) -> String {
    match a {
        HintArg::Number(n) | HintArg::Name(n) => n.clone(),
        HintArg::Text(t) => format!("{t:?}"),
        HintArg::Named(n, x) => format!("{n} = {}", hint_arg(x)),
        HintArg::List(xs) => list(xs, hint_arg),
    }
}

fn binop(b: &BinOp) -> String {
    format!("{b:?}")
}

fn cmpop(c: &CmpOp) -> String {
    format!("{c:?}")
}

fn split(s: &SplitBy) -> String {
    format!("{s:?}")
}

/// An operation and its operands.
pub fn op_text(op: &Op) -> String {
    use Op::*;
    match op {
        Const(c) => format!("Const {}", const_text(c)),
        Arg(i) => format!("Arg {i}"),
        Param(p) => format!("Param P{}", p.0),
        Get(l) => format!("Get l{}", l.0),
        Set(l, x) => format!("Set l{} {}", l.0, v(x)),
        Binary(b, x, y) => format!("Binary {} {} {}", binop(b), v(x), v(y)),
        Compare(c, x, y) => format!("Compare {} {} {}", cmpop(c), v(x), v(y)),
        LogicAnd(x, y) => format!("LogicAnd {} {}", v(x), v(y)),
        LogicOr(x, y) => format!("LogicOr {} {}", v(x), v(y)),
        If { cond } => format!("If {{ cond {} }}", v(cond)),
        Else => "Else".into(),
        ForRange { n } => format!("ForRange {{ n {} }}", v(n)),
        ForStream { stream } => format!("ForStream {{ stream {} }}", v(stream)),
        ForZip { streams } => format!(
            "ForZip {{ {} }}",
            streams.iter().map(v).collect::<Vec<_>>().join(", ")
        ),
        ForArrive { broadcast } => format!("ForArrive {{ broadcast {} }}", v(broadcast)),
        While => "While".into(),
        WhileTest { cond } => format!("WhileTest {{ cond {} }}", v(cond)),
        End => "End".into(),
        Return(None) => "Return".into(),
        Return(Some(x)) => format!("Return {}", v(x)),
        Convert(s, x) => format!("Convert {s} {}", v(x)),
        Unary(u, x) => format!("Unary {u:?} {}", v(x)),
        Select { cmp, a, b, x, y } => format!(
            "Select {{ cmp {}, a {}, b {}, x {}, y {} }}",
            cmpop(cmp),
            v(a),
            v(b),
            v(x),
            v(y)
        ),
        Reduce(r, x, a) => format!("Reduce {r:?} {} {}", v(x), axis(*a)),
        SpreadCol(x) => format!("SpreadCol {}", v(x)),
        SpreadRow(x) => format!("SpreadRow {}", v(x)),
        Matmul(x, y) => format!("Matmul {} {}", v(x), v(y)),
        Transpose(x) => format!("Transpose {}", v(x)),
        Zeros => "Zeros".into(),
        CallFunc(f, args) => format!("CallFunc f{} {}", f.0, values(args)),
        Tiles { tensor, first, n } => {
            format!(
                "Tiles {{ tensor {}, first {}, n {} }}",
                v(tensor),
                v(first),
                v(n)
            )
        }
        Rect { tensor, rows, cols } => format!(
            "Rect {{ tensor {}, rows {}, cols {} }}",
            v(tensor),
            range(rows),
            range(cols)
        ),
        Turn(x) => format!("Turn {}", v(x)),
        Items {
            tensor,
            range,
            size,
        } => match range {
            Some((f, n)) => format!(
                "Items {{ tensor {}, first {}, n {}, size {} }}",
                v(tensor),
                v(f),
                v(n),
                v(size)
            ),
            None => format!("Items {{ tensor {}, size {} }}", v(tensor), v(size)),
        },
        ReadTile(x, i) => format!("ReadTile {} {}", v(x), v(i)),
        Words(x) => format!("Words {}", v(x)),
        Word(x, i) => format!("Word {} {}", v(x), v(i)),
        Records(x, r, c) => format!("Records {} R{} {}", v(x), r.0, v(c)),
        RecordAt(x, i) => format!("RecordAt {} {}", v(x), v(i)),
        RecordField(x, k) => format!("RecordField {} {k}", v(x)),
        Put { items, src, at } => {
            format!("Put {{ items {}, src s{}, at {} }}", v(items), src.0, v(at))
        }
        Partition(x, r, s) => format!("Partition {} {} {}", v(x), v(r), v(s)),
        Stream(x, n) => format!("Stream {} {}", v(x), v(n)),
        Sink(x, n) => format!("Sink {} {}", v(x), v(n)),
        Push(s, x) => format!("Push {} {}", v(s), v(x)),
        Store(s, x) => format!("Store {} {}", v(s), v(x)),
        StoreScalars(s, x) => format!("StoreScalars {} {}", v(s), from(x)),
        BlockAt(x, i) => format!("BlockAt {} {}", v(x), v(i)),
        Fill(m, x) => format!("Fill s{} {}", m.0, v(x)),
        ZeroState(m) => format!("ZeroState s{}", m.0),
        StateSet(m, i, x) => format!("StateSet s{} {} {}", m.0, v(i), v(x)),
        StateGet(m, i) => format!("StateGet s{} {}", m.0, v(i)),
        Bump(m, i) => format!("Bump s{} {}", m.0, v(i)),
        Rank => "Rank".into(),
        IsLeader => "IsLeader".into(),
        WorkerId => "WorkerId".into(),
        Row => "Row".into(),
        Col => "Col".into(),
        GroupSize => "GroupSize".into(),
        GroupId(x) => format!("GroupId {}", v(x)),
        Split(s, x) => format!("Split {} {}", split(s), v(x)),
        Leaders(s, x) => format!("Leaders {} {}", split(s), v(x)),
        ReduceGroup {
            x,
            with,
            within,
            among,
            publish,
        } => format!(
            "ReduceGroup {{ x {}, with {}, within {}, among {}, publish {} }}",
            v(x),
            combiner(with),
            opt(within),
            opt(among),
            values(publish)
        ),
        AllReduce { x, with, publish } => format!(
            "AllReduce {{ x {}, with {}, publish {} }}",
            v(x),
            combiner(with),
            values(publish)
        ),
        Gather(x) => format!("Gather {}", v(x)),
        AllGather { value, publish } => format!(
            "AllGather {{ value {}, publish {} }}",
            v(value),
            values(publish)
        ),
        Broadcast(a, x) => format!("Broadcast {} {}", axis(*a), v(x)),
        IsSender(x) => format!("IsSender {}", v(x)),
        Cast(b, x, w) => format!("Cast {} {} {}", v(b), v(x), opt(w)),
        Sync(xs) => format!("Sync {}", values(xs)),
        Arrived => "Arrived".into(),
        MakeMessage(m, xs) => format!("MakeMessage m{} {}", m.0, values(xs)),
        Send {
            message,
            to,
            body,
            publish,
            used,
        } => format!(
            "Send {{ message {}, to {}{}, publish {}, used {} }}",
            v(message),
            v(to),
            body.map_or(String::new(), |b| format!(", body {}", from(&b))),
            values(publish),
            yes(*used)
        ),
        Forward {
            message,
            as_type,
            with,
            to,
            publish,
        } => format!(
            "Forward {{ message {}{}, with {}, to {}, publish {} }}",
            v(message),
            as_type.map_or(String::new(), |t| format!(", as m{}", t.0)),
            list(with, |(k, x)| format!("{k} {}", v(x))),
            v(to),
            values(publish)
        ),
        Consume(x) => format!("Consume {}", v(x)),
        Offer(x, to, None) => format!("Offer {} {}", v(x), v(to)),
        Offer(x, to, Some(b)) => format!("Offer {} {} body {}", v(x), v(to), from(b)),
        Field(x, k) => format!("Field {} {k}", v(x)),
        Op::Body(x) => format!("Body {}", v(x)),
        MakeGroup(first, n) => format!("MakeGroup {} {}", opt(first), v(n)),
        Invoke {
            entry: e,
            args,
            on,
            after,
            after_end,
        } => format!(
            "Invoke {{ entry {}, args {}, on {}, after {}, after_end {} }}",
            entry(e),
            values(args),
            v(on),
            values(after),
            values(after_end)
        ),
        Await(x) => format!("Await {}", v(x)),
        Outcome(x) => format!("Outcome {}", v(x)),
        Failed(x) => format!("Failed {}", v(x)),
        Reason(x) => format!("Reason {}", v(x)),
        OutcomeValue(x) => format!("OutcomeValue {}", v(x)),
        FailureName(f) => format!("FailureName {}", f.name()),
        IslandCount => "IslandCount".into(),
        IslandOf(x) => format!("IslandOf {}", v(x)),
        IslandCrosses(x) => format!("IslandCrosses {}", v(x)),
        IslandHops(x) => format!("IslandHops {}", v(x)),
        ReduceTo(t, to, with) => format!("ReduceTo {} {} {}", v(t), v(to), combiner(with)),
        BroadcastFrom(t, from) => format!("BroadcastFrom {} {}", v(t), v(from)),
    }
}

impl Printer {
    fn line(&mut self, depth: usize, text: &str) {
        for _ in 0..depth {
            self.out.push_str("  ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    /// A list under a heading: `name: []` when empty, else the heading and one line
    /// for each item, one level deeper.
    fn section(&mut self, depth: usize, name: &str, items: Vec<String>) {
        if items.is_empty() {
            self.line(depth, &format!("{name}: []"));
        } else {
            self.line(depth, &format!("{name}:"));
            for item in items {
                self.line(depth + 1, &item);
            }
        }
    }

    fn inst(&mut self, depth: usize, i: usize, inst: &Inst, body: &Body) {
        let place = format!("i{i}");
        let mut text = format!("{place:<4}");
        if !text.ends_with(' ') {
            text.push(' ');
        }
        if !inst.defs.is_empty() {
            let defs: Vec<String> = inst
                .defs
                .iter()
                .map(|d| {
                    let t = body.values.get(d.0 as usize);
                    format!("{}: {}", v(d), t.map_or("?".to_string(), type_text))
                })
                .collect();
            text.push_str(&defs.join(", "));
            text.push_str(" = ");
        }
        text.push_str(&op_text(&inst.op));
        self.line(depth, &text);
    }

    fn body(&mut self, depth: usize, body: &Body) {
        if !body.locals.is_empty() {
            let locals: Vec<String> = body
                .locals
                .iter()
                .enumerate()
                .map(|(k, t)| format!("l{k} {}", type_text(t)))
                .collect();
            self.line(depth, &format!("locals: {}", locals.join(", ")));
        }
        for (i, inst) in body.insts.iter().enumerate() {
            self.inst(depth, i, inst, body);
        }
    }

    fn effects(&mut self, depth: usize, e: &Effects) {
        self.line(depth, "effects:");
        let tensors = e
            .tensors
            .iter()
            .map(|a| {
                let body = body_ref(&a.body);
                let (view, at) = (a.view.0, a.at.0);
                format!(
                    "param {}  body {body}  view i{view}  {:?}  at i{at}",
                    a.param, a.access
                )
            })
            .collect();
        self.section(depth + 1, "tensors", tensors);
        let state = e
            .state
            .iter()
            .map(|s| {
                format!(
                    "s{}  body {}  {:?}  at i{}",
                    s.member.0,
                    body_ref(&s.body),
                    s.how,
                    s.at.0
                )
            })
            .collect();
        self.section(depth + 1, "state", state);
        let collectives = e
            .collectives
            .iter()
            .map(|(b, i)| format!("{} i{}", body_ref(b), i.0))
            .collect();
        self.section(depth + 1, "collectives", collectives);
        let calls = list(&e.calls, |f| format!("f{}", f.0));
        self.line(depth + 1, &format!("calls: {calls}"));
    }

    fn module(&mut self, m: &Module) {
        let formats = list(&m.formats, |f| f.name().to_string());
        let tiles: Vec<String> = m
            .tiles
            .iter()
            .map(|t| format!("[{}, {}]", t.rows, t.cols))
            .collect();
        let head = format!(
            "module {:?}  tile {}  formats {formats}",
            m.name,
            tiles.join(", ")
        );
        self.line(0, &head);
        let params = m
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let binding = match &p.binding {
                    Binding::Build(c) => format!("build {}", const_text(c)),
                    Binding::Load(None) => "load fit".into(),
                    Binding::Load(Some(c)) => format!("load {}", const_text(c)),
                    Binding::Invoke(c) => format!("invoke at most {}", const_text(c)),
                };
                format!("P{i} {}: {}  {binding}", p.name, p.ty)
            })
            .collect();
        self.section(0, "params", params);
        let program =
            |p: Option<super::ProgramId>| p.map_or(String::new(), |p| format!("  in p{}", p.0));
        let fields = |fs: &[(String, crate::types::ScalarType)]| {
            let parts: Vec<String> = fs.iter().map(|(n, s)| format!("{n}: {s}")).collect();
            format!("{{ {} }}", parts.join(", "))
        };
        let records = m
            .records
            .iter()
            .enumerate()
            .map(|(i, r)| {
                format!(
                    "R{i} {} {}{}",
                    r.name,
                    fields(&r.fields),
                    program(r.program)
                )
            })
            .collect();
        self.section(0, "records", records);
        let messages = m
            .messages
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let body = match x.body {
                    0 => String::new(),
                    n => format!(" body {n}"),
                };
                format!(
                    "m{i} {} {}{body}{}",
                    x.name,
                    fields(&x.fields),
                    program(x.program)
                )
            })
            .collect();
        self.section(0, "messages", messages);
        let requirements = m
            .requirements
            .iter()
            .map(|r| {
                let ability = match &r.ability {
                    Ability::LocalState { amount, tiles } => {
                        format!("LocalState {amount} tiles {}", yes(*tiles))
                    }
                    Ability::Grid => "Grid".into(),
                    Ability::Dynamic => "Dynamic".into(),
                    Ability::Islands(n) => format!("Islands {n}"),
                    Ability::Impl(name) => format!("Impl {name:?}"),
                };
                let causes = list(&r.causes, |c| match c {
                    Cause::Stated => "Stated".into(),
                    Cause::GridHandler(p, h) => format!("GridHandler p{}.h{}", p.0, h.0),
                    Cause::DynamicRun(e) => format!("DynamicRun {}", entry(e)),
                    Cause::ExternWithoutFallback(f) => format!("ExternWithoutFallback f{}", f.0),
                });
                format!("{ability}  causes {causes}")
            })
            .collect();
        self.section(0, "requirements", requirements);
        self.funcs(m);
        self.programs(m);
        self.controls(m);
        self.notes(m);
    }

    fn funcs(&mut self, m: &Module) {
        if m.funcs.is_empty() {
            self.line(0, "funcs: []");
            return;
        }
        self.line(0, "funcs:");
        for (i, f) in m.funcs.iter().enumerate() {
            let kind = match &f.kind {
                FuncKind::Plain => "Plain".to_string(),
                FuncKind::Combine { commutative } => {
                    format!("Combine commutative {}", yes(*commutative))
                }
                FuncKind::Extern {
                    implementation,
                    reads,
                    writes,
                } => format!(
                    "Extern {implementation:?} reads {} writes {}",
                    list(reads, u32::to_string),
                    list(writes, u32::to_string)
                ),
            };
            let head = format!(
                "f{i} {}  {kind}  params {} -> {}{}",
                f.name,
                types(&f.params),
                type_text(&f.ret),
                shape(m, f.tile)
            );
            self.line(1, &head);
            if let Some(body) = &f.body {
                self.body(2, body);
                if matches!(f.kind, FuncKind::Extern { .. }) {
                    self.effects(2, &f.effects);
                }
            }
        }
    }

    fn programs(&mut self, m: &Module) {
        if m.programs.is_empty() {
            self.line(0, "programs: []");
            return;
        }
        self.line(0, "programs:");
        for (p, prog) in m.programs.iter().enumerate() {
            self.line(1, &format!("p{p} {}", prog.name));
            let state = prog
                .state
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let kind = match s.kind {
                        MemberKind::Tiles(f) => format!("Tiles {f}"),
                        MemberKind::Scalars(t) => format!("Scalars {t}"),
                    };
                    format!(
                        "s{i} {}  {kind}  count {}  scope {:?}  lifetime {:?}  export {}  by_element {}{}",
                        s.name,
                        count(&s.count),
                        s.scope,
                        s.lifetime,
                        yes(s.export),
                        yes(s.by_element),
                        shape(m, s.tile)
                    )
                })
                .collect();
            self.section(2, "state", state);
            for (h, handler) in prog.handlers.iter().enumerate() {
                let on = match handler.on {
                    On::Group => "Group".to_string(),
                    On::Grid(r, c) => format!("Grid {r}x{c}"),
                };
                let ret = handler
                    .ret
                    .map_or("none".to_string(), |s| s.name().to_string());
                let head = format!(
                    "h{h} {}  params {}  ret {ret}  on {on}  exclusive {}{}",
                    handler.name,
                    types(&handler.params),
                    yes(handler.exclusive),
                    shape(m, handler.tile)
                );
                self.line(2, &head);
                self.body(3, &handler.body);
                self.effects(3, &handler.effects);
            }
            for (r, run) in prog.runs.iter().enumerate() {
                let peers = match &run.peers {
                    Peers::None => "None".to_string(),
                    Peers::Offsets(o) => format!("Offsets {}", list(o, u64::to_string)),
                    Peers::Pairs(ps) => format!("Pairs {}", list(ps, |(a, b)| format!("{a} {b}"))),
                };
                let head = format!(
                    "r{r} {}  params {}  peers {peers}{}",
                    run.name,
                    types(&run.params),
                    shape(m, run.tile)
                );
                self.line(2, &head);
                self.line(3, "seed:");
                self.body(4, &run.seed);
                for (k, (message, body)) in run.on_message.iter().enumerate() {
                    self.line(3, &format!("on{k} m{}:", message.0));
                    self.body(4, body);
                }
                self.effects(3, &run.effects);
            }
            for (k, chain) in prog.chains.iter().enumerate() {
                let on = match chain.on {
                    On::Group => "Group".to_string(),
                    On::Grid(r, c) => format!("Grid {r}x{c}"),
                };
                let head = format!(
                    "k{k} {}  params {}  on {on}",
                    chain.name,
                    types(&chain.params)
                );
                self.line(2, &head);
                let calls = chain
                    .calls
                    .iter()
                    .map(|c| {
                        let args = list(&c.args, |a| match a {
                            CallArg::Param(i) => format!("param {i}"),
                            CallArg::Past(i) => format!("past {i}"),
                            CallArg::Const(x) => format!("const {}", const_text(x)),
                        });
                        match &c.part {
                            None => format!("h{} {args}", c.handler.0),
                            Some(p) => format!(
                                "h{} {args}  part {}..{}",
                                c.handler.0,
                                count(&p.first),
                                count(&p.end)
                            ),
                        }
                    })
                    .collect();
                self.section(3, "calls", calls);
                let edges = chain
                    .deps
                    .edges
                    .iter()
                    .map(|e| {
                        format!(
                            "writer {} reader {} param {} write {} read {} past {}",
                            e.writer,
                            e.reader,
                            e.param,
                            e.write,
                            e.read,
                            yes(e.past)
                        )
                    })
                    .collect();
                self.section(3, "deps", edges);
            }
        }
    }

    fn controls(&mut self, m: &Module) {
        if m.controls.is_empty() {
            self.line(0, "controls: []");
            return;
        }
        self.line(0, "controls:");
        for (c, control) in m.controls.iter().enumerate() {
            let ret = control
                .ret
                .map_or("none".to_string(), |s| s.name().to_string());
            let head = format!(
                "c{c} {}  params {}  ret {ret}",
                control.name,
                types(&control.params)
            );
            self.line(1, &head);
            self.body(2, &control.body);
        }
    }

    fn notes(&mut self, m: &Module) {
        let n = &m.notes;
        self.line(0, "notes:");
        let hints = n
            .hints
            .iter()
            .map(|h| {
                format!(
                    "{}  {} {}",
                    anchor(&h.anchor),
                    h.path.join("."),
                    list(&h.args, hint_arg)
                )
            })
            .collect();
        self.section(1, "hints", hints);
        let spans = n
            .spans
            .iter()
            .map(|(a, s)| format!("{}  {}..{}", anchor(a), s.start, s.end))
            .collect();
        self.section(1, "spans", spans);
        let names = n
            .names
            .iter()
            .map(|x| match x {
                Named::Value(b, v, s) => format!("{} v{} {s:?}", body_ref(b), v.0),
                Named::Local(b, l, s) => format!("{} l{} {s:?}", body_ref(b), l.0),
            })
            .collect();
        self.section(1, "names", names);
        let params = n
            .param_names
            .iter()
            .map(|(a, s)| format!("{}  {s:?}", anchor(a)))
            .collect();
        self.section(1, "param_names", params);
    }
}
