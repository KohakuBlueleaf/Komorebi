//! The writer: a module to its three parts and the file. One walk in the order of the
//! module's lists; every integer fixed width and little-endian, so a module has one
//! byte string. `docs/ir/bytecode.md` gives the layout field by field.

use super::version::{self as v, Table};
use crate::program::body::{Body, Op, Range};
use crate::program::derive::{Cause, ChainDeps, Effects, Requirement};
use crate::program::notes::{Anchor, HintArg, Named, Owner};
use crate::program::{
    BodyRef, CallArg, Control, Func, FuncKind, Handler, Lifetime, Member, MemberKind, Module,
    Program, Run, Scope,
};
use crate::types::{
    Ability, Axis, BinOp, Binding, CmpOp, Combiner, ConstValue, Count, Entry, Failure, Format, On,
    Peers, ReduceOp, ScalarType, SplitBy, Type, UnaryOp,
};

/// A module's three parts: S, the program (everything but the notes); H, the hints;
/// P, the positions and the names of values and parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parts {
    /// S: the program.
    pub program: Vec<u8>,
    /// H: the hints.
    pub hints: Vec<u8>,
    /// P: the positions and the names.
    pub positions: Vec<u8>,
}

/// The byte-code program of a module: the header, then S, H and P.
pub fn encode(module: &Module) -> Vec<u8> {
    let parts = parts(module);
    let mut out = Vec::with_capacity(
        v::HEADER_LEN + parts.program.len() + parts.hints.len() + parts.positions.len(),
    );
    out.extend_from_slice(&v::MAGIC);
    out.extend_from_slice(&v::VERSION.to_le_bytes());
    for part in [&parts.program, &parts.hints, &parts.positions] {
        out.extend_from_slice(&(part.len() as u64).to_le_bytes());
    }
    out.extend_from_slice(&parts.program);
    out.extend_from_slice(&parts.hints);
    out.extend_from_slice(&parts.positions);
    out
}

/// The three parts of a module, each the walk of its fields.
pub fn parts(module: &Module) -> Parts {
    let mut s = Writer::default();
    s.program(module);
    let mut h = Writer::default();
    h.list(&module.notes.hints, |w, hint| {
        w.anchor(&hint.anchor);
        w.list(&hint.path, |w, p| w.str(p));
        w.list(&hint.args, Writer::hint_arg);
    });
    let mut p = Writer::default();
    p.list(&module.notes.spans, |w, (anchor, span)| {
        w.anchor(anchor);
        w.u32(span.start);
        w.u32(span.end);
    });
    p.list(&module.notes.names, |w, named| match named {
        Named::Value(body, value, name) => {
            w.tag(v::NAMED, "Value");
            w.body_ref(body);
            w.u32(value.0);
            w.str(name);
        }
        Named::Local(body, local, name) => {
            w.tag(v::NAMED, "Local");
            w.body_ref(body);
            w.u32(local.0);
            w.str(name);
        }
    });
    p.list(&module.notes.param_names, |w, (anchor, name)| {
        w.anchor(anchor);
        w.str(name);
    });
    Parts {
        program: s.out,
        hints: h.out,
        positions: p.out,
    }
}

/// The bytes written so far.
#[derive(Default)]
pub(super) struct Writer {
    pub(super) out: Vec<u8>,
}

impl Writer {
    fn u8(&mut self, x: u8) {
        self.out.push(x);
    }

    fn u32(&mut self, x: u32) {
        self.out.extend_from_slice(&x.to_le_bytes());
    }

    fn u64(&mut self, x: u64) {
        self.out.extend_from_slice(&x.to_le_bytes());
    }

    fn bool(&mut self, x: bool) {
        self.u8(u8::from(x));
    }

    fn len(&mut self, n: usize) {
        match u32::try_from(n) {
            Ok(n) => self.u32(n),
            Err(_) => panic!("a list or a string of {n} items does not fit a byte-code program"),
        }
    }

    fn str(&mut self, s: &str) {
        self.len(s.len());
        self.out.extend_from_slice(s.as_bytes());
    }

    fn tag(&mut self, table: Table, name: &str) {
        self.u8(v::number(table, name));
    }

    fn list<T>(&mut self, items: &[T], mut each: impl FnMut(&mut Self, &T)) {
        self.len(items.len());
        for item in items {
            each(self, item);
        }
    }

    fn opt<T>(&mut self, x: Option<&T>, each: impl FnOnce(&mut Self, &T)) {
        match x {
            None => self.u8(0),
            Some(x) => {
                self.u8(1);
                each(self, x);
            }
        }
    }

    fn values(&mut self, xs: &[crate::program::body::ValueId]) {
        self.list(xs, |w, x| w.u32(x.0));
    }

    fn opt_value(&mut self, x: Option<crate::program::body::ValueId>) {
        self.opt(x.as_ref(), |w, x| w.u32(x.0));
    }

    // ---- the module ----

    fn program(&mut self, m: &Module) {
        self.str(&m.name);
        self.u64(m.tile.rows);
        self.u64(m.tile.cols);
        self.list(&m.formats, |w, f| w.format(*f));
        self.list(&m.params, |w, p| {
            w.str(&p.name);
            w.scalar(p.ty);
            w.binding(&p.binding);
        });
        self.list(&m.records, |w, r| {
            w.str(&r.name);
            w.opt(r.program.as_ref(), |w, p| w.u32(p.0));
            w.list(&r.fields, |w, (name, ty)| {
                w.str(name);
                w.scalar(*ty);
            });
        });
        self.list(&m.messages, |w, r| {
            w.str(&r.name);
            w.opt(r.program.as_ref(), |w, p| w.u32(p.0));
            w.list(&r.fields, |w, (name, ty)| {
                w.str(name);
                w.scalar(*ty);
            });
        });
        self.list(&m.funcs, Writer::func);
        self.list(&m.programs, Writer::program_decl);
        self.list(&m.controls, Writer::control);
        self.list(&m.requirements, Writer::requirement);
    }

    fn func(&mut self, f: &Func) {
        self.str(&f.name);
        match &f.kind {
            FuncKind::Plain => self.tag(v::FUNC_KIND, "Plain"),
            FuncKind::Combine { commutative } => {
                self.tag(v::FUNC_KIND, "Combine");
                self.bool(*commutative);
            }
            FuncKind::Extern {
                implementation,
                reads,
                writes,
            } => {
                self.tag(v::FUNC_KIND, "Extern");
                self.str(implementation);
                self.list(reads, |w, x| w.u32(*x));
                self.list(writes, |w, x| w.u32(*x));
            }
        }
        self.list(&f.params, Writer::ty);
        self.ty(&f.ret);
        self.opt(f.body.as_ref(), Writer::body);
        self.effects(&f.effects);
    }

    fn program_decl(&mut self, p: &Program) {
        self.str(&p.name);
        self.list(&p.state, Writer::member);
        self.list(&p.handlers, Writer::handler);
        self.list(&p.runs, Writer::run);
        self.list(&p.chains, |w, c| {
            w.str(&c.name);
            w.list(&c.params, Writer::ty);
            w.on(&c.on);
            w.list(&c.calls, |w, call| {
                w.u32(call.handler.0);
                w.list(&call.args, Writer::call_arg);
            });
            w.deps(&c.deps);
        });
    }

    fn member(&mut self, m: &Member) {
        self.str(&m.name);
        match &m.kind {
            MemberKind::Tiles(f) => {
                self.tag(v::MEMBER_KIND, "Tiles");
                self.format(*f);
            }
            MemberKind::Scalars(s) => {
                self.tag(v::MEMBER_KIND, "Scalars");
                self.scalar(*s);
            }
        }
        self.count(&m.count);
        match m.scope {
            Scope::Worker => self.tag(v::SCOPE, "Worker"),
        }
        match m.lifetime {
            Lifetime::Program => self.tag(v::LIFETIME, "Program"),
        }
        self.bool(m.export);
        self.bool(m.by_element);
    }

    fn handler(&mut self, h: &Handler) {
        self.str(&h.name);
        self.list(&h.params, Writer::ty);
        self.opt(h.ret.as_ref(), |w, s| w.scalar(*s));
        self.on(&h.on);
        self.bool(h.exclusive);
        self.body(&h.body);
        self.effects(&h.effects);
    }

    fn run(&mut self, r: &Run) {
        self.str(&r.name);
        self.list(&r.params, Writer::ty);
        self.peers(&r.peers);
        self.body(&r.seed);
        self.list(&r.on_message, |w, (message, body)| {
            w.u32(message.0);
            w.body(body);
        });
        self.effects(&r.effects);
    }

    fn call_arg(&mut self, a: &CallArg) {
        match a {
            CallArg::Param(i) => {
                self.tag(v::CALL_ARG, "Param");
                self.u32(*i);
            }
            CallArg::Past(i) => {
                self.tag(v::CALL_ARG, "Past");
                self.u32(*i);
            }
            CallArg::Const(c) => {
                self.tag(v::CALL_ARG, "Const");
                self.const_value(c);
            }
        }
    }

    fn deps(&mut self, d: &ChainDeps) {
        self.list(&d.edges, |w, e| {
            w.u32(e.writer);
            w.u32(e.reader);
            w.u32(e.param);
            w.u32(e.write);
            w.u32(e.read);
            w.bool(e.past);
        });
    }

    fn control(&mut self, c: &Control) {
        self.str(&c.name);
        self.list(&c.params, Writer::ty);
        self.opt(c.ret.as_ref(), |w, s| w.scalar(*s));
        self.body(&c.body);
    }

    fn requirement(&mut self, r: &Requirement) {
        match &r.ability {
            Ability::LocalState { amount, tiles } => {
                self.tag(v::ABILITY, "LocalState");
                self.u64(*amount);
                self.bool(*tiles);
            }
            Ability::Grid => self.tag(v::ABILITY, "Grid"),
            Ability::Dynamic => self.tag(v::ABILITY, "Dynamic"),
            Ability::Islands(n) => {
                self.tag(v::ABILITY, "Islands");
                self.u64(*n);
            }
            Ability::Impl(name) => {
                self.tag(v::ABILITY, "Impl");
                self.str(name);
            }
        }
        self.list(&r.causes, |w, c| match c {
            Cause::Stated => w.tag(v::CAUSE, "Stated"),
            Cause::GridHandler(p, h) => {
                w.tag(v::CAUSE, "GridHandler");
                w.u32(p.0);
                w.u32(h.0);
            }
            Cause::DynamicRun(e) => {
                w.tag(v::CAUSE, "DynamicRun");
                w.entry(e);
            }
            Cause::ExternWithoutFallback(f) => {
                w.tag(v::CAUSE, "ExternWithoutFallback");
                w.u32(f.0);
            }
        });
    }

    fn effects(&mut self, e: &Effects) {
        self.list(&e.tensors, |w, t| {
            w.u32(t.param);
            w.body_ref(&t.body);
            w.u32(t.view.0);
            w.tag(v::ACCESS, &format!("{:?}", t.access));
            w.u32(t.at.0);
        });
        self.list(&e.state, |w, s| {
            w.u32(s.member.0);
            w.body_ref(&s.body);
            w.tag(v::STATE_USE, &format!("{:?}", s.how));
            w.u32(s.at.0);
        });
        self.list(&e.collectives, |w, (body, inst)| {
            w.body_ref(body);
            w.u32(inst.0);
        });
        self.list(&e.calls, |w, f| w.u32(f.0));
    }

    // ---- references ----

    fn body_ref(&mut self, b: &BodyRef) {
        match b {
            BodyRef::Func(f) => {
                self.tag(v::BODY_REF, "Func");
                self.u32(f.0);
            }
            BodyRef::Handler(p, h) => {
                self.tag(v::BODY_REF, "Handler");
                self.u32(p.0);
                self.u32(h.0);
            }
            BodyRef::Seed(p, r) => {
                self.tag(v::BODY_REF, "Seed");
                self.u32(p.0);
                self.u32(r.0);
            }
            BodyRef::On(p, r, i) => {
                self.tag(v::BODY_REF, "On");
                self.u32(p.0);
                self.u32(r.0);
                self.u32(*i);
            }
            BodyRef::Control(c) => {
                self.tag(v::BODY_REF, "Control");
                self.u32(c.0);
            }
        }
    }

    fn entry(&mut self, e: &Entry) {
        let (name, p, x) = match e {
            Entry::Handler(p, h) => ("Handler", p.0, h.0),
            Entry::Run(p, r) => ("Run", p.0, r.0),
            Entry::Chain(p, c) => ("Chain", p.0, c.0),
        };
        self.tag(v::ENTRY, name);
        self.u32(p);
        self.u32(x);
    }

    fn anchor(&mut self, a: &Anchor) {
        match a {
            Anchor::Module => self.tag(v::ANCHOR, "Module"),
            Anchor::Func(f) => {
                self.tag(v::ANCHOR, "Func");
                self.u32(f.0);
            }
            Anchor::Program(p) => {
                self.tag(v::ANCHOR, "Program");
                self.u32(p.0);
            }
            Anchor::Handler(p, h) => {
                self.tag(v::ANCHOR, "Handler");
                self.u32(p.0);
                self.u32(h.0);
            }
            Anchor::Run(p, r) => {
                self.tag(v::ANCHOR, "Run");
                self.u32(p.0);
                self.u32(r.0);
            }
            Anchor::Member(p, m) => {
                self.tag(v::ANCHOR, "Member");
                self.u32(p.0);
                self.u32(m.0);
            }
            Anchor::Control(c) => {
                self.tag(v::ANCHOR, "Control");
                self.u32(c.0);
            }
            Anchor::Chain(p, c) => {
                self.tag(v::ANCHOR, "Chain");
                self.u32(p.0);
                self.u32(c.0);
            }
            Anchor::Call(p, c, i) => {
                self.tag(v::ANCHOR, "Call");
                self.u32(p.0);
                self.u32(c.0);
                self.u32(*i);
            }
            Anchor::Param(p) => {
                self.tag(v::ANCHOR, "Param");
                self.u32(p.0);
            }
            Anchor::Message(m) => {
                self.tag(v::ANCHOR, "Message");
                self.u32(m.0);
            }
            Anchor::Record(r) => {
                self.tag(v::ANCHOR, "Record");
                self.u32(r.0);
            }
            Anchor::ParamOf(owner, i) => {
                self.tag(v::ANCHOR, "ParamOf");
                match owner {
                    Owner::Func(f) => {
                        self.tag(v::OWNER, "Func");
                        self.u32(f.0);
                    }
                    Owner::Handler(p, h) => {
                        self.tag(v::OWNER, "Handler");
                        self.u32(p.0);
                        self.u32(h.0);
                    }
                    Owner::Run(p, r) => {
                        self.tag(v::OWNER, "Run");
                        self.u32(p.0);
                        self.u32(r.0);
                    }
                    Owner::Chain(p, c) => {
                        self.tag(v::OWNER, "Chain");
                        self.u32(p.0);
                        self.u32(c.0);
                    }
                    Owner::Control(c) => {
                        self.tag(v::OWNER, "Control");
                        self.u32(c.0);
                    }
                }
                self.u32(*i);
            }
            Anchor::Inst(body, inst) => {
                self.tag(v::ANCHOR, "Inst");
                self.body_ref(body);
                self.u32(inst.0);
            }
        }
    }

    pub(super) fn hint_arg(&mut self, a: &HintArg) {
        match a {
            HintArg::Number(s) => {
                self.tag(v::HINT_ARG, "Number");
                self.str(s);
            }
            HintArg::Text(s) => {
                self.tag(v::HINT_ARG, "Text");
                self.str(s);
            }
            HintArg::Name(s) => {
                self.tag(v::HINT_ARG, "Name");
                self.str(s);
            }
            HintArg::Named(s, x) => {
                self.tag(v::HINT_ARG, "Named");
                self.str(s);
                self.hint_arg(x);
            }
            HintArg::List(xs) => {
                self.tag(v::HINT_ARG, "List");
                self.list(xs, Writer::hint_arg);
            }
        }
    }

    // ---- the words of types ----

    fn format(&mut self, f: Format) {
        self.tag(v::FORMAT, &format!("{f:?}"));
    }

    fn scalar(&mut self, s: ScalarType) {
        self.tag(v::SCALAR_TYPE, &format!("{s:?}"));
    }

    fn axis(&mut self, a: Axis) {
        self.tag(v::AXIS, &format!("{a:?}"));
    }

    fn count(&mut self, c: &Count) {
        match c {
            Count::Const(n) => {
                self.tag(v::COUNT, "Const");
                self.u64(*n);
            }
            Count::Param(p) => {
                self.tag(v::COUNT, "Param");
                self.u32(p.0);
            }
            Count::GroupSize => self.tag(v::COUNT, "GroupSize"),
        }
    }

    fn const_value(&mut self, c: &ConstValue) {
        match c {
            ConstValue::Int(s, bits) => {
                self.tag(v::CONST_VALUE, "Int");
                self.scalar(*s);
                self.u64(*bits);
            }
            ConstValue::Float(s, x) => {
                self.tag(v::CONST_VALUE, "Float");
                self.scalar(*s);
                self.u64(x.to_bits());
            }
            ConstValue::Bool(b) => {
                self.tag(v::CONST_VALUE, "Bool");
                self.bool(*b);
            }
        }
    }

    fn binding(&mut self, b: &Binding) {
        match b {
            Binding::Build(c) => {
                self.tag(v::BINDING, "Build");
                self.const_value(c);
            }
            Binding::Load(c) => {
                self.tag(v::BINDING, "Load");
                self.opt(c.as_ref(), Writer::const_value);
            }
            Binding::Invoke(c) => {
                self.tag(v::BINDING, "Invoke");
                self.const_value(c);
            }
        }
    }

    fn on(&mut self, o: &On) {
        match o {
            On::Group => self.tag(v::ON, "Group"),
            On::Grid(r, c) => {
                self.tag(v::ON, "Grid");
                self.u64(*r);
                self.u64(*c);
            }
        }
    }

    fn peers(&mut self, p: &Peers) {
        match p {
            Peers::None => self.tag(v::PEERS, "None"),
            Peers::Offsets(xs) => {
                self.tag(v::PEERS, "Offsets");
                self.list(xs, |w, x| w.u64(*x));
            }
            Peers::Pairs(xs) => {
                self.tag(v::PEERS, "Pairs");
                self.list(xs, |w, (s, r)| {
                    w.u64(*s);
                    w.u64(*r);
                });
            }
        }
    }

    pub(super) fn ty(&mut self, t: &Type) {
        match t {
            Type::Scalar(s) => {
                self.tag(v::TYPE, "Scalar");
                self.scalar(*s);
            }
            Type::Bool => self.tag(v::TYPE, "Bool"),
            Type::Tile(f) => {
                self.tag(v::TYPE, "Tile");
                self.format(*f);
            }
            Type::Reduced(f, a) => {
                self.tag(v::TYPE, "Reduced");
                self.format(*f);
                self.axis(*a);
            }
            Type::Block(f, n) => {
                self.tag(v::TYPE, "Block");
                self.format(*f);
                self.count(n);
            }
            Type::Tensor(f, replicated) => {
                self.tag(v::TYPE, "Tensor");
                self.format(*f);
                self.bool(*replicated);
            }
            Type::View(f) => {
                self.tag(v::TYPE, "View");
                self.opt(f.as_ref(), |w, f| w.format(*f));
            }
            Type::Items(n) => {
                self.tag(v::TYPE, "Items");
                self.count(n);
            }
            Type::Stream(f, n) => {
                self.tag(v::TYPE, "Stream");
                self.format(*f);
                self.count(n);
            }
            Type::Zip(a, b) => {
                self.tag(v::TYPE, "Zip");
                self.ty(a);
                self.ty(b);
            }
            Type::Sink(f, n) => {
                self.tag(v::TYPE, "Sink");
                self.format(*f);
                self.count(n);
            }
            Type::Words(s) => {
                self.tag(v::TYPE, "Words");
                self.scalar(*s);
            }
            Type::Records(r) => {
                self.tag(v::TYPE, "Records");
                self.u32(r.0);
            }
            Type::RecordValue(r) => {
                self.tag(v::TYPE, "RecordValue");
                self.u32(r.0);
            }
            Type::Message(m) => {
                self.tag(v::TYPE, "Message");
                self.u32(m.0);
            }
            Type::SubGroups => self.tag(v::TYPE, "SubGroups"),
            Type::Leaders => self.tag(v::TYPE, "Leaders"),
            Type::Broadcast(f) => {
                self.tag(v::TYPE, "Broadcast");
                self.format(*f);
            }
            Type::Group => self.tag(v::TYPE, "Group"),
            Type::Handle(s) => {
                self.tag(v::TYPE, "Handle");
                self.opt(s.as_ref(), |w, s| w.scalar(*s));
            }
            Type::Outcome(s) => {
                self.tag(v::TYPE, "Outcome");
                self.opt(s.as_ref(), |w, s| w.scalar(*s));
            }
            Type::Reason => self.tag(v::TYPE, "Reason"),
        }
    }

    // ---- bodies ----

    fn body(&mut self, b: &Body) {
        self.list(&b.values, Writer::ty);
        self.list(&b.locals, Writer::ty);
        self.list(&b.insts, |w, inst| {
            w.op(&inst.op);
            w.values(&inst.defs);
        });
    }

    fn range(&mut self, r: &Range) {
        self.u32(r.lo.0);
        self.u32(r.hi.0);
        self.opt_value(r.step);
    }

    fn combiner(&mut self, c: &Combiner) {
        match c {
            Combiner::Func(f) => {
                self.tag(v::COMBINER, "Func");
                self.u32(f.0);
            }
            other => self.tag(v::COMBINER, &format!("{other:?}")),
        }
    }

    fn bin_op(&mut self, o: BinOp) {
        self.tag(v::BIN_OP, &format!("{o:?}"));
    }

    fn cmp_op(&mut self, o: CmpOp) {
        self.tag(v::CMP_OP, &format!("{o:?}"));
    }

    fn unary_op(&mut self, o: UnaryOp) {
        self.tag(v::UNARY_OP, &format!("{o:?}"));
    }

    fn reduce_op(&mut self, o: ReduceOp) {
        self.tag(v::REDUCE_OP, &format!("{o:?}"));
    }

    fn split_by(&mut self, s: SplitBy) {
        self.tag(v::SPLIT_BY, &format!("{s:?}"));
    }

    fn failure(&mut self, f: Failure) {
        self.tag(v::FAILURE, &format!("{f:?}"));
    }

    /// An operation: its number, then its operands in the order the variant declares
    /// them.
    pub(super) fn op(&mut self, op: &Op) {
        let t = |w: &mut Self, name: &str| w.tag(v::OP, name);
        match op {
            Op::Const(c) => {
                t(self, "Const");
                self.const_value(c);
            }
            Op::Arg(i) => {
                t(self, "Arg");
                self.u32(*i);
            }
            Op::Param(p) => {
                t(self, "Param");
                self.u32(p.0);
            }
            Op::Get(l) => {
                t(self, "Get");
                self.u32(l.0);
            }
            Op::Set(l, x) => {
                t(self, "Set");
                self.u32(l.0);
                self.u32(x.0);
            }
            Op::Binary(o, a, b) => {
                t(self, "Binary");
                self.bin_op(*o);
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::Compare(o, a, b) => {
                t(self, "Compare");
                self.cmp_op(*o);
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::LogicAnd(a, b) => {
                t(self, "LogicAnd");
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::LogicOr(a, b) => {
                t(self, "LogicOr");
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::If { cond } => {
                t(self, "If");
                self.u32(cond.0);
            }
            Op::Else => t(self, "Else"),
            Op::ForRange { n } => {
                t(self, "ForRange");
                self.u32(n.0);
            }
            Op::ForStream { stream } => {
                t(self, "ForStream");
                self.u32(stream.0);
            }
            Op::ForZip { a, b } => {
                t(self, "ForZip");
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::ForArrive { broadcast } => {
                t(self, "ForArrive");
                self.u32(broadcast.0);
            }
            Op::While => t(self, "While"),
            Op::WhileTest { cond } => {
                t(self, "WhileTest");
                self.u32(cond.0);
            }
            Op::End => t(self, "End"),
            Op::Return(x) => {
                t(self, "Return");
                self.opt_value(*x);
            }
            Op::Convert(s, x) => {
                t(self, "Convert");
                self.scalar(*s);
                self.u32(x.0);
            }
            Op::Unary(o, x) => {
                t(self, "Unary");
                self.unary_op(*o);
                self.u32(x.0);
            }
            Op::Select { cmp, a, b, x, y } => {
                t(self, "Select");
                self.cmp_op(*cmp);
                for v in [a, b, x, y] {
                    self.u32(v.0);
                }
            }
            Op::Reduce(o, x, a) => {
                t(self, "Reduce");
                self.reduce_op(*o);
                self.u32(x.0);
                self.axis(*a);
            }
            Op::SpreadCol(x) => {
                t(self, "SpreadCol");
                self.u32(x.0);
            }
            Op::SpreadRow(x) => {
                t(self, "SpreadRow");
                self.u32(x.0);
            }
            Op::Matmul(a, b) => {
                t(self, "Matmul");
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::Transpose(x) => {
                t(self, "Transpose");
                self.u32(x.0);
            }
            Op::Zeros => t(self, "Zeros"),
            Op::CallFunc(f, args) => {
                t(self, "CallFunc");
                self.u32(f.0);
                self.values(args);
            }
            Op::Tiles { tensor, first, n } => {
                t(self, "Tiles");
                self.u32(tensor.0);
                self.u32(first.0);
                self.u32(n.0);
            }
            Op::Rect { tensor, rows, cols } => {
                t(self, "Rect");
                self.u32(tensor.0);
                self.range(rows);
                self.range(cols);
            }
            Op::Turn(x) => {
                t(self, "Turn");
                self.u32(x.0);
            }
            Op::Items {
                tensor,
                range,
                size,
            } => {
                t(self, "Items");
                self.u32(tensor.0);
                self.opt(range.as_ref(), |w, (a, b)| {
                    w.u32(a.0);
                    w.u32(b.0);
                });
                self.u32(size.0);
            }
            Op::ReadTile(a, b) => {
                t(self, "ReadTile");
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::Words(x) => {
                t(self, "Words");
                self.u32(x.0);
            }
            Op::Word(a, b) => {
                t(self, "Word");
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::Records(a, r, b) => {
                t(self, "Records");
                self.u32(a.0);
                self.u32(r.0);
                self.u32(b.0);
            }
            Op::RecordAt(a, b) => {
                t(self, "RecordAt");
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::RecordField(x, i) => {
                t(self, "RecordField");
                self.u32(x.0);
                self.u32(*i);
            }
            Op::Put { items, src, at } => {
                t(self, "Put");
                self.u32(items.0);
                self.u32(src.0);
                self.u32(at.0);
            }
            Op::Partition(a, b, c) => {
                t(self, "Partition");
                self.u32(a.0);
                self.u32(b.0);
                self.u32(c.0);
            }
            Op::Stream(a, b) => {
                t(self, "Stream");
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::Sink(a, b) => {
                t(self, "Sink");
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::Push(a, b) => {
                t(self, "Push");
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::Store(a, b) => {
                t(self, "Store");
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::BlockAt(a, b) => {
                t(self, "BlockAt");
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::Fill(m, x) => {
                t(self, "Fill");
                self.u32(m.0);
                self.u32(x.0);
            }
            Op::ZeroState(m) => {
                t(self, "ZeroState");
                self.u32(m.0);
            }
            Op::StateSet(m, a, b) => {
                t(self, "StateSet");
                self.u32(m.0);
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::StateGet(m, x) => {
                t(self, "StateGet");
                self.u32(m.0);
                self.u32(x.0);
            }
            Op::Bump(m, x) => {
                t(self, "Bump");
                self.u32(m.0);
                self.u32(x.0);
            }
            Op::Rank => t(self, "Rank"),
            Op::IsLeader => t(self, "IsLeader"),
            Op::WorkerId => t(self, "WorkerId"),
            Op::Row => t(self, "Row"),
            Op::Col => t(self, "Col"),
            Op::GroupSize => t(self, "GroupSize"),
            Op::GroupId(x) => {
                t(self, "GroupId");
                self.u32(x.0);
            }
            Op::Split(s, x) => {
                t(self, "Split");
                self.split_by(*s);
                self.u32(x.0);
            }
            Op::Leaders(s, x) => {
                t(self, "Leaders");
                self.split_by(*s);
                self.u32(x.0);
            }
            Op::ReduceGroup {
                x,
                with,
                within,
                among,
                publish,
            } => {
                t(self, "ReduceGroup");
                self.u32(x.0);
                self.combiner(with);
                self.opt_value(*within);
                self.opt_value(*among);
                self.values(publish);
            }
            Op::AllReduce { x, with, publish } => {
                t(self, "AllReduce");
                self.u32(x.0);
                self.combiner(with);
                self.values(publish);
            }
            Op::Gather(x) => {
                t(self, "Gather");
                self.u32(x.0);
            }
            Op::Broadcast(a, x) => {
                t(self, "Broadcast");
                self.axis(*a);
                self.u32(x.0);
            }
            Op::IsSender(x) => {
                t(self, "IsSender");
                self.u32(x.0);
            }
            Op::Cast(a, b, c) => {
                t(self, "Cast");
                self.u32(a.0);
                self.u32(b.0);
                self.opt_value(*c);
            }
            Op::Sync(xs) => {
                t(self, "Sync");
                self.values(xs);
            }
            Op::Arrived => t(self, "Arrived"),
            Op::MakeMessage(m, xs) => {
                t(self, "MakeMessage");
                self.u32(m.0);
                self.values(xs);
            }
            Op::Send {
                message,
                to,
                publish,
                used,
            } => {
                t(self, "Send");
                self.u32(message.0);
                self.u32(to.0);
                self.values(publish);
                self.bool(*used);
            }
            Op::Forward {
                message,
                with,
                to,
                publish,
            } => {
                t(self, "Forward");
                self.u32(message.0);
                self.list(with, |w, (i, x)| {
                    w.u32(*i);
                    w.u32(x.0);
                });
                self.u32(to.0);
                self.values(publish);
            }
            Op::Consume(x) => {
                t(self, "Consume");
                self.u32(x.0);
            }
            Op::Offer(a, b) => {
                t(self, "Offer");
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::Field(x, i) => {
                t(self, "Field");
                self.u32(x.0);
                self.u32(*i);
            }
            Op::MakeGroup(a, b) => {
                t(self, "MakeGroup");
                self.opt_value(*a);
                self.u32(b.0);
            }
            Op::Invoke {
                entry,
                args,
                on,
                after,
                after_end,
            } => {
                t(self, "Invoke");
                self.entry(entry);
                self.values(args);
                self.u32(on.0);
                self.values(after);
                self.values(after_end);
            }
            Op::Await(x) => {
                t(self, "Await");
                self.u32(x.0);
            }
            Op::Outcome(x) => {
                t(self, "Outcome");
                self.u32(x.0);
            }
            Op::Failed(x) => {
                t(self, "Failed");
                self.u32(x.0);
            }
            Op::Reason(x) => {
                t(self, "Reason");
                self.u32(x.0);
            }
            Op::OutcomeValue(x) => {
                t(self, "OutcomeValue");
                self.u32(x.0);
            }
            Op::FailureName(f) => {
                t(self, "FailureName");
                self.failure(*f);
            }
            Op::IslandCount => t(self, "IslandCount"),
            Op::IslandOf(x) => {
                t(self, "IslandOf");
                self.u32(x.0);
            }
            Op::IslandCrosses(x) => {
                t(self, "IslandCrosses");
                self.u32(x.0);
            }
            Op::IslandHops(x) => {
                t(self, "IslandHops");
                self.u32(x.0);
            }
            Op::ReduceTo(a, b, c) => {
                t(self, "ReduceTo");
                self.u32(a.0);
                self.u32(b.0);
                self.combiner(c);
            }
            Op::BroadcastFrom(a, b) => {
                t(self, "BroadcastFrom");
                self.u32(a.0);
                self.u32(b.0);
            }
        }
    }
}
