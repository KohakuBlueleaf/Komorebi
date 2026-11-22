//! The reader: a byte-code program back to the module, trusting nothing. It refuses a
//! file that is not one, a version it does not read, and any byte the writer would not
//! have written; then `program::validate` runs, and a module that does not validate is
//! refused. No input makes it panic: a length is checked against the bytes left before
//! anything is made, and nesting is bounded.

use super::version::{self as v, Table};
use super::Refusal;
use crate::program::body::{Body, Inst, LocalId, Op, Range, ScalarsFrom, ValueId};
use crate::program::derive::{
    Access, Cause, ChainDeps, ChainEdge, Effects, Requirement, StateAccess, StateUse, TensorAccess,
};
use crate::program::notes::{Anchor, Hint, HintArg, Named, Notes, Origin, Owner, SourceSpan};
use crate::program::{
    validate, BodyRef, Call, CallArg, Chain, Control, Func, FuncKind, Handler, Lifetime, Member,
    MemberKind, Message, Module, Param, Part, Program, Record, Run, Scope,
};
use crate::types::{
    Ability, Axis, BinOp, Binding, ChainId, CmpOp, Combiner, ConstValue, ControlId, Count, Entry,
    Failure, Format, FuncId, HandlerId, MemberId, MessageId, On, ParamId, Peers, ProgramId,
    RecordId, ReduceOp, RunId, ScalarType, SplitBy, TileShape, Type, UnaryOp,
};

/// How deep a type or a hint's argument may nest: the figure of the text's own limit.
const MAX_DEPTH: u32 = 256;

type Read<T> = Result<T, Refusal>;

/// The module a byte-code program holds, validated; or why the bytes are refused.
pub fn decode(bytes: &[u8]) -> Result<Module, Refusal> {
    let module = read(bytes)?;
    let violations = validate(&module);
    if violations.is_empty() {
        return Ok(module);
    }
    let what: Vec<String> = violations
        .iter()
        .map(|x| match (x.body, x.inst) {
            (Some(body), Some(inst)) => format!("{body:?}, instruction {}: {}", inst.0, x.what),
            (Some(body), None) => format!("{body:?}: {}", x.what),
            _ => x.what.clone(),
        })
        .collect();
    Err(Refusal {
        code: v::INVALID,
        what: format!("the module does not validate: {}", what.join("; ")),
        at: None,
    })
}

/// The three parts of a byte-code program, S, H and P, after its header is checked;
/// nothing inside them is read.
pub fn split(bytes: &[u8]) -> Result<[&[u8]; 3], Refusal> {
    if bytes.len() < 8 || bytes[..4] != v::MAGIC {
        return Err(Refusal {
            code: v::NOT_BYTE_CODE,
            what: "not a byte-code program: it does not open with `KOMC` and a version".into(),
            at: Some(0),
        });
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version > v::VERSION {
        return Err(Refusal {
            code: v::NEWER,
            what: format!(
                "a byte-code program of version {version}; this build reads version {}: \
                 compile it again from its .kom with this komo, or use a newer one",
                v::VERSION
            ),
            at: Some(4),
        });
    }
    if version < v::VERSION {
        return Err(Refusal {
            code: v::OLDER,
            what: format!(
                "a byte-code program of version {version}; this build reads version {} \
                 only: compile it again from its .kom",
                v::VERSION
            ),
            at: Some(4),
        });
    }
    if bytes.len() < v::HEADER_LEN {
        return Err(malformed(
            bytes.len(),
            format!(
                "the header is cut short: {} bytes of {}",
                bytes.len(),
                v::HEADER_LEN
            ),
        ));
    }
    let mut lens = [0u64; 3];
    for (i, len) in lens.iter_mut().enumerate() {
        let at = 8 + 8 * i;
        let mut b = [0u8; 8];
        b.copy_from_slice(&bytes[at..at + 8]);
        *len = u64::from_le_bytes(b);
    }
    let body = (bytes.len() - v::HEADER_LEN) as u64;
    let sum = lens[0]
        .checked_add(lens[1])
        .and_then(|x| x.checked_add(lens[2]));
    if sum != Some(body) {
        return Err(malformed(
            8,
            format!(
                "the lengths of the parts ({}, {}, {}) do not sum to the {body} bytes after \
                 the header",
                lens[0], lens[1], lens[2]
            ),
        ));
    }
    // Each length is now at most `body`, which fits `usize`.
    let s_end = v::HEADER_LEN + lens[0] as usize;
    let h_end = s_end + lens[1] as usize;
    Ok([
        &bytes[v::HEADER_LEN..s_end],
        &bytes[s_end..h_end],
        &bytes[h_end..],
    ])
}

/// The module the bytes hold, read strictly, not validated.
pub(super) fn read(bytes: &[u8]) -> Result<Module, Refusal> {
    let [s, h, p] = split(bytes)?;
    let s_at = v::HEADER_LEN;
    let h_at = s_at + s.len();
    let p_at = h_at + h.len();

    let mut r = Reader::new(s, s_at, "the program (S)");
    let mut module = r.program()?;
    r.finish()?;

    let mut r = Reader::new(h, h_at, "the hints (H)");
    let hints = r.list(|r| {
        Ok(Hint {
            anchor: r.anchor()?,
            path: r.list(Reader::string)?,
            args: r.list(Reader::hint_arg)?,
        })
    })?;
    r.finish()?;

    let mut r = Reader::new(p, p_at, "the positions (P)");
    let spans = r.list(|r| {
        let anchor = r.anchor()?;
        let start = r.u32()?;
        let end = r.u32()?;
        Ok((anchor, SourceSpan { start, end }))
    })?;
    let names = r.list(|r| {
        let at = r.at();
        match r.tag(v::NAMED)? {
            "Value" => Ok(Named::Value(r.body_ref()?, ValueId(r.u32()?), r.string()?)),
            "Local" => Ok(Named::Local(r.body_ref()?, LocalId(r.u32()?), r.string()?)),
            other => Err(r.unread(at, other)),
        }
    })?;
    let param_names = r.list(|r| Ok((r.anchor()?, r.string()?)))?;
    let origin = r.opt(|r| {
        Ok(Origin {
            files: r.list(Reader::string)?,
            lines: r.list(|r| Ok((r.u32()?, r.u32()?, r.string()?)))?,
            places: r.list(|r| Ok((r.anchor()?, r.u32()?, r.u32()?, r.u32()?)))?,
        })
    })?;
    r.finish()?;

    module.notes = Notes {
        hints,
        spans,
        names,
        param_names,
        origin,
    };
    Ok(module)
}

fn malformed(at: usize, what: String) -> Refusal {
    Refusal {
        code: v::MALFORMED,
        what,
        at: Some(at as u64),
    }
}

/// A cursor in one part.
pub(super) struct Reader<'b> {
    bytes: &'b [u8],
    pos: usize,
    /// The file offset of the part's first byte.
    base: usize,
    /// The part, for refusals.
    part: &'static str,
    depth: u32,
}

impl<'b> Reader<'b> {
    pub(super) fn new(bytes: &'b [u8], base: usize, part: &'static str) -> Self {
        Reader {
            bytes,
            pos: 0,
            base,
            part,
            depth: 0,
        }
    }

    fn at(&self) -> usize {
        self.base + self.pos
    }

    fn left(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn bad(&self, at: usize, what: String) -> Refusal {
        malformed(at, format!("{}: {what}", self.part))
    }

    /// The refusal for a name of a table that the reader has no arm for: a fault of
    /// this crate, refused rather than trusted.
    fn unread(&self, at: usize, name: &str) -> Refusal {
        self.bad(at, format!("the variant {name} is not read by this build"))
    }

    /// The part read whole: its walk ends exactly at its length.
    pub(super) fn finish(&self) -> Read<()> {
        if self.left() == 0 {
            Ok(())
        } else {
            Err(self.bad(
                self.at(),
                format!(
                    "its walk ends {} bytes before the part's length says",
                    self.left()
                ),
            ))
        }
    }

    fn take(&mut self, n: usize) -> Read<&'b [u8]> {
        if n > self.left() {
            return Err(self.bad(
                self.at(),
                format!(
                    "cut short: {n} bytes wanted, {} left in the part",
                    self.left()
                ),
            ));
        }
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn u8(&mut self) -> Read<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Read<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Read<u64> {
        let b = self.take(8)?;
        let mut x = [0u8; 8];
        x.copy_from_slice(b);
        Ok(u64::from_le_bytes(x))
    }

    fn bool(&mut self) -> Read<bool> {
        let at = self.at();
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            x => Err(self.bad(at, format!("a bool of {x}: only 0 and 1 are written"))),
        }
    }

    /// A count of items: each item takes at least one byte, so a count above the bytes
    /// left is refused before anything is made.
    fn len(&mut self) -> Read<usize> {
        let at = self.at();
        let n = self.u32()? as usize;
        if n > self.left() {
            return Err(self.bad(
                at,
                format!(
                    "a count of {n}, with {} bytes left in the part",
                    self.left()
                ),
            ));
        }
        Ok(n)
    }

    pub(super) fn string(&mut self) -> Read<String> {
        let n = self.len()?;
        let at = self.at();
        let b = self.take(n)?;
        match std::str::from_utf8(b) {
            Ok(s) => Ok(s.to_string()),
            Err(_) => Err(self.bad(at, "a string that is not UTF-8".into())),
        }
    }

    fn tag(&mut self, table: Table) -> Read<&'static str> {
        let at = self.at();
        let n = self.u8()?;
        match v::name(table, n) {
            Some(name) => Ok(name),
            None => {
                let which = v::TABLES
                    .iter()
                    .find(|(_, t)| std::ptr::eq(*t, table))
                    .map_or("an enum", |(name, _)| name);
                Err(self.bad(at, format!("{n} is no number of {which}")))
            }
        }
    }

    pub(super) fn list<T>(&mut self, mut each: impl FnMut(&mut Self) -> Read<T>) -> Read<Vec<T>> {
        let n = self.len()?;
        let mut out = Vec::new();
        for _ in 0..n {
            out.push(each(self)?);
        }
        Ok(out)
    }

    fn opt<T>(&mut self, each: impl FnOnce(&mut Self) -> Read<T>) -> Read<Option<T>> {
        let at = self.at();
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(each(self)?)),
            x => Err(self.bad(at, format!("an option of {x}: only 0 and 1 are written"))),
        }
    }

    fn value(&mut self) -> Read<ValueId> {
        Ok(ValueId(self.u32()?))
    }

    fn values(&mut self) -> Read<Vec<ValueId>> {
        self.list(Reader::value)
    }

    fn opt_value(&mut self) -> Read<Option<ValueId>> {
        self.opt(Reader::value)
    }

    fn deeper(&mut self) -> Read<()> {
        if self.depth >= MAX_DEPTH {
            return Err(self.bad(self.at(), format!("nested deeper than {MAX_DEPTH}")));
        }
        self.depth += 1;
        Ok(())
    }

    // ---- the module ----

    fn program(&mut self) -> Read<Module> {
        let name = self.string()?;
        let tiles = self.list(|r| {
            Ok(TileShape {
                rows: r.u64()?,
                cols: r.u64()?,
            })
        })?;
        let formats = self.list(Reader::format)?;
        let params = self.list(|r| {
            Ok(Param {
                name: r.string()?,
                ty: r.scalar()?,
                binding: r.binding()?,
            })
        })?;
        let records = self.list(|r| {
            Ok(Record {
                name: r.string()?,
                program: r.opt(|r| Ok(ProgramId(r.u32()?)))?,
                fields: r.list(|r| Ok((r.string()?, r.scalar()?)))?,
            })
        })?;
        let messages = self.list(|r| {
            Ok(Message {
                name: r.string()?,
                program: r.opt(|r| Ok(ProgramId(r.u32()?)))?,
                fields: r.list(|r| Ok((r.string()?, r.scalar()?)))?,
                body: r.u32()?,
            })
        })?;
        let funcs = self.list(Reader::func)?;
        let programs = self.list(Reader::program_decl)?;
        let controls = self.list(|r| {
            Ok(Control {
                name: r.string()?,
                params: r.list(Reader::ty)?,
                ret: r.opt(Reader::scalar)?,
                body: r.body()?,
            })
        })?;
        let requirements = self.list(Reader::requirement)?;
        Ok(Module {
            name,
            tiles,
            formats,
            params,
            records,
            messages,
            funcs,
            programs,
            controls,
            requirements,
            notes: Notes::default(),
        })
    }

    fn func(&mut self) -> Read<Func> {
        let name = self.string()?;
        let at = self.at();
        let kind = match self.tag(v::FUNC_KIND)? {
            "Plain" => FuncKind::Plain,
            "Combine" => FuncKind::Combine {
                commutative: self.bool()?,
            },
            "Extern" => FuncKind::Extern {
                implementation: self.string()?,
                reads: self.list(Reader::u32)?,
                writes: self.list(Reader::u32)?,
            },
            other => return Err(self.unread(at, other)),
        };
        Ok(Func {
            name,
            kind,
            params: self.list(Reader::ty)?,
            ret: self.ty()?,
            body: self.opt(Reader::body)?,
            effects: self.effects()?,
            tile: self.u32()?,
        })
    }

    fn program_decl(&mut self) -> Read<Program> {
        Ok(Program {
            name: self.string()?,
            state: self.list(Reader::member)?,
            handlers: self.list(Reader::handler)?,
            runs: self.list(Reader::run)?,
            chains: self.list(|r| {
                Ok(Chain {
                    name: r.string()?,
                    params: r.list(Reader::ty)?,
                    on: r.on()?,
                    calls: r.list(|r| {
                        Ok(Call {
                            handler: HandlerId(r.u32()?),
                            args: r.list(Reader::call_arg)?,
                            part: r.part()?,
                        })
                    })?,
                    deps: r.deps()?,
                })
            })?,
        })
    }

    fn member(&mut self) -> Read<Member> {
        let name = self.string()?;
        let at = self.at();
        let kind = match self.tag(v::MEMBER_KIND)? {
            "Tiles" => MemberKind::Tiles(self.format()?),
            "Scalars" => MemberKind::Scalars(self.scalar()?),
            other => return Err(self.unread(at, other)),
        };
        let count = self.count()?;
        let at = self.at();
        let scope = match self.tag(v::SCOPE)? {
            "Worker" => Scope::Worker,
            other => return Err(self.unread(at, other)),
        };
        let at = self.at();
        let lifetime = match self.tag(v::LIFETIME)? {
            "Program" => Lifetime::Program,
            other => return Err(self.unread(at, other)),
        };
        Ok(Member {
            name,
            kind,
            count,
            scope,
            lifetime,
            export: self.bool()?,
            by_element: self.bool()?,
            tile: self.u32()?,
        })
    }

    fn handler(&mut self) -> Read<Handler> {
        Ok(Handler {
            name: self.string()?,
            params: self.list(Reader::ty)?,
            ret: self.opt(Reader::scalar)?,
            on: self.on()?,
            exclusive: self.bool()?,
            body: self.body()?,
            effects: self.effects()?,
            tile: self.u32()?,
        })
    }

    fn run(&mut self) -> Read<Run> {
        Ok(Run {
            name: self.string()?,
            params: self.list(Reader::ty)?,
            peers: self.peers()?,
            seed: self.body()?,
            on_message: self.list(|r| Ok((MessageId(r.u32()?), r.body()?)))?,
            effects: self.effects()?,
            tile: self.u32()?,
        })
    }

    fn call_arg(&mut self) -> Read<CallArg> {
        let at = self.at();
        match self.tag(v::CALL_ARG)? {
            "Param" => Ok(CallArg::Param(self.u32()?)),
            "Past" => Ok(CallArg::Past(self.u32()?)),
            "Const" => Ok(CallArg::Const(self.const_value()?)),
            other => Err(self.unread(at, other)),
        }
    }

    fn deps(&mut self) -> Read<ChainDeps> {
        Ok(ChainDeps {
            edges: self.list(|r| {
                Ok(ChainEdge {
                    writer: r.u32()?,
                    reader: r.u32()?,
                    param: r.u32()?,
                    write: r.u32()?,
                    read: r.u32()?,
                    past: r.bool()?,
                })
            })?,
        })
    }

    fn requirement(&mut self) -> Read<Requirement> {
        let at = self.at();
        let ability = match self.tag(v::ABILITY)? {
            "LocalState" => Ability::LocalState {
                amount: self.u64()?,
                tiles: self.bool()?,
            },
            "Grid" => Ability::Grid,
            "Dynamic" => Ability::Dynamic,
            "Islands" => Ability::Islands(self.u64()?),
            "Impl" => Ability::Impl(self.string()?),
            other => return Err(self.unread(at, other)),
        };
        let causes = self.list(|r| {
            let at = r.at();
            match r.tag(v::CAUSE)? {
                "Stated" => Ok(Cause::Stated),
                "GridHandler" => Ok(Cause::GridHandler(ProgramId(r.u32()?), HandlerId(r.u32()?))),
                "DynamicRun" => Ok(Cause::DynamicRun(r.entry()?)),
                "ExternWithoutFallback" => Ok(Cause::ExternWithoutFallback(FuncId(r.u32()?))),
                other => Err(r.unread(at, other)),
            }
        })?;
        Ok(Requirement { ability, causes })
    }

    fn effects(&mut self) -> Read<Effects> {
        let tensors = self.list(|r| {
            let param = r.u32()?;
            let body = r.body_ref()?;
            let view = crate::program::body::InstId(r.u32()?);
            let at = r.at();
            let access = match r.tag(v::ACCESS)? {
                "Stream" => Access::Stream,
                "Read" => Access::Read,
                "Sink" => Access::Sink,
                "Store" => Access::Store,
                "Put" => Access::Put,
                "ExternRead" => Access::ExternRead,
                "ExternWrite" => Access::ExternWrite,
                other => return Err(r.unread(at, other)),
            };
            Ok(TensorAccess {
                param,
                body,
                view,
                access,
                at: crate::program::body::InstId(r.u32()?),
            })
        })?;
        let state = self.list(|r| {
            let member = MemberId(r.u32()?);
            let body = r.body_ref()?;
            let at = r.at();
            let how = match r.tag(v::STATE_USE)? {
                "Read" => StateUse::Read,
                "WriteElement" => StateUse::WriteElement,
                "FillWhole" => StateUse::FillWhole,
                other => return Err(r.unread(at, other)),
            };
            Ok(StateAccess {
                member,
                body,
                how,
                at: crate::program::body::InstId(r.u32()?),
            })
        })?;
        let collectives =
            self.list(|r| Ok((r.body_ref()?, crate::program::body::InstId(r.u32()?))))?;
        let calls = self.list(|r| Ok(FuncId(r.u32()?)))?;
        Ok(Effects {
            tensors,
            state,
            collectives,
            calls,
        })
    }

    // ---- references ----

    fn body_ref(&mut self) -> Read<BodyRef> {
        let at = self.at();
        match self.tag(v::BODY_REF)? {
            "Func" => Ok(BodyRef::Func(FuncId(self.u32()?))),
            "Handler" => Ok(BodyRef::Handler(
                ProgramId(self.u32()?),
                HandlerId(self.u32()?),
            )),
            "Seed" => Ok(BodyRef::Seed(ProgramId(self.u32()?), RunId(self.u32()?))),
            "On" => Ok(BodyRef::On(
                ProgramId(self.u32()?),
                RunId(self.u32()?),
                self.u32()?,
            )),
            "Control" => Ok(BodyRef::Control(ControlId(self.u32()?))),
            other => Err(self.unread(at, other)),
        }
    }

    fn entry(&mut self) -> Read<Entry> {
        let at = self.at();
        match self.tag(v::ENTRY)? {
            "Handler" => Ok(Entry::Handler(
                ProgramId(self.u32()?),
                HandlerId(self.u32()?),
            )),
            "Run" => Ok(Entry::Run(ProgramId(self.u32()?), RunId(self.u32()?))),
            "Chain" => Ok(Entry::Chain(ProgramId(self.u32()?), ChainId(self.u32()?))),
            other => Err(self.unread(at, other)),
        }
    }

    fn anchor(&mut self) -> Read<Anchor> {
        let at = self.at();
        match self.tag(v::ANCHOR)? {
            "Module" => Ok(Anchor::Module),
            "Func" => Ok(Anchor::Func(FuncId(self.u32()?))),
            "Program" => Ok(Anchor::Program(ProgramId(self.u32()?))),
            "Handler" => Ok(Anchor::Handler(
                ProgramId(self.u32()?),
                HandlerId(self.u32()?),
            )),
            "Run" => Ok(Anchor::Run(ProgramId(self.u32()?), RunId(self.u32()?))),
            "Member" => Ok(Anchor::Member(
                ProgramId(self.u32()?),
                MemberId(self.u32()?),
            )),
            "Control" => Ok(Anchor::Control(ControlId(self.u32()?))),
            "Chain" => Ok(Anchor::Chain(ProgramId(self.u32()?), ChainId(self.u32()?))),
            "Call" => Ok(Anchor::Call(
                ProgramId(self.u32()?),
                ChainId(self.u32()?),
                self.u32()?,
            )),
            "Param" => Ok(Anchor::Param(ParamId(self.u32()?))),
            "Message" => Ok(Anchor::Message(MessageId(self.u32()?))),
            "Record" => Ok(Anchor::Record(RecordId(self.u32()?))),
            "ParamOf" => {
                let at = self.at();
                let owner = match self.tag(v::OWNER)? {
                    "Func" => Owner::Func(FuncId(self.u32()?)),
                    "Handler" => Owner::Handler(ProgramId(self.u32()?), HandlerId(self.u32()?)),
                    "Run" => Owner::Run(ProgramId(self.u32()?), RunId(self.u32()?)),
                    "Chain" => Owner::Chain(ProgramId(self.u32()?), ChainId(self.u32()?)),
                    "Control" => Owner::Control(ControlId(self.u32()?)),
                    other => return Err(self.unread(at, other)),
                };
                Ok(Anchor::ParamOf(owner, self.u32()?))
            }
            "Inst" => Ok(Anchor::Inst(
                self.body_ref()?,
                crate::program::body::InstId(self.u32()?),
            )),
            other => Err(self.unread(at, other)),
        }
    }

    pub(super) fn hint_arg(&mut self) -> Read<HintArg> {
        self.deeper()?;
        let at = self.at();
        let arg = match self.tag(v::HINT_ARG)? {
            "Number" => HintArg::Number(self.string()?),
            "Text" => HintArg::Text(self.string()?),
            "Name" => HintArg::Name(self.string()?),
            "Named" => HintArg::Named(self.string()?, Box::new(self.hint_arg()?)),
            "List" => HintArg::List(self.list(Reader::hint_arg)?),
            other => return Err(self.unread(at, other)),
        };
        self.depth -= 1;
        Ok(arg)
    }

    // ---- the words of types ----

    /// A variant without fields, found in `all` by its table name.
    fn plain<T: Copy + std::fmt::Debug>(&mut self, table: Table, all: &[T]) -> Read<T> {
        let at = self.at();
        let name = self.tag(table)?;
        match all.iter().find(|x| format!("{x:?}") == name) {
            Some(x) => Ok(*x),
            None => Err(self.unread(at, name)),
        }
    }

    fn format(&mut self) -> Read<Format> {
        self.plain(v::FORMAT, &Format::ALL)
    }

    fn scalar(&mut self) -> Read<ScalarType> {
        self.plain(v::SCALAR_TYPE, &ScalarType::ALL)
    }

    fn axis(&mut self) -> Read<Axis> {
        self.plain(v::AXIS, &[Axis::Row, Axis::Col])
    }

    fn bin_op(&mut self) -> Read<BinOp> {
        use BinOp::*;
        self.plain(
            v::BIN_OP,
            &[Add, Sub, Mul, Div, Rem, And, Or, Xor, Shl, Shr, Min, Max],
        )
    }

    fn cmp_op(&mut self) -> Read<CmpOp> {
        use CmpOp::*;
        self.plain(v::CMP_OP, &[Gt, Lt, Ge, Le, Eq, Ne])
    }

    fn unary_op(&mut self) -> Read<UnaryOp> {
        use UnaryOp::*;
        self.plain(
            v::UNARY_OP,
            &[
                Neg, Abs, Relu, Square, Floor, Exp, Exp2, Tanh, Erf, Log, Sigmoid, Silu, Sqrt,
                Rsqrt, Reciprocal,
            ],
        )
    }

    fn reduce_op(&mut self) -> Read<ReduceOp> {
        self.plain(v::REDUCE_OP, &[ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min])
    }

    fn split_by(&mut self) -> Read<SplitBy> {
        self.plain(v::SPLIT_BY, &[SplitBy::Parts, SplitBy::Size])
    }

    fn failure(&mut self) -> Read<Failure> {
        self.plain(v::FAILURE, &Failure::ALL)
    }

    fn combiner(&mut self) -> Read<Combiner> {
        let at = self.at();
        match self.tag(v::COMBINER)? {
            "Func" => Ok(Combiner::Func(FuncId(self.u32()?))),
            "Sum" => Ok(Combiner::Sum),
            "Max" => Ok(Combiner::Max),
            "Min" => Ok(Combiner::Min),
            "And" => Ok(Combiner::And),
            "Or" => Ok(Combiner::Or),
            other => Err(self.unread(at, other)),
        }
    }

    fn count(&mut self) -> Read<Count> {
        let at = self.at();
        match self.tag(v::COUNT)? {
            "Const" => Ok(Count::Const(self.u64()?)),
            "Param" => Ok(Count::Param(ParamId(self.u32()?))),
            "GroupSize" => Ok(Count::GroupSize),
            other => Err(self.unread(at, other)),
        }
    }

    /// A constant: an integer's bits above its type's width, and a float that is a
    /// NaN, are refused (no text writes either).
    fn const_value(&mut self) -> Read<ConstValue> {
        let at = self.at();
        match self.tag(v::CONST_VALUE)? {
            "Int" => {
                let s = self.scalar()?;
                let bits_at = self.at();
                let bits = self.u64()?;
                if s.bits() < 64 && bits >> s.bits() != 0 {
                    return Err(self.bad(
                        bits_at,
                        format!("an integer of {s} with bits above its width ({bits:#x})"),
                    ));
                }
                Ok(ConstValue::Int(s, bits))
            }
            "Float" => {
                let s = self.scalar()?;
                let bits_at = self.at();
                let x = f64::from_bits(self.u64()?);
                if x.is_nan() {
                    return Err(self.bad(bits_at, "a float constant that is a NaN".into()));
                }
                Ok(ConstValue::Float(s, x))
            }
            "Bool" => Ok(ConstValue::Bool(self.bool()?)),
            other => Err(self.unread(at, other)),
        }
    }

    fn binding(&mut self) -> Read<Binding> {
        let at = self.at();
        match self.tag(v::BINDING)? {
            "Build" => Ok(Binding::Build(self.const_value()?)),
            "Load" => Ok(Binding::Load(self.opt(Reader::const_value)?)),
            "Invoke" => Ok(Binding::Invoke(self.const_value()?)),
            other => Err(self.unread(at, other)),
        }
    }

    fn scalars_from(&mut self) -> Read<ScalarsFrom> {
        let at = self.at();
        match self.tag(v::SCALARS_FROM)? {
            "Value" => Ok(ScalarsFrom::Value(self.value()?)),
            "Member" => Ok(ScalarsFrom::Member(MemberId(self.u32()?))),
            other => Err(self.unread(at, other)),
        }
    }

    fn part(&mut self) -> Read<Option<Part>> {
        let at = self.at();
        match self.tag(v::PART)? {
            "Whole" => Ok(None),
            "Part" => Ok(Some(Part {
                first: self.count()?,
                end: self.count()?,
            })),
            other => Err(self.unread(at, other)),
        }
    }

    fn on(&mut self) -> Read<On> {
        let at = self.at();
        match self.tag(v::ON)? {
            "Group" => Ok(On::Group),
            "Grid" => Ok(On::Grid(self.u64()?, self.u64()?)),
            other => Err(self.unread(at, other)),
        }
    }

    fn peers(&mut self) -> Read<Peers> {
        let at = self.at();
        match self.tag(v::PEERS)? {
            "None" => Ok(Peers::None),
            "Offsets" => Ok(Peers::Offsets(self.list(Reader::u64)?)),
            "Pairs" => Ok(Peers::Pairs(self.list(|r| Ok((r.u64()?, r.u64()?)))?)),
            other => Err(self.unread(at, other)),
        }
    }

    pub(super) fn ty(&mut self) -> Read<Type> {
        self.deeper()?;
        let at = self.at();
        let t = match self.tag(v::TYPE)? {
            "Scalar" => Type::Scalar(self.scalar()?),
            "Bool" => Type::Bool,
            "Tile" => Type::Tile(self.format()?),
            "Reduced" => Type::Reduced(self.format()?, self.axis()?, self.count()?, self.count()?),
            "Block" => Type::Block(self.format()?, self.count()?, self.count()?),
            "Tensor" => Type::Tensor(self.format()?, self.bool()?),
            "View" => Type::View(self.opt(Reader::format)?),
            "Items" => Type::Items(self.count()?),
            "Stream" => Type::Stream(self.format()?, self.count()?, self.count()?),
            "Zip" => Type::Zip(Box::new(self.ty()?), Box::new(self.ty()?)),
            "Sink" => Type::Sink(self.format()?, self.count()?, self.count()?),
            "Words" => Type::Words(self.scalar()?),
            "Records" => Type::Records(RecordId(self.u32()?)),
            "RecordValue" => Type::RecordValue(RecordId(self.u32()?)),
            "Message" => Type::Message(MessageId(self.u32()?)),
            "Body" => Type::Body,
            "SubGroups" => Type::SubGroups,
            "Leaders" => Type::Leaders,
            "Broadcast" => Type::Broadcast(self.format()?),
            "Group" => Type::Group,
            "Handle" => Type::Handle(self.opt(Reader::scalar)?),
            "Outcome" => Type::Outcome(self.opt(Reader::scalar)?),
            "Reason" => Type::Reason,
            other => return Err(self.unread(at, other)),
        };
        self.depth -= 1;
        Ok(t)
    }

    // ---- bodies ----

    fn body(&mut self) -> Read<Body> {
        Ok(Body {
            values: self.list(Reader::ty)?,
            locals: self.list(Reader::ty)?,
            insts: self.list(|r| {
                Ok(Inst {
                    op: r.op()?,
                    defs: r.values()?,
                })
            })?,
        })
    }

    fn range(&mut self) -> Read<Range> {
        Ok(Range {
            lo: self.value()?,
            hi: self.value()?,
            step: self.opt_value()?,
        })
    }

    /// An operation: its number, then its operands in the order the variant declares
    /// them.
    pub(super) fn op(&mut self) -> Read<Op> {
        let at = self.at();
        let op = match self.tag(v::OP)? {
            "Const" => Op::Const(self.const_value()?),
            "Arg" => Op::Arg(self.u32()?),
            "Param" => Op::Param(ParamId(self.u32()?)),
            "Get" => Op::Get(LocalId(self.u32()?)),
            "Set" => Op::Set(LocalId(self.u32()?), self.value()?),
            "Binary" => Op::Binary(self.bin_op()?, self.value()?, self.value()?),
            "Compare" => Op::Compare(self.cmp_op()?, self.value()?, self.value()?),
            "LogicAnd" => Op::LogicAnd(self.value()?, self.value()?),
            "LogicOr" => Op::LogicOr(self.value()?, self.value()?),
            "If" => Op::If {
                cond: self.value()?,
            },
            "Else" => Op::Else,
            "ForRange" => Op::ForRange { n: self.value()? },
            "ForStream" => Op::ForStream {
                stream: self.value()?,
            },
            "ForZip" => Op::ForZip {
                streams: self.values()?,
            },
            "ForArrive" => Op::ForArrive {
                broadcast: self.value()?,
            },
            "While" => Op::While,
            "WhileTest" => Op::WhileTest {
                cond: self.value()?,
            },
            "End" => Op::End,
            "Return" => Op::Return(self.opt_value()?),
            "Convert" => Op::Convert(self.scalar()?, self.value()?),
            "Unary" => Op::Unary(self.unary_op()?, self.value()?),
            "Select" => Op::Select {
                cmp: self.cmp_op()?,
                a: self.value()?,
                b: self.value()?,
                x: self.value()?,
                y: self.value()?,
            },
            "Reduce" => Op::Reduce(self.reduce_op()?, self.value()?, self.axis()?),
            "SpreadCol" => Op::SpreadCol(self.value()?),
            "SpreadRow" => Op::SpreadRow(self.value()?),
            "Matmul" => Op::Matmul(self.value()?, self.value()?),
            "Transpose" => Op::Transpose(self.value()?),
            "Zeros" => Op::Zeros,
            "CallFunc" => Op::CallFunc(FuncId(self.u32()?), self.values()?),
            "Tiles" => Op::Tiles {
                tensor: self.value()?,
                first: self.value()?,
                n: self.value()?,
            },
            "Rect" => Op::Rect {
                tensor: self.value()?,
                rows: self.range()?,
                cols: self.range()?,
            },
            "Turn" => Op::Turn(self.value()?),
            "Items" => Op::Items {
                tensor: self.value()?,
                range: self.opt(|r| Ok((r.value()?, r.value()?)))?,
                size: self.value()?,
            },
            "ReadTile" => Op::ReadTile(self.value()?, self.value()?),
            "Words" => Op::Words(self.value()?),
            "Word" => Op::Word(self.value()?, self.value()?),
            "Records" => Op::Records(self.value()?, RecordId(self.u32()?), self.value()?),
            "RecordAt" => Op::RecordAt(self.value()?, self.value()?),
            "RecordField" => Op::RecordField(self.value()?, self.u32()?),
            "Put" => Op::Put {
                items: self.value()?,
                src: MemberId(self.u32()?),
                at: self.value()?,
            },
            "Partition" => Op::Partition(self.value()?, self.value()?, self.value()?),
            "Stream" => Op::Stream(self.value()?, self.value()?),
            "Sink" => Op::Sink(self.value()?, self.value()?),
            "Push" => Op::Push(self.value()?, self.value()?),
            "Store" => Op::Store(self.value()?, self.value()?),
            "StoreScalars" => Op::StoreScalars(self.value()?, self.scalars_from()?),
            "BlockAt" => Op::BlockAt(self.value()?, self.value()?),
            "Fill" => Op::Fill(MemberId(self.u32()?), self.value()?),
            "ZeroState" => Op::ZeroState(MemberId(self.u32()?)),
            "StateSet" => Op::StateSet(MemberId(self.u32()?), self.value()?, self.value()?),
            "StateGet" => Op::StateGet(MemberId(self.u32()?), self.value()?),
            "Bump" => Op::Bump(MemberId(self.u32()?), self.value()?),
            "Rank" => Op::Rank,
            "IsLeader" => Op::IsLeader,
            "WorkerId" => Op::WorkerId,
            "Row" => Op::Row,
            "Col" => Op::Col,
            "GroupSize" => Op::GroupSize,
            "GroupId" => Op::GroupId(self.value()?),
            "Split" => Op::Split(self.split_by()?, self.value()?),
            "Leaders" => Op::Leaders(self.split_by()?, self.value()?),
            "ReduceGroup" => Op::ReduceGroup {
                x: self.value()?,
                with: self.combiner()?,
                within: self.opt_value()?,
                among: self.opt_value()?,
                publish: self.values()?,
            },
            "AllReduce" => Op::AllReduce {
                x: self.value()?,
                with: self.combiner()?,
                publish: self.values()?,
            },
            "Gather" => Op::Gather(self.value()?),
            "AllGather" => Op::AllGather {
                value: self.value()?,
                publish: self.values()?,
            },
            "Broadcast" => Op::Broadcast(self.axis()?, self.value()?),
            "IsSender" => Op::IsSender(self.value()?),
            "Cast" => Op::Cast(self.value()?, self.value()?, self.opt_value()?),
            "Sync" => Op::Sync(self.values()?),
            "Arrived" => Op::Arrived,
            "MakeMessage" => Op::MakeMessage(MessageId(self.u32()?), self.values()?),
            "Send" => Op::Send {
                message: self.value()?,
                to: self.value()?,
                body: self.opt(Reader::scalars_from)?,
                publish: self.values()?,
                used: self.bool()?,
            },
            "Forward" => Op::Forward {
                message: self.value()?,
                as_type: self.opt(|r| Ok(MessageId(r.u32()?)))?,
                with: self.list(|r| Ok((r.u32()?, r.value()?)))?,
                to: self.value()?,
                publish: self.values()?,
            },
            "Consume" => Op::Consume(self.value()?),
            "Offer" => Op::Offer(
                self.value()?,
                self.value()?,
                self.opt(Reader::scalars_from)?,
            ),
            "Field" => Op::Field(self.value()?, self.u32()?),
            "Body" => Op::Body(self.value()?),
            "MakeGroup" => Op::MakeGroup(self.opt_value()?, self.value()?),
            "Invoke" => Op::Invoke {
                entry: self.entry()?,
                args: self.values()?,
                on: self.value()?,
                after: self.values()?,
                after_end: self.values()?,
            },
            "Await" => Op::Await(self.value()?),
            "Outcome" => Op::Outcome(self.value()?),
            "Failed" => Op::Failed(self.value()?),
            "Reason" => Op::Reason(self.value()?),
            "OutcomeValue" => Op::OutcomeValue(self.value()?),
            "FailureName" => Op::FailureName(self.failure()?),
            "IslandCount" => Op::IslandCount,
            "IslandOf" => Op::IslandOf(self.value()?),
            "IslandCrosses" => Op::IslandCrosses(self.value()?),
            "IslandHops" => Op::IslandHops(self.value()?),
            "ReduceTo" => Op::ReduceTo(self.value()?, self.value()?, self.combiner()?),
            "BroadcastFrom" => Op::BroadcastFrom(self.value()?, self.value()?),
            other => return Err(self.unread(at, other)),
        };
        Ok(op)
    }
}
