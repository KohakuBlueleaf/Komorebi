//! Encode: the program structure to a byte-code program (`.komc`) and back, and its
//! hashes. It reads the program structure and the types only.
//!
//! A byte-code program is a header and three parts: S, the program (everything but
//! `Module::notes`); H, the hints; P, the source positions and the names of values and
//! parameters. The header is the magic `KOMC`, the version, and the length of each part.
//! Every integer is fixed width and little-endian, every enum's number is written in
//! `version`, and the reader refuses any byte the writer would not have written: one
//! module has one byte string, so the bytes are what the hashes cover.
//! `docs/ir/bytecode.md` gives the layout.

pub mod decode;
pub mod encode;
pub mod hash;
pub mod version;

pub use decode::{decode, split};
pub use encode::{encode, parts, Parts};
pub use hash::{hashes, hex, Hashes};

/// Why a byte-code program is refused: the code, what is wrong, and the byte of the
/// file it is about (none for a module that reads whole and does not validate).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// The code: one of `version`'s.
    pub code: &'static str,
    /// What is wrong, in words.
    pub what: String,
    /// The offset in the file.
    pub at: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::decode::Reader;
    use super::encode::Writer;
    use super::version::{self as v, TABLES};
    use crate::program::body::{Op, Range, ScalarsFrom, ValueId};
    use crate::program::notes::HintArg;
    use crate::types::{
        Axis, BinOp, ChainId, CmpOp, Combiner, ConstValue, Count, Entry, Failure, Format, FuncId,
        MemberId, MessageId, ParamId, ProgramId, RecordId, ReduceOp, ScalarType, SplitBy, Type,
        UnaryOp,
    };

    #[test]
    fn every_table_numbers_each_variant_once() {
        for (enum_name, table) in TABLES {
            for (i, (n, name)) in table.iter().enumerate() {
                for (m, other) in &table[i + 1..] {
                    assert_ne!(n, m, "{enum_name}: two variants numbered {n}");
                    assert_ne!(name, other, "{enum_name}: {name} numbered twice");
                }
            }
        }
    }

    fn variant(x: &impl std::fmt::Debug) -> String {
        let text = format!("{x:?}");
        text.split(|c: char| !c.is_alphanumeric())
            .next()
            .unwrap_or_default()
            .to_string()
    }

    /// Every operation, each with distinct operands, written and read back.
    #[test]
    fn every_operation_reads_back_as_written() {
        let x = ValueId;
        let range = Range {
            lo: x(1),
            hi: x(2),
            step: Some(x(3)),
        };
        let ops = vec![
            Op::Const(ConstValue::Float(ScalarType::F32, -0.5)),
            Op::Arg(3),
            Op::Param(ParamId(2)),
            Op::Get(crate::program::body::LocalId(4)),
            Op::Set(crate::program::body::LocalId(4), x(5)),
            Op::Binary(BinOp::Shr, x(1), x(2)),
            Op::Compare(CmpOp::Ne, x(1), x(2)),
            Op::LogicAnd(x(1), x(2)),
            Op::LogicOr(x(2), x(1)),
            Op::If { cond: x(7) },
            Op::Else,
            Op::ForRange { n: x(8) },
            Op::ForStream { stream: x(9) },
            Op::ForZip {
                streams: vec![x(1), x(2), x(3)],
            },
            Op::ForArrive { broadcast: x(3) },
            Op::While,
            Op::WhileTest { cond: x(4) },
            Op::End,
            Op::Return(None),
            Op::Convert(ScalarType::I16, x(1)),
            Op::Unary(UnaryOp::Reciprocal, x(1)),
            Op::Select {
                cmp: CmpOp::Le,
                a: x(1),
                b: x(2),
                x: x(3),
                y: x(4),
            },
            Op::Reduce(ReduceOp::Min, x(1), Axis::Col),
            Op::SpreadCol(x(1)),
            Op::SpreadRow(x(2)),
            Op::Matmul(x(1), x(2)),
            Op::Transpose(x(3)),
            Op::Zeros,
            Op::CallFunc(FuncId(2), vec![x(1), x(2)]),
            Op::Tiles {
                tensor: x(1),
                first: x(2),
                n: x(3),
            },
            Op::Rect {
                tensor: x(1),
                rows: range,
                cols: Range {
                    step: None,
                    ..range
                },
            },
            Op::Turn(x(1)),
            Op::Items {
                tensor: x(1),
                range: Some((x(2), x(3))),
                size: x(4),
            },
            Op::ReadTile(x(1), x(2)),
            Op::Words(x(1)),
            Op::Word(x(1), x(2)),
            Op::Records(x(1), RecordId(2), x(3)),
            Op::RecordAt(x(1), x(2)),
            Op::RecordField(x(1), 5),
            Op::Put {
                items: x(1),
                src: MemberId(2),
                at: x(3),
            },
            Op::Partition(x(1), x(2), x(3)),
            Op::Stream(x(1), x(2)),
            Op::Sink(x(1), x(2)),
            Op::Push(x(1), x(2)),
            Op::Store(x(1), x(2)),
            Op::BlockAt(x(1), x(2)),
            Op::Fill(MemberId(1), x(2)),
            Op::ZeroState(MemberId(3)),
            Op::StateSet(MemberId(1), x(2), x(3)),
            Op::StateGet(MemberId(1), x(2)),
            Op::Bump(MemberId(1), x(2)),
            Op::Rank,
            Op::IsLeader,
            Op::WorkerId,
            Op::Row,
            Op::Col,
            Op::GroupSize,
            Op::GroupId(x(1)),
            Op::Split(SplitBy::Size, x(1)),
            Op::Leaders(SplitBy::Parts, x(1)),
            Op::ReduceGroup {
                x: x(1),
                with: Combiner::Func(FuncId(4)),
                within: Some(x(2)),
                among: None,
                publish: vec![x(3)],
            },
            Op::AllReduce {
                x: x(1),
                with: Combiner::Or,
                publish: vec![],
            },
            Op::Gather(x(1)),
            Op::AllGather {
                value: x(1),
                publish: vec![x(2)],
            },
            Op::Broadcast(Axis::Row, x(1)),
            Op::IsSender(x(1)),
            Op::Cast(x(1), x(2), Some(x(3))),
            Op::Sync(vec![x(1), x(2)]),
            Op::Arrived,
            Op::MakeMessage(MessageId(1), vec![x(2)]),
            Op::Send {
                message: x(1),
                to: x(2),
                body: None,
                publish: vec![x(3)],
                used: true,
            },
            Op::Send {
                message: x(1),
                to: x(2),
                body: Some(ScalarsFrom::Member(MemberId(3))),
                publish: vec![],
                used: false,
            },
            Op::Forward {
                message: x(1),
                as_type: None,
                with: vec![(0, x(2))],
                to: x(3),
                publish: vec![],
            },
            Op::Forward {
                message: x(1),
                as_type: Some(MessageId(2)),
                with: vec![(0, x(2)), (1, x(3))],
                to: x(4),
                publish: vec![x(5)],
            },
            Op::Consume(x(1)),
            Op::Offer(x(1), x(2), None),
            Op::Offer(x(1), x(2), Some(ScalarsFrom::Value(x(3)))),
            Op::Field(x(1), 2),
            Op::Body(x(1)),
            Op::MakeGroup(Some(x(1)), x(2)),
            Op::Invoke {
                entry: Entry::Chain(ProgramId(1), ChainId(2)),
                args: vec![x(1)],
                on: x(2),
                after: vec![x(3)],
                after_end: vec![x(4)],
            },
            Op::Await(x(1)),
            Op::Outcome(x(1)),
            Op::Failed(x(1)),
            Op::Reason(x(1)),
            Op::OutcomeValue(x(1)),
            Op::FailureName(Failure::OutOfRange),
            Op::IslandCount,
            Op::IslandOf(x(1)),
            Op::IslandCrosses(x(1)),
            Op::IslandHops(x(1)),
            Op::ReduceTo(x(1), x(2), Combiner::Max),
            Op::BroadcastFrom(x(1), x(2)),
        ];
        let mut named: Vec<String> = ops.iter().map(variant).collect();
        named.sort();
        named.dedup();
        assert_eq!(named.len(), v::OP.len(), "one instance of every operation");
        for op in &ops {
            let mut w = Writer::default();
            w.op(op);
            assert_eq!(v::name(v::OP, w.out[0]), Some(variant(op).as_str()));
            let mut r = Reader::new(&w.out, 0, "test");
            assert_eq!(r.op().as_ref(), Ok(op));
            r.finish().unwrap_or_else(|e| panic!("{op:?}: {e:?}"));
        }
    }

    /// Every type, every format and every scalar type, written and read back.
    #[test]
    fn every_type_reads_back_as_written() {
        let f = Format::Bfp4;
        let s = ScalarType::U64;
        let mut types = vec![
            Type::Scalar(s),
            Type::Bool,
            Type::Tile(f),
            Type::Reduced(f, Axis::Row, Count::Const(2), Count::Const(3)),
            Type::Block(f, Count::Const(4), Count::Const(9)),
            Type::Tensor(f, true),
            Type::View(None),
            Type::Items(Count::Param(ParamId(1))),
            Type::Stream(f, Count::Const(1), Count::GroupSize),
            Type::Zip(Box::new(Type::Bool), Box::new(Type::Tile(f))),
            Type::Sink(f, Count::Const(1), Count::Const(2)),
            Type::Words(s),
            Type::Records(RecordId(1)),
            Type::RecordValue(RecordId(2)),
            Type::Message(MessageId(3)),
            Type::Body,
            Type::SubGroups,
            Type::Leaders,
            Type::Broadcast(f),
            Type::Group,
            Type::Handle(Some(s)),
            Type::Outcome(None),
            Type::Reason,
        ];
        let mut named: Vec<String> = types.iter().map(variant).collect();
        named.sort();
        named.dedup();
        assert_eq!(named.len(), v::TYPE.len(), "one instance of every type");
        types.extend(Format::ALL.iter().map(|f| Type::Tile(*f)));
        types.extend(ScalarType::ALL.iter().map(|s| Type::Scalar(*s)));
        for ty in &types {
            let mut w = Writer::default();
            w.ty(ty);
            let mut r = Reader::new(&w.out, 0, "test");
            assert_eq!(r.ty().as_ref(), Ok(ty));
            r.finish().unwrap_or_else(|e| panic!("{ty:?}: {e:?}"));
        }
    }

    #[test]
    fn a_type_nested_too_deep_is_refused() {
        let mut ty = Type::Bool;
        for _ in 0..300 {
            ty = Type::Zip(Box::new(ty), Box::new(Type::Bool));
        }
        let mut w = Writer::default();
        w.ty(&ty);
        let mut r = Reader::new(&w.out, 0, "test");
        let refusal = r.ty().expect_err("refused");
        assert_eq!(refusal.code, v::MALFORMED);
        assert!(
            refusal.what.contains("nested deeper than 256"),
            "{refusal:?}"
        );
    }

    #[test]
    fn a_hint_argument_nested_too_deep_is_refused() {
        let mut arg = HintArg::Name("x".into());
        for _ in 0..300 {
            arg = HintArg::List(vec![arg]);
        }
        let mut w = Writer::default();
        w.hint_arg(&arg);
        let mut r = Reader::new(&w.out, 0, "test");
        assert_eq!(r.hint_arg().expect_err("refused").code, v::MALFORMED);
    }
}
