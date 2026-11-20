//! The program structure's validator: a valid module gives nothing, and each kind of
//! violation, made by hand in a small module, is found.

use komorebi_ir::program::body::{Body, Inst, InstId, LocalId, Op, ValueId};
use komorebi_ir::program::derive::{Cause, Requirement};
use komorebi_ir::program::notes::{Anchor, Hint, Named, SourceSpan};
use komorebi_ir::program::{
    validate, BodyRef, Func, FuncKind, Lifetime, Member, MemberKind, Message, Module, Program, Run,
    Scope,
};
use komorebi_ir::types::{
    Ability, BinOp, CmpOp, ConstValue, Count, Format, FuncId, MessageId, ParamId, Peers, ProgramId,
    RecordId, ScalarType, TileShape, Type,
};

const U32: Type = Type::Scalar(ScalarType::U32);
const F32: Type = Type::Scalar(ScalarType::F32);

fn inst(op: Op, defs: &[u32]) -> Inst {
    Inst {
        op,
        defs: defs.iter().map(|d| ValueId(*d)).collect(),
    }
}

fn v(i: u32) -> ValueId {
    ValueId(i)
}

/// A module of one function `f(params) -> ret` with this body.
fn module(params: Vec<Type>, ret: Type, values: Vec<Type>, insts: Vec<Inst>) -> Module {
    Module {
        name: "m".into(),
        tile: TileShape { rows: 32, cols: 32 },
        formats: vec![Format::F32],
        params: Vec::new(),
        records: Vec::new(),
        messages: Vec::new(),
        funcs: vec![Func {
            name: "f".into(),
            kind: FuncKind::Plain,
            params,
            ret,
            body: Some(Body {
                values,
                locals: Vec::new(),
                insts,
            }),
            effects: Default::default(),
        }],
        programs: Vec::new(),
        controls: Vec::new(),
        requirements: Vec::new(),
        notes: Default::default(),
    }
}

/// `f(n: u32) -> u32 { var s = 0; for i in range(n) { s = s + i } return s }`.
fn valid() -> Module {
    let mut m = module(
        vec![U32],
        U32,
        vec![U32, U32, U32, U32, U32, U32],
        vec![
            inst(Op::Arg(0), &[0]),
            inst(Op::Const(ConstValue::Int(ScalarType::U32, 0)), &[1]),
            inst(Op::Set(LocalId(0), v(1)), &[]),
            inst(Op::ForRange { n: v(0) }, &[2]),
            inst(Op::Get(LocalId(0)), &[3]),
            inst(Op::Binary(BinOp::Add, v(3), v(2)), &[4]),
            inst(Op::Set(LocalId(0), v(4)), &[]),
            inst(Op::End, &[]),
            inst(Op::Get(LocalId(0)), &[5]),
            inst(Op::Return(Some(v(5))), &[]),
        ],
    );
    m.funcs[0].body.as_mut().unwrap().locals.push(U32);
    m
}

fn body(m: &mut Module) -> &mut Body {
    m.funcs[0].body.as_mut().unwrap()
}

/// The violations' texts, for a message on failure.
fn found(m: &Module) -> Vec<String> {
    validate(m).into_iter().map(|x| x.what).collect()
}

fn has(m: &Module, part: &str) {
    let all = found(m);
    assert!(
        all.iter().any(|w| w.contains(part)),
        "{part:?} not in {all:?}"
    );
}

#[test]
fn a_valid_module_has_no_violation() {
    assert_eq!(found(&valid()), Vec::<String>::new());
}

#[test]
fn constructs_balance() {
    let mut m = valid();
    body(&mut m).insts.remove(7); // the loop's End
    has(&m, "left open");

    let mut m = valid();
    body(&mut m).insts.insert(8, inst(Op::End, &[]));
    has(&m, "nothing open");

    let mut m = valid();
    body(&mut m).insts.insert(4, inst(Op::Else, &[]));
    has(&m, "`Else` outside an `If`");

    let mut m = module(
        vec![],
        U32,
        vec![Type::Bool, U32],
        vec![
            inst(Op::While, &[]),
            inst(Op::Const(ConstValue::Bool(true)), &[0]),
            inst(Op::End, &[]),
            inst(Op::Const(ConstValue::Int(ScalarType::U32, 0)), &[1]),
            inst(Op::Return(Some(v(1))), &[]),
        ],
    );
    has(&m, "closed before its `WhileTest`");
    body(&mut m)
        .insts
        .insert(2, inst(Op::WhileTest { cond: v(0) }, &[]));
    assert_eq!(found(&m), Vec::<String>::new());
}

#[test]
fn an_operand_is_defined_before_its_use() {
    let mut m = valid();
    body(&mut m).insts[5].op = Op::Binary(BinOp::Add, v(3), v(5));
    has(&m, "v5 is not defined here");
}

#[test]
fn an_operand_is_not_used_after_its_construct_ends() {
    // the loop's i (v2) read after the loop's End
    let mut m = valid();
    body(&mut m).insts[9].op = Op::Return(Some(v(2)));
    has(&m, "v2 is not defined here");

    // a value of the `then` branch read in the `else` branch
    let m = module(
        vec![Type::Bool],
        U32,
        vec![Type::Bool, U32, U32],
        vec![
            inst(Op::Arg(0), &[0]),
            inst(Op::If { cond: v(0) }, &[]),
            inst(Op::Const(ConstValue::Int(ScalarType::U32, 1)), &[1]),
            inst(Op::Else, &[]),
            inst(Op::Binary(BinOp::Add, v(1), v(1)), &[2]),
            inst(Op::End, &[]),
            inst(Op::Return(None), &[]),
        ],
    );
    has(&m, "v1 is not defined here");
}

#[test]
fn values_are_numbered_in_order() {
    let mut m = valid();
    body(&mut m).insts[1].defs = vec![v(2)];
    has(&m, "v2 is not the next value");

    let mut m = valid();
    body(&mut m).values.push(U32);
    has(&m, "7 values typed, 6 defined");
}

#[test]
fn operand_types_fit_the_operation() {
    let mut m = valid();
    body(&mut m).values[2] = F32; // the loop's i, as an f32: range of a u32 gives a u32
    has(&m, "defines");

    let mut m = valid();
    body(&mut m).insts[1].op = Op::Const(ConstValue::Float(ScalarType::F32, 0.0));
    body(&mut m).values[1] = F32;
    has(&m, "a `Set` of another type");

    let m = module(
        vec![U32, F32],
        U32,
        vec![U32, F32, U32],
        vec![
            inst(Op::Arg(0), &[0]),
            inst(Op::Arg(1), &[1]),
            inst(Op::Binary(BinOp::Add, v(0), v(1)), &[2]),
            inst(Op::Return(Some(v(2))), &[]),
        ],
    );
    has(&m, "binary operation do not fit");

    let m = module(
        vec![U32],
        U32,
        vec![U32],
        vec![
            inst(Op::Arg(0), &[0]),
            inst(Op::If { cond: v(0) }, &[]),
            inst(Op::End, &[]),
            inst(Op::Return(Some(v(0))), &[]),
        ],
    );
    has(&m, "not a bool");

    let m = module(
        vec![U32],
        F32,
        vec![U32],
        vec![inst(Op::Arg(0), &[0]), inst(Op::Return(Some(v(0))), &[])],
    );
    has(&m, "`return` of another type");

    let m = module(
        vec![U32, U32],
        Type::Bool,
        vec![U32, U32, Type::Bool],
        vec![
            inst(Op::Arg(0), &[0]),
            inst(Op::Arg(1), &[1]),
            inst(Op::Compare(CmpOp::Lt, v(0), v(1)), &[2]),
            inst(Op::Return(Some(v(2))), &[]),
        ],
    );
    assert_eq!(found(&m), Vec::<String>::new());
}

#[test]
fn a_gather_yields_a_block_of_the_groups_size() {
    let tile = Type::Tile(Format::Bf16);
    let m = module(
        vec![tile.clone()],
        Type::Block(Format::Bf16, Count::GroupSize),
        vec![tile.clone(), Type::Block(Format::Bf16, Count::GroupSize)],
        vec![
            inst(Op::Arg(0), &[0]),
            inst(Op::Gather(v(0)), &[1]),
            inst(Op::Return(Some(v(1))), &[]),
        ],
    );
    assert_eq!(found(&m), Vec::<String>::new());

    let mut m = m;
    body(&mut m).values[1] = Type::Block(Format::Bf16, Count::Const(4));
    has(&m, "defines");
}

#[test]
fn ids_are_in_range() {
    let mut m = valid();
    body(&mut m).insts[4].op = Op::Get(LocalId(3));
    has(&m, "local 3 out of range");

    let mut m = valid();
    body(&mut m).insts[5] = inst(Op::CallFunc(FuncId(9), vec![v(3), v(2)]), &[4]);
    has(&m, "function 9 out of range");

    let mut m = valid();
    m.programs.push(Program {
        name: "p".into(),
        state: vec![Member {
            name: "s".into(),
            kind: MemberKind::Scalars(ScalarType::U32),
            count: Count::Param(ParamId(2)),
            scope: Scope::Worker,
            lifetime: Lifetime::Program,
            export: false,
            by_element: false,
        }],
        handlers: Vec::new(),
        runs: Vec::new(),
        chains: Vec::new(),
    });
    has(&m, "parameter 2 out of range");
    m.programs[0].state[0].count = Count::GroupSize;
    has(&m, "a member counted by the group's size");

    let mut m = valid();
    m.requirements.push(Requirement {
        ability: Ability::Dynamic,
        causes: vec![Cause::ExternWithoutFallback(FuncId(9))],
    });
    has(&m, "a requirement's cause that names nothing");

    // a run's `on M` names a message of the module (one numbering), of its own program
    let mut m = valid();
    m.messages.push(Message {
        name: "M".into(),
        program: None,
        fields: vec![("x".into(), ScalarType::U32)],
    });
    let empty = || Body {
        values: Vec::new(),
        locals: Vec::new(),
        insts: Vec::new(),
    };
    let on = Body {
        values: vec![Type::Message(MessageId(0))],
        locals: Vec::new(),
        insts: vec![inst(Op::Arrived, &[0])],
    };
    m.programs.push(Program {
        name: "p".into(),
        state: Vec::new(),
        handlers: Vec::new(),
        runs: vec![Run {
            name: "r".into(),
            params: Vec::new(),
            peers: Peers::None,
            seed: empty(),
            on_message: vec![(MessageId(0), on)],
            effects: Default::default(),
        }],
        chains: Vec::new(),
    });
    has(&m, "run r: message 0 not of its program");
    m.messages[0].program = Some(ProgramId(0));
    assert_eq!(found(&m), Vec::<String>::new());
    m.messages[0].program = Some(ProgramId(3));
    has(&m, "M: declared in a program out of range");
    m.messages[0].program = Some(ProgramId(0));
    m.programs[0].runs[0].on_message[0].0 = MessageId(1);
    has(&m, "run r: message 1 not of its program");
}

#[test]
fn a_body_opens_with_its_args() {
    let mut m = valid();
    body(&mut m).insts.swap(0, 1);
    body(&mut m).insts[0].defs = vec![v(0)];
    body(&mut m).insts[1].defs = vec![v(1)];
    has(&m, "does not open with `Arg 0`");
}

#[test]
fn every_anchor_names_something() {
    let hint = |anchor| Hint {
        anchor,
        path: vec!["tt".into(), "cb".into()],
        args: Vec::new(),
    };
    let mut m = valid();
    m.notes.hints.push(hint(Anchor::Func(FuncId(5))));
    has(&m, "an anchor that names nothing");

    let mut m = valid();
    let at = Anchor::Inst(BodyRef::Func(FuncId(0)), InstId(10));
    m.notes.spans.push((at, SourceSpan { start: 0, end: 1 }));
    has(&m, "an anchor that names nothing");

    let mut m = valid();
    m.notes.hints.push(hint(Anchor::Chain(
        ProgramId(0),
        komorebi_ir::types::ChainId(0),
    )));
    has(&m, "an anchor that names nothing");

    let mut m = valid();
    let f = BodyRef::Func(FuncId(0));
    m.notes.names.push(Named::Value(f, v(6), "x".into()));
    has(&m, "a name of a value or a slot that does not exist");

    let mut m = valid();
    m.notes
        .param_names
        .push((Anchor::Func(FuncId(0)), "n".into()));
    has(&m, "a parameter's name not on a parameter");

    for anchor in [
        Anchor::Param(ParamId(0)),
        Anchor::Record(RecordId(0)),
        Anchor::Message(MessageId(0)),
    ] {
        let mut m = valid();
        m.notes.hints.push(hint(anchor));
        has(&m, "an anchor that names nothing");
    }
}

#[test]
fn notes_are_in_their_order() {
    let mut m = valid();
    let f = BodyRef::Func(FuncId(0));
    let span = SourceSpan { start: 0, end: 1 };
    m.notes.spans.push((Anchor::Inst(f, InstId(3)), span));
    m.notes.spans.push((Anchor::Inst(f, InstId(1)), span));
    has(&m, "spans: not in the order of their anchors");

    let mut m = valid();
    m.notes.names.push(Named::Local(f, LocalId(0), "s".into()));
    m.notes.names.push(Named::Value(f, v(2), "i".into()));
    has(&m, "names: not in the order");
    m.notes.names.reverse();
    assert_eq!(found(&m), Vec::<String>::new());
}

#[test]
fn the_printer_writes_one_line_an_instruction() {
    let text = komorebi_ir::program::print(&valid());
    let expected = "\
module \"m\"  tile [32, 32]  formats [f32]
params: []
records: []
messages: []
requirements: []
funcs:
  f0 f  Plain  params [u32] -> u32
    locals: l0 u32
    i0  v0: u32 = Arg 0
    i1  v1: u32 = Const u32 0
    i2  Set l0 v1
    i3  v2: u32 = ForRange { n v0 }
    i4  v3: u32 = Get l0
    i5  v4: u32 = Binary Add v3 v2
    i6  Set l0 v4
    i7  End
    i8  v5: u32 = Get l0
    i9  Return v5
programs: []
controls: []
notes:
  hints: []
  spans: []
  names: []
  param_names: []
";
    assert_eq!(text, expected);
}

#[test]
fn a_conversion_names_its_type() {
    let m = module(
        vec![U32],
        F32,
        vec![U32, F32],
        vec![
            inst(Op::Arg(0), &[0]),
            inst(Op::Convert(ScalarType::F32, v(0)), &[1]),
            inst(Op::Return(Some(v(1))), &[]),
        ],
    );
    assert_eq!(found(&m), Vec::<String>::new());
    let mut m = m;
    body(&mut m).values[1] = U32;
    has(&m, "defines");
}

#[test]
fn float_formats_are_a_notation_integer_formats_are_not() {
    let returns = |given: Format, declared: Format| {
        module(
            vec![Type::Tile(given)],
            Type::Tile(declared),
            vec![Type::Tile(given)],
            vec![inst(Op::Arg(0), &[0]), inst(Op::Return(Some(v(0))), &[])],
        )
    };
    assert_eq!(
        found(&returns(Format::Bf16, Format::F32)),
        Vec::<String>::new()
    );
    assert_eq!(
        found(&returns(Format::Bfp8, Format::F16)),
        Vec::<String>::new()
    );
    has(
        &returns(Format::U32, Format::I32),
        "`return` of another type",
    );
    has(
        &returns(Format::U32, Format::F32),
        "`return` of another type",
    );
    has(
        &returns(Format::F32, Format::U8),
        "`return` of another type",
    );
}

#[test]
fn a_function_on_blocks_reads_one_tile_and_its_blocks_have_one_count() {
    let block = |n| Type::Block(Format::F32, Count::Const(n));
    let tile = Type::Tile(Format::F32);
    // `f(x: tile<f32>[3], k: f32) -> tile<f32>[3] { return x * k }`: a tile of the block
    let make = |x: Type, ret: Type| {
        module(
            vec![x, F32],
            ret,
            vec![tile.clone(), F32, tile.clone()],
            vec![
                inst(Op::Arg(0), &[0]),
                inst(Op::Arg(1), &[1]),
                inst(Op::Binary(BinOp::Mul, v(0), v(1)), &[2]),
                inst(Op::Return(Some(v(2))), &[]),
            ],
        )
    };
    let m = make(block(3), block(3));
    assert!(validate(&m).is_empty(), "{:?}", found(&m));
    has(&make(block(3), block(4)), "on blocks");
    has(&make(block(3), tile.clone()), "on blocks");
    let mut m = make(block(3), block(3));
    m.funcs[0].kind = FuncKind::Combine { commutative: false };
    has(&m, "on blocks");
}
