//! The format's version and magic, the codes of a refused file, and the number of every
//! variant of every enum the format writes. `docs/ir/bytecode.md` gives the same tables,
//! and a test checks that the two agree.
//!
//! A number is written here and nowhere else: never Rust's own order. It is never
//! reused or moved: a new variant takes a new number, a removed one leaves its number
//! unused, and either change is a new `VERSION`.

/// The first four bytes of a byte-code program.
pub const MAGIC: [u8; 4] = *b"KOMC";

/// The version this build writes, and the only one it reads. Any change to the layout
/// or to a table below is a new version. Version 6 is frozen: a test holds its tables.
pub const VERSION: u32 = 6;

/// The bytes of the header: the magic, the version (u32), and the length of each part,
/// S, H and P (three u64).
pub const HEADER_LEN: usize = 32;

/// A file that is not a byte-code program: too short for the magic and the version, or
/// another magic.
pub const NOT_BYTE_CODE: &str = "K950";

/// A byte-code program of a newer version than this build reads.
pub const NEWER: &str = "K951";

/// A byte-code program of an older version than this build reads.
pub const OLDER: &str = "K952";

/// A byte-code program whose bytes are not one: cut short, a number out of range, a
/// part whose walk does not end at its length, bytes after the last part.
pub const MALFORMED: &str = "K953";

/// A byte-code program that reads whole and does not validate.
pub const INVALID: &str = "K954";

/// A byte-code program whose derived parts (effects, a chain's dependencies, the
/// requirement set) are not the ones Derive gives its module again.
pub const DERIVED: &str = "K955";

/// The variants of one enum: each one's number and its name.
pub type Table = &'static [(u8, &'static str)];

/// The number of the variant `name` in `table`. A name missing from its table is a
/// fault of this crate, found by the test that encodes every variant.
pub fn number(table: Table, name: &str) -> u8 {
    match table.iter().find(|(_, n)| *n == name) {
        Some((number, _)) => *number,
        None => panic!("the variant {name} has no number in its table"),
    }
}

/// The name of the variant numbered `number` in `table`, if there is one.
pub fn name(table: Table, number: u8) -> Option<&'static str> {
    table
        .iter()
        .find(|(n, _)| *n == number)
        .map(|(_, name)| *name)
}

/// `program::BodyRef`.
pub const BODY_REF: Table = &[
    (0, "Func"),
    (1, "Handler"),
    (2, "Seed"),
    (3, "On"),
    (4, "Control"),
];

/// `program::FuncKind`.
pub const FUNC_KIND: Table = &[(0, "Plain"), (1, "Combine"), (2, "Extern")];

/// `program::MemberKind`.
pub const MEMBER_KIND: Table = &[(0, "Tiles"), (1, "Scalars")];

/// `program::Scope`.
pub const SCOPE: Table = &[(0, "Worker")];

/// `program::Lifetime`.
pub const LIFETIME: Table = &[(0, "Program")];

/// `program::CallArg`.
pub const CALL_ARG: Table = &[(0, "Param"), (1, "Past"), (2, "Const")];

/// `program::Op`.
pub const OP: Table = &[
    (0, "Const"),
    (1, "Arg"),
    (2, "Param"),
    (3, "Get"),
    (4, "Set"),
    (5, "Binary"),
    (6, "Compare"),
    (7, "LogicAnd"),
    (8, "LogicOr"),
    (9, "If"),
    (10, "Else"),
    (11, "ForRange"),
    (12, "ForStream"),
    (13, "ForZip"),
    (14, "ForArrive"),
    (15, "While"),
    (16, "WhileTest"),
    (17, "End"),
    (18, "Return"),
    (19, "Convert"),
    (20, "Unary"),
    (21, "Select"),
    (22, "Reduce"),
    (23, "SpreadCol"),
    (24, "SpreadRow"),
    (25, "Matmul"),
    (26, "Transpose"),
    (27, "Zeros"),
    (28, "CallFunc"),
    (29, "Tiles"),
    (30, "Rect"),
    (31, "Turn"),
    (32, "Items"),
    (33, "ReadTile"),
    (34, "Words"),
    (35, "Word"),
    (36, "Records"),
    (37, "RecordAt"),
    (38, "RecordField"),
    (39, "Put"),
    (40, "Partition"),
    (41, "Stream"),
    (42, "Sink"),
    (43, "Push"),
    (44, "Store"),
    (45, "BlockAt"),
    (46, "Fill"),
    (47, "ZeroState"),
    (48, "StateSet"),
    (49, "StateGet"),
    (50, "Bump"),
    (51, "Rank"),
    (52, "IsLeader"),
    (53, "WorkerId"),
    (54, "Row"),
    (55, "Col"),
    (56, "GroupSize"),
    (57, "GroupId"),
    (58, "Split"),
    (59, "Leaders"),
    (60, "ReduceGroup"),
    (61, "AllReduce"),
    (62, "Gather"),
    (63, "Broadcast"),
    (64, "IsSender"),
    (65, "Cast"),
    (66, "Sync"),
    (67, "Arrived"),
    (68, "MakeMessage"),
    (69, "Send"),
    (70, "Forward"),
    (71, "Consume"),
    (72, "Offer"),
    (73, "Field"),
    (74, "MakeGroup"),
    (75, "Invoke"),
    (76, "Await"),
    (77, "Outcome"),
    (78, "Failed"),
    (79, "Reason"),
    (80, "OutcomeValue"),
    (81, "FailureName"),
    (82, "IslandCount"),
    (83, "IslandOf"),
    (84, "IslandCrosses"),
    (85, "IslandHops"),
    (86, "ReduceTo"),
    (87, "BroadcastFrom"),
    (88, "Body"),
    (89, "AllGather"),
    (90, "StoreScalars"),
];

/// `program::Access`.
pub const ACCESS: Table = &[
    (0, "Stream"),
    (1, "Read"),
    (2, "Sink"),
    (3, "Store"),
    (4, "Put"),
    (5, "ExternRead"),
    (6, "ExternWrite"),
];

/// `program::StateUse`.
pub const STATE_USE: Table = &[(0, "Read"), (1, "WriteElement"), (2, "FillWhole")];

/// `program::Cause`.
pub const CAUSE: Table = &[
    (0, "Stated"),
    (1, "GridHandler"),
    (2, "DynamicRun"),
    (3, "ExternWithoutFallback"),
];

/// `program::Anchor`.
pub const ANCHOR: Table = &[
    (0, "Module"),
    (1, "Func"),
    (2, "Program"),
    (3, "Handler"),
    (4, "Run"),
    (5, "Member"),
    (6, "Control"),
    (7, "Chain"),
    (8, "Call"),
    (9, "Param"),
    (10, "Message"),
    (11, "Record"),
    (12, "ParamOf"),
    (13, "Inst"),
];

/// `program::Owner`.
pub const OWNER: Table = &[
    (0, "Func"),
    (1, "Handler"),
    (2, "Run"),
    (3, "Chain"),
    (4, "Control"),
];

/// `program::HintArg`.
pub const HINT_ARG: Table = &[
    (0, "Number"),
    (1, "Text"),
    (2, "Name"),
    (3, "Named"),
    (4, "List"),
];

/// `program::Named`.
pub const NAMED: Table = &[(0, "Value"), (1, "Local")];

/// `types::Type`.
pub const TYPE: Table = &[
    (0, "Scalar"),
    (1, "Bool"),
    (2, "Tile"),
    (3, "Reduced"),
    (4, "Block"),
    (5, "Tensor"),
    (6, "View"),
    (7, "Items"),
    (8, "Stream"),
    (9, "Zip"),
    (10, "Sink"),
    (11, "Words"),
    (12, "Records"),
    (13, "RecordValue"),
    (14, "Message"),
    (15, "SubGroups"),
    (16, "Leaders"),
    (17, "Broadcast"),
    (18, "Group"),
    (19, "Handle"),
    (20, "Outcome"),
    (21, "Reason"),
    (22, "Body"),
];

/// `types::ConstValue`.
pub const CONST_VALUE: Table = &[(0, "Int"), (1, "Float"), (2, "Bool")];

/// `types::Binding`.
pub const BINDING: Table = &[(0, "Build"), (1, "Load"), (2, "Invoke")];

/// `types::On`.
pub const ON: Table = &[(0, "Group"), (1, "Grid")];

/// A chain call's `Option<program::Part>`.
pub const PART: Table = &[(0, "Whole"), (1, "Part")];

/// `program::body::ScalarsFrom`: where a message's body comes from.
pub const SCALARS_FROM: Table = &[(0, "Value"), (1, "Member")];

/// `types::Peers`.
pub const PEERS: Table = &[(0, "None"), (1, "Offsets"), (2, "Pairs")];

/// `types::Ability`.
pub const ABILITY: Table = &[
    (0, "LocalState"),
    (1, "Grid"),
    (2, "Dynamic"),
    (3, "Islands"),
    (4, "Impl"),
];

/// `types::Count`.
pub const COUNT: Table = &[(0, "Const"), (1, "Param"), (2, "GroupSize")];

/// `types::Axis`.
pub const AXIS: Table = &[(0, "Row"), (1, "Col")];

/// `types::Format`.
pub const FORMAT: Table = &[
    (0, "U8"),
    (1, "U16"),
    (2, "U32"),
    (3, "U64"),
    (4, "I8"),
    (5, "I16"),
    (6, "I32"),
    (7, "I64"),
    (8, "F16"),
    (9, "F32"),
    (10, "F64"),
    (11, "Bf16"),
    (12, "Bfp8"),
    (13, "Bfp4"),
    (14, "Bfp2"),
];

/// `types::ScalarType`.
pub const SCALAR_TYPE: Table = &[
    (0, "U8"),
    (1, "U16"),
    (2, "U32"),
    (3, "U64"),
    (4, "I8"),
    (5, "I16"),
    (6, "I32"),
    (7, "I64"),
    (8, "F16"),
    (9, "F32"),
    (10, "F64"),
];

/// `types::BinOp`.
pub const BIN_OP: Table = &[
    (0, "Add"),
    (1, "Sub"),
    (2, "Mul"),
    (3, "Div"),
    (4, "Rem"),
    (5, "And"),
    (6, "Or"),
    (7, "Xor"),
    (8, "Shl"),
    (9, "Shr"),
    (10, "Min"),
    (11, "Max"),
];

/// `types::CmpOp`.
pub const CMP_OP: Table = &[
    (0, "Gt"),
    (1, "Lt"),
    (2, "Ge"),
    (3, "Le"),
    (4, "Eq"),
    (5, "Ne"),
];

/// `types::UnaryOp`.
pub const UNARY_OP: Table = &[
    (0, "Neg"),
    (1, "Abs"),
    (2, "Relu"),
    (3, "Square"),
    (4, "Floor"),
    (5, "Exp"),
    (6, "Exp2"),
    (7, "Tanh"),
    (8, "Erf"),
    (9, "Log"),
    (10, "Sigmoid"),
    (11, "Silu"),
    (12, "Sqrt"),
    (13, "Rsqrt"),
    (14, "Reciprocal"),
];

/// `types::ReduceOp`.
pub const REDUCE_OP: Table = &[(0, "Sum"), (1, "Max"), (2, "Min")];

/// `types::SplitBy`.
pub const SPLIT_BY: Table = &[(0, "Parts"), (1, "Size")];

/// `types::Combiner`.
pub const COMBINER: Table = &[
    (0, "Func"),
    (1, "Sum"),
    (2, "Max"),
    (3, "Min"),
    (4, "And"),
    (5, "Or"),
];

/// `types::Failure`.
pub const FAILURE: Table = &[
    (0, "ViewEnd"),
    (1, "EmptyState"),
    (2, "SendRefused"),
    (3, "Capacity"),
    (4, "GivenUp"),
    (5, "Dependency"),
    (6, "NotAPeer"),
    (7, "OverBound"),
    (8, "DivideByZero"),
    (9, "OutOfRange"),
];

/// `types::Entry`.
pub const ENTRY: Table = &[(0, "Handler"), (1, "Run"), (2, "Chain")];

/// Every table, by the name of its enum, in the order of the page.
pub const TABLES: &[(&str, Table)] = &[
    ("BodyRef", BODY_REF),
    ("FuncKind", FUNC_KIND),
    ("MemberKind", MEMBER_KIND),
    ("Scope", SCOPE),
    ("Lifetime", LIFETIME),
    ("CallArg", CALL_ARG),
    ("Op", OP),
    ("Access", ACCESS),
    ("StateUse", STATE_USE),
    ("Cause", CAUSE),
    ("Anchor", ANCHOR),
    ("Owner", OWNER),
    ("HintArg", HINT_ARG),
    ("Named", NAMED),
    ("Type", TYPE),
    ("ConstValue", CONST_VALUE),
    ("Binding", BINDING),
    ("On", ON),
    ("Part", PART),
    ("ScalarsFrom", SCALARS_FROM),
    ("Peers", PEERS),
    ("Ability", ABILITY),
    ("Count", COUNT),
    ("Axis", AXIS),
    ("Format", FORMAT),
    ("ScalarType", SCALAR_TYPE),
    ("BinOp", BIN_OP),
    ("CmpOp", CMP_OP),
    ("UnaryOp", UNARY_OP),
    ("ReduceOp", REDUCE_OP),
    ("SplitBy", SPLIT_BY),
    ("Combiner", COMBINER),
    ("Failure", FAILURE),
    ("Entry", ENTRY),
];
