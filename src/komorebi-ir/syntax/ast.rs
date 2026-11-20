//! The syntax tree of a Komorebi IR module, as the text says it.
//!
//! Every node keeps its source position (a [`Span`]) and every comment of the text is
//! kept beside the node it belongs to, so the writer gives the text back.

pub use crate::span::Span;

/// A name written in the source.
#[derive(Clone, Debug, PartialEq)]
pub struct Ident {
    /// The name.
    pub name: String,
    /// Where it is written.
    pub span: Span,
    /// Written between backticks: the author's item of that name, never the language's
    /// (R50).
    pub quoted: bool,
}

/// A `//` comment. `text` holds it from the `//` to the end of its line, with trailing
/// spaces removed.
#[derive(Clone, Debug, PartialEq)]
pub struct Comment {
    /// The comment's text, starting with `//`.
    pub text: String,
    /// Where it is written.
    pub span: Span,
}

/// What stands on the lines before a part: one blank line (any run of blank lines
/// counts as one), or a comment alone on its line.
#[derive(Clone, Debug, PartialEq)]
pub enum Lead {
    /// One or more blank lines.
    Blank,
    /// A comment on a line of its own.
    Comment(Comment),
}

/// The comments of one part that takes a line or more: the lines before it, and the
/// comment at the end of its last line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trivia {
    /// Blank lines and comment lines before the part. Never two `Blank` in a row.
    pub leading: Vec<Lead>,
    /// The comment after the part on its last line.
    pub trailing: Option<Comment>,
}

/// One part of a sequence of lines (an item, a statement, a state member, ...).
#[derive(Clone, Debug, PartialEq)]
pub struct Line<T> {
    /// Its comments.
    pub trivia: Trivia,
    /// The part.
    pub node: T,
    /// Where it is written, without its comments.
    pub span: Span,
}

/// A sequence of lines between braces, or the items of a module.
#[derive(Clone, Debug, PartialEq)]
pub struct Lines<T> {
    /// The comment after the opening brace, on its line.
    pub open: Option<Comment>,
    /// The parts, in order.
    pub lines: Vec<Line<T>>,
    /// Comment lines (and blank lines between them) after the last part, before the
    /// closing brace or the end of file. Never ends with `Blank`.
    pub end: Vec<Lead>,
}

impl<T> Lines<T> {
    /// The parts without their comments.
    pub fn nodes(&self) -> impl Iterator<Item = &T> {
        self.lines.iter().map(|line| &line.node)
    }
}

/// One element of a bracketed, comma-separated list.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry<T> {
    /// Comment lines before the element.
    pub leading: Vec<Comment>,
    /// The element.
    pub value: T,
    /// The comment after the element (and its comma) on its line.
    pub trailing: Option<Comment>,
}

/// A bracketed, comma-separated list: parameters, operands, fields, list elements.
#[derive(Clone, Debug, PartialEq)]
pub struct List<T> {
    /// The comment after the opening bracket, on its line.
    pub open: Option<Comment>,
    /// The elements.
    pub entries: Vec<Entry<T>>,
    /// Comment lines after the last element, before the closing bracket.
    pub end: Vec<Comment>,
    /// Where it is written, brackets included.
    pub span: Span,
}

impl<T> List<T> {
    /// The elements without their comments.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.entries.iter().map(|entry| &entry.value)
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the list has no element.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// True when a comment stands anywhere in the list.
    pub fn has_comments(&self) -> bool {
        self.open.is_some()
            || !self.end.is_empty()
            || self
                .entries
                .iter()
                .any(|entry| !entry.leading.is_empty() || entry.trailing.is_some())
    }
}

/// A whole module: one `.kom` file. The crate's entry points work on a tree on the
/// stack its depth needs; cloning, comparing or dropping a deep tree on a caller's
/// own small stack is the caller's business (`crate::deep::on_deep_stack` gives one).
#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    /// Comment lines before `module`, and the comment on its line.
    pub trivia: Trivia,
    /// The module's name.
    pub name: Ident,
    /// The items, in order. `open` is always `None`.
    pub items: Lines<Item>,
    /// The whole text.
    pub span: Span,
}

/// An item of a module.
#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    /// `tile [R, C]`.
    Tile(TileDecl),
    /// `format f, g`.
    Format(FormatDecl),
    /// `requires { ... }`.
    Requires(Requires),
    /// `param NAME: T = value @bind(...)`.
    Param(Param),
    /// `message M { fields }`.
    Message(Message),
    /// `record R { fields }`.
    Record(Record),
    /// `func f(params) -> T { ... }`.
    Func(Func),
    /// `extern func f(params) -> T @impl("...") [fallback { ... }]`.
    Extern(Extern),
    /// `combine f(params) -> T @commutative { ... }`.
    Combine(Func),
    /// `program P { ... }`.
    Program(Program),
    /// `control c(params) [-> T] { ... }`.
    Control(Control),
}

/// `tile [R, C]`: the module's block of elements.
#[derive(Clone, Debug, PartialEq)]
pub struct TileDecl {
    /// Rows, as written.
    pub rows: IntLit,
    /// Columns, as written.
    pub cols: IntLit,
    /// Where it is written.
    pub span: Span,
}

/// An integer literal as written (decimal or `0x` hexadecimal).
#[derive(Clone, Debug, PartialEq)]
pub struct IntLit {
    /// The digits as written.
    pub text: String,
    /// Where it is written.
    pub span: Span,
}

/// `format f, g, ...`: the element formats the module uses.
#[derive(Clone, Debug, PartialEq)]
pub struct FormatDecl {
    /// The format names.
    pub formats: Vec<Ident>,
    /// Where it is written.
    pub span: Span,
}

/// `requires { ... }`: what the machine must be able to do.
#[derive(Clone, Debug, PartialEq)]
pub struct Requires {
    /// One requirement a line.
    pub body: Lines<Requirement>,
    /// Where it is written.
    pub span: Span,
}

/// One requirement.
#[derive(Clone, Debug, PartialEq)]
pub enum Requirement {
    /// `local_state >= tiles(n)` or `local_state >= bytes(n)`.
    LocalState {
        /// The unit.
        unit: StateUnit,
        /// The amount.
        amount: IntLit,
        /// Where it is written.
        span: Span,
    },
    /// `group = line` or `group = grid`.
    Group {
        /// The shape.
        shape: GroupShape,
        /// Where it is written.
        span: Span,
    },
    /// `dynamic`.
    Dynamic {
        /// Where it is written.
        span: Span,
    },
    /// `islands >= n`.
    Islands {
        /// The least number of islands.
        count: IntLit,
        /// Where it is written.
        span: Span,
    },
    /// `impl("name")`.
    Impl {
        /// The implementation's name.
        name: StrLit,
        /// Where it is written.
        span: Span,
    },
}

/// The unit of a `local_state` requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateUnit {
    /// `tiles(n)`.
    Tiles,
    /// `bytes(n)`.
    Bytes,
}

/// The shape of a `group` requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupShape {
    /// `line`.
    Line,
    /// `grid`.
    Grid,
}

/// A string literal: `text` is what stands between the quotes, escapes as written.
#[derive(Clone, Debug, PartialEq)]
pub struct StrLit {
    /// The characters between the quotes, as written.
    pub text: String,
    /// Where it is written, quotes included.
    pub span: Span,
}

/// `param NAME: T = value {attr}`.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    /// The name.
    pub name: Ident,
    /// The type.
    pub ty: Type,
    /// The value: an expression, or `fit`.
    pub value: ParamValue,
    /// The attributes after it (`@bind(...)`).
    pub attrs: Vec<Attr>,
    /// Where it is written.
    pub span: Span,
}

/// The value of a `param`.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    /// An expression.
    Expr(Expr),
    /// `fit`: the largest value whose declared state fits the worker memory.
    Fit(Span),
}

/// `message M {attr} { fields }`.
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    /// The message type's name.
    pub name: Ident,
    /// The attributes after the name.
    pub attrs: Vec<Attr>,
    /// The fields.
    pub fields: List<Field>,
    /// Where it is written.
    pub span: Span,
}

/// `record R { fields }`.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    /// The record's name.
    pub name: Ident,
    /// The fields.
    pub fields: List<Field>,
    /// Where it is written.
    pub span: Span,
}

/// `name: T`, a field of a message or a record.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    /// The field's name.
    pub name: Ident,
    /// Its type.
    pub ty: Type,
    /// Where it is written.
    pub span: Span,
}

/// `func` or `combine`: a pure function with a body.
#[derive(Clone, Debug, PartialEq)]
pub struct Func {
    /// The name.
    pub name: Ident,
    /// The parameters.
    pub params: List<ParamDecl>,
    /// The result type.
    pub ret: Type,
    /// The attributes before the body (a `func`'s: hints only, the checker refuses
    /// the language's own there).
    pub attrs: Vec<Attr>,
    /// The body.
    pub body: Block,
    /// Where it is written.
    pub span: Span,
}

/// `extern func`: a function the IR names and types, resolved by a backend.
#[derive(Clone, Debug, PartialEq)]
pub struct Extern {
    /// The name.
    pub name: Ident,
    /// The parameters.
    pub params: List<ParamDecl>,
    /// The result type.
    pub ret: Type,
    /// The attributes (`@reads`, `@writes`, `@impl`).
    pub attrs: Vec<Attr>,
    /// The `fallback { ... }` body, when written.
    pub fallback: Option<Block>,
    /// Where it is written.
    pub span: Span,
}

/// One parameter: `name: T {attr}`.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamDecl {
    /// The name.
    pub name: Ident,
    /// The type.
    pub ty: Type,
    /// The attributes after the type (`@replicated`).
    pub attrs: Vec<Attr>,
    /// Where it is written.
    pub span: Span,
}

/// `program P { ... }`: behaviour loaded together, sharing state.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    /// The program's name.
    pub name: Ident,
    /// Its parts, in order.
    pub body: Lines<ProgramItem>,
    /// Where it is written.
    pub span: Span,
}

/// A part of a program.
#[derive(Clone, Debug, PartialEq)]
pub enum ProgramItem {
    /// `message M { ... }`.
    Message(Message),
    /// `record R { ... }`.
    Record(Record),
    /// `state {attr} { ... }`.
    State(State),
    /// `handler h(...) { ... }`.
    Handler(Handler),
    /// `run r(...) @dynamic { seed { ... } on M(m) { ... } }`.
    Run(Run),
    /// `chain c(...) { call ... }`.
    Chain(Chain),
}

/// `state {attr} { member: T {attr} ... }`: what a worker keeps.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    /// The attributes of the block (`@scope`, `@lifetime`).
    pub attrs: Vec<Attr>,
    /// The members, one a line.
    pub members: Lines<StateMember>,
    /// Where it is written.
    pub span: Span,
}

/// `name: T {attr}`, a member of a state block.
#[derive(Clone, Debug, PartialEq)]
pub struct StateMember {
    /// The member's name.
    pub name: Ident,
    /// Its type.
    pub ty: Type,
    /// Its own attributes (`@export`, `@filled(element)`).
    pub attrs: Vec<Attr>,
    /// Where it is written.
    pub span: Span,
}

/// `handler h(params) [-> T] {attr} { ... }`.
#[derive(Clone, Debug, PartialEq)]
pub struct Handler {
    /// The handler's name.
    pub name: Ident,
    /// The parameters.
    pub params: List<ParamDecl>,
    /// The scalar result type, when written.
    pub ret: Option<Type>,
    /// The attributes (`@on(...)`, `@exclusive`, hints).
    pub attrs: Vec<Attr>,
    /// The body.
    pub body: Block,
    /// Where it is written.
    pub span: Span,
}

/// `run r(params) {attr} { seed { ... } on M(m) { ... } ... }`.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    /// The run's name.
    pub name: Ident,
    /// The parameters.
    pub params: List<ParamDecl>,
    /// The attributes (`@dynamic`, `@peers(...)`).
    pub attrs: Vec<Attr>,
    /// The `seed` block first, then the `on` blocks.
    pub body: Lines<RunPart>,
    /// Where it is written.
    pub span: Span,
}

/// A part of a run's body.
#[derive(Clone, Debug, PartialEq)]
pub enum RunPart {
    /// `seed { ... }`: once on every worker when the run starts.
    Seed(Block),
    /// `on M(m) { ... }`: once for each message of type M that arrives.
    On(OnBlock),
}

/// `on M(m) { ... }`.
#[derive(Clone, Debug, PartialEq)]
pub struct OnBlock {
    /// The message type.
    pub message: Ident,
    /// The name the body gives the message.
    pub binding: Ident,
    /// The body.
    pub body: Block,
    /// Where it is written.
    pub span: Span,
}

/// `chain c(params) {attr} { call h(args) ... }`.
#[derive(Clone, Debug, PartialEq)]
pub struct Chain {
    /// The chain's name.
    pub name: Ident,
    /// The parameters.
    pub params: List<ParamDecl>,
    /// The attributes (`@on(...)`).
    pub attrs: Vec<Attr>,
    /// The calls, in order.
    pub calls: Lines<Call>,
    /// Where it is written.
    pub span: Span,
}

/// `call h(args) {attr}`: one of the program's handlers, inside a chain.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    /// The handler called.
    pub handler: Ident,
    /// Its operands.
    pub args: List<Arg>,
    /// The attributes (`@on(group[a:b])`, R47).
    pub attrs: Vec<Attr>,
    /// Where it is written.
    pub span: Span,
}

/// `control c(params) [-> T] {attr} { ... }`: decides the next invocation from values.
#[derive(Clone, Debug, PartialEq)]
pub struct Control {
    /// The control program's name.
    pub name: Ident,
    /// The parameters.
    pub params: List<ParamDecl>,
    /// The scalar result type, when written.
    pub ret: Option<Type>,
    /// The attributes (hints such as `@place.control`).
    pub attrs: Vec<Attr>,
    /// The body.
    pub body: Block,
    /// Where it is written.
    pub span: Span,
}

/// `@name(args)` or `@backend.name(args)`: a property of what it follows. One name
/// is one of the language's own; a dotted name is a backend's hint.
#[derive(Clone, Debug, PartialEq)]
pub struct Attr {
    /// The name's parts: one for the language's own, more for a hint.
    pub path: Vec<Ident>,
    /// The operands, when the attribute has parentheses.
    pub args: Option<List<Arg>>,
    /// Where it is written.
    pub span: Span,
}

impl Attr {
    /// True for a dotted name, a backend's hint.
    pub fn is_hint(&self) -> bool {
        self.path.len() > 1
    }
}

/// A type.
#[derive(Clone, Debug, PartialEq)]
pub struct Type {
    /// What type.
    pub kind: TypeKind,
    /// `[n]` after a scalar or a tile type: n of them; of `tile<F>[r, c]`, c.
    pub count: Option<Box<Expr>>,
    /// Of `tile<F>[r, c]`, a block of r rows of c tiles (R45): r. `[n]` writes none.
    pub rows: Option<Box<Expr>>,
    /// Where it is written.
    pub span: Span,
}

/// The kinds of type.
#[derive(Clone, Debug, PartialEq)]
pub enum TypeKind {
    /// An integer or float scalar: `u8` to `u64`, `i8` to `i64`, `f16`, `f32`, `f64`.
    Scalar(Scalar),
    /// `bool`.
    Bool,
    /// `tile<F>`.
    Tile {
        /// The element format.
        format: Ident,
    },
    /// `tensor<F>`.
    Tensor {
        /// The element format.
        format: Ident,
    },
    /// `view`.
    View,
    /// A record or message type, by name.
    Named(Ident),
}

/// The scalar types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    /// `u8`.
    U8,
    /// `u16`.
    U16,
    /// `u32`.
    U32,
    /// `u64`.
    U64,
    /// `i8`.
    I8,
    /// `i16`.
    I16,
    /// `i32`.
    I32,
    /// `i64`.
    I64,
    /// `f16`.
    F16,
    /// `f32`.
    F32,
    /// `f64`.
    F64,
}

impl Scalar {
    /// Every scalar type, in the order of the enum.
    pub const ALL: [Scalar; 11] = [
        Scalar::U8,
        Scalar::U16,
        Scalar::U32,
        Scalar::U64,
        Scalar::I8,
        Scalar::I16,
        Scalar::I32,
        Scalar::I64,
        Scalar::F16,
        Scalar::F32,
        Scalar::F64,
    ];

    /// The type's name in the text.
    pub fn name(self) -> &'static str {
        match self {
            Scalar::U8 => "u8",
            Scalar::U16 => "u16",
            Scalar::U32 => "u32",
            Scalar::U64 => "u64",
            Scalar::I8 => "i8",
            Scalar::I16 => "i16",
            Scalar::I32 => "i32",
            Scalar::I64 => "i64",
            Scalar::F16 => "f16",
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
        }
    }

    /// The scalar type of a name, if it is one.
    pub fn from_name(name: &str) -> Option<Scalar> {
        Scalar::ALL.into_iter().find(|s| s.name() == name)
    }
}

/// `{ statements }`: the body of a function, handler, control program or statement.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    /// The statements, one a line.
    pub stmts: Lines<Stmt>,
    /// Where it is written, braces included.
    pub span: Span,
}

/// A statement.
#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    /// `let name [: T] = value {attr}`.
    Let {
        /// The name bound.
        name: Ident,
        /// The type, when written.
        ty: Option<Type>,
        /// The value.
        value: Expr,
        /// The attributes after the value.
        attrs: Vec<Attr>,
    },
    /// `var name: T = value`.
    Var {
        /// The name bound.
        name: Ident,
        /// The type.
        ty: Type,
        /// The first value.
        value: Expr,
    },
    /// `target = value`, `target += value`, `target -= value`.
    Assign {
        /// A name, with members and indexes after it.
        target: Expr,
        /// The operator.
        op: AssignOp,
        /// The value.
        value: Expr,
    },
    /// `for a, b in iter {attr} { ... }`.
    For {
        /// The names bound for each step.
        vars: Vec<Ident>,
        /// What is walked.
        iter: Expr,
        /// The attributes after it (hints).
        attrs: Vec<Attr>,
        /// The body.
        body: Block,
    },
    /// `while cond { ... }`.
    While {
        /// The condition.
        cond: Expr,
        /// The body.
        body: Block,
    },
    /// `if cond { ... } [else { ... }]`.
    If {
        /// The condition.
        cond: Expr,
        /// The body when true.
        then: Block,
        /// The body when false, when written.
        otherwise: Option<Block>,
    },
    /// `store view <- value`.
    Store {
        /// Where it writes.
        target: Expr,
        /// What it writes.
        value: Expr,
    },
    /// `consume m`.
    Consume(Ident),
    /// `return [value]`.
    Return(Option<Expr>),
    /// An expression alone, such as a call.
    Expr(Expr),
}

/// The operator of an assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignOp {
    /// `=`.
    Set,
    /// `+=`.
    Add,
    /// `-=`.
    Sub,
}

impl AssignOp {
    /// The operator's text.
    pub fn text(self) -> &'static str {
        match self {
            AssignOp::Set => "=",
            AssignOp::Add => "+=",
            AssignOp::Sub => "-=",
        }
    }
}

/// An expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    /// What expression.
    pub kind: ExprKind,
    /// Where it is written.
    pub span: Span,
}

/// The kinds of expression.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    /// An integer literal, as written.
    Int(String),
    /// A float literal, as written.
    Float(String),
    /// A string literal: the characters between the quotes, as written.
    Str(String),
    /// `true` or `false`.
    Bool(bool),
    /// A name.
    Name(Ident),
    /// `(e)`: kept, so the text keeps its parentheses.
    Paren(Box<Expr>),
    /// `(a, b, ...)`, two elements or more.
    Tuple(List<Expr>),
    /// `[a, b, ...]`.
    List(List<Expr>),
    /// `M { field: value, ... }`, or `{ field: value, ... }` without a type name.
    Init {
        /// The message type, when written.
        ty: Option<Ident>,
        /// The fields given.
        fields: List<FieldInit>,
    },
    /// `base.name`.
    Member {
        /// What it is a member of.
        base: Box<Expr>,
        /// The member.
        name: Ident,
    },
    /// `callee(args)`.
    Call {
        /// What is called.
        callee: Box<Expr>,
        /// The operands.
        args: List<Arg>,
    },
    /// `base[i, ...]`, each operand an index or a slice `lo:hi[:step]`.
    Index {
        /// What is indexed.
        base: Box<Expr>,
        /// The indexes.
        index: List<IndexItem>,
    },
    /// `-operand`.
    Neg(Box<Expr>),
    /// `lhs op rhs`.
    Binary {
        /// The operator.
        op: BinOp,
        /// The left operand.
        lhs: Box<Expr>,
        /// The right operand.
        rhs: Box<Expr>,
    },
    /// `await h`.
    Await(Ident),
    /// `past(y)`: y as an earlier call of the chain wrote it.
    Past(Ident),
}

/// `name: value` in a message or `with` initialiser.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldInit {
    /// The field.
    pub name: Ident,
    /// Its value.
    pub value: Expr,
    /// Where it is written.
    pub span: Span,
}

/// An operand: `value`, or `name = value` when given by name.
#[derive(Clone, Debug, PartialEq)]
pub struct Arg {
    /// The operand's name, when given by name.
    pub name: Option<Ident>,
    /// The value.
    pub value: Expr,
    /// Where it is written.
    pub span: Span,
}

/// One operand of an index: a value, or a slice.
#[derive(Clone, Debug, PartialEq)]
pub enum IndexItem {
    /// `i`.
    At(Expr),
    /// `lo:hi` or `lo:hi:step`.
    Slice {
        /// The first index.
        lo: Box<Expr>,
        /// One past the last index.
        hi: Box<Expr>,
        /// The step, when written.
        step: Option<Box<Expr>>,
    },
}

/// The binary operators, from the loosest binding to the tightest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// `||`.
    Or,
    /// `&&`.
    And,
    /// `==`.
    Eq,
    /// `!=`.
    Ne,
    /// `<`.
    Lt,
    /// `<=`.
    Le,
    /// `>`.
    Gt,
    /// `>=`.
    Ge,
    /// `|`.
    BitOr,
    /// `^`.
    BitXor,
    /// `&`.
    BitAnd,
    /// `<<`.
    Shl,
    /// `>>`.
    Shr,
    /// `+`.
    Add,
    /// `-`.
    Sub,
    /// `*`.
    Mul,
    /// `/`.
    Div,
    /// `%`.
    Rem,
}

impl BinOp {
    /// Every binary operator.
    pub const ALL: [BinOp; 18] = [
        BinOp::Or,
        BinOp::And,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::Lt,
        BinOp::Le,
        BinOp::Gt,
        BinOp::Ge,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::BitAnd,
        BinOp::Shl,
        BinOp::Shr,
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Rem,
    ];

    /// The operator's text.
    pub fn text(self) -> &'static str {
        match self {
            BinOp::Or => "||",
            BinOp::And => "&&",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::BitAnd => "&",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
        }
    }

    /// How tightly it binds: a larger number binds tighter. Every operator groups
    /// from the left.
    pub fn precedence(self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => 3,
            BinOp::BitOr => 4,
            BinOp::BitXor => 5,
            BinOp::BitAnd => 6,
            BinOp::Shl | BinOp::Shr => 7,
            BinOp::Add | BinOp::Sub => 8,
            BinOp::Mul | BinOp::Div | BinOp::Rem => 9,
        }
    }
}
