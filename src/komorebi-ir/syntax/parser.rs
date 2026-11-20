//! Text to syntax tree: the pest grammar, then a walk over its pairs that builds
//! [`ast`](super::ast) nodes with their positions and comments.

use std::cell::Cell;
use std::sync::OnceLock;

use pest::error::{ErrorVariant, InputLocation};
use pest::iterators::{Pair, Pairs};
use pest::pratt_parser::{Assoc, Op, PrattParser};
use pest::Parser;

use super::ast::*;
use crate::errors::{Diagnostic, STATE_NOT_BLOCK, SYNTAX, TOO_DEEP};

pub use grammar::{KomorebiParser, Rule};

/// The pest grammar; its `Rule` enum names the rules of `komorebi.pest`.
mod grammar {
    #![allow(missing_docs)]
    use pest_derive::Parser;

    /// The grammar of the text form, compiled by pest.
    #[derive(Parser)]
    #[grammar = "komorebi-ir/syntax/komorebi.pest"]
    pub struct KomorebiParser;
}

/// The deepest expression tree the parser reads (every operand, operator, sign,
/// member, call and bracket is one level), and the deepest nesting of blocks.
pub const MAX_DEPTH: usize = 256;

/// Parses a whole module. A text that does not follow the grammar gives one
/// diagnostic, at the first place the grammar cannot go on; this never panics. Its
/// walks recurse as deep as the tree: call it on the stack `crate::deep` gives, as
/// `crate::api::parse` does.
pub fn parse_module(source: &str) -> Result<Module, Vec<Diagnostic>> {
    if let Some(diagnostic) = check_limits(source) {
        return Err(vec![diagnostic]);
    }
    let mut pairs = KomorebiParser::parse(Rule::module, source)
        .map_err(|error| vec![from_pest(source, error)])?;
    let module = pairs.next().ok_or_else(|| {
        vec![Diagnostic::error(
            source,
            SYNTAX,
            "empty text".into(),
            Span::new(0, 0),
        )]
    })?;
    let builder = Builder {
        source,
        blocks: Cell::new(0),
    };
    builder.module(module).map_err(|d| vec![d])
}

/// The refusal of a text nested deeper than the parser reads.
fn too_deep(source: &str, span: Span) -> Diagnostic {
    let message = format!("nested deeper than {MAX_DEPTH}, deeper than this tool reads");
    Diagnostic::error(source, TOO_DEEP, message, span)
}

/// Refuses, before the grammar runs, a text whose brackets nest or whose signs run
/// deeper than any tree the parser accepts can be; the grammar's walk stays bounded.
fn check_limits(source: &str) -> Option<Diagnostic> {
    let bytes = source.as_bytes();
    let mut depth = 0usize;
    let mut minus_run = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            b'"' => {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' && bytes[i] != b'\n' {
                    i += if bytes[i] == b'\\' { 2 } else { 1 };
                }
            }
            b'(' | b'[' | b'{' => {
                depth += 1;
                if depth > 2 * MAX_DEPTH + 8 {
                    return Some(too_deep(source, Span::new(i, i + 1)));
                }
            }
            b')' | b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
        if c == b'-' {
            minus_run += 1;
            if minus_run > MAX_DEPTH {
                return Some(too_deep(source, Span::new(i, i + 1)));
            }
        } else if c != b' ' && c != b'\t' {
            minus_run = 0;
        }
        i += 1;
    }
    None
}

/// A pest error as a diagnostic: where the grammar stopped, and what it expected.
fn from_pest(source: &str, error: pest::error::Error<Rule>) -> Diagnostic {
    let (start, end) = match error.location {
        InputLocation::Pos(at) => (at, at),
        InputLocation::Span((start, end)) => (start, end),
    };
    let message = match &error.variant {
        ErrorVariant::ParsingError { positives, .. } => {
            let mut names: Vec<&str> = positives.iter().filter_map(|r| label(*r)).collect();
            names.sort_unstable();
            names.dedup();
            let found = match source[start..].chars().next() {
                None => "the end of the text".to_string(),
                Some('\n') | Some('\r') => "the end of the line".to_string(),
                Some(c) => format!("`{c}`"),
            };
            match names.len() {
                0 => format!("unexpected {found}"),
                1 => format!("expected {}, found {found}", names[0]),
                n => format!(
                    "expected {} or {}, found {found}",
                    names[..n - 1].join(", "),
                    names[n - 1]
                ),
            }
        }
        ErrorVariant::CustomError { message } => message.clone(),
    };
    Diagnostic::error(source, SYNTAX, message, Span::new(start, end))
}

/// The words a diagnostic uses for what the grammar expected.
fn label(rule: Rule) -> Option<&'static str> {
    Some(match rule {
        Rule::EOI => "the end of the text",
        Rule::ident | Rule::arg_name | Rule::named_type => "a name",
        Rule::keyword => "a keyword",
        Rule::comment | Rule::gap => "a line break",
        Rule::ogap => return None,
        Rule::int_lit => "an integer",
        Rule::float_lit => "a number",
        Rule::string_lit => "a string",
        Rule::true_lit | Rule::false_lit => "`true` or `false`",
        Rule::module | Rule::kw_module => "`module`",
        Rule::item => "an item",
        Rule::tile_decl | Rule::kw_tile => "`tile`",
        Rule::format_decl | Rule::kw_format => "`format`",
        Rule::requires_decl | Rule::kw_requires => "`requires`",
        Rule::requirement => "a requirement",
        Rule::req_local => "`local_state`",
        Rule::req_unit => "`tiles` or `bytes`",
        Rule::req_group => "`group`",
        Rule::req_shape => "`line` or `grid`",
        Rule::req_dynamic => "`dynamic`",
        Rule::req_islands => "`islands`",
        Rule::req_impl => "`impl`",
        Rule::param_decl | Rule::kw_param => "`param`",
        Rule::fit | Rule::kw_fit => "`fit`",
        Rule::message_decl | Rule::kw_message => "`message`",
        Rule::record_decl | Rule::kw_record => "`record`",
        Rule::fields | Rule::inits | Rule::anon_init => "`{`",
        Rule::field | Rule::init => "a field",
        Rule::func_decl | Rule::kw_func => "`func`",
        Rule::combine_decl | Rule::kw_combine => "`combine`",
        Rule::extern_decl | Rule::kw_extern => "`extern`",
        Rule::fallback | Rule::kw_fallback => "`fallback`",
        Rule::params | Rule::args | Rule::call_args => "`(`",
        Rule::one_param => "a parameter",
        Rule::program_decl | Rule::kw_program => "`program`",
        Rule::program_item => "a part of a program",
        Rule::state_decl | Rule::state_single | Rule::kw_state => "`state`",
        Rule::state_member => "a state member",
        Rule::handler_decl | Rule::kw_handler => "`handler`",
        Rule::run_decl | Rule::kw_run => "`run`",
        Rule::seed_block | Rule::kw_seed => "`seed`",
        Rule::on_block | Rule::kw_on => "`on`",
        Rule::chain_decl | Rule::kw_chain => "`chain`",
        Rule::call_stmt | Rule::kw_call => "`call`",
        Rule::control_decl | Rule::kw_control => "`control`",
        Rule::attr | Rule::attr_name => "an attribute",
        Rule::type_
        | Rule::tile_type
        | Rule::tensor_type
        | Rule::view_type
        | Rule::bool_type
        | Rule::scalar_type => "a type",
        Rule::array | Rule::shape | Rule::index | Rule::list_lit => "`[`",
        Rule::block => "`{`",
        Rule::stmt
        | Rule::let_stmt
        | Rule::var_stmt
        | Rule::for_stmt
        | Rule::while_stmt
        | Rule::if_stmt
        | Rule::store_stmt
        | Rule::consume_stmt
        | Rule::return_stmt
        | Rule::assign_stmt
        | Rule::expr_stmt => "a statement",
        Rule::kw_let => "`let`",
        Rule::kw_var => "`var`",
        Rule::kw_if => "`if`",
        Rule::kw_else => "`else`",
        Rule::kw_for => "`for`",
        Rule::kw_in => "`in`",
        Rule::kw_while => "`while`",
        Rule::kw_return => "`return`",
        Rule::kw_store => "`store`",
        Rule::kw_consume => "`consume`",
        Rule::kw_await | Rule::await_expr => "`await`",
        Rule::kw_past | Rule::past_expr => "`past`",
        Rule::lvalue => "a name",
        Rule::assign_op => "`=`, `+=` or `-=`",
        Rule::expr | Rule::postfix_expr | Rule::primary | Rule::struct_init => "an expression",
        Rule::neg => "`-`",
        Rule::paren_or_tuple => "`(`",
        Rule::member => "`.`",
        Rule::index_item | Rule::slice => "an index",
        Rule::arg => "an operand",
        Rule::or_op
        | Rule::and_op
        | Rule::eq_op
        | Rule::ne_op
        | Rule::le_op
        | Rule::ge_op
        | Rule::shl_op
        | Rule::shr_op
        | Rule::lt_op
        | Rule::gt_op
        | Rule::bor_op
        | Rule::bxor_op
        | Rule::band_op
        | Rule::add_op
        | Rule::sub_op
        | Rule::mul_op
        | Rule::div_op
        | Rule::rem_op => "an operator",
        _ => return None,
    })
}

fn pratt() -> &'static PrattParser<Rule> {
    static PRATT: OnceLock<PrattParser<Rule>> = OnceLock::new();
    PRATT.get_or_init(|| {
        PrattParser::new()
            .op(Op::infix(Rule::or_op, Assoc::Left))
            .op(Op::infix(Rule::and_op, Assoc::Left))
            .op(Op::infix(Rule::eq_op, Assoc::Left)
                | Op::infix(Rule::ne_op, Assoc::Left)
                | Op::infix(Rule::lt_op, Assoc::Left)
                | Op::infix(Rule::le_op, Assoc::Left)
                | Op::infix(Rule::gt_op, Assoc::Left)
                | Op::infix(Rule::ge_op, Assoc::Left))
            .op(Op::infix(Rule::bor_op, Assoc::Left))
            .op(Op::infix(Rule::bxor_op, Assoc::Left))
            .op(Op::infix(Rule::band_op, Assoc::Left))
            .op(Op::infix(Rule::shl_op, Assoc::Left) | Op::infix(Rule::shr_op, Assoc::Left))
            .op(Op::infix(Rule::add_op, Assoc::Left) | Op::infix(Rule::sub_op, Assoc::Left))
            .op(Op::infix(Rule::mul_op, Assoc::Left)
                | Op::infix(Rule::div_op, Assoc::Left)
                | Op::infix(Rule::rem_op, Assoc::Left))
            .op(Op::prefix(Rule::neg))
    })
}

fn bin_op(rule: Rule) -> BinOp {
    match rule {
        Rule::or_op => BinOp::Or,
        Rule::and_op => BinOp::And,
        Rule::eq_op => BinOp::Eq,
        Rule::ne_op => BinOp::Ne,
        Rule::lt_op => BinOp::Lt,
        Rule::le_op => BinOp::Le,
        Rule::gt_op => BinOp::Gt,
        Rule::ge_op => BinOp::Ge,
        Rule::bor_op => BinOp::BitOr,
        Rule::bxor_op => BinOp::BitXor,
        Rule::band_op => BinOp::BitAnd,
        Rule::shl_op => BinOp::Shl,
        Rule::shr_op => BinOp::Shr,
        Rule::add_op => BinOp::Add,
        Rule::sub_op => BinOp::Sub,
        Rule::mul_op => BinOp::Mul,
        Rule::div_op => BinOp::Div,
        _ => BinOp::Rem,
    }
}

fn is_keyword(rule: Rule) -> bool {
    matches!(
        rule,
        Rule::kw_module
            | Rule::kw_tile
            | Rule::kw_format
            | Rule::kw_requires
            | Rule::kw_param
            | Rule::kw_message
            | Rule::kw_record
            | Rule::kw_func
            | Rule::kw_extern
            | Rule::kw_combine
            | Rule::kw_fallback
            | Rule::kw_program
            | Rule::kw_control
            | Rule::kw_state
            | Rule::kw_handler
            | Rule::kw_run
            | Rule::kw_seed
            | Rule::kw_on
            | Rule::kw_chain
            | Rule::kw_call
            | Rule::kw_let
            | Rule::kw_var
            | Rule::kw_if
            | Rule::kw_else
            | Rule::kw_for
            | Rule::kw_in
            | Rule::kw_while
            | Rule::kw_return
            | Rule::kw_store
            | Rule::kw_consume
            | Rule::kw_await
            | Rule::kw_past
            | Rule::kw_fit
    )
}

/// The children of a pair, keywords left out.
fn kids(pair: Pair<'_, Rule>) -> Vec<Pair<'_, Rule>> {
    pair.into_inner()
        .filter(|p| !is_keyword(p.as_rule()))
        .collect()
}

fn span_of(pair: &Pair<'_, Rule>) -> Span {
    let span = pair.as_span();
    Span::new(span.start(), span.end())
}

/// What a gap between two parts holds: the comment that ends the line before it, and
/// the blank lines and comment lines after that line.
struct GapText {
    trailing: Option<Comment>,
    leads: Vec<Lead>,
}

/// Reads a `gap` or `ogap`. `line_start` is true when the gap starts a line (the top
/// of the file), so that its first line is a line of its own.
fn read_gap(pair: &Pair<'_, Rule>, line_start: bool) -> GapText {
    let text = pair.as_str();
    let base = pair.as_span().start();
    let mut out = GapText {
        trailing: None,
        leads: Vec::new(),
    };
    let mut offset = 0;
    let pieces: Vec<&str> = text.split('\n').collect();
    let last = pieces.len() - 1;
    for (k, piece) in pieces.iter().enumerate() {
        let at = base + offset;
        offset += piece.len() + 1;
        if k == last {
            break;
        }
        let comment = piece.find("//").map(|i| {
            let body = piece[i..].trim_end();
            Comment {
                text: body.to_string(),
                span: Span::new(at + i, at + i + body.len()),
            }
        });
        if k == 0 && !line_start {
            out.trailing = comment;
            continue;
        }
        match comment {
            Some(comment) => out.leads.push(Lead::Comment(comment)),
            None => {
                if out.leads.last() != Some(&Lead::Blank) {
                    out.leads.push(Lead::Blank);
                }
            }
        }
    }
    out
}

/// Collects comments and parts into a [`Lines`].
struct LinesBuilder<T> {
    lines: Lines<T>,
    pending: Vec<Lead>,
    keep_first_blank: bool,
}

impl<T> LinesBuilder<T> {
    fn new(keep_first_blank: bool) -> LinesBuilder<T> {
        LinesBuilder {
            lines: Lines {
                open: None,
                lines: Vec::new(),
                end: Vec::new(),
            },
            pending: Vec::new(),
            keep_first_blank,
        }
    }

    fn gap(&mut self, gap: GapText) {
        if let Some(comment) = gap.trailing {
            match self.lines.lines.last_mut() {
                Some(line) if line.trivia.trailing.is_none() && self.pending.is_empty() => {
                    line.trivia.trailing = Some(comment)
                }
                None if self.lines.open.is_none() && self.pending.is_empty() => {
                    self.lines.open = Some(comment)
                }
                _ => self.pending.push(Lead::Comment(comment)),
            }
        }
        for lead in gap.leads {
            if lead == Lead::Blank && self.pending.last() == Some(&Lead::Blank) {
                continue;
            }
            self.pending.push(lead);
        }
    }

    fn push(&mut self, node: T, span: Span) {
        let mut leading = std::mem::take(&mut self.pending);
        if self.lines.lines.is_empty() && !self.keep_first_blank {
            while leading.first() == Some(&Lead::Blank) {
                leading.remove(0);
            }
        }
        self.lines.lines.push(Line {
            trivia: Trivia {
                leading,
                trailing: None,
            },
            node,
            span,
        });
    }

    fn finish(mut self) -> Lines<T> {
        let mut end = std::mem::take(&mut self.pending);
        while end.last() == Some(&Lead::Blank) {
            end.pop();
        }
        if self.lines.lines.is_empty() && !self.keep_first_blank {
            while end.first() == Some(&Lead::Blank) {
                end.remove(0);
            }
        }
        self.lines.end = end;
        self.lines
    }
}

/// Collects comments and elements into a [`List`].
struct ListBuilder<T> {
    list: List<T>,
    pending: Vec<Comment>,
}

impl<T> ListBuilder<T> {
    fn new(span: Span) -> ListBuilder<T> {
        ListBuilder {
            list: List {
                open: None,
                entries: Vec::new(),
                end: Vec::new(),
                span,
            },
            pending: Vec::new(),
        }
    }

    fn gap(&mut self, gap: GapText) {
        if let Some(comment) = gap.trailing {
            match self.list.entries.last_mut() {
                Some(entry) if entry.trailing.is_none() && self.pending.is_empty() => {
                    entry.trailing = Some(comment)
                }
                None if self.list.open.is_none() && self.pending.is_empty() => {
                    self.list.open = Some(comment)
                }
                _ => self.pending.push(comment),
            }
        }
        for lead in gap.leads {
            if let Lead::Comment(comment) = lead {
                self.pending.push(comment);
            }
        }
    }

    fn push(&mut self, value: T) {
        self.list.entries.push(Entry {
            leading: std::mem::take(&mut self.pending),
            value,
            trailing: None,
        });
    }

    fn finish(mut self) -> List<T> {
        self.list.end = std::mem::take(&mut self.pending);
        self.list
    }
}

type Built<T> = Result<T, Diagnostic>;

struct Builder<'s> {
    source: &'s str,
    /// How many blocks enclose the part being built.
    blocks: Cell<usize>,
}

impl<'s> Builder<'s> {
    fn fail(&self, span: Span, message: impl Into<String>) -> Diagnostic {
        Diagnostic::error(self.source, SYNTAX, message.into(), span)
    }

    fn ident(&self, pair: Pair<'_, Rule>) -> Ident {
        Ident {
            name: pair.as_str().to_string(),
            span: span_of(&pair),
        }
    }

    fn int(&self, pair: Pair<'_, Rule>) -> IntLit {
        IntLit {
            text: pair.as_str().to_string(),
            span: span_of(&pair),
        }
    }

    fn string(&self, pair: Pair<'_, Rule>) -> StrLit {
        let text = pair.as_str();
        StrLit {
            text: text[1..text.len() - 1].to_string(),
            span: span_of(&pair),
        }
    }

    fn module(&self, pair: Pair<'_, Rule>) -> Built<Module> {
        let span = span_of(&pair);
        let mut trivia = Trivia::default();
        let mut name = None;
        let mut items = LinesBuilder::new(true);
        for p in kids(pair) {
            match p.as_rule() {
                Rule::ogap | Rule::gap if name.is_none() => {
                    let gap = read_gap(&p, true);
                    trivia.leading = gap.leads;
                    while trivia.leading.first() == Some(&Lead::Blank) {
                        trivia.leading.remove(0);
                    }
                }
                Rule::ogap | Rule::gap => {
                    let mut gap = read_gap(&p, false);
                    if items.lines.lines.is_empty() && trivia.trailing.is_none() {
                        trivia.trailing = gap.trailing.take();
                    }
                    items.gap(gap);
                }
                Rule::ident => name = Some(self.ident(p)),
                Rule::item => {
                    let span = span_of(&p);
                    let item = self.item(p)?;
                    items.push(item, span);
                }
                _ => {}
            }
        }
        let name = name.ok_or_else(|| self.fail(span, "a module needs a name"))?;
        Ok(Module {
            trivia,
            name,
            items: items.finish(),
            span,
        })
    }

    fn item(&self, pair: Pair<'_, Rule>) -> Built<Item> {
        let p = only(pair);
        Ok(match p.as_rule() {
            Rule::tile_decl => {
                let span = span_of(&p);
                let mut ints = kids(p).into_iter();
                let rows = self.int(next(&mut ints));
                let cols = self.int(next(&mut ints));
                Item::Tile(TileDecl { rows, cols, span })
            }
            Rule::format_decl => {
                let span = span_of(&p);
                let formats = kids(p).into_iter().map(|q| self.ident(q)).collect();
                Item::Format(FormatDecl { formats, span })
            }
            Rule::requires_decl => {
                let span = span_of(&p);
                let body = self.lines(p, |b, q| b.requirement(q))?;
                Item::Requires(Requires { body, span })
            }
            Rule::param_decl => Item::Param(self.param(p)?),
            Rule::message_decl => Item::Message(self.message(p)?),
            Rule::record_decl => Item::Record(self.record(p)?),
            Rule::func_decl | Rule::combine_decl => {
                let combine = p.as_rule() == Rule::combine_decl;
                let func = self.func(p)?;
                if combine {
                    Item::Combine(func)
                } else {
                    Item::Func(func)
                }
            }
            Rule::extern_decl => Item::Extern(self.extern_func(p)?),
            Rule::program_decl => Item::Program(self.program(p)?),
            _ => Item::Control(self.control(p)?),
        })
    }

    /// Builds a [`Lines`] from the children of `pair`: gaps and parts, the parts made
    /// by `part`; other children are left out.
    fn lines<T>(
        &self,
        pair: Pair<'_, Rule>,
        part: impl Fn(&Self, Pair<'_, Rule>) -> Built<T>,
    ) -> Built<Lines<T>> {
        let mut builder = LinesBuilder::new(false);
        for p in kids(pair) {
            match p.as_rule() {
                Rule::gap | Rule::ogap => builder.gap(read_gap(&p, false)),
                Rule::ident | Rule::attr | Rule::params => {}
                _ => {
                    let span = span_of(&p);
                    let node = part(self, p)?;
                    builder.push(node, span);
                }
            }
        }
        Ok(builder.finish())
    }

    /// Builds a [`List`] from a bracketed pair: gaps and elements made by `element`.
    fn list<T>(
        &self,
        pair: Pair<'_, Rule>,
        element: impl Fn(&Self, Pair<'_, Rule>) -> Built<T>,
    ) -> Built<List<T>> {
        let mut builder = ListBuilder::new(span_of(&pair));
        for p in kids(pair) {
            match p.as_rule() {
                Rule::gap | Rule::ogap => builder.gap(read_gap(&p, false)),
                _ => {
                    let value = element(self, p)?;
                    builder.push(value);
                }
            }
        }
        Ok(builder.finish())
    }

    fn requirement(&self, pair: Pair<'_, Rule>) -> Built<Requirement> {
        let p = only(pair);
        let span = span_of(&p);
        Ok(match p.as_rule() {
            Rule::req_local => {
                let mut it = kids(p).into_iter();
                let unit = match next(&mut it).as_str() {
                    "tiles" => StateUnit::Tiles,
                    _ => StateUnit::Bytes,
                };
                let amount = self.int(next(&mut it));
                Requirement::LocalState { unit, amount, span }
            }
            Rule::req_group => {
                let shape = match next(&mut kids(p).into_iter()).as_str() {
                    "line" => GroupShape::Line,
                    _ => GroupShape::Grid,
                };
                Requirement::Group { shape, span }
            }
            Rule::req_dynamic => Requirement::Dynamic { span },
            Rule::req_islands => Requirement::Islands {
                count: self.int(next(&mut kids(p).into_iter())),
                span,
            },
            _ => Requirement::Impl {
                name: self.string(next(&mut kids(p).into_iter())),
                span,
            },
        })
    }

    fn attrs(&self, pairs: &mut Vec<Pair<'_, Rule>>) -> Built<Vec<Attr>> {
        let mut out = Vec::new();
        for p in pairs.drain(..) {
            out.push(self.attr(p)?);
        }
        Ok(out)
    }

    fn attr(&self, pair: Pair<'_, Rule>) -> Built<Attr> {
        let span = span_of(&pair);
        let mut it = kids(pair).into_iter();
        let name = next(&mut it);
        let base = name.as_span().start() + 1;
        let mut path = Vec::new();
        let mut at = base;
        for part in name.as_str()[1..].split('.') {
            path.push(Ident {
                name: part.to_string(),
                span: Span::new(at, at + part.len()),
            });
            at += part.len() + 1;
        }
        let args = match it.next() {
            Some(args) => Some(self.args(args)?),
            None => None,
        };
        Ok(Attr { path, args, span })
    }

    fn param(&self, pair: Pair<'_, Rule>) -> Built<Param> {
        let span = span_of(&pair);
        let mut it = kids(pair).into_iter();
        let name = self.ident(next(&mut it));
        let ty = self.ty(next(&mut it))?;
        let v = next(&mut it);
        let value = match v.as_rule() {
            Rule::fit => ParamValue::Fit(span_of(&v)),
            _ => ParamValue::Expr(self.expr(v)?),
        };
        let mut rest: Vec<_> = it.collect();
        let attrs = self.attrs(&mut rest)?;
        Ok(Param {
            name,
            ty,
            value,
            attrs,
            span,
        })
    }

    fn fields(&self, pair: Pair<'_, Rule>) -> Built<List<Field>> {
        self.list(pair, |b, p| {
            let span = span_of(&p);
            let mut it = kids(p).into_iter();
            let name = b.ident(next(&mut it));
            let ty = b.ty(next(&mut it))?;
            Ok(Field { name, ty, span })
        })
    }

    fn message(&self, pair: Pair<'_, Rule>) -> Built<Message> {
        let span = span_of(&pair);
        let mut it = kids(pair).into_iter();
        let name = self.ident(next(&mut it));
        let mut attrs = Vec::new();
        let mut fields = None;
        for p in it {
            match p.as_rule() {
                Rule::attr => attrs.push(self.attr(p)?),
                _ => fields = Some(self.fields(p)?),
            }
        }
        let fields = fields.ok_or_else(|| self.fail(span, "a message needs fields"))?;
        Ok(Message {
            name,
            attrs,
            fields,
            span,
        })
    }

    fn record(&self, pair: Pair<'_, Rule>) -> Built<Record> {
        let span = span_of(&pair);
        let mut it = kids(pair).into_iter();
        let name = self.ident(next(&mut it));
        let fields = self.fields(next(&mut it))?;
        Ok(Record { name, fields, span })
    }

    fn params(&self, pair: Pair<'_, Rule>) -> Built<List<ParamDecl>> {
        self.list(pair, |b, p| {
            let span = span_of(&p);
            let mut it = kids(p).into_iter();
            let name = b.ident(next(&mut it));
            let ty = b.ty(next(&mut it))?;
            let mut rest: Vec<_> = it.collect();
            let attrs = b.attrs(&mut rest)?;
            Ok(ParamDecl {
                name,
                ty,
                attrs,
                span,
            })
        })
    }

    /// The parts shared by functions, handlers and control programs: the name, the
    /// parameters, an optional result type, the attributes and the remaining pairs.
    fn head<'p>(&self, pair: Pair<'p, Rule>) -> Built<Head<'p>> {
        let mut it = kids(pair).into_iter();
        let name = self.ident(next(&mut it));
        let params = self.params(next(&mut it))?;
        let mut head = Head {
            name,
            params,
            ret: None,
            attrs: Vec::new(),
            rest: Vec::new(),
        };
        for p in it {
            match p.as_rule() {
                Rule::type_ => head.ret = Some(self.ty(p)?),
                Rule::scalar_type => head.ret = Some(self.scalar_type(p, None)),
                Rule::attr => head.attrs.push(self.attr(p)?),
                _ => head.rest.push(p),
            }
        }
        Ok(head)
    }

    fn func(&self, pair: Pair<'_, Rule>) -> Built<Func> {
        let span = span_of(&pair);
        let mut head = self.head(pair)?;
        let body = self.block(head.rest.remove(0))?;
        let ret = head
            .ret
            .ok_or_else(|| self.fail(span, "a function needs a result type"))?;
        Ok(Func {
            name: head.name,
            params: head.params,
            ret,
            attrs: head.attrs,
            body,
            span,
        })
    }

    fn extern_func(&self, pair: Pair<'_, Rule>) -> Built<Extern> {
        let span = span_of(&pair);
        let head = self.head(pair)?;
        let fallback = match head.rest.into_iter().next() {
            Some(p) => Some(self.block(next(&mut kids(p).into_iter()))?),
            None => None,
        };
        let ret = head
            .ret
            .ok_or_else(|| self.fail(span, "a function needs a result type"))?;
        Ok(Extern {
            name: head.name,
            params: head.params,
            ret,
            attrs: head.attrs,
            fallback,
            span,
        })
    }

    fn control(&self, pair: Pair<'_, Rule>) -> Built<Control> {
        let span = span_of(&pair);
        let mut head = self.head(pair)?;
        let body = self.block(head.rest.remove(0))?;
        Ok(Control {
            name: head.name,
            params: head.params,
            ret: head.ret,
            attrs: head.attrs,
            body,
            span,
        })
    }

    fn program(&self, pair: Pair<'_, Rule>) -> Built<Program> {
        let span = span_of(&pair);
        let name = self.ident(next(&mut kids(pair.clone()).into_iter()));
        let body = self.lines(pair, |b, p| b.program_item(p))?;
        Ok(Program { name, body, span })
    }

    fn program_item(&self, pair: Pair<'_, Rule>) -> Built<ProgramItem> {
        let p = only(pair);
        Ok(match p.as_rule() {
            Rule::message_decl => ProgramItem::Message(self.message(p)?),
            Rule::record_decl => ProgramItem::Record(self.record(p)?),
            Rule::state_decl => {
                let span = span_of(&p);
                let mut attrs = Vec::new();
                for q in kids(p.clone()) {
                    if q.as_rule() == Rule::attr {
                        attrs.push(self.attr(q)?);
                    }
                }
                let members = self.lines(p, |b, q| {
                    let span = span_of(&q);
                    let mut it = kids(q).into_iter();
                    let name = b.ident(next(&mut it));
                    let ty = b.ty(next(&mut it))?;
                    let mut rest: Vec<_> = it.collect();
                    let attrs = b.attrs(&mut rest)?;
                    Ok(StateMember {
                        name,
                        ty,
                        attrs,
                        span,
                    })
                })?;
                ProgramItem::State(State {
                    attrs,
                    members,
                    span,
                })
            }
            Rule::state_single => {
                let message = "a state is always a block: `state @attrs { member: T }`";
                return Err(Diagnostic::error(
                    self.source,
                    STATE_NOT_BLOCK,
                    message.into(),
                    span_of(&p),
                ));
            }
            Rule::handler_decl => {
                let span = span_of(&p);
                let mut head = self.head(p)?;
                let body = self.block(head.rest.remove(0))?;
                ProgramItem::Handler(Handler {
                    name: head.name,
                    params: head.params,
                    ret: head.ret,
                    attrs: head.attrs,
                    body,
                    span,
                })
            }
            Rule::run_decl => {
                let span = span_of(&p);
                let head = self.head(p.clone())?;
                let body = self.lines(p, |b, q| b.run_part(q))?;
                ProgramItem::Run(Run {
                    name: head.name,
                    params: head.params,
                    attrs: head.attrs,
                    body,
                    span,
                })
            }
            _ => {
                let span = span_of(&p);
                let head = self.head(p.clone())?;
                let calls = self.lines(p, |b, q| {
                    // to its last part: the spaces `attr*` passes over are not the call's
                    let end = q
                        .clone()
                        .into_inner()
                        .next_back()
                        .map(|k| k.as_span().end());
                    let span = Span::new(span_of(&q).start, end.unwrap_or(span_of(&q).end));
                    let mut it = kids(q).into_iter();
                    let handler = b.ident(next(&mut it));
                    let args = b.args(next(&mut it))?;
                    let attrs = it.map(|a| b.attr(a)).collect::<Built<Vec<_>>>()?;
                    Ok(Call {
                        handler,
                        args,
                        attrs,
                        span,
                    })
                })?;
                ProgramItem::Chain(Chain {
                    name: head.name,
                    params: head.params,
                    attrs: head.attrs,
                    calls,
                    span,
                })
            }
        })
    }

    fn run_part(&self, pair: Pair<'_, Rule>) -> Built<RunPart> {
        let span = span_of(&pair);
        let rule = pair.as_rule();
        let mut it = kids(pair).into_iter();
        if rule == Rule::seed_block {
            return Ok(RunPart::Seed(self.block(next(&mut it))?));
        }
        let message = self.ident(next(&mut it));
        let binding = self.ident(next(&mut it));
        let body = self.block(next(&mut it))?;
        Ok(RunPart::On(OnBlock {
            message,
            binding,
            body,
            span,
        }))
    }

    fn scalar_type(&self, pair: Pair<'_, Rule>, count: Option<Box<Expr>>) -> Type {
        let span = span_of(&pair);
        let scalar = Scalar::from_name(pair.as_str()).unwrap_or(Scalar::U32);
        Type {
            kind: TypeKind::Scalar(scalar),
            count,
            rows: None,
            span,
        }
    }

    fn ty(&self, pair: Pair<'_, Rule>) -> Built<Type> {
        let span = span_of(&pair);
        let mut it = kids(pair).into_iter();
        let p = next(&mut it);
        let count = match it.next() {
            Some(array) => Some(Box::new(self.expr(next(&mut kids(array).into_iter()))?)),
            None => None,
        };
        let kind = match p.as_rule() {
            Rule::scalar_type => {
                let mut ty = self.scalar_type(p, count);
                ty.span = span;
                return Ok(ty);
            }
            Rule::bool_type => TypeKind::Bool,
            Rule::view_type => TypeKind::View,
            Rule::named_type => TypeKind::Named(self.ident(only(p))),
            Rule::tensor_type => TypeKind::Tensor {
                format: self.ident(only(p)),
            },
            _ => {
                let mut inner = kids(p).into_iter();
                let format = self.ident(next(&mut inner));
                let (rows, count) = match inner.next() {
                    Some(shape) if shape.as_rule() == Rule::shape => {
                        let mut ends = kids(shape).into_iter();
                        let rows = self.expr(next(&mut ends))?;
                        let cols = self.expr(next(&mut ends))?;
                        (Some(Box::new(rows)), Some(Box::new(cols)))
                    }
                    Some(array) => (None, Some(Box::new(self.expr(only(array))?))),
                    None => (None, None),
                };
                return Ok(Type {
                    kind: TypeKind::Tile { format },
                    count,
                    rows,
                    span,
                });
            }
        };
        Ok(Type {
            kind,
            count,
            rows: None,
            span,
        })
    }

    fn block(&self, pair: Pair<'_, Rule>) -> Built<Block> {
        let span = span_of(&pair);
        if self.blocks.get() >= MAX_DEPTH {
            return Err(too_deep(self.source, span));
        }
        self.blocks.set(self.blocks.get() + 1);
        let stmts = self.lines(pair, |b, p| b.stmt(p));
        self.blocks.set(self.blocks.get() - 1);
        Ok(Block {
            stmts: stmts?,
            span,
        })
    }

    fn stmt(&self, pair: Pair<'_, Rule>) -> Built<Stmt> {
        let p = only(pair);
        let rule = p.as_rule();
        let mut it = kids(p).into_iter();
        Ok(match rule {
            Rule::let_stmt => {
                let name = self.ident(next(&mut it));
                let mut q = next(&mut it);
                let ty = if q.as_rule() == Rule::type_ {
                    let ty = self.ty(q)?;
                    q = next(&mut it);
                    Some(ty)
                } else {
                    None
                };
                let value = self.expr(q)?;
                let mut rest: Vec<_> = it.collect();
                let attrs = self.attrs(&mut rest)?;
                Stmt::Let {
                    name,
                    ty,
                    value,
                    attrs,
                }
            }
            Rule::var_stmt => Stmt::Var {
                name: self.ident(next(&mut it)),
                ty: self.ty(next(&mut it))?,
                value: self.expr(next(&mut it))?,
            },
            Rule::for_stmt => {
                let mut vars = Vec::new();
                let mut iter = None;
                let mut attrs = Vec::new();
                let mut body = None;
                for q in it {
                    match q.as_rule() {
                        Rule::ident => vars.push(self.ident(q)),
                        Rule::expr => iter = Some(self.expr(q)?),
                        Rule::attr => attrs.push(self.attr(q)?),
                        _ => body = Some(self.block(q)?),
                    }
                }
                let (Some(iter), Some(body)) = (iter, body) else {
                    return Err(self.fail(Span::default(), "a `for` needs a stream and a body"));
                };
                Stmt::For {
                    vars,
                    iter,
                    attrs,
                    body,
                }
            }
            Rule::while_stmt => Stmt::While {
                cond: self.expr(next(&mut it))?,
                body: self.block(next(&mut it))?,
            },
            Rule::if_stmt => {
                let cond = self.expr(next(&mut it))?;
                let then = self.block(next(&mut it))?;
                let otherwise = match it.next() {
                    Some(q) => Some(self.block(q)?),
                    None => None,
                };
                Stmt::If {
                    cond,
                    then,
                    otherwise,
                }
            }
            Rule::store_stmt => Stmt::Store {
                target: self.expr(next(&mut it))?,
                value: self.expr(next(&mut it))?,
            },
            Rule::consume_stmt => Stmt::Consume(self.ident(next(&mut it))),
            Rule::return_stmt => Stmt::Return(match it.next() {
                Some(q) => Some(self.expr(q)?),
                None => None,
            }),
            Rule::assign_stmt => {
                let target = self.lvalue(next(&mut it))?;
                let op = match next(&mut it).as_str() {
                    "+=" => AssignOp::Add,
                    "-=" => AssignOp::Sub,
                    _ => AssignOp::Set,
                };
                let value = self.expr(next(&mut it))?;
                Stmt::Assign { target, op, value }
            }
            _ => Stmt::Expr(self.expr(next(&mut it))?),
        })
    }

    fn lvalue(&self, pair: Pair<'_, Rule>) -> Built<Expr> {
        let mut it = kids(pair).into_iter();
        let first = self.ident(next(&mut it));
        let mut expr = Expr {
            span: first.span,
            kind: ExprKind::Name(first),
        };
        self.postfix_ops(&mut expr, it)?;
        Ok(expr)
    }

    fn postfix_ops<'p>(
        &self,
        expr: &mut Expr,
        ops: impl Iterator<Item = Pair<'p, Rule>>,
    ) -> Built<()> {
        for op in ops {
            let op_span = span_of(&op);
            let base = Box::new(std::mem::replace(
                expr,
                Expr {
                    kind: ExprKind::Bool(false),
                    span: Span::default(),
                },
            ));
            let span = Span::new(base.span.start, op_span.end);
            let kind = match op.as_rule() {
                Rule::member => ExprKind::Member {
                    base,
                    name: self.ident(only(op)),
                },
                Rule::call_args => ExprKind::Call {
                    callee: base,
                    args: self.args(only(op))?,
                },
                _ => ExprKind::Index {
                    base,
                    index: self.list(op, |b, p| b.index_item(p))?,
                },
            };
            *expr = Expr { kind, span };
            if expr_depth(expr) > MAX_DEPTH {
                return Err(too_deep(self.source, span));
            }
        }
        Ok(())
    }

    fn index_item(&self, pair: Pair<'_, Rule>) -> Built<IndexItem> {
        let p = only(pair);
        if p.as_rule() == Rule::expr {
            return Ok(IndexItem::At(self.expr(p)?));
        }
        let mut it = kids(p).into_iter();
        let lo = self.expr(next(&mut it))?;
        let hi = self.expr(next(&mut it))?;
        let step = match it.next() {
            Some(q) => Some(self.expr(q)?),
            None => None,
        };
        Ok(IndexItem::Slice {
            lo: Box::new(lo),
            hi: Box::new(hi),
            step: step.map(Box::new),
        })
    }

    fn args(&self, pair: Pair<'_, Rule>) -> Built<List<Arg>> {
        self.list(pair, |b, p| {
            let span = span_of(&p);
            let mut it = kids(p).into_iter();
            let first = next(&mut it);
            if first.as_rule() == Rule::arg_name {
                let name = Some(b.ident(only(first)));
                let value = b.expr(next(&mut it))?;
                return Ok(Arg { name, value, span });
            }
            Ok(Arg {
                name: None,
                value: b.expr(first)?,
                span,
            })
        })
    }

    fn expr(&self, pair: Pair<'_, Rule>) -> Built<Expr> {
        self.pratt_expr(pair.into_inner()).map(|(expr, _)| expr)
    }

    /// The expression and its depth. The depth is counted as the tree is built, so a
    /// chain too deep is refused before it is whole.
    fn pratt_expr(&self, pairs: Pairs<'_, Rule>) -> Built<(Expr, usize)> {
        pratt()
            .map_primary(|p| {
                let expr = self.postfix_expr(p)?;
                let depth = expr_depth(&expr);
                self.within(expr, depth)
            })
            .map_prefix(|op, operand| {
                let (operand, depth) = operand?;
                let expr = Expr {
                    span: Span::new(op.as_span().start(), operand.span.end),
                    kind: ExprKind::Neg(Box::new(operand)),
                };
                self.within(expr, depth + 1)
            })
            .map_infix(|lhs, op, rhs| {
                let ((lhs, left), (rhs, right)) = (lhs?, rhs?);
                let expr = Expr {
                    span: Span::new(lhs.span.start, rhs.span.end),
                    kind: ExprKind::Binary {
                        op: bin_op(op.as_rule()),
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                };
                self.within(expr, left.max(right) + 1)
            })
            .parse(pairs)
    }

    /// The expression and its depth, or K011 when the depth is over the limit.
    fn within(&self, expr: Expr, depth: usize) -> Built<(Expr, usize)> {
        if depth > MAX_DEPTH {
            Err(too_deep(self.source, expr.span))
        } else {
            Ok((expr, depth))
        }
    }

    fn postfix_expr(&self, pair: Pair<'_, Rule>) -> Built<Expr> {
        let mut it = kids(pair).into_iter();
        let mut expr = self.primary(next(&mut it))?;
        self.postfix_ops(&mut expr, it)?;
        Ok(expr)
    }

    fn primary(&self, pair: Pair<'_, Rule>) -> Built<Expr> {
        let p = only(pair);
        let span = span_of(&p);
        let kind = match p.as_rule() {
            Rule::paren_or_tuple => {
                let list = self.list(p, |b, q| b.expr(q))?;
                if list.len() == 1 && !list.has_comments() {
                    let mut entries = list.entries;
                    ExprKind::Paren(Box::new(entries.remove(0).value))
                } else if list.len() == 1 {
                    return Err(self.fail(span, "comments inside parentheses"));
                } else {
                    ExprKind::Tuple(list)
                }
            }
            Rule::list_lit => ExprKind::List(self.list(p, |b, q| b.expr(q))?),
            Rule::anon_init => ExprKind::Init {
                ty: None,
                fields: self.inits(only(p))?,
            },
            Rule::struct_init => {
                let mut it = kids(p).into_iter();
                let ty = Some(self.ident(next(&mut it)));
                ExprKind::Init {
                    ty,
                    fields: self.inits(next(&mut it))?,
                }
            }
            Rule::float_lit => ExprKind::Float(p.as_str().to_string()),
            Rule::int_lit => ExprKind::Int(p.as_str().to_string()),
            Rule::string_lit => ExprKind::Str(self.string(p).text),
            Rule::true_lit => ExprKind::Bool(true),
            Rule::false_lit => ExprKind::Bool(false),
            Rule::await_expr => ExprKind::Await(self.ident(only(p))),
            Rule::past_expr => ExprKind::Past(self.ident(only(p))),
            _ => ExprKind::Name(self.ident(p)),
        };
        Ok(Expr { kind, span })
    }

    fn inits(&self, pair: Pair<'_, Rule>) -> Built<List<FieldInit>> {
        self.list(pair, |b, p| {
            let span = span_of(&p);
            let mut it = kids(p).into_iter();
            let name = b.ident(next(&mut it));
            let value = b.expr(next(&mut it))?;
            Ok(FieldInit { name, value, span })
        })
    }
}

/// The depth of an expression tree, counted without recursion.
fn expr_depth(expr: &Expr) -> usize {
    let mut deepest = 0;
    let mut stack = vec![(expr, 1)];
    while let Some((e, d)) = stack.pop() {
        deepest = deepest.max(d);
        stack.extend(children(e).into_iter().map(|child| (child, d + 1)));
    }
    deepest
}

/// The expressions directly inside an expression.
fn children(e: &Expr) -> Vec<&Expr> {
    match &e.kind {
        ExprKind::Paren(inner) | ExprKind::Neg(inner) => vec![inner],
        ExprKind::Member { base, .. } => vec![base],
        ExprKind::Binary { lhs, rhs, .. } => vec![lhs, rhs],
        ExprKind::Tuple(list) | ExprKind::List(list) => list.values().collect(),
        ExprKind::Init { fields, .. } => fields.values().map(|f| &f.value).collect(),
        ExprKind::Call { callee, args } => {
            let mut out = vec![&**callee];
            out.extend(args.values().map(|a| &a.value));
            out
        }
        ExprKind::Index { base, index } => {
            let mut out = vec![&**base];
            for item in index.values() {
                match item {
                    IndexItem::At(at) => out.push(at),
                    IndexItem::Slice { lo, hi, step } => {
                        out.push(lo);
                        out.push(hi);
                        out.extend(step.as_deref());
                    }
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

/// The parts of a function, handler, run, chain or control program before its body.
struct Head<'p> {
    name: Ident,
    params: List<ParamDecl>,
    ret: Option<Type>,
    attrs: Vec<Attr>,
    rest: Vec<Pair<'p, Rule>>,
}

/// The next pair of a sequence the grammar guarantees.
fn next<'p>(it: &mut impl Iterator<Item = Pair<'p, Rule>>) -> Pair<'p, Rule> {
    it.next().expect("the grammar guarantees this pair")
}

/// The one child, keywords left out, of a pair the grammar gives exactly one.
fn only(pair: Pair<'_, Rule>) -> Pair<'_, Rule> {
    next(&mut kids(pair).into_iter())
}
