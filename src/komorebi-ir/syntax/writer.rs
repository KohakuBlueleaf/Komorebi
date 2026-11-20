//! Syntax tree to canonical text.
//!
//! The layout: braces; four spaces a level; a form that does not fit in 88 columns is
//! broken at a bracket as PEP 8 lays out a call (one element a line, a trailing comma,
//! the closing bracket on its own line); attributes after the head they mark.

use super::ast::*;

/// The widest line the writer makes where the form can be broken.
pub const WIDTH: usize = 88;
const INDENT: &str = "    ";

/// The canonical text of a module. Its walks recurse as deep as the tree: call it
/// on the stack `crate::deep` gives, as `crate::api::write` does.
pub fn write_module(module: &Module) -> String {
    let mut w = Writer { out: String::new() };
    w.module(module);
    w.out
}

/// A part of a line: text, or a bracketed list that may be broken over lines.
#[derive(Clone, Debug)]
enum Piece {
    Text(String),
    List(ListDoc),
}

#[derive(Clone, Debug)]
struct ListDoc {
    open: &'static str,
    close: &'static str,
    /// `{ a, b }` rather than `{a, b}`.
    pad: bool,
    /// The operands of an attribute: broken only when nothing else can be.
    in_attr: bool,
    open_comment: Option<String>,
    items: Vec<ItemDoc>,
    end: Vec<String>,
}

#[derive(Clone, Debug)]
struct ItemDoc {
    leading: Vec<String>,
    pieces: Vec<Piece>,
    trailing: Option<String>,
}

impl ListDoc {
    fn has_comments(&self) -> bool {
        self.open_comment.is_some()
            || !self.end.is_empty()
            || self
                .items
                .iter()
                .any(|item| !item.leading.is_empty() || item.trailing.is_some())
    }
}

/// The pieces on one line, or `None` when a comment forces a break.
fn flat(pieces: &[Piece]) -> Option<String> {
    let mut s = String::new();
    for piece in pieces {
        match piece {
            Piece::Text(text) => s.push_str(text),
            Piece::List(list) => {
                if list.has_comments() {
                    return None;
                }
                s.push_str(list.open);
                if list.pad && !list.items.is_empty() {
                    s.push(' ');
                }
                for (k, item) in list.items.iter().enumerate() {
                    if k > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&flat(&item.pieces)?);
                }
                if list.pad && !list.items.is_empty() {
                    s.push(' ');
                }
                s.push_str(list.close);
            }
        }
    }
    Some(s)
}

fn width(text: &str) -> usize {
    text.chars().count()
}

fn text(s: impl Into<String>) -> Piece {
    Piece::Text(s.into())
}

struct Writer {
    out: String,
}

impl Writer {
    fn line(&mut self, indent: usize, content: &str, trailing: Option<&str>) {
        if !content.is_empty() {
            for _ in 0..indent {
                self.out.push_str(INDENT);
            }
            self.out.push_str(content);
        }
        if let Some(comment) = trailing {
            if content.is_empty() {
                for _ in 0..indent {
                    self.out.push_str(INDENT);
                }
            } else {
                self.out.push_str("  ");
            }
            self.out.push_str(comment);
        }
        self.out.push('\n');
    }

    fn comment_lines(&mut self, indent: usize, comments: &[String]) {
        for comment in comments {
            self.line(indent, "", Some(comment));
        }
    }

    fn leads(&mut self, indent: usize, leads: &[Lead]) {
        for lead in leads {
            match lead {
                Lead::Blank => self.out.push('\n'),
                Lead::Comment(comment) => self.line(indent, "", Some(&comment.text)),
            }
        }
    }

    /// Writes `pieces` as one line when they fit, and broken at a list otherwise.
    /// `suffix` follows the last line; `trailing` is the comment on the last line.
    fn emit(&mut self, pieces: &[Piece], indent: usize, suffix: &str, trailing: Option<&str>) {
        let room = WIDTH.saturating_sub(indent * INDENT.len() + width(suffix));
        if let Some(s) = flat(pieces) {
            if width(&s) <= room {
                self.line(indent, &format!("{s}{suffix}"), trailing);
                return;
            }
        }
        let Some(k) = choose_break(pieces, room) else {
            let s = flat(pieces).unwrap_or_default();
            self.line(indent, &format!("{s}{suffix}"), trailing);
            return;
        };
        let Piece::List(list) = &pieces[k] else {
            return;
        };
        let head = flat(&pieces[..k]).unwrap_or_default();
        self.line(
            indent,
            &format!("{head}{}", list.open),
            list.open_comment.as_deref(),
        );
        for item in &list.items {
            self.comment_lines(indent + 1, &item.leading);
            self.emit(&item.pieces, indent + 1, ",", item.trailing.as_deref());
        }
        self.comment_lines(indent + 1, &list.end);
        let mut tail = vec![text(list.close)];
        tail.extend(pieces[k + 1..].iter().cloned());
        self.emit(&tail, indent, suffix, trailing);
    }

    fn module(&mut self, module: &Module) {
        self.leads(0, &module.trivia.leading);
        let head = format!("module {}", module.name.name);
        self.line(0, &head, comment_text(&module.trivia.trailing));
        for line in &module.items.lines {
            self.leads(0, &line.trivia.leading);
            self.item(&line.node, 0, comment_text(&line.trivia.trailing));
        }
        self.leads(0, &module.items.end);
    }

    fn item(&mut self, item: &Item, indent: usize, trailing: Option<&str>) {
        match item {
            Item::Tile(tile) => {
                let s = format!("tile [{}, {}]", tile.rows.text, tile.cols.text);
                self.line(indent, &s, trailing);
            }
            Item::Format(format) => {
                let names: Vec<&str> = format.formats.iter().map(|f| f.name.as_str()).collect();
                let s = format!("format {}", names.join(", "));
                self.line(indent, &s, trailing);
            }
            Item::Requires(requires) => {
                let head = vec![text("requires")];
                self.lines_block(head, &requires.body, indent, trailing, |w, r, i, t| {
                    w.line(i, &requirement(r), t)
                });
            }
            Item::Param(param) => {
                let mut pieces = vec![text(format!("param {}: ", param.name.name))];
                pieces.extend(ty(&param.ty));
                pieces.push(text(" = "));
                match &param.value {
                    ParamValue::Fit(_) => pieces.push(text("fit")),
                    ParamValue::Expr(e) => pieces.extend(expr(e)),
                }
                pieces.extend(attrs(&param.attrs));
                self.emit(&pieces, indent, "", trailing);
            }
            Item::Message(message) => self.message(message, indent, trailing),
            Item::Record(record) => self.record(record, indent, trailing),
            Item::Func(func) => self.func("func", func, indent, trailing),
            Item::Combine(func) => self.func("combine", func, indent, trailing),
            Item::Extern(ext) => {
                let mut head = vec![text(format!("extern func {}", ext.name.name))];
                head.push(params(&ext.params));
                head.push(text(" -> "));
                head.extend(ty(&ext.ret));
                head.extend(attrs(&ext.attrs));
                match &ext.fallback {
                    None => self.emit(&head, indent, "", trailing),
                    Some(body) => {
                        head.push(text(" fallback"));
                        self.blocky(head, &[("", body)], indent, trailing);
                    }
                }
            }
            Item::Program(program) => {
                let head = vec![text(format!("program {}", program.name.name))];
                self.lines_block(head, &program.body, indent, trailing, |w, p, i, t| {
                    w.program_item(p, i, t)
                });
            }
            Item::Control(control) => {
                let mut head = vec![text(format!("control {}", control.name.name))];
                head.push(params(&control.params));
                if let Some(ret) = &control.ret {
                    head.push(text(" -> "));
                    head.extend(ty(ret));
                }
                head.extend(attrs(&control.attrs));
                self.blocky(head, &[("", &control.body)], indent, trailing);
            }
        }
    }

    fn message(&mut self, message: &Message, indent: usize, trailing: Option<&str>) {
        let mut pieces = vec![text(format!("message {}", message.name.name))];
        pieces.extend(attrs(&message.attrs));
        pieces.push(text(" "));
        pieces.push(fields(&message.fields));
        self.emit(&pieces, indent, "", trailing);
    }

    fn record(&mut self, record: &Record, indent: usize, trailing: Option<&str>) {
        let pieces = vec![
            text(format!("record {} ", record.name.name)),
            fields(&record.fields),
        ];
        self.emit(&pieces, indent, "", trailing);
    }

    fn func(&mut self, word: &str, func: &Func, indent: usize, trailing: Option<&str>) {
        let mut head = vec![text(format!("{word} {}", func.name.name))];
        head.push(params(&func.params));
        head.push(text(" -> "));
        head.extend(ty(&func.ret));
        head.extend(attrs(&func.attrs));
        self.blocky(head, &[("", &func.body)], indent, trailing);
    }

    fn program_item(&mut self, item: &ProgramItem, indent: usize, trailing: Option<&str>) {
        match item {
            ProgramItem::Message(message) => self.message(message, indent, trailing),
            ProgramItem::Record(record) => self.record(record, indent, trailing),
            ProgramItem::State(state) => {
                let mut head = vec![text("state")];
                head.extend(attrs(&state.attrs));
                self.lines_block(head, &state.members, indent, trailing, |w, m, i, t| {
                    let mut pieces = vec![text(format!("{}: ", m.name.name))];
                    pieces.extend(ty(&m.ty));
                    pieces.extend(attrs(&m.attrs));
                    w.emit(&pieces, i, "", t);
                });
            }
            ProgramItem::Handler(handler) => {
                let mut head = vec![text(format!("handler {}", handler.name.name))];
                head.push(params(&handler.params));
                if let Some(ret) = &handler.ret {
                    head.push(text(" -> "));
                    head.extend(ty(ret));
                }
                head.extend(attrs(&handler.attrs));
                self.blocky(head, &[("", &handler.body)], indent, trailing);
            }
            ProgramItem::Run(run) => {
                let mut head = vec![text(format!("run {}", run.name.name))];
                head.push(params(&run.params));
                head.extend(attrs(&run.attrs));
                self.lines_block(
                    head,
                    &run.body,
                    indent,
                    trailing,
                    |w, part, i, t| match part {
                        RunPart::Seed(body) => w.blocky(vec![text("seed")], &[("", body)], i, t),
                        RunPart::On(on) => {
                            let head = format!("on {}({})", on.message.name, on.binding.name);
                            w.blocky(vec![text(head)], &[("", &on.body)], i, t);
                        }
                    },
                );
            }
            ProgramItem::Chain(chain) => {
                let mut head = vec![text(format!("chain {}", chain.name.name))];
                head.push(params(&chain.params));
                head.extend(attrs(&chain.attrs));
                self.lines_block(head, &chain.calls, indent, trailing, |w, call, i, t| {
                    let pieces = vec![
                        text(format!("call {}", call.handler.name)),
                        args(&call.args, false),
                    ];
                    w.emit(&pieces, i, "", t);
                });
            }
        }
    }

    /// A head and a braced sequence of lines that is always written one a line.
    fn lines_block<T>(
        &mut self,
        mut head: Vec<Piece>,
        body: &Lines<T>,
        indent: usize,
        trailing: Option<&str>,
        mut part: impl FnMut(&mut Writer, &T, usize, Option<&str>),
    ) {
        head.push(text(" {"));
        self.emit(&head, indent, "", comment_text(&body.open));
        for line in &body.lines {
            self.leads(indent + 1, &line.trivia.leading);
            part(
                self,
                &line.node,
                indent + 1,
                comment_text(&line.trivia.trailing),
            );
        }
        self.leads(indent + 1, &body.end);
        self.line(indent, "}", trailing);
    }

    /// A head followed by blocks (`if ... { } else { }`): each `(joiner, block)` adds
    /// the joiner and the block, always broken, one statement a line.
    fn blocky(
        &mut self,
        head: Vec<Piece>,
        blocks: &[(&str, &Block)],
        indent: usize,
        trailing: Option<&str>,
    ) {
        let mut current = head;
        for (joiner, block) in blocks {
            if !joiner.is_empty() {
                current.push(text(*joiner));
            }
            current.push(text(" {"));
            self.emit(&current, indent, "", comment_text(&block.stmts.open));
            for line in &block.stmts.lines {
                self.leads(indent + 1, &line.trivia.leading);
                self.stmt(&line.node, indent + 1, comment_text(&line.trivia.trailing));
            }
            self.leads(indent + 1, &block.stmts.end);
            current = vec![text("}")];
        }
        self.emit(&current, indent, "", trailing);
    }

    fn stmt(&mut self, stmt: &Stmt, indent: usize, trailing: Option<&str>) {
        match stmt {
            Stmt::For {
                vars,
                iter,
                attrs: a,
                body,
            } => self.blocky(for_head(vars, iter, a), &[("", body)], indent, trailing),
            Stmt::While { cond, body } => {
                let mut head = vec![text("while ")];
                head.extend(expr(cond));
                self.blocky(head, &[("", body)], indent, trailing);
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                let mut head = vec![text("if ")];
                head.extend(expr(cond));
                match otherwise {
                    None => self.blocky(head, &[("", then)], indent, trailing),
                    Some(other) => {
                        self.blocky(head, &[("", then), (" else", other)], indent, trailing)
                    }
                }
            }
            _ => {
                let pieces = simple_stmt(stmt);
                self.emit(&pieces, indent, "", trailing);
            }
        }
    }
}

fn comment_text(comment: &Option<Comment>) -> Option<&str> {
    comment.as_ref().map(|c| c.text.as_str())
}

fn for_head(vars: &[Ident], iter: &Expr, a: &[Attr]) -> Vec<Piece> {
    let names: Vec<&str> = vars.iter().map(|v| v.name.as_str()).collect();
    let mut head = vec![text(format!("for {} in ", names.join(", ")))];
    head.extend(expr(iter));
    head.extend(attrs(a));
    head
}

fn simple_stmt(stmt: &Stmt) -> Vec<Piece> {
    let mut p = Vec::new();
    match stmt {
        Stmt::Let {
            name,
            ty: t,
            value,
            attrs: a,
        } => {
            p.push(text(format!("let {}", name.name)));
            if let Some(t) = t {
                p.push(text(": "));
                p.extend(ty(t));
            }
            p.push(text(" = "));
            p.extend(expr(value));
            p.extend(attrs(a));
        }
        Stmt::Var { name, ty: t, value } => {
            p.push(text(format!("var {}: ", name.name)));
            p.extend(ty(t));
            p.push(text(" = "));
            p.extend(expr(value));
        }
        Stmt::Assign { target, op, value } => {
            p.extend(expr(target));
            p.push(text(format!(" {} ", op.text())));
            p.extend(expr(value));
        }
        Stmt::Store { target, value } => {
            p.push(text("store "));
            p.extend(expr(target));
            p.push(text(" <- "));
            p.extend(expr(value));
        }
        Stmt::Consume(name) => p.push(text(format!("consume {}", name.name))),
        Stmt::Return(None) => p.push(text("return")),
        Stmt::Return(Some(value)) => {
            p.push(text("return "));
            p.extend(expr(value));
        }
        Stmt::Expr(e) => p.extend(expr(e)),
        Stmt::For { .. } | Stmt::While { .. } | Stmt::If { .. } => {}
    }
    p
}

fn requirement(r: &Requirement) -> String {
    match r {
        Requirement::LocalState { unit, amount, .. } => {
            let unit = match unit {
                StateUnit::Tiles => "tiles",
                StateUnit::Bytes => "bytes",
            };
            format!("local_state >= {unit}({})", amount.text)
        }
        Requirement::Group { shape, .. } => match shape {
            GroupShape::Line => "group = line".to_string(),
            GroupShape::Grid => "group = grid".to_string(),
        },
        Requirement::Dynamic { .. } => "dynamic".to_string(),
        Requirement::Islands { count, .. } => format!("islands >= {}", count.text),
        Requirement::Impl { name, .. } => format!("impl(\"{}\")", name.text),
    }
}

fn list_doc<T>(
    list: &List<T>,
    open: &'static str,
    close: &'static str,
    pad: bool,
    element: impl Fn(&T) -> Vec<Piece>,
) -> ListDoc {
    ListDoc {
        open,
        close,
        pad,
        in_attr: false,
        open_comment: list.open.as_ref().map(|c| c.text.clone()),
        items: list
            .entries
            .iter()
            .map(|entry| ItemDoc {
                leading: entry.leading.iter().map(|c| c.text.clone()).collect(),
                pieces: element(&entry.value),
                trailing: entry.trailing.as_ref().map(|c| c.text.clone()),
            })
            .collect(),
        end: list.end.iter().map(|c| c.text.clone()).collect(),
    }
}

fn params(list: &List<ParamDecl>) -> Piece {
    Piece::List(list_doc(list, "(", ")", false, |p| {
        let mut pieces = vec![text(format!("{}: ", p.name.name))];
        pieces.extend(ty(&p.ty));
        pieces.extend(attrs(&p.attrs));
        pieces
    }))
}

fn fields(list: &List<Field>) -> Piece {
    Piece::List(list_doc(list, "{", "}", true, |f| {
        let mut pieces = vec![text(format!("{}: ", f.name.name))];
        pieces.extend(ty(&f.ty));
        pieces
    }))
}

fn args(list: &List<Arg>, in_attr: bool) -> Piece {
    let mut doc = list_doc(list, "(", ")", false, |a| {
        let mut pieces = Vec::new();
        if let Some(name) = &a.name {
            pieces.push(text(format!("{} = ", name.name)));
        }
        pieces.extend(expr(&a.value));
        pieces
    });
    doc.in_attr = in_attr;
    Piece::List(doc)
}

fn attrs(list: &[Attr]) -> Vec<Piece> {
    let mut pieces = Vec::new();
    for attr in list {
        let path: Vec<&str> = attr.path.iter().map(|p| p.name.as_str()).collect();
        pieces.push(text(format!(" @{}", path.join("."))));
        if let Some(a) = &attr.args {
            pieces.push(args(a, true));
        }
    }
    pieces
}

fn ty(t: &Type) -> Vec<Piece> {
    let mut pieces = vec![text(match &t.kind {
        TypeKind::Scalar(s) => s.name().to_string(),
        TypeKind::Bool => "bool".to_string(),
        TypeKind::Tile { format } => format!("tile<{}>", format.name),
        TypeKind::Tensor { format } => format!("tensor<{}>", format.name),
        TypeKind::View => "view".to_string(),
        TypeKind::Named(name) => name.name.clone(),
    })];
    if let Some(count) = &t.count {
        pieces.push(text("["));
        if let Some(rows) = &t.rows {
            pieces.extend(expr(rows));
            pieces.push(text(", "));
        }
        pieces.extend(expr(count));
        pieces.push(text("]"));
    }
    pieces
}

fn expr(e: &Expr) -> Vec<Piece> {
    let mut p = Vec::new();
    push_expr(&mut p, e);
    p
}

fn push_expr(p: &mut Vec<Piece>, e: &Expr) {
    match &e.kind {
        ExprKind::Int(s) | ExprKind::Float(s) => p.push(text(s.clone())),
        ExprKind::Str(s) => p.push(text(format!("\"{s}\""))),
        ExprKind::Bool(b) => p.push(text(if *b { "true" } else { "false" })),
        ExprKind::Name(name) => p.push(text(name.name.clone())),
        ExprKind::Paren(inner) => {
            p.push(text("("));
            push_expr(p, inner);
            p.push(text(")"));
        }
        ExprKind::Tuple(list) => p.push(Piece::List(list_doc(list, "(", ")", false, expr))),
        ExprKind::List(list) => p.push(Piece::List(list_doc(list, "[", "]", false, expr))),
        ExprKind::Init { ty, fields } => {
            if let Some(ty) = ty {
                p.push(text(format!("{} ", ty.name)));
            }
            p.push(Piece::List(list_doc(fields, "{", "}", true, |f| {
                let mut pieces = vec![text(format!("{}: ", f.name.name))];
                pieces.extend(expr(&f.value));
                pieces
            })));
        }
        ExprKind::Member { base, name } => {
            push_expr(p, base);
            p.push(text(format!(".{}", name.name)));
        }
        ExprKind::Call { callee, args: a } => {
            push_expr(p, callee);
            p.push(args(a, false));
        }
        ExprKind::Index { base, index } => {
            push_expr(p, base);
            p.push(Piece::List(list_doc(
                index,
                "[",
                "]",
                false,
                |item| match item {
                    IndexItem::At(e) => expr(e),
                    IndexItem::Slice { lo, hi, step } => {
                        let mut pieces = expr(lo);
                        pieces.push(text(":"));
                        pieces.extend(expr(hi));
                        if let Some(step) = step {
                            pieces.push(text(":"));
                            pieces.extend(expr(step));
                        }
                        pieces
                    }
                },
            )));
        }
        ExprKind::Neg(operand) => {
            p.push(text("-"));
            push_expr(p, operand);
        }
        ExprKind::Binary { op, lhs, rhs } => {
            push_expr(p, lhs);
            p.push(text(format!(" {} ", op.text())));
            push_expr(p, rhs);
        }
        ExprKind::Await(name) => p.push(text(format!("await {}", name.name))),
        ExprKind::Past(name) => p.push(text(format!("past({})", name.name))),
    }
}

/// The list to break a line at: the first that holds a comment; else the last list
/// outside attributes whose line up to its bracket fits, else the first such list;
/// else the same among the attributes' lists.
fn choose_break(pieces: &[Piece], room: usize) -> Option<usize> {
    let lists: Vec<(usize, &ListDoc)> = pieces
        .iter()
        .enumerate()
        .filter_map(|(k, p)| match p {
            Piece::List(list) => Some((k, list)),
            Piece::Text(_) => None,
        })
        .collect();
    let forced = lists
        .iter()
        .find(|(k, _)| flat(&pieces[*k..*k + 1]).is_none());
    if let Some((k, _)) = forced {
        return Some(*k);
    }
    for in_attr in [false, true] {
        let candidates: Vec<usize> = lists
            .iter()
            .filter(|(_, list)| list.in_attr == in_attr && !list.items.is_empty())
            .map(|(k, _)| *k)
            .collect();
        let fits = candidates.iter().rev().find(|&&k| {
            let Piece::List(list) = &pieces[k] else {
                return false;
            };
            flat(&pieces[..k]).is_some_and(|head| width(&head) + width(list.open) <= room)
        });
        if let Some(k) = fits.or(candidates.first()) {
            return Some(*k);
        }
    }
    None
}
