//! The operations of a body: their operands (by position or by name), and the type
//! each yields from its operands.

use std::collections::HashSet;

use super::ty::{Count, Shape, Ty};
use crate::syntax;
use crate::syntax::ast::{Arg, Expr, ExprKind, Ident, List, Span};
use crate::types::{Axis, Format, ScalarType};

use super::body::{
    as_int, as_name, count_of, named, named_opt, pos, Cx, Family, Kind, Operand, PROVIDED, UNARY,
};
use super::codes;
use super::env::FuncKind;
use super::op::OpKind;
use super::{Checker, Name};
use crate::types::{
    BinOp, ChainId, Combiner, Entry, FuncId, HandlerId, MemberId, ProgramId, ReduceOp, RunId,
    SplitBy, UnaryOp,
};

impl Checker<'_> {
    // ---- Operands -----------------------------------------------------------------

    /// Matches the operands of a call to the operation's form: each by position or by
    /// name (`K111`), every one that may not be left out given (`K103`). `None` after a
    /// diagnostic.
    pub(super) fn operands<'a>(
        &mut self,
        what: &str,
        span: Span,
        args: &'a List<Arg>,
        form: &[Operand],
    ) -> Option<Vec<Option<&'a Expr>>> {
        let mut given: Vec<Option<&Expr>> = vec![None; form.len()];
        let positional: Vec<usize> = (0..form.len()).filter(|&i| !form[i].named).collect();
        let mut next = 0;
        for arg in args.values() {
            match &arg.name {
                None => {
                    if next < positional.len() {
                        given[positional[next]] = Some(&arg.value);
                        next += 1;
                    } else {
                        let by_name: Vec<String> = form
                            .iter()
                            .filter(|o| o.named)
                            .map(|o| format!("`{} = ...`", o.name))
                            .collect();
                        let text = if by_name.is_empty() {
                            format!("`{what}` takes {} operands by position", positional.len())
                        } else {
                            format!(
                                "`{what}` takes {} operands by position; give {} by name",
                                positional.len(),
                                by_name.join(", ")
                            )
                        };
                        self.error(codes::OPERAND_FORM, text, arg.span);
                        return None;
                    }
                }
                // an operand's name between backticks is the module's, and no item of
                // the module names an operand (R50)
                Some(name) if name.quoted => {
                    self.unknown(name);
                    return None;
                }
                Some(name) => match form.iter().position(|o| o.name == name.name) {
                    Some(i) if form[i].named && given[i].is_none() => {
                        given[i] = Some(&arg.value);
                    }
                    Some(i) if form[i].named => {
                        self.error(
                            codes::OPERAND_FORM,
                            format!("`{}` is given twice", name.name),
                            arg.span,
                        );
                        return None;
                    }
                    Some(_) => {
                        self.error(
                            codes::OPERAND_FORM,
                            format!("`{what}` takes `{}` by position", name.name),
                            arg.span,
                        );
                        return None;
                    }
                    None => {
                        self.error(
                            codes::OPERAND_FORM,
                            format!("`{what}` takes no operand named `{}`", name.name),
                            arg.span,
                        );
                        return None;
                    }
                },
            }
        }
        let missing: Vec<&str> = form
            .iter()
            .zip(&given)
            .filter(|(o, g)| !o.optional && g.is_none())
            .map(|(o, _)| o.name)
            .collect();
        if !missing.is_empty() {
            self.error(
                codes::MISSING_OPERAND,
                format!("`{what}` is not given `{}`", missing.join("`, `")),
                span,
            );
            return None;
        }
        Some(given)
    }

    /// Checks an operand against the type it must have.
    pub(super) fn operand(&mut self, cx: &mut Cx, e: Option<&Expr>, want: &Ty, what: &str) -> Ty {
        let Some(e) = e else {
            return Ty::Unknown;
        };
        let ty = self.expr(cx, e);
        self.fit(e, &ty, want, what)
    }

    /// An operand that names one of `words` (`axis = row`, `along = col`).
    pub(super) fn word_operand(&mut self, e: Option<&Expr>, words: &[&str], what: &str) {
        let Some(e) = e else {
            return;
        };
        match as_name(e) {
            Some(n) if !n.quoted && words.contains(&n.name.as_str()) => {
                self.note_word(n);
            }
            _ => self.error(
                codes::WRONG_TYPE,
                format!("{what} is one of `{}`", words.join("`, `")),
                e.span,
            ),
        }
    }

    /// The combination of a reduce: a `combine`, or one of the provided ones.
    pub(super) fn combination(&mut self, cx: &Cx, e: Option<&Expr>) {
        let Some(e) = e else {
            return;
        };
        let Some(n) = as_name(e) else {
            self.error(
                codes::REDUCE_WITH,
                "a reduce combines with a `combine` or one of `sum max min and or`".into(),
                e.span,
            );
            return;
        };
        if !n.quoted && PROVIDED.contains(&n.name.as_str()) {
            self.note_word(n);
            return;
        }
        match cx.env.funcs.get(&n.name) {
            Some(f) if f.kind == FuncKind::Combine => self.note_func(cx, n),
            Some(_) => self.error(
                codes::REDUCE_WITH,
                format!(
                    "`{}` is not a `combine`; a reduce combines with a `combine` or one of \
                     `sum max min and or`",
                    n.name
                ),
                n.span,
            ),
            None => self.unknown(n),
        }
    }

    /// Each element of a list of views (`publish = [v]`).
    pub(super) fn view_list(&mut self, cx: &mut Cx, e: Option<&Expr>) {
        let Some(e) = e else {
            return;
        };
        match &e.kind {
            ExprKind::List(items) => {
                for item in items.values() {
                    let ty = self.expr(cx, item);
                    self.expect_fit(
                        &ty,
                        &Ty::View {
                            format: None,
                            tiles: Count::Unknown,
                        },
                        item.span,
                        "a published view",
                    );
                }
                self.note_type(e, &Ty::List);
            }
            _ => self.error(
                codes::WRONG_TYPE,
                "`publish` is a list of views".into(),
                e.span,
            ),
        }
    }

    // ---- Calls --------------------------------------------------------------------

    pub(super) fn call(&mut self, cx: &mut Cx, callee: &Expr, args: &List<Arg>, span: Span) -> Ty {
        match &callee.kind {
            ExprKind::Name(n) => self.free_call(cx, n, args, span),
            ExprKind::Member { base, name } => self.method(cx, base, name, args, span),
            _ => {
                self.error(codes::WRONG_TYPE, "this is not called".into(), callee.span);
                Ty::Unknown
            }
        }
    }

    pub(super) fn free_call(&mut self, cx: &mut Cx, n: &Ident, args: &List<Arg>, span: Span) -> Ty {
        let what = n.name.as_str();
        // a bare name of the language's calls is the language's; one between backticks
        // is the module's only (R50)
        let language =
            !n.quoted && (syntax::CALLS.contains(&what) || syntax::CONVERSIONS.contains(&what));
        if !language && (cx.lookup(what).is_some() || cx.state(what).is_some()) {
            self.error(codes::WRONG_TYPE, format!("`{what}` is not called"), n.span);
            return Ty::Unknown;
        }
        if let Some(sig) = cx.env.funcs.get(what).filter(|_| !language).cloned() {
            if !self.allowed(cx, Family::Compute, what, span) {
                return Ty::Unknown;
            }
            self.note_func(cx, n);
            let lifted = self.positional_call(cx, what, span, args, &sig.params);
            if let Some(&f) = cx.env.ix.funcs.get(what) {
                let parts: Vec<Option<&Expr>> = args.values().map(|a| Some(&a.value)).collect();
                self.note_op(span, OpKind::CallFunc(FuncId(f)), &parts);
            }
            // a function on tiles run on each tile of blocks of one shape (R45)
            return match (sig.ret, lifted) {
                (Ty::Tile(f), Some(shape)) => Ty::Block(f, shape),
                (ret, _) => ret,
            };
        }
        if n.quoted {
            self.unknown(n);
            return Ty::Unknown;
        }
        if let Some(target) = ScalarType::from_name(what) {
            if !self.allowed(cx, Family::Value, what, span) {
                return Ty::Unknown;
            }
            self.note_name(n.span, Name::Operation);
            return self.convert(cx, target, args, span);
        }
        let family = match what {
            "min" | "max" | "range" => Family::Value,
            "where" | "maximum" | "minimum" | "matmul" | "reduce_sum" | "reduce_max"
            | "reduce_min" => Family::Compute,
            _ if UNARY.contains(&what) => Family::Compute,
            "partition" | "stream" | "sink" | "zip" | "reduce" | "gather" | "broadcast"
            | "sync" => Family::Worker,
            "all_reduce" => {
                if cx.kind == Kind::Control {
                    Family::Islands
                } else {
                    Family::Worker
                }
            }
            "send" | "forward" | "offer" => Family::Dynamic,
            "invoke" | "outcome" | "group" => Family::Control,
            "reduce_to" | "broadcast_from" => Family::Islands,
            "all_gather" => {
                self.error(
                    codes::NOT_BUILT,
                    "`all_gather` is named in the language and not built yet".into(),
                    span,
                );
                return Ty::Unknown;
            }
            _ => {
                self.unknown(n);
                return Ty::Unknown;
            }
        };
        if !self.allowed(cx, family, what, span) {
            return Ty::Unknown;
        }
        self.note_name(n.span, Name::Operation);
        if what == "forward"
            && args
                .values()
                .any(|a| a.name.as_ref().is_some_and(|n| n.name == "as" && !n.quoted))
        {
            self.error(
                codes::NOT_BUILT,
                "`as =` on a `forward` is named in the language and not built yet".into(),
                span,
            );
            return Ty::Unknown;
        }
        self.builtin(cx, what, args, span)
    }

    pub(super) fn builtin(&mut self, cx: &mut Cx, what: &str, args: &List<Arg>, span: Span) -> Ty {
        let view = Ty::View {
            format: None,
            tiles: Count::Unknown,
        };
        match what {
            "min" | "max" => {
                let Some(o) = self.operands(what, span, args, &[pos("a"), pos("b")]) else {
                    return Ty::Unknown;
                };
                let (Some(a), Some(b)) = (o[0], o[1]) else {
                    return Ty::Unknown;
                };
                let op = if what == "min" {
                    BinOp::Min
                } else {
                    BinOp::Max
                };
                self.note_op(span, OpKind::Binary(op), &o);
                let ta = self.expr(cx, a);
                let tb = self.expr(cx, b);
                self.expect_fit(&ta, &Ty::AnyNumber, a.span, "an operand");
                self.expect_fit(&tb, &Ty::AnyNumber, b.span, "an operand");
                self.one_type(what, a, &ta, b, &tb, span)
            }
            "range" => {
                let Some(o) = self.operands(what, span, args, &[pos("n")]) else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::ForRange, &o);
                match self.operand(cx, o[0], &Ty::AnyInt, "the count") {
                    Ty::Scalar(s) => Ty::Range(s),
                    _ => Ty::Unknown,
                }
            }
            "where" => {
                let Some(o) = self.operands(what, span, args, &[pos("c"), pos("x"), pos("y")])
                else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::Select, &o);
                if let Some(c) = o[0] {
                    let ty = self.expr(cx, c);
                    if !matches!(ty, Ty::Mask | Ty::Bool | Ty::Unknown) {
                        self.error(
                            codes::WRONG_TYPE,
                            format!("`where` takes a comparison first, not {ty}"),
                            c.span,
                        );
                    }
                }
                let x = self.tile_operand(cx, o[1]);
                let y = self.tile_operand(cx, o[2]);
                self.elementwise_of(what, &[o[1], o[2]], &[x, y], span)
            }
            "maximum" | "minimum" => {
                let Some(o) = self.operands(what, span, args, &[pos("a"), pos("b")]) else {
                    return Ty::Unknown;
                };
                let op = if what == "maximum" {
                    BinOp::Max
                } else {
                    BinOp::Min
                };
                self.note_op(span, OpKind::Binary(op), &o);
                let a = self.tile_operand(cx, o[0]);
                let b = self.tile_operand(cx, o[1]);
                self.elementwise_of(what, &[o[0], o[1]], &[a, b], span)
            }
            "matmul" => {
                let Some(o) = self.operands(what, span, args, &[pos("a"), pos("b")]) else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::Matmul, &o);
                if self.on_blocks {
                    return self.block_product(cx, o[0], o[1], span);
                }
                let a = self.operand(cx, o[0], &Ty::Tile(None), "the left operand");
                let b = self.operand(cx, o[1], &Ty::Tile(None), "the right operand");
                tile_of(&[a, b])
            }
            "reduce_sum" | "reduce_max" | "reduce_min" => {
                let Some(o) = self.operands(what, span, args, &[pos("a"), named("axis")]) else {
                    return Ty::Unknown;
                };
                // in a function on blocks (R45), of a block too: of each of its tiles
                let a = match o[0] {
                    Some(e) if self.on_blocks => self.tile_operand(cx, Some(e)),
                    _ => self.operand(cx, o[0], &Ty::Tile(None), "the operand"),
                };
                self.word_operand(o[1], &["row", "col"], "the axis");
                let axis = match o[1]
                    .and_then(as_name)
                    .filter(|n| !n.quoted)
                    .map(|n| n.name.as_str())
                {
                    Some("row") => Axis::Row,
                    Some("col") => Axis::Col,
                    _ => return Ty::Unknown,
                };
                let op = match what {
                    "reduce_sum" => ReduceOp::Sum,
                    "reduce_max" => ReduceOp::Max,
                    _ => ReduceOp::Min,
                };
                self.note_op(span, OpKind::Reduce(op, axis), &o[..1]);
                match a {
                    Ty::Tile(f) => Ty::Reduced(f, axis, Shape::TILE),
                    Ty::Block(f, shape) => Ty::Reduced(f, axis, shape),
                    _ => Ty::Unknown,
                }
            }
            _ if UNARY.contains(&what) => {
                let Some(o) = self.operands(what, span, args, &[pos("a")]) else {
                    return Ty::Unknown;
                };
                if let Some(u) = unary_op(what) {
                    self.note_op(span, OpKind::Unary(u), &o);
                }
                let a = self.tile_operand(cx, o[0]);
                self.elementwise_of(what, &[o[0]], &[a], span)
            }
            "partition" => {
                let form = [pos("view"), pos("rank"), pos("size")];
                let Some(o) = self.operands(what, span, args, &form) else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::Partition, &o);
                let v = self.operand(cx, o[0], &view, "the view");
                self.operand(cx, o[1], &Ty::AnyInt, "the rank");
                self.operand(cx, o[2], &Ty::AnyInt, "the size");
                match v {
                    Ty::View { format, .. } => Ty::View {
                        format,
                        tiles: Count::Unknown,
                    },
                    _ => Ty::Unknown,
                }
            }
            "stream" | "sink" => {
                let Some(o) = self.operands(what, span, args, &[pos("view"), named("block")])
                else {
                    return Ty::Unknown;
                };
                let kind = if what == "sink" {
                    OpKind::Sink
                } else {
                    OpKind::Stream
                };
                // `block = [r, c]` (R45): blocks of r rows of c tiles, the parts the
                // view, c and r; `block = n` is `[1, n]`
                let (o, rows) = match o[1].map(|b| &b.kind) {
                    Some(ExprKind::List(ends)) => {
                        let ends: Vec<&Expr> = ends.values().collect();
                        let [r, c] = ends[..] else {
                            self.error(
                                codes::STREAM_BLOCK,
                                format!("the block of a `{what}` is `n` or `[rows, columns]`"),
                                o[1].map_or(span, |b| b.span),
                            );
                            return Ty::Unknown;
                        };
                        // a number written or a module parameter bound at build
                        let rows = match count_of(cx, r) {
                            Count::Known(n) if n > 0 => n,
                            _ => {
                                self.error(
                                    codes::WRONG_TYPE,
                                    format!(
                                        "the rows of a `{what}`'s block `[r, c]` are a number \
                                         written in the text or a module parameter bound at \
                                         build (one bound at load or at the invocation, or \
                                         `group.size`, there is not built)"
                                    ),
                                    r.span,
                                );
                                return Ty::Unknown;
                            }
                        };
                        self.note_type(r, &Ty::Scalar(ScalarType::U32));
                        (vec![o[0], Some(c), Some(r)], rows)
                    }
                    _ => (o, 1),
                };
                self.note_op(span, kind, &o);
                let v = self.operand(cx, o[0], &view, "the view");
                self.operand(cx, o[1], &Ty::AnyInt, "the block");
                let format = match v {
                    Ty::View { format, .. } => format,
                    _ => None,
                };
                let block = o[1].map_or(Count::Unknown, |b| count_of(cx, b));
                if let (Some(b), Count::Unknown | Count::GroupSize) = (o[1], block) {
                    self.error(
                        codes::STREAM_BLOCK,
                        format!(
                            "the block of a `{what}` is a number written in the text or a module \
                             parameter, known before the run"
                        ),
                        b.span,
                    );
                    return Ty::Unknown;
                }
                let shape = Shape {
                    rows: Count::Known(rows),
                    cols: block,
                };
                if what == "sink" {
                    if rows != 1 {
                        self.error(
                            codes::NOT_BUILT,
                            "a `sink` of blocks of several rows `[r, c]` is named in the \
                             language and not built yet"
                                .into(),
                            span,
                        );
                        return Ty::Unknown;
                    }
                    return Ty::Sink(format, shape);
                }
                let elem = match shape {
                    Shape::TILE => Ty::Tile(format),
                    shape => Ty::Block(format, shape),
                };
                let tiles = match v {
                    Ty::View { tiles, .. } => tiles,
                    _ => Count::Unknown,
                };
                Ty::Stream(Box::new(elem), tiles)
            }
            "zip" => {
                if let Some(arg) = args.values().find(|a| a.name.is_some()) {
                    self.error(
                        codes::OPERAND_FORM,
                        "the streams of `zip` are given by position".into(),
                        arg.span,
                    );
                    return Ty::Unknown;
                }
                if args.len() < 2 {
                    self.error(
                        codes::MISSING_OPERAND,
                        "`zip` takes two streams or more".into(),
                        span,
                    );
                    return Ty::Unknown;
                }
                let parts: Vec<Option<&Expr>> = args.values().map(|a| Some(&a.value)).collect();
                self.note_op(span, OpKind::ForZip, &parts);
                let mut streams = Vec::new();
                let mut counts = Vec::new();
                for arg in args.values() {
                    let ty = self.expr(cx, &arg.value);
                    match &ty {
                        // its blocks, where the compile knows its tiles and its block
                        Ty::Stream(elem, Count::Known(n)) => {
                            let blocks = match &**elem {
                                Ty::Block(_, s) => match (s.rows, s.cols) {
                                    (Count::Known(1), Count::Known(c)) if c > 0 => {
                                        Some(n.div_ceil(c))
                                    }
                                    (Count::Known(r), Count::Known(c)) if r * c > 0 => {
                                        Some(n / (r * c))
                                    }
                                    _ => None,
                                },
                                _ => Some(*n),
                            };
                            counts.extend(blocks);
                        }
                        Ty::Stream(_, _) | Ty::Unknown => {}
                        _ => self.error(
                            codes::WRONG_TYPE,
                            format!("`zip` takes streams, not {ty}"),
                            arg.value.span,
                        ),
                    }
                    streams.push(ty);
                }
                if counts.windows(2).any(|w| w[0] != w[1]) {
                    let counts: Vec<String> = counts.iter().map(u64::to_string).collect();
                    self.error(
                        codes::OUT_OF_RANGE,
                        format!(
                            "`zip` walks streams of different lengths: {} blocks",
                            counts.join(", ")
                        ),
                        span,
                    );
                    return Ty::Unknown;
                }
                Ty::Zip(streams)
            }
            "reduce" => {
                let form = [
                    pos("x"),
                    named("with"),
                    named_opt("within"),
                    named_opt("among"),
                    named_opt("publish"),
                ];
                let Some(o) = self.operands(what, span, args, &form) else {
                    return Ty::Unknown;
                };
                if let Some(c) = o[1].and_then(|w| combiner(cx, w)) {
                    self.note_op(span, OpKind::ReduceGroup(c), &[o[0], o[2], o[3], o[4]]);
                }
                let x = self.operand(cx, o[0], &Ty::Unknown, "the value");
                self.combination(cx, o[1]);
                self.operand(cx, o[2], &Ty::SubGroups, "`within`");
                self.operand(cx, o[3], &Ty::Leaders, "`among`");
                self.view_list(cx, o[4]);
                x
            }
            "all_reduce" => {
                let form = [pos("x"), named("with"), named_opt("publish")];
                let Some(o) = self.operands(what, span, args, &form) else {
                    return Ty::Unknown;
                };
                if let Some(c) = o[1].and_then(|w| combiner(cx, w)) {
                    self.note_op(span, OpKind::AllReduce(c), &[o[0], o[2]]);
                }
                let x = self.operand(cx, o[0], &Ty::Unknown, "the value");
                if cx.kind == Kind::Control {
                    self.replicated(x.clone(), o[0], what);
                }
                self.combination(cx, o[1]);
                self.view_list(cx, o[2]);
                if cx.kind == Kind::Control {
                    Ty::Void
                } else {
                    x
                }
            }
            "gather" => {
                let Some(o) = self.operands(what, span, args, &[pos("x")]) else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::Gather, &o);
                match self.operand(cx, o[0], &Ty::Unknown, "the value") {
                    Ty::Tile(f) => Ty::Block(f, Shape::row(Count::GroupSize)),
                    _ => Ty::Unknown,
                }
            }
            "broadcast" => {
                let form = [named("along"), named_opt("sender")];
                let Some(o) = self.operands(what, span, args, &form) else {
                    return Ty::Unknown;
                };
                if !cx.grid {
                    self.error(
                        codes::BROADCAST_ON_LINE,
                        "a broadcast needs a handler on a grid (`@on(grid[R, C])`)".into(),
                        span,
                    );
                    return Ty::Unknown;
                }
                self.word_operand(o[0], &["row", "col"], "`along`");
                match o[0]
                    .and_then(as_name)
                    .filter(|n| !n.quoted)
                    .map(|n| n.name.as_str())
                {
                    Some("row") => self.note_op(span, OpKind::Broadcast(Axis::Row), &o[1..]),
                    Some("col") => self.note_op(span, OpKind::Broadcast(Axis::Col), &o[1..]),
                    _ => {}
                }
                self.operand(cx, o[1], &Ty::AnyInt, "the sender");
                self.made.push(span);
                Ty::Broadcast((span.start, span.end))
            }
            "sync" => {
                let Some(o) = self.operands(what, span, args, &[named("publish")]) else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::Sync, &o);
                self.view_list(cx, o[0]);
                Ty::Void
            }
            "send" | "offer" => {
                let form: &[Operand] = if what == "send" {
                    &[pos("message"), named("to"), named_opt("publish")]
                } else {
                    &[pos("message"), named("to")]
                };
                let Some(o) = self.operands(what, span, args, form) else {
                    return Ty::Unknown;
                };
                let kind = if what == "send" {
                    OpKind::Send
                } else {
                    OpKind::Offer
                };
                self.note_op(span, kind, &o);
                let m = self.operand(cx, o[0], &Ty::Unknown, "the message");
                if !matches!(m, Ty::Message(_) | Ty::Unknown) {
                    self.error(
                        codes::WRONG_TYPE,
                        format!("`{what}` takes a message, not {m}"),
                        o[0].map(|e| e.span).unwrap_or(span),
                    );
                }
                self.operand(cx, o[1], &Ty::AnyInt, "the rank");
                if what == "send" {
                    self.view_list(cx, o[2]);
                } else if let Some(to) = o[1] {
                    self.offer_peer(cx, to, span);
                }
                Ty::Bool
            }
            "forward" => self.forward(cx, args, span),
            "invoke" => self.invoke(cx, args, span),
            "outcome" => {
                let Some(o) = self.operands(what, span, args, &[pos("h")]) else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::Outcome, &o);
                match self.operand(cx, o[0], &Ty::Handle(None), "the handle") {
                    Ty::Handle(ret) => Ty::Outcome(ret),
                    _ => Ty::Unknown,
                }
            }
            "group" => {
                let form: &[Operand] = if args.len() == 1 {
                    &[pos("count")]
                } else {
                    &[pos("first"), pos("count")]
                };
                let Some(o) = self.operands(what, span, args, form) else {
                    return Ty::Unknown;
                };
                let parts = if o.len() == 1 {
                    vec![None, o[0]]
                } else {
                    o.clone()
                };
                self.note_op(span, OpKind::MakeGroup, &parts);
                for e in o {
                    self.operand(cx, e, &Ty::AnyInt, "a count of workers");
                }
                Ty::Group
            }
            "reduce_to" => {
                let Some(o) =
                    self.operands(what, span, args, &[pos("t"), named("to"), named("with")])
                else {
                    return Ty::Unknown;
                };
                if let Some(c) = o[2].and_then(|w| combiner(cx, w)) {
                    self.note_op(span, OpKind::ReduceTo(c), &o[..2]);
                }
                let t = self.operand(cx, o[0], &Ty::Unknown, "the tensor");
                self.replicated(t, o[0], what);
                self.operand(cx, o[1], &Ty::AnyInt, "the island");
                self.combination(cx, o[2]);
                Ty::Void
            }
            "broadcast_from" => {
                let Some(o) = self.operands(what, span, args, &[pos("t"), named("from")]) else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::BroadcastFrom, &o);
                let t = self.operand(cx, o[0], &Ty::Unknown, "the tensor");
                self.replicated(t, o[0], what);
                self.operand(cx, o[1], &Ty::AnyInt, "the island");
                Ty::Void
            }
            _ => Ty::Unknown,
        }
    }

    /// An operand that is a tile, or a scalar standing for one.
    /// `matmul(a, b)` in a function on blocks (R45): the product of the two blocks as
    /// matrices of tiles, a `[r, k]` by a `[k, c]` giving a `[r, c]` (a tile is
    /// `[1, 1]`); a's columns other than b's rows are `K102`.
    fn block_product(&mut self, cx: &mut Cx, a: Option<&Expr>, b: Option<&Expr>, span: Span) -> Ty {
        let mut operand = |this: &mut Self, e: Option<&Expr>, what: &str| {
            let ty = this.tile_operand(cx, e);
            match ty {
                Ty::Tile(f) => Some((f, Shape::TILE)),
                Ty::Block(f, s) => Some((f, s)),
                Ty::Unknown => None,
                other => {
                    this.error(
                        codes::WRONG_TYPE,
                        format!("{what} of `matmul` is {other}, where a tile or a block is asked"),
                        e.map_or(span, |e| e.span),
                    );
                    None
                }
            }
        };
        let left = operand(self, a, "the left operand");
        let right = operand(self, b, "the right operand");
        let (Some((f, l)), Some((_, r))) = (left, right) else {
            return Ty::Unknown;
        };
        if !Shape::row(l.cols).agrees(Shape::row(r.rows)) {
            self.error(
                codes::WRONG_TYPE,
                format!(
                    "`matmul` of a block {l} by a block {r}: the left's columns of tiles are the \
                     right's rows"
                ),
                span,
            );
            return Ty::Unknown;
        }
        match (Shape {
            rows: l.rows,
            cols: r.cols,
        }) {
            Shape::TILE => Ty::Tile(f),
            shape => Ty::Block(f, shape),
        }
    }

    pub(super) fn tile_operand(&mut self, cx: &mut Cx, e: Option<&Expr>) -> Ty {
        let Some(e) = e else {
            return Ty::Unknown;
        };
        let ty = self.expr(cx, e);
        let block = self.on_blocks && matches!(ty, Ty::Block(..));
        if !(ty.is_tile_operand() || block || matches!(ty, Ty::Unknown | Ty::Reduced(..))) {
            self.error(
                codes::WRONG_TYPE,
                format!("a tile operation takes a tile, not {ty}"),
                e.span,
            );
            return Ty::Unknown;
        }
        ty
    }

    /// `K208`: `items`, `records` and `words` only on a tensor whose format has stated
    /// bytes (an integer type, `f32` or `f64`).
    pub(super) fn stated_bytes(&mut self, format: Option<Format>, what: &str, span: Span) -> bool {
        match format {
            Some(f) if f.bytes().is_none() => {
                self.error(
                    codes::NO_STATED_BYTES,
                    format!(
                        "`{what}` reads the bytes of a tensor, and `{f}` has no stated bytes: \
                         only an integer type, `f32` or `f64`"
                    ),
                    span,
                );
                false
            }
            _ => true,
        }
    }

    /// The result of an element-by-element compute function from its operands' types.
    pub(super) fn elementwise_of(
        &mut self,
        what: &str,
        exprs: &[Option<&Expr>],
        operands: &[Ty],
        span: Span,
    ) -> Ty {
        let ty = match self.elementwise(what, operands, span) {
            Some(Ty::AnyNumber) | None => tile_of(operands),
            Some(ty) => ty,
        };
        self.beside(exprs, &ty);
        ty
    }

    pub(super) fn replicated(&mut self, ty: Ty, e: Option<&Expr>, what: &str) {
        let Some(e) = e else {
            return;
        };
        if !matches!(
            ty,
            Ty::Tensor {
                replicated: true,
                ..
            } | Ty::Unknown
        ) {
            self.error(
                codes::WRONG_TYPE,
                format!("`{what}` in a control program takes a tensor marked `@replicated`"),
                e.span,
            );
        }
    }

    pub(super) fn make_message(
        &mut self,
        cx: &mut Cx,
        name: &Ident,
        fields: &List<crate::syntax::ast::FieldInit>,
        span: Span,
    ) -> Ty {
        if !self.allowed(
            cx,
            Family::Dynamic,
            &format!("{} {{ ... }}", name.name),
            span,
        ) {
            return Ty::Unknown;
        }
        let Some(decl) = cx.env.message(cx.program.as_deref(), &name.name).cloned() else {
            self.error(
                codes::UNKNOWN_NAME,
                format!("`{}` is not a message type", name.name),
                name.span,
            );
            return Ty::Unknown;
        };
        if let Some(place) = cx.env.ix.message_ref(cx.program.as_deref(), &name.name) {
            self.note_name(name.span, Name::Message(place));
        }
        if let Some(place) = cx.env.ix.message_ref(cx.program.as_deref(), &name.name) {
            let mut parts: Vec<Option<&Expr>> = vec![None; decl.len()];
            for f in fields.values() {
                if let Some(at) = decl.iter().position(|(n, _)| *n == f.name.name) {
                    parts[at] = parts[at].or(Some(&f.value));
                }
            }
            self.note_op(span, OpKind::MakeMessage(place), &parts);
        }
        let mut seen = HashSet::new();
        for f in fields.values() {
            let ty = self.expr(cx, &f.value);
            match decl.iter().position(|(n, _)| *n == f.name.name) {
                Some(at) => {
                    let want = decl[at].1.clone();
                    self.note_name(f.name.span, Name::Field(at as u32));
                    if !seen.insert(f.name.name.clone()) {
                        self.error(
                            codes::NAME_TWICE,
                            format!("the field `{}` is given twice", f.name.name),
                            f.span,
                        );
                    }
                    self.fit(
                        &f.value,
                        &ty,
                        &want,
                        &format!("the field `{}`", f.name.name),
                    );
                }
                None => self.error(
                    codes::WRONG_TYPE,
                    format!("the message {} has no field `{}`", name.name, f.name.name),
                    f.name.span,
                ),
            }
        }
        let missing: Vec<&str> = decl
            .iter()
            .filter(|(n, _)| !seen.contains(n))
            .map(|(n, _)| n.as_str())
            .collect();
        if !missing.is_empty() {
            self.error(
                codes::MESSAGE_FIELD_MISSING,
                format!(
                    "the message {} is made without its field `{}`",
                    name.name,
                    missing.join("`, `")
                ),
                span,
            );
        }
        Ty::Message(name.name.clone())
    }

    pub(super) fn forward(&mut self, cx: &mut Cx, args: &List<Arg>, span: Span) -> Ty {
        let form = [
            pos("m"),
            named_opt("with"),
            named("to"),
            named_opt("publish"),
        ];
        let Some(o) = self.operands("forward", span, args, &form) else {
            return Ty::Unknown;
        };
        self.note_op(span, OpKind::Forward, &o);
        let mut message = None;
        if let Some(m) = o[0] {
            match as_name(m).and_then(|n| cx.lookup(&n.name).cloned().map(|l| (n, l))) {
                Some((n, l)) if l.binding == Name::Arrived => {
                    self.note_name(n.span, Name::Arrived);
                    self.note_type(m, &l.ty);
                    if let Ty::Message(name) = l.ty {
                        message = Some(name);
                    }
                }
                _ => {
                    self.error(
                        codes::WRONG_TYPE,
                        "`forward` takes the message this body handles".into(),
                        m.span,
                    );
                    return Ty::Unknown;
                }
            }
        }
        if let Some(with) = o[1] {
            match &with.kind {
                ExprKind::Init { ty: None, fields } => {
                    if let Some(m) = &message {
                        self.note_type(with, &Ty::Message(m.clone()));
                    }
                    let decl = message
                        .as_deref()
                        .and_then(|m| cx.env.message(cx.program.as_deref(), m))
                        .cloned();
                    for f in fields.values() {
                        let ty = self.expr(cx, &f.value);
                        let Some(decl) = &decl else {
                            continue;
                        };
                        match decl.iter().position(|(n, _)| *n == f.name.name) {
                            Some(at) => {
                                let want = &decl[at].1;
                                self.note_name(f.name.span, Name::Field(at as u32));
                                self.fit(
                                    &f.value,
                                    &ty,
                                    want,
                                    &format!("the field `{}`", f.name.name),
                                );
                            }
                            None => self.error(
                                codes::WRONG_TYPE,
                                format!("the message has no field `{}`", f.name.name),
                                f.name.span,
                            ),
                        }
                    }
                }
                _ => self.error(
                    codes::WRONG_TYPE,
                    "`with` gives fields: `with = { field: value }`".into(),
                    with.span,
                ),
            }
        }
        self.operand(cx, o[2], &Ty::AnyInt, "the rank");
        self.view_list(cx, o[3]);
        Ty::Void
    }

    pub(super) fn invoke(&mut self, cx: &mut Cx, args: &List<Arg>, span: Span) -> Ty {
        let form = [
            pos("call"),
            named("on"),
            named_opt("after"),
            named_opt("after_end"),
        ];
        let Some(o) = self.operands("invoke", span, args, &form) else {
            return Ty::Unknown;
        };
        let mut ret = None;
        if let Some(call) = o[0] {
            let ExprKind::Call { callee, args } = &call.kind else {
                self.error(
                    codes::WRONG_TYPE,
                    "`invoke` takes a call of a program's handler: `P.h(...)`".into(),
                    call.span,
                );
                return Ty::Unknown;
            };
            let ExprKind::Member { base, name } = &callee.kind else {
                self.error(
                    codes::WRONG_TYPE,
                    "`invoke` takes a call of a program's handler: `P.h(...)`".into(),
                    call.span,
                );
                return Ty::Unknown;
            };
            let Some(program) = as_name(base) else {
                self.error(
                    codes::WRONG_TYPE,
                    "`invoke` takes a call of a program's handler: `P.h(...)`".into(),
                    call.span,
                );
                return Ty::Unknown;
            };
            let Some(info) = cx.env.programs.get(&program.name) else {
                self.error(
                    codes::UNKNOWN_NAME,
                    format!("`{}` is not a program", program.name),
                    program.span,
                );
                return Ty::Unknown;
            };
            let Some(target) = info.invocables.get(&name.name).cloned() else {
                self.error(
                    codes::UNKNOWN_NAME,
                    format!(
                        "`{}` is not a handler, run or chain of `{}`",
                        name.name, program.name
                    ),
                    name.span,
                );
                return Ty::Unknown;
            };
            if let Some(&place) = cx.env.ix.programs.get(&program.name) {
                self.note_name(program.span, Name::Program(place));
            }
            if let Some(place) = cx.env.ix.invocable_of(&program.name, &name.name) {
                if let Some(entry) = entry_of(&place) {
                    let mut parts = vec![o[1], o[2], o[3]];
                    parts.extend(args.values().map(|a| Some(&a.value)));
                    self.note_op(span, OpKind::Invoke(entry), &parts);
                }
                self.note_name(name.span, place);
            }
            let what = format!("{}.{}", program.name, name.name);
            self.positional_call(cx, &what, call.span, args, &target.params);
            ret = target.ret;
            self.note_type(call, &ret.clone().unwrap_or(Ty::Void));
        }
        self.operand(cx, o[1], &Ty::Group, "`on`");
        for list in [o[2], o[3]].into_iter().flatten() {
            match &list.kind {
                ExprKind::List(items) => {
                    for item in items.values() {
                        let ty = self.expr(cx, item);
                        if !matches!(ty, Ty::Handle(_) | Ty::Unknown) {
                            self.error(
                                codes::WRONG_TYPE,
                                format!("`after` lists handles, not {ty}"),
                                item.span,
                            );
                        }
                    }
                    self.note_type(list, &Ty::List);
                }
                _ => self.error(
                    codes::WRONG_TYPE,
                    "`after` is a list of handles".into(),
                    list.span,
                ),
            }
        }
        Ty::Handle(ret.map(Box::new))
    }

    pub(super) fn method(
        &mut self,
        cx: &mut Cx,
        base: &Expr,
        name: &Ident,
        args: &List<Arg>,
        span: Span,
    ) -> Ty {
        let what = name.name.as_str();
        // a method between backticks is the module's, and the module has no methods
        // (R50)
        if name.quoted {
            self.unknown(name);
            return Ty::Unknown;
        }
        if let Some(word) = as_name(base) {
            // a bare `group` or `island` is the language's, whatever the module names;
            // the module's value of that name is written between backticks (R50)
            if !word.quoted {
                match word.name.as_str() {
                    "group" => return self.group_method(cx, word, name, args, span),
                    "island" => {
                        if !self.allowed(cx, Family::IslandQuery, &format!("island.{what}"), span) {
                            return Ty::Unknown;
                        }
                        if cx.kind == Kind::Control && matches!(what, "crosses" | "hops") {
                            self.error(
                                codes::CONTROL_NOT_VALUES,
                                format!(
                                    "`island.{what}` asks about the asking worker's own island, \
                                     and a control program runs on no worker; `island.count` \
                                     and `island.of(id)` are allowed here"
                                ),
                                span,
                            );
                            return Ty::Unknown;
                        }
                        self.note_word(word);
                        let Some(o) = self.operands(what, span, args, &[pos("id")]) else {
                            return Ty::Unknown;
                        };
                        let kind = match what {
                            "of" => Some(OpKind::IslandOf),
                            "hops" => Some(OpKind::IslandHops),
                            "crosses" => Some(OpKind::IslandCrosses),
                            _ => None,
                        };
                        if let Some(kind) = kind {
                            self.note_op(span, kind, &o);
                        }
                        self.operand(cx, o[0], &Ty::AnyInt, "a worker's id");
                        return match what {
                            "of" | "hops" => Ty::Scalar(ScalarType::U32),
                            "crosses" => Ty::Bool,
                            _ => self.no_member("island", name),
                        };
                    }
                    _ => {}
                }
            }
            // A state member's own operations.
            if cx.lookup(&word.name).is_none() {
                if let Some(member) = cx.state(&word.name).cloned() {
                    if matches!(what, "fill" | "zero" | "bump") {
                        if !self.allowed(cx, Family::Worker, &format!("s.{what}"), span) {
                            return Ty::Unknown;
                        }
                        if let Some(state) = cx.member_name(&word.name) {
                            self.note_name(word.span, state);
                        }
                        return self.state_method(cx, word, &member.ty, name, args, span);
                    }
                }
            }
        }
        let ty = self.expr(cx, base);
        match (&ty, what) {
            (Ty::Unknown, _) => {
                for arg in args.values() {
                    self.expr(cx, &arg.value);
                }
                Ty::Unknown
            }
            (Ty::Tensor { format, .. }, "tiles") => {
                if !self.allowed(cx, Family::Worker, "t.tiles", span) {
                    return Ty::Unknown;
                }
                let Some(o) = self.operands("tiles", span, args, &[pos("first"), pos("n")]) else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::Tiles, &[Some(base), o[0], o[1]]);
                self.operand(cx, o[0], &Ty::AnyInt, "the first tile");
                self.operand(cx, o[1], &Ty::AnyInt, "the count");
                let tiles = match o[1].and_then(as_int) {
                    Some(n) => Count::Known(n),
                    None => Count::Unknown,
                };
                Ty::View {
                    format: *format,
                    tiles,
                }
            }
            (Ty::Tensor { format, .. }, "items") => {
                if !self.allowed(cx, Family::Worker, "t.items", span) {
                    return Ty::Unknown;
                }
                let positional = args.values().filter(|a| a.name.is_none()).count();
                let form: &[Operand] = if positional == 0 {
                    &[named("size")]
                } else {
                    &[pos("first"), pos("n"), named("size")]
                };
                let Some(o) = self.operands("items", span, args, form) else {
                    return Ty::Unknown;
                };
                let parts = if positional == 0 {
                    [Some(base), None, None, o[0]]
                } else {
                    [Some(base), o[0], o[1], o[2]]
                };
                self.note_op(span, OpKind::Items, &parts);
                if !self.stated_bytes(*format, "items", span) {
                    return Ty::Unknown;
                }
                for e in &o {
                    self.operand(cx, *e, &Ty::AnyInt, "an operand of `items`");
                }
                let count = match (positional, o.get(1).copied().flatten().and_then(as_int)) {
                    (2, Some(n)) => Count::Known(n),
                    _ => Count::Unknown,
                };
                let size = parts[3].map_or(Count::Unknown, |b| count_of(cx, b));
                Ty::Items(count, size)
            }
            (Ty::View { format, tiles }, "words") => {
                let Some(_) = self.operands("words", span, args, &[]) else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::Words, &[Some(base)]);
                if tiles.more_than_one() {
                    self.error(
                        codes::WORDS_MANY_TILES,
                        "`.words()` reads one tile, and this view holds more".into(),
                        span,
                    );
                    return Ty::Unknown;
                }
                if !self.stated_bytes(*format, "words", span) {
                    return Ty::Unknown;
                }
                Ty::Words(format.and_then(Format::as_scalar))
            }
            (Ty::View { format, tiles }, "records") => {
                let Some(o) = self.operands("records", span, args, &[pos("R"), named("count")])
                else {
                    return Ty::Unknown;
                };
                if tiles.more_than_one() {
                    self.error(
                        codes::WORDS_MANY_TILES,
                        "`.records()` reads one tile, and this view holds more".into(),
                        span,
                    );
                    return Ty::Unknown;
                }
                if !self.stated_bytes(*format, "records", span) {
                    return Ty::Unknown;
                }
                self.operand(cx, o[1], &Ty::AnyInt, "the count");
                let Some(r) = o[0] else {
                    return Ty::Unknown;
                };
                let name = match as_name(r) {
                    Some(n) if cx.env.record(cx.program.as_deref(), &n.name).is_some() => {
                        if let Some(place) = cx.env.ix.record_ref(cx.program.as_deref(), &n.name) {
                            self.note_name(n.span, Name::Record(place));
                            self.note_op(span, OpKind::Records(place), &[Some(base), o[1]]);
                        }
                        n.name.clone()
                    }
                    Some(n) => {
                        self.error(
                            codes::UNKNOWN_NAME,
                            format!("`{}` is not a record type", n.name),
                            n.span,
                        );
                        return Ty::Unknown;
                    }
                    None => {
                        self.error(
                            codes::WRONG_TYPE,
                            "`.records` takes a record type".into(),
                            r.span,
                        );
                        return Ty::Unknown;
                    }
                };
                let count = o[1].and_then(as_int);
                let tile_bytes = match (
                    cx.env.tile,
                    format.and_then(Format::bytes).map(|b| b as u64),
                ) {
                    (Some((rows, cols)), Some(width)) => Some(rows * cols * width),
                    _ => None,
                };
                let record = cx.env.record_bytes.get(&name).copied();
                if let (Some(c), Some(tile), Some(record)) = (count, tile_bytes, record) {
                    if c.saturating_mul(record) > tile {
                        self.error(
                            codes::OUT_OF_RANGE,
                            format!(
                                "{c} records {name} of {record} bytes do not fit one tile of \
                                 {tile} bytes"
                            ),
                            span,
                        );
                        return Ty::Unknown;
                    }
                }
                let count = count.map_or(Count::Unknown, Count::Known);
                Ty::Records(name, count)
            }
            (Ty::Items(count, _), "put") => {
                let Some(o) = self.operands("put", span, args, &[pos("src"), named("at")]) else {
                    return Ty::Unknown;
                };
                if let Some(Name::State { member, .. }) =
                    o[0].and_then(as_name).and_then(|n| cx.member_name(&n.name))
                {
                    self.note_op(
                        span,
                        OpKind::Put(MemberId(member)),
                        &[Some(base), o[0], o[1]],
                    );
                }
                self.operand(cx, o[0], &Ty::Unknown, "the source");
                self.operand(cx, o[1], &Ty::AnyInt, "the item");
                if let (Count::Known(n), Some(i)) = (count, o[1].and_then(as_int)) {
                    if i >= *n {
                        self.error(
                            codes::OUT_OF_RANGE,
                            format!("the item {i} is past the end: the view holds {n}"),
                            span,
                        );
                    }
                }
                Ty::Void
            }
            (Ty::Sink(_, _), "push") => {
                let Some(o) = self.operands("push", span, args, &[pos("t")]) else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::Push, &[Some(base), o[0]]);
                let before = self.diagnostics.len();
                let t = self.operand(cx, o[0], &Ty::Unknown, "the block");
                if let Ty::Reduced(..) = t {
                    self.expect_fit(&t, &Ty::Tile(None), span, "the block pushed");
                } else if !matches!(t, Ty::Tile(_) | Ty::Block(_, _) | Ty::Unknown) {
                    self.error(
                        codes::WRONG_TYPE,
                        format!("`push` writes a tile, not {t}"),
                        span,
                    );
                } else if let (Ty::Sink(_, want), true) = (&ty, self.diagnostics.len() == before) {
                    // a sink takes blocks of its own shape; a shape the text does not
                    // show is the run's to check, and an operand already refused says
                    // nothing more here
                    let got = match &t {
                        Ty::Tile(_) => Some(Shape::TILE),
                        Ty::Block(_, s) => Some(*s),
                        _ => None,
                    };
                    let known =
                        |s: Shape| matches!((s.rows, s.cols), (Count::Known(_), Count::Known(_)));
                    if let Some(got) = got.filter(|g| known(*g) && known(*want) && g != want) {
                        let show = |s: Shape| match (s.rows, s.cols) {
                            (Count::Known(1), Count::Known(c)) => format!("{c}"),
                            (Count::Known(r), Count::Known(c)) => format!("[{r}, {c}]"),
                            _ => String::new(),
                        };
                        self.error(
                            codes::WRONG_TYPE,
                            format!(
                                "a sink of `block = {}` takes blocks of that shape, and this \
                                 is a block of {}",
                                show(*want),
                                show(got)
                            ),
                            span,
                        );
                    }
                }
                // K206 (a stream writing where it reads) is Verify's, on the derived effects
                Ty::Void
            }
            (Ty::Tile(f), "col" | "row") => {
                if !self.allowed(cx, Family::Compute, what, span) {
                    return Ty::Unknown;
                }
                let Some(_) = self.operands(what, span, args, &[]) else {
                    return Ty::Unknown;
                };
                self.note_op(span, spread(what), &[Some(base)]);
                Ty::Tile(*f)
            }
            // in a function on blocks (R45), on each tile of the block
            (Ty::Block(f, shape), "col" | "row") if self.on_blocks => {
                if !self.allowed(cx, Family::Compute, what, span) {
                    return Ty::Unknown;
                }
                let Some(_) = self.operands(what, span, args, &[]) else {
                    return Ty::Unknown;
                };
                self.note_op(span, spread(what), &[Some(base)]);
                Ty::Block(*f, *shape)
            }
            (Ty::Reduced(format, axis, shape), "col" | "row") => {
                if !self.allowed(cx, Family::Compute, what, span) {
                    return Ty::Unknown;
                }
                let Some(_) = self.operands(what, span, args, &[]) else {
                    return Ty::Unknown;
                };
                let spreads = matches!((axis, what), (Axis::Row, "col") | (Axis::Col, "row"));
                if !spreads {
                    let other = if what == "col" { "row" } else { "col" };
                    self.error(
                        codes::REDUCED_VALUE,
                        format!("this is {ty}; `.{other}()` spreads it into a tile"),
                        span,
                    );
                    return Ty::Unknown;
                }
                self.note_op(span, spread(what), &[Some(base)]);
                // a block's values spread back over the block's shape
                if *shape == Shape::TILE {
                    Ty::Tile(*format)
                } else {
                    Ty::Block(*format, *shape)
                }
            }
            (Ty::Broadcast(at), "cast") => {
                let at = *at;
                let Some(o) = self.operands("cast", span, args, &[pos("block"), named_opt("word")])
                else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::Cast, &[Some(base), o[0], o[1]]);
                let sent = self.operand(cx, o[0], &Ty::Tile(None), "the block");
                self.operand(cx, o[1], &Ty::AnyInt, "the word");
                if let (Ty::Tile(Some(f)) | Ty::Block(Some(f), _), Some(block)) = (sent, o[0]) {
                    self.cast_format(at, f, block.span);
                }
                Ty::Void
            }
            (Ty::Broadcast(at), "arrive") => {
                let at = *at;
                let Some(_) = self.operands("arrive", span, args, &[]) else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::ForArrive, &[Some(base)]);
                Ty::Arrive(at)
            }
            _ => self.no_member(&ty.to_string(), name),
        }
    }

    pub(super) fn group_method(
        &mut self,
        cx: &mut Cx,
        word: &Ident,
        name: &Ident,
        args: &List<Arg>,
        span: Span,
    ) -> Ty {
        let what = name.name.as_str();
        if !self.allowed(cx, Family::Worker, &format!("group.{what}"), span) {
            return Ty::Unknown;
        }
        self.note_word(word);
        match what {
            "id" => {
                let Some(o) = self.operands("group.id", span, args, &[pos("rank")]) else {
                    return Ty::Unknown;
                };
                self.note_op(span, OpKind::GroupId, &o);
                self.operand(cx, o[0], &Ty::AnyInt, "the rank");
                Ty::Scalar(ScalarType::U32)
            }
            "split" | "leaders" => {
                let full = format!("group.{what}");
                let Some(o) =
                    self.operands(&full, span, args, &[named_opt("parts"), named_opt("size")])
                else {
                    return Ty::Unknown;
                };
                match (o[0], o[1]) {
                    (Some(_), Some(_)) => {
                        self.error(
                            codes::OPERAND_FORM,
                            format!("`{full}` takes `parts` or `size`, not both"),
                            span,
                        );
                        return Ty::Unknown;
                    }
                    (None, None) => {
                        self.error(
                            codes::MISSING_OPERAND,
                            format!("`{full}` is not given `parts` or `size`"),
                            span,
                        );
                        return Ty::Unknown;
                    }
                    _ => {}
                }
                let by = if o[0].is_some() {
                    SplitBy::Parts
                } else {
                    SplitBy::Size
                };
                let kind = if what == "split" {
                    OpKind::Split(by)
                } else {
                    OpKind::Leaders(by)
                };
                self.note_op(span, kind, &[o[0].or(o[1])]);
                for e in o {
                    self.operand(cx, e, &Ty::AnyInt, "a count");
                }
                if what == "split" {
                    Ty::SubGroups
                } else {
                    Ty::Leaders
                }
            }
            _ => self.no_member("group", name),
        }
    }

    pub(super) fn state_method(
        &mut self,
        cx: &mut Cx,
        word: &Ident,
        ty: &Ty,
        name: &Ident,
        args: &List<Arg>,
        span: Span,
    ) -> Ty {
        let what = name.name.as_str();
        let member = match cx.member_name(&word.name) {
            Some(Name::State { member, .. }) => Some(MemberId(member)),
            _ => None,
        };
        match what {
            "fill" => {
                let Some(o) = self.operands("fill", span, args, &[pos("stream")]) else {
                    return Ty::Unknown;
                };
                if let Some(m) = member {
                    self.note_op(span, OpKind::Fill(m), &o);
                }
                let s = self.operand(cx, o[0], &Ty::Unknown, "the stream");
                if !matches!(s, Ty::Stream(_, _) | Ty::Unknown) {
                    self.error(
                        codes::WRONG_TYPE,
                        format!("`.fill` takes a stream, not {s}"),
                        span,
                    );
                }
                let member = match ty {
                    Ty::Tile(_) => Some(1),
                    Ty::Block(
                        _,
                        Shape {
                            rows: Count::Known(1),
                            cols: Count::Known(n),
                        },
                    ) => Some(*n),
                    _ => None,
                };
                if let (Some(member), Ty::Stream(_, Count::Known(n))) = (member, &s) {
                    if member != *n {
                        self.error(
                            codes::OUT_OF_RANGE,
                            format!(
                                "`{}` holds {member} tiles and the stream that fills it {n}",
                                word.name
                            ),
                            span,
                        );
                    }
                }
                Ty::Void
            }
            "zero" => {
                let Some(_) = self.operands("zero", span, args, &[]) else {
                    return Ty::Unknown;
                };
                if let Some(m) = member {
                    self.note_op(span, OpKind::ZeroState(m), &[]);
                }
                Ty::Void
            }
            _ => {
                let Some(o) = self.operands("bump", span, args, &[pos("i")]) else {
                    return Ty::Unknown;
                };
                if let Some(m) = member {
                    self.note_op(span, OpKind::Bump(m), &o);
                }
                self.operand(cx, o[0], &Ty::AnyInt, "the index");
                let elem = match ty {
                    Ty::Scalars(elem, _) if elem.is_integer() => (**elem).clone(),
                    _ => {
                        self.error(
                            codes::WRONG_TYPE,
                            format!(
                                "`.bump` counts in a member of integers, and `{}` is not one",
                                word.name
                            ),
                            span,
                        );
                        Ty::Unknown
                    }
                };
                elem
            }
        }
    }

    /// `T(x)`: `x` converted to the scalar type `T`. Integer to integer wraps in the new
    /// width; integer to float is its value; float to integer drops the fraction. A
    /// number written in the text takes `T` as its place, so one `T` cannot hold is
    /// `K115`.
    fn convert(&mut self, cx: &mut Cx, target: ScalarType, args: &List<Arg>, span: Span) -> Ty {
        let what = target.name();
        let Some(o) = self.operands(what, span, args, &[pos("x")]) else {
            return Ty::Unknown;
        };
        let Some(x) = o[0] else {
            return Ty::Unknown;
        };
        self.note_op(span, OpKind::Convert(target), &o);
        let ty = self.expr(cx, x);
        if let Ty::Number(_) = ty {
            self.fit(x, &ty, &Ty::Scalar(target), "the number converted");
            return Ty::Scalar(target);
        }
        match ty {
            Ty::Scalar(_) => Ty::Scalar(target),
            Ty::Unknown => Ty::Unknown,
            other => {
                self.error(
                    codes::WRONG_TYPE,
                    format!("`{what}(x)` converts a scalar, not {other}"),
                    x.span,
                );
                Ty::Unknown
            }
        }
    }
}

/// The function of one tile `name` names.
fn unary_op(name: &str) -> Option<UnaryOp> {
    Some(match name {
        "neg" => UnaryOp::Neg,
        "abs" => UnaryOp::Abs,
        "relu" => UnaryOp::Relu,
        "square" => UnaryOp::Square,
        "floor" => UnaryOp::Floor,
        "exp" => UnaryOp::Exp,
        "exp2" => UnaryOp::Exp2,
        "tanh" => UnaryOp::Tanh,
        "erf" => UnaryOp::Erf,
        "log" => UnaryOp::Log,
        "sigmoid" => UnaryOp::Sigmoid,
        "silu" => UnaryOp::Silu,
        "sqrt" => UnaryOp::Sqrt,
        "rsqrt" => UnaryOp::Rsqrt,
        "reciprocal" => UnaryOp::Reciprocal,
        _ => return None,
    })
}

/// `.col()` or `.row()`.
fn spread(what: &str) -> OpKind {
    if what == "col" {
        OpKind::SpreadCol
    } else {
        OpKind::SpreadRow
    }
}

/// What a collective's `with` combines with: a `combine` of the module, or one of
/// the provided ones.
fn combiner(cx: &Cx, e: &Expr) -> Option<Combiner> {
    let n = as_name(e)?;
    if n.quoted {
        return Some(Combiner::Func(FuncId(*cx.env.ix.funcs.get(&n.name)?)));
    }
    Some(match n.name.as_str() {
        "sum" => Combiner::Sum,
        "max" => Combiner::Max,
        "min" => Combiner::Min,
        "and" => Combiner::And,
        "or" => Combiner::Or,
        name => Combiner::Func(FuncId(*cx.env.ix.funcs.get(name)?)),
    })
}

/// The entry an invocation names, from the name of its handler, run or chain.
pub(super) fn entry_of(name: &Name) -> Option<Entry> {
    Some(match *name {
        Name::Handler { program, index } => Entry::Handler(ProgramId(program), HandlerId(index)),
        Name::Run { program, index } => Entry::Run(ProgramId(program), RunId(index)),
        Name::Chain { program, index } => Entry::Chain(ProgramId(program), ChainId(index)),
        _ => return None,
    })
}

/// The tile a compute operation yields from its operands.
pub(super) fn tile_of(operands: &[Ty]) -> Ty {
    for ty in operands {
        if let Ty::Tile(f) = ty {
            return Ty::Tile(*f);
        }
    }
    if operands.iter().any(|t| matches!(t, Ty::Unknown)) {
        return Ty::Unknown;
    }
    Ty::Tile(None)
}
