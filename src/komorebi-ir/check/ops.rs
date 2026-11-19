//! The operations of a body: their operands (by position or by name), and the type
//! each yields from its operands.

use std::collections::HashSet;

use crate::syntax::ast::{Arg, Expr, ExprKind, Ident, List, Span};
use crate::types::{Axis, Count, Format, Ty};

use super::body::{
    as_int, as_name, named, named_opt, pos, Cx, Family, Kind, Operand, PROVIDED, UNARY,
};
use super::codes;
use super::env::FuncKind;
use super::{Binding, Checker};

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
        self.expect_fit(&ty, want, e.span, what);
        ty
    }

    /// An operand that names one of `words` (`axis = row`, `along = col`).
    pub(super) fn word_operand(&mut self, e: Option<&Expr>, words: &[&str], what: &str) {
        let Some(e) = e else {
            return;
        };
        match as_name(e) {
            Some(n) if words.contains(&n.name.as_str()) => {
                self.note_name(n.span, Binding::Word);
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
        if PROVIDED.contains(&n.name.as_str()) {
            self.note_name(n.span, Binding::Word);
            return;
        }
        match cx.env.funcs.get(&n.name) {
            Some(f) if f.kind == FuncKind::Combine => self.note_name(n.span, Binding::Func),
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
        if cx.lookup(what).is_some() || cx.state(what).is_some() {
            self.error(codes::WRONG_TYPE, format!("`{what}` is not called"), n.span);
            return Ty::Unknown;
        }
        if let Some(sig) = cx.env.funcs.get(what).cloned() {
            if !self.allowed(cx, Family::Compute, what, span) {
                return Ty::Unknown;
            }
            self.note_name(n.span, Binding::Func);
            self.positional_call(cx, what, span, args, &sig.params);
            return sig.ret;
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
        self.note_name(n.span, Binding::Word);
        if what == "forward"
            && args
                .values()
                .any(|a| a.name.as_ref().is_some_and(|n| n.name == "as"))
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
                let a = self.operand(cx, o[0], &Ty::Float, "an operand");
                let b = self.operand(cx, o[1], &Ty::Float, "an operand");
                if a == Ty::Float || b == Ty::Float {
                    Ty::Float
                } else {
                    Ty::Int
                }
            }
            "range" => {
                let Some(o) = self.operands(what, span, args, &[pos("n")]) else {
                    return Ty::Unknown;
                };
                self.operand(cx, o[0], &Ty::Int, "the count");
                Ty::Range
            }
            "where" => {
                let Some(o) = self.operands(what, span, args, &[pos("c"), pos("x"), pos("y")])
                else {
                    return Ty::Unknown;
                };
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
                self.elementwise_of(what, &[x, y], span)
            }
            "maximum" | "minimum" => {
                let Some(o) = self.operands(what, span, args, &[pos("a"), pos("b")]) else {
                    return Ty::Unknown;
                };
                let a = self.tile_operand(cx, o[0]);
                let b = self.tile_operand(cx, o[1]);
                self.elementwise_of(what, &[a, b], span)
            }
            "matmul" => {
                let Some(o) = self.operands(what, span, args, &[pos("a"), pos("b")]) else {
                    return Ty::Unknown;
                };
                let a = self.operand(cx, o[0], &Ty::Tile(None), "the left operand");
                let b = self.operand(cx, o[1], &Ty::Tile(None), "the right operand");
                tile_of(&[a, b])
            }
            "reduce_sum" | "reduce_max" | "reduce_min" => {
                let Some(o) = self.operands(what, span, args, &[pos("a"), named("axis")]) else {
                    return Ty::Unknown;
                };
                let a = self.operand(cx, o[0], &Ty::Tile(None), "the operand");
                self.word_operand(o[1], &["row", "col"], "the axis");
                let axis = match o[1].and_then(as_name).map(|n| n.name.as_str()) {
                    Some("row") => Axis::Row,
                    Some("col") => Axis::Col,
                    _ => return Ty::Unknown,
                };
                match a {
                    Ty::Tile(_) => Ty::Reduced(axis),
                    _ => Ty::Unknown,
                }
            }
            _ if UNARY.contains(&what) => {
                let Some(o) = self.operands(what, span, args, &[pos("a")]) else {
                    return Ty::Unknown;
                };
                let a = self.tile_operand(cx, o[0]);
                self.elementwise_of(what, &[a], span)
            }
            "partition" => {
                let form = [pos("view"), pos("rank"), pos("size")];
                let Some(o) = self.operands(what, span, args, &form) else {
                    return Ty::Unknown;
                };
                let v = self.operand(cx, o[0], &view, "the view");
                self.operand(cx, o[1], &Ty::Int, "the rank");
                self.operand(cx, o[2], &Ty::Int, "the size");
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
                let v = self.operand(cx, o[0], &view, "the view");
                self.operand(cx, o[1], &Ty::Int, "the block");
                if what == "sink" {
                    return Ty::Sink;
                }
                let format = match v {
                    Ty::View { format, .. } => format,
                    _ => None,
                };
                let elem = match o[1].and_then(as_int) {
                    Some(1) => Ty::Tile(format),
                    Some(n) => Ty::Block(format, Count::Known(n)),
                    None => Ty::Unknown,
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
                let mut streams = Vec::new();
                let mut counts = Vec::new();
                for arg in args.values() {
                    let ty = self.expr(cx, &arg.value);
                    match &ty {
                        Ty::Stream(_, Count::Known(n)) => counts.push(*n),
                        Ty::Stream(_, Count::Unknown) | Ty::Unknown => {}
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
                            "`zip` walks streams of different lengths: {} tiles",
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
                let x = self.operand(cx, o[0], &Ty::Unknown, "the value");
                self.combination(cx, o[1]);
                self.operand(cx, o[2], &Ty::Parts, "`within`");
                self.operand(cx, o[3], &Ty::Parts, "`among`");
                self.view_list(cx, o[4]);
                x
            }
            "all_reduce" => {
                let form = [pos("x"), named("with"), named_opt("publish")];
                let Some(o) = self.operands(what, span, args, &form) else {
                    return Ty::Unknown;
                };
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
                match self.operand(cx, o[0], &Ty::Unknown, "the value") {
                    Ty::Tile(f) => Ty::Block(f, Count::Unknown),
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
                self.operand(cx, o[1], &Ty::Int, "the sender");
                Ty::Broadcast
            }
            "sync" => {
                let Some(o) = self.operands(what, span, args, &[named("publish")]) else {
                    return Ty::Unknown;
                };
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
                let m = self.operand(cx, o[0], &Ty::Unknown, "the message");
                if !matches!(m, Ty::Message(_) | Ty::Unknown) {
                    self.error(
                        codes::WRONG_TYPE,
                        format!("`{what}` takes a message, not {m}"),
                        o[0].map(|e| e.span).unwrap_or(span),
                    );
                }
                self.operand(cx, o[1], &Ty::Int, "the rank");
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
                let h = self.operand(cx, o[0], &Ty::Handle(None), "the handle");
                let _ = h;
                Ty::Outcome
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
                for e in o {
                    self.operand(cx, e, &Ty::Int, "a count of workers");
                }
                Ty::Group
            }
            "reduce_to" => {
                let Some(o) =
                    self.operands(what, span, args, &[pos("t"), named("to"), named("with")])
                else {
                    return Ty::Unknown;
                };
                let t = self.operand(cx, o[0], &Ty::Unknown, "the tensor");
                self.replicated(t, o[0], what);
                self.operand(cx, o[1], &Ty::Int, "the island");
                self.combination(cx, o[2]);
                Ty::Void
            }
            "broadcast_from" => {
                let Some(o) = self.operands(what, span, args, &[pos("t"), named("from")]) else {
                    return Ty::Unknown;
                };
                let t = self.operand(cx, o[0], &Ty::Unknown, "the tensor");
                self.replicated(t, o[0], what);
                self.operand(cx, o[1], &Ty::Int, "the island");
                Ty::Void
            }
            _ => Ty::Unknown,
        }
    }

    /// An operand that is a tile, or a scalar standing for one.
    pub(super) fn tile_operand(&mut self, cx: &mut Cx, e: Option<&Expr>) -> Ty {
        let Some(e) = e else {
            return Ty::Unknown;
        };
        let ty = self.expr(cx, e);
        if !(ty.is_tile_operand() || matches!(ty, Ty::Unknown | Ty::Reduced(_))) {
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
    pub(super) fn elementwise_of(&mut self, what: &str, operands: &[Ty], span: Span) -> Ty {
        match self.elementwise(what, operands, span) {
            Some(Ty::Int | Ty::Float) | None => tile_of(operands),
            Some(ty) => ty,
        }
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
        let mut seen = HashSet::new();
        for f in fields.values() {
            let ty = self.expr(cx, &f.value);
            match decl.iter().find(|(n, _)| *n == f.name.name) {
                Some((_, want)) => {
                    if !seen.insert(f.name.name.clone()) {
                        self.error(
                            codes::NAME_TWICE,
                            format!("the field `{}` is given twice", f.name.name),
                            f.span,
                        );
                    }
                    self.expect_fit(
                        &ty,
                        want,
                        f.value.span,
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
        let mut message = None;
        if let Some(m) = o[0] {
            match as_name(m).and_then(|n| cx.lookup(&n.name).cloned().map(|l| (n, l))) {
                Some((n, l)) if l.binding == Binding::Message => {
                    self.note_name(n.span, Binding::Message);
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
                    let decl = message
                        .as_deref()
                        .and_then(|m| cx.env.message(cx.program.as_deref(), m))
                        .cloned();
                    for f in fields.values() {
                        let ty = self.expr(cx, &f.value);
                        let Some(decl) = &decl else {
                            continue;
                        };
                        match decl.iter().find(|(n, _)| *n == f.name.name) {
                            Some((_, want)) => self.expect_fit(
                                &ty,
                                want,
                                f.value.span,
                                &format!("the field `{}`", f.name.name),
                            ),
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
        self.operand(cx, o[2], &Ty::Int, "the rank");
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
            self.note_name(program.span, Binding::Program);
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
        if let Some(word) = as_name(base) {
            if cx.lookup(&word.name).is_none() && cx.state(&word.name).is_none() {
                match word.name.as_str() {
                    "group" => return self.group_method(cx, word, name, args, span),
                    "island" => {
                        if !self.allowed(cx, Family::IslandQuery, &format!("island.{what}"), span) {
                            return Ty::Unknown;
                        }
                        self.note_name(word.span, Binding::Word);
                        let Some(o) = self.operands(what, span, args, &[pos("id")]) else {
                            return Ty::Unknown;
                        };
                        self.operand(cx, o[0], &Ty::Int, "a worker's id");
                        return match what {
                            "of" | "hops" => Ty::Int,
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
                        self.note_name(word.span, Binding::State);
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
                self.operand(cx, o[0], &Ty::Int, "the first tile");
                self.operand(cx, o[1], &Ty::Int, "the count");
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
                if !self.stated_bytes(*format, "items", span) {
                    return Ty::Unknown;
                }
                for e in &o {
                    self.operand(cx, *e, &Ty::Int, "an operand of `items`");
                }
                let count = match (positional, o.get(1).copied().flatten().and_then(as_int)) {
                    (2, Some(n)) => Count::Known(n),
                    _ => Count::Unknown,
                };
                Ty::Items(count)
            }
            (Ty::View { format, tiles }, "words") => {
                let Some(_) = self.operands("words", span, args, &[]) else {
                    return Ty::Unknown;
                };
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
                Ty::Words
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
                self.operand(cx, o[1], &Ty::Int, "the count");
                let Some(r) = o[0] else {
                    return Ty::Unknown;
                };
                let name = match as_name(r) {
                    Some(n) if cx.env.record(cx.program.as_deref(), &n.name).is_some() => {
                        self.note_name(n.span, Binding::Word);
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
                let tile_bytes = match (cx.env.tile, format.and_then(Format::bytes)) {
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
            (Ty::Items(count), "put") => {
                let Some(o) = self.operands("put", span, args, &[pos("src"), named("at")]) else {
                    return Ty::Unknown;
                };
                self.operand(cx, o[0], &Ty::Unknown, "the source");
                self.operand(cx, o[1], &Ty::Int, "the item");
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
            (Ty::Sink, "push") => {
                let Some(o) = self.operands("push", span, args, &[pos("t")]) else {
                    return Ty::Unknown;
                };
                let t = self.operand(cx, o[0], &Ty::Unknown, "the block");
                if let Ty::Reduced(_) = t {
                    self.expect_fit(&t, &Ty::Tile(None), span, "the block pushed");
                } else if !matches!(t, Ty::Tile(_) | Ty::Block(_, _) | Ty::Unknown) {
                    self.error(
                        codes::WRONG_TYPE,
                        format!("`push` writes a tile, not {t}"),
                        span,
                    );
                }
                self.stream_overlap(cx, base, span);
                Ty::Void
            }
            (Ty::Tile(f), "col" | "row") => {
                if !self.allowed(cx, Family::Compute, what, span) {
                    return Ty::Unknown;
                }
                let Some(_) = self.operands(what, span, args, &[]) else {
                    return Ty::Unknown;
                };
                Ty::Tile(*f)
            }
            (Ty::Reduced(axis), "col" | "row") => {
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
                Ty::Tile(None)
            }
            (Ty::Broadcast, "cast") => {
                let Some(o) = self.operands("cast", span, args, &[pos("block"), named_opt("word")])
                else {
                    return Ty::Unknown;
                };
                self.operand(cx, o[0], &Ty::Tile(None), "the block");
                self.operand(cx, o[1], &Ty::Int, "the word");
                Ty::Void
            }
            (Ty::Broadcast, "arrive") => {
                let Some(_) = self.operands("arrive", span, args, &[]) else {
                    return Ty::Unknown;
                };
                Ty::Arrive
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
        self.note_name(word.span, Binding::Word);
        match what {
            "id" => {
                let Some(o) = self.operands("group.id", span, args, &[pos("rank")]) else {
                    return Ty::Unknown;
                };
                self.operand(cx, o[0], &Ty::Int, "the rank");
                Ty::Int
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
                for e in o {
                    self.operand(cx, e, &Ty::Int, "a count");
                }
                Ty::Parts
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
        match what {
            "fill" => {
                let Some(o) = self.operands("fill", span, args, &[pos("stream")]) else {
                    return Ty::Unknown;
                };
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
                    Ty::Block(_, Count::Known(n)) => Some(*n),
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
                self.state_use.1.insert(word.name.clone());
                Ty::Void
            }
            "zero" => {
                let Some(_) = self.operands("zero", span, args, &[]) else {
                    return Ty::Unknown;
                };
                self.state_use.1.insert(word.name.clone());
                Ty::Void
            }
            _ => {
                let Some(o) = self.operands("bump", span, args, &[pos("i")]) else {
                    return Ty::Unknown;
                };
                self.operand(cx, o[0], &Ty::Int, "the index");
                match ty {
                    Ty::Scalars(elem, _) if **elem == Ty::Int => {}
                    _ => self.error(
                        codes::WRONG_TYPE,
                        format!(
                            "`.bump` counts in a member of integers, and `{}` is not one",
                            word.name
                        ),
                        span,
                    ),
                }
                self.state_use.0.push((word.name.clone(), span));
                Ty::Int
            }
        }
    }
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
