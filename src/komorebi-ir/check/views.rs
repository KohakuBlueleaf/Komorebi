//! What the text shows of ranks and views: the rank an `offer` goes to, and whether a
//! stream writes a view overlapping the one it reads.

use crate::syntax::ast::{Attr, BinOp, Expr, ExprKind, IndexItem, Span};

use super::body::{as_int, as_name, is_word_member, Cx, Peers, Rank, ViewKey};
use super::codes;
use super::env::parse_int;
use super::Checker;

impl Checker<'_> {
    // ---- Ranks and views ----------------------------------------------------------

    /// The rank an expression stands for, when the text shows it: a constant, or
    /// `(worker.rank + d) % group.size` with a constant d, through a `let`.
    pub(super) fn rank_of(&self, cx: &Cx, e: &Expr) -> Option<Rank> {
        match &e.kind {
            ExprKind::Int(text) => parse_int(text).map(Rank::Const),
            ExprKind::Paren(inner) => self.rank_of(cx, inner),
            ExprKind::Name(n) => cx.lookup(&n.name).and_then(|l| l.rank),
            ExprKind::Binary {
                op: BinOp::Rem,
                lhs,
                rhs,
            } if is_word_member(rhs, "group", "size") => {
                let mut sum = lhs.as_ref();
                while let ExprKind::Paren(inner) = &sum.kind {
                    sum = inner;
                }
                match &sum.kind {
                    ExprKind::Binary {
                        op: BinOp::Add,
                        lhs: a,
                        rhs: b,
                    } => {
                        if is_word_member(a, "worker", "rank") {
                            as_int(b).map(Rank::Offset)
                        } else if is_word_member(b, "worker", "rank") {
                            as_int(a).map(Rank::Offset)
                        } else {
                            None
                        }
                    }
                    _ if is_word_member(sum, "worker", "rank") => Some(Rank::Offset(0)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// `K502`: an `offer` to a rank the text shows, that is not a declared peer.
    pub(super) fn offer_peer(&mut self, cx: &Cx, to: &Expr, span: Span) {
        let Some(rank) = self.rank_of(cx, to) else {
            return;
        };
        let refused = match (&cx.peers, rank) {
            (None, _) => true,
            (Some(p), Rank::Offset(d)) if !p.offsets.is_empty() || p.pairs.is_empty() => {
                !p.offsets.contains(&d)
            }
            (Some(p), Rank::Const(c)) if !p.pairs.is_empty() || p.offsets.is_empty() => {
                !p.pairs.iter().any(|&(_, r)| r == c)
            }
            _ => false,
        };
        if refused {
            let text = match (&cx.peers, rank) {
                (None, _) => "the run declares no peers (`@peers`)".to_string(),
                (_, Rank::Offset(d)) => format!("the offset {d} is not among the run's `@peers`"),
                (_, Rank::Const(c)) => format!("the rank {c} is not among the run's `@peers`"),
            };
            self.error(
                codes::OFFER_NOT_PEER,
                format!("an `offer` goes to a declared peer; {text}"),
                span,
            );
        }
    }

    /// The view an expression names, written out.
    pub(super) fn view_key(&self, cx: &Cx, e: &Expr) -> Option<ViewKey> {
        match &e.kind {
            ExprKind::Paren(inner) => self.view_key(cx, inner),
            ExprKind::Name(n) => cx.lookup(&n.name).and_then(|l| l.view.clone()),
            ExprKind::Member { base, name } if name.name == "T" => {
                let inner = self.view_key(cx, base)?;
                Some(ViewKey {
                    tensor: inner.tensor,
                    text: format!("{}.T", inner.text),
                    range: None,
                })
            }
            ExprKind::Index { base, .. } => {
                let ExprKind::Member { base: t, name } = &base.kind else {
                    return None;
                };
                if name.name != "tiles" {
                    return None;
                }
                let tensor = as_name(t)?.name.clone();
                Some(ViewKey {
                    tensor,
                    text: render(e),
                    range: None,
                })
            }
            ExprKind::Call { callee, args } => {
                let first = args.values().next().map(|a| &a.value);
                match &callee.kind {
                    ExprKind::Name(f) if !f.quoted && (f.name == "stream" || f.name == "sink") => {
                        self.view_key(cx, first?)
                    }
                    ExprKind::Name(f) if !f.quoted && f.name == "partition" => {
                        let inner = self.view_key(cx, first?)?;
                        let rest: Vec<String> =
                            args.values().skip(1).map(|a| render(&a.value)).collect();
                        let wrap = format!("partition(_, {})", rest.join(", "));
                        Some(ViewKey {
                            tensor: inner.tensor,
                            text: format!("partition({}, {})", inner.text, rest.join(", ")),
                            range: inner
                                .range
                                .map(|(w, lo, hi)| (format!("{wrap}{w}"), lo, hi)),
                        })
                    }
                    ExprKind::Member { base, name }
                        if name.name == "tiles" || name.name == "items" =>
                    {
                        let tensor = as_name(base)?.name.clone();
                        let range = if name.name == "tiles" {
                            let vals: Vec<Option<u64>> =
                                args.values().map(|a| as_int(&a.value)).collect();
                            match vals.as_slice() {
                                [Some(lo), Some(n)] => Some((String::new(), *lo, lo + n)),
                                _ => None,
                            }
                        } else {
                            None
                        };
                        Some(ViewKey {
                            tensor,
                            text: render(e),
                            range,
                        })
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// The peers `@peers(offsets = [...])` or `@peers(pairs = [(s, r), ...])` declares.
pub(super) fn peers_of(attr: &Attr) -> Peers {
    let mut peers = Peers::default();
    let Some(args) = &attr.args else {
        return peers;
    };
    for arg in args.values() {
        let ExprKind::List(items) = &arg.value.kind else {
            continue;
        };
        match arg.name.as_ref().map(|n| n.name.as_str()) {
            Some("offsets") => peers.offsets.extend(items.values().filter_map(as_int)),
            Some("pairs") => {
                for item in items.values() {
                    if let ExprKind::Tuple(pair) = &item.kind {
                        let v: Vec<Option<u64>> = pair.values().map(as_int).collect();
                        if let [Some(s), Some(r)] = v.as_slice() {
                            peers.pairs.push((*s, *r));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    peers
}

/// An expression written out, spaces aside, to compare two views.
pub(super) fn render(e: &Expr) -> String {
    match &e.kind {
        ExprKind::Int(t) | ExprKind::Float(t) | ExprKind::Str(t) => t.clone(),
        ExprKind::Bool(b) => b.to_string(),
        ExprKind::Name(n) => n.name.clone(),
        ExprKind::Paren(inner) => format!("({})", render(inner)),
        ExprKind::Member { base, name } => format!("{}.{}", render(base), name.name),
        ExprKind::Call { callee, args } => {
            let args: Vec<String> = args
                .values()
                .map(|a| match &a.name {
                    Some(n) => format!("{}={}", n.name, render(&a.value)),
                    None => render(&a.value),
                })
                .collect();
            format!("{}({})", render(callee), args.join(","))
        }
        ExprKind::Index { base, index } => {
            let items: Vec<String> = index
                .values()
                .map(|i| match i {
                    IndexItem::At(e) => render(e),
                    IndexItem::Slice { lo, hi, step } => match step {
                        Some(s) => format!("{}:{}:{}", render(lo), render(hi), render(s)),
                        None => format!("{}:{}", render(lo), render(hi)),
                    },
                })
                .collect();
            format!("{}[{}]", render(base), items.join(","))
        }
        ExprKind::Neg(inner) => format!("-{}", render(inner)),
        ExprKind::Binary { op, lhs, rhs } => {
            format!("({}{}{})", render(lhs), op.text(), render(rhs))
        }
        _ => "?".into(),
    }
}
