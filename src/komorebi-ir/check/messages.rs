//! `K501`: every arrived message is forwarded or consumed exactly once, on every path
//! through its `on M(m)` body.

use crate::syntax::ast::{Block, Expr, ExprKind, OnBlock, Span, Stmt};

use super::codes;
use super::Checker;

/// The numbers of times a path may have handled the message: bit 0 for none, bit 1
/// for once, bit 2 for twice or more.
type Counts = u8;

const NONE: Counts = 1;
const ONCE: Counts = 2;
const MORE: Counts = 4;

/// The counts after one more action.
fn after_action(counts: Counts) -> Counts {
    let mut out = 0;
    if counts & NONE != 0 {
        out |= ONCE;
    }
    if counts & (ONCE | MORE) != 0 {
        out |= MORE;
    }
    out
}

/// True for `forward(m, ...)` of the message named `m`.
fn is_forward_of(expr: &Expr, m: &str) -> bool {
    let ExprKind::Call { callee, args } = &expr.kind else {
        return false;
    };
    let is_forward = matches!(&callee.kind, ExprKind::Name(n) if n.name == "forward" && !n.quoted);
    let first = args.values().next().map(|a| &a.value.kind);
    is_forward && matches!(first, Some(ExprKind::Name(n)) if n.name == m)
}

impl Checker<'_> {
    /// Checks one `on M(m)` body.
    pub(crate) fn message_once(&mut self, on: &OnBlock) {
        let mut twice = false;
        let counts = self.handled(&on.body, &on.binding.name, NONE, &mut twice);
        if counts & NONE != 0 {
            self.error(
                codes::MESSAGE_ONCE,
                format!(
                    "on some path `{}` is neither forwarded nor consumed; every arrived message \
                     is forwarded or consumed exactly once",
                    on.binding.name
                ),
                on.span,
            );
        }
    }

    fn handled(&mut self, block: &Block, m: &str, mut counts: Counts, twice: &mut bool) -> Counts {
        for line in &block.stmts.lines {
            counts = self.handled_stmt(&line.node, line.span, m, counts, twice);
        }
        counts
    }

    fn handled_stmt(
        &mut self,
        stmt: &Stmt,
        span: Span,
        m: &str,
        counts: Counts,
        twice: &mut bool,
    ) -> Counts {
        let action = match stmt {
            Stmt::Consume(name) => name.name == m,
            Stmt::Expr(e) => is_forward_of(e, m),
            Stmt::If {
                then, otherwise, ..
            } => {
                let a = self.handled(then, m, counts, twice);
                let b = match otherwise {
                    Some(otherwise) => self.handled(otherwise, m, counts, twice),
                    None => counts,
                };
                return a | b;
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => {
                let once = self.handled(body, m, counts, twice);
                let again = self.handled(body, m, once, twice);
                return counts | once | again;
            }
            _ => false,
        };
        if !action {
            return counts;
        }
        if counts & (ONCE | MORE) != 0 && !*twice {
            *twice = true;
            self.error(
                codes::MESSAGE_ONCE,
                format!(
                    "`{m}` may already be forwarded or consumed here; every arrived message is \
                     forwarded or consumed exactly once"
                ),
                span,
            );
        }
        after_action(counts)
    }
}
