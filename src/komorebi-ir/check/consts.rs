//! Constant expressions: whole numbers written in the text, module parameters bound at
//! build, the operators `+ - * / % << >> & | ^`, `min(a, b)`, `max(a, b)` and brackets,
//! evaluated at program compile with the language's integer rules (`/` rounds toward
//! zero). At a place of a float type: numbers with a point, float parameters, and
//! `+ - * /`. The one evaluator: the checker reports what it refuses, and Lower takes
//! the same values from it.

use crate::syntax::ast::{BinOp, Expr, ExprKind, Ident, Span};
use crate::types::ScalarType;

use super::body::holds;
use super::codes;
use super::Checker;

/// The value of a constant expression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Num {
    /// A whole number.
    Int(i128),
    /// A float.
    Float(f64),
}

/// What a name in a constant expression stands for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Known {
    /// A module parameter bound at build, with its value (`None` when its own value is
    /// refused, which is reported there).
    Build(Option<Num>),
    /// A module parameter bound at load or at the invocation.
    Unbuilt,
    /// A module parameter declared after the place that names it.
    Later,
    /// No module parameter (or one a local of the body hides).
    NotParam,
}

/// Why an expression has no value as a constant expression.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Fault {
    /// A form a constant expression does not take.
    NotConstant(Span),
    /// A name that is no module parameter.
    NotParam(Span, String),
    /// A parameter bound at load or at the invocation.
    Unbuilt(Span, String),
    /// A parameter named before its own declaration.
    Later(Span, String),
    /// A division or a remainder by zero.
    ByZero(Span),
    /// A value past what the evaluation holds.
    TooLarge(Span),
    /// A parameter whose own value is refused (reported at its declaration).
    Silent,
}

/// A whole number written in the text, decimal or `0x` hexadecimal.
fn whole(text: &str) -> Option<u128> {
    match text.strip_prefix("0x") {
        Some(hex) => u128::from_str_radix(hex, 16).ok(),
        None => text.parse().ok(),
    }
}

/// The value of the constant expression `e`, a float one when `float`; `look` says what
/// each name stands for.
pub(crate) fn eval(e: &Expr, float: bool, look: &dyn Fn(&Ident) -> Known) -> Result<Num, Fault> {
    let int = |n: Num| match n {
        Num::Int(v) => Ok(v),
        Num::Float(_) => Err(Fault::NotConstant(e.span)),
    };
    let real = |n: Num| match n {
        Num::Float(v) => Ok(v),
        Num::Int(v) => Ok(v as f64),
    };
    match &e.kind {
        ExprKind::Int(text) => {
            let v = whole(text).ok_or(Fault::TooLarge(e.span))?;
            let v = i128::try_from(v).map_err(|_| Fault::TooLarge(e.span))?;
            Ok(if float {
                Num::Float(v as f64)
            } else {
                Num::Int(v)
            })
        }
        ExprKind::Float(text) if float => text
            .parse::<f64>()
            .map(Num::Float)
            .map_err(|_| Fault::NotConstant(e.span)),
        ExprKind::Paren(inner) => eval(inner, float, look),
        ExprKind::Neg(inner) => match eval(inner, float, look)? {
            Num::Int(v) => v.checked_neg().map(Num::Int).ok_or(Fault::TooLarge(e.span)),
            Num::Float(v) => Ok(Num::Float(-v)),
        },
        ExprKind::Name(n) => match look(n) {
            Known::Build(Some(Num::Int(v))) if !float => Ok(Num::Int(v)),
            Known::Build(Some(Num::Float(v))) if float => Ok(Num::Float(v)),
            Known::Build(Some(_)) => Err(Fault::NotConstant(e.span)),
            Known::Build(None) => Err(Fault::Silent),
            Known::Unbuilt => Err(Fault::Unbuilt(n.span, n.name.clone())),
            Known::Later => Err(Fault::Later(n.span, n.name.clone())),
            Known::NotParam => Err(Fault::NotParam(n.span, n.name.clone())),
        },
        ExprKind::Binary { op, lhs, rhs } => {
            let a = eval(lhs, float, look)?;
            let b = eval(rhs, float, look)?;
            if float {
                let (a, b) = (real(a)?, real(b)?);
                return match op {
                    BinOp::Add => Ok(Num::Float(a + b)),
                    BinOp::Sub => Ok(Num::Float(a - b)),
                    BinOp::Mul => Ok(Num::Float(a * b)),
                    BinOp::Div if b == 0.0 => Err(Fault::ByZero(rhs.span)),
                    BinOp::Div => Ok(Num::Float(a / b)),
                    _ => Err(Fault::NotConstant(e.span)),
                };
            }
            let (a, b) = (int(a)?, int(b)?);
            let large = Fault::TooLarge(e.span);
            let v = match op {
                BinOp::Add => a.checked_add(b).ok_or(large)?,
                BinOp::Sub => a.checked_sub(b).ok_or(large)?,
                BinOp::Mul => a.checked_mul(b).ok_or(large)?,
                BinOp::Div | BinOp::Rem if b == 0 => return Err(Fault::ByZero(rhs.span)),
                // `/` rounds toward zero, and `%` is the remainder of that division
                BinOp::Div => a.checked_div(b).ok_or(large)?,
                BinOp::Rem => a.checked_rem(b).ok_or(large)?,
                BinOp::Shl | BinOp::Shr => {
                    let s = u32::try_from(b)
                        .ok()
                        .filter(|s| *s < 127)
                        .ok_or(Fault::TooLarge(rhs.span))?;
                    if *op == BinOp::Shr {
                        a >> s
                    } else {
                        let v = a << s;
                        if v >> s != a {
                            return Err(large);
                        }
                        v
                    }
                }
                BinOp::BitAnd => a & b,
                BinOp::BitOr => a | b,
                BinOp::BitXor => a ^ b,
                _ => return Err(Fault::NotConstant(e.span)),
            };
            Ok(Num::Int(v))
        }
        ExprKind::Call { callee, args } if !float => {
            let ExprKind::Name(f) = &callee.kind else {
                return Err(Fault::NotConstant(e.span));
            };
            let operands: Vec<&Expr> = args
                .values()
                .filter(|a| a.name.is_none())
                .map(|a| &a.value)
                .collect();
            let (min, max) = (f.name == "min", f.name == "max");
            match operands.as_slice() {
                [a, b] if !f.quoted && (min || max) && args.len() == 2 => {
                    let a = int(eval(a, false, look)?)?;
                    let b = int(eval(b, false, look)?)?;
                    Ok(Num::Int(if min { a.min(b) } else { a.max(b) }))
                }
                _ => Err(Fault::NotConstant(e.span)),
            }
        }
        _ => Err(Fault::NotConstant(e.span)),
    }
}

/// Whether the scalar type `s` holds the value `n`.
pub(crate) fn fits(n: Num, s: ScalarType) -> bool {
    match n {
        Num::Float(v) => s.is_float() && holds(s, v),
        Num::Int(_) if s.is_float() => false,
        Num::Int(v) => {
            let bits = s.bits();
            let (lo, hi) = if s.is_signed() {
                (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
            } else {
                (0, (1i128 << bits) - 1)
            };
            v >= lo && v <= hi
        }
    }
}

/// The value of the constant expression `e` as a `u32` count, size, rank or grid side,
/// where it is one that fits.
pub(crate) fn count(e: &Expr, look: &dyn Fn(&Ident) -> Known) -> Option<u64> {
    match eval(e, false, look) {
        Ok(Num::Int(v)) if fits(Num::Int(v), ScalarType::U32) => Some(v as u64),
        _ => None,
    }
}

impl Checker<'_> {
    /// Reports why `e` has no value as a constant expression, at the place `what`.
    pub(crate) fn constant_fault(&mut self, fault: Fault, what: &str) {
        match fault {
            Fault::NotConstant(span) => self.error(
                codes::WRONG_TYPE,
                format!(
                    "{what} is a constant expression: whole numbers written in the text, module \
                     parameters bound at build, `+ - * / % << >> & | ^`, `min`, `max` and \
                     brackets (for a float, numbers with a point, float parameters and \
                     `+ - * /`)"
                ),
                span,
            ),
            Fault::NotParam(span, name) => self.error(
                codes::UNKNOWN_NAME,
                format!("`{name}` is not a module parameter"),
                span,
            ),
            Fault::Unbuilt(span, name) => self.error(
                codes::WRONG_TYPE,
                format!(
                    "a constant expression names only parameters bound at build, and `{name}` \
                     is not"
                ),
                span,
            ),
            Fault::Later(span, name) => self.error(
                codes::UNKNOWN_NAME,
                format!("`{name}` is named before its own declaration"),
                span,
            ),
            Fault::ByZero(span) => self.error(
                codes::DIVIDE_BY_ZERO,
                format!("{what} divides by zero"),
                span,
            ),
            Fault::TooLarge(span) => self.error(
                codes::OUT_OF_RANGE,
                format!("{what} is past what a constant expression holds"),
                span,
            ),
            Fault::Silent => {}
        }
    }

    /// The value of the constant expression `e` at a place of type `s` (`what`), with
    /// what it refuses reported: its own faults, and a value `s` does not hold (`K115`).
    pub(crate) fn constant_at(
        &mut self,
        e: &Expr,
        s: ScalarType,
        what: &str,
        look: &dyn Fn(&Ident) -> Known,
    ) -> Option<Num> {
        match eval(e, s.is_float(), look) {
            Ok(n) if fits(n, s) => Some(n),
            Ok(n) => {
                let shown = match n {
                    Num::Int(v) => v.to_string(),
                    Num::Float(v) => v.to_string(),
                };
                self.error(
                    codes::OUT_OF_RANGE,
                    format!("{what} is {shown}, which {s} does not hold"),
                    e.span,
                );
                None
            }
            Err(fault) => {
                self.constant_fault(fault, what);
                None
            }
        }
    }
}
