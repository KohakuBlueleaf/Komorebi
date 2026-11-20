//! A tile function's words for the math CPU's vector unit: `tile_<f>.hpp` and its
//! listing `tile_<f>.txt`. These are NOT made by a port of today's tile compiler
//! (`komorebi.dsl.tile`): they are its output for the one shape `return a * k` (a bf16
//! tile times a run-time `f32`), kept as a template with two names filled in. Their
//! identity with today's files proves nothing about a tile compiler. Any other body
//! is refused by name (K964); the next tile function starts the tile compiler's port.

use crate::backend::{Refusal, NOT_BUILT};
use crate::program::body::Op;
use crate::program::notes::{Anchor, Owner};
use crate::program::{FuncKind, Module};
use crate::types::{BinOp, Format, FuncId, ScalarType, Type};

use super::plan::param_name;

/// The words of a bf16 tile times a run-time scalar (fp32 Dst): the tile compiler's
/// output for `return a * k`, `@FN@` the function's name and `@K@` the scalar's.
const SCALE_BY_WORD_HPP: &str = include_str!("units/scale_by_word.hpp.in");

/// Its listing: the unit program, the passes, the checks and the costs.
const SCALE_BY_WORD_TXT: &str = include_str!("units/scale_by_word.txt.in");

/// The templates by name, as the backend's version covers them.
pub const TEMPLATES: &[(&str, &str)] = &[
    ("scale_by_word.hpp.in", SCALE_BY_WORD_HPP),
    ("scale_by_word.txt.in", SCALE_BY_WORD_TXT),
];

/// A tile function's two files: its words and its listing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    /// The function's name: `km_tile::<name>`.
    pub name: String,
    /// `tile_<name>.hpp`.
    pub words: String,
    /// `tile_<name>.txt`.
    pub listing: String,
}

/// The unit of function `f`, or why it is not built.
pub fn unit(module: &Module, f: FuncId) -> Result<Unit, Refusal> {
    let refused = |what: &str| Refusal {
        code: NOT_BUILT,
        what: format!("the Tensix backend does not emit {what} yet"),
        at: Some(Anchor::Func(f)),
    };
    let Some(func) = module.funcs.get(f.0 as usize) else {
        return Err(refused("a call of no function"));
    };
    let insts = func
        .body
        .as_ref()
        .map(|b| b.insts.as_slice())
        .unwrap_or(&[]);
    let shape = matches!(func.kind, FuncKind::Plain)
        && func.inner(Some(0)) == Some(Type::Tile(Format::Bf16))
        && func.inner(Some(1)) == Some(Type::Scalar(ScalarType::F32))
        && func.inner(None) == Some(Type::Tile(Format::Bf16))
        && matches!(
            insts,
            [a, k, mul, ret]
                if a.op == Op::Arg(0)
                    && k.op == Op::Arg(1)
                    && matches!(&mul.op, Op::Binary(BinOp::Mul, x, y)
                        if a.defs == [*x] && k.defs == [*y])
                    && matches!(&ret.op, Op::Return(Some(r)) if mul.defs == [*r])
        );
    if !shape {
        return Err(refused(
            "a tile function other than `return a * k` of a bf16 tile and a f32 (that one is today's output kept as a template; no tile compiler is ported)",
        ));
    }
    let k = param_name(module, Owner::Func(f), 1).unwrap_or("k");
    let fill = |text: &str| text.replace("@FN@", &func.name).replace("@K@", k);
    Ok(Unit {
        name: func.name.clone(),
        words: fill(SCALE_BY_WORD_HPP),
        listing: fill(SCALE_BY_WORD_TXT),
    })
}
