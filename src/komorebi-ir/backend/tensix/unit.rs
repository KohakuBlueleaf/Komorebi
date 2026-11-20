//! A tile function's words for the math CPU's vector unit: `tile_<f>.hpp` and its
//! listing `tile_<f>.txt`, made by the tile compiler's port (`tile`) as today's class
//! DSL makes them for a call on one tile at a time: the function traced, its unit
//! program lowered with a load macro, emitted, and listed with the emit's checks.

use crate::backend::{Refusal, NOT_BUILT, OVER_LIMIT};
use crate::program::notes::{Anchor, HintArg};
use crate::program::Module;
use crate::types::FuncId;

use super::tile::chain::ChainError;
use super::tile::emit::emit;
use super::tile::ir::{kernel_ir, KernelIR};
use super::tile::lower::{unit_program, A_DTYPES, FIDELITY};
use super::tile::matrix::phases;
use super::tile::unit::UnitProgram;

/// A tile function's two files: its words and its listing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    /// The function's name: `km_tile::<name>`.
    pub name: String,
    /// `tile_<name>.hpp`.
    pub words: String,
    /// `tile_<name>.txt`.
    pub listing: String,
    /// Its unit program's operands.
    operands: usize,
    /// Its unit program's results.
    results: usize,
}

impl Unit {
    /// The Dst tiles a call takes (today's `max(len(unit.operands), unit.results)`):
    /// a batch of calls is the Dst half's tiles over this.
    pub fn dst_tiles(&self) -> usize {
        self.operands.max(self.results)
    }
}

/// The Dst a tile function runs on (today's `tile_dest`): `fp16a` when every
/// argument is a tile of the A family, else `fp32` (a run-time scalar among them
/// included).
fn dest(ir: &KernelIR) -> &'static str {
    let a_family = ir.values.is_empty()
        && ir
            .tensors
            .iter()
            .filter(|t| !t.output)
            .all(|t| t.dtype.as_deref().is_some_and(|d| A_DTYPES.contains(&d)));
    if a_family {
        "fp16a"
    } else {
        "fp32"
    }
}

/// Why the handler's compute loop, which reads its tiles into Dst tiles 0.. of each
/// call and packs one result from it, does not take `unit`, if it does not.
fn not_printed(unit: &UnitProgram) -> Option<&'static str> {
    if unit.operands.is_empty() || unit.operands.iter().any(Option::is_none) {
        Some("a tile function whose unit reads other than tiles")
    } else if unit.results != 1 || !unit.products.is_empty() || !unit.parts.is_empty() {
        Some("a tile function of more than one result or with tile products")
    } else {
        None
    }
}

/// The fidelity of function `f`'s tile products: its hint `@tensix.fidelity(<name>)`,
/// one of today's names (another is refused with today's text, `K962`), or with no
/// hint today's default (`HiFi4`). A second such hint, or one not of one name, is
/// refused by name (`K964`).
pub fn fidelity(module: &Module, f: FuncId) -> Result<String, Refusal> {
    let at = Some(Anchor::Func(f));
    let mut hints = module
        .notes
        .hints
        .iter()
        .filter(|h| h.anchor == Anchor::Func(f) && h.path == ["tensix", "fidelity"]);
    let Some(hint) = hints.next() else {
        return Ok(FIDELITY.to_string());
    };
    let not_built = |what: &str| Refusal {
        code: NOT_BUILT,
        what: format!("the Tensix backend does not take {what} yet"),
        at,
    };
    if hints.next().is_some() {
        return Err(not_built("two `tensix.fidelity` hints on a function"));
    }
    let [HintArg::Name(name)] = hint.args.as_slice() else {
        return Err(not_built("a `tensix.fidelity` hint other than of one name"));
    };
    match phases(name) {
        Ok(_) => Ok(name.clone()),
        Err(e) => Err(Refusal {
            code: OVER_LIMIT,
            what: e.0,
            at,
        }),
    }
}

/// The unit of function `f` called on one tile at a time, as today's `tile_unit` makes
/// it at the 1-tile width, or why it is not built.
pub fn unit(module: &Module, f: FuncId) -> Result<Unit, Refusal> {
    let at = Some(Anchor::Func(f));
    let Some(func) = module.funcs.get(f.0 as usize) else {
        return Err(Refusal {
            code: NOT_BUILT,
            what: "the Tensix backend does not emit a call of no function yet".into(),
            at,
        });
    };
    let name = func.name.clone();
    let ir = kernel_ir(module, f, &name)?;
    let fidelity = fidelity(module, f)?;
    let mut made =
        unit_program(&ir, dest(&ir), true, false, &fidelity).map_err(|e| e.refusal(at))?;
    if let Some(what) = not_printed(&made) {
        return Err(Refusal {
            code: NOT_BUILT,
            what: format!("the Tensix backend's compute loop does not call {what} yet"),
            at,
        });
    }
    let refused = |e| ChainError::from(e).refusal(at);
    let words = emit(&mut made).map_err(refused)?;
    let listing = made.dump().map_err(refused)? + "\n";
    Ok(Unit {
        name,
        words,
        listing,
        operands: made.operands.len(),
        results: usize::try_from(made.results).unwrap_or(0),
    })
}
