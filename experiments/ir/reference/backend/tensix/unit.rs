//! A tile function's words for the math CPU's vector unit: `tile_<f>.hpp` and its
//! listing `tile_<f>.txt`, made by the tile compiler's port (`tile`) as today's class
//! DSL makes them for a call on one tile at a time: the function traced, its unit
//! program lowered with a load macro, emitted, and listed with the emit's checks.

use crate::backend::tensix::plan::param_name;
use crate::backend::{Refusal, NOT_BUILT, OVER_LIMIT};
use crate::program::notes::{Anchor, HintArg, Owner};
use crate::program::Module;
use crate::types::FuncId;

use super::tile::chain::ChainError;
use super::tile::emit::emit;
use super::tile::ir::{kernel_ir, KernelIR};
use super::tile::lower::{unit_program, A_DTYPES, FIDELITY};
use super::tile::matrix::phases;
use super::tile::rows::CHUNK;
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
    /// Where a call's parameters and result sit in Dst.
    layout: DstLayout,
}

impl Unit {
    /// The Dst tiles a call takes (today's `max(len(unit.operands), unit.results)`):
    /// a batch of calls is the Dst half's tiles over this.
    pub fn dst_tiles(&self) -> usize {
        self.operands.max(self.results)
    }

    /// Where a call's parameters and result sit in Dst, from the call's first tile:
    /// the tiles the caller copies each parameter into and packs the result from.
    pub fn dst_layout(&self) -> &DstLayout {
        &self.layout
    }
}

/// Where a call of a unit holds its parameters and its result in Dst, counted from the
/// call's first Dst tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DstLayout {
    /// The Dst tiles of each parameter, in the function's order: none for a parameter
    /// not read as a tile (a run-time scalar), several for one read in several ways.
    pub params: Vec<Vec<usize>>,
    /// The Dst tiles the result is packed from, in order.
    pub result: Vec<usize>,
}

/// The stream form's layout (today's handler, a call on one tile): Dst tile k holds the
/// parameter its unit's operand k names, and result k is in Dst tile k.
fn stream_layout(names: &[&str], unit: &UnitProgram) -> DstLayout {
    let params = names
        .iter()
        .map(|name| {
            (unit.operands.iter().enumerate())
                .filter(|(_, o)| o.as_deref() == Some(*name))
                .map(|(k, _)| k)
                .collect()
        })
        .collect();
    let results = usize::try_from(unit.results).unwrap_or(0);
    DstLayout {
        params,
        result: (0..results).collect(),
    }
}

/// The row form's layout on a row of `width` tiles (today's handler): the first
/// streamed parameter's row in Dst 0 .. width - 1 (a chunk of 2, Dst 0 and 1, when the
/// row is wider than 2), resident row k (the `resident` parameters, the last ones) in
/// Dst 2 width + k (4 + k in the chunk form), the result over the row's tiles. The
/// compute loop does not call the row form yet; this says where it will put them.
pub fn row_layout(resident: &[bool], width: usize) -> DstLayout {
    // unlike the stream form's, read off the unit, this restates today's handler: the
    // unit of the row form names only the row's tensor, and today's compute loop puts
    // the rest where compile/tiles/functions.py writes it (the row's copies and field k
    // at 2 width + k in tile_unit's call; row_chunks' chunk in 0 and 1, field k at 4 + k)
    let chunk = usize::try_from(CHUNK).unwrap_or(2);
    let (tiles, fields) = if width > chunk {
        (chunk, 4)
    } else {
        (width, 2 * width)
    };
    let first = resident.iter().position(|r| !r);
    let mut k = 0;
    let params = (resident.iter().enumerate())
        .map(|(i, &r)| {
            if r {
                k += 1;
                vec![fields + k - 1]
            } else if Some(i) == first {
                (0..tiles).collect()
            } else {
                Vec::new()
            }
        })
        .collect();
    DstLayout {
        params,
        result: (0..tiles).collect(),
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
/// one of today's names (another is refused with today's whole text, `K962`), or with no
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
        Err(e) => {
            // today's unit_program refuses it at its entry, by `fail`: the prefix, the
            // function's name and a line (today's caller's; here the function's own)
            let func = module
                .funcs
                .get(f.0 as usize)
                .map_or("", |x| x.name.as_str());
            let line = func_line(module, f).map_or(String::new(), |l| format!(" (at {l})"));
            Err(Refusal {
                code: OVER_LIMIT,
                what: format!("komorebi.dsl.tile: {func}: {}{line}", e.0),
                at,
            })
        }
    }
}

/// `file:line` of function `f`'s place in the module's origin, when it has one.
fn func_line(module: &Module, f: FuncId) -> Option<String> {
    let origin = module.notes.origin.as_ref()?;
    let &(_, file, first, _) = origin.places.iter().find(|p| p.0 == Anchor::Func(f))?;
    let path = origin.files.get(file as usize)?;
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
    Some(format!("{base}:{first}"))
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
    let names: Vec<&str> = (0..func.params.len())
        .map(|i| param_name(module, Owner::Func(f), i as u32).unwrap_or(""))
        .collect();
    Ok(Unit {
        name,
        words,
        listing,
        operands: made.operands.len(),
        results: usize::try_from(made.results).unwrap_or(0),
        layout: stream_layout(&names, &made),
    })
}
