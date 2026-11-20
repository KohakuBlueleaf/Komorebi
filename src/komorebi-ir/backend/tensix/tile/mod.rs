//! The tile compiler's port (today's `komorebi.dsl.tile`): a tile function's words for
//! the math CPU's vector unit and their listing, stage by stage, each stage's text the
//! same as today's for the same function. Ported so far: the tile IR of a
//! function (`ir`), its vector-unit function (`chain`, `vector`), its unit program
//! (`lower`, `macro`) with its hazard checks (`checks`) and the unit program's data
//! (`unit`).

pub mod chain;
pub mod checks;
pub mod ir;
pub mod lower;
pub mod r#macro;
pub mod py;
pub mod unit;
pub mod vector;
