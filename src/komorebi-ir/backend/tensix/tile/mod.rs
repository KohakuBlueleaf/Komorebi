//! The tile compiler's port (today's `komorebi.dsl.tile`): a tile function's words for
//! the math CPU's vector unit and their listing, stage by stage, each stage's text the
//! same as today's for the same function: the tile IR of a function (`ir`), its
//! vector-unit function (`chain`, `vector`), its unit program (`lower`, `macro`) with
//! its hazard checks (`checks`), the unit program's data (`unit`), Wormhole's forms of
//! its words (`wormhole`) and its header (`emit`). What is not ported yet is refused by
//! name.

pub mod chain;
pub mod checks;
pub mod emit;
pub mod ir;
pub mod lower;
pub mod r#macro;
pub mod py;
pub mod unit;
pub mod vector;
pub mod wormhole;
