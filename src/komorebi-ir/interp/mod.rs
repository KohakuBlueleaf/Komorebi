//! The reference interpreter: the values (`memory`), the language's operations on them
//! (`ops`), the workers and the bodies they run (`workers`), the meetings of a group's
//! workers at a collective (`delivery`), and a loaded module with the invocations made
//! on it and the run's report (`control`). It reads the program structure, never the
//! syntax tree.

pub mod control;
pub mod delivery;
pub mod memory;
pub mod ops;
pub mod workers;
