//! What a declaration states beside its types: a constant, when a parameter is
//! fixed, the group shape an invocable runs on, the peers a run declares, and an
//! ability of the machine a module may require.

use super::ScalarType;

/// A constant of a type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstValue {
    /// An integer of a type, as its bits masked to the type's width.
    Int(ScalarType, u64),
    /// A float of a type, as written, held as an `f64` (the number the text wrote).
    Float(ScalarType, f64),
    /// `true` or `false`.
    Bool(bool),
}

/// When a module parameter is fixed: `@bind(...)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Binding {
    /// `@bind(build)`: fixed when the module is built; every use is this constant.
    Build(ConstValue),
    /// `@bind(load)` with a value, or with `fit` (`None`): fixed when the module is
    /// loaded on a machine.
    Load(Option<ConstValue>),
    /// `@bind(invoke)`: given at each invocation, at most this value (else the
    /// failure `over_bound`).
    Invoke(ConstValue),
}

/// The group shape a handler or a chain runs on: `@on(...)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum On {
    /// `@on(group)`: a line of ranks.
    Group,
    /// `@on(grid[R, C])`: R rows of C workers.
    Grid(u64, u64),
}

/// The peers a dynamic run declares for `offer`: `@peers(...)`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Peers {
    /// No `@peers`.
    #[default]
    None,
    /// `@peers(offsets = [d, ...])`.
    Offsets(Vec<u64>),
    /// `@peers(pairs = [(s, r), ...])`.
    Pairs(Vec<(u64, u64)>),
}

/// An ability of the machine a module may require: a line of its `requires` block.
/// `group = line` is every target's and is no ability.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ability {
    /// `local_state >= tiles(n)` or `local_state >= bytes(n)`.
    LocalState {
        /// The amount.
        amount: u64,
        /// In tiles (else in bytes).
        tiles: bool,
    },
    /// `group = grid`.
    Grid,
    /// `dynamic`.
    Dynamic,
    /// `islands >= n`.
    Islands(u64),
    /// `impl("name")`.
    Impl(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_has_no_peers_unless_it_states_them() {
        assert_eq!(Peers::default(), Peers::None);
        assert_ne!(On::Group, On::Grid(4, 8));
        assert_eq!(
            Binding::Build(ConstValue::Int(ScalarType::U32, 64)),
            Binding::Build(ConstValue::Int(ScalarType::U32, 64))
        );
    }
}
