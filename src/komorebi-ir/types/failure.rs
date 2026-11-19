//! The named failures of an invocation: what `o.reason` yields and a control program
//! compares (`o.reason == capacity`).

use std::fmt;

/// A failure by name: the invocation ends without a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Failure {
    /// `view_end`: a view past its tensor's end.
    ViewEnd,
    /// `empty_state`: a read of state that is not filled.
    EmptyState,
    /// `send_refused`: a refused `send` whose result is not used.
    SendRefused,
    /// `capacity`: one of the backend's limits exceeded.
    Capacity,
    /// `given_up`: the service gave the invocation up.
    GivenUp,
    /// `dependency`: an invocation it waited for failed.
    Dependency,
    /// `not_a_peer`: an `offer` to a rank that is not a declared peer.
    NotAPeer,
    /// `over_bound`: a parameter above its declared largest value.
    OverBound,
    /// `divide_by_zero`: an integer divided by zero, by `/` or `%`.
    DivideByZero,
    /// `out_of_range`: an index or a count that does not fit what it addresses, or a
    /// negative shift.
    OutOfRange,
}

impl Failure {
    /// Every failure, in the order the language lists them.
    pub const ALL: [Failure; 10] = [
        Failure::ViewEnd,
        Failure::EmptyState,
        Failure::SendRefused,
        Failure::Capacity,
        Failure::GivenUp,
        Failure::Dependency,
        Failure::NotAPeer,
        Failure::OverBound,
        Failure::DivideByZero,
        Failure::OutOfRange,
    ];

    /// The failure's name, as a control program writes it.
    pub fn name(self) -> &'static str {
        match self {
            Failure::ViewEnd => "view_end",
            Failure::EmptyState => "empty_state",
            Failure::SendRefused => "send_refused",
            Failure::Capacity => "capacity",
            Failure::GivenUp => "given_up",
            Failure::Dependency => "dependency",
            Failure::NotAPeer => "not_a_peer",
            Failure::OverBound => "over_bound",
            Failure::DivideByZero => "divide_by_zero",
            Failure::OutOfRange => "out_of_range",
        }
    }

    /// The failure of a name, if the name is one.
    pub fn from_name(name: &str) -> Option<Failure> {
        Failure::ALL.into_iter().find(|f| f.name() == name)
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_names_round_trip() {
        for failure in Failure::ALL {
            assert_eq!(Failure::from_name(failure.name()), Some(failure));
        }
        assert_eq!(Failure::from_name("too_big"), None);
    }
}
