//! A module's output on a backend named by the caller: what `komo` and the Python face
//! hand a loader. Nothing is written and nothing run; the output is the backend's
//! `compile`. No backend is part of the build: every name is refused.

use crate::backend::{Backend, Output, Refusal, Target};
use crate::program::Module;

/// The backends a caller may name.
pub const BACKENDS: &[&str] = &[];

/// Why a module has no output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// No backend has the name.
    NoBackend(String),
    /// The backend refuses the module: every refusal.
    Backend(Vec<Refusal>),
}

/// The backend of a name: none, since no backend is part of the build.
pub fn backend(_name: &str) -> Option<Box<dyn Backend>> {
    None
}

/// The output of `module` on backend `name`, for architecture `arch` (or the host's),
/// with no feature.
pub fn emit(module: &Module, name: &str, arch: Option<&str>) -> Result<Output, Refused> {
    let Some(backend) = backend(name) else {
        return Err(Refused::NoBackend(name.to_string()));
    };
    let target = Target {
        arch: arch.unwrap_or(std::env::consts::ARCH).to_string(),
        features: Default::default(),
    };
    backend.compile(module, &target).map_err(Refused::Backend)
}
