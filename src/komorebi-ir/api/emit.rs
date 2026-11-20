//! A module's output on a backend named by the caller: what `komo` and the Python face
//! hand a loader. Nothing is written and nothing run; the output is the backend's
//! `compile`, on the target its name gives when the caller names none.

use crate::backend::cpu::Cpu;
use crate::backend::tensix::{chip, Tensix};
use crate::backend::{Backend, Output, Refusal, Target};
use crate::program::Module;

/// The backends a caller may name.
pub const BACKENDS: &[&str] = &["cpu", "tensix"];

/// Why a module has no output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// No backend has the name.
    NoBackend(String),
    /// The backend refuses the module: every refusal.
    Backend(Vec<Refusal>),
}

/// The backend of a name.
pub fn backend(name: &str) -> Option<Box<dyn Backend>> {
    match name {
        "cpu" => Some(Box::new(Cpu)),
        "tensix" => Some(Box::new(Tensix)),
        _ => None,
    }
}

/// The target a backend compiles for when the caller names no architecture: the
/// Tensix backend's names no chip (one text for every chip); the CPU's is the host's.
pub fn default_arch(backend: &str) -> &'static str {
    match backend {
        "tensix" => chip::ARCH,
        _ => std::env::consts::ARCH,
    }
}

/// The output of `module` on backend `name`, for architecture `arch` (or the backend's
/// default), with no feature.
pub fn emit(module: &Module, name: &str, arch: Option<&str>) -> Result<Output, Refused> {
    let Some(backend) = backend(name) else {
        return Err(Refused::NoBackend(name.to_string()));
    };
    let target = Target {
        arch: arch.unwrap_or(default_arch(name)).to_string(),
        features: Default::default(),
    };
    backend.compile(module, &target).map_err(Refused::Backend)
}
