//! What a load of a module on Tensix makes from the tensors it is given, beside the
//! compile's output (`api::emit`): the thread header, the kernels with each tensor's
//! layout filled in, and the buffers by the loader's names of their formats
//! (`chip::dtype`). Nothing is written and nothing run.

use crate::backend::tensix::load::{self, LoadTensor, LoadedKernel};
use crate::backend::tensix::{chip, Tensix};
use crate::backend::{Backend, Build, Refusal, Target, NOT_BUILT};
use crate::program::Module;

/// A buffer as the loader makes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadBuffer {
    /// Its index.
    pub index: u32,
    /// The loader's name of its format (`bfloat16`).
    pub dtype: &'static str,
    /// The bytes of a page.
    pub page_bytes: u64,
    /// Its pages.
    pub pages: u64,
    /// The other half of its pair, when it is one.
    pub pair: Option<u32>,
}

/// What a load makes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loaded {
    /// The thread header's name (`<program>_thread.h`) and text.
    pub thread: (String, String),
    /// The kernels, each tensor's layout in its place.
    pub kernels: Vec<LoadedKernel>,
    /// The buffers.
    pub buffers: Vec<LoadBuffer>,
}

fn refused(what: String) -> Vec<Refusal> {
    vec![Refusal {
        code: NOT_BUILT,
        what,
        at: None,
    }]
}

/// The load of `module` given `tensors` by parameter name, each name once and every
/// tensor parameter named; on `chip` (`blackhole`, `wormhole_b0`) when one is named, a
/// chip of another name refused. A load names no chip when nothing is loaded yet (a
/// make with no session).
pub fn load(
    module: &Module,
    tensors: &[(String, LoadTensor)],
    chip: Option<&str>,
) -> Result<Loaded, Vec<Refusal>> {
    if let Some(arch) = chip {
        let target = Target {
            arch: arch.to_string(),
            features: Default::default(),
        };
        load::chip(&target).map_err(|r| vec![r])?;
    }
    let output = Tensix.compile(
        module,
        &Target {
            arch: chip::ARCH.to_string(),
            features: Default::default(),
        },
    )?;
    let Build::Kernels { kernels, buffers } = &output.build else {
        return Err(refused("a Tensix output built by steps".into()));
    };
    let names = load::tensor_names(module)?;
    for (name, _) in tensors {
        if !names.contains(name) {
            return Err(refused(format!(
                "a tensor `{name}`: the program takes none of that name"
            )));
        }
        if tensors.iter().filter(|(n, _)| n == name).count() > 1 {
            return Err(refused(format!("the tensor `{name}` given twice")));
        }
    }
    let mut ordered = Vec::new();
    for name in &names {
        match tensors.iter().find(|(n, _)| n == name) {
            Some((_, t)) => ordered.push(t.clone()),
            None => return Err(refused(format!("no tensor given for `{name}`"))),
        }
    }
    let header = load::thread_header(module, &ordered)?;
    let kernels = load::kernels(module, kernels, &ordered)?;
    let mut out = Vec::new();
    for b in buffers {
        let Some(dtype) = chip::dtype(b.format) else {
            return Err(refused(format!(
                "a buffer of format {}, which the loader does not name",
                b.format.name()
            )));
        };
        out.push(LoadBuffer {
            index: b.index,
            dtype,
            page_bytes: b.page_bytes,
            pages: b.pages,
            pair: b.pair,
        });
    }
    let name = module.programs.first().map_or("", |p| p.name.as_str());
    Ok(Loaded {
        thread: (format!("{}_thread.h", name.to_lowercase()), header),
        kernels,
        buffers: out,
    })
}
