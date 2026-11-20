//! Building the CPU backend's output, in a cache of two levels: the sources under the
//! output's key (`<cache>/cpu/<key>/src`), and what the compiler made under the key and
//! the compiler (`<cache>/cpu/<key>/<compiler>`), so another compiler rebuilds only
//! that. A build is made in a directory of its own and renamed into place when it is
//! whole; one already in place is used as it is.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::backend::Output;
use crate::bytecode::hash::{hex, sha256};

use super::{DRIVER, LIBRARY};

/// The cache: `KOMO_IR_CACHE`, else `$XDG_CACHE_HOME/komorebi-ir`, else
/// `$HOME/.cache/komorebi-ir`.
pub fn default_cache() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("KOMO_IR_CACHE") {
        return Some(PathBuf::from(dir));
    }
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(PathBuf::from(dir).join("komorebi-ir"));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/komorebi-ir"))
}

/// The C++ compiler: `$CXX`, else `c++`.
pub fn compiler() -> String {
    std::env::var("CXX").unwrap_or_else(|_| "c++".to_string())
}

/// A built output: the directory that holds the library and the driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Built {
    /// The directory.
    pub dir: PathBuf,
}

impl Built {
    /// The shared library.
    pub fn library(&self) -> PathBuf {
        self.dir.join(LIBRARY)
    }

    /// The driver.
    pub fn driver(&self) -> PathBuf {
        self.dir.join(DRIVER)
    }
}

fn write_files(output: &Output, dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for (name, bytes) in &output.files {
        let path = dir.join(name);
        std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

/// Makes `staged` whole, then renames it to `dir`; when another build put `dir` in
/// place first, that one is kept and `staged` removed.
fn settle(staged: &Path, dir: &Path) -> Result<(), String> {
    if std::fs::rename(staged, dir).is_err() {
        if !dir.is_dir() {
            return Err(format!("{} could not be put in place", dir.display()));
        }
        let _ = std::fs::remove_dir_all(staged);
    }
    Ok(())
}

fn staging(dir: &Path) -> PathBuf {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let name = dir
        .file_name()
        .map_or(String::new(), |n| n.to_string_lossy().into_owned());
    dir.with_file_name(format!("{name}.part-{}-{n}", std::process::id()))
}

/// The first line of the compiler's `--version`: part of the second level's key.
fn compiler_line(program: &str) -> Result<String, String> {
    let out = Command::new(program)
        .arg("--version")
        .output()
        .map_err(|e| format!("the C++ compiler `{program}`: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(text.lines().next().unwrap_or_default().to_string())
}

/// Builds `output` in `cache` with the compiler `compiler()`, or finds it built.
pub fn build(output: &Output, cache: &Path) -> Result<Built, String> {
    let base = cache.join("cpu").join(output.key.hex());
    let sources = base.join("src");
    if !sources.is_dir() {
        let staged = staging(&sources);
        write_files(output, &staged)?;
        settle(&staged, &sources)?;
    }
    let program = compiler();
    let mut key = compiler_line(&program)?.into_bytes();
    for step in &output.build {
        key.push(0);
        key.extend_from_slice(step.program.as_bytes());
        for arg in &step.args {
            key.push(0);
            key.extend_from_slice(arg.as_bytes());
        }
    }
    let dir = base.join(&hex(&sha256(&key))[..16]);
    if dir.is_dir() {
        return Ok(Built { dir });
    }
    let staged = staging(&dir);
    write_files(output, &staged)?;
    for step in &output.build {
        let run = if step.program == "c++" {
            program.as_str()
        } else {
            step.program.as_str()
        };
        let done = Command::new(run)
            .args(&step.args)
            .current_dir(&staged)
            .output()
            .map_err(|e| format!("the C++ compiler `{run}`: {e}"))?;
        if !done.status.success() {
            return Err(format!(
                "`{run} {}` failed:\n{}",
                step.args.join(" "),
                String::from_utf8_lossy(&done.stderr)
            ));
        }
    }
    settle(&staged, &dir)?;
    Ok(Built { dir })
}
