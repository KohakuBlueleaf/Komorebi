//! Building the CPU backend's output, in a cache of two levels: the sources under the
//! output's key (`<cache>/cpu/<key>/src`), and what the compiler made under the key and
//! the compiler (`<cache>/cpu/<key>/<compiler>`), so another compiler rebuilds only
//! that. A build is made in a staging directory this call creates and renamed into
//! place when it is whole; one already in place is used as it is. The only directory
//! ever removed is a staging directory the same call created and lost the race with.

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

/// A staging directory: made only by `stage`, which created it, so it is the only
/// kind of path `place` ever removes.
struct Staged(PathBuf);

/// Creates a new staging directory beside `dir`: `<dir>.part-<pid>-<n>`. It fails
/// rather than reuse a directory that is there already.
fn stage(dir: &Path) -> Result<Staged, String> {
    let name = dir
        .file_name()
        .map_or(String::new(), |n| n.to_string_lossy().into_owned());
    let parent = dir
        .parent()
        .ok_or_else(|| format!("{} has no parent", dir.display()))?;
    std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let path = parent.join(format!("{name}.part-{}-{n}", std::process::id()));
    std::fs::create_dir(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Staged(path))
}

/// Puts the directory `dir` in place: `fill` fills a staging directory this call
/// creates beside it, which is then renamed to `dir`. When another build put `dir` in
/// place first, that one is kept whole and this call's staging directory is removed.
/// A `fill` that fails leaves its staging directory, and `dir` is not made.
pub fn place(dir: &Path, fill: impl FnOnce(&Path) -> Result<(), String>) -> Result<(), String> {
    let staged = stage(dir)?;
    fill(&staged.0)?;
    if std::fs::rename(&staged.0, dir).is_err() {
        if !dir.is_dir() {
            return Err(format!("{} could not be put in place", dir.display()));
        }
        let _ = std::fs::remove_dir_all(&staged.0);
    }
    Ok(())
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
        place(&sources, |staged| write_files(output, staged))?;
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
    place(&dir, |staged| {
        write_files(output, staged)?;
        for step in &output.build {
            let run = if step.program == "c++" {
                program.as_str()
            } else {
                step.program.as_str()
            };
            let done = Command::new(run)
                .args(&step.args)
                .current_dir(staged)
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
        Ok(())
    })?;
    Ok(Built { dir })
}
