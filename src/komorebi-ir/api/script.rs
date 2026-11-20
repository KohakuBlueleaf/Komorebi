//! The run script (docs/ir/run-scripts.md): the tensors a host makes, the invocations
//! and control programs it runs, and the tensors it reads, as plain text. The same
//! script runs on the interpreter (`komo run`) and on a backend's driver.

use crate::interp::control::{Arg, Islands, Machine, Options, Report};
use crate::interp::memory::{Format, Scalar, Tensor, TensorId, Tile};
use crate::program::Module;

/// A run script, read.
#[derive(Clone, Debug, PartialEq)]
pub struct Script {
    /// The workers the module is loaded on: 1 unless an `options` line says.
    pub workers: usize,
    /// How many workers each island holds, in order; none: one island.
    pub islands: Option<Vec<usize>>,
    /// The other lines, in order.
    pub lines: Vec<Line>,
}

/// One line of a run script after its options.
#[derive(Clone, Debug, PartialEq)]
pub enum Line {
    /// `tensor NAME FORMAT ROWS COLS [fill VALUE | fill tile+K]`.
    Tensor {
        /// Its name.
        name: String,
        /// Its format.
        format: Format,
        /// Rows of elements.
        rows: usize,
        /// Columns of elements.
        cols: usize,
        /// What its elements hold at first.
        fill: Fill,
    },
    /// `invoke Program.handler ARG... on FIRST COUNT`.
    Invoke {
        /// `Program.handler`.
        entry: String,
        /// The arguments.
        args: Vec<ScriptArg>,
        /// The first worker of the group.
        first: usize,
        /// The workers in the group.
        count: usize,
    },
    /// `control NAME ARG...`.
    Control {
        /// The control program.
        name: String,
        /// The arguments.
        args: Vec<ScriptArg>,
    },
    /// `read NAME`: the tensor as it is now, in the report.
    Read(String),
}

/// What a tensor's elements hold at first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fill {
    /// Every element this value.
    Value(f64),
    /// Every element of tile `i` the value `i + k`.
    TilePlus(f64),
}

/// An argument as a script writes it.
#[derive(Clone, Debug, PartialEq)]
pub enum ScriptArg {
    /// A tensor the script made, by name.
    Tensor(String),
    /// `VALUE:TYPE`, `true` or `false`.
    Scalar(Scalar),
}

/// A scalar as a script (and the report) writes it: `VALUE:TYPE`, `true`, `false`.
pub fn scalar(word: &str) -> Option<Scalar> {
    match word {
        "true" => return Some(Scalar::Bool(true)),
        "false" => return Some(Scalar::Bool(false)),
        _ => {}
    }
    let (value, ty) = word.rsplit_once(':')?;
    let format = Format::from_name(ty)?;
    format.as_scalar()?;
    if format.is_integer() {
        Scalar::int(format, value.parse::<i128>().ok()?)
    } else {
        Scalar::float(format, value.parse::<f64>().ok()?)
    }
}

fn number<T: std::str::FromStr>(word: Option<&str>, what: &str) -> Result<T, String> {
    let word = word.ok_or_else(|| format!("{what} is missing"))?;
    word.parse::<T>()
        .map_err(|_| format!("`{word}` is not {what}"))
}

fn args(words: &[&str]) -> Result<Vec<ScriptArg>, String> {
    words
        .iter()
        .map(|w| {
            if w.contains(':') || *w == "true" || *w == "false" {
                scalar(w)
                    .map(ScriptArg::Scalar)
                    .ok_or_else(|| format!("`{w}` is not a scalar"))
            } else {
                Ok(ScriptArg::Tensor((*w).to_string()))
            }
        })
        .collect()
}

fn fill(words: &[&str]) -> Result<Fill, String> {
    match words {
        [] => Ok(Fill::Value(0.0)),
        ["fill", f] => match f.strip_prefix("tile+") {
            Some(k) => number(Some(k), "a number").map(Fill::TilePlus),
            None if *f == "tile" => Ok(Fill::TilePlus(0.0)),
            None => number(Some(f), "a number").map(Fill::Value),
        },
        _ => Err(format!(
            "`{}` is not `fill VALUE` or `fill tile+K`",
            words.join(" ")
        )),
    }
}

/// Reads a run script; an error names the line.
pub fn parse(text: &str) -> Result<Script, String> {
    let mut script = Script {
        workers: 1,
        islands: None,
        lines: Vec::new(),
    };
    for (n, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or_default();
        let words: Vec<&str> = line.split_whitespace().collect();
        let at = |e: String| format!("line {}: {e}", n + 1);
        let parsed = match words.as_slice() {
            [] => continue,
            ["options", rest @ ..] => {
                if !script.lines.is_empty() {
                    return Err(at("`options` comes before every other line".into()));
                }
                let mut i = 0;
                while i < rest.len() {
                    match rest[i] {
                        "workers" => {
                            script.workers =
                                number(rest.get(i + 1).copied(), "a count").map_err(at)?
                        }
                        "islands" => {
                            let sizes = rest
                                .get(i + 1)
                                .ok_or_else(|| at("the islands' sizes are missing".into()))?
                                .split(',')
                                .map(|s| number(Some(s), "a count"))
                                .collect::<Result<Vec<usize>, String>>()
                                .map_err(at)?;
                            script.islands = Some(sizes);
                        }
                        other => return Err(at(format!("`{other}` is not an option"))),
                    }
                    i += 2;
                }
                continue;
            }
            ["tensor", name, format, rows, cols, rest @ ..] => Line::Tensor {
                name: (*name).to_string(),
                format: Format::from_name(format)
                    .ok_or_else(|| at(format!("`{format}` is not a format")))?,
                rows: number(Some(rows), "a count").map_err(at)?,
                cols: number(Some(cols), "a count").map_err(at)?,
                fill: fill(rest).map_err(at)?,
            },
            ["invoke", entry, rest @ ..] => {
                let Some(on) = rest.iter().position(|w| *w == "on") else {
                    return Err(at("`invoke` ends with `on FIRST COUNT`".into()));
                };
                let [first, count] = &rest[on + 1..] else {
                    return Err(at("`invoke` ends with `on FIRST COUNT`".into()));
                };
                Line::Invoke {
                    entry: (*entry).to_string(),
                    args: args(&rest[..on]).map_err(at)?,
                    first: number(Some(first), "a worker").map_err(at)?,
                    count: number(Some(count), "a count").map_err(at)?,
                }
            }
            ["control", name, rest @ ..] => Line::Control {
                name: (*name).to_string(),
                args: args(rest).map_err(at)?,
            },
            ["read", name] => Line::Read((*name).to_string()),
            _ => {
                return Err(at(format!(
                    "`{}` is not a line of a run script",
                    line.trim()
                )))
            }
        };
        script.lines.push(parsed);
    }
    Ok(script)
}

/// A tensor of the module's tile, filled as the script says.
fn made(m: &Module, name: &str, format: Format, rows: usize, cols: usize, fill: Fill) -> Tensor {
    let mut t = Tensor::zeros(name, format, rows, cols, m.tile);
    let count = t.tiles.first().map_or(0, |x| x.elements.len());
    for (i, tile) in t.tiles.iter_mut().enumerate() {
        let v = match fill {
            Fill::Value(v) => v,
            Fill::TilePlus(k) => i as f64 + k,
        };
        let filled = if format.is_integer() {
            Tile::from_ints(format, m.tile, &vec![v as i128; count])
        } else {
            Tile::from_f64(format, m.tile, &vec![v; count])
        };
        if let Some(filled) = filled {
            *tile = filled;
        }
    }
    t
}

/// Runs a script on the interpreter: its report, with each tensor read shown where the
/// script reads it. An error names what the script names that is not there.
pub fn run(m: &Module, script: &Script) -> Result<Report, String> {
    let options = Options {
        islands: script
            .islands
            .clone()
            .map(|sizes| Islands { sizes, links: None }),
        ..Options::default()
    };
    let mut machine = Machine::load(m, script.workers, &[], options).map_err(|r| r.to_string())?;
    let mut tensors: Vec<(String, TensorId)> = Vec::new();
    let find = |tensors: &[(String, TensorId)], name: &str| {
        tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| *t)
            .ok_or_else(|| format!("no tensor {name}"))
    };
    let given = |tensors: &[(String, TensorId)], args: &[ScriptArg]| {
        args.iter()
            .map(|a| match a {
                ScriptArg::Tensor(n) => find(tensors, n).map(Arg::Tensor),
                ScriptArg::Scalar(s) => Ok(Arg::Scalar(*s)),
            })
            .collect::<Result<Vec<Arg>, String>>()
    };
    for line in &script.lines {
        match line {
            Line::Tensor {
                name,
                format,
                rows,
                cols,
                fill,
            } => {
                let id = machine.tensor(made(m, name, *format, *rows, *cols, *fill));
                tensors.push((name.clone(), id));
            }
            Line::Invoke {
                entry,
                args,
                first,
                count,
            } => {
                let args = given(&tensors, args)?;
                machine.invoke(entry, &args, (*first, *count), &[]);
            }
            Line::Control { name, args } => {
                let args = given(&tensors, args)?;
                machine.control(name, &args);
            }
            Line::Read(name) => machine.show_tensor(find(&tensors, name)?)?,
        }
    }
    Ok(machine.report)
}
