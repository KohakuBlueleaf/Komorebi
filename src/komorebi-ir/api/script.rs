//! The run script (docs/ir/run-scripts.md): the tensors a host makes, the invocations
//! and control programs it runs, and the tensors it reads, as plain text. The same
//! script runs on the interpreter (`komo run`) and on a backend's driver.

use crate::interp::control::{Arg, Machine, Options, Report};
use crate::interp::memory::{Extent, Format, Scalar, Tensor, TensorId, Tile};
use crate::program::Module;
use crate::types::IslandGraph;

/// A run script, read.
#[derive(Clone, Debug, PartialEq)]
pub struct Script {
    /// The workers the module is loaded on: 1 unless an `options` line says.
    pub workers: usize,
    /// How many workers each island holds, in order; none: one island.
    pub islands: Option<Vec<usize>>,
    /// The pairs of islands joined; none: every pair.
    pub links: Option<Vec<(usize, usize)>>,
    /// The bytes of a worker's memory; none: the interpreter's default.
    pub memory: Option<u64>,
    /// The bound on a dynamic run's live messages; none: no bound.
    pub live: Option<u64>,
    /// The instructions an invocation may run on each worker; none: the default.
    pub steps: Option<u64>,
    /// The parameters bound at load, by name (`load` lines).
    pub bindings: Vec<(String, Scalar)>,
    /// The other lines, in order.
    pub lines: Vec<Line>,
}

/// One line of a run script after its options and its `load` lines.
#[derive(Clone, Debug, PartialEq)]
pub enum Line {
    /// `tensor NAME FORMAT ROWS COLS [replicated] [fill ...]`.
    Tensor {
        /// Its name.
        name: String,
        /// Its format.
        format: Format,
        /// Rows of elements.
        rows: usize,
        /// Columns of elements.
        cols: usize,
        /// One copy for each island.
        replicated: bool,
        /// What its elements hold at first.
        fill: Fill,
    },
    /// `invoke Program.name ARG... on FIRST COUNT [with NAME=VALUE:TYPE ...]`.
    Invoke {
        /// `Program.name`: a handler, a chain or a dynamic run.
        entry: String,
        /// The arguments.
        args: Vec<ScriptArg>,
        /// The first worker of the group.
        first: usize,
        /// The workers in the group.
        count: usize,
        /// The parameters bound at the invocation, by name.
        params: Vec<(String, Scalar)>,
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
    /// `state Program.member`: the exported state member as it is now, in the report.
    State(String),
}

/// What a tensor's elements hold at first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fill {
    /// Every element this value.
    Value(f64),
    /// Every element of tile `i` the value `i + k`.
    TilePlus(f64),
    /// Element `i`, counted row by row over the logical shape, the value `i + k`; an
    /// element past the shape 0.
    ElementPlus(f64),
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

/// `NAME=VALUE:TYPE`: a parameter's name and its value.
fn binding(word: &str) -> Result<(String, Scalar), String> {
    let (name, value) = word
        .split_once('=')
        .ok_or_else(|| format!("`{word}` is not NAME=VALUE:TYPE"))?;
    if name.is_empty() {
        return Err(format!("`{word}` is not NAME=VALUE:TYPE"));
    }
    let value = scalar(value).ok_or_else(|| format!("`{value}` is not a scalar"))?;
    Ok((name.to_string(), value))
}

/// `A-B,C-D,...`: the pairs of islands joined.
fn links(word: Option<&str>) -> Result<Vec<(usize, usize)>, String> {
    let word = word.ok_or_else(|| "the links are missing".to_string())?;
    word.split(',')
        .map(|pair| {
            let (a, b) = pair
                .split_once('-')
                .ok_or_else(|| format!("`{pair}` is not a link A-B"))?;
            Ok((number(Some(a), "an island")?, number(Some(b), "an island")?))
        })
        .collect()
}

fn fill(words: &[&str]) -> Result<Fill, String> {
    match words {
        [] => Ok(Fill::Value(0.0)),
        ["fill", f] => {
            if let Some(k) = f.strip_prefix("tile+") {
                number(Some(k), "a number").map(Fill::TilePlus)
            } else if let Some(k) = f.strip_prefix("element+") {
                number(Some(k), "a number").map(Fill::ElementPlus)
            } else if *f == "tile" {
                Ok(Fill::TilePlus(0.0))
            } else {
                number(Some(f), "a number").map(Fill::Value)
            }
        }
        _ => Err(format!(
            "`{}` is not `fill VALUE`, `fill tile+K` or `fill element+K`",
            words.join(" ")
        )),
    }
}

/// Reads a run script; an error names the line.
pub fn parse(text: &str) -> Result<Script, String> {
    let mut script = Script {
        workers: 1,
        islands: None,
        links: None,
        memory: None,
        live: None,
        steps: None,
        bindings: Vec::new(),
        lines: Vec::new(),
    };
    // whether a line other than `options` came: `options` comes first
    let mut begun = false;
    for (n, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or_default();
        let words: Vec<&str> = line.split_whitespace().collect();
        let at = |e: String| format!("line {}: {e}: `{}`", n + 1, line.trim());
        let parsed = match words.as_slice() {
            [] => continue,
            ["options", rest @ ..] => {
                if begun {
                    return Err(at("`options` comes before every other line".into()));
                }
                begun = true;
                let mut i = 0;
                while i < rest.len() {
                    let value = rest.get(i + 1).copied();
                    match rest[i] {
                        "workers" => script.workers = number(value, "a count").map_err(at)?,
                        "islands" => {
                            let sizes = value
                                .ok_or_else(|| at("the islands' sizes are missing".into()))?
                                .split(',')
                                .map(|s| number(Some(s), "a count"))
                                .collect::<Result<Vec<usize>, String>>()
                                .map_err(at)?;
                            script.islands = Some(sizes);
                        }
                        "links" => script.links = Some(links(value).map_err(at)?),
                        "memory" => {
                            script.memory = Some(number(value, "a count of bytes").map_err(at)?)
                        }
                        "live" => script.live = Some(number(value, "a count").map_err(at)?),
                        "steps" => script.steps = Some(number(value, "a count").map_err(at)?),
                        other => return Err(at(format!("`{other}` is not an option"))),
                    }
                    i += 2;
                }
                if script.links.is_some() && script.islands.is_none() {
                    return Err(at(
                        "`links` joins the islands an `islands` option gives".into()
                    ));
                }
                continue;
            }
            ["load", name, value] => {
                if !script.lines.is_empty() {
                    return Err(at(
                        "`load` comes before every `tensor`, `invoke`, `control`, `read` and \
                         `state` line"
                            .into(),
                    ));
                }
                begun = true;
                let value =
                    scalar(value).ok_or_else(|| at(format!("`{value}` is not a scalar")))?;
                script.bindings.push(((*name).to_string(), value));
                continue;
            }
            ["tensor", name, format, rows, cols, rest @ ..] => {
                let (replicated, rest) = match rest {
                    ["replicated", more @ ..] => (true, more),
                    _ => (false, rest),
                };
                Line::Tensor {
                    name: (*name).to_string(),
                    format: Format::from_name(format)
                        .ok_or_else(|| at(format!("`{format}` is not a format")))?,
                    rows: number(Some(rows), "a count").map_err(at)?,
                    cols: number(Some(cols), "a count").map_err(at)?,
                    replicated,
                    fill: fill(rest).map_err(at)?,
                }
            }
            ["invoke", entry, rest @ ..] => {
                let Some(on) = rest.iter().position(|w| *w == "on") else {
                    return Err(at("`invoke` names its group with `on FIRST COUNT`".into()));
                };
                let after = &rest[on + 1..];
                let (group, params) = match after.iter().position(|w| *w == "with") {
                    Some(w) if w + 1 == after.len() => {
                        return Err(at("`with` is followed by NAME=VALUE:TYPE ...".into()))
                    }
                    Some(w) => (&after[..w], &after[w + 1..]),
                    None => (after, &[][..]),
                };
                let [first, count] = group else {
                    return Err(at("`invoke` names its group with `on FIRST COUNT`".into()));
                };
                Line::Invoke {
                    entry: (*entry).to_string(),
                    args: args(&rest[..on]).map_err(at)?,
                    first: number(Some(first), "a worker").map_err(at)?,
                    count: number(Some(count), "a count").map_err(at)?,
                    params: params
                        .iter()
                        .map(|w| binding(w))
                        .collect::<Result<_, _>>()
                        .map_err(at)?,
                }
            }
            ["control", name, rest @ ..] => Line::Control {
                name: (*name).to_string(),
                args: args(rest).map_err(at)?,
            },
            ["read", name] => Line::Read((*name).to_string()),
            ["state", name] => Line::State((*name).to_string()),
            _ => return Err(at("not a line of a run script".into())),
        };
        begun = true;
        script.lines.push(parsed);
    }
    Ok(script)
}

/// The least common multiple of two counts above 0.
fn lcm(a: usize, b: usize) -> usize {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a.checked_div(x).map_or(0, |q| q * b)
}

/// A tensor stored in the module's default tile shape, filled as the script says. Its
/// rows are padded up to a multiple of every row count among the module's tile shapes
/// (their least common multiple), its columns likewise: whole tiles of every shape.
fn made(m: &Module, name: &str, format: Format, rows: usize, cols: usize, fill: Fill) -> Tensor {
    let shape = m.default_tile();
    let side = |f: fn(&crate::types::TileShape) -> u64| {
        m.tiles
            .iter()
            .map(|t| usize::try_from(f(t)).unwrap_or(1).max(1))
            .fold(1, lcm)
    };
    let (by_rows, by_cols) = (side(|t| t.rows), side(|t| t.cols));
    let padded = (
        rows.div_ceil(by_rows) * by_rows,
        cols.div_ceil(by_cols) * by_cols,
    );
    let mut t = Tensor::zeros(name, format, padded.0, padded.1, shape);
    t.shape = (rows, cols);
    let (h, w) = (shape.height(), shape.width());
    let grid_cols = t.grid.1;
    for (i, tile) in t.tiles.iter_mut().enumerate() {
        let values: Vec<f64> = (0..h.saturating_mul(w))
            .map(|k| match fill {
                Fill::Value(v) => v,
                Fill::TilePlus(k) => i as f64 + k,
                Fill::ElementPlus(plus) => {
                    let r = (i / grid_cols) * h + k / w;
                    let c = (i % grid_cols) * w + k % w;
                    if r < rows && c < cols {
                        (r * cols + c) as f64 + plus
                    } else {
                        0.0
                    }
                }
            })
            .collect();
        let filled = if format.is_integer() {
            let ints: Vec<i128> = values.iter().map(|v| *v as i128).collect();
            Tile::from_ints(format, shape, &ints)
        } else {
            Tile::from_f64(format, shape, &values)
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
    let defaults = Options::default();
    let options = Options {
        islands: script.islands.clone().map(|sizes| IslandGraph {
            sizes,
            links: script.links.clone(),
        }),
        worker_memory: script.memory.unwrap_or(defaults.worker_memory),
        live_bound: script.live.or(defaults.live_bound),
        step_budget: script.steps.unwrap_or(defaults.step_budget),
    };
    let bindings: Vec<(&str, Scalar)> = script
        .bindings
        .iter()
        .map(|(n, v)| (n.as_str(), *v))
        .collect();
    let mut machine =
        Machine::load(m, script.workers, &bindings, options).map_err(|r| r.to_string())?;
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
                replicated,
                fill,
            } => {
                let tensor = made(m, name, *format, *rows, *cols, *fill);
                let id = if *replicated {
                    machine.replicated(tensor)
                } else {
                    machine.tensor(tensor)
                };
                tensors.push((name.clone(), id));
            }
            Line::Invoke {
                entry,
                args,
                first,
                count,
                params,
            } => {
                let args = given(&tensors, args)?;
                let params: Vec<(&str, Scalar)> =
                    params.iter().map(|(n, v)| (n.as_str(), *v)).collect();
                machine.invoke(entry, &args, (*first, *count), &params);
            }
            Line::Control { name, args } => {
                let args = given(&tensors, args)?;
                machine.control(name, &args);
            }
            Line::Read(name) => machine.show_tensor(find(&tensors, name)?)?,
            Line::State(name) => machine.show_state(name)?,
        }
    }
    Ok(machine.report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32s(v: u64) -> Scalar {
        Scalar::Int(Format::U32, v)
    }

    /// The error of a script that does not read, which names the line.
    fn refused(text: &str) -> String {
        parse(text).expect_err("refused")
    }

    const COUNT: &str = "module count

tile [32, 32]
format u32

param ROOM: u32 = fit @bind(load)
param STEP: u32 = 8 @bind(invoke)

program P {
    state @scope(worker) @lifetime(program) {
        seen: u32[ROOM] @filled(element) @export
    }

    handler mark(x: tensor<u32> @replicated) -> u32 @on(group) {
        seen[0] = STEP
        let words = x.tiles(0, 1).words()
        return words[1]
    }
}
";

    #[test]
    fn options_take_every_word_in_any_order() {
        let s = parse("options steps 9 links 0-1,1-2 live 3 memory 4096 islands 1,1,1 workers 3\n")
            .expect("reads");
        assert_eq!(s.workers, 3);
        assert_eq!(s.islands, Some(vec![1, 1, 1]));
        assert_eq!(s.links, Some(vec![(0, 1), (1, 2)]));
        assert_eq!((s.memory, s.live, s.steps), (Some(4096), Some(3), Some(9)));
    }

    #[test]
    fn a_link_that_is_not_a_pair_is_refused() {
        assert_eq!(
            refused("options islands 1,1 links 0+1"),
            "line 1: `0+1` is not a link A-B: `options islands 1,1 links 0+1`"
        );
    }

    #[test]
    fn links_without_islands_are_refused() {
        assert_eq!(
            refused("options workers 2 links 0-1"),
            "line 1: `links` joins the islands an `islands` option gives: \
             `options workers 2 links 0-1`"
        );
    }

    #[test]
    fn options_after_another_line_are_refused() {
        assert_eq!(
            refused("load ROOM 2:u32\noptions workers 2"),
            "line 2: `options` comes before every other line: `options workers 2`"
        );
    }

    #[test]
    fn a_load_line_binds_a_parameter() {
        let s = parse("options workers 2\nload ROOM 2:u32\ntensor x u32 32 32\n").expect("reads");
        assert_eq!(s.bindings, vec![("ROOM".to_string(), u32s(2))]);
    }

    #[test]
    fn a_load_line_after_a_tensor_is_refused() {
        assert_eq!(
            refused("tensor x u32 32 32\nload ROOM 2:u32"),
            "line 2: `load` comes before every `tensor`, `invoke`, `control`, `read` and \
             `state` line: `load ROOM 2:u32`"
        );
    }

    #[test]
    fn a_load_line_without_a_scalar_is_refused() {
        assert_eq!(
            refused("load ROOM 2"),
            "line 1: `2` is not a scalar: `load ROOM 2`"
        );
    }

    #[test]
    fn a_tensor_can_be_replicated_and_filled_element_by_element() {
        let s = parse("tensor w f32 2 3 replicated fill element+1\n").expect("reads");
        assert_eq!(
            s.lines,
            vec![Line::Tensor {
                name: "w".into(),
                format: Format::F32,
                rows: 2,
                cols: 3,
                replicated: true,
                fill: Fill::ElementPlus(1.0),
            }]
        );
    }

    #[test]
    fn an_element_fill_counts_row_by_row_over_the_shape() {
        let m = crate::api::build(COUNT).expect("builds");
        let t = made(&m, "x", Format::U32, 33, 34, Fill::ElementPlus(5.0));
        let first = t.tiles[0].to_ints().expect("integers");
        assert_eq!(first[..3], [5, 6, 7]);
        // row 1, column 0 of the shape
        assert_eq!(first[32], 5 + 34);
        // tile 1 starts at column 32; its column 2 is past the shape
        let next = t.tiles[1].to_ints().expect("integers");
        assert_eq!(next[..3], [5 + 32, 5 + 33, 0]);
        // tile 3: row 32, columns 32 and 33; row 33 is past the shape
        let last = t.tiles[3].to_ints().expect("integers");
        assert_eq!(last[..3], [5 + 32 * 34 + 32, 5 + 32 * 34 + 33, 0]);
        assert_eq!(last[32], 0);
    }

    #[test]
    fn a_fill_that_is_not_one_of_the_three_is_refused() {
        assert_eq!(
            refused("tensor x u32 32 32 fill row+1"),
            "line 1: `row+1` is not a number: `tensor x u32 32 32 fill row+1`"
        );
        assert_eq!(
            refused("tensor x u32 32 32 fill 1 replicated"),
            "line 1: `fill 1 replicated` is not `fill VALUE`, `fill tile+K` or \
             `fill element+K`: `tensor x u32 32 32 fill 1 replicated`"
        );
    }

    #[test]
    fn an_invoke_binds_parameters_by_name() {
        let s = parse("invoke P.mark x on 0 2 with STEP=3:u32 OTHER=true\n").expect("reads");
        assert_eq!(
            s.lines,
            vec![Line::Invoke {
                entry: "P.mark".into(),
                args: vec![ScriptArg::Tensor("x".into())],
                first: 0,
                count: 2,
                params: vec![
                    ("STEP".to_string(), u32s(3)),
                    ("OTHER".to_string(), Scalar::Bool(true)),
                ],
            }]
        );
    }

    #[test]
    fn a_with_that_binds_nothing_is_refused() {
        assert_eq!(
            refused("invoke P.mark x on 0 2 with"),
            "line 1: `with` is followed by NAME=VALUE:TYPE ...: `invoke P.mark x on 0 2 with`"
        );
        assert_eq!(
            refused("invoke P.mark x on 0 2 with STEP"),
            "line 1: `STEP` is not NAME=VALUE:TYPE: `invoke P.mark x on 0 2 with STEP`"
        );
    }

    #[test]
    fn a_state_line_names_a_member() {
        let s = parse("state P.seen\n").expect("reads");
        assert_eq!(s.lines, vec![Line::State("P.seen".into())]);
    }

    #[test]
    fn an_unknown_line_names_its_number_and_its_text() {
        assert_eq!(
            refused("tensor x u32 32 32\n\nshow x"),
            "line 3: not a line of a run script: `show x`"
        );
    }

    #[test]
    fn the_new_lines_run() {
        let m = crate::api::build(COUNT).expect("builds");
        let s = parse(
            "options workers 2 islands 1,1 links 0-1 memory 64 live 4 steps 1000
load ROOM 2:u32
tensor x u32 32 32 replicated fill element+0
invoke P.mark x on 0 2 with STEP=3:u32
state P.seen
",
        )
        .expect("reads");
        let report = run(&m, &s).expect("runs");
        assert_eq!(
            report.print(),
            "#0 P.mark group(0, 2): completed 1:u32\n\
             state P.seen\n\
             \x20 worker 0: 3:u32, _\n\
             \x20 worker 1: 3:u32, _\n"
        );
    }
}
