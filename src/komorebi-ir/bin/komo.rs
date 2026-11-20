//! `komo`: the command of the Komorebi IR.
//!
//! `komo fmt [--check] <file.kom>` prints the canonical text (with `--check`, exits 1
//! when the file differs from it); `komo parse <file.kom>` exits 0 or prints diagnostics;
//! `komo check <file.kom>` runs program compile's checks: exits 0, or prints one line a
//! diagnostic, `file:line:col: code: text`, and exits 1.
//!
//! `komo compile <file.kom> [-o <file.komc>]` runs the same checks and writes the
//! byte-code program (by default beside the text, as `.komc`). `komo hashes <file>` prints
//! the three hashes, `semantic`, `build` and `full`, one a line; `komo print <file>`
//! prints the program structure. Both take a `.kom` (checked, as `komo check` does) or a
//! `.komc` (read, validated, derived again and verified: a refusal is
//! `file.komc: byte N: code: text`).
//!
//! `komo run <file> <script>` runs a run script (docs/ir/run-scripts.md) on the
//! interpreter and prints its report: each invocation's end, and each tensor it reads.

use std::process::ExitCode;

const USAGE: &str = "usage: komo fmt [--check] <file.kom>
       komo parse <file.kom>
       komo check <file.kom>
       komo compile <file.kom> [-o <file.komc>]
       komo hashes <file.kom|file.komc>
       komo print <file.kom|file.komc>
       komo run <file.kom|file.komc> <script>";

fn main() -> ExitCode {
    komorebi_ir::deep::on_deep_stack(run)
}

fn run() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    match words.as_slice() {
        ["fmt", file] => fmt(file, false),
        ["fmt", "--check", file] | ["fmt", file, "--check"] => fmt(file, true),
        ["parse", file] => parse(file),
        ["check", file] => check(file),
        ["compile", file] => compile(file, None),
        ["compile", file, "-o", out] | ["compile", "-o", out, file] => compile(file, Some(out)),
        ["hashes", file] => with_module(file, |m| {
            let h = komorebi_ir::bytecode::hashes(m);
            println!("semantic {}", komorebi_ir::bytecode::hex(&h.semantic));
            println!("build    {}", komorebi_ir::bytecode::hex(&h.build));
            println!("full     {}", komorebi_ir::bytecode::hex(&h.full));
        }),
        ["print", file] => with_module(file, |m| print!("{}", komorebi_ir::program::print(m))),
        ["run", file, script] => run_script(file, script),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn read(file: &str) -> Result<String, ExitCode> {
    std::fs::read_to_string(file).map_err(|error| {
        eprintln!("{file}: {error}");
        ExitCode::from(2)
    })
}

fn report(file: &str, diagnostics: &[komorebi_ir::Diagnostic]) -> ExitCode {
    for diagnostic in diagnostics {
        eprintln!("{}", diagnostic.render(file));
    }
    ExitCode::from(1)
}

fn fmt(file: &str, check: bool) -> ExitCode {
    let source = match read(file) {
        Ok(source) => source,
        Err(code) => return code,
    };
    match komorebi_ir::format(&source) {
        Err(diagnostics) => report(file, &diagnostics),
        Ok(text) if check => {
            if text == source {
                ExitCode::SUCCESS
            } else {
                eprintln!("{file}: not in canonical layout");
                ExitCode::from(1)
            }
        }
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
    }
}

fn parse(file: &str) -> ExitCode {
    let source = match read(file) {
        Ok(source) => source,
        Err(code) => return code,
    };
    match komorebi_ir::parse(&source) {
        Ok(_) => ExitCode::SUCCESS,
        Err(diagnostics) => report(file, &diagnostics),
    }
}

fn check(file: &str) -> ExitCode {
    let source = match read(file) {
        Ok(source) => source,
        Err(code) => return code,
    };
    match komorebi_ir::check(&source) {
        Ok(_) => ExitCode::SUCCESS,
        Err(diagnostics) => report(file, &diagnostics),
    }
}

fn compile(file: &str, out: Option<&str>) -> ExitCode {
    let source = match read(file) {
        Ok(source) => source,
        Err(code) => return code,
    };
    let bytes = match komorebi_ir::compile(&source) {
        Ok(bytes) => bytes,
        Err(diagnostics) => return report(file, &diagnostics),
    };
    let out = match out {
        Some(out) => out.to_string(),
        None => format!("{}.komc", file.strip_suffix(".kom").unwrap_or(file)),
    };
    match std::fs::write(&out, bytes) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{out}: {error}");
            ExitCode::from(2)
        }
    }
}

fn run_script(file: &str, script: &str) -> ExitCode {
    let text = match read(script) {
        Ok(text) => text,
        Err(code) => return code,
    };
    let script_read = match komorebi_ir::api::script::parse(&text) {
        Ok(s) => s,
        Err(error) => {
            eprintln!("{script}: {error}");
            return ExitCode::from(2);
        }
    };
    let mut failed = None;
    let code = with_module(file, |m| {
        match komorebi_ir::api::script::run(m, &script_read) {
            Ok(report) => print!("{}", report.print()),
            Err(error) => failed = Some(error),
        }
    });
    match failed {
        Some(error) => {
            eprintln!("{script}: {error}");
            ExitCode::from(2)
        }
        None => code,
    }
}

/// Runs `then` on the program structure of a `.kom` (checked) or of a `.komc` (loaded:
/// `api::load`).
fn with_module(file: &str, then: impl FnOnce(&komorebi_ir::program::Module)) -> ExitCode {
    let module = if file.ends_with(".komc") {
        let bytes = match std::fs::read(file) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("{file}: {error}");
                return ExitCode::from(2);
            }
        };
        match komorebi_ir::api::load(&bytes) {
            Ok(module) => module,
            Err(refusals) => {
                for refusal in refusals {
                    match refusal.at {
                        Some(at) => {
                            eprintln!("{file}: byte {at}: {}: {}", refusal.code, refusal.what)
                        }
                        None => eprintln!("{file}: {}: {}", refusal.code, refusal.what),
                    }
                }
                return ExitCode::from(1);
            }
        }
    } else {
        let source = match read(file) {
            Ok(source) => source,
            Err(code) => return code,
        };
        match komorebi_ir::api::build(&source) {
            Ok(module) => module,
            Err(diagnostics) => return report(file, &diagnostics),
        }
    };
    then(&module);
    ExitCode::SUCCESS
}
