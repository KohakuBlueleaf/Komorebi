//! `komo`: the command of the Komorebi IR.
//!
//! `komo fmt [--check] <file.kom>` prints the canonical text (with `--check`, exits 1
//! when the file differs from it); `komo parse <file.kom>` exits 0 or prints diagnostics;
//! `komo check <file.kom>` runs program compile's checks: exits 0, or prints one line a
//! diagnostic, `file:line:col: code: text`, and exits 1.

use std::process::ExitCode;

const USAGE: &str =
    "usage: komo fmt [--check] <file.kom>\n       komo parse <file.kom>\n       komo check <file.kom>";

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
