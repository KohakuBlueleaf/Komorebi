# Komorebi IR for VS Code

Colours the text form of a Komorebi IR module (`.kom`): a TextMate grammar in `syntaxes/`
and the language's comment, brackets and quotes in `language-configuration.json`. No code runs.

Install by copying or linking this folder into `~/.vscode/extensions/` (a remote window reads
`~/.vscode-server/extensions/`), then reload the window:

    ln -s "$PWD" ~/.vscode/extensions/komorebi-ir

`cargo test --test editors` fails, naming the word, when the grammar misses a word of
`src/komorebi-ir/syntax/komorebi.pest` or of the checker (keywords, built-in functions and values, formats, failure names); the member names after a dot (`.rank`, `.tiles`, ...) are listed by hand from `docs/ir/operations.md` and the test does not cover them.
