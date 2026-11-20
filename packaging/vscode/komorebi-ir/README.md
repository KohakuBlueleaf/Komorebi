# Komorebi IR for VS Code

Colours the text form of a Komorebi IR module (`.kom`): a TextMate grammar in `syntaxes/`
and the language's comment, brackets and quotes in `language-configuration.json`. No code runs.

Build the extension file and install it (in a remote window, run the second line in its terminal;
it installs on the remote side), or use "Extensions: Install from VSIX..." and pick the file:

    python3 packaging/vscode/build_vsix.py /tmp
    code --install-extension /tmp/komorebi-ir-0.1.0.vsix

Then reload the window. A linked folder is picked up only when it sits in the extensions folder
the running server uses (and then it follows the checkout); the VSIX is the documented way.

`cargo test --test editors` fails, naming the word, when the grammar misses a word of
`src/komorebi-ir/syntax/komorebi.pest` or of the checker (keywords, built-in functions and values, formats, failure names); the member names after a dot (`.rank`, `.tiles`, ...) are listed by hand from `docs/ir/operations.md` and the test does not cover them.
It also fails when a rule gives a scope the default themes (Dark+, Light+ and their Modern forms)
do not colour, so every name shows in a colour: a plain name is `variable.other.readwrite`, a
name written in capitals `variable.other.constant`, and the language's own words in an attribute
(`group`, `grid`, `build`, `worker`, ...) are `constant.language`; that list is by hand.
