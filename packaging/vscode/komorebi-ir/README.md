# Komorebi IR for VS Code

Colours the text form of a Komorebi IR module (`.kom`): a TextMate grammar in `syntaxes/`
and the language's comment, brackets and quotes in `language-configuration.json`. No code runs.

Build the extension file and install it (in a remote window, run the second line in its terminal;
it installs on the remote side), or use "Extensions: Install from VSIX..." and pick the file:

    python3 packaging/vscode/build_vsix.py /tmp
    code --install-extension /tmp/komorebi-ir-0.1.0.vsix

Then reload the window. Linking this folder into an extensions folder by hand is not reliable:
VS Code registers an extension only through its own install.

`cargo test --test editors` fails, naming the word, when the grammar misses a word of
`src/komorebi-ir/syntax/komorebi.pest` or of the checker (keywords, built-in functions and values, formats, failure names); the member names after a dot (`.rank`, `.tiles`, ...) are listed by hand from `docs/ir/operations.md` and the test does not cover them.
