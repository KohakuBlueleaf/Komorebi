# tree-sitter grammar for Komorebi IR

The text form of a Komorebi IR module (`.kom`), as a tree-sitter grammar: `grammar.js` follows
`src/komorebi-ir/syntax/komorebi.pest` rule by rule, `queries/highlights.scm` colours it.
The generated parser (`src/`) is not kept in the repository.

    npm install tree-sitter-cli     # or: cargo install tree-sitter-cli
    npx tree-sitter generate        # writes src/parser.c in this folder
    npx tree-sitter parse some.kom  # the syntax tree; exit status 1 on an ERROR node

An editor that takes tree-sitter grammars builds `src/parser.c` (a C compiler) under the name
`komorebi_ir`, maps `kom` files to it and reads `queries/highlights.scm`. It parses every module
the parser of `komo check` accepts; it is looser in one place: any postfix expression may be assigned.
