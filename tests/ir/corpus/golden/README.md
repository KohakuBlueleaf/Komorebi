# The IR's golden corpus

Programs in the Komorebi IR's text form (`.kom`), used by the tests of the `komorebi-ir` crate.

| Folder | Holds |
|---|---|
| `lessons/` | The IR lessons of the tutorial, T1 to T13, T15 and T16 (T14 is host code), each exactly as the specification writes it. Every one checks clean |
| `reference/` | The other whole modules of the specification's reference sections. The first line is a comment naming the section; every one checks clean |
| `invalid/` | One small module for each thing program compile refuses. Beside each `<name>.kom`, a `<name>.expect` |

## The `.expect` files

One line for each diagnostic program compile gives for the module beside it: the diagnostic's code and
the line it points at, separated by one space.

```
K301 13
```

The set of lines is the whole expectation: a check of the module gives exactly these codes at exactly
these lines, no more. The codes are listed, each with the rule it stands for, in
[the catalogue of checks](../../../../docs/ir/checks.md).

The first line of each invalid module is a comment that says what it breaks and which section of the
specification states the rule.
