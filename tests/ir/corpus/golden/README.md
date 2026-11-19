# The IR's golden corpus

Programs in the Komorebi IR's text form (`.kom`), used by the tests of the `komorebi-ir` crate.

| Folder | Holds |
|---|---|
| `lessons/` | The IR lessons of the tutorial, T1 to T13, T15 and T16 (T14 is host code), each exactly as the specification writes it. Every one checks clean |
| `reference/` | The other whole modules of the specification's reference. The first line is a comment naming the topic it comes from; every one checks clean |
| `invalid/` | One small module for each thing program compile refuses. Beside each `<name>.kom`, a `<name>.expect` |
| `run_fails/` | Modules that check clean and whose run ends in a named failure. Beside each `<name>.kom`, a `<name>.expect` |

## The `.expect` files

One line for each diagnostic, given as the diagnostic's code and the line it points at, separated by
one space.

```
K301 13
```

In `invalid/`, the lines are what program compile gives for the module: exactly these codes at exactly
these lines, no more. In `run_fails/`, program compile gives nothing, and the line is the failure the
run ends in: its code and the line of the operation that fails. The second comment line of a module
in `run_fails/` says what to run.

A diagnostic points at the line of the operation that completes the violation: the later of two
invocations that race, the write that overlaps a read, the second action on a message handled twice,
the head `on M(m)` of a body that leaves its message neither forwarded nor consumed.

The codes are listed, each with the rule it stands for, in
[the catalogue of checks](../../../../docs/ir/checks.md).

The first line of each module in `invalid/` and `run_fails/` is a comment that says what it breaks
and, in brackets, the topic of the specification that states the rule.
