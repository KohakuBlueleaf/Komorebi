# Komorebi IR reference material

The backends and the frontend here were written before the Komorebi IR was finished. They are
kept to be read when the backend and frontend frameworks are designed. They are not built and
not tested, and they write and read forms the IR no longer has.

| Here | Came from |
|---|---|
| `backend/tensix/`, `backend/cpu/` | `src/komorebi-ir/backend/` |
| `api/load.rs` | `src/komorebi-ir/api/load.rs` |
| `frontend/dsl_ir/` | The Python package `src/komorebi/dsl/ir/` (the DSL frontend) |
| `examples/` | The eight modules the DSL frontend wrote, once directly in `examples/ir/` |
| `docs/` | `docs/ir/backends.md` and `docs/ir/frontends.md` |

Their tests are in `tests/ir/reference/` (no test target builds them, and pytest does not
collect them).
