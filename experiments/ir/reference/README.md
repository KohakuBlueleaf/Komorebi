# Komorebi IR reference material

The backends and the frontend material here were written before the Komorebi IR was finished.
They are kept to be read when the backend and frontend frameworks are designed; they are not built and not tested.
Where each part came from: `backend/tensix/` and `backend/cpu/` from `src/komorebi-ir/backend/`, `api/load.rs` from `src/komorebi-ir/api/load.rs`.
`tests/` from `tests/ir/` (`tensix.rs`, `tensix_tile.rs`, `tensix_rows.rs`, `tensix_chunks.rs`, `frontend.rs`, `parity.rs`, `tile_common.rs`, the folders `tensix/` (less `t1_class.kom` and `t1_places.txt`, now in `tests/ir/origin/`), `frontend/`, `parity/`, and the tests of `backend.rs` that name a concrete backend).
`examples/` from the eight `.kom` files directly in `examples/ir/` (`p1_...` to `p7_...`, `t1_...`): what the DSL frontend writes.
