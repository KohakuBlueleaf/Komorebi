# komorebi-ir

The Komorebi IR's Python package: the Rust core (`src/komorebi-ir/`, built with the
crate feature `python`) as the extension module `komorebi_ir._rs`, and the thin package
`src/komorebi_ir/` around it. It is not published.

## Building

```
pip install maturin
maturin build --release -m packaging/ir/pyproject.toml --out dist     # a wheel
maturin develop -m packaging/ir/pyproject.toml                          # into the venv
python -m pytest tests/ir/python
```

The tests compare the hashes with the command's, `komo hashes` (`cargo build`).
