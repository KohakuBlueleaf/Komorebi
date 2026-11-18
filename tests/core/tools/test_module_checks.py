"""The module checks: the statement caps, import cycles over every import (importlib
calls too), imports inside functions, the pending edges, on made-up packages."""

import ast

from komorebi.tools import module_checks

BIG = "src/komorebi/big.py"


def lines(count):
    return "x = 1\n" * count


def errors(sources, config=None, optional=()):
    return module_checks.check(sources, config or {}, set(optional))[0]


def test_statements_leave_out_imports_docstrings_and_layout():
    source = '"""doc"""\nimport os\n\n# note\nD = {\n    1: 2,\n}\n\n\ndef f():\n'
    source += '    """doc"""\n    from sys import path\n    if a:\n        return 1\n'
    assert module_checks.statements(ast.parse(source)) == 4  # D =, def, if, return


def test_over_the_soft_cap_warns_and_over_the_hard_cap_fails_unless_listed():
    assert module_checks.size_warnings({BIG: 1000}) == []
    assert module_checks.size_warnings({BIG: 1001}) != []
    assert errors({BIG: lines(2000)}) == []
    assert errors({BIG: lines(2001)}) == [
        f"{BIG}: 2001 statements, over the hard cap of 2000"
    ]
    hard = {"over_hard_cap": {BIG: {"statements": 2100, "why": "w"}}}
    assert errors({BIG: lines(2001)}, hard) == []


def test_a_folder_of_more_than_12_python_files_is_an_error_unless_listed():
    files = [f"tests/core/x/test_{i}.py" for i in range(12)] + ["tests/core/x/a.c"]
    assert module_checks.flat_errors(files) == []
    files.append("tests/core/x/test_12.py")
    assert module_checks.flat_errors([*files, "tests/core/x/y/one.py"]) == [
        "tests/core/x/: 13 files side by side: group them by concern"
    ]
    assert module_checks.flat_errors(files, {"tests/core/x": "one family"}) == []


def test_a_test_module_imported_by_another_test_file_is_an_error():
    sources = {
        "tests/x/helper_fixture.py": "import os\n",
        "tests/x/test_a.py": "from helper_fixture import made\n",
        "tests/x/test_b.py": "import pytest\n\nfrom test_a import made\n",
        "tests/x/test_c.py": "import tests.x.test_a\n",
    }
    assert module_checks.test_module_imports(sources) == [
        "tests/x/test_b.py:3: `from test_a import made`: a test module imported; move"
        " what is shared into a helper module beside it",
        "tests/x/test_c.py:1: `import tests.x.test_a`: a test module imported; move"
        " what is shared into a helper module beside it",
    ]


def test_a_listed_file_that_grew_or_fell_under_the_cap_is_an_error():
    hard = {"over_hard_cap": {BIG: {"statements": 2100, "why": "w"}}}
    assert errors({BIG: lines(2101)}, hard) == [
        f"{BIG}: 2101 statements, grew past its 2100"
    ]
    assert errors({BIG: lines(2000)}, hard) == [
        f"{BIG}: 2000 statements, at 2000 or under: take it off"
    ]


def test_a_cycle_is_named_counting_imports_inside_functions():
    sources = {
        "src/komorebi/__init__.py": "",
        "src/komorebi/a.py": "from komorebi import b\n",
        "src/komorebi/b.py": "import importlib\n\ndef f():\n"
        "    return importlib.import_module('komorebi.a')\n",
    }
    found = errors(sources)
    assert "import cycle among komorebi.a, komorebi.b" in found
    assert any("import_module('komorebi.a')` inside f" in e for e in found)


def test_an_import_inside_a_function_only_of_an_optional_package():
    sources = {
        "src/komorebi/c.py": "class K:\n    def f(self):\n        import os\n",
        "src/komorebi/d.py": "def g():\n    import torch\n",
        "src/komorebi/e.py": "import importlib\n\ndef h(name):\n"
        "    return importlib.import_module(name)\n",
    }
    found = errors(sources, optional={"torch"})
    assert len(found) == 2 and "`import os` inside K.f" in found[0]
    assert (
        "src/komorebi/e.py:4: `importlib.import_module(name)`: not a literal" in found
    )


def test_a_pending_edge_is_skipped_and_one_gone_is_an_error():
    sources = {
        "src/komorebi/a.py": "from komorebi import b\n",
        "src/komorebi/b.py": "from komorebi import a\n",
    }
    entry = {"path": "src/komorebi/b.py", "imports": "komorebi.a", "owner": "o"}
    assert errors(sources, {"pending": [entry]}) == []
    gone = errors({"src/komorebi/b.py": ""}, {"pending": [entry]})
    assert gone == [
        "src/komorebi/b.py: the edge to komorebi.a (o) is gone: take it off the"
        " pending list"
    ]


def test_the_package_lazy_names_are_exempt():
    lazy = "import importlib\n\ndef __getattr__(name):\n"
    lazy += "    return importlib.import_module('komorebi.a')\n"
    sources = {
        "src/komorebi/__init__.py": lazy,
        "src/komorebi/a.py": "import komorebi\n",
    }
    assert errors(sources) == []
    assert errors({"src/komorebi/x.py": lazy, "src/komorebi/a.py": ""}) != []


def test_without_check_the_table_of_the_repository_prints(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["module_checks"])
    assert module_checks.main() == 0
    out = capsys.readouterr().out
    assert out.startswith("statements | file\n") and "src/komorebi/" in out
