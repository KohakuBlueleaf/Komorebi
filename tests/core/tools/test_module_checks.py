"""The module checks: the statement caps, the allowlist that only shrinks, cycles and
imports inside functions, on small made-up packages."""

import ast

from komorebi.tools import module_checks

BIG = "src/komorebi/big.py"


def lines(count):
    return "x = 1\n" * count


def errors(sources, allow=None):
    return module_checks.check(sources, allow or {})[0]


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


def test_a_listed_file_that_grew_or_fell_under_the_cap_is_an_error():
    hard = {"over_hard_cap": {BIG: {"statements": 2100, "why": "w"}}}
    assert errors({BIG: lines(2101)}, hard) == [
        f"{BIG}: 2101 statements, grew past its 2100"
    ]
    assert errors({BIG: lines(2000)}, hard) == [
        f"{BIG}: 2000 statements, at 2000 or under: take it off"
    ]


def test_a_cycle_is_named_and_a_listed_one_must_still_be_one():
    sources = {
        "src/komorebi/__init__.py": "",
        "src/komorebi/a.py": "from komorebi import b\n",
        "src/komorebi/b.py": "from . import a\n",
        "src/komorebi/c.py": "def f():\n    from komorebi import a\n",
    }
    found = errors(sources)
    assert "import cycle among komorebi.a, komorebi.b" in found
    listed = {"cycle": [{"modules": ["komorebi.b", "komorebi.a"], "why": "w"}]}
    assert "import cycle among komorebi.a, komorebi.b" not in errors(sources, listed)
    gone = {"cycle": [{"modules": ["komorebi.a", "komorebi.c"], "why": "w"}]}
    assert any("komorebi.a, komorebi.c is not a cycle" in e for e in errors({}, gone))


def test_an_import_inside_a_function_must_be_listed():
    sources = {"src/komorebi/c.py": "class K:\n    def f(self):\n        import os\n"}
    assert errors(sources) == [
        "src/komorebi/c.py: `import os` inside K.f: not allowlisted"
    ]
    entry = {"path": "src/komorebi/c.py", "function": "K.f", "import": "import os"}
    assert errors(sources, {"function_import": [entry]}) == []
