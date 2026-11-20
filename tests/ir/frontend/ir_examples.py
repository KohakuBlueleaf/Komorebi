"""The IR examples (examples/ir/): each program the DSL frontend takes end to end, its
module's text as `komorebi.dsl.ir.built.module` writes it, under a first line naming
the class. `python tests/ir/frontend/ir_examples.py [--write]`, in a venv with komorebi
and the komorebi-ir wheel: names each file that is not that text (exit 1), or writes."""

import importlib
import sys
from pathlib import Path

from komorebi.dsl.ir.built import module

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "tests" / "ir" / "tensix"
FOLDER = ROOT / "examples" / "ir"

# (file, script in tests/ir/tensix, class): T1, then the corpus's programs 1 to 7
EXAMPLES = [
    ("t1_scale_by_a_number.kom", "t1_class", "Scale"),
    ("p1_scalar_line.kom", "scalars_line", "Scalars"),
    ("p2_step_sum.kom", "steps_sum", "StepHandlers"),
    ("p3_step_add.kom", "steps_add", "StepHandlers"),
    ("p4_gate.kom", "gate_class", "Gate"),
    ("p5_movement_read_and_copy.kom", "movement_class", "Movement"),
    ("p6_tile_product.kom", "small_class", "Small"),
    ("p7_swiglu.kom", "swiglu_class", "SwiGLU"),
]


def text(script, name):
    """The example of class `name` of `script`: its first line, then the module."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    loaded = importlib.import_module(script)
    given = (
        loaded.made_with()
        if hasattr(loaded, "made_with")
        else {"l1": 1 << 20, "x": loaded.tensor(), "y": loaded.tensor()}
    )
    made = module(getattr(loaded, name), **given)
    first = f"// {name} of tests/ir/tensix/{script}.py, as the DSL frontend writes it"
    return f"{first}\n{made.text}"


def stale():
    """The examples whose file is not the frontend's text today, by file name."""
    return [
        file
        for file, script, name in EXAMPLES
        if not (FOLDER / file).is_file()
        or (FOLDER / file).read_text() != text(script, name)
    ]


def main(argv):
    if argv == ["--write"]:
        for file, script, name in EXAMPLES:
            (FOLDER / file).write_text(text(script, name))
        return 0
    if argv:
        print("usage: ir_examples.py [--write]", file=sys.stderr)
        return 2
    differ = stale()
    for file in differ:
        print(f"examples/ir/{file} is not the frontend's text: run with --write")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
