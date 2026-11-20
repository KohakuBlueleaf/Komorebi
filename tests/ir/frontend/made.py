"""A class form made both ways, for tests/ir/frontend.rs: today's emitter (its build
folder, kernels and buffers) and the DSL's frontend (its module's text and origin).
Run as `made.py <script> <class>` in the folder that holds a copy of the script from
tests/ir/tensix/, `src` on the path."""

import importlib
import sys

sys.path.insert(0, ".")

from komorebi.dsl.ir import module


def made_with(script):
    """The make's keywords: the script's `made_with()`, or the first lesson's."""
    if hasattr(script, "made_with"):
        return script.made_with()
    return {"l1": 1 << 20, "x": script.tensor(), "y": script.tensor()}


def main(name, class_name):
    script = importlib.import_module(name)
    cls = getattr(script, class_name)
    given = made_with(script)
    built = cls(log=lambda line: None, **given)
    print(f"FOLDER {built.built.folder}")
    # The kernels the program hands the runtime (role, source, describe's processor,
    # compile arguments, defines, options), its buffers (halves), then its pairs.
    for number, role in enumerate(("reader", "writer", "compute")):
        options = built.compute if role == "compute" else {}
        print(
            "KERNEL",
            role,
            built.sources()[role].name,
            number,
            "args=" + ",".join(str(word) for word in built.kernel_compile_args(role)),
            "defines=" + ",".join(f"{k}:{v}" for k, v in sorted(built.defines.items())),
            "options="
            + ",".join(
                f"{k}:{str(v).lower() if isinstance(v, bool) else v}"
                for k, v in sorted(options.items())
            ),
        )
    for buffer in built.buffers:
        print("BUFFER", buffer.index, buffer.format, buffer.page_bytes, buffer.pages)
    for pair in built.pairs:
        print("PAIR", pair.mover_a, pair.mover_b)
    made = module(cls, **given)
    for file in made.files:
        print(f"FILE {file}")
    for file, number, text in made.lines:
        print(f"LINE {file} {number} {text}")
    for row in made.map:
        print("MAP " + " ".join(str(n) for n in row))
    for safe, real in made.renames:
        print(f"RENAME {safe} {real}")
    print("TEXT")
    sys.stdout.write(made.text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
