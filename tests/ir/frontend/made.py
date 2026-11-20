"""The class form of the first lesson made both ways, for tests/ir/frontend.rs: today's
emitter (its build folder) and the DSL's frontend (its module's text and origin). Run
in the folder that holds a copy of tests/ir/tensix/t1_class.py, `src` on the path.
"""

import sys

sys.path.insert(0, ".")

import t1_class

from komorebi.dsl.ir import module


def main():
    made_with = {"l1": 1 << 20, "x": t1_class.tensor(), "y": t1_class.tensor()}
    built = t1_class.Scale(log=lambda line: None, **made_with)
    print(f"FOLDER {built.built.folder}")
    made = module(t1_class.Scale, **made_with)
    for name in made.files:
        print(f"FILE {name}")
    for file, number, text in made.lines:
        print(f"LINE {file} {number} {text}")
    for row in made.map:
        print("MAP " + " ".join(str(n) for n in row))
    for safe, real in made.renames:
        print(f"RENAME {safe} {real}")
    print("TEXT")
    sys.stdout.write(made.text)


if __name__ == "__main__":
    main()
