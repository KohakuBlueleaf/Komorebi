"""Movement on the host: the pattern word, the movement form of a program's kernels,
buffer pairs with the check of the halves' roles in the sources, and KM_PAIR_CHECK."""

import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

import komorebi as km
from komorebi.device import mock
from komorebi.errors import ProgramError


def write(folder, **sources):
    for name, text in sources.items():
        (folder / f"{name}.cpp").write_text(text)
    return folder


def test_a_pattern_gives_its_word_and_the_movers_shares():
    pattern = km.Pattern("AAAB")
    assert pattern.word == 4 | 0b1000 << 8 and pattern.define() == "0x804"
    assert pattern.share("A") == Fraction(3, 4) and pattern.blocks("B", 9) == 2
    for letters in ("", "AC", "A" * 25):
        with pytest.raises(ValueError, match="letters A and B"):
            km.Pattern(letters)


def test_one_movement_source_serves_both_data_movers(tmp_path):
    folder = write(tmp_path, movement="", compute="")
    program = km.Program("p", kernels=folder, compile_args={"movement": [7]})
    sources = program.sources()
    assert sources["reader"] == sources["writer"] == folder / "movement.cpp"
    assert program.kernel_compile_args("writer") == [7]
    assert program.kernel_compile_args("compute") == []
    named = {"movement": folder / "movement.cpp", "compute": folder / "compute.cpp"}
    assert km.Program("p", kernels=named).sources()["reader"] == named["movement"]
    write(folder, reader="")
    with pytest.raises(ProgramError, match="keep one form"):
        km.Program("p", kernels=folder).sources()
    with pytest.raises(ProgramError, match="not both"):
        km.Program("p", kernels=folder, opt={"movement": "Os", "reader": "O2"})


def test_a_pair_is_two_buffers_and_a_half_keeps_one_filler_and_one_drainer(tmp_path):
    pair = km.BufferPair(3, 4, "bfloat16", 2048, (6, 2))
    program = km.Program("p", kernels=tmp_path, buffers=[pair])
    assert program.buffers == [
        km.Buffer(3, "bfloat16", 2048, 6),
        km.Buffer(4, "bfloat16", 2048, 2),
    ]
    with pytest.raises(ProgramError, match="declared twice"):
        km.Program("p", kernels=tmp_path, buffers=[pair, km.Buffer(4, "uint32", 64, 1)])
    header = "constexpr uint32_t kHalfB = 4;\n"
    (tmp_path / "pairs.h").write_text(header)
    compute = '#include "pairs.h"\ncb_wait_front(kHalfB, 1); cb_pop_front(kHalfB, 1);\n'
    write(tmp_path, movement="", compute=compute)
    assert program.sources()
    write(tmp_path, compute=compute + "cb_push_back(kHalfB, 1);\n")
    with pytest.raises(ProgramError, match="both fills and drains buffer 4"):
        program.sources()


# The movement example's kernels with their input pair at buffers 40 and 41, loaded on
# a mock card: with no define (as before), then with KM_PAIR_CHECK 1.
PAIR_LOAD = """
import shutil, sys
from pathlib import Path
import numpy as np
import komorebi as km

example, folder = Path(sys.argv[1]), Path(sys.argv[2])
shutil.copytree(example, folder)
header = folder / "movement.h"
text = header.read_text().replace("MoverA = 0;", "MoverA = 40;")
header.write_text(text.replace("MoverB = 1;", "MoverB = 41;"))
constants = km.abi.constants(header, known=km.abi.headers().constants)
with km.Session() as session:
    data = np.zeros((64, 1024), np.uint16)
    tensors = [session.tensor(data, "uint16"), session.tensor(data, "uint16")]
    args = tensors[0].tensor_args() + tensors[1].tensor_args()
    for defines in ({}, {"KM_PAIR_CHECK": 1}):
        pairs = [
            km.BufferPair(constants[f"k{n}MoverA"], constants[f"k{n}MoverB"],
                          "bfloat16", 2048, 4)
            for n in ("Input", "Output")
        ]
        program = km.Program(
            f"pairs{len(defines)}", kernels=folder, buffers=pairs, defines=defines,
            compile_args={"movement": args},
            handlers=header,
        )
        try:
            session.load(program).stop()
            print("BUILT", defines)
        except km.ProgramError as error:
            print("REFUSED", defines, str(error))
"""


@pytest.mark.tree
def test_a_pair_past_the_32_buffers_fails_the_build_only_with_the_check(tmp_path):
    example = Path(__file__).parents[3] / "examples" / "11_movement" / "kernels"
    env = dict(os.environ, **{mock.ACTIVE: "1"})
    env[mock.MOCK] = str(mock.descriptor((0,), "blackhole"))
    env[mock.MEMORY] = str(tmp_path / "kernel_buffer.json")
    env["TT_VISIBLE_DEVICES"] = "0"
    env.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(km.__file__).parents[1]), env.get("PYTHONPATH", "")]
    )
    done = subprocess.run(
        [sys.executable, "-c", PAIR_LOAD, str(example), str(tmp_path / "kernels")],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert "BUILT {}" in done.stdout, done.stdout[-3000:]
    refused = done.stdout.split("REFUSED {'KM_PAIR_CHECK': 1}", 1)[-1]
    assert "build failed" in refused, done.stdout[-3000:]
    assert "array subscript value '40'" in refused and "kInputMoverA" in refused
