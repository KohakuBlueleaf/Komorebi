"""A class's command from its thread on a tensor other than the make's: <handler>_with
takes each tensor parameter's load-word place (the make's tensors and thread_tensors=)
and refuses one of another kind than the make's (no card)."""

from pathlib import Path

from class_thread_header_fixture import COMPILER, Counted, Pages, Plain, quiet

import komorebi as km
from komorebi.device.tensor import Tensor

HERE = Path(__file__).resolve().parent / "class_thread"


def grid(columns, address, dtype="bfloat16", page=2048):
    return Tensor(Pages(page, columns, address), (32, columns * 32), dtype, "tile")


class Other(Plain):
    spare = km.region(64)  # a thread takes one region at least
    thread = km.x280_thread(str(HERE / "other.c"), cluster=1)


@COMPILER
def test_a_thread_names_another_tensor_and_one_of_another_kind_is_refused():
    z, wide = grid(8, 0x11000), grid(8, 0x21000, "float32", 4096)
    program = Other(l1=1 << 20, log=quiet, a=grid(8, 0x1000), out=grid(8, 0x9000),
                    thread_tensors={"z": z, "wide": wide})  # fmt: skip
    thread = program.threads[min(program.threads)]
    text = (Path(thread.includes[0]) / "other_thread.h").read_text()
    assert "#define OTHER_WORD_Z (OTHER_WORDS_OWN + 2u)" in text
    assert "    case OTHER_WORD_Z: return 1u;" in text  # the make's a and out: kind 1
    assert "    case OTHER_WORD_WIDE: return 2u;" in text
    assert "static inline int other_scale_with(" in text
    assert "    km_command->a1 = other_word(out_place);" in text
    assert thread.words[-2:] == [0x11000, 0x21000]  # thread_tensors= as load words


@COMPILER
def test_without_a_tensor_parameter_there_is_no_with_function():
    program = Counted(l1=1 << 20, log=quiet, SIZE=8192)
    assert "_with(" not in (program.built.folder / "counted_thread.h").read_text()
