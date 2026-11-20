"""The chain's walk in a class's kernels (km.chain method or chain=True): its handler,
buffer 28, km_chain.hpp's walk in each kernel, nothing of it in a class without; the
same walk in the runtime's form; a chain build without a walk refused (no card)."""

from types import SimpleNamespace

import pytest
from chain_fixture import Handlers, Steps, made, quiet, tensors

from komorebi import chain
from komorebi.dsl.program import walk_refused
from komorebi.errors import ProgramError

GIVEN = tensors()
WORDS = chain.words()["kChainBuffer"]  # 28: the top of a program's 0 to 28


def test_a_class_without_a_chain_has_none_of_the_walk():
    program = made(Handlers, GIVEN)
    assert program.chain_handler == 0 and "km_chain" not in program.handlers
    assert all(buffer.index != WORDS for buffer in program.buffers)


def test_a_chain_build_walks_the_records_in_each_kernel():
    program = made(Steps, GIVEN)
    number = program.handlers["km_chain"]
    assert number == max(program.handlers.values()) == program.chain_handler
    chain_buffer = [b for b in program.buffers if b.index == WORDS == 28]
    assert len(chain_buffer) == 1 and chain_buffer[0].page_bytes == 2816


def test_chain_true_builds_the_walk_in_a_class_without_a_method():
    program = Handlers(
        l1=1 << 20, log=print, chain=True, a=GIVEN["x"], b=GIVEN["x"],
        out=GIVEN["y"], s=GIVEN["s"], t=GIVEN["t"],
    )  # fmt: skip
    assert program.chain_handler == program.handlers["km_chain"]


def made_runtime(cls, **options):
    x = GIVEN["x"]
    return cls(
        l1=1 << 20, log=quiet, form="runtime", a=x, b=x, out=GIVEN["y"],
        s=GIVEN["s"], t=GIVEN["t"], **options,
    )  # fmt: skip


def test_the_runtime_form_walks_the_records_in_its_parts():
    program = made_runtime(Steps)
    movement = (program.built.folder / "movement.cpp").read_text()
    compute = (program.built.folder / "compute.cpp").read_text()
    assert "if (km::handler(command) != kHandlerKmChain) {" in movement
    assert "km::chain_reader(movement, command, km_case);" in movement
    assert "km::chain_writer(km::ChainShape{}, movement, command," in movement
    assert "return km_chain_end(movement, command);" in movement
    assert "km::chain_compute(compute, command, km_case);" in compute
    assert "bool km_program(km::Writer& writer" in movement


def test_a_runtime_form_class_without_a_chain_has_no_walk():
    program = made_runtime(Handlers)
    movement = (program.built.folder / "movement.cpp").read_text()
    compute = (program.built.folder / "compute.cpp").read_text()
    assert "km_chain" not in movement and "km_chain" not in compute


def test_a_chain_build_whose_kernels_have_no_walk_is_refused():
    compiled = SimpleNamespace(chain=6, name="Prod")
    plain = {"movement.cpp": "switch", "compute.cpp": "km::chain_compute("}
    with pytest.raises(ProgramError, match="no chain walk") as error:
        walk_refused(compiled, plain, "runtime")
    assert "movement.cpp" in str(error.value)
    assert "done with no record run" in str(error.value)
    walk_refused(SimpleNamespace(chain=0, name="Prod"), plain, "runtime")


def test_chain_stamps_in_the_runtime_form_are_refused():
    with pytest.raises(ProgramError, match="not compiled into a program's parts"):
        made_runtime(Steps, chain_stamps=True)
