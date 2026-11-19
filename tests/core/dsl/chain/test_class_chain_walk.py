"""The chain's walk in a class's kernels: built with a km.chain method or chain=True
(its handler after the class's, the chain's buffer 28, km_chain.hpp's reader, writer
and compute walk, the input's waits, the marks, a tree's credit), and nothing of it in
a class without; an overlay handler's case in it (no card)."""

from chain_fixture import Handlers, Steps, made, tensors

from komorebi import chain

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
