"""run.submit of a class's handler, not exclusive, warns when it reads a tensor that a
command on overlapping cores without its event yet writes (no card)."""

import warnings

import numpy as np
import pytest

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import moved_tensors
from komorebi.session.run import Run


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, address):
        self.page_size, self.num_pages, self.address = 4096, 64, address

    def accessor_args(self):
        return [2, self.page_size]


class Counts(km.Program):
    @km.handler
    def add(self, counts: km.Tensor, ones: km.Tensor):
        acc = km.tile.zeros(km.fp32)
        for tile in km.Stream(counts.pages(self.rank, 1)):
            acc += tile
        for tile in km.Stream(ones.pages(self.rank, 1)):
            acc += tile
        counts.pages(self.rank, 1).write(acc)


def tensor(address):
    return Tensor(Pages(address), (2048, 32), "float32", "tile")


def held():
    """A Counts as Run's order check reads it: made with no session."""
    tensors = {"counts": tensor(0x1000), "ones": tensor(0x80000)}
    program = Counts.__new__(Counts)
    program.moved_tensors = moved_tensors(compile_class(Counts, {}, tensors))
    return program


def run_of(program):
    """A Run with no card: only the columns the order check reads."""
    run = Run.__new__(Run)
    run.programs = [program]
    run._writers, run._base, run._handed, run._next = {}, 1, 1, 1
    run._order_methods = {}
    run._has = np.zeros(0, bool)
    return run


def test_a_handler_s_reads_and_writes_by_its_lines():
    run = run_of(held())
    assert run.programs[0].moved_tensors["add"] == (
        frozenset({"counts", "ones"}),
        frozenset({"counts"}),
    )


def test_a_read_of_what_a_command_still_out_writes_is_named():
    run = run_of(held())
    words, other = [0x1000, 0x80000], [0x100000, 0x80000]  # (counts, ones)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        written = run._check_order(1, words, 0, (0, 8), False)
        assert list(written.values()) == ["counts"]
        run._writers[1], run._next = ((0, 8), written), 2
        run._check_order(1, words, 0, (0, 8), True)  # exclusive
        run._check_order(1, words, 0, (8, 8), False)  # other cores
        run._check_order(1, other, 0, (0, 8), False)  # another tensor
    match = r"command 2 \(Counts.add, cores 4 to 11\) reads counts, which command 1"
    with pytest.warns(km.CommandOrderWarning, match=match):
        run._check_order(1, words, 0, (4, 8), False)


def test_a_command_whose_event_came_is_no_longer_named():
    run = run_of(held())
    words = [0x1000, 0x80000]
    run._writers[1] = ((0, 8), run._check_order(1, words, 0, (0, 8), False))
    run._handed, run._next, run._has = 2, 2, np.array([True])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run._check_order(1, words, 0, (0, 8), False)
    assert 1 not in run._writers
