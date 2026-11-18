"""A broadcast's sender / receiver split in a chain (no card): the sender's cast of
pages read is described as every rank's read (a receiver waits on its sender's
pages, or rectangle's); a cast of another form, or a read in the receiver's
branch, still refused."""

import ast

import komorebi as km
from komorebi.dsl.chain.chain_calls import handler_lines


class Casts(km.Program):
    @km.handler
    def spread(self, a: km.Tensor, out: km.Tensor, width: int):
        along = km.Broadcast(self, along="row", window=1)
        if along.sender():
            block = along.cast(a.pages(0, width).read())
        else:
            block = along.arrive()
        out.pages(self.rank * width, width).write(block)

    @km.handler
    def block(self, a: km.Tensor, out: km.Tensor, width: int):
        along = km.Broadcast(self, along="row", window=1)
        if along.sender():
            block = along.cast(a.tiles[0:1, 0:width].read())
        else:
            block = along.arrive()
        out.pages(self.rank * width, width).write(block)

    @km.handler
    def taken(self, a: km.Tensor, out: km.Tensor, width: int):
        along = km.Broadcast(self, along="row", window=1)
        if along.sender():
            block = along.cast(a.pages(0, width).read())
        else:
            block = along.arrive()
            out.pages(0, width).write(block)


def lines_of(method):
    """The handler's lines, its words as the compiler sets them at the make."""
    method.words = [("a", "tensor"), ("out", "tensor"), ("width", "int")]
    return handler_lines(method)


def test_a_senders_cast_is_every_ranks_read():
    lines = lines_of(Casts.spread)
    assert lines.refused is None
    read, write = lines.accesses
    assert (read.kind, read.tensor, read.leader) == ("read", "a", False)
    assert (write.kind, write.tensor) == ("write", "out")


def test_a_senders_cast_of_a_rectangle_is_every_ranks_read():
    lines = lines_of(Casts.block)
    assert lines.refused is None
    read = lines.accesses[0]
    assert (read.kind, read.tensor, read.leader) == ("read", "a", False)
    assert ast.unparse(read.rect[1]) == "1" and ast.unparse(read.rect[3]) == "width"


def test_a_read_or_write_in_the_receivers_branch_is_refused():
    lines = lines_of(Casts.taken)
    assert "a broadcast's if is" in lines.refused
