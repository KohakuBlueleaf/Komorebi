"""A broadcast's sender / receiver split in a chain (no card): the sender's cast of
pages read is described as every rank's read (a receiver waits on its sender's
pages, or rectangle's); a cast of another form, or a read in the receiver's
branch, still refused."""

import ast

import komorebi as km
from komorebi.dsl.chain.chain_calls import handler_lines, table_read


class Job(km.Entry):
    """A job of a table: its first row of O, its row step."""

    o: int
    stride: int


class Tabled(km.Program):
    @km.handler
    def rows(self, out: km.Tensor, tables: km.Tensor, count: int):
        jobs = tables.pages(self.rank, 1).entries(Job, first=0, size=2, count=0)
        for job in jobs[0:count]:
            o0 = job.o // 2
            results = km.Stream(
                out.tiles[o0 : o0 + 4 * job.stride : job.stride, 0:2],
                write=True,
                block=2,
            )
            for _ in range(4):
                results.push(km.tile.zeros(km.fp32, 2))


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
    def pair(self, a: km.Tensor, out: km.Tensor, width: int):
        along = km.Broadcast(self, along="row", window=1)
        if along.sender():
            block = along.cast(a.tiles[0:1, 0:width].T, a.tiles[1:2, 0:width])
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


def test_a_cast_of_two_rectangles_turned_or_not_is_two_reads():
    lines = lines_of(Casts.pair)
    assert lines.refused is None
    reads = [a for a in lines.accesses if a.kind == "read"]
    assert [(a.tensor, ast.unparse(a.rect[0])) for a in reads] == [
        ("a", "0"),
        ("a", "1"),
    ]


def test_a_read_or_write_in_the_receivers_branch_is_refused():
    lines = lines_of(Casts.taken)
    assert "a broadcast's if is" in lines.refused


def test_a_stepped_write_at_a_tables_words_is_a_write_at_a_tables_words():
    method = Tabled.rows
    method.words = [("out", "tensor"), ("tables", "tensor"), ("count", "int")]
    lines = handler_lines(method)
    assert lines.refused is None
    (write,) = [a for a in lines.accesses if a.kind == "write"]
    assert write.tensor == "out" and write.step is not None
    assert table_read(write, lines)
