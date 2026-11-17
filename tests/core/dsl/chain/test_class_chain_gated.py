"""A chain build's loop over streams in step: each read stream held by the line's gate
(km::Gated, km::ChainGate) until the record's input is marked, so a stream of a tensor
the chain does not write goes on meanwhile; a line of one stream, a prefetch=False
stream and a build without a chain keep their text (no card)."""

from chain_fixture import Handlers, Steps, made, tensors

import komorebi as km

GIVEN = tensors()


class LateZip(km.Program):
    """A loop over streams in step, one of them prefetch=False (its own file's class:
    the lines are this file's)."""

    @km.handler
    def add_late(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for x, y in zip(
            km.Stream(a.pages(first, width)),
            km.Stream(b.pages(first, width), prefetch=False),
        ):
            results.push(x + y)


def case(program, handler):
    text = (program.built.folder / "movement.cpp").read_text()
    return text.split(f"case kHandler{handler}: {{")[1].split("case kHandler")[0]


def chained(cls):
    x = GIVEN["x"]
    return cls(l1=1 << 20, log=print, chain=True, a=x, b=x, out=GIVEN["y"])


def test_a_loop_over_streams_in_step_holds_each_stream_by_the_line_s_gate():
    add = case(made(Steps, GIVEN), "Add")
    gate = (
        "km::ChainGate km_gate0 = km::chain_gate(km::ChainShape{}, movement, "
        "command.a4, 3u, km_chain_waited);"
    )
    move = (
        "if (!movement.move(km::Gated(km_read0, km_gate0, 1u), "
        "km::Gated(km_read1, km_gate0, 2u), km_write0)) {"
    )
    assert gate in add and move in add
    assert add.index(gate) < add.index("const km::Stream km_read0(") < add.index(move)
    assert "km_chain_read" not in add


def test_a_line_of_one_stream_keeps_the_record_s_whole_wait():
    scale = case(made(Steps, GIVEN), "Scale")
    assert "if (!km_chain_read(movement, command.a4, 1u)) {" in scale
    assert "km::Gated" not in scale and "km::chain_gate" not in scale


def test_a_prefetch_false_stream_in_step_keeps_the_record_s_whole_wait():
    add = case(chained(LateZip), "AddLate")
    wait = add.index("if (!km_chain_read(movement, ~0u, 3u)) {")  # the whole input
    assert wait < add.index("const km::Stream km_read0(")
    assert "km::Gated" not in add and "km::chain_gate" not in add


def test_a_build_without_a_chain_holds_no_stream():
    x = GIVEN["x"]
    program = Handlers(
        l1=1 << 20, log=print, a=x, b=x, out=GIVEN["y"], s=GIVEN["s"], t=GIVEN["t"]
    )
    text = (program.built.folder / "movement.cpp").read_text()
    assert "km::Gated" not in text and "km::chain_gate" not in text
    assert "movement.move(km_read0, km_read1, km_write0)" in text
