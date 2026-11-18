"""A chain build's loop over streams in step: each read stream held by the line's gate
(km::Gated, km::ChainGate) until the record's input is marked, so a stream of a tensor
the chain does not write goes on meanwhile; a line of one stream, a prefetch=False
stream and a build without a chain keep their text (no card)."""

from chain_fixture import Steps, made, tensors

GIVEN = tensors()


def case(program, handler):
    text = (program.built.folder / "movement.cpp").read_text()
    return text.split(f"case kHandler{handler}: {{")[1].split("case kHandler")[0]


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
    # streams held by the gate until the record's input is marked: a race values miss
    assert gate in add and move in add
