"""The chain's walk in a class's kernels: built with a km.chain method or chain=True
(its handler after the class's, the chain's buffer 32, km_chain.hpp's reader, writer
and compute walk, the input's waits, the marks, a tree's credit), and nothing of it in
a class without; an overlay handler's case in it (no card)."""

from chain_fixture import Handlers, Steps, made, tensors

import komorebi as km

GIVEN = tensors()


class OverlaySteps(Steps):
    """Steps with SCALE and SUM as overlays (ADD resident)."""

    scale = km.handler(overlay=True)(Steps.scale.function)
    sum = km.handler(overlay=True)(Steps.sum.function)


def sources(program):
    folder = program.built.folder
    return {name: (folder / name).read_text() for name in program.built.files}


def test_a_class_without_a_chain_has_none_of_the_walk():
    program = made(Handlers, GIVEN)
    text = sources(program)
    assert program.chain_handler == 0 and "km_chain" not in program.handlers
    assert not any(
        word in body
        for body in text.values()
        for word in ("km_chain", "km::chain_", "KmChain")
    )
    assert all(buffer.index != 32 for buffer in program.buffers)


def test_a_chain_build_walks_the_records_in_each_kernel():
    program = made(Steps, GIVEN)
    text = sources(program)
    number = program.handlers["km_chain"]
    assert number == max(program.handlers.values()) == program.chain_handler
    assert f"constexpr uint32_t kHandlerKmChain = {number};" in text["steps.h"]
    movement, compute = text["movement.cpp"], text["compute.cpp"]
    assert "km::chain_reader(movement, command, km_case);" in movement
    assert "km::chain_writer(km::ChainShape{}, movement, command, km_case);" in movement
    assert movement.count("return km_chain_end(movement, command);") == len(
        [name for name in program.handlers if name != "km_chain"]
    )
    assert "!km::chain_credit(movement) || !km::reduce_emit(" in movement
    assert "km::chain_compute(compute, command, km_case);" in compute
    chain_buffer = [b for b in program.buffers if b.index == 32]
    assert len(chain_buffer) == 1 and chain_buffer[0].page_bytes == 2816


def test_chain_true_builds_the_walk_in_a_class_without_a_method():
    program = Handlers(
        l1=1 << 20, log=print, chain=True, a=GIVEN["x"], b=GIVEN["x"],
        out=GIVEN["y"], s=GIVEN["s"], t=GIVEN["t"],
    )  # fmt: skip
    assert program.chain_handler == program.handlers["km_chain"]


def test_an_overlay_handler_s_case_loads_waits_for_the_input_calls_and_marks():
    program = made(OverlaySteps, GIVEN)
    text = sources(program)
    movement = text["movement.cpp"]
    case = movement.split("case kHandlerScale: {")[1].split("case kHandler")[0]
    order = [
        "km::Overlay km_part(movement, kOverlayScale);",
        "if (!km_part.load()) {",
        "if (!km_chain_read(movement, command.a4, 1u)) {",
        "if (km_part.call(command) == 0) {",
        "return km_chain_end(movement, command);",
    ]
    assert [case.index(line) for line in order] == sorted(
        case.index(line) for line in order
    )
    assert "km_rank" not in case  # the body is the part's
    resident = movement.split("case kHandlerAdd: {")[1].split("case kHandler")[0]
    assert "km::Overlay" not in resident and "km_chain_read" in resident
    part = text["sum/movement.cpp"]
    assert '#include "komorebi/device/km_chain.hpp"' in part
    assert "!km::chain_credit(movement) || !km::reduce_emit(" in part
    assert "kChainArea" not in part and "chain_now" not in part
    assert "km::chain_compute(compute, command, km_case);" in text["compute.cpp"]
    assert "km::Overlay km_part(compute, kOverlaySum);" in text["compute.cpp"]
