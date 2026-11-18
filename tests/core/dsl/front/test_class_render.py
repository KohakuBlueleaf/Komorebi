"""The class compiler's renderer: runs of one mover's nodes under #if / #else (no
card)."""

from komorebi.dsl.front.nodes import Line, render


def guards(movers):
    """The preprocessor lines of a run of one-line nodes on `movers`."""
    nodes = [Line(f"x{i};", mover=mover) for i, mover in enumerate(movers)]
    return [line for line in render(nodes, 0) if line.startswith("#")]


def test_a_then_b_share_one_if():
    assert guards("AB") == [
        "#if defined(COMPILE_FOR_BRISC)  // mover A",
        "#else  // mover B",
        "#endif",
    ]


def test_a_b_a_opens_a_new_if_not_a_second_else():
    assert guards("ABA") == [
        "#if defined(COMPILE_FOR_BRISC)  // mover A",
        "#else  // mover B",
        "#endif",
        "#if defined(COMPILE_FOR_BRISC)  // mover A",
        "#endif",
    ]
    lines = guards("BABA")
    for at, line in enumerate(lines):  # never two #else under one #if
        if line.startswith("#else"):
            assert lines[at - 1].startswith("#if")
