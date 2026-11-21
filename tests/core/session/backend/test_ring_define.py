"""The ring bound a load's kernels are built for (KM_RING_DEPTH_MAX): the load's depth
rounded up to a power of two, or a bound a program declares; a depth past a declared
bound is refused by name before the build (no card)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from komorebi.errors import ProgramError
from komorebi.program import overlay
from komorebi.session import sizes


def program(name="sum", **defines):
    return SimpleNamespace(name=name, defines=defines)


@pytest.mark.parametrize(
    "depth, bound", [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (16, 16), (64, 64)]
)
def test_the_bound_is_the_depth_rounded_up_to_a_power_of_two(depth, bound):
    assert sizes.ring_define([program()], depth) == ("KM_RING_DEPTH_MAX", str(bound))


def test_a_declared_bound_is_kept_and_the_least_of_several_taken():
    placed = [program("a", KM_RING_DEPTH_MAX=16), program("b", KM_RING_DEPTH_MAX=8)]
    assert sizes.ring_define(placed, 4) == ("KM_RING_DEPTH_MAX", "8")


def test_a_depth_past_the_build_s_bound_is_refused_by_name():
    placed = [program("a"), program("small", KM_RING_DEPTH_MAX=8)]
    with pytest.raises(ProgramError, match="small: ring depth 16: .* at most 8 slots"):
        sizes.ring_define(placed, 16)


@pytest.mark.parametrize("bound", [0, 6, 128])
def test_a_declared_bound_not_a_power_of_two_in_the_ring_is_refused(bound):
    with pytest.raises(ProgramError, match=f"KM_RING_DEPTH_MAX={bound}: not a power"):
        sizes.ring_define([program(KM_RING_DEPTH_MAX=bound)], 1)


def recipe(*defines, folder=Path(".")):
    elf = folder / "k.elf"
    return overlay.Recipe("brisc", elf, ("g++",), "Os", (), (), (), "Os", defines)


def test_a_part_is_compiled_with_its_kernel_s_ring_bound():
    kernel = recipe("-DKM_RING_DEPTH_MAX=4", "-DFULL_KERNEL_NAME=x")
    words = kernel.compile_command("part.cpp", "part.o", {"KM_OVERLAY_PART": 1})
    assert "-DKM_RING_DEPTH_MAX=4" in words
    overlay.check_ring("o", "reader", kernel, {"KM_OVERLAY_PART": 1})


def test_a_part_s_ring_bound_is_in_its_build_key(tmp_path):
    (tmp_path / "k.elf").write_bytes(b"elf")
    (tmp_path / "reader.cpp").write_text("")
    made = overlay.declare("o", {"reader": tmp_path / "reader.cpp"}, 2)
    keys = {
        overlay.built_key(
            recipe(f"-DKM_RING_DEPTH_MAX={bound}", folder=tmp_path), made, "reader"
        )
        for bound in (4, 64)
    }
    assert len(keys) == 2


@pytest.mark.parametrize("kernel", [("-DKM_RING_DEPTH_MAX=4",), ()])
def test_a_part_with_another_ring_bound_than_its_kernel_is_refused(kernel):
    match = "o, reader part: KM_RING_DEPTH_MAX=64"
    with pytest.raises(overlay.OverlayError, match=match):
        overlay.check_ring("o", "reader", recipe(*kernel), {"KM_RING_DEPTH_MAX": 64})
