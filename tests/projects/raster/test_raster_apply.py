"""APPLY in the frame program, on the host: the windows' layout over the frame's buffers
(apply.layout, frame_hosts) and its refusal, the defines the kernels read, the update
region's header words (apply.header) against thread/frame.h."""

import re
from types import SimpleNamespace

import pytest

from projects.raster import apply, card_thread, lists

KERNELS = lists.HERE / "kernels"
# plan.sizes of the two sets the README measures (only the numbers a layout reads)
SPONZA = {"own_most": 16, "k_most": 72, "rank_most": 5, "table_pages": 4}
BISTRO = {"own_most": 40, "k_most": 83, "rank_most": 12, "table_pages": 4}


def spans(sizes, windows):
    """Each window as (host, start, end) bytes."""
    rows = apply.buffers(sizes)
    return {
        name: (host, at, at + rows[name][1] * pages)
        for name, (host, at, pages) in windows.items()
    }


@pytest.mark.parametrize("sizes", [SPONZA, BISTRO])
def test_frame_layout_fits_apart(sizes):
    """Every APPLY buffer has a window inside one host buffer, on 64 bytes, its whole
    pages; no two windows overlap; x's pair lies in the page slots."""
    hosts = apply.frame_hosts()
    size_of = {index: size for _, index, size in hosts}
    windows = apply.layout(sizes, hosts)
    assert list(windows) == list(apply.buffers(sizes))
    placed = spans(sizes, windows)
    for name, (host, start, end) in placed.items():
        assert start % apply.WINDOW_ALIGN == 0
        assert 0 <= start < end <= size_of[host]
        assert windows[name][2] == apply.buffers(sizes)[name][2]
    slots = card_thread.SH["kCbShadeSlots"]
    assert placed["kCbApplyX"][0] == slots and placed["kCbApplyXB"][0] == slots
    ordered = sorted(placed.values())
    for (host_a, _, end_a), (host_b, start_b, _) in zip(ordered, ordered[1:]):
        assert host_a != host_b or end_a <= start_b


def test_frame_hosts_are_the_frame_buffers():
    """The hosts are the frame program's buffers of those indices and bytes."""
    frame = {
        buffer.index: buffer.page_bytes * buffer.pages
        for buffer in card_thread.shade_buffers()
    }
    show = card_thread.SHOW
    frame[show["kCbShow"]] = show["kShowPageBytes"] * show["kShowPages"]
    for _, index, size in apply.frame_hosts():
        assert frame[index] == size


def test_layout_refuses_with_numbers():
    """A set whose buffers fit no host is refused, naming the buffer and the bytes."""
    large = {**BISTRO, "own_most": 200}
    with pytest.raises(apply.SetRefused, match=r"kCbApplySums \(819,200 B\)"):
        apply.layout(large, apply.frame_hosts())
    wide = {**SPONZA, "k_most": 300}  # x's pair past the page slots
    with pytest.raises(apply.SetRefused, match="x's pair goes in the first"):
        apply.layout(wide, apply.frame_hosts())


def test_window_defines_name_every_buffer():
    """The defines give each buffer's host, byte and pages, and apply_window.hpp
    windows every buffer apply.py declares."""
    windows = apply.layout(BISTRO, apply.frame_hosts())
    defines = apply.window_defines(windows)
    source = (KERNELS / "apply_window.hpp").read_text()
    named = re.findall(r"APPLY_WINDOW\((kCb\w+)\);", source)
    assert sorted(named) == sorted(apply.buffers(BISTRO))
    for name, (host, at, pages) in windows.items():
        assert defines[f"APPLY_HOST_{name}"] == str(host)
        assert defines[f"APPLY_AT_{name}"] == str(at)
        assert defines[f"APPLY_PAGES_{name}"] == str(pages)
    parts = apply.frame_parts(BISTRO)
    assert all(buffer.pages == 1 for buffer in parts["buffers"])
    assert parts["defines"]["APPLY_OWN"] == "40"


def test_header_words():
    """The update region: a0 to a5 the five tensors and 0, a6 to a9 0, the group the
    cores from 0; its size and indices frame.h's."""
    F = card_thread.F
    made = {
        name: SimpleNamespace(address=0x1000 * (i + 1))
        for i, name in enumerate(apply.ORDER)
    }
    words = apply.header(made, 120)
    assert len(words) * 4 == F["RF_UPDATE_BYTES"]
    at = F["RF_U_A0"]
    tensors = [0x1000, 0x2000, 0x3000, 0x4000, 0x5000]
    assert list(words[at : at + 10]) == tensors + [0] * 5
    assert words[F["RF_U_GROUP"]] == 120 << 16
    assert not words[F["RF_U_GROUP"] + 1 :].any()
    assert F["RF_R_UPDATE"] == 10 and F["RF_REGIONS"] == 11
    assert F["RF_U_OTHER"] == 16 and F["RF_U_CODE"] < F["RF_U_OTHER"]
    assert F["RF_HANDLER_APPLY"] == card_thread.A["kHandlerApply"]
