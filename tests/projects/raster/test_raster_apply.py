"""APPLY in the frame program, on the host: the windows' layout over the frame's buffers
(apply.layout, frame_hosts) and its refusal, the defines the kernels read, the update
region's header words (apply.header) against thread/frame.h."""

import re
from types import SimpleNamespace

import numpy as np
import pytest

from projects.raster import apply, card_thread, lists

KERNELS = lists.HERE / "kernels"
# plan.sizes of the two sets the README measures (only the numbers a layout reads)
SPONZA = {"own_most": 16, "k_most": 72, "rank_most": 5, "table_pages": 4}
BISTRO = {"own_most": 40, "k_most": 83, "rank_most": 12, "table_pages": 4}
HALL = {"own_most": 35, "k_most": 95, "rank_most": 11, "table_pages": 4}


def frame_hosts():
    """The hosts of the default lit frame's program."""
    held = card_thread.capacity([(1024, 1024)], True)
    return apply.frame_hosts(card_thread.frame_buffers(held))


def spans(sizes, windows):
    """Each window as (host, start, end) bytes."""
    rows = apply.buffers(sizes)
    return {
        name: (host, at, at + rows[name][1] * pages)
        for name, (host, at, pages) in windows.items()
    }


@pytest.mark.parametrize("sizes", [SPONZA, BISTRO, HALL])
def test_frame_layout_fits_apart(sizes):
    """Every APPLY buffer has a window inside one host range, on 64 bytes, its whole
    pages; no two windows overlap; x's pair lies in the page slots' run."""
    hosts = frame_hosts()
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


def test_frame_hosts_are_runs_of_the_frame_buffers():
    """The default frame's hosts: the page slots, SHADE's and LIGHT's work (22 to 24)
    as one run, SHOW's ring, the planes and the table (13 to 15) as another; each run's
    bytes the frame program's buffers' end to end."""
    sh, show = card_thread.SH, card_thread.SHOW
    frame = {
        buffer.index: buffer.page_bytes * buffer.pages
        for buffer in card_thread.shade_buffers()
    }
    frame[show["kCbShow"]] = show["kShowPageBytes"] * show["kShowPages"]
    hosts = frame_hosts()
    assert [index for _, index, _ in hosts] == [sh["kCbShadeSlots"], show["kCbShow"]]
    assert hosts[0][2] == sum(frame[index] for index in (22, 23, 24))
    assert hosts[1][2] == sum(frame[index] for index in (13, 14, 15))
    assert hosts[0][0] == "kCbShadeSlots+kCbShadeWork+kCbLightWork"


def test_frame_runs_follow_the_addresses():
    """A run of three adjacent dead buffers (an alignment gap inside it unused), a live
    buffer between breaking a run, the run of the first dead one first."""
    cbs = [
        (5, "uint32", 4096, 1),  # dead, alone: 6 is live
        (6, "uint32", 100, 1),
        (7, "uint32", 40, 1),  # dead: 7, 8, 9 a run (8 starts on 64 bytes)
        (8, "uint32", 64, 2),
        (9, "uint32", 64, 1),
        (10, "uint32", 64, 1),  # live
    ]
    dead = {7: "a", 8: "b", 9: "c", 5: "d"}
    runs = apply.frame_runs(cbs, dead)
    assert runs == [("a+b+c", 7, 64 + 128 + 64), ("d", 5, 4096)]
    assert apply.frame_runs(cbs, {5: "d", 7: "a"}) == [("d", 5, 4096), ("a", 7, 40)]


def test_frame_runs_take_the_addresses_not_the_indices():
    """Consecutive indices with a live buffer of another index placed between them are
    no run: the addresses decide."""
    cbs = [(1, "uint32", 64, 1), (3, "uint32", 64, 1), (2, "uint32", 64, 1)]
    assert apply.frame_runs(cbs, {1: "a", 3: "c"}) == [("a", 1, 64), ("c", 3, 64)]
    assert apply.frame_runs(cbs, {1: "a", 2: "b"}) == [("a+b", 1, 128)]


def test_layout_refuses_with_numbers():
    """A set whose buffers fit no host is refused, naming the buffer and the bytes."""
    large = {**BISTRO, "own_most": 200}
    with pytest.raises(apply.SetRefused, match=r"kCbApplySums \(819,200 B\)"):
        apply.layout(large, frame_hosts())
    wide = {**SPONZA, "k_most": 400}  # x's pair past the page slots' run
    with pytest.raises(apply.SetRefused, match="x's pair goes in the first"):
        apply.layout(wide, frame_hosts())


def test_window_defines_name_every_buffer():
    """The defines give each buffer's host, byte and pages, and apply_window.hpp
    windows every buffer apply.py declares."""
    windows = apply.layout(BISTRO, frame_hosts())
    defines = apply.window_defines(windows)
    source = (KERNELS / "apply_window.hpp").read_text()
    named = re.findall(r"APPLY_WINDOW\((kCb\w+)\);", source)
    assert sorted(named) == sorted(apply.buffers(BISTRO))
    for name, (host, at, pages) in windows.items():
        assert defines[f"APPLY_HOST_{name}"] == str(host)
        assert defines[f"APPLY_AT_{name}"] == str(at)
        assert defines[f"APPLY_PAGES_{name}"] == str(pages)
    held = card_thread.capacity([(1024, 1024)], True)
    parts = apply.frame_parts(BISTRO, card_thread.frame_buffers(held))
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


def test_field_set_names_the_files_missing(tmp_path):
    """The field's update on the card without the scene's transport set: refused, the
    missing files named."""
    folder = tmp_path / "public" / "sponza_transport"
    folder.mkdir(parents=True)
    (folder / "perm.npy").write_bytes(b"")
    with pytest.raises(FileNotFoundError) as refused:
        apply.field_set("sponza_overview", tmp_path, 120)
    reason = str(refused.value)
    assert "blocks.npz" in reason and "weights.f32" in reason
    assert "perm.npy" not in reason


def test_the_update_region_joins_both_headers():
    """card_thread.update_words: words 0 to 15 APPLY's header's, the others the
    field's update's; its bytes alone when neither is given."""
    F = card_thread.F
    words = F["RF_UPDATE_BYTES"] // 4
    field = {"update": np.arange(words, dtype=np.uint32) + 1000}
    header = np.arange(words, dtype=np.uint32) + 1
    joined = card_thread.update_words(field, header)
    own = F["RF_U_OTHER"]
    assert list(joined[:own]) == list(header[:own])
    assert list(joined[own:]) == list(field["update"][own:])
    assert not card_thread.update_words(None, header)[own:].any()
    assert card_thread.update_words() == F["RF_UPDATE_BYTES"]


def test_the_weights_cache_is_keyed_by_what_w_depends_on(tmp_path):
    """The cache folder: <data>/cache/apply by default, none for "none"; a set's file
    named by its scene and a key that changes with the set's files, the cores and the
    encoding's version."""
    assert apply.cache_folder(None, tmp_path) == tmp_path / "cache" / "apply"
    assert apply.cache_folder("none", tmp_path) is None
    assert apply.cache_folder(str(tmp_path / "w"), tmp_path) == tmp_path / "w"
    folder = tmp_path / "public" / "sponza_transport"
    folder.mkdir(parents=True)
    for name in apply.SET_KEYED:
        (folder / name).write_bytes(b"1234")
    cache = tmp_path / "w"
    first = apply.weights_file(folder, 120, cache)
    assert first.parent == cache and first.name.startswith("sponza_")
    assert apply.weights_file(folder, 120, cache) == first
    assert apply.weights_file(folder, 64, cache) != first
    assert apply.weights_file(folder, 120, None) is None
    (folder / "weights.f32").write_bytes(b"12345")
    assert apply.weights_file(folder, 120, cache) != first
