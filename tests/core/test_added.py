"""Code added to a live run (Run.addon) on the host: the load's reserve, the numbers an
add takes, the key of the cache of built parts, the entry command's words, the
numbers of several programs' own overlays, and each refusal. No card and no
compiler."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

import komorebi as km
from komorebi import abi, added, overlay, programs
from komorebi.errors import ProgramError

SERVICE = overlay.constants()["kOverlayService"]
BANKS = overlay.constants()["kOverlayBanks"]


def parts(folder, kernels=("reader",)):
    folder.mkdir(parents=True, exist_ok=True)
    for kernel in kernels:
        (folder / f"{kernel}.cpp").write_text("")
    return folder


def linked(size=128):
    return overlay.Linked(
        base=overlay.LINK_BASE,
        entry=overlay.LINK_BASE,
        relocations=(),
        symbols={"km_overlay_entry": (overlay.LINK_BASE, "FUNC", "GLOBAL")},
        undefined=(),
        extent=size,
        image=bytes(range(256))[:size],
    )


class Image:
    """The overlays' DRAM tensor: `rows` rows of 8 pages of `width` bytes."""

    def __init__(self, rows, width=256):
        self.pages, self.page_bytes = rows * BANKS, width
        self.address, self.aligned_page_bytes = 0x10000, width


def fake_run(room, slots=None):
    """A live run whose writes and commands are recorded."""
    writes, queued = [], []
    session = SimpleNamespace(
        log=lambda message: None,
        write=lambda image, pages, first: writes.append((pages.copy(), first)),
        tree=None,
    )
    run = SimpleNamespace(
        backend=SimpleNamespace(room=room, session=session),
        _check_alive=lambda: None,
        _queue_command=lambda handler, cores, words: queued.append(
            (handler, cores, words)
        )
        or len(queued),
        wait=lambda handle, timeout: SimpleNamespace(kind="done"),
    )
    return run, writes, queued


def a_room(
    rows=1, end=5, numbers=(1,), handlers=(1, 2), entered=True, groups=1, messages=None
):
    slots = {"reader": (2, 1024), "writer": (0, 0)}
    slots |= {part: (3, 1024) for part in ("unpack", "math", "pack")}
    recipes = dict.fromkeys(slots)
    made = [
        added.Group(16 * g, 16, recipes, dict(slots), set(numbers))
        for g in range(groups)
    ]
    return added.Room(Image(end), rows, end, made, set(handlers), entered, messages)


@pytest.fixture
def cached(monkeypatch, tmp_path):
    """build_part_cached answering from the cache with a part of 128 bytes."""
    built = []

    def build(recipe, made, kernel, part, cache, slot_bytes):
        built.append((made.name, part, slot_bytes))
        return linked(), tmp_path / "part.elf", True

    monkeypatch.setattr(overlay, "build_part_cached", build)
    monkeypatch.setattr(added.native, "cache_directory", lambda: tmp_path)
    return built


def test_the_reserve_is_zero_rows_after_the_parts():
    built = overlay.Built({(1, "reader"): bytes(256), (1, "writer"): bytes(128)}, {})
    rows, placed = overlay.dram_rows(built)
    assert rows.shape == (2 * BANKS, 64)
    rows, again = overlay.dram_rows(built, reserve=1000)  # 4 rows of 256 bytes
    assert rows.shape == (6 * BANKS, 64) and again == placed
    assert not rows[2 * BANKS :].any()


def test_a_room_is_the_rows_after_the_parts_and_none_without_a_reserve(tmp_path):
    program = km.Program(
        "p", kernels=parts(tmp_path / "k", overlay.KERNELS), handlers={"run": 1}
    )
    assert added.room(None, Image(3), 3, [], [program]) is None
    room = added.room(None, Image(7), 3, [], [program])
    assert (room.next, room.end, room.left(), room.handlers) == (3, 7, 1024, {1})
    assert not room.entered  # its program names no slot bytes
    program.slots(bytes={"writer": 2048})
    assert program.defines["KM_ADDED_HANDLERS"] == "1"
    assert added.room(None, Image(7), 3, [], [program]).entered
    program.slots(count=2)
    assert "KM_ADDED_HANDLERS" not in program.defines


def test_slots_named_by_bytes_are_reserved():
    slots = overlay.Slots(count={"reader": 4}, bytes={"writer": 2048})
    assert slots.reserved("writer") and not slots.reserved("reader")
    assert slots.of("writer", []) == (1, 2048)  # a slot though no overlay has a part
    assert overlay.Slots(bytes=1024).reserved("compute")


def test_an_add_takes_the_lowest_free_number_and_a_handler_skips_handlers():
    room = a_room(numbers=(1, 3))
    assert added._number(room, "x", False, None) == 2
    assert added._number(room, "x", True, None) == 4  # 2 is a handler of the run
    assert added._number(room, "x", False, 40) == 40


def test_each_number_refusal_has_its_message():
    room = a_room(numbers=(1,))
    with pytest.raises(added.AddError, match=r"number 0 is not in 1\.\.62"):
        added._number(room, "x", False, 0)
    with pytest.raises(added.AddError, match="not in 1"):
        added._number(room, "x", False, SERVICE)
    with pytest.raises(added.AddError, match="is an overlay of the run already"):
        added._number(room, "x", False, 1)
    with pytest.raises(added.AddError, match="is a handler of the run's programs"):
        added._number(room, "x", True, 2)
    full = a_room(numbers=range(1, SERVICE - 1), handlers=(SERVICE - 1,))
    assert added._number(full, "x", False, None) == SERVICE - 1
    with pytest.raises(added.AddError, match=r"no overlay number is left.*1 by han"):
        added._number(full, "x", True, None)


def test_the_entry_words_carry_number_mark_addresses_and_sizes():
    h = abi.headers()
    placed = {"reader": (0x1000, 1088), "pack": (0x2000, 1728)}
    words = added.entry_words(5, True, placed)
    assert words[h["KM_ENTRY_NUMBER"]] == 5 | h["KM_ENTRY_HANDLER"]
    assert words[h["KM_ENTRY_ADDRESS"]] == 0x1000
    assert words[h["KM_ENTRY_ADDRESS"] + 4] == 0x2000
    assert words[h["KM_ENTRY_BYTES"]] == 1088 // 64  # reader, writer (none)
    assert words[h["KM_ENTRY_BYTES"] + 2] == 1728 // 64  # pack alone in its word
    assert added.entry_words(5, False, {})[h["KM_ENTRY_NUMBER"]] == 5


def test_an_add_writes_its_parts_into_the_room_and_queues_an_entry_a_group(
    tmp_path, cached
):
    room = a_room(rows=1, end=5, groups=2)
    run, writes, queued = fake_run(room)
    number = added.addon(run, "mix", parts(tmp_path / "mix"))
    assert number == 2 and room.next == 2  # one row; both groups share its bytes
    ((pages, first),) = writes
    assert first == BANKS and pages.shape == (BANKS, 256)
    assert bytes(pages[0, :128]) == linked().image and (pages == pages[0]).all()
    entry = abi.headers()["KM_HANDLER_ENTRY"]
    assert [(h, cores) for h, cores, _ in queued] == [
        (entry, (0, 16)),
        (entry, (16, 16)),
    ]
    assert queued[0][2] == added.entry_words(2, False, {"reader": (0x10100, 128)})
    assert all(2 in each.numbers for each in room.groups)
    made = run.backend.room.added["mix"]
    assert (made.number, made.handler, made.message, made.parts, made.cached) == (
        2,
        False,
        False,
        {"reader": 128},
        True,
    )
    assert [part for _, part, _ in cached] == ["reader", "reader"]


def test_each_add_refusal_writes_nothing(tmp_path, cached):
    folder = parts(tmp_path / "mix")
    run, writes, queued = fake_run(None)
    with pytest.raises(added.AddError, match="reserved no room"):
        added.addon(run, "mix", folder)
    run, writes, queued = fake_run(a_room(entered=False))
    for kind in ({}, {"handler": True}, {"message": True}):
        with pytest.raises(added.AddError, match="without the path for added code"):
            added.addon(run, "mix", folder, **kind)
    assert not writes and not queued
    run, writes, queued = fake_run(a_room())
    added.addon(run, "mix", folder)
    with pytest.raises(added.AddError, match="added to this run already, as overlay 2"):
        added.addon(run, "mix", folder)
    with pytest.raises(added.AddError, match=r"gave the writer no overlay slots"):
        added.addon(run, "both", parts(tmp_path / "both", ("reader", "writer")))
    run, writes, queued = fake_run(a_room(rows=1, end=1 + 1))
    added.addon(run, "one", folder)  # the last row
    with pytest.raises(added.AddError, match=r"take 256 B .* and 0 B are left"):
        added.addon(run, "two", folder)
    assert len(writes) == 1 and len(queued) == 1


def test_a_dynamic_handler_takes_its_number_below_17_and_marks_its_entry(
    tmp_path, cached
):
    folder = parts(tmp_path / "report", ("compute",))
    room = a_room(numbers=(1, 2), handlers=(1, 2, 3), messages={1, 2})
    run, writes, queued = fake_run(room)
    number = added.addon(run, "report", folder, message=True)
    assert number == 3  # a command handler 3 that is not dynamic does not refuse it
    words = queued[0][2]
    h = abi.headers()
    assert words[h["KM_ENTRY_NUMBER"]] == 3 | h["KM_ENTRY_MESSAGE"]
    assert room.added["report"].message and [p for _, p, _ in cached][-1] == "pack"
    with pytest.raises(added.AddError, match="is a dynamic handler or an overlay"):
        added.addon(run, "again", folder, message=True, number=2)
    with pytest.raises(added.AddError, match=r"dynamic handler 17 is not in 1\.\.16"):
        added.addon(run, "again", folder, message=True, number=17)
    with pytest.raises(added.AddError, match="its body is one compute part"):
        added.addon(run, "reader", parts(tmp_path / "r"), message=True)
    with pytest.raises(added.AddError, match="not both"):
        added.addon(run, "both", folder, handler=True, message=True)
    run, *_ = fake_run(a_room())
    with pytest.raises(added.AddError, match="no dynamic handlers as parts"):
        added.addon(run, "report", folder, message=True)


def test_the_hosts_copy_of_the_area_gets_the_entry_for_the_reset_lists(
    tmp_path, cached
):
    c, h = overlay.constants(), abi.headers()
    start = 8 + h["KM_CONTROL_BYTES"] // 4  # a control offset of 8 words
    room = a_room(groups=2, messages={1})
    run, *_ = fake_run(room)
    rows = np.zeros((40, start + c["kOverlayAreaBytes"] // 4), np.uint32)
    run.backend.control_rows, run.backend.control_offset = rows, 8
    scale = added.addon(run, "scale", parts(tmp_path / "s", ("reader",)), number=40)
    report = added.addon(
        run, "report", parts(tmp_path / "r", ("compute",)), message=True
    )
    area = rows[:, start:]
    table = c["kOverlayTable"] // 4 + 2 * len(overlay.PARTS) * scale
    assert (area[:32, table] == 0x10100).all() and (area[:32, table + 1] == 128).all()
    assert (area[32:] == 0).all()  # cores of no group
    assert (area[:32, c["kOverlayCount"]] == 41).all()
    assert (area[:32, c["kOverlayMessages"] // 4] == 1 << report).all()
    assert not area[
        :32, c["kOverlayHandlers"] // 4 : c["kOverlayHandlers"] // 4 + 2
    ].any()


def index_of(tmp_path, included_text="int a;"):
    header = tmp_path / "src" / "part.h"
    header.parent.mkdir(exist_ok=True)
    header.write_text(included_text)
    preprocessed = tmp_path / "src" / "part.ii"
    preprocessed.write_text('# 1 "part.h"\n# 1 "<built-in>"\nint a;\n')
    index = tmp_path / "built" / "key-reader.json"
    overlay.keep(index, linked(), preprocessed, tmp_path / "part.elf")
    return index, header


def test_a_kept_part_comes_back_until_a_file_it_includes_changes(tmp_path):
    index, header = index_of(tmp_path)
    found, elf = overlay.built(index)
    assert found == linked() and elf == tmp_path / "part.elf"
    header.write_text("int b;")
    assert overlay.built(index) is None
    assert overlay.built(tmp_path / "built" / "none.json") is None


def recipe(tmp_path, **changes):
    elf = tmp_path / "kernel.elf"
    if not elf.exists():
        elf.write_bytes(b"\x7fELF one")
    made = overlay.Recipe("brisc", elf, ("g++",), "-Os", (), (), (), "-Os", ())
    return dataclasses.replace(made, **changes)


def test_the_cache_key_is_what_decides_the_bytes(tmp_path):
    folder = parts(tmp_path / "mix")
    made = overlay.declare("mix", folder, 2)
    key = overlay.built_key(recipe(tmp_path), made, "reader")
    assert key == overlay.built_key(recipe(tmp_path), made, "reader")
    other_number = overlay.declare("again", folder, 7)  # the name and number are not
    assert overlay.built_key(recipe(tmp_path), other_number, "reader") == key
    changed = [
        overlay.declare("mix", folder, 2, {"SCALE": 2}),
        overlay.declare("mix", parts(tmp_path / "other"), 2),
    ]
    for each in changed:
        assert overlay.built_key(recipe(tmp_path), each, "reader") != key
    assert overlay.built_key(recipe(tmp_path, opt="-O2"), made, "reader") != key
    (tmp_path / "kernel.elf").write_bytes(b"\x7fELF two")  # the loaded kernels
    assert overlay.built_key(recipe(tmp_path), made, "reader") != key


def program_with(tmp_path, name, numbers, count=None):
    made = km.Program(
        name, kernels=parts(tmp_path / name, overlay.KERNELS), handlers={"run": 1}
    )
    for number in numbers:
        made.overlay(f"o{number}", parts(tmp_path / f"{name}-{number}"), number=number)
    if count is not None:
        made.slots(count=count)
    return made


def test_several_programs_share_the_numbers_below_theirs(tmp_path):
    first = program_with(tmp_path, "a", (1, 2), count=2)
    second = program_with(tmp_path, "b", (1,))
    programs.shift_overlays([first, second])
    assert (first.overlay_shift, second.overlay_shift) == (0, 2)
    (own,) = programs.own_overlays(second)
    assert own.number == 3 and own.defines["KM_OVERLAY_SHIFT"] == "2"
    assert programs.program_overlay(second).defines["KM_OVERLAY_SHIFT"] == "2"


def test_several_programs_refuse_one_slot_and_numbers_that_run_out(tmp_path):
    one = program_with(tmp_path, "a", (1,), count=1)
    with pytest.raises(ProgramError, match="one reader slot"):
        programs.shift_overlays([one, program_with(tmp_path, "b", ())])
    high = program_with(tmp_path, "c", (SERVICE - 2,), count=2)
    with pytest.raises(ProgramError, match=r"take numbers 1 to 61 .* 2 programs 61"):
        programs.shift_overlays([high, program_with(tmp_path, "d", ())])
