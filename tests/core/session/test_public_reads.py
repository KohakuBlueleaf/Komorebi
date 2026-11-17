"""Public reads that scripts once took from a run's or a session's inside: overlay
epochs, the dispatcher's give-ups, held blocks, region writes, DRAM banks, bindings."""

import types

import numpy as np
import pytest
from fake_port import FakeBackend, FakePort
from fake_session import CONSTANTS, FakeDevice, load_session

from komorebi.bench import power
from komorebi.device import native
from komorebi.engines.graph import Graph
from komorebi.program import overlay
from komorebi.program import thread as thread_module
from komorebi.session import tensor_bytes
from komorebi.session.run import Run
from komorebi.session.session import Session
from komorebi.session.x280 import _Regions, _ThreadRegions

GRID = (12, 10)
MASK = overlay.constants()["kOverlayEpochMask"]


def test_the_read_back_keeps_each_cores_overlay_epoch(monkeypatch, tmp_path):
    session, log, _ = load_session(monkeypatch, tmp_path)
    sess = session.Session(device=FakeDevice(log), grid_timeout=0.5)
    sess._module = native.load_module()
    port = sess._module.Dispatch("dev", 1750)
    port.close()
    backend = sess._module.card_backend(sess._card(), port, 1350, 0)
    backend.control, backend.control_addr = "control", 0x20000
    backend._wait_grid = lambda timeout: True
    epoch = (CONSTANTS["KM_CONTROL_BYTES"] + overlay.constants()["kOverlayEpoch"]) // 4
    rows = np.zeros((120, epoch + 8), np.int64)
    rows[:, epoch] = np.arange(120) * 3
    monkeypatch.setattr(tensor_bytes, "_read_uint32", lambda tensor, *args: rows)
    backend.read_back()
    assert backend.overlay_epochs is None  # no overlay area
    backend.area_bytes = 4096
    blocks, _ = backend.read_back()
    assert blocks.shape == (120, CONSTANTS["KM_CONTROL_BYTES"] // 4)
    assert list(backend.overlay_epochs[:4]) == [0, 3, 6, 9]


def test_state_gives_the_overlay_epoch_masked_or_none():
    port = FakePort()
    run = Run(FakeBackend(port), GRID)
    run.stop()
    assert {core["overlay_epoch"] for core in run.state()} == {None}
    backend = FakeBackend(FakePort())
    backend.overlay_epochs = np.arange(120, dtype=np.uint32) | 0x7000000
    run = Run(backend, GRID)
    run.stop()
    epochs = [core["overlay_epoch"] for core in run.state()]
    assert MASK == 0xFFFFFF and epochs[:3] == [0, 1, 2]  # the high byte is not counted


def test_the_dispatchers_give_ups_are_read_without_a_command():
    port = FakePort()
    run = Run(FakeBackend(port), GRID)
    assert run.dispatcher_given_up() == 0
    port.given_up = 3
    before = len(port.submitted)
    assert run.dispatcher_given_up() == 3 and len(port.submitted) == before


def test_blocks_held_counts_blocks_until_their_events():
    class Block:
        address, freed = 0x4000_0000, False

        def free(self):
            self.freed = True

    backend = FakeBackend(FakePort())
    backend.block = lambda data: Block()
    backend.fence = lambda: None
    run = Run(backend, GRID)
    handles = [run.submit(5, (0, 2), block=b"\x01" * 64) for _ in range(2)]
    assert run.blocks_held == 2
    run.wait(handles[0])
    run.wait(handles[1])
    assert run.blocks_held == 0


def regions():
    """A _Regions of one 100-byte region at page 2 of bank 3, over a module that logs
    its DRAM writes and barriers."""
    calls = []
    module = types.SimpleNamespace(
        write_dram=lambda device, banks, address, data: calls.append(
            ("write", banks, address, bytes(data))
        ),
        dram_barrier=lambda device: calls.append(("barrier",)),
    )
    made = _Regions.__new__(_Regions)
    made.session = types.SimpleNamespace(_module=module, device="device")
    made.bank, made.base = 3, 0x10000
    made.placed = {(1, "list"): thread_module.Placement("list", 2, 1, 100)}
    return made, calls


def test_a_region_write_lands_at_its_page_and_offset_with_its_barrier():
    made, calls = regions()
    made.write((1, "list"), np.array([1, 2], np.uint32), offset=8)
    address = 0x10000 + 2 * thread_module.PAGE + 8
    assert calls == [("write", [3], address, b"\x01\0\0\0\x02\0\0\0"), ("barrier",)]
    calls.clear()
    made.write((1, "list"), b"\x05" * 4, offset=96, barrier=False)
    assert calls == [("write", [3], address + 88, b"\x05" * 4)]


@pytest.mark.parametrize("offset, size", [(-1, 4), (97, 4), (0, 101)])
def test_a_region_write_outside_the_region_is_refused(offset, size):
    made, calls = regions()
    with pytest.raises(ValueError, match="region list: bytes"):
        made.write((1, "list"), np.zeros(size, np.uint8), offset=offset)
    assert calls == []


def test_a_threads_write_names_its_region_and_a_fallback_has_none():
    made, calls = regions()
    view = _ThreadRegions.__new__(_ThreadRegions)
    view.all, view.keys = made, {"list": (1, "list")}
    run = thread_module.ThreadRun(None, None, view)
    run.write("list", b"\x07" * 4, offset=4)
    assert calls[0][2] == 0x10000 + 2 * thread_module.PAGE + 4 and len(calls) == 2
    fallback = thread_module.ThreadRun(None, None, None, on_card=False)
    with pytest.raises(RuntimeError, match="fallback has arrays"):
        fallback.write("list", b"\x07")


def test_dram_banks_asks_the_open_device():
    session = Session.__new__(Session)
    session._module = types.SimpleNamespace(dram_banks=lambda device: 8)
    session.device = "device"
    assert session.dram_banks == 8


def test_binding_names_follow_the_registers():
    graph = Graph()
    graph.bindings("it", "step")
    graph.bindings("again")
    assert graph.binding_names == ("it", "step", "again")


def test_visible_card_is_the_first_visible_device(monkeypatch):
    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)
    assert power.visible_card() == 0
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "2,3")
    assert power.visible_card() == 2
