"""The host side of an island session as one card's (no card): an island tensor's
read() and write() by its type, an island run's wait as one event of its parts."""

from types import SimpleNamespace

import numpy as np

from komorebi import formats
from komorebi.island_session import IslandEvent, IslandTensor
from komorebi.run import Event


def test_an_island_tensor_reads_and_writes_as_a_one_card_tensor():
    data = (np.arange(64 * 32, dtype=np.float32).reshape(64, 32) % 7) - 3  # bf16 exact
    raw, _ = formats.to_bytes(data, "bfloat16", "tile")
    part = SimpleNamespace(page_bytes=2048, aligned_page_bytes=2048, address=0)
    tensor = IslandTensor(
        [part, part], SimpleNamespace(pages=2), (64, 32), "bfloat16", "tile", 3
    )
    written = []
    tensor.session = SimpleNamespace(
        write=lambda held, rows: written.append((held, rows.copy())),
        read=lambda held: (raw.reshape(2, 2048),),
    )
    assert (tensor.read() == data).all()
    tensor.write(data)
    ((held, rows),) = written
    assert (
        held is tensor
        and rows.shape == (2, 2048)
        and (rows == raw.reshape(2, 2048)).all()
    )


def test_an_island_run_s_wait_is_one_event_and_its_parts():
    done = Event(4, "done", 0, 500)
    slow = Event(7, "done", 0, 900, 12)
    event = IslandEvent([done, slow])
    assert event.kind == "done" and event.microseconds == 18.0 and event.ticks == 900
    assert event.handle == 4 and event.result is None
    first, second = event  # the tuple of its parts, as before
    assert (
        (first, second) == (done, slow) and event[1] is slow and event == (done, slow)
    )
    assert [part.kind for part in event] == ["done", "done"]
    refused = IslandEvent([done, Event(9, "rejected", 1, 0)])
    assert refused.kind == "rejected" and refused.reason == 1
