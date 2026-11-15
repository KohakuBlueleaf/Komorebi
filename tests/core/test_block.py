"""The command block: the layout follows the slot count, and a block's words."""

import re

import numpy as np
import pytest
from fake_port import FakeBackend, FakePort

from komorebi import abi
from komorebi import thread as thread_module
from komorebi.run import Run

CONSTANTS = abi.headers().constants
GRID = (12, 10)
ADDRESS = 0x4000_0000


class Block:
    def __init__(self, data):
        self.address, self.data, self.freed = ADDRESS, bytes(data), False

    def free(self):
        self.freed = True


class BlockBackend(FakeBackend):
    """A fake backend that keeps every block it was given."""

    def __init__(self, port):
        super().__init__(port)
        self.blocks, self.fences = [], 0

    def block(self, data):
        self.blocks.append(Block(data))
        return self.blocks[-1]

    def fence(self):
        self.fences += 1


def make_run():
    port = FakePort()
    backend = BlockBackend(port)
    return Run(backend, GRID), port, backend


def test_the_control_block_follows_the_slot_count():
    slots = CONSTANTS["KM_RING_SLOTS"]
    assert slots == 64
    assert CONSTANTS["KM_CONTROL_BOOT"] == CONSTANTS["KM_CONTROL_DONE"] + 16 * slots
    assert CONSTANTS["KM_STATE_WORDS"] == 2 * slots + 16
    assert (
        CONSTANTS["KM_CONTROL_JOB"]
        == CONSTANTS["KM_CONTROL_STATE"] + 4 * CONSTANTS["KM_STATE_WORDS"]
    )
    area = CONSTANTS["KM_BLOCK_AREA_BYTES"]
    assert (
        area % 64 == 0
        and CONSTANTS["KM_BLOCK_BYTES"] <= CONSTANTS["KM_BLOCK_BYTES_MASK"]
    )
    assert CONSTANTS["KM_CONTROL_BLOCK"] + 2 * area == CONSTANTS["KM_CONTROL_ABORT"]
    assert CONSTANTS["KM_CONTROL_LIVE"] == CONSTANTS["KM_CONTROL_ABORT"] + 4 * slots
    islands = CONSTANTS["KM_CONTROL_LIVE"] + CONSTANTS["KM_LIVE_BYTES"]
    assert islands == CONSTANTS["KM_CONTROL_ISLANDS"]
    ring = islands + CONSTANTS["KM_ISLANDS_BYTES"]
    assert ring == CONSTANTS["KM_CONTROL_RING"]
    assert ring + 64 * slots == CONSTANTS["KM_CONTROL_BYTES"]
    assert CONSTANTS["KM_CONTROL_BLOCK"] % 64 == 0


def test_a_block_goes_in_a7_and_a8_and_is_freed_after_its_event():
    run, port, backend = make_run()
    weights = np.arange(64, dtype=np.uint32) * 3
    handle = run.submit(5, (0, 2), [11, 12], block=weights)
    assert run._kept[handle] is backend.blocks[0]
    assert backend.blocks[0].data == weights.tobytes()
    second = run.submit(5, (2, 2), block=weights)
    assert backend.fences == 0
    event = run.wait(handle)
    assert backend.fences == 1  # both handed to the port
    words = port.ring[handle]
    at = run._arg_at
    assert [words[at[0]], words[at[1]]] == [11, 12]
    assert [words[at[7]], words[at[8]]] == [ADDRESS, 256]
    assert event.kind == "done" and backend.blocks[0].freed
    run.wait(second)
    assert backend.fences == 1 and backend.blocks[1].freed and not run._kept


def test_a_block_of_bytes_and_one_without_arguments():
    run, port, backend = make_run()
    handle = run.submit(5, (0, 1), block=b"\x01\x02\x03")
    run.wait(handle)
    at = run._arg_at
    assert [port.ring[handle][at[i]] for i in range(9)] == [0] * 7 + [ADDRESS, 3]
    assert backend.blocks[0].data == b"\x01\x02\x03"


@pytest.mark.parametrize(
    "args, block, cluster, message",
    [
        (list(range(8)), b"x", None, "8 arguments"),
        ([], b"", None, "0 bytes"),
        ([], bytes(CONSTANTS["KM_BLOCK_BYTES"] + 1), None, "1025 bytes"),
        ([], b"x", 1, "not to an X280 thread"),
    ],
)
def test_a_block_is_refused(args, block, cluster, message):
    run, _, backend = make_run()
    with pytest.raises(ValueError, match=message):
        if cluster is None:
            run.submit(5, (0, 1), args, block=block)
        else:
            run._block_words(block, args, cluster)
    assert backend.blocks == []


def test_a_thread_block_names_the_bank_of_its_cluster():
    text = (thread_module.native.kmx_directory() / "kmx_layout.h").read_text()
    one, other = re.search(
        r"KMX_BANK_OF_CLUSTER\(c\) \(\(c\) == 1u \? (\d+)u : (\d+)u\)", text
    ).groups()
    banks = thread_module.BANK_OF_CLUSTER
    assert (int(one), int(other), int(other)) == (banks[1], banks[2], banks[3])
