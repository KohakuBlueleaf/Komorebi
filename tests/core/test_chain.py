"""The chain on the host: a call's description as records, the earlier input's proof,
the direct input's checks (a core's share from one root or several), and each refusal
with the call, tensor and core it names."""

import numpy as np
import pytest
from fake_port import FakeBackend, FakePort

from komorebi import abi
from komorebi import chain as kc
from komorebi.run import Run

W = kc.words()
ADDRESS = 0x10_0000


class Block:
    def __init__(self, data):
        self.address, self.data, self.freed = ADDRESS, bytes(data), False

    def free(self):
        self.freed = True


class BlockBackend(FakeBackend):
    def __init__(self, port):
        super().__init__(port)
        self.blocks = []

    def block(self, data, place=None):
        self.blocks.append(Block(data if place is None else place(ADDRESS)))
        return self.blocks[-1]

    def fence(self):
        pass


def steps(add_range=None, earlier=True):
    """Scale x into y, scale y into z, add y to z into s (y earlier), sum s to a root:
    eight cores, one tile each (block g of y, z, s written by core g)."""
    one = kc.Range(0, 1, 1)
    add = kc.Read("y", ranges=(add_range or one,), earlier=earlier)
    return [
        kc.Call(1, (0, 2), (0, 8), 1, 1, "y", name="scale"),
        kc.Call(1, (1, 3), (0, 8), 1, 1, "z", "y", ranges=(one,), name="scale"),
        kc.Call(1, (2,), (0, 8), 1, 1, "s", "z", ranges=(one,), reads=(add,)),
        kc.Call(1, (3,), (0, 8), 8, 1, "t", "s", k=(0, 8), sends=True, word=("n", 9)),
    ]


def test_the_records_of_a_chain():
    records = kc.records(steps(), ADDRESS)
    assert records.shape == (4, W["kChainWords"]) and records.dtype == np.uint32
    assert list(records[1, :2]) == [1, 3] and records[1, W["kChainGroup"]] == 8 << 16
    info = [int(word) for word in records[:, W["kChainInfo"]]]
    assert info[0] == 1 << W["kChainSizeShift"] | 1 << W["kChainWidthShift"]
    assert info[1] & W["kChainMarked"] and (info[1] >> W["kChainNeedsShift"]) & 1
    last = info[3]
    assert last & W["kChainSends"] and (last >> W["kChainTokenShift"]) & 0xF == 15
    assert (last >> W["kChainSizeShift"]) & W["kChainSizeMask"] == 8
    assert records[1, W["kChainRanges"]] == 0 | 1 << 12 | 1 << 20
    assert records[3, W["kChainRanges"]] == 8 << 20
    nexts = [int(word) for word in records[:, W["kChainNext"]]]
    assert nexts == [ADDRESS + 64, ADDRESS + 128, ADDRESS + 192, 0]


def test_an_earlier_input_is_proven():
    assert kc.prove(steps())


def test_an_earlier_input_out_of_reach_is_refused():
    shifted = kc.Range(1, 1, 1)  # core g reads y's tile g + 1: not what z's marks say
    with pytest.raises(
        kc.ChainRefused, match="core 0 reads y tile 1, written by call 0"
    ):
        kc.prove(steps(shifted))


def test_an_input_the_call_before_does_not_write_is_refused():
    calls = steps()
    calls[2] = calls[2]._replace(input="y")
    with pytest.raises(kc.ChainRefused, match=r"call 2: reads y as the call before"):
        kc.prove(calls)


def test_an_outside_tensor_a_call_writes_is_refused():
    with pytest.raises(
        kc.ChainRefused, match="reads y as an outside tensor, but call 0"
    ):
        kc.prove(steps(earlier=False))


def test_a_direct_input_is_checked_and_reported():
    calls = steps()
    calls[3] = calls[3]._replace(direct=kc.Direct(1, 4096))
    assert kc.check_direct(calls) == {3: [(4096, 1)] * 8}
    calls[3] = calls[3]._replace(direct=kc.Direct(1, 4096), size=4, k=(0, 8))
    with pytest.raises(kc.ChainRefused, match="core 0 reads 2 tiles of s; its arrival"):
        kc.check_direct(calls)
    calls[3] = calls[3]._replace(direct=kc.Direct(2, 4096))  # two roots a core
    assert kc.check_direct(calls) == {3: [(8192, 2)] * 8}
    calls[3] = calls[3]._replace(k=None, ranges=(kc.Range(0),))
    with pytest.raises(kc.ChainRefused, match="read by a share of K"):
        kc.check_direct(calls)


def test_run_chain_writes_once_and_submit_takes_its_word():
    port = FakePort()
    backend = BlockBackend(port)
    run = Run(backend, (12, 10))
    chain = run.chain(steps(), mode=3)
    assert len(backend.blocks) == 1 and chain.group == (0, 8)
    words = np.frombuffer(backend.blocks[0].data, np.uint32).reshape(4, 16)
    assert words[0, W["kChainNext"]] == ADDRESS + 64
    handle = run.submit(chain, n=7)
    run.wait(handle)
    at = run._arg_at
    sent = port.ring[handle]
    bank = 1 << abi.headers()["KM_BLOCK_BANK_SHIFT"]
    assert [sent[at[0]], sent[at[4]], sent[at[7]], sent[at[8]]] == [
        7,
        3,
        ADDRESS,
        64 | bank,
    ]
    with pytest.raises(ValueError, match="per-submit word is n"):
        run.submit(chain, m=7)


def test_a_record_names_its_handler_and_its_stamp_row():
    calls = steps()
    calls[1] = calls[1]._replace(handler=2, row=5)
    records = kc.records(calls, ADDRESS, [1, 2, 1, 1])
    handlers = [int(word) >> W["kChainHandlerShift"] for word in records[:, 11]]
    assert handlers == [1, 2, 1, 1]
    assert records[1, W["kChainNext"]] == (ADDRESS + 128) | 5
    assert kc.records(calls, ADDRESS)[1, W["kChainInfo"]] >> 24 == 0  # the chain's
    with pytest.raises(kc.ChainRefused, match="stamp row 0 to 63"):
        kc.records([calls[0]._replace(row=64)], ADDRESS)
