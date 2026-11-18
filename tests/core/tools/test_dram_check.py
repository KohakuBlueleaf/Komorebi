"""kmr dram-check against a fake session whose reads give the written words back with
chosen words flipped: a stable flip, a flip in one read only, none; the refusal without
a host channel, --all's tensors held together, and the command's exit codes."""

import contextlib

import numpy as np
import pytest

from komorebi import cli
from komorebi.device import tensor as tensor_module
from komorebi.errors import ChannelError
from komorebi.tools import dram_check


class FakeTensor:
    def __init__(self, data):
        self.data, self.freed = data, False

    def free(self):
        self.freed = True


class FakeSession:
    """A session with a host channel (or not): its tensors keep the words, its reads
    flip `flips[i]` ((page, word, mask) each) in read i. `room` pages fit in all."""

    dram_banks = 8

    def __init__(self, flips=((), ()), channel=True, room=None):
        self.flips, self.channel = flips, (b"" if channel else None)
        self.room, self.tensors, self.reads = room, [], 0

    def _map_channel(self):
        pass

    def _mock_mapped(self):
        return False

    def tensor(self, data, dtype, layout="row"):
        assert (dtype, layout, data.shape[1]) == ("uint32", "row", 1024)
        held = sum(len(t.data) for t in self.tensors if not t.freed)
        if self.room is not None and held + len(data) > self.room:
            raise tensor_module.DramFull("full", "", self.room - held)
        self.tensors.append(FakeTensor(data.copy()))
        return self.tensors[-1]

    def read(self, tensor):
        words = tensor.data.copy()
        for page, word, mask in self.flips[self.reads % 2]:
            words[page, word] ^= mask
        self.reads += 1
        return [words.view(np.uint8)]


def test_no_flip_reads_clean_and_frees_the_tensor():
    session = FakeSession()
    result = dram_check.check_dram(session, megabytes=1)
    assert result["megabytes"] == 1 and result["words"] == 256 * 1024
    assert (result["wrong_read1"], result["wrong_read2"], result["reads_differ"]) == (
        0,
        0,
        0,
    )
    assert result["banks"] == {} and result["bits"] == {}
    assert session.reads == 2 and session.tensors[0].freed


def test_a_stable_flip_is_wrong_in_both_reads_and_attributed_to_its_bank():
    flips = [(11, 5, 1 << 3), (19, 0, (1 << 31) | 1)]  # pages 11, 19: bank 3
    result = dram_check.check_dram(FakeSession((flips, flips)), megabytes=1)
    assert (result["wrong_read1"], result["wrong_read2"], result["reads_differ"]) == (
        2,
        2,
        0,
    )
    assert result["banks"] == {3: 2} and result["bank_count"] == 8
    assert result["bits"] == {0: 2, 3: 2, 31: 2}


def test_a_flip_in_one_read_only_differs_between_the_reads():
    result = dram_check.check_dram(FakeSession(([], [(6, 9, 1 << 17)])), megabytes=1)
    assert (result["wrong_read1"], result["wrong_read2"], result["reads_differ"]) == (
        0,
        1,
        1,
    )
    assert result["banks"] == {6: 1} and result["bits"] == {17: 1}


def test_the_bank_count_is_the_sessions():
    session = FakeSession(([(13, 0, 1)], [(13, 0, 1)]))
    session.dram_banks = 12
    assert dram_check.check_dram(session, megabytes=1)["banks"] == {1: 1}


def test_no_host_channel_is_refused_before_anything_is_written():
    session = FakeSession(channel=False)
    with pytest.raises(ChannelError, match="no host channel"):
        dram_check.check_dram(session)
    assert session.tensors == []


def test_all_holds_its_tensors_until_the_last_then_frees_them():
    session = FakeSession(room=600)  # pages: 256 + 256, then the 88 that still fit
    seen = []
    results = dram_check.check_all(session, megabytes=1, said=seen.append)
    assert [r["words"] // 1024 for r in results] == [256, 256, 88]
    assert seen == results and all(t.freed for t in session.tensors)
    assert len(session.tensors) == 3


@pytest.mark.parametrize(
    ("flips", "code", "said"),
    [
        (((), ()), 0, "card 0: 1 MiB in 1 tensor(s), 0 wrong words"),
        (([(3, 1, 4)], [(3, 1, 4)]), 1, "reset the card again (tt-smi -r 0)"),
    ],
)
def test_the_command_exits_1_on_any_wrong_word(monkeypatch, capsys, flips, code, said):
    session = FakeSession(flips)
    opened = lambda card: contextlib.nullcontext(session)  # noqa: E731
    monkeypatch.setattr(dram_check.km, "Session", opened, raising=False)
    assert cli.main(["dram-check", "--card", "0", "--megabytes", "1"]) == code
    out = capsys.readouterr().out
    assert said in out
    if code:
        assert "  bank 3 of 8: 1 wrong words" in out and "bit 2: 2" in out
