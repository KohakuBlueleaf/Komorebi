"""session.channel_transfer: the command words of the session's own transfers, a slot
of the region each, the armed wait, the refusals (no card)."""

from types import SimpleNamespace

import numpy as np
import pytest

from komorebi import ChannelError, KomorebiError, Session, abi
from komorebi.device import channel as ch
from komorebi.islands.same import SessionCalls
from komorebi.session.channel_transfer import PUSH_SALT

H = abi.headers()
REGION = ch.check_region(4 << 20)
NONCE = H["KM_TRANSFER_NONCE"]


class Run:
    """A live run whose transfer() writes each rank's sequence word, as the cores do."""

    def __init__(self, session):
        self.session, self.cores, self.sent = session, 120, []
        self.backend = SimpleNamespace(
            transfer=SimpleNamespace(chunk=16 << 10, address=0x50000)
        )

    def holds(self):
        return None

    def transfer(self, handler, group, args, timeout):
        self.sent.append((handler, group, tuple(args)))
        t = ch.Transfer(
            *[args[H[f"KM_TRANSFER_{n.upper()}"]] for n in ch.Transfer._fields]
        )
        want = ch.sequence_words(t.nonce, t.pages, t.page, group[1])
        ch.words_view(self.session.channel, REGION, group[1])[:, :4] = want


def session_with_run():
    session = Session(channel=REGION.bytes)
    session.channel = np.zeros(REGION.bytes, np.uint8)
    session._dram_banks = 8  # as the open reads them from a Blackhole card
    session.run = Run(session)
    return session


def tensor(pages, address=0x1000, page=2048):
    return SimpleNamespace(
        in_dram=True, address=address, aligned_page_bytes=page, page_bytes=page,
        pages=pages,
    )  # fmt: skip


@pytest.mark.parametrize("push", [False, True])
def test_the_words_are_the_sessions_own_transfer_but_the_nonce(push):
    session, x = session_with_run(), tensor(72)
    session._transfer(x, 0, x.pages, "run", push)
    handler, group, args = session.run.sent[0]
    made = session.channel_transfer(x, push)
    assert made.entry.handler == handler and made.entry.group == group[:2]
    assert made.entry.field == group[2]
    assert made.entry.args[:NONCE] + made.entry.args[NONCE + 1 :] == (
        args[:NONCE] + args[NONCE + 1 :]
    )
    assert made.entry.args[NONCE] == (PUSH_SALT if push else 0)


def nn_words(session, inputs, outputs):
    """The entries projects/nn built by hand before this call: its tensors in order,
    pages and words packed from the region's start."""
    run, region, out = session.run, session._region, []
    stride = H["KM_TRANSFER_WORD_BYTES"]
    offset, words = 0, region.words
    for push, tensors in ((False, inputs), (True, outputs)):
        for t in tensors:
            page = t.aligned_page_bytes
            chunk = ch.chunk_bytes(page, run.backend.transfer.chunk)
            wanted = ch.ranks(t.pages, page, chunk, session.transfer_cores)
            first, count, field = ch.transfer_group(run.holds(), run.cores, wanted)
            stored = 0x5A5A0001 if push else 0
            transfer = ch.Transfer(
                t.address, page, 0, t.pages, offset, words, region.bytes, stored,
                run.backend.transfer.address, chunk,
            )  # fmt: skip
            handler = H["KM_HANDLER_PUSH" if push else "KM_HANDLER_PULL"]
            out.append((handler, (first, count), tuple(transfer.args()), field))
            offset += -(-t.pages * page // 16) * 16
            words += count * stride
    return out


def test_a_list_of_transfers_has_the_words_nn_built_by_hand():
    session = session_with_run()
    inputs, outputs = [tensor(72), tensor(3, 0x9000)], [tensor(64, 0x20000)]
    made = [session.channel_transfer(t, False) for t in inputs]
    made += [session.channel_transfer(t, True) for t in outputs]
    assert [tuple(m.entry) for m in made] == nn_words(session, inputs, outputs)


def test_the_data_is_the_region_and_the_wait_is_armed_each_issue():
    session = session_with_run()
    first = session.channel_transfer(tensor(4), False)
    made = session.channel_transfer(tensor(5), True)
    made.data[:] = 7
    assert session.channel[4 * 2048 : 9 * 2048].min() == 7  # after the first's pages
    assert first.data.base is not None and first.data.shape == (4, 2048)
    assert made.arm() == 1 and made.arm() == 2 and made.arm(9) == 9
    assert not made.wait(0.01, spin=0.0)
    nonce = PUSH_SALT ^ 9
    count = made.entry.group[1]
    made.words[:, :4] = ch.sequence_words(nonce, 5, 2048, count)
    assert made.wait(0.01)
    made.arm()
    assert not made.words.any()


def test_a_slot_is_given_back_by_close_or_a_new_run():
    session = session_with_run()
    big = tensor(REGION.data // 2048 - 1)
    made = session.channel_transfer(big, False)
    with pytest.raises(ChannelError, match="no room"):
        session.channel_transfer(tensor(2), False)
    made.close()
    session.channel_transfer(tensor(2), False).close()
    session.channel_transfer(big, False)
    session.run = Run(session)  # the next load's run
    assert session.channel_transfer(big, True).entry.args[H["KM_TRANSFER_OFFSET"]] == 0


def test_refused_without_a_channel_a_run_its_buffer_or_a_dram_tensor():
    session = session_with_run()
    l1 = tensor(1)
    l1.in_dram = False
    with pytest.raises(ChannelError, match="in DRAM"):
        session.channel_transfer(l1, False)
    session.run.backend.transfer = None
    with pytest.raises(ChannelError, match="transfer buffer"):
        session.channel_transfer(tensor(1), False)
    session.run = None
    with pytest.raises(ChannelError, match="transfer buffer"):
        session.channel_transfer(tensor(1), False)
    session.channel = None
    with pytest.raises(ChannelError, match="no channel"):
        session.channel_transfer(tensor(1), False)


def test_islands_refuse_it_by_name():
    with pytest.raises(KomorebiError, match=r"session.island\(i\).channel_transfer"):
        SessionCalls.channel_transfer(None, tensor(1), False)
