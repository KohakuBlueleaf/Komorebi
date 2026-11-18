"""A transfer of a whole DRAM tensor through the host channel as an entry of a
program's own command list (Session.channel_transfer): its slot of the region, its
command words, its data and its wait."""

from typing import NamedTuple

from komorebi import abi
from komorebi.device import channel as channel_module
from komorebi.errors import ChannelError
from komorebi.session import sizes

PUSH_SALT = 0x5A5A0001  # a push's nonce word: its sequence words differ from a pull's


class Entry(NamedTuple):
    """A PULL's or PUSH's command words, as a list carries them: the handler, the
    group (first, count), the ten argument words and the program field."""

    handler: int
    group: tuple
    args: tuple
    field: int


def _first_fit(taken, size, start, end):
    """The lowest offset from `start` where `size` bytes fit before `end` among the
    (offset, size) spans `taken`, or None."""
    at = start
    for offset, length in sorted(taken):
        if offset - at >= size:
            break
        at = max(at, offset + length)
    return at if at + size <= end else None


class ChannelTransfer:
    """A whole DRAM tensor's PULL (host to card) or PUSH (card to host) through the
    host channel, for a list that issues it: `entry` its command words, `data` its
    pages in the region, arm() before each issue, wait() for its sequence words."""

    def __init__(self, session, tensor, push):
        run, region = session.run, session._region
        if session.channel is None or region is None:
            raise ChannelError("session.channel_transfer: the session has no channel")
        buffer = getattr(getattr(run, "backend", None), "transfer", None)
        if run is None or buffer is None:
            raise ChannelError(
                "session.channel_transfer: no loaded run with a transfer buffer"
            )
        if not tensor.in_dram:
            raise ChannelError("session.channel_transfer: a tensor in DRAM, not in L1")
        h = abi.headers()
        page, stride = tensor.aligned_page_bytes, h["KM_TRANSFER_WORD_BYTES"]
        chunk = channel_module.chunk_bytes(page, buffer.chunk)
        wanted = channel_module.ranks(tensor.pages, page, chunk, session.transfer_cores)
        first, count, field = channel_module.transfer_group(
            run.holds(), run.cores, wanted
        )
        if session.island is not None:  # the run takes flat ranks
            graph, here = session.island
            first += graph.first(here)
        live = session._channel_transfers = [
            made for made in session._channel_transfers if made.run is run
        ]
        size = -(-tensor.pages * page // 16) * 16
        offset = _first_fit([m.span for m in live], size, 0, region.data)
        words = _first_fit(
            [m.word_span for m in live], count * stride, region.words, region.bytes
        )
        if offset is None or words is None:
            raise ChannelError(
                f"session.channel_transfer: the host channel's region has no room for "
                f"{size} B of pages and {count} sequence words beside "
                f"{len(live)} channel transfers (its data {region.data} B)"
            )
        self.salt = PUSH_SALT if push else 0
        transfer = channel_module.Transfer(
            tensor.address, page, 0, tensor.pages, offset, words, region.bytes,
            self.salt, buffer.address, chunk,
        )  # fmt: skip
        channel_module.check(
            transfer, count, sizes.L1_BYTES, session.dram_banks, tensor.pages
        )
        handler = h["KM_HANDLER_PUSH" if push else "KM_HANDLER_PULL"]
        self.entry = Entry(handler, (first, count), tuple(transfer.args()), field)
        self.run, self.session, self.tensor = run, session, tensor
        self.span, self.word_span = (offset, size), (words, count * stride)
        rows = session.channel[offset : offset + tensor.pages * page]
        self.data = rows.reshape(tensor.pages, page)[:, : tensor.page_bytes]
        raw = session.channel[words : words + count * stride]
        self.words = raw.view("<u4").reshape(count, stride // 4)
        self.key = 0
        self.want = channel_module.sequence_words(self.salt, tensor.pages, page, count)
        live.append(self)

    def arm(self, key=None):
        """Before an issue: clears the sequence words and steps the key (or takes
        `key`), which the list's runner XORs into the nonce word; returns the key."""
        self.key = ((self.key + 1) & 0xFFFFFFFF or 1) if key is None else key
        nonce = self.salt ^ self.key
        self.want[:, 0] = nonce
        self.want[:, 3] = abi.headers()["KM_TRANSFER_MARK"] ^ nonce
        self.words[:] = 0
        return self.key

    def wait(self, timeout, spin=None):
        """Whether every rank's sequence word of the armed issue arrived within
        `timeout` s, looked at as a session transfer's are (`spin` s without a pause,
        None: the session's)."""
        return self.session._words_arrived(self.words, self.want, timeout, spin)

    def close(self):
        """Gives its slot of the region back (a new run's load gives them all)."""
        live = self.session._channel_transfers
        if self in live:
            live.remove(self)
