"""channel.bound: each transfer bound named on the host; the DRAM form's 64 B lines
and a page range past the tensor refused, a good transfer taken (no card)."""

from komorebi.device import channel
from komorebi.errors import ChannelError

L1 = 1 << 20
good = channel.Transfer(1 << 24, 2048, 0, 64, 0, 1 << 20, 2 << 20, 7, 0x20000, 8192)
assert channel.bound(good, 4, L1, 64) is None, channel.bound(good, 4, L1, 64)
cases = {
    "address 16-aligned": (good._replace(address=(1 << 24) + 16), "whole 64 B"),
    "page 1040": (good._replace(page=1040, chunk=1040 * 4), "whole 64 B"),
    "buffer 16-aligned": (good._replace(buffer=0x20010), "whole 64 B"),
    "pages past the tensor": (good._replace(first=8), "inside the tensor's 64"),
    "pages past 4 GiB": (good._replace(address=(1 << 32) - 4096), "4 GiB"),
    "chunk not whole pages": (good._replace(chunk=3000), "chunk of whole pages"),
    "words over the pages": (good._replace(words=0), "apart from pages"),
}
bad = 0
for name, (transfer, want) in cases.items():
    try:
        channel.check(transfer, 4, L1, 64)
        said = None
    except ChannelError as error:
        said = str(error)
    ok = said is not None and want in said
    bad += not ok
    print("ok  " if ok else "FAIL", name, "->", said)
print("all ok" if not bad else f"FAILED {bad}")
raise SystemExit(bad != 0)
