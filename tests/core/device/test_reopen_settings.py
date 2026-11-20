"""A session reopened for a larger kernel buffer keeps every setting it was made
with but the kernel buffer's size: the card, the clock, the time-outs, the log, the
channel, the slow code; an island session's cards opened again as at the first open
(no card: the close and the open stood in for)."""

from komorebi.device import kernel_buffer
from komorebi.islands.parts import load as island_load
from komorebi.session import session

SETTINGS = (
    "card",
    "pll_mhz",
    "_default_clock",
    "tree",
    "log",
    "log_given",
    "grid_timeout",
    "slow_code",
    "_region",
)


def test_a_larger_reopen_keeps_the_constructor_settings(monkeypatch):
    lines = []
    made = session.Session(
        card=1,
        pll_mhz=800,
        log=lines.append,
        grid_timeout=3.5,
        channel=None,
        slow_code=True,
    )
    before = {name: getattr(made, name) for name in SETTINGS}
    opened = []
    monkeypatch.setattr(made, "_close_device", lambda: None)
    monkeypatch.setattr(made, "open", lambda: opened.append(made._worker_l1))
    monkeypatch.setattr(made, "l1", lambda: (70656, 0))
    kernel_buffer.reopen(made, 93248, 123456, "the kernels take 92,224 B")
    assert opened == [123456]
    assert {name: getattr(made, name) for name in SETTINGS} == before
    assert made.card == 1 and made.pll_mhz == 800 and made.grid_timeout == 3.5
    assert made.slow_code is True and made.log == lines.append
    assert any(line.startswith("kernel buffer 70,656 -> 93,248 B") for line in lines)


def test_an_island_reopen_opens_the_same_cards_as_the_first(monkeypatch):
    seen = []

    class Mesh:
        def close(self):
            seen.append(("close",))

        def device(self, column):
            seen.append(("device", column))

    def fabric(module, visible):
        seen.append(("fabric", visible))
        opened.mesh = Mesh()

    monkeypatch.setattr(island_load.mock, "active", lambda: True)
    monkeypatch.setattr(island_load.native, "load_module", lambda tree: object())
    opened = type("Opened", (), {})()
    opened.log, opened.kernel_bytes, opened.cards = (lambda line: None), 70656, [0, 1]
    opened.tree, opened.packet, opened.sessions, opened.links = None, 4096, (), ()
    opened.mesh, opened._open_fabric = Mesh(), fabric
    opened._visible, opened._columns = [3, 0, 1], (1, 2)
    opened.arch = "blackhole"
    resize = type("Resize", (), {"size": 93248, "why": "the kernels take 92,224 B"})
    island_load.LoadMixin._cards_reopen(opened, resize)
    assert seen == [("close",), ("fabric", [3, 0, 1])]  # the first open's chips
    assert opened.cards == [0, 1] and opened.packet == 4096
