"""clock_said: the line below the busy clock, none above, none on a mock."""

from komorebi.session import process

lines = []
process.mock.active = lambda: False
process.aiclk_mhz = lambda card=None: 800
print(
    "open slow:",
    process.clock_said(1, "the open", "opened after card 0 ...", lines.append),
)
print("close slow:", process.clock_said(1, "the close", None, lines.append))
process.aiclk_mhz = lambda card=None: 1350
print("busy:", process.clock_said(0, "the open", None, lines.append))


def bad(card=None):
    raise OSError("no driver")


process.aiclk_mhz = bad
print("unreadable:", process.clock_said(0, "the open", None, lines.append))
print(*lines, sep="\n")
assert len(lines) == 2
