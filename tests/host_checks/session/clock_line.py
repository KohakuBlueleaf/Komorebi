"""clock_said: the line below the architecture's busy clock, none above, none on a
mock; 1000 MHz is a Wormhole chip's busy clock and a slow Blackhole card's."""

from komorebi.session import process

lines = []
process.mock.active = lambda: False
process.aiclk_mhz = lambda card=None, driver=None: 800
print(
    "open slow:",
    process.clock_said(
        1, "the open", "blackhole", "opened after card 0 ...", lines.append
    ),
)
print(
    "close slow:", process.clock_said(1, "the close", "blackhole", None, lines.append)
)
process.aiclk_mhz = lambda card=None, driver=None: 1350
print("busy:", process.clock_said(0, "the open", "blackhole", None, lines.append))
process.aiclk_mhz = lambda card=None, driver=None: 1000
print(
    "wormhole busy:",
    process.clock_said(0, "the open", "wormhole_b0", None, lines.append),
)


def bad(card=None, driver=None):
    raise OSError("no driver")


process.aiclk_mhz = bad
print("unreadable:", process.clock_said(0, "the open", "blackhole", None, lines.append))
process.aiclk_mhz = lambda card=None, driver=None: 800  # a slow card, a script's mock
process.os.environ[process.MOCK] = "p150.yaml"
print("own mock:", process.clock_said(0, "the close", "blackhole", None, lines.append))
del process.os.environ[process.MOCK]
print(*lines, sep="\n")
assert len(lines) == 2 and all("below 1350 MHz" in line for line in lines)
