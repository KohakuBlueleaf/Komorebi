"""What test_added.py and other tests share: BANKS, BACKEND, Image, a_room, fake_run,
parts."""

from types import SimpleNamespace

from komorebi.program import overlay
from komorebi.program.addon import code as added

BANKS = overlay.constants()["kOverlayBanks"]
BACKEND = SimpleNamespace(card=SimpleNamespace(dram_banks=BANKS))  # a Blackhole card


def parts(folder, kernels=("reader",)):
    folder.mkdir(parents=True, exist_ok=True)
    for kernel in kernels:
        (folder / f"{kernel}.cpp").write_text("")
    return folder


class Image:
    """The overlays' DRAM tensor: `rows` rows of 8 pages of `width` bytes."""

    def __init__(self, rows, width=256):
        self.pages, self.page_bytes = rows * BANKS, width
        self.address, self.aligned_page_bytes = 0x10000, width


def fake_run(room, slots=None):
    """A live run whose writes and commands are recorded."""
    writes, queued = [], []
    card = SimpleNamespace(
        log=lambda message: None,
        write=lambda image, pages, first: writes.append((pages.copy(), first)),
        tree=None,
        dram_banks=BANKS,  # a Blackhole card's
    )
    run = SimpleNamespace(
        backend=SimpleNamespace(room=room, card=card),
        programs=[],  # none built with guard=False
        _check_alive=lambda: None,
        island_first=0,  # one island: its ranks are its cores
        island_ranks=120,
        cores=120,
        _workers=120,  # the X280 dispatcher's run: every core
        arch=None,  # a chip that runs rewritten code as written
        _queue_command=lambda handler, cores, words, *flags: queued.append(
            (handler, cores, words)
        )
        or len(queued),
        wait=lambda handle, timeout: SimpleNamespace(kind="done"),
    )
    return run, writes, queued


def a_room(
    rows=1, end=5, numbers=(1,), handlers=(1, 2), entered=True, groups=1, messages=None
):
    slots = {"reader": (2, 1024), "writer": (0, 0)}
    slots |= {part: (3, 1024) for part in ("unpack", "math", "pack")}
    recipes = dict.fromkeys(slots)
    made = [
        added.Group(16 * g, 16, recipes, dict(slots), set(numbers))
        for g in range(groups)
    ]
    return added.Room(Image(end), rows, end, made, set(handlers), entered, messages)
