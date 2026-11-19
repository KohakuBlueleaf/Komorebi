"""km.Session: one card, or islands; its constructor picks. Above both the one-card
session (komorebi.session.session) and the island session (komorebi.islands.session),
which name neither this module nor each other's class."""

from komorebi.errors import NotServed
from komorebi.islands.session import IslandSession
from komorebi.session import dispatch
from komorebi.session import session as card_session


class Session(card_session.Session):
    """One card: `card` selects it, `device` uses an open one (`cards`, or `islands` 2
    to 4 beside `card`: an IslandSession); `grid_timeout` bounds every grid wait. As a
    context manager it stops a loaded run at the exit (through the runtime on an
    exception) and closes the device it opened; so do the process's exit and SIGTERM."""

    def __new__(cls, *args, cards=None, islands=None, **options):
        asked = options.pop("dispatcher", None)
        several = cards is not None or islands is not None
        if several and dispatch.choose(asked, True) == dispatch.TENSIX:
            raise NotServed(dispatch.TENSIX, "islands and links (cards=, islands=)")
        if cards is not None:
            if args or "card" in options or "device" in options:
                raise ValueError("cards= is given alone, without card or device")
            return IslandSession(cards, islands=islands, **options)
        if islands is not None:
            if args or "card" not in options or "device" in options:
                raise ValueError("islands= is given with card=, without device")
            card = options.pop("card")
            return IslandSession([card], islands=islands, **options)
        return super().__new__(cls)
