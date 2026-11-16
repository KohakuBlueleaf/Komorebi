"""A finished command as the dispatcher reports it: one card's event, and an island
run's, the tuple of its parts' events."""

from typing import NamedTuple


class Event(NamedTuple):
    """A finished command: its handle, kind (done, timeout, pong, rejected), the
    auxiliary (done: the group's first core; timeout: the slot; rejected: the reason),
    submit-to-completion time in X280 timer ticks, and for a command submitted with
    `result=True` the leader's result word (sent as the ticks)."""

    handle: int
    kind: str
    auxiliary: int
    ticks: int
    result: int | None = None

    @property
    def microseconds(self):
        return self.ticks / 50.0  # the X280 timer runs at 50 MHz


class IslandEvent(tuple):
    """A finished command of an island run: the tuple of its parts' events in island
    order, which also reads as one card's event (docs/reference/islands.md#runs)."""

    @property
    def parts(self):
        return tuple(self)

    def _worst(self):
        """The first part that is not done, else the slowest."""
        undone = [event for event in self if event.kind != "done"]
        return undone[0] if undone else max(self, key=lambda event: event.ticks)

    kind = property(lambda self: self._worst().kind)  # "done" only if every part is
    ticks = property(lambda self: max(event.ticks for event in self))
    microseconds = property(lambda self: max(event.microseconds for event in self))
    reason = property(lambda self: self._worst().auxiliary)  # rejected: its reason
    handle = property(lambda self: self[0].handle)  # the group's first rank's part
    auxiliary = property(lambda self: self[0].auxiliary)
    result = property(lambda self: self[0].result)
