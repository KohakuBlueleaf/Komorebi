"""A card session's and run's calls on islands (IslandSession, IslandRun): one value
where every island's is the same, a call over every island, else refused by name."""

import time

import numpy as np

from komorebi.errors import KomorebiError, WaitTimeout
from komorebi.session.parts import load as card_load

ENGINE = (
    "an engine's stream or graph runs on the island it is uploaded to (uploads over "
    "islands are not built): run.island(i).{name}(...)"
)
# A card run's names an island run does not offer, and what to call instead
RUN_REFUSED = {
    **{
        name: ENGINE
        for name in (
            "upload",
            "start",
            "restart",
            "release",
            "cancel",
            "wait_engine",
            "run_hostfed",
            "check_templates",
        )
    },
    "sequence": (
        "a ticket is a part an island, each with its own sequence number: "
        "run.trace().command(ticket) gives its rows, run.island(i).sequence(handle) "
        "island i's part's"
    ),
    "holds": "each island's cores by its own core numbers: run.island(i).holds()",
    "state_words": (
        "each island's raw words: run.island(i).state_words(); run.state() is every "
        "rank's"
    ),
    "backend": "each island's run has its own: run.island(i).backend",
    "inspect_ticks": (
        "each island's dispatcher reads its own: run.island(i).inspect_ticks"
    ),
    "stored": "each island stores its own commands: run.island(i).stored",
    "blocks_held": "each island's run holds its own blocks: run.island(i).blocks_held",
    "dispatcher_given_up": (
        "each island's dispatcher counts its own: run.island(i).dispatcher_given_up()"
    ),
}
TENSOR_REFUSED = {"buffer": "each island's part has its own: tensor.parts[i].buffer"}


def refused(owner, table, kind, name):
    """The AttributeError for `name` of `owner` (so hasattr stays False): what to call
    instead when `table` names it."""
    why = table.get(name)
    if why is None:
        return AttributeError(
            f"'{type(owner).__name__}' object has no attribute '{name}'"
        )
    return AttributeError(f"{kind}.{name} on islands: {why.format(name=name)}")


def same(what, values, own):
    """The one value of `values` (one an island) when every island's is the same,
    else KomorebiError naming `own`, which gives island i's."""
    first = values[0]
    for value in values[1:]:
        if value is not first and value != first:
            raise KomorebiError(f"{what}: the islands' differ; {own} gives island i's")
    return first


class SessionCalls:
    """IslandSession's calls of a card session: each island's when they are all the
    same (as `grid`), refused by name when they differ (two cards' devices)."""

    def _same(self, name, read):
        if not self.sessions:
            raise KomorebiError(f"session.{name}: the session is not open")
        own = f"session.island(i).{name}"
        values = [read(session) for session in self.sessions]
        return same(f"session.{name}", values, own)

    def nocs(self):
        """The grid's NoC coordinates by core (Session.nocs), each island's the same."""
        return self._same("nocs()", lambda session: session.nocs())

    def l1(self):
        """(kernel buffer, L1 left for buffers) of each core (Session.l1), each
        island's the same."""
        return self._same("l1()", lambda session: session.l1())

    def launch(self):
        """Session.launch on the one device of a split card."""
        return self._same("launch()", lambda session: session).launch()

    @property
    def dram_banks(self):
        """The DRAM banks of the open device (Session.dram_banks), each island's the
        same."""
        return self._same("dram_banks", lambda session: session.dram_banks)

    @property
    def pll_mhz(self):
        """The X280 clock the islands' runtimes were given (Session.pll_mhz)."""
        if not self.sessions:
            asked = self._options["pll_mhz"]
            return card_load.X280_MHZ if asked is None else asked
        return self._same("pll_mhz", lambda session: session.pll_mhz)

    @property
    def card(self):
        """The card (Session.card): one card, or one card split into islands."""
        if len(self.cards) != 1:
            raise KomorebiError(
                f"session.card: the session has cards {list(self.cards)}; "
                "session.cards, or session.island(i).card"
            )
        return self.cards[0]

    @property
    def device(self):
        """The open device (Session.device) of one card, or of a split card."""
        return self._same("device", lambda session: session.device)

    @property
    def channel(self):
        """The host channel (Session.channel) of one card, or of a split card."""
        return self._same("channel", lambda session: session.channel)


class RunCalls:
    """IslandRun's calls of a card run that every island answers the same, or that
    act on every island; a card run's name it does not offer is refused by name."""

    def __getattr__(self, name):
        raise refused(self, RUN_REFUSED, "run", name)

    def _same(self, name, read):
        values = []
        for island, run in enumerate(self.runs):
            try:
                values.append(read(run))
            except KomorebiError as error:
                raise self._error(error, island) from None
        return same(f"run.{name}", values, f"run.island(i).{name}")

    def alive(self):
        """Whether every island's run takes commands."""
        return all(run.alive() for run in self.runs)

    @property
    def added(self):
        """What run.addon placed (Run.added), one numbering on every island."""
        self._same("added", lambda run: {k: v.number for k, v in run.added.items()})
        return self.runs[0].added

    @property
    def programs(self):
        """The run's programs by program field (Run.programs), the same on every
        island."""
        self._same("programs", lambda run: [id(one) for one in run.programs])
        return self.runs[0].programs

    @property
    def h(self):
        return self._same("h", lambda run: run.h)

    def program_field(self, program):
        return self._same("program_field()", lambda run: run.program_field(program))

    def load_field(self, program):
        return self._same("load_field()", lambda run: run.load_field(program))

    def field_handler(self, number, program_field):
        return self._same(
            "field_handler()", lambda run: run.field_handler(number, program_field)
        )

    def thread_words(self, handler, args=()):
        """Run.thread_words: a command's words as submit would send them, the same on
        every island (one field and handler numbering, a tensor's table word)."""
        return self._same("thread_words()", lambda run: run.thread_words(handler, args))

    def store(self):
        """Stores what is queued on every island now, without waiting (send)."""
        self.send()

    def wait_any(self, handles, timeout=10.0):
        """Sends what is queued and waits for the first ticket among `handles` whose
        every part arrived; returns its event (the lowest ticket if several) as wait()
        does, and raises as wait() does for it."""
        wanted = np.unique(np.asarray(list(handles), np.int64))
        if not wanted.size or wanted.min() < 1 or wanted.max() >= self._next:
            raise ValueError("no handles, or a handle that was never submitted")
        started = time.perf_counter()
        deadline = time.monotonic() + timeout
        self.send()
        try:
            while True:
                done = np.ones(wanted.size, bool)
                for island, run in enumerate(self.runs):
                    parts = self._parts[island][wanted]
                    held = parts != 0
                    done[held] &= run._arrived(parts[held])
                if done.any():
                    break
                if time.monotonic() > deadline:
                    raise WaitTimeout(
                        f"none of {wanted.size} tickets done in {timeout} s",
                        wanted.tolist(),
                    )
                for island, run in enumerate(self.runs):
                    parts = self._parts[island][wanted]
                    parts = np.unique(parts[parts != 0])
                    pending = parts[~run._arrived(parts)]
                    if pending.size:
                        try:
                            run._check_alive()
                            run._step(pending, int(pending[-1]), 0.0)
                        except KomorebiError as error:
                            raise self._error(error, island) from None
        finally:
            self.wait_s += time.perf_counter() - started
        return self.wait(int(wanted[done][0]), max(deadline - time.monotonic(), 0.0))

    def keep_resets(self, exclude=(), timeout=10.0):
        """Run.keep_resets on every island within one `timeout`: `exclude` flat ranks
        (each island's link cores get none, as at the load); True when every island
        kept its lists (else each island's reason in `recovery_refused`)."""
        deadline = time.monotonic() + timeout
        excluded = set(exclude)
        kept = True
        for island, run in enumerate(self.runs):
            first = self.graph.first(island)
            mine = self.graph.rank_cores[island]
            cores = set(range(run.cores)) - set(mine)
            cores |= {
                mine[rank - first] for rank in excluded if 0 <= rank - first < len(mine)
            }
            try:
                kept &= bool(
                    run.keep_resets(cores, max(deadline - time.monotonic(), 0.0))
                )
            except KomorebiError as error:
                raise self._error(error, island) from None
        return kept

    def group(self, handle):
        """The flat group (first, count) of ticket `handle` (Run.group), or None."""
        if not 0 < int(handle) < self._next:
            return None
        for island, run in enumerate(self.runs):
            part = int(self._parts[island][handle])
            if part:
                return run.group(part)
        return None

    @property
    def recovery_refused(self):
        """Why an island kept no reset lists (Run.recovery_refused), naming the first
        such island; None when every island kept them."""
        for island, run in enumerate(self.runs):
            if run.recovery_refused is not None:
                return f"island {island}: {run.recovery_refused}"
        return None

    @property
    def stalls(self):
        """Every island's watcher stalls (Run.stalls), each with its `island`."""
        return [
            {**stall, "island": island}
            for island, run in enumerate(self.runs)
            for stall in run.stalls
        ]

    @property
    def native_s(self):
        """Host time in every island's native loop (Run.native_s), summed."""
        return sum(run.native_s for run in self.runs)
