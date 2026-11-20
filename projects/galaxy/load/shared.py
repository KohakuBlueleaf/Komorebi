"""The one load the box pieces share (README.md). Each piece's
`make(session, W, layers, run=run)` hands its programs to it (`take`) instead of loading
its own; the driver then loads once (`start`), and the pieces switch and submit through
it as through a run of their own."""

import komorebi as km

# the DRAM room for the added programs' parts: rows of the widest part named ahead
# (a Projections-size class: 5 rows of 63,488 B), every program of a token's 62
RESERVE = 23_000_000

SHARED = {"math_fidelity": "HiFi2", "fp32_dest_acc_en": True, "dst_full_sync_en": True}


class Hold(km.Program, compute=SHARED):
    """The dict's program of a load with no hand program: a page copy, its buffers at
    index 0 and 16 only (bfloat16); an added class keeps its own formats there."""

    @km.handler
    def copy(self, x: km.Tensor, y: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for xb in km.Stream(x.pages(0, 1)):
            total += xb
        y.pages(0, 1).write(total)


def hold(session):
    """A Hold of the session's, in the runtime's form."""
    page = lambda: session.empty((32, 32), "bfloat16", "tile")  # noqa: E731
    return Hold(x=page(), y=page(), form="runtime", session=session)


class Shared:
    """{program: group} from every piece, in the order taken. `start` makes one program
    the dict's, on every rank (a hand program must be: it cannot be added), names the
    others ahead of it and adds them on every island."""

    def __init__(self, session, reserve=RESERVE):
        self.session, self.reserve = session, reserve
        self.groups = {}
        self.run = self.first = None
        self.held = []  # each rank's program, as the switches left it

    def take(self, groups):
        """A piece's {program: group}, before `start`."""
        if self.run is not None:
            raise km.KomorebiError("a program taken after the shared load was made")
        for program, group in groups.items():
            self.groups.setdefault(program, group)

    def start(self, first=None):
        """THE load: `first` (else the first program taken) the dict's, the others
        named ahead and added; one slot a kernel, the service part in the reader's.
        Returns the run."""
        first = next(iter(self.groups)) if first is None else first
        rest = [program for program in self.groups if program is not first]
        for program in rest:
            first.ahead(program)
        first.slots(count=1, service="shared")
        self.first = first
        self.run = self.session.load(
            {first: (0, self.session.ranks)}, reserve=self.reserve
        )
        for program in rest:
            self.run.addon(program)
        self.held = [first] * self.session.ranks
        return self.run

    def load(self, program, group, **options):
        """A switch: the run's load of `program` on `group` (first, count), kept as
        what those ranks hold."""
        ticket = self.__getattr__("load")(program, group, **options)
        first, count = group
        self.held[first : first + count] = [program] * count
        return ticket

    def holds(self, program, group):
        """Whether every rank of `group` holds `program`: a piece asks this before a
        switch, since another piece may have switched those ranks since."""
        first, count = group
        return all(held is program for held in self.held[first : first + count])

    def __getattr__(self, name):  # load, submit, wait, store, island, stop: the run's
        run = self.__dict__.get("run")
        if run is None:
            raise km.KomorebiError(f"{name}: the shared load is not made (start)")
        return getattr(run, name)
