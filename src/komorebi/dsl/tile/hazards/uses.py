"""Each value's readers and writers in a unit program's code, the index the algebra
passes of lower.py keep while they change the code."""


class _Uses:
    """Each value's readers and writers in `code`, built once for a pass, in code
    order; the pass `drop`s an instruction it changes or deletes and `add`s it back
    once changed (deleting keeps the order of the rest)."""

    def __init__(self, code):
        self.at = {id(x): i for i, x in enumerate(code)}
        self.read, self.written = {}, {}
        for x in code:
            self.add(x)

    def add(self, insn):
        """Indexes `insn` by the values it reads and writes now."""
        for v in insn.reads():
            self.read.setdefault(v, {})[id(insn)] = insn
        for v in insn.writes():
            self.written.setdefault(v, {})[id(insn)] = insn

    def drop(self, insn):
        """Forgets `insn` under the values it reads and writes now."""
        for v in insn.reads():
            self.read.get(v, {}).pop(id(insn), None)
        for v in insn.writes():
            self.written.get(v, {}).pop(id(insn), None)

    def readers(self, value):
        """The instructions that read `value`, in code order."""
        return sorted(self.read.get(value, {}).values(), key=self._order)

    def writers(self, value):
        """The instructions that write `value`, in code order."""
        return sorted(self.written.get(value, {}).values(), key=self._order)

    def _order(self, insn):
        return self.at[id(insn)]
