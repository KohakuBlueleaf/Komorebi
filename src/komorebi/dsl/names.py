"""Every name a program written as a class uses, in one table: the compiler finds each
construct by its spelling here, and `komorebi` exports the module-level ones, so a
rename is one edit of a row. Released words keep their released spelling.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Name:
    """A spelling in Python, what it is, and the nearest released word (None: none,
    a new name)."""

    spelling: str
    meaning: str
    released: str | None


# The key is what the compiler calls the construct; `km.` marks a module-level name.
NAMES = {
    # the class
    "handler": Name("handler", "km.handler: a method that is a handler", "handler"),
    "field": Name("L1", "km.L1[format, shape]: a field resident in the L1", None),
    "tensor": Name("Tensor", "km.Tensor: a parameter that is a tensor", "tensor"),
    "pattern": Name("Pattern", "km.Pattern: a parameter that is a pattern", "pattern"),
    "args": Name("args", "a handler's command words from its arguments", "argument"),
    "plan": Name("plan", "the program's sizes, given or solved at build", None),
    # what a core is in the command
    "rank": Name("rank", "self.rank: this core's rank in the command's group", "rank"),
    "size": Name("size", "self.size: the cores of the command's group", "group"),
    "leader": Name("leader", "self.leader: the group's rank 0", "leader"),
    "partition": Name("partition", "self.partition(pages): this core's part", None),
    "x": Name("x", "self.x: this core's column in the broadcasts' rectangle", None),
    "y": Name("y", "self.y: this core's row in the broadcasts' rectangle", None),
    # moves
    "stream": Name("Stream", "km.Stream(pages, block=, pattern=): a stream", "stream"),
    "write_stream": Name("write", "km.Stream(..., write=True): a write stream", None),
    "block": Name("block", "a stream's pages a block", "block"),
    "repeat": Name("repeat", "a stream's repetitions", "repetitions"),
    "push": Name("push", "out.push(tile): a tile into a write stream", None),
    "pages": Name("pages", "tensor.pages(first, count): a range of pages", "page"),
    "read": Name("read", "pages.read(): the pages, read", "read"),
    "write": Name("write", "pages.write(tile): the tile, written there", "write"),
    "reduce": Name("reduce", "km.reduce(tile, form=): the group's sum", "reduce"),
    "form": Name("form", "a reduce's form: owner or tree", "form"),
    "broadcast": Name("Broadcast", "km.Broadcast(self, along=): a broadcast", None),
    "along": Name("along", "a broadcast's direction: row or column", None),
    "window": Name("window", "a broadcast's blocks in flight", "window"),
    "sender": Name("sender", "broadcast.sender(): this core sends", "sender"),
    "cast": Name("cast", "broadcast.cast(pages): the block, to the others", "cast"),
    "arrive": Name("arrive", "broadcast.arrive(): the block, arrived", "arrive"),
    "rect": Name("rect", "made with rect=(x, y, columns, rows): its rectangle", "rect"),
    "nocs": Name("nocs", "made with nocs=session.nocs(): the grid's NoCs", "nocs"),
    "l1": Name("l1", "made with l1=session.l1()[1]: the L1 left for buffers", "l1"),
    # tiles and bodies
    "tile": Name("tile", "km.tile: tile values of compute", "tile"),
    "zeros": Name("zeros", "km.tile.zeros(format): a tile of zeros", None),
    "cxx": Name("cxx", "km.cxx(text, takes=, gives=, on=): an inline body", None),
    "takes": Name("takes", "the names an inline body reads", None),
    "gives": Name("gives", "what an inline body gives", None),
    "on": Name("on", "the CPU of an inline body", None),
    "result": Name("result", "self.result(word): the command's result word", "result"),
    # formats
    "fp32": Name("fp32", "km.fp32: float32", "float32"),
    "bf16": Name("bf16", "km.bf16: bfloat16", "bfloat16"),
    "u32": Name("u32", "km.u32: a 32-bit word", "uint32"),
}


def spelling(kind):
    """The Python spelling of construct `kind`."""
    return NAMES[kind].spelling


def kind_of(name, kinds):
    """The construct among `kinds` spelled `name`, or None."""
    for kind in kinds:
        if NAMES[kind].spelling == name:
            return kind
    return None
