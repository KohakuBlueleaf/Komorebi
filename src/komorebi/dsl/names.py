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
    "overlay": Name(
        "overlay",
        "km.handler(overlay=True): its code an overlay, loaded and called",
        None,
    ),
    "stamps": Name(
        "stamps",
        'km.handler(overlay=True, stamps="t"): its load and call cycles, a page a '
        'mover of tensor t; km.handler(stamps="t") of a handler calling a tile '
        "function: the math CPU's cycles of each call, a page a rank",
        None,
    ),
    "chain": Name(
        "chain",
        "km.chain: a method whose lines are calls of the class's handlers, a chain",
        "chain",
    ),
    "past": Name(
        "past", "km.past(t) at a chain's call: written further back, proven", None
    ),
    "direct": Name(
        "direct",
        "km.direct(t) at a chain's call: sent into the cores' L1; "
        'km.handler(direct=("t",)): the inputs that may be',
        None,
    ),
    "at_submit": Name(
        "at_submit",
        'km.at_submit("n") at a chain\'s call: an int word given at each '
        "run.submit(chain, n=value)",
        None,
    ),
    "prefetch": Name(
        "prefetch",
        "km.Stream(..., prefetch=False): read only once the chain's input is marked",
        None,
    ),
    "field": Name("L1", "km.L1[format, *shape]: a field resident in the L1", None),
    "symbol": Name(
        "symbol", "K = km.symbol(): a size the plan gives or the build solves", None
    ),
    "constant": Name(
        "constant",
        "K = km.constant(n): a number of the class fixed at its build; "
        "p: km.constant[int] = n: a handler's",
        None,
    ),
    "handlers": Name(
        "handlers",
        'Class(handlers={"h": [{"p": 4}, {"p": 8}]}): the sets of handler h\'s '
        "constant parameters built, each a handler of its own",
        None,
    ),
    "first_use": Name(
        "ANY",
        'Class(handlers={"h": km.ANY}): any set of handler h\'s constant parameters, '
        "a set not built compiled at its first submit and added to the run; "
        "km.ANY(p=n): the load's buffers sized for p=n too",
        None,
    ),
    "up_to": Name(
        "UPTO",
        'Class(handlers={"h": [km.UPTO(block=8)]}): block given at each submit, one '
        "build serving 1 to 8; a tensor parameter's name: its tile columns; rect: "
        "the rectangle, an argument after the parameters",
        None,
    ),
    "own": Name("own", "f: km.L1[...] = km.own(push=): a field only bodies use", None),
    "tensor": Name("Tensor", "km.Tensor: a parameter that is a tensor", "tensor"),
    "pattern": Name("Pattern", "km.Pattern: a parameter that is a pattern", "pattern"),
    "region": Name(
        "Region",
        "km.Region: a parameter that is an X280 thread's region, given by its name",
        "region",
    ),
    "class_thread": Name(
        "x280_thread",
        'thread = km.x280_thread("t.c", cluster=1): the class\'s X280 thread, C built '
        "at the make with the class's thread header "
        "(komorebi.program.thread is the module)",
        "X280 thread",
    ),
    "thread_region": Name(
        "region",
        "groups = km.region(bytes): a region of the class's thread (an int or the "
        "class's constants)",
        "region",
    ),
    "thread_words": Name(
        "thread_words",
        "Class(thread_words=[...]): the words of the class's thread, before the load's",
        None,
    ),
    "thread_tensors": Name(
        "thread_tensors",
        "Class(thread_tensors={...}): tensors the thread's chains name beside the "
        "make's",
        None,
    ),
    "args": Name("args", "a handler's command words from its arguments", "argument"),
    "plan": Name("plan", "the program's sizes, given or solved at build", None),
    # what a core is in the command
    "rank": Name("rank", "self.rank: this core's rank in the command's group", "rank"),
    "size": Name("size", "self.size: the cores of the command's group", "group"),
    "leader": Name("leader", "self.leader: the group's rank 0", "leader"),
    "partition": Name(
        "partition",
        "self.partition(pages): this core's part by rank; b.partition(pages): "
        "by its place along broadcast b's row or column",
        None,
    ),
    "core": Name(
        "core", "self.core: this core's place in the grid (its worker index)", "me"
    ),
    "x": Name("x", "self.x: this core's column in the broadcasts' rectangle", None),
    "y": Name("y", "self.y: this core's row in the broadcasts' rectangle", None),
    # messages (a dynamic handler: compute alone, the runtime's movers)
    "message": Name(
        "Message",
        "class Visit(km.Message): a message's named words (int annotations)",
        "message",
    ),
    "arrive_messages": Name(
        "arrive",
        "for message in self.arrive(): each message this core is given; the loop's "
        "handler is dynamic",
        None,
    ),
    "seeding": Name(
        "seeding", "if self.seeding(): the run's first messages, sent from here", None
    ),
    "send": Name("send", "self.send(rank, message): a first message to rank", "send"),
    "forward": Name(
        "forward", "self.forward(rank, message): the message, on to rank", "forward"
    ),
    "consume": Name("consume", "self.consume(): the message, ended here", "consume"),
    "direct_message": Name(
        "direct",
        "went = self.direct(rank, message): the message into its slot at rank, a "
        "declared pair; 0 (nothing sent) while the slot is busy",
        "direct",
    ),
    # a page's words as a table (every CPU of the handler)
    "entry": Name(
        "Entry",
        "class Job(km.Entry): an entry's named words (int annotations), 0, 1, ...",
        None,
    ),
    "entries": Name(
        "entries",
        "pages.entries(Job, first=, size=, count=): a page's words as a table of "
        "entries, read on every CPU of the handler",
        None,
    ),
    "entries_first": Name("first", "entries(first=w): the first entry's word", None),
    "entries_size": Name("size", "entries(size=n): the words an entry", None),
    "entries_count": Name(
        "count", "entries(count=w): the word that holds the entries' count", None
    ),
    "entries_fixed": Name(
        "fixed", "entries(fixed=n): n entries, known at build (no count word)", None
    ),
    # moves
    "stream": Name("Stream", "km.Stream(pages, block=, pattern=): a stream", "stream"),
    "write_stream": Name("write", "km.Stream(..., write=True): a write stream", None),
    "block": Name("block", "a stream's pages a block", "block"),
    "repeat": Name("repeat", "a stream's repetitions", "repetitions"),
    "drain": Name("drain", "km.Stream(pages).drain(): read, the tiles dropped", None),
    "push": Name("push", "out.push(tile): a tile into a write stream", None),
    "fill": Name("fill", "self.field.fill(km.Stream(pages)): pages into a field", None),
    "zero": Name(
        "zero", "self.field.zero(): the field's tiles zeroed on the core", None
    ),
    "pages": Name("pages", "tensor.pages(first, count): a range of pages", "page"),
    "items": Name(
        "items",
        "tensor.items(first, count, size=B): items of B bytes packed in its pages, "
        "for a stream",
        None,
    ),
    "item_size": Name("size", "items(..., size=B): the bytes an item", None),
    "item_step": Name(
        "step",
        "items(first=, size=, count=, step=s): item first, first + s, ...: one item "
        "every s",
        None,
    ),
    "put": Name(
        "put",
        "t.items(size=B).put(source, at=i): B bytes of L1 at item i of t, by a mover",
        None,
    ),
    "put_at": Name("at", "put(source, at=i): the item the bytes go to", None),
    "stream_end": Name(
        "end",
        "out.end(): compute's line, no page of drained stream out follows",
        None,
    ),
    "bump": Name(
        "bump",
        "slot = self.f.bump(i): word i of field f, then one more, on the CPU that "
        "takes it",
        None,
    ),
    "tiles": Name(
        "tiles",
        "tensor.tiles[r0:r1, c0:c1]: a rectangle of tiles, row-major; [r0:r1:s, "
        "c0:c1] its rows s apart",
        None,
    ),
    "transposed": Name(
        "T", "tensor.tiles[...].T: a cast's part, the rectangle's columns in turn", None
    ),
    "read": Name(
        "read",
        "pages.read(): the pages, read (a cast's block); word = pages.read() of one "
        "page: its first word, on mover B; a, b = pages.read()[0:2] of one page: its "
        "words 0 and 1, on both movers and compute",
        "read",
    ),
    "write": Name(
        "write",
        "pages.write(tile): the tile, written there; pages.write(word): a mover's "
        "word as the page's first word",
        "write",
    ),
    "reduce": Name("reduce", "km.reduce(tile, form=): the group's sum", "reduce"),
    "all_reduce": Name(
        "all_reduce",
        "km.all_reduce(tile, form=): the group's sum at every rank of it",
        "all_reduce",
    ),
    "form": Name("form", "a reduce's form: owner or tree", "form"),
    "group": Name(
        "group", "km.reduce(t, group=S): in groups of S consecutive ranks", None
    ),
    "combine": Name(
        "combine",
        "km.reduce(self.state, combine=MERGE): each arrival merged into the state "
        "by an inline body, not added",
        None,
    ),
    "broadcast": Name("Broadcast", "km.Broadcast(self, along=): a broadcast", None),
    "along": Name("along", "a broadcast's direction: row or column", None),
    "window": Name("window", "a broadcast's blocks in flight", "window"),
    "broadcast_mover": Name(
        "mover", 'a broadcast\'s mover, "A" (NoC 0) or "B" (NoC 1)', None
    ),
    "broadcast_unread": Name(
        "unread",
        "a broadcast's word: when not 0, each core's mover pushes each block unread",
        None,
    ),
    "broadcast_length": Name(
        "length",
        "a broadcast's band: so many cores of the row or column a sender",
        None,
    ),
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
    "srcb": Name("srcb", "x @ km.srcb(w): w is the product's SrcB operand", None),
    "srca": Name("srca", "km.srca(x) @ w: x is the product's SrcA operand", None),
    "takes": Name("takes", "the names an inline body reads", None),
    "gives": Name("gives", "what an inline body gives: int or km.Tile", None),
    "body_reads": Name(
        "reads",
        "km.cxx(..., takes=[t], reads=[t]): the tensors a body reads, for a chain an "
        "outside input of the call",
        None,
    ),
    "body_writes": Name(
        "writes",
        "km.cxx(..., takes=[t], writes=[t]): the tensors a body writes, for a chain "
        "written by the call",
        None,
    ),
    "tile_value": Name(
        "Tile", "gives=km.Tile: the tile a body leaves in DST 0", "tile"
    ),
    "on": Name(
        "on",
        'the CPU of an inline body: "mover_a", "mover_b", "unpack", "math", "pack"; '
        "compute's three when not given",
        None,
    ),
    "include": Name(
        "include",
        "km.cxx(..., include=[...]): #include lines for the body's CPU alone",
        None,
    ),
    "at": Name(
        "at",
        'km.cxx(..., at="start"): the body at its kernel\'s start, before the '
        'commands; at="end": after the run',
        None,
    ),
    "defines": Name(
        "defines",
        "class P(km.Program, defines={...}): the kernels' defines; made with "
        "defines= too",
        None,
    ),
    "program_form": Name(
        "form",
        'P(..., form="runtime"): the class built as a program in the runtime\'s form '
        '(km_program parts under the runtime\'s kernels); "kernel" the default',
        None,
    ),
    "startup": Name(
        "Startup",
        'class P(km.Program, startup=km.Startup("a", "b", "out", order="reverse")): '
        "compute's start-up by those fields",
        None,
    ),
    "compute_options": Name(
        "compute",
        "class P(km.Program, compute={...}): the compute kernel's options, by the "
        "host API's names",
        None,
    ),
    "tile_function": Name(
        "tile_function",
        "@km.tile_function: a function of tiles, traced and emitted for compute by "
        "the tile compiler; a handler calls it on the tiles of its streams",
        None,
    ),
    # a tile function's arithmetic: the tile compiler's trace functions, by these
    # names (error= and over= pass through to exp and exp2)
    **{
        f"math_{name}": Name(
            name, f"km.{name}: {what} (in a tile function)", f"trace.{name}"
        )
        for name, what in (
            ("exp", "e^x, exp(x, error=, over=)"),
            ("exp2", "2^x, exp2(x, error=, over=)"),
            ("sigmoid", "1 / (1 + e^-x)"),
            ("silu", "x sigmoid(x)"),
            ("tanh", "tanh(x), tanh(x, error=, over=)"),
            ("erf", "erf(x), erf(x, error=, over=)"),
            ("log", "the natural log, log(x, error=, over=)"),
            ("sqrt", "the square root"),
            ("rsqrt", "1 / sqrt(x)"),
            ("reciprocal", "1 / x"),
            ("relu", "max(x, 0)"),
            ("square", "x x"),
            ("maximum", "the larger of two, value by value"),
            ("minimum", "the smaller of two, value by value"),
            ("reduce_sum", "a row's sum (the row form), reduce_sum(x, exact=)"),
            ("reduce_max", "a row's largest value (the row form)"),
            (
                "lagging_max",
                "a reference that moves only past a lag, lagging_max(m, r, lag) "
                "(the step form)",
            ),
            (
                "word_max",
                "a per-row value's largest over the rows, one word (the step form)",
            ),
        )
    },
    "tolerance": Name(
        "tolerance",
        "@km.tile_function(tolerance=1e-6): its stated largest relative error "
        "against float64, printed with its cost",
        None,
    ),
    "row": Name(
        "row",
        "km.Stream(pages, row=2): each item a row of 2 tiles, for a tile function "
        "in the row form",
        "row",
    ),
    "to": Name(
        "to",
        "self.forward(rank, message, to=Class.handler), self.send(..., to=): the "
        "dynamic handler the message names there",
        "handler",
    ),
    "min": Name("min", "min(a, b): the smaller of ints of the handler, a word", None),
    "max": Name("max", "max(a, b): the larger of ints of the handler, a word", None),
    "zip": Name(
        "zip",
        "for a, b in zip(km.Stream(..), km.Stream(..)): streams read in step, a "
        "tile of each a turn",
        None,
    ),
    "tile_buffer": Name(
        "_buffer",
        "tile_buffer in a body: a stream tile's buffer (the loop's name)",
        None,
    ),
    "tile_index": Name(
        "_index", "tile_index in a body: a stream tile's place in its block", None
    ),
    "may_cross": Name(
        "may_cross", "program.may_cross: the report of what may cross a link", None
    ),
    "result": Name(
        "result", "self.result: refused; `return word` is the result", "result"
    ),
    # formats
    "fp32": Name("fp32", "km.fp32: float32", "float32"),
    "bf16": Name("bf16", "km.bf16: bfloat16", "bfloat16"),
    "u32": Name("u32", "km.u32: a 32-bit word", "uint32"),
    "fp16": Name("fp16", "km.fp16: float16, the A family's IEEE half", "float16"),
    "bfp8": Name("bfp8", "km.bfp8: block float, 7-bit magnitudes, A family", "bfp8"),
    "bfp4": Name("bfp4", "km.bfp4: block float, 3-bit magnitudes, A family", "bfp4"),
    "bfp2": Name("bfp2", "km.bfp2: block float, 1-bit magnitudes, A family", "bfp2"),
    "bfp8_b": Name("bfp8_b", "km.bfp8_b: block float, 7-bit magnitudes", "bfp8_b"),
    "bfp4_b": Name("bfp4_b", "km.bfp4_b: block float, 3-bit magnitudes", "bfp4_b"),
    "bfp2_b": Name("bfp2_b", "km.bfp2_b: block float, 1-bit magnitudes", "bfp2_b"),
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
