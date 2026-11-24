"""The program's buffers and their circular buffer indices: one registry, the compiled
class's list, which every handler's buffers join in the order they are asked for (a
buffer is shared by name across handlers and grown in place)."""

from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.records import Buffer, Refused
from komorebi.dsl.front.names import spelling

# Circular buffer indices: reads from 0, writes from 16, the others from 8, to 25 (26
# to 31 are the runtime's: messages 26-28, tokens 29-30, trace 31).
FIRST_READ, FIRST_OTHER, FIRST_WRITE, LAST_INDEX = 0, 8, 16, 25
PAGE_BYTES = {"bfloat16": 2048, "float32": 4096}  # a tile page of each format
# Items a read half holds (double buffering on each mover and one more each), and a
# write half.
READ_ITEMS = 4
WRITE_ITEMS = 2
REGION_PAGE = 4096  # an X280 thread's region: whole pages of one bank (thread.PAGE)


def free_index(compiled, first, pair=False):
    """The first index from `first` that no buffer and no reserved index takes; for a
    `pair`, the first even step from `first` whose index and the next are both free."""
    taken = {b.index_a for b in compiled.buffers} | {
        b.index_b for b in compiled.buffers
    }
    taken |= compiled.reserved
    index = first
    if pair:
        while index in taken or index + 1 in taken:
            index += 2
    else:
        while index in taken:
            index += 1
    return index


def added(compiled, buffer):
    """`buffer` joins the program's buffers."""
    compiled.buffers.append(buffer)
    return buffer


def next_index(compiled):
    """The next index of a field or another single buffer."""
    index = free_index(compiled, FIRST_OTHER)
    if index > LAST_INDEX:
        raise Refused(f"{compiled.name}: more buffers and fields than the indices")
    return index


class SharedPair:
    """A handler's view of a read pair another handler made with another block: the
    pair itself (KM_RING_PADDED: its rings padded at their limits), this block."""

    def __init__(self, buffer, block):
        object.__setattr__(self, "buffer", buffer)
        object.__setattr__(self, "block", block)

    def __getattr__(self, name):
        return getattr(self.buffer, name)

    def __setattr__(self, name, value):
        setattr(self.buffer, name, value)


def pair(compiled, role, number, dtype, page_bytes, block, items, what, handler=None):
    """The program's pair `number` of `role`, made or grown; a read pair of another
    handler's block shared (SharedPair), its halves `items` of the largest block."""
    name = f"k{role.capitalize()}{number}"
    for buffer in compiled.buffers:
        if buffer.name == name:
            if (buffer.dtype, buffer.page_bytes, buffer.block) != (
                dtype,
                page_bytes,
                block,
            ):
                if not shareable(compiled, buffer, role, (dtype, page_bytes), handler):
                    break
                most = max(buffer.pages[0], items * block)
                buffer.pages, buffer.padded = (most, most), True
                if handler not in buffer.handlers:
                    buffer.handlers.append(handler)
                first = buffer.what
                if first.startswith("shared by "):
                    first = first.split("; ", 1)[1]
                buffer.what = f"shared by {', '.join(buffer.handlers)}; {first}"
                return SharedPair(buffer, block)
            if handler not in buffer.handlers:
                buffer.handlers.append(handler)
            return buffer
    else:
        first = FIRST_READ if role == "read" else FIRST_WRITE
        index = free_index(compiled, first, pair=True)
        buffer = Buffer(
            name,
            role,
            index,
            index + 1,
            dtype,
            page_bytes,
            (items * block, items * block),
            block,
            what,
        )
        if index + 1 > LAST_INDEX:
            raise Refused(no_pair(compiled, name, role, handler, first))
        buffer.handlers = [handler]
        return added(compiled, buffer)
    return pair(
        compiled, role, f"{number}b", dtype, page_bytes, block, items, what, handler
    )


def no_pair(compiled, name, role, handler, first):
    """The refusal of a pair the indices have no room for: the count, the limit and
    each handler's pairs (a pair of one name serves every handler of its format)."""
    pairs = [b for b in compiled.buffers if b.index_b is not None]
    held = "; ".join(
        f"{b.name} {b.index_a}-{b.index_b} ({b.dtype}, {', '.join(map(str, users))})"
        for b, users in ((b, b.handlers) for b in pairs)
    )
    singles = len(compiled.buffers) - len(pairs) + len(compiled.reserved)
    return (
        f"{compiled.name}: {handler}'s {role} stream needs pair {name}, and indices "
        f"{first} to {LAST_INDEX} have no two free side by side: {len(pairs)} pairs "
        f"({held}) and {singles} other buffers, fields and reserved indices take "
        f"them; open fewer streams at once in one handler, give a stream the "
        "format and page of another handler's (a pair of one name serves both), or "
        "split the class"
    )


def declared_pair(compiled, attribute, role, dtype, page_bytes, pages, what):
    """The pair of a km.Pair of the class (role "read" or "write"), by the stream
    pairs' rule: the first two free indices side by side from their role's first."""
    name = f"kPair{camel(attribute)}"
    first = FIRST_READ if role == "read" else FIRST_WRITE
    index = free_index(compiled, first, pair=True)
    if index + 1 > LAST_INDEX:
        pairs = [b for b in compiled.buffers if b.index_b is not None]
        held = "; ".join(
            f"{b.name} {b.index_a}-{b.index_b} ({b.dtype}, "
            f"{', '.join(map(str, b.handlers)) or 'declared'})"
            for b in pairs
        )
        singles = len(compiled.buffers) - len(pairs) + len(compiled.reserved)
        raise Refused(
            f"{compiled.name}: km.{spelling('pair')} {attribute} ({role}) needs pair "
            f"{name}, and "
            f"indices {first} to {LAST_INDEX} have no two free side by side: "
            f"{len(pairs)} pairs ({held}) and {singles} other buffers, fields and "
            "reserved indices take them; declare fewer pairs, open fewer streams at "
            "once in one handler, or split the class"
        )
    buffer = Buffer(
        name, role, index, index + 1, dtype, page_bytes, (pages, pages), 1, what
    )
    buffer.declared = True
    return added(compiled, buffer)


def padded_rings(compiled):
    """The KM_RING_PADDED mask: a bit for each buffer of a shared read pair."""
    mask = 0
    for buffer in compiled.buffers:
        if buffer.padded:
            mask |= (1 << buffer.index_a) | (1 << buffer.index_b)
    return mask


def shareable(compiled, buffer, role, page_format, handler):
    """Whether a handler's read stream of another block may take `buffer`: a make
    that shares pairs, a read pair of the same format and page, streamed by other
    handlers only, none with a block given at each submit (handler None)."""
    return (
        compiled.share_pairs
        and role == "read"
        and buffer.role == "read"
        and (buffer.dtype, buffer.page_bytes) == page_format
        and handler not in buffer.handlers
        and None not in buffer.handlers + [handler]
    )


def out_buffer(compiled, name, dtype, page_bytes, pages, what, item=None):
    """The buffer `name` of pages compute packs and mover B writes, items of `item`
    pages (else `pages`), grown to `pages` when it is there in this format with items
    of that width only (its positions last across commands: an item of another width
    passes the ring's end, bad_item); else the next name that does, or a new one."""
    item = item or pages
    taken = {b.name: b for b in compiled.buffers}
    while name in taken:
        buffer = taken[name]
        if (buffer.dtype, buffer.page_bytes) == (dtype, page_bytes):
            most = max(buffer.pages[0], pages)
            if (buffer.items or [buffer.pages[0]]) == [item] and most % item == 0:
                buffer.pages = (most,)
                return buffer
        name += "b"
    index = free_index(compiled, FIRST_WRITE)
    if index > LAST_INDEX:
        raise Refused(f"{compiled.name}: more buffers than the indices")
    buffer = Buffer(name, "out", index, None, dtype, page_bytes, (pages,), 1, what)
    buffer.items = [item]
    return added(compiled, buffer)


def other_buffer(compiled, name, role, dtype, page_bytes, pages, what):
    """The buffer `name` (an arrival, a broadcast's, a field's ...), made once."""
    for buffer in compiled.buffers:
        if buffer.name == name:
            return buffer
    index = free_index(compiled, FIRST_OTHER)
    if index >= FIRST_WRITE and role != "field":
        raise Refused(f"{compiled.name}: more buffers than the indices")
    buffer = Buffer(name, role, index, None, dtype, page_bytes, (pages,), 1, what)
    return added(compiled, buffer)


def block_step(rows, row_step):
    """The tensor pages from one block of `rows` rows to the next: an int, or C++ for a
    row step that is a word."""
    return rows * row_step if isinstance(row_step, int) else f"{rows} * {row_step}"
