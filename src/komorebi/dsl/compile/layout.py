"""The program's buffers and their circular buffer indices: one registry, the compiled
class's list, which every handler's buffers join in the order they are asked for (a
buffer is shared by name across handlers and grown in place)."""

from komorebi.dsl.compile.records import Buffer, Refused

# Circular buffer indices: reads from 0, writes from 16, the others from 8 (24 and up
# are the runtime's).
FIRST_READ, FIRST_OTHER, FIRST_WRITE, LAST_INDEX = 0, 8, 16, 25
PAGE_BYTES = {"bfloat16": 2048, "float32": 4096}  # a tile page of each format


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


def pair(compiled, role, number, dtype, page_bytes, block, items, what):
    """The program's pair `number` of `role`, made or grown."""
    name = f"k{role.capitalize()}{number}"
    for buffer in compiled.buffers:
        if buffer.name == name:
            if (buffer.dtype, buffer.page_bytes, buffer.block) != (
                dtype,
                page_bytes,
                block,
            ):
                break
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
            raise Refused(f"{compiled.name}: more pairs than buffers")
        return added(compiled, buffer)
    return pair(compiled, role, f"{number}b", dtype, page_bytes, block, items, what)


def out_buffer(compiled, name, dtype, page_bytes, pages, what):
    """The buffer `name` of pages compute packs and mover B writes, grown to `pages`
    when it is there in this format; else a new one (under another name if taken)."""
    for buffer in compiled.buffers:
        if buffer.name == name and (buffer.dtype, buffer.page_bytes) == (
            dtype,
            page_bytes,
        ):
            buffer.pages = (max(buffer.pages[0], pages),)
            return buffer
    while any(b.name == name for b in compiled.buffers):
        name += "b"
    index = free_index(compiled, FIRST_WRITE)
    if index > LAST_INDEX:
        raise Refused(f"{compiled.name}: more buffers than the indices")
    buffer = Buffer(name, "out", index, None, dtype, page_bytes, (pages,), 1, what)
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
