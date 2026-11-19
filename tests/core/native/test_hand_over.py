"""The Tensix dispatcher's hand-over between its two loops (td_slots.hpp), over every
interleaving of their L1 accesses and the workers' acks on a small ring: out fills a
slot's record, then its issued word; in reads issued, then the record and the ack, and
frees the slot by its done word, which out reads before it reuses it."""

from typing import NamedTuple

import pytest


class Out(NamedTuple):
    command: int = 1  # the next command (its request and sequence alike)
    step: int = 0  # 0: pick a slot; 1, 2: the record's two words; 3: issued
    slot: int = -1
    busy: tuple = ()  # out's own copy: the sequence of each busy slot, 0 free


class In(NamedTuple):
    slot: int = 0  # the slot the sweep is at
    step: int = 0  # 0: read issued; 1, 2: the record; 3: the ack; 4: done
    issued: int = 0
    record: tuple = (0, 0)
    seen: tuple = ()  # the sequence in last freed in each slot


class Memory(NamedTuple):
    issued: tuple
    done: tuple
    record: tuple  # (request, sequence) of each slot
    ack: tuple


def put(values, index, value):
    return values[:index] + (value,) + values[index + 1 :]


def out_steps(out, memory, commands):
    """Out's next access, or None once every command went out."""
    if out.command > commands:
        return None
    s = out.slot
    if out.step == 0:  # out reads each busy slot's done word, then takes the first free
        busy = tuple(
            0 if b and memory.done[i] == b else b for i, b in enumerate(out.busy)
        )
        free = [i for i, b in enumerate(busy) if b == 0]
        if not free:
            return out._replace(busy=busy), memory
        return out._replace(busy=busy, slot=free[0], step=1), memory
    if out.step == 1:
        record = put(memory.record, s, (out.command, memory.record[s][1]))
        return out._replace(step=2), memory._replace(record=record)
    if out.step == 2:
        record = put(memory.record, s, (out.command, out.command))
        return out._replace(step=3), memory._replace(record=record)
    issued = put(memory.issued, s, out.command)
    busy = put(out.busy, s, out.command)
    out = out._replace(command=out.command + 1, step=0, slot=-1, busy=busy)
    return out, memory._replace(issued=issued)


def in_steps(inside, memory, events):
    """In's next access: its sweep goes round the slots for ever."""
    s, depth = inside.slot, len(memory.issued)
    nxt = inside._replace(slot=(s + 1) % depth, step=0)
    if inside.step == 0:
        issued = memory.issued[s]
        if issued == inside.seen[s]:
            return nxt, memory, events
        return inside._replace(step=1, issued=issued), memory, events
    if inside.step in (1, 2):
        record = put(inside.record, inside.step - 1, memory.record[s][inside.step - 1])
        return inside._replace(step=inside.step + 1, record=record), memory, events
    if inside.step == 3:
        if memory.ack[s] != inside.issued:
            return nxt, memory, events
        return inside._replace(step=4), memory, events
    request, sequence = inside.record
    assert sequence == inside.issued, "a record read torn: not the issued command's"
    done = put(memory.done, s, inside.issued)
    seen = put(inside.seen, s, inside.issued)
    return nxt._replace(seen=seen), memory._replace(done=done), events + (request,)


def explore(depth, commands):
    """Every state reachable by out, in and the workers (a worker acks a slot's line
    once out picked the slot for it: the line goes out before its record), and the
    states from which every command's event can still come."""
    zero = (0,) * depth
    memory = Memory(zero, zero, ((0, 0),) * depth, zero)
    start = (Out(busy=zero), In(seen=zero), memory, ())
    seen, frontier, back, finished = {start}, [start], {}, set()
    while frontier:
        state = frontier.pop()
        out, inside, memory, events = state
        if len(events) == commands:
            finished.add(state)
        following = []
        stepped = out_steps(out, memory, commands)
        if stepped is not None:
            following.append((stepped[0], inside, stepped[1], events))
        following.append((out, *in_steps(inside, memory, events)))
        for slot in range(depth):
            line = out.busy[slot] or (out.command if out.slot == slot else 0)
            if line and memory.ack[slot] != line and memory.done[slot] != line:
                acked = memory._replace(ack=put(memory.ack, slot, line))
                following.append((out, inside, acked, events))
        for moved in following:
            back.setdefault(moved, set()).add(state)
            if moved not in seen:
                seen.add(moved)
                frontier.append(moved)
    can_end, frontier = set(finished), list(finished)
    while frontier:
        for earlier in back.get(frontier.pop(), ()):
            if earlier not in can_end:
                can_end.add(earlier)
                frontier.append(earlier)
    return seen, finished, can_end


@pytest.mark.parametrize("depth, commands", [(1, 3), (2, 4), (3, 4)])
def test_every_command_gives_one_event_and_no_record_is_read_torn(depth, commands):
    states, finished, can_end = explore(depth, commands)
    assert finished and can_end == states  # no interleaving gets stuck
    for state in states:  # never two events for one command, never a wrong request
        assert len(set(state[3])) == len(state[3])
        assert set(state[3]) <= set(range(1, commands + 1))
