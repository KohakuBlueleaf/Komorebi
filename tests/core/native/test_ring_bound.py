"""The ring bound (KM_CONTROL_BOUND), checked over every interleaving on a small ring.

A model of one core's pass (`km::scan`, `Reader::next`) against the X280's grow and
shrink, with the NoC's writes landing in order; its scan is checked against the header.
"""

import itertools
import json
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[3]
INCLUDE = ROOT / "src" / "komorebi_native" / "include"

# A pass reads the bound, each slot's word 0, a new slot's deciding words (land1 to
# land3, group, word 0 again), the bound again when it holds a pick, then copies it.
START, SLOT, LAND1, LAND2, LAND3, GROUP, AGAIN, POST, COPY_HEAD, COPY_TAIL = range(10)


class Reader(NamedTuple):
    phase: int = START
    bound: int = 0  # the bound word read at the start of the pass
    slot: int = 0
    sequence: int = 0  # the slot's word 0 as first read
    words: tuple = (0, 0, 0, False)  # land1 to land3, own
    pick: int = -1
    pick_sequence: int = 0
    partial: bool = False
    copy: tuple = (0, False)
    visited: int = 0  # bit s: the pass read slot s's word 0
    behind: int = 0  # bit s: a line's word 0 landed in slot s after that


class Model:
    """One core against a dispatcher submitting `own` in order (True: the core's own
    command, False: another group's), sequences 1, 2, ... `guard`: the second read of
    the bound; `late_bound`: a bound write may land after the writes behind it."""

    def __init__(self, depth, step, own, guard=True, late_bound=False):
        self.depth, self.step, self.own = depth, step, own
        self.guard, self.late_bound = guard, late_bound

    def start(self):
        d = self.depth
        memory = (((0, False, 0),) * d, d)  # slots (word 0, own, land words), bound
        mine = ((0,) * d, (0,) * d, 0, ())  # last, waiting sequences, mask, taken
        dispatcher = (0, (None,) * d, d, ())  # next command, busy, bound, writes
        return memory, Reader(), mine, dispatcher

    def arrive(self, memory, reader, write):
        slots, bound = memory
        if write[0] == "bound":
            return (slots, write[1]), reader
        part, slot, sequence, own = write
        head, head_own, land = slots[slot]
        if part == "head":
            new = (sequence, own, land)
            if reader.visited >> slot & 1:
                reader = reader._replace(behind=reader.behind | 1 << slot)
        else:
            new = (head, head_own, sequence)
        return (slots[:slot] + (new,) + slots[slot + 1 :], bound), reader

    def dispatcher_moves(self, state):
        """Every next state the dispatcher or the NoC can make."""
        memory, reader, mine, (k, busy, bound, writes) = state
        out = []
        free = [s for s in range(self.depth) if busy[s] is None]
        if k < len(self.own) and free:
            slot, sequence, own = free[0], k + 1, self.own[k]  # the lowest free slot
            new_busy = busy[:slot] + ((sequence, own),) + busy[slot + 1 :]
            new_bound, grow = bound, ()
            if slot >= bound:  # the bound is written first, on the line's path
                high = 1 + max(s for s in range(self.depth) if new_busy[s] is not None)
                new_bound = min(self.depth, -(-high // self.step) * self.step)
                grow = (("bound", new_bound),)
            for parts in (("head", "land"), ("land", "head")):  # a line lands in parts
                line = tuple((part, slot, sequence, own) for part in parts)
                after = (k + 1, new_busy, new_bound, writes + grow + line)
                out.append((memory, reader, mine, after))
        for slot in range(self.depth):  # another group's: any time; own: once taken
            if busy[slot] is not None and (
                not busy[slot][1] or busy[slot][0] in mine[3]
            ):
                new_busy = busy[:slot] + (None,) + busy[slot + 1 :]
                out.append((memory, reader, mine, (k, new_busy, bound, writes)))
        high = 1 + max(
            (s for s in range(self.depth) if busy[s] is not None), default=-1
        )
        for lower in range(max(high, 1), bound):  # any shrink over free slots only
            after = (k, busy, lower, writes + (("bound", lower),))
            out.append((memory, reader, mine, after))
        for index, write in enumerate(writes):  # in order; a late bound may be passed
            if index and not (self.late_bound and writes[index - 1][0] == "bound"):
                break
            rest = writes[:index] + writes[index + 1 :]
            landed, seen = self.arrive(memory, reader, write)
            out.append((landed, seen, mine, (k, busy, bound, rest)))
        return out

    def reader_move(self, state):
        """The reader's next read: (state, the sequence taken or None)."""
        memory, r, mine, dispatcher = state
        slots, bound_word = memory
        last, waiting, mask, taken = mine
        bound = r.bound if 0 < r.bound <= self.depth else self.depth

        def go(reader, mine=mine, took=None):
            return (memory, reader, mine, dispatcher), took

        if r.phase == START:
            return go(Reader(SLOT, bound_word))
        if r.phase == SLOT:
            if r.slot == bound:
                return go(r._replace(phase=POST))
            sequence = slots[r.slot][0]
            seen = r._replace(slot=r.slot + 1, visited=r.visited | 1 << r.slot)
            if sequence <= last[r.slot]:
                return go(seen)
            if mask >> r.slot & 1 and waiting[r.slot] == sequence:
                if r.pick < 0 or r.pick_sequence > sequence:
                    seen = seen._replace(pick=r.slot, pick_sequence=sequence)
                return go(seen)
            mine = (last, waiting, mask & ~(1 << r.slot), taken)
            deep = r._replace(phase=LAND1, sequence=sequence, visited=seen.visited)
            return go(deep, mine)
        if r.phase in (LAND1, LAND2, LAND3):
            words = list(r.words)
            words[r.phase - LAND1] = slots[r.slot][2]
            return go(r._replace(phase=r.phase + 1, words=tuple(words)))
        if r.phase == GROUP:
            return go(r._replace(phase=AGAIN, words=r.words[:3] + (slots[r.slot][1],)))
        if r.phase == AGAIN:
            head, slot, sequence = slots[r.slot][0], r.slot, r.sequence
            nxt = r._replace(phase=SLOT, slot=slot + 1)
            if head != sequence or not r.words[0] == r.words[1] == r.words[2] == head:
                return go(nxt._replace(partial=True))
            if not r.words[3]:
                last = last[:slot] + (sequence,) + last[slot + 1 :]
                return go(nxt, (last, waiting, mask, taken))
            waiting = waiting[:slot] + (sequence,) + waiting[slot + 1 :]
            if r.pick < 0 or r.pick_sequence > sequence:
                nxt = nxt._replace(pick=slot, pick_sequence=sequence)
            return go(nxt, (last, waiting, mask | 1 << slot, taken))
        if r.phase == POST:
            whole = not 0 < r.bound < self.depth  # a pass over the depth: no raise
            kept = not self.guard or whole or bound_word == r.bound
            if r.pick >= 0 and not r.partial and kept:
                return go(r._replace(phase=COPY_HEAD))
            return go(Reader())
        if r.phase == COPY_HEAD:
            return go(r._replace(phase=COPY_TAIL, copy=slots[r.pick][:2]))
        if r.copy[0] != r.pick_sequence or slots[r.pick][2] != r.copy[0]:
            return go(Reader())
        last = last[: r.pick] + (r.pick_sequence,) + last[r.pick + 1 :]
        return go(Reader(), (last, waiting, mask, taken + (r.pick_sequence,)), r.pick)

    def check(self):
        """Every interleaving. Returns (states, None), or (states, the first fault): a
        take of another group's command, an own command lost, or a take before an
        earlier own command that did not land behind the pass (`today`: allowed)."""
        own_sequences = [k + 1 for k, mine in enumerate(self.own) if mine]
        first = self.start()
        seen, stack = {first}, [first]
        while stack:
            state = stack.pop()
            nexts = self.dispatcher_moves(state)
            after, slot = self.reader_move(state)
            if slot is not None:
                fault = self.take_fault(state, after, slot, own_sequences)
                if fault:
                    return len(seen), fault
            nexts.append(after)
            k, _, _, writes = state[3]
            if k == len(self.own) and not writes and state[1].phase == START:
                lost = self.settle(state, own_sequences)
                if lost:
                    return len(seen), lost
            for candidate in nexts:
                if candidate not in seen:
                    seen.add(candidate)
                    stack.append(candidate)
        return len(seen), None

    def take_fault(self, state, after, slot, own_sequences):
        taken, before = after[2][3][-1], state[2][3]
        if not self.own[taken - 1]:
            return f"took {taken}, another group's"
        if taken in before:
            return f"took {taken} twice"
        earlier = [s for s in own_sequences if s < taken and s not in before]
        busy = state[3][1]
        for sequence in earlier:
            held = [s for s in range(self.depth) if busy[s] and busy[s][0] == sequence]
            if not held or not state[1].behind >> held[0] & 1:
                return f"took {taken} before {sequence}"
        return None

    def settle(self, state, own_sequences):
        """Everything written and landed: the reader alone takes every own command."""
        for _ in range(64 * (len(own_sequences) + 2)):
            state, _ = self.reader_move(state)
        missing = [s for s in own_sequences if s not in state[2][3]]
        return f"never took {missing}" if missing else None


# Larger cases (a ring of 3 with 4 commands: 20.8 million states) take minutes.
@pytest.mark.parametrize("depth, step, most", [(3, 1, 3), (4, 1, 2), (4, 2, 2)])
def test_every_interleaving_loses_nothing_and_adds_no_reorder(depth, step, most):
    for count in range(1, most + 1):
        for own in itertools.product((True, False), repeat=count):
            states, fault = Model(depth, step, own).check()
            assert fault is None, (own, fault)
            assert states > 1


def test_a_bound_that_lands_late_reorders_without_the_rule():
    # The order rule is needed: a bound overtaken by the line lets 3 go before 2.
    _, fault = Model(3, 1, (False, True, True), late_bound=True).check()
    assert fault == "took 3 before 2"


def test_a_bound_raised_in_the_pass_reorders_without_the_check():
    # The second read of the bound is needed: without it 3 goes before 2.
    _, fault = Model(3, 1, (False, True, True), guard=False).check()
    assert fault == "took 3 before 2"


SCAN_CHECK = r"""
#include <cstdio>
#include <cstdlib>
#include "komorebi/device/core/km_command.hpp"

// Per input line: the bound, then 6 slots of (word 0, land words, own); one pass.
int main() {
    unsigned bound, words[24];
    while (std::scanf("%u", &bound) == 1) {
        km_command_t ring[KM_RING_SLOTS] = {};
        uint32_t last[KM_RING_SLOTS] = {};
        for (unsigned k = 0; k < 24; k++) {
            std::scanf("%u", &words[k]);
        }
        for (unsigned s = 0; s < 6; s++) {
            ring[s].sequence = words[4 * s];
            ring[s].land1 = ring[s].land2 = ring[s].land3 = words[4 * s + 1];
            ring[s].group = km::make_group(words[4 * s + 2] ? 5 : 9, 1);
        }
        const km::Pass found = km::scan(ring, last, bound, 5);
        std::printf("[%d, %u, %d, [", found.pick, found.pick_sequence, found.partial);
        for (unsigned s = 0; s < 6; s++) {
            std::printf(s ? ", %u" : "%u", last[s]);
        }
        std::printf("]]\n");
    }
    return 0;
}
"""


def model_scan(slots, bound):
    """The model's pass with nothing landing: [pick, sequence, partial, last]."""
    model = Model(len(slots), 1, ())
    _, reader, mine, dispatcher = model.start()
    state = ((tuple(slots), bound), reader, mine, dispatcher)
    state, _ = model.reader_move(state)
    while state[1].phase != POST:
        state, _ = model.reader_move(state)
    r = state[1]
    return [
        r.pick,
        r.pick_sequence if r.pick >= 0 else 0,
        int(r.partial),
        list(state[2][0]),
    ]


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
def test_the_model_scan_is_the_header_scan(tmp_path):
    source_file = tmp_path / "scan.cpp"
    source_file.write_text(SCAN_CHECK)
    exe = tmp_path / "scan"
    command = ["c++", "-std=c++17", "-Wall", "-Werror", "-I", str(INCLUDE)]
    subprocess.run([*command, str(source_file), "-o", str(exe)], check=True)
    contents = [(0, False, 0), (3, True, 3), (2, False, 2), (4, True, 0), (1, True, 1)]
    cases = [
        (slots, bound)
        for slots in itertools.product(contents, repeat=6)
        if len({s[0] for s in slots if s[0]}) == len([s for s in slots if s[0]])
        for bound in (2, 6)
    ]
    lines = [
        " ".join(
            map(
                str,
                [bound, *(w for s, own, land in slots for w in (s, land, int(own), 0))],
            )
        )
        for slots, bound in cases
    ]
    out = subprocess.run(
        [str(exe)], input="\n".join(lines), capture_output=True, text=True, check=True
    )
    got = [json.loads(line) for line in out.stdout.splitlines()]
    assert got == [model_scan(slots, bound) for slots, bound in cases]
