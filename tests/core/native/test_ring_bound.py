"""The ring bound (KM_CONTROL_BOUND) and the reader's list of its own waiting commands,
checked over every interleaving on a small ring: a model of one core's pass against the
X280's grow and shrink, the NoC's writes landing in order; checked against the header.
"""

import itertools
import json
import random
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
WRAP = 1 << 32


def newer(sequence, other):
    """km::newer: the ring's wrapping comparison of two sequence numbers."""
    return 0 < (sequence - other) % WRAP < WRAP // 2


def listed(order, seen, slot, sequence):
    """km::waiting_add: `slot` after every listed slot whose sequence is not newer."""
    at = len(order)
    while at > 0 and newer(seen[order[at - 1]], sequence):
        at -= 1
    return order[:at] + (slot,) + order[at:]


def put(values, index, value):
    return values[:index] + (value,) + values[index + 1 :]


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
    command), sequences `first` and on, 0 skipped (the X280's), after consumed ones.
    `guard`: the bound read again; `late_bound`: it may land late; `give_ups`: slots
    given up (rule 8), used again; `stale_drop`: a pass drops such a listed command."""

    def __init__(
        self,
        depth,
        step,
        own,
        guard=True,
        late_bound=False,
        first=1,
        give_ups=0,
        stale_drop=True,
    ):
        self.depth, self.step, self.own = depth, step, own
        self.guard, self.late_bound, self.give_ups = guard, late_bound, give_ups
        self.stale_drop = stale_drop
        self.sequences, sequence = [], first % WRAP
        for _ in own:
            sequence = sequence or 1
            self.sequences.append(sequence)
            sequence = (sequence + 1) % WRAP
        self.prior = ((first - 1) % WRAP,) * depth

    def start(self):
        d = self.depth
        memory = (tuple((p, False, p) for p in self.prior), d)  # (word 0, own, land)
        mine = (self.prior, self.prior, (), ())  # last, seen, listed slots, taken
        dispatcher = (0, (None,) * d, d, (), ())  # next, busy, bound, writes, given up
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
        memory, reader, mine, (k, busy, bound, writes, given) = state
        out = []
        free = [s for s in range(self.depth) if busy[s] is None]
        if k < len(self.own) and free:
            slot, own = free[0], self.own[k]  # the lowest free slot
            sequence = self.sequences[k]
            new_busy = busy[:slot] + ((sequence, own),) + busy[slot + 1 :]
            new_bound, grow = bound, ()
            if slot >= bound:  # the bound is written first, on the line's path
                high = 1 + max(s for s in range(self.depth) if new_busy[s] is not None)
                new_bound = min(self.depth, -(-high // self.step) * self.step)
                grow = (("bound", new_bound),)
            for parts in (("head", "land"), ("land", "head")):  # a line lands in parts
                line = tuple((part, slot, sequence, own) for part in parts)
                after = (k + 1, new_busy, new_bound, writes + grow + line, given)
                out.append((memory, reader, mine, after))
        for slot in range(self.depth):  # another group's: any time; own: once taken
            if busy[slot] is None:
                continue
            new_busy = put(busy, slot, None)
            if not busy[slot][1] or busy[slot][0] in mine[3]:
                out.append((memory, reader, mine, (k, new_busy, bound, writes, given)))
            elif len(given) < self.give_ups:  # given up before the core took it
                after = (k, new_busy, bound, writes, given + (busy[slot][0],))
                out.append((memory, reader, mine, after))
        high = 1 + max(
            (s for s in range(self.depth) if busy[s] is not None), default=-1
        )
        for lower in range(max(high, 1), bound):  # any shrink over free slots only
            after = (k, busy, lower, writes + (("bound", lower),), given)
            out.append((memory, reader, mine, after))
        for index, write in enumerate(writes):  # in order; a late bound may be passed
            if index and not (self.late_bound and writes[index - 1][0] == "bound"):
                break
            rest = writes[:index] + writes[index + 1 :]
            landed, seen = self.arrive(memory, reader, write)
            out.append((landed, seen, mine, (k, busy, bound, rest, given)))
        return out

    def reader_move(self, state):
        """The reader's next read: (state, the slot taken from or None)."""
        memory, r, mine, dispatcher = state
        slots, bound_word = memory
        last, seen, order, taken = mine
        bound = r.bound if 0 < r.bound <= self.depth else self.depth

        def go(reader, mine=mine, took=None):
            return (memory, reader, mine, dispatcher), took

        def drop(slot):  # km::waiting_drop
            if not self.stale_drop:
                return mine
            return (
                last,
                put(seen, slot, last[slot]),
                tuple(s for s in order if s != slot),
                taken,
            )

        if r.phase == START:
            return go(Reader(SLOT, bound_word))
        if r.phase == SLOT:
            if r.slot == bound:
                return go(r._replace(phase=POST))
            sequence = slots[r.slot][0]
            nxt = r._replace(slot=r.slot + 1, visited=r.visited | 1 << r.slot)
            if not newer(sequence, seen[r.slot]):
                return go(nxt)
            if seen[r.slot] != last[r.slot]:  # listed, and the slot used again
                mine = drop(r.slot)
            deep = r._replace(phase=LAND1, sequence=sequence, visited=nxt.visited)
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
                last, seen = put(last, slot, sequence), put(seen, slot, sequence)
                return go(nxt, (last, seen, order, taken))
            order = listed(order, seen, slot, sequence)
            return go(nxt, (last, put(seen, slot, sequence), order, taken))
        if r.phase == POST:
            whole = not 0 < r.bound < self.depth  # a pass over the depth: no raise
            kept = not self.guard or whole or bound_word == r.bound
            if order and not r.partial and kept:
                pick = r._replace(phase=COPY_HEAD, pick=order[0])
                return go(pick._replace(pick_sequence=seen[order[0]]))
            return go(Reader())
        if r.phase == COPY_HEAD:
            return go(r._replace(phase=COPY_TAIL, copy=slots[r.pick][:2]))
        if r.copy[0] != r.pick_sequence or slots[r.pick][2] != r.copy[0]:
            return go(Reader())
        last = put(last, r.pick, r.pick_sequence)
        mine = (last, seen, order[1:], taken + (r.pick_sequence,))
        return go(Reader(), mine, r.pick)

    def check(self):
        """Every interleaving. Returns (states, None), or (states, the first fault): a
        take of another group's command, an own command lost, or a take before an
        earlier own command that did not land behind the pass (`today`: allowed)."""
        own_sequences = [q for q, mine in zip(self.sequences, self.own) if mine]
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
            k, _, _, writes, _ = state[3]
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
        index = self.sequences.index(taken)
        if not self.own[index]:
            return f"took {taken}, another group's"
        if taken in before:
            return f"took {taken} twice"
        busy, given = state[3][1], state[3][4]
        earlier = [
            s
            for s in own_sequences[: own_sequences.index(taken)]
            if s not in before and s not in given
        ]
        for sequence in earlier:
            held = [s for s in range(self.depth) if busy[s] and busy[s][0] == sequence]
            if not held or not state[1].behind >> held[0] & 1:
                return f"took {taken} before {sequence}"
        return None

    def settle(self, state, own_sequences):
        """Everything written and landed: the reader alone takes every own command."""
        for _ in range(64 * (len(own_sequences) + 2)):
            state, _ = self.reader_move(state)
        given = state[3][4]
        missing = [s for s in own_sequences if s not in state[2][3] and s not in given]
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


@pytest.mark.parametrize("first", [WRAP - 2, WRAP - 1])
def test_the_list_orders_by_the_wrapping_comparison(first):
    # Sequences that wrap (0 is skipped): 1 is newer than WRAP - 1, and is taken after.
    for own in itertools.product((True, False), repeat=3):
        states, fault = Model(3, 1, own, first=first).check()
        assert fault is None, (own, fault)
        assert states > 1


def test_a_given_up_slot_used_again_drops_its_listed_command():
    # Rule 8: a busy slot given up before the core took its command, used again while
    # the core still lists the old one: the stale entry goes, nothing else is lost.
    for own in [(True, True), (True, False, True), (True, True, True)]:
        states, fault = Model(2, 1, own, give_ups=1).check()
        assert fault is None, (own, fault)
        assert states > 1


def test_a_stale_entry_kept_takes_wrongly():
    _, fault = Model(2, 1, (True, False, True), give_ups=1, stale_drop=False).check()
    assert fault is not None


SCAN_CHECK = r"""
#include <cstdio>
#include <cstdlib>
#include "komorebi/device/core/km_command.hpp"

// Per input line: every slot's last sequence, the bound, then 6 slots of (word 0, land
// words, own); one pass.
int main() {
    unsigned prior, bound, words[24];
    while (std::scanf("%u %u", &prior, &bound) == 2) {
        km_command_t ring[KM_RING_SLOTS] = {};
        uint32_t last[KM_RING_SLOTS];
        for (unsigned s = 0; s < KM_RING_SLOTS; s++) {
            last[s] = prior;
        }
        for (unsigned k = 0; k < 24; k++) {
            std::scanf("%u", &words[k]);
        }
        for (unsigned s = 6; s < KM_RING_SLOTS; s++) {
            ring[s].sequence = prior;
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


def model_scan(slots, bound, prior):
    """The model's pass with nothing landing: [pick, sequence, partial, last]."""
    model = Model(len(slots), 1, (), first=prior + 1)
    _, reader, mine, dispatcher = model.start()
    state = ((tuple(slots), bound), reader, mine, dispatcher)
    state, _ = model.reader_move(state)
    while state[1].phase != POST:
        state, _ = model.reader_move(state)
    last, seen, order, _ = state[2]
    return [
        order[0] if order else -1,
        seen[order[0]] if order else 0,
        int(state[1].partial),
        list(last),
    ]


# Built as a load's kernels are (a bound of 8 slots) and as a build without the define.
BOUNDS = [[], ["-DKM_RING_DEPTH_MAX=8"]]


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
@pytest.mark.parametrize("bound", BOUNDS)
def test_the_model_scan_is_the_header_scan(tmp_path, bound):
    source_file = tmp_path / "scan.cpp"
    source_file.write_text(SCAN_CHECK)
    exe = tmp_path / "scan"
    command = ["c++", "-std=c++17", "-Wall", "-Werror", *bound, "-I", str(INCLUDE)]
    subprocess.run([*command, str(source_file), "-o", str(exe)], check=True)
    cases = []
    for prior in (0, WRAP - 3):  # the second: sequences that wrap past 0
        sequence = Model(1, 1, (True,) * 4, first=prior + 1).sequences
        sequence = [prior, *sequence]  # the X280's: each next one but 0
        contents = [
            (sequence[0], False, sequence[0]),
            (sequence[3], True, sequence[3]),
            (sequence[2], False, sequence[2]),
            (sequence[4], True, prior),
            (sequence[1], True, sequence[1]),
        ]
        cases += [
            (slots, bound, prior)
            for slots in itertools.product(contents, repeat=6)
            if len({s[0] for s in slots if s[0] != prior})
            == len([s for s in slots if s[0] != prior])
            for bound in (2, 6)
        ]
    lines = [
        " ".join(
            map(
                str,
                [
                    prior,
                    bound,
                    *(w for s, own, land in slots for w in (s, land, int(own), 0)),
                ],
            )
        )
        for slots, bound, prior in cases
    ]
    out = subprocess.run(
        [str(exe)], input="\n".join(lines), capture_output=True, text=True, check=True
    )
    got = [json.loads(line) for line in out.stdout.splitlines()]
    assert got == [model_scan(slots, bound, prior) for slots, bound, prior in cases]


LIST_CHECK = r"""
#include <cstdio>
#include "komorebi/device/core/km_command.hpp"

// Per input line: n, then n operations (slot, sequence): the slot dropped, then listed
// with that sequence unless it is 0. Prints the list, first to last.
int main() {
    unsigned n, slot, sequence;
    while (std::scanf("%u", &n) == 1) {
        km::Waiting waiting = {};
        waiting.head = 61;  // the list wraps around the end of `order`
        for (unsigned k = 0; k < n; k++) {
            std::scanf("%u %u", &slot, &sequence);
            km::waiting_drop(waiting, slot, 7);  // as a pass does before it lists
            if (sequence != 0) {
                km::waiting_add(waiting, slot, sequence);
            }
        }
        for (unsigned k = 0; k < waiting.count; k++) {
            const unsigned at = (waiting.head + k) & (KM_RING_DEPTH_MAX - 1);
            std::printf(k ? " %u" : "%u", waiting.order[at]);
        }
        std::printf("\n");
    }
    return 0;
}
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
@pytest.mark.parametrize("bound", BOUNDS)
def test_the_header_list_is_the_model_list(tmp_path, bound):
    source_file = tmp_path / "list.cpp"
    source_file.write_text(LIST_CHECK)
    exe = tmp_path / "list"
    command = ["c++", "-std=c++17", "-Wall", "-Werror", *bound, "-I", str(INCLUDE)]
    subprocess.run([*command, str(source_file), "-o", str(exe)], check=True)
    rng = random.Random(1)
    cases = []
    for _ in range(400):
        base, operations = rng.choice([1, WRAP - 6]), []
        for _ in range(rng.randrange(1, 12)):
            slot = rng.randrange(8)
            sequence = (base + rng.randrange(10)) % WRAP or 1
            operations.append((slot, 0 if rng.random() < 0.3 else sequence))
        cases.append(operations)
    lines = [
        " ".join(map(str, [len(ops), *(w for op in ops for w in op)])) for ops in cases
    ]
    out = subprocess.run(
        [str(exe)], input="\n".join(lines), capture_output=True, text=True, check=True
    )
    want = []
    for operations in cases:
        order, seen = (), (0,) * 8
        for slot, sequence in operations:
            order = tuple(s for s in order if s != slot)  # dropped, then listed
            if sequence:
                order = listed(order, seen, slot, sequence)
                seen = put(seen, slot, sequence)
        want.append(" ".join(map(str, order)))
    assert out.stdout.splitlines() == want
