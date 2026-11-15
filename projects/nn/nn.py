"""Komorebi-nn's host side: kernel calls as entries of a list in DRAM (nn.h), the X280
executor thread that walks it (thread/executor.c), and the host calls that drive it:
a closed list run N times, an open list appended while it runs, or the host submitting
every call itself. The decode project's layer as calls is at the end."""

import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

import komorebi as km
from komorebi import abi
from projects.decode import card

HERE = Path(__file__).resolve().parent
NN = abi.constants(HERE / "nn.h")
FAILED = 0xFFFFFFF0  # nn.h's NN_FAILED: a run's result word with its status below
ENTRY_WORDS, ENTRY_BYTES = NN["NN_ENTRY_WORDS"], NN["NN_ENTRY_BYTES"]
LINE_BYTES = 64  # an entry's second line (words 16 to 31): what a link rewrites
CLUSTER = 2  # the executor's work cluster, beside the decode's own thread on 1
TICKS_PER_MICROSECOND = 50  # kmx_time() and the events count the 50 MHz timer
# A run's terms in a stats slot, NN_TERM_* in order
TERMS = ("total", "fetch", "submit", "wait", "command", "idle", "check")
RECORD = abi.constants(abi.headers().chain)  # an entry's first line: the record
if RECORD["kChainRecordBytes"] != LINE_BYTES or RECORD["kChainWords"] != 16:
    raise ImportError("an entry's first line is not the chain's record (km_chain.h)")


# An entry's check (nn.h): the sum of word i x (2 i + 1), the link and the check out
FACTORS = 2 * np.arange(ENTRY_WORDS, dtype=np.uint32) + 1
FACTORS[[NN["NN_ENTRY_NEXT"], NN["NN_ENTRY_CHECK"]]] = 0


def checks(rows):
    """Each entry's check word, for entries [n, 32] uint32 (sums mod 2 ** 32, as C)."""
    return (np.asarray(rows, np.uint32) * FACTORS).sum(axis=-1, dtype=np.uint32)


class Call(NamedTuple):
    """One kernel call: `handler` (a number) on cores `cores` (first, count) with ten
    words `args`; `flags` the command's (KM_FLAG_*), `program` its program field,
    `overlay` the operation it calls (the host's note); `record` the chain's four words
    (info, three ranges) and `row` its stamp row when the call can be a chain's record.
    """

    handler: int
    cores: tuple
    args: tuple
    flags: int = 0
    program: int = 0
    overlay: int = 0
    record: tuple = (0, 0, 0, 0)
    row: int = 0


def entry(call, kind):
    """The 32 words of `call`'s entry of `kind` (NN_KIND_*), its links 0."""
    if len(call.args) != 10 or len(call.record) != 4:
        raise ValueError("a call has ten words and a record of four")
    first, count = call.cores
    words = np.zeros(ENTRY_WORDS, np.uint32)
    words[RECORD["kChainArguments"] : RECORD["kChainGroup"]] = call.args
    words[RECORD["kChainGroup"]] = first | count << 16
    words[RECORD["kChainInfo"] : RECORD["kChainNext"]] = call.record
    words[RECORD["kChainNext"]] = call.row  # its links' low bits (km_chain.h)
    words[NN["NN_ENTRY_HEADER"]] = call.handler | call.flags << 8 | call.program << 24
    words[NN["NN_ENTRY_KIND"]] = kind
    words[NN["NN_ENTRY_OVERLAY"]] = call.overlay
    return words


class Entries:
    """A list's entries from slot `first` of the executor's list region, in order: a
    call a command, or a run of calls as one chain command."""

    def __init__(self, first=0):
        self.first = first
        self.rows = []  # the entries' words
        self.items = []  # (first row, rows): what the executor runs, one at a time

    def command(self, call, kind=NN["NN_KIND_COMMAND"]):
        """`call` as one command, submitted and waited for (`kind` NN_KIND_TRANSFER: a
        PUSH or PULL whose nonce word the executor XORs with RUN's a3)."""
        self.items.append((len(self.rows), 1))
        self.rows.append(entry(call, kind))

    def chain(self, calls, mode, flags=0, token=0, ahead=False):
        """`calls` as one chain command: the kernel's chain mode `mode` (its a4) on
        the cores of every call, the first call's handler and program; `flags`; a0
        `token` (chain.h's token word), a1 kChainAheadOff unless `ahead`."""
        if not calls:
            raise ValueError("a chain of no calls")
        head = calls[0]
        count = max(call.cores[0] + call.cores[1] for call in calls)
        words = entry(head, NN["NN_KIND_CHAIN"])
        words[NN["NN_ENTRY_HEADER"]] = head.handler | flags << 8 | head.program << 24
        words[NN["NN_ENTRY_CHAIN_MODE"]] = mode
        words[NN["NN_ENTRY_CHAIN_GROUP"]] = count << 16
        words[NN["NN_ENTRY_CHAIN_TOKEN"]] = token
        words[NN["NN_ENTRY_CHAIN_AHEAD"]] = 0 if ahead else card.CHAIN["kChainAheadOff"]
        self.items.append((len(self.rows), len(calls)))
        self.rows.append(words)
        self.rows += [entry(call, NN["NN_KIND_NONE"]) for call in calls[1:]]

    def __len__(self):
        return len(self.rows)

    def words(self, base, sequence=0):
        """[entries, 32] uint32 with the links: each item's next item (the list's
        end 0), each chain record's next record (its last 0); `sequence` (an open
        list's number) and the check. `base`: the region's bank address."""
        if self.first + len(self.rows) > NN["NN_ENTRIES_MAX"]:
            raise ValueError(
                f"{len(self.rows)} entries from {self.first}: past the list"
            )
        out = np.array(self.rows, np.uint32).reshape(-1, ENTRY_WORDS)
        address = [base + (self.first + i) * ENTRY_BYTES for i in range(len(out) + 1)]
        for number, (row, count) in enumerate(self.items):
            last = number + 1 == len(self.items)
            out[row, NN["NN_ENTRY_NEXT"]] = (
                0 if last else address[self.items[number + 1][0]]
            )
            for record in range(row, row + count - 1):
                out[record, RECORD["kChainNext"]] |= address[record + 1]
        out[:, NN["NN_ENTRY_SEQUENCE"]] = sequence
        out[:, NN["NN_ENTRY_CHECK"]] = checks(out)
        return out


def executor_thread(program, cluster=CLUSTER):
    """Adds the executor to `program` on work cluster `cluster`; returns the program."""
    program.thread(
        cluster=cluster,
        sources=[HERE / "thread" / "executor.c"],
        includes=[HERE],
        regions={
            "list": NN["NN_LIST_BYTES"],
            "stats": 4 * NN["NN_SLOT_WORDS"] * NN["NN_SLOTS"],
        },
        handlers={"run": NN["NN_HANDLER_RUN"], "drain": NN["NN_HANDLER_DRAIN"]},
    )
    return program


class Executor:
    """The executor of a live run: writes lists into its region (the host's DRAM
    writes, as a command's block is written), runs them, reads the stats after stop."""

    def __init__(self, session, run, cluster=CLUSTER):
        self.session, self.run, self.cluster = session, run, cluster
        self.bank, self.base = run.threads[cluster].regions.address("list")
        self.slot = 0
        self.sequence = 0  # the last open list's number

    def address(self, index):
        """The bank address of the list's entry `index`."""
        return self.base + index * ENTRY_BYTES

    def store(self, index, words, offset=0):
        """Words (uint32) into the list from entry `index` on (`offset` bytes into
        it); no barrier."""
        data = np.ascontiguousarray(words, np.uint32).view(np.uint8).reshape(-1)
        address = self.address(index) + offset
        self.session._module.write_dram(self.session.device, [self.bank], address, data)

    def fence(self):
        """Returns once every word stored so far is in DRAM."""
        self.session._module.dram_barrier(self.session.device)

    def write(self, entries):
        """A closed list into the region, in DRAM on return."""
        self.store(entries.first, entries.words(self.base))
        self.fence()

    def next_slot(self):
        """A stats slot for the next run."""
        if self.slot == NN["NN_SLOTS"]:
            raise ValueError(f"{self.slot} runs: the stats hold {NN['NN_SLOTS']}")
        self.slot += 1
        return self.slot - 1

    def closed(self, entries, times=1, timeout=10.0):
        """RUN: the written list `entries` from its first entry to its end, `times`
        times, waited for: (the thread's ticks, its stats slot)."""
        slot = self.next_slot()
        args = [self.address(entries.first), times, slot]
        handle = self.run.submit("run", cluster=self.cluster, args=args, result=True)
        return ended(self.run.wait(handle, timeout=timeout)), slot

    def start(self, entries, times=1, nonce=0):
        """RUN of the written list `entries` (its transfers' nonce `nonce`), stored on
        the card and not waited for: (its handle, its stats slot)."""
        slot = self.next_slot()
        args = [self.address(entries.first), times, slot, nonce]
        handle = self.run.submit("run", cluster=self.cluster, args=args, result=True)
        self.run.store()
        return handle, slot

    def open(self, at, idle=1.0, fence=False):
        """An open list after an anchor at entry `at` (OpenList), its drain started;
        `fence`: a DRAM barrier between an append's entries and its link."""
        self.sequence += 1
        return OpenList(self, at, idle, fence)

    def stats(self):
        """Every stats slot after run.stop(): [slots, words] uint32."""
        words = self.run.threads[self.cluster].read("stats").view(np.uint32)
        return words.reshape(NN["NN_SLOTS"], NN["NN_SLOT_WORDS"])


def ended(event):
    """A run's ticks from its event; KomorebiError when it failed."""
    if event.kind != "done" or event.result >= FAILED:
        raise km.KomorebiError(f"executor: {event}")
    return int(event.result)


class OpenList:
    """An open list: the executor's DRAIN runs each entry appended after the anchor at
    entry `at` once its link is in DRAM and the entry is new (its sequence, its check);
    sync() appends an END and waits for it. `times`: the host's seconds of all appends
    by term (build, store, fence, link); `fence`: a DRAM barrier before each link."""

    def __init__(self, executor, at, idle, fence):
        self.executor, self.fence, self.sequence = executor, fence, executor.sequence
        self.last_at, self.next = at, at + 1  # the entry the thread polls; the free one
        self.last = entry(NOTHING, NN["NN_KIND_NONE"])
        executor.store(at, self.last)
        executor.fence()
        self.slot = executor.next_slot()
        args = [executor.address(at), self.slot, round(idle * 50e6), self.sequence]
        run = executor.run
        self.handle = run.submit(
            "drain", cluster=executor.cluster, args=args, result=True
        )
        run.store()
        self.times = dict.fromkeys(("build", "store", "fence", "link"), 0.0)
        self.appended = 0

    def append(self, call=None, chain=None, mode=0, flags=0, kind=None):
        """Appends `call` as one command, or `chain` (calls) as one chain command in
        `mode` with `flags`, or an entry of `kind` (END) alone: the new entries into
        DRAM, then the link in the last entry's second line (after a barrier with
        `fence`; without, the thread waits until the entries are new)."""
        started = time.perf_counter()
        entries = Entries(self.next)
        if chain is not None:
            entries.chain(chain, mode, flags)
        elif call is not None:
            entries.command(call)
        else:
            entries.rows.append(entry(NOTHING, kind))
            entries.items.append((0, 1))
        words = entries.words(self.executor.base, self.sequence)
        built = time.perf_counter()
        self.executor.store(self.next, words)
        stored = time.perf_counter()
        if self.fence:
            self.executor.fence()
        fenced = time.perf_counter()
        self.last[NN["NN_ENTRY_NEXT"]] = self.executor.address(self.next)
        self.executor.store(self.last_at, self.last[ENTRY_WORDS // 2 :], LINE_BYTES)
        linked = time.perf_counter()
        spans = (built - started, stored - built, fenced - stored, linked - fenced)
        for name, seconds in zip(self.times, spans, strict=True):
            self.times[name] += seconds
        self.appended += 1
        self.last, self.last_at = words[0].copy(), self.next
        self.next += len(words)

    def sync(self, timeout=10.0):
        """Appends an END and waits for the drain's event: the thread's ticks."""
        self.append(kind=NN["NN_KIND_END"])
        return ended(self.executor.run.wait(self.handle, timeout=timeout))


NOTHING = Call(0, (0, 0), (0,) * 10)  # an anchor's or an END's words


def drive(run, calls, timeout=5.0):
    """The host submits every call itself and waits for each (run.submit, run.wait):
    [(submit seconds, wait seconds, event ticks)] a call."""
    out = []
    exclusive = abi.headers().constants["KM_FLAG_EXCLUSIVE"]
    for call in calls:
        started = time.perf_counter()
        handle = run.submit(
            int(call.handler),
            tuple(int(word) for word in call.cores),
            [int(word) for word in call.args],
            exclusive=bool(call.flags & exclusive),
        )
        submitted = time.perf_counter()
        event = run.wait(handle, timeout=timeout)
        if event.kind != "done":
            raise km.KomorebiError(f"host call: {event}")
        out.append((submitted - started, time.perf_counter() - submitted, event.ticks))
    return out


def slot_terms(words):
    """A stats slot: status, entries, polls and each term in microseconds."""
    out = {
        "status": int(words[NN["NN_ST_STATUS"]]),
        "entries": int(words[NN["NN_ST_ENTRIES"]]),
        "polls": int(words[NN["NN_ST_POLLS"]]),
        "stale": int(words[NN["NN_ST_STALE"]]),
    }
    for index, name in enumerate(TERMS):
        at = NN["NN_ST_TERMS"] + 2 * index
        ticks = int(words[at]) | int(words[at + 1]) << 32
        out[name] = ticks / TICKS_PER_MICROSECOND
    return out


def entry_ticks(words, count, first=0):
    """The command ticks of entries first .. first + count - 1 of a stats slot."""
    at = NN["NN_ST_ENTRY_TICKS"] + first
    return words[at : at + count].astype(np.int64)


# The decode layer as calls: what thread/layers.c builds for a layer, a command a step.
STEPS = ("qkv", "attend", "o", "gate_up", "down")
D, S = card.D, card.S
EXCLUSIVE = card.EXCLUSIVE


class Decode(card.Decode):
    """card.Decode with the executor on CLUSTER beside the decode's own thread: one
    program, the hand's thread and the list's in one run."""

    def program(self):
        return executor_thread(super().program())


def nop_call(decode, cores):
    """The cheapest call of the decode program: SPLIT in its operation mode with no
    operation, on cores (first, count): each core enters the handler and returns."""
    words = decode.words()
    args = (0, 0, 0, words[D["DC_WORD_REDUCE"]], S["kSplitOp"], 0, 0, 0, 0, 0)
    return Call(words[D["DC_WORD_SPLIT"]], cores, args, EXCLUSIVE)


def layer_calls(decode, layer, position, rows=1, token=False):
    """Layer `layer`'s five calls (STEPS) for `rows` tokens from `position`, each with
    its chain record (the fused layer; layers.c's step_command, chain_write); `token`:
    qkv's and attend's a9 take the position from the chain's a0 (theirs 0)."""
    words = decode.words()
    t = {name: tensor.address for name, tensor in decode.tensors.items()}
    w = {name: tensor.address for name, tensor in decode.weights[layer].items()}
    reduce = words[D["DC_WORD_REDUCE"]] | rows << D["DC_ROWS_SHIFT"]
    epsilon = words[D["DC_WORD_EPSILON"]]
    marked, sends = RECORD["kChainMarked"], RECORD["kChainSends"]
    from_a0 = RECORD["kChainTokenMask"] << RECORD["kChainTokenShift"]
    if token:
        position = 0

    def product(a, weight, y, plan, operation, other, a3=0, info=marked | sends):
        plan = words[D[plan]]
        operation = words[D[operation]]
        args = (a, w[weight], y, reduce | a3, 0, plan & 0xFF, operation, *other)
        record = (info, 0, 0, 0)
        return Call(
            words[D["DC_WORD_SPLIT"]],
            (0, plan >> 8),
            args,
            EXCLUSIVE,
            0,
            operation & 0xFF,
            record,
        )

    qkv = product(
        t["x"],
        "qkv",
        t["qkv"],
        "DC_WORD_PLAN_QKV",
        "DC_WORD_SCALE",
        (t["squares1"], t["qk"], position | layer << 16),
        D["DC_QK"],
        sends | (from_a0 if token else 0),
    )
    plan = words[D["DC_WORD_PLAN_ATTEND"]]
    attend = Call(
        words[D["DC_WORD_SPLIT"]],
        (0, plan >> 8),
        (
            t["qkv"],
            w["keys"],
            t["a"],
            reduce,
            S["kSplitOp"],
            plan & 0xFF,
            words[D["DC_WORD_ATTEND"]],
            w["values"],
            t["attend"],
            position | words[D["DC_WORD_CACHE_TILES"]] << 16,
        ),
        EXCLUSIVE,
        0,
        D["DC_OVERLAY_ATTEND"],
        (
            marked
            | sends
            | RECORD["kChainNeedsRanges"] << RECORD["kChainNeedsShift"]
            | (from_a0 if token else 0),
            *words[D["DC_WORD_ATTEND_NEEDS"] : D["DC_WORD_ATTEND_NEEDS"] + 3],
        ),
    )
    o = product(
        t["a"],
        "o",
        t["x1"],
        "DC_WORD_PLAN_Q",
        "DC_WORD_ADD",
        (t["x"], t["constants"], t["squares2"]),
    )
    gate_up = product(
        t["x1"],
        "gate_up",
        t["gated"],
        "DC_WORD_PLAN_GATE_UP",
        "DC_WORD_GATE",
        (t["squares2"], t["constants"], epsilon),
        info=marked | sends | words[D["DC_WORD_GATE_TILES"]] << 16,
    )
    down = product(
        t["gated"],
        "down",
        t["x"],
        "DC_WORD_PLAN_DOWN",
        "DC_WORD_ADD",
        (t["x1"], t["constants"], t["squares1"]),
    )
    calls = [qkv, attend, o, gate_up, down]
    return [  # each record's stamp row: its step's
        call._replace(row=row) for row, call in enumerate(calls)
    ]


def token_calls(decode, layers, rows=1):
    """A token's calls through layers 0 .. layers - 1 as one chain's records (layers.c's
    token form): each qkv after the first marked, the last record's stamp row its own;
    the position is the chain's a0."""
    calls = [
        call
        for layer in range(layers)
        for call in layer_calls(decode, layer, 0, rows, True)
    ]
    marked = RECORD["kChainMarked"]
    for at in range(len(STEPS), len(calls), len(STEPS)):
        info, *ranges = calls[at].record
        calls[at] = calls[at]._replace(record=(info | marked, *ranges))
    if layers > 1:
        calls[-1] = calls[-1]._replace(row=D["DC_CHAIN_LAST_ROW"])
    return calls


def decode_entries(
    decode, tokens, layers, position, chain, first=0, rows=1, token=False, ahead=False
):
    """`tokens` steps of `rows` tokens through layers 0 .. layers - 1 from `position`
    as one closed list: a command a call; with `chain` a layer as one chain, with
    `token` a token as one (`ahead`: a record's W read in the record before)."""
    entries = Entries(first)
    mode = card.CHAIN["kChainMode"]
    for step in range(tokens):
        at = position + step * rows
        if token:
            calls = token_calls(decode, layers, rows)
            entries.chain(calls, mode, EXCLUSIVE, at, ahead)
            continue
        for layer in range(layers):
            calls = layer_calls(decode, layer, at, rows)
            if chain:
                entries.chain(calls, mode, EXCLUSIVE, ahead=ahead)
            else:
                for call in calls:
                    entries.command(call)
    return entries
