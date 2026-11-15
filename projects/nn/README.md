# nn: a list of kernel calls and one X280 thread that runs it

The first piece of a small network runtime on Komorebi: the host describes work as a
list of kernel calls in DRAM, and one generic X280 thread walks the list and submits
each call, so the host is not on the path between two kernels. The question is what a
call costs this way against the host submitting every call itself, and whether a real
model runs through it as fast as the hand-written thread that knows the model.

The calls here are the decode project's own kernels (`../decode`): its program, its
plans and its words, called and not copied. The thread knows no model.

## The entry

A list is entries of 128 bytes (two 64-byte lines) in a region of the thread's DRAM
bank, each one kernel call (`nn.h`). The first line is the split's chain record
(`../dense/kernels/split/chain.h`) as it is, so a run of entries is passed to the chain
unchanged; the second line is the thread's.

| Words | Line | What |
|---|---|---|
| 0 to 9 | record | the command's a0 to a9 |
| 10 | record | its cores: first \| count << 16 |
| 11 to 14 | record | the chain's info word and three range words (a chain's record only) |
| 15 | record | the next record of the same chain, its bank address; 0: the chain's last |
| 16 | thread | the header: handler \| flags << 8 \| program << 24 |
| 17 | thread | the kind: COMMAND, CHAIN, END, NONE |
| 18 | thread | the list's next entry, its bank address; 0: the end |
| 19, 20 | thread | a chain: its command's a4 (the kernel's chain mode) and its cores |
| 21 | thread | the operation overlay the call names (kept for the host; the kernel reads its own word, SPLIT's a6) |
| 22, 23 | thread | the sequence (an open list's number; 0 in a closed list) and the check |

The chain's next (word 15) and the list's next (word 18) are two words because a chain
ends where the list goes on. The check is the sum of word i x (2i + 1) over the entry,
words 18 and 23 left out, modulo 2^32.

## The thread

`thread/executor.c`, on work cluster 2 beside the decode's own thread on cluster 1 (one
command ring each; the same program, so both run in one session). For each entry it
builds the command from the words (a COMMAND), or one command in the chain's mode whose
block is the entry's first line (a CHAIN, the records after it read by the cores), then
submits it and waits for its event. It is driven three ways:

- **Closed list** (`run`: a0 the first entry, a1 the times, a2 a stats slot): the host
  writes the whole list once with a DRAM barrier, the thread drops its cached lines of
  the region once and walks the list from the first entry to a next of 0, N times.
- **Open list** (`drain`: a0 an anchor entry, a1 a stats slot, a2 the ticks it waits
  idle at most, a3 the open list's number): the host appends entries while the thread
  runs; the thread polls the next word of the last entry it ran (the line dropped from
  its cache, read again), then reads the linked entry, and a chain's records, until their
  sequence and check hold, runs it, and polls its next. An END entry ends the command, so
  the host waits only when it asks for a result.
- **The host's calls**, for comparison: the host submits each call and waits for it
  (`run.submit`, `run.wait`), as every example does.

Each run leaves a stats slot (its status, entries, polls, stale reads, each term's
ticks, and each entry's command ticks by its index), read after `run.stop()`.

## The host calls (the interface)

`nn.py`:

- `Call(handler, cores, args, flags, program, overlay, record)`: one kernel call; `record`
  the chain's four words when the call can be a chain's record.
- `Entries(first)`: a list from entry `first` of the region. `command(call)` appends a call
  as one command; `chain(calls, mode, flags)` a run of calls as one chain command.
  `words(base, sequence)` gives the entries' words with the links and the checks.
- `executor_thread(program, cluster=2)` adds the thread to any program.
- `Executor(session, run)`: `write(entries)` (the host's DRAM write and a barrier),
  `closed(entries, times)` (one `run` command, waited for: the thread's ticks and the
  slot), `open(at, idle, fence)` (an `OpenList`), `stats()` after the stop.
- `OpenList`: `append(call)`, `append(chain=calls, mode=, flags=)`, `sync()` (an END,
  then the event).
- `drive(run, calls)`: the host's own calls.
- `layer_calls(decode, layer, position, rows)`: the decode layer's five calls (qkv,
  attend, o, gate_up, down) with their chain records, from `Decode`'s words and tensors,
  as the decode thread builds them; `decode_entries(...)` a closed list of T tokens of
  L layers; `nop_call(decode, cores)` the decode program's cheapest call. `Decode` is the
  decode project's with the executor added.

## Run

```
python projects/nn/bench.py --check --costs --decode --json out.json
python projects/nn/bench.py --layers 1 --check --costs
```

`--check` runs one token through layer 0 at T 4096 six ways from the same x (the decode
thread a command a step and as a chain; the list closed both ways; the open list; the
host's calls) and compares qkv, a, x1, gated, x and the cache's new row with the first,
bit for bit. `--costs` times 200 cheapest calls a list on one core and on all 120,
`--decode` 10 tokens of 28 layers at T 4096, 1024 and 256 the three ways.

## The check

Every tensor and the cache row of every way equal the decode thread's commands bit for
bit, at T 4096 (before and after the removal below).

## The decode layer through the list

Qwen3-1.7B, 28 layers, random weights as the decode bench's, 10 tokens a run; us a
layer, from the thread's clock (the host's for the host's calls); the hand's columns are
the decode project's `results/qwen3-1.7b-chain-x10.json`. Run 2026-11-15 18:18 on one
p150a, AICLK 1,350 MHz (one sample of 1,343), host load 1.2, `results/qwen3-1.7b-list-x10.json`.

| | T 4096 | T 1024 | T 256 |
|---|---:|---:|---:|
| (a) closed list, a command a step | 410.94 | 373.50 | 364.85 |
| hand, a command a step | 410.26 | 373.00 | 364.22 |
| (a) its commands' events | 399.19 | 361.80 | 353.09 |
| hand's events | 399.02 | 361.78 | 353.00 |
| (a) the thread (5 entries) | 11.75 | 11.70 | 11.76 |
| (a') closed list, the layer as a chain | 347.28 | 318.22 | 307.92 |
| hand, the chain | 356.57 | 326.41 | 317.61 |
| (a') the chain command | 344.74 | 315.69 | 305.38 |
| hand's chain command | 345.42 | 315.27 | 306.47 |
| (a') the thread (1 entry) | 2.54 | 2.53 | 2.54 |
| hand's thread, the chain | 11.15 | 11.14 | 11.14 |
| (c) the host's calls | 1,168.05 | 1,148.62 | 1,115.85 |
| (c) its host submit and wait less the events | 767.5 | 785.0 | 761.3 |

The hand's run-to-run spread over its recorded runs is 0.1 to 0.3 us a layer for the
chain and 0.6 to 1.6 for a command a step. (a) is the hand within it: its events within
0.2, its thread's 2.35 us a command (fetch 0.55, submit 0.51, wake 1.01, the rest 0.28)
against the hand's 2.26. (a') is 8.2 to 9.7 us under the hand: the chain command takes
the same time (within 1.1), and the thread's share is 2.5 us where the hand's is 11.1,
most of which is the hand writing the five records itself before each layer; here the
host wrote them once. (c) runs the same events and spends 153 to 157 us a call on the
host's submit and wait.

## What an entry costs

200 calls of the decode program's cheapest kernel, SPLIT in its operation mode with no
operation (each core of the call enters the handler and returns), on one core / on all
120. us a call, same run as above.

| Term | (a) closed | (b) open, fenced | (b) open, checked | (c) host |
|---|---:|---:|---:|---:|
| host: building the entry | | 14.0 / 13.9 | 12.4 / 12.4 | |
| host: the entry's write | | 4.1 / 4.1 | 4.0 / 4.1 | |
| host: the DRAM barrier | | 51.7 / 51.9 | | |
| host: the link's write | | 4.2 / 4.3 | 4.1 / 3.8 | |
| host: submit | | | | 11.8 / 11.7 |
| host: wait, less the command | | | | 126.7 / 126.6 |
| thread: polls for the link (idle) | | 71.8 / 71.5 | 18.0 / 17.5 | |
| thread: reading the entry until new | | 1.3 / 1.3 | 1.3 / 1.3 | |
| thread: fetch and build | 0.31 / 0.31 | 0.27 / 0.27 | 0.27 / 0.27 | |
| thread: submit | 0.51 / 0.51 | 0.52 / 0.52 | 0.51 / 0.51 | |
| the command (the dispatcher's submit to completion) | 1.89 / 2.25 | 1.94 / 2.30 | 1.89 / 2.27 | 1.86 / 2.26 |
| thread: the wake (the wait less the command) | 1.01 / 1.05 | 0.98 / 1.00 | 0.97 / 1.02 | |
| thread: the rest | 0.27 / 0.28 | 0.32 / 0.32 | 0.32 / 0.32 | |
| a call | 3.99 / 4.40 | 76.7 / 76.8 | 23.1 / 23.0 | 140.6 / 140.9 |

(a) is the replays of a written list (5 x 200); its first run fetches the dropped lines,
0.54 us an entry, and takes 4.36 / 4.52. In (b) the host appends as fast as it can and
the thread follows: a call takes the host's append, and the thread waits in polls of
about 1.03 us each (a line dropped and read from DRAM). Not one entry was read before it
was new. From the host's link write to the kernel's start, summed from the terms (the
host's and the thread's clocks are not one, so this is not measured end to end): the
write 4.1, half a poll 0.5, the entry's read 1.3, fetch and submit 0.8, and the
dispatcher's start, a part of the wake's 1.0: 6.7 to 7.7 us.

A sync and a read back on the host after the last entry: the open list's END and its
event 131 to 190 us; one tile of x read 213 to 239 us, x's 64 tiles 360 to 409 us. A
closed list's write: 200 entries in 0.22 to 0.30 ms.

## What was removed and what it gave

The largest term on the package's own paths was the open list's DRAM barrier, 51.7 us
of an append of 76.7 (the barrier waits on all eight DRAM channels): it kept the thread
from reading an entry before it was in DRAM. It is gone: each entry carries its open
list's number and a check, and the thread, once it sees a link, reads the entry (and a
chain's records) again until both hold, so the host writes the entry and the link with
no barrier between them. An append went from 76.7 to 23.1 us, a sync from 190 to 132 us;
the thread's read of a linked entry until it is new, 1.3 us, is the same in both forms.
The largest term of all is still the host's own wait in (c), 127 us
a call above its command; the lists exist to keep it off the path between kernels. In
the open list the next largest is now the host's building of an entry in Python (12.4
us), then its two writes (4.0 and 4.1 us).

## What the next packages take from this

- The entry (the table above, `nn.h`): a call is ten words, its cores, its header and an
  overlay note; a chainable call also carries its chain record words. An op table maps an
  operation to a `Call` (or a run of them for a chain) from its tensors and shapes, as
  `layer_calls` does for the decode layer.
- The host calls: `Entries` for a closed list built once and run N times (a compiled
  graph), `OpenList.append` / `sync` for eager calls, `drive` to compare.
- An open list's entries take a slot each of 2,048 in the region, in order; a new open
  list starts at a new anchor. A closed list is written whole before its run.
- Not here: placing a program that is not on a call's cores (the entry names its program
  in the header; the decode program is one program), several calls in flight, a ring
  that reuses the region's slots, a value read on the host without stopping the list.
