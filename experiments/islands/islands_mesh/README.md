# Islands of one mesh

Komorebi runs a program on one card. A run over several cards treats each card's chip
as an island: its own grid, its own dispatcher, and links to the other islands. This
experiment asks whether one process can hold both cards that way:

- one 1 x 2 mesh with tt-metal's 1-D fabric on;
- on each card, Komorebi's dispatcher on X280 cluster 0 and Komorebi's persistent
  kernels on the whole grid;
- and a core of one island, inside a handler, writing into the L1 of a core of the
  other island over the link, with the connection open across commands.

Answer, measured on two Blackhole p150a cards joined by four Ethernet links: yes, with
no change to tt-metal or to its X280 runtime.

| Step | Result |
|---|---|
| 1. One fabric mesh, one X280 runtime over it, a dispatcher on each card answering NOP | done: 1.131 and 1.159 µs a NOP command on island 0 and 1 alone, 1.200 and 1.206 µs with both at once |
| 2. The persistent kernels on both grids, example 01 on each | done: MAP 2.197 and 2.182 µs a command alone, 2.235 and 2.222 at once, every tile exact on both |
| 3. A handler opens a fabric connection, writes 64 bytes into the other island's L1, a handler there reads them back; both ways; the connection open across commands | done: 18 slots of 64 bytes exact (8 rounds each way plus the last of 1000 pipelined writes each way), 1.685 and 1.585 µs a SEND command |

## How the islands are opened

The native module's `Islands(count)` sets the 1-D fabric and opens one 1 x count mesh
in slow dispatch; tt-metal starts the fabric routers on both chips in that open. It
then makes one 1 x 1 submesh of each chip. A submesh has its own command queue, and in
slow dispatch a queue waits only for the cores of its own earlier workloads: a launch
on island 1 does not wait for the persistent grid of island 0. So each island's grid
is launched by its own session, one after the other, and both run at once; one
workload over both grids is not needed.

`open_x280()` opens tt-metal's X280 runtime once over the whole mesh. The runtime binds
one mesh a process, and that mesh is the parent, not a submesh. `device(i)` returns
island i as a native `Device` that holds the submesh and the shared runtime. A
`Session(card=chip, device=islands.device(i))` then loads and runs a program on that
island as on a card opened alone: its `Dispatch` uses the shared runtime and only the
X280 tiles of its own chip (boot, clock, cluster 0, the host window, the host buffer).
A card opened alone takes the same code with its own runtime, as before.

## 1 and 2. A dispatcher and example 01 on each island

`islands.py` opens the islands, loads example 01's program on each (120 cores, its
own dispatcher on its own chip's cluster 0, the X280 at 1350 MHz), then times, on
each alone and on both at once (one host thread an island): 5 rounds of 20,000 NOP to
the whole grid and 3 rounds of 2,000 MAP (each core adds one tile). Both runs stop,
and the last MAP round's 120 tiles of each are checked exactly.

2026-11-14, host load 6.6, AICLK 1350 MHz, microseconds a command (submit to event,
pipelined; the median of the rounds):

| Case | island 0 NOP | island 0 MAP | island 1 NOP | island 1 MAP | wrong tiles |
|---|---|---|---|---|---|
| each alone | 1.131 | 2.197 | 1.159 | 2.182 | - |
| both at once, one process | 1.200 | 2.235 | 1.206 | 2.222 | 0 / 0 |

The kernels on each island ended with no error after 212,001 commands. Both at once
costs 4 to 6 % a NOP command here, against under 1 % with one process a card in the
`two_cards` experiment: the two islands' host threads share one process. The MAP
rounds are short (2,000 commands), so their figures include each round's start.

## 3. Across the link from inside a run

`kernels/` is a program of its own: reader and compute do nothing; the writer of one
core of each island, its link core, holds a fabric connection.

- The link core is the row-0 Tensix core nearest the link's Ethernet core (here logical
  (11, 0) on both islands). Its writer arguments are the fabric's Ethernet channel of
  link 0 toward the other island, from `Islands.link(from, to, link)`, and the L1
  addresses of its teardown word and producer cursor. The connection is built from
  them with tt-metal's `WorkerToFabricEdmSender::build_from_args` taking L1 addresses.
  No semaphore of the program is used.
- Every core holds a row of 512 bytes of L1 (an L1 tensor): the connection's two
  words, a packet header, the 64 bytes to send, and the slot the other island writes.
- OPEN opens the connection. SEND fills 64 bytes from a seed and sends one packet,
  one hop, a write into the other island's link core's slot; it reports once the
  packet's writes have left the core. COPY waits (bounded) until its slot holds the
  seed asked for, then copies the slot into a DRAM page that the host reads. CLOSE
  closes the connection and waits for the router's answer. Every wait is bounded by
  the run's wait limit and a bound that runs out gives the command no report, which
  the dispatcher turns into a timeout event.

`link.py`: OPEN on both; 8 rounds of SEND on island 0 then COPY on island 1, and the
other way; 1,000 SEND commands each way in one `submit_many` (pipelined, timed), then a
COPY of the last; CLOSE on both; stop; read back and compare every copied slot.

2026-11-14, AICLK 1350 MHz:

| What | Result |
|---|---|
| events of every OPEN, SEND, COPY and CLOSE | all done |
| slots read back and checked | 18 of 64 bytes, 0 wrong |
| SEND, 1,000 pipelined to one core | 1.685 µs a command island 0 to 1, 1.585 island 1 to 0 |
| kernels at the end | no error on either island |

On each island the connection was opened by one command, carried the 1,008 SEND
commands after it, and was closed by another.

## What the build needs

- **Native calls.** `Islands(count, worker_l1_size)`: the fabric set and the 1 x count
  mesh opened, a submesh a chip. `open_x280()`: one runtime over the mesh. `device(i)`:
  island i for a session. `link(from, to, link)`: links found, the Ethernet channel, the
  Ethernet core. `close()`: drops the runtime (every island's dispatcher closed first),
  closes the mesh and sets the fabric off. `Dispatch` over an island's device takes
  only its chip's X280 tiles.
- **Order of starts.** The fabric is set before the mesh opens; the X280 runtime opens
  after the mesh, once, before any island's dispatcher; then each island loads in turn
  (its grid, then its dispatcher), each on its own queue. A link connection opens in a
  command after both islands are loaded, and the first write across follows it; the
  other island's receiving side is its persistent grid, already running. Closing runs
  the other way: every connection closed by a command, every run stopped, then the
  mesh.
- **What a session over several islands holds.** The mesh (`Islands`) and the one X280
  runtime; per island a session as today (device, dispatcher, run, tensors, the X280
  clock of its chip); per link, the link core of each side, its channel and its L1
  words. Tensors are per island: an island's tensors live on its own submesh.

Not tried here: links other than link 0, several links at once, a packet's own count
at the far core (the experiment's COPY waits on the payload's seed), and the
dispatcher of one island sending commands to the other.

## Running it

Both cards in one job, through the hang watcher, with tt-metal's environment set and
the repository's `src` on the path:

    python islands.py                       # steps 1 and 2
    python islands.py --no-load             # the open and the close alone
    python link.py --rounds 8 --sends 1000  # step 3

Each prints one JSON line and exits nonzero if a check failed.

Card only: both scripts open tt-metal's mesh and X280 runtime themselves, not through a
Komorebi session, so a prebuild or a mock device cannot run them.
