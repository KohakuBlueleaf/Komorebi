# 11 movement

Both data-movement CPUs of each core carry one stream ([Movement](../../docs/reference/movement.md)).
The program has no reader and no writer of its own: one source, `movement.cpp`, is
built for both CPUs, the movers. Mover A (RISCV_0) issues on NoC 0, mover B (RISCV_1) on
NoC 1. Each core reads its tiles in blocks; a pattern given on the command line says
which mover reads block k (`AAAB`: mover A three of every four blocks, mover B the
fourth), each mover reads into its own half of a buffer pair, and compute takes the
blocks in order, each from the half its owner filled.

| Handler | Arguments | What a core does |
|---|---|---|
| `READ` | a0 `input_address`, a2 `tiles`, a3 `tiles_per_block`, a4 pattern word, a5 `repetitions` | the movers read the core's `tiles` pages (from page rank x `tiles`) a5 times through the input pair; compute takes each block and drops it |
| `COPY` | as `READ`, with a1 `output_address`, once | compute copies each block from the input pair into the same half of the output pair, and the movers write it to the same pages of the output |

The host checks every word: for each pattern a `COPY` of a block and a half per core (a
ragged last block) goes through both streams, and the output read back must equal the
input. Then it times `READ` from the card's clock, by the difference of R and 4R
repetitions (best of three), and prints each pattern's rate beside pattern `A`, mover A
alone.

## The program, from the host

```python
import komorebi as km

program = km.Program(
    "movement",
    kernels="kernels/",  # movement.cpp for both movers, compute.cpp
    buffers=[
        km.BufferPair(0, 1, "bfloat16", 2048, pages=64),  # input: mover A's, mover B's
        km.BufferPair(16, 17, "bfloat16", 2048, pages=32),  # output
    ],
    compile_args={"movement": source.tensor_args() + target.tensor_args()},
    handlers="kernels/movement.h",  # kHandlerRead = 1, kHandlerCopy = 2
)
pattern = km.Pattern("AAAB")
run.submit("read", group=(0, cores), args=[address, 0, 256, 16, pattern.word, 1])
```

```c++
// movement.cpp, the same source on both movers
using Input = km::Tensor<0>;  // the input's tensor_args(), then the output's
km::Movement movement;
movement.run([&](const km_command_t& command) {
    const km::Stream read(km::kMoveRead, Input(command.a0), rank * tiles, tiles,
                          tiles_per_block, repetitions, kInput,
                          km::Pattern::of(command.a4), kAskPort);
    return movement.move(read);  // this mover's blocks of the stream
});

// compute.cpp: block k from the half its owner filled
const uint32_t half = km::block_half(kInput, pattern, block);
```

## Files

| File | What it is |
|---|---|
| `kernels/movement.cpp` | the data-movement kernel of both movers: a `km::Movement` that moves the read stream (and, for `COPY`, the write stream) |
| `kernels/compute.cpp` | takes the blocks in order; `COPY` copies each through the vector unit's registers |
| `kernels/movement.h` | handlers, the buffer indices of the two pairs and the command arguments |
| `movement.py` | checks the copies, times the reads, prints one line a pattern and one JSON line |
| `islands.py` | the same kernels over two cards: island 0's cores copy and read an input island 1 holds alone, or (`--write-across`) copy into an output island 1 holds ([Islands](../../docs/reference/islands.md#a-kernels-page-reads-and-writes)) |

## What it needs

A Komorebi install and a card prepared as in [Running on a card](../../docs/install.md#running-on-a-card).

```
python movement.py                       # pattern AB beside A, every core of the grid
python movement.py --pattern AAAB B --cores 8 --tiles 256 --block 32
python movement.py --pattern A B AB AAAB --cores 4 --tiles 48 --block 8 --check-only
```

`--cores 0` (the default) is every core of the grid. The script exits non-zero if a copy
differs from its input.

## Numbers

A Blackhole p150a, AICLK 1,350 MHz, host load 4.7, the whole grid (120 cores), 256 tiles
of 2 KB a core in blocks of 16, every copy exact (`python movement.py --pattern AB AAAB
B`):

| Pattern | GB/s | Beside `A` |
|---|---|---|
| `A` | 427.8 | 1.00 |
| `AB` | 485.9 | 1.14 |
| `AAAB` | 502.6 | 1.17 |
| `B` | 247.4 | 0.58 |

On the full grid NoC 0 reads DRAM faster than NoC 1, so the best share gives mover B one
block in four. On fewer cores the routes differ and so does the best pattern; the
[reference](../../docs/reference/movement.md#choosing-a-pattern) has the measured table.

Over two p150a cards, island 0's first 4 cores read 48 tiles each in blocks of 8 from an
input that island 1 holds alone; the kernels are the same, the pages cross the link
(`python islands.py`, one page link): the copy is exact, COPY takes 316.4 us and READ
275.8 us, 1.43 GB/s. With `--write-across` the input is island 0's and the output
island 1's, so every page of the copy is put across: exact, COPY 313.4 us. The kernels
take 87,360 bytes of the two-card session's kernel buffer (112 KiB).
