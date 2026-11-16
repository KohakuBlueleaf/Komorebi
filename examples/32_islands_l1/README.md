# 32 islands L1

A tensor held in the cores' L1, moved across islands. Each rank holds one float32
tile of the tensor in its L1, at one address on every core; one command moves every
rank's tile to the rank `shift` after it (mod the group), whether that rank is on its
own island or another, and each rank writes the tile it got to the output.

The move is an example function in the kernel (`kernels/reader.cpp`), not a runtime
call: two calls of the runtime make it.

```cpp
bool move_l1(km::Reader& reader, const km_command_t& command, uint32_t from_rank,
             uint32_t to_rank, uint32_t from, uint32_t to, uint32_t bytes,
             uint32_t port) {
    const uint32_t me = reader.me();
    if (me == from_rank && !km::write_l1(reader, command, to_rank, to,
                                         reinterpret_cast<const void*>(from), bytes,
                                         static_cast<int>(port))) {
        return false;
    }
    return me != to_rank || km::port_wait(reader, command, port, 1);
}
```

The sender's `km::write_l1` writes the bytes into the receiver's L1 (on its island by
the NoC, on another through the links) and then counts 1 at its port; it returns once
the write has landed. The receiver waits for that count. Every rank calls it twice, as
the sender of its own tile, then as the receiver of the one before it, so the tile it
sends from (`held`) and the one it gets (`landing`) are two buffers: a tile that lands
before its rank has sent its own never overwrites it.

```
python l1.py                          # cards 0 and 1, shift 1 and half the ranks
python l1.py --islands 2 --shifts 1 0 # card 0 split in 2
```

`--shifts 0` is half the ranks: every rank's tile crosses to the other island. With
shift 1 only the last rank of each island crosses.

## Results

2026-11-16, Blackhole p150a, AICLK 1350 MHz, the median of 10 commands on the card,
every rank's tile exact:

| Cards | Ranks | Shift 1 | Shift: half the ranks |
|---|---:|---:|---:|
| 0 and 1 (one link) | 236 | 10.8 µs | 578.6 µs (118 tiles each way) |
| 0 split in 2 | 118 | 8.5 µs | 231.3 µs (59 tiles each way) |

A crossing write is a confirmed one, and an island's ranks' writes toward the other
share its link cores: 118 tiles of 4 KiB each way in 578.6 µs is about 1.7 GB/s each
way.

## Files

| File | What it is |
|---|---|
| `kernels/l1.h` | the handler, its words, the two buffers |
| `kernels/reader.cpp` | `move_l1` and the handler |
| `kernels/writer.cpp`, `kernels/compute.cpp` | nothing to do |
| `l1.py` | the tiles, the moves, the check; one JSON line |
