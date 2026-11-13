# Reading a tensor back from DRAM

How long the host takes to read one interleaved bfloat16 tile tensor back: bank by bank
under one DRAM barrier (`Session.read_bf16`: one read per bank, then the pages put in
order and untiled on the host) against the whole-buffer read (`Tensor.read`: tt-metal's
read of the mesh buffer, which waits for the launched cores). The two ways are read in
turn, three times each, and must give the same array.

## Result

Measured when the other way was `ttnn.to_torch`, before the host left ttnn. On one
Blackhole p150a, 2,048 tiles (4.19 MB), host load about 8: bank by bank 2.56,
2.24 and 1.96 s, `ttnn.to_torch` 3.28, 2.80 and 3.03 s; the arrays equal. The numbers
move with the host's load, and the first read in a process pays a warm-up (another run
gave 10.7 s for its first bank-by-bank read, then 3.7 s through ttnn).

## Running it

On a card, with tt-metal's environment set and the repository's `src` on the path:

    python experiments/dram_read/read_rate.py --tiles 64 --reps 1   # small, a second
    python experiments/dram_read/read_rate.py                       # 2,048 tiles, 3 x 2

It prints one JSON line: the seconds of every read each way (`banks_s`, `whole_s`), whether they agree, and
the host's load.
