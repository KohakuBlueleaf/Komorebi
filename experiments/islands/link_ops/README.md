# Writes into another rank's L1 and adds to its word

Two calls a kernel makes with a flat rank, wherever that rank runs
(`komorebi/device/km_far.hpp`):

- `km::l1_write(core, command, rank, address, from, bytes, port)`: `bytes` from this
  core's L1 at `from` into the rank's L1 at `address` (the same offset in 16 bytes);
  with `port` 0 or more, 1 is added to that port of the rank after the bytes landed.
- `km::word_add(core, command, rank, address, value)`: `value` added to the 4-byte word
  at `address` of the rank's L1.

Each is done when it returns, so what one rank sends to one rank arrives in the order
sent. A rank of the caller's own island is reached by the NoC directly. A rank of
another island is not reached yet: the call returns false and the command gives up in
`far` (the gave-up word's top byte is the rank's island).

The question: are the calls exact and in order, and what does one cost, to a rank of
the same island and to ranks one and two links away.

`ops.py` runs, for each case `S:T`, `--calls` calls from island S's first rank to
island T's last rank: each call writes `--bytes` into slot `call % --slots` of a ring in
the target's L1, counted at `--port`, and adds 1 to its counter word. The target's
ring, counter and port count are then read back and compared with what the calls
should have left.

    python ops.py --card 0 --islands 3 --shape line --cases 0:0,0:1,0:2
    python ops.py --cards 0,1 --cases 0:0,0:1

## Results

None yet.
