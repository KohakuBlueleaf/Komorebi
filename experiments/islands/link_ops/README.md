# Writes into another rank's L1 and adds to its word

Two calls a kernel makes with a flat rank, wherever that rank runs
(`komorebi/device/links/km_far.hpp`):

- `km::write_l1(core, command, rank, address, from, bytes, port)`: `bytes` (whole 4s)
  from this core's L1 at `from` into the rank's L1 at `address` (both 16-aligned, on
  any island: a guarded build refuses others by name); with `port` 0 or more, 1 is
  added to that port of the rank after the bytes landed.
- `km::add_word(core, command, rank, address, value)`: `value` added to the 4-byte word
  at `address` of the rank's L1; `km::add_word(..., km::posted)` returns once sent.

A confirmed call is done when it returns. Posted adds from one rank land in the order
sent; a later confirmed call or `km::put_wait` waits for them. A rank of the caller's
own island is reached by the NoC directly, a rank of another island by the link cores,
any number of links away. A call that cannot be sent gives up in `far` (the gave-up
word's top byte is the rank's island).

The question: are the calls exact and in order, and what does one cost, to a rank of
the same island and to ranks one and two links away.

`ops.py` runs, for each case `S:T`, `--calls` calls from island S's first rank to
island T's last rank: each call writes `--bytes` into slot `call % --slots` of a ring in
the target's L1, counted at `--port`, and adds 1 to its counter word. The target's
ring, counter and port count are then read back and compared with what the calls
should have left.

    python ops.py --card 0 --islands 3 --cases 0:0,0:1,0:2
    python ops.py --cards 0,1 --cases 0:0,0:1

## Results

Commit aec998ab, every case exact (ring, counter, port count), µs a call:

| where | case | hops | 16 B write + add | posted add | 1,044 B write |
|---|---|---|---|---|---|
| card 0 split, line of 3 | 0:0 | 0 | 1.07 | 0.12 | |
| | 0:1 | 1 | 8.58 | 1.54 | 9.89 |
| | 0:2 | 2 | 18.84 | 4.04 | 20.15 |
| cards 0,1 | 0:0 | 0 | 1.09 | | |
| | 0:1 | 1 | 11.81 | 2.16 | 13.10 |
| | 1:0 | 1 | 11.79 | | |

10^5 calls a case (the 1,044-byte writes and the split card's 2:0: 1,000; the split
card's 0:0 and posted 0:1: at bba122ea).
