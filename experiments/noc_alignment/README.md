# Small NoC writes and the low four address bits

A NoC write shorter than 16 bytes from one Tensix core's L1 to another's is only correct
when the source and destination addresses have the same low four bits. The NoC
interface's own header (`noc.h` of the Blackhole tree) says so and adds that "the one
from destination address is assumed" when they differ. This experiment shows what that
means on a card: which bytes arrive, and whether the sender's acknowledgement returns.

## Result

When the low four bits differ, the write lands at the destination address, and the
bytes it carries are the ones at the destination's offset within the source's 16-byte
line, not the bytes at the source address. The write is silently wrong: no error, and a
non-posted write is acknowledged as usual, so a barrier after it returns.

Measured on core (0, 0) writing 4 bytes into core (1, 0) of a Blackhole p150a (one card,
slow dispatch, no load from other jobs on that card). The source line holds
S[i] = 0xA0A00000 | i in its 16 words; "word" counts 4-byte words.

| source word | destination word | write | arrived in the target slot | acknowledged |
|---|---|---|---|---|
| 0 | 0 | posted | word 0 = S[0] | (posted) |
| 1 | 1 | posted | word 1 = S[1] | (posted) |
| 3 | 3 | posted | word 3 = S[3] | (posted) |
| 1 | 1 | non-posted | word 1 = S[1] | yes |
| 1 | 0 | posted | word 0 = S[0] | (posted) |
| 2 | 1 | posted | word 1 = S[1] | (posted) |
| 3 | 0 | posted | word 0 = S[0] | (posted) |
| 0 | 3 | posted | word 3 = S[3] | (posted) |
| 1 | 0 | non-posted | word 0 = S[0] | yes |

Every other word of each 64-byte target slot stayed zero.

Rule that follows: a small write goes from a 16-byte slot to a 16-byte slot (or, more
generally, source and destination share their low four address bits). The runtime's
small writes all go from a 16-byte slot of the sender's control block to a 16-byte slot
of the receiver.

## What runs

| part | file |
|---|---|
| Kernel on core (0, 0), RISCV_0: fills the source line, submits one 4-byte write per test, then waits 20,000 cycles (posted) or polls the acknowledgement count for at most 1,000,000 cycles (non-posted) | `probe.cpp` |
| Driver: an int32 tensor sharded over cores (0, 0) and (1, 0), 128 words each, zero filled; one launch; reads both shards back | `probe.py` |

Core (1, 0) runs no kernel, so nothing on the card waits for the writes to arrive, and
every wait in the kernel is bounded. The tests come in two sets, each one launch: the
aligned controls first (`--set controls`), then the mismatched writes
(`--set mismatched`).

## Running it

    TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=20 \
        python probe.py --set controls
    TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=20 \
        python probe.py --set mismatched

Each run prints progress lines and one JSON line: per test the slot, the source and
destination words, the words that arrived (as S[i]) and `done` (`yes` for a posted
write, `acked` or `no ack` for a non-posted one). It needs the `komorebi` package.
