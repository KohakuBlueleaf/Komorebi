# The collectives' account

Where does the time of `km::all_gather` and `km::all_reduce` go, and how does it grow
with the tensor? `account.py` runs the two plain calls from a kernel (GATHER and
REDUCE, one call each in `kernels/reader.cpp`) and from the host (`run.all_gather`,
`run.all_reduce`) on pages of a float32 copy tensor, checks every island's copy, and
prints one JSON line a size, call and form: the call's time on the card (submit to
done, its slowest island), the median of `--rounds` calls.

```
python account.py --sizes 4096,16384 --calls gather,reduce --forms kernel --rounds 3
python account.py --calls nop,gather --sizes 2     # and a NOP command beside them
```

Copies of up to 256 pages are read back whole; larger ones by sample (the first,
middle and last 8 pages of every island's share, on every island). `--stamps` builds
the calls with their step stamps (`KM_COLL_STAMPS`: each rank's wall clock at each
step, written at the call's return to its page of a copy tensor) and adds each
island's steps, from its first rank's entry, to the row. A NoC write reads L1 only, and
a kernel's statics are the core's own local memory: so the service overlay keeps the
stamps in its slot, and a kernel gives the calls 128 bytes of its L1
(`km::coll_stamps_in`, here the buffer `kCbStamps`). Both forms' stamps were run
(the account in `docs/reference/islands.md`). `--store` stores the kernel form's
command by the wait (one native call for every island), or island by island from
Python (`old`); `--inspect` reads every rank's state once after the load
(`run.inspect()`), each island's in its own ranks.

## Results

Two Blackhole p150a cards joined by four links, 2026-11-15, AICLK 1350 MHz on both,
host load 4.6, from a kernel, the median of 3 calls, every sample exact:

| Call | Pages (float32 tiles) | On the card |
|---|---|---|
| all-gather | 4,096 (16 MiB) | 132.4 µs |
| all-gather | 16,384 (64 MiB) | 381.7 µs |
| all-reduce | 4,096 (16 MiB) | 1,488.3 µs |
| all-reduce | 16,384 (64 MiB) | 5,556.8 µs |

From 16 to 64 MiB the gather moves 25.2 MB more each way in 249.3 µs (101 GB/s each
way, the links' rate); the reduce adds 80.8 µs a MB of tensor (12.4 GB/s), most of it
the adds. The reference page `docs/reference/islands.md` ("Collectives across
islands") has the account.
