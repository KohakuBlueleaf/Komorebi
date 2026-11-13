# Overlays: kernels loaded at run time

Whether a persistent kernel can act as a loader: a kernel body compiled on the host as
an **overlay**, linked for a fixed slot in L1 against the persistent kernel's symbols,
kept in DRAM, read into the slot by the core's own NoC read when a command names it,
and called. What that costs, what must be flushed, how the three compute cores are
parked while their code is replaced, and what a swap on all 120 cores takes.

## Results

One Blackhole p150a, slow dispatch, AICLK 1350 MHz (driver, every run), host load 7 to
14. Card times are wall-clock cycles of the core; µs at 1350 MHz.

| Question | Answer |
|---|---|
| Build | Overlays build with the persistent kernel's own compile command and flags, linked at the slot with `--just-symbols` of its ELF: entry at the slot's first word, `.bss` zeroed by the entry, no relocations, no undefined symbol. Data symbols resolve (the persistent kernel's globals in the core's local memory, the firmware's). Text symbols do not: the runtime loads kernel text at another address than it is linked for (execute in place), so the builder refuses a call into the persistent kernel's text; its functions are passed as pointers |
| One core, two overlays, 1,000 swaps | Exact: every result, the persistent kernel's call counter, a load at every swap |
| Load time, 4 to 80 KiB | 540 cycles + 21.1 cycles a KiB (fit over the five sizes): 0.45 µs for 4 KiB, 1.64 µs for 80 KiB |
| A command naming the overlay held | 128 to 153 cycles (0.1 µs) besides the body itself |
| The host's ordinary launch of one core | 67.6 ms median from the launch call to the core's return (slow dispatch, no program cache, as a session runs); 21 ms with the program cache in one run, 68 ms in another |
| Stale code after a rewrite | Yes. Without the i-cache invalidated after a load, 10,000 of 20,000 runs of a 576-byte body ran the other body's code; a 3.5 KiB body after it ran 20,000 of 20,000 wrong. With the invalidation (the firmware's register write, one per load): 0 wrong in each |
| Compute cores | Parked in the persistent compute kernel's wait loop, outside every slot; the loader sees them parked when each one's done word equals the go word, writes their three parts, invalidates their i-caches, then sets go. A vector-unit overlay (copy, times or plus a scalar, pack), swapped with a second 1,000 times: exact |
| All 120 cores, one 20 KiB overlay | Started together (within 0.05 µs): the last core done 44 µs after the start with one DRAM copy (one bank; loads 1.9 to 39 µs), 11.5 µs with a copy in each of the 8 banks (loads 0.6 to 6.5 µs). Naming the overlay held: 4.96 µs, which is the body's own run |

## What it means for the design

- **Slot size.** A slot of 96 KiB was used; at 21 cycles a KiB even 80 KiB loads in
  1.6 µs on one core, so the slot is limited by L1, not by load time. Code runs from L1
  at about 0.32 cycles a byte for straight-line code (25,800 cycles for 80 KiB), so a
  body's own run is usually longer than its load.
- **Many cores.** Loads compete for the DRAM bank: keep one copy of an overlay per bank
  (core i reads bank i % 8), which cut the 120-core swap from 44 to 11.5 µs.
- **Flush.** Every load is followed by the i-cache invalidation of the cores that will
  run the slot. Without it a core runs stale lines for as long as they stay cached.
- **Parking.** The compute cores need a wait loop in the persistent image and a done
  word each; nothing else. Their state is shared through the persistent image's
  symbols, so the persistent compute kernel exports the library state its overlays use
  (the runtime's whole-program build otherwise removes or hides it), and a library's
  file-static cache that an overlay holds a copy of is reset at the overlay's entry
  (`builder.py` refuses an overlay with a second copy of any other state).
- **Against a launch.** A load and a call cost microseconds where the host's launch of a
  kernel costs tens of milliseconds.

## Running it

On a card, with tt-metal's environment set (`--card`, default 1):

    python experiments/overlay_probe/probe.py capture     # the runtime's build commands
    python experiments/overlay_probe/probe.py swap        # 1,000 swaps on one core
    python experiments/overlay_probe/probe.py cost        # 4, 10, 20, 40, 80 KiB
    python experiments/overlay_probe/probe.py launch      # the host's launch of one core
    python experiments/overlay_probe/probe.py stale       # without, with invalidation
    python experiments/overlay_probe/probe.py compute     # compute overlay, 1,000 swaps
    python experiments/overlay_probe/probe.py grid --copies 8   # all cores, 8 copies

Each job prints progress lines and one JSON line, also kept in `build/overlay_probe/`.
`capture` builds the kernels once with the runtime's compile commands logged; the
builder reuses those commands with the run's own persistent ELFs.

## Files

- `loader.cpp`: the persistent kernel on data-movement core 0 (commands: run, alternate,
  compute, clock, exit); `compute.cpp`: the persistent compute kernel.
- `overlay.h`, `overlay_entry.h`: the call between them and an overlay, and its entry.
- `chain.cpp`: a data-movement overlay of generated straight-line code (the sizes);
  `scalar.cpp` with `compute_part.h`: the compute overlay.
- `builder.py`: compile, link, checks (slot overlap, entry, relocations, calls into the
  persistent text, duplicated state) and the loaded image; `probe.py`: the jobs.
