# What an overlay costs inside the runtime

[overlay_probe](../overlay_probe/README.md) and [remote_load](../remote_load/README.md)
showed that a core can load a kernel body (an **overlay**) into a fixed L1 **slot** while
a run is live and call it. This page measures four costs that decide how a runtime
carries overlays:

1. **Runtime calls from an overlay.** An overlay cannot call the persistent kernel's
   functions by name, so it reaches the runtime through a table of pointers, or it
   compiles the header-only calls in and reaches only shared state. What does each cost
   in a tight loop, and which calls must be compiled in?
2. **Linking for a slot.** One body linked at one fixed slot, once per slot, or
   position-independent (one image for any slot): bytes, load time, speed, what the
   builder needs, and whether a core can hold two overlays at once.
3. **Parking the compute cores.** Replacing a core's parts needs its compute cores
   parked. What does a command cost when it names the program the core holds (the check
   only), and when all five parts change?
4. **Buffers set up again.** Can a live core switch between two programs' circular
   buffer configurations between commands, and at what cost?

## Results

One Blackhole p150a, slow dispatch, 2026-11-14. AICLK was 1350 MHz in every sample (read
between commands, every job), host load 4.3 to 5.5. Card times are wall-clock cycles of
the core; µs at 1350 MHz. Each row names its job (`costs.py`, below); every number is the
median over cores and repetitions unless the row says otherwise.

| Question | Job | Number |
|---|---|---|
| 1. Reader loop: reserve, read a 2 KiB page from DRAM, wait, push (2,048 blocks, 5 repetitions) | `calls --cores one`, `row`, `grid` | Cycles a block, whole kernel / overlay through the table / overlay with the calls compiled in: one core 616.7 / 676.3 / 624.4; 12 cores 621.2 / 675.4 / 624.4; 120 cores (DRAM-bound) 782.5 / 779.5 / 780.3. Every block's first word checked: 0 wrong |
| 1. Compute loop: wait for the front, copy into DST, times a scalar on the vector unit, pack, pop, push (1,024 tiles, 5 repetitions) | same | Cycles a tile: 305.3 / 320.2 / 300.9 on 1, 12 and 120 cores alike. 16 values of every tile checked: 0 wrong; with wrong expected values all 1,024 (12,288; 122,880) tiles are reported wrong |
| 1. Overlay size | same | Reader 320 B through the table, 832 B compiled in; compute parts (unpack, math, pack) 576 B each through the table, 1,216 / 832 / 896 B compiled in |
| 1. Shared state the compiled-in overlay refers to | same | Reader: `cb_interface`, `noc_reads_num_issued`, `bank_to_dram_offset`, `dram_bank_to_noc_xy`, all the firmware's (fixed addresses). Compute: the firmware's `cb_interface`, `cfg_state_id`, `dest_offset_id`, `__instrn_buffer`, and the compute kernel's own `unp_cfg_context`, `math_sync_tile_dst_index`, `mailbox_base`, `pc_buf_base` |
| 2. Bytes | `slots` | The reader body (compiled-in form): 832 B linked at a fixed slot, 832 B for each further slot; position-independent 832 B once (4 more bytes of code and a 28-byte table of addresses, `.got`, within the same 64-byte padding) |
| 2. Load time, 832 B | `slots --cores one`, `row`, `grid` | Fixed 557 cycles, position-independent 557 (one core); 576 and 576 to 578 (12 cores); 614 and 614 (120 cores, a copy per bank). The one core's 450 cycles for its second-slot copy came from another DRAM bank |
| 2. Cycles a block (256 blocks, 20 repetitions) | same | Fixed 624.3, position-independent 624.2 to 624.5 (one core); 625.0 to 625.6 (12 cores); 807.2 to 808.6 (120 cores). 0 wrong |
| 2. Two overlays held at once | same | The position-independent reader in slot 0 and its table form in a second slot, alternated 1,000 times (200 on 120 cores): 2 loads per core, 0 wrong |
| 3. A command naming the program held (the check) | `park --cores one`, `row`, `grid` | 142 cycles (0.11 µs) from the command seen to the go, 199 after a 60 KiB part; the last of the four parked cores starts 60 to 78 cycles after the go. Same on 12 and 120 cores |
| 3. A command naming another program (five parts of 4, 20, 60 KiB) | same | Command seen to go (park, five loads, five i-caches invalidated): one core 1,061 / 2,364 / 5,656 cycles (0.79 / 1.75 / 4.19 µs); 12 cores, last core 1,650 / 5,880 / 16,789; 120 cores with a copy per bank, median core 5,778 / 26,382 / 77,475, last core 9,421 / 42,945 / 128,007 (7.0 / 31.8 / 94.8 µs). Release as above. Every part's result checked: 0 wrong in 100 switches (one core), 50 (12 and 120) |
| 4. Buffers set up again | `layouts --cores one`, `row`, `grid` | Possible. 918.4 cycles (0.68 µs) a switch, 877 to 926; 10,000 switches on one core (24,000 on 12, 240,000 on 120), 8 tiles through compute between switches, every value of every tile checked: 0 wrong, 0 buffer pointers outside the configuration in force |

## What it means

- **Calls.** Through the table every runtime call is an indirect call: about 12 cycles
  each in the reader (+60 cycles, +9.7% a block, five calls), +15 cycles (+4.9%) a tile
  in compute (eleven calls a tile on each compute core). Compiled in, the overlay runs
  as fast as the whole kernel (+1.3% and −1.4%, code placement). Waiting on DRAM on 120
  cores hides the reader's difference. So every header-only call is compiled into the
  overlay (buffer waits, pushes and pops, NoC reads and barriers, the compute calls);
  the table carries only state. The data-movement side refers to the firmware's state
  only, at addresses fixed by the firmware, not by the kernel build. The compute side
  also refers to four symbols the compute kernel defines (`unp_cfg_context`,
  `math_sync_tile_dst_index`, `mailbox_base`, `pc_buf_base`): the table carries those, or
  the runtime's persistent compute kernel keeps them at fixed places.
- **Linking.** Position-independent code costs nothing measurable here (same bytes,
  load and speed) and needs one copy in DRAM for any slot. The builder needs: `-fpie`;
  an output section for the table of addresses (`.got`) inside the slot; hidden visibility on
  the overlay's own linker-script symbols and entry; and, as its check, the overlay
  linked at two slot addresses with byte-identical images. Without the hidden visibility
  the table holds the slot's addresses and the two images differ, which that check
  catches. A core then holds two overlays in two slots at once.
- **Parking.** Checking the program before every command costs about 0.1 µs plus the
  release of the parked cores (about 75 cycles); a switch costs the five loads, which
  on 120 cores are bound by DRAM (35 MiB for 60 KiB parts, 95 µs for the last core).
- **Buffers.** A live core switches buffer configurations in 0.68 µs when all five
  cores are parked and every page of the old configuration is popped. The sequence
  below worked with the same data format in both configurations; a configuration with
  other formats also needs the compute kernel's unpacker and packer set up again (not
  tried).

### Setting up a program's buffers again

The second configuration lives in L1 of the core (here an L1 buffer written by the
host): four words a circular buffer (address, bytes, pages, page bytes), at the buffer's
index, the format the firmware reads at a launch. With the four other cores parked,
data-movement core 0:

1. waits until each buffer of either configuration is empty: its hardware `received`
   and `acked` counters equal (the packer stores its counter after the core moves on,
   so the core's own done word is not enough);
2. zeroes both counters of every buffer in either configuration (the firmware zeroes
   them only at a launch; the set-up below zeroes each core's own copies);
3. calls `setup_local_cb_read_write_interfaces<true, true, false>(configuration, 0,
   mask)` for itself, then lets the other cores run the same call on their own state:
   data-movement core 1 `<true, true, false>`, unpack `<true, false, false>`, pack
   `<false, true, true>`, math nothing;
4. waits until they are parked again. The next command runs on the new buffers.

Not covered: the runtime's own buffers (its token and trace buffers) are not in these
kernels; the number of buffers stays the same (two), only their places and page counts
change (2 and 2 pages against 6 and 4).

## Running it

On a card with tt-metal's environment set (`--card`, default 1):

    python experiments/overlay_costs/costs.py capture         # the kernels' build commands
    python experiments/overlay_costs/costs.py calls --cores one   # one, row or grid
    python experiments/overlay_costs/costs.py slots --cores one
    python experiments/overlay_costs/costs.py park --cores one   # --reps 50 on row, grid
    python experiments/overlay_costs/costs.py layouts --cores one   # --commands 2 on row, grid

Each job prints progress lines and one JSON line, also kept in `build/overlay_costs/`.
The grid jobs keep a copy of every overlay in each of the 8 DRAM banks (core i reads
copy i % 8).

## Files

- `host.cpp`: the persistent kernel on data-movement core 0 (commands: read, compute,
  switch, layouts, clock, exit); `drain.cpp` on data-movement core 1 and `compute.cpp`
  on the compute cores, parked between commands.
- `costs.h`: the control words, the overlay's call, the tables of runtime calls;
  `api.h`: the runtime calls and the two loops, header-only.
- `reader.cpp`, `compute_body.cpp`: the loops as overlays (table or compiled in);
  `part.cpp`: generated straight-line parts of a given size; `entry.h`, `mover_part.h`,
  `compute_part.h`: the entry and what the runtime's build puts before a kernel.
- `costs.py`: the jobs; it builds the overlays with
  [overlay_probe](../overlay_probe/README.md)'s builder and its own linker script.
