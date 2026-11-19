# Links between Wormhole chips

Komorebi's island links run on tt-metal's 1-D fabric. The routers run on the
Ethernet cores. A link core is a Tensix core that holds a connection to a router and
sends packets through `komorebi/device/links/km_transport.hpp`. This experiment asks
whether that transport, unchanged, carries what an island needs between the Wormhole
chips of a Galaxy: a write, a stream of pages, and a command's line with its answer.
It also asks what this costs to a neighbour and to chips two and three links away.

The ends are this folder's kernels. The runtime's link kernels (`kernels/link/*.cpp`)
do not build on Wormhole yet: their second data-movement CPU is over its 16 KiB of
code (below).

## The program

`host/link_wormhole.cpp` opens a 1 x N mesh with the fabric (`FABRIC_1D`, slow
dispatch). It puts `kernels/end.cpp` on one Tensix core of chip `--from` (the near
end) and on one core of chip `--to` (the far end), and launches both as one workload.
Each end connects to the router of link `--link` toward the other with
`km::LinkSender::connect`, using the channel tt-metal names for that link. Its headers
are routed `|to - from|` chips on. The two ends share one L1 layout (`kernels/wl.h`),
so an end names its partner's places by its own addresses.

The run has three phases, in this order:

| Phase | What crosses |
|---|---|
| ping | the near end sends 64 bytes and an add to the far end's count in one packet (`link_counted_write`); the far end checks it and sends the same back; 1,000 rounds |
| line | the near end sends a 128-byte line (`link_write`), then an add to the far end's line count as a second packet (`link_increment`); the far end checks the line and answers with one word written into the near end's DONE word (`link_inline`); 1,000 rounds |
| pages | the near end streams `--pages` pages of `--page` bytes into `--window` slots on the far end; every `--batch`th page is a counted write; the far end checks each batch and adds the slots it freed to the near end's credit word |

Every wait has a limit (`--limit-s`, 5 s). Both ends check every word of every ping
and line. The receiver of the pages checks the first and last word of each page, or
every word with `--check 1`. The host reads both ends' status words and prints one
JSON line. Round-trip times are from the near end's wall clock. The page rate is the
receiver's: the bytes after the first counted batch, over the time from that batch's
count to the end. Ticks are converted at 985.5 MHz, Wormhole's measured wall clock.

## Results

Galaxy Wormhole chips 16 to 19 form a line: four links between each pair of
neighbours. tt-metal v0.81.1, firmware 19.15.0, slow dispatch, one core on each end,
logical core (0, 0) on both chips. Every row below was exact at both ends, and ran
once. Round trips are the minimum and the mean of 1,000.

| Path | Ping round trip (µs) | Line and DONE (µs) | Pages (GB/s) |
|---|---|---|---|
| chip 16 to 17, link 0 | 3.58 / 3.72 | 3.41 / 3.47 | 8.16 (4,096 B), 8.66 (4,352 B), 10.19 (7,616 B) |
| chip 17 to 16 | 3.58 / 3.72 | 3.41 / 3.54 | 8.15 |
| chip 16 to 17, link 3 | 3.91 / 4.05 | 3.63 / 3.75 | 8.15 |
| chip 16 to 18 through 17 (two links) | 4.94 / 5.20 | 4.80 / 5.06 | 8.16 (4,096 B), 9.60 (7,616 B) |
| chip 16 to 19 (three links) | 6.43 / 6.71 | 6.28 / 6.49 | 8.16 |

- Each extra link adds about 1.4 µs to a round trip, about 0.7 µs each way.
- The page rate is the same at every distance. Up to 4,352 bytes it is about 0.50 µs
  a page: this sender waits for its writes to leave before it reuses its one page
  header, so the rate is a count of packets.
- With every word of 4,096 pages checked, the receiver's check is what sets the rate
  (0.19 GB/s). All words were exact.
- A raw sender with no such wait ([link_rate](../link_rate/README.md) run on chips
  16 and 17, 4,352-byte payloads, one count a batch of 64) carries 11.61 GB/s on one
  link, from a core four or five NoC hops from the Ethernet tile. Four links one way
  carry 46.15 GB/s, and both ways at once 40.90 and 42.31. A fused write and
  increment with every packet carries 6.46 GB/s. On Blackhole the same runs give 27 to
  29 GB/s a link and 107 to 111 over four. One Wormhole link carries about 11.6 GB/s.

## What Wormhole asks of a link

- The routers take 4,352-byte payloads unless the open tells them more
  (`FabricRouterConfig.max_packet_payload_size_bytes`, `--router-payload`). Wormhole's
  largest is 7,616 bytes (`chips.py`'s `fabric_payload_bytes`), against 15,232 on
  Blackhole. The island session's default packet of 8,192 bytes is over it.
- A 1 x 3 mesh does not open on Wormhole at this tt-metal version. With three chips
  visible, auto-discovery falls back to 2 x 1 ("requires 3 devices, but only 2 devices
  are available in the system mesh MeshShape([2, 1])"). A 1 x 3 mesh graph descriptor
  is refused too ("For WORMHOLE_B0 architecture, if both mesh dimensions are odd, they
  must both be 1"). A line of three chips is a 1 x 4 mesh with one chip unused, as the
  two-link row above.
- No link packet may target a DRAM bank of the far chip
  ([hazards](../../../docs/reference/hazards.md#a-link-packet-for-a-far-wormhole-chips-dram)).
- The runtime's link kernels and a rank's writer in an island build are over the
  second data-movement CPU's 16,384 bytes of code on Wormhole. Measured on the
  Wormhole stand-in, the link core's writer is 43,640 bytes (32,104 at `-Os`), and
  26_islands_sum's writer with `KM_PAGE_LINKS` is 30,164 (24,072). The readers (50,680
  and 32,268 bytes) are on BRISC, which has no such limit.

## Running it

Build the host program with `kmr build` against a configured tt-metal tree, naming
this folder's `kernels` (an absolute path). Run it with the chips visible, holding
them, under a time limit:

    kmr build host build/link_wormhole host/link_wormhole.cpp \
        -D 'KM_LINK_KERNELS="<this folder>/kernels"'
    TT_VISIBLE_DEVICES=16,17 build/link_wormhole --pages 16384 --window 32 --batch 8
    TT_VISIBLE_DEVICES=16,17 build/link_wormhole --pages 16384 --window 16 --batch 4 \
        --page 7616 --router-payload 7616
    TT_VISIBLE_DEVICES=16,17,18,19 build/link_wormhole --chips 4 --to 2 \
        --pages 16384 --window 32 --batch 8

Options: `--chips`, `--from`, `--to` (default: the other end of the line), `--link`,
`--phases` (1 ping, 2 line, 4 pages, summed), `--rounds`, `--pages`, `--page`,
`--window`, `--batch`, `--check`, `--router-payload`, `--base` (the area's L1
address on both ends, 0x100000), `--near-core`, `--far-core` (logical `x,y`),
`--limit-s`, `--tick-mhz`.
