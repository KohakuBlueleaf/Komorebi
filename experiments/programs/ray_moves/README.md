# ray_moves

Would the ray tracer be faster if the scene stayed where it is and the rays moved? Today
(`projects/raytrace`, main) every core keeps the rays of its own pixels and reads
treelets from DRAM into a few L1 slots. Sponza's overview takes 348.5 ms a frame; the
bounce trace alone is 194.6 ms, and only 6.0 of 32 lanes are alive at an inner node
visit. The design measured here is the other way round:

- every treelet stays in the L1 of one or more owner cores for the whole frame; the top
  tree is in every core;
- a ray's home core (the core of its pixel tile) walks the top tree and puts the ray in
  the bin of the owner of the nearest treelet it enters; a bin is sent as a bundle of up
  to R rays (48 bytes a ray, 16 bytes of header);
- the owner walks the treelet with the bundle's rays as its 32 lanes; each ray then goes
  on to its next treelet if that one's entry is not beyond its best hit, or back home.
  A shadow ray is done at its first hit;
- a home core is done when every ray it sent out has come back.

This experiment builds no tracer. It measures the rays' paths on the host (part A), what
moving bundles costs on the card (part B), and puts the two together into a predicted
frame beside today's (part C).

```
python experiments/programs/ray_moves/paths.py --data DATA --json paths.json        # part A
python experiments/programs/ray_moves/paths.py --tables paths.json                  # its tables
python experiments/programs/ray_moves/transport.py run --variant pages --json p.json
python experiments/programs/ray_moves/transport.py run --variant zero-copy --sizes 2048 4096 \
    --json z.json                                                          # part B
python experiments/programs/ray_moves/transport.py run --variant bodies --sizes 2048 4096
python experiments/programs/ray_moves/predict.py results/paths.json results/today.json \
    results/transport-pages.json results/transport-zero-copy.json         # part C
```

`DATA` is the folder of the repository's scene downloader (`SCENE.npz` and
`SCENE_tree.npz`). Part A takes about 35 minutes on 32 host threads; each part B run
takes 6 to 22 seconds on the card. The runs below wrote their JSON to `results/`,
which git does not keep (`paths.json` without its per-core lists).

## Answer

- **Sponza overview: 292 ms predicted against 348.5 measured**, with zero-copy pages,
  R = 32 and the "full" flush rule, if the owners' walks overlap the transport. If they
  do not overlap, it is 437 ms: slower than today.
- **Bistro interior overview: 2256 ms predicted against 523.7 measured.** It does not
  beat today. Its treelets take 156 MiB and the cores' L1 holds 118 MiB, so the design's
  first premise fails, and with no room for second owners the busiest owner core walks
  11 times the mean (bounce, R = 64: 1167 ms on the busiest core against 101 ms on the
  mean one).
- Part B's page-sized copy of the runtime's messages is bound by the cores copying each
  page twice; pages that no core copies double the rate (12.7 million 2 KiB bundles a
  second over the grid). The runtime's own bodies, a 64-byte line and the bundle as
  its body, move 30.3 million 2 KiB bundles a second: no core copies them either, and
  nothing reads them on the way.

## Part A: the rays' paths (host)

`paths.py` takes the reference's rays and tree and the treelets as `projects/raytrace`
cuts them (`bvh.cut_treelets`, budgets 32, 64 and 128 KiB) and follows every ray of the
four traces (primary, shadow, bounce, second shadow) at 1024 x 1024. The rays of the
later traces are made from the reference's own hits. The walk inside a treelet is the
reference's walk started at the treelet's root; across treelets the ray visits them in
order of entry and stops when the next entry is beyond its best hit. Checked against
`reference.raytrace` on 8192 rays per trace of the 10^5 scene: no hit, triangle or
occlusion differs.

- **Visits**: treelets a ray visits; "saved" is the share of entered treelets the early
  exit skips.
- **Skew**: the share of all visits taken by the busiest 1 % and 10 % of treelets.
- **Owners**: treelets given to 118 cores by balanced predicted load (longest first,
  each to the least loaded core with room for it in its 1 MiB of L1). "One owner": one
  copy a treelet. "Several": a treelet gets one copy per half a core's mean load, 1 to
  8. A ray is sent to the copy `source core mod copies`, or to the sender's own copy if
  it holds one. The column shows visits per core, max over mean.
- **Bundles**: a round-based run of every core's bins. Rule "full": a bin is sent when
  it holds R rays, and the rest at the end of the round when the core has nothing else
  to do. Rule "each": every bin is sent after each input bundle. The bundle counts
  include the bundles that return rays home, not those a core sends to itself. "MB
  moved" is ray records and headers at R = 32, rule "full".
- **Lanes per inner node**: the bundles' rays walked as the card walks a packet (one
  stack per packet, children ordered by the packet's majority direction, a lane tests
  each popped node against its own best hit), sampled at 64 packets a treelet (at most
  16384 a trace), R = 32, rule "full". Today: 6.0 on Sponza's bounce.

The owner placement used by the bundles and the walks is balanced by predicted cycles
(below), not visits. Several owners is used where they fit in L1; for the 10^6 scene and
Bistro they do not, so the bundle columns use one owner.

Calibration: the same emulation on today's packets (Sponza bounce, sorted as `--sort`
sorts them, 1024 packets) against the card's profile. Visits a packet 60.0 (card 62.3),
lanes per inner node 6.50 (6.03), inner node visits a packet 1183 (1338), pops 2426
(2740), leaves 117 (138). The emulation counts about 12 % fewer node visits than the card
does.

10^5 scene (102,308 triangles)

| budget KiB | trace | visits a ray, mean / median / p99 | saved by the early exit | busiest 1 % / 10 % of treelets | visits per core max / mean, one owner / several | L1 fits | bundles, R = 32 / 64 / 128 (fill) | fill under the each rule | lanes per inner node | MB moved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | primary | 2.13 / 2 / 9 | 40 % | 27 % / 75 % | 16.97 / 2.12 | yes | 92k (0.98) / 47k (0.96) / 25k (0.91) | 0.782 / 0.674 / 0.525 | 23.30 | 140 |
| 128 | shadow | 1.60 / 1 / 4 | 14 % | 27 % / 81 % | 16.30 / 2.04 | yes | 57k (0.97) / 29k (0.95) / 15k (0.89) | 0.752 / 0.655 / 0.535 | 20.74 | 86 |
| 128 | bounce | 2.32 / 2 / 7 | 16 % | 24 % / 77 % | 15.37 / 1.92 | yes | 96k (0.75) / 61k (0.59) / 45k (0.40) | 0.127 / 0.083 / 0.053 | 5.14 | 112 |
| 128 | shadow2 | 1.39 / 1 / 4 | 28 % | 19 % / 70 % | 12.95 / 1.62 | yes | 27k (0.60) / 20k (0.41) / 17k (0.25) | 0.087 / 0.058 / 0.039 | 3.74 | 26 |
| 64 | primary | 2.15 / 2 / 10 | 46 % | 37 % / 80 % | 16.78 / 2.10 | yes | 96k (0.97) / 50k (0.94) / 27k (0.87) | 0.726 / 0.595 / 0.439 | 22.75 | 144 |
| 64 | shadow | 1.69 / 2 / 4 | 21 % | 35 % / 83 % | 15.42 / 1.93 | yes | 61k (0.96) / 32k (0.92) / 17k (0.85) | 0.694 / 0.580 / 0.451 | 20.20 | 91 |
| 64 | bounce | 2.48 / 2 / 8 | 20 % | 31 % / 79 % | 14.21 / 1.78 | yes | 112k (0.68) / 77k (0.49) / 60k (0.32) | 0.113 / 0.071 / 0.044 | 5.46 | 118 |
| 64 | shadow2 | 1.44 / 1 / 4 | 39 % | 23 % / 71 % | 11.06 / 1.38 | yes | 33k (0.52) / 25k (0.33) / 22k (0.19) | 0.074 / 0.047 / 0.031 | 3.99 | 26 |
| 32 | primary | 2.16 / 2 / 10 | 50 % | 46 % / 83 % | 16.39 / 2.05 | yes | 98k (0.95) / 52k (0.90) / 29k (0.81) | 0.654 / 0.506 / 0.350 | 22.20 | 145 |
| 32 | shadow | 1.80 / 2 / 5 | 29 % | 42 % / 86 % | 14.05 / 1.76 | yes | 64k (0.94) / 34k (0.88) / 19k (0.78) | 0.625 / 0.499 / 0.367 | 19.71 | 94 |
| 32 | bounce | 2.72 / 2 / 9 | 23 % | 39 % / 81 % | 11.77 / 1.47 | yes | 139k (0.58) / 103k (0.39) / 86k (0.23) | 0.101 / 0.061 / 0.037 | 5.77 | 126 |
| 32 | shadow2 | 1.57 / 1 / 5 | 49 % | 28 % / 74 % | 8.40 / 1.05 | yes | 41k (0.43) / 34k (0.26) / 31k (0.14) | 0.062 / 0.038 / 0.024 | 4.31 | 28 |

10^6 scene (997,806 triangles)

| budget KiB | trace | visits a ray, mean / median / p99 | saved by the early exit | busiest 1 % / 10 % of treelets | visits per core max / mean, one owner / several | L1 fits | bundles, R = 32 / 64 / 128 (fill) | fill under the each rule | lanes per inner node | MB moved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | primary | 1.90 / 2 / 10 | 65 % | 53 % / 94 % | 22.73 / 2.84 | no | 90k (0.96) / 47k (0.92) / 26k (0.84) | 0.701 / 0.560 / 0.399 | 18.56 | 134 |
| 128 | shadow | 1.63 / 1 / 5 | 38 % | 54 % / 95 % | 21.01 / 2.63 | no | 64k (0.95) / 34k (0.91) / 19k (0.82) | 0.652 / 0.523 / 0.385 | 15.78 | 95 |
| 128 | bounce | 2.46 / 2 / 8 | 34 % | 48 % / 87 % | 17.09 / 2.15 | no | 134k (0.60) / 97k (0.41) / 80k (0.25) | 0.094 / 0.057 / 0.034 | 4.42 | 126 |
| 128 | shadow2 | 1.45 / 1 / 5 | 53 % | 39 % / 82 % | 10.45 / 1.32 | no | 51k (0.46) / 42k (0.28) / 37k (0.16) | 0.060 / 0.037 / 0.023 | 2.81 | 36 |
| 64 | primary | 1.96 / 2 / 11 | 68 % | 55 % / 96 % | 16.33 / 2.04 | no | 94k (0.94) / 51k (0.88) / 29k (0.76) | 0.620 / 0.468 / 0.313 | 17.94 | 138 |
| 64 | shadow | 1.78 / 1 / 5 | 44 % | 55 % / 97 % | 14.14 / 1.77 | no | 69k (0.93) / 37k (0.87) / 22k (0.75) | 0.581 / 0.446 / 0.311 | 15.53 | 101 |
| 64 | bounce | 2.72 / 2 / 9 | 36 % | 50 % / 89 % | 12.76 / 1.60 | no | 167k (0.52) / 129k (0.33) / 111k (0.19) | 0.084 / 0.049 / 0.029 | 4.35 | 136 |
| 64 | shadow2 | 1.56 / 1 / 5 | 59 % | 41 % / 84 % | 7.72 / 1.00 | no | 63k (0.38) / 54k (0.22) / 51k (0.12) | 0.052 / 0.031 / 0.019 | 2.97 | 38 |
| 32 | primary | 1.91 / 2 / 11 | 72 % | 55 % / 97 % | 13.59 / 1.70 | no | 96k (0.90) / 54k (0.80) / 33k (0.65) | 0.525 / 0.371 / 0.233 | 17.39 | 134 |
| 32 | shadow | 1.76 / 1 / 6 | 50 % | 58 % / 98 % | 12.41 / 1.55 | no | 72k (0.89) / 40k (0.79) / 25k (0.63) | 0.499 / 0.362 / 0.239 | 14.88 | 99 |
| 32 | bounce | 2.75 / 2 / 10 | 40 % | 52 % / 90 % | 11.07 / 1.39 | no | 205k (0.42) / 169k (0.26) / 153k (0.14) | 0.075 / 0.043 / 0.024 | 4.54 | 136 |
| 32 | shadow2 | 1.59 / 1 / 6 | 64 % | 44 % / 86 % | 6.62 / 1.00 | no | 79k (0.31) / 71k (0.17) / 68k (0.09) | 0.047 / 0.027 / 0.016 | 3.03 | 38 |

Sponza overview (262,266 triangles)

| budget KiB | trace | visits a ray, mean / median / p99 | saved by the early exit | busiest 1 % / 10 % of treelets | visits per core max / mean, one owner / several | L1 fits | bundles, R = 32 / 64 / 128 (fill) | fill under the each rule | lanes per inner node | MB moved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | primary | 12.48 / 12 / 26 | 42 % | 24 % / 65 % | 9.45 / 1.18 | yes | 460k (0.95) / 243k (0.90) / 135k (0.81) | 0.496 / 0.335 / 0.204 | 22.10 | 681 |
| 128 | shadow | 8.62 / 8 / 19 | 31 % | 21 % / 66 % | 8.94 / 1.12 | yes | 314k (0.96) / 164k (0.92) / 89k (0.85) | 0.537 / 0.382 / 0.245 | 21.38 | 469 |
| 128 | bounce | 15.67 / 15 / 29 | 30 % | 10 % / 56 % | 4.19 / 1.10 | yes | 765k (0.69) / 519k (0.51) / 400k (0.33) | 0.074 / 0.041 / 0.022 | 7.02 | 826 |
| 128 | shadow2 | 9.82 / 10 / 23 | 32 % | 18 % / 62 % | 7.67 / 1.07 | yes | 411k (0.80) / 255k (0.64) / 179k (0.46) | 0.046 / 0.026 / 0.015 | 9.01 | 509 |
| 64 | primary | 10.90 / 10 / 26 | 51 % | 29 % / 69 % | 10.83 / 1.35 | yes | 413k (0.93) / 223k (0.87) / 129k (0.75) | 0.459 / 0.302 / 0.180 | 21.97 | 600 |
| 64 | shadow | 7.18 / 7 / 18 | 28 % | 20 % / 67 % | 5.24 / 1.08 | yes | 278k (0.94) / 149k (0.88) / 85k (0.77) | 0.517 / 0.361 / 0.228 | 21.57 | 406 |
| 64 | bounce | 14.89 / 14 / 29 | 37 % | 18 % / 62 % | 4.25 / 1.01 | yes | 800k (0.64) / 568k (0.45) / 457k (0.28) | 0.070 / 0.038 / 0.020 | 7.84 | 795 |
| 64 | shadow2 | 8.35 / 8 / 20 | 32 % | 20 % / 65 % | 4.97 / 1.01 | yes | 422k (0.68) / 290k (0.49) / 227k (0.32) | 0.045 / 0.025 / 0.014 | 9.95 | 446 |
| 32 | primary | 9.68 / 9 / 27 | 58 % | 36 % / 79 % | 12.19 / 1.52 | yes | 380k (0.91) / 210k (0.83) / 126k (0.69) | 0.443 / 0.290 / 0.171 | 21.71 | 539 |
| 32 | shadow | 6.35 / 6 / 17 | 29 % | 27 % / 78 % | 4.32 / 1.00 | yes | 257k (0.91) / 142k (0.83) / 85k (0.69) | 0.477 / 0.324 / 0.199 | 21.53 | 365 |
| 32 | bounce | 13.13 / 13 / 27 | 44 % | 23 % / 70 % | 3.78 / 1.00 | yes | 807k (0.56) / 608k (0.37) / 513k (0.22) | 0.065 / 0.035 / 0.018 | 8.24 | 706 |
| 32 | shadow2 | 7.66 / 7 / 19 | 34 % | 24 % / 71 % | 4.89 / 1.00 | yes | 438k (0.60) / 320k (0.41) / 264k (0.25) | 0.042 / 0.023 / 0.013 | 10.73 | 414 |

Bistro interior overview (1,020,903 triangles)

| budget KiB | trace | visits a ray, mean / median / p99 | saved by the early exit | busiest 1 % / 10 % of treelets | visits per core max / mean, one owner / several | L1 fits | bundles, R = 32 / 64 / 128 (fill) | fill under the each rule | lanes per inner node | MB moved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | primary | 12.35 / 11 / 32 | 36 % | 55 % / 94 % | 9.56 / 1.20 | no | 462k (0.95) / 246k (0.89) / 139k (0.79) | 0.513 / 0.357 / 0.222 | 26.31 | 678 |
| 128 | shadow | 10.96 / 11 / 32 | 27 % | 59 % / 96 % | 10.54 / 1.32 | no | 403k (0.94) / 215k (0.88) / 123k (0.77) | 0.352 / 0.230 / 0.138 | 24.54 | 588 |
| 128 | bounce | 13.72 / 13 / 32 | 21 % | 38 % / 83 % | 7.61 / 1.03 | no | 824k (0.57) / 616k (0.38) / 518k (0.23) | 0.066 / 0.036 / 0.020 | 8.07 | 729 |
| 128 | shadow2 | 13.32 / 13 / 35 | 25 % | 55 % / 89 % | 8.21 / 1.05 | no | 564k (0.78) / 353k (0.63) / 250k (0.44) | 0.041 / 0.022 / 0.012 | 13.38 | 686 |
| 64 | primary | 11.86 / 10 / 32 | 39 % | 63 % / 97 % | 9.95 / 1.24 | no | 459k (0.92) / 253k (0.83) / 151k (0.70) | 0.461 / 0.310 / 0.187 | 25.83 | 653 |
| 64 | shadow | 9.53 / 9 / 31 | 26 % | 64 % / 97 % | 10.90 / 1.36 | no | 368k (0.91) / 204k (0.82) / 124k (0.67) | 0.324 / 0.206 / 0.121 | 23.60 | 518 |
| 64 | bounce | 13.73 / 13 / 33 | 23 % | 47 % / 88 % | 6.58 / 1.00 | no | 934k (0.50) / 732k (0.32) / 637k (0.18) | 0.062 / 0.034 / 0.018 | 9.10 | 731 |
| 64 | shadow2 | 11.67 / 11 / 32 | 26 % | 62 % / 91 % | 8.57 / 1.09 | no | 562k (0.69) / 379k (0.51) / 291k (0.33) | 0.040 / 0.022 / 0.012 | 13.20 | 608 |
| 32 | primary | 15.54 / 14 / 36 | 35 % | 78 % / 99 % | 7.59 / 1.00 | no | 597k (0.91) / 332k (0.81) / 201k (0.67) | 0.497 / 0.341 / 0.210 | 25.33 | 840 |
| 32 | shadow | 8.80 / 8 / 27 | 28 % | 71 % / 98 % | 10.86 / 1.36 | no | 356k (0.87) / 205k (0.75) / 131k (0.59) | 0.316 / 0.199 / 0.117 | 23.44 | 482 |
| 32 | bounce | 14.20 / 13 / 33 | 25 % | 57 % / 92 % | 6.21 / 1.00 | no | 1067k (0.45) / 863k (0.28) / 766k (0.16) | 0.060 / 0.032 / 0.017 | 9.09 | 755 |
| 32 | shadow2 | 11.47 / 11 / 30 | 25 % | 67 % / 93 % | 8.56 / 1.08 | no | 605k (0.64) / 428k (0.45) / 342k (0.28) | 0.039 / 0.021 / 0.012 | 13.69 | 599 |

The 10^6 scene and Bistro take 156 to 160 MiB of treelets at every budget: they do not
fit in 118 cores of 1 MiB. The 10^5 scene (16 MiB) and Sponza (40.5 MiB) do.

What part A says:

1. Visits a ray, Sponza's bounce at 128 KiB: 15.7 (median 15, p99 29); the early exit
   skips 30 % of the 22.3 treelets a ray enters.
2. The skew is high on primary and shadow rays (the busiest 10 % of treelets take 65 %
   of Sponza's visits, 94 % of Bistro's) and lower on the bounce (56 %, 83 %).
3. With one owner a treelet the busiest core has 4 to 23 times the mean visits; with up
   to 8 owners for the busy ones, 1.0 to 1.5 on Sponza and up to 2.1 on the 10^5 scene
   (128 treelets at 128 KiB, too few to spread).
4. Rule "full" sends Sponza's bounce in 765k bundles of 32 (69 % full); 64 and 128 are
   emptier (51 %, 33 %) and save fewer bundles than they cost in page size. Rule "each"
   fills 2 to 7 % of a bundle on the secondary traces: unusable.
5. Lanes per inner node from bundles: 7.0 on Sponza's bounce (today 6.0), 9.0 on its
   second shadow, 22 on primary and shadow rays. The bundle does not fix the divergence
   of the bounce: rays that share a treelet do not share a direction.
6. Sponza's frame moves 2.5 GB of ray records at R = 32 (bounce: 826 MB).

## Part B: the transport (card)

`transport.py run` loads one dynamic program on the whole grid (120 cores: 118 workers
and the last 2 as counter tiles) whose handler takes a message, checks its checksum,
and forwards it to the next rank until its hop count runs out. Ranks are drawn from
`kernels/destinations.h`: the share of bundles each core receives in Sponza's bounce
(128 KiB, R = 32, rule "full"; the busiest core 2.3 times the mean). One run seeds M
messages and forwards each F times; the rate is M (F2 - F1) over the difference of the
card's own times of two runs (F = 100 and 400), so launch and dispatch drop out. Best of
3 pairs. Every run checks: the command finished, every core ran it, M (F + 1) visits and
M messages consumed, every checksum exact, nothing rejected, the counter tiles took no
messages.

Four sets of kernels:

- **runtime**: `km::Messages` as it is, 64 and 256 bytes (1 and 4 rays a message).
- **pages**: the runtime's message kernels copied into `kernels/` (`page_messages.hpp`,
  `pages_reader.cpp`, `pages_writer.cpp`) with a 4096-byte message maximum and 4 messages
  a packet; nothing else changes.
- **zero-copy**: chosen after "pages" was clearly the bound (below), stated before it was
  built: the same DRAM queues and counter-tile reservations, but the compute hands the
  writer the address of a page where it arrived and the writer sends it from there with
  one NoC write; the reader checks each page's checksum as it lands, so only the NoC
  moves a page's bytes (`zero_copy_*`). No core that answers atomics issues any; no NoC
  register is read.
- **bodies**: the runtime's own reader and writer and `km::Messages` with
  `body_bytes`: a 64-byte line (id, hops, state, the body's checksum) and the bundle
  as its body, forwarded as it arrived (`compute_bodies.cpp`). Only the core where a
  message ends reads its body and checks the checksum, so a visit costs no read of the
  page; 4 messages a packet.

Blackhole card, AICLK 1350 MHz before and after every run (one sample 1343), host load
9.4 to 12.4. Messages over the grid a second:

| kernels | message bytes | rays a message | messages | million messages a second | GB a second | compute cycles a visit, check / forward |
| --- | --- | --- | --- | --- | --- | --- |
| runtime | 64 | 1 | 8192 | 48.78 | 3.1 | 146 / 371 |
| runtime | 256 | 4 | 8192 | 36.32 | 9.3 | 626 / 867 |
| pages | 2048 | 32 (1552 bytes) | 2048 | 6.32 | 12.9 | 5108 / 5754 |
| pages | 4096 | 64 (3088 bytes) | 2048 | 3.19 | 13.1 | 10229 / 11372 |
| zero-copy | 2048 | 32 | 2048 | 12.74 | 26.1 | 4 / 114 |
| zero-copy | 4096 | 64 | 2048 | 6.53 | 26.7 | 5 / 114 |
| bodies | 64 + 2048 | 32 | 2048 | 30.31 | 62.1 | 13 / 2091 |
| bodies | 64 + 4096 | 64 | 2048 | 29.35 | 120.2 | 26 / 2205 |
| bodies, overlapped writer | 64 + 2048 | 32 | 2048 | 34.63 | 70.9 | not read |
| bodies, overlapped writer | 64 + 4096 | 64 | 2048 | 33.75 | 138.2 | not read |

The bodies rows: card 1, 2026-11-16, AICLK 1350 MHz before and after every run, host
load 4.4 to 4.1, best of 3 pairs (the pairs within 0.2 %); the check column is the
cycles of the end check over all visits. The overlapped-writer rows: the same command
after the writer's two-stage flush (below), card 1, one run each size, exact.

- One message per ray: Sponza's bounce moves 16.9 million rays, 347 ms at 64 bytes,
  more than today's whole bounce trace; 117 ms at 256 bytes if every message carries 4
  rays, against 60 ms for the same rays in 2 KiB zero-copy bundles.
- **What caps "pages"**: the compute core copies every page word by word, once into the
  check and once out to the writer, at about 10 cycles a word: 21.6k cycles a 4 KiB visit
  on the busiest core, which receives 2.3 times the mean. The rate halves when the page
  doubles. Not the queue memory (4096 messages a queue were enough) and not the counter
  tiles (one reservation a page). At R = 32 Sponza's bounce needs 121 ms of transport
  against 107 ms of walking on the busiest owner: the transport is the bound.
- Zero-copy also halves with the page size, at 26 GB/s over the grid: the bound is
  then the bytes the busiest core's reader and writer move through the DRAM queues
  (about 8 µs a 4 KiB page there), not the cores' compute.
- R = 128 needs 6160 bytes, more than the 4096-byte maximum of these kernels' message
  bins; it is not measured.
- The runtime's bodies do not halve with the page: 30.3 and 29.4 M a second. The bound
  is each core's writer: `forward` with a body waits until the writer has reserved the
  position, written the body and the line and committed (about 2,100 cycles a visit);
  the busiest core's 16,010 visits at that cost are the run. Zero-copy's own bound was
  its reader's checksum of every page as it landed, which the bodies runs leave out.
- The writer's two-stage flush commits the last bin while the next bin's reservation
  travels, and a forward of a body that arrived no longer waits for its bin (the packet's
  `done` does): 34.6 and 33.8 M a second, 14 to 15 % more.

After `transport.py destinations PATHS_JSON` (a part A report of one scene with its
per-core lists) regenerates `kernels/destinations.h`, run the repository's C++
formatter on it.

## Part C: the prediction

`predict.py`, per trace:

- **top walk**: the home core's top-tree walk and its sends, taken as today's phase 1
  on the slowest core (the same walk over the same top tree);
- **owner walks**: the busiest owner core's cycles: the sampled packets' pops, inner
  nodes and leaves times today's measured cycles per step (36 a pop, 362 an inner node,
  737 a nearest-hit leaf, 485 an any-hit one) plus 250 a visit for packet state in and
  out, at 1350 MHz;
- **transport**: the trace's bundles over the measured rate of the page that holds them;
- **output** and **rest**: today's phase 3 on the slowest core, and today's command time
  minus its slowest core's total (dispatch and the tail), kept as they are;
- overlapped: top walk + max(owner walks, transport) + output + rest; serial: the sum.

The frame adds today's GEN, SHADE1, SHADE2 and IMAGE; SORT and UNSORT are gone.

| scene | transport, R | primary | shadow | bounce | second shadow | frame overlapped | frame serial | today |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sponza overview | today | 50.10 | 27.86 | 194.64 | 56.38 | | | 348.53 |
| Sponza overview | pages, 32 | 93.82 | 62.70 | 149.64 | 80.70 | 391.8 | 592.5 | |
| Sponza overview | pages, 64 | 97.16 | 64.38 | 191.29 | 95.53 | 453.3 | 668.8 | |
| Sponza overview | zero-copy, 32 | 57.19 | 37.71 | 135.21 | 57.24 | **292.3** | 437.2 | |
| Sponza overview | zero-copy, 64 | 58.66 | 38.15 | 140.32 | 60.64 | 302.7 | 479.6 | |
| Bistro overview | today | 56.76 | 41.36 | 307.96 | 98.34 | | | 523.72 |
| Bistro overview | zero-copy, 32 | 392.17 | 254.23 | 1456.27 | 528.32 | 2635.9 | 2812.7 | |
| Bistro overview | zero-copy, 64 | 486.76 | 150.18 | 1211.64 | 402.82 | **2256.3** | 2475.4 | |

Sponza, zero-copy, R = 32, by term (ms, four traces):

| trace | top walk | owner walks (busiest / mean core) | transport | output | rest |
| --- | --- | --- | --- | --- | --- |
| primary | 9.30 | 31.64 / 14.66 | 36.11 | 2.51 | 9.28 |
| shadow | 7.57 | 20.95 / 9.11 | 24.64 | 0.87 | 4.64 |
| bounce | 14.75 | 106.60 / 67.49 | 60.09 | 2.51 | 11.35 |
| second shadow | 8.24 | 41.58 / 24.72 | 32.29 | 0.87 | 6.55 |

The three largest terms on Sponza:

1. **Owner walks, 200.8 ms** (bounce 106.6). Rests on the emulated walk: about 12 %
   fewer node visits than the card makes on today's packets, so this term is more likely
   low than high. It also rests on today's per-step cycles holding at 7 alive lanes,
   and on the placement by sampled cycles (the busiest core is 1.6 times the mean on
   the bounce).
2. **Transport, 153.1 ms**, hidden except where it exceeds the walk (primary and shadow).
   Rests on the overlap: the reader and writer move bundles while the compute walks,
   which no run here measured. Without it Sponza is 437 ms. It also rests on the rate
   measured with Sponza bounce's destinations and nothing else running on the cores.
3. **Top walk, 39.9 ms.** Rests on the home core's walk of the top tree costing what
   today's phase 1 costs; today's phase 1 also writes packets, the redo's fills bins.

Bistro is sunk by the owner walks: 1167 ms on the busiest core in the bounce against
101 ms on the mean core, because its treelets do not fit in L1 even once and no hot
treelet can have a second owner. The prediction for Bistro is also only an indication,
because a design whose scene does not fit in L1 is not the design measured.

## What the redo should use

Zero-copy pages of 2 KiB with R = 32 and the "full" flush rule: the only option that is
faster than today on Sponza, and only with the owners' walks overlapping the transport.
The largest risk is that the design does not hold the larger scenes at all: Bistro and
the 10^6 scene need 156 MiB of L1 for their treelets against 118 MiB, and without room
for copies of hot treelets one owner core takes 6 to 23 times the mean visits. The second
is the overlap: serial, Sponza is 437 ms. The bundle does not raise the alive lanes of
the bounce (7.0 against 6.0); the gain on Sponza comes from no treelet reads, no SORT
and the balance of copies, not from fuller packets.
