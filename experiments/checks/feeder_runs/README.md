# Longer feeder runs

Runs of the map/collect example under the test feeder that are too long or too broad
for the card tests: many jobs, every group size, timing of groups in flight. Each is a
command line of `mapcollect_fed.py` in this folder, which prints one JSON line.
Run them by hand on a card, one at a time, under a watcher that ends a run that stops
printing.

| run | command arguments | what to look for |
|---|---|---|
| a thousand jobs | `--jobs 1000 --groups random --out-pages 2048 --out-tensors 32 --seed 7` | `done` 1000, `jobs_bad` 0, `gave_up_cores` 0, every kernel ended |
| random groups, four in flight | `--jobs 100 --groups random --depth 4 --out-tensors 4` | `jobs_bad` 0, `random_total_rel_rms_max` under 1e-2 |
| group sizes | `--jobs 20 --count C --out-tensors 2`, C = 1, 8, 119 | `jobs_bad` 0 |
| groups across rows | `--jobs 20 --first F --count C --out-tensors 2`, (F, C) = (6, 13), (0, 60) | `jobs_bad` 0, `ranks_bad` empty |
| disjoint groups at once | `--jobs 40 --groups pair --depth 2 --trace --out-tensors 2` | `overlap.overlapping` above 0 |
| overlapping groups, not held | `--jobs 40 --groups overlap --trace --out-tensors 2` | `overlapping_groups.together` above 0: the check sees overlap |
| EXIT held until no slot is busy | `--jobs 10 --count 60 --exit-at 5` | `exit.after_every_ack` true, `exit.reported_by` 119 |
| a long command at the stop | `--group-list 1:1,0:1 --stall 0 --tiles 16384 --deadline 0.001` | core 0's compute gave up in `cinput` on the stop |

## Result

Each of these passed on one Blackhole p150a (slow dispatch, 119 workers and the
feeder) while it was part of the card tests. The thousand jobs took about 190 s, most
of it reading the 32 output tensors back.
