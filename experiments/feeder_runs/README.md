# Longer feeder runs

Runs of the map/collect example under the test feeder that are too long or too broad
for the card tests: many jobs, every team size, timing of teams in flight. Each is a
command line of `tests/hardware/feeder/mapcollect_fed.py`, which prints one JSON line.
Run them by hand on a card, one at a time, under a watcher that ends a run that stops
printing.

| run | command arguments | what to look for |
|---|---|---|
| a thousand jobs | `--jobs 1000 --teams random --out-pages 2048 --out-tensors 32 --seed 7` | `done` 1000, `jobs_bad` 0, `gave_up_cores` 0, every kernel ended |
| random teams, four in flight | `--jobs 100 --teams random --depth 4 --out-tensors 4` | `jobs_bad` 0, `random_total_rel_rms_max` under 1e-2 |
| team sizes | `--jobs 20 --count C --out-tensors 2`, C = 1, 8, 119 | `jobs_bad` 0 |
| teams across rows | `--jobs 20 --first F --count C --out-tensors 2`, (F, C) = (6, 13), (0, 60) | `jobs_bad` 0, `membership_bad` empty |
| disjoint teams at once | `--jobs 40 --teams pair --depth 2 --trace --out-tensors 2` | `overlap.overlapping` above 0 |
| overlapping teams, not held | `--jobs 40 --teams overlap --trace --out-tensors 2` | `overlapping_teams.together` above 0: the check sees overlap |
| EXIT held until no slot is busy | `--jobs 10 --count 60 --exit-at 5` | `exit.after_every_ack` true, `exit.reported_by` 119 |
| a long record at the stop | `--team-list 1:1,0:1 --stall 0 --tiles 16384 --deadline 0.001` | core 0's compute gave up in `cinput` on the stop |

## Result

Each of these passed on one Blackhole p150a (slow dispatch, 119 workers and the
feeder) while it was part of the card tests. The thousand jobs took about 190 s, most
of it reading the 32 output tensors back.
