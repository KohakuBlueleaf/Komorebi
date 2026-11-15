# The link service

The first measure of messages between islands: two islands (the chips of a 1 x 2
fabric mesh, one card each) joined by one link, a core on each island running a link
reader and writer of its own as a rank of the run's group, its own connection opened
at start-up, a hello both ways.

The service now runs on the island's link core 0 (`Session(cards=..., links=...)`), on
the CPU that holds link 0, and the run's ranks are only the flat ranks (see
`docs/concepts/islands.md` and `examples/05_forward/islands.py`). The script and kernels
that measured the table below are in the repository's history, at the commit
"Experiments: link_service, the cost of a crossing and messages a second"
(`git log -- experiments/islands/link_service`).

## Results

Two Blackhole p150a cards joined by four Ethernet links, one link used (channel 4,
link cores (11, 0) on both); 64-byte messages, AICLK 1350 MHz, outbox 128 entries,
inbox 4 packets. Every case below: both runs done, the partners' visits and every
origin's count and id sum exact. Times are from the origins' first seed to their last
end (wall clock of island 0).

| Case | Result |
|---|---|
| across:1:1:1:54, one message to island 1 and back | 13.07 µs the round trip, 6.5 µs a crossing |
| local:1:1:1000:54, one message, 1000 round trips inside island 0 | 2.135 µs a hop |
| across:1:1:1000:54, one message, 1000 round trips across | 4.714 µs a crossing |

Messages a second each way (`origins:128:trips`; single: a packet a message; batched:
up to 54 messages a packet, as many as wait):

| Origins on island 0 | Round trips | Single | Batched |
|---|---|---|---|
| 1 | 16 | 601,697 | 2,935,150 |
| 12 | 8 | 606,123 | 4,691,237 |
| 117 (all) | 4 | 604,923 | 4,759,470 |

A single message costs the link about 1.66 µs a direction whatever the number of
senders: its data packet and its count packet, each waited out of the core. Batched,
one origin is limited by its own round trip (128 messages in flight) and 12 or more
reach about 4.7 million messages a second each way, 300 MB/s of messages.
