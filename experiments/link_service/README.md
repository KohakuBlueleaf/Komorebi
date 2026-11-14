# The link service

Two islands (the chips of a 1 x 2 fabric mesh, one card each) are joined by a link:
on each island one core of the run's group, the link core, is a rank no handler
addresses; it runs the link's reader and writer (`km_link.hpp`).

- The other cores' writers send a message for the far island to the link core as to a
  queue: an atomic add on its outbox tail reserves entries, the message goes into its
  entry, the first mark is set last.
- The link writer sends the outbox's ready entries over the fabric, many to a packet
  (at most 4,352 bytes): a plain write into a free slot of the far link core's inbox,
  then a count packet (an atomic add on the far landed word). Credits come back as a
  16-byte write of the far reader's freed count.
- The link reader takes each landed packet in sequence and issues its entries again
  inside its island, as an ordinary forward to the rank each route word names.
- A link is up only after a hello both ways: each link writer opens its connection at
  start-up, zeroes its region and writes this load's token into the far region; it
  holds its outbox and repeats the token until it has read the far one, then sends it
  once more. No hello within the wait limit ends the run (where 60, "link not up").
- Every wait is bounded; a give-up ends the run with its reason and the link's words.

The link core is a member of the run's group, so its status line counts: sent, the
messages it issued again locally; consumed, the messages it sent across. Each island
then balances by itself. The link's line goes idle once the run's expected crossings
(from the command words) are reached: a run over two islands has no sound quiet end
per island while a message is on the fabric.

A run over two islands needs both RUN commands on the cards at once: `Run.submit`
only queues a command, which reaches the card in a wait or in `Run.store()`.
`cross.py` stores both before it drains either; an island session's waits store
every island's part first.

## Running

```
python cross.py --case across:1:1:1:54
```

A case is `mode:origins:messages:trips:packet`: `origins` cores of island 0 seed
`messages` each; each message goes to its partner (the same rank of island 1 through
both link cores in mode `across`, rank + ranks / 2 of the same island in mode `local`)
and back, `trips` round trips, then ends at its origin. `packet` is the most entries a
packet. The script prints one JSON line.

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
