# Ranges of a page across islands

Two cards opened as the islands of one session (`km.Session(cards=[0, 1], links=4)`).
When a kernel reads a few bytes out of a page that the other island holds, with
`km::read_bytes`, do exactly those bytes land at the L1 address it names, and nothing
beside them? And what does one such range cost?

## The read

`km::read_bytes(core, command, tensor, page, offset, bytes, into)` reads a page this
island holds with one NoC read. For a page the other island holds it writes one get
line (64 bytes) into the island's link core 0, which sends it across like a page ask.
A page CPU of the holder (the lines in turn over the CPUs that do not carry the lines)
reads the range from DRAM into its slots, at the same low 6 address bits as the DRAM
address, and writes it into the asker's L1: each word before `into`'s first 16-byte
line and after its last as an inline word write (a write's two ends share their low 4
bits, and the router writes a payload from a 16-byte line), then the lines between as
one write fused with an add of 1 to the asking CPU's page port. A range inside one
16-byte line goes as words, then an add alone. `km::ask_bytes` does the same without
waiting and `km::bytes_wait` waits for all of them, so a core keeps several ranges in
flight. Offset and bytes are multiples of 4, and `into` has the offset's low 6 bits.

## Result

`ranges.py`: a uint32 tensor of 256 pages of 4,096 bytes, word w of page p
`(p << 16 | w) ^ 0x9E3779B9`, held by island 1 ("across") and a copy held by island 0
("here"). Each of island 0's 118 ranks reads 64 ranges with one `READ` command a row,
each range at a page and offset of its own (a hash of the rank and the range's
number), into a slot of its L1 whose other words hold a mark. The core checks every
word of every range against the pattern and the 16 bytes either side against the mark;
the host checks each rank's last range against its own copy of the data. "us a range"
is the cycles a rank spent inside the read calls over its 64 ranges, at the clock read
after the command (1350 MHz in every row), mean and largest over the ranks; "command"
is the dispatcher's span of the command (submit to completion).

2026-11-15 05:10, cards 0 and 1 (p150a), four links, host load 11.7, one run:

| Place | Bytes | In flight | Command, ms | us a range, mean / max | Words wrong | Guards changed |
|---|---|---|---|---|---|---|
| across | 4 | 1 | 12.833 | 187.3 / 199.0 | 0 | 0 |
| across | 4 | 32 | 12.897 | 196.9 / 200.1 | 0 | 0 |
| across | 64 | 1 | 14.457 | 215.1 / 224.2 | 0 | 0 |
| across | 64 | 32 | 14.429 | 220.4 / 223.8 | 0 | 0 |
| across | 128 | 1 | 14.422 | 215.8 / 223.5 | 0 | 0 |
| across | 128 | 32 | 14.503 | 221.5 / 224.8 | 0 | 0 |
| across | 2048 | 1 | 14.559 | 211.1 / 220.6 | 0 | 0 |
| across | 2048 | 32 | 14.718 | 217.8 / 223.1 | 0 | 0 |
| here | 4 | 1 | 0.128 | 0.476 / 0.507 | 0 | 0 |
| here | 4 | 32 | 0.106 | 0.185 / 0.197 | 0 | 0 |
| here | 64 | 1 | 0.139 | 0.477 / 0.513 | 0 | 0 |
| here | 64 | 32 | 0.117 | 0.181 / 0.196 | 0 | 0 |
| here | 128 | 1 | 0.150 | 0.478 / 0.497 | 0 | 0 |
| here | 128 | 32 | 0.128 | 0.191 / 0.206 | 0 | 0 |
| here | 2048 | 1 | 0.477 | 0.539 / 0.556 | 0 | 0 |
| here | 2048 | 32 | 0.463 | 0.295 / 0.399 | 0 | 0 |

Every rank's last range equals the host's data in every row. Each "across" row crossed
7,552 get lines (483,328 bytes) one way and 7,552 ranges back.

Across, the other island answers one range every 1.70 µs (4 bytes) to 1.95 µs (2,048
bytes): 7,552 ranges in 12.8 to 14.7 ms, the same with 32 in flight as with one. So
with 118 ranks reading at once, a rank's 190 to 220 µs a range is its place in that
queue, not a range's own time; how long one range takes with the island otherwise idle
was not measured. The 4-byte rows are words and an add only, so no add arrived before
the words it counts in 15,104 ranges.

## Running it

    python ranges.py            # cards 0,1; --sizes, --flights, --ranges, --pages

It needs the `komorebi` package and both cards, prints progress lines and one JSON
line, and exits 1 unless every rank's ranges are exact.
