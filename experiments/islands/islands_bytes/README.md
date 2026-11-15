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

## Two more forms: pages of the command's size, and a range of one bank

`forms.py`: can a kernel on either island read and write a tensor in pages whose size
the command gives (`km::SizedTensor(word, page_bytes)`), and a range of one DRAM bank
of the other island (`session.bank_range`, `km::BankRange(word, bank, page_bytes)`),
with the page and byte calls, every word landing where it should?

A uint32 tensor of 256 pages of 4,096 bytes is held by island 1; its word at bank
address `x` of bank `b` is `(b << 20 | x >> 2) ^ 0x5A17C0DE`. All 236 ranks of both
islands run each command, so island 0's ranks go across and island 1's read their own
DRAM. PAGES: rank r reads pages 2r and 2r + 1 in pages of the command's size, checks
every word and writes them xor a mark at pages 472 + 2r on; once with 1,024-byte
pages and once with 192-byte pages. BYTES: rank r reads four 128-byte items of pages
of 8,192 bytes (larger than a packet) with `km::ask_bytes`, checks them and puts them
xor the mark at items of its own further on with `km::put_bytes`. RANGE: a 64-page
tensor held by island 0 stands in for an X280 thread's region; its pages in bank 6
(32 KiB) hold `(x >> 2) ^ 0x6A09E667` at their offset `x` in the range, the other
banks other words. Rank r reads pages 2r and 2r + 1 of 64 bytes with `km::read_pages`
and 40 bytes at byte 20 of page 2r with `km::read_bytes`, checks them, and writes page
2r back xor the mark with `km::write_pages` and page 2r + 1 with `km::write_bytes`. The
host then reads both tensors back and compares every word with the writes replayed in
order. "ms" is the dispatcher's span of the command, the slower island's.

2026-11-15 05:53, cards 0 and 1 (p150a), four links, host load 6.2, one run:

| Command | Pages, bytes | ms | Words checked | Wrong | 0 to 1: asks, pages, puts, bytes | 1 to 0: asks, pages, puts, bytes |
|---|---|---|---|---|---|---|
| PAGES | 1,024 | 0.511 | 120,832 | 0 | 118, 118, 118, 256,768 | 0, 236, 118, 242,608 |
| PAGES | 192 | 0.502 | 22,656 | 0 | 118, 118, 118, 60,416 | 0, 236, 118, 46,256 |
| BYTES | 8,192 | 0.915 | 30,208 | 0 | 472, 118, 118, 279,424 | 0, 472, 118, 60,888 |
| RANGE | 64 | 1.294 | 9,912 | 0 | 0, 354, 236, 21,240 | 354, 236, 236, 286,976 |

The counts are the links' (`run.crossings()`, [What crossed](../../../docs/reference/islands.md#what-crossed)):
island 0's 118 ranks send one get line for their two pages and one put line, island
1's answer two pages; in BYTES four range lines and one batch a rank; in RANGE island
1's ranks send three get lines (a line a page of the range, one range) and two puts
(a page and a batch).

The host read the tensor (1 MiB) and the range's tensor back: 0 words wrong in each,
the range's other banks unchanged. Both kernels ended with no error. The clock was
800 MHz idle before and after; the ranks' cycles inside the calls over the command's
span (at most 672,366 cycles in 0.511 ms) put it near 1,350 MHz during the commands.

    python forms.py             # cards 0,1; --sizes, --item-page, --items, --range-page

It needs the `komorebi` package and both cards, prints progress lines and one JSON
line, and exits 1 unless every word is exact.
