"""Two tensor forms across islands: a tensor island 1 holds, read and written in pages
of the command's size (km::SizedTensor) by every rank of both islands, and a range of
one DRAM bank of island 0 (km::BankRange) read and written back by them. Every word
checked on the cores and on the host. Prints one JSON line. See README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

KERNELS = Path(__file__).resolve().parent / "forms"
PAGE = 4096  # the tensors' own pages
BANKS = 8  # kFormsBanks
MARK = 0xA5A5A5A5  # kFormsMark
MAGIC = 0x464F524D  # kFormsMagic
ITEM = 128  # kFormsItem
DATA = 8192  # kFormsData
RANGE_OFFSET, RANGE_BYTES = 20, 40  # kFormsRangeOffset, kFormsRangeBytes
REPORT = 64  # kFormsReport
NUMBERS = {"pages": 1, "bytes": 2, "range": 3}  # kHandlerPages, kHandlerBytes, ..
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument("--pages", type=int, default=256, help="the tensor's pages")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 192])
    parser.add_argument("--item-page", type=int, default=8192, help="BYTES' pages")
    parser.add_argument("--items", type=int, default=4, help="BYTES: items a rank")
    parser.add_argument("--range-page", type=int, default=64, help="RANGE's pages")
    parser.add_argument("--bank", type=int, default=6, help="the range's bank")
    return parser.parse_args()


def forms_word(bank, offset):
    """forms_word: the tensor's word at bank address `offset` of bank `bank`."""
    return ((bank << 20) | (offset >> 2)) ^ 0x5A17C0DE


def range_word(offset):
    """forms_range_word: the bank range's word at its byte `offset`."""
    return (offset >> 2) ^ 0x6A09E667


def span(first, pages, size):
    """[low, high) of the bank addresses pages first .. of `size` bytes cover."""
    return first // BANKS * size, -(-(first + pages) // BANKS) * size


def banks_model(bank_bytes):
    """The tensor's words by bank and bank address: [banks, bank_bytes / 4]."""
    words = np.arange(bank_bytes // 4, dtype=np.uint32)[None, :]
    banks = np.arange(BANKS, dtype=np.uint32)[:, None]
    return ((banks << np.uint32(20)) | words) ^ np.uint32(0x5A17C0DE)


def as_pages(model, page):
    """The banks' words as the interleaved pages of `page` bytes (a divisor of a bank's
    bytes): [pages, page / 4]."""
    per = model.shape[1] * 4 // page
    return (
        model.reshape(BANKS, per, page // 4).transpose(1, 0, 2).reshape(-1, page // 4)
    )


def from_pages(pages, page):
    """as_pages' inverse."""
    per = pages.shape[0] // BANKS
    return pages.reshape(per, BANKS, page // 4).transpose(1, 0, 2).reshape(BANKS, -1)


def spot(model, page, index, offset, size):
    """`size` bytes at `offset` of page `index` of `page` bytes, as a view of words."""
    at = (index // BANKS * page + offset) // 4
    return model[index % BANKS, at : at + size // 4]


def apply_pages(model, page, ranks, count, target):
    """PAGES' writes: page i to page target + i, each word xor MARK (i < ranks x
    count), every source read before any write."""
    done = ranks * count
    read = [spot(model, page, i, 0, page).copy() for i in range(done)]
    for i, words in enumerate(read):
        spot(model, page, target + i, 0, page)[:] = words ^ np.uint32(MARK)


def apply_bytes(model, page, ranks, count, sources, target):
    """BYTES' puts: item g of pages `target` on is source item g % sources, xor MARK."""
    per = page // ITEM
    read = [
        spot(model, page, s // per, s % per * ITEM, ITEM).copy()
        for s in np.arange(ranks * count) % sources
    ]
    for g, words in enumerate(read):
        at = target * per + g
        spot(model, page, at // per, at % per * ITEM, ITEM)[:] = words ^ np.uint32(MARK)


def region_pages(pages, bank):
    """The region tensor's words: bank `bank`'s pages hold range_word of the region's
    offsets, the others forms_word's complement."""
    out = np.empty((pages, PAGE // 4), np.uint32)
    for j in range(pages):
        offsets = (j // BANKS) * PAGE + 4 * np.arange(PAGE // 4, dtype=np.uint32)
        if j % BANKS == bank:
            out[j] = (offsets >> np.uint32(2)) ^ np.uint32(0x6A09E667)
        else:
            out[j] = ~(((j % BANKS) << 20 | (offsets >> 2)) ^ np.uint32(0x5A17C0DE))
    return out


def apply_range(region, bank, page, ranks):
    """RANGE's writes: region pages 0 .. 2 ranks - 1 of `page` bytes xor MARK."""
    out = region.copy()
    mine = out[bank::BANKS].reshape(-1)  # the bank's pages, the range in order
    mine[: 2 * ranks * page // 4] ^= np.uint32(MARK)
    out[bank::BANKS] = mine.reshape(-1, PAGE // 4)
    return out


def check_reports(rows, handler, checked):
    """Each rank's report of one command: (bad ranks, words wrong, islands seen)."""
    bad, wrong, islands = [], 0, set()
    for rank, row in enumerate(rows):
        ok = (
            int(row[0]) == MAGIC
            and int(row[1]) == rank
            and int(row[2]) == handler
            and int(row[3]) == checked
            and int(row[4]) == 0
            and int(row[5]) == 0
        )
        wrong += int(row[4])
        islands.add(int(row[7]))
        if not ok:
            bad.append(rank)
    return bad, wrong, sorted(islands)


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    bank_bytes = -(-args.pages // BANKS) * PAGE
    report = {"cards": cards, "links": args.links, "bank_bytes": bank_bytes, "rows": []}
    with km.Session(cards=cards, links=args.links, log=log) as session:
        graph = session.graph
        own = [(graph.first(i), graph.count(i)) for i in range(2)]
        ranks = graph.ranks
        count = 2  # PAGES: pages a rank
        target = -(-ranks * count // BANKS) * BANKS
        sources = BANKS * (args.item_page // ITEM)  # BYTES: the items of pages 0..7
        item_target = 3 * BANKS
        item_pages = -(-ranks * args.items * ITEM // args.item_page)
        # each command reads bytes no earlier command wrote; all inside the banks
        reads = [span(0, ranks * count, size) for size in args.sizes]
        writes = [span(target, ranks * count, size) for size in args.sizes]
        reads.append(span(0, BANKS, args.item_page))
        writes.append(span(item_target, item_pages, args.item_page))
        for number, (low, high) in enumerate(reads):
            hit = [w for w in writes[:number] if w[0] < high and low < w[1]]
            if hit or writes[number][1] > bank_bytes or count * max(args.sizes) > DATA:
                raise SystemExit(f"command {number}: its pages do not fit ({hit})")
        model = banks_model(bank_bytes)
        tensor = session.tensor(as_pages(model, PAGE), "uint32", group=own[1])
        region_data = region_pages(64, args.bank)
        region_tensor = session.tensor(region_data, "uint32", group=own[0])
        base = region_tensor.parts[0].address
        region = session.bank_range(0, args.bank, base, 8 * PAGE)
        reports = [  # a page a rank, on its island, for each command
            session.empty((ranks, REPORT // 4), "uint32")
            for _ in range(len(args.sizes) + 2)
        ]
        log(f"tensor {tensor.placement.parts}, word {tensor.address}")
        log(f"range: bank {args.bank} at {base:#x} of island 0, word {region.address}")
        program = km.Program(
            "forms",
            kernels=KERNELS,
            handlers=KERNELS / "forms.h",
            buffers=[
                km.Buffer(
                    index=0, format="bfloat16", page_bytes=64 + DATA + REPORT, pages=1
                )
            ],
            compile_args={"reader": reports[0].accessor_args()},
        )
        run = session.load(program, deadline=2.0)
        log(f"loaded on both islands, {ranks} ranks ({own})")
        commands = [
            ("pages", size, [tensor.address, size, target, count])
            for size in args.sizes
        ]
        commands.append(
            (
                "bytes",
                args.item_page,
                [
                    tensor.address,
                    args.item_page,
                    sources,
                    args.items,
                    0,
                    0,
                    item_target,
                ],
            )
        )
        commands.append(
            ("range", args.range_page, [region.address, args.bank, args.range_page])
        )
        events = []
        for number, (name, size, words) in enumerate(commands):
            words = (words + [0] * 7)[:7]
            words[4], words[5] = reports[number].address, 0
            before = run.crossings()
            ticket = run.submit(program.handler(name), (0, ranks), words)
            parts = run.wait(ticket, timeout=30.0)
            moved = run.crossings() - before
            ms = max(part.ticks for part in parts) / 50e3  # the X280's mtime: 50 MHz
            events.append((ms, moved))
            log(f"{name.upper()} {size} B pages: {ms:.3f} ms")
        try:
            results = run.stop()
            report["kernel_errors"] = [result["error_name"] for result in results]
        except km.KomorebiError as error:
            report["kernel_errors"] = str(error)[:300]
        log(f"stopped: {report['kernel_errors']}")
        checked = {"pages": None, "bytes": args.items * ITEM // 4}
        checked["range"] = 2 * args.range_page // 4 + RANGE_BYTES // 4
        for number, (name, size, _) in enumerate(commands):
            (got,) = session.read(reports[number])
            rows = got.view(np.uint32).reshape(ranks, -1)
            want = count * size // 4 if name == "pages" else checked[name]
            bad, wrong, islands = check_reports(rows, NUMBERS[name], want)
            ms, moved = events[number]
            cycles = rows[:, 6].astype(np.float64)
            row = {
                "handler": name,
                "page_bytes": size,
                "command_ms": round(ms, 3),
                "cycles_mean": round(float(cycles.mean()), 1),
                "cycles_max": int(cycles.max()),
                "words_checked": int(rows[:, 3].astype(np.int64).sum()),
                "words_wrong": wrong,
                "islands": islands,
                "crossed": {
                    f"{a}->{b}": moved.total(a, b) for a, b in ((0, 1), (1, 0))
                },
                "bad_ranks": bad[:8],
            }
            log(
                f"{name} {size} B: wrong {wrong}, bad ranks {len(bad)}, "
                f"{row['words_checked']} words checked"
            )
            report["rows"].append(row)
        for size in args.sizes:
            apply_pages(model, size, ranks, count, target)
        apply_bytes(model, args.item_page, ranks, args.items, sources, item_target)
        log("reading the tensor and the range's tensor back")
        (back,) = session.read(tensor)
        back = from_pages(back.view(np.uint32).reshape(-1, PAGE // 4), PAGE)
        report["tensor_words_wrong"] = int((back != model).sum())
        (region_back,) = session.read(region_tensor)
        region_back = region_back.view(np.uint32).reshape(-1, PAGE // 4)
        want = apply_range(region_data, args.bank, args.range_page, ranks)
        report["region_words_wrong"] = int((region_back != want).sum())
        report["region_other_banks_wrong"] = int(
            sum(
                (region_back[j] != region_data[j]).sum()
                for j in range(region_back.shape[0])
                if j % BANKS != args.bank
            )
        )
        log(
            f"host: tensor words wrong {report['tensor_words_wrong']}, range's "
            f"tensor {report['region_words_wrong']}"
        )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    report["ok"] = bool(
        report["kernel_errors"] == ["none", "none"]
        and all(not row["bad_ranks"] for row in report["rows"])
        and report["tensor_words_wrong"] == 0
        and report["region_words_wrong"] == 0
    )
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
