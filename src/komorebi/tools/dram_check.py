"""A card's DRAM read back against random data (the hazards page, "A chip whose DRAM
reads wrong after a reset"): written and read twice through the host channel, wrong
words counted by bank and bit. Usage: kmr dram-check --card N [--megabytes M] [--all]
"""

import time

import numpy as np

import komorebi as km
from komorebi.device import tensor as tensor_module
from komorebi.errors import ChannelError

PAGE = 4096  # bytes of a row, one page of the tensor
WORDS = PAGE // 4  # 32-bit words a page
MEGABYTES = 256  # the check's default, and the size of each tensor of --all
RULE = "reset the card again (tt-smi -r {card}), then check again"


def _refuse_without_channel(session):
    """ChannelError when the session has no host channel: through the device API a
    read of 256 MB takes about two minutes (2 MB/s)."""
    session._map_channel()  # a session given device= maps it at its first transfer
    if session.channel is None and not session._mock_mapped():
        raise ChannelError(
            "dram-check: this session has no host channel (opened with channel=0, or "
            "the driver mapped no hugepage), so its reads would go through the device "
            "API at about 2 MB/s: open the card with the channel"
        )


def _check_tensor(session, pages, generator):
    """Random words in a row tensor of `pages` 4,096-byte pages, read back twice: the
    result (check_dram's keys) and the tensor, still held."""
    data = generator.integers(0, 1 << 32, size=(pages, WORDS), dtype=np.uint32)
    started = time.perf_counter()
    tensor = session.tensor(data, "uint32", "row")
    write_s = time.perf_counter() - started
    reads, seconds = [], []
    for _ in range(2):
        started = time.perf_counter()
        (raw,) = session.read(tensor)
        seconds.append(time.perf_counter() - started)
        reads.append(np.ascontiguousarray(raw).view(np.uint32).reshape(pages, WORDS))
    flips = [read ^ data for read in reads]
    wrong = [flip != 0 for flip in flips]
    banks = session.dram_banks
    per_page = (wrong[0] | wrong[1]).sum(axis=1)
    by_bank = np.bincount(np.arange(pages) % banks, weights=per_page, minlength=banks)
    bits = np.zeros(32, np.int64)
    for flip, bad in zip(flips, wrong, strict=True):
        words = flip[bad]
        for bit in range(32):
            bits[bit] += int(np.count_nonzero(words & np.uint32(1 << bit)))
    result = {
        "megabytes": pages * PAGE / (1 << 20),
        "words": pages * WORDS,
        "wrong_read1": int(wrong[0].sum()),
        "wrong_read2": int(wrong[1].sum()),
        "reads_differ": int(np.count_nonzero(reads[0] != reads[1])),
        "banks": {bank: int(n) for bank, n in enumerate(by_bank) if n},
        "bank_count": banks,
        "bits": {bit: int(n) for bit, n in enumerate(bits) if n},
        "write_s": write_s,
        "read_s": tuple(seconds),
    }
    return result, tensor


def check_dram(session, megabytes=MEGABYTES, seed=0):
    """`megabytes` MiB of seeded random words in 4,096-byte pages through the host
    channel (ChannelError without one), read twice, freed: wrong words by read, by
    bank (words wrong in either read; page p in bank p % banks) and by bit, words the
    reads differ in, and the seconds."""
    _refuse_without_channel(session)
    pages = int(megabytes * (1 << 20)) // PAGE
    if pages < 1:
        raise ValueError(f"megabytes {megabytes}: at least one page of {PAGE} bytes")
    result, tensor = _check_tensor(session, pages, np.random.default_rng(seed))
    tensor.free()
    return result


def check_all(session, seed=0, megabytes=MEGABYTES, said=None):
    """check_dram over as much DRAM as the allocator gives: tensors of `megabytes` MiB
    held together until one does not fit, then the most pages that still fit; each
    checked as it is made, all freed at the end. The results, each passed to `said`."""
    _refuse_without_channel(session)
    generator = np.random.default_rng(seed)
    pages = int(megabytes * (1 << 20)) // PAGE
    held, results = [], []
    try:
        while pages > 0:
            try:
                result, tensor = _check_tensor(session, pages, generator)
            except tensor_module.DramFull as full:
                pages = min(full.fits, pages - 1)  # the rest, in one last tensor
                continue
            held.append(tensor)
            results.append(result)
            if said is not None:
                said(result)
    finally:
        for tensor in held:
            tensor.free()
    return results


def wrong_words(result):
    """The wrong words of a result in either read."""
    return result["wrong_read1"] + result["wrong_read2"]


def result_lines(result, label):
    """A line a result, then a line a bank with wrong words and one of the bits."""
    read1, read2 = result["read_s"]
    lines = [
        f"{label}: {result['megabytes']:g} MiB, wrong words {result['wrong_read1']} "
        f"(read 1), {result['wrong_read2']} (read 2), {result['reads_differ']} differ "
        f"between the reads; write {result['write_s']:.2f} s, reads {read1:.2f} s "
        f"and {read2:.2f} s"
    ]
    for bank, count in result["banks"].items():
        lines.append(f"  bank {bank} of {result['bank_count']}: {count} wrong words")
    if result["bits"]:
        bits = ", ".join(f"bit {bit}: {n}" for bit, n in result["bits"].items())
        lines.append(f"  wrong bits: {bits}")
    return lines


def run_cli(args, out):
    """`kmr dram-check`: opens the card, checks, prints; 1 on any wrong word."""
    started = time.perf_counter()
    with km.Session(card=args.card) as session:
        if args.all:
            results = check_all(
                session,
                args.seed,
                args.megabytes,
                lambda result: print(
                    "\n".join(result_lines(result, f"card {args.card}")),
                    file=out,
                    flush=True,
                ),
            )
        else:
            results = [check_dram(session, args.megabytes, args.seed)]
            print("\n".join(result_lines(results[0], f"card {args.card}")), file=out)
    total = sum(result["megabytes"] for result in results)
    wrong = sum(wrong_words(result) for result in results)
    print(
        f"card {args.card}: {total:g} MiB in {len(results)} tensor(s), {wrong} wrong "
        f"words, {time.perf_counter() - started:.1f} s with the open",
        file=out,
    )
    if wrong:
        print(f"card {args.card}: " + RULE.format(card=args.card), file=out)
        return 1
    return 0
