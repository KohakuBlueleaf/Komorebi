"""A tensor's bytes between the host and the card: pages split into pieces, the
dispatcher's state area and control bytes, the transfer chunk, reads by DRAM banks."""

import numpy as np

from komorebi import abi
from komorebi.device import channel as channel_module
from komorebi.device import chips, dram
from komorebi.errors import StartError
from komorebi.program import overlay
from komorebi.program import several as programs

CONTROL_ALIGN = 64  # the dispatcher stores whole 64-byte lines into the control block


IPI_ACK_DDR = 0x470020  # the X280 runtime's IPI ack word in the cluster's DRAM bank


STATE_PAGES_A_BANK = 4  # pages of KMD_STATE_BYTES in each DRAM bank (interleaved)


DRAM_PIECE_ALIGN = 64  # a piece of a DRAM page written apart starts this aligned


def _state_page():
    """The page of the dispatcher's state area (_state_area): KMD_STATE_BYTES."""
    return abi.headers()["KMD_STATE_BYTES"] // 4 * 4


def _state_area(session):
    """A DRAM buffer whose bank address holds the dispatcher's state in every
    bank (KMD_STATE_BYTES at least); rejected if it covers the X280 runtime's IPI word
    (a chip with an X280: chips.x280)."""
    abi_headers = abi.headers()
    # each bank holds what the kernel keeps cached and flushes (KMD_STATE_CACHED)
    if STATE_PAGES_A_BANK * _state_page() < abi_headers["KMD_STATE_CACHED"]:
        raise StartError(
            f"{STATE_PAGES_A_BANK} state pages a bank: under KMD_STATE_CACHED"
        )
    pages = STATE_PAGES_A_BANK * session.dram_banks  # the same pages in every bank
    rows = programs.words_tensor(session, np.zeros((pages, _state_page() // 4)))
    address = rows.address
    covers = address <= IPI_ACK_DDR < address + pages * abi_headers["KMD_STATE_BYTES"]
    if covers and chips.chip(session.arch).x280:
        raise StartError(f"the kernel state at {address:#x} covers the X280 IPI word")
    return rows


def _control_bytes(most=0):
    """A core's control block and overlay area (Run._control_blocks's words), the
    records of `most` slots of a part past 4 after it."""
    abi_headers = abi.headers()
    area = overlay.area_bytes(most)
    return (abi_headers["KM_CONTROL_BYTES"] + area + CONTROL_ALIGN) // 4 * 4


def write_pieces(aligned, chunk):
    """The pieces a DRAM page of `aligned` bytes is written in while a run lives, its
    transfer moving at most `chunk` bytes a page: 1 when it fits, else the fewest equal
    pieces of whole DRAM_PIECE_ALIGN units within the chunk; None when none are."""
    if aligned <= chunk:
        return 1
    for pieces in range(2, aligned // DRAM_PIECE_ALIGN + 1):
        piece = aligned // pieces
        if aligned % pieces == 0 and piece % DRAM_PIECE_ALIGN == 0 and piece <= chunk:
            return pieces
    return None


def split_pages(first, rows, pieces, banks):
    """Pages first .. of an interleaved DRAM tensor (`rows`, [n, aligned] uint8) as
    the same bytes in pages of aligned / pieces: piece j of page p is page
    banks * ((p // banks) * pieces + j) + p % banks of that view (a bank holds its pages
    back to back). [(first page of the view, [m, piece] rows)], each a run of them."""
    count, aligned = rows.shape
    pages = first + np.arange(count)
    at = (
        banks * ((pages // banks)[:, None] * pieces + np.arange(pieces))
        + (pages % banks)[:, None]
    )
    order = np.argsort(at, axis=None, kind="stable")
    at = at.reshape(-1)[order]
    pieced = rows.reshape(count * pieces, aligned // pieces)[order]
    cuts = np.flatnonzero(np.diff(at) != 1) + 1
    return [
        (int(run[0]), part)
        for run, part in zip(np.split(at, cuts), np.split(pieced, cuts), strict=True)
    ]


def _transfer_chunk_of(largest_page, l1, asked, held, switch, trace_cap, control_bytes):
    """The chunk of a run's transfer buffer: CHUNK_DEFAULT, `asked` where it fits, or
    the largest DRAM page (the session's tensors' and the state area's) when two take
    at most half of what a core has left after its buffers (`l1`: the session's). A run
    with overlays, or whose groups switch program, keeps the default: slots to place."""
    page = max(largest_page, _state_page())
    if switch or any(each.overlays for each in held):
        return channel_module.CHUNK_DEFAULT
    buffers = max(
        sum(buffer.page_bytes * buffer.pages for buffer in each.buffers)
        for each in held
    )
    buffers += 2 * abi.headers()["kTokenBytes"] + trace_cap * 16
    spare = l1()[1] - buffers - control_bytes  # `asked`: over two cards LINK_CHUNK
    word = abi.headers()["KM_TRANSFER_WORD_BYTES"]
    if asked and asked >= page and 2 * asked + word <= spare // 2:
        return asked
    if page <= channel_module.CHUNK_DEFAULT:
        return channel_module.CHUNK_DEFAULT
    return channel_module.chunk_for(page, spare // 2) or channel_module.CHUNK_DEFAULT


def read_tensors(device, module, tensors, first=0, pages=None):
    """The raw pages first .. first + pages - 1 (pages=None: to the end) of interleaved
    DRAM tensors as [pages, page_bytes] uint8 each: one DRAM barrier, one read per bank
    and tensor, no wait for idle cores."""
    banks = module.dram_banks(device)
    spans, layout = [], []
    for tensor in tensors:
        if not tensor.in_dram:
            raise ValueError("only interleaved DRAM tensors are read by banks")
        page, aligned = tensor.page_bytes, tensor.aligned_page_bytes
        start_page, count = dram.page_range(first, pages, tensor.pages)
        mine = dram.bank_spans(tensor.address, aligned, count, banks, start_page)
        layout.append((len(spans), len(mine), page, aligned, count))
        spans += mine
    chunks = module.read_dram(device, spans)
    return [
        dram.pages_in_order(
            chunks[first_span : first_span + span_count], page, aligned, pages, banks
        )
        for first_span, span_count, page, aligned, pages in layout
    ]


def _read_uint32(tensor, device=None, module=None):
    """A row-major 32-bit tensor's words as int64: by banks when it is in DRAM and a
    module is given, else as one buffer read (which waits for the launched cores)."""
    if module is not None and tensor.in_dram:
        (pages,) = read_tensors(device, module, [tensor])
        words = pages.reshape(-1).view(np.uint32).reshape(tuple(tensor.shape))
        return words.astype(np.int64)
    return tensor.read_bytes().view(np.uint32).reshape(tensor.shape).astype(np.int64)
