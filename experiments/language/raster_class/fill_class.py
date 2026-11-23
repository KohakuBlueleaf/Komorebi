"""The rasterizer's fill pass (FILL) as a class: each core's directory words, its list
pages streamed to the pack CPU, the hand program's fill on the vector unit as inline
bodies, the empty tile's planes as a kept field. README.md says what each line lowers
to and which moves are still bodies."""

import komorebi as km
from projects.raster import card

PLANES = 5  # a tile's planes (kPlanes), one 4096-byte page each
# The fill's compute options, the hand program's: the pack CPU's vector unit.
FILL = {"math_fidelity": "HiFi4", "fp32_dest_acc_en": True, "enable_trisc2_rvv": True}


# Mover B: one item of the directory written to the frame (the hand's writer.cpp).
ITEM = """
const uint32_t bases[kDirFrameMax] = {
    frames_base0, frames_base1, frames_base2, frames_base3, frames_base4,
    frames_base5, frames_base6, frames_base7, frames_base8, frames_base9,
    frames_base10, frames_base11, frames_base12, frames_base13, frames_base14
};
const uint32_t tensor_pages = frames_pages;
const bool is_empty = (item_id & kItemEmpty) != 0;
const uint32_t asked = km::wall();
if (!is_empty && !movement.wait_front(out, kPlanes)) {
    return true;
}
const uint32_t ready = km::wall();
const uint32_t page = (item_id & kItemSlot) * kPlanes;
if (page / tensor_pages >= kDirFrameMax) {
    bad_item(kWhereBadSend);
    return false;
}
const uint32_t first_row = item_id >> kItemRow0Shift & kItemRowMask;
const uint32_t end_row = item_id >> kItemRow1Shift & kItemRowMask;
const uint32_t byte_offset = first_row * kRowBytes;
const uint32_t bytes = (
    end_row > 0 ? (end_row - first_row) * kRowBytes : kOutPageBytes
);
uint32_t source = (is_empty ? empty : get_read_ptr(out)) + byte_offset;
if (bytes == kOutPageBytes) {  // whole pages: the page calls
    const Frame chunk(bases[page / tensor_pages]);
    for (uint32_t plane = 0; plane < kPlanes; ++plane) {
        if (!km::write_pages(
            movement, command, chunk, (page + plane) % tensor_pages, 1, source
        )) {
            return false;
        }
        source += kOutPageBytes;
    }
} else if constexpr (Frame::kLocal) {  // a band: part of each page
    for (uint32_t plane = 0; plane < kPlanes; ++plane) {
        const InterleavedAddrGen<true> frame_tensor = {
            .bank_base_address = bases[(page + plane) / tensor_pages],
            .page_size = kOutPageBytes
        };
        noc_async_write(
            source,
            get_noc_addr((page + plane) % tensor_pages, frame_tensor, byte_offset),
            bytes
        );
        source += kOutPageBytes;
    }
} else {  // a band over islands: no call writes part of a page
    bad_item(kWhereBandFar);
    return false;
}
if (!noc_done(
    [] { return ncrisc_noc_nonposted_writes_sent(noc_index); }, kWhereFlush
)) {
    return false;
}
if (!is_empty) {
    cb_pop_front(out, kPlanes);
}
writer_wait += ready - asked;
writer_write += km::wall() - ready;
"""


class DirHead(km.Entry):
    """A core's directory page, words 0 to 2: its first list page, their count and
    its items' count."""

    first: int
    pages: int
    items: int


class DirItem(km.Entry):
    """An item of the directory page (from word 3): its tile, empty bit and rows."""

    id: int


class DirFrames(km.Entry):
    """The frame's tensors (from word 1,000): pages each, then each one's address."""

    pages: int
    base0: int
    base1: int
    base2: int
    base3: int
    base4: int
    base5: int
    base6: int
    base7: int
    base8: int
    base9: int
    base10: int
    base11: int
    base12: int
    base13: int
    base14: int


# the default frame's tiles and tables (kernels/grid.h)
class FillPass(km.Program, compute=FILL, defines=card.capacity_defines()):
    """load_empty: the empty tile's planes into their field, kept. fill_tiles: the hand
    program's FILL on every core: its list pages walked on the pack CPU into each
    item's planes, written into the frame's pages, then its profile page."""

    empty: km.L1[km.u32, PLANES, 32, 32]  # the empty tile's planes
    out: km.L1[km.u32, 3 * PLANES, 32, 32] = km.own(push="pack")  # filled items
    handoff: km.L1[km.u32, 1, 32, 32] = km.own(push="mover_a")  # the reader's words

    @km.handler
    def load_empty(self, planes: km.Tensor):
        self.empty.fill(km.Stream(planes.pages(0, PLANES)))

    @km.handler
    def fill_tiles(
        self,
        directory: km.Tensor,
        lists: km.Tensor,
        frame: km.Tensor,
        profile: km.Tensor,
    ):
        km.cxx(
            """
            #include "../../../../../projects/raster/kernels/raster.h"
            const uint32_t reader_started = km::wall();
            """,
            on="mover_a",
        )
        km.cxx(
            """
            const uint32_t writer_started = km::wall();
            uint32_t writer_wait = 0, writer_write = 0, writer_items = 0;
            """,
            on="mover_b",
        )
        for head in directory.pages(self.rank, 1).entries(DirHead, size=3, fixed=1):
            # The fill's state on the pack CPU, seen by its bodies below: the hand's
            # fill_list, its slot walk resumed at each list page.
            km.cxx(
                """
                constexpr uint32_t background[3] = {0x3D4CCCCD, 0x3D4CCCCD, 0x3DA3D70A};
                // A give-up into the writer's gave-up word (the first one is kept) and
                // the stop word, as the hand's compute.
                auto give_up = [&](uint32_t code, uint32_t where) {
                    volatile uint32_t* state = reinterpret_cast<volatile uint32_t*>(
                        compute.control() + KM_CONTROL_STATE
                    );
                    if (state[km::kStWriter] == 0) {
                        state[km::kStWriter] = code | where << 8;
                    }
                    *reinterpret_cast<volatile uint32_t*>(
                        compute.control() + KM_CONTROL_STOP
                    ) = 1;
                };
                // Rows [first_row, end_row) of each plane to the same place in the
                // item at `destination`, from the Dst window.
                auto copy_band = [&](uint32_t destination, uint32_t first_row,
                                     uint32_t end_row) {
                    const uint32_t first_word = kRowBytes / 4 * first_row;
                    const uint32_t words = kRowBytes / 4 * (end_row - first_row);
                    for (uint32_t plane = 0; plane < kPlanes; ++plane) {
                #if defined(__riscv_vector)
                        fill::copy_words_rvv(
                            reinterpret_cast<uint32_t*>(
                                destination + 4 * (plane * kPlaneWords + first_word)
                            ),
                            const_cast<const uint32_t*>(
                                fill::win() + plane * kPlaneWords + first_word
                            ),
                            words
                        );
                #else
                        fill::copy_words(
                            reinterpret_cast<volatile uint32_t*>(
                                destination + 4 * (plane * kPlaneWords + first_word)
                            ),
                            fill::win() + plane * kPlaneWords + first_word,
                            words
                        );
                #endif
                    }
                };
                fill::ProfileCounters counters;
                fill::setup();
                fill::edges(true);
                const uint32_t fill_started = fill::wall();
                uint32_t page_done = fill_started;
                uint32_t tile_x = 0, tile_y = 0, records_left = 0;
                int32_t first_row = 0, end_row = 32;
                bool listing = true, ended = false;
                // The item's planes out of Dst into the next 5 pages of `out`, pushed.
                auto tile_out = [&](uint32_t out) {
                    uint32_t before = fill::wall();
                    ckernel::tensix_sync();
                    uint32_t after = fill::wall();
                    counters.drain += after - before;
                    if (!compute.reserve_back(out, kPlanes)) {
                        return false;
                    }
                    before = fill::wall();
                    counters.wait_out += before - after;
                    const uint32_t destination = get_local_cb_interface(out).fifo_wr_ptr
                        << 4;
                    copy_band(destination, first_row, end_row);
                    const uint32_t words = (
                        (kPlanes - 1) * kPlaneWords + kRowBytes / 4 * end_row
                    );
                    asm volatile("fence" ::: "memory");
                    const uint32_t last_word = (
                        reinterpret_cast<volatile uint32_t*>(destination)[words - 1]
                    );
                    asm volatile("and x0, x0, %0" : : "r"(last_word) : "memory");
                    cb_push_back(out, kPlanes);
                    counters.out += fill::wall() - before;
                    counters.tiles++;
                    return true;
                };
                """,
                on="pack",
                include=["../../../../../projects/raster/kernels/fill.hpp"],
            )
            km.cxx("const uint32_t reader_moving = km::wall();", on="mover_a")
            for page in km.Stream(lists.pages(head.first, head.pages), pattern="A"):
                km.cxx(
                    """
                    asm volatile("fence" ::: "memory");
                    counters.wait_in += fill::wall() - page_done;
                    const volatile uint32_t* slots = (
                        reinterpret_cast<const volatile uint32_t*>(page)
                    );
                    for (uint32_t slot = 0; listing && slot < kSlotsPerPage; ++slot) {
                        const volatile uint32_t* header = slots + kSlotWords * slot;
                        if (records_left == 0) {  // a tile header, or the end
                            if (header[0] == kMarkEnd) {
                                listing = false;
                                ended = true;
                                break;
                            }
                            const uint32_t tile = header[kHdrTile];
                            records_left = header[kHdrCount];
                            first_row = header[kHdrRow0];
                            end_row = header[kHdrRow1];
                            if (!(0 <= first_row && first_row < end_row &&
                                  end_row <= 32 && tile < 1024)) {
                                give_up(kGaveBadItem, kWhereBadHeader);
                                listing = false;
                                break;
                            }
                            tile_x = tile % 32;
                            tile_y = tile / 32;
                            const uint32_t before = fill::wall();
                            fill::init_tile(tile_x, tile_y, background);
                            counters.init += fill::wall() - before;
                            if (records_left == 0 && !tile_out(out)) {
                                listing = false;
                            }
                            continue;
                        }
                        const volatile uint32_t* record = header;
                        uint32_t before = fill::wall();
                        uint32_t record_words[kSlotWords];
                    #if defined(__riscv_vector)
                        fill::copy_words_rvv(
                            record_words,
                            const_cast<const uint32_t*>(record),
                            kSlotWords
                        );
                    #else
                        fill::fetch(record_words, record);
                    #endif
                        const int32_t box_first_row = (
                            static_cast<int32_t>(record_words[kRecY0]) - 32 * tile_y
                        );
                        const int32_t box_last_row = (
                            static_cast<int32_t>(record_words[kRecY1]) - 32 * tile_y
                        );
                        const int32_t first_covered = (
                            box_first_row < first_row ? first_row : box_first_row
                        );
                        const int32_t last_covered = (
                            box_last_row > end_row - 1 ? end_row - 1 : box_last_row
                        );
                        fill::load_triangle(record_words);
                        uint32_t after = fill::wall();
                        counters.triangle_setup += after - before;
                        for (
                            int32_t pixel_row = first_covered;
                            pixel_row <= last_covered;
                            ++pixel_row
                        ) {
                            fill::row(pixel_row);
                            counters.row_count++;
                        }
                        counters.rows += fill::wall() - after;
                        counters.triangles++;
                        if (--records_left == 0 && !tile_out(out)) {
                            listing = false;
                        }
                    }
                    page_done = fill::wall();
                    """,
                    takes=[page, self.out],
                    on="pack",
                )
            # The reader's words (wait, read, total, pages) to mover B by the own field.
            km.cxx(
                """
                const uint32_t span = km::wall() - reader_moving;
                if (cb_pages_reservable_at_back(handoff, 1)) {
                    cb_reserve_back(handoff, 1);
                    volatile tt_l1_ptr uint32_t* words = km::l1(get_write_ptr(handoff));
                    words[0] = span - page_cycles;
                    words[1] = page_cycles;
                    words[2] = km::wall() - reader_started;
                    words[3] = page_tiles;
                    cb_push_back(handoff, 1);
                }
                """,
                takes=[page, self.handoff],
                on="mover_a",
            )
            # Mover B: the hand's writer loop, an item of the directory at a time
            # (ITEM: its planes, or the empty tile's, into the frame's pages).
            km.cxx(
                """
                auto noc_done = [&](auto ready, uint32_t where) {
                    const uint32_t code = movement.poll(
                        ready, movement.args().wait_limit
                    );
                    if (code == km::kGaveNone) {
                        return true;
                    }
                    movement.gave_up(km::kStWriter, code, where, 0, command.sequence);
                    return false;
                };
                auto bad_item = [&](uint32_t where) {
                    movement.gave_up(
                        km::kStWriter, kGaveBadItem, where, 0, command.sequence
                    );
                    *movement.control(KM_CONTROL_STOP) = 1;
                };
                """,
                on="mover_b",
            )
            for frames in directory.pages(self.rank, 1).entries(
                DirFrames, first=1000, size=16, fixed=1
            ):
                km.cxx(
                    """
                    if (frames_pages == 0 || frames_pages % kPlanes != 0 ||
                        head_items > kDirMaxItems) {
                        bad_item(kWhereBadSend);
                        return false;
                    }
                    """,
                    takes=[head, frames],
                    on="mover_b",
                )
                for item in directory.pages(self.rank, 1).entries(
                    DirItem, first=3, size=1, count=2
                ):
                    km.cxx(ITEM, takes=[item, frames, frame, self.out, self.empty],
                           on="mover_b")  # fmt: skip
                km.cxx("writer_items = head_items;", takes=[head], on="mover_b")
            report = km.Stream(profile.pages(self.rank, 1), write=True, pattern="B")
            # The pack CPU's counters at words 0 to 10 of the profile page, as the
            # hand's.
            km.cxx(
                """
                fill::finish();
                if (listing && !ended) {  // the list's pages ended before its end slot
                    give_up(kGaveWait, kWhereWaitPage);
                }
                counters.total = fill::wall() - fill_started;
                if (report != 0) {
                    volatile uint32_t* words = (
                        reinterpret_cast<volatile uint32_t*>(report)
                    );
                    const uint32_t values[11] = {
                        counters.wait_in,
                        counters.init,
                        counters.triangle_setup,
                        counters.rows,
                        counters.drain,
                        counters.out,
                        counters.wait_out,
                        counters.tiles,
                        counters.triangles,
                        counters.row_count,
                        counters.total
                    };
                    for (uint32_t word = 0; word < kOutPageBytes / 4; ++word) {
                        words[word] = word < 11 ? values[word] : 0;
                    }
                    asm volatile("fence" ::: "memory");
                    const uint32_t last = words[kOutPageBytes / 4 - 1];
                    asm volatile("and x0, x0, %0" : : "r"(last) : "memory");
                }
                """,
                takes=[report],
                on="pack",
            )
            # Mover B's page: the reader's words, then the writer's (wait, write, total,
            # items), as the hand's writer adds them; the movers write it after this.
            km.cxx(
                """
                volatile tt_l1_ptr uint32_t* words = km::l1(report);
                const uint64_t limit = km::wall64() + (1ull << 30);
                bool handed = cb_pages_available_at_front(handoff, 1);
                while (!handed && km::wall64() < limit) {
                    handed = cb_pages_available_at_front(handoff, 1);
                }
                if (handed) {
                    cb_wait_front(handoff, 1);
                    invalidate_l1_cache();
                    volatile tt_l1_ptr uint32_t* reader = km::l1(get_read_ptr(handoff));
                    for (uint32_t word = 0; word < 4; ++word) {
                        words[kProfReader + word] = reader[word];
                    }
                    cb_pop_front(handoff, 1);
                }
                words[kProfWriter + 0] = writer_wait;
                words[kProfWriter + 1] = writer_write;
                words[kProfWriter + 2] = km::wall() - writer_started;
                words[kProfWriter + 3] = writer_items;
                """,
                takes=[report, self.handoff],
                on="mover_b",
            )


def made(session, tensors, log):
    """The class made with example tensors of each name, one command at a time as
    the hand program."""
    return FillPass(log=log, session=session, depth=1, **tensors)
