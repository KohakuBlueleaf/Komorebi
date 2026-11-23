"""The rasterizer's per-triangle passes (COUNT, HIST, SETUP) as a class: the camera's
page and each core's part of the mesh blocks streamed to the pack CPU, the hand
program's vector-unit setup and its pass as inline bodies, the writer's counting and
sending as bodies on mover B. README.md says which moves are still bodies."""

import komorebi as km

# The passes' compute options, the hand program's: the pack CPU's vector unit.
PASSES = {"fp32_dest_acc_en": True, "enable_trisc2_rvv": True}
# The hand's setup_compute.cpp on the pack CPU: its helpers alone, its out buffer the
# class's field `out` (buffer 10, after HIST's table's two; a start body checks it).
PASS_HELPERS = {"RASTER_PASS_HELPERS": 1, "RASTER_NO_ENTRY": 1, "RASTER_OUT_CB": 10}
HEADERS = """
#include "../../../../../projects/raster/kernels/setup.h"
#include "../../../../../projects/raster/kernels/setup_noc.hpp"
"""
# The camera's page into every lane of its slots, and each lane's number (the hand's
# first page of a pass, setup_compute.cpp).
CAMERA = """
const uint32_t* camera_words = reinterpret_cast<const uint32_t*>(camera_page);
asm volatile("fence" ::: "memory");
for (uint32_t i = 0; i < kCamWords; ++i) {
    setup_pass::fill_slot(setup_pass::kSlotCamera + i, camera_words[i]);
}
setup_pass::order(setup_pass::word(setup_pass::kSlotCamera + kCamWords - 1) + 62);
const uint32_t lane_numbers = setup_pass::word(setup_pass::kSlotLaneNumber);
for (uint32_t lane = 0; lane < 32; ++lane) {
    setup_pass::window()[lane_numbers + 2 * lane] = float_bits(lane);
}
setup_pass::order(lane_numbers + 62);
"""
# One block of 32 triangles on the vector unit, then the hand's pass over its lanes.
BLOCK = """
if (pass.succeeded) {
    const uint32_t* mesh_words = reinterpret_cast<const uint32_t*>(block);
    asm volatile("fence" ::: "memory");
    setup_pass::block_in(mesh_words);
    setup_pass::camera_space();
    setup_pass::triangles(pass.handler != kHandlerCount);
    const uint32_t first_triangle = 32 * (first + block_index);
    setup_pass::finish_lanes(
        pass.handler == kHandlerSetup, float_bits(first_triangle)
    );
    ckernel::tensix_sync();
    pass.block_finished(mesh_words, cb_ptr(work), first_triangle);
}
++block_index;
"""
# After the blocks: the last out page tells mover B to stop.
END = """
if (pass.succeeded) {
    pass.out.end(compute);
}
setup_pass::finish();
"""
# Mover B's give-up, as the hand's writer records it.
GIVE_UP = """
auto give_up = [&](uint32_t code, uint32_t where) {
    volatile tt_l1_ptr uint32_t* state = (
        km::l1(movement.args().control + KM_CONTROL_STATE)
    );
    if (state[km::kStWriter] == 0) {
        state[km::kStWriter] = code | where << 8;
    }
    *movement.control(KM_CONTROL_STOP) = 1;
};
auto flushed = [&] {
    return movement.poll(
        [] { return ncrisc_noc_nonposted_writes_flushed(noc_index); },
        movement.args().wait_limit
    ) == km::kGaveNone;
};
"""


class Heavy(km.Entry):
    """A tile HIST writes: its number (the heavy page's words from word 1, word 0
    their count)."""

    tile: int


class SetupPasses(km.Program, compute=PASSES, defines=PASS_HELPERS):
    """count_tiles, hist_tiles and setup_lists: the hand program's COUNT, HIST and
    SETUP on every core, each over its part of the mesh blocks (rank x blocks /
    size up to the next rank's), as the hand's command words name them."""

    out: km.L1[km.u32, 4, 32, 32] = km.own(push="pack")  # pack to mover B
    work: km.L1[km.u32, 3, 32, 32] = km.own(push=None)  # the pack CPU's space
    counts: km.L1[km.u32, 2, 32, 32] = km.own(push=None)  # entries, rows a tile
    hist: km.L1[km.u32, 16, 32, 32] = km.own(push=None)  # COUNT's, kept for HIST
    next_slots: km.L1[km.u32, 2, 32, 32]  # SETUP: each item's next slot
    item_table: km.L1[km.u32, 2, 32, 32]  # SETUP: each tile's items and their rows

    @km.handler
    def count_tiles(
        self, mesh: km.Tensor, counters: km.Tensor, blocks: int, camera: km.Tensor
    ):
        # Once a kernel: the hand's pack helpers, and mover B's give-up and flush.
        km.cxx(
            'static_assert(out == RASTER_OUT_CB, "the out field is RASTER_OUT_CB");',
            takes=[self.out],
            on="pack",
            include=["../../../../../projects/raster/kernels/setup_compute.cpp"],
            at="start",
        )
        km.cxx(HEADERS + GIVE_UP, on="mover_b", at="start")
        me = self.rank
        first = self.rank * blocks // self.size
        end = (self.rank + 1) * blocks // self.size
        km.cxx(
            """
            setup_pass::setup();
            Pass pass{compute, kHandlerCount};
            uint32_t block_index = 0;
            """,
            on="pack",
        )
        for camera_page in km.Stream(camera.pages(0, 1), pattern="A"):
            km.cxx(CAMERA, takes=[camera_page], on="pack")
        for block in km.Stream(mesh.pages(first, end - first), pattern="A"):
            km.cxx(BLOCK, takes=[block, first, self.work], on="pack")
        km.cxx(END, on="pack")
        # Mover B: the hand's count_stage. Per tile the entries and rows, and the
        # 2-row histograms kept for HIST, from the pack CPU's box words.
        km.cxx(
            """
            volatile tt_l1_ptr uint32_t* counter_words = km::l1(get_write_ptr(counts));
            volatile tt_l1_ptr uint32_t* histogram_words = km::l1(get_write_ptr(hist));
            volatile tt_l1_ptr uint16_t* histogram = (
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(histogram_words)
            );
            for (uint32_t i = 0; i < 2 * kTiles; ++i) {
                counter_words[i] = 0;
            }
            for (uint32_t i = 0; i < kHistBytes / 4; ++i) {
                histogram_words[i] = 0;
            }
            for (;;) {
                if (!movement.wait_front(out, 1, 4)) {
                    return true;
                }
                invalidate_l1_cache();
                volatile tt_l1_ptr uint32_t* page_words = km::l1(get_read_ptr(out));
                const uint32_t head = page_words[0], words = head & ~kOutLast;
                for (uint32_t i = 0; i < words && i < kOutWords; ++i) {
                    const uint32_t box_word = page_words[1 + i];
                    const uint32_t first_tile_x = box_word & 0x3F;
                    const uint32_t last_tile_x = box_word >> 6 & 0x3F;
                    const uint32_t first_row = box_word >> 12 & 0x3FF;
                    const uint32_t last_row = box_word >> 22 & 0x3FF;
                    for (
                        uint32_t tile_y = first_row >> 5; tile_y <= last_row >> 5;
                        ++tile_y
                    ) {
                        const uint32_t top = 32 * tile_y;
                        const uint32_t row0 = first_row > top ? first_row : top;
                        const uint32_t row1 = last_row < top + 31 ? last_row : top + 31;
                        for (
                            uint32_t tile_x = first_tile_x; tile_x <= last_tile_x;
                            ++tile_x
                        ) {
                            const uint32_t tile = kTilesX * tile_y + tile_x;
                            counter_words[tile] += 1;
                            counter_words[kTiles + tile] += row1 - row0 + 1;
                            volatile tt_l1_ptr uint16_t* tile_histogram = (
                                histogram + 32 * tile
                            );
                            const uint16_t starts = (
                                tile_histogram[(row0 - top) >> 1] + 1
                            );
                            tile_histogram[(row0 - top) >> 1] = starts;
                            const uint16_t ends = (
                                tile_histogram[16 + ((row1 - top) >> 1)] + 1
                            );
                            tile_histogram[16 + ((row1 - top) >> 1)] = ends;
                            if (starts == 0 || ends == 0) {
                                give_up(kGaveBadHist, tile);
                                cb_pop_front(out, 1);
                                return true;
                            }
                        }
                    }
                }
                cb_pop_front(out, 1);
                if (head & kOutLast) {
                    break;
                }
            }
            asm volatile("fence" ::: "memory");
            if (!km::write_pages(
                movement, command, counters, me, 1, get_write_ptr(counts)
            ) || !flushed()) {
                return false;
            }
            """,
            takes=[counters, me, self.out, self.counts, self.hist],
            on="mover_b",
        )

    @km.handler
    def hist_tiles(self, heavy: km.Tensor, histograms: km.Tensor, page_bytes: int):
        me = self.rank
        # Mover B: the hand's HIST. Each heavy tile's 64 bytes of the histograms COUNT
        # kept, to this core's page of histograms at the tile's place in the list.
        km.cxx("uint32_t heavy_index = 0;", on="mover_b")
        for cut in heavy.pages(0, 1).entries(Heavy, first=1, size=1, count=0):
            km.cxx(
                """
                if (64 * (heavy_index + 1) > page_bytes) {
                    give_up(kGaveBadHeavy, heavy_index + 1);
                    return true;
                }
                noc_async_write(
                    get_write_ptr(hist) + 64 * (cut_tile % kTiles),
                    setup_pass::noc_address(
                        0, histograms.word, me, page_bytes, 64 * heavy_index
                    ),
                    64
                );
                ++heavy_index;
                """,
                takes=[cut, histograms, page_bytes, me, self.hist],
                on="mover_b",
            )
        km.cxx("if (!flushed()) {\n    return false;\n}", on="mover_b")

    @km.handler
    def setup_lists(
        self,
        mesh: km.Tensor,
        lists: km.Tensor,
        blocks: int,
        camera: km.Tensor,
        slot_pages: km.Tensor,
        list_slots: int,
        item_words: km.Tensor,
    ):
        first = self.rank * blocks // self.size
        end = (self.rank + 1) * blocks // self.size
        self.next_slots.fill(km.Stream(slot_pages.pages(2 * self.rank, 2)))
        self.item_table.fill(km.Stream(item_words.pages(0, 2)))
        km.cxx(
            """
            setup_pass::setup();
            Pass pass{compute, kHandlerSetup};
            uint32_t block_index = 0;
            """,
            on="pack",
        )
        for camera_page in km.Stream(camera.pages(0, 1), pattern="A"):
            km.cxx(CAMERA, takes=[camera_page], on="pack")
        for block in km.Stream(mesh.pages(first, end - first), pattern="A"):
            km.cxx(BLOCK, takes=[block, first, self.work], on="pack")
        km.cxx(END, on="pack")
        # Mover B: the hand's setup_stage. Each record of the pack CPU's pages to the
        # next slot of every item (a tile, or a band of its rows) its box reaches.
        km.cxx(
            """
            volatile tt_l1_ptr uint32_t* next_slot = km::l1(next_slots);
            volatile tt_l1_ptr uint32_t* items = km::l1(item_table);
            const volatile tt_l1_ptr uint16_t* rows = (
                reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(
                    item_table + kItemRows
                )
            );
            const InterleavedAddrGen<true> list_pages = {
                .bank_base_address = lists.word, .page_size = kListPageBytes
            };
            bool succeeded = true;
            for (;;) {
                if (!movement.wait_front(out, 1, 4)) {
                    return true;
                }
                invalidate_l1_cache();
                const uint32_t page = get_read_ptr(out);
                volatile tt_l1_ptr uint32_t* page_words = km::l1(page);
                const uint32_t head = page_words[0], entries = head & ~kOutLast;
                for (uint32_t i = 0; i < entries && i < kOutEntries && succeeded; ++i) {
                    const uint32_t box_word = page_words[1 + i];
                    const uint32_t first_tile_x = box_word & 0x3F;
                    const uint32_t last_tile_x = box_word >> 6 & 0x3F;
                    const uint32_t first_row = box_word >> 12 & 0x3FF;
                    const uint32_t last_row = box_word >> 22 & 0x3FF;
                    for (
                        uint32_t tile_y = first_row >> 5;
                        tile_y <= last_row >> 5 && succeeded;
                        ++tile_y
                    ) {
                        const uint32_t top = 32 * tile_y;
                        const uint32_t row0 = first_row > top ? first_row : top;
                        const uint32_t row1 = last_row < top + 31 ? last_row : top + 31;
                        for (
                            uint32_t tile_x = first_tile_x;
                            tile_x <= last_tile_x && succeeded;
                            ++tile_x
                        ) {
                            const uint32_t tile = kTilesX * tile_y + tile_x;
                            const uint32_t first_item = items[tile] & 0xFFFF;
                            const uint32_t item_count = items[tile] >> 16;
                            if (first_item + item_count > kSlotsBytes / 4) {
                                give_up(kGaveBadSlot, tile);
                                succeeded = false;
                                break;
                            }
                            for (
                                uint32_t item = first_item;
                                item < first_item + item_count;
                                ++item
                            ) {
                                const uint32_t item_first_row = rows[item] & 0xFF;
                                const uint32_t item_end_row = rows[item] >> 8;
                                if (
                                    ((row0 - top) >> 1) >= (item_end_row >> 1) ||
                                    ((row1 - top) >> 1) < (item_first_row >> 1)
                                ) {
                                    continue;
                                }
                                const uint32_t entry_slot = next_slot[item];
                                next_slot[item] = entry_slot + 1;
                                if (entry_slot >= list_slots) {
                                    give_up(kGaveBadSlot, tile);
                                    succeeded = false;
                                    break;
                                }
                                noc_async_write(
                                    page + 128 * (1 + i),
                                    get_noc_addr(
                                        entry_slot / 16,
                                        list_pages,
                                        (entry_slot % 16) * 128
                                    ),
                                    128
                                );
                            }
                        }
                    }
                }
                succeeded = flushed() && succeeded;
                cb_pop_front(out, 1);
                if (!succeeded) {
                    return false;
                }
                if (head & kOutLast) {
                    break;
                }
            }
            """,
            takes=[lists, list_slots, self.out, self.next_slots, self.item_table],
            on="mover_b",
        )


def made(session, tensors, log):
    """The class made with example tensors of each name, one command at a time as
    the hand program."""
    return SetupPasses(log=log, session=session, depth=1, **tensors)
