"""Where SUM, HSUM and PLACE spend their time: an empty handler (what any command of a
class costs), the region class's route with wall-clock stamps on each CPU, the same
route in blocks of the hand's batch, and the hand's route as one body on mover B."""

from passes_class import GIVE_UP, HEADERS
from region_class import (
    COUNTER_ITEMS,
    HEAVY_AT,
    HPREFIX,
    HPREFIX_ITEMS,
    HSUM_STATE,
    HSUM_ZERO,
    HTOTALS,
    PLACE,
    PREFIX,
    REGION_PAGES,
    SLICES,
    SUM_STATE,
    SUM_ZERO,
    TOTALS,
)

import komorebi as km

SPACE_PAGES = 12  # the hand's PLACE space: region, directory page, end slot, headers
HIGH_PAGES = 5  # place_ranges: the region from page 4 (frame, item and core words)
# A mover's or pack's stamps (wall-clock words) and the mark of a stage's start.
STAMPS = """uint32_t km_stamp[16] = {};
uint32_t km_mark = km::wall();
km_stamp[0] = km_mark;"""
MARK = "km_mark = km::wall();"
# A stage's wall time added to a stamp; a loop's moved cycles to the next one.
SPENT_1 = "km_stamp[1] += km::wall() - km_mark;"
SPENT_2 = "km_stamp[2] += km::wall() - km_mark;"
SPENT_3 = "km_stamp[3] += km::wall() - km_mark;"
SPENT_7 = "km_stamp[7] += km::wall() - km_mark;"
MOVED_ENTRY = "km_stamp[5] += km::wall() - km_mark;\nkm_stamp[6] += entry_cycles;"
MOVED_COUNTS = "km_stamp[5] += km::wall() - km_mark;\nkm_stamp[6] += counts_cycles;"
# The stamps into an item: mover A's item 0 of st, mover B's item 1, pack's page.
KEEP_OPEN = "km_stamp[15] = km::wall();\nvolatile tt_l1_ptr uint32_t* kept = km::l1("
KEEP_CLOSE = """);
for (uint32_t i = 0; i < 16; ++i) {
    kept[i] = km_stamp[i];
}
asm volatile("fence" ::: "memory");
"""
KEEP_A = KEEP_OPEN + "get_write_ptr(st)" + KEEP_CLOSE
KEEP_B = KEEP_OPEN + "get_write_ptr(st) + 64" + KEEP_CLOSE
PACK_KEEP = """
km_stamp[15] = km::wall();
if (out != 0) {
    volatile uint32_t* kept = reinterpret_cast<volatile uint32_t*>(out);
    for (uint32_t i = 0; i < 16; ++i) {
        kept[i] = km_stamp[i];
    }
    asm volatile("fence" ::: "memory");
}
"""
# Pack's item body timed: the first start (stamp 1), the cycles inside (2), the last
# end (3), the items (4).
TIME_OPEN = """{
const uint32_t km_begun = km::wall();
if (km_stamp[4]++ == 0) {
    km_stamp[1] = km_begun;
}
"""
TIME_CLOSE = """
km_stamp[3] = km::wall();
km_stamp[2] += km_stamp[3] - km_begun;
}
"""
ARRIVED = """
auto arrived = [&] {
    return movement.poll(
        [] { return ncrisc_noc_reads_flushed(noc_index); },
        movement.args().wait_limit
    ) == km::kGaveNone;
};
"""
# The hand's SUM (setup_writer.cpp) on mover B, 32 cores at a time through staging;
# stamps: 1 read wait, 2 flush wait, 3 total.
SUM_HAND = """
const uint32_t begun = km::wall();
uint32_t read_wait = 0, flush_wait = 0, mark = 0;
constexpr uint32_t kBatch = 32;
const uint32_t staging = get_write_ptr(stage);
for (uint32_t slice = rank; slice < kTiles / 16; slice += cores) {
    uint32_t entries[16] = {}, row_counts[16] = {};
    for (uint32_t first_core = 0; first_core < counting; first_core += kBatch) {
        const uint32_t batch_cores = (
            counting - first_core < kBatch ? counting - first_core : kBatch
        );
        for (uint32_t batch_core = 0; batch_core < batch_cores; ++batch_core) {
            const uint64_t address = setup_pass::noc_address(
                0, counters.word, first_core + batch_core, kCountBytes, 64 * slice
            );
            noc_async_read(address, staging + 64 * batch_core, 64);
            noc_async_read(address + 4 * kTiles, staging + 2048 + 64 * batch_core, 64);
        }
        mark = km::wall();
        if (!arrived()) {
            return false;
        }
        read_wait += km::wall() - mark;
        invalidate_l1_cache();
        for (uint32_t batch_core = 0; batch_core < batch_cores; ++batch_core) {
            volatile tt_l1_ptr uint32_t* entry_words = (
                km::l1(staging + 64 * batch_core)
            );
            volatile tt_l1_ptr uint32_t* row_words = (
                km::l1(staging + 2048 + 64 * batch_core)
            );
            for (uint32_t j = 0; j < 16; ++j) {
                const uint32_t entry = entry_words[j];
                entry_words[j] = entries[j];
                entries[j] += entry;
                row_counts[j] += row_words[j];
            }
            asm volatile("fence" ::: "memory");
            noc_async_write(
                staging + 64 * batch_core,
                setup_pass::noc_address(
                    0, prefixes.word, first_core + batch_core, kSlotsBytes, 64 * slice
                ),
                64
            );
        }
        mark = km::wall();
        if (!flushed()) {
            return false;
        }
        flush_wait += km::wall() - mark;
    }
    volatile tt_l1_ptr uint32_t* total_words = km::l1(staging);
    for (uint32_t j = 0; j < 16; ++j) {
        total_words[j] = entries[j];
        total_words[16 + j] = row_counts[j];
    }
    asm volatile("fence" ::: "memory");
    noc_async_write(staging, sums.get_noc_addr(0, 64 * slice), 64);
    noc_async_write(staging + 64, sums.get_noc_addr(0, 4 * kTiles + 64 * slice), 64);
    mark = km::wall();
    if (!flushed()) {
        return false;
    }
    flush_wait += km::wall() - mark;
}
km_stamp[1] = read_wait;
km_stamp[2] = flush_wait;
km_stamp[3] = km::wall() - begun;
"""
# The hand's HSUM on mover B, 16 cores at a time; the same stamps.
HSUM_HAND = """
const uint32_t begun = km::wall();
uint32_t read_wait = 0, flush_wait = 0, mark = 0;
constexpr uint32_t kBatch = 16;
const uint32_t staging = get_write_ptr(stage);
const uint32_t prefix_space = staging + 64 * kBatch;
for (uint32_t heavy_tile = rank; heavy_tile < heavy_tiles; heavy_tile += cores) {
    uint32_t totals[32] = {};
    for (uint32_t first_core = 0; first_core < counting; first_core += kBatch) {
        const uint32_t batch_cores = (
            counting - first_core < kBatch ? counting - first_core : kBatch
        );
        for (uint32_t batch_core = 0; batch_core < batch_cores; ++batch_core) {
            noc_async_read(
                setup_pass::noc_address(
                    0, histograms.word, first_core + batch_core, 64 * step,
                    64 * heavy_tile
                ),
                staging + 64 * batch_core,
                64
            );
        }
        mark = km::wall();
        if (!arrived()) {
            return false;
        }
        read_wait += km::wall() - mark;
        invalidate_l1_cache();
        for (uint32_t batch_core = 0; batch_core < batch_cores; ++batch_core) {
            volatile tt_l1_ptr uint16_t* count_words = (
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                    staging + 64 * batch_core
                )
            );
            volatile tt_l1_ptr uint32_t* prefix_words = (
                km::l1(prefix_space + 128 * batch_core)
            );
            uint32_t starts = 0, ends = 0;
            for (uint32_t i = 0; i < 16; ++i) {
                starts += totals[i];
                ends += totals[16 + i];
                prefix_words[i] = starts;
                prefix_words[16 + i] = ends;
            }
            for (uint32_t i = 0; i < 32; ++i) {
                totals[i] += count_words[i];
            }
            asm volatile("fence" ::: "memory");
            noc_async_write(
                prefix_space + 128 * batch_core,
                setup_pass::noc_address(
                    0, hprefix.word, first_core + batch_core, kHprefixPageBytes,
                    128 * heavy_tile
                ),
                128
            );
        }
        mark = km::wall();
        if (!flushed()) {
            return false;
        }
        flush_wait += km::wall() - mark;
    }
    volatile tt_l1_ptr uint32_t* total_words = km::l1(staging);
    for (uint32_t i = 0; i < 32; ++i) {
        total_words[i] = totals[i];
    }
    asm volatile("fence" ::: "memory");
    noc_async_write(
        staging, sums.get_noc_addr(0, kSumsHeavyAt + 128 * heavy_tile), 128
    );
    mark = km::wall();
    if (!flushed()) {
        return false;
    }
    flush_wait += km::wall() - mark;
}
km_stamp[1] = read_wait;
km_stamp[2] = flush_wait;
km_stamp[3] = km::wall() - begun;
"""
# The hand's PLACE on mover B (its two ranges, directory, end slot, headers 32 at a
# time); stamps: 1 read wait, 2 flush wait, 3 total, 4 the reads' end.
PLACE_HAND = """
const uint32_t begun = km::wall();
uint32_t flush_wait = 0, mark = 0;
const uint32_t space = get_write_ptr(room);
noc_async_read(items.get_noc_addr(0, 0), space, kItemsBytes);
noc_async_read(
    items.get_noc_addr(0, kFrameWordsAt), space + kFrameWordsAt,
    kItemsRegionBytes - kFrameWordsAt
);
mark = km::wall();
if (!arrived()) {
    return false;
}
km_stamp[1] = km::wall() - mark;
km_stamp[4] = km::wall() - begun;
invalidate_l1_cache();
volatile tt_l1_ptr uint32_t* tile_items = km::l1(space);
const volatile tt_l1_ptr uint16_t*
    rows = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(space + kItemRows);
volatile tt_l1_ptr uint32_t* item_words = km::l1(space + kItemWordsAt);
volatile tt_l1_ptr uint32_t* core_words = km::l1(space + kCoreWordsAt);
volatile tt_l1_ptr uint32_t* frame_words = km::l1(space + kFrameWordsAt);
volatile tt_l1_ptr uint16_t*
    tile_of = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(space + kBaseAt);
uint32_t item_count = 0;
for (uint32_t tile = 0; tile < kTiles; ++tile) {
    const uint32_t first = tile_items[tile] & 0xFFFF;
    const uint32_t items_in_tile = tile_items[tile] >> 16;
    for (uint32_t i = first; i < first + items_in_tile && i < kSlotsBytes / 4; ++i) {
        tile_of[i] = tile;
    }
    item_count = first + items_in_tile;
}
if (item_count > kSlotsBytes / 4 || fill_cores > kMaxFillCores) {
    give_up(kGaveBadPlace, 1);
    return true;
}
const InterleavedAddrGen<true> list_pages = {
    .bank_base_address = lists.word, .page_size = kListPageBytes
};
const uint32_t me = rank, directory_space = space + kItemsRegionBytes;
auto build_slot = [&](uint32_t slot_address, uint32_t mark, uint32_t i) {
    volatile tt_l1_ptr uint32_t* slot_words = km::l1(slot_address);
    for (uint32_t word = 0; word < 32; ++word) {
        slot_words[word] = 0;
    }
    slot_words[0] = mark;
    if (mark == kLayMarkTile) {
        slot_words[kLayHdrTile] = tile_of[i];
        slot_words[kLayHdrCount] = item_words[kItemWords * i] >> 8;
        slot_words[kLayHdrRow0] = rows[i] & 0xFF;
        slot_words[kLayHdrRow1] = rows[i] >> 8;
    }
};
if (me < fill_cores) {
    volatile tt_l1_ptr uint32_t* directory_words = km::l1(directory_space);
    for (uint32_t word = 0; word < kDirPageBytes / 4; ++word) {
        directory_words[word] = 0;
    }
    uint32_t directory_items = 0;
    for (
        uint32_t i = 0; i < item_count && directory_items < kLayDirFrame - kLayDirIds;
        ++i
    ) {
        const uint32_t item_word = item_words[kItemWords * i];
        if ((item_word & 0xFF) != me) {
            continue;
        }
        uint32_t id = tile_of[i];
        if (rows[i] != (32u << 8)) {
            id |= (rows[i] & 0xFF) << kLayRow0Shift | (rows[i] >> 8) << kLayRow1Shift;
        }
        directory_words[kLayDirIds + directory_items++] = (
            (item_word >> 8) == 0 ? id | kLayItemEmpty : id
        );
    }
    directory_words[0] = core_words[kCoreWords * me];
    directory_words[1] = core_words[kCoreWords * me + 1];
    directory_words[2] = directory_items;
    for (uint32_t word = 0; word < 16; ++word) {
        directory_words[kLayDirFrame + word] = frame_words[word];
    }
    asm volatile("fence" ::: "memory");
    noc_async_write(
        directory_space,
        setup_pass::noc_address(0, directory.word, me, kDirPageBytes),
        kDirPageBytes
    );
    const uint32_t end_slot = core_words[kCoreWords * me + 3];
    const uint32_t slot_address = directory_space + kDirPageBytes;
    build_slot(slot_address, kLayMarkEnd, 0);
    asm volatile("fence" ::: "memory");
    noc_async_write(
        slot_address,
        get_noc_addr(end_slot / 16, list_pages, (end_slot % 16) * 128),
        128
    );
    mark = km::wall();
    if (!flushed()) {
        return false;
    }
    flush_wait += km::wall() - mark;
}
uint32_t batched = 0;
const uint32_t header_space = directory_space + kDirPageBytes + 128;
for (uint32_t i = me; i < item_count; i += cores) {
    if ((item_words[kItemWords * i] >> 8) == 0) {
        continue;
    }
    const uint32_t slot_address = header_space + 128 * batched;
    const uint32_t slot = item_words[kItemWords * i + 1];
    build_slot(slot_address, kLayMarkTile, i);
    asm volatile("fence" ::: "memory");
    noc_async_write(
        slot_address, get_noc_addr(slot / 16, list_pages, (slot % 16) * 128), 128
    );
    if (++batched == kOutBytes / 128) {
        mark = km::wall();
        if (!flushed()) {
            return false;
        }
        flush_wait += km::wall() - mark;
        batched = 0;
    }
}
mark = km::wall();
if (!flushed()) {
    return false;
}
flush_wait += km::wall() - mark;
km_stamp[2] = flush_wait;
km_stamp[3] = km::wall() - begun;
"""


class RegionRoute(km.Program):
    """idle (any class's command); sum_route, hsum_route, place_route: the region
    class's route, its CPUs stamped (stamps given when the word stamping is 1)."""

    region: km.L1[km.u32, REGION_PAGES, 32, 32]  # PLACE: the items region, read
    page: km.L1[km.u32, 1, 32, 32] = km.own(push=None)  # PLACE: a directory page
    slots: km.L1[km.u32, 2, 32, 32] = km.own(push=None)  # PLACE: end slot, headers
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)  # movers A and B's stamps

    @km.handler
    def idle(self):
        pass

    @km.handler
    def sum_route(
        self,
        counters: km.Tensor,
        prefixes: km.Tensor,
        sums: km.Region,
        counting: int,
        stamping: int,
        stamps: km.Tensor,
    ):
        km.cxx(HEADERS + GIVE_UP, on="mover_b", at="start")
        km.cxx(STAMPS, on="mover_a")
        km.cxx(STAMPS, on="mover_b")
        km.cxx(STAMPS, on="pack")
        km.cxx(SUM_STATE, on="pack")
        for turn in range((SLICES + self.size - 1 - self.rank) // self.size):
            part = self.rank + turn * self.size
            km.cxx(SUM_ZERO, on="pack")
            prefix = km.Stream(
                prefixes.items(first=part, count=counting, size=64, step=COUNTER_ITEMS),
                write=True,
            )
            km.cxx(MARK, on="mover_a")
            km.cxx(MARK, on="mover_b")
            for entry, row in zip(
                km.Stream(
                    counters.items(
                        first=part, count=counting, size=64, step=COUNTER_ITEMS
                    )
                ),
                km.Stream(
                    counters.items(
                        first=SLICES + part, count=counting, size=64, step=COUNTER_ITEMS
                    )
                ),
            ):
                km.cxx(
                    TIME_OPEN + PREFIX + TIME_CLOSE,
                    takes=[entry, row, prefix],
                    on="pack",
                )
            km.cxx(MOVED_ENTRY, takes=[entry], on="mover_a")
            km.cxx(MOVED_ENTRY, takes=[entry], on="mover_b")
            km.cxx(MARK, on="mover_b")
            totals = km.Stream(
                sums.items(first=part, count=2, size=64, step=SLICES),
                write=True,
                block=2,
            )
            km.cxx(TOTALS, takes=[totals], on="pack")
            km.cxx(SPENT_7, on="mover_b")
        km.cxx(KEEP_A, takes=[self.st], on="mover_a")
        km.cxx(KEEP_B, takes=[self.st], on="mover_b")
        if stamping:
            stamps.items(size=64).put(self.st[0], at=self.rank * 4, on="mover_a")
            stamps.items(size=64).put(self.st[1], at=self.rank * 4 + 1)
            out = km.Stream(
                stamps.items(first=self.rank * 4 + 2, count=1, size=64), write=True
            )
            km.cxx(PACK_KEEP, takes=[out], on="pack")

    @km.handler
    def hsum_route(
        self,
        heavy_tiles: int,
        histograms: km.Tensor,
        hprefix: km.Tensor,
        sums: km.Region,
        counting: int,
        step: int,
        stamping: int,
        stamps: km.Tensor,
    ):
        km.cxx(STAMPS, on="mover_a")
        km.cxx(STAMPS, on="mover_b")
        km.cxx(STAMPS, on="pack")
        km.cxx(HSUM_STATE, on="pack")
        for turn in range((heavy_tiles + self.size - 1 - self.rank) // self.size):
            tile = self.rank + turn * self.size
            km.cxx(HSUM_ZERO, on="pack")
            prefix = km.Stream(
                hprefix.items(first=tile, count=counting, size=128, step=HPREFIX_ITEMS),
                write=True,
            )
            km.cxx(MARK, on="mover_a")
            km.cxx(MARK, on="mover_b")
            for counts in km.Stream(
                histograms.items(first=tile, count=counting, size=64, step=step)
            ):
                km.cxx(
                    TIME_OPEN + HPREFIX + TIME_CLOSE, takes=[counts, prefix], on="pack"
                )
            km.cxx(MOVED_COUNTS, takes=[counts], on="mover_a")
            km.cxx(MOVED_COUNTS, takes=[counts], on="mover_b")
            km.cxx(MARK, on="mover_b")
            total = km.Stream(
                sums.items(first=HEAVY_AT + tile, count=1, size=128), write=True
            )
            km.cxx(HTOTALS, takes=[total], on="pack")
            km.cxx(SPENT_7, on="mover_b")
        km.cxx(KEEP_A, takes=[self.st], on="mover_a")
        km.cxx(KEEP_B, takes=[self.st], on="mover_b")
        if stamping:
            stamps.items(size=64).put(self.st[0], at=self.rank * 4, on="mover_a")
            stamps.items(size=64).put(self.st[1], at=self.rank * 4 + 1)
            out = km.Stream(
                stamps.items(first=self.rank * 4 + 2, count=1, size=64), write=True
            )
            km.cxx(PACK_KEEP, takes=[out], on="pack")

    @km.handler
    def place_route(
        self,
        items: km.Region,
        directory: km.Tensor,
        fill_cores: int,
        lists: km.Tensor,
        stamping: int,
        stamps: km.Tensor,
    ):
        km.cxx(STAMPS, on="mover_a")
        km.cxx(STAMPS, on="mover_b")
        self.region.fill(km.Stream(items.pages(0, REGION_PAGES)))
        km.cxx(SPENT_1, on="mover_a")
        cores = self.size
        km.cxx(MARK, on="mover_b")
        end_slot = km.cxx(
            "km_stamp[1] = km::wall() - km_mark;\nkm_mark = km::wall();\n" + PLACE,
            takes=[self.region, self.page, self.slots, fill_cores, lists, cores],
            on="mover_b",
            gives=int,
        )
        km.cxx(SPENT_2, on="mover_b")
        km.cxx(MARK, on="mover_b")
        if self.rank < fill_cores:
            directory.items(size=4096).put(self.page, at=self.rank)
            lists.items(size=128).put(self.slots, at=end_slot)
        km.cxx(SPENT_3, on="mover_b")
        km.cxx(KEEP_A, takes=[self.st], on="mover_a")
        km.cxx(KEEP_B, takes=[self.st], on="mover_b")
        if stamping:
            stamps.items(size=64).put(self.st[0], at=self.rank * 4, on="mover_a")
            stamps.items(size=64).put(self.st[1], at=self.rank * 4 + 1)


class RegionBlock(km.Program):
    """sum_block, hsum_block: the region class's route in the hand's batches (blocks of
    32 and of 16 items)."""

    @km.handler
    def sum_block(
        self, counters: km.Tensor, prefixes: km.Tensor, sums: km.Region, counting: int
    ):
        km.cxx(SUM_STATE, on="pack")
        for turn in range((SLICES + self.size - 1 - self.rank) // self.size):
            part = self.rank + turn * self.size
            km.cxx(SUM_ZERO, on="pack")
            prefix = km.Stream(
                prefixes.items(first=part, count=counting, size=64, step=COUNTER_ITEMS),
                write=True,
                block=32,
            )
            for entry, row in zip(
                km.Stream(
                    counters.items(
                        first=part, count=counting, size=64, step=COUNTER_ITEMS
                    ),
                    block=32,
                ),
                km.Stream(
                    counters.items(
                        first=SLICES + part, count=counting, size=64, step=COUNTER_ITEMS
                    ),
                    block=32,
                ),
            ):
                km.cxx(PREFIX, takes=[entry, row, prefix], on="pack")
            totals = km.Stream(
                sums.items(first=part, count=2, size=64, step=SLICES),
                write=True,
                block=2,
            )
            km.cxx(TOTALS, takes=[totals], on="pack")

    @km.handler
    def hsum_block(
        self,
        heavy_tiles: int,
        histograms: km.Tensor,
        hprefix: km.Tensor,
        sums: km.Region,
        counting: int,
        step: int,
    ):
        km.cxx(HSUM_STATE, on="pack")
        for turn in range((heavy_tiles + self.size - 1 - self.rank) // self.size):
            tile = self.rank + turn * self.size
            km.cxx(HSUM_ZERO, on="pack")
            prefix = km.Stream(
                hprefix.items(first=tile, count=counting, size=128, step=HPREFIX_ITEMS),
                write=True,
                block=16,
            )
            for counts in km.Stream(
                histograms.items(first=tile, count=counting, size=64, step=step),
                block=16,
            ):
                km.cxx(HPREFIX, takes=[counts, prefix], on="pack")
            total = km.Stream(
                sums.items(first=HEAVY_AT + tile, count=1, size=128), write=True
            )
            km.cxx(HTOTALS, takes=[total], on="pack")


class RegionHand(km.Program):
    """sum_hand, hsum_hand, place_hand: the hand's route, one body on mover B, its
    stamps given when the word stamping is 1."""

    room: km.L1[km.u32, SPACE_PAGES, 32, 32] = km.own(push=None)  # PLACE's space
    stage: km.L1[km.u32, 1, 32, 32] = km.own(push=None)  # SUM and HSUM's staging
    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)  # mover B's stamps, item 1

    @km.handler
    def sum_hand(
        self,
        counters: km.Tensor,
        prefixes: km.Tensor,
        sums: km.Region,
        counting: int,
        stamping: int,
        stamps: km.Tensor,
    ):
        km.cxx(HEADERS + GIVE_UP + ARRIVED, on="mover_b", at="start")
        km.cxx(STAMPS, on="mover_b")
        rank = self.rank
        cores = self.size
        km.cxx(
            SUM_HAND,
            takes=[counters, prefixes, sums, counting, rank, cores, self.stage],
            on="mover_b",
        )
        km.cxx(KEEP_B, takes=[self.st], on="mover_b")
        if stamping:
            stamps.items(size=64).put(self.st[1], at=self.rank * 4 + 1)

    @km.handler
    def hsum_hand(
        self,
        heavy_tiles: int,
        histograms: km.Tensor,
        hprefix: km.Tensor,
        sums: km.Region,
        counting: int,
        step: int,
        stamping: int,
        stamps: km.Tensor,
    ):
        km.cxx(STAMPS, on="mover_b")
        rank = self.rank
        cores = self.size
        km.cxx(
            HSUM_HAND,
            takes=[
                heavy_tiles,
                histograms,
                hprefix,
                sums,
                counting,
                step,
                rank,
                cores,
                self.stage,
            ],
            on="mover_b",
        )
        km.cxx(KEEP_B, takes=[self.st], on="mover_b")
        if stamping:
            stamps.items(size=64).put(self.st[1], at=self.rank * 4 + 1)

    @km.handler
    def place_hand(
        self,
        items: km.Region,
        directory: km.Tensor,
        fill_cores: int,
        lists: km.Tensor,
        stamping: int,
        stamps: km.Tensor,
    ):
        km.cxx(STAMPS, on="mover_b")
        rank = self.rank
        cores = self.size
        km.cxx(
            PLACE_HAND,
            takes=[items, directory, fill_cores, lists, rank, cores, self.room],
            on="mover_b",
        )
        km.cxx(KEEP_B, takes=[self.st], on="mover_b")
        if stamping:
            stamps.items(size=64).put(self.st[1], at=self.rank * 4 + 1)


def plain(text):
    """A stamped body without its stamps: the hand's text as it is."""
    words = ("km_stamp", "begun", "read_wait", "flush_wait", "mark = km::wall()")
    return "\n".join(
        line for line in text.split("\n") if not any(word in line for word in words)
    )


SUM_PLAIN = plain(SUM_HAND)
HSUM_PLAIN = plain(HSUM_HAND)
PLACE_PLAIN = plain(PLACE_HAND)
# PLACE from the two ranges the hand reads, filled by mover A into low (the items
# table) and high (from page 4: frame, item and core words); tile_of in its own field.
RANGES_HEAD = """
volatile tt_l1_ptr uint32_t* tile_items = km::l1(low);
const volatile tt_l1_ptr uint16_t*
    rows = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(low + kItemRows);
volatile tt_l1_ptr uint32_t* item_words = km::l1(high + kItemWordsAt - 4 * 4096);
volatile tt_l1_ptr uint32_t* core_words = km::l1(high + kCoreWordsAt - 4 * 4096);
volatile tt_l1_ptr uint32_t* frame_words = km::l1(high + kFrameWordsAt - 4 * 4096);
volatile tt_l1_ptr uint16_t*
    tile_of = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(tiles));
"""
PLACE_RANGES = RANGES_HEAD + PLACE[PLACE.index("uint32_t item_count = 0;") :]


class RegionPlain(km.Program):
    """sum_plain, hsum_plain, place_plain: the hand's text as one body on mover B;
    place_ranges: the region class's PLACE reading only the hand's two ranges."""

    low: km.L1[km.u32, 2, 32, 32]  # place_ranges: the items table
    high: km.L1[km.u32, HIGH_PAGES, 32, 32]  # place_ranges: from page 4
    tiles: km.L1[km.u32, 2, 32, 32] = km.own(push=None)  # place_ranges: tile_of
    page: km.L1[km.u32, 1, 32, 32] = km.own(push=None)  # place_ranges: directory
    slots: km.L1[km.u32, 2, 32, 32] = km.own(push=None)  # place_ranges: end, headers
    room: km.L1[km.u32, SPACE_PAGES, 32, 32] = km.own(push=None)  # place_plain
    stage: km.L1[km.u32, 1, 32, 32] = km.own(push=None)  # SUM and HSUM's staging

    @km.handler
    def sum_plain(
        self, counters: km.Tensor, prefixes: km.Tensor, sums: km.Region, counting: int
    ):
        km.cxx(HEADERS + GIVE_UP + ARRIVED, on="mover_b", at="start")
        rank = self.rank
        cores = self.size
        km.cxx(
            SUM_PLAIN,
            takes=[counters, prefixes, sums, counting, rank, cores, self.stage],
            on="mover_b",
        )

    @km.handler
    def hsum_plain(
        self,
        heavy_tiles: int,
        histograms: km.Tensor,
        hprefix: km.Tensor,
        sums: km.Region,
        counting: int,
        step: int,
    ):
        rank = self.rank
        cores = self.size
        km.cxx(
            HSUM_PLAIN,
            takes=[
                heavy_tiles,
                histograms,
                hprefix,
                sums,
                counting,
                step,
                rank,
                cores,
                self.stage,
            ],
            on="mover_b",
        )

    @km.handler
    def place_plain(
        self, items: km.Region, directory: km.Tensor, fill_cores: int, lists: km.Tensor
    ):
        rank = self.rank
        cores = self.size
        km.cxx(
            PLACE_PLAIN,
            takes=[items, directory, fill_cores, lists, rank, cores, self.room],
            on="mover_b",
        )

    @km.handler
    def place_ranges(
        self, items: km.Region, directory: km.Tensor, fill_cores: int, lists: km.Tensor
    ):
        self.low.fill(km.Stream(items.pages(0, 2)))
        self.high.fill(km.Stream(items.pages(4, HIGH_PAGES)))
        cores = self.size
        end_slot = km.cxx(
            PLACE_RANGES,
            takes=[
                self.low,
                self.high,
                self.tiles,
                self.page,
                self.slots,
                fill_cores,
                lists,
                cores,
            ],
            on="mover_b",
            gives=int,
        )
        if self.rank < fill_cores:
            directory.items(size=4096).put(self.page, at=self.rank)
            lists.items(size=128).put(self.slots, at=end_slot)


class RegionBlockTimed(km.Program):
    """sum_btimed, hsum_btimed: the route in the hand's batches, its CPUs stamped as
    RegionRoute's."""

    st: km.L1[km.u32, 1, 32, 32] = km.own(push=None)  # movers A and B's stamps

    @km.handler
    def sum_btimed(
        self,
        counters: km.Tensor,
        prefixes: km.Tensor,
        sums: km.Region,
        counting: int,
        stamping: int,
        stamps: km.Tensor,
    ):
        km.cxx(HEADERS + GIVE_UP, on="mover_b", at="start")
        km.cxx(STAMPS, on="mover_a")
        km.cxx(STAMPS, on="mover_b")
        km.cxx(STAMPS, on="pack")
        km.cxx(SUM_STATE, on="pack")
        for turn in range((SLICES + self.size - 1 - self.rank) // self.size):
            part = self.rank + turn * self.size
            km.cxx(SUM_ZERO, on="pack")
            prefix = km.Stream(
                prefixes.items(first=part, count=counting, size=64, step=COUNTER_ITEMS),
                write=True,
                block=32,
            )
            km.cxx(MARK, on="mover_a")
            km.cxx(MARK, on="mover_b")
            for entry, row in zip(
                km.Stream(
                    counters.items(
                        first=part, count=counting, size=64, step=COUNTER_ITEMS
                    ),
                    block=32,
                ),
                km.Stream(
                    counters.items(
                        first=SLICES + part, count=counting, size=64, step=COUNTER_ITEMS
                    ),
                    block=32,
                ),
            ):
                km.cxx(
                    TIME_OPEN + PREFIX + TIME_CLOSE,
                    takes=[entry, row, prefix],
                    on="pack",
                )
            km.cxx(MOVED_ENTRY, takes=[entry], on="mover_a")
            km.cxx(MOVED_ENTRY, takes=[entry], on="mover_b")
            km.cxx(MARK, on="mover_b")
            totals = km.Stream(
                sums.items(first=part, count=2, size=64, step=SLICES),
                write=True,
                block=2,
            )
            km.cxx(TOTALS, takes=[totals], on="pack")
            km.cxx(SPENT_7, on="mover_b")
        km.cxx(KEEP_A, takes=[self.st], on="mover_a")
        km.cxx(KEEP_B, takes=[self.st], on="mover_b")
        if stamping:
            stamps.items(size=64).put(self.st[0], at=self.rank * 4, on="mover_a")
            stamps.items(size=64).put(self.st[1], at=self.rank * 4 + 1)
            out = km.Stream(
                stamps.items(first=self.rank * 4 + 2, count=1, size=64), write=True
            )
            km.cxx(PACK_KEEP, takes=[out], on="pack")

    @km.handler
    def hsum_btimed(
        self,
        heavy_tiles: int,
        histograms: km.Tensor,
        hprefix: km.Tensor,
        sums: km.Region,
        counting: int,
        step: int,
        stamping: int,
        stamps: km.Tensor,
    ):
        km.cxx(STAMPS, on="mover_a")
        km.cxx(STAMPS, on="mover_b")
        km.cxx(STAMPS, on="pack")
        km.cxx(HSUM_STATE, on="pack")
        for turn in range((heavy_tiles + self.size - 1 - self.rank) // self.size):
            tile = self.rank + turn * self.size
            km.cxx(HSUM_ZERO, on="pack")
            prefix = km.Stream(
                hprefix.items(first=tile, count=counting, size=128, step=HPREFIX_ITEMS),
                write=True,
                block=16,
            )
            km.cxx(MARK, on="mover_a")
            km.cxx(MARK, on="mover_b")
            for counts in km.Stream(
                histograms.items(first=tile, count=counting, size=64, step=step),
                block=16,
            ):
                km.cxx(
                    TIME_OPEN + HPREFIX + TIME_CLOSE, takes=[counts, prefix], on="pack"
                )
            km.cxx(MOVED_COUNTS, takes=[counts], on="mover_a")
            km.cxx(MOVED_COUNTS, takes=[counts], on="mover_b")
            km.cxx(MARK, on="mover_b")
            total = km.Stream(
                sums.items(first=HEAVY_AT + tile, count=1, size=128), write=True
            )
            km.cxx(HTOTALS, takes=[total], on="pack")
            km.cxx(SPENT_7, on="mover_b")
        km.cxx(KEEP_A, takes=[self.st], on="mover_a")
        km.cxx(KEEP_B, takes=[self.st], on="mover_b")
        if stamping:
            stamps.items(size=64).put(self.st[0], at=self.rank * 4, on="mover_a")
            stamps.items(size=64).put(self.st[1], at=self.rank * 4 + 1)
            out = km.Stream(
                stamps.items(first=self.rank * 4 + 2, count=1, size=64), write=True
            )
            km.cxx(PACK_KEEP, takes=[out], on="pack")


# The hand's PLACE text in a function of its own (a lambda the compiler keeps apart),
# so its registers are allocated as the hand's place() are: 0 gave up, 1 done.
PLACE_FN = (
    "auto place_body = [&]() __attribute__((noinline)) -> uint32_t {\n"
    + PLACE_PLAIN.replace("return false;", "return 0;").replace(
        "return true;", "return 1;"
    )
    + "\nreturn 1;\n};\nif (place_body() == 0) {\n    return false;\n}\n"
)


class RegionFn(km.Program):
    """place_fn: the hand's PLACE text as a class body, compiled in a function of its
    own."""

    room: km.L1[km.u32, SPACE_PAGES, 32, 32] = km.own(push=None)  # PLACE's space

    @km.handler
    def place_fn(
        self, items: km.Region, directory: km.Tensor, fill_cores: int, lists: km.Tensor
    ):
        km.cxx(HEADERS + GIVE_UP + ARRIVED, on="mover_b", at="start")
        rank = self.rank
        cores = self.size
        km.cxx(
            PLACE_FN,
            takes=[items, directory, fill_cores, lists, rank, cores, self.room],
            on="mover_b",
        )
