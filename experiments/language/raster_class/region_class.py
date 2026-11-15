"""The rasterizer's three passes that write or read an X280 thread's region (SUM, HSUM,
PLACE) as a class: the counters' and histograms' slices as items with a step, their
prefixes and totals packed by bodies on pack, PLACE's directory and end slot put by
mover B. README.md says which moves are still bodies."""

from passes_class import GIVE_UP, HEADERS

import komorebi as km

SLICES = 64  # SUM: a slice is 16 tiles' words, 64 B; kTiles / 16 of them
COUNTER_ITEMS = 128  # 64 B items a counters (and a slots) page of 8,192 B
HPREFIX_ITEMS = 256  # 128 B items a heavy prefixes page of 32,768 B
HEAVY_AT = 64  # HSUM's totals in SUM's: from byte 8,192, 128 B items
REGION_PAGES = 9  # PLACE: the items region's 36,864 B, in pages of 4,096 B
# SUM's sums of one slice over the cores, on pack.
SUM_STATE = "uint32_t entries[16], row_counts[16];"
SUM_ZERO = "for (uint32_t j = 0; j < 16; ++j) {\n    entries[j] = row_counts[j] = 0;\n}"
# One core's slice: its entries' exclusive prefix to the slots page, both sums added.
PREFIX = """
volatile uint32_t* entry_words = reinterpret_cast<volatile uint32_t*>(entry);
volatile uint32_t* row_words = reinterpret_cast<volatile uint32_t*>(row);
volatile uint32_t* prefix_words = reinterpret_cast<volatile uint32_t*>(prefix);
for (uint32_t j = 0; j < 16; ++j) {
    const uint32_t count = entry_words[j];
    if (prefix != 0) {
        prefix_words[j] = entries[j];
    }
    entries[j] += count;
    row_counts[j] += row_words[j];
}
asm volatile("fence" ::: "memory");
"""
# The slice's totals: entries' at item slice, rows' at item 64 + slice.
TOTALS = """
if (totals != 0) {
    volatile uint32_t* total_words = reinterpret_cast<volatile uint32_t*>(totals);
    for (uint32_t j = 0; j < 16; ++j) {
        total_words[j] = entries[j];
        total_words[16 + j] = row_counts[j];
    }
    asm volatile("fence" ::: "memory");
}
"""
HSUM_STATE = "uint32_t totals[32];"
HSUM_ZERO = "for (uint32_t i = 0; i < 32; ++i) {\n    totals[i] = 0;\n}"
# One core's 32 band counts: the 16 starts' and 16 ends' running sums before it.
HPREFIX = """
volatile uint16_t* count_words = reinterpret_cast<volatile uint16_t*>(counts);
volatile uint32_t* prefix_words = reinterpret_cast<volatile uint32_t*>(prefix);
uint32_t starts = 0, ends = 0;
for (uint32_t i = 0; i < 16; ++i) {
    starts += totals[i];
    ends += totals[16 + i];
    if (prefix != 0) {
        prefix_words[i] = starts;
        prefix_words[16 + i] = ends;
    }
}
for (uint32_t i = 0; i < 32; ++i) {
    totals[i] += count_words[i];
}
asm volatile("fence" ::: "memory");
"""
HTOTALS = """
if (total != 0) {
    volatile uint32_t* total_words = reinterpret_cast<volatile uint32_t*>(total);
    for (uint32_t i = 0; i < 32; ++i) {
        total_words[i] = totals[i];
    }
    asm volatile("fence" ::: "memory");
}
"""
# PLACE on mover B, the hand's: each item's tile, this core's directory page and end
# slot into own fields (put after), the headers written 32 at a time.
PLACE = """
volatile tt_l1_ptr uint32_t* tile_items = km::l1(region);
const volatile tt_l1_ptr uint16_t*
    rows = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(region + kItemRows);
volatile tt_l1_ptr uint32_t* item_words = km::l1(region + kItemWordsAt);
volatile tt_l1_ptr uint32_t* core_words = km::l1(region + kCoreWordsAt);
volatile tt_l1_ptr uint32_t* frame_words = km::l1(region + kFrameWordsAt);
volatile tt_l1_ptr uint16_t*
    tile_of = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(region + kBaseAt);
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
const uint32_t me = movement.me();
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
    volatile tt_l1_ptr uint32_t* directory = km::l1(get_write_ptr(page));
    for (uint32_t word = 0; word < kDirPageBytes / 4; ++word) {
        directory[word] = 0;
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
        directory[kLayDirIds + directory_items++] = (
            (item_word >> 8) == 0 ? id | kLayItemEmpty : id
        );
    }
    directory[0] = core_words[kCoreWords * me];
    directory[1] = core_words[kCoreWords * me + 1];
    directory[2] = directory_items;
    for (uint32_t word = 0; word < 16; ++word) {
        directory[kLayDirFrame + word] = frame_words[word];
    }
    end_slot = core_words[kCoreWords * me + 3];
    build_slot(get_write_ptr(slots), kLayMarkEnd, 0);
    asm volatile("fence" ::: "memory");
}
uint32_t batched = 0;
const uint32_t header_space = get_write_ptr(slots) + 128 * 32;
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
    if (++batched == 32) {
        if (!flushed()) {
            return false;
        }
        batched = 0;
    }
}
if (!flushed()) {
    return false;
}
"""


class RegionPasses(km.Program):
    """sum_tiles, hsum_tiles and place_items: the hand program's SUM, HSUM and PLACE
    on every core, each core's slices or heavy tiles by rank, as the hand's words
    name them; the region a parameter given by its name."""

    region: km.L1[km.u32, REGION_PAGES, 32, 32]  # PLACE: the items region, read
    page: km.L1[km.u32, 1, 32, 32] = km.own(push=None)  # PLACE: a directory page
    slots: km.L1[km.u32, 2, 32, 32] = km.own(push=None)  # PLACE: end slot, headers

    @km.handler
    def sum_tiles(
        self, counters: km.Tensor, prefixes: km.Tensor, sums: km.Region, counting: int
    ):
        km.cxx(HEADERS + GIVE_UP, on="mover_b", at="start")
        km.cxx(SUM_STATE, on="pack")
        for turn in range((SLICES + self.size - 1 - self.rank) // self.size):
            part = self.rank + turn * self.size
            km.cxx(SUM_ZERO, on="pack")
            prefix = km.Stream(
                prefixes.items(first=part, count=counting, size=64, step=COUNTER_ITEMS),
                write=True,
            )
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
                km.cxx(PREFIX, takes=[entry, row, prefix], on="pack")
            totals = km.Stream(
                sums.items(first=part, count=2, size=64, step=SLICES),
                write=True,
                block=2,
            )
            km.cxx(TOTALS, takes=[totals], on="pack")

    @km.handler
    def hsum_tiles(
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
            )
            for counts in km.Stream(
                histograms.items(first=tile, count=counting, size=64, step=step)
            ):
                km.cxx(HPREFIX, takes=[counts, prefix], on="pack")
            total = km.Stream(
                sums.items(first=HEAVY_AT + tile, count=1, size=128), write=True
            )
            km.cxx(HTOTALS, takes=[total], on="pack")

    @km.handler
    def place_items(
        self,
        items: km.Region,
        directory: km.Tensor,
        fill_cores: int,
        lists: km.Tensor,
    ):
        self.region.fill(km.Stream(items.pages(0, REGION_PAGES)))
        cores = self.size
        end_slot = km.cxx(
            PLACE,
            takes=[self.region, self.page, self.slots, fill_cores, lists, cores],
            on="mover_b",
            gives=int,
        )
        if self.rank < fill_cores:
            directory.items(size=4096).put(self.page, at=self.rank)
            lists.items(size=128).put(self.slots, at=end_slot)
