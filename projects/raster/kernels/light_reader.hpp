// LIGHT's reader (shade.h): table, camera, items' words and planes, asked pages;
// kLightGlass: SHADE's table too, an item the glass planes, C and the merged 1/z.
#include "shade.h"

void light_reader_command(const km::Reader& reader, const km_command_t& command) {
    const uint32_t directory_address = command.a0;
    const uint32_t scratch = get_write_ptr(kCbDirReader);
    volatile tt_l1_ptr uint32_t* directory = km::l1(scratch);
    uint32_t surface[kSurfaceWords + 3] = {};
    if (!surface_words(reader, command, scratch, surface)) {
        return;
    }
    const bool glass = command.a9 == kLightGlass;
    uint32_t glass_frame[kDirFrameMax + 1] = {};  // the glass layer's frame words
    if (glass) {
        if (!km::read_pages(
            reader, command, Directories(command.a6), reader.me(), 1, scratch
        ) ||
            !read_done(reader, command.sequence)) {
            return;
        }
        invalidate_l1_cache();
        for (uint32_t i = 0; i <= kDirFrameMax; ++i) {
            glass_frame[i] = directory[kDirFrame + i];
        }
        if (glass_frame[0] == 0 || glass_frame[0] % kPlanes != 0) {
            return;
        }
    }
    const uint32_t item_pages = glass ? kLightGlassPages : kLightItemPages;
    const uint32_t item_planes = glass ? kLightGlassPlanes : kLightItemPlanes;
    const uint32_t table_address = surface[kSurfaceWords];
    const uint32_t pages_address = surface[kSurfaceWords + 2];
    if (!km::read_pages(
        reader, command, Directories(directory_address), reader.me(), 1, scratch
    ) ||
        !read_done(reader, command.sequence)) {
        return;
    }
    invalidate_l1_cache();
    const uint32_t items = directory[kDirItems], tensor_pages = directory[kDirFrame];
    if (tensor_pages == 0 || tensor_pages % kPlanes != 0 || items > kDirMaxItems) {
        return;  // the pack thread waits for its table and gives up
    }
    if (!reader.reserve(kCbShadeTable, 1)) {
        return;
    }
    // the table: a1's in the bank a8 - 1 (the sun region), else the surface's tensor
    const uint64_t table_source = command.a1 != 0
        ? get_noc_addr_from_bank_id<true>(command.a8 - 1, command.a1)
        : get_noc_addr(
            0,
            InterleavedAddrGen<true>{
                .bank_base_address = table_address, .page_size = kLightTableBytes
            }
        );
    noc_async_read(table_source, get_write_ptr(kCbShadeTable), kLightTableBytes);
    noc_async_read_barrier();  // the camera after the table's words (a5, bank a8 - 1)
    noc_async_read(
        get_noc_addr_from_bank_id<true>(command.a8 - 1, command.a5),
        get_write_ptr(kCbShadeTable) + 4 * kLtCamera,
        4 * kCamWordsShade
    );
    if (glass) {  // SHADE's table (a2) after LIGHT's: the materials' tau and weight
        noc_async_read(
            get_noc_addr(
                0,
                InterleavedAddrGen<true>{
                    .bank_base_address = command.a2, .page_size = kShadeTableBytes
                }
            ),
            get_write_ptr(kCbShadeTable) + 4 * kLightShadeTableAt,
            kLightShadeTableBytes
        );
    }
    if (!read_done(reader, command.sequence)) {
        return;
    }
    cb_push_back(kCbShadeTable, 1);
    const InterleavedAddrGen<true> light_pages = {
        .bank_base_address = pages_address, .page_size = kTexPageBytes
    };
    const uint32_t slots = get_write_ptr(kCbShadeSlots);
    const uint32_t ring = get_local_cb_interface(kCbShadePlanes).fifo_size;
    const uint32_t ring_end = get_local_cb_interface(kCbShadePlanes).fifo_limit;
    uint32_t item = 0;
    bool ended = false, failed = false;
    // One step at a time, asks first: true once the pack thread's end has come.
    auto step = [&]() -> bool {
        switch (serve_ask(reader, command, light_pages, slots)) {
            case kAskEnd:
                ended = true;
                return true;
            case kAskFailed:
                failed = true;
                return true;
            case kAskServed:
            case kAskBlocked:
                return false;
            case kAskNone:
                break;
        }
        if (item > items || !cb_pages_reservable_at_back(kCbIn, 1)) {
            return false;
        }
        if (item == items) {  // the end's word
            cb_reserve_back(kCbIn, 1);
            *km::l1(get_write_ptr(kCbIn)) = kShadeAskEnd;
            cb_push_back(kCbIn, 1);
            ++item;
            return false;
        }
        if (!cb_pages_reservable_at_back(kCbShadePlanes, item_pages)) {
            return false;
        }
        const uint32_t id = directory[kDirIds + item];
        const uint32_t tile = id & kItemSlot;
        const uint32_t page = tile * kPlanes;
        const uint32_t* second_words = surface + kDirFrameMax + 1;
        if (
            tile >= 1024 || page / tensor_pages >= kDirFrameMax ||
            page / surface[0] >= kDirFrameMax ||
            page / second_words[0] >= kDirFrameMax ||
            (glass && page / glass_frame[0] >= kDirFrameMax)
        ) {
            failed = true;
            return true;
        }
        const InterleavedAddrGen<true> tensors[4] = {
            {
                .bank_base_address = surface[1 + page / surface[0]],
                .page_size = kOutPageBytes
            },
            {
                .bank_base_address = second_words[1 + page / second_words[0]],
                .page_size = kOutPageBytes
            },
            {
                .bank_base_address = directory[kDirFrame + 1 + page / tensor_pages],
                .page_size = kOutPageBytes
            },
            {
                .bank_base_address = glass ? glass_frame[1 + page / glass_frame[0]] : 0,
                .page_size = kOutPageBytes
            }
        };
        const uint32_t tensor_size[4] = {
            surface[0], second_words[0], tensor_pages, glass ? glass_frame[0] : 1
        };
        cb_reserve_back(kCbShadePlanes, item_pages);
        uint32_t destination = get_write_ptr(kCbShadePlanes);
        // the surface's planes 0 to 9, then the frame's 1/z and material word (glass:
        // the glass layer's, then the opaque frame's colour planes and 1/z)
        for (uint32_t at = 0; at < item_planes; ++at) {
            uint32_t source = at < 2 * kPlanes ? at / kPlanes : glass ? 3 : 2;
            uint32_t plane = (
                at < 2 * kPlanes ? at % kPlanes : (at == 2 * kPlanes ? 0 : kPlanes - 1)
            );
            if (at >= kLightGlassC) {  // C, then the merged 1/z
                source = 2;
                plane = at < kLightGlassDepth ? 1 + at - kLightGlassC : 0;
            }
            noc_async_read(
                get_noc_addr((page + plane) % tensor_size[source], tensors[source]),
                destination,
                kOutPageBytes
            );
            destination += kOutPageBytes;
            if (destination >= ring_end) {
                destination -= ring;
            }
        }
        if (!read_done(reader, command.sequence)) {
            failed = true;
            return true;
        }
        for (uint32_t pushed = 0; pushed < item_pages; pushed += kPlanes) {
            cb_push_back(kCbShadePlanes, kPlanes);  // five at a time: none wraps
        }
        const uint32_t first_row = id >> kItemRow0Shift & kItemRowMask;
        const uint32_t end_row = id >> kItemRow1Shift & kItemRowMask;
        cb_reserve_back(kCbIn, 1);
        volatile tt_l1_ptr uint32_t* words = km::l1(get_write_ptr(kCbIn));
        words[0] = end_row > 0 ? first_row | end_row << 8 : 32u << 8;
        words[1] = tile;
        cb_push_back(kCbIn, 1);
        ++item;
        return false;
    };
    const uint32_t code = reader.poll(step, reader.args().wait_limit);
    if (code != km::kGaveNone || failed || !ended) {
        reader.gave_up(
            km::kStReader,
            code != km::kGaveNone ? code : kGaveBadItem,
            kWhereShadeReader,
            0,
            command.sequence
        );
    }
}
