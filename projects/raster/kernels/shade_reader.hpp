// SHADE's reader (shade.h): the table, list pages, each item's planes (a6: then the
// opaque layer's of its tile), the texture pages the pack thread asks for into slots.
#include "shade.h"

void shade_reader_command(const km::Reader& reader, const km_command_t& command) {
    const uint32_t directory_address = command.a0, lists_address = command.a1;
    const uint32_t textures_address = command.a2, table_address = command.a3;
    const uint32_t opaque_address = command.a6;
    const uint32_t item_pages = opaque_address != 0 ? 2 * kPlanes : kPlanes;
    const uint32_t scratch = get_write_ptr(kCbDirReader);
    volatile tt_l1_ptr uint32_t* directory = km::l1(scratch);
    uint32_t opaque[kDirFrameMax + 1] = {};  // the opaque layer's frame words
    if (opaque_address != 0) {
        if (!km::read_pages(
            reader, command, Directories(opaque_address), reader.me(), 1, scratch
        ) ||
            !read_done(reader, command.sequence)) {
            return;
        }
        invalidate_l1_cache();
        for (uint32_t i = 0; i <= kDirFrameMax; ++i) {
            opaque[i] = directory[kDirFrame + i];
        }
    }
    if (!km::read_pages(
        reader, command, Directories(directory_address), reader.me(), 1, scratch
    ) ||
        !read_done(reader, command.sequence)) {
        return;
    }
    invalidate_l1_cache();
    const uint32_t first = directory[kDirFirst], pages = directory[kDirPages];
    const uint32_t items = directory[kDirItems], tensor_pages = directory[kDirFrame];
    if (
        tensor_pages == 0 || tensor_pages % kPlanes != 0 || items > kDirMaxItems ||
        (opaque_address != 0 && (opaque[0] == 0 || opaque[0] % kPlanes != 0))
    ) {
        return;  // the pack thread finds the same from its list and gives up
    }
    if (!reader.reserve(kCbShadeTable, 1)) {
        return;
    }
    const InterleavedAddrGen<true> table = {
        .bank_base_address = table_address, .page_size = kShadeTableBytes
    };
    noc_async_read(
        get_noc_addr(0, table), get_write_ptr(kCbShadeTable), kShadeTableBytes
    );
    if (!read_done(reader, command.sequence)) {
        return;
    }
    cb_push_back(kCbShadeTable, 1);
    const Lists lists(lists_address);
    const InterleavedAddrGen<true> textures = {
        .bank_base_address = textures_address, .page_size = kTexPageBytes
    };
    const uint32_t slots = get_write_ptr(kCbShadeSlots);
    uint32_t page = 0, item = 0;
    bool ended = false, failed = false;
    // One step at a time, asks first: true once the pack thread's end has come.
    auto step = [&]() -> bool {
        if (cb_pages_available_at_front(kCbShadeAsk, 1)) {
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* ask = km::l1(get_read_ptr(kCbShadeAsk));
            const uint32_t count = ask[0];
            if (count == kShadeAskEnd) {
                cb_pop_front(kCbShadeAsk, 1);
                ended = true;
                return true;
            }
            if (count > kShadeSlots || !cb_pages_reservable_at_back(kCbShadeDone, 1)) {
                failed = count > kShadeSlots;
                return failed;
            }
            for (uint32_t slot = 0; slot < count; ++slot) {
                noc_async_read(
                    get_noc_addr(ask[1 + slot], textures),
                    slots + slot * kTexPageBytes,
                    kTexPageBytes
                );
            }
            if (!read_done(reader, command.sequence)) {
                failed = true;
                return true;
            }
            cb_pop_front(kCbShadeAsk, 1);
            cb_reserve_back(kCbShadeDone, 1);
            cb_push_back(kCbShadeDone, 1);
            return false;
        }
        if (page < pages && cb_pages_reservable_at_back(kCbIn, 1)) {
            cb_reserve_back(kCbIn, 1);
            if (!km::read_pages(
                reader, command, lists, first + page, 1, get_write_ptr(kCbIn)
            ) ||
                !read_done(reader, command.sequence)) {
                failed = true;
                return true;
            }
            cb_push_back(kCbIn, 1);
            ++page;
            return false;
        }
        while (item < items && (directory[kDirIds + item] & kItemEmpty) != 0) {
            ++item;
        }
        if (item < items && cb_pages_reservable_at_back(kCbShadePlanes, item_pages)) {
            const uint32_t tile_page = (
                (directory[kDirIds + item] & kItemSlot) * kPlanes
            );
            if (
                tile_page / tensor_pages >= kDirFrameMax ||
                (opaque_address != 0 && tile_page / opaque[0] >= kDirFrameMax)
            ) {
                failed = true;
                return true;
            }
            cb_reserve_back(kCbShadePlanes, item_pages);
            const InterleavedAddrGen<true> frame_tensor = {
                .bank_base_address = directory
                    [kDirFrame + 1 + tile_page / tensor_pages],
                .page_size = kOutPageBytes
            };
            const uint32_t destination = get_write_ptr(kCbShadePlanes);
            for (uint32_t plane = 0; plane < kPlanes; ++plane) {
                noc_async_read(
                    get_noc_addr((tile_page + plane) % tensor_pages, frame_tensor),
                    destination + plane * kOutPageBytes,
                    kOutPageBytes
                );
            }
            if (opaque_address != 0) {
                // the second five pages, after the first in the ring: its write place
                // is a multiple of five pages (the opaque layer's command's items)
                uint32_t second = destination + kPlanes * kOutPageBytes;
                if (second >= get_local_cb_interface(kCbShadePlanes).fifo_limit) {
                    second -= get_local_cb_interface(kCbShadePlanes).fifo_size;
                }
                const InterleavedAddrGen<true> opaque_tensor = {
                    .bank_base_address = opaque[1 + tile_page / opaque[0]],
                    .page_size = kOutPageBytes
                };
                for (uint32_t plane = 0; plane < kPlanes; ++plane) {
                    noc_async_read(
                        get_noc_addr((tile_page + plane) % opaque[0], opaque_tensor),
                        second + plane * kOutPageBytes,
                        kOutPageBytes
                    );
                }
            }
            if (!read_done(reader, command.sequence)) {
                failed = true;
                return true;
            }
            for (uint32_t pushed = 0; pushed < item_pages; pushed += kPlanes) {
                cb_push_back(kCbShadePlanes, kPlanes);  // five at a time: none wraps
            }
            ++item;
        }
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
