// LIGHT's reader (shade.h): the lights' table, then for every item of the directory
// its rows (an answer page) and its planes: the surface's, the frame's material word.
#include "shade.h"

void light_reader_command(const km::Reader& reader, const km_command_t& command) {
    const uint32_t directory_address = command.a0;
    const uint32_t scratch = get_write_ptr(kCbDirReader);
    volatile tt_l1_ptr uint32_t* directory = km::l1(scratch);
    uint32_t surface[kSurfaceWords + 2] = {};
    if (!surface_words(reader, command, scratch, surface)) {
        return;
    }
    const uint32_t table_address = surface[kSurfaceWords];
    if (!km::read_pages(
        reader, command, Directories(directory_address), reader.me(), 1, scratch
    ) ||
        !read_done(reader, command.sequence)) {
        return;
    }
    invalidate_l1_cache();
    const uint32_t items = directory[kDirItems], tensor_pages = directory[kDirFrame];
    if (tensor_pages == 0 || tensor_pages % kPlanes != 0 || items > kDirMaxItems) {
        return;  // the pack thread waits for its answer and gives up
    }
    if (!reader.reserve(kCbShadeTable, 1)) {
        return;
    }
    noc_async_read(
        get_noc_addr(
            0,
            InterleavedAddrGen<true>{
                .bank_base_address = table_address, .page_size = kLightTableBytes
            }
        ),
        get_write_ptr(kCbShadeTable),
        kLightTableBytes
    );
    if (!read_done(reader, command.sequence)) {
        return;
    }
    cb_push_back(kCbShadeTable, 1);
    uint32_t item = 0;
    bool failed = false;
    // An item when its answer and planes fit; true once the end's answer is pushed.
    auto step = [&]() -> bool {
        if (!cb_pages_reservable_at_back(kCbShadeDone, 1)) {
            return false;
        }
        if (item == items) {
            cb_reserve_back(kCbShadeDone, 1);
            *km::l1(get_write_ptr(kCbShadeDone)) = kShadeAskEnd;
            cb_push_back(kCbShadeDone, 1);
            return true;
        }
        if (!cb_pages_reservable_at_back(kCbShadePlanes, kLightItemPages)) {
            return false;
        }
        const uint32_t id = directory[kDirIds + item];
        const uint32_t slot = id & kItemSlot;
        const uint32_t page = slot * kPlanes;
        const uint32_t* second_words = surface + kDirFrameMax + 1;
        if (
            page / tensor_pages >= kDirFrameMax || page / surface[0] >= kDirFrameMax ||
            page / second_words[0] >= kDirFrameMax
        ) {
            failed = true;
            return true;
        }
        const InterleavedAddrGen<true> frame_tensor = {
            .bank_base_address = directory[kDirFrame + 1 + page / tensor_pages],
            .page_size = kOutPageBytes
        };
        const InterleavedAddrGen<true> first_tensor = {
            .bank_base_address = surface[1 + page / surface[0]],
            .page_size = kOutPageBytes
        };
        const InterleavedAddrGen<true> second_tensor = {
            .bank_base_address = second_words[1 + page / second_words[0]],
            .page_size = kOutPageBytes
        };
        cb_reserve_back(kCbShadePlanes, kLightItemPages);
        const uint32_t destination = get_write_ptr(kCbShadePlanes);
        uint32_t second = destination + kPlanes * kOutPageBytes;
        if (second >= get_local_cb_interface(kCbShadePlanes).fifo_limit) {
            second -= get_local_cb_interface(kCbShadePlanes).fifo_size;
        }
        // the surface's planes 0 to 4 (a, normal x, y); the material word, normal z,
        // visibility (the second frame's first two)
        for (uint32_t plane = 0; plane < kPlanes; ++plane) {
            noc_async_read(
                get_noc_addr((page + plane) % surface[0], first_tensor),
                destination + plane * kOutPageBytes,
                kOutPageBytes
            );
        }
        noc_async_read(
            get_noc_addr((page + kPlanes - 1) % tensor_pages, frame_tensor),
            second,
            kOutPageBytes
        );
        for (uint32_t plane = 0; plane < 2; ++plane) {
            noc_async_read(
                get_noc_addr((page + plane) % second_words[0], second_tensor),
                second + (1 + plane) * kOutPageBytes,
                kOutPageBytes
            );
        }
        if (!read_done(reader, command.sequence)) {
            failed = true;
            return true;
        }
        cb_push_back(kCbShadePlanes, kPlanes);  // five at a time: none wraps
        cb_push_back(kCbShadePlanes, kPlanes);
        const uint32_t first_row = id >> kItemRow0Shift & kItemRowMask;
        const uint32_t end_row = id >> kItemRow1Shift & kItemRowMask;
        cb_reserve_back(kCbShadeDone, 1);
        *km::l1(
            get_write_ptr(kCbShadeDone)
        ) = (end_row > 0 ? first_row | end_row << 8 : 32u << 8);
        cb_push_back(kCbShadeDone, 1);
        ++item;
        return false;
    };
    const uint32_t code = reader.poll(step, reader.args().wait_limit);
    if (code != km::kGaveNone || failed) {
        reader.gave_up(
            km::kStReader,
            code != km::kGaveNone ? code : kGaveBadItem,
            kWhereShadeReader,
            0,
            command.sequence
        );
    }
}
