// COUNT and SETUP compute: unpack hands each mesh page to pack, which runs the vector
// unit, keeps or clips each triangle, and counts (COUNT) or sends (SETUP) its records.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "komorebi/device/km_worker.hpp"
#include "setup.h"
#ifdef TRISC_PACK
#include "setup.hpp"
#endif

namespace {

    // A give-up into the writer's gave-up word (the first one is kept) and the stop
    // word.
    void give_up(uint32_t code, uint32_t where) {
        const uint32_t control = get_arg_val<uint32_t>(0);
        volatile uint32_t* state = (
            reinterpret_cast<volatile uint32_t*>(control + KM_CONTROL_STATE)
        );
        if (state[km::kStWriter] == 0) {
            state[km::kStWriter] = code | where << 8;
        }
        *reinterpret_cast<volatile uint32_t*>(control + KM_CONTROL_STOP) = 1;
    }

    // Polls ready() for at most kWaitCycles. False: gave up (recorded at `where`).
    template <typename Ready>
    bool timed(Ready ready, uint32_t where) {
        const uint32_t started = km::wall();
        while (!ready()) {
            if (km::wall() - started > kWaitCycles) {
                give_up(kGaveWait, where);
                return false;
            }
        }
        return true;
    }

    bool mail_from(ckernel::ThreadId thread, uint32_t where) {
        return timed(
            [&] {
                return ckernel::mailbox_not_empty(thread);
            },
            where
        );
    }

#ifdef TRISC_UNPACK
    // Waits for a page of `cb` and gives its address to the pack thread.
    bool hand(uint32_t cb, uint32_t where) {
        const bool page = timed(
            [&] {
                const uint16_t received = (
                    reg_read(reinterpret_cast<uint32_t>(get_cb_tiles_received_ptr(cb)))
                );
                const uint16_t acked = get_local_cb_interface(cb).tiles_acked;
                return static_cast<uint16_t>(received - acked) >= 1;
            },
            where
        );
        if (!page) {
            return false;
        }
        cb_wait_front(cb, 1);
        ckernel::mailbox_write(
            ckernel::ThreadId::PackThreadId, get_tile_l1_byte_address(cb, 0)
        );
        return true;
    }
#endif

}  // namespace

#ifdef TRISC_PACK
namespace {

    // float32 operations one at a time, so that nothing is fused. The cores' floats
    // round to nearest and flush results below the normal range; they cannot divide.
    inline float fsub(float a, float b) {
        volatile float result = a - b;
        return result;
    }

    inline float fmul(float a, float b) {
        volatile float result = a * b;
        return result;
    }

    inline float fadd(float a, float b) {
        volatile float result = a + b;
        return result;
    }

    inline uint32_t bits(float number) {
        uint32_t pattern;
        __builtin_memcpy(&pattern, &number, 4);
        return pattern;
    }

    inline float value(uint32_t pattern) {
        float number;
        __builtin_memcpy(&number, &pattern, 4);
        return number;
    }

    // a / b rounded to nearest even, in integers: a normal or 0, b normal, a result
    // below the normal range flushed to a signed 0 as the cores' floats do.
    inline float fdiv(float a, float b) {
        const uint32_t bits_a = bits(a), bits_b = bits(b);
        const uint32_t sign = (bits_a ^ bits_b) & 0x80000000;
        if ((bits_a & 0x7FFFFFFF) == 0) {
            return value(sign);
        }
        uint32_t significand_a = (bits_a & 0x7FFFFF) | 0x800000;
        const uint32_t significand_b = (bits_b & 0x7FFFFF) | 0x800000;
        int32_t exponent = (
            static_cast<int32_t>(bits_a >> 23 & 0xFF) -
            static_cast<int32_t>(bits_b >> 23 & 0xFF) + 127
        );
        if (significand_a < significand_b) {
            significand_a <<= 1;
            exponent -= 1;
        }
        // 25 quotient bits (24 and a guard bit), the remainder as the sticky bit
        uint32_t quotient = 0, remainder = significand_a;
        for (uint32_t i = 0; i < 25; ++i) {
            quotient <<= 1;
            if (remainder >= significand_b) {
                remainder -= significand_b;
                quotient |= 1;
            }
            remainder <<= 1;
        }
        uint32_t significand = quotient >> 1;
        if ((quotient & 1) && (remainder != 0 || (significand & 1))) {
            ++significand;
        }
        if (significand == 0x1000000) {
            significand >>= 1;
            ++exponent;
        }
        if (exponent <= 0) {
            return value(sign);
        }
        return (
            value(
                sign | static_cast<uint32_t>(exponent) << 23 | (significand & 0x7FFFFF)
            )
        );
    }

    // An integer below 2^24 as float32 bits (exact).
    inline uint32_t float_bits(uint32_t integer) {
        if (integer == 0) {
            return 0;
        }
        const uint32_t exponent = 31 - __builtin_clz(integer);
        return (127 + exponent) << 23 | ((integer << (23 - exponent)) & 0x7FFFFF);
    }

    // A corner: camera-space x, y, z, then red, green, blue.
    struct Corner {
        float values[6];
    };

    // The polygon of a triangle cut at z = near (ref/raster.py clip_polygon): each
    // corner beyond, and a point where an edge crosses, from its corner beyond.
    uint32_t clip(const Corner* corners, float near, Corner* polygon) {
        uint32_t count = 0;
        for (uint32_t corner = 0; corner < 3; ++corner) {
            const Corner& start = corners[corner];
            const Corner& end = corners[(corner + 1) % 3];
            // z > near on the bits (near > 0, z not NaN): no float compare, whose flag
            // save and restore (frflags, fsflags) the cores' documentation leaves out.
            const int32_t near_bits = static_cast<int32_t>(bits(near));
            const bool start_beyond = (
                static_cast<int32_t>(bits(start.values[2])) > near_bits
            );
            const bool end_beyond = (
                static_cast<int32_t>(bits(end.values[2])) > near_bits
            );
            if (start_beyond) {
                polygon[count++] = start;
            }
            if (start_beyond != end_beyond) {
                const Corner& inside = start_beyond ? start : end;
                const Corner& outside = start_beyond ? end : start;
                const float fraction = (
                    fdiv(
                        fsub(near, inside.values[2]),
                        fsub(outside.values[2], inside.values[2])
                    )
                );
                for (uint32_t i = 0; i < 6; ++i) {
                    polygon[count].values[i] = (
                        fadd(
                            inside.values[i],
                            fmul(fraction, fsub(outside.values[i], inside.values[i]))
                        )
                    );
                }
                polygon[count].values[2] = near;
                ++count;
            }
        }
        return count;
    }

    // An integer in [0, 2048) as float16 bits.
    inline uint32_t half(uint32_t integer) {
        if (integer == 0) {
            return 0;
        }
        const uint32_t exponent = 31 - __builtin_clz(integer);
        return (exponent + 15) << 10 | ((integer << (10 - exponent)) & 0x3FF);
    }

    // The box of a raw record: columns x0, x1 and rows y0, y1.
    struct Box {
        int32_t first_column, last_column, first_row, last_row;
    };

    inline Box box_of(const uint32_t* words) {
        return {
            static_cast<int32_t>(words[kRawX0] - kMagic),
            static_cast<int32_t>(words[kRawX1] - kMagic),
            static_cast<int32_t>(words[kRawY0] - kMagic),
            static_cast<int32_t>(words[kRawY1] - kMagic)
        };
    }

    // Kept by the reference's cull: a non-zero area and a box that is not empty.
    inline bool kept(const uint32_t* words, const Box& box) {
        return (
            (words[kRawArea] & 0x7FFFFFFF) != 0 &&
            box.first_column <= box.last_column && box.first_row <= box.last_row
        );
    }

    // The words the vector unit leaves raw: thresholds (top-left), box, triangle.
    void finish(uint32_t* words, const Box& box, uint32_t triangle) {
        for (uint32_t corner = 0; corner < 3; ++corner) {
            const int32_t edge_a = static_cast<int32_t>(words[corner]);
            const int32_t edge_b = static_cast<int32_t>(words[3 + corner]);
            const bool top_left = (
                edge_a > 0 || ((words[corner] & 0x7FFFFFFF) == 0 && edge_b > 0)
            );
            words[9 + corner] = top_left ? 0x80000001 : 0;
        }
        words[24] = half(box.first_column);
        words[25] = half(box.last_column + 1);
        words[26] = box.first_row;
        words[27] = box.last_row;
        words[28] = float_bits(triangle);
        words[29] = 0;
        words[30] = 0;
        words[31] = 0;
    }

    uint32_t* cb_ptr(uint32_t cb) {
        return reinterpret_cast<uint32_t*>(get_local_cb_interface(cb).fifo_wr_ptr << 4);
    }

    // Zeroes n words (a multiple of 32), 32 per vector store.
    void zero(uint32_t* words, uint32_t count) {
#if defined(__riscv_vector)
        const size_t vector_length = __riscv_vsetvl_e32m8(32);
        const vuint32m8_t zeros = __riscv_vmv_v_x_u32m8(0, vector_length);
        for (uint32_t i = 0; i < count; i += 32) {
            __riscv_vse32_v_u32m8(words + i, zeros, vector_length);
        }
        asm volatile("" ::: "memory");
#else
        for (uint32_t i = 0; i < count; ++i) {
            words[i] = 0;
        }
#endif
    }

    // SETUP's out pages: up to kOutEntries records, each with its target slot.
    struct Out {
        uint32_t* page = nullptr;
        uint32_t filled = 0;
        uint32_t wait = 0;     // cycles waiting for a free out page
        uint32_t entries = 0;  // records sent

        bool open(const km::Compute& compute) {
            if (page == nullptr) {
                const uint32_t asked = km::wall();
                const bool reserved = compute.reserve_back(kCbOut, 1);
                wait += km::wall() - asked;
                if (!reserved) {
                    return false;
                }
                page = cb_ptr(kCbOut);
                filled = 0;
            }
            return true;
        }

        bool add(const km::Compute& compute, const uint32_t* record, uint32_t slot) {
            if (!open(compute)) {
                return false;
            }
            uint32_t* destination = page + 32 * (1 + filled);
#if defined(__riscv_vector)
            const size_t vector_length = __riscv_vsetvl_e32m8(32);
            __riscv_vse32_v_u32m8(
                destination, __riscv_vle32_v_u32m8(record, vector_length), vector_length
            );
            asm volatile("" ::: "memory");
#else
            for (uint32_t word = 0; word < 32; ++word) {
                destination[word] = record[word];
            }
#endif
            page[1 + filled] = slot;
            ++entries;
            if (++filled == kOutEntries) {
                push(false);
            }
            return true;
        }

        void push(bool last) {
            page[0] = filled | (last ? kOutLast : 0);
            asm volatile("fence" ::: "memory");
            const uint32_t last_word = reinterpret_cast<volatile uint32_t*>(page)[0];
            asm volatile("and x0, x0, %0" : : "r"(last_word) : "memory");
            cb_push_back(kCbOut, 1);
            page = nullptr;
        }

        // The last page (possibly without entries) tells the writer to stop.
        bool end(const km::Compute& compute) {
            if (!open(compute)) {
                return false;
            }
            push(true);
            return true;
        }
    };

    // Corners beyond the near distance (z > near > 0: float32 order as integers).
    uint32_t beyond_of(const uint32_t* words, uint32_t near) {
        uint32_t count = 0;
        for (uint32_t corner = 0; corner < 3; ++corner) {
            count += (
                static_cast<int32_t>(words[kRawZ + corner]) > static_cast<int32_t>(near)
            );
        }
        return count;
    }

    // One pass's state: the mode, the counters (COUNT) or the tiles' next slots and
    // the out pages (SETUP).
    struct Pass {
        const km::Compute& compute;
        uint32_t handler;
        uint32_t* counters = nullptr;  // COUNT: entries of tile t at t, rows at kTiles + t
        uint16_t* histogram = nullptr;  // COUNT: tile t's start and end counts at 32 t
        uint32_t* next_slot = nullptr;  // SETUP: item i's next slot
        const uint32_t* items = nullptr;  // SETUP: the items page (setup.h)
        uint32_t list_slots = 0;          // SETUP: the lists' slots
        Out out;
        bool succeeded = true;
        uint32_t kept_count = 0, clipped = 0;  // the profile's counts

        // A finished record that touches `box` (inside the image, or a give-up).
        void record(const uint32_t* words, const Box& box) {
            if (
                box.first_column < 0 || box.first_row < 0 ||
                box.last_column >= static_cast<int32_t>(kWidth) ||
                box.last_row >= static_cast<int32_t>(kHeight)
            ) {
                give_up(kGaveBadBox, words[28] & 0xFFFF);
                succeeded = false;
                return;
            }
            for (
                int32_t tile_y = box.first_row >> 5; tile_y <= (box.last_row >> 5);
                ++tile_y
            ) {
                const int32_t top = 32 * tile_y;
                const int32_t first_row = box.first_row > top ? box.first_row : top;
                const int32_t last_row = (
                    box.last_row < top + 31 ? box.last_row : top + 31
                );
                for (
                    int32_t tile_x = box.first_column >> 5;
                    tile_x <= (box.last_column >> 5);
                    ++tile_x
                ) {
                    const uint32_t tile = kTilesX * tile_y + tile_x;
                    if (handler == kHandlerCount) {
                        counters[tile] += 1;
                        counters[kTiles + tile] += last_row - first_row + 1;
                        uint16_t* tile_histogram = histogram + 32 * tile;
                        if (
                            ++tile_histogram[(first_row - top) >> 1] == 0 ||
                            ++tile_histogram[16 + ((last_row - top) >> 1)] == 0
                        ) {
                            give_up(kGaveBadHist, tile);
                            succeeded = false;
                            return;
                        }
                        continue;
                    }
                    const uint32_t first = items[tile] & 0xFFFF;
                    const uint32_t item_count = items[tile] >> 16;
                    if (first + item_count > kSlotsBytes / 4) {
                        give_up(kGaveBadSlot, tile);
                        succeeded = false;
                        return;
                    }
                    const uint16_t* rows = (
                        reinterpret_cast<const uint16_t*>(items + kItemRows / 4)
                    );
                    for (uint32_t i = first; i < first + item_count; ++i) {
                        // the item's rows [item_first_row, item_end_row) against the
                        // entry's 2-row groups
                        const uint32_t item_first_row = rows[i] & 0xFF;
                        const uint32_t item_end_row = rows[i] >> 8;
                        if (
                            ((first_row - top) >> 1) >=
                                static_cast<int32_t>(item_end_row >> 1) ||
                            ((last_row - top) >> 1) <
                                static_cast<int32_t>(item_first_row >> 1)
                        ) {
                            continue;
                        }
                        const uint32_t entry_slot = next_slot[i]++;
                        if (entry_slot >= list_slots) {
                            give_up(kGaveBadSlot, tile);
                            succeeded = false;
                            return;
                        }
                        if (!out.add(compute, words, entry_slot)) {
                            succeeded = false;
                            return;
                        }
                    }
                }
            }
        }

        // Clips a crossing triangle and sends its pieces (one or two) through the
        // vector unit; their records go out as the triangle's. `records`: free space.
        void pieces(
            const Corner* corners, float near, uint32_t* records, uint32_t triangle
        ) {
            Corner poly[6];  // at most 4; room for 6 so the check below comes first
            const uint32_t polygon_size = clip(corners, near, poly);
            if (polygon_size < 3 || polygon_size > 4) {
                give_up(kGaveBadClip, triangle & 0xFFFF);
                succeeded = false;
                return;
            }
            constexpr uint32_t fan[2][3] = {{0, 1, 2}, {0, 2, 3}};
            const uint32_t piece_count = polygon_size - 2;
            ++clipped;
            volatile uint32_t* window = setup_pass::window();
            for (uint32_t j = 0; j < piece_count; ++j) {
                for (uint32_t vertex = 0; vertex < 3; ++vertex) {
                    const Corner& corner = poly[fan[j][vertex]];
                    for (uint32_t i = 0; i < 3; ++i) {
                        window
                            [setup_pass::word(
                                setup_pass::kSlotCameraSpace + 3 * vertex + i
                            ) +
                             2 * j] = bits(corner.values[i]);
                        window
                            [setup_pass::word(
                                setup_pass::kSlotColour + 3 * vertex + i
                            ) +
                             2 * j] = bits(corner.values[3 + i]);
                    }
                }
            }
            setup_pass::order(
                setup_pass::word(setup_pass::kSlotColour + 8) + 2 * (piece_count - 1)
            );
            setup_pass::triangles(handler == kHandlerSetup);
            ckernel::tensix_sync();
            setup_pass::records_out(records);
            for (uint32_t j = 0; j < piece_count && succeeded; ++j) {
                uint32_t* words = records + 32 * j;
                const Box box = box_of(words);
                if (!kept(words, box)) {
                    continue;
                }
                finish(words, box, triangle);
                ++kept_count;
                record(words, box);
            }
        }

        // The block's kept triangles in order, from raw records at `records` (Dst holds
        // the block), a crossing one's pieces at its place; `work`: 2,048 words.
        void block(
            uint32_t* records,
            const uint32_t* mesh,
            uint32_t* work,
            uint32_t first_triangle
        ) {
            const uint32_t near = (
                setup_pass::window()[setup_pass::word(setup_pass::kSlotCamera + 15)]
            );
            Corner* corners = reinterpret_cast<Corner*>(work + 1024);
            uint32_t crossing = 0;
            for (uint32_t lane = 0; lane < 32; ++lane) {
                const uint32_t beyond = beyond_of(records + 32 * lane, near);
                if (beyond != 1 && beyond != 2) {
                    continue;
                }
                crossing |= 1u << lane;
                for (uint32_t vertex = 0; vertex < 3; ++vertex) {
                    Corner& corner = corners[3 * lane + vertex];
                    for (uint32_t i = 0; i < 3; ++i) {
                        const uint32_t camera_slot = (
                            setup_pass::word(
                                setup_pass::kSlotCameraSpace + 3 * vertex + i
                            ) +
                            2 * lane
                        );
                        const uint32_t colour_word = (
                            setup_pass::word(setup_pass::kSlotColour + 3 * vertex + i) +
                            2 * lane
                        );
                        corner.values[i] = value(setup_pass::window()[camera_slot]);
                        corner.values[3 + i] = value(mesh[colour_word]);
                    }
                }
            }
            for (uint32_t lane = 0; lane < 32 && succeeded; ++lane) {
                uint32_t* words = records + 32 * lane;
                if (crossing >> lane & 1) {
                    pieces(
                        corners + 3 * lane, value(near), work, first_triangle + lane
                    );
                    continue;
                }
                if (beyond_of(words, near) != 3) {
                    continue;
                }
                const Box box = box_of(words);
                if (!kept(words, box)) {
                    continue;
                }
                finish(words, box, first_triangle + lane);
                ++kept_count;
                record(words, box);
            }
        }
    };

}  // namespace
#endif

// The passes' work for one command; the frame program's compute kernel calls it as
// well.
void setup_compute_command(const km::Compute& compute, const km_command_t& command) {
    const uint32_t handler = km::handler(command);
    const uint32_t cores = command.a5;
    if (
        (
            handler != kHandlerLanes && handler != kHandlerCount &&
            handler != kHandlerSetup
        ) ||
        cores == 0 || compute.me() >= cores
    ) {
        return;
    }
    const uint32_t blocks_word = command.a2, list_slots = command.a6;
    const uint32_t profile_address = command.a9;
    // with kRanges the camera's page brings the core's range (setup_reader.cpp)
    const bool ranged = (blocks_word & kRanges) != 0;
    const uint32_t blocks = blocks_word & ~kRanges;
    uint32_t first_block = compute.me() * blocks / cores;
    uint32_t pages = 1 + ((compute.me() + 1) * blocks / cores - first_block);
#ifdef TRISC_UNPACK
    // SETUP: the slots and items pages first, held until the pass ends.
    const bool held = (
        handler != kHandlerSetup || (hand(kCbSlots, 4) && hand(kCbItems, 4))
    );
    // Hands each page to the pack thread; frees it when the pack thread is done.
    for (uint32_t page = 0; held && page < pages; ++page) {
        if (!hand(kCbMesh, 1)) {
            break;
        }
        if (page == 0 && ranged) {
            asm volatile("fence" ::: "memory");
            const volatile uint32_t*
                camera = (
                    reinterpret_cast<const volatile uint32_t*>(
                        get_tile_l1_byte_address(kCbMesh, 0)
                    )
                );
            pages = 1 + (camera[kCamWords + 1] - camera[kCamWords]);
        }
        if (!mail_from(ckernel::ThreadId::PackThreadId, 2)) {
            break;
        }
        (void)ckernel::mailbox_read(ckernel::ThreadId::PackThreadId);
        cb_pop_front(kCbMesh, 1);
    }
    if (handler == kHandlerSetup && held) {
        cb_pop_front(kCbSlots, 1);
        cb_pop_front(kCbItems, 1);
    }
#endif
#ifdef TRISC_PACK
    const uint32_t started = km::wall();
    uint32_t mail = 0, vector_cycles = 0, scalar_cycles = 0;
    setup_pass::setup();
    Pass pass{compute, handler};
    if (handler == kHandlerSetup) {
        if (!mail_from(ckernel::ThreadId::UnpackThreadId, 3)) {
            return;
        }
        pass.next_slot = (
            reinterpret_cast<uint32_t*>(
                ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)
            )
        );
        if (!mail_from(ckernel::ThreadId::UnpackThreadId, 3)) {
            return;
        }
        pass.items = (
            reinterpret_cast<const uint32_t*>(
                ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)
            )
        );
        pass.list_slots = list_slots & ~kSetupPrefix;
    }
    if (handler == kHandlerCount) {
        if (!compute.reserve_back(kCbCount, 1)) {
            return;
        }
        pass.counters = cb_ptr(kCbCount);
        zero(pass.counters, 2 * kTiles);
        // never pushed: the buffer's space, kept for HIST
        uint32_t* histogram_words = cb_ptr(kCbHist);
        zero(histogram_words, kHistBytes / 4);
        pass.histogram = reinterpret_cast<uint16_t*>(histogram_words);
    }
    uint32_t* work = cb_ptr(kCbWork);
    for (uint32_t page = 0; page < pages; ++page) {
        const uint32_t asked = km::wall();
        if (!mail_from(ckernel::ThreadId::UnpackThreadId, 3)) {
            pass.succeeded = false;
            break;
        }
        mail += km::wall() - asked;
        const uint32_t* mesh_words = (
            reinterpret_cast<const uint32_t*>(
                ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)
            )
        );
        asm volatile("fence" ::: "memory");
        if (page == 0) {
            if (ranged) {
                first_block = mesh_words[kCamWords];
                pages = 1 + (mesh_words[kCamWords + 1] - first_block);
            }
            // the camera: each number to every lane of its slot
            for (uint32_t i = 0; i < kCamWords; ++i) {
                setup_pass::fill_slot(setup_pass::kSlotCamera + i, mesh_words[i]);
            }
            setup_pass::order(
                setup_pass::word(setup_pass::kSlotCamera + kCamWords - 1) + 62
            );
            ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
            continue;
        }
        const uint32_t vector_started = km::wall();
        setup_pass::block_in(mesh_words);
        setup_pass::camera_space();
        setup_pass::triangles(handler != kHandlerCount);
        ckernel::tensix_sync();
        if (handler == kHandlerLanes) {
            ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
            if (!compute.reserve_back(kCbRec, 1)) {
                break;
            }
            uint32_t* records = cb_ptr(kCbRec);
            setup_pass::records_out(records);
            asm volatile("fence" ::: "memory");
            const uint32_t last_word = (
                reinterpret_cast<volatile uint32_t*>(records)[1023]
            );
            asm volatile("and x0, x0, %0" : : "r"(last_word) : "memory");
            cb_push_back(kCbRec, 1);
            continue;
        }
        setup_pass::records_out(work);
        const uint32_t scalar_started = km::wall();
        vector_cycles += scalar_started - vector_started;
        pass.block(work, mesh_words, work + 1024, 32 * (first_block + page - 1));
        scalar_cycles += km::wall() - scalar_started;
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
        if (!pass.succeeded) {
            break;
        }
    }
    if (profile_address != 0) {
        volatile uint32_t* profile = cb_ptr(kCbSProf);
        profile[kSProfPack + 0] = mail;
        profile[kSProfPack + 1] = vector_cycles;
        profile[kSProfPack + 2] = scalar_cycles;
        profile[kSProfPack + 3] = km::wall() - started;
        profile[kSProfPack + 4] = pass.kept_count;
        profile[kSProfPack + 5] = pass.clipped;
        profile[kSProfPack + 6] = pass.out.entries;
        profile[kSProfPack + 7] = pass.out.wait;
        asm volatile("fence" ::: "memory");
    }
    if (handler == kHandlerCount && pass.counters != nullptr) {
        asm volatile("fence" ::: "memory");
        const uint32_t counters_last = (
            reinterpret_cast<volatile uint32_t*>(pass.counters)[2 * kTiles - 1]
        );
        const uint32_t histogram_last = (
            reinterpret_cast<volatile uint32_t*>(pass.histogram)[kHistBytes / 4 - 1]
        );
        asm volatile("and x0, x0, %0" : : "r"(counters_last) : "memory");
        asm volatile("and x0, x0, %0" : : "r"(histogram_last) : "memory");
        cb_push_back(kCbCount, 1);
    }
    if (handler == kHandlerSetup && pass.succeeded) {
        pass.out.end(compute);
    }
    setup_pass::finish();
#endif
}

#ifndef RASTER_NO_ENTRY
void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kCbMesh, kCbRec);
    compute.run([&](const km_command_t& command) {
        setup_compute_command(compute, command);
    });
}
#endif
