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
#ifndef RASTER_OUT_CB  // the out buffer; a class that includes this file names its own
#define RASTER_OUT_CB kCbOut
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

    // The polygon of a triangle cut at z = near (reference/raster.py clip_polygon):
    // each corner beyond, and a point where an edge crosses, from its corner beyond.
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

    // A shaded record's word (setup.h kRecShadeWord) from the triangle's shade words
    // (32 apart); a level = #k < levels with texels > area 2^(2k - 1), as integers.
    inline uint32_t shade_word(uint32_t area_bits, const uint32_t* shade) {
        const uint32_t material = shade[32];
        uint32_t word = material & 0xFFFF;
        for (uint32_t texture = 0; texture < 3; ++texture) {
            const uint32_t levels = material >> (16 + 4 * texture) & 0xF;
            const uint32_t texels_bits = shade[texture == 0 ? 0 : 32 + 32 * texture];
            uint32_t level = 0;
            for (uint32_t k = 1; k < levels; ++k) {
                const uint32_t exponent = (area_bits >> 23) + 2 * k - 1;
                const uint32_t threshold = (
                    exponent >= 255 ? 0x7F800000 : area_bits + ((2 * k - 1) << 23)
                );
                level += (
                    static_cast<int32_t>(texels_bits) > static_cast<int32_t>(threshold)
                );
            }
            word |= level << (16 + 4 * texture);
        }
        return word;
    }

    uint32_t* cb_ptr(uint32_t cb) {
        return reinterpret_cast<uint32_t*>(get_local_cb_interface(cb).fifo_wr_ptr << 4);
    }

    // The out pages to the writer: SETUP up to kOutEntries records, each with its box
    // word; COUNT up to kOutWords box words.
    struct Out {
        uint32_t* page = nullptr;
        uint32_t filled = 0;
        uint32_t wait = 0;     // cycles waiting for a free out page
        uint32_t entries = 0;  // records sent

        bool open(const km::Compute& compute) {
            if (page == nullptr) {
                const uint32_t asked = km::wall();
                const bool reserved = compute.reserve_back(RASTER_OUT_CB, 1);
                wait += km::wall() - asked;
                if (!reserved) {
                    return false;
                }
                page = cb_ptr(RASTER_OUT_CB);
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

        // COUNT: one box word, up to kOutWords a page.
        bool add_word(const km::Compute& compute, uint32_t box_word) {
            if (!open(compute)) {
                return false;
            }
            page[1 + filled] = box_word;
            ++entries;
            if (++filled == kOutWords) {
                push(false);
            }
            return true;
        }

        void push(bool last) {
            page[0] = filled | (last ? kOutLast : 0);
            asm volatile("fence" ::: "memory");
            const uint32_t last_word = reinterpret_cast<volatile uint32_t*>(page)[0];
            asm volatile("and x0, x0, %0" : : "r"(last_word) : "memory");
            cb_push_back(RASTER_OUT_CB, 1);
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

    // One pass's state: the mode and the out pages to the writer.
    struct Pass {
        const km::Compute& compute;
        uint32_t handler;
        Out out;
        bool succeeded = true;
        bool weighted = false;  // SETUP on a shaded frame (setup.h kShadeAddrAt)
        uint32_t kept_count = 0, clipped = 0;  // the profile's counts
        uint32_t block_cost = 0;  // COUNT: the block's predicted cost (setup.h kCostBlock)
        // COUNT with ranges: the costs summed after j x blocks / kCostKnots blocks
        uint32_t* knots = nullptr;
        uint32_t blocks = 0, block = 0, next_knot = 1, cost_sum = 0;

        // knot j once j x blocks / kCostKnots (rounded down) blocks are summed
        void advance() {
            while (
                next_knot <= kCostKnots &&
                next_knot * blocks <= kCostKnots * block + kCostKnots - 1
            ) {
                knots[next_knot++] = cost_sum;
            }
        }

        // A finished record that touches `box` (inside the image, or a give-up); COUNT
        // reads no word of it.
        void record(const uint32_t* words, const Box& box, uint32_t triangle) {
            if (
                box.first_column < 0 || box.first_row < 0 ||
                box.last_column >= static_cast<int32_t>(kWidth) ||
                box.last_row >= static_cast<int32_t>(kHeight)
            ) {
                give_up(kGaveBadBox, triangle & 0xFFFF);
                succeeded = false;
                return;
            }
            // the writer counts it (COUNT) or sends it to every item it reaches (SETUP)
            const uint32_t box_word = (
                static_cast<uint32_t>(box.first_column >> 5) |
                static_cast<uint32_t>(box.last_column >> 5) << 5 |
                static_cast<uint32_t>(box.first_row) << 10 |
                static_cast<uint32_t>(box.last_row) << 20
            );
            if (knots != nullptr) {
                block_cost += (
                    kCostKept +
                    kCostEntry *
                        ((box.last_column >> 5) - (box.first_column >> 5) + 1) *
                        ((box.last_row >> 5) - (box.first_row >> 5) + 1)
                );
            }
            const bool handed = (
                handler == kHandlerCount
                    ? out.add_word(compute, box_word)
                    : out.add(compute, words, box_word)
            );
            succeeded = succeeded && handed;
        }

        // Clips a crossing triangle and sends its pieces (one or two) through the
        // vector unit; their records go out as the triangle's. `records`: free space.
        void pieces(
            const Corner* corners,
            float near,
            uint32_t* records,
            uint32_t triangle,
            const uint32_t* shade_words
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
            setup_pass::triangles(handler == kHandlerSetup, weighted);
            ckernel::tensix_sync();
            setup_pass::records_out(records);
            const volatile uint32_t* areas = (
                setup_pass::window() + setup_pass::word(setup_pass::kSlotArea)
            );
            for (uint32_t j = 0; j < piece_count && succeeded; ++j) {
                uint32_t* words = records + 32 * j;
                const Box box = box_of(words);
                if (!kept(words, box)) {
                    continue;
                }
                finish(words, box, triangle);
                if (weighted) {
                    words[kRecShadeWord] = (shade_word(areas[2 * j], shade_words));
                }
                ++kept_count;
                record(words, box, triangle);
            }
        }

        // The same after finish_lanes. `work`: records from 0, pieces' records from
        // 1,024, corners from 2,048, the lanes' flags and boxes from 2,688.
        void block_finished(
            const uint32_t* mesh, uint32_t* work, uint32_t first_triangle
        ) {
            // pair-major (setup_pass::pairs_out): flags with first columns, last
            // columns with lane numbers, then first with last rows
            uint32_t* lane_words = work + 2688;
            setup_pass::pairs_out(setup_pass::kSlotFlags, 2, lane_words);
            setup_pass::pairs_out(
                setup_pass::kSlotRecords + kRawY0, 1, lane_words + 128
            );
            const uint32_t* rows = lane_words + 128;
            uint32_t kept_lanes = 0, crossing = 0;
            for (uint32_t lane = 0; lane < 32; ++lane) {
                const uint32_t flags = lane_words[2 * lane];
                kept_lanes |= (flags & 1) << lane;
                crossing |= (flags >> 1 & 1) << lane;
            }
            // SETUP: each kept lane's record by segment loads from the window,
            // before any clipping reuses the window
            if (handler == kHandlerSetup) {
                for (uint32_t left = kept_lanes; left != 0; left &= left - 1) {
                    const uint32_t lane = __builtin_ctz(left);
                    setup_pass::lane_record(
                        const_cast<const uint32_t*>(
                            setup_pass::window() +
                            setup_pass::word(setup_pass::kSlotRecords)
                        ),
                        lane,
                        work + 32 * lane
                    );
                }
            }
            // a shaded frame: each kept record's material and level, before any
            // clipping reuses the window (the texel areas and materials after the mesh)
            const uint32_t* shade_words = mesh + kMeshBytes / 4;
            if (handler == kHandlerSetup && weighted) {
                const volatile uint32_t* areas = (
                    setup_pass::window() + setup_pass::word(setup_pass::kSlotArea)
                );
                for (uint32_t left = kept_lanes; left != 0; left &= left - 1) {
                    const uint32_t lane = __builtin_ctz(left);
                    work[32 * lane + kRecShadeWord] = (
                        shade_word(areas[2 * lane], shade_words + lane)
                    );
                }
            }
            const uint32_t near = (
                setup_pass::window()[setup_pass::word(setup_pass::kSlotCamera + 15)]
            );
            Corner* corners = reinterpret_cast<Corner*>(work + 2048);
            for (uint32_t left = crossing; left != 0; left &= left - 1) {
                const uint32_t lane = __builtin_ctz(left);
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
            for (
                uint32_t left = kept_lanes | crossing; left != 0 && succeeded;
                left &= left - 1
            ) {
                const uint32_t lane = __builtin_ctz(left);
                if (crossing >> lane & 1) {
                    pieces(
                        corners + 3 * lane,
                        value(near),
                        work + 1024,
                        first_triangle + lane,
                        shade_words + lane
                    );
                    continue;
                }
                const Box box{
                    static_cast<int32_t>(lane_words[2 * lane + 1]),
                    static_cast<int32_t>(lane_words[64 + 2 * lane]),
                    static_cast<int32_t>(rows[2 * lane]),
                    static_cast<int32_t>(rows[2 * lane + 1])
                };
                ++kept_count;
                record(work + 32 * lane, box, first_triangle + lane);
            }
            // COUNT: the block's cost, for the next frame's ranges
            if (knots != nullptr) {
                cost_sum += (
                    block_cost + kCostBlock +
                    kCostClipped * static_cast<uint32_t>(__builtin_popcount(crossing))
                );
                ++block;
                advance();
            }
            block_cost = 0;
        }
    };

}  // namespace
#endif

#ifndef RASTER_PASS_HELPERS  // a class takes the helpers above alone
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
    const uint32_t blocks_word = command.a2;
    const uint32_t profile_address = command.a9;
    // with kRanges the camera's page brings the core's range (setup_reader.cpp)
    const bool ranged = (blocks_word & kRanges) != 0;
    const uint32_t blocks = blocks_word & ~kRanges;
    uint32_t first_block = compute.me() * blocks / cores;
    uint32_t pages = 1 + ((compute.me() + 1) * blocks / cores - first_block);
#ifdef TRISC_UNPACK
    // Hands each page to the pack thread; frees it when the pack thread is done.
    for (uint32_t page = 0; page < pages; ++page) {
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
#endif
#ifdef TRISC_PACK
    const uint32_t started = km::wall();
    uint32_t mail = 0, vector_cycles = 0, scalar_cycles = 0;
    setup_pass::setup();
    Pass pass{compute, handler};
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
                if (handler == kHandlerCount) {
                    pass.knots = work + kWorkCostsAt / 4;
                    pass.blocks = pages - 1;
                    pass.advance();
                }
            }
            // the camera: each number to every lane of its slot
            for (uint32_t i = 0; i < kCamWords; ++i) {
                setup_pass::fill_slot(setup_pass::kSlotCamera + i, mesh_words[i]);
            }
            setup_pass::order(
                setup_pass::word(setup_pass::kSlotCamera + kCamWords - 1) + 62
            );
            // each lane's number as a float, for finish_lanes
            const uint32_t lane_numbers = setup_pass::word(setup_pass::kSlotLaneNumber);
            for (uint32_t lane = 0; lane < 32; ++lane) {
                setup_pass::window()[lane_numbers + 2 * lane] = float_bits(lane);
            }
            setup_pass::order(lane_numbers + 62);
            pass.weighted = handler == kHandlerSetup && mesh_words[kCamWords + 2] != 0;
            ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
            continue;
        }
        const uint32_t vector_started = km::wall();
        setup_pass::block_in(mesh_words);
        setup_pass::camera_space();
        setup_pass::triangles(handler != kHandlerCount, pass.weighted);
        const uint32_t first_triangle = 32 * (first_block + page - 1);
        if (handler != kHandlerLanes) {
            setup_pass::finish_lanes(
                handler == kHandlerSetup, float_bits(first_triangle)
            );
        }
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
        const uint32_t scalar_started = km::wall();
        vector_cycles += scalar_started - vector_started;
        pass.block_finished(mesh_words, work, first_triangle);
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
    if (pass.knots != nullptr) {
        // the blocks summed: the thread uses the costs only when they are the range's
        pass.knots[0] = pass.block;
        asm volatile("fence" ::: "memory");
    }
    if ((handler == kHandlerSetup || handler == kHandlerCount) && pass.succeeded) {
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
#endif
