// Lanes probe compute: the unpack thread hands each input page to the pack thread,
// which fills the tile: row records as the rasterizer does, then the lane pages.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "komorebi/device/km_worker.hpp"
#include "lanes.h"
#include "raster.h"
#ifdef TRISC_PACK
#include "fill.hpp"
#endif

namespace {

    // A give-up into the writer's gave-up word (the first one is kept), and the stop.
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

}  // namespace

#ifdef TRISC_PACK
namespace {

    using namespace ckernel;
    using fill::L0;
    using fill::L1;
    using fill::L2;
    using fill::L3;
    using fill::L4;
    using fill::L5;
    using fill::NZ;
    using fill::ONE;
    using fill::PY;
    using fill::ZERO;

    constexpr uint32_t kSetVd = 8;  // SFPGT: -1 or 0 into VD
    // Upper halves of float32 0, 1, 2, 3 (offsets within a box).
    constexpr uint32_t kSmallFloat[kMaxSide] = {0x0000, 0x3F80, 0x4000, 0x4040};

    // Input pages from the unpack thread (nullptr: the wait gave up).
    struct Input {
        const volatile uint32_t* page = nullptr;
        uint32_t wait = 0;

        const volatile uint32_t* next() {
            if (page != nullptr) {
                ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
            }
            const uint32_t asked = fill::wall();
            if (!mail_from(ckernel::ThreadId::UnpackThreadId, kWhereWaitPage)) {
                page = nullptr;
                return nullptr;
            }
            page = (
                reinterpret_cast<const volatile uint32_t*>(
                    ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)
                )
            );
            asm volatile("fence" ::: "memory");
            wait += fill::wall() - asked;
            return page;
        }

        // The last page goes back with "no more".
        void end() {
            ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 0);
        }
    };

    void load_lane(uint32_t lreg, uint32_t word) {
        fill::load_vec(lreg, kVecLaneIn + word);
    }

    // value = ((A px + -0) + (B py + -0)) + C of the lane words from `word` into L0, as
    // the fill's planes; `positive_zero`: + 0 after A px, as the fill's edge 0.
    void lane_plane(uint32_t word, bool positive_zero) {
        load_lane(L0, word);
        TTI_SFPMAD(L0, L4, NZ, L0, 0);
        if (positive_zero) {
            TTI_SFPMAD(L0, ONE, ZERO, L0, 0);
        }
        load_lane(L1, word + 1);
        TTI_SFPMAD(L1, PY, NZ, L1, 0);
        TTI_SFPMAD(L0, ONE, L1, L0, 0);
        load_lane(L1, word + 2);
        TTI_SFPMAD(L0, ONE, L1, L0, 0);
    }

    // Offset `offset` = (column, row) of the box in every lane: L5 the inside mask,
    // then w (0 outside) and the three colours into the offset's vectors.
    void lane_offset(uint32_t offset, uint32_t column, uint32_t row) {
        load_lane(L4, kLanePx);
        TT_SFPLOADI(L0, fill::kFloatB, kSmallFloat[column & 3]);
        TTI_SFPMAD(L4, ONE, L0, L4, 0);
        load_lane(PY, kLanePy);
        TT_SFPLOADI(L0, fill::kFloatB, kSmallFloat[row & 3]);
        TTI_SFPMAD(PY, ONE, L0, PY, 0);
        load_lane(L5, kLanePxEnd);
        TTI_SFPGT(0, L4, L5, kSetVd);  // px < px end
        load_lane(L1, kLanePyEnd);
        TTI_SFPGT(0, PY, L1, kSetVd);
        TTI_SFPAND(0, L1, L5, 0);
        for (uint32_t edge = 0; edge < 3; ++edge) {
            // a, b, c of edge k are words k, 3 + k, 6 + k: three loads of stride 3.
            load_lane(L0, kRecA + edge);
            TTI_SFPMAD(L0, L4, NZ, L0, 0);
            if (edge == 0) {
                TTI_SFPMAD(L0, ONE, ZERO, L0, 0);
            }
            load_lane(L1, kRecB + edge);
            TTI_SFPMAD(L1, PY, NZ, L1, 0);
            TTI_SFPMAD(L0, ONE, L1, L0, 0);
            load_lane(L1, kRecC + edge);
            TTI_SFPMAD(L0, ONE, L1, L0, 0);
            load_lane(L1, kRecT + edge);
            TTI_SFPGT(0, L1, L0, kSetVd);  // e > T
            TTI_SFPAND(0, L0, L5, 0);
        }
        offset &= kMaxOffsets - 1;
        TT_SFPLOADI(L2, fill::kUShort, 1u << offset);
        TTI_SFPAND(0, L5, L2, 0);
        TTI_SFPOR(0, L2, L3, 0);  // L3: the offsets each lane is inside at
        lane_plane(kRecW, false);
        TTI_SFPAND(0, L5, L0, 0);
        fill::store_vec(L0, kVecLaneW + offset);
        for (uint32_t colour = 0; colour < 3; ++colour) {
            lane_plane(kRecCol + 3 * colour, false);
            fill::store_vec(L0, kVecLaneColour + 3 * offset + colour);
        }
    }

    struct Counters {
        uint32_t values[kProfWords] = {};
    };

    // Window word of vector `vector`, lane `lane`.
    inline uint32_t window_word(uint32_t vector, uint32_t lane) {
        return 64 * (vector >> 1) + 2 * lane + (vector & 1);
    }

    // One lane page into Dst, its offsets on the vector unit, then each inside pixel
    // against the planes: larger w, or equal w and lower number, wins (order-free).
    bool lane_page(const volatile uint32_t* page, Counters& counters) {
        uint32_t* values = counters.values;
        const uint32_t shape = page[kLaneShape];
        const uint32_t columns = shape & 0xF, rows = (shape >> 4) & 0xF;
        if (columns < 1 || columns > kMaxSide || rows < 1 || rows > kMaxSide) {
            give_up(kGaveBadItem, kWhereBadLanes);
            return false;
        }
        const uint32_t offsets = columns * rows;
        volatile uint32_t* window = fill::win();
        uint32_t before = fill::wall();
        ckernel::tensix_sync();
        constexpr uint32_t kInWord = 64 * (kVecLaneIn >> 1);
        constexpr uint32_t kInWords = 64 * (kLaneWords >> 1);
        fill::copy_words_rvv(
            const_cast<uint32_t*>(window + kInWord),
            const_cast<const uint32_t*>(page),
            kInWords
        );
        fill::order(kInWord + kInWords - 1);
        uint32_t after = fill::wall();
        values[kProfCopyIn] += after - before;
        uint32_t column_of[kMaxOffsets], row_of[kMaxOffsets];
        TTI_SFPLOADI(L3, fill::kUShort, 0);
        for (uint32_t row = 0, offset = 0; row < rows; ++row) {
            for (uint32_t column = 0; column < columns; ++column, ++offset) {
                column_of[offset] = column;
                row_of[offset] = row;
                lane_offset(offset, column, row);
            }
        }
        fill::store_vec(L3, kVecLaneInside);
        ckernel::tensix_sync();
        before = fill::wall();
        values[kProfVector] += before - after;
        const volatile uint32_t* corner = page + window_word(kLaneCorner, 0);
        const volatile uint32_t* triangle = page + window_word(kLaneTri, 0);
        const uint32_t inside_limit = (1u << offsets) - 1;
        for (uint32_t lane = 0; lane < 32; ++lane) {
            uint32_t inside = window[window_word(kVecLaneInside, lane)] & inside_limit;
            if (inside == 0) {
                continue;
            }
            const uint32_t first = corner[2 * lane], number = triangle[2 * lane];
            while (inside != 0) {
                const uint32_t at = __builtin_ctz(inside);
                inside &= inside - 1;
                const uint32_t w = window[window_word(kVecLaneW + at, lane)];
                const uint32_t column = ((first & 0x1F) + column_of[at]) & 0x1F;
                const uint32_t row = (((first >> 8) & 0x1F) + row_of[at]) & 0x1F;
                const uint32_t pixel = window_word(row, column);
                const uint32_t depth = window[pixel];
                bool wins = static_cast<int32_t>(w) > static_cast<int32_t>(depth);
                if (!wins && w == depth) {
                    const uint32_t winner = window[pixel + 64 * 64];
                    wins = static_cast<int32_t>(number) < static_cast<int32_t>(winner);
                }
                values[kProfFragments]++;
                if (wins) {
                    const uint32_t vector = kVecLaneColour + 3 * at;
                    const uint32_t red = window[window_word(vector, lane)];
                    const uint32_t green = window[window_word(vector + 1, lane)];
                    const uint32_t blue = window[window_word(vector + 2, lane)];
                    window[pixel] = w;
                    window[pixel + 64 * 64] = number;
                    window[pixel + 1024] = red;
                    window[pixel + 2048] = green;
                    window[pixel + 3072] = blue;
                    values[kProfWins]++;
                }
            }
        }
        values[kProfResolve] += fill::wall() - before;
        values[kProfLanePages]++;
        values[kProfOffsets] += offsets;
        return true;
    }

    // The access costs of the pack thread: 64 of each kind, timed.
    void access_times(const volatile uint32_t* head, uint32_t* values) {
        volatile uint32_t* window = fill::win();
        ckernel::tensix_sync();
        uint32_t chained = 0, sum = 0;
        uint32_t before = fill::wall();
        for (uint32_t i = 0; i < 64; ++i) {
            chained = window[2 * i + (chained >> 31)];
        }
        uint32_t after = fill::wall();
        values[kProfWindowChain] = after - before;
        for (uint32_t i = 0; i < 64; i += 8) {
            uint32_t loaded[8];
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                loaded[j] = window[2 * (i + j)];
            }
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                sum += loaded[j];
            }
        }
        before = fill::wall();
        values[kProfWindowLoads] = before - after;
        for (uint32_t i = 0; i < 64; ++i) {
            window[64 * 115 + i] = i;
        }
        fill::order(64 * 115 + 63);
        after = fill::wall();
        values[kProfWindowStores] = after - before;
        for (uint32_t i = 0; i < 64; ++i) {
            chained = head[2 * i + (chained >> 31)];
        }
        values[kProfL1Chain] = fill::wall() - after;
        asm volatile("and x0, x0, %0" : : "r"(chained + sum) : "memory");
    }

    // LANES_BROADCAST: a row record's numbers reach every lane through Dst (eight
    // copies written by the RISC-V, SFPLOAD, SFPCONFIG) instead of immediates.
#ifndef LANES_BROADCAST
#define LANES_BROADCAST 0
#endif
    constexpr bool kBroadcast = LANES_BROADCAST != 0;
    constexpr uint32_t kBroadcastReg = 11;  // SFPCONFIG writes only LRegs 11 to 14
    // Record words a, b, c, w and colours (0 to 8, 12 to 23); the number by immediates.
    constexpr uint32_t kBroadcastWords[21] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    };
    // Two sets in turn (rewritten two triangles on, past the 32-instruction queue):
    // numbers 0 to 19 a pair a group of four Dst rows, number 20 in a shared group.
    constexpr uint32_t kBroadcastGroups[2] = {107, 118};
    constexpr uint32_t kBroadcastShared = 117;

    inline uint32_t broadcast_vector(uint32_t set, uint32_t number) {
        return (
            number < 20
                ? 2 * (kBroadcastGroups[set] + number / 2) + (number & 1)
                : 2 * kBroadcastShared + set
        );
    }

    // The 21 numbers of `record` into set `set`: eight copies, Dst lanes 0 to 7.
    void broadcast_write(const uint32_t* record, uint32_t set) {
        volatile uint32_t* window = fill::win();
        const size_t length = __riscv_vsetvl_e32m4(16);
        const vbool8_t odd = (
            __riscv_vmsne_vx_u32m4_b8(
                __riscv_vand_vx_u32m4(__riscv_vid_v_u32m4(length), 1, length), 0, length
            )
        );
        for (uint32_t pair = 0; pair < 10; ++pair) {
            vuint32m4_t copies = (
                __riscv_vmv_v_x_u32m4(record[kBroadcastWords[2 * pair]], length)
            );
            copies = (
                __riscv_vmerge_vxm_u32m4(
                    copies, record[kBroadcastWords[2 * pair + 1]], odd, length
                )
            );
            __riscv_vse32_v_u32m4(
                const_cast<uint32_t*>(window + 64 * (kBroadcastGroups[set] + pair)),
                copies,
                length
            );
        }
        const uint32_t last = record[kBroadcastWords[20]];
        for (uint32_t lane = 0; lane < 8; ++lane) {
            window[64 * kBroadcastShared + 2 * lane + set] = last;
        }
        asm volatile("" ::: "memory");
        fill::order(64 * kBroadcastShared + 14 + set);
    }

    // Number `number` of set `set` into every lane of LReg 11.
    inline void broadcast(uint32_t set, uint32_t number) {
        TT_SFPLOAD(
            L0,
            fill::kInt32,
            fill::kAddressMode,
            fill::dst_address(broadcast_vector(set & 1, number)) & 0x3FF
        );
        TTI_SFPCONFIG(0, kBroadcastReg, 0);
    }

    // fill::load_triangle with the numbers of set `set` (written) in place of
    // immediates; the same operations in the same order.
    void load_triangle_broadcast(const uint32_t* record, uint32_t set) {
        using fill::constant_address;
        using fill::kAddressMode;
        using fill::kInt32;
        using fill::push;
        TTI_SFPLOAD(L5, kInt32, kAddressMode, constant_address(fill::kConstPx));
        for (uint32_t edge = 0; edge < 3; ++edge) {
            broadcast(set, edge);
            TTI_SFPMAD(kBroadcastReg, L5, NZ, L1, 0);
            if (edge == 0) {
                push(TT_OP_SFPLOADI(L2, fill::kFloatA, record[kRecX0] & 0xFFFF));
                push(TT_OP_SFPLOADI(L4, fill::kFloatA, record[kRecX1] & 0xFFFF));
                TTI_SFPLOADI(L3, fill::kFloatB, 0xFF80);
                TTI_SFPGT(0, L2, L5, fill::kSetCC);
                TTI_SFPGT(0, L5, L4, fill::kSetCC);
                TTI_SFPLOADI(L3, fill::kFloatB, 0);
                TTI_SFPENCC(3, 0, 0, 10);
                TTI_SFPMAD(L1, ONE, L3, L1, 0);
            }
            TT_SFPSTORE(
                L1, kInt32, kAddressMode, constant_address(fill::kConstA + edge)
            );
            broadcast(set, 3 + edge);
            TT_SFPSTORE(
                kBroadcastReg,
                kInt32,
                kAddressMode,
                constant_address(fill::kConstB + edge)
            );
            broadcast(set, 6 + edge);
            TT_SFPSTORE(
                kBroadcastReg,
                kInt32,
                kAddressMode,
                constant_address(fill::kConstC + edge)
            );
        }
        broadcast(set, 9);
        TTI_SFPMAD(kBroadcastReg, L5, NZ, L1, 0);
        TTI_SFPSTORE(L1, kInt32, kAddressMode, constant_address(fill::kConstW));
        broadcast(set, 10);
        TTI_SFPSTORE(
            kBroadcastReg, kInt32, kAddressMode, constant_address(fill::kConstW + 1)
        );
        broadcast(set, 11);
        TTI_SFPSTORE(
            kBroadcastReg, kInt32, kAddressMode, constant_address(fill::kConstW + 2)
        );
        for (uint32_t colour = 0; colour < 3; ++colour) {
            broadcast(set, 12 + 3 * colour);
            TTI_SFPMAD(kBroadcastReg, L5, NZ, L1, 0);
            TT_SFPSTORE(
                L1, kInt32, kAddressMode, constant_address(fill::kConstColourA + colour)
            );
            broadcast(set, 13 + 3 * colour);
            TT_SFPSTORE(
                kBroadcastReg,
                kInt32,
                kAddressMode,
                constant_address(fill::kConstColourB + colour)
            );
            broadcast(set, 14 + 3 * colour);
            TT_SFPSTORE(
                kBroadcastReg,
                kInt32,
                kAddressMode,
                constant_address(fill::kConstColourC + colour)
            );
        }
        for (uint32_t edge = 0; edge < 3; ++edge) {
            push(TT_OP_SFPMOV(
                0, record[kRecT + edge] != 0 ? fill::TNEG : ZERO, L2 + edge, 0
            ));
        }
        fill::immediate(L5, record[kRecTri]);
    }

    // The broadcast's bits: 64 patterns (special values, then a sequence) each through
    // Dst lanes 0 to 7, SFPLOAD and SFPCONFIG to every lane; lanes that differ.
    uint32_t broadcast_bits() {
        constexpr uint32_t kSpecial[16] = {
            0x00000000,
            0x80000000,
            0x80000001,
            0x00000001,
            0x007FFFFF,
            0x807FFFFF,
            0x00800000,
            0x3F800000,
            0xBF800000,
            0x7F7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0x7F800001,
            0xFFFFFFFF,
            0x4B400000
        };
        volatile uint32_t* window = fill::win();
        constexpr uint32_t kOutVector = 250;
        uint32_t wrong = 0, pattern = 0x12345679;
        for (uint32_t index = 0; index < 64; ++index) {
            pattern = pattern * 1664525 + 1013904223;
            const uint32_t value = index < 16 ? kSpecial[index] : pattern;
            for (uint32_t lane = 0; lane < 8; ++lane) {
                window[64 * kBroadcastGroups[0] + 2 * lane] = value;
            }
            fill::order(64 * kBroadcastGroups[0] + 14);
            broadcast(0, 0);
            TTI_SFPSTORE(
                kBroadcastReg,
                fill::kInt32,
                fill::kAddressMode,
                fill::dst_address(kOutVector)
            );
            ckernel::tensix_sync();
            for (uint32_t lane = 0; lane < 32; ++lane) {
                wrong += window[window_word(kOutVector, lane)] != value;
            }
        }
        return wrong;
    }

    // Row record `record` (a local copy) over its rows of tile row `tile_y`.
    void row_record(const uint32_t* record, uint32_t tile_y, Counters& counters) {
        int32_t first_row = static_cast<int32_t>(record[kRecY0]) - 32 * tile_y;
        int32_t last_row = static_cast<int32_t>(record[kRecY1]) - 32 * tile_y;
        first_row = first_row < 0 ? 0 : first_row;
        last_row = last_row > 31 ? 31 : last_row;
        const uint32_t before = fill::wall();
        if constexpr (kBroadcast) {
            const uint32_t set = counters.values[kProfRowRecords] & 1;
            broadcast_write(record, set);
            counters.values[kProfBroadcastWrite] += fill::wall() - before;
            load_triangle_broadcast(record, set);
        } else {
            fill::load_triangle(record);
        }
        counters.values[kProfSetup] += fill::wall() - before;
        for (int32_t pixel_row = first_row; pixel_row <= last_row; ++pixel_row) {
            fill::row(pixel_row);
            counters.values[kProfRowCount]++;
        }
        counters.values[kProfRowRecords]++;
    }

    void tile_command() {
        constexpr uint32_t background[3] = {0x3D4CCCCD, 0x3D4CCCCD, 0x3DA3D70A};
        Counters counters;
        uint32_t* values = counters.values;
        const uint32_t started = fill::wall();
        fill::setup();
        fill::edges(true);
        Input input;
        const volatile uint32_t* head = input.next();
        if (head == nullptr) {
            return;
        }
        const uint32_t tile = head[kHeadTile], records = head[kHeadRowRecords];
        const uint32_t lane_pages = head[kHeadLanePages];
        if (tile >= kTileCount || records > kMaxRecords || lane_pages > kMaxRecords) {
            give_up(kGaveBadItem, kWhereBadLanes);
            input.end();
            return;
        }
        uint32_t before = fill::wall();
        fill::init_tile(tile % 32, tile / 32, background);
        uint32_t after = fill::wall();
        values[kProfInit] = after - before;
        access_times(head, values);
        if constexpr (kBroadcast) {
            values[kProfBitsWrong] = broadcast_bits();
        }
        after = fill::wall();
        const volatile uint32_t* page = nullptr;
        for (uint32_t index = 0; index < records; ++index) {
            if (index % 32 == 0) {
                page = input.next();
                if (page == nullptr) {
                    return;
                }
            }
            uint32_t record[kSlotWords];
            fill::copy_words_rvv(
                record,
                const_cast<const uint32_t*>(page + kSlotWords * (index % 32)),
                kSlotWords
            );
            row_record(record, tile / 32, counters);
        }
        values[kProfRows] = fill::wall() - after;
        for (uint32_t number = 0; number < lane_pages; ++number) {
            page = input.next();
            if (page == nullptr || !lane_page(page, counters)) {
                if (page != nullptr) {
                    input.end();
                }
                return;
            }
        }
        ckernel::tensix_sync();
        input.end();
        fill::finish();
        before = fill::wall();
        cb_reserve_back(kLanesCbOut, kLanesOutPages);
        const uint32_t out = get_local_cb_interface(kLanesCbOut).fifo_wr_ptr << 4;
        fill::copy_words_rvv(
            reinterpret_cast<uint32_t*>(out),
            const_cast<const uint32_t*>(fill::win()),
            kPlanes * kPlaneWords
        );
        values[kProfOut] = fill::wall() - before;
        values[kProfWait] = input.wait;
        values[kProfTotal] = fill::wall() - started;
        volatile uint32_t* profile = (
            reinterpret_cast<volatile uint32_t*>(out + kPlanes * kLanesPageBytes)
        );
        for (uint32_t word = 0; word < 32; ++word) {
            profile[word] = word < kProfWords ? values[word] : 0;
        }
        asm volatile("fence" ::: "memory");
        const uint32_t last_word = profile[31];
        asm volatile("and x0, x0, %0" : : "r"(last_word) : "memory");
        cb_push_back(kLanesCbOut, kLanesOutPages);
    }

}  // namespace
#endif

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kLanesCbIn, kLanesCbOut);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerTile) {
            return;
        }
#ifdef TRISC_UNPACK
        // Hands each page to the pack thread; frees it when the pack thread is done.
        for (;;) {
            const bool arrived = timed(
                [] {
                    const uint16_t received = (
                        reg_read(
                            reinterpret_cast<uint32_t>(
                                get_cb_tiles_received_ptr(kLanesCbIn)
                            )
                        )
                    );
                    return (
                        static_cast<uint16_t>(
                            received - get_local_cb_interface(kLanesCbIn).tiles_acked
                        ) >= 1
                    );
                },
                kWhereWaitList
            );
            if (!arrived) {
                break;
            }
            cb_wait_front(kLanesCbIn, 1);
            ckernel::mailbox_write(
                ckernel::ThreadId::PackThreadId, get_tile_l1_byte_address(kLanesCbIn, 0)
            );
            if (!mail_from(ckernel::ThreadId::PackThreadId, kWhereWaitDone)) {
                break;
            }
            const uint32_t more = (
                ckernel::mailbox_read(ckernel::ThreadId::PackThreadId)
            );
            cb_pop_front(kLanesCbIn, 1);
            if (more == 0) {
                break;
            }
        }
#endif
#ifdef TRISC_PACK
        tile_command();
#endif
    });
}
