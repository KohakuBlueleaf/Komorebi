// Tile-cost compute: one section (MODE) repeated, each repeat started by a barrier of
// the three compute CPUs, stamped on each CPU, and read out of the tile's counters.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "cost.hpp"
#include "counters_table.hpp"
#if PB_VEC
#include "vec.hpp"
#endif

#if defined(TRISC_UNPACK)
constexpr uint32_t kMe = 0;
#elif defined(TRISC_MATH)
constexpr uint32_t kMe = 1;
#else
constexpr uint32_t kMe = 2;
#endif

namespace {

    // MODE 0: a cycle of acquire, copies, body, commit, wait, packs, release.
    // MODE 1: products by matmul_block. MODE 2: format changes.
    constexpr uint32_t kTiles = PB_TILES;
    constexpr uint32_t kGroup = PB_GROUP;
    constexpr uint32_t kCycles = kTiles / kGroup;
#if PB_VEC
    constexpr uint32_t kVecDst[] = {PB_VEC_DSTS};
#endif

    volatile uint32_t* info;

    inline void fence() {
        asm volatile("fence" ::: "memory");
    }

    // The L1 page is not cleared between runs: a tag is the run's nonce plus a count,
    // and only a word within 65,535 above the nonce counts as posted.
    uint32_t g_nonce = 0, g_failed = 0;

    // All three CPUs at one point: each posts the tag, then waits for the others'.
    inline void barrier(uint32_t tag) {
        fence();
        info[cost::kBar + kMe] = g_nonce + tag;
        fence();
        for (uint32_t spin = 0; spin < cost::kSpinLimit; ++spin) {
            fence();
            const uint32_t a = info[cost::kBar] - g_nonce;
            const uint32_t b = info[cost::kBar + 1] - g_nonce;
            const uint32_t c = info[cost::kBar + 2] - g_nonce;
            if (
                a >= tag && a < 0x10000 && b >= tag && b < 0x10000 && c >= tag &&
                c < 0x10000
            ) {
                return;
            }
        }
        g_failed = tag;
    }

    inline void section_init() {
#if PB_MODE == 0 || PB_MODE == 2
        copy_init(cost::kCbA);
#if PB_VEC
        MATH((km_tile::vec_init()));
#endif
#elif PB_MODE == 1
        matmul_block_init(cost::kCbA, cost::kCbB, 0, PB_CT, PB_RT, PB_KT);
#endif
    }

    inline void section() {
#if PB_MODE == 0
        PACK((cb_reserve_back(cost::kCbOut, kTiles)));
        for (uint32_t c = 0; c < kCycles; ++c) {
            tile_regs_acquire();
#if PB_COPY
#if PB_VEC
            for (uint32_t g = 0; g < sizeof(kVecDst) / sizeof(kVecDst[0]); ++g) {
                copy_tile(cost::kCbA, (c * kGroup + g) % kTiles, kVecDst[g]);
            }
#else
            for (uint32_t g = 0; g < kGroup; ++g) {
                copy_tile(cost::kCbA, c * kGroup + g, g);
            }
#endif
#endif
#if PB_VEC
            MATH((km_tile::vec(0)));
#endif
            tile_regs_commit();
            tile_regs_wait();
#if PB_PACK
            for (uint32_t g = 0; g < kGroup; ++g) {
                pack_tile<true>(g, cost::kCbOut, c * kGroup + g);
            }
#endif
            tile_regs_release();
        }
        PACK((cb_push_back(cost::kCbOut, kTiles)));
#elif PB_MODE == 1
        PACK((cb_reserve_back(cost::kCbOut, PB_RT * PB_CT * kCycles)));
        for (uint32_t c = 0; c < kCycles; ++c) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < PB_KT; ++k) {
                matmul_block(
                    cost::kCbA, cost::kCbB, k, k * PB_CT, 0, 0, PB_CT, PB_RT, PB_KT
                );
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < PB_RT * PB_CT; ++t) {
                pack_tile<true>(t, cost::kCbOut, c * PB_RT * PB_CT + t);
            }
            tile_regs_release();
        }
        PACK((cb_push_back(cost::kCbOut, PB_RT * PB_CT * kCycles)));
#elif PB_MODE == 2
        PACK((cb_reserve_back(cost::kCbOut, kTiles / 2)));
        PACK((cb_reserve_back(cost::kCbOutAlt, kTiles / 2)));
        for (uint32_t i = 0; i < kTiles; ++i) {
            const uint32_t src = (i % 2 == 0) ? cost::kCbA : cost::kCbAlt;
            const uint32_t dst = (i % 2 == 0) ? cost::kCbOut : cost::kCbOutAlt;
            tile_regs_acquire();
#if PB_RECONF_UNPACK
            reconfig_data_format_srca(src);
#endif
            copy_tile(src, i / 2, 0);
            tile_regs_commit();
            tile_regs_wait();
#if PB_RECONF_PACK
            pack_reconfig_data_format(dst);
#endif
            pack_tile<true>(0, dst, i / 2);
            tile_regs_release();
        }
        PACK((cb_push_back(cost::kCbOut, kTiles / 2)));
        PACK((cb_push_back(cost::kCbOutAlt, kTiles / 2)));
#elif PB_MODE == 4
        // The clock: the math CPU spins PB_TILES million wall-clock cycles, the host
        // reads the card's clock meanwhile and times the command.
#if defined(TRISC_MATH)
        const uint32_t spin_start = cost::wall();
        while (cost::wall() - spin_start < PB_TILES * 1000000u) {
            ;
        }
#endif
#endif
    }

    // After the barrier that ends a repeat: the unpack CPU takes the output pages back,
    // so the next repeat's reserve finds the room (a consumer in place of the writer).
    inline void section_tail() {
#if PB_MODE == 0
        UNPACK((cb_wait_front(cost::kCbOut, kTiles)));
        UNPACK((cb_pop_front(cost::kCbOut, kTiles)));
#elif PB_MODE == 1
        UNPACK((cb_wait_front(cost::kCbOut, PB_RT * PB_CT * kCycles)));
        UNPACK((cb_pop_front(cost::kCbOut, PB_RT * PB_CT * kCycles)));
#elif PB_MODE == 2
        UNPACK((cb_wait_front(cost::kCbOut, kTiles / 2)));
        UNPACK((cb_pop_front(cost::kCbOut, kTiles / 2)));
        UNPACK((cb_wait_front(cost::kCbOutAlt, kTiles / 2)));
        UNPACK((cb_pop_front(cost::kCbOutAlt, kTiles / 2)));
#endif
    }

    // The math CPU starts the four banks; the pack CPU stops them and reads the tables.
    inline void counters_start() {
#if defined(TRISC_MATH)
        for (uint32_t b = 0; b < 4; ++b) {
            cost::start_bank(cost::kBanks[b]);
        }
#endif
    }

    inline void counters_read(uint32_t base) {
#if defined(TRISC_PACK)
        for (uint32_t b = 0; b < 4; ++b) {
            cost::stop_bank(cost::kBanks[b]);
        }
        const uint32_t read_start = cost::wall();
        uint32_t word = base + cost::kCounters;
        for (uint32_t b = 0; b < 4; ++b) {
            const cost::Bank& bank = cost::kBanks[b];
            for (uint32_t i = 0; i < counters::kSizes[b]; ++i) {
                const uint32_t mode = (
                    static_cast<uint32_t>(counters::kSelects[b][i]) << 8
                );
                cost::put(bank.mode, mode);
                uint32_t spin = 0;
                while (cost::get(bank.mode) != mode && ++spin < 1024) {
                    ;
                }
                if (i == 0) {
                    info[base + cost::kRefs + b] = cost::get(bank.out_l);
                }
                info[word++] = spin < 1024 ? cost::get(bank.out_h) : 0xDEAD0000u;
            }
        }
        info[base + cost::kReadTime] = cost::wall() - read_start;
#endif
    }

}  // namespace

void kernel_main() {
    const uint32_t nonce = get_arg_val<uint32_t>(0);
    const uint32_t repeats = get_arg_val<uint32_t>(1);
    g_nonce = nonce;
    info = reinterpret_cast<volatile uint32_t*>(get_tile_address(cost::kCbInfo, 0));
#if PB_MODE == 1
    compute_kernel_hw_startup<SrcOrder::Reverse>(cost::kCbA, cost::kCbB, cost::kCbOut);
#else
    compute_kernel_hw_startup(cost::kCbA, cost::kCbOut);
#endif
    section_init();
#if PB_MODE == 0 || PB_MODE == 1
    UNPACK((cb_wait_front(cost::kCbA, kTiles)));
#endif
#if PB_MODE == 2
    UNPACK((cb_wait_front(cost::kCbA, kTiles / 2)));
#endif
#if PB_MODE == 1
    UNPACK((cb_wait_front(cost::kCbB, kTiles)));
#endif
#if PB_MODE == 2
    UNPACK((cb_wait_front(cost::kCbAlt, kTiles / 2)));
#endif
#if defined(TRISC_MATH)
    info[cost::kThrottle] = cost::get(cost::kArcScratch);
    info[cost::kDbgDisable] = cost::get(cost::kDbgFeatureDisable);
#endif
    // The empty span: a wall-clock pair around one tensix_sync, on each CPU.
    for (uint32_t i = 0; i < cost::kSyncs; ++i) {
        const uint32_t start = cost::wall();
        tensix_sync();
        const uint32_t span = cost::wall() - start;
        if (kMe == 1) {
            info[cost::kSync + i] = span;
        }
    }
    uint32_t tag = 0;
    for (uint32_t r = 0; r < repeats && r < cost::kMaxRepeats; ++r) {
        const uint32_t base = cost::kRep + r * cost::kRepStride;
        tensix_sync();
        barrier(++tag);
        counters_start();
        barrier(++tag);
        const uint32_t started = cost::wall();
        section();
        tensix_sync();
        const uint32_t ended = cost::wall();
        info[base + cost::kStamps + 2 * kMe] = started;
        info[base + cost::kStamps + 2 * kMe + 1] = ended;
        barrier(++tag);
        section_tail();
        counters_read(base);
        barrier(++tag);
    }
    info[cost::kFail + kMe] = g_failed;
#if defined(TRISC_PACK)
    info[cost::kRepeats] = repeats;
    info[cost::kTiles] = kTiles;
    fence();
    info[cost::kNonce] = nonce;
    fence();
    cb_reserve_back(cost::kCbInfo, 1);
    cb_push_back(cost::kCbInfo, 1);
#endif
}
