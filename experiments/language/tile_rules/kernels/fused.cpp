// Tile-rules bodies with products inside (BODY 28, 29, 30), timed on the math CPU;
// every unit's tiles are in L1 before the first: language/tile_rules/README.md.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "rules.hpp"
#if BODY == 28
#include "mm_silu.hpp"
#define FUSED_INIT km_tile::mm_silu_init
#define FUSED_BODY(rows) km_tile::mm_silu(0)
#elif BODY == 29
#include "mm_gate.hpp"
#define FUSED_INIT km_tile::mm_gate_init
#define FUSED_BODY(rows) km_tile::mm_gate(0)
#else
#include "mm_gate_band.hpp"
#define FUSED_INIT km_tile::mm_gate_band_init
#define FUSED_BODY(rows) km_tile::mm_gate_band(0, rows)
#endif

void kernel_main() {
    const uint32_t nonce = get_arg_val<uint32_t>(0);
    const uint32_t units = get_arg_val<uint32_t>(1);
    [[maybe_unused]] const uint32_t rows = get_arg_val<uint32_t>(2);
    constexpr bool kThird = BODY != 28;  // W3 in kCbP
    volatile uint32_t* info = (
        reinterpret_cast<volatile uint32_t*>(get_tile_address(rules::kCbInfo, 0))
    );
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        rules::kCbA, rules::kCbB, rules::kCbOut
    );
    mm_no_mop_init_short(rules::kCbA, rules::kCbB, false, 1, 1, 1);
    MATH((FUSED_INIT()));
    // The empty span: a wall-clock read pair around one tensix_sync, on the math CPU.
    for (uint32_t i = 0; i < rules::kSyncs; ++i) {
        MATH(
            const uint32_t start = rules::wall(); tensix_sync();
            info[rules::kSync + i] = rules::wall() - start
        );
    }
    const uint32_t count = units < rules::kMaxUnits ? units : rules::kMaxUnits;
    cb_wait_front(rules::kCbA, count);
    cb_wait_front(rules::kCbB, count);
    if constexpr (kThird) {
        cb_wait_front(rules::kCbP, count);
    }
    for (uint32_t u = 0; u < count; ++u) {
        const uint32_t started = rules::wall();
        tile_regs_acquire();
        // No sync before the products: the unpacker is ahead holding both Src banks.
        MATH(const uint32_t start = rules::wall());
        // a into SrcB, w1 into SrcA: the product of Dst tile 0 (then w3: tile 1)
        UNPACK((llk_unpack_AB_matmul(rules::kCbA, rules::kCbB, u, u, 1, 1, 1)));
        if constexpr (kThird) {
            UNPACK((llk_unpack_AB_matmul(rules::kCbA, rules::kCbP, u, u, 1, 1, 1)));
        }
        MATH((FUSED_BODY(rows)));
        MATH(tensix_sync(); info[rules::kBody + u] = rules::wall() - start);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(rules::kCbOut, 1);
        pack_tile(0, rules::kCbOut);
        tile_regs_release();
        cb_push_back(rules::kCbOut, 1);
        PACK(info[rules::kLoop + u] = rules::wall() - started);
    }
    cb_pop_front(rules::kCbA, count);
    cb_pop_front(rules::kCbB, count);
    if constexpr (kThird) {
        cb_pop_front(rules::kCbP, count);
    }
    MATH(asm volatile("fence" :: : "memory"); info[rules::kMathDone] = nonce);
#ifdef TRISC_PACK
    uint32_t error = 0, polls = 0;
    while (true) {
        asm volatile("fence" ::: "memory");
        if (info[rules::kMathDone] == nonce) {
            break;
        }
        if (++polls >= rules::kSpinLimit) {
            error = rules::kErrorSpin;
            break;
        }
    }
    info[rules::kUnits] = units;
    info[rules::kError] = error;
    asm volatile("fence" ::: "memory");
    info[rules::kNonce] = nonce;
    asm volatile("fence" ::: "memory");
    cb_reserve_back(rules::kCbInfo, 1);
    cb_push_back(rules::kCbInfo, 1);
#endif
}
