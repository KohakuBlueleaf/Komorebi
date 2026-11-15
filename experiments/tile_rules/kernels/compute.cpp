// Tile-rules compute: per unit (a tile, or a row of UNIT_TILES) its tiles copied into
// Dst, one body (BODY: rules.py's table) timed on the math CPU, the unit packed.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "rules.hpp"
#if BODY == 1
#include "silu.hpp"
#elif BODY == 3
#include "add.hpp"
#elif BODY == 4
#include "add_macro.hpp"
#elif BODY == 6
#include "scale.hpp"
#elif BODY == 7
#include "scale_macro.hpp"
#elif BODY == 9
#include "gate.hpp"
#elif BODY == 11
#include "norm.hpp"
#endif

namespace {

    constexpr uint32_t kUnitTiles = UNIT_TILES;

#if BODY == 8 || BODY == 10
    constexpr auto kReuse = EltwiseBinaryReuseDestType::DEST_TO_SRCA;

    // Decode's product at HiFi4 (ops.hpp), Dst tile `dst` times `tile` of the buffer;
    // its state_configure does nothing here, so the source formats are set explicitly.
    inline void mul_hifi4_init(uint32_t input_buffer) {
        reconfig_data_format(input_buffer, input_buffer);
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, kReuse>(
            false, false, input_buffer
        )));
        MATH(
            (
                llk_math_eltwise_binary_init<
                    EltwiseBinaryType::ELWMUL,
                    BroadcastType::NONE,
                    MathFidelity::HiFi4,
                    kReuse>(input_buffer, input_buffer, false)
            )
        );
    }

    inline void mul_hifi4(uint32_t input_buffer, uint32_t tile, uint32_t dst) {
        UNPACK((llk_unpack_A<BroadcastType::NONE, true, kReuse>(input_buffer, tile)));
        MATH(
            (
                llk_math_eltwise_binary<
                    EltwiseBinaryType::ELWMUL,
                    BroadcastType::NONE,
                    DST_ACCUM_MODE,
                    MathFidelity::HiFi4,
                    kReuse>(input_buffer, input_buffer, dst, true)
            )
        );
    }
#endif

#if BODY == 10
    // Dst 0 into the one page of `buffer`, then waits for it (decode's pack_one).
    inline void pack_one(uint32_t buffer) {
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(buffer, 1);
        pack_tile(0, buffer);
        cb_push_back(buffer, 1);
        tile_regs_release();
        cb_wait_front(buffer, 1);
    }

    // The decode project's norm (its add's sums of squares, its scale's rms_scale and
    // product) for one row of tiles, then the product with the weight tile.
    inline void norm_unit() {
        cb_wait_front(rules::kCbY, kUnitTiles);
        reconfig_data_format(rules::kCbY, rules::kCbY);
        tile_regs_acquire();
        matmul_init(rules::kCbY, rules::kCbY, 1);
        for (uint32_t t = 0; t < kUnitTiles; ++t) {
            matmul_tiles(rules::kCbY, rules::kCbY, t, t, 0);
        }
        mul_hifi4_init(rules::kCbP);
        mul_hifi4(rules::kCbP, rules::kPIdentity, 0);
        pack_one(rules::kCbSquares);
        cb_pop_front(rules::kCbY, kUnitTiles);
        reconfig_data_format(rules::kCbP, rules::kCbSquares);
        tile_regs_acquire();
        matmul_init(rules::kCbSquares, rules::kCbP, 0);
        matmul_tiles(rules::kCbSquares, rules::kCbP, 0, rules::kPMean, 0);
        binop_with_scalar_tile_init();
        add_unary_tile(0, EPS_BITS);
        rsqrt_tile_init();
        rsqrt_tile(0);
        cb_pop_front(rules::kCbSquares, 1);
        pack_one(rules::kCbScale);
        for (uint32_t t = 0; t < kUnitTiles; ++t) {
            tile_regs_acquire();
            reconfig_data_format(rules::kCbA, rules::kCbA);
            copy_init(rules::kCbA);
            copy_tile(rules::kCbA, t, 0);
            mul_hifi4_init(rules::kCbScale);
            mul_hifi4(rules::kCbScale, 0, 0);
            mul_hifi4_init(rules::kCbP);
            mul_hifi4(rules::kCbP, rules::kPWeight + t, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(rules::kCbOut, 1);
            pack_tile(0, rules::kCbOut);
            cb_push_back(rules::kCbOut, 1);
            tile_regs_release();
        }
        cb_pop_front(rules::kCbScale, 1);
    }
#endif

    // Once per stream (bodies whose units re-initialise per unit do it in body()).
    inline void body_init() {
#if BODY == 0
        silu_tile_init();
#elif BODY == 1
        MATH((km_tile::silu_init()));
#elif BODY == 2
        add_binary_tile_init();
#elif BODY == 3
        MATH((km_tile::add_init()));
#elif BODY == 4
        MATH((km_tile::add_macro_init()));
#elif BODY == 5
        binop_with_scalar_tile_init();
#elif BODY == 6
        MATH((km_tile::scale_init()));
#elif BODY == 7
        MATH((km_tile::scale_macro_init()));
#elif BODY == 9
        MATH((km_tile::gate_init()));
#elif BODY == 11
        MATH((km_tile::norm_init()));
#endif
    }

    inline void body() {
#if BODY == 0
        silu_tile(0);
#elif BODY == 1
        MATH((km_tile::silu(0)));
#elif BODY == 2
        add_binary_tile(0, 1, 0);
#elif BODY == 3
        MATH((km_tile::add(0)));
#elif BODY == 4
        MATH((km_tile::add_macro(0)));
#elif BODY == 5
        mul_unary_tile(0, SCALE_BITS);
#elif BODY == 6
        MATH((km_tile::scale(0)));
#elif BODY == 7
        MATH((km_tile::scale_macro(0)));
#elif BODY == 8
        // The decode project's gate (its gate compute.cpp, less the norm's scale): silu
        // on Dst, then the product with b from its buffer on the matrix unit.
        silu_tile_init();
        silu_tile(0);
        mul_hifi4_init(rules::kCbB);
        mul_hifi4(rules::kCbB, 0, 0);
#elif BODY == 9
        MATH((km_tile::gate(0)));
#elif BODY == 11
        MATH((km_tile::norm(0)));
#endif
    }

}  // namespace

void kernel_main() {
    const uint32_t nonce = get_arg_val<uint32_t>(0);
    const uint32_t units = get_arg_val<uint32_t>(1);
    volatile uint32_t* info = (
        reinterpret_cast<volatile uint32_t*>(get_tile_address(rules::kCbInfo, 0))
    );
    compute_kernel_hw_startup(rules::kCbA, rules::kCbOut);
    copy_init(rules::kCbA);
#if HAS_P
    cb_wait_front(rules::kCbP, HAS_P);
#endif
    body_init();
    // The empty span: a wall-clock read pair around one tensix_sync, on the math CPU.
    for (uint32_t i = 0; i < rules::kSyncs; ++i) {
        MATH(
            const uint32_t start = rules::wall(); tensix_sync();
            info[rules::kSync + i] = rules::wall() - start
        );
    }
    for (uint32_t u = 0; u < units && u < rules::kMaxUnits; ++u) {
        const uint32_t started = rules::wall();
        cb_wait_front(rules::kCbA, kUnitTiles);
#if INPUTS == 2
        cb_wait_front(rules::kCbB, 1);
#endif
#if BODY == 10
        // The hand norm packs on its own (it goes through two one-page buffers).
        MATH(tensix_sync(); const uint32_t start = rules::wall());
        norm_unit();
        MATH(tensix_sync(); info[rules::kBody + u] = rules::wall() - start);
        cb_pop_front(rules::kCbA, kUnitTiles);
#else
        tile_regs_acquire();
#if BODY == 8
        reconfig_data_format(rules::kCbA, rules::kCbA);
        copy_init(rules::kCbA);
#endif
        for (uint32_t t = 0; t < kUnitTiles; ++t) {
            copy_tile(rules::kCbA, t, t);
        }
#if COPY_B
        copy_tile(rules::kCbB, 0, 1);
#endif
#if HAS_P
        copy_tile(rules::kCbP, 0, P_TILE);
#endif
        MATH(tensix_sync(); const uint32_t start = rules::wall());
        body();
        MATH(tensix_sync(); info[rules::kBody + u] = rules::wall() - start);
        tile_regs_commit();
        cb_pop_front(rules::kCbA, kUnitTiles);
#if INPUTS == 2
        cb_pop_front(rules::kCbB, 1);
#endif
        tile_regs_wait();
        cb_reserve_back(rules::kCbOut, kUnitTiles);
        for (uint32_t t = 0; t < kUnitTiles; ++t) {
            pack_tile(t, rules::kCbOut);
        }
        tile_regs_release();
        cb_push_back(rules::kCbOut, kUnitTiles);
#endif
        PACK(info[rules::kLoop + u] = rules::wall() - started);
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
