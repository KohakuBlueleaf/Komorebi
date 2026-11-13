// Matrix product compute: per owned row tile m and column block j0, the K chunks'
// products accumulate in DST (fp32) and the block is packed to the output buffer.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "dense.h"
#include "komorebi/device/km_worker.hpp"

namespace {

#if defined(TRISC_UNPACK)
    constexpr uint32_t kBody = kPrUnBody;
#elif defined(TRISC_MATH)
    constexpr uint32_t kBody = kPrMathBody;
#else
    constexpr uint32_t kBody = kPrPkBody;
#endif

    constexpr auto kReuse = EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    constexpr auto kMul = EltwiseBinaryType::ELWMUL;

    // eltwise_binary.h's mul_reuse_dest_init and _tiles at HiFi4 whatever the program's
    // fidelity: at HiFi2 an element-wise product loses its operands' low bits.
    inline void mul_hifi4_init(uint32_t input_buffer) {
        state_configure(input_buffer, __LINE__);
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, kReuse>(
            false, false, input_buffer
        )));
        MATH(
            (
                llk_math_eltwise_binary_init<
                    kMul,
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
                    kMul,
                    BroadcastType::NONE,
                    DST_ACCUM_MODE,
                    MathFidelity::HiFi4,
                    kReuse>(input_buffer, input_buffer, dst, true)
            )
        );
    }

    // LAYER: per owned row and column block, A's row times W's block in DST (the gate
    // pass: A's block copied), the epilogue on DST, the pack to Y's buffer.
    void layer(
        const km::Compute& compute,
        const km_command_t& command,
        volatile tt_l1_ptr uint32_t* profile
    ) {
        const uint32_t m_tiles = command.a2, k_tiles = command.a3, n_tiles = command.a4;
        const uint32_t block_width = command.a5, w_buffer = command.a6;
        const uint32_t epilogue = command.a7, held_buffers = command.a8;
        const km::Partition owned_rows = (
            km::partition(
                m_tiles, compute.me() - km::first(command), km::count(command)
            )
        );
        const uint32_t a_location = held_buffers & 0xFF;
        const uint32_t b_buffer = (held_buffers >> 8) & 0xFF;
        const uint32_t y_location = (held_buffers >> 16) & 0xFF;
        const uint32_t a_buffer = a_location == kDram ? kCbA : a_location;
        const uint32_t y_buffer = y_location == kDram ? kCbOut : y_location;
        const bool gate = w_buffer == 0;
        const bool silu = epilogue == kEpSilu || epilogue == kEpGate;
        const bool multiply = epilogue == kEpMul || gate;
        uint32_t wait_for_a = 0, matmul_cycles = 0, epilogue_cycles = 0;
        uint32_t wait_math = 0, wait_out = 0;
        uint32_t pack = 0, tiles = 0;
        bool ok = true, init = true;
        for (
            uint32_t row = owned_rows.start;
            ok && row < owned_rows.start + owned_rows.len;
            ++row
        ) {
            if (!gate) {
                const uint32_t asked = km::wall();
                ok = compute.wait_front(a_buffer, k_tiles);
                wait_for_a += km::wall() - asked;
            }
            for (uint32_t column = 0; ok && column < n_tiles; column += block_width) {
                const uint32_t block_started = km::wall();
                tile_regs_acquire();
                if (gate) {
                    ok = compute.wait_front(a_buffer, block_width);
                    if (ok) {
                        copy_init(a_buffer);
                        for (uint32_t tile = 0; tile < block_width; ++tile) {
                            copy_tile(a_buffer, tile, tile);
                        }
                    }
                } else {
                    if (init) {  // first block, or an epilogue changed the units' state
                        matmul_block_init(
                            a_buffer, w_buffer, 0, block_width, 1, k_tiles
                        );
                        init = epilogue != kEpNone;
                    }
                    for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                        matmul_block(
                            a_buffer,
                            w_buffer,
                            k_tile,
                            k_tile * n_tiles + column,
                            0,
                            0,
                            block_width,
                            1,
                            k_tiles
                        );
                    }
                }
                const uint32_t product_issued = km::wall();
                if (ok && silu) {
                    silu_tile_init();
                    for (uint32_t tile = 0; tile < block_width; ++tile) {
                        silu_tile(tile);
                    }
                }
                if (ok && multiply) {
                    ok = compute.wait_front(b_buffer, block_width);
                    if (ok) {
                        mul_hifi4_init(b_buffer);
                        for (uint32_t tile = 0; tile < block_width; ++tile) {
                            mul_hifi4(b_buffer, tile, tile);
                        }
                        cb_pop_front(b_buffer, block_width);
                    }
                }
                if (ok && gate) {
                    cb_pop_front(a_buffer, block_width);
                }
                tile_regs_commit();
                const uint32_t epilogue_issued = km::wall();
                tile_regs_wait();
                const uint32_t math_done = km::wall();
                const bool room = ok && compute.reserve_back(y_buffer, block_width);
                const uint32_t output_reserved = km::wall();
                if (room) {
                    for (uint32_t tile = 0; tile < block_width; ++tile) {
                        pack_tile(tile, y_buffer);
                    }
                }
                tile_regs_release();
                if (room) {
                    cb_push_back(y_buffer, block_width);
                    tiles += block_width;
                }
                matmul_cycles += product_issued - block_started;
                epilogue_cycles += epilogue_issued - product_issued;
                wait_math += math_done - epilogue_issued;
                wait_out += output_reserved - math_done;
                pack += km::wall() - output_reserved;
            }
            if (ok && !gate) {
                cb_pop_front(a_buffer, k_tiles);
            }
        }
        UNPACK((km::add_sat(profile + kPrUnWaitA, wait_for_a)));
        MATH((km::add_sat(profile + kPrMathMm, matmul_cycles)));
        MATH((km::add_sat(profile + kPrMathEp, epilogue_cycles)));
        PACK((km::add_sat(profile + kPrPkWaitMath, wait_math)));
        PACK((km::add_sat(profile + kPrPkWaitOut, wait_out)));
        PACK((km::add_sat(profile + kPrPkPack, pack)));
        PACK((km::add_sat(profile + kPrPkTiles, tiles)));
        (void)wait_for_a, (void)matmul_cycles, (void)epilogue_cycles, (void)wait_math,
            (void)wait_out, (void)pack;
        (void)tiles;
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    // Every operand is bf16; A's buffer stands for W's, which the streamed form lacks.
    compute_kernel_hw_startup<SrcOrder::Reverse>(kCbA, kCbA, kCbOut);
    volatile tt_l1_ptr uint32_t* profile = km::l1(get_tile_address(kCbProf, 0));
    uint32_t loaded[kCbW2 + 1] = {};  // W pages held in each W buffer
    compute.run([&](const km_command_t& command) {
        const uint32_t started = km::wall();
        const uint32_t k_tiles = command.a3, n_tiles = command.a4;
        if (km::handler(command) == kHandlerLoad) {
            const uint32_t w_buffer = command.a1;
            if (w_buffer != kCbW && w_buffer != kCbW3 && w_buffer != kCbW2) {
                return;
            }
            if (loaded[w_buffer] != 0) {
                cb_pop_front(w_buffer, loaded[w_buffer]);
                loaded[w_buffer] = 0;
            }
            if (compute.wait_front(w_buffer, k_tiles * n_tiles)) {
                loaded[w_buffer] = k_tiles * n_tiles;
            }
            return;
        }
        if (km::handler(command) == kHandlerLayer) {
            layer(compute, command, profile);
            km::add_sat(profile + kBody, km::wall() - started);
            return;
        }
        if (km::handler(command) != kHandlerProduct) {
            return;
        }
        const uint32_t m_tiles = command.a2, chunk_tiles = command.a5;
        const uint32_t block_width = command.a6, mode = command.a7, form = command.a8;
        const km::Partition owned_rows = (
            km::partition(
                m_tiles, compute.me() - km::first(command), km::count(command)
            )
        );
        const bool stream = form == kFormStream;
        const bool row_item = stream || chunk_tiles == k_tiles;
        const bool reads = mode != kModeComputeOnly;
        const bool math = mode == kModeFull || mode == kModeComputeOnly;
        const uint32_t w_buffer = stream ? kCbWs : kCbW;
        if (math) {
            matmul_block_init(kCbA, w_buffer, 0, block_width, 1, chunk_tiles);
        }
        uint32_t wait_for_a = 0, wait_for_w = 0, matmul_cycles = 0;
        uint32_t wait_math = 0, wait_out = 0;
        uint32_t pack = 0, tiles = 0;
        bool ok = true;
        for (
            uint32_t row = owned_rows.start;
            ok && row < owned_rows.start + owned_rows.len;
            ++row
        ) {
            if (row_item && reads) {
                const uint32_t asked = km::wall();
                ok = compute.wait_front(kCbA, k_tiles);
                wait_for_a += km::wall() - asked;
            }
            for (uint32_t column = 0; ok && column < n_tiles; column += block_width) {
                const uint32_t block_started = km::wall();
                if (math) {
                    tile_regs_acquire();
                }
                for (uint32_t chunk = 0; ok && chunk < k_tiles; chunk += chunk_tiles) {
                    if (!row_item && reads) {
                        const uint32_t asked = km::wall();
                        ok = compute.wait_front(kCbA, chunk_tiles);
                        wait_for_a += km::wall() - asked;
                    }
                    if (ok && stream && reads) {
                        const uint32_t asked = km::wall();
                        ok = compute.wait_front(kCbWs, chunk_tiles * block_width);
                        wait_for_w += km::wall() - asked;
                    }
                    if (ok && math) {
                        // A: the chunk's tiles (or the row's); W: chunk_tiles rows of
                        // n_tiles (resident) or block_width (a streamed chunk).
                        const uint32_t a_first = row_item ? chunk : 0;
                        for (uint32_t k_tile = 0; k_tile < chunk_tiles; ++k_tile) {
                            const uint32_t w_tile = (
                                stream
                                    ? k_tile * block_width
                                    : (chunk + k_tile) * n_tiles + column
                            );
                            matmul_block(
                                kCbA,
                                w_buffer,
                                a_first + k_tile,
                                w_tile,
                                0,
                                0,
                                block_width,
                                1,
                                chunk_tiles
                            );
                        }
                    }
                    if (ok && !row_item && reads) {
                        cb_pop_front(kCbA, chunk_tiles);
                    }
                    if (ok && stream && reads) {
                        cb_pop_front(kCbWs, chunk_tiles * block_width);
                    }
                }
                if (!math) {
                    continue;
                }
                tile_regs_commit();
                const uint32_t product_issued = km::wall();
                matmul_cycles += product_issued - block_started;
                tile_regs_wait();
                const uint32_t math_done = km::wall();
                const bool room = ok && compute.reserve_back(kCbOut, block_width);
                const uint32_t output_reserved = km::wall();
                if (room) {
                    for (uint32_t tile = 0; tile < block_width; ++tile) {
                        pack_tile(tile, kCbOut);
                    }
                }
                tile_regs_release();
                if (room) {
                    cb_push_back(kCbOut, block_width);
                    tiles += block_width;
                }
                wait_math += math_done - product_issued;
                wait_out += output_reserved - math_done;
                pack += km::wall() - output_reserved;
            }
            if (ok && row_item && reads) {
                cb_pop_front(kCbA, k_tiles);
            }
        }
        UNPACK((km::add_sat(profile + kPrUnWaitA, wait_for_a)));
        UNPACK((km::add_sat(profile + kPrUnWaitW, wait_for_w)));
        MATH((km::add_sat(profile + kPrMathMm, matmul_cycles)));
        PACK((km::add_sat(profile + kPrPkWaitMath, wait_math)));
        PACK((km::add_sat(profile + kPrPkWaitOut, wait_out)));
        PACK((km::add_sat(profile + kPrPkPack, pack)));
        PACK((km::add_sat(profile + kPrPkTiles, tiles)));
        km::add_sat(profile + kBody, km::wall() - started);
        (void)pack;
    });
}
