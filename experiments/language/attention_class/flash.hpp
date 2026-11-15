// The attention project's compute (projects/attention/kernels/compute.cpp, the body of
// Kohaku-TT's ktt_flash compute) for the class's bodies: the same steps, the buffers
// the template's arguments (the bodies' names of the class's buffers, constants) but
// Q's and the output's, the sizes the class's defines, no profile words. finalize packs
// a row into the write stream's block the line reserved.
#pragma once

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "../../../projects/attention/kernels/sfpu.hpp"
#include "llk_math_eltwise_unary_sfpu_init.h"
#endif

namespace flash_class {

    constexpr uint32_t HEAD_TILES = FLASH_HEAD_TILES;
    constexpr uint32_t VALUE_TILES = FLASH_VALUE_TILES;
    constexpr uint32_t BLOCK_KEYS = FLASH_BLOCK_KEYS;
    constexpr uint32_t THRESHOLD_BITS = FLASH_THRESHOLD_BITS;
    // Accumulator columns per row: O, then the row sum l (column VALUE_TILES).
    constexpr uint32_t ACCUMULATOR_COLUMNS = VALUE_TILES + 1;
    constexpr uint32_t SUB_CHUNK = FLASH_SUB_CHUNK;  // the hand's kSubChunk
    constexpr uint32_t ANY_WORD = 257;  // fp32 state tile: row 0, column 17 (the flag)
    constexpr uint32_t FIRST_VALUE_PAGE = HEAD_TILES * BLOCK_KEYS;  // its first V page

    static_assert(BLOCK_KEYS % SUB_CHUNK == 0, "key block must be whole sub-chunks");

    // Q's buffer and the output's: the stream's halves, set by the bodies.
    inline uint32_t cb_queries = 0, cb_output = 0;

    struct Step {
        uint32_t row;          // query tile row
        uint32_t chunk;        // chunk index along the keys (0 = first)
        uint32_t chunk_start;  // first key tile of the chunk within the block
        uint32_t key_tiles;    // key tiles in the chunk
        bool last;             // the row's last chunk
        bool
            diagonal;  // the chunk's last tile is the row's diagonal tile (causal mask)
    };

    inline uint32_t slot(uint32_t row, uint32_t chunk) {
        return 2 * row + (chunk & 1);
    }

    // Math thread: broadcast-wise row maxima of the section.
    inline void state_vector(uint32_t key_tiles) {
#ifdef TRISC_MATH
        if (key_tiles == 3) {
            flash::state_max<3>();
        } else if (key_tiles == 2) {
            flash::state_max<2>();
        } else {
            flash::state_max<1>();
        }
#endif
    }

    // Pack thread: row state update and P of the section.
    inline void exp_vector(bool first, uint32_t key_tiles) {
#ifdef TRISC_PACK
        flash::exp_section_begin();
        if (first) {
            flash::state_update<true>(THRESHOLD_BITS);
        } else {
            flash::state_update<false>(THRESHOLD_BITS);
        }
        if (key_tiles == 3) {
            flash::exp_step<3>();
        } else if (key_tiles == 2) {
            flash::exp_step<2>();
        } else {
            flash::exp_step<1>();
        }
        flash::exp_section_end();
#endif
    }

    // Math thread: vector work after the section's matmuls, holding the SFPU mutex.
    inline void math_vector_begin() {
        MATH((TTI_STALLWAIT(p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::MATH)));
        MATH(
            (
                math::set_dst_write_addr<
                    DstTileShape::Tile32x32,
                    UnpackDestination::SrcRegs>(0)
            )
        );
        MATH((t6_mutex_acquire(mutex::SFPU)));
    }

    inline void math_vector_end() {
        MATH((TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::WAIT_SFPU)));
        MATH((t6_mutex_release(mutex::SFPU)));
    }

    // Pack thread: vector work waits for the math section and holds the SFPU mutex;
    // packs wait for the vector work.
    inline void vector_begin() {
        PACK((TTI_SEMWAIT(
            p_stall::STALL_SFPU | p_stall::STALL_SYNC,
            semaphore::t6_sem(semaphore::MATH_PACK),
            p_stall::STALL_ON_ZERO
        )));
        PACK(
            (
                math::set_dst_write_addr<
                    DstTileShape::Tile32x32,
                    UnpackDestination::SrcRegs>(0)
            )
        );
        PACK((t6_mutex_acquire(mutex::SFPU)));
    }

    inline void vector_end() {
        PACK((
            TTI_STALLWAIT(p_stall::STALL_SYNC | p_stall::STALL_PACK, p_stall::WAIT_SFPU)
        ));
        PACK((t6_mutex_release(mutex::SFPU)));
    }

    // Packer output format (fp32: cb_accumulator, cb_state; bf16: cb_probabilities,
    // cb_output) and L1 accumulation, changed only on transitions.
    inline bool pack_fp32 = false;
    inline bool pack_accumulating = false;

    template <uint32_t cb_probabilities, uint32_t cb_accumulator>
    inline void pack_format(bool fp32) {
        if (fp32 != pack_fp32) {
            pack_reconfig_data_format(
                fp32 ? cb_probabilities : cb_accumulator,
                fp32 ? cb_accumulator : cb_probabilities
            );
            pack_fp32 = fp32;
        }
    }

    inline void pack_accumulate(bool accumulate) {
        if (accumulate != pack_accumulating) {
            pack_reconfig_l1_acc(accumulate ? 1 : 0);
            pack_accumulating = accumulate;
        }
    }

    // The constant tiles, built by the vector unit and packed once: ones (cb_ones);
    // with FLASH_CAUSAL the diagonal mask (cb_causal).
    template <
        uint32_t cb_ones,
        uint32_t cb_causal,
        uint32_t cb_probabilities,
        uint32_t cb_accumulator>
    inline void constant_tiles() {
        tile_regs_acquire_math_clear();
        math_vector_begin();
        MATH((math::clear_dst_reg_addr()));
        MATH((flash::const_lanes()));
        MATH((flash::const_tile(flash::Const::Ones, 0, 0)));
#ifdef FLASH_CAUSAL
        MATH((flash::const_tile(flash::Const::Causal, 1, 0)));
#endif
        MATH((math::clear_dst_reg_addr()));
        math_vector_end();
        tile_regs_commit();
        tile_regs_wait();
        pack_format<cb_probabilities, cb_accumulator>(false);
        pack_accumulate(false);
        cb_reserve_back(cb_ones, 1);
        pack_tile(0, cb_ones);
        cb_push_back(cb_ones, 1);
#ifdef FLASH_CAUSAL
        cb_reserve_back(cb_causal, 1);
        pack_tile(1, cb_causal);
        cb_push_back(cb_causal, 1);
#endif
        tile_regs_release_math_clear();
        cb_wait_front(cb_ones, 1);
#ifdef FLASH_CAUSAL
        cb_wait_front(cb_causal, 1);
#endif
    }

    // The steps of the buffers K / V (any bf16 buffer where no block is held), the
    // ones, P, the row state, O and the row sums, the two tokens and the causal mask.
    template <
        uint32_t cb_keys_values,
        uint32_t cb_ones,
        uint32_t cb_probabilities,
        uint32_t cb_state,
        uint32_t cb_accumulator,
        uint32_t cb_state_token,
        uint32_t cb_accumulator_token,
        uint32_t cb_causal>
    struct Flash {
        static void pack_format(bool fp32) {
            flash_class::pack_format<cb_probabilities, cb_accumulator>(fp32);
        }

        // S = Q_r K^T (raw scores) and the row state into Dst; state update and P on
        // the pack thread.
        static void score(const Step& step) {
            const bool first = step.chunk == 0;
            tile_regs_acquire_math_clear();
            // First chunk: any slot (defines every state row).
            reconfig_data_format_srca(cb_keys_values, cb_state);
            copy_init(cb_state);
            copy_tile(
                cb_state, first ? slot(step.row, 1) : slot(step.row, step.chunk - 1), 3
            );
            reconfig_data_format_srca(cb_state, cb_keys_values);
            // The causal mask is copied into the diagonal score tile first; the score
            // matmul accumulates onto it.
#ifdef FLASH_CAUSAL
            if (step.diagonal) {
                copy_init(cb_causal);
                copy_tile(cb_causal, 0, step.key_tiles - 1);
            }
#endif
            mm_no_mop_init_short(
                cb_queries, cb_keys_values, true, step.key_tiles, 1, HEAD_TILES
            );
            for (uint32_t head_tile = 0; head_tile < HEAD_TILES; ++head_tile) {
                matmul_block_no_mop(
                    cb_queries,
                    cb_keys_values,
                    step.row * HEAD_TILES + head_tile,
                    head_tile * BLOCK_KEYS + step.chunk_start,
                    0,
                    true,
                    step.key_tiles,
                    1,
                    HEAD_TILES
                );
            }
            math_vector_begin();
            state_vector(step.key_tiles);
            math_vector_end();
            tile_regs_commit();

            cb_reserve_back(cb_probabilities, SUB_CHUNK);
            cb_reserve_back(cb_state_token, 1);
            tile_regs_wait();
            vector_begin();
            exp_vector(first, step.key_tiles);
            vector_end();
            pack_format(false);
            pack_accumulate(false);
            for (uint32_t tile = 0; tile < step.key_tiles; ++tile) {
                pack_tile(tile, cb_probabilities);
            }
            pack_format(true);
            pack_tile<true>(3, cb_state, slot(step.row, step.chunk));
            tile_regs_release_math_clear();
            cb_push_back(cb_probabilities, SUB_CHUNK);
            cb_push_back(cb_state_token, 1);
        }

        // Dst 0..: P V columns [j0, j0 + wv) of the block, then the row-sum column if
        // `sum`.
        static void pv_columns(
            const Step& step, uint32_t first_column, uint32_t value_columns, bool sum
        ) {
            if (value_columns > 0) {
                mm_no_mop_init_short(
                    cb_probabilities,
                    cb_keys_values,
                    false,
                    value_columns,
                    1,
                    step.key_tiles
                );
                for (uint32_t key = 0; key < step.key_tiles; ++key) {
                    matmul_block_no_mop(
                        cb_probabilities,
                        cb_keys_values,
                        key,
                        FIRST_VALUE_PAGE + (step.chunk_start + key) * VALUE_TILES +
                            first_column,
                        0,
                        false,
                        value_columns,
                        1,
                        step.key_tiles
                    );
                }
            }
            if (sum) {
                mm_no_mop_init_short(
                    cb_probabilities, cb_ones, false, 1, 1, step.key_tiles
                );
                for (uint32_t key = 0; key < step.key_tiles; ++key) {
                    matmul_block_no_mop(
                        cb_probabilities,
                        cb_ones,
                        key,
                        0,
                        value_columns,
                        false,
                        1,
                        1,
                        step.key_tiles
                    );
                }
            }
        }

        // O_r (and l_r) += P V (and P 1); scaled by alpha first when the tile's
        // reference moved.
        static void output(const Step& step) {
            const bool first = step.chunk == 0;
            cb_wait_front(cb_state_token, 1);
            cb_wait_front(cb_probabilities, SUB_CHUNK);
            const uint32_t moved = (
                first
                    ? 0
                    : read_tile_value(cb_state, slot(step.row, step.chunk), ANY_WORD)
            );
            if (moved == 0) {
                for (
                    uint32_t first_column = 0; first_column < ACCUMULATOR_COLUMNS;
                    first_column += 4
                ) {
                    const uint32_t columns = (
                        ACCUMULATOR_COLUMNS - first_column < 4
                            ? ACCUMULATOR_COLUMNS - first_column
                            : 4
                    );
                    const uint32_t value_columns = (
                        first_column < VALUE_TILES
                            ? (VALUE_TILES - first_column < columns
                                     ? VALUE_TILES - first_column
                                     : columns)
                            : 0
                    );
                    tile_regs_acquire_math_clear();
                    pv_columns(
                        step,
                        first_column,
                        value_columns,
                        first_column <= VALUE_TILES &&
                            VALUE_TILES < first_column + columns
                    );
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_format(true);
                    pack_accumulate(!first);
                    for (uint32_t tile = 0; tile < columns; ++tile) {
                        pack_tile<true>(
                            tile,
                            cb_accumulator,
                            step.row * ACCUMULATOR_COLUMNS + first_column + tile
                        );
                    }
                    tile_regs_release_math_clear();
                }
            } else {
                for (uint32_t column = 0; column < ACCUMULATOR_COLUMNS; ++column) {
                    tile_regs_acquire_math_clear();
                    pv_columns(
                        step,
                        column,
                        column < VALUE_TILES ? 1 : 0,
                        column == VALUE_TILES
                    );
                    reconfig_data_format_srca(cb_keys_values, cb_state);
                    copy_init(cb_state);
                    copy_tile(cb_state, slot(step.row, step.chunk), 2);
                    copy_tile(cb_state, slot(step.row, step.chunk - 1), 3);
                    copy_init(cb_accumulator);
                    copy_tile(
                        cb_accumulator, step.row * ACCUMULATOR_COLUMNS + column, 1
                    );
                    reconfig_data_format_srca(cb_accumulator, cb_keys_values);
                    tile_regs_commit();
                    tile_regs_wait();
                    vector_begin();
                    PACK((flash::rescale_step()));
                    vector_end();
                    pack_format(true);
                    pack_accumulate(false);
                    pack_tile<true>(
                        0, cb_accumulator, step.row * ACCUMULATOR_COLUMNS + column
                    );
                    tile_regs_release_math_clear();
                }
            }
            cb_pop_front(cb_probabilities, SUB_CHUNK);
            cb_pop_front(cb_state_token, 1);
            if (step.last) {
                cb_reserve_back(cb_accumulator_token, 1);
                cb_push_back(cb_accumulator_token, 1);
            }
        }

        // Output row r = O_r / l_r (bf16) into cb_output's block (the line's reserve
        // and push around it).
        static void finalize(uint32_t row) {
            cb_wait_front(cb_accumulator_token, 1);
            cb_pop_front(cb_accumulator_token, 1);
            for (
                uint32_t first_column = 0; first_column < VALUE_TILES; first_column += 3
            ) {
                const uint32_t columns = (
                    VALUE_TILES - first_column < 3 ? VALUE_TILES - first_column : 3
                );
                tile_regs_acquire_math_clear();
                reconfig_data_format_srca(cb_keys_values, cb_accumulator);
                copy_init(cb_accumulator);
                for (uint32_t tile = 0; tile < columns; ++tile) {
                    copy_tile(
                        cb_accumulator,
                        row * ACCUMULATOR_COLUMNS + first_column + tile,
                        tile
                    );
                }
                copy_tile(cb_accumulator, row * ACCUMULATOR_COLUMNS + VALUE_TILES, 3);
                reconfig_data_format_srca(cb_accumulator, cb_keys_values);
                tile_regs_commit();
                tile_regs_wait();
                vector_begin();
                PACK((flash::normalize_step(columns)));
                vector_end();
                pack_format(false);
                pack_accumulate(false);
                for (uint32_t tile = 0; tile < columns; ++tile) {
                    pack_tile(tile, cb_output);
                }
                tile_regs_release_math_clear();
            }
        }

        // One K / V block of a job (key tiles [block_start, ...) of its `key_tiles`):
        // its steps in (chunk, row) order, pairs of two different rows; `chunk_base`
        // the job's chunks before it.
        static void block(
            uint32_t rows,
            uint32_t first_row,
            uint32_t key_tiles,
            uint32_t stride,
            uint32_t block_start,
            uint32_t chunk_base
        ) {
            const uint32_t keys_in_block = (
                key_tiles - block_start < BLOCK_KEYS
                    ? key_tiles - block_start
                    : BLOCK_KEYS
            );
            const uint32_t sub_chunks = (keys_in_block + SUB_CHUNK - 1) / SUB_CHUNK;
            Step pending{};
            bool has_pending = false;
            for (uint32_t sub_chunk = 0; sub_chunk < sub_chunks; ++sub_chunk) {
                const uint32_t chunk_start = sub_chunk * SUB_CHUNK;
                const uint32_t chunk_first_key = block_start + chunk_start;
                const uint32_t chunk_keys = (
                    keys_in_block - chunk_start < SUB_CHUNK
                        ? keys_in_block - chunk_start
                        : SUB_CHUNK
                );
                for (uint32_t row = 0; row < rows; ++row) {
#ifdef FLASH_CAUSAL
                    const uint32_t global_row = first_row + row * stride;
                    const uint32_t last_key = (
                        global_row < key_tiles - 1 ? global_row : key_tiles - 1
                    );
#else
                    const uint32_t last_key = key_tiles - 1;
#endif
                    if (chunk_first_key > last_key) {
                        continue;
                    }
                    const uint32_t keys = (
                        last_key - chunk_first_key + 1 < chunk_keys
                            ? last_key - chunk_first_key + 1
                            : chunk_keys
                    );
                    const bool last = chunk_first_key + keys - 1 == last_key;
#ifdef FLASH_CAUSAL
                    const bool diagonal = last && last_key == global_row;
#else
                    const bool diagonal = false;
#endif
                    const Step step{
                        row, chunk_base + sub_chunk, chunk_start, keys, last, diagonal
                    };
                    if (has_pending && pending.row != step.row) {
                        score(pending);
                        score(step);
                        output(pending);
                        output(step);
                        has_pending = false;
                    } else {
                        if (has_pending) {
                            score(pending);
                            output(pending);
                        }
                        pending = step;
                        has_pending = true;
                    }
                }
            }
            if (has_pending) {
                score(pending);
                output(pending);
            }
        }
    };

}  // namespace flash_class
