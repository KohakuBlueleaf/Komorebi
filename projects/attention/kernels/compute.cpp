// Attention compute, the body of Kohaku-TT's ktt_flash compute (Apache-2.0): per job,
// up to MAX_ROWS query tile rows against the job's key blocks from the broadcast.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "attention.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "sfpu.hpp"
#endif

namespace {

    constexpr uint32_t HEAD_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t VALUE_TILES = get_compile_time_arg_val(1);
    constexpr uint32_t MAX_ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK_KEYS = get_compile_time_arg_val(3);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(4);
    constexpr uint32_t THRESHOLD_BITS = get_compile_time_arg_val(5);
    // Accumulator columns per row: O, then the row sum l (column VALUE_TILES).
    constexpr uint32_t ACCUMULATOR_COLUMNS = VALUE_TILES + 1;
    constexpr uint32_t SUB_CHUNK = kSubChunk;
    constexpr uint32_t ANY_WORD = 257;  // fp32 state tile: row 0, column 17 (the flag)
    constexpr uint32_t BLOCK_PAGES = HEAD_TILES * BLOCK_KEYS +
        BLOCK_KEYS * VALUE_TILES;  // pages of a K / V block
    constexpr uint32_t FIRST_VALUE_PAGE = HEAD_TILES * BLOCK_KEYS;  // its first V page

    constexpr uint32_t cb_queries = kCbQ;
    constexpr uint32_t cb_keys_values = kCbKv;
    constexpr uint32_t cb_ones = kCbOnes;
    constexpr uint32_t cb_probabilities = kCbP;
    constexpr uint32_t cb_state = kCbSt;
    constexpr uint32_t cb_accumulator = kCbO;
    constexpr uint32_t cb_state_token = kCbStok;
    constexpr uint32_t cb_accumulator_token = kCbOtok;
    constexpr uint32_t cb_causal = kCbCausal, cb_output = kCbOut;

    // This thread's profile counters for the record (attention.h, kPr*): wall-clock
    // cycles at its own submit, added to the profile page at the command's end.
    uint32_t profile_counters[kProfWords];

    void flush(volatile tt_l1_ptr uint32_t* profile, uint32_t first, uint32_t count) {
        for (uint32_t word = first; word < first + count; ++word) {
            km::add_sat(profile + word, profile_counters[word]);
        }
    }

    static_assert(BLOCK_KEYS % SUB_CHUNK == 0, "key block must be whole sub-chunks");

    struct Step {
        uint32_t row;          // query tile row
        uint32_t chunk;        // chunk index along the keys (0 = first)
        uint32_t chunk_start;  // first key tile of the chunk within the block
        uint32_t key_tiles;    // key tiles in the chunk
        bool last;             // the row's last chunk
        bool
            diagonal;  // the chunk's last tile is the row's diagonal tile (causal mask)
    };

    uint32_t slot(uint32_t row, uint32_t chunk) {
        return 2 * row + (chunk & 1);
    }

    // Math thread: broadcast-wise row maxima of the section.
    void state_vector(uint32_t key_tiles) {
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
    void exp_vector(bool first, uint32_t key_tiles) {
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
    void math_vector_begin() {
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

    void math_vector_end() {
        MATH((TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::WAIT_SFPU)));
        MATH((t6_mutex_release(mutex::SFPU)));
    }

    // Packer output format (fp32: cb_accumulator, cb_state; bf16: cb_probabilities,
    // cb_output) and L1 accumulation, changed only on transitions.
    bool pack_fp32 = false;
    bool pack_accumulating = false;

    void pack_format(bool fp32) {
        if (fp32 != pack_fp32) {
            pack_reconfig_data_format(
                fp32 ? cb_probabilities : cb_accumulator,
                fp32 ? cb_accumulator : cb_probabilities
            );
            pack_fp32 = fp32;
        }
    }

    void pack_accumulate(bool accumulate) {
        if (accumulate != pack_accumulating) {
            pack_reconfig_l1_acc(accumulate ? 1 : 0);
            pack_accumulating = accumulate;
        }
    }

    // Pack thread: vector work waits for the math section and holds the SFPU mutex;
    // packs wait for the vector work.
    void vector_begin() {
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

    void vector_end() {
        PACK((
            TTI_STALLWAIT(p_stall::STALL_SYNC | p_stall::STALL_PACK, p_stall::WAIT_SFPU)
        ));
        PACK((t6_mutex_release(mutex::SFPU)));
    }

    // S = Q_r K^T (raw scores) and the row state into Dst; state update and P on the
    // pack thread.
    void score(const Step& step) {
        const bool first = step.chunk == 0;
        const uint32_t started = km::wall();
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
        const uint32_t matmul_issued = km::wall();
        math_vector_begin();
        state_vector(step.key_tiles);
        math_vector_end();
        tile_regs_commit();
        MATH((profile_counters[kPrMaMax] += km::wall() - matmul_issued));
        MATH((profile_counters[kPrMaScore] += km::wall() - started));

        cb_reserve_back(cb_probabilities, SUB_CHUNK);
        cb_reserve_back(cb_state_token, 1);
        tile_regs_wait();
        const uint32_t math_ready = km::wall();
        vector_begin();
        exp_vector(first, step.key_tiles);
        vector_end();
        const uint32_t exp_done = km::wall();
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
        PACK((profile_counters[kPrPkWait] += math_ready - started));
        PACK((profile_counters[kPrPkExp] += exp_done - math_ready));
        PACK((profile_counters[kPrPkPack] += km::wall() - exp_done));
        PACK((profile_counters[kPrPkTiles] += step.key_tiles));
    }

    // Dst 0..: P V columns [j0, j0 + wv) of the block, then the row-sum column if
    // `sum`.
    void pv_columns(
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

    // O_r (and l_r) += P V (and P 1); scaled by alpha first when the tile's reference
    // moved.
    void output(const Step& step) {
        const bool first = step.chunk == 0;
        const uint32_t started = km::wall();
        cb_wait_front(cb_state_token, 1);
        cb_wait_front(cb_probabilities, SUB_CHUNK);
        const uint32_t moved = (
            first ? 0 : read_tile_value(cb_state, slot(step.row, step.chunk), ANY_WORD)
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
                    first_column <= VALUE_TILES && VALUE_TILES < first_column + columns
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
                    step, column, column < VALUE_TILES ? 1 : 0, column == VALUE_TILES
                );
                reconfig_data_format_srca(cb_keys_values, cb_state);
                copy_init(cb_state);
                copy_tile(cb_state, slot(step.row, step.chunk), 2);
                copy_tile(cb_state, slot(step.row, step.chunk - 1), 3);
                copy_init(cb_accumulator);
                copy_tile(cb_accumulator, step.row * ACCUMULATOR_COLUMNS + column, 1);
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
        MATH((profile_counters[kPrMaPv] += km::wall() - started));
        PACK((profile_counters[kPrPkOut] += km::wall() - started));
    }

    // Output row r = O_r / l_r (bf16), to the writer. Pack waits (with a timeout) for
    // room; without it the row is not packed and the command ends in a give-up.
    void finalize(const km::Compute& compute, uint32_t row) {
        const uint32_t started = km::wall();
        cb_wait_front(cb_accumulator_token, 1);
        cb_pop_front(cb_accumulator_token, 1);
        const bool room = compute.reserve_back(cb_output, VALUE_TILES);
        for (uint32_t first_column = 0; first_column < VALUE_TILES; first_column += 3) {
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
            for (uint32_t tile = 0; room && tile < columns; ++tile) {
                pack_tile(tile, cb_output);
            }
            tile_regs_release_math_clear();
        }
        if (room) {
            cb_push_back(cb_output, VALUE_TILES);
        }
        PACK((profile_counters[kPrPkFinal] += km::wall() - started));
    }

    // The constant tiles, built by the vector unit and packed once: ones (cb_ones);
    // with FLASH_CAUSAL the diagonal mask (cb_causal).
    void constant_tiles() {
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
        pack_format(false);
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

    // One job: its rows against key tiles [0, key_tiles) in BLOCK_KEYS blocks from the
    // broadcast, then finalized (`work` false: blocks popped only). False: gave up.
    bool job(
        const km::Compute& compute,
        uint32_t rows,
        uint32_t first_row,
        uint32_t key_tiles,
        uint32_t stride,
        uint32_t& blocks,
        bool work
    ) {
        uint32_t started = km::wall();
        if (!compute.wait_front(cb_queries, MAX_ROWS * HEAD_TILES)) {
            return false;
        }
        UNPACK((profile_counters[kPrUnWaitQ] += km::wall() - started));
        if (!work) {
            rows = 0;
        }
        uint32_t chunk_base = 0;
        for (
            uint32_t block_start = 0; block_start < key_tiles; block_start += BLOCK_KEYS
        ) {
            const uint32_t keys_in_block = (
                key_tiles - block_start < BLOCK_KEYS
                    ? key_tiles - block_start
                    : BLOCK_KEYS
            );
            const uint32_t sub_chunks = (keys_in_block + SUB_CHUNK - 1) / SUB_CHUNK;
            started = km::wall();
            if (!km::broadcast_receive(compute, cb_keys_values, BLOCK_PAGES)) {
                return false;
            }
            UNPACK((profile_counters[kPrUnWaitKv] += km::wall() - started));
            // Visible steps in (chunk, row) order; pairs hold two different rows.
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
            km::broadcast_done(cb_keys_values, BLOCK_PAGES);
            ++blocks;
            chunk_base += sub_chunks;
        }
        for (uint32_t row = 0; row < rows; ++row) {
            finalize(compute, row);
        }
        cb_pop_front(cb_queries, MAX_ROWS * HEAD_TILES);
        return true;
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    const uint32_t window = compute.arg(kArgWindow);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_queries, cb_keys_values, cb_probabilities
    );
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, DST_ACCUM_MODE>()));
    MATH((flash::init(__builtin_bit_cast(float, SCALE_BITS))));
    MATH((flash::macro_init()));
    PACK((flash::addr_init()));
    constant_tiles();
    // Row state and O are slots written in place, reserved once and never pushed.
    cb_reserve_back(cb_state, 2 * MAX_ROWS);
    cb_reserve_back(cb_accumulator, MAX_ROWS * ACCUMULATOR_COLUMNS);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerAttend) {
            return;
        }
        const uint32_t started = km::wall();
        const uint32_t mode = command.a5, requested_passes = command.a6;
        const uint32_t first_job = command.a7;
        for (uint32_t word = 0; word < kProfWords; ++word) {
            profile_counters[word] = 0;
        }
        if (!compute.wait_front(kCbJobs, 1)) {
            return;
        }
        const uint32_t table = get_tile_address(kCbJobs, 0);
        invalidate_l1_cache();
        const uint32_t jobs = km::l1(table)[kTabJobs];
        const uint32_t passes = requested_passes == 0 ? 1 : requested_passes;
        const bool work = mode != kModeBroadcastOnly;
        uint32_t blocks = 0;
        bool ok = jobs <= kJobsMax;
        for (uint32_t pass = 0; ok && pass < passes; ++pass) {
            for (
                uint32_t job_index = first_job;
                ok && job_index < jobs_end(command, jobs);
                ++job_index
            ) {
                const volatile tt_l1_ptr uint32_t* job_words = (
                    km::l1(table + 4 * (kTabFirst + kJobWords * job_index))
                );
                ok = (
                    job(
                        compute,
                        job_words[kJobRows],
                        job_words[kJobStart],
                        job_words[kJobKeys],
                        job_words[kJobStride],
                        blocks,
                        work
                    )
                );
            }
        }
        if (ok) {
            km::broadcast_end(compute, cb_keys_values, BLOCK_PAGES, window, blocks);
        }
        cb_pop_front(kCbJobs, 1);
        // Each thread adds its own words (the writer may have zeroed the page).
        const uint32_t body = km::wall() - started;
        volatile tt_l1_ptr uint32_t* profile = km::l1(get_tile_address(kCbProf, 0));
        invalidate_l1_cache();
        UNPACK((profile_counters[kPrUnBody] = body));
        MATH((profile_counters[kPrMaBody] = body));
        PACK((profile_counters[kPrPkBody] = body));
        UNPACK((flush(profile, kPrUnBody, 4)));
        MATH((flush(profile, kPrMaBody, 4)));
        PACK((flush(profile, kPrPkBody, 8)));
    });
    PACK((flash::restore_constants()));
}
