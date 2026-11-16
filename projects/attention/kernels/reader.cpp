// Attention reader (NoC0): the core's job table to compute, then per job its Q rows
// and, unless BROADCAST_ON_WRITER, the job's K / V blocks (the sender multicasts them).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "attention.h"
#include "komorebi/device/movement/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

namespace {

    constexpr uint32_t HEAD_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t VALUE_TILES = get_compile_time_arg_val(1);
    constexpr uint32_t MAX_ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK_KEYS = get_compile_time_arg_val(3);
    constexpr uint32_t WINDOW_BLOCKS = get_compile_time_arg_val(4);  // broadcast window
    constexpr uint32_t BLOCK_PAGES = HEAD_TILES * BLOCK_KEYS +
        BLOCK_KEYS * VALUE_TILES;  // pages of a K / V block

    // Rows `rows` of a job's Q (HEAD_TILES tiles each, `stride` tile rows apart) into
    // the Q buffer, as one item of MAX_ROWS x HEAD_TILES pages. False: gave up.
    template <typename Queries>
    bool read_q(
        const km::Reader& reader,
        const km_command_t& command,
        const Queries& queries,
        uint32_t first_tile,
        uint32_t rows,
        uint32_t stride,
        uint32_t* counters
    ) {
        const uint32_t started = km::wall();
        if (!reader.reserve(kCbQ, MAX_ROWS * HEAD_TILES)) {
            return false;
        }
        const uint32_t reserved = km::wall();
        const uint32_t destination = get_write_ptr(kCbQ);
        for (uint32_t row = 0; row < rows; ++row) {
            if (!km::read_pages(
                reader,
                command,
                queries,
                first_tile + row * stride * HEAD_TILES,
                HEAD_TILES,
                destination + row * HEAD_TILES * kTileBytes
            )) {
                return false;
            }
        }
        if (!flushed(reader, km::kStReader, km::kFlushReads)) {
            return false;
        }
        cb_push_back(kCbQ, MAX_ROWS * HEAD_TILES);
        counters[kPrRdQWait] += reserved - started;
        counters[kPrRdQRead] += km::wall() - reserved;
        return true;
    }

    // Compute only: `pages` pushed as they are in the broadcast buffer (no read, no
    // broadcast).
    bool push_block(const km::Reader& reader, uint32_t pages) {
        if (!reader.reserve(kCbKv, pages)) {
            return false;
        }
        cb_push_back(kCbKv, pages);
        return true;
    }

    // The command's jobs, `passes` times: Q rows, then K / V blocks through `broadcast`
    // (null: not this kernel's), or pushed as they are (compute only).
    template <typename Queries, typename Keys, typename Values>
    void run_jobs(
        const km::Reader& reader,
        const km_command_t& command,
        uint32_t table,
        const Queries& queries,
        const Keys& keys,
        const Values& values,
        km::Broadcast* broadcast,
        volatile tt_l1_ptr uint32_t* profile,
        uint32_t started
    ) {
        const uint32_t mode = command.a5, requested_passes = command.a6;
        const uint32_t first_job = command.a7;
        const bool reads = mode != kModeComputeOnly;
        const uint32_t jobs = km::l1(table)[kTabJobs];
        const uint32_t passes = requested_passes == 0 ? 1 : requested_passes;
        if (jobs > kJobsMax) {
            return;
        }
        uint32_t counters[kProfWords] = {};
        uint32_t blocks = 0;  // blocks of this command so far, over all jobs and passes
        for (uint32_t pass = 0; pass < passes; ++pass) {
            for (
                uint32_t job_index = first_job; job_index < jobs_end(command, jobs);
                ++job_index
            ) {
                const volatile tt_l1_ptr uint32_t* job_words = (
                    km::l1(table + 4 * (kTabFirst + kJobWords * job_index))
                );
                const uint32_t rows = reads ? job_words[kJobRows] : 0;
                if (!read_q(
                    reader,
                    command,
                    queries,
                    job_words[kJobQ],
                    rows,
                    job_words[kJobStride],
                    counters
                )) {
                    return;
                }
                if (
                    broadcast != nullptr &&
                    !broadcast_job<HEAD_TILES, VALUE_TILES, BLOCK_KEYS>(
                        reader,
                        command,
                        *broadcast,
                        keys,
                        values,
                        job_words,
                        blocks,
                        counters
                    )
                ) {
                    return;
                }
                for (
                    uint32_t block_start = 0;
                    !reads && block_start < job_words[kJobKeys];
                    block_start += BLOCK_KEYS, ++blocks
                ) {
                    if (!push_block(reader, BLOCK_PAGES)) {
                        return;
                    }
                }
            }
        }
        const uint32_t finish_started = km::wall();
        if (!reads) {
            const uint32_t pad = (
                (WINDOW_BLOCKS - blocks % WINDOW_BLOCKS) % WINDOW_BLOCKS
            );
            if (pad != 0 && !push_block(reader, pad * BLOCK_PAGES)) {
                return;
            }
        } else if (broadcast != nullptr) {
            broadcast->finish();
        }
        counters[kPrRdFinish] += km::wall() - finish_started;
        counters[kPrRdBody] += km::wall() - started;
        // Without the broadcast here, its words are the writer's.
        const uint32_t last = broadcast != nullptr || !reads ? kPrRdFinish : kPrRdQRead;
        for (uint32_t word = kPrRdBody; word <= last; ++word) {
            km::add_sat(profile + word, counters[word]);
        }
    }

}  // namespace

void kernel_main() {
    // Q, K, V: bf16 tiles in DRAM, after the five sizes; then the job tables
    using Q = km::Tensor<5>;
    using K = km::Tensor<Q::next_compile_time_args_offset()>;
    using V = km::Tensor<K::next_compile_time_args_offset()>;
    using T = km::Tensor<V::next_compile_time_args_offset()>;
    km::Reader reader;
    volatile tt_l1_ptr uint32_t* profile = km::l1(get_write_ptr(kCbProf));
    for (uint32_t word = 0; word < kProfWords; ++word) {
        profile[word] = 0;
    }
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerAttend) {
            return;
        }
        const uint32_t started = km::wall();
        const uint32_t q_address = command.a0, k_address = command.a1;
        const uint32_t v_address = command.a2, job_tables = command.a4;
        const uint32_t mode = command.a5;
        if (!reader.reserve(kCbJobs, 1)) {
            return;
        }
        const uint32_t table = get_write_ptr(kCbJobs);
        if (!read_table(
            reader,
            command,
            km::kStReader,
            T(job_tables),
            reader.me() - km::first(command),
            table
        )) {
            return;
        }
        cb_push_back(kCbJobs, 1);  // compute pops it at the command's end
        const Q queries(q_address);
        const K keys(k_address);
        const V values(v_address);
#ifndef BROADCAST_ON_WRITER
        if (mode != kModeComputeOnly) {
            km::Broadcast broadcast(reader, command, kArgBroadcast);
            if (broadcast.ok() && broadcast.included()) {
                run_jobs(
                    reader,
                    command,
                    table,
                    queries,
                    keys,
                    values,
                    &broadcast,
                    profile,
                    started
                );
            }
            return;
        }
#endif
        run_jobs(
            reader, command, table, queries, keys, values, nullptr, profile, started
        );
    });
}
