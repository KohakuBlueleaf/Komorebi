// Attention writer (NoC1): its copy of the job table, each job's rows to O, and with
// BROADCAST_ON_WRITER first the job's K / V blocks (broadcast). PROF: the profile page.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "attention.h"
#include "komorebi/device/movement/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

namespace {

    constexpr uint32_t HEAD_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t VALUE_TILES = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_KEYS = get_compile_time_arg_val(3);

    // The command's jobs, `passes` times: each job's blocks through `broadcast` (null:
    // none), its rows written (full), popped (compute only) or none. False: gave up.
    template <typename Outputs, typename Keys, typename Values>
    bool run_jobs(
        const km::Writer& writer,
        const km_command_t& command,
        uint32_t table,
        const Outputs& outputs,
        const Keys& keys,
        const Values& values,
        km::Broadcast* broadcast,
        volatile tt_l1_ptr uint32_t* profile,
        uint32_t started
    ) {
        const uint32_t mode = command.a5, requested_passes = command.a6;
        const uint32_t first_job = command.a7;
        const uint32_t jobs = km::l1(table)[kTabJobs];
        const uint32_t passes = requested_passes == 0 ? 1 : requested_passes;
        if (jobs > kJobsMax) {
            return false;
        }
        const uint32_t outputs_per_row = mode == kModeBroadcastOnly ? 0 : 1;
        const bool writes = mode == kModeFull;
        uint32_t counters[kProfWords] = {};
        uint32_t blocks = 0;  // broadcast blocks of this command so far
        for (uint32_t pass = 0; pass < passes; ++pass) {
            for (
                uint32_t job_index = first_job; job_index < jobs_end(command, jobs);
                ++job_index
            ) {
                const volatile tt_l1_ptr uint32_t* job_words = (
                    km::l1(table + 4 * (kTabFirst + kJobWords * job_index))
                );
                if (
                    broadcast != nullptr &&
                    !broadcast_job<HEAD_TILES, VALUE_TILES, BLOCK_KEYS>(
                        writer,
                        command,
                        *broadcast,
                        keys,
                        values,
                        job_words,
                        blocks,
                        counters
                    )
                ) {
                    return false;
                }
                const uint32_t first_page = job_words[kJobO];
                const uint32_t rows = job_words[kJobRows] * outputs_per_row;
                const uint32_t stride = job_words[kJobStride];
                for (uint32_t row = 0; row < rows; ++row) {
                    const uint32_t wait_started = km::wall();
                    if (!writer.wait_front(kCbOut, VALUE_TILES)) {
                        return false;  // nothing to report: the tile is stopping
                    }
                    const uint32_t wait_done = km::wall();
                    const uint32_t source = get_read_ptr(kCbOut);
                    if (
                        writes &&
                        !km::write_pages(
                            writer,
                            command,
                            outputs,
                            first_page + row * stride * VALUE_TILES,
                            VALUE_TILES,
                            source
                        )
                    ) {
                        return false;
                    }
                    if (!flushed(writer, km::kStWriter, km::kFlushSent)) {
                        return false;
                    }
                    cb_pop_front(kCbOut, VALUE_TILES);
                    counters[kPrWrWait] += wait_done - wait_started;
                    counters[kPrWrWrite] += km::wall() - wait_done;
                }
            }
        }
        if (broadcast != nullptr) {
            const uint32_t finish_started = km::wall();
            broadcast->finish();
            counters[kPrRdFinish] += km::wall() - finish_started;
            for (uint32_t word = kPrRdKvFree; word <= kPrRdFinish; ++word) {
                km::add_sat(
                    profile + word, counters[word]
                );  // the broadcast's words, the reader's otherwise
            }
        }
        km::add_sat(profile + kPrWrWait, counters[kPrWrWait]);
        km::add_sat(profile + kPrWrWrite, counters[kPrWrWrite]);
        km::add_sat(profile + kPrWrBody, km::wall() - started);
        return true;
    }

}  // namespace

void kernel_main() {
    // O, K, V: bf16 tiles in DRAM, after the five sizes; then the job tables and the
    // profile pages
    using O = km::Tensor<5>;
    using K = km::Tensor<O::next_compile_time_args_offset()>;
    using V = km::Tensor<K::next_compile_time_args_offset()>;
    using T = km::Tensor<V::next_compile_time_args_offset()>;
    using P = km::Tensor<T::next_compile_time_args_offset()>;
    km::Writer writer;
    // The table copy's page: reserved once, used in place, never pushed.
    cb_reserve_back(kCbWriterJobs, 1);
    const uint32_t table = get_write_ptr(kCbWriterJobs);
    volatile tt_l1_ptr uint32_t* profile = km::l1(get_write_ptr(kCbProf));
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerProf) {
            const uint32_t profile_address = command.a0, zero_profile = command.a1;
            const uint32_t rank = writer.me() - km::first(command);
            if (!km::write_pages(
                writer, command, P(profile_address), rank, 1, get_write_ptr(kCbProf)
            )) {
                return false;
            }
            if (!flushed(writer, km::kStWriter, km::kFlushSent)) {
                return false;
            }
            for (uint32_t word = 0; zero_profile != 0 && word < kProfWords; ++word) {
                profile[word] = 0;
            }
            return true;
        }
        if (km::handler(command) != kHandlerAttend) {
            return true;
        }
        const uint32_t started = km::wall();
        const uint32_t k_address = command.a1, v_address = command.a2;
        const uint32_t o_address = command.a3, job_tables = command.a4;
        const uint32_t mode = command.a5;
        if (!read_table(
            writer,
            command,
            km::kStWriter,
            T(job_tables),
            writer.me() - km::first(command),
            table
        )) {
            return false;
        }
        const K keys(k_address);
        const V values(v_address);
        const O outputs(o_address);
#ifdef BROADCAST_ON_WRITER
        if (mode != kModeComputeOnly) {
            km::Broadcast broadcast(writer, command, kArgBroadcast);
            if (!broadcast.ok()) {
                return false;
            }
            if (!broadcast.included()) {
                return true;  // in no broadcast: it still reports (false: it would not)
            }
            return (
                run_jobs(
                    writer,
                    command,
                    table,
                    outputs,
                    keys,
                    values,
                    &broadcast,
                    profile,
                    started
                )
            );
        }
#endif
        return (
            run_jobs(
                writer, command, table, outputs, keys, values, nullptr, profile, started
            )
        );
    });
}
