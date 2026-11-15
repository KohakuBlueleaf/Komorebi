// The dispatcher's side of a dynamic run: the two-pass check of the status lines and
// the end command it submits to the run's group once every line is idle and balanced.
#include <stdint.h>

#include "sim_impl.hpp"

namespace kmsim {

    void Sim::Impl::messages_submitted(const Job& job, size_t k, uint32_t sequence) {
        if (cfg.messages_handler != 0 && job.handler == cfg.messages_handler) {
            messages_k = static_cast<int32_t>(k);
            messages_sequence = sequence;
            messages_ended = false;
        }
    }

    void Sim::Impl::messages_watch(std::vector<Job>& jobs) {
        if (messages_k < 0 || messages_ended) {
            return;
        }
        const Job& run = jobs[messages_k];
        sim.messages.checks++;
        std::vector<uint32_t> first(4 * run.count);
        bool idle = true;
        for (uint32_t k = 0; k < run.count; k++) {
            for (uint32_t w = 0; w < 4; w++) {
                first[4 * k + w] = disp_word(kDispLines + 16 * (run.first + k) + 4 * w);
            }
            const uint32_t state = first[4 * k + KM_MESSAGES_LINE_STATE];
            if (first[4 * k + KM_MESSAGES_LINE_RUN] != messages_sequence) {
                idle = false;
            } else if (state == KM_MESSAGES_GAVE_UP) {
                sim.messages.gave = run.first + k + 1;
                sim.timed_out = true;
                stop_all();
                return;
            } else if (state != KM_MESSAGES_IDLE) {
                idle = false;
            }
        }
        if (!idle) {
            return;
        }
        yield();  // the second pass starts after the first has finished
        uint64_t sent = 0, consumed = 0;
        for (uint32_t k = 0; k < run.count; k++) {
            for (uint32_t w = 0; w < 4; w++) {
                if (
                    disp_word(kDispLines + 16 * (run.first + k) + 4 * w) !=
                    first[4 * k + w]
                ) {
                    return;
                }
            }
            sent += first[4 * k + KM_MESSAGES_LINE_SENT];
            consumed += first[4 * k + KM_MESSAGES_LINE_CONSUMED];
        }
        if (sent != consumed) {
            return;
        }
        sim.messages.quiet_at = clock;
        sim.messages.sent = sent;
        Job end;
        end.handler = KM_HANDLER_MESSAGES_END;
        end.first = run.first;
        end.count = run.count;
        end.a[0] = messages_sequence;
        jobs.push_back(end);
        sim.cfg.jobs.push_back(end);  // the checks read the groups from here
        sim.jobs.push_back(JobResult{});
        messages_ended = true;
    }

}  // namespace kmsim
