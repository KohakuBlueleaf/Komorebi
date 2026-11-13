// The dispatcher's side of a dynamic run: the two-pass check of the status lines and
// the end command it submits to the run's group once every line is idle and balanced.
#include <stdint.h>

#include "sim_impl.hpp"

namespace kmsim {

    void Sim::Impl::dyn_submitted(const Job& job, size_t k, uint32_t seq) {
        if (cfg.dyn_handler != 0 && job.handler == cfg.dyn_handler) {
            dyn_k = static_cast<int32_t>(k);
            dyn_seq = seq;
            dyn_ended = false;
        }
    }

    void Sim::Impl::dyn_watch(std::vector<Job>& jobs) {
        if (dyn_k < 0 || dyn_ended) {
            return;
        }
        const Job& run = jobs[dyn_k];
        sim.dyn.checks++;
        std::vector<uint32_t> first(4 * run.count);
        bool idle = true;
        for (uint32_t k = 0; k < run.count; k++) {
            for (uint32_t w = 0; w < 4; w++) {
                first[4 * k + w] = disp_word(kDispLines + 16 * (run.first + k) + 4 * w);
            }
            const uint32_t state = first[4 * k + KM_DYN_LINE_STATE];
            if (first[4 * k + KM_DYN_LINE_RUN] != dyn_seq) {
                idle = false;
            } else if (state == KM_DYN_GAVE_UP) {
                sim.dyn.gave = run.first + k + 1;
                sim.timed_out = true;
                stop_all();
                return;
            } else if (state != KM_DYN_IDLE) {
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
                if (disp_word(kDispLines + 16 * (run.first + k) + 4 * w) !=
                    first[4 * k + w]) {
                    return;
                }
            }
            sent += first[4 * k + KM_DYN_LINE_SENT];
            consumed += first[4 * k + KM_DYN_LINE_CONSUMED];
        }
        if (sent != consumed) {
            return;
        }
        sim.dyn.quiet_at = clock;
        sim.dyn.sent = sent;
        Job end;
        end.handler = KM_HANDLER_DYN_END;
        end.first = run.first;
        end.count = run.count;
        end.a[0] = dyn_seq;
        jobs.push_back(end);
        sim.cfg.jobs.push_back(end);  // the checks read the groups from here
        sim.jobs.push_back(JobResult{});
        dyn_ended = true;
    }

}  // namespace kmsim
