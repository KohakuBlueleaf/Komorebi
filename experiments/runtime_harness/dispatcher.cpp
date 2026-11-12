// The dispatcher model and the grid's set-up: records issued as the dispatch kernel
// does, MAP inputs prepared and outputs checked, every core's buffers and arguments.
#include <stdint.h>

#include <cstring>

#include "sim_impl.hpp"

namespace kmsim {

    void Sim::Impl::prepare_map(Job& job, uint32_t k, uint32_t j) {
        const uint32_t tiles = job.a[1];
        job.a[0] = kDramIn + j * kInTiles * kTileBytes;
        job.a[2] = kDramOut + j * kOutPages * kTileBytes;
        job.a[3] = 0;
        for (uint32_t i = 0; i < tiles && i < kInTiles; i++) {
            uint16_t* t = (
                reinterpret_cast<uint16_t*>(dram.data() + job.a[0] + i * kTileBytes)
            );
            for (uint32_t e = 0; e < kTileBytes / 2; e++) {
                t[e] = to_bf16(input_value(k, i, e));
            }
        }
        std::memset(dram.data() + job.a[2], 0xFF, kOutPages * kTileBytes);
    }

    uint32_t Sim::Impl::check_map(const Job& job, uint32_t k) {
        uint32_t bad = 0;
        for (uint32_t m = 0; m < job.count && m < kOutPages; m++) {
            const km::Share s = km::share(job.a[1], m, job.count);
            const uint16_t* t = (
                reinterpret_cast<const uint16_t*>(
                    dram.data() + job.a[2] + (job.a[3] + m) * kTileBytes
                )
            );
            for (uint32_t e = 0; e < kTileBytes / 2; e++) {
                float want = 0;
                for (uint32_t i = s.start; i < s.start + s.len; i++) {
                    want += input_value(k, i, e);
                }
                bad += from_bf16(t[e]) != want;
            }
        }
        return bad;
    }

    uint32_t Sim::Impl::issue(const Job& job, int32_t k, uint32_t j) {
        const uint32_t s = next_seq();
        uint32_t w[16] = {};
        w[0] = w[4] = w[8] = w[12] = s;
        w[1] = (km::make_hdr(job.seg, job.flags, job.hdr_slot >= 0 ? job.hdr_slot : j));
        w[2] = km::make_team(job.first, job.count);
        const uint32_t at[10] = {3, 5, 6, 7, 9, 10, 11, 13, 14, 15};
        for (uint32_t i = 0; i < 10; i++) {
            w[at[i]] = job.a[i];
        }
        for (uint32_t c = 0; c < n; c++) {
            send(
                n,
                self,
                c,
                kCtlOff + KM_CTL_RING + j * 64,
                reinterpret_cast<const uint8_t*>(w),
                64,
                k,
                0,
                s
            );
        }
        if (cfg.on_issue && k >= 0) {
            cfg.on_issue(static_cast<uint32_t>(k), s, j);
        }
        return s;
    }

    bool Sim::Impl::booting() {
        const uint64_t t0 = clock;
        const uint32_t groups = (n + cfg.row - 1) / cfg.row;
        for (;;) {
            bool ok = disp_word(kDispMailbox) == 1;
            for (uint32_t g = 0; g < groups; g++) {
                ok = ok && disp_word(kDispReady + g * 16) == KM_READY;
            }
            if (ok) {
                return true;
            }
            if (clock - t0 > cfg.deadline || abort) {
                return false;
            }
            yield();
        }
    }

    bool Sim::Impl::held(
        const std::vector<Job>& jobs, size_t k, const int32_t* busy, uint32_t d
    ) {
        if ((jobs[k].flags & KM_F_EXCL) == 0) {
            return false;
        }
        const uint32_t team = km::make_team(jobs[k].first, jobs[k].count);
        for (uint32_t j = 0; j < d; j++) {
            if (busy[j] >= 0 &&
                km::teams_overlap(
                    team, km::make_team(jobs[busy[j]].first, jobs[busy[j]].count)
                )) {
                return true;
            }
        }
        return false;
    }

    void Sim::Impl::dispatcher() {
        if (!cfg.dispatch) {
            return;
        }
        if (!booting()) {
            stop_all();
            sim.timed_out = true;
            return;
        }
        booted = sim.booted = true;
        std::vector<Job> jobs = cfg.jobs;
        sim.jobs.assign(jobs.size(), JobResult{});
        int32_t busy[KM_RING_SLOTS];
        for (auto& b : busy) {
            b = -1;
        }
        const uint32_t depth = cfg.depth == 0 || cfg.depth > 16 ? 16 : cfg.depth;
        size_t next = 0, done = 0;
        while (done < jobs.size()) {
            if (abort) {
                return;
            }
            for (uint32_t j = 0; j < depth; j++) {
                if (busy[j] < 0) {
                    continue;
                }
                JobResult& r = sim.jobs[busy[j]];
                if (disp_word(kDispAck + j * 16) == KM_ACK(r.seq)) {
                    r.acked = clock;
                    r.result = disp_word(kDispAck + j * 16 + 4);
                    r.status = kDone;
                    if (jobs[busy[j]].map) {
                        r.data_bad = check_map(jobs[busy[j]], busy[j]);
                    }
                    busy[j] = -1;
                    done++;
                } else if (clock - r.issued > cfg.deadline) {
                    r.status = kTimedOut;
                    sim.timed_out = true;
                    stop_all();
                    return;
                }
            }
            const uint64_t last = next ? sim.jobs[next - 1].issued : 0;
            if (next < jobs.size() && clock >= last + jobs[next].gap) {
                uint32_t j = 0;
                while (j < depth && busy[j] >= 0) {
                    j++;
                }
                if (j < depth && held(jobs, next, busy, depth)) {
                    j = depth;
                }
                if (j < depth) {
                    Job& job = jobs[next];
                    if (job.map) {
                        prepare_map(job, next, j);
                    }
                    JobResult& r = sim.jobs[next];
                    r.slot = j;
                    r.issued = clock;
                    r.status = kIssued;
                    r.seq = issue(job, static_cast<int32_t>(next), j);
                    busy[j] = static_cast<int32_t>(next);
                    next++;
                }
            }
            yield();
        }
        if (!cfg.exit) {
            return;
        }
        Job x;
        x.seg = KM_SEG_EXIT;
        x.first = cfg.exit_team & 0xFFFFu;
        x.count = cfg.exit_team ? cfg.exit_team >> 16 : n;
        JobResult& r = sim.exit_job;
        r.slot = 0;
        r.issued = clock;
        r.status = kIssued;
        r.seq = issue(x, -1, 0);
        for (;;) {
            if (disp_word(kDispAck) == KM_ACK(r.seq)) {
                r.acked = clock;
                r.status = kDone;
                return;
            }
            if (clock - r.issued > cfg.deadline) {
                r.status = kTimedOut;
                sim.timed_out = true;
                stop_all();
                return;
            }
            if (abort) {
                return;
            }
            yield();
        }
    }

    void Sim::Impl::setup() {
        const size_t bytes = static_cast<size_t>(n + 1) * kL1Bytes;
        void* p = (
            mmap(
                nullptr,
                bytes,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT,
                -1,
                0
            )
        );
        if (p == MAP_FAILED) {
            std::perror("mmap");
            std::exit(2);
        }
        region = static_cast<uint8_t*>(p);
        region_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
        dram.assign(kDramBytes, 0);
        cores.resize(n + 1);
        for (uint32_t c = 0; c <= n; c++) {
            CoreState& s = cores[c];
            s.mem = region + static_cast<size_t>(c) * kL1Bytes;
            s.base = region_base + c * kL1Bytes;
            s.x = c < n ? c % cfg.row + 1 : 100;
            s.y = c < n ? c / cfg.row + 1 : 100;
            for (auto& d : s.dst) {
                d.assign(kTileBytes / 2, 0.0f);
            }
            const auto cb =
                [&](uint32_t id, uint32_t off, uint32_t page, uint32_t pages) {
                    s.cb[id] = Cb{s.base + off, page, pages};
                };
            cb(km::kCbTokenCompute, 0x2000, km::kTokenBytes, 1);
            cb(km::kCbTokenWriter, 0x2040, km::kTokenBytes, 1);
            cb(0, 0x3000, kTileBytes, 16);
            cb(16, 0xB000, kTileBytes, 2);
            for (const CbSpec& x : cfg.cbs) {
                cb(x.id, x.off, x.page, x.pages);
            }
            if (c < n) {
                uint32_t* table = reinterpret_cast<uint32_t*>(dram.data() + kDramTable);
                table[c] = s.x | s.y << 16;
            }
        }
        const CoreState& d = cores[n];
        for (uint32_t c = 0; c < n; c++) {
            CoreState& s = cores[c];
            const uint32_t leader = c / cfg.row * cfg.row;
            const uint32_t row_end = leader + cfg.row < n ? leader + cfg.row : n;
            km::Args a = {};
            a.ctl = s.base + kCtlOff;
            a.me = c;
            a.cores = kDramTable;
            a.disp_xy = d.x | d.y << 16;
            a.ack_base = d.base + kDispAck;
            a.mailbox = c == 0 ? d.base + kDispMailbox : 0;
            a.leader = leader;
            a.rank = c == leader ? 0 : c - leader - 1;
            a.members = row_end - leader - 1;
            a.ready_slot = leader / cfg.row;
            a.ready_base = d.base + kDispReady;
            a.tag = 0x5A5A;
            a.depth = cfg.depth;
            a.wait_limit = cfg.wait_limit;
            a.idle_limit = cfg.idle_limit;
            a.row = cfg.tree_row;
            a.workers = n;
            const uint32_t* f = &a.ctl;
            s.args.assign(f, f + km::kArgs);
            s.compute_args = {a.ctl, a.me, a.wait_limit, a.idle_limit};
        }
    }

}  // namespace kmsim
