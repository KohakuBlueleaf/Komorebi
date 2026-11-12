// The simulator: scheduler, L1 images, circular buffers, NoC delivery, compute threads
// and the dispatcher model. One thread runs at a time, in a fixed rotation.
#include <pthread.h>
#include <sched.h>
#include <linux/futex.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <queue>
#include <random>
#include <thread>

#include "api/dataflow/dataflow_api.h"
#include "harness.hpp"
#include "komorebi/device/km_core.hpp"

namespace kmsim {

    volatile uint32_t wall_clock[2];

    const char* role_name(uint32_t role) {
        static const char* names[] = {"reader", "writer", "unpack", "math", "pack"};
        return role < kRoles ? names[role] : "?";
    }

    namespace {

        // Image offset of the buffers' counter registers (8 bytes per buffer).
        constexpr uint32_t kCbRegs = 0x200;

        struct Abort {};

        // A thread's turn: set by the thread before it, waited for without spinning.
        class Gate {
        public:
            void open() {
                flag_.store(1, std::memory_order_release);
                syscall(SYS_futex, &flag_, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
            }

            void pass() {
                while (flag_.exchange(0, std::memory_order_acquire) == 0) {
                    syscall(
                        SYS_futex, &flag_, FUTEX_WAIT_PRIVATE, 0, nullptr, nullptr, 0
                    );
                }
            }

        private:
            std::atomic<int> flag_{0};
        };

        struct Thread {
            Gate sem;
            std::thread th;
            bool alive = true;
            bool kernel = false;
            bool returned = false;
            uint32_t core = 0;
            Role role = kReader;
            void (*fn)() = nullptr;
            const char* last = "start";
            uint32_t last_cb = ~0u;
            uint64_t frozen_until = 0;
            uint32_t writes = 0;   // writes issued and not yet landed
            uint32_t atomics = 0;  // atomics issued and not yet landed
        };

        struct Cb {
            uint32_t base = 0, page = 0, pages = 0;
            uint32_t wr = 0, rd = 0, received = 0, acked = 0, reserved = 0;
        };

        struct CoreState {
            uint8_t* mem = nullptr;
            uint32_t base = 0;  // host address of the image (below 4 GiB)
            uint32_t x = 0, y = 0;
            Cb cb[32];
            std::deque<uint32_t> mbox[4][4];  // [from][to], thread ids 1 to 3
            std::deque<std::vector<float>> src;
            std::vector<float> dst[16];
            int dst_state = 0;  // 0 free, 1 math, 2 committed, 3 pack
            uint32_t writer_seq = 0;
            uint32_t reader_seq = 0, compute_seq = 0;  // record each side works on
            std::vector<uint32_t> args, compute_args;
        };

        struct Pending {
            uint64_t due, ord;
            uint32_t src, dst, off, bytes;
            uint8_t data[16];
            bool atomic;
            uint32_t value;
            Thread* from;
            uint32_t kind;  // 0 plain, 1 report, 2 ack, 3 stop
            uint32_t seq;

            bool operator>(const Pending& o) const {
                return due != o.due ? due > o.due : ord > o.ord;
            }
        };

        thread_local Thread* self = nullptr;

        uint16_t to_bf16(float f) {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            return static_cast<uint16_t>(u >> 16);
        }

        float from_bf16(uint16_t h) {
            const uint32_t u = static_cast<uint32_t>(h) << 16;
            float f;
            std::memcpy(&f, &u, 4);
            return f;
        }

        // Input element e of tile i of job k: small integers, exact in bfloat16 sums.
        float input_value(uint32_t k, uint32_t i, uint32_t e) {
            return static_cast<float>((k + i + e) % 4);
        }

    }  // namespace

    struct Sim::Impl {
        Sim& sim;
        Config& cfg;
        uint32_t n;  // workers; index n is the dispatcher
        uint8_t* region = nullptr;
        uint32_t region_base = 0;
        std::vector<CoreState> cores;
        std::vector<uint8_t> dram;
        std::vector<std::unique_ptr<Thread>> threads;
        size_t cur = 0;
        uint32_t alive = 0, kernels_alive = 0;
        bool dispatcher_alive = true;
        uint64_t clock = 0;
        std::atomic<bool> abort{false};
        bool booted = false;
        std::priority_queue<Pending, std::vector<Pending>, std::greater<Pending>>
            pending;
        uint64_t ord = 0;
        std::mt19937_64 rng;
        std::mutex mu;
        std::condition_variable cv;
        bool finished = false;

        Impl(Sim& s, Config& c) : sim(s), cfg(c), n(c.cores), rng(c.seed) {
        }

        void violation(const std::string& what) {
            char head[96];
            std::snprintf(
                head,
                sizeof head,
                "t=%llu core %u %s: ",
                static_cast<unsigned long long>(clock),
                self ? self->core : 0,
                self ? (self->kernel ? role_name(self->role) : "harness") : "main"
            );
            if (sim.violations.size() < 64) {
                sim.violations.push_back(head + what);
            }
        }

        void log(uint32_t core, uint32_t kind, uint32_t a, uint32_t b) {
            sim.events.push_back({clock, core, kind, a, b});
        }

        CoreState& me() {
            return cores[self->core];
        }

        uint8_t* host(uint32_t addr) {
            return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(addr));
        }

        // ---- scheduler ----

        void tick() {
            clock += cfg.tick;
            wall_clock[0] = static_cast<uint32_t>(clock);
            wall_clock[1] = static_cast<uint32_t>(clock >> 32);
            if (!abort && clock > cfg.bound) {
                abort = true;
            }
        }

        size_t next_alive() const {
            for (size_t k = 1; k <= threads.size(); k++) {
                const size_t i = (cur + k) % threads.size();
                if (threads[i]->alive) {
                    return i;
                }
            }
            return threads.size();
        }

        void yield() {
            tick();
            const size_t next = next_alive();
            if (next != cur) {
                Thread* t = threads[next].get();
                cur = next;
                t->sem.open();
                self->sem.pass();
            }
            if (abort && self->kernel) {
                throw Abort{};
            }
        }

        void finish() {
            self->alive = false;
            alive--;
            if (self->kernel) {
                kernels_alive--;
            }
            const size_t next = next_alive();
            if (next < threads.size()) {
                cur = next;
                threads[next]->sem.open();
            } else {
                std::lock_guard<std::mutex> lock(mu);
                finished = true;
                cv.notify_all();
            }
        }

        void kernel_thread(Thread* t) {
            self = t;
            t->sem.pass();
            try {
                if (abort) {
                    throw Abort{};
                }
                t->fn();
                t->returned = true;
                if (t->writes != 0 || t->atomics != 0) {  // the card may end the core
                    violation(
                        "returned with " + std::to_string(t->writes) + " writes and " +
                        std::to_string(t->atomics) + " atomics not landed"
                    );
                }
            } catch (const Abort&) {
            }
            finish();
        }

        // ---- addresses ----

        // Offset in a core's image of a host address inside any image; ~0u: none.
        uint32_t image_off(uint32_t addr) const {
            const uint64_t end = region_base + static_cast<uint64_t>(n + 1) * kL1Bytes;
            if (addr < region_base || addr >= end) {
                return ~0u;
            }
            return (addr - region_base) % kL1Bytes;
        }

        uint32_t core_at(uint32_t x, uint32_t y) const {
            for (uint32_t c = 0; c <= n; c++) {
                if (cores[c].x == x && cores[c].y == y) {
                    return c;
                }
            }
            return ~0u;
        }

        // ---- NoC ----

        void send(
            uint32_t src,
            Thread* from,
            uint32_t dst,
            uint32_t off,
            const uint8_t* data,
            uint32_t bytes,
            int32_t job,
            uint32_t kind,
            uint32_t seq
        ) {
            const uint32_t first = off / 16, last = (off + bytes - 1) / 16;
            const uint32_t parts = last - first + 1;
            for (uint32_t p = 0; p < parts; p++) {
                const uint32_t lo = p == 0 ? off : (first + p) * 16;
                const uint32_t hi = p + 1 == parts ? off + bytes : (first + p + 1) * 16;
                const Chunk c = {src, dst, lo, hi - lo, p, parts, false, job};
                const int64_t d = cfg.delay ? cfg.delay(c) : 32 + (rng() & 63);
                if (d < 0) {
                    continue;
                }
                Pending q = {};
                q.due = clock + static_cast<uint64_t>(d);
                q.ord = ord++;
                q.src = src;
                q.dst = dst;
                q.off = lo;
                q.bytes = hi - lo;
                std::memcpy(q.data, data + (lo - off), hi - lo);
                q.from = from;
                q.kind = kind;
                q.seq = seq;
                from->writes++;
                pending.push(q);
            }
        }

        // What a core may receive from the NoC; anything else is a violation.
        bool may_land(const Pending& q) const {
            if (q.dst == kDram) {
                return true;
            }
            const uint32_t at = q.off - kCtlOff;
            if (q.dst == n) {
                const auto in = [&](uint32_t base, uint32_t count, uint32_t bytes) {
                    return (
                        q.off >= base && q.off < base + count * 16 &&
                        (q.off - base) % 16 + q.bytes <= bytes
                    );
                };
                return (
                    !q.atomic &&
                    (in(kDispAck, KM_RING_SLOTS, 4) || in(kDispReady, 16, 4) ||
                     (q.off == kDispMailbox && q.bytes <= 8))
                );
            }
            if (q.off < kCtlOff) {
                return false;
            }
            if (q.atomic) {
                return (
                    q.src < n && at >= KM_CTL_DONE && at < KM_CTL_DONE + 16 * 16 &&
                    (at - KM_CTL_DONE) % 16 == 0
                );
            }
            if (q.src == n) {
                return (
                    at + q.bytes <= KM_CTL_RSVD || (at == KM_CTL_STOP && q.bytes == 4)
                );
            }
            return (
                !booted && at >= KM_CTL_BOOT && at < KM_CTL_STATE &&
                (at - KM_CTL_BOOT) % 16 == 0 && q.bytes == 4
            );
        }

        void deliver() {
            while (!pending.empty() && pending.top().due <= clock) {
                const Pending q = pending.top();
                pending.pop();
                if (!may_land(q)) {
                    char what[160];
                    std::snprintf(
                        what,
                        sizeof what,
                        "%s from core %u landed outside what core %u receives: offset "
                        "0x%x, %u bytes",
                        q.atomic ? "atomic" : "write",
                        q.src,
                        q.dst,
                        q.off,
                        q.bytes
                    );
                    violation(what);
                }
                uint8_t* to =
                    (q.dst == kDram ? dram.data() + q.off : cores[q.dst].mem + q.off);
                if (q.atomic) {
                    uint32_t v;
                    std::memcpy(&v, to, 4);
                    v += q.value;
                    std::memcpy(to, &v, 4);
                    q.from->atomics--;
                } else {
                    std::memcpy(to, q.data, q.bytes);
                    q.from->writes--;
                }
                if (q.kind == 1) {
                    log(q.src, kEvReportLand, q.seq, q.dst);
                } else if (q.kind == 2) {
                    log(q.src, kEvAckLand, q.seq, (q.off - kDispAck) / 16);
                } else if (q.kind == 3) {
                    log(q.dst, kEvStop, 0, q.src);
                }
            }
        }

        void delivery_thread(Thread* t) {
            self = t;
            t->sem.pass();
            for (;;) {
                deliver();
                if (kernels_alive == 0 && !dispatcher_alive) {
                    break;
                }
                yield();
            }
            finish();
        }

        // ---- dispatcher ----

        uint32_t& disp_word(uint32_t off) {
            return *reinterpret_cast<uint32_t*>(cores[n].mem + off);
        }

        void stop_all() {
            const uint32_t one = 1;
            for (uint32_t w = 0; w < n; w++) {
                send(
                    n,
                    self,
                    w,
                    kCtlOff + KM_CTL_STOP,
                    reinterpret_cast<const uint8_t*>(&one),
                    4,
                    -1,
                    3,
                    0
                );
            }
        }

        uint32_t seq = 0;

        uint32_t next_seq() {
            seq =
                (seq == 0 ? cfg.first_seq
                          : (cfg.next_seq ? cfg.next_seq(seq) : seq + 1));
            return seq;
        }

        void prepare_map(Job& job, uint32_t k, uint32_t j) {
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

        uint32_t check_map(const Job& job, uint32_t k) {
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

        uint32_t issue(const Job& job, int32_t k, uint32_t j) {
            const uint32_t s = next_seq();
            uint32_t w[16] = {};
            w[0] = w[4] = w[8] = w[12] = s;
            w[1] = (
                km::make_hdr(job.seg, job.flags, job.hdr_slot >= 0 ? job.hdr_slot : j)
            );
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

        bool booting() {
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

        // True when a KM_F_EXCL record's team overlaps a team in flight, as the
        // dispatch kernel holds it.
        bool held(
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

        void dispatcher() {
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

        void dispatcher_thread(Thread* t) {
            self = t;
            t->sem.pass();
            dispatcher();
            dispatcher_alive = false;
            finish();
        }

        // ---- set-up and run ----

        void setup() {
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
                    uint32_t* table = (
                        reinterpret_cast<uint32_t*>(dram.data() + kDramTable)
                    );
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

        void add(void (*fn)(), uint32_t core, Role role, bool kernel) {
            auto t = std::make_unique<Thread>();
            t->fn = fn;
            t->core = core;
            t->role = role;
            t->kernel = kernel;
            threads.push_back(std::move(t));
        }

        void run() {
            add(nullptr, n, kReader, false);  // dispatcher
            add(nullptr, n, kReader, false);  // delivery
            for (uint32_t c = 0; c < n; c++) {
                for (uint32_t r = 0; r < kRoles; r++) {
                    add(cfg.kernels[r], c, static_cast<Role>(r), true);
                }
            }
            alive = static_cast<uint32_t>(threads.size());
            kernels_alive = alive - 2;
            threads[0]->th = std::thread([this] {
                dispatcher_thread(threads[0].get());
            });
            threads[1]->th = std::thread([this] {
                delivery_thread(threads[1].get());
            });
            for (size_t i = 2; i < threads.size(); i++) {
                Thread* t = threads[i].get();
                t->th = std::thread([this, t] {
                    kernel_thread(t);
                });
            }
            // One thread runs at a time: all on one CPU, a hand-over stays on it.
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(sched_getcpu(), &one);
            for (auto& t : threads) {
                pthread_setaffinity_np(t->th.native_handle(), sizeof one, &one);
            }
            const auto t0 = std::chrono::steady_clock::now();
            cur = 0;
            threads[0]->sem.open();
            {
                std::unique_lock<std::mutex> lock(mu);
                const auto limit = std::chrono::duration<double>(cfg.host_seconds);
                if (!cv.wait_for(lock, limit, [&] {
                        return finished;
                    })) {
                    abort = true;  // read by the running thread at its next step
                    if (!cv.wait_for(lock, std::chrono::seconds(5), [&] {
                            return finished;
                        })) {
                        std::fprintf(
                            stderr, "a thread runs without scheduling points\n"
                        );
                        std::fflush(stderr);
                        std::_Exit(3);
                    }
                    sim.violations.push_back("host-time watchdog passed");
                }
            }
            for (auto& t : threads) {
                t->th.join();
            }
            sim.host_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0
            )
                              .count();
            sim.end_cycles = clock;
            sim.all_returned = true;
            for (auto& t : threads) {
                if (t->kernel && !t->returned) {
                    sim.all_returned = false;
                    char line[128];
                    std::snprintf(
                        line,
                        sizeof line,
                        "core %u %s did not return; last call %s(cb %d)",
                        t->core,
                        role_name(t->role),
                        t->last,
                        static_cast<int>(t->last_cb)
                    );
                    sim.stuck.push_back(line);
                }
            }
        }
    };

    namespace {
        Sim::Impl* g = nullptr;
    }

    Sim::Sim(Config c) : cfg(std::move(c)), impl_(std::make_unique<Impl>(*this, cfg)) {
        impl_->setup();
    }

    Sim::~Sim() {
        munmap(impl_->region, static_cast<size_t>(impl_->n + 1) * kL1Bytes);
        g = nullptr;
    }

    void Sim::run() {
        g = impl_.get();
        wall_clock[0] = 0;
        wall_clock[1] = 0;
        impl_->run();
    }

    void Sim::freeze(uint32_t core, Role role, uint64_t cycles) {
        for (auto& t : impl_->threads) {
            if (t->kernel && t->core == core && t->role == role) {
                t->frozen_until = impl_->clock + cycles;
            }
        }
    }

    void Sim::set_stop(uint32_t core) {
        *reinterpret_cast<uint32_t*>(
            impl_->cores[core].mem + kCtlOff + KM_CTL_STOP
        ) = 1;
        impl_->log(core, kEvStop, 0, ~0u);
    }

    uint64_t Sim::now() const {
        return impl_->clock;
    }

    uint32_t Sim::word(uint32_t core, uint32_t off) const {
        uint32_t v;
        std::memcpy(&v, impl_->cores[core].mem + kCtlOff + off, 4);
        return v;
    }

    uint32_t Sim::mailbox_words() const {
        uint32_t n = 0;
        for (const auto& c : impl_->cores) {
            for (const auto& row : c.mbox) {
                for (const auto& q : row) {
                    n += static_cast<uint32_t>(q.size());
                }
            }
        }
        return n;
    }

    uint32_t Sim::popped(uint32_t core, uint32_t cb) const {
        return impl_->cores[core].cb[cb & 31].acked;
    }

    uint32_t Sim::state(uint32_t core, uint32_t index) const {
        return word(core, KM_CTL_STATE + 4 * index);
    }

    // ---- the calls behind the stub headers ----

    void point(const char* name, uint32_t cb) {
        self->last = name;
        self->last_cb = cb;
        if (!g->cfg.triggers.empty()) {
            const Probe p = {self->core, self->role, name, cb, g->clock};
            for (auto& t : g->cfg.triggers) {
                t(p);
            }
        }
        while (g->clock < self->frozen_until) {
            g->yield();
        }
    }

    void yield() {
        g->yield();
    }

    uint32_t arg(uint32_t i) {
        CoreState& c = g->me();
        const auto& a = self->role >= kUnpack ? c.compute_args : c.args;
        if (i >= a.size()) {
            g->violation("argument " + std::to_string(i) + " was not given");
            return 0;
        }
        return a[i];
    }

    uint32_t compile_arg(uint32_t i) {
        return i == 0 ? kTileBytes : 0;
    }

    namespace {

        Cb& cb_of(uint32_t id) {
            Cb& b = g->me().cb[id & 31];
            if (id >= 32 || b.pages == 0) {
                g->violation(
                    "circular buffer " + std::to_string(id) + " is not configured"
                );
            }
            return b;
        }

        uint32_t first_word(uint32_t addr) {
            uint32_t v;
            std::memcpy(&v, g->host(addr), 4);
            return v;
        }

    }  // namespace

    bool cb_room(uint32_t cb, uint32_t pages) {
        const Cb& b = cb_of(cb);
        return b.pages - (b.received - b.acked) >= pages;
    }

    bool cb_ready(uint32_t cb, uint32_t pages) {
        const Cb& b = cb_of(cb);
        return b.received - b.acked >= pages;
    }

    namespace {

        // The record the calling kernel works on (0: none yet).
        uint32_t record_of() {
            const CoreState& c = g->me();
            switch (self->role) {
                case kReader:
                    return c.reader_seq;
                case kWriter:
                    return c.writer_seq;
                default:
                    return c.compute_seq;
            }
        }

        void cb_violation(uint32_t cb, bool producer, const std::string& what) {
            const bool compute = self->role >= kUnpack;
            g->violation(
                "buffer " + std::to_string(cb) + ", " +
                (compute ? "compute" : "data-movement") +
                (producer ? " producer" : " consumer") + ", record " +
                std::to_string(record_of()) + ": " + what
            );
        }

        // Compute wraps at >= (llk_io_pack.h, llk_io_unpack.h: llk_push/pop_tiles),
        // data movement only at == (dataflow_api.h cb_push_back, cb_pop_front).
        void advance(uint32_t cb, Cb& b, uint32_t& ptr, uint32_t pages, bool producer) {
            if (ptr + pages > b.pages) {
                cb_violation(
                    cb,
                    producer,
                    "item of " + std::to_string(pages) + " pages at page " +
                        std::to_string(ptr) + " passes the end of " +
                        std::to_string(b.pages)
                );
            }
            ptr += pages;
            if (self->role >= kUnpack ? ptr >= b.pages : ptr == b.pages) {
                ptr -= b.pages;
            } else if (ptr > b.pages) {
                cb_violation(
                    cb,
                    producer,
                    "pointer past the limit at page " + std::to_string(ptr)
                );
            }
        }

    }  // namespace

    void note_record(uint32_t seq) {
        CoreState& c = g->me();
        (self->role == kReader       ? c.reader_seq
             : self->role == kWriter ? c.writer_seq
                                     : c.compute_seq) = seq;
    }

    void cb_reserve(uint32_t cb, uint32_t pages) {
        while (!cb_room(cb, pages)) {
            g->yield();
        }
        Cb& b = cb_of(cb);
        if (b.wr + pages > b.pages) {
            cb_violation(
                cb,
                true,
                "reservation of " + std::to_string(pages) + " pages at page " +
                    std::to_string(b.wr) + " passes the end of " +
                    std::to_string(b.pages)
            );
        }
        b.reserved = pages;
    }

    void cb_push(uint32_t cb, uint32_t pages) {
        Cb& b = cb_of(cb);
        if (pages > b.reserved) {
            cb_violation(cb, true, "push of pages not reserved");
        }
        b.reserved = pages > b.reserved ? 0 : b.reserved - pages;
        if (cb == km::kCbTokenCompute && self->role == kReader) {
            g->me().reader_seq = first_word(b.base + b.wr * b.page);
            g->log(self->core, kEvTake, g->me().reader_seq, 0);
        }
        b.received += pages;
        *cb_received_word(cb) = b.received;
        uint32_t wr = b.wr;
        advance(cb, b, wr, pages, true);
        b.wr = wr;
    }

    void cb_wait(uint32_t cb, uint32_t pages) {
        while (!cb_ready(cb, pages)) {
            g->yield();
        }
        const Cb& b = cb_of(cb);
        if (b.rd + pages > b.pages) {
            cb_violation(
                cb,
                false,
                "wait for " + std::to_string(pages) + " pages at page " +
                    std::to_string(b.rd) + " passes the end of " +
                    std::to_string(b.pages)
            );
        }
    }

    void cb_pop(uint32_t cb, uint32_t pages) {
        Cb& b = cb_of(cb);
        if (b.received - b.acked < pages) {
            cb_violation(cb, false, "pop of pages not ready");
            return;
        }
        if (cb == km::kCbTokenWriter && self->role == kWriter) {
            g->me().writer_seq = first_word(b.base + b.rd * b.page);
            g->log(self->core, kEvWriterTake, g->me().writer_seq, 0);
        }
        if (cb == km::kCbTokenCompute && self->role == kUnpack) {
            g->me().compute_seq = first_word(b.base + b.rd * b.page);
        }
        b.acked += pages;
        *cb_acked_word(cb) = b.acked;
        uint32_t rd = b.rd;
        advance(cb, b, rd, pages, false);
        b.rd = rd;
    }

    uint32_t cb_write_ptr(uint32_t cb) {
        const Cb& b = cb_of(cb);
        return b.base + b.wr * b.page;
    }

    uint32_t cb_read_ptr(uint32_t cb) {
        const Cb& b = cb_of(cb);
        return b.base + b.rd * b.page;
    }

    CbView cb_view(uint32_t cb) {
        const Cb& b = cb_of(cb);
        return {
            b.pages,
            b.page,
            b.received,
            b.acked,
            b.base + b.pages * b.page,
            b.base + b.wr * b.page,
            b.base + b.rd * b.page
        };
    }

    volatile uint32_t* cb_received_word(uint32_t cb) {
        return reinterpret_cast<uint32_t*>(g->me().mem + kCbRegs + (cb & 31) * 8);
    }

    volatile uint32_t* cb_acked_word(uint32_t cb) {
        return reinterpret_cast<uint32_t*>(g->me().mem + kCbRegs + (cb & 31) * 8 + 4);
    }

    uint32_t cb_page(uint32_t cb) {
        return cb_of(cb).page;
    }

    uint64_t noc_core(uint32_t x, uint32_t y, uint32_t addr) {
        return (
            static_cast<uint64_t>(0x10000u | (y & 0xFF) << 8 | (x & 0xFF)) << 32 | addr
        );
    }

    uint64_t noc_dram(uint32_t addr) {
        return static_cast<uint64_t>(0xFFFFFu) << 32 | addr;
    }

    namespace {

        // Receiver and offset of a NoC address; false (and a violation) if it is none.
        bool target(uint64_t a, uint32_t bytes, uint32_t& core, uint32_t& off) {
            const uint32_t tag = static_cast<uint32_t>(a >> 32),
                           addr = static_cast<uint32_t>(a);
            if (tag == 0xFFFFFu) {
                core = kDram;
                off = addr;
                if (static_cast<uint64_t>(addr) + bytes > kDramBytes) {
                    g->violation("DRAM address out of range");
                    return false;
                }
                return true;
            }
            core = g->core_at(tag & 0xFF, (tag >> 8) & 0xFF);
            off = g->image_off(addr);
            if (core == ~0u || off == ~0u || off + bytes > kL1Bytes) {
                char what[96];
                std::snprintf(
                    what,
                    sizeof what,
                    "NoC address 0x%llx names no L1 word",
                    static_cast<unsigned long long>(a)
                );
                g->violation(what);
                return false;
            }
            return true;
        }

        // The caller's own L1: an address inside its image.
        bool own(uint32_t addr, uint32_t bytes) {
            const CoreState& c = g->me();
            if (addr < c.base || addr + bytes > c.base + kL1Bytes) {
                g->violation("local address outside this core's L1");
                return false;
            }
            return true;
        }

    }  // namespace

    void noc_read(uint64_t src, uint32_t dst, uint32_t bytes) {
        uint32_t core, off;
        if (!target(src, bytes, core, off) || !own(dst, bytes)) {
            return;
        }
        const uint8_t* from = (
            core == kDram ? g->dram.data() + off : g->cores[core].mem + off
        );
        std::memcpy(g->host(dst), from, bytes);
    }

    void noc_write(uint32_t src, uint64_t dst, uint32_t bytes) {
        uint32_t core, off;
        if (bytes == 0 || !target(dst, bytes, core, off) || !own(src, bytes)) {
            return;
        }
        uint32_t kind = 0, seq = 0;
        if (self->role == kWriter && core == g->n && off >= kDispAck &&
            off < kDispAck + 16 * KM_RING_SLOTS) {
            kind = 2;
            seq = first_word(src);
            g->log(self->core, kEvAck, seq, (off - kDispAck) / 16);
        }
        g->send(self->core, self, core, off, g->host(src), bytes, -1, kind, seq);
    }

    void noc_inc(uint64_t dst, uint32_t value) {
        uint32_t core, off;
        if (!target(dst, 4, core, off)) {
            return;
        }
        Pending q = {};
        const Chunk c = {self->core, core, off, 4, 0, 1, true, -1};
        const int64_t d = g->cfg.delay ? g->cfg.delay(c) : 32 + (g->rng() & 63);
        uint32_t seq = 0;
        if (self->role == kWriter) {
            seq = g->me().writer_seq;
            g->log(self->core, kEvReport, seq, core);
        }
        if (d < 0) {
            return;
        }
        q.due = g->clock + static_cast<uint64_t>(d);
        q.ord = g->ord++;
        q.src = self->core;
        q.dst = core;
        q.off = off;
        q.bytes = 4;
        q.atomic = true;
        q.value = value;
        q.from = self;
        q.kind = self->role == kWriter ? 1 : 0;
        q.seq = seq;
        self->atomics++;
        g->pending.push(q);
    }

    void noc_read_barrier() {
    }

    void noc_write_barrier() {
        while (self->writes != 0) {
            g->yield();
        }
    }

    void noc_writes_flushed() {
    }

    void noc_atomic_barrier() {
        while (self->atomics != 0) {
            g->yield();
        }
    }

    namespace {

        uint32_t thread_id() {
            return self->role - kUnpack + 1;
        }

    }  // namespace

    void mailbox_write(uint32_t to, uint32_t value) {
        auto& q = g->me().mbox[thread_id() & 3][to & 3];
        while (q.size() >= g->cfg.mailbox_depth) {
            g->yield();
        }
        q.push_back(value);
        g->sim.mailbox_peak = (
            std::max(g->sim.mailbox_peak, static_cast<uint32_t>(q.size()))
        );
    }

    uint32_t mailbox_read(uint32_t from) {
        auto& q = g->me().mbox[from & 3][thread_id() & 3];
        while (q.empty()) {
            g->yield();
        }
        const uint32_t v = q.front();
        q.pop_front();
        return v;
    }

    void dst_acquire() {
        while (g->me().dst_state != 0) {
            g->yield();
        }
        g->me().dst_state = 1;
    }

    void dst_commit() {
        if (g->me().dst_state != 1) {
            g->violation("tile_regs_commit without tile_regs_acquire");
        }
        g->me().dst_state = 2;
    }

    void dst_wait() {
        while (g->me().dst_state != 2) {
            g->yield();
        }
        g->me().dst_state = 3;
    }

    void dst_release() {
        if (g->me().dst_state != 3) {
            g->violation("tile_regs_release without tile_regs_wait");
        }
        g->me().dst_state = 0;
    }

    void unpack_tile(uint32_t cb, uint32_t tile) {
        if (!cb_ready(cb, tile + 1)) {
            g->violation(
                "copy_tile of a tile not at the front of buffer " + std::to_string(cb)
            );
        }
        const uint16_t* t = (
            reinterpret_cast<const uint16_t*>(
                g->host(cb_read_ptr(cb) + tile * cb_page(cb))
            )
        );
        std::vector<float> v(kTileBytes / 2);
        for (uint32_t e = 0; e < v.size(); e++) {
            v[e] = from_bf16(t[e]);
        }
        while (g->me().src.size() >= 2) {
            g->yield();
        }
        g->me().src.push_back(std::move(v));
    }

    void math_take(uint32_t dst) {
        while (g->me().src.empty()) {
            g->yield();
        }
        g->me().dst[dst & 15] = std::move(g->me().src.front());
        g->me().src.pop_front();
    }

    void math_add(uint32_t a, uint32_t b, uint32_t out) {
        auto& d = g->me().dst;
        for (size_t e = 0; e < d[out & 15].size(); e++) {
            d[out & 15][e] = d[a & 15][e] + d[b & 15][e];
        }
    }

    void math_fill(uint32_t dst, float value) {
        for (auto& e : g->me().dst[dst & 15]) {
            e = value;
        }
    }

    void pack_tile(uint32_t dst, uint32_t cb) {
        const Cb& b = cb_of(cb);
        if (b.reserved == 0) {
            g->violation(
                "pack_tile into buffer " + std::to_string(cb) + " without a reservation"
            );
        }
        uint16_t* t = reinterpret_cast<uint16_t*>(g->host(cb_write_ptr(cb)));
        const auto& d = g->me().dst[dst & 15];
        for (size_t e = 0; e < d.size(); e++) {
            t[e] = to_bf16(d[e]);
        }
    }

}  // namespace kmsim
