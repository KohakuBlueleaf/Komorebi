// The simulator's state shared by its two translation units: the scheduler, L1 images,
// circular buffers, NoC delivery, compute threads and the dispatcher model.
#pragma once

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

    inline const char* role_name(uint32_t role) {
        static const char* names[] = {"reader", "writer", "unpack", "math", "pack"};
        return role < kRoles ? names[role] : "?";
    }

    namespace detail {

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
            uint32_t ret = ~0u;  // atomic: image offset of the value before the add

            bool operator>(const Pending& o) const {
                return due != o.due ? due > o.due : ord > o.ord;
            }
        };

        inline thread_local Thread* self = nullptr;

        inline uint16_t to_bf16(float f) {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            return static_cast<uint16_t>(u >> 16);
        }

        inline float from_bf16(uint16_t h) {
            const uint32_t u = static_cast<uint32_t>(h) << 16;
            float f;
            std::memcpy(&f, &u, 4);
            return f;
        }

        // Input element e of tile i of job k: small integers, exact in bfloat16 sums.
        inline float input_value(uint32_t k, uint32_t i, uint32_t e) {
            return static_cast<float>((k + i + e) % 4);
        }

    }  // namespace detail

    using namespace detail;

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
                    (in(kDispAck, KM_RING_SLOTS, 8) || in(kDispReady, 16, 4) ||
                     (q.off == kDispMailbox && q.bytes <= 8) ||
                     (in(kDispLines, n, 16) && q.bytes == 16))
                );
            }
            if (q.off < kCtlOff) {
                return false;
            }
            if (q.atomic) {
                return (
                    q.src < n &&
                    ((at >= KM_CTL_DONE && at < KM_CTL_DONE + 16 * 16 &&
                      (at - KM_CTL_DONE) % 16 == 0) ||
                     at == KM_CTL_DYN + KM_DYN_IN_TAIL)
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
                    if (q.ret != ~0u) {  // the response carries the old value
                        std::memcpy(cores[q.src].mem + q.ret, &v, 4);
                    }
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

        void prepare_map(Job& job, uint32_t k, uint32_t j);

        uint32_t check_map(const Job& job, uint32_t k);

        uint32_t issue(const Job& job, int32_t k, uint32_t j);

        bool booting();

        // True when a KM_F_EXCL record's team overlaps a team in flight, as the
        // dispatch kernel holds it.
        bool held(
            const std::vector<Job>& jobs, size_t k, const int32_t* busy, uint32_t d
        );

        // A dynamic run in flight (dyn_dispatch.cpp): job k was issued with sequence
        // seq; the end record is appended to `jobs` once the status lines are quiet.
        int32_t dyn_k = -1;
        uint32_t dyn_seq = 0;
        bool dyn_ended = true;
        void dyn_issued(const Job& job, size_t k, uint32_t seq);
        void dyn_watch(std::vector<Job>& jobs);

        void dispatcher();

        void dispatcher_thread(Thread* t) {
            self = t;
            t->sem.pass();
            dispatcher();
            dispatcher_alive = false;
            finish();
        }

        // ---- set-up and run ----

        void setup();

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

    inline Sim::Impl* g = nullptr;

}  // namespace kmsim
