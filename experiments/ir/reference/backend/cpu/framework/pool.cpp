// Resident workers behind komo_cpu.h, one thread each; komo_invoke waits for its end,
// which is the lowest rank's that did not complete, else completed.
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "komo_cpu.hpp"

namespace {

    using komo_cpu::Status;

    struct Job {
        komo_cpu::Handler handler = nullptr;
        const komo_cpu::Args* args = nullptr;
        komo_cpu::Worker worker{};
        Status* out = nullptr;
    };

    komo_end end_of(uint32_t state, uint32_t failure) {
        return komo_end{state, failure, 0, 0};
    }

    // Whether the arguments are the entry's parameters, kind by kind.
    bool fits(const komo_cpu::EntryPoint& e, const komo_arg* args, uint32_t count) {
        if (count != e.count_params) {
            return false;
        }
        for (uint32_t i = 0; i < count; ++i) {
            const komo_cpu::ArgKind& want = e.params[i];
            if (args[i].kind != want.kind) {
                return false;
            }
            if (
                want.kind == KOMO_ARG_TENSOR &&
                (args[i].tensor == nullptr || args[i].tensor->format != want.number)
            ) {
                return false;
            }
        }
        return true;
    }

}  // namespace

struct komo_pool {
    std::vector<std::thread> threads;
    std::vector<Job> jobs;
    std::vector<bool> posted;
    std::mutex lock;
    std::condition_variable work;
    std::condition_variable done;
    uint32_t left = 0;
    bool closing = false;

    void run(uint32_t id) {
        std::unique_lock<std::mutex> held(lock);
        for (;;) {
            work.wait(held, [&] {
                return closing || posted[id];
            });
            if (closing) {
                return;
            }
            Job job = jobs[id];
            posted[id] = false;
            held.unlock();
            *job.out = job.handler(job.worker, *job.args);
            held.lock();
            left -= 1;
            if (left == 0) {
                done.notify_all();
            }
        }
    }
};

extern "C" komo_pool* komo_open(uint32_t workers) {
    if (workers == 0) {
        return nullptr;
    }
    komo_pool* pool = new komo_pool();
    pool->jobs.resize(workers);
    pool->posted.assign(workers, false);
    for (uint32_t id = 0; id < workers; ++id) {
        pool->threads.emplace_back([pool, id] {
            pool->run(id);
        });
    }
    return pool;
}

extern "C" void komo_close(komo_pool* pool) {
    if (pool == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> held(pool->lock);
        pool->closing = true;
    }
    pool->work.notify_all();
    for (std::thread& t : pool->threads) {
        t.join();
    }
    delete pool;
}

extern "C" int32_t komo_entry(const char* name) {
    for (uint32_t i = 0; i < komo_module.count_entries; ++i) {
        if (std::strcmp(komo_module.entries[i].name, name) == 0) {
            return int32_t(i);
        }
    }
    return -1;
}

extern "C" komo_end komo_invoke(
    komo_pool* pool,
    int32_t entry,
    const komo_arg* args,
    uint32_t count_args,
    uint32_t first,
    uint32_t count
) {
    if (
        pool == nullptr || entry < 0 || uint32_t(entry) >= komo_module.count_entries ||
        count == 0 || first > pool->threads.size() ||
        count > pool->threads.size() - first
    ) {
        return end_of(KOMO_INVALID, 0);
    }
    const komo_cpu::EntryPoint& e = komo_module.entries[entry];
    if (!fits(e, args, count_args)) {
        return end_of(KOMO_INVALID, 0);
    }
    komo_cpu::Args given{args, count_args};
    std::vector<Status> ends(count);
    {
        std::lock_guard<std::mutex> held(pool->lock);
        pool->left = count;
        for (uint32_t rank = 0; rank < count; ++rank) {
            pool->jobs[first + rank] = Job{
                e.handler,
                &given,
                komo_cpu::Worker{rank, count, first + rank},
                &ends[rank]
            };
            pool->posted[first + rank] = true;
        }
    }
    pool->work.notify_all();
    {
        std::unique_lock<std::mutex> held(pool->lock);
        pool->done.wait(held, [&] {
            return pool->left == 0;
        });
    }
    for (const Status& s : ends) {
        if (s.kind == Status::Kind::failed) {
            return end_of(KOMO_FAILED, uint32_t(s.failure));
        }
        if (s.kind == Status::Kind::invalid) {
            return end_of(KOMO_INVALID, 0);
        }
    }
    return end_of(KOMO_COMPLETED, 0);
}
