// Timed waits for the ray tracer's kernels: each gives up after a wall-clock timeout or
// once the stop flag is set, and the caller records it and leaves the command.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_worker.hpp"

namespace timed {

#if defined(COMPILE_FOR_TRISC)
    constexpr uint32_t kCycles = 2700000000u;  // 2 s at 1350 MHz; a page wait is < 1 ms
    constexpr uint32_t kAbort = 0xFFFFFFFF;    // a mailbox value: the sender gave up

    inline uint32_t clock() {
        return *reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
    }

    template <typename Ready>
    bool poll(Ready ready) {
        const volatile uint32_t* stop = (
            reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(0) + KM_CTL_STOP)
        );
        const uint32_t t0 = clock();
        for (uint32_t i = 0;; ++i) {
            if (ready()) {
                return true;
            }
            if ((i & 63) == 63) {
                asm volatile("fence" ::: "memory");
                if (*stop != 0 || clock() - t0 > kCycles) {
                    return false;
                }
            }
        }
    }

    // Unpack: cb_wait_front once the pages are there.
    inline bool front(uint32_t cb, uint32_t pages) {
        volatile uint32_t* got = get_cb_tiles_received_ptr(cb);
        const bool ok = poll([&] {
            const uint32_t r = (
                ckernel::reg_read(
                    static_cast<uint32_t>(reinterpret_cast<uintptr_t>(got))
                )
            );
            const uint16_t n = (
                static_cast<uint16_t>(r - get_local_cb_interface(cb).tiles_acked)
            );
            return n >= pages;
        });
        if (ok) {
            cb_wait_front(cb, pages);
        }
        return ok;
    }

    // Pack: cb_reserve_back once the space is there.
    inline bool back(uint32_t cb, uint32_t pages) {
        volatile uint32_t* acked = get_cb_tiles_acked_ptr(cb);
        const bool ok = poll([&] {
            const uint32_t a = (
                ckernel::reg_read(
                    static_cast<uint32_t>(reinterpret_cast<uintptr_t>(acked))
                )
            );
            const uint16_t used = (
                static_cast<uint16_t>(get_local_cb_interface(cb).tiles_received - a)
            );
            return get_local_cb_interface(cb).fifo_num_pages - used >= pages;
        });
        if (ok) {
            cb_reserve_back(cb, pages);
        }
        return ok;
    }

    // A mailbox value from thread `from`; false when none came in time.
    inline bool mail(uint8_t from, uint32_t& v) {
        if (!poll([&] {
            return ckernel::mailbox_not_empty(from);
        })) {
            return false;
        }
        v = ckernel::mailbox_read(from);
        return true;
    }

    // Bytes of a buffer from its write pointer to its end (compute's 16-byte units).
    inline uint32_t room(uint32_t cb) {
        const auto& i = get_local_cb_interface(cb);
        return (i.fifo_limit - i.fifo_wr_ptr) << 4;
    }
#else
    // NoC reads submitted so far have landed (data-movement kernels). False: gave up,
    // recorded in the state word pair `word`.
    inline bool reads_done(const km::Core& c, uint32_t word) {
        const uint32_t code = c.poll(
            [] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            c.args().wait_limit
        );
        if (code != km::kGaveNone) {
            c.gave_up(word, code, km::kWhereData, 0, 0);
            return false;
        }
        noc_async_read_barrier();
        return true;
    }

    // NoC writes submitted so far have left the core.
    inline bool writes_sent(const km::Core& c, uint32_t word) {
        const uint32_t code = c.poll(
            [] {
                return ncrisc_noc_nonposted_writes_sent(noc_index);
            },
            c.args().wait_limit
        );
        if (code != km::kGaveNone) {
            c.gave_up(word, code, km::kWhereData, 0, 0);
            return false;
        }
        noc_async_writes_flushed();
        return true;
    }

    // NoC writes submitted so far have landed (acknowledged): the next command may read
    // them.
    inline bool writes_landed(const km::Core& c, uint32_t word) {
        const uint32_t code = c.poll(
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            c.args().wait_limit
        );
        if (code != km::kGaveNone) {
            c.gave_up(word, code, km::kWhereData, 0, 0);
            return false;
        }
        noc_async_write_barrier();
        return true;
    }

    // The buffer holds `bytes` contiguous from its write pointer.
    inline bool fits(uint32_t cb, uint32_t bytes) {
        const auto& i = get_local_cb_interface(cb);
        return i.fifo_wr_ptr + bytes <= i.fifo_limit;
    }
#endif

}  // namespace timed
