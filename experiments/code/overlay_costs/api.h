// The runtime's calls of a reader loop and of a compute loop, header-only: each wait
// is timed and gives up on the stop word. Inline here; the tables wrap the same calls.
#pragma once

#include <stdint.h>

#include "costs.h"

namespace cs {

    constexpr uint32_t kWaitLimit = 50000000;  // cycles (about 37 ms) for one wait

}  // namespace cs

#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"

namespace cs {

    inline uint32_t wall() {
        return (
            *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                RISCV_DEBUG_REG_WALL_CLOCK_L
            )
        );
    }

    // Timed cb_reserve_back: false on the stop word or after kWaitLimit cycles.
    inline bool reserve(uint32_t cb, uint32_t pages, volatile uint32_t* stop) {
        if (!cb_pages_reservable_at_back(cb, pages)) {
            const uint32_t t0 = wall();
            do {
                invalidate_l1_cache();
                if (*stop != 0 || wall() - t0 > kWaitLimit) {
                    return false;
                }
            } while (!cb_pages_reservable_at_back(cb, pages));
        }
        cb_reserve_back(cb, pages);
        return true;
    }

    // Timed cb_wait_front: false on the stop word or after kWaitLimit cycles.
    inline bool wait_front(uint32_t cb, uint32_t pages, volatile uint32_t* stop) {
        if (!cb_pages_available_at_front(cb, pages)) {
            const uint32_t t0 = wall();
            do {
                invalidate_l1_cache();
                if (*stop != 0 || wall() - t0 > kWaitLimit) {
                    return false;
                }
            } while (!cb_pages_available_at_front(cb, pages));
        }
        cb_wait_front(cb, pages);
        return true;
    }

    // Starts the read of an interleaved DRAM tensor's page (page i in bank i % 8).
    inline void read_page(uint32_t tensor, uint32_t page, uint32_t bytes, uint32_t l1) {
        noc_async_read(
            get_noc_addr_from_bank_id<true>(page & 7u, tensor + (page >> 3) * bytes),
            l1,
            bytes
        );
    }

    // Timed noc_async_read_barrier.
    inline bool read_barrier() {
        const uint32_t t0 = wall();
        while (!ncrisc_noc_reads_flushed(noc_index)) {
            if (wall() - t0 > kWaitLimit) {
                return false;
            }
        }
        return true;
    }

    // The reader loop: `count` pages from `first` (wrapping at 1024) into kCbRead,
    // each read, waited for and pushed. Returns a status.
    template <typename Api>
    inline uint32_t reader_loop(const Call& call, const Api& api) {
        for (uint32_t i = 0; i < call.count; i++) {
            if (!api.reserve(kCbRead, 1, call.stop)) {
                return kWaitTimeout;
            }
            const uint32_t page = (call.arg + i) & 1023u;
            api.read_page(call.tensor, page, call.page_bytes, api.write_ptr(kCbRead));
            if (!api.read_barrier()) {
                return kReadTimeout;
            }
            api.push(kCbRead, 1);
        }
        return kOk;
    }

    // The reader's calls inline, in the shape of a MoverApi.
    struct InlineMover {
        static bool reserve(uint32_t cb, uint32_t pages, volatile uint32_t* stop) {
            return cs::reserve(cb, pages, stop);
        }

        static uint32_t write_ptr(uint32_t cb) {
            return get_write_ptr(cb);
        }

        static void read_page(
            uint32_t tensor, uint32_t page, uint32_t bytes, uint32_t l1
        ) {
            cs::read_page(tensor, page, bytes, l1);
        }

        static bool read_barrier() {
            return cs::read_barrier();
        }

        static void push(uint32_t cb, uint32_t pages) {
            cb_push_back(cb, pages);
        }
    };

}  // namespace cs

#else
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

namespace cs {

    inline uint32_t wall() {
        return ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    }

    // Timed cb_wait_front: unpack polls and decides, math and pack get its answer by
    // mailbox. False on the stop word or after kWaitLimit cycles, on all three.
    inline bool wait_front(uint32_t cb, uint32_t pages, volatile uint32_t* stop) {
        uint32_t ok = 1;
        UNPACK(({
            volatile tt_l1_ptr uint32_t* received = get_cb_tiles_received_ptr(cb);
            const uint32_t t0 = wall();
            for (;;) {
                const uint16_t available = (
                    static_cast<uint16_t>(
                        ckernel::reg_read(reinterpret_cast<uint32_t>(received)) -
                        get_local_cb_interface(cb).tiles_acked
                    )
                );
                if (available >= pages) {
                    break;
                }
                if (*stop != 0 || wall() - t0 > kWaitLimit) {
                    ok = 0;
                    break;
                }
            }
            if (ok) {
                ckernel::cb_wait_front(cb, pages);
            }
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, ok);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, ok);
        }));
        MATH((ok = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
        PACK((ok = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
        return ok != 0;
    }

    // Timed cb_reserve_back, decided by pack: false there on the stop word or after
    // kWaitLimit cycles; true on unpack and math.
    inline bool reserve_back(uint32_t cb, uint32_t pages, volatile uint32_t* stop) {
        bool ok = true;
        PACK(({
            volatile tt_reg_ptr uint32_t* acked = get_cb_tiles_acked_ptr(cb);
            const uint32_t t0 = wall();
            for (;;) {
                const uint16_t used = (
                    static_cast<uint16_t>(
                        get_local_cb_interface(cb).tiles_received -
                        ckernel::reg_read(reinterpret_cast<uint32_t>(acked))
                    )
                );
                if (get_local_cb_interface(cb).fifo_num_pages - used >= pages) {
                    break;
                }
                if (*stop != 0 || wall() - t0 > kWaitLimit) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                ckernel::cb_reserve_back(cb, pages);
            }
        }));
        return ok;
    }

    // The compute loop: per tile, wait for kCbIn's front, the tile into DST, times
    // the scalar (fp32 bits) on the vector unit, packed into kCbOut. Returns a status.
    template <typename Api>
    inline uint32_t compute_loop(const Call& call, const Api& api) {
        api.copy_init(kCbIn);
        api.scalar_init();
        for (uint32_t i = 0; i < call.count; i++) {
            if (!api.wait_front(kCbIn, 1, call.stop)) {
                return kWaitTimeout;
            }
            api.acquire();
            api.copy(kCbIn, 0, 0);
            api.mul(0, call.arg);
            api.commit();
            api.wait();
            if (!api.reserve_back(kCbOut, 1, call.stop)) {
                return kWaitTimeout;
            }
            api.pack(0, kCbOut);
            api.release();
            api.pop(kCbIn, 1);
            api.push(kCbOut, 1);
        }
        return kOk;
    }

    // The compute calls inline, in the shape of a ComputeApi.
    struct Inline {
        static bool wait_front(uint32_t cb, uint32_t pages, volatile uint32_t* stop) {
            return cs::wait_front(cb, pages, stop);
        }

        static bool reserve_back(uint32_t cb, uint32_t pages, volatile uint32_t* stop) {
            return cs::reserve_back(cb, pages, stop);
        }

        static void copy_init(uint32_t cb) {
            ckernel::copy_init(cb);
        }

        static void scalar_init() {
            ckernel::binop_with_scalar_tile_init();
        }

        static void acquire() {
            ckernel::tile_regs_acquire();
        }

        static void copy(uint32_t cb, uint32_t tile, uint32_t dst) {
            ckernel::copy_tile(cb, tile, dst);
        }

        static void mul(uint32_t dst, uint32_t scalar) {
            ckernel::mul_unary_tile(dst, scalar);
        }

        static void commit() {
            ckernel::tile_regs_commit();
        }

        static void wait() {
            ckernel::tile_regs_wait();
        }

        static void pack(uint32_t dst, uint32_t cb) {
            ckernel::pack_tile(dst, cb);
        }

        static void release() {
            ckernel::tile_regs_release();
        }

        static void pop(uint32_t cb, uint32_t pages) {
            ckernel::cb_pop_front(cb, pages);
        }

        static void push(uint32_t cb, uint32_t pages) {
            ckernel::cb_push_back(cb, pages);
        }
    };

}  // namespace cs
#endif
