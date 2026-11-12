// Per-record trace: kTraceWords wall-clock words per sequence number in L1 (runtime
// buffer kCbTrace), stamped by the reader and the writer, written to DRAM at exit.
#pragma once

#include <stdint.h>

#include "km_core.hpp"

namespace km {

    class Trace {
    public:
        explicit Trace(const Args& a)
            : dram_(a.trace),
              cap_(a.trace_cap),
              base_(a.trace ? get_write_ptr(kCbTrace) : 0) {
        }

        bool on() const {
            return dram_ != 0;
        }

        // Clears the L1 records (reader, at start-up, before any stamp).
        void clear() const {
            if (on()) {
                volatile tt_l1_ptr uint32_t* t = l1(base_);
                for (uint32_t w = 0; w < cap_ * kTraceWords; w++) {
                    t[w] = 0;
                }
            }
        }

        void put(uint32_t seq, uint32_t field, uint32_t value) const {
            if (on() && seq - 1 < cap_) {
                l1(base_)[(seq - 1) * kTraceWords + field] = value;
            }
        }

        void stamp(uint32_t seq, uint32_t field) const {
            put(seq, field, wall());
        }

        // Writes this core's records to page `page` of the trace (writer, at exit).
        void flush(uint32_t page) const {
            if (!on()) {
                return;
            }
            invalidate_l1_cache();
            const uint32_t bytes = cap_ * kTraceWords * 4;
            const InterleavedAddrGen<true> dst = {
                .bank_base_address = dram_, .page_size = bytes
            };
            noc_async_write(base_, get_noc_addr(page, dst), bytes);
            noc_async_write_barrier();
        }

    private:
        uint32_t dram_;
        uint32_t cap_;
        uint32_t base_;
    };

}  // namespace km
