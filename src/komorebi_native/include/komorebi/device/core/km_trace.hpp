// Per-command trace: kTraceWords wall-clock words per sequence number in L1 (runtime
// buffer kCbTrace), timestamped by the reader and the writer, written to DRAM at exit.
#pragma once

#include <stdint.h>

#include "km_core.hpp"

namespace km {

    class Trace {
    public:
        explicit Trace(const Args& args) :
            dram_(args.trace),
            capacity_(args.trace_cap),
            base_(args.trace ? get_write_ptr(kCbTrace) : 0),
            state_(args.control + KM_CONTROL_STATE) {
        }

        bool on() const {
            return dram_ != 0;
        }

        // Clears the L1 commands (reader, at start-up, before any timestamp).
        void clear() const {
            if (on()) {
                volatile tt_l1_ptr uint32_t* words = l1(base_);
                for (uint32_t word = 0; word < capacity_ * kTraceWords; word++) {
                    words[word] = 0;
                }
            }
        }

        // A sequence past the capacity is dropped; kStTraceDropped counts the commands
        // (the writer's kTraceDone timestamp, so one kernel alone writes the count).
        void put(uint32_t sequence, uint32_t field, uint32_t value) const {
            if (on() && sequence - 1 < capacity_) {
                l1(base_)[(sequence - 1) * kTraceWords + field] = value;
            } else if (on() && field == kTraceDone) {
                add_sat(l1(state_) + kStTraceDropped, 1);
            }
        }

        void timestamp(uint32_t sequence, uint32_t field) const {
            put(sequence, field, wall());
        }

        uint32_t base() const {
            return base_;
        }

        // Writes this core's commands to page `page` of the trace (writer, at exit),
        // then waits through `core`'s timed write barrier.
        template <typename Writer>
        void flush(uint32_t page, const Writer& core) const {
            if (!on()) {
                return;
            }
            invalidate_l1_cache();
            const uint32_t bytes = capacity_ * kTraceWords * 4;
            const InterleavedAddrGen<true> destination = {
                .bank_base_address = dram_, .page_size = bytes
            };
            noc_async_write(base_, destination.get_noc_addr(page), bytes);
            core.flushed(kFlushWrites, kStWriter);
        }

    private:
        uint32_t dram_;
        uint32_t capacity_;
        uint32_t base_;
        uint32_t state_;
    };

}  // namespace km
