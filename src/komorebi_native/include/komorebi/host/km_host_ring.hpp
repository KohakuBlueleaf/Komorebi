// The host's side of a dispatcher's host ring and event ring (x280/kernel/README.md):
// both dispatchers' ports; each says how a line and the read line reach its ring.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <immintrin.h>

#include "km_dispatch.h"
#include "komorebi/abi/km_abi.h"

namespace km {

    class HostRing {
    public:
        using Clock = std::chrono::steady_clock;

        virtual ~HostRing() = default;

        // Stores the read count and a fence: an engine's posts wait for event room.
        void publish() {
            require_open();
            if (seen_ != published_) {
                settle();
                store_read();
                _mm_sfence();
                fenced_ = Clock::now();
            }
        }

        uint32_t host_word(uint32_t word) const {
            require_open();
            if (word >= KMD_HOST_RING / 4) {
                throw std::runtime_error("no such host word");
            }
            return host_[word];
        }

        // Waits until the kernel reports READY or a final state; returns that state, or
        // 0 after the timeout.
        uint32_t wait_ready(double timeout_s) const {
            require_open();
            const auto deadline = after(timeout_s);
            for (;;) {
                const uint32_t state = host_[KMD_HOST_STATE];
                if (state >= KMD_STATE_READY) {
                    return state;
                }
                if (Clock::now() > deadline) {
                    return 0;
                }
            }
        }

        // Commands that fit now: free host ring slots, and room in the event ring, so
        // that unread events plus commands in flight never exceed it. Counters wrap.
        uint32_t room() const {
            require_open();
            const uint32_t taken = host_[KMD_HOST_CONSUMED];
            const int64_t ring = (
                static_cast<int64_t>(host_depth_) -
                static_cast<int32_t>(next_ - 1u - taken)
            );
            const int64_t events = (
                static_cast<int64_t>(event_depth_) -
                static_cast<int32_t>(next_ - 1u - seen_)
            );
            return static_cast<uint32_t>(std::max<int64_t>(0, std::min(ring, events)));
        }

        // Stores the read count line (KMD_LIM_READ: events the host has read), without
        // a fence. The kernel takes a command only while its events fit the ring.
        void store_read() {
            alignas(64) uint32_t line[16] = {seen_};
            store_read_line(line);
            published_ = seen_;
        }

        // While waiting: the read count and a fence, only when the kernel may hold a
        // command for it (command i waits only while i - 1 - published reaches depth).
        void publish_if_held() {
            if (
                seen_ != published_ && next_ - 1u - published_ >= event_depth_ &&
                Clock::now() >= settled()
            ) {
                store_read();
                _mm_sfence();
                fenced_ = Clock::now();
            }
        }

        // Spins until settle_microseconds_ passed since the last fence: a store that
        // follows a fence too closely often stalls about 12 us.
        void settle() const {
            const auto until = settled();
            while (Clock::now() < until) {
                _mm_pause();
            }
        }

        Clock::time_point settled() const {
            return (
                fenced_ +
                std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double, std::micro>(settle_microseconds_)
                )
            );
        }

        // Stores the first k commands that fit as commands next .. next + k - 1 (number
        // in sequence and the landing words), the read count, one fence. Returns k.
        uint32_t push(const uint32_t* commands, size_t count) {
            const uint32_t stored = (
                static_cast<uint32_t>(std::min<size_t>(count, room()))
            );
            store(commands, stored);
            return stored;
        }

        // Queues n commands for wait(), stored in order as the ring has room (the
        // kernel orders KM_FLAG_EXCLUSIVE ones). Returns the number queued.
        size_t enqueue(const uint32_t* commands, size_t count) {
            require_open();
            queue_.insert(queue_.end(), commands, commands + 16 * count);
            return queued();
        }

        size_t queued() const {
            return queue_.size() / 16 - head_;
        }

        // Stores queued commands up to `upto` and reads events until all `wanted` have
        // one, a non-DONE event or a final kernel state arrives, or timeout_s passes.
        std::vector<uint32_t> wait(
            const std::vector<uint32_t>& wanted, uint32_t upto, double timeout_s
        ) {
            require_open();
            // wanted is in command order: offsets from its first fit one table.
            const uint32_t first_wanted = wanted.empty() ? 0 : wanted.front();
            std::vector<uint8_t> need(
                wanted.empty() ? 0 : wanted.back() - first_wanted + 1u, 0
            );
            size_t left = 0;
            for (const uint32_t command : wanted) {
                left += need[command - first_wanted] == 0;
                need[command - first_wanted] = 1;
            }
            const auto deadline = after(timeout_s);
            std::vector<uint32_t> out;
            for (;;) {
                store_queued(upto);
                const size_t had = out.size();
                read_events(out);
                publish_if_held();
                bool other = false;
                for (size_t event = had; event < out.size(); event += 4) {
                    const uint32_t offset = out[event + 1] - first_wanted;
                    if (offset < need.size() && need[offset] != 0) {
                        need[offset] = 0;
                        left--;
                    }
                    other = other || (out[event + 2] & 0xFFu) != KM_EVENT_DONE;
                }
                if (
                    left == 0 || other || host_[KMD_HOST_STATE] >= KMD_STATE_DONE ||
                    Clock::now() > deadline
                ) {
                    return out;
                }
                if (out.size() == had) {
                    _mm_pause();
                }
            }
        }

        // The events that have arrived since the last call, in order.
        std::vector<uint32_t> poll() {
            require_open();
            std::vector<uint32_t> out;
            read_events(out);
            return out;
        }

        // Appends the events that have arrived to out, in order.
        void read_events(std::vector<uint32_t>& out) {
            const auto* ring = (
                reinterpret_cast<const volatile uint32_t*>(
                    reinterpret_cast<const volatile uint8_t*>(host_) + KMD_HOST_RING
                )
            );
            for (;;) {
                // A sequence lock: index, fields, index again; a change means the
                // event was being written, so it is read again.
                const uint32_t number = seen_ + 1;
                const volatile uint32_t* line = ring + (number % event_depth_) * 4;
                if (line[0] != number) {
                    return;
                }
                _mm_lfence();
                const uint32_t request = line[1], status = line[2], ticks = line[3];
                _mm_lfence();
                if (line[0] != number) {
                    continue;
                }
                out.insert(out.end(), {number, request, status, ticks});
                seen_ = number;
            }
        }

        // poll() until at least one event arrives, the kernel reaches a final state, or
        // timeout_s passes.
        std::vector<uint32_t> wait_events(double timeout_s) {
            const auto deadline = after(timeout_s);
            for (;;) {
                auto out = poll();
                publish_if_held();
                if (
                    !out.empty() || host_[KMD_HOST_STATE] >= KMD_STATE_DONE ||
                    Clock::now() > deadline
                ) {
                    auto rest = poll();
                    out.insert(out.end(), rest.begin(), rest.end());
                    return out;
                }
                _mm_pause();
            }
        }

        // The batch store_queued(upto) takes from the queue: those up to upto that fit.
        uint32_t batch_upto(uint32_t upto) const {
            const int64_t due = static_cast<int32_t>(upto - next_) + int64_t{1};
            size_t count = std::min<size_t>({queued(), batch_, room()});
            count = (
                std::min<int64_t>(
                    static_cast<int64_t>(count), std::max<int64_t>(0, due)
                )
            );
            return static_cast<uint32_t>(count);
        }

        uint32_t batch_ = 64;  // commands per store: every line, then one fence
        uint32_t host_depth_ = 1, event_depth_ = 1;
        uint32_t next_ = 1;                 // number of the next command
        uint32_t seen_ = 0;                 // events read from the ring
        uint32_t published_ = 0;            // the read count last stored
        double settle_microseconds_ = 1.0;  // pause between a fence and the next store
        Clock::time_point fenced_{};

    protected:
        static Clock::time_point after(double seconds) {
            return (
                Clock::now() +
                std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(seconds)
                )
            );
        }

        // Throws when the port is closed.
        virtual void require_open() const = 0;
        // The lines of k commands as commands next .. next + k - 1, no fence.
        virtual void store_lines(const uint32_t* commands, uint32_t count) const = 0;
        // The read count's 64-byte line into the ring's read line, no fence.
        virtual void store_read_line(const uint32_t* line) = 0;

        // A new run: numbering from 1, nothing read, nothing queued.
        void restart_ring(uint32_t host_depth, uint32_t event_depth) {
            host_depth_ = host_depth;
            event_depth_ = event_depth;
            next_ = 1;
            seen_ = published_ = 0;
            queue_.clear();
            head_ = 0;
        }

        // Stores k commands as the next commands, every line then one fence.
        void store(const uint32_t* commands, uint32_t count) {
            if (count == 0) {
                return;
            }
            settle();
            store_lines(commands, count);
            store_read();
            _mm_sfence();
            fenced_ = Clock::now();
            next_ += count;
        }

        // One batch from the queue: the commands up to upto that fit. An exclusive one
        // is stored as any other; the kernel's KM_FLAG_EXCLUSIVE check orders it.
        void store_queued(uint32_t upto) {
            const uint32_t count = batch_upto(upto);
            store(queue_.data() + 16 * head_, count);
            taken(count);
        }

        // `count` commands left the queue's head.
        void taken(size_t count) {
            head_ += count;
            if (head_ == queue_.size() / 16) {
                queue_.clear();
                head_ = 0;
            } else if (head_ > 4096 && 2 * head_ > queue_.size() / 16) {
                queue_.erase(queue_.begin(), queue_.begin() + 16 * head_);
                head_ = 0;
            }
        }

        volatile uint32_t* host_ = nullptr;  // the host words, then the event ring
        std::vector<uint32_t> queue_;        // commands not stored yet, 16 words each
        size_t head_ = 0;                    // the first command not stored
    };

}  // namespace km
