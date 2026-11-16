// Far order, reader: in one command a rank puts a page into another island's DRAM and
// posts an add to a far rank, which reads that page the moment its count rises.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_far.hpp"
#include "komorebi/device/km_worker.hpp"
#include "order.h"

// The wall clock's ticks since `last`, added to `elapsed`; true while under `limit`.
inline bool under(uint32_t& last, uint64_t& elapsed, uint64_t limit) {
    const uint32_t now = km::wall();
    elapsed += now - last;
    last = now;
    return elapsed < limit;
}

// The sender (order.h PAIR): once the watcher runs, each round a put of its page, a
// posted add to the watcher's counter, then the put waited for.
template <typename Args>
void send(km::Reader& reader, const km_command_t& command, const Args& args) {
    const uint32_t watcher = command.a0, rounds = command.a1;
    const uint32_t tensor = command.a2, pages = command.a3;
    const uint64_t limit = uint64_t{command.a5} * kTicksAMillisecond;
    const uint32_t source = get_write_ptr(kCbPage);
    const uint32_t bytes = TensorAccessor(args, 0).get_aligned_page_size();
    const uint32_t counter = get_write_ptr(kCbArea) + kAreaCounter * 4;
    volatile tt_l1_ptr uint32_t* words = km::l1(source);
    volatile tt_l1_ptr uint32_t* own = km::l1(get_write_ptr(kCbArea));
    uint32_t last = km::wall();
    uint64_t elapsed = 0;
    bool ready = false;
    while (!ready && under(last, elapsed, limit)) {
        invalidate_l1_cache();
        ready = own[kAreaReady] != 0;
    }
    uint32_t sent = 0;
    own[kAreaGaveUp] = ready ? 0 : 2;
    for (uint32_t round = 0; ready && round < rounds; round++) {
        for (uint32_t w = 0; w < bytes / 4; w++) {
            words[w] = order_word(round, w);
        }
        const km::PageAt at = km::page_at(tensor, round % pages);
        if (!km::put_pages(
            reader, command, at.island, at.address, bytes, at.index, 1, source
        ) ||
            !km::add_word(reader, command, watcher, counter, 1, km::posted) ||
            !km::put_wait(reader, command, 1, at.island)) {
            own[kAreaGaveUp] = 1;
            break;
        }
        sent++;
    }
    own[kAreaDone] = sent;
}

// The watcher (order.h PAIR): tells the sender it runs, then at each rise of its
// counter reads the page of the round that add follows and checks every word.
template <typename Args>
void watch(km::Reader& reader, const km_command_t& command, const Args& args) {
    const uint32_t sender = command.a4, rounds = command.a1;
    const uint32_t tensor = command.a2, pages = command.a3;
    const uint64_t limit = uint64_t{command.a5} * kTicksAMillisecond;
    const uint32_t buffer = get_write_ptr(kCbPage);
    const uint32_t bytes = TensorAccessor(args, 0).get_aligned_page_size();
    volatile tt_l1_ptr uint32_t* own = km::l1(get_write_ptr(kCbArea));
    const volatile tt_l1_ptr uint32_t* got = km::l1(buffer);
    const uint32_t ready = get_write_ptr(kCbArea) + kAreaReady * 4;
    if (!km::add_word(reader, command, sender, ready, 1)) {
        own[kAreaGaveUp] = 1;
        return;
    }
    uint32_t seen = 0, rises = 0, checked = 0, early = 0, stale = 0;
    uint32_t last = km::wall();
    uint64_t elapsed = 0;
    while (seen < rounds && under(last, elapsed, limit)) {
        invalidate_l1_cache();
        const uint32_t now = own[kAreaCounter];
        if (now == seen) {
            continue;
        }
        seen = now;
        rises++;
        const uint32_t round = now - 1;
        if (!km::read_pages(reader, command, args, tensor, round % pages, 1, buffer)) {
            own[kAreaGaveUp] = 1;
            break;
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (own[kAreaCounter] - round >= pages - 1) {
            stale++;  // the sender may have begun round + pages' put of this page
            continue;
        }
        checked++;
        for (uint32_t w = 0; w < bytes / 4; w++) {
            if (got[w] != order_word(round, w)) {
                if (early == 0) {
                    own[kAreaFirst] = round;
                    own[kAreaFirst + 1] = got[0];
                }
                early++;
                break;
            }
        }
    }
    if (seen < rounds && own[kAreaGaveUp] == 0) {
        own[kAreaGaveUp] = 2;
    }
    own[kAreaDone] = checked;
    own[kAreaEarly] = early;
    own[kAreaStale] = stale;
    own[kAreaSeen] = rises;
}

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();  // the data's and reports' (one shape)
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        const uint32_t area = get_write_ptr(kCbArea);
        volatile tt_l1_ptr uint32_t* words = km::l1(area);
        if (km::handler(command) == kHandlerClear) {
            for (uint32_t w = 0; w < kAreaBytes / 4; w++) {
                words[w] = 0;
            }
            return;
        }
        if (km::handler(command) == kHandlerPair) {
            const uint32_t me = reader.me();  // the flat rank
            if (me == command.a4) {
                send(reader, command, args);
            } else if (me == command.a0) {
                watch(reader, command, args);
            }
            return;
        }
        if (km::handler(command) == kHandlerReport) {
            invalidate_l1_cache();
            km::write_pages(reader, command, args, command.a0, command.a1, 1, area);
            noc_async_write_barrier();
        }
    });
}
