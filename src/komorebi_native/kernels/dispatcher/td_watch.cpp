// The Tensix dispatcher's watch on TRISC 2 (the X280's watch.h, no interrupt): a loop
// without a beat for 100 ms is a stall that in reports (SEEN, then BACK).
#include <stdint.h>

#include "api/compute/common.h"
#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"

void kernel_main() {
#ifdef TRISC_PACK
    const uint32_t block = get_arg_val<uint32_t>(kTdBlock);
    const uint32_t tag = get_arg_val<uint32_t>(kTdTag);
    const uint64_t ticks_us = get_arg_val<uint32_t>(kTdTicksUs);  // 16.16
    const uint64_t stall_ticks = (kTdStallUs * ticks_us) >> 16;
    const uint64_t look_ticks = (kTdLookUs * ticks_us) >> 16;
    volatile tt_l1_ptr uint32_t* hand = km::l1(block + kTbHand + kTxWords);
    volatile tt_l1_ptr uint32_t* line = km::l1(block + kTbWatch);
    // out's start, or its end before it (a start-up that failed); none within the
    // start-up wait: out did not run (its block not placed)
    const uint64_t boot = (
        static_cast<uint64_t>(get_arg_val<uint32_t>(kTdBootLimit)) << 10
    );
    const uint64_t started = km::wall64();
    for (;;) {
        invalidate_l1_cache();
        if (hand[kTwGo] == tag || hand[kTwEnd] == tag) {
            break;
        }
        if (km::wall64() - started > boot) {
            return;
        }
    }
    line[kTkState] = KMD_WATCH_ON;
    uint32_t beats[2] = {line[kTkOut], line[kTkIn]};
    uint64_t since[2], last = km::wall64();
    since[0] = since[1] = last;
    uint32_t stalled = 0, seen = 0, back = 0, gap = 0, looks = 0;
    uint64_t seen_at = 0;
    while (hand[kTwGo] == tag && hand[kTwEnd] != tag) {
        invalidate_l1_cache();
        const uint64_t now = km::wall64();
        if (now - last < look_ticks) {
            for (uint32_t i = 0; i < 256; i++) {
                asm volatile("nop");
            }
            continue;
        }
        last = now;
        line[kTkLooks] = ++looks;
        const uint32_t loops[2] = {kTkOut, kTkIn};
        uint32_t quiet = 0;
        for (uint32_t loop = 0; loop < 2; loop++) {
            const uint32_t beat = line[loops[loop]];
            const uint64_t still = now - since[loop];
            if (beat != beats[loop]) {
                gap = still > gap ? static_cast<uint32_t>(still) : gap;
                beats[loop] = beat;
                since[loop] = now;
            } else if (still > stall_ticks) {
                quiet |= 1u << loop;
                gap = still > gap ? static_cast<uint32_t>(still) : gap;
            }
        }
        line[kTkGap] = gap;
        if (quiet != 0 && stalled == 0) {  // SEEN: its loops, the gap
            stalled = quiet;
            seen_at = now;
            line[kTkLoops] = quiet;
            line[kTkTicks] = (
                static_cast<uint32_t>(now - (quiet & 1u ? since[0] : since[1]))
            );
            asm volatile("" ::: "memory");
            line[kTkSeen] = ++seen;
        } else if (stalled != 0 && quiet == 0) {  // BACK: the ticks since SEEN
            stalled = 0;
            line[kTkTicks] = static_cast<uint32_t>(now - seen_at);
            asm volatile("" ::: "memory");
            line[kTkBack] = ++back;
        }
    }
    line[kTkState] = KMD_WATCH_OFF;
#endif
}
