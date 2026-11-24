// APPLY's buffers as windows over other buffers' memory (APPLY_WINDOWS; README.md
// "APPLY"): each CPU points its own views there at the start of every APPLY command.
#pragma once

#include "apply.h"

#if defined(APPLY_WINDOWS)
// Buffer `cb`'s view on this CPU: `pages` pages from byte `at` of buffer `host` (both
// in this CPU's units: bytes on a mover, 16 bytes on compute, cb_addr_shift).
inline void apply_window(uint32_t cb, uint32_t host, uint32_t at, uint32_t pages) {
    const LocalCBInterface& under = get_local_cb_interface(host);
    LocalCBInterface& view = get_local_cb_interface(cb);
    const uint32_t start = under.fifo_limit - under.fifo_size + (at >> cb_addr_shift);
    view.fifo_size = pages * view.fifo_page_size;
    view.fifo_limit = start + view.fifo_size;
    view.fifo_num_pages = pages;
    view.fifo_rd_ptr = start;
    view.fifo_wr_ptr = start;
    view.fifo_wr_tile_ptr = 0;
}

#define APPLY_WINDOW(name) \
    apply_window(name, APPLY_HOST_##name, APPLY_AT_##name, APPLY_PAGES_##name)

// Every view to its window (apply.py's layout; each buffer is empty then, so its counts
// stay); math keeps no buffer views.
inline void apply_windows() {
#if !defined(TRISC_MATH)
    APPLY_WINDOW(kCbApplyX);
    APPLY_WINDOW(kCbApplyXB);
    APPLY_WINDOW(kCbApplyW);
    APPLY_WINDOW(kCbApplyWB);
    APPLY_WINDOW(kCbApplyY);
    APPLY_WINDOW(kCbApplyProf);
    APPLY_WINDOW(kCbApplyEntries);
    APPLY_WINDOW(kCbApplyEntriesB);
    APPLY_WINDOW(kCbApplyHeadPage);
    APPLY_WINDOW(kCbApplyHeadPageB);
    APPLY_WINDOW(kCbApplySums);
    APPLY_WINDOW(kCbApplyHidden);
    APPLY_WINDOW(kCbApplyReady);
#endif
}
#else
inline void apply_windows() {
}
#endif
