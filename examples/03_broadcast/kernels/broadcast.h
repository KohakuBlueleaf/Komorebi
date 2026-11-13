// Broadcast example: handlers, buffers and arguments its kernels share, and the two
// ways a member gets its blocks (broadcast broadcast, or its own DRAM reads).
#pragma once

#include <stdint.h>

#include "komorebi/device/km_broadcast.hpp"

// CAST: a0 A (rows), a1 W (columns), a2 output, a3 blocks n, a4 member that stalls at
// row block a5 (a test; kNone: none), a6 output page of member 0, a7 data blocks d.
constexpr uint32_t kHandlerBroadcast = 1;
// READ: as CAST, but every member reads both of its blocks from DRAM itself.
constexpr uint32_t kHandlerRead = 2;

constexpr uint32_t kCbRow = 0;   // row broadcast buffer: B blocks of P tiles of A
constexpr uint32_t kCbCol = 1;   // column broadcast buffer: the same for W
constexpr uint32_t kCbOut = 16;  // one fp32 tile
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kNone = 0xFFFFFFFFu;

// Program arguments of reader and writer: the broadcast's words, then the core's row
// (or column) in the rectangle. Compute: member (1 inside the rectangle), B, P.
constexpr uint32_t kArgBroadcast = 0;
constexpr uint32_t kArgIndex = km::kBroadcastArgs;

#if !defined(COMPILE_FOR_TRISC)

// Bounded wait for this kernel's reads; gives up into state word `word`.
inline bool reads_landed(const km::Core& c, uint32_t word, const km_command_t& r) {
    const uint32_t code = c.poll(
        [] {
            return ncrisc_noc_reads_flushed(noc_index);
        },
        c.args().wait_limit,
        c.args().wait_limit
    );
    if (code != km::kGaveNone) {
        c.gave_up(word, code, km::kWhereData, km::slot(r), r.seq);
    }
    return code == km::kGaveNone;
}

// Blocks 0 to n - 1: the sender reads block k (tiles first + k % d * P onwards) into
// its slot and broadcasts it; a receiver waits for it. Stops at block `stall`.
template <typename Src, typename Stall>
bool broadcast_blocks(
    km::Broadcast& ln,
    const Src& src,
    uint32_t first,
    uint32_t n,
    uint32_t d,
    Stall stall
) {
    const uint32_t p = ln.pages();
    for (uint32_t k = 0; k < n; k++) {
        if (stall(k)) {
            return false;
        }
        if (!ln.sender()) {
            if (!ln.arrive(k)) {
                return false;
            }
            continue;
        }
        const uint32_t addr = ln.reserve(k);
        if (addr == 0) {
            return false;
        }
        for (uint32_t i = 0; i < p; i++) {
            noc_async_read_page(first + k % d * p + i, src, addr + i * kTileBytes);
        }
        if (!ln.cast(k)) {
            return false;
        }
    }
    return ln.finish();
}

// READ: the member reads its blocks itself into the same buffer, then pads it to
// whole windows as Broadcast::finish does (compute runs the same code for both).
template <typename Src>
bool read_blocks(
    const km::Core& c,
    uint32_t word,
    const km_command_t& r,
    const km::Broadcast& ln,
    uint32_t cb,
    const Src& src,
    uint32_t first,
    uint32_t n,
    uint32_t d
) {
    const uint32_t p = ln.pages(), w = ln.window();
    const uint32_t pad = (w - n % w) % w * p;
    for (uint32_t k = 0; k <= n; k++) {
        const uint32_t pages = k < n ? p : pad;
        if (pages == 0) {
            break;
        }
        const uint32_t code = c.poll(
            [&] {
                return cb_pages_reservable_at_back(cb, pages);
            },
            c.args().wait_limit
        );
        if (code != km::kGaveNone || !km::item_fits(cb, get_write_ptr(cb), pages)) {
            const uint32_t why = code != km::kGaveNone ? code : km::kGaveBadItem;
            c.gave_up(word, why, km::kWhereData, km::slot(r), r.seq);
            return false;
        }
        cb_reserve_back(cb, pages);
        const uint32_t dst = get_write_ptr(cb);
        for (uint32_t i = 0; k < n && i < p; i++) {
            noc_async_read_page(first + k % d * p + i, src, dst + i * kTileBytes);
        }
        if (!reads_landed(c, word, r)) {
            return false;
        }
        cb_push_back(cb, pages);
    }
    return true;
}

#endif
