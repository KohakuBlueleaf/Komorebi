// Linear layer Y = X x W for islands, by its tensors' placements (split along K, along
// N, or copied): words, buffers, ports.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_placement.hpp"

// LINEAR, reference words a0 X ([M, K] tiles), a1 W ([K, N]), a2 Y ([M, N]), a3 S (a
// copy: partials, below), a6 T (0: none; a 64 B row a rank); a4 K, a5 N tiles.
constexpr uint32_t kHandlerLinear = 1;

// Built for the input dim sharded form's addition (LINEAR_ADD: two islands, Y a copy):
// partials to S, the other's gathered here (km::gather_pages), added in fp32, to Y.
#if defined(LINEAR_ADD)
#define LINEAR_ADDS true
#else
#define LINEAR_ADDS false
#endif
// The other island's partial of this rank's blocks, counted here (km::keeper).
constexpr uint32_t kPortAdd = 7;
// Built with W's read spread over the column's ranks (LINEAR_SPREAD, grid plan): each
// rank of the column broadcast reads its K rows of a block, the sender casts it.
#if defined(LINEAR_SPREAD)
#define LINEAR_SPREADS true
#else
#define LINEAR_SPREADS false
#endif
constexpr uint32_t kCbWSlice = 13;  // bf16, a receiver's K rows of a W block
constexpr uint32_t kPortReady = 0;  // a receiver: the sender's slot reserved (a marker)
constexpr uint32_t kPortSlices = 1;  // the sender: the receivers' rows landed
// Built with X's read spread over the row's ranks (LINEAR_XSPREAD, grid plan): each
// rank of the row broadcast reads its share of a block's tiles, the sender casts it.
#if defined(LINEAR_XSPREAD)
#define LINEAR_XSPREADS true
#else
#define LINEAR_XSPREADS false
#endif
constexpr uint32_t kCbXSlice = 14;    // bf16, a receiver's tiles of an X block
constexpr uint32_t kPortXReady = 2;   // as kPortReady, the row broadcast's
constexpr uint32_t kPortXSlices = 3;  // as kPortSlices, the row broadcast's
// Built for the output dim sharded form into a copy of Y (LINEAR_GATHER, K-split
// plan): each island writes its columns, then every rank's km::all_gather.
#if defined(LINEAR_GATHER)
#define LINEAR_GATHERS true
#else
#define LINEAR_GATHERS false
#endif

// S's set of island `island`'s partial in a command of `parity` (two, so that a
// command's partial never lands on the one the other island still reads).
inline uint32_t add_set(uint32_t parity, uint32_t island) {
    return parity * km::islands() + island;
}

// Buffers: the grid plan's (row and column broadcasts, bf16 output subblocks, fp32
// partial sums), then the partials' subblocks added (LINEAR_ADD), a token, T's row.
constexpr uint32_t kCbRow = 0;
constexpr uint32_t kCbCol = 1;
constexpr uint32_t kCbYOut = 16;
constexpr uint32_t kCbPart = 24;
constexpr uint32_t kCbSent = 5;  // bf16, a subblock of a partial (LINEAR_ADD)
constexpr uint32_t kCbToken = 6;
constexpr uint32_t kCbArrive = 7;
constexpr uint32_t kCbStamp = 8;  // the writer's row of T
constexpr uint32_t kBf16Bytes = 2048;
constexpr uint32_t kPartBytes = 4096;  // fp32
constexpr uint32_t kSignalBytes = 32;
constexpr uint32_t kStampBytes = 128;
// T's words: the command taken, the crossing begun (the far partial or the gather),
// the crossing done, the command ended (the wall clock, AICLK cycles).
constexpr uint32_t kStampStart = 0;
constexpr uint32_t kStampPut = 1;
constexpr uint32_t kStampSynced = 2;
constexpr uint32_t kStampEnd = 3;
// Inside the crossing: the far partial landed (Up and Down the same).
constexpr uint32_t kStampSettled = 4;
constexpr uint32_t kStampUp = 5;
constexpr uint32_t kStampDown = 6;
// The account (LINEAR_ACCOUNT): AICLK cycles of each CPU's phases a command, written at
// its end: the movers' frees, reads, casts and writes; compute's waits, math and pack.
constexpr uint32_t kAcWFree = 16;
constexpr uint32_t kAcWRead = 17;
constexpr uint32_t kAcWCast = 18;
constexpr uint32_t kAcWTiles = 19;
constexpr uint32_t kAcYWait = 20;
constexpr uint32_t kAcYWrite = 21;
constexpr uint32_t kAcXFree = 22;
constexpr uint32_t kAcXRead = 23;
constexpr uint32_t kAcXCast = 24;
constexpr uint32_t kAcReaderEnd = 25;
constexpr uint32_t kAcInputWait = 26;
constexpr uint32_t kAcMath = 27;
constexpr uint32_t kAcPack = 28;
constexpr uint32_t kAcProducts = 29;
constexpr uint32_t kAcComputeEnd = 30;
constexpr uint32_t kAcComputeStart = 31;
#if defined(LINEAR_ACCOUNT)
#define LINEAR_ACCOUNT_DO(statement) statement
#else
#define LINEAR_ACCOUNT_DO(statement)
#endif
// A phase's cycles into `counter`: from `since` to now, `since` then now.
inline void lap(uint32_t& counter, uint32_t& since) {
    const uint32_t now = km::wall();
    counter += now - since;
    since = now;
}

#if defined(COMPILE_FOR_TRISC)
// Compute's view of T's row (the unpacker's and packer's threads): each sets up only
// its own end of a buffer, the unpacker the read pointer, the packer the write pointer.
inline volatile uint32_t* stamp_row() {
    const LocalCBInterface& row = get_local_cb_interface(kCbStamp);
#if defined(TRISC_UNPACK)
    return reinterpret_cast<volatile uint32_t*>(row.fifo_rd_ptr << 4);
#else
    return reinterpret_cast<volatile uint32_t*>(row.fifo_wr_ptr << 4);
#endif
}
#endif

#if !defined(COMPILE_FOR_TRISC)

// Waits until port `port` of this core has counted `wanted` arrivals, then clears
// it. False: gave up (recorded in state word `state`).
inline bool counted(
    const km::Core& mover,
    uint32_t state,
    const km_command_t& command,
    uint32_t port,
    uint32_t wanted
) {
    volatile tt_l1_ptr uint32_t* words = km::port_words(mover, port);
    words[KM_PORT_WANT] = wanted;
    const uint32_t code = mover.poll_at(
        km::kWherePort,
        [&] {
            invalidate_l1_cache();
            return static_cast<int32_t>(words[KM_PORT_COUNT] - wanted) >= 0;
        },
        mover.args().wait_limit
    );
    if (code != km::kGaveNone) {
        mover.gave_up(
            state, code, km::kWherePort, km::slot(command) | port << 8, command.sequence
        );
        return false;
    }
    words[KM_PORT_COUNT] = 0;
    words[KM_PORT_WANT] = 0;
    return true;
}

// Waits until word COUNT of port `port` of this core holds `marker`. False: gave
// up (recorded in state word `state`).
inline bool marked(
    const km::Core& mover,
    uint32_t state,
    const km_command_t& command,
    uint32_t port,
    uint32_t marker
) {
    volatile tt_l1_ptr uint32_t* words = km::port_words(mover, port);
    const uint32_t code = mover.poll_at(
        km::kWherePort,
        [&] {
            invalidate_l1_cache();
            return words[KM_PORT_COUNT] == marker;
        },
        mover.args().wait_limit
    );
    if (code != km::kGaveNone) {
        mover.gave_up(
            state, code, km::kWherePort, km::slot(command) | port << 8, command.sequence
        );
        return false;
    }
    return true;
}

// This CPU's writes have landed (acknowledged), so other cores may read them. False:
// gave up (recorded in state word `state`).
inline bool landed(const km::Core& mover, uint32_t state, const km_command_t& command) {
    const uint32_t code = mover.poll_at(
        km::kWhereData,
        [] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        },
        mover.args().wait_limit
    );
    if (code != km::kGaveNone) {
        mover.gave_up(state, code, km::kWhereData, km::slot(command), command.sequence);
        return false;
    }
    return true;
}

#endif

// The K-split plan (LINEAR_SPLIT, few rows of X): groups of S ranks, a block a group,
// each rank 1/S of K on both movers, the parts summed into the group's first rank.
constexpr uint32_t kCbKa = 2;        // the reader's X chunks: oh x kc tiles
constexpr uint32_t kCbKw = 3;        // the reader's W chunks: kc x ow tiles
constexpr uint32_t kCbKaB = 9;       // the writer's X chunks
constexpr uint32_t kCbKwB = 10;      // the writer's W chunks
constexpr uint32_t kCbKArrive = 11;  // fp32, a slot of oh x ow tiles a sender
constexpr uint32_t kCbKSum = 25;     // fp32, this rank's part: oh x ow tiles
constexpr uint32_t kCbKOut = 18;     // bf16, the group's sum at its first rank
constexpr uint32_t kCbKAdd = 12;     // bf16, its block of both partials (LINEAR_ADD)
constexpr uint32_t kCbKX = 15;       // bf16, the whole-K plan's X: all its K chunks
// Its words (every CPU's): S, oh, ow, kc, subblock sh x sw, blocks a row of the window,
// arrival pages, reduce form, M tiles, a set of S's pages, the chunks' pattern word.
constexpr uint32_t kKsS = 0;
constexpr uint32_t kKsOh = 1;
constexpr uint32_t kKsOw = 2;
constexpr uint32_t kKsKc = 3;
constexpr uint32_t kKsSh = 4;
constexpr uint32_t kKsSw = 5;
constexpr uint32_t kKsNbw = 6;
constexpr uint32_t kKsArrive = 7;
constexpr uint32_t kKsForm = 8;
constexpr uint32_t kKsMt = 9;
constexpr uint32_t kKsSetPages = 10;
constexpr uint32_t kKsPattern = 11;
// The whole-K plan (S 1, Kohaku-TT's decode form): X multicast once a panel by the
// panel's sender (its writer, on its NoC) into every rank's kCbKX, a marker after each.
constexpr uint32_t kKsXCast = 12;    // 1: this plan
constexpr uint32_t kKsXRole = 13;    // this core's: 0 none, 1 the sender, 2 a receiver
constexpr uint32_t kKsXWaits = 14;   // sender: the receivers whose free it counts
constexpr uint32_t kKsXDests = 15;   // sender: the multicast's cores but itself
constexpr uint32_t kKsXSender = 16;  // receiver: its sender's core
constexpr uint32_t kKsXStart = 17;   // the cast's corners (writer's NoC), x | y << 16
constexpr uint32_t kKsXEnd = 18;
constexpr uint32_t kKsXChunks = 19;  // K chunks a multicast
constexpr uint32_t kKsWords = 20;
constexpr uint32_t kKXSender = 1;
constexpr uint32_t kKXReceiver = 2;
// Ports (the split plan's grid ports): a receiver's X buffer free, X's marker
constexpr uint32_t kPortKXFree = 2;
constexpr uint32_t kPortKXGot = 3;
// T's words of the K-split plan (with the account): the movers' chunks moved (reader,
// writer), the reduce (writer: its part sent, or the sum taken), Y written.
constexpr uint32_t kAcKMoveA = 16;
constexpr uint32_t kAcKMoveB = 17;
constexpr uint32_t kAcKReduce = 18;
constexpr uint32_t kAcKWrite = 19;
constexpr uint32_t kAcKReaderReduce = 20;
// wall clocks: compute's own part packed, each round's part taken and added; the
// writer's part to send in hand
constexpr uint32_t kAcKOwn = 21;
constexpr uint32_t kAcKTaken = 22;  // rounds 0 to kKRounds - 1
constexpr uint32_t kAcKAdded = 26;  // the same
constexpr uint32_t kAcKHeld = 30;
constexpr uint32_t kKRounds = 4;
constexpr uint32_t kBroadcastRow = 0;
constexpr uint32_t kBroadcastCol = 1;

// Ports: the writer's go to its own reader (LINEAR_ADD: both partials are in S).
constexpr uint32_t kPortGo = 5;

// Arguments as the grid form's: reader, writer: broadcast words, then core words;
// compute: core words, then the two broadcast windows.
constexpr uint32_t kArgBroadcast = 0;
constexpr uint32_t kArgCore = km::kBroadcastArgs;
constexpr uint32_t kCoreRank = 0;
constexpr uint32_t kCoreM0 = 1;
constexpr uint32_t kCoreN0 = 2;
constexpr uint32_t kCoreSent = 3;      // its first page in a set of S
constexpr uint32_t kCoreSetPages = 4;  // the pages of a set
constexpr uint32_t kCorePlan = 5;
constexpr uint32_t kPlMt = 0;
constexpr uint32_t kPlKt = 1;
constexpr uint32_t kPlNt = 2;
constexpr uint32_t kPlPcM = 3;
constexpr uint32_t kPlPcN = 4;
constexpr uint32_t kPlBm = 5;
constexpr uint32_t kPlBn = 6;
constexpr uint32_t kPlKb = 7;
constexpr uint32_t kPlNbh = 10;
constexpr uint32_t kPlNbw = 11;
constexpr uint32_t kPlSh = 12;
constexpr uint32_t kPlSw = 13;
constexpr uint32_t kPlanWords = 14;
constexpr uint32_t kCoreWords = kCorePlan + kPlanWords;

// What island `at` holds of a tensor (its reference word, `pages` pages): its part's
// bank address, steps first .. of `all` steps of `inner` pages, in `blocks` blocks.
struct Holding {
    uint32_t address;
    uint32_t first;
    uint32_t steps;
    uint32_t all;
    uint32_t inner;
    uint32_t pages;  // the part's

    // Page `index` of the part in the whole tensor, and back.
    uint32_t page(uint32_t index) const {
        const uint32_t block = index / (steps * inner), rest = index % (steps * inner);
        return (block * all + first) * inner + rest;
    }

    uint32_t index(uint32_t page) const {
        const uint32_t block = page / (all * inner), step = page / inner % all;
        return (block * steps + step - first) * inner + page % inner;
    }

    // Whether pages page .. page + count - 1 are all of the part, so at consecutive
    // indices from index(page).
    bool holds(uint32_t page, uint32_t count) const {
        const uint32_t last = page + count - 1, span = all * inner;
        return (
            page / span == last / span && page / inner % all - first < steps &&
            last / inner % all - first < steps
        );
    }
};

[[gnu::noinline]] inline Holding holding(uint32_t tensor, uint32_t at, uint32_t pages) {
    const km::Islands view(km::island_area());
    const volatile uint32_t* entry = km::tensor_entry(view, tensor);
    if (entry == nullptr) {  // a bank address: one island holds every page
        return {tensor, 0, 1, 1, pages, pages};
    }
    const volatile uint32_t*
        part = (entry + KM_TENSOR_HEADER_BYTES / 4 + at * km::kPartWords);
    if (entry[3] & 1u) {  // a copy: every island all of it
        return {part[0], 0, 1, 1, pages, pages};
    }
    const uint32_t all = entry[1], inner = entry[2];
    return {
        part[0], part[1], part[2], all, inner, pages / (all * inner) * part[2] * inner
    };
}

// Tiles first .. first + count - 1 of a dimension.
struct Span {
    uint32_t first;
    uint32_t count;
};

// The K tiles of W island `at` holds: whole rows of W ([K, N] tiles, split along K),
// or all of it (all of it too for its columns of a W split along N).
inline Span k_part(uint32_t w, uint32_t at, uint32_t k_whole, uint32_t n_whole) {
    const Holding part = holding(w, at, k_whole * n_whole);
    if (part.pages == k_whole * n_whole) {
        return {0, k_whole};
    }
    if (part.pages != 0 && part.all == k_whole && part.inner == n_whole) {
        return {part.first, part.steps};
    }
    if (part.pages != 0 && part.all == n_whole && part.inner == 1) {
        return {0, k_whole};
    }
    return {0, 0};
}

// The columns of W island `at` holds: its columns of a W split along N, else all.
inline Span w_columns(uint32_t w, uint32_t at, uint32_t k_whole, uint32_t n_whole) {
    const Holding part = holding(w, at, k_whole * n_whole);
    if (
        part.pages != 0 && part.pages != k_whole * n_whole && part.all == n_whole &&
        part.inner == 1
    ) {
        return {part.first, part.steps};
    }
    return {0, n_whole};
}

// The columns of Y ([M, N] tiles: split along N, along M, or all of it) island `at`
// keeps of the rows it keeps.
inline Span columns(uint32_t y, uint32_t at, uint32_t m_tiles, uint32_t n_whole) {
    const Holding part = holding(y, at, m_tiles * n_whole);
    if (part.pages == m_tiles * n_whole) {
        return {0, n_whole};
    }
    if (part.pages != 0 && part.all == n_whole && part.inner == 1) {
        return {part.first, part.steps};
    }
    if (part.pages != 0 && part.all == m_tiles && part.inner == n_whole) {
        return {0, n_whole};
    }
    return {0, 0};
}

// The rows of Y island `at` keeps: its whole rows when Y is split along M, else all
// (none where it holds nothing).
inline Span kept_rows(uint32_t y, uint32_t at, uint32_t m_tiles, uint32_t n_whole) {
    const Holding part = holding(y, at, m_tiles * n_whole);
    if (part.pages == 0) {
        return {0, 0};
    }
    if (
        part.pages != m_tiles * n_whole && part.all == m_tiles && part.inner == n_whole
    ) {
        return {part.first, part.steps};
    }
    return {0, m_tiles};
}

// The rows of the product island `at` makes: the rows of X ([M, K] tiles) it holds
// when X is split along M, else every row (its part of K for all of them).
inline Span made_rows(uint32_t x, uint32_t at, uint32_t m_tiles, uint32_t k_whole) {
    const Holding part = holding(x, at, m_tiles * k_whole);
    if (
        part.pages != 0 && part.pages != m_tiles * k_whole && part.all == m_tiles &&
        part.inner == k_whole
    ) {
        return {part.first, part.steps};
    }
    return {0, part.pages != 0 ? m_tiles : 0};
}

// Whether tensor `tensor` (its reference word) is a copy on every island.
inline bool copied(uint32_t tensor) {
    const km::Islands view(km::island_area());
    const volatile uint32_t* entry = km::tensor_entry(view, tensor);
    return entry != nullptr && (entry[3] & 1u) != 0;
}

// The columns island `to` keeps of the rows `from` makes (all of them kept, W's columns
// held by `from`); of a Y copied on every island, its own only (its partial).
[[gnu::noinline]] inline Span window(
    uint32_t x,
    uint32_t w,
    uint32_t y,
    uint32_t from,
    uint32_t to,
    uint32_t m,
    uint32_t k,
    uint32_t n
) {
    const Span made = made_rows(x, from, m, k), kept = kept_rows(y, to, m, n);
    if (
        (to != from && copied(y)) || made.count == 0 || made.first < kept.first ||
        made.first + made.count > kept.first + kept.count
    ) {
        return {0, 0};
    }
    const Span kept_columns = columns(y, to, m, n), held = w_columns(w, from, k, n);
    if (
        kept_columns.first < held.first ||
        kept_columns.first + kept_columns.count > held.first + held.count
    ) {
        // a copy of Y and W split along N: the island's own columns into its copy
        return copied(y) ? held : Span{0, 0};
    }
    return kept_columns;
}

// The island `turn` places after island `here` (1: the next one).
inline uint32_t island_at(uint32_t here, uint32_t turn) {
    return (here + turn) % km::islands();
}

inline uint32_t blocks_of(uint32_t numerator, uint32_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

// S: a set an island and command parity (add_set), in a set every core's output
// blocks in the order it makes them (its first page a core word), tiles as packed.
struct Sent {
    uint32_t set_pages;
    uint32_t first;

    uint32_t page(uint32_t set, uint32_t tile) const {
        return set * set_pages + first + tile;
    }
};
