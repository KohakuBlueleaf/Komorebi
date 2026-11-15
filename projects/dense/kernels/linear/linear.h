// Linear layer Y = X x W for islands, by its tensors' placements (split along K, along
// N, or copied): words, buffers, ports.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_placement.hpp"

// LINEAR, reference words a0 X ([M, K] tiles), a1 W ([K, N]), a2 Y ([M, N]), a3 S (a
// copy: tiles put here, below), a6 T (0: none; a 64 B row a rank); a4 K, a5 N tiles.
constexpr uint32_t kHandlerLinear = 1;
// Built for a plan that exchanges nothing (LINEAR_OWN): each island its own window
// only, no puts and no sync in the build.
#if defined(LINEAR_OWN)
#define LINEAR_EXCHANGES false
#else
#define LINEAR_EXCHANGES true
#endif

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
// partial sums), then a block of the columns put across, the subblocks put here.
constexpr uint32_t kCbRow = 0;
constexpr uint32_t kCbCol = 1;
constexpr uint32_t kCbYOut = 16;
constexpr uint32_t kCbPart = 24;
constexpr uint32_t kCbSend = 17;  // bf16, two chunks of the tiles put (put_chunk)
constexpr uint32_t kCbSent = 5;   // bf16, an output subblock from one other island
constexpr uint32_t kCbToken = 6;
constexpr uint32_t kCbArrive = 7;
constexpr uint32_t kCbStamp = 8;  // the writer's row of T
constexpr uint32_t kBf16Bytes = 2048;
constexpr uint32_t kPartBytes = 4096;  // fp32
constexpr uint32_t kSignalBytes = 32;
constexpr uint32_t kStampBytes = 128;
// A put's tiles at least (64 KiB): a chunk is the smallest whole number of subblocks
// at or over it.
constexpr uint32_t kPutTiles = 32;
// T's words: the command taken, the other islands' columns put (sent), the sync done,
// its own columns written (the wall clock, AICLK cycles).
constexpr uint32_t kStampStart = 0;
constexpr uint32_t kStampPut = 1;
constexpr uint32_t kStampSynced = 2;
constexpr uint32_t kStampEnd = 3;
// Inside the sync: the last put counted, the island's tree up, the leaders' crossing
// (the leader) or the tree's word down.
constexpr uint32_t kStampSettled = 4;
constexpr uint32_t kStampUp = 5;
constexpr uint32_t kStampDown = 6;
// The puts' cycles a command: waiting for tiles, in put_pages, in put_piece, its
// pieces' slot waits and writes (PutRun turn, write); and the pieces.
constexpr uint32_t kStampFed = 7;
constexpr uint32_t kStampLine = 8;
constexpr uint32_t kStampPiece = 9;
constexpr uint32_t kStampTurn = 10;
constexpr uint32_t kStampWrite = 11;
constexpr uint32_t kStampPieces = 12;
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
// Its words (every CPU's, from 0): S, oh, ow, kc, subblock sh x sw, blocks a row of the
// window, the arrival buffer's pages, the reduce form, M tiles, a set of S's pages.
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
constexpr uint32_t kKsWords = 11;
// T's words of the K-split plan (with the account): the movers' chunks moved (reader,
// writer), the reduce (writer: its part sent, or the sum taken), Y written.
constexpr uint32_t kAcKMoveA = 16;
constexpr uint32_t kAcKMoveB = 17;
constexpr uint32_t kAcKReduce = 18;
constexpr uint32_t kAcKWrite = 19;
constexpr uint32_t kAcKReaderReduce = 20;
constexpr uint32_t kBroadcastRow = 0;
constexpr uint32_t kBroadcastCol = 1;

// Ports: the island's tree up and down, the leaders' crossing, the writer's go to its
// own reader (the writer's puts count at its page port, km::pages_port).
constexpr uint32_t kPortUp = 2;
constexpr uint32_t kPortCross = 3;
constexpr uint32_t kPortDown = 4;
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

// The islands in the order an island computes their columns: the next one first, its
// own last (`turn` 1 .. islands).
inline uint32_t island_at(uint32_t here, uint32_t turn) {
    return (here + turn) % km::islands();
}

// Whether any island puts to another: one holds part of K and another keeps columns
// of the rows it makes. The same answer on every island (no sync when false).
[[gnu::noinline]] inline bool exchanging(
    uint32_t x, uint32_t w, uint32_t y, uint32_t m, uint32_t k, uint32_t n
) {
    for (uint32_t from = 0; from < km::islands(); ++from) {
        if (k_part(w, from, k, n).count == 0) {
            continue;
        }
        for (uint32_t to = 0; to < km::islands(); ++to) {
            if (to != from && window(x, w, y, from, to, m, k, n).count != 0) {
                return true;
            }
        }
    }
    return false;
}

// The other islands that put to island `here` (each a K block of a block: k blocks
// 0 ..): those holding part of K that make rows `here` keeps columns of.
[[gnu::noinline]] inline uint32_t senders_to(
    uint32_t x,
    uint32_t w,
    uint32_t y,
    uint32_t here,
    uint32_t m,
    uint32_t k,
    uint32_t n
) {
    uint32_t senders = 0;
    for (uint32_t turn = 1; turn < km::islands(); ++turn) {
        const uint32_t from = island_at(here, turn);
        senders += (
            k_part(w, from, k, n).count != 0 &&
            window(x, w, y, from, here, m, k, n).count != 0
        );
    }
    return senders;
}

inline uint32_t blocks_of(uint32_t numerator, uint32_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

// S: a set a sending island (sent_set), in a set every core's output blocks in the
// order it makes them (its first page a core word), tiles as packed.
struct Sent {
    uint32_t set_pages;
    uint32_t first;

    uint32_t page(uint32_t set, uint32_t tile) const {
        return set * set_pages + first + tile;
    }
};

inline uint32_t put_chunk(uint32_t subblock_tiles) {
    return blocks_of(kPutTiles, subblock_tiles) * subblock_tiles;
}

inline uint32_t sent_set(uint32_t from, uint32_t to) {
    return (from + km::islands() - to - 1) % km::islands();
}
