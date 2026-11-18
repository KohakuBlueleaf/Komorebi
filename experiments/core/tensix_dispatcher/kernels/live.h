// The dispatcher on one Tensix core, fed live: its arguments, the block in its L1 the
// host writes, the host lines in the channel and the results it writes at the end.
#pragma once

#include <stdint.h>

// Runtime arguments.
constexpr uint32_t kLvControl = 0;     // control block address (the workers' too)
constexpr uint32_t kLvWorkers = 1;     // cores 0 to workers - 1 get every command
constexpr uint32_t kLvCores = 2;       // DRAM address of the core table
constexpr uint32_t kLvBlock = 3;       // L1 address of the live block (below)
constexpr uint32_t kLvHostDepth = 4;   // host ring lines, a power of two, at most 64
constexpr uint32_t kLvEventDepth = 5;  // event ring, a power of two, at most 128
constexpr uint32_t kLvChannel = 6;     // channel offset of the host lines (below)
constexpr uint32_t kLvDepth = 7;       // slots used, 1 to KM_RING_SLOTS
constexpr uint32_t kLvDeadline = 8;    // per command, submit to ack, 1024-cycle units
constexpr uint32_t kLvResults = 9;     // DRAM address of the results, 64-byte pages
constexpr uint32_t kLvGroups = 10;     // start-up groups: READY words to wait for
constexpr uint32_t kLvBootLimit = 11;  // start-up wait, 1024-cycle units
constexpr uint32_t kLvIdleLimit = 12;  // nothing taken nor busy this long: the end
constexpr uint32_t kLvTraceCap = 13;   // commands 1 to cap get a trace row
constexpr uint32_t kLvRects = 14;      // multicast rectangles (0: unicast to each)
// Rectangle r at kLvRect + 3 r: NoC 0 start and end (x | y << 16), receivers.
constexpr uint32_t kLvRect = 15;
constexpr uint32_t kLvRectsMax = 4;
constexpr uint32_t kLvHelper = 27;  // nonzero: the other mover issues rectangle 1
constexpr uint32_t kLvArgs = 28;

// The live block in the dispatcher's L1 (every core has one; only the dispatcher's is
// used). The host writes the ring and the read line; the rest is the kernel's.
constexpr uint32_t kLbRing = 0;        // host_depth lines of 64 bytes
constexpr uint32_t kLbRead = 4096;     // word 0: events the host has read
constexpr uint32_t kLbCounts = 4160;   // staging of the counts line (16 bytes)
constexpr uint32_t kLbEvents = 4224;   // staging of the events, 16 bytes each
constexpr uint32_t kLbPending = 6272;  // the command taken and not yet submitted
constexpr uint32_t kLbHand = 6336;     // the helper's word pair (below)
constexpr uint32_t kLbBytes = 6400;

// The host lines at kLvChannel in the session's region of the host channel.
constexpr uint32_t kLhConsumed = 0;  // word 0: commands taken
constexpr uint32_t kLhTold = 1;      // word 1: events written
constexpr uint32_t kLhEvents = 64;   // byte offset of the event ring

// Results: two summary rows of 16 words, then a trace row of 4 words a command.
constexpr uint32_t kLsMagic = 0;     // kLiveMagic
constexpr uint32_t kLsStatus = 1;    // kLe*
constexpr uint32_t kLsSlot = 2;      // timed-out command: slot
constexpr uint32_t kLsSequence = 3;  // its sequence number
constexpr uint32_t kLsRequest = 4;   // its request (host number)
constexpr uint32_t kLsTaken = 5;     // commands taken from the ring
constexpr uint32_t kLsDone = 6;      // commands acknowledged
constexpr uint32_t kLsBoot = 7;      // start-up wait, cycles / 16
constexpr uint32_t kLsFirstLo = 8;   // wall clock at the first take
constexpr uint32_t kLsFirstHi = 9;
constexpr uint32_t kLsLastLo = 10;  // wall clock at the last event
constexpr uint32_t kLsLastHi = 11;
constexpr uint32_t kLsMaxBusy = 12;    // most commands in flight at once
constexpr uint32_t kLsExitAcked = 13;  // 1: the EXIT command was completed
constexpr uint32_t kLsReady = 14;      // READY words seen at start-up (bit per group)
constexpr uint32_t kLsPushed = 15;     // events written
// Row 1: what each part of the loop took, summed, cycles / 16 (LIVE_PARTS builds).
constexpr uint32_t kLpTake = 16;    // the landing words and the line copied
constexpr uint32_t kLpSlot = 17;    // the checks and the search for a free slot
constexpr uint32_t kLpWrite = 18;   // the command's writes to the workers, flushed
constexpr uint32_t kLpSweep = 19;   // the sweep of the busy slots' ack words
constexpr uint32_t kLpEvent = 20;   // the events' and the counts line's writes
constexpr uint32_t kLpPasses = 21;  // loop passes
constexpr uint32_t kLpEmpty = 22;   // passes that looked at the ring and found no line
constexpr uint32_t kLpHeld = 23;    // passes with a command taken and not submitted
constexpr uint32_t kLpCounts = 24;  // counts lines written
constexpr uint32_t kLpRejected = 25;    // commands rejected
constexpr uint32_t kLpRoom = 26;        // passes held by event room
constexpr uint32_t kLpHelperWait = 27;  // cycles / 16 waiting for the helper
constexpr uint32_t kLpIdle = 28;        // passes' looks at an empty ring, cycles / 16
// Row 2: the helper (V3) and the loop's L1 polls (LIVE_PARTS builds), cycles / 16.
constexpr uint32_t kLpHelperIssue = 32;  // the helper: token seen to its done word
constexpr uint32_t kLpHelperPolls = 33;  // the helper's polls of the word pair
constexpr uint32_t kLpHandOver = 34;   // the loop's token written to the helper's look
constexpr uint32_t kLpHandBack = 35;   // the helper's done written to the loop's look
constexpr uint32_t kLpLoopPolls = 36;  // the loop's L1 polls: ring, acks, done word
constexpr uint32_t kLpHelperLines = 37;  // lines the helper multicast
constexpr uint32_t kLsRows = 3;
// Trace row of command (request) r, 1 to cap: sequence, slot | status << 8 | held
// << 16, wall clock at submit, at ack (low words).
constexpr uint32_t kLtWords = 4;

constexpr uint32_t kLiveMagic = 0x11FE0000;
constexpr uint32_t kLeNone = 0;
constexpr uint32_t kLeDone = 1;     // ended after EXIT
constexpr uint32_t kLeTimeout = 2;  // a deadline missed: stop flag written
constexpr uint32_t kLeBoot = 3;     // the start-up wait gave up: stop flag written
constexpr uint32_t kLeIdle = 4;     // the idle limit passed: stop flag written

// The helper's word pair: the loop writes sequence << 8 | slot into word 0, the helper
// the same into word 1 once its multicast has left; kHandStop in word 0: stop.
constexpr uint32_t kHandLine = 0;
constexpr uint32_t kHandDone = 1;
constexpr uint32_t kHandStop = 0xFFFFFFFFu;
constexpr uint32_t kHandAt = 2;      // wall clock when the loop wrote the token
constexpr uint32_t kHandDoneAt = 3;  // wall clock when the helper wrote its done
constexpr uint32_t kHandStats = 4;   // at the stop: issue, polls, hand-over, lines

#if !defined(LIVE_PACE)
#define LIVE_PACE 0
#endif
// Empty cycles between two polls of L1 (the movers share L1 with TRISC 0): LIVE_PACE.
inline void live_pace() {
    for (uint32_t i = 0; i < LIVE_PACE; i++) {
        asm volatile("nop");
    }
}
