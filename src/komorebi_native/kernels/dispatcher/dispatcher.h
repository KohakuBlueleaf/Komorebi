// The Tensix dispatcher's arguments, its block in L1 (the host ring) and its area of
// the host channel; the host reads the same constants (komorebi.session.tensix).
#pragma once

#include <stdint.h>

#include "komorebi/abi/km_stream.h"

// Runtime arguments of the dispatcher kernel.
constexpr uint32_t kTdControl = 0;     // control block address (every core's)
constexpr uint32_t kTdRanks = 1;       // the workers: ranks 0 to ranks - 1
constexpr uint32_t kTdCores = 2;       // DRAM address of the core table
constexpr uint32_t kTdBlock = 3;       // L1 address of the block below, 64-byte aligned
constexpr uint32_t kTdHostDepth = 4;   // host ring lines, a power of two, at most 64
constexpr uint32_t kTdEventDepth = 5;  // event ring, a power of two, at most 128
constexpr uint32_t kTdArea = 6;        // channel offset of the area below
constexpr uint32_t kTdDepth = 7;       // slots used, 1 to KM_RING_SLOTS
constexpr uint32_t kTdDeadline = 8;    // per command, submit to ack, 1024-tick units
constexpr uint32_t kTdBootLimit = 9;   // start-up wait, 1024-tick units
constexpr uint32_t kTdIdleLimit = 10;  // nothing taken nor busy this long: 0 never
constexpr uint32_t kTdGroups = 11;     // start-up groups: READY words to wait for
constexpr uint32_t kTdRects = 12;      // multicast rectangles (0: unicast to each)
// Rectangle r at kTdRect + 3 r: NoC 0 start and end (x | y << 16), receivers.
constexpr uint32_t kTdRect = 13;
constexpr uint32_t kTdRectsMax = 4;
constexpr uint32_t kTdLoopback = 25;  // 1: a rectangle's multicast counts its sender
constexpr uint32_t kTdTicksUs = 26;   // wall-clock ticks a microsecond, times 65536
constexpr uint32_t kTdPrograms = 27;  // 1: the table below holds each core's program
// The program field each core holds at the start, core c in byte c % 4 of word c / 4
// (0xFF: none); the X280's KMD_LIM_PROGRAMS.
constexpr uint32_t kTdHeld = 28;
constexpr uint32_t kTdTag = 60;  // the load's tag: the hand-over's Go, End and Over
constexpr uint32_t kTdMessages = 61;  // the dynamic handlers, one a byte (0: none)
constexpr uint32_t kTdArgs = 62;
// The block's buffer, on the dispatcher's core alone: the block at its first 64-byte
// boundary.
constexpr uint32_t kTdBlockCb = 0;
constexpr uint32_t kTdProgramsMax = 64;  // program fields 0 to 63 (KMD_PROGRAMS_MAX)

// The block in the dispatcher's L1: the host writes the ring and the read line.
constexpr uint32_t kTbRing = 0;        // host_depth lines of 64 bytes
constexpr uint32_t kTbRead = 4096;     // word 0: events the host has read
constexpr uint32_t kTbWords = 4160;    // staging of the host words (KMD_HOST_*)
constexpr uint32_t kTbEvents = 4224;   // staging of the events, 16 bytes each
constexpr uint32_t kTbPending = 6272;  // the command taken and not yet submitted
constexpr uint32_t kTbResult = 6336;   // staging of the result (kThResult)
constexpr uint32_t kTbHand = 6592;     // the hand-over between the two loops (kTx*)
constexpr uint32_t kTbQueue = 9792;    // the job queue's jobs (td_queue.hpp)
constexpr uint32_t kTbLines = 15936;   // a dynamic run's status lines, 16 bytes a core
constexpr uint32_t kTbSample = 17984;  // in's first look at them: {sent, consumed} a core
constexpr uint32_t kTbScratch = 19008;  // out's NoC reads of a core's live words
constexpr uint32_t kTbInspect = 19072;  // the inspect block (KMD_LIM_INSPECT's form)
constexpr uint32_t kTbBytes = 125632;

// The hand-over (td_slots.hpp), each word written by one loop: "out" (takes, submits)
// and "in" (sweeps, writes the events and the host words).
constexpr uint32_t kTxIssued = 0;  // out: slot j's sequence once its record is written
constexpr uint32_t kTxDone = 256;  // in: slot j's sequence once its event is written
constexpr uint32_t kTxRecords = 512;  // out: slot j's record, 32 bytes (td::Record)
constexpr uint32_t kTxEvents = 2560;  // out: events of no slot, kTxEventsMax x 16 bytes
constexpr uint32_t kTxEventsMax = 32;
constexpr uint32_t kTxWords = 3072;  // the words below, 4 bytes each
constexpr uint32_t kTxBytes = 3200;
// Words of out.
constexpr uint32_t kTwGo = 0;        // 1: started, in may write the host words
constexpr uint32_t kTwEnd = 1;       // 1: finishing, in ends its loop
constexpr uint32_t kTwConsumed = 2;  // the host's commands taken
constexpr uint32_t kTwPosted = 3;    // events posted at kTxEvents
// Words of in.
constexpr uint32_t kTwPushed = 8;  // events written to the host's ring
constexpr uint32_t kTwTaken = 10;  // posted events written
constexpr uint32_t kTwLate = 11;   // 1 + the slot whose deadline passed first
constexpr uint32_t kTwOver = 12;   // 1: in ended, its words final
constexpr uint32_t kTwResults = 13;
constexpr uint32_t kTwLastResult = 14;
constexpr uint32_t kTwLastLo = 15;  // the last event's wall clock
constexpr uint32_t kTwLastHi = 16;
constexpr uint32_t kTwPasses = 17;  // in's passes
constexpr uint32_t kTwSweeps = 18;  // of those, passes that swept
// The dynamic run in flight: out's, its slot and group, then its sequence (0: none).
constexpr uint32_t kTwRunSlot = 19;
constexpr uint32_t kTwRunGroup = 20;
constexpr uint32_t kTwRunSequence = 21;
constexpr uint32_t kTwRunQuiet = 22;  // in: the run's sequence once quiet, its end owed
constexpr uint32_t kTwRunGave = 23;   // in: 1 + the core whose line said gave up
constexpr uint32_t kTwRunChecks = 24;  // in: two-look checks
constexpr uint32_t kTwRunQuietAt = 25;  // in: the wall clock's low word at the last quiet
// A deadline's give-up: in asks it (the slot, then its sequence), out answers it.
constexpr uint32_t kTwGiveSlot = 26;
constexpr uint32_t kTwGiveSequence = 27;
constexpr uint32_t kTwGivenUp = 28;   // out: the sequence once every core left it
constexpr uint32_t kTwGiveKept = 29;  // out: the sequence a core did not leave (fatal)
constexpr uint32_t kTwGiveUps = 30;   // in: the commands given up (KMD_H_GIVEN_UP)

// The job queue: a job's line and its numbers (td_queue.hpp).
constexpr uint32_t kTqJobBytes = 96;

// The area in the host channel: the host words and the event ring as the X280's host
// buffer (KMD_HOST_*), then the result block (KMD_R_*), then kTr* words.
constexpr uint32_t kThResult = 4096;
constexpr uint32_t kThBytes = 8192;
constexpr uint32_t kTrAbandoned = 32;    // commands in flight at the finish
constexpr uint32_t kTrBoundWrites = 33;  // ring bound changes
constexpr uint32_t kTrBoundHigh = 34;    // the highest bound written
constexpr uint32_t kTrBoundLow = 35;     // the lowest bound written (0: none)
constexpr uint32_t kTrInPasses = 36;     // in's passes
constexpr uint32_t kTrInSweeps = 37;     // of those, passes that swept the slots
constexpr uint32_t kTrRunGave = 38;      // KMD_H_MESSAGES_GAVE: 1 + the core (0: none)
constexpr uint32_t kTrRunChecks = 39;    // KMD_H_MESSAGES_CHECKS
constexpr uint32_t kTrRunQuietAt = 48;   // KMD_H_MESSAGES_QUIET
constexpr uint32_t kTrParts = 40;  // a KM_TD_PARTS build: the loop's ticks by part
constexpr uint32_t kTrPartsCount = 8;
constexpr uint32_t kTrWords = 64;
constexpr uint32_t kTdHostDepthMax = 64;
constexpr uint32_t kTdEventDepthMax = 128;

// Empty cycles between two polls of L1 on a pass that found nothing.
constexpr uint32_t kTdPace = 16;
// Passes between two looks at the dispatcher's own stop flag while commands flow.
constexpr uint32_t kTdStopLook = 64;

static_assert(kTdHeld + KM_CORES_MAX / 4 == kTdTag, "a byte a core");
static_assert(kTbHand + kTxBytes == kTbQueue && kTbHand % 64 == 0, "the hand-over");
static_assert(kTbQueue + KM_QUEUE_JOBS * kTqJobBytes == kTbLines, "the job queue");
static_assert(kTbLines + 16 * KM_CORES_MAX == kTbSample, "the status lines");
static_assert(kTbSample + 8 * KM_CORES_MAX == kTbScratch, "the first looks");
static_assert(kTbScratch + 64 == kTbInspect && kTbInspect % 64 == 0, "the inspect");
static_assert(kTxWords + 4 * (kTwGiveUps + 1) <= kTxBytes, "the hand-over's words");

// A KM_TD_ACK_COUNT bench build: the workers' count of acks, word 3 of the last slot's
// ack line (the acks' base + this), which no ack writes.
constexpr uint32_t kTdAckCountAt = 1020;
static_assert(kTdAckCountAt == (KM_RING_SLOTS - 1) * 16 + 12, "the last slot's word 3");
