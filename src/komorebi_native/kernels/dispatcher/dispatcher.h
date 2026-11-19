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
constexpr uint32_t kTdMessages = 61;      // the dynamic handlers, one a byte (0: none)
constexpr uint32_t kTdStageBytes = 2048;  // kTbStage's bytes; the kept lists' DRAM page
// A thread on TRISC 1 (td_thread.hpp): the X280's 50 MHz ticks a wall-clock tick, times
// 2^32 (0: none); its area is right after the block (kTa*).
constexpr uint32_t kTdThreadScale = 62;
// An island's flat ranks, first | count << 16 (0: one island), and its first core: the
// X280's `island` word. Its link cores are the cores after its ranks.
constexpr uint32_t kTdIsland = 63;
constexpr uint32_t kTdIslandBase = 64;
// A thread on an X280 cluster: its tile (x | y << 16) and its KMD_WORK_* area's address
// there, where in and out copy the area's words (0: the thread is on TRISC 1).
constexpr uint32_t kTdThreadTile = 65;
constexpr uint32_t kTdThreadWork = 66;
// Commands across islands (td_link.hpp, the host's link line).
// this island | islands << 16 (0: no link) | 1 << 31 when messages cross as deliveries
constexpr uint32_t kTdLinkIsland = 67;
constexpr uint32_t kTdLinkCore = 68;   // link core 0: its index in the cores
constexpr uint32_t kTdLinkArea = 69;   // and its link area
constexpr uint32_t kTdLinkBase = 70;   // the link block below, one address everywhere
constexpr uint32_t kTdLinkSlots = 71;  // an inbox part's packets
// 3 words an island: its ranks, its threads (first | count << 16), its dispatcher's
// tile
constexpr uint32_t kTdLinkEntries = 72;
constexpr uint32_t kTdLinkIslandsMax = 8;
// Threads on X280 clusters 2 and 3: tile and work area as kTdThread*, two words each
constexpr uint32_t kTdFarThread = 96;
constexpr uint32_t kTdThreads = 100;  // bit c: a thread on cluster c (sources 1 to 3)
// A dynamic run's X280 ranks (KMD_X280_RANKS >> 32): cluster | count << 4, a part's
// counter tiles << 8 (0: none)
constexpr uint32_t kTdX280Ranks = 101;
constexpr uint32_t kTdArgs = 102;
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
constexpr uint32_t kTbStage = 19072;    // a reset list's bytes on their way to a core
constexpr uint32_t kTbWatch = 21120;  // the watch's line (kTk*), TRISC 2 (td_watch.hpp)
constexpr uint32_t kTbOut = 21184;    // out's large state, off its stack (OutState)
constexpr uint32_t kTbInspect = 23232;  // the inspect block (KMD_LIM_INSPECT's form)
constexpr uint32_t kTbBytes = 129792;
// The areas of threads on clusters 2 and 3, after the engines' (td::thread_area).
constexpr uint32_t kTbFar = 211904;
// The X280 ranks' line (td_ranks.hpp), after those areas, then the link block; kept
// without either, so the link block's address is the same on every island
constexpr uint32_t kTbRanks = 220480;

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

// The watch's line: the two loops' beats, then the watch's words (each one writer).
constexpr uint32_t kTkOut = 0;    // out: its passes, and its beats in a long wait
constexpr uint32_t kTkIn = 1;     // in: its passes
constexpr uint32_t kTkState = 4;  // the watch: KMD_WATCH_ON while it looks, then OFF
constexpr uint32_t kTkSeen = 5;   // the watch: stalls seen (a loop without a beat)
constexpr uint32_t kTkBack = 6;   // the watch: of those, stalls whose loops beat again
constexpr uint32_t kTkLoops = 7;  // the watch: the last stall's loops, 1 out | 2 in
constexpr uint32_t kTkTicks = 8;  // the watch: the last stall's gap (SEEN), then length
constexpr uint32_t kTkGap = 9;    // the watch: the longest gap between two beats
constexpr uint32_t kTkLooks = 10;  // the watch: its looks
// A stall: no beat of a loop for 100 ms (the X280's KMD_WATCH_STALL_TICKS); a look a
// ms.
constexpr uint32_t kTdStallUs = 100000;
constexpr uint32_t kTdLookUs = 1000;

// A thread's area, after the block: the start line (the host writes it last), then the
// X280's LIM block at the X280's offsets (KMX_LIM_*, KMX_LIM_BYTES of them).
constexpr uint32_t kTaStart = 0;
constexpr uint32_t kTaStarted = 1;  // the thread's: the wall clock at kmx_main, 2 words
constexpr uint32_t kTaCodeSet = 3;  // the thread's: 1 once kmx_result set the code
constexpr uint32_t kTaLim = 64;
// Its command ring (source 1 of out, td_sources.hpp): the lines, the thread's read line
// (word 0: events it took), the work area (td::ThreadArea), the thread's ring state.
constexpr uint32_t kTaRing = 1472;
constexpr uint32_t kTaRingDepth = 16;
constexpr uint32_t kTaRead = 2496;
constexpr uint32_t kTaWork = 2560;
constexpr uint32_t kTaEventDepth = 32;
constexpr uint32_t kTaState = 3712;
constexpr uint32_t kTaBytes = 4288;

// The resident engines on TRISC 0 (td_engine.hpp), source 4 of out, after the thread's
// area: ring, read line, out's words, its words, its work area (td::ThreadArea's form).
constexpr uint32_t kTeRing = 0;
constexpr uint32_t kTeRingDepth = 32;
constexpr uint32_t kTeRead = 2048;
constexpr uint32_t kTeOut = 2112;
constexpr uint32_t kTeOwn = 2176;
constexpr uint32_t kTeWork = 2240;
constexpr uint32_t kTeEventDepth = 64;
// Out's graph table (td::GraphTable), TRISC 0's state, then the upload region.
constexpr uint32_t kTeTable = 3392;
constexpr uint32_t kTeState = 4096;
constexpr uint32_t kTeUpload = 12288;
constexpr uint32_t kTeUploadBytes = 65536;
constexpr uint32_t kTeBytes = 77824;
// Out's words: the host's deadline in microseconds, the jobs queued, of those the
// engines' jobs taken in ever (the X280's KMD_ERING_QUEUED and KMD_ERING_TAKEN).
constexpr uint32_t kToDeadline = 0;
constexpr uint32_t kToQueued = 1;
constexpr uint32_t kToTaken = 2;
// TRISC 0's words: its state (1 running, 2 ended), passes, commands, posts, runs.
constexpr uint32_t kTnState = 0;
constexpr uint32_t kTnPasses = 1;
constexpr uint32_t kTnCommands = 2;
constexpr uint32_t kTnPosts = 3;
constexpr uint32_t kTnRuns = 4;
constexpr uint32_t kRanksBytes = 64;  // the X280 ranks' line (kTbRanks)
// The link block, after the X280 ranks' line: the ring of commands from other islands
// (source 5), its events, then out's words, then the inbox parts.
constexpr uint32_t kTlRing = 0;
constexpr uint32_t kTlDepth = 16;  // the X280's KMD_LINK_DEPTH
constexpr uint32_t kTlEvents = 1024;
constexpr uint32_t kTlEventDepth = 32;  // the X280's KMD_LINK_EVENTS
constexpr uint32_t kTlTold = 1536;  // in's: its events written, then the words below
// A dynamic run across islands (td_link.hpp): in's samples of its lines for out (their
// count, 1 idle and unchanged, the sums), and out's word: the run whose lines it wants.
constexpr uint32_t kTlSamples = 1;
constexpr uint32_t kTlQuiet = 2;
constexpr uint32_t kTlSent = 3;      // and its high word
constexpr uint32_t kTlConsumed = 5;  // and its high word
constexpr uint32_t kTlSpans = 8;
constexpr uint32_t kTlWaited = 9;  // out's: 1 + the island the run waits for (0: none)
constexpr uint32_t kTlCount = 1600;
constexpr uint32_t kTlState = 1664;
constexpr uint32_t kTlInbox = 8192;

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
static_assert(kTbScratch + 64 == kTbStage && kTbStage + kTdStageBytes == kTbWatch);
static_assert(
    kTbWatch + 64 == kTbOut && kTbOut + 2048 == kTbInspect, "the watch's line"
);
static_assert(kTbInspect % 64 == 0, "the inspect");
static_assert(kTxWords + 4 * (kTwGiveUps + 1) <= kTxBytes, "the hand-over's words");
static_assert(kTaRing + 64 * kTaRingDepth == kTaRead && kTaRead + 64 == kTaWork);
static_assert(kTaBytes % 64 == 0, "the thread's area is whole lines");
static_assert(kTeRing + 64 * kTeRingDepth == kTeRead && kTeRead + 64 == kTeOut);
static_assert(kTeOut + 64 == kTeOwn && kTeOwn + 64 == kTeWork, "the engines' words");
static_assert(kTeTable % 64 == 0 && kTeState % 64 == 0 && kTeUpload % 64 == 0);
static_assert(kTeUpload + kTeUploadBytes == kTeBytes, "the upload region ends it");
static_assert(kTlRing + 64 * kTlDepth == kTlEvents, "the link ring");
static_assert(kTlEvents + 16 * kTlEventDepth == kTlTold && kTlTold + 64 == kTlCount);
static_assert(kTdLinkEntries + 3 * kTdLinkIslandsMax == kTdFarThread, "the link's");
static_assert(kTdFarThread + 4 == kTdThreads && kTdThreads + 1 == kTdX280Ranks);
static_assert(kTdX280Ranks + 1 == kTdArgs, "the X280 ranks' word");
static_assert(kTbBytes + kTaBytes + kTeBytes == kTbFar, "clusters 2 and 3's areas");
static_assert(kTbFar + 2 * kTaBytes == kTbRanks, "the X280 ranks' line");

// A KM_TD_ACK_COUNT bench build: the workers' count of acks, word 3 of the last slot's
// ack line (the acks' base + this), which no ack writes.
constexpr uint32_t kTdAckCountAt = 1020;
static_assert(kTdAckCountAt == (KM_RING_SLOTS - 1) * 16 + 12, "the last slot's word 3");
