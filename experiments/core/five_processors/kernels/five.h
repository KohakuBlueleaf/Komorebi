// Five processors of one core: the handlers, the processors and the words of a core's
// row (an L1 tensor, one row of kRowWords words in every core) its kernels share.
#pragma once

#include <stdint.h>

// PROBE (X1): a0 mode (kProbe*), a1 the NoC, a2 the target's translated x | y << 16,
// a3 the TRISC (0 unpack, 1 math, 2 pack); the movers wait, then retake counters.
constexpr uint32_t kHandlerProbe = 1;
constexpr uint32_t kProbeRegister = 1;  // read NOC_NODE_ID and NOC_ID_LOGICAL
constexpr uint32_t kProbeWrite = 2;     // 16 bytes to the target's kTrisLanding
constexpr uint32_t kProbeRead = 3;      // 16 bytes from the target's kReadSource
// LOOP (X2): a0 the processors (bit p), a1 iterations over each one's block; a2 not 0:
// run until a2 x 1024 cycles have passed instead.
constexpr uint32_t kHandlerLoop = 2;
// NOC (X3): a0 processors (1 BRISC, 2 NCRISC, 3 both halves, 4 TRISC 0), a1 kind, a2 N,
// a3 repetitions, a4 first rank or DRAM bank, a9 DRAM address; ticks at kTimes (m).
constexpr uint32_t kHandlerNoc = 3;
constexpr uint32_t kNocWriteOne = 1;   // one 64-byte write to rank a4, flushed
constexpr uint32_t kNocReadOne = 2;    // one 64-byte read from rank a4
constexpr uint32_t kNocWriteDram = 3;  // one 64-byte write to bank a4 at a9, flushed
constexpr uint32_t kNocReadDram = 4;   // one 64-byte read from bank a4 at a9
constexpr uint32_t kNocWriteMany = 5;  // N writes to ranks a4.., then one flush
constexpr uint32_t kNocReadMany = 6;   // N reads from ranks a4.., then one barrier
// one multicast write of 64 bytes to a2 cores: NoC 0 corners a5 (start) and a6 (end),
// NoC 1 corners a7 and a8, each translated x | y << 16
constexpr uint32_t kNocMulticast = 7;
// STAMP (X4): the reader writes the 64-bit wall clock to its result words a0, a0 + 1.
constexpr uint32_t kHandlerStamp = 4;
// PING (X5): a0 pair (kPair*), a1 round trips. The first of the pair writes k to the
// ping word, the second answers k in the pong word; the first counts the ticks.
constexpr uint32_t kHandlerPing = 5;
constexpr uint32_t kPairBriscNcrisc = 0;
constexpr uint32_t kPairBriscTrisc = 1;  // BRISC and TRISC 0 (unpack)
constexpr uint32_t kPairTriscTrisc = 2;  // TRISC 0 (unpack) and TRISC 2 (pack)
constexpr uint32_t kPairTriscBrisc = 3;  // TRISC 0 starts, BRISC answers

// Processors, as bit p of a mask and as the index of a result block.
constexpr uint32_t kBrisc = 0;
constexpr uint32_t kNcrisc = 1;
constexpr uint32_t kTrisc0 = 2;  // unpack, then math, then pack
constexpr uint32_t kProcessors = 5;

// A core's row (uint32 words).
constexpr uint32_t kRowWords = 2304;
constexpr uint32_t kResult = 0;  // processor p's 32 result words at p x 32
constexpr uint32_t kResultWords = 32;
// Words of a result block.
constexpr uint32_t kRStart = 0;  // LOOP: start, end (64-bit, low first), checksum,
constexpr uint32_t kREnd = 2;    // iterations done
constexpr uint32_t kRSum = 4;
constexpr uint32_t kRIterations = 5;
constexpr uint32_t kRProbe = 8;   // PROBE: 8 words (kernels/compute.cpp, mover.hpp)
constexpr uint32_t kRStamp = 16;  // STAMP: 4 words (two stamps)
constexpr uint32_t kRPing = 20;   // PING: ticks (64-bit), round trips, timeouts
constexpr uint32_t kRNoc = 24;    // NOC: repetitions, failures
constexpr uint32_t kSync = 192;   // words of the handshakes, a 16-byte line each
constexpr uint32_t kPing = kSync;
constexpr uint32_t kPong = kSync + 4;
constexpr uint32_t kGo = kSync + 8;  // NOC with both movers: BRISC's repetition + 1,
constexpr uint32_t kGoTime = kSync + 9;       // its start time (low wall-clock word),
constexpr uint32_t kGoBack = kSync + 12;      // NCRISC's repetitions done
constexpr uint32_t kTrisDone = kSync + 16;    // PROBE: the TRISC's done (the sequence)
constexpr uint32_t kMoverReady = kSync + 20;  // PROBE: mover m's ready at + m x 4
constexpr uint32_t kTimes = 256;  // mover m's ticks (TRISC 0: m 2) at + m x 256
constexpr uint32_t kTimesWords = 256;
constexpr uint32_t kLanding = 1024;      // 64-byte landing of m at + m x 16
constexpr uint32_t kTrisLanding = 1072;  // PROBE: a TRISC's 16-byte write lands here
constexpr uint32_t kMulticastLanding = 1088;
constexpr uint32_t kReadSource = 1152;    // 16 words the host fills: rank x 256 + word
constexpr uint32_t kSource = 1216;        // 64-byte source of m at + m x 16
constexpr uint32_t kTrisSource = 1264;    // PROBE: the TRISC's 16-byte source
constexpr uint32_t kReadInto = 1280;      // 64-byte read of m lands at + m x 16
constexpr uint32_t kTrisReadInto = 1328;  // PROBE: the TRISC's 16-byte read lands here
constexpr uint32_t kTable = 1536;         // NoC addresses of m at + m x kTableWords
constexpr uint32_t kTableWords = 256;     // 2 words a core
constexpr uint32_t kBlockWords = 1024;    // LOOP: 4 KiB a processor
