// The texture lookup probe's handlers, arguments and layouts (C++ and the host read
// this file).
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerLoop = 1;    // a0 cpu, a1 form, a2 stream, a3 slots, a4 reps
constexpr uint32_t kHandlerMiss = 2;    // a0 cpu, a1 in flight, a2 bytes, a3 reps
constexpr uint32_t kHandlerColour = 3;  // a0 form, a1 cpu, a2 reps
constexpr uint32_t kHandlerReplay = 4;  // a0 slots, a1 form, a2 passes
constexpr uint32_t kHandlerReport = 5;  // the writer: results and out to DRAM

// The CPU that runs a command: the two data-movement CPUs, two compute CPUs.
constexpr uint32_t kCpuReader = 0;
constexpr uint32_t kCpuWriter = 1;
constexpr uint32_t kCpuUnpack = 2;
constexpr uint32_t kCpuPack = 3;

// How a pixel's page is found: its number is the page's place in L1, a hash of its
// key into a table of 4-way sets, or the same with the last page found kept aside.
constexpr uint32_t kFormDirect = 0;
constexpr uint32_t kFormHash = 1;
constexpr uint32_t kFormHashLast = 2;

// The colour: integers on a CPU, or the vector unit (unpack, multiply, pack), or the
// vector unit's path without the multiply (unpack and pack only).
constexpr uint32_t kColourInteger = 0;
constexpr uint32_t kColourVector = 1;
constexpr uint32_t kColourCopy = 2;

// A pixel's word: the page key above bit 10, the texel's place in its page below.
constexpr uint32_t kOffsetBits = 10;
constexpr uint32_t kOffsetMask = 1023;
constexpr uint32_t kPageBytes = 4096;
constexpr uint32_t kPageShift = 12;
constexpr uint32_t kPixels = 1024;
constexpr uint32_t kSlotsMax = 32;
constexpr uint32_t kWays = 4;
constexpr uint32_t kStreams = 8;      // synthetic streams of kPixels words each
constexpr uint32_t kRepsMax = 64;     // timed repetitions kept per command
constexpr uint32_t kInFlightMax = 8;  // DRAM reads in flight at once
constexpr uint32_t kOutWords = 2048;  // the replay's output ring
constexpr uint32_t kItemsMax = 64;    // replay: items a core

// L1 tensors of every core (the host passes their addresses as arguments 0 to 6, the
// same to reader, writer and compute).
constexpr uint32_t kArgPages = 0;    // kSlotsMax pages: the texels of the loop
constexpr uint32_t kArgSlots = 1;    // kSlotsMax pages: the table's pages (misses)
constexpr uint32_t kArgStreams = 2;  // kStreams x kPixels words, then the keys
constexpr uint32_t kArgPlanes = 3;   // texel plane, light plane (float32 bits)
constexpr uint32_t kArgOut = 4;      // kOutWords words: a plane per command
constexpr uint32_t kArgResults = 5;  // kResultWords words
constexpr uint32_t kArgReplay = 6;   // replay: item table, then the words of all items
constexpr uint32_t kArgPool = 7;     // the DRAM pool's address (reader and writer)
constexpr uint32_t kArgPoolPages = 8;
constexpr uint32_t kArgReport = 9;  // the DRAM report (writer)

// The DRAM report, a page of 4 KB each: core c's results at page c, its out words at
// pages kReportOut + 2 c and + 1 (the host reads it while the run is live).
constexpr uint32_t kReportCores = 12;
constexpr uint32_t kReportOut = kReportCores;
constexpr uint32_t kReportPages = 3 * kReportCores;

// Keys of the hash forms: kSlotsMax words after the streams, the table's pages in slot
// order (key + 1 in a tag; 0 is an empty way).
constexpr uint32_t kKeysAt = kStreams * kPixels;

// Result words of a core.
constexpr uint32_t kResultError = 0;  // nonzero: a wait gave up, or a bad argument
constexpr uint32_t kResultRun = 1;    // the command's a9 (the host's run number)
constexpr uint32_t kResultSum = 2;    // the sum of the texels read
constexpr uint32_t kResultMisses = 3;
constexpr uint32_t kResultReps = 16;    // cycles of each repetition
constexpr uint32_t kResultItems = 128;  // replay: cycles, misses, sum of each item
constexpr uint32_t kResultWords = 1024;

// The replay's item table (words at kArgReplay): items, then per item first word and
// words; the words of every item follow at kReplayWordsAt.
constexpr uint32_t kReplayWordsAt = 1 + 2 * kItemsMax;

// Errors.
constexpr uint32_t kErrorArgs = 1;
constexpr uint32_t kErrorRead = 2;  // a DRAM read did not land in time
constexpr uint32_t kErrorWait = 4;  // a buffer wait gave up

// Circular buffers: the colour planes for the unpacker, the packed colour plane.
constexpr uint32_t kCbColour = 0;
constexpr uint32_t kCbPacked = 16;
