// Test feeder: its arguments, buffers and the layout of the results it writes.
#pragma once

#include <stdint.h>

// Runtime arguments.
constexpr uint32_t kFdCtl = 0;        // control block address (the workers' too)
constexpr uint32_t kFdWorkers = 1;    // cores 0 to workers - 1 get every record
constexpr uint32_t kFdCores = 2;      // DRAM address of the core table
constexpr uint32_t kFdTable = 3;      // DRAM address of the records, 64 bytes each
constexpr uint32_t kFdRecords = 4;    // number of records
constexpr uint32_t kFdDepth = 5;      // slots used, 1 to KM_RING_SLOTS
constexpr uint32_t kFdDeadline = 6;   // per record, issue to ack, 1024-cycle units
constexpr uint32_t kFdResults = 7;    // DRAM address of the results, 64 bytes a row
constexpr uint32_t kFdGroups = 8;     // start-up groups: READY words to wait for
constexpr uint32_t kFdBootLimit = 9;  // bound of the start-up wait, 1024-cycle units
constexpr uint32_t kFdExit = 10;      // nonzero: EXIT last (if the table has none)
constexpr uint32_t kFdRelease = 11;  // nonzero: a9 = issue time + this (cycles, low word)
// Nonzero: record kFdRaw - 1 is issued with the slot and team it names (test only).
constexpr uint32_t kFdRaw = 12;
constexpr uint32_t kFdArgs = 13;

// Buffers on the feeder core: the records plus a spare (n + 1 rows), the results.
constexpr uint32_t kFdCbTable = 0;
constexpr uint32_t kFdCbResults = 1;

// Results row 0: the summary.
constexpr uint32_t kFrMagic = 0;    // kFeederMagic
constexpr uint32_t kFrStatus = 1;   // kFs*
constexpr uint32_t kFrSlot = 2;     // timed-out record: slot
constexpr uint32_t kFrSeq = 3;      // its sequence number
constexpr uint32_t kFrIndex = 4;    // its index in the table
constexpr uint32_t kFrIssued = 5;   // records issued
constexpr uint32_t kFrDone = 6;     // records acknowledged
constexpr uint32_t kFrBoot = 7;     // start-up wait, cycles / 16
constexpr uint32_t kFrFirst = 8;    // wall clock at the first issue (low word)
constexpr uint32_t kFrLast = 9;     // wall clock at the last acknowledgement (low word)
constexpr uint32_t kFrSpanLo = 10;  // first issue to last acknowledgement, cycles
constexpr uint32_t kFrSpanHi = 11;
constexpr uint32_t kFrIssueCycles = 12;  // issuing (all writes of a record), summed
constexpr uint32_t kFrMaxBusy = 13;      // most records in flight at once
constexpr uint32_t kFrExitAcked = 14;    // 1: the EXIT record was completed
constexpr uint32_t kFrReady = 15;        // READY words seen at start-up (bit per group)
// Results row 1 + i: record i.
constexpr uint32_t kFrRecSeq = 0;
constexpr uint32_t kFrRecSlot = 1;
constexpr uint32_t kFrRecStatus = 2;  // kFs*
constexpr uint32_t kFrRecIssue = 3;   // wall clock before the first write (low word)
constexpr uint32_t kFrRecIssued = 4;  // after the last write left
constexpr uint32_t kFrRecAcked = 5;   // when the feeder saw the ack word
constexpr uint32_t kFrRecBusy = 6;    // records in flight after issuing it
constexpr uint32_t kFrRecHeld = 7;    // 1: KM_F_EXCL held it while a team overlapped
constexpr uint32_t kFrRecResult = 8;  // KM_F_RESULT: the result word acked with it

constexpr uint32_t kFeederMagic = 0xFEED0000;
constexpr uint32_t kFsNone = 0;     // not issued
constexpr uint32_t kFsDone = 1;     // acknowledged
constexpr uint32_t kFsTimeout = 2;  // deadline missed: stop word written
constexpr uint32_t kFsBoot = 3;     // start-up wait gave up: stop word written
