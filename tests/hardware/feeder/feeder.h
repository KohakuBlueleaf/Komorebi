// Test feeder: its arguments, buffers and the layout of the results it writes.
#pragma once

#include <stdint.h>

// Runtime arguments.
constexpr uint32_t kFdControl = 0;    // control block address (the workers' too)
constexpr uint32_t kFdWorkers = 1;    // cores 0 to workers - 1 get every command
constexpr uint32_t kFdCores = 2;      // DRAM address of the core table
constexpr uint32_t kFdTable = 3;      // DRAM address of the commands, 64 bytes each
constexpr uint32_t kFdCommands = 4;   // number of commands
constexpr uint32_t kFdDepth = 5;      // slots used, 1 to KM_RING_SLOTS
constexpr uint32_t kFdDeadline = 6;   // per command, submit to ack, 1024-cycle units
constexpr uint32_t kFdResults = 7;    // DRAM address of the results, 64 bytes a row
constexpr uint32_t kFdGroups = 8;     // start-up groups: READY words to wait for
constexpr uint32_t kFdBootLimit = 9;  // timeout of the start-up wait, 1024-cycle units
constexpr uint32_t kFdExit = 10;      // nonzero: EXIT last (if the table has none)
constexpr uint32_t kFdRelease = 11;  // nonzero: a9 = submit time + this (cycles, low word)
// Nonzero: command kFdRaw - 1 is submitted with the slot and group it names (test
// only).
constexpr uint32_t kFdRaw = 12;
constexpr uint32_t kFdMessagesHandler = 13;       // nonzero: commands of this handler are dynamic runs
constexpr uint32_t kFdLines = 14;  // L1 address of the cores' status lines (16 bytes)
constexpr uint32_t kFdArgs = 15;

// Buffers on the feeder core: the commands plus two spares (EXIT, the end of a dynamic
// run), the results.
constexpr uint32_t kFdCbTable = 0;
constexpr uint32_t kFdCbResults = 1;

// Results row 0: the summary.
constexpr uint32_t kFrMagic = 0;      // kFeederMagic
constexpr uint32_t kFrStatus = 1;     // kFs*
constexpr uint32_t kFrSlot = 2;       // timed-out command: slot
constexpr uint32_t kFrSequence = 3;   // its sequence number
constexpr uint32_t kFrIndex = 4;      // its index in the table
constexpr uint32_t kFrSubmitted = 5;  // commands submitted
constexpr uint32_t kFrDone = 6;       // commands acknowledged
constexpr uint32_t kFrBoot = 7;       // start-up wait, cycles / 16
constexpr uint32_t kFrFirst = 8;      // wall clock at the first submit (low word)
constexpr uint32_t kFrLast = 9;     // wall clock at the last acknowledgement (low word)
constexpr uint32_t kFrSpanLo = 10;  // first submit to last acknowledgement, cycles
constexpr uint32_t kFrSpanHi = 11;
constexpr uint32_t kFrSubmitCycles = 12;              // submitting (all writes of a command), summed
constexpr uint32_t kFrMaxBusy = 13;    // most commands in flight at once
constexpr uint32_t kFrExitAcked = 14;  // 1: the EXIT command was completed
constexpr uint32_t kFrReady = 15;      // READY words seen at start-up (bit per group)
// Results row 1 + i: command i.
constexpr uint32_t kFrRecSequence = 0;
constexpr uint32_t kFrRecSlot = 1;
constexpr uint32_t kFrRecStatus = 2;     // kFs*
constexpr uint32_t kFrCmdSubmit = 3;     // wall clock before the first write (low word)
constexpr uint32_t kFrCmdSubmitted = 4;  // after the last write left
constexpr uint32_t kFrRecAcked = 5;      // when the feeder saw the ack word
constexpr uint32_t kFrRecBusy = 6;       // commands in flight after submitting it
constexpr uint32_t kFrRecHeld = 7;  // 1: KM_FLAG_EXCLUSIVE held it while a group overlapped
constexpr uint32_t kFrRecResult = 8;  // KM_FLAG_RESULT: the result word acked with it
// A dynamic run's row, from word 9: the quiet end, the submitter's checks, the sums.
constexpr uint32_t kFrRecMessagesEnd = 9;      // wall clock when the quiet end was seen
constexpr uint32_t kFrRecMessagesChecks = 10;  // two-pass checks made
constexpr uint32_t kFrRecMessagesSent = 11;    // sum of the lines' sent at the end
constexpr uint32_t kFrRecMessagesGave = 12;  // 1 + the core whose line said it gave up
constexpr uint32_t kFrRecMessagesEndSequence = 13;  // sequence of the end command
constexpr uint32_t kFrRecMessagesEndAck = 14;  // wall clock when the end command was acked

constexpr uint32_t kFeederMagic = 0xFEED0000;
constexpr uint32_t kFsNone = 0;     // not submitted
constexpr uint32_t kFsDone = 1;     // acknowledged
constexpr uint32_t kFsTimeout = 2;  // deadline missed: stop flag written
constexpr uint32_t kFsBoot = 3;     // start-up wait gave up: stop flag written
constexpr uint32_t kFsMessagesGave = 4;  // a core's status line said it gave up: stop written
