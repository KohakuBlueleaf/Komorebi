// Attention on resident workers: segments, record words, buffers and the job table that
// the host and the three kernels share. Sizes are in 32 x 32 tiles.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_lane.hpp"

// ATTEND: a0 Q, a1 K, a2 V, a3 O (bf16 tiles, [B, H, S, D] in tile order), a4 the job
// tables (a page per member), a5 mode, a6 passes over the table (0: one).
constexpr uint32_t kSegAttend = 1;
// PROF: a0 profile tensor; the writer stores this core's profile page at the member's
// page, then (a1 != 0) zeroes it.
constexpr uint32_t kSegProf = 2;

// ATTEND modes: the full call; cast only (Q reads and the K / V broadcast, compute pops
// the blocks); compute only (Q, K, V as they are in L1: no reads, lane or writes).
constexpr uint32_t kModeFull = 0;
constexpr uint32_t kModeCastOnly = 1;
constexpr uint32_t kModeComputeOnly = 2;

// Buffers: a job's Q rows; the lane buffer, B blocks of K [DT][KB] then V [KB][DV];
// the compute's own as ktt_flash's (ones, P, row state, O, tokens, causal mask).
constexpr uint32_t kCbQ = 0;
constexpr uint32_t kCbKv = 1;
constexpr uint32_t kCbOnes = 3;
constexpr uint32_t kCbP = 4;
constexpr uint32_t kCbSt = 5;
constexpr uint32_t kCbO = 6;
constexpr uint32_t kCbStok = 7;
constexpr uint32_t kCbOtok = 8;
constexpr uint32_t kCbCausal = 10;
constexpr uint32_t kCbOut = 16;          // output rows to the writer
constexpr uint32_t kCbJobs = 17;         // the job table, reader to compute
constexpr uint32_t kCbWriterJobs = 18;   // the writer's own copy of its table
constexpr uint32_t kTileBytes = 2048;    // bf16
constexpr uint32_t kTile32Bytes = 4096;  // fp32
constexpr uint32_t kTokenBytes = 64;
constexpr uint32_t kSubChunk = 3;  // key tiles per compute step

// Job table page: the job count, then per job from kTabFirst: first Q, K, V and O
// pages, query tile rows (at most R), the first row in its head, key tiles, row step.
constexpr uint32_t kTableBytes = 4096;
constexpr uint32_t kTabJobs = 0;
constexpr uint32_t kTabFirst = 8;
constexpr uint32_t kJobWords = 8;
constexpr uint32_t kJobQ = 0;
constexpr uint32_t kJobK = 1;
constexpr uint32_t kJobV = 2;
constexpr uint32_t kJobO = 3;
constexpr uint32_t kJobRows = 4;
constexpr uint32_t kJobStart = 5;
constexpr uint32_t kJobKeys = 6;
constexpr uint32_t kJobStride = 7;
constexpr uint32_t kJobsMax = (kTableBytes / 4 - kTabFirst) / kJobWords;

// Profile page: 32-bit words of wall-clock cycles summed over the records, each word
// written by one thread.
constexpr uint32_t kCbProf = 19;
constexpr uint32_t kProfBytes = 1024;
constexpr uint32_t kProfWords = 32;
// Reader: body, waits for Q space, Q reads, the sender's slot waits, K / V read issue,
// casts (read barrier, receivers' stamps, multicast), a receiver's arrivals, finish.
constexpr uint32_t kPrRdBody = 0;
constexpr uint32_t kPrRdQWait = 1;
constexpr uint32_t kPrRdQRead = 2;
constexpr uint32_t kPrRdKvFree = 3;
constexpr uint32_t kPrRdKvRead = 4;
constexpr uint32_t kPrRdCast = 5;
constexpr uint32_t kPrRdArrive = 6;
constexpr uint32_t kPrRdFinish = 7;
// Unpack: body, waits for Q, for K / V blocks. Math: body, the score sections (matmul
// and row maxima), the row maxima alone, the P V sections.
constexpr uint32_t kPrUnBody = 8;
constexpr uint32_t kPrUnWaitQ = 9;
constexpr uint32_t kPrUnWaitKv = 10;
constexpr uint32_t kPrMaBody = 12;
constexpr uint32_t kPrMaScore = 13;
constexpr uint32_t kPrMaMax = 14;
constexpr uint32_t kPrMaPv = 15;
// Pack: body, waits for math, the exponential sections, packing P and the state, the
// output sections (rescale, packing O), the rows' normalization, score tiles.
constexpr uint32_t kPrPkBody = 16;
constexpr uint32_t kPrPkWait = 17;
constexpr uint32_t kPrPkExp = 18;
constexpr uint32_t kPrPkPack = 19;
constexpr uint32_t kPrPkOut = 20;
constexpr uint32_t kPrPkFinal = 21;
constexpr uint32_t kPrPkTiles = 22;
// Writer: body, waits for output rows, writes.
constexpr uint32_t kPrWrBody = 24;
constexpr uint32_t kPrWrWait = 25;
constexpr uint32_t kPrWrWrite = 26;

// Program arguments. Reader: the lane's words. Compute: the lane window B.
constexpr uint32_t kArgLane = 0;
constexpr uint32_t kArgWindow = 0;

#if !defined(COMPILE_FOR_TRISC)

// Bounded wait until this kernel's reads have landed (km::kFlushReads) or its writes
// have left L1 (km::kFlushSent); a give-up goes into state word `word`.
inline bool flushed(const km::Core& c, uint32_t word, uint32_t what) {
    const uint32_t code = c.poll(
        [&] {
            if constexpr (noc_mode == DM_DYNAMIC_NOC) {
                return (
                    what == km::kFlushReads
                        ? ncrisc_dynamic_noc_reads_flushed(noc_index)
                        : ncrisc_dynamic_noc_nonposted_writes_sent(noc_index)
                );
            } else {
                return (
                    what == km::kFlushReads
                        ? ncrisc_noc_reads_flushed(noc_index)
                        : ncrisc_noc_nonposted_writes_sent(noc_index)
                );
            }
        },
        c.args().wait_limit,
        c.args().wait_limit
    );
    if (code != km::kGaveNone) {
        c.gave_up(word, code, km::kWhereData, 0, 0);
    }
    return code == km::kGaveNone;
}

// Reads this member's job table page (page `member` of the tables at `base`) to L1
// `dst`. False: the read gave up (recorded).
inline bool read_table(
    const km::Core& c, uint32_t word, uint32_t base, uint32_t member, uint32_t dst
) {
    const InterleavedAddrGen<true> pages = {
        .bank_base_address = base, .page_size = kTableBytes
    };
    noc_async_read(get_noc_addr(member, pages), dst, kTableBytes);
    return flushed(c, word, km::kFlushReads);
}

#endif
