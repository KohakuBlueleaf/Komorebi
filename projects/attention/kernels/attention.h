// Attention on persistent kernels: handlers, command words, buffers and the job table
// that the host and the three kernels share. Sizes are in 32 x 32 tiles.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_broadcast.hpp"

// ATTEND: a0 q_address, a1 k_address, a2 v_address, a3 o_address (bf16 tiles), a4
// job_tables, a5 mode, a6 requested_passes (0: one), a7 first_job, a8 job_count.
constexpr uint32_t kHandlerAttend = 1;
// PROF: a0 profile_address (the profile tensor); the writer stores this core's profile
// page at its rank's page, then (zero_profile, a1 != 0) zeroes it.
constexpr uint32_t kHandlerProf = 2;

// ATTEND modes: the full call; broadcast only (Q reads, the K / V broadcast, compute
// pops the blocks); compute only (Q, K, V in L1: no reads, broadcast or writes).
constexpr uint32_t kModeFull = 0;
constexpr uint32_t kModeBroadcastOnly = 1;
constexpr uint32_t kModeComputeOnly = 2;

// Buffers: a job's Q rows; the broadcast buffer, B blocks of K [DT][KB], V [KB][DV];
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

// Profile page: 32-bit words of wall-clock cycles summed over the commands, each word
// written by one thread.
constexpr uint32_t kCbProf = 19;
constexpr uint32_t kProfBytes = 1024;
constexpr uint32_t kProfWords = 32;
// Reader: body, waits for Q space, Q reads, the sender's slot waits, K / V read submit,
// multicasts (read barrier, receivers' markers, multicast), arrivals, finish.
constexpr uint32_t kPrRdBody = 0;
constexpr uint32_t kPrRdQWait = 1;
constexpr uint32_t kPrRdQRead = 2;
constexpr uint32_t kPrRdKvFree = 3;
constexpr uint32_t kPrRdKvRead = 4;
constexpr uint32_t kPrRdBroadcast = 5;
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

// The command's end in the core's `jobs` jobs (it starts at job first_job).
inline uint32_t jobs_end(const km_command_t& command, uint32_t jobs) {
    const uint32_t first_job = command.a7, job_count = command.a8;
    return (
        job_count == 0 || first_job + job_count > jobs ? jobs : first_job + job_count
    );
}

// Program arguments. The kernel that runs the broadcast (the reader on NoC0, the writer
// with BROADCAST_ON_WRITER on NoC1): the broadcast's words. Compute: the window B.
constexpr uint32_t kArgBroadcast = 0;
constexpr uint32_t kArgWindow = 0;

#if !defined(COMPILE_FOR_TRISC)

// Timed wait until this kernel's reads have landed (km::kFlushReads) or its writes
// have left L1 (km::kFlushSent); a give-up goes into state word `word`.
inline bool flushed(const km::Core& core, uint32_t word, uint32_t what) {
    const uint32_t code = core.poll(
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
        core.args().wait_limit,
        core.args().wait_limit
    );
    if (code != km::kGaveNone) {
        core.gave_up(word, code, km::kWhereData, 0, 0);
    }
    return code == km::kGaveNone;
}

// Reads this core's job table page (page `rank` of the tables at `base`) to L1
// `dst`. False: the read gave up (recorded).
inline bool read_table(
    const km::Core& core, uint32_t word, uint32_t base, uint32_t rank, uint32_t dst
) {
    const InterleavedAddrGen<true> pages = {
        .bank_base_address = base, .page_size = kTableBytes
    };
    noc_async_read(get_noc_addr(rank, pages), dst, kTableBytes);
    return flushed(core, word, km::kFlushReads);
}

// Sender: K tiles [block_start, block_start + keys_in_block) of every head-dim tile
// (head-tile-major) and their V rows (key-major) into the block at `address`.
template <
    uint32_t HEAD_TILES,
    uint32_t VALUE_TILES,
    uint32_t BLOCK_KEYS,
    typename Accessor>
void read_kv(
    const Accessor& keys,
    const Accessor& values,
    uint32_t first_key_page,
    uint32_t first_value_page,
    uint32_t block_start,
    uint32_t keys_in_block,
    uint32_t address
) {
    for (uint32_t key = 0; key < keys_in_block; ++key) {
        for (uint32_t head_tile = 0; head_tile < HEAD_TILES; ++head_tile) {
            noc_async_read_page(
                first_key_page + (block_start + key) * HEAD_TILES + head_tile,
                keys,
                address + (head_tile * BLOCK_KEYS + key) * kTileBytes
            );
        }
    }
    for (uint32_t key = 0; key < keys_in_block; ++key) {
        for (uint32_t column = 0; column < VALUE_TILES; ++column) {
            noc_async_read_page(
                first_value_page + (block_start + key) * VALUE_TILES + column,
                values,
                address +
                    (HEAD_TILES * BLOCK_KEYS + key * VALUE_TILES + column) * kTileBytes
            );
        }
    }
}

// One job's K / V blocks through the broadcast (`job_words`; `blocks`, `counters`):
// the sender reads and multicasts, a receiver waits. False: gave up.
template <
    uint32_t HEAD_TILES,
    uint32_t VALUE_TILES,
    uint32_t BLOCK_KEYS,
    typename Accessor>
bool broadcast_job(
    km::Broadcast& broadcast,
    const Accessor& keys,
    const Accessor& values,
    const volatile tt_l1_ptr uint32_t* job_words,
    uint32_t& blocks,
    uint32_t* counters
) {
    const uint32_t first_key_page = job_words[kJobK];
    const uint32_t first_value_page = job_words[kJobV];
    const uint32_t key_tiles = job_words[kJobKeys];
    for (
        uint32_t block_start = 0; block_start < key_tiles;
        block_start += BLOCK_KEYS, ++blocks
    ) {
        const uint32_t started = km::wall();
        if (!broadcast.sender()) {
            if (!broadcast.arrive(blocks)) {
                return false;
            }
            counters[kPrRdArrive] += km::wall() - started;
            continue;
        }
        const uint32_t address = broadcast.reserve(blocks);
        if (address == 0) {
            return false;
        }
        const uint32_t reserved = km::wall();
        const uint32_t keys_in_block = (
            key_tiles - block_start < BLOCK_KEYS ? key_tiles - block_start : BLOCK_KEYS
        );
        read_kv<HEAD_TILES, VALUE_TILES, BLOCK_KEYS>(
            keys,
            values,
            first_key_page,
            first_value_page,
            block_start,
            keys_in_block,
            address
        );
        const uint32_t read_issued = km::wall();
        if (!broadcast.cast(blocks)) {
            return false;
        }
        counters[kPrRdKvFree] += reserved - started;
        counters[kPrRdKvRead] += read_issued - reserved;
        counters[kPrRdBroadcast] += km::wall() - read_issued;
    }
    return true;
}

#endif
