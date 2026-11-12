// The simulated grid behind the stub: cores with L1 images, a thread per kernel, the
// delivery thread, the dispatcher model, fault injection and the log the cases check.
#pragma once

#include <stdint.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "kmsim/sim.hpp"

// The kernels under test, built with kernel_main renamed to these.
void kmsim_reader_main();
void kmsim_writer_main();
void kmsim_unpack_main();
void kmsim_math_main();
void kmsim_pack_main();

// Rows of the completion tree the build is checked against (0: one level).
#if !defined(KMSIM_TREE_ROW)
#define KMSIM_TREE_ROW 0
#endif

namespace kmsim {

    constexpr uint32_t kL1Bytes = 0x10000;  // one core's L1 image
    constexpr uint32_t kCtlOff = 0x1000;    // the control block in every image
    constexpr uint32_t kDram = 0xFFFFu;     // Chunk::dst of a DRAM write
    constexpr int64_t kDrop = -1;           // a delay that drops the chunk
    constexpr uint32_t kTileBytes = 2048;   // a page of the example's buffers

    // Dispatcher image: ack word j, READY word g (16 bytes apart), the hello mailbox.
    constexpr uint32_t kDispAck = 0x100;
    constexpr uint32_t kDispReady = 0x300;
    constexpr uint32_t kDispMailbox = 0x500;

    // DRAM: core table, the input tiles and the output pages of each ring slot.
    constexpr uint32_t kDramBytes = 4u << 20;
    constexpr uint32_t kDramTable = 0;
    constexpr uint32_t kDramIn = 0x10000;
    constexpr uint32_t kInTiles = 64;  // most input tiles of a record
    constexpr uint32_t kDramOut = kDramIn + 16 * kInTiles * kTileBytes;
    constexpr uint32_t kOutPages = 16;  // output pages of a slot

    // One piece of a NoC write or one atomic on its way: a write is cut at the
    // receiver's 16-byte boundaries, each piece gets its own delay.
    struct Chunk {
        uint32_t src;    // sending core; the dispatcher is Config::cores
        uint32_t dst;    // receiving core, the dispatcher, or kDram
        uint32_t off;    // offset in the receiver's image (DRAM: the address)
        uint32_t bytes;  // at most 16
        uint32_t part;   // index of this piece in its write
        uint32_t parts;  // pieces in the write
        bool atomic;
        int32_t job;  // dispatcher writes: the record's index; else -1
    };

    // A stub call of a kernel thread, as the case's triggers see it; cb is the buffer,
    // or for a mailbox call the other thread's id (1 unpack, 2 math, 3 pack).
    struct Probe {
        uint32_t core;
        Role role;
        const char* name;
        uint32_t cb;
        uint64_t t;
    };

    // Log entries: a is a sequence number unless said otherwise.
    enum Ev : uint32_t {
        kEvTake,        // reader handed record a to compute and writer
        kEvWriterTake,  // writer took the token of record a
        kEvReport,      // writer sent its increment for a to core b
        kEvReportLand,  // that increment landed on core b
        kEvAck,         // writer wrote ack word b with value a
        kEvAckLand,     // that ack landed
        kEvStop,        // the stop word of the core was set (landed)
    };

    struct Event {
        uint64_t t;
        uint32_t core;
        uint32_t kind;
        uint32_t a;
        uint32_t b;
    };

    // One record for the dispatcher. map: a0 to a3 are filled in (inputs written,
    // outputs checked); hdr_slot >= 0 replaces the slot field of the header.
    struct Job {
        uint32_t seg = 0;
        uint32_t first = 0;
        uint32_t count = 1;
        uint32_t a[10] = {};
        uint32_t flags = 0;  // header flags: KM_F_EXCL
        bool map = false;
        int32_t hdr_slot = -1;
        uint64_t gap = 0;  // cycles after the previous issue before this one
    };

    enum JobStatus : uint32_t { kNotIssued, kIssued, kDone, kTimedOut };

    struct JobResult {
        uint32_t seq = 0;
        uint32_t slot = 0;
        uint64_t issued = 0;
        uint64_t acked = 0;
        uint32_t status = kNotIssued;
        uint32_t data_bad = 0;  // output elements that differ from the expectation
        uint32_t result = 0;    // word 1 of its ack line when it was acknowledged
    };

    // A circular buffer of every core: id, offset in the L1 image, page size, pages.
    struct CbSpec {
        uint32_t id, off, page, pages;
    };

    struct Config {
        uint32_t cores = 4;                  // workers 0 to cores - 1
        uint32_t row = 2;                    // cores per grid row: start-up groups
        uint32_t tree_row = KMSIM_TREE_ROW;  // completion tree rows; 0: one level
        uint32_t mailbox_depth = 4;          // words per compute mailbox
        uint32_t depth = 4;
        uint32_t wait_limit = 16;  // 1024-cycle units
        uint32_t idle_limit = 512;
        uint32_t tick = 16;            // cycles per scheduling step
        uint64_t deadline = 1u << 20;  // cycles from issue to ack (also start-up)
        uint64_t bound = 1ull << 24;   // cycles by which every thread has returned
        double host_seconds = 60;      // host-time watchdog
        uint64_t seed = 1;
        std::vector<Job> jobs;
        bool dispatch = true;     // false: no dispatcher (a case's own kernels)
        std::vector<CbSpec> cbs;  // buffers added to or replacing the example's
        void (*kernels[kRoles])() = {
            kmsim_reader_main,
            kmsim_writer_main,
            kmsim_unpack_main,
            kmsim_math_main,
            kmsim_pack_main
        };
        bool exit = true;        // EXIT after the last job
        uint32_t exit_team = 0;  // its team field; 0: every core
        uint32_t first_seq = 1;
        std::function<uint32_t(uint32_t)> next_seq;  // default: + 1
        std::function<int64_t(const Chunk&)> delay;  // default: 32 to 95 cycles
        std::vector<std::function<void(const Probe&)>> triggers;
        std::function<void(uint32_t job, uint32_t seq, uint32_t slot)> on_issue;
    };

    class Sim {
    public:
        explicit Sim(Config cfg);
        ~Sim();
        void run();

        // Fault injection, from a trigger or on_issue.
        void freeze(uint32_t core, Role role, uint64_t cycles);
        void set_stop(uint32_t core);
        uint64_t now() const;

        // After run().
        uint32_t word(uint32_t core, uint32_t off) const;  // ctl-relative, 32-bit
        uint32_t state(uint32_t core, uint32_t index) const;
        uint32_t mailbox_words() const;  // words left in compute mailboxes, all cores
        uint32_t mailbox_peak = 0;       // most words ever queued in one mailbox
        uint32_t popped(uint32_t core, uint32_t cb) const;  // pages popped so far

        Config cfg;
        bool all_returned = false;
        std::vector<std::string> stuck;       // kernels that did not return, and where
        std::vector<std::string> violations;  // stub contract or landing violations
        std::vector<Event> events;
        std::vector<JobResult> jobs;
        JobResult exit_job;
        bool booted = false;
        bool timed_out = false;
        uint64_t end_cycles = 0;
        double host_ms = 0;

        struct Impl;

    private:
        std::unique_ptr<Impl> impl_;
    };

    const char* role_name(uint32_t role);

}  // namespace kmsim
