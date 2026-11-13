// The calls behind the stub headers: points, arguments, buffers, NoC, mailboxes and
// the compute threads' tiles, each acting on the calling kernel's core.
#include <stdint.h>

#include <cstring>
#include <string>

#include "sim_impl.hpp"

namespace kmsim {

    void point(const char* name, uint32_t cb) {
        self->last = name;
        self->last_cb = cb;
        if (!g->cfg.triggers.empty()) {
            const Probe p = {self->core, self->role, name, cb, g->clock};
            for (auto& t : g->cfg.triggers) {
                t(p);
            }
        }
        while (g->clock < self->frozen_until) {
            g->yield();
        }
    }

    void yield() {
        g->yield();
    }

    uint32_t arg(uint32_t i) {
        CoreState& c = g->me();
        const auto& a = self->role >= kUnpack ? c.compute_args : c.args;
        if (i >= a.size()) {
            g->violation("argument " + std::to_string(i) + " was not given");
            return 0;
        }
        return a[i];
    }

    uint32_t compile_arg(uint32_t i) {
        return i == 0 ? kTileBytes : 0;
    }

    namespace {

        Cb& cb_of(uint32_t id) {
            Cb& b = g->me().cb[id & 31];
            if (id >= 32 || b.pages == 0) {
                g->violation(
                    "circular buffer " + std::to_string(id) + " is not configured"
                );
            }
            return b;
        }

        uint32_t first_word(uint32_t addr) {
            uint32_t v;
            std::memcpy(&v, g->host(addr), 4);
            return v;
        }

    }  // namespace

    bool cb_room(uint32_t cb, uint32_t pages) {
        const Cb& b = cb_of(cb);
        return b.pages - (b.received - b.acked) >= pages;
    }

    bool cb_ready(uint32_t cb, uint32_t pages) {
        const Cb& b = cb_of(cb);
        return b.received - b.acked >= pages;
    }

    namespace {

        // The command the calling kernel works on (0: none yet).
        uint32_t command_of() {
            const CoreState& c = g->me();
            switch (self->role) {
                case kReader:
                    return c.reader_seq;
                case kWriter:
                    return c.writer_seq;
                default:
                    return c.compute_seq;
            }
        }

        void cb_violation(uint32_t cb, bool producer, const std::string& what) {
            const bool compute = self->role >= kUnpack;
            g->violation(
                "buffer " + std::to_string(cb) + ", " +
                (compute ? "compute" : "data-movement") +
                (producer ? " producer" : " consumer") + ", command " +
                std::to_string(command_of()) + ": " + what
            );
        }

        // Compute wraps at >= (llk_io_pack.h, llk_io_unpack.h: llk_push/pop_tiles),
        // data movement only at == (dataflow_api.h cb_push_back, cb_pop_front).
        void advance(uint32_t cb, Cb& b, uint32_t& ptr, uint32_t pages, bool producer) {
            if (ptr + pages > b.pages) {
                cb_violation(
                    cb,
                    producer,
                    "item of " + std::to_string(pages) + " pages at page " +
                        std::to_string(ptr) + " passes the end of " +
                        std::to_string(b.pages)
                );
            }
            ptr += pages;
            if (self->role >= kUnpack ? ptr >= b.pages : ptr == b.pages) {
                ptr -= b.pages;
            } else if (ptr > b.pages) {
                cb_violation(
                    cb,
                    producer,
                    "pointer past the limit at page " + std::to_string(ptr)
                );
            }
        }

    }  // namespace

    void note_command(uint32_t seq) {
        CoreState& c = g->me();
        (self->role == kReader       ? c.reader_seq
             : self->role == kWriter ? c.writer_seq
                                     : c.compute_seq) = seq;
    }

    void cb_reserve(uint32_t cb, uint32_t pages) {
        while (!cb_room(cb, pages)) {
            g->yield();
        }
        Cb& b = cb_of(cb);
        if (b.wr + pages > b.pages) {
            cb_violation(
                cb,
                true,
                "reservation of " + std::to_string(pages) + " pages at page " +
                    std::to_string(b.wr) + " passes the end of " +
                    std::to_string(b.pages)
            );
        }
        b.reserved = pages;
    }

    void cb_push(uint32_t cb, uint32_t pages) {
        Cb& b = cb_of(cb);
        if (pages > b.reserved) {
            cb_violation(cb, true, "push of pages not reserved");
        }
        b.reserved = pages > b.reserved ? 0 : b.reserved - pages;
        if (cb == km::kCbTokenCompute && self->role == kReader) {
            g->me().reader_seq = first_word(b.base + b.wr * b.page);
            g->log(self->core, kEvTake, g->me().reader_seq, 0);
        }
        b.received += pages;
        *cb_received_word(cb) = b.received;
        uint32_t wr = b.wr;
        advance(cb, b, wr, pages, true);
        b.wr = wr;
    }

    void cb_wait(uint32_t cb, uint32_t pages) {
        while (!cb_ready(cb, pages)) {
            g->yield();
        }
        const Cb& b = cb_of(cb);
        if (b.rd + pages > b.pages) {
            cb_violation(
                cb,
                false,
                "wait for " + std::to_string(pages) + " pages at page " +
                    std::to_string(b.rd) + " passes the end of " +
                    std::to_string(b.pages)
            );
        }
    }

    void cb_pop(uint32_t cb, uint32_t pages) {
        Cb& b = cb_of(cb);
        if (b.received - b.acked < pages) {
            cb_violation(cb, false, "pop of pages not ready");
            return;
        }
        if (cb == km::kCbTokenWriter && self->role == kWriter) {
            g->me().writer_seq = first_word(b.base + b.rd * b.page);
            g->log(self->core, kEvWriterTake, g->me().writer_seq, 0);
        }
        if (cb == km::kCbTokenCompute && self->role == kUnpack) {
            g->me().compute_seq = first_word(b.base + b.rd * b.page);
        }
        b.acked += pages;
        *cb_acked_word(cb) = b.acked;
        uint32_t rd = b.rd;
        advance(cb, b, rd, pages, false);
        b.rd = rd;
    }

    uint32_t cb_write_ptr(uint32_t cb) {
        const Cb& b = cb_of(cb);
        return b.base + b.wr * b.page;
    }

    uint32_t cb_read_ptr(uint32_t cb) {
        const Cb& b = cb_of(cb);
        return b.base + b.rd * b.page;
    }

    CbView cb_view(uint32_t cb) {
        const Cb& b = cb_of(cb);
        return {
            b.pages,
            b.page,
            b.received,
            b.acked,
            b.base + b.pages * b.page,
            b.base + b.wr * b.page,
            b.base + b.rd * b.page
        };
    }

    volatile uint32_t* cb_received_word(uint32_t cb) {
        return reinterpret_cast<uint32_t*>(g->me().mem + kCbRegs + (cb & 31) * 8);
    }

    volatile uint32_t* cb_acked_word(uint32_t cb) {
        return reinterpret_cast<uint32_t*>(g->me().mem + kCbRegs + (cb & 31) * 8 + 4);
    }

    uint32_t cb_page(uint32_t cb) {
        return cb_of(cb).page;
    }

    uint64_t noc_core(uint32_t x, uint32_t y, uint32_t addr) {
        return (
            static_cast<uint64_t>(0x10000u | (y & 0xFF) << 8 | (x & 0xFF)) << 32 | addr
        );
    }

    uint64_t noc_dram(uint32_t addr) {
        return static_cast<uint64_t>(0xFFFFFu) << 32 | addr;
    }

    namespace {

        // Receiver and offset of a NoC address; false (and a violation) if it is none.
        bool target(uint64_t a, uint32_t bytes, uint32_t& core, uint32_t& off) {
            const uint32_t tag = static_cast<uint32_t>(a >> 32),
                           addr = static_cast<uint32_t>(a);
            if (tag == 0xFFFFFu) {
                core = kDram;
                off = addr;
                if (static_cast<uint64_t>(addr) + bytes > kDramBytes) {
                    g->violation("DRAM address out of range");
                    return false;
                }
                return true;
            }
            core = g->core_at(tag & 0xFF, (tag >> 8) & 0xFF);
            off = g->image_off(addr);
            if (core == ~0u || off == ~0u || off + bytes > kL1Bytes) {
                char what[96];
                std::snprintf(
                    what,
                    sizeof what,
                    "NoC address 0x%llx names no L1 word",
                    static_cast<unsigned long long>(a)
                );
                g->violation(what);
                return false;
            }
            return true;
        }

        // The caller's own L1: an address inside its image.
        bool own(uint32_t addr, uint32_t bytes) {
            const CoreState& c = g->me();
            if (addr < c.base || addr + bytes > c.base + kL1Bytes) {
                g->violation("local address outside this core's L1");
                return false;
            }
            return true;
        }

    }  // namespace

    void noc_read(uint64_t src, uint32_t dst, uint32_t bytes) {
        uint32_t core, off;
        if (!target(src, bytes, core, off) || !own(dst, bytes)) {
            return;
        }
        if (core == kDram && (off ^ g->image_off(dst)) % 64 != 0) {
            g->violation("a DRAM read whose source and destination differ in bits 5:0");
        }
        const uint8_t* from = (
            core == kDram ? g->dram.data() + off : g->cores[core].mem + off
        );
        std::memcpy(g->host(dst), from, bytes);
    }

    void noc_write(uint32_t src, uint64_t dst, uint32_t bytes) {
        uint32_t core, off;
        if (bytes == 0 || !target(dst, bytes, core, off) || !own(src, bytes)) {
            return;
        }
        uint32_t kind = 0, seq = 0;
        if (self->role == kWriter && core == g->n && off >= kDispAck &&
            off < kDispAck + 16 * KM_RING_SLOTS) {
            kind = 2;
            seq = first_word(src);
            g->log(self->core, kEvAck, seq, (off - kDispAck) / 16);
        }
        g->send(self->core, self, core, off, g->host(src), bytes, -1, kind, seq);
    }

    void noc_inc(uint64_t dst, uint32_t value) {
        uint32_t core, off;
        if (!target(dst, 4, core, off)) {
            return;
        }
        Pending q = {};
        const Chunk c = {self->core, core, off, 4, 0, 1, true, -1};
        const int64_t d = g->cfg.delay ? g->cfg.delay(c) : 32 + (g->rng() & 63);
        uint32_t seq = 0;
        if (self->role == kWriter) {
            seq = g->me().writer_seq;
            g->log(self->core, kEvReport, seq, core);
        }
        if (d < 0) {
            return;
        }
        q.due = g->clock + static_cast<uint64_t>(d);
        q.ord = g->ord++;
        q.src = self->core;
        q.dst = core;
        q.off = off;
        q.bytes = 4;
        q.atomic = true;
        q.value = value;
        q.from = self;
        q.kind = self->role == kWriter ? 1 : 0;
        q.seq = seq;
        self->atomics++;
        g->pending.push(q);
    }

    void noc_fetch_add(uint64_t dst, uint32_t value, uint32_t ret_addr) {
        uint32_t core, off;
        if (!target(dst, 4, core, off) || (ret_addr != ~0u && !own(ret_addr, 4))) {
            return;
        }
        const Chunk c = {self->core, core, off, 4, 0, 1, true, -1};
        const int64_t d = g->cfg.delay ? g->cfg.delay(c) : 32 + (g->rng() & 63);
        if (d < 0) {
            return;
        }
        Pending q = {};
        q.due = g->clock + static_cast<uint64_t>(d);
        q.ord = g->ord++;
        q.src = self->core;
        q.dst = core;
        q.off = off;
        q.bytes = 4;
        q.atomic = true;
        q.value = value;
        q.from = self;
        q.ret = ret_addr == ~0u ? ~0u : g->image_off(ret_addr);
        self->atomics++;
        g->pending.push(q);
    }

    bool noc_atomics_done() {
        g->yield();
        return self->atomics == 0;
    }

    bool noc_writes_done() {
        g->yield();
        return self->writes == 0;
    }

    void noc_read_barrier() {
    }

    void noc_write_barrier() {
        while (self->writes != 0) {
            g->yield();
        }
    }

    void noc_writes_flushed() {
    }

    void noc_atomic_barrier() {
        while (self->atomics != 0) {
            g->yield();
        }
    }

    namespace {

        uint32_t thread_id() {
            return self->role - kUnpack + 1;
        }

    }  // namespace

    void mailbox_write(uint32_t to, uint32_t value) {
        auto& q = g->me().mbox[thread_id() & 3][to & 3];
        while (q.size() >= g->cfg.mailbox_depth) {
            g->yield();
        }
        q.push_back(value);
        g->sim.mailbox_peak = (
            std::max(g->sim.mailbox_peak, static_cast<uint32_t>(q.size()))
        );
    }

    uint32_t mailbox_read(uint32_t from) {
        auto& q = g->me().mbox[from & 3][thread_id() & 3];
        while (q.empty()) {
            g->yield();
        }
        const uint32_t v = q.front();
        q.pop_front();
        return v;
    }

    void dst_acquire() {
        while (g->me().dst_state != 0) {
            g->yield();
        }
        g->me().dst_state = 1;
    }

    void dst_commit() {
        if (g->me().dst_state != 1) {
            g->violation("tile_regs_commit without tile_regs_acquire");
        }
        g->me().dst_state = 2;
    }

    void dst_wait() {
        while (g->me().dst_state != 2) {
            g->yield();
        }
        g->me().dst_state = 3;
    }

    void dst_release() {
        if (g->me().dst_state != 3) {
            g->violation("tile_regs_release without tile_regs_wait");
        }
        g->me().dst_state = 0;
    }

    void unpack_tile(uint32_t cb, uint32_t tile) {
        if (!cb_ready(cb, tile + 1)) {
            g->violation(
                "copy_tile of a tile not at the front of buffer " + std::to_string(cb)
            );
        }
        const uint16_t* t = (
            reinterpret_cast<const uint16_t*>(
                g->host(cb_read_ptr(cb) + tile * cb_page(cb))
            )
        );
        std::vector<float> v(kTileBytes / 2);
        for (uint32_t e = 0; e < v.size(); e++) {
            v[e] = from_bf16(t[e]);
        }
        while (g->me().src.size() >= 2) {
            g->yield();
        }
        g->me().src.push_back(std::move(v));
    }

    void math_take(uint32_t dst) {
        while (g->me().src.empty()) {
            g->yield();
        }
        g->me().dst[dst & 15] = std::move(g->me().src.front());
        g->me().src.pop_front();
    }

    void math_add(uint32_t a, uint32_t b, uint32_t out) {
        auto& d = g->me().dst;
        for (size_t e = 0; e < d[out & 15].size(); e++) {
            d[out & 15][e] = d[a & 15][e] + d[b & 15][e];
        }
    }

    void math_fill(uint32_t dst, float value) {
        for (auto& e : g->me().dst[dst & 15]) {
            e = value;
        }
    }

    void pack_tile(uint32_t dst, uint32_t cb) {
        const Cb& b = cb_of(cb);
        if (b.reserved == 0) {
            g->violation(
                "pack_tile into buffer " + std::to_string(cb) + " without a reservation"
            );
        }
        uint16_t* t = reinterpret_cast<uint16_t*>(g->host(cb_write_ptr(cb)));
        const auto& d = g->me().dst[dst & 15];
        for (size_t e = 0; e < d.size(); e++) {
            t[e] = to_bf16(d[e]);
        }
    }

}  // namespace kmsim
