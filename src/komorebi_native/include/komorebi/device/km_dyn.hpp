// Dynamic crossing: an item moves to the queue of another core in DRAM, reserved by
// one atomic add on the receiver; the generic reader and writer, and compute's calls.
#pragma once

#include <stdint.h>

#include "km_core.hpp"
#include "km_record.hpp"

namespace km {

    // Arguments of a dynamic program: reader and writer program arguments 0 to
    // kDynArgs - 1 (the program's own follow), compute program arguments 0 and 1.
    struct DynArgs {
        uint32_t seg;         // the dynamic segment
        uint32_t queue;       // DRAM address of the queues, kDynPageItems items a page
        uint32_t cap;         // items per queue: a power of two, at least kDynPageItems
        uint32_t item_bytes;  // a multiple of 64, at most kDynItemMax
        uint32_t lines;       // status lines at the issuer: core me at lines + 16 me
        uint32_t salt;        // per-process random bits mixed into every mark
        uint32_t idle;        // cycles the idle condition holds before a line says so
        uint32_t bins;        // writer bins, 1 to kDynBinsMax
    };

    constexpr uint32_t kDynArgs = sizeof(DynArgs) / 4;
    constexpr uint32_t kDynCArgItem = 0;   // compute: item bytes
    constexpr uint32_t kDynCArgQuota = 1;  // compute: spawns at the start of a run
    constexpr uint32_t kDynCArgs = 2;

    // Buffers: packets reader -> compute, outgoing items compute -> writer, bins.
    constexpr uint32_t kCbDynIn = 26;
    constexpr uint32_t kCbDynOut = 27;
    constexpr uint32_t kCbDynBins = 28;
    constexpr uint32_t kDynPacket = 32;     // items of a packet and of a bin
    constexpr uint32_t kDynPageItems = 32;  // items of a DRAM page of a queue
    constexpr uint32_t kDynHeader = 64;     // packet header (n, first position): DRAM
                                            // reads land 64-byte aligned
    constexpr uint32_t kDynItemMax = 256;   // bytes
    constexpr uint32_t kDynBinsMax = 16;
    constexpr uint32_t kDynEnd = 0xFFFFFFFF;  // packet n / outgoing dest: the run ended
    constexpr uint32_t kDynWaiting = 1u << 31;  // idle word bit (compute, writer)

    // Idle words (KM_DYN_FLAGS): word 0 the run, word 1 the state.
    constexpr uint32_t kDynFlagReader = 0x00;
    constexpr uint32_t kDynFlagCompute = 0x10;  // waiting in next() | items pushed
    constexpr uint32_t kDynFlagWriter = 0x20;   // every bin sent | items taken

    // Where a dynamic run gave up (KM_DYN_GAVE word 0); word 1 position, word 2 run,
    // word 3: for a gap the word 0 found there, for a lost answer the destination.
    constexpr uint32_t kDynGap = 1;       // a reserved item did not land in time
    constexpr uint32_t kDynOverflow = 2;  // IN_TAIL - IN_HEAD passed the capacity
    constexpr uint32_t kDynAtomic = 3;    // a reservation was not answered in time
    constexpr uint32_t kDynFlush = 4;     // the writer's item writes did not leave
    constexpr uint32_t kDynRead = 5;      // the reader's item reads did not land
    constexpr uint32_t kDynInPage = 6;    // the reader waited for a free packet page
    constexpr uint32_t kDynLine = 7;      // the status line write did not leave
    constexpr uint32_t kDynBadDest = 8;   // hop or spawn to a core outside the team
    constexpr uint32_t kDynNext = 9;      // compute's wait for a packet
    constexpr uint32_t kDynOutPage = 10;  // compute's wait for an outgoing page
    constexpr uint32_t kDynOutbox = 11;   // the writer's wait for an outgoing item
    constexpr uint32_t kDynStopped = 12;  // the stop word ended the run
    constexpr uint32_t kDynAlign = 13;    // packet pages or items not 64-byte aligned
    constexpr uint32_t kDynLost = 14;     // a reservation's answer never landed

    // The mark of the item at queue position `pos`. It is word 0 of every 16-byte
    // chunk of the item: chunks of one write land in no defined order.
    inline uint32_t dyn_mark(uint32_t pos, uint32_t salt) {
        return (pos + 1) ^ salt;
    }

    // Writes the mark into every chunk of the item at `item`.
    inline void dyn_set_mark(uint32_t item, uint32_t bytes, uint32_t mark) {
        for (uint32_t c = 0; c < bytes; c += 16) {
            l1(item + c)[0] = mark;
        }
    }

    // True when every chunk of the item at `item` carries the mark.
    inline bool dyn_landed(uint32_t item, uint32_t bytes, uint32_t mark) {
        for (uint32_t c = 0; c < bytes; c += 16) {
            if (l1(item + c)[0] != mark) {
                return false;
            }
        }
        return true;
    }

    // Queue position `pos` of core `core`: DRAM page and byte offset in it.
    inline uint32_t dyn_page(uint32_t core, uint32_t pos, uint32_t cap) {
        return core * (cap / kDynPageItems) + (pos & (cap - 1)) / kDynPageItems;
    }

    // Records the first give-up of a dynamic run (where, position, run).
    inline void dyn_gave_up(
        uint32_t ctl, uint32_t where, uint32_t pos, uint32_t run, uint32_t seen = 0
    ) {
        volatile tt_l1_ptr uint32_t* g = l1(ctl + KM_CTL_DYN + KM_DYN_GAVE);
        if (g[0] == 0) {
            g[1] = pos;
            g[2] = run;
            g[3] = seen;
            g[0] = where;
        }
    }

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
namespace km {

    // Reads the dynamic arguments that follow the start-up arguments.
    inline DynArgs dyn_args(const Core& c) {
        DynArgs d;
        uint32_t* f = &d.seg;
        for (uint32_t k = 0; k < kDynArgs; k++) {
            f[k] = c.arg(k);
        }
        return d;
    }

    // Polls a NoC counter `done()` for at most the wait limit. False: it did not.
    template <typename Done>
    inline bool noc_bounded(const Core& c, Done done) {
        const uint64_t end = (
            wall64() + (static_cast<uint64_t>(c.args().wait_limit) << 10)
        );
        for (uint32_t i = 0;; i++) {
            if (done()) {
                return true;
            }
            if ((i & 15) == 15 && wall64() > end) {
                return false;
            }
        }
    }

    // Generic reader: this core's queue in order, to compute in packets; idle and busy
    // lines to the issuer; leaves on the run's end record, the stop word or a give-up.
    template <typename Rd>
    inline void dyn_reader(Rd& rd, const km_record_t& r) {
        const Args& a = rd.args();
        const DynArgs d = dyn_args(rd);
        const uint32_t run = r.seq;
        volatile tt_l1_ptr uint32_t* dyn = rd.ctl(KM_CTL_DYN);
        volatile tt_l1_ptr uint32_t* tail = dyn + KM_DYN_IN_TAIL / 4;
        volatile tt_l1_ptr uint32_t* head_w = dyn + KM_DYN_IN_HEAD / 4;
        volatile tt_l1_ptr uint32_t* cf = dyn + (KM_DYN_FLAGS + kDynFlagCompute) / 4;
        volatile tt_l1_ptr uint32_t* wf = dyn + (KM_DYN_FLAGS + kDynFlagWriter) / 4;
        volatile tt_l1_ptr uint32_t* rf = dyn + (KM_DYN_FLAGS + kDynFlagReader) / 4;
        volatile tt_l1_ptr uint32_t* out = dyn + KM_DYN_OUT / 4;
        const InterleavedAddrGen<true> q = {
            .bank_base_address = d.queue, .page_size = kDynPageItems * d.item_bytes
        };
        const uint32_t in_pages = get_local_cb_interface(kCbDynIn).fifo_num_pages;
        const uint64_t line = rd.disp_noc(d.lines + 16 * a.me);
        const uint64_t wait = static_cast<uint64_t>(a.wait_limit) << 10;
        rf[1] = 0;
        rf[0] = run;
        uint32_t head = *head_w, where = 0;
        if ((get_write_ptr(kCbDynIn) | d.item_bytes) % 64 != 0) {
            where = kDynAlign;  // a DRAM read lands 64-byte aligned or not as asked
        }
        bool idle_shown = false, busy = true, armed = false;
        uint32_t key[3] = {0, 0, 0};
        uint64_t since = 0, waiting_from = 0;

        // One 16-byte store of {run, state, sent, consumed} at the issuer.
        auto status = [&](uint32_t state, uint32_t sent, uint32_t consumed) {
            out[0] = run;
            out[1] = state;
            out[2] = sent;
            out[3] = consumed;
            invalidate_l1_cache();  // a fence: the NoC reads the stored line
            noc_async_write(a.ctl + KM_CTL_DYN + KM_DYN_OUT, line, 16);
            return noc_bounded(rd, [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            });
        };
        // The idle condition, as one comparable sample (compute and writer words).
        auto sample = [&](uint32_t t, uint32_t* k) {
            k[0] = t;
            k[1] = cf[1];
            k[2] = wf[1];
            return (
                t == head && cf[0] == run && wf[0] == run && (k[1] & kDynWaiting) &&
                (k[2] & kDynWaiting) && k[1] == k[2] &&
                cb_pages_reservable_at_back(kCbDynIn, in_pages)
            );
        };
        // True when the next record of this core in the ring is this run's end.
        auto end_record = [&] {
            const volatile km_record_t*
                ring = (
                    reinterpret_cast<const volatile km_record_t*>(rd.ctl(KM_CTL_RING))
                );
            volatile tt_l1_ptr uint32_t* last = rd.state() + kStLast;
            for (uint32_t j = 0; j < a.depth; j++) {
                const uint32_t s = ring[j].seq;
                if (!newer(s, last[j])) {
                    continue;
                }
                km_record_t e;
                copy_record(&ring[j], &e);
                if (e.seq == s && KM_LANDED(&e) && seg(e) == KM_SEG_DYN_END &&
                    e.a0 == run && in_team(e, a.me)) {
                    return true;
                }
            }
            return false;
        };

        while (where == 0) {
            invalidate_l1_cache();
            if (rd.stop()) {
                where = kDynStopped;
                break;
            }
            const uint32_t t = *tail;
            if (t - head > d.cap) {
                where = kDynOverflow;
                break;
            }
            if (t == head) {
                if (end_record()) {
                    break;
                }
                uint32_t k[3];
                if (!busy || !sample(t, k)) {
                    armed = false;
                    continue;
                }
                const uint32_t sent = dyn[KM_DYN_SENT / 4];
                const uint32_t consumed = dyn[KM_DYN_CONSUMED / 4];
                uint32_t k2[3];
                invalidate_l1_cache();
                if (!sample(*tail, k2) || k2[0] != k[0] || k2[1] != k[1] ||
                    k2[2] != k[2]) {
                    armed = false;
                    continue;
                }
                if (!armed || key[0] != k[0] || key[1] != k[1] || key[2] != k[2]) {
                    armed = true;
                    key[0] = k[0];
                    key[1] = k[1];
                    key[2] = k[2];
                    since = wall64();
                    continue;
                }
                if (wall64() - since < d.idle) {
                    continue;
                }
                if (!status(KM_DYN_IDLE, sent, consumed)) {
                    where = kDynLine;
                    break;
                }
                rf[1] = 1;
                idle_shown = true;
                busy = false;  // until the next item is taken
                continue;
            }
            busy = true;
            armed = false;
            if (idle_shown) {  // busy has landed before the first item is taken
                if (!status(KM_DYN_BUSY, 0, 0)) {
                    where = kDynLine;
                    break;
                }
                rf[1] = 0;
                idle_shown = false;
            }
            if (rd.poll(
                    [&] {
                        return cb_pages_reservable_at_back(kCbDynIn, 1);
                    },
                    a.wait_limit
                ) != kGaveNone) {
                where = kDynInPage;
                break;
            }
            cb_reserve_back(kCbDynIn, 1);
            const uint32_t page = get_write_ptr(kCbDynIn);
            uint32_t k = t - head < kDynPacket ? t - head : kDynPacket;
            for (uint32_t i = 0; i < k;) {
                const uint32_t pos = (head + i) & (d.cap - 1);
                const uint32_t in_page = pos % kDynPageItems;
                uint32_t n = kDynPageItems - in_page;
                n = n < k - i ? n : k - i;
                noc_async_read(
                    get_noc_addr(dyn_page(a.me, pos, d.cap), q, in_page * d.item_bytes),
                    page + kDynHeader + i * d.item_bytes,
                    n * d.item_bytes
                );
                i += n;
            }
            if (!noc_bounded(rd, [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                })) {
                where = kDynRead;
                break;
            }
            invalidate_l1_cache();
            uint32_t m = 0;
            while (
                m < k &&
                dyn_landed(
                    page + kDynHeader + m * d.item_bytes,
                    d.item_bytes,
                    dyn_mark(head + m, d.salt)
                )
            ) {
                m++;
            }
            if (m == 0) {  // the item at head has not landed yet
                const uint64_t now = wall64();
                if (waiting_from == 0) {
                    waiting_from = now;
                } else if (now - waiting_from > wait) {
                    where = kDynGap;
                    break;
                }
                continue;
            }
            waiting_from = 0;
            l1(page)[0] = m;
            l1(page)[1] = head;
            cb_push_back(kCbDynIn, 1);
            head += m;
            *head_w = head;
        }
        if (where != 0) {
            // a gap: word 0 of the item found at the head, for the report
            dyn_gave_up(
                a.ctl, where, head, run, l1(get_write_ptr(kCbDynIn))[kDynHeader / 4]
            );
            status(KM_DYN_GAVE_UP, head, where);
            rd.gave_up(kStReader, kGaveStop, kWhereData, slot(r), run);
        }
        // Compute leaves its loop on the end packet (or on the stop word).
        if (rd.poll(
                [] {
                    return cb_pages_reservable_at_back(kCbDynIn, 1);
                },
                a.wait_limit,
                a.wait_limit
            ) == kGaveNone) {
            cb_reserve_back(kCbDynIn, 1);
            l1(get_write_ptr(kCbDynIn))[0] = kDynEnd;
            cb_push_back(kCbDynIn, 1);
        }
    }

    // Generic writer: outgoing items in bins by destination; one atomic add per bin
    // reserves positions, then the items go to that queue. True: the run ended.
    template <typename Wr>
    inline bool dyn_writer(Wr& wr, const km_record_t& r) {
        const Args& a = wr.args();
        const DynArgs d = dyn_args(wr);
        const uint32_t run = r.seq, words = d.item_bytes / 4;
        const uint32_t bins = (
            d.bins == 0 || d.bins > kDynBinsMax ? kDynBinsMax : d.bins
        );
        volatile tt_l1_ptr uint32_t* dyn = wr.ctl(KM_CTL_DYN);
        volatile tt_l1_ptr uint32_t* wf = dyn + (KM_DYN_FLAGS + kDynFlagWriter) / 4;
        volatile tt_l1_ptr uint32_t* cf = dyn + (KM_DYN_FLAGS + kDynFlagCompute) / 4;
        volatile tt_l1_ptr uint32_t* ret = dyn + KM_DYN_RET / 4;
        const uint32_t ret_addr = a.ctl + KM_CTL_DYN + KM_DYN_RET;
        const InterleavedAddrGen<true> q = {
            .bank_base_address = d.queue, .page_size = kDynPageItems * d.item_bytes
        };
        const uint32_t store = get_write_ptr(kCbDynBins);
        uint32_t dest[kDynBinsMax], fill[kDynBinsMax];
        for (uint32_t b = 0; b < kDynBinsMax; b++) {
            dest[b] = kDynEnd;
            fill[b] = 0;
        }
        uint32_t taken = 0, sent = 0, held = 0, where = 0, flag = 0, lost_to = 0;
        dyn[KM_DYN_SENT / 4] = 0;
        wf[1] = 0;
        wf[0] = run;

        // Bin b to its destination: reserve, mark, write, wait for the writes.
        auto flush = [&](uint32_t b) {
            const uint32_t n = fill[b];
            ret[0] = kDynEnd;
            invalidate_l1_cache();  // a fence: the store lands before the answer can
            noc_fast_atomic_increment<noc_mode, true>(
                noc_index,
                write_at_cmd_buf,
                wr.core_noc(dest[b], a.ctl + KM_CTL_DYN + KM_DYN_IN_TAIL),
                NOC_UNICAST_WRITE_VC,
                n,
                31,
                false,
                false,
                ret_addr
            );
            if (!noc_bounded(wr, [] {
                    return ncrisc_noc_nonposted_atomics_flushed(noc_index);
                })) {
                where = kDynAtomic;
                return false;
            }
            // On the card the answer can stay away although its count moved.
            if (!noc_bounded(wr, [&] {
                    invalidate_l1_cache();
                    return ret[0] != kDynEnd;
                })) {
                where = kDynLost;
                lost_to = dest[b];
                return false;
            }
            const uint32_t p = ret[0], base = store + b * kDynPacket * d.item_bytes;
            for (uint32_t i = 0; i < n; i++) {
                dyn_set_mark(
                    base + i * d.item_bytes, d.item_bytes, dyn_mark(p + i, d.salt)
                );
            }
            invalidate_l1_cache();  // a fence: the NoC reads the stored marks
            for (uint32_t i = 0; i < n;) {
                const uint32_t pos = (p + i) & (d.cap - 1);
                const uint32_t in_page = pos % kDynPageItems;
                uint32_t m = kDynPageItems - in_page;
                m = m < n - i ? m : n - i;
                noc_async_write(
                    base + i * d.item_bytes,
                    get_noc_addr(
                        dyn_page(dest[b], pos, d.cap), q, in_page * d.item_bytes
                    ),
                    m * d.item_bytes
                );
                i += m;
            }
            if (!noc_bounded(wr, [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                })) {
                where = kDynFlush;
                return false;
            }
            sent += n;
            dyn[KM_DYN_SENT / 4] = sent;
            fill[b] = 0;
            dest[b] = kDynEnd;
            held--;
            return true;
        };
        auto flush_all = [&] {
            for (uint32_t b = 0; b < bins; b++) {
                if (fill[b] != 0 && !flush(b)) {
                    return false;
                }
            }
            return true;
        };

        bool ended = false;
        const uint64_t limit =
            (a.idle_limit == 0 ? ~0ull : static_cast<uint64_t>(a.idle_limit) << 10);
        uint64_t quiet_from = wall64();
        for (uint32_t loop = 0; where == 0 && !ended; loop++) {
            invalidate_l1_cache();
            if (wr.stop()) {
                where = kDynStopped;
                break;
            }
            if (!cb_pages_available_at_front(kCbDynOut, 1)) {
                if (held != 0 && cf[0] == run && (cf[1] & kDynWaiting)) {
                    flush_all();  // compute waits: nothing stays in a bin
                    continue;
                }
                const uint32_t f = held == 0 ? (taken | kDynWaiting) : taken;
                if (f != flag) {
                    flag = f;
                    wf[1] = f;
                }
                if ((loop & 15) == 15 && wall64() - quiet_from > limit) {
                    where = kDynOutbox;
                }
                continue;
            }
            quiet_from = wall64();
            cb_wait_front(kCbDynOut, 1);
            volatile tt_l1_ptr uint32_t* it = l1(get_read_ptr(kCbDynOut));
            const uint32_t to = it[0];
            if (to == kDynEnd) {
                cb_pop_front(kCbDynOut, 1);
                ended = flush_all();
                break;
            }
            const uint32_t b = to % bins;
            if (dest[b] != to && fill[b] != 0 && !flush(b)) {
                break;
            }
            if (fill[b] == 0) {
                dest[b] = to;
                held++;
            }
            volatile tt_l1_ptr uint32_t* slot_w = (
                l1(store + (b * kDynPacket + fill[b]) * d.item_bytes)
            );
            for (uint32_t w = 1; w < words; w++) {
                slot_w[w] = it[w];
            }
            fill[b]++;
            cb_pop_front(kCbDynOut, 1);
            taken++;
            flag = taken;
            wf[1] = taken;
            if (fill[b] == kDynPacket && !flush(b)) {
                break;
            }
        }
        if (where != 0) {
            dyn_gave_up(a.ctl, where, taken, run, lost_to);
            wr.gave_up(kStWriter, kGaveStop, kWhereData, slot(r), run);
            return false;
        }
        return ended;
    }

}  // namespace km

#else
namespace km {

    // A packet of items in L1 (the input buffer's page), valid until done().
    struct Packet {
        uint32_t n;
        uint32_t base;  // byte address of item 0
        uint32_t item_bytes;

        explicit operator bool() const {
            return n != 0;
        }

        const volatile tt_l1_ptr uint32_t* operator[](uint32_t i) const {
            return l1(base + i * item_bytes);
        }
    };

    // Compute's side of a dynamic run. The body runs on all three threads alike; pack
    // writes the outgoing items, unpack takes and frees the packets.
    class Dyn {
    public:
        template <typename Cp>
        Dyn(const Cp& cp, const km_record_t& r)
            : ctl_(get_arg_val<uint32_t>(kCArgCtl)),
              idle_(get_arg_val<uint32_t>(kCArgIdle)),
              wait_(get_arg_val<uint32_t>(kCArgWait)),
              run_(r.seq),
              first_(km::first(r)),
              count_(km::count(r)),
              item_bytes_(cp.arg(kDynCArgItem)),
              quota_(cp.arg(kDynCArgQuota)) {
            UNPACK(({
                volatile tt_l1_ptr uint32_t* dyn = l1(ctl_ + KM_CTL_DYN);
                dyn[KM_DYN_CONSUMED / 4] = 0;
                dyn[KM_DYN_QUOTA / 4] = quota_;
                dyn[(KM_DYN_FLAGS + kDynFlagCompute) / 4 + 1] = 0;
                dyn[(KM_DYN_FLAGS + kDynFlagCompute) / 4] = run_;
            }));
        }

        // True until seeded(): the run's own spawns go here, once per run and core.
        bool seeding() const {
            return !seeded_;
        }

        void seeded() {
            seeded_ = true;
        }

        uint32_t count() const {
            return count_;
        }

        uint32_t quota() const {
            return quota_;
        }

        // The next packet; empty when the run ended, the stop word was set or a wait
        // gave up. Unpack waits and tells math and pack.
        Packet next() {
            seeded_ = true;
            uint32_t n = 0, addr = 0;
            PACK((ckernel::mailbox_write(
                ckernel::ThreadId::UnpackThreadId, pushed_ | (failed_ ? kDynWaiting : 0)
            )));
            UNPACK(({
                const uint32_t p = (
                    ckernel::mailbox_read(ckernel::ThreadId::PackThreadId)
                );
                volatile tt_l1_ptr uint32_t* dyn = l1(ctl_ + KM_CTL_DYN);
                volatile tt_l1_ptr uint32_t*
                    cf = (dyn + (KM_DYN_FLAGS + kDynFlagCompute) / 4);
                if (!(p & kDynWaiting) && !failed_) {
                    pushed_ = p;
                    cf[1] = pushed_ | kDynWaiting;
                    volatile tt_l1_ptr uint32_t* got = (
                        get_cb_tiles_received_ptr(kCbDynIn)
                    );
                    const uint32_t code = poll_stop(
                        l1(ctl_ + KM_CTL_STOP),
                        [&] {
                            return (
                                static_cast<uint16_t>(
                                    ckernel::reg_read(
                                        static_cast<uint32_t>(
                                            reinterpret_cast<uintptr_t>(got)
                                        )
                                    ) -
                                    get_local_cb_interface(kCbDynIn).tiles_acked
                                ) >= 1
                            );
                        },
                        idle_,
                        0
                    );
                    if (code == kGaveNone) {
                        cb_wait_front(kCbDynIn, 1);
                        invalidate_l1_cache();
                        addr = ckernel::get_tile_l1_byte_address(kCbDynIn, 0);
                        n = l1(addr)[0];
                        if (n == kDynEnd || n == 0 || n > kDynPacket) {
                            cb_pop_front(kCbDynIn, 1);
                            n = 0;
                        } else {
                            cf[1] = pushed_;
                        }
                    } else {
                        dyn_gave_up(
                            ctl_, code == kGaveStop ? kDynStopped : kDynNext, 0, run_
                        );
                        record_gave_up(ctl_, kStCompute, code, kWhereCToken, 0, run_);
                    }
                }
                ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, n);
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, n);
                ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, addr);
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, addr);
            }));
            MATH(({
                n = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
                addr = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
            }));
            PACK(({
                n = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
                addr = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
            }));
            if (n == 0) {
                PACK((end_marker()));
                return {0, 0, item_bytes_};
            }
            return {n, addr + kDynHeader, item_bytes_};
        }

        // The item goes on at member `dest` of the run's team. item: item_bytes / 4
        // words; word 0 of each 16-byte chunk (words 0, 4, 8, ...) is the runtime's.
        void hop(uint32_t dest, const uint32_t* item) {
            if (dest >= count_) {
                failed_ = true;
                PACK((dyn_gave_up(ctl_, kDynBadDest, dest, run_)));
                PACK((
                    record_gave_up(ctl_, kStCompute, kGaveStop, kWhereCOutput, 0, run_)
                ));
                return;
            }
            PACK(({
                if (!failed_) {
                    send(first_ + dest, item);
                }
            }));
        }

        // The item ends here: one spawn comes back to this core.
        void retire() {
            quota_++;
        }

        // A new item at member `dest`; false when this core has no quota left.
        bool spawn(uint32_t dest, const uint32_t* item) {
            if (quota_ == 0) {
                return false;
            }
            quota_--;
            hop(dest, item);
            return true;
        }

        // The packet's page is free; CONSUMED counts its items.
        void done(const Packet& pk) {
            MATH((ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1)));
            PACK((ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1)));
            UNPACK(({
                (void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId);
                (void)ckernel::mailbox_read(ckernel::ThreadId::PackThreadId);
                cb_pop_front(kCbDynIn, 1);
                consumed_ += pk.n;
                volatile tt_l1_ptr uint32_t* dyn = l1(ctl_ + KM_CTL_DYN);
                dyn[KM_DYN_CONSUMED / 4] = consumed_;
                dyn[KM_DYN_QUOTA / 4] = quota_;
            }));
        }

        // Waits for the stop word (bounded by the idle limit): a body that stalls.
        bool wait_stop() const {
            return poll_stop(
                       l1(ctl_ + KM_CTL_STOP),
                       [] {
                           return false;
                       },
                       idle_,
                       0
                   ) == kGaveStop;
        }

    private:
        // Pack: one outgoing item, word 0 the destination core.
        void send(uint32_t to, const uint32_t* item) {
            if (!reserve()) {
                return;
            }
            volatile tt_l1_ptr uint32_t* o = (
                l1(get_local_cb_interface(kCbDynOut).fifo_wr_ptr << cb_addr_shift)
            );
            for (uint32_t w = 1; w < item_bytes_ / 4; w++) {
                o[w] = item[w];
            }
            o[0] = to;
            __sync_synchronize();  // the item is in L1 before the push
            cb_push_back(kCbDynOut, 1);
            pushed_++;
        }

        // Pack: the writer leaves its loop on this item.
        void end_marker() {
            if (!reserve()) {
                return;
            }
            l1(get_local_cb_interface(kCbDynOut).fifo_wr_ptr
               << cb_addr_shift)[0] = kDynEnd;
            __sync_synchronize();  // the item is in L1 before the push
            cb_push_back(kCbDynOut, 1);
        }

        // Pack: a free outgoing page (bounded, observes the stop word).
        bool reserve() {
            volatile tt_reg_ptr uint32_t* acked = get_cb_tiles_acked_ptr(kCbDynOut);
            const uint32_t code = poll_stop(
                l1(ctl_ + KM_CTL_STOP),
                [&] {
                    const uint16_t used = (
                        static_cast<uint16_t>(
                            get_local_cb_interface(kCbDynOut).tiles_received -
                            ckernel::reg_read(
                                static_cast<uint32_t>(
                                    reinterpret_cast<uintptr_t>(acked)
                                )
                            )
                        )
                    );
                    return get_local_cb_interface(kCbDynOut).fifo_num_pages - used >= 1;
                },
                wait_,
                0
            );
            if (code != kGaveNone) {
                failed_ = true;
                dyn_gave_up(ctl_, kDynOutPage, pushed_, run_);
                record_gave_up(ctl_, kStCompute, code, kWhereCOutput, 0, run_);
                return false;
            }
            cb_reserve_back(kCbDynOut, 1);
            return true;
        }

        uint32_t ctl_;
        uint32_t idle_;
        uint32_t wait_;
        uint32_t run_;
        uint32_t first_;
        uint32_t count_;
        uint32_t item_bytes_;
        uint32_t quota_;
        uint32_t pushed_ = 0;
        uint32_t consumed_ = 0;
        bool seeded_ = false;
        bool failed_ = false;
    };

}  // namespace km
#endif
