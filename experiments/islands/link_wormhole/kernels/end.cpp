// One end of a link over km_transport.hpp, its partner this kernel on another chip:
// pings, command lines with their DONE, a page stream; every wait bounded.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../../../../src/komorebi_native/include/komorebi/device/links/km_transport.hpp"
#include "wl.h"

namespace {

    constexpr uint32_t kArgs = 17;  // the host's count comes as compile-time arg 0

    struct End {
        uint32_t base, near, hops, peer_x, peer_y, rounds, pages, page, window, batch;
        uint32_t check, seed;
        uint64_t limit;  // ticks a wait may take
        volatile tt_l1_ptr uint32_t* status;
        km::LinkSender* sender;

        volatile tt_l1_ptr uint32_t* word(uint32_t offset) const {
            return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + offset);
        }

        uint64_t far(uint32_t offset) const {
            return km::link_noc(peer_x, peer_y, base + offset);
        }

        km::LinkHeaderAt header(uint32_t k) const {
            return km::link_header(base + kHeaders + 64 * k, hops);
        }

        // Until `ready()`, or false after the limit with `code` set.
        template <typename Ready>
        bool wait(Ready ready, uint32_t code) {
            const uint64_t end = km::wall64() + limit;
            uint32_t polls = 0;
            while (!ready()) {
                if ((++polls & 63) == 0 && km::wall64() > end) {
                    status[kWordCode] = code;
                    return false;
                }
            }
            return true;
        }

        // The header and payload handed to the router once it has a slot; the header's
        // and payload's words not touched again until they left (sent()).
        bool send(km::LinkHeaderAt header, uint32_t from, uint32_t bytes) {
            if (!wait(
                [this] {
                    return sender->space();
                },
                kCodeSpace
            )) {
                return false;
            }
            sender->send(header, from, bytes);
            return true;
        }

        bool sent() {
            return wait(
                [] {
                    return km::link_sent();
                },
                kCodeSpace
            );
        }

        bool count_reaches(uint32_t offset, uint32_t value) {
            volatile tt_l1_ptr uint32_t* count = word(kCounts + offset);
            return wait(
                [count, value] {
                    invalidate_l1_cache();
                    return *count >= value;
                },
                kCodeWait
            );
        }

        void timed(uint32_t at, uint64_t ticks) {
            const uint32_t t = static_cast<uint32_t>(ticks);
            if (status[at] == 0 || t < status[at]) {
                status[at] = t;
            }
            status[at + 1] += t;
            if (t > status[at + 2]) {
                status[at + 2] = t;
            }
        }

        // Ping: 64 bytes and an add on the partner's count in one packet (the fused
        // write and increment of a counted write); the far end sends them back.
        bool ping(uint32_t r) {
            volatile tt_l1_ptr uint32_t* out = word(kPing);
            if (!sent()) {
                return false;
            }
            for (uint32_t w = 0; w < kPingBytes / 4; w++) {
                out[w] = w == 0 ? r : pattern(seed, r, w);
            }
            km::LinkHeaderAt h = header(0);
            km::link_counted_write(
                h, far(kPong), far(kCounts + kCountPing), 1, kPingBytes
            );
            return send(h, base + kPing, kPingBytes);
        }

        bool pong_landed(uint32_t r) {
            if (!count_reaches(kCountPing, r)) {
                return false;
            }
            volatile tt_l1_ptr uint32_t* in = word(kPong);
            for (uint32_t w = 0; w < kPingBytes / 4; w++) {
                if (in[w] != (w == 0 ? r : pattern(seed, r, w))) {
                    status[kWordEarly]++;
                    return true;
                }
            }
            return true;
        }

        // A command's line: 128 bytes into the partner's landing, then (another
        // header) an add on its line count: the count may only land after the line.
        bool line(uint32_t r) {
            volatile tt_l1_ptr uint32_t* out = word(kStage);
            if (!sent()) {
                return false;
            }
            for (uint32_t w = 0; w < kLineBytes / 4; w++) {
                out[w] = w == 0 ? r : pattern(seed, r + 0x10000, w);
            }
            km::LinkHeaderAt data = header(1);
            km::link_write(data, far(kLanding), kLineBytes);
            if (!send(data, base + kStage, kLineBytes)) {
                return false;
            }
            km::LinkHeaderAt count = header(2);
            km::link_increment(count, far(kCounts + kCountLine), 1);
            return send(count, 0, 0);
        }

        bool line_landed(uint32_t r) {
            if (!count_reaches(kCountLine, r)) {
                return false;
            }
            volatile tt_l1_ptr uint32_t* in = word(kLanding);
            for (uint32_t w = 0; w < kLineBytes / 4; w++) {
                if (in[w] != (w == 0 ? r : pattern(seed, r + 0x10000, w))) {
                    status[kWordEarly]++;
                    break;
                }
            }
            return true;
        }

        // DONE: one word written into the partner's done word (an inline write).
        bool done(uint32_t r) {
            if (!sent()) {
                return false;
            }
            km::LinkHeaderAt h = header(2);
            km::link_inline(h, far(kCounts + kCountDone), r);
            return send(h, 0, 0);
        }

        // `pages` pages of `page` bytes into the partner's `window` slots, a count
        // every `batch` pages; the receiver frees slots a batch at a time.
        bool send_pages() {
            for (uint32_t s = 0; s < window; s++) {
                volatile tt_l1_ptr uint32_t* slot = word(kSource + s * page);
                for (uint32_t w = 0; w < page / 4; w++) {
                    slot[w] = pattern(seed, s, w);
                }
            }
            volatile tt_l1_ptr uint32_t* credit = word(kCounts + kCountCredit);
            const uint64_t first = km::wall64();
            for (uint32_t p = 0; p < pages; p++) {
                if (!wait(
                    [credit, p, this] {
                        invalidate_l1_cache();
                        return p < *credit + window;
                    },
                    kCodeWait
                )) {
                    return false;
                }
                const uint32_t slot = p % window;
                const bool counted = (p + 1) % batch == 0 || p + 1 == pages;
                if (!sent()) {
                    return false;
                }
                km::LinkHeaderAt h = header(3);
                if (counted) {
                    km::link_counted_write(
                        h,
                        far(kSource + slot * page),
                        far(kCounts + kCountPages),
                        1,
                        page
                    );
                } else {
                    km::link_write(h, far(kSource + slot * page), page);
                }
                if (!send(h, base + kSource + slot * page, page)) {
                    return false;
                }
                status[kWordPages] = p + 1;
            }
            const uint64_t span = km::wall64() - first;
            status[kWordSpanLo] = static_cast<uint32_t>(span);
            status[kWordSpanHi] = static_cast<uint32_t>(span >> 32);
            return sent();
        }

        bool take_pages() {
            const uint32_t batches = (pages + batch - 1) / batch;
            uint64_t first = 0;
            for (uint32_t b = 0; b < batches; b++) {
                if (!count_reaches(kCountPages, b + 1)) {
                    return false;
                }
                if (b == 0) {
                    first = km::wall64();
                }
                const uint32_t from = b * batch;
                const uint32_t to = from + batch < pages ? from + batch : pages;
                for (uint32_t p = from; p < to; p++) {
                    const uint32_t s = p % window;
                    volatile tt_l1_ptr uint32_t* slot = word(kSource + s * page);
                    const uint32_t last = page / 4 - 1;
                    for (uint32_t w = 0; w <= last; w += check ? 1 : last) {
                        if (slot[w] != pattern(seed, s, w)) {
                            status[kWordWrong]++;
                        }
                    }
                }
                status[kWordPages] = to;
                if (!sent()) {
                    return false;
                }
                km::LinkHeaderAt h = header(3);
                km::link_increment(h, far(kCounts + kCountCredit), to - from);
                if (!send(h, 0, 0)) {
                    return false;
                }
            }
            const uint64_t span = km::wall64() -
                first;  // the first batch's count to now
            status[kWordSpanLo] = static_cast<uint32_t>(span);
            status[kWordSpanHi] = static_cast<uint32_t>(span >> 32);
            return true;
        }
    };

}  // namespace

void kernel_main() {
    size_t i = 0;
    End e{};
    e.base = get_arg_val<uint32_t>(i++);
    e.near = get_arg_val<uint32_t>(i++);  // 1: this end starts each round, sends pages
    const uint32_t channel = get_arg_val<uint32_t>(i++);
    e.hops = get_arg_val<uint32_t>(i++);
    e.peer_x = get_arg_val<uint32_t>(i++);
    e.peer_y = get_arg_val<uint32_t>(i++);
    const uint32_t phases = get_arg_val<uint32_t>(i++);
    e.rounds = get_arg_val<uint32_t>(i++);
    e.pages = get_arg_val<uint32_t>(i++);
    e.page = get_arg_val<uint32_t>(i++);
    e.window = get_arg_val<uint32_t>(i++);
    e.batch = get_arg_val<uint32_t>(i++);
    e.check = get_arg_val<uint32_t>(i++);
    e.seed = get_arg_val<uint32_t>(i++);
    e.limit = static_cast<uint64_t>(get_arg_val<uint32_t>(i++)) << 20;
    const uint32_t delay = get_arg_val<uint32_t>(i++);   // ticks before the first send
    const uint32_t expect = get_arg_val<uint32_t>(i++);  // kArgs again
    e.status = e.word(kStatus);
    for (uint32_t w = 0; w < kStatusWords; w++) {
        e.status[w] = 0;
    }
    e.status[kWordMagic] = kMagic;
    if (
        i != kArgs || get_compile_time_arg_val(0) != kArgs || expect != kArgs ||
        e.page % 16 != 0 || e.window == 0 || e.batch == 0 || e.batch > e.window
    ) {
        e.status[kWordCode] = kCodeArgs;
        return;
    }
    km::LinkSender& sender = km::link_sender();
    e.sender = &sender;
    sender.connect(channel, e.base + kSender);
    sender.open();
    const uint64_t start = km::wall64() + delay;
    while (km::wall64() < start) {
        ;
    }
    bool ok = true;
    if (phases & kPhasePing) {
        e.status[kWordStep] = kPhasePing;
        for (uint32_t r = 1; ok && r <= e.rounds; r++) {
            const uint64_t t0 = km::wall64();
            ok = e.near ? e.ping(r) && e.pong_landed(r) : e.pong_landed(r) && e.ping(r);
            if (ok && e.near) {
                e.timed(kWordPingMin, km::wall64() - t0);
                e.status[kWordRounds] = r;
            }
        }
    }
    if (ok && (phases & kPhaseLine)) {
        e.status[kWordStep] = kPhaseLine;
        volatile tt_l1_ptr uint32_t* done = e.word(kCounts + kCountDone);
        for (uint32_t r = 1; ok && r <= e.rounds; r++) {
            const uint64_t t0 = km::wall64();
            if (e.near) {
                ok = e.line(r) &&
                    e.wait(
                        [done, r] {
                            invalidate_l1_cache();
                            return *done == r;
                        },
                        kCodeWait
                    );
            } else {
                ok = e.line_landed(r) && e.done(r);
            }
            if (ok) {
                if (e.near) {
                    e.timed(kWordLineMin, km::wall64() - t0);
                }
                e.status[kWordLines] = r;
            }
        }
    }
    if (ok && (phases & kPhasePages)) {
        e.status[kWordStep] = kPhasePages;
        ok = e.near ? e.send_pages() : e.take_pages();
    }
    e.sent();
    sender.close_start();
    if (!e.wait(
        [&sender, &e] {
            return sender.closed(e.base + kSender);
        },
        kCodeClose
    )) {
        return;
    }
}
