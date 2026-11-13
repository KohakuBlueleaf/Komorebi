// A core's port to the X280 core that serves it: the X280 core's tells and answers land
// in this core's KM_CTL_X280 lines; the core's asks and tells go to its mailbox line.
#pragma once

#include <stdint.h>

#include "km_core.hpp"

namespace km {

    // Where codes of the X280 port (the gave-up word of the kernel that sent).
    constexpr uint32_t kWhereX280Send = 36;    // a store to the X280 core did not leave
    constexpr uint32_t kWhereX280Answer = 37;  // no answer to an ask in time
    constexpr uint32_t kWhereX280Tell = 38;    // no tell from the X280 core in time
    constexpr uint32_t kWhereX280None = 39;  // the program gave this core no X280 core
    static_assert(kWhereX280Send > kWhereDynCounter, "after the dynamic where codes");

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
namespace km {

    // One message: a sequence number and three words.
    struct X280Msg {
        uint32_t seq;
        uint32_t w[3];
    };

    // The port, owned by one data-movement kernel (`word`: its gave-up state word,
    // kStReader or kStWriter). Only that kernel sends on it; any may look.
    class X280Port {
    public:
        explicit X280Port(const Core& c, uint32_t word = kStReader)
            : c_(c), word_(word) {
        }

        // Whether the program gave this core an X280 core.
        bool served() const {
            return c_.ctl(KM_CTL_X280 + KM_X280_TO)[KM_X280_TO_BOX] != 0;
        }

        // The X280 core's newest tell, if it is newer than the last one looked at: true
        // and `m`. Never waits; tells in between are lost (latest wins).
        bool told(X280Msg& m) {
            invalidate_l1_cache();
            if (!read(KM_X280_TELL, m) || m.seq == told_) {
                return false;
            }
            told_ = m.seq;
            return true;
        }

        // A tell newer than the last one looked at, waiting at most the wait limit.
        // False: gave up (recorded).
        bool wait_told(X280Msg& m) {
            const uint32_t code = c_.poll(
                [&] {
                    return read(KM_X280_TELL, m) && m.seq != told_;
                },
                c_.args().wait_limit
            );
            if (code != kGaveNone) {
                c_.gave_up(word_, code, kWhereX280Tell, 0, told_);
                return false;
            }
            told_ = m.seq;
            return true;
        }

        // Tells the X280 core three words; it sees the newest at its next look. False:
        // gave up (recorded).
        bool tell(uint32_t a, uint32_t b, uint32_t d) {
            return send(KM_BOX_TELL, ++tells_, a, b, d);
        }

        // Asks the X280 core: sends the ask, returns its sequence number (0: gave up).
        uint32_t ask(uint32_t a, uint32_t b, uint32_t d) {
            return send(KM_BOX_ASK, ++asks_, a, b, d) ? asks_ : 0;
        }

        // The answer to ask `seq`, if it landed: true and `m`. Never waits.
        bool answered(uint32_t seq, X280Msg& m) const {
            invalidate_l1_cache();
            return read(KM_X280_ANSWER, m) && m.seq == seq;
        }

        // Waits (bounded) for the answer to ask `seq`. False: gave up (recorded).
        bool wait_answer(uint32_t seq, X280Msg& m) const {
            const uint32_t code = c_.poll(
                [&] {
                    return read(KM_X280_ANSWER, m) && m.seq == seq;
                },
                c_.args().wait_limit
            );
            if (code != kGaveNone) {
                c_.gave_up(word_, code, kWhereX280Answer, 0, seq);
                return false;
            }
            return true;
        }

        // ask, then wait_answer. False: gave up (recorded).
        bool call(uint32_t a, uint32_t b, uint32_t d, X280Msg& m) {
            const uint32_t seq = ask(a, b, d);
            return seq != 0 && wait_answer(seq, m);
        }

    private:
        // A 16-byte line of KM_CTL_X280 as one message; false while seq is 0.
        bool read(uint32_t at, X280Msg& m) const {
            volatile tt_l1_ptr uint32_t* p = c_.ctl(KM_CTL_X280 + at);
            m.seq = p[0];
            m.w[0] = p[1];
            m.w[1] = p[2];
            m.w[2] = p[3];
            return m.seq != 0;
        }

        // Stores {seq, a, b, d} into line `at` of the mailbox: the last store must
        // have left before KM_X280_OUT is written again.
        bool send(uint32_t at, uint32_t seq, uint32_t a, uint32_t b, uint32_t d) {
            volatile tt_l1_ptr uint32_t* to = c_.ctl(KM_CTL_X280 + KM_X280_TO);
            const uint32_t xy = to[KM_X280_TO_TILE], box = to[KM_X280_TO_BOX];
            if (box == 0) {
                c_.gave_up(word_, kGaveBadItem, kWhereX280None, 0, seq);
                return false;
            }
            const uint32_t code = c_.poll(
                [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                },
                c_.args().wait_limit
            );
            if (code != kGaveNone) {
                c_.gave_up(word_, code, kWhereX280Send, 0, seq);
                return false;
            }
            volatile tt_l1_ptr uint32_t* out = c_.ctl(KM_CTL_X280 + KM_X280_OUT);
            out[0] = seq;
            out[1] = a;
            out[2] = b;
            out[3] = d;
            noc_async_write(
                c_.args().ctl + KM_CTL_X280 + KM_X280_OUT,
                get_noc_addr(xy & 0xFFFFu, xy >> 16, box + at),
                16
            );
            return true;
        }

        const Core& c_;
        uint32_t word_;
        uint32_t told_ = 0;
        uint32_t tells_ = 0;
        uint32_t asks_ = 0;
    };

}  // namespace km
#endif
