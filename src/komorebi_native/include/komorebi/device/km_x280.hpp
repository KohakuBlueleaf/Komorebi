// A core's port to the X280 core that serves it: its tells and answers land in this
// core's KM_CONTROL_X280 lines; the core's asks and tells go to its mailbox.
#pragma once

#include <stdint.h>

#include "km_core.hpp"

namespace km {

    // Where codes of the X280 port (the gave-up word of the kernel that sent).
    constexpr uint32_t kWhereX280Send = 36;    // a store to the X280 core did not leave
    constexpr uint32_t kWhereX280Answer = 37;  // no answer to an ask in time
    constexpr uint32_t kWhereX280Tell = 38;    // no tell from the X280 core in time
    constexpr uint32_t kWhereX280None = 39;  // the program gave this core no X280 core
    static_assert(
        kWhereX280Send > kWhereMessagesCounter, "after the dynamic where codes"
    );

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
namespace km {

    // One message: a sequence number and three words.
    struct X280Msg {
        uint32_t sequence;
        uint32_t w[3];
    };

    // The port, owned by one data-movement kernel (`word`: its gave-up state word,
    // kStReader or kStWriter). Only that kernel sends on it; any may look.
    class X280Port {
    public:
        explicit X280Port(const Core& core, uint32_t word = kStReader) :
            core_(core), word_(word) {
        }

        // Whether the program gave this core an X280 core.
        bool served() const {
            return core_.control(KM_CONTROL_X280 + KM_X280_TO)[KM_X280_TO_BOX] != 0;
        }

        // The X280 core's newest tell, if it is newer than the last one looked at: true
        // and `message`. Never waits; tells in between are lost (latest wins).
        bool told(X280Msg& message) {
            invalidate_l1_cache();
            if (!read(KM_X280_TELL, message) || message.sequence == told_) {
                return false;
            }
            told_ = message.sequence;
            return true;
        }

        // A tell newer than the last one looked at, waiting at most the wait limit.
        // False: gave up (recorded).
        bool wait_told(X280Msg& message) {
            const uint32_t code = core_.poll_at(
                kWhereX280Tell,
                [&] {
                    return read(KM_X280_TELL, message) && message.sequence != told_;
                },
                core_.args().wait_limit
            );
            if (code != kGaveNone) {
                core_.gave_up(word_, code, kWhereX280Tell, 0, told_);
                return false;
            }
            told_ = message.sequence;
            return true;
        }

        // Tells the X280 core three words; it sees the newest at its next look. False:
        // gave up (recorded).
        bool tell(uint32_t first_word, uint32_t second_word, uint32_t third_word) {
            return send(KM_BOX_TELL, ++tells_, first_word, second_word, third_word);
        }

        // Asks the X280 core: sends the ask, returns its sequence number (0: gave up).
        uint32_t ask(uint32_t first_word, uint32_t second_word, uint32_t third_word) {
            return (
                send(KM_BOX_ASK, ++asks_, first_word, second_word, third_word)
                    ? asks_
                    : 0
            );
        }

        // The answer to ask `sequence`, if it landed: true and `message`. Never waits.
        bool answered(uint32_t sequence, X280Msg& message) const {
            invalidate_l1_cache();
            return read(KM_X280_ANSWER, message) && message.sequence == sequence;
        }

        // Waits (with a timeout) for the answer to ask `sequence`. False: gave up
        // (recorded).
        bool wait_answer(uint32_t sequence, X280Msg& message) const {
            const uint32_t code = core_.poll_at(
                kWhereX280Answer,
                [&] {
                    return (
                        read(KM_X280_ANSWER, message) && message.sequence == sequence
                    );
                },
                core_.args().wait_limit
            );
            if (code != kGaveNone) {
                core_.gave_up(word_, code, kWhereX280Answer, 0, sequence);
                return false;
            }
            return true;
        }

        // The pull: the next range of units the X280 core hands this core, `first` and
        // the count; 0 when none is left or a wait gave up (pull_gave_up(), recorded).
        uint32_t next(uint32_t& first) {
            uint32_t word;
            return next(first, word);
        }

        // The same, and the word the X280 core's pick_word gave with the range (0 from
        // the counter, the log or a pick without a word).
        uint32_t next(uint32_t& first, uint32_t& word) {
            word = 0;
            if (pull_ == 0) {
                pull_ = ask(KM_PULL_ASK, 0, 0);
            }
            X280Msg message;
            const uint32_t sequence = pull_;
            pull_ = 0;
            if (sequence == 0 || !wait_answer(sequence, message)) {
                gave_ = true;
                return 0;
            }
            if (message.w[1] == 0) {
                return 0;
            }
            // asked now, answered while this range runs: one ask in flight
            pull_ = ask(KM_PULL_ASK, 0, 0);
            first = message.w[0];
            word = message.w[2];
            return message.w[1];
        }

        // Whether the last next() returned 0 because a wait gave up.
        bool pull_gave_up() const {
            return gave_;
        }

        // ask, then wait_answer. False: gave up (recorded).
        bool call(
            uint32_t first_word,
            uint32_t second_word,
            uint32_t third_word,
            X280Msg& message
        ) {
            const uint32_t sequence = ask(first_word, second_word, third_word);
            return sequence != 0 && wait_answer(sequence, message);
        }

    private:
        // A 16-byte line of KM_CONTROL_X280 as one message; false while sequence is 0.
        bool read(uint32_t line, X280Msg& message) const {
            volatile tt_l1_ptr uint32_t* words = core_.control(KM_CONTROL_X280 + line);
            message.sequence = words[0];
            message.w[0] = words[1];
            message.w[1] = words[2];
            message.w[2] = words[3];
            return message.sequence != 0;
        }

        // Stores {sequence, three words} into mailbox line `line`: the last store must
        // have left before KM_X280_OUT is written again.
        bool send(
            uint32_t line,
            uint32_t sequence,
            uint32_t first_word,
            uint32_t second_word,
            uint32_t third_word
        ) {
            volatile tt_l1_ptr uint32_t* to_x280 = (
                core_.control(KM_CONTROL_X280 + KM_X280_TO)
            );
            const uint32_t coordinates = to_x280[KM_X280_TO_TILE];
            const uint32_t box = to_x280[KM_X280_TO_BOX];
            if (box == 0) {
                core_.gave_up(word_, kGaveBadItem, kWhereX280None, 0, sequence);
                return false;
            }
            const uint32_t code = core_.poll_at(
                kWhereX280Send,
                [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                },
                core_.args().wait_limit
            );
            if (code != kGaveNone) {
                core_.gave_up(word_, code, kWhereX280Send, 0, sequence);
                return false;
            }
            volatile tt_l1_ptr uint32_t* out = (
                core_.control(KM_CONTROL_X280 + KM_X280_OUT)
            );
            out[0] = sequence;
            out[1] = first_word;
            out[2] = second_word;
            out[3] = third_word;
            noc_async_write(
                core_.args().control + KM_CONTROL_X280 + KM_X280_OUT,
                get_noc_addr(coordinates & 0xFFFFu, coordinates >> 16, box + line),
                16
            );
            return true;
        }

        const Core& core_;
        uint32_t word_;
        uint32_t told_ = 0;
        uint32_t tells_ = 0;
        uint32_t asks_ = 0;
        uint32_t pull_ = 0;  // the pull ask in flight, 0: none
        bool gave_ = false;
    };

}  // namespace km
#endif
