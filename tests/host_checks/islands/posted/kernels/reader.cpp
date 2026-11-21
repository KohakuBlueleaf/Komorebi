// A posted write into another island's L1, reader: rank 0 of the group sends rows to
// rank 1 by km::write_l1 (confirmed or km::posted), rank 1 checks them and answers.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/links/km_far.hpp"
#include "komorebi/device/movement/km_port.hpp"
#include "komorebi/device/tensors/km_ask.hpp"
#include "posted.h"

namespace {

    using Pages = km::Tensor<0>;

    constexpr uint32_t kPortRow = 1;    // the receiver's: a row landed
    constexpr uint32_t kPortPing = 2;   // the receiver's: the clock's ping
    constexpr uint32_t kPortOrder = 3;  // the receiver's: the second of two posted
    constexpr uint32_t kPortFence = 4;  // the receiver's: the confirmed call after
    constexpr uint32_t kPortBack = 5;   // the sender's: an answer landed

    // the small words' places, 16 B each, at one address on every core
    constexpr uint32_t kPing = 0, kPong = 16, kAnswer = 32, kSent = 48, kFence = 64;

    // word i of the row made for `salt`: different in every round and every word
    uint32_t pattern(uint32_t salt, uint32_t i) {
        uint32_t x = salt * 0x9E3779B1u + i * 0x85EBCA6Bu + 0x7F4A7C15u;
        return x ^ x >> 15;
    }

    void fill(uint32_t at, uint32_t bytes, uint32_t salt) {
        volatile tt_l1_ptr uint32_t* row = km::l1(at);
        for (uint32_t i = 0; i < bytes / 4; i++) {
            row[i] = pattern(salt, i);
        }
    }

    // the words of `bytes` at `at` that are not the row made for `salt`
    uint32_t wrong(uint32_t at, uint32_t bytes, uint32_t salt) {
        invalidate_l1_cache();
        const volatile tt_l1_ptr uint32_t* row = km::l1(at);
        uint32_t bad = 0;
        for (uint32_t i = 0; i < bytes / 4; i++) {
            bad += row[i] != pattern(salt, i) ? 1 : 0;
        }
        return bad;
    }

    // 16 B at `words` + `at` into the same place of `rank`, 1 at its port (confirmed)
    bool tell(
        km::Reader& reader,
        const km_command_t& command,
        uint32_t rank,
        uint32_t words,
        uint32_t at,
        uint32_t port
    ) {
        return (
            km::write_l1(
                reader,
                command,
                rank,
                words + at,
                reinterpret_cast<const void*>(words + at),
                16,
                static_cast<int>(port)
            )
        );
    }

    void put4(uint32_t at, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        volatile tt_l1_ptr uint32_t* w = km::l1(at);
        w[0] = a, w[1] = b, w[2] = c, w[3] = d;
    }

    uint64_t wide(uint32_t at) {
        invalidate_l1_cache();
        return static_cast<uint64_t>(km::l1(at)[1]) << 32 | km::l1(at)[0];
    }

    // the sender: per round the clocks' offset (a ping and its pong), then the row
    bool send(km::Reader& reader, const km_command_t& command, uint32_t to) {
        const bool posted = command.a1 != 0;
        const uint32_t bytes = command.a2, rounds = command.a3;
        const uint32_t rows = get_write_ptr(kCbRows);
        const uint32_t landing = get_write_ptr(kCbLanding);
        const uint32_t words = get_write_ptr(kCbWords);
        const uint32_t results = get_write_ptr(kCbResults);
        put4(words + kSent, 0, 0, 0, 0);
        for (uint32_t k = 0; k < rounds; k++) {
            fill(rows, bytes, k);  // the last row landed: its bytes are free
            const uint64_t a0 = km::wall64();
            if (
                !tell(reader, command, to, words, kPing, kPortPing) ||
                !km::port_wait(reader, command, kPortBack, 1)
            ) {
                return false;
            }
            const uint64_t a3 = km::wall64();
            const int64_t offset = (
                (
                    static_cast<int64_t>(wide(words + kPong) - a0) +
                    static_cast<int64_t>(wide(words + kPong + 8) - a3)
                ) >>
                1
            );
            const uint64_t c0 = km::wall64();
            const bool ok = (
                posted
                    ? km::write_l1(
                        reader,
                        command,
                        to,
                        landing,
                        reinterpret_cast<const void*>(rows),
                        bytes,
                        static_cast<int>(kPortRow),
                        km::posted
                    )
                    : km::write_l1(
                        reader,
                        command,
                        to,
                        landing,
                        reinterpret_cast<const void*>(rows),
                        bytes,
                        static_cast<int>(kPortRow)
                    )
            );
            const uint64_t c1 = km::wall64();
            if (!ok || !km::port_wait(reader, command, kPortBack, 1)) {
                return false;
            }
            const uint64_t e = km::wall64();
            const int64_t got = static_cast<int64_t>(wide(words + kAnswer) - c0);
            put4(
                results + k * 16,
                static_cast<uint32_t>(c1 - c0),
                static_cast<uint32_t>(got - offset),
                static_cast<uint32_t>(e - c0),
                km::l1(words + kAnswer)[2]
            );
        }
        if (posted) {  // two posted, the second's port: both landed; then the fence
            const uint32_t second = rows + kRowBytes, third = second + kRowBytes;
            fill(rows, bytes, 0xA000u), fill(second, bytes, 0xB000u);
            const uint64_t f0 = km::wall64();
            if (!km::write_l1(
                reader,
                command,
                to,
                landing,
                reinterpret_cast<const void*>(rows),
                bytes,
                -1,
                km::posted
            ) ||
                !km::write_l1(
                    reader,
                    command,
                    to,
                    landing + kRowBytes,
                    reinterpret_cast<const void*>(second),
                    bytes,
                    static_cast<int>(kPortOrder),
                    km::posted
                )) {
                return false;
            }
            const uint64_t f1 = km::wall64();
            if (!km::port_wait(reader, command, kPortBack, 1)) {  // checked there
                return false;
            }
            fill(third, bytes, 0xC000u), fill(rows, bytes, 0xD000u);
            if (!km::write_l1(
                reader,
                command,
                to,
                landing,
                reinterpret_cast<const void*>(third),
                bytes,
                -1,
                km::posted
            ) ||
                !km::write_l1(
                    reader,
                    command,
                    to,
                    landing + kRowBytes,
                    reinterpret_cast<const void*>(rows),
                    bytes,
                    -1,
                    km::posted
                )) {
                return false;
            }
            const uint64_t g0 = km::wall64();
            if (!tell(reader, command, to, words, kFence, kPortFence)) {
                return false;
            }
            const uint64_t g1 = km::wall64();
            fill(third, bytes, 0xEEEEu), fill(rows, bytes, 0xEEEEu);  // free now
            put4(
                words + kSent,
                static_cast<uint32_t>(f1 - f0),
                static_cast<uint32_t>(g1 - g0),
                0,
                0
            );
            if (!km::port_wait(reader, command, kPortBack, 1)) {
                return false;
            }
        }
        if (!km::write_pages(
            reader, command, Pages(command.a0), 0, kResultPages, results
        ) ||
            !km::write_pages(reader, command, Pages(command.a0), kOutWords, 1, words)) {
            return false;
        }
        noc_async_writes_flushed();
        return true;
    }

    // the receiver: answers the ping with its clock, checks each row and answers
    bool receive(km::Reader& reader, const km_command_t& command, uint32_t from) {
        const bool posted = command.a1 != 0;
        const uint32_t bytes = command.a2, rounds = command.a3;
        const uint32_t landing = get_write_ptr(kCbLanding);
        const uint32_t words = get_write_ptr(kCbWords);
        uint32_t bad = 0, order = 0, fence = 0;
        for (uint32_t k = 0; k < rounds; k++) {
            if (!km::port_wait(reader, command, kPortPing, 1)) {
                return false;
            }
            const uint64_t b1 = km::wall64();
            const uint64_t b2 = km::wall64();
            put4(
                words + kPong,
                static_cast<uint32_t>(b1),
                static_cast<uint32_t>(b1 >> 32),
                static_cast<uint32_t>(b2),
                static_cast<uint32_t>(b2 >> 32)
            );
            if (
                !tell(reader, command, from, words, kPong, kPortBack) ||
                !km::port_wait(reader, command, kPortRow, 1)
            ) {
                return false;
            }
            const uint64_t d1 = km::wall64();
            const uint32_t here = wrong(landing, bytes, k);
            bad += here;
            put4(
                words + kAnswer,
                static_cast<uint32_t>(d1),
                static_cast<uint32_t>(d1 >> 32),
                here,
                k
            );
            if (!tell(reader, command, from, words, kAnswer, kPortBack)) {
                return false;
            }
        }
        if (posted) {
            if (!km::port_wait(reader, command, kPortOrder, 1)) {
                return false;
            }
            order = (
                wrong(landing, bytes, 0xA000u) +
                wrong(landing + kRowBytes, bytes, 0xB000u)
            );
            if (
                !tell(reader, command, from, words, kAnswer, kPortBack) ||
                !km::port_wait(reader, command, kPortFence, 1)
            ) {
                return false;
            }
            fence = (
                wrong(landing, bytes, 0xC000u) +
                wrong(landing + kRowBytes, bytes, 0xD000u)
            );
            if (!tell(reader, command, from, words, kAnswer, kPortBack)) {
                return false;
            }
        }
        put4(words + kSent, bad, order, fence, rounds);
        if (
            !km::write_pages(reader, command, Pages(command.a0), kOutWords, 1, words) ||
            !km::write_pages(
                reader, command, Pages(command.a0), kOutRow, kRowBytes / kPage, landing
            )
        ) {
            return false;
        }
        noc_async_writes_flushed();
        return true;
    }

}  // namespace

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerRounds) {
            return;
        }
        const km::Group group = km::group_of(command, reader.me());
        if (command.a2 > kRowBytes || command.a3 > kRoundsMax || group.size != 2) {
            return;
        }
        if (group.rank == 0) {
            send(reader, command, group.base + 1);
        } else {
            receive(reader, command, group.base);
        }
    });
}
