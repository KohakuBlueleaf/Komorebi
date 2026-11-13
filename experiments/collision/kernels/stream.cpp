// Broadcast side, stream mode (README "Stream mode"): a fixed body multicast back to
// back with a window; bounded waits, state words stored before each library barrier.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "broadcast.h"

namespace {

    volatile tt_l1_ptr uint32_t* words(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

    inline uint64_t wall() {
        const uint32_t lo = (
            *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                RISCV_DEBUG_REG_WALL_CLOCK_L
            )
        );
        const uint32_t hi = (
            *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                RISCV_DEBUG_REG_WALL_CLOCK_H
            )
        );
        return (static_cast<uint64_t>(hi) << 32) | lo;
    }

    // True when a word with this run's tag carries a block number at or past k (16-bit
    // block numbers; the window keeps every comparison within 2^15).
    inline bool reached(uint32_t v, uint32_t k, uint32_t tag) {
        return (
            (v >> 17) == (tag & 0x7FFFu) &&
            static_cast<int16_t>(static_cast<uint16_t>(v - k)) >= 0
        );
    }

    struct Broadcast {
        uint32_t base;
        uint64_t limit;
        uint64_t wait_max = 0;

        volatile tt_l1_ptr uint32_t& at(uint32_t off) const {
            return *words(base + off);
        }

        void state(uint32_t phase, uint32_t k) const {
            at(BROADCAST_STATE) = phase;
            at(BROADCAST_STATE + 4) = k;
        }

        bool read_peer(uint32_t x, uint32_t y) {
            at(BROADCAST_PEER) = 0xFFFFFFFFu;
            at(BROADCAST_PEER + 4) = 0xFFFFFFFFu;
            noc_async_read(
                get_noc_addr(x, y, base + BROADCAST_STATE), base + BROADCAST_PEER, 16
            );
            const uint64_t t0 = wall();
            while (!ncrisc_noc_reads_flushed(noc_index)) {
                if (wall() - t0 > limit) {
                    return false;
                }
            }
            invalidate_l1_cache();
            return true;
        }

        void note_wait(uint64_t t0) {
            const uint64_t dt = wall() - t0;
            wait_max = dt > wait_max ? dt : wait_max;
        }
    };

}  // namespace

void kernel_main() {
    uint32_t a = 0;
    const uint32_t cnt_dram = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t tag = get_arg_val<uint32_t>(a++);
    const uint32_t blocks = get_arg_val<uint32_t>(a++);
    const uint32_t block_bytes = get_arg_val<uint32_t>(a++);
    const bool until = get_arg_val<uint32_t>(a++) != 0;
    const uint32_t role = get_arg_val<uint32_t>(a++);
    const uint32_t rank = get_arg_val<uint32_t>(a++);
    const uint32_t receivers = get_arg_val<uint32_t>(a++);
    const uint32_t snd_x = get_arg_val<uint32_t>(a++);
    const uint32_t snd_y = get_arg_val<uint32_t>(a++);
    // 1: rank r at snd_y + r, 0: at snd_y - r; 3, 2: a row broadcast (peers not read).
    const uint32_t step = get_arg_val<uint32_t>(a++);
    const uint32_t mx0 = get_arg_val<uint32_t>(a++), my0 = get_arg_val<uint32_t>(a++);
    const uint32_t mx1 = get_arg_val<uint32_t>(a++), my1 = get_arg_val<uint32_t>(a++);
    const uint32_t wait_cycles = get_arg_val<uint32_t>(a++);
    const uint32_t window = get_arg_val<uint32_t>(a++);
    const uint32_t every = get_arg_val<uint32_t>(a++);  // full check every N blocks

    Broadcast broadcast{get_write_ptr(kBroadcastCb), wait_cycles};
    volatile tt_l1_ptr uint32_t* done = words(get_write_ptr(kShareCb));
    const uint32_t body_words = (block_bytes - STREAM_HEADER) / 4;
    auto slot = [&](uint32_t k) {
        return broadcast.base + BROADCAST_BLOCK + (k % window) * block_bytes;
    };

    uint32_t n_ok = 0, bad_blocks = 0, bad_words = 0, bad_k = 0, bad_w = 0, bad_v = 0;
    uint32_t where = 0, where_k = 0, missing = 0, peer = 0, peer_phase = 0, peer_k = 0;
    uint32_t stopped = 0, full = 0;
    uint64_t t_begin = 0, t_end = 0, busy = 0, check = 0;

    broadcast.state(LP_START, 0);
    if (role == 1) {
        // Sender. All stamps at least k: true, or false after the limit.
        auto wait_markers = [&](uint32_t k) -> bool {
            const uint64_t t0 = wall();
            uint32_t r = 1;
            for (;;) {
                invalidate_l1_cache();
                while (
                    r <= receivers &&
                    reached(broadcast.at(BROADCAST_DONE + 16 * r), k, tag)
                ) {
                    r++;
                }
                if (r > receivers) {
                    broadcast.note_wait(t0);
                    return true;
                }
                if (wall() - t0 > broadcast.limit) {
                    missing = 0;
                    for (uint32_t q = 1; q <= receivers; q++) {
                        missing += (
                            !reached(broadcast.at(BROADCAST_DONE + 16 * q), k, tag)
                        );
                    }
                    peer = r;
                    return false;
                }
            }
        };
        const uint64_t mvalid = (
            get_noc_multicast_addr(mx0, my0, mx1, my1, broadcast.base + BROADCAST_VALID)
        );
        for (uint32_t s = 0; s < window; s++) {
            volatile tt_l1_ptr uint32_t* body = words(slot(s) + STREAM_HEADER);
            for (uint32_t w = 0; w < body_words; w++) {
                body[w] = stream_body(w, tag);
            }
        }
        broadcast.state(LP_WAIT_MARKERS, 0);
        bool ok = wait_markers(0);
        if (!ok) {
            where = LW_READY;
        }
        t_begin = wall();
        uint32_t k = 1;
        for (; ok && k <= blocks; k++) {
            if (k > window) {
                broadcast.state(LP_WAIT_MARKERS, k - window);
                ok = wait_markers(k - window);
                if (!ok) {
                    where = LW_MARKER;
                    break;
                }
            }
            if (until) {
                invalidate_l1_cache();
                if (*done == broadcast_commands_done(tag)) {
                    stopped = 1;
                    break;
                }
            }
            const uint32_t src = slot(k);
            words(src)[0] = broadcast_marker(k, tag);
            words(src)[1] = k;
            broadcast.state(LP_BARRIER_BLOCK, k);
            const uint64_t t0 = wall();
            noc_async_write_multicast(
                src,
                get_noc_multicast_addr(mx0, my0, mx1, my1, src),
                block_bytes,
                receivers
            );
            noc_async_write_barrier();
            busy += wall() - t0;
            broadcast.at(BROADCAST_VALID) = broadcast_marker(k, tag);
            noc_semaphore_set_multicast(
                broadcast.base + BROADCAST_VALID, mvalid, receivers
            );
        }
        const uint32_t last = k - 1;  // blocks sent
        broadcast.state(LP_BARRIER_VALID, last);
        noc_async_write_barrier();
        if (ok && stopped) {
            broadcast.at(BROADCAST_VALID) = broadcast_stop(last, tag);
            noc_semaphore_set_multicast(
                broadcast.base + BROADCAST_VALID, mvalid, receivers
            );
            noc_async_write_barrier();
        }
        if (ok) {
            broadcast.state(LP_WAIT_MARKERS, last);
            ok = wait_markers(last);
            if (!ok) {
                where = LW_MARKER;
            }
        }
        t_end = wall();
        n_ok = ok ? last : 0;
        if (!ok) {
            where_k = where == LW_READY ? 0 : k;
            broadcast.state(LP_GAVE_UP, where_k);
            const uint32_t py = step & 1 ? snd_y + peer : snd_y - peer;
            if ((step & 2) == 0 && broadcast.read_peer(snd_x, py)) {
                peer_phase = broadcast.at(BROADCAST_PEER);
                peer_k = broadcast.at(BROADCAST_PEER + 4);
            } else {
                peer_phase = 0xFFFFFFFFu;  // the read did not complete
                peer_k = 0xFFFFFFFFu;
            }
        }
    } else if (role == 2) {
        // Receiver: clear VALID, poison every slot, stamp block 0 (ready).
        auto poison = [&](uint32_t s) {
            volatile tt_l1_ptr uint32_t* body = words(slot(s) + STREAM_HEADER);
            for (uint32_t w = 0; w < body_words; w++) {
                body[w] = ~stream_body(w, tag);
            }
        };
        broadcast.at(BROADCAST_VALID) = 0;
        for (uint32_t s = 0; s < window; s++) {
            poison(s);
        }
        const uint64_t my_slot = (
            get_noc_addr(snd_x, snd_y, broadcast.base + BROADCAST_DONE + 16 * rank)
        );
        auto stamp = [&](uint32_t k) {
            broadcast.at(BROADCAST_SRC) = broadcast_marker(k, tag);
            broadcast.state(LP_BARRIER_MARKER, k);
            noc_async_write(broadcast.base + BROADCAST_SRC, my_slot, 4);
            noc_async_write_barrier();
        };
        stamp(0);
        t_begin = wall();
        for (uint32_t k = 1; k <= blocks; k++) {
            broadcast.state(LP_WAIT_VALID, k);
            const uint64_t t0 = wall();
            uint32_t v = 0;
            bool have = false;
            for (;;) {
                invalidate_l1_cache();
                v = broadcast.at(BROADCAST_VALID);
                have = reached(v, k, tag);
                if (have || (v & (1u << 16)) != 0 || wall() - t0 > broadcast.limit) {
                    break;
                }
            }
            if (!have && (v >> 17) == (tag & 0x7FFFu) && (v & (1u << 16)) != 0) {
                stopped = 1;  // the stop flag: block k was never sent
                break;
            }
            if (!have) {
                where = LW_VALID;
                where_k = k;
                broadcast.state(LP_GAVE_UP, k);
                peer = 0;
                if (broadcast.read_peer(snd_x, snd_y)) {
                    peer_phase = broadcast.at(BROADCAST_PEER);
                    peer_k = broadcast.at(BROADCAST_PEER + 4);
                } else {
                    peer_phase = 0xFFFFFFFFu;  // the read did not complete
                    peer_k = 0xFFFFFFFFu;
                }
                break;
            }
            broadcast.note_wait(t0);
            broadcast.state(LP_VERIFY, k);
            volatile tt_l1_ptr uint32_t* blk = words(slot(k));
            uint32_t bad = 0;
            if (blk[0] != broadcast_marker(k, tag) || blk[1] != k) {
                if (bad_words == 0) {
                    bad_k = k;
                    bad_w = blk[0] != broadcast_marker(k, tag) ? 0 : 1;
                    bad_v = blk[bad_w];
                }
                bad++;
                bad_words++;
            }
            if (k <= window || k % every == 0) {
                const uint64_t c0 = wall();
                volatile tt_l1_ptr uint32_t* body = blk + STREAM_HEADER / 4;
                for (uint32_t w = 0; w < body_words; w++) {
                    const uint32_t got = body[w];
                    if (got != stream_body(w, tag)) {
                        if (bad_words == 0) {
                            bad_k = k;
                            bad_w = w + STREAM_HEADER / 4;
                            bad_v = got;
                        }
                        bad++;
                        bad_words++;
                    }
                }
                full++;
                check += wall() - c0;
            }
            // Block k + W lands in this slot next: poison it if that one is checked.
            if ((k + window) % every == 0) {
                const uint64_t c0 = wall();
                poison(k);
                check += wall() - c0;
            }
            bad_blocks += bad != 0;
            n_ok++;
            stamp(k);
        }
        t_end = wall();
    }
    broadcast.state(where ? LP_GAVE_UP : LP_END, where_k);

    volatile tt_l1_ptr uint32_t* c = words(get_write_ptr(kBroadcastCnt));
    for (uint32_t w = 0; w < LN_WORDS; w++) {
        c[w] = 0;
    }
    const uint64_t span = t_end - t_begin;
    c[LN_ROLE] = role;
    c[LN_BLOCKS] = n_ok;
    c[LN_BAD_BLOCKS] = bad_blocks;
    c[LN_BAD_WORDS] = bad_words;
    c[LN_BAD_K] = bad_k;
    c[LN_BAD_W] = bad_w;
    c[LN_BAD_V] = bad_v;
    c[LN_WHERE] = where;
    c[LN_WHERE_K] = where_k;
    c[LN_MISSING] = missing;
    c[LN_PEER] = peer;
    c[LN_PEER_PHASE] = peer_phase;
    c[LN_PEER_K] = peer_k;
    c[LN_T_LO] = static_cast<uint32_t>(span);
    c[LN_T_HI] = static_cast<uint32_t>(span >> 32);
    c[LN_STOPPED] = stopped;
    c[LN_TAG] = tag;
    c[LN_WAIT_MAX] = static_cast<uint32_t>(broadcast.wait_max);
    c[LN_BUSY_LO] = static_cast<uint32_t>(busy);
    c[LN_BUSY_HI] = static_cast<uint32_t>(busy >> 32);
    c[LN_CHECK_LO] = static_cast<uint32_t>(check);
    c[LN_CHECK_HI] = static_cast<uint32_t>(check >> 32);
    c[LN_FULL] = full;
    c[LN_MAGIC] = BROADCAST_MAGIC | me;
    const InterleavedAddrGen<true> dst = {
        .bank_base_address = cnt_dram, .page_size = LN_WORDS * 4
    };
    noc_async_write(get_write_ptr(kBroadcastCnt), get_noc_addr(me, dst), LN_WORDS * 4);
    noc_async_write_barrier();
}
