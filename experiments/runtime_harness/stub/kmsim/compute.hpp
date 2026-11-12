// Stub of the compute kernel API. The kernel is built once per thread (TRISC_UNPACK,
// TRISC_MATH, TRISC_PACK); each call acts only on the thread that runs it on the card.
#pragma once

#include <stdint.h>

#include "kmsim/cb_interface.hpp"
#include "kmsim/sim.hpp"

#if defined(TRISC_UNPACK)
#define UNPACK(...) __VA_ARGS__
#else
#define UNPACK(...)
#endif
#if defined(TRISC_MATH)
#define MATH(...) __VA_ARGS__
#else
#define MATH(...)
#endif
#if defined(TRISC_PACK)
#define PACK(...) __VA_ARGS__
#else
#define PACK(...)
#endif

// Runtime argument i of this kernel (the same list on all three threads).
template <typename T>
static inline T get_arg_val(int i) {
    kmsim::point("get_arg_val");
    return static_cast<T>(kmsim::arg(static_cast<uint32_t>(i)));
}

// A scheduling point; the L1 data cache itself is not modelled.
static inline void invalidate_l1_cache() {
    kmsim::point("invalidate_l1_cache");
    kmsim::yield();
}

// Counter registers of buffer cb, kept in step by every push and pop of the stub.
static inline volatile uint32_t* get_cb_tiles_received_ptr(uint32_t cb) {
    return kmsim::cb_received_word(cb);
}

static inline volatile uint32_t* get_cb_tiles_acked_ptr(uint32_t cb) {
    return kmsim::cb_acked_word(cb);
}

namespace ckernel {

    // A register read: the 32-bit word at addr (the stub's registers live in L1).
    static inline uint32_t reg_read(uint32_t addr) {
        kmsim::point("reg_read");
        return *reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(addr));
    }

    // L1 byte address of tile `tile` at the front of cb (the operand id is the cb id).
    static inline uint32_t get_tile_l1_byte_address(uint32_t cb, uint32_t tile) {
        return kmsim::cb_read_ptr(cb) + tile * kmsim::cb_page(cb);
    }

    enum ThreadId {
        BriscThreadId = 0,
        UnpackThreadId = 1,
        MathThreadId = 2,
        PackThreadId = 3
    };

    // A FIFO of four words from this thread to `thread`; a write to a full one waits.
    static inline void mailbox_write(uint32_t thread, uint32_t data) {
        kmsim::point("mailbox_write", thread);
        kmsim::mailbox_write(thread, data);
    }

    // Waits (unbounded) for the next word from `thread`.
    static inline uint32_t mailbox_read(uint32_t thread) {
        kmsim::point("mailbox_read", thread);
        return kmsim::mailbox_read(thread);
    }

    // Unpack reads word `element` of tile `tile` at the front and sends it to math and
    // pack, which wait for it; the front is not checked for readiness.
    static inline uint32_t read_tile_value(
        uint32_t cb, uint32_t tile, uint32_t element
    ) {
        uint32_t value = 0;
        UNPACK({
            const uint32_t at = (
                kmsim::cb_read_ptr(cb) + tile * kmsim::cb_page(cb) + element * 4
            );
            value = *reinterpret_cast<volatile uint32_t*>(at);
            mailbox_write(MathThreadId, value);
            mailbox_write(PackThreadId, value);
        })
        MATH(value = mailbox_read(UnpackThreadId);)
        PACK(value = mailbox_read(UnpackThreadId);)
        return value;
    }

}  // namespace ckernel

// Unpack: waits (unbounded) for pages at the front. Other threads: nothing.
static inline void cb_wait_front(uint32_t cb, uint32_t pages) {
    UNPACK((kmsim::point("cb_wait_front", cb), kmsim::cb_wait(cb, pages)));
}

// Unpack: frees pages at the front. Other threads: nothing.
static inline void cb_pop_front(uint32_t cb, uint32_t pages) {
    UNPACK((kmsim::point("cb_pop_front", cb), kmsim::cb_pop(cb, pages)));
}

// Pack: waits (unbounded) for free pages at the back. Other threads: nothing.
static inline void cb_reserve_back(uint32_t cb, uint32_t pages) {
    PACK((kmsim::point("cb_reserve_back", cb), kmsim::cb_reserve(cb, pages)));
}

// Pack: hands pages to the consumer. Other threads: nothing.
static inline void cb_push_back(uint32_t cb, uint32_t pages) {
    PACK((kmsim::point("cb_push_back", cb), kmsim::cb_push(cb, pages)));
}

// Hardware set-up and per-operation initialisation: nothing to model.
static inline void compute_kernel_hw_startup(uint32_t, uint32_t) {
}

static inline void copy_init(uint32_t) {
}

static inline void add_binary_tile_init() {
}

static inline void fill_tile_init() {
}

// Math: waits (unbounded) until pack released the destination registers.
static inline void tile_regs_acquire() {
    MATH((kmsim::point("tile_regs_acquire"), kmsim::dst_acquire()));
}

// Math: hands the destination registers to pack.
static inline void tile_regs_commit() {
    MATH((kmsim::point("tile_regs_commit"), kmsim::dst_commit()));
}

// Pack: waits (unbounded) until math committed the destination registers.
static inline void tile_regs_wait() {
    PACK((kmsim::point("tile_regs_wait"), kmsim::dst_wait()));
}

// Pack: gives the destination registers back to math.
static inline void tile_regs_release() {
    PACK((kmsim::point("tile_regs_release"), kmsim::dst_release()));
}

// Unpack sends the front tile to math (two source slots, waits when both are full);
// math waits (unbounded) for it and stores it in register `dst`. No data formats.
static inline void copy_tile(uint32_t cb, uint32_t tile, uint32_t dst) {
    UNPACK((kmsim::point("copy_tile", cb), kmsim::unpack_tile(cb, tile)));
    MATH((kmsim::point("copy_tile", cb), kmsim::math_take(dst)));
    (void)cb;
    (void)tile;
    (void)dst;
}

// Math: register out = a + b, element-wise in single precision (not the vector unit).
static inline void add_binary_tile(uint32_t a, uint32_t b, uint32_t out) {
    MATH((kmsim::point("add_binary_tile"), kmsim::math_add(a, b, out)));
    (void)a;
    (void)b;
    (void)out;
}

// Math: every element of register `dst` = value.
static inline void fill_tile(uint32_t dst, float value) {
    MATH((kmsim::point("fill_tile"), kmsim::math_fill(dst, value)));
    (void)dst;
    (void)value;
}

// Pack: register `dst` to the back page of cb as bfloat16 (truncated).
static inline void pack_tile(uint32_t dst, uint32_t cb) {
    PACK((kmsim::point("pack_tile", cb), kmsim::pack_tile(dst, cb)));
    (void)dst;
    (void)cb;
}
