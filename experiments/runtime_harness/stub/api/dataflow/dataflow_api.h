// Stub of the data-movement kernel API: the calls the runtime and the example use, each
// with what the simulator models and what it leaves out.
#pragma once

#include <stdint.h>

#include "kmsim/cb_interface.hpp"
#include "kmsim/sim.hpp"

// Runtime argument i of this kernel; no bounds check beyond the simulator's table.
template <typename T>
inline T get_arg_val(int i) {
    kmsim::point("get_arg_val");
    return static_cast<T>(kmsim::arg(static_cast<uint32_t>(i)));
}

// NoC address of addr on the core at x, y; any address within the receiver's L1 image.
inline uint64_t get_noc_addr(uint32_t x, uint32_t y, uint32_t addr) {
    return kmsim::noc_core(x, y, addr);
}

// Interleaved DRAM pages: one flat DRAM, no banks; page p at base + p * page_size.
template <bool DRAM>
struct InterleavedAddrGen {
    uint32_t bank_base_address;
    uint32_t page_size;
};

template <bool DRAM>
inline uint64_t get_noc_addr(uint32_t page, const InterleavedAddrGen<DRAM>& g) {
    return kmsim::noc_dram(g.bank_base_address + page * g.page_size);
}

// Tensor accessor: compile-time argument Index holds the page size; flat DRAM as above.
template <uint32_t Index>
struct TensorAccessorArgs {
    static constexpr uint32_t index = Index;
};

template <typename Args>
struct TensorAccessor {
    TensorAccessor(Args, uint32_t base)
        : base(base), page_size(kmsim::compile_arg(Args::index)) {
    }

    uint32_t base;
    uint32_t page_size;
};

// Read: the bytes are copied at the call (no read latency is modelled).
inline void noc_async_read(uint64_t src, uint32_t dst, uint32_t bytes) {
    kmsim::point("noc_async_read");
    kmsim::noc_read(src, dst, bytes);
}

template <typename Args>
inline void noc_async_read_page(
    uint32_t page, const TensorAccessor<Args>& t, uint32_t dst
) {
    kmsim::point("noc_async_read_page");
    kmsim::noc_read(kmsim::noc_dram(t.base + page * t.page_size), dst, t.page_size);
}

// Returns once this kernel's reads landed (always: reads land at the call).
inline void noc_async_read_barrier() {
    kmsim::point("noc_async_read_barrier");
    kmsim::noc_read_barrier();
}

// Write: the source is read at the call, delivered later in 16-byte chunks by the
// delivery thread, each with the case's delay; a dropped chunk never lands.
inline void noc_async_write(uint32_t src, uint64_t dst, uint32_t bytes) {
    kmsim::point("noc_async_write");
    kmsim::noc_write(src, dst, bytes);
}

template <typename Args>
inline void noc_async_write_page(
    uint32_t page, const TensorAccessor<Args>& t, uint32_t src
) {
    kmsim::point("noc_async_write_page");
    kmsim::noc_write(src, kmsim::noc_dram(t.base + page * t.page_size), t.page_size);
}

// Waits until every write of this kernel has landed or was dropped.
inline void noc_async_write_barrier() {
    kmsim::point("noc_async_write_barrier");
    kmsim::noc_write_barrier();
}

// Returns at once: the source was already read when the write was issued.
inline void noc_async_writes_flushed() {
    kmsim::point("noc_async_writes_flushed");
    kmsim::noc_writes_flushed();
}

// Atomic add on a 32-bit word of another core, delivered like a write chunk.
inline void noc_semaphore_inc(uint64_t addr, uint32_t incr) {
    kmsim::point("noc_semaphore_inc");
    kmsim::noc_inc(addr, incr);
}

// Waits until every atomic of this kernel has landed; writes are not covered.
inline void noc_async_atomic_barrier() {
    kmsim::point("noc_async_atomic_barrier");
    kmsim::noc_atomic_barrier();
}

// A scheduling point; the L1 data cache itself is not modelled.
inline void invalidate_l1_cache() {
    kmsim::point("invalidate_l1_cache");
    kmsim::yield();
}

// Free pages at the back; no other effect.
inline bool cb_pages_reservable_at_back(uint32_t cb, uint32_t pages) {
    kmsim::point("cb_pages_reservable_at_back", cb);
    return kmsim::cb_room(cb, pages);
}

// Waits (unbounded) for free pages; a reservation past the buffer's end is reported.
inline void cb_reserve_back(uint32_t cb, uint32_t pages) {
    kmsim::point("cb_reserve_back", cb);
    kmsim::cb_reserve(cb, pages);
}

// Hands reserved pages to the consumer; pushing more than was reserved is reported.
inline void cb_push_back(uint32_t cb, uint32_t pages) {
    kmsim::point("cb_push_back", cb);
    kmsim::cb_push(cb, pages);
}

// Pages ready at the front; no other effect.
inline bool cb_pages_available_at_front(uint32_t cb, uint32_t pages) {
    kmsim::point("cb_pages_available_at_front", cb);
    return kmsim::cb_ready(cb, pages);
}

// Waits (unbounded) for pages at the front.
inline void cb_wait_front(uint32_t cb, uint32_t pages) {
    kmsim::point("cb_wait_front", cb);
    kmsim::cb_wait(cb, pages);
}

// Frees pages at the front; popping more than are ready is reported.
inline void cb_pop_front(uint32_t cb, uint32_t pages) {
    kmsim::point("cb_pop_front", cb);
    kmsim::cb_pop(cb, pages);
}

// L1 address of the back (producer) and of the front (consumer) page.
inline uint32_t get_write_ptr(uint32_t cb) {
    return kmsim::cb_write_ptr(cb);
}

inline uint32_t get_read_ptr(uint32_t cb) {
    return kmsim::cb_read_ptr(cb);
}
