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
inline uint64_t get_noc_addr(uint32_t x, uint32_t y, uint32_t addr, uint8_t = 0) {
    return kmsim::noc_core(x, y, addr);
}

// NoC address of addr in the calling core's own L1.
inline uint64_t get_noc_addr(uint32_t addr) {
    return kmsim::noc_self(addr);
}

// Interleaved DRAM pages: one flat DRAM, no banks; page p at base + p * page_size.
template <bool DRAM>
struct InterleavedAddrGen {
    uint32_t bank_base_address;
    uint32_t page_size;

    uint64_t get_noc_addr(uint32_t page, uintptr_t offset = 0, uint8_t = 0) const {
        return kmsim::noc_dram(bank_base_address + page * page_size + offset);
    }
};

template <bool DRAM>
inline uint64_t get_noc_addr(uint32_t page, const InterleavedAddrGen<DRAM>& g) {
    return kmsim::noc_dram(g.bank_base_address + page * g.page_size);
}

template <bool DRAM>
inline uint64_t get_noc_addr(
    uint32_t page, const InterleavedAddrGen<DRAM>& g, uint32_t offset
) {
    return kmsim::noc_dram(g.bank_base_address + page * g.page_size + offset);
}

// An address in a DRAM bank: one flat DRAM, so every bank's copy is the same bytes.
template <bool DRAM>
inline uint64_t get_noc_addr_from_bank_id(uint32_t, uint32_t addr, uint8_t = 0) {
    return kmsim::noc_dram(addr);
}

// Tensor accessor: compile-time argument Index holds the page size; flat DRAM as above.
template <uint32_t Index>
struct TensorAccessorArgs {
    static constexpr uint32_t index = Index;
};

template <typename Args>
struct TensorAccessor {
    TensorAccessor(Args, uint32_t base) :
        base(base), page_size(kmsim::compile_arg(Args::index)) {
    }

    uint32_t get_aligned_page_size() const {
        return page_size;
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

// Returns at once: the source was already read when the write was submitted.
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

// Counter registers of buffer cb (pages pushed, pages popped), kept by the stub.
inline volatile uint32_t* get_cb_tiles_received_ptr(uint32_t cb) {
    return kmsim::cb_received_word(cb);
}

inline volatile uint32_t* get_cb_tiles_acked_ptr(uint32_t cb) {
    return kmsim::cb_acked_word(cb);
}

// Free pages at the back; no other effect.
inline bool cb_pages_reservable_at_back(uint32_t cb, uint32_t pages) {
    kmsim::point("cb_pages_reservable_at_back", cb);
    return kmsim::cb_room(cb, pages);
}

// Waits (without a timeout) for free pages; a reservation past the buffer's end is
// reported.
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

// Waits (without a timeout) for pages at the front.
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

// The NoC and command buffer of the calling kernel: one NoC per kernel in the model.
constexpr uint8_t noc_index = 0;
constexpr uint8_t DM_DEDICATED_NOC = 0;
constexpr uint8_t DM_DYNAMIC_NOC = 1;
constexpr uint8_t noc_mode = DM_DEDICATED_NOC;
constexpr uint32_t write_at_cmd_buf = 0;
constexpr uint32_t NOC_UNICAST_WRITE_VC = 0;

// The command buffer's registers (Blackhole's names and offsets): the model has none, so
// a buffer is always ready, a read gives 0 and a write is dropped.
constexpr uint32_t NOC_RET_ADDR_LO = 0xFFB20000 + 0xC;
constexpr uint32_t NOC_NODE_ID = 0xFFB20000 + 0x44;
constexpr uint32_t NOC_ADDR_NODE_ID_BITS = 6;
constexpr uint64_t NOC_NODE_ID_MASK = (uint64_t{1} << NOC_ADDR_NODE_ID_BITS) - 1;
constexpr uint32_t MEM_NOC_ATOMIC_RET_VAL_ADDR = 4;

inline bool noc_cmd_buf_ready(uint32_t, uint32_t) { return true; }
inline uint32_t NOC_CMD_BUF_READ_REG(uint32_t, uint32_t, uint32_t) { return 0; }
inline void NOC_CMD_BUF_WRITE_REG(uint32_t, uint32_t, uint32_t, uint32_t) {}
inline uint64_t NOC_XY_ADDR(uint64_t x, uint64_t y, uint32_t addr) {
    return kmsim::noc_core(static_cast<uint32_t>(x), static_cast<uint32_t>(y), addr);
}

// A non-posted atomic add delivered like an increment; with ret_addr programmed, the
// value before the add is written to this core's ret_addr when it is acknowledged.
template <uint8_t NocMode, bool ProgramRet>
inline void noc_fast_atomic_increment(
    uint32_t,
    uint32_t,
    uint64_t addr,
    uint32_t,
    uint32_t incr,
    uint32_t,
    bool,
    bool,
    uint32_t ret_addr
) {
    kmsim::point("noc_fast_atomic_increment");
    kmsim::noc_fetch_add(addr, incr, ProgramRet ? ret_addr : ~0u);
}

// The counters the barriers poll: true once every atomic (write) of this kernel was
// acknowledged (landed); reads land at the call.
inline bool ncrisc_noc_nonposted_atomics_flushed(uint32_t) {
    kmsim::point("ncrisc_noc_nonposted_atomics_flushed");
    return kmsim::noc_atomics_done();
}

inline bool ncrisc_noc_nonposted_writes_flushed(uint32_t) {
    kmsim::point("ncrisc_noc_nonposted_writes_flushed");
    return kmsim::noc_writes_done();
}

inline bool ncrisc_noc_reads_flushed(uint32_t) {
    kmsim::point("ncrisc_noc_reads_flushed");
    return true;
}

// Sources are read when a write is submitted; the dynamic NoC is not modelled.
inline bool ncrisc_noc_nonposted_writes_sent(uint32_t) {
    kmsim::point("ncrisc_noc_nonposted_writes_sent");
    return true;
}

inline bool ncrisc_dynamic_noc_nonposted_writes_sent(uint32_t noc) {
    return ncrisc_noc_nonposted_writes_sent(noc);
}

inline bool ncrisc_dynamic_noc_nonposted_writes_flushed(uint32_t noc) {
    return ncrisc_noc_nonposted_writes_flushed(noc);
}

inline bool ncrisc_dynamic_noc_nonposted_atomics_flushed(uint32_t noc) {
    return ncrisc_noc_nonposted_atomics_flushed(noc);
}

inline bool ncrisc_dynamic_noc_reads_flushed(uint32_t noc) {
    return ncrisc_noc_reads_flushed(noc);
}

// The host channel behind the PCIe tile is not modelled: no scenario sends a transfer,
// so these calls only compile the runtime's transfer code (they do nothing).
#ifndef MEM_L1_SIZE
#define MEM_L1_SIZE (1536 * 1024)
#endif
#ifndef NOC_INDEX
#define NOC_INDEX 0
#endif
#define PCIE_NOC_X 0
#define PCIE_NOC_Y 0
#define NOC_XY_PCIE_ENCODING(x, y) (uint64_t{1} << 60)

inline void noc_async_write_set_pcie_state(uint64_t) {
    kmsim::point("noc_async_write_set_pcie_state");
}

inline void noc_async_write_with_state(uint32_t, uint32_t, uint32_t) {
    kmsim::point("noc_async_write_with_state");
}

inline void noc_async_write_clear_pcie_state() {
    kmsim::point("noc_async_write_clear_pcie_state");
}

inline void noc_async_write_pcie(uint32_t, uint64_t, uint32_t) {
    kmsim::point("noc_async_write_pcie");
}

inline void noc_async_read_set_pcie_state(uint64_t) {
    kmsim::point("noc_async_read_set_pcie_state");
}

inline void noc_async_read_with_state(uint32_t, uint32_t, uint32_t) {
    kmsim::point("noc_async_read_with_state");
}

inline void noc_async_read_clear_pcie_state() {
    kmsim::point("noc_async_read_clear_pcie_state");
}

inline bool noc_cmd_bufs_mid_clear() {
    return true;
}

// The banks a copy's bounds name (no scenario copies).
#ifndef NUM_DRAM_BANKS
#define NUM_DRAM_BANKS 8
#endif

// Not modelled, only compiled: the link cores' page path, the wire between islands,
// overlays and multicasts (no scenario runs a link, loads an overlay or multicasts).
#define NOC_MAX_TRANSACTION_ID 0xF
#define NIU_MST_NONPOSTED_WR_REQ_SENT 0
#define NOC_STATUS_READ_REG(noc, reg) 0u
#define NOC_UNICAST_ADDR_X(addr) 0u
#define NOC_UNICAST_ADDR_Y(addr) 0u
#ifndef MEM_MAP_END
#define MEM_MAP_END (64 * 1024)
#endif
inline uint32_t noc_nonposted_writes_num_issued[2] = {};
inline uint32_t kmsim_config[64] = {};
#define TENSIX_CFG_BASE (reinterpret_cast<uintptr_t>(kmsim_config))
#define RISCV_IC_INVALIDATE_InvalidateAll_ADDR32 0

inline void noc_async_read_set_trid(uint32_t, uint8_t = 0) {
    kmsim::point("noc_async_read_set_trid");
}

inline void noc_inline_dw_write(uint64_t, uint32_t, uint8_t = 0xF, uint8_t = 0) {
    kmsim::point("noc_inline_dw_write");
}

inline uint64_t get_noc_multicast_addr(
    uint32_t x_start, uint32_t y_start, uint32_t, uint32_t, uint32_t addr
) {
    return kmsim::noc_core(x_start, y_start, addr);
}

inline void noc_async_write_multicast(
    uint32_t, uint64_t, uint32_t, uint32_t, bool = false
) {
    kmsim::point("noc_async_write_multicast");
}
