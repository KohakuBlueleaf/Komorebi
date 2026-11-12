// The simulator behind the stub headers: every stub call of a kernel lands here. One
// kernel thread runs at a time; the wall clock advances at each scheduling point.
#pragma once

#include <stdint.h>

// Pointer qualifiers of the device compilers: nothing on the host.
#define tt_l1_ptr
#define tt_reg_ptr

namespace kmsim {

    // Low and high word of the wall clock (cycles); advanced by the scheduler only.
    extern volatile uint32_t wall_clock[2];

    // The kernels of one core, in the order the simulator starts them.
    enum Role : uint32_t { kReader, kWriter, kUnpack, kMath, kPack, kRoles };

    // Names a stub call for the probes: records it, runs the case's triggers, holds a
    // frozen kernel there. Ends the thread when the case is aborted.
    void point(const char* name, uint32_t cb = ~0u);

    // A scheduling point: the clock advances one step and the next thread runs.
    void yield();

    // The calling kernel's runtime argument i and compile-time argument i.
    uint32_t arg(uint32_t i);
    uint32_t compile_arg(uint32_t i);

    // Circular buffers of the calling core: one producer and one consumer, page counts.
    bool cb_room(uint32_t cb, uint32_t pages);
    bool cb_ready(uint32_t cb, uint32_t pages);
    void cb_reserve(uint32_t cb, uint32_t pages);
    void cb_push(uint32_t cb, uint32_t pages);
    void cb_wait(uint32_t cb, uint32_t pages);
    void cb_pop(uint32_t cb, uint32_t pages);
    uint32_t cb_write_ptr(uint32_t cb);
    uint32_t cb_read_ptr(uint32_t cb);
    uint32_t cb_page(uint32_t cb);

    // The record the calling kernel works on, for violation reports (the stub also
    // takes it from the tokens; only a case's own kernels need to say it).
    void note_record(uint32_t seq);

    // The buffer as the kernel APIs' local interface shows it, and the two counter
    // registers (pages pushed, pages popped) as words in the core's L1 image.
    struct CbView {
        uint32_t fifo_num_pages;
        uint32_t fifo_page_size;
        uint32_t tiles_received;
        uint32_t tiles_acked;
        uint32_t fifo_limit;  // byte address of the end
        uint32_t fifo_wr_ptr;
        uint32_t fifo_rd_ptr;
    };

    CbView cb_view(uint32_t cb);
    volatile uint32_t* cb_received_word(uint32_t cb);
    volatile uint32_t* cb_acked_word(uint32_t cb);

    // NoC: a 64-bit address names a core by x, y or the DRAM, and a 32-bit address.
    uint64_t noc_core(uint32_t x, uint32_t y, uint32_t addr);
    uint64_t noc_dram(uint32_t addr);
    void noc_read(uint64_t src, uint32_t dst, uint32_t bytes);
    void noc_write(uint32_t src, uint64_t dst, uint32_t bytes);
    void noc_inc(uint64_t dst, uint32_t value);
    void noc_read_barrier();
    void noc_write_barrier();
    void noc_writes_flushed();
    void noc_atomic_barrier();
    void noc_fetch_add(uint64_t dst, uint32_t value, uint32_t ret_addr);
    bool noc_atomics_done();
    bool noc_writes_done();

    // Compute: mailboxes between the three threads, the destination registers and the
    // tile moves of the unpacker, the math engine and the packer.
    void mailbox_write(uint32_t to, uint32_t value);
    uint32_t mailbox_read(uint32_t from);
    void dst_acquire();
    void dst_commit();
    void dst_wait();
    void dst_release();
    void unpack_tile(uint32_t cb, uint32_t tile);
    void math_take(uint32_t dst);
    void math_add(uint32_t a, uint32_t b, uint32_t out);
    void math_fill(uint32_t dst, float value);
    void pack_tile(uint32_t dst, uint32_t cb);

}  // namespace kmsim

// The wall-clock registers read by km::wall64() and km::wall(): low and live high.
#define RISCV_DEBUG_REG_WALL_CLOCK_L \
    (reinterpret_cast<uintptr_t>(&kmsim::wall_clock[0]))
#define RISCV_DEBUG_REG_WALL_CLOCK_1 \
    (reinterpret_cast<uintptr_t>(&kmsim::wall_clock[1]))
