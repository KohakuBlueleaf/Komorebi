/* Reset of one core during a run: its RISC CPUs in reset through its soft-reset
 * register, the host's write list (firmware, tables, kernels) into its L1, released. */
#pragma once

#include "event_ring.h"
#include "kernel.h"

/* Window `slot` aimed at `address` of core `core` (the host's core table), non-posted
 * and read back; 0 when it is refused or reads otherwise. */
static volatile uint8_t* reset_aim(
    kmd_t* dispatcher, uint32_t slot, uint32_t core, uint32_t address
) {
    const uint32_t coordinates = (
        ((volatile const uint32_t*)(dispatcher->lim + KMD_LIM_COORDS))[core]
    );
    if (coordinates == 0) {
        return 0;
    }
    noc_tlb_2m_t programmed;
    volatile uint8_t* window = (
        (volatile uint8_t*)noc_configure_tlb_2m_checked(
            slot,
            coordinates & 0xFFFFu,
            coordinates >> 16,
            address,
            NOC_ORDER_STRICT,
            &programmed
        )
    );
    if (window == 0 || noc_verify_tlb_2m(slot, &programmed) != 0) {
        return 0;
    }
    return window;
}

/* One store to the soft-reset register, then a fence. */
static inline void reset_register(volatile uint8_t* reg, uint32_t value) {
    *(volatile uint32_t*)reg = value;
    x280_fence_iorw();
}

/* Runs the list of `bytes` at `offset` of the upload region on core `core`. Returns
 * KMD_RESET_*; every wait is bounded, both windows are cleared on every path. */
static uint32_t reset_run(
    kmd_t* dispatcher, uint32_t core, uint32_t offset, uint32_t bytes
) {
    if (
        core >= KM_CORES_MAX || offset % 16u != 0 || bytes % 16u != 0 ||
        offset > dispatcher->upload_bytes || bytes > dispatcher->upload_bytes - offset
    ) {
        return KMD_RESET_BAD_LIST;
    }
    volatile uint8_t* l1 = reset_aim(dispatcher, KMD_WIN_RESET, core, 0);
    volatile uint8_t* reg = (
        reset_aim(dispatcher, KMD_WIN_RESET_REGISTER, core, KMD_SOFT_RESET_ADDR)
    );
    uint32_t status = l1 == 0 || reg == 0 ? KMD_RESET_NO_WINDOW : KMD_RESET_OK;
    volatile const uint32_t* list = (
        (volatile const uint32_t*)(dispatcher->lim + KMD_LIM_UPLOAD + offset)
    );
    x280_fence_ir();
    uint32_t at = 0; /* words */
    while (status == KMD_RESET_OK && at * 4u + KMD_RESET_ENTRY <= bytes) {
        const uint32_t kind = list[at], address = list[at + 1];
        const uint32_t size = list[at + 2], value = list[at + 3];
        at += KMD_RESET_ENTRY / 4u;
        if (kind == KMD_RESET_END) {
            break;
        }
        if (kind == KMD_RESET_ASSERT) {
            reset_register(reg, KMD_SOFT_RESET_RISCS);
        } else if (kind == KMD_RESET_RELEASE) {
            reset_register(reg, KMD_SOFT_RESET_RELEASE);
        } else if (kind == KMD_RESET_WRITE) {
            const uint32_t padded = (size + 15u) & ~15u;
            if (
                address % 4u != 0 || size % 4u != 0 || address > KMD_L1_BYTES ||
                size > KMD_L1_BYTES - address || padded > bytes - at * 4u
            ) {
                status = KMD_RESET_BAD_LIST;
                break;
            }
            volatile uint32_t* to = (volatile uint32_t*)(l1 + address);
            for (uint32_t word = 0; word < size / 4u; word++) {
                to[word] = list[at + word];
            }
            x280_fence_ow();
            at += padded / 4u;
        } else if (kind == KMD_RESET_WAIT) {
            if (address % 4u != 0 || address > KMD_L1_BYTES - 4u) {
                status = KMD_RESET_BAD_LIST;
                break;
            }
            volatile const uint32_t* word = (volatile const uint32_t*)(l1 + address);
            const uint64_t until = x280_mtime() + KMD_RESET_WAIT_TICKS;
            while ((*word & size) != value) {
                if (x280_mtime() > until) {
                    status = KMD_RESET_NO_FIRMWARE;
                    break;
                }
            }
        } else {
            status = KMD_RESET_BAD_LIST;
        }
    }
    noc_clear_tlb_2m(KMD_WIN_RESET);
    noc_clear_tlb_2m(KMD_WIN_RESET_REGISTER);
    return status;
}

/* The host's reset command: runs the list, then PONG {status, the ticks it took}. */
static inline void reset_command(
    kmd_t* dispatcher, uint32_t request, uint32_t core, uint32_t offset, uint32_t bytes
) {
    const uint64_t start = x280_mtime();
    const uint32_t status = reset_run(dispatcher, core, offset, bytes);
    const uint64_t now = x280_mtime();
    host_push(
        dispatcher, request, KM_EVENT_PONG | status << 8, (uint32_t)(now - start)
    );
    dispatcher->last_event_mtime = now;
}
