// A tensor table row written while the run lives: a tensor made after an island load
// (ENTRY with KM_ENTRY_TABLE, served by the service overlay on every rank).
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "km_island.hpp"
#if defined(KM_TABLE_CACHE)
#include "km_table_cache.hpp"
#endif

namespace km {

    // Row a1 of this island's table from DRAM bank 0 at a2 (the row's own offset mod
    // 64 there, so the read keeps its alignment); a word outside the table gives up.
    template <typename Kernel>
    inline void write_table_row(Kernel& reader, const km_command_t& command) {
        const Islands view(island_area());
        const uint32_t table = view.tensors();
        if (table == 0 || command.a1 >= KM_TENSORS_MAX) {
            reader.gave_up(
                kStReader, kGaveBadItem, kWhereFromHost, command.a1, command.sequence
            );
            return;
        }
#if defined(KM_TABLE_CACHE)
        forget_entry(view, table, command.a1);  // the host wrote its DRAM row
#else
        const uint32_t bytes = (
            KM_TENSOR_HEADER_BYTES + view.count() * KM_TENSOR_PART_BYTES
        );
        noc_async_read(
            get_noc_addr_from_bank_id<true>(0, command.a2),
            table + command.a1 * bytes,
            bytes
        );
        noc_async_read_barrier();
        invalidate_l1_cache();  // the row is read from L1 by the next lookup
#endif
    }

}  // namespace km
