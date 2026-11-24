// The record page's write as the hand kernel of the class's handler `note`: mover B
// writes the page pack filled in the class's raw field to the core's page of records.
#pragma once

namespace note_k {

    // NOTE on mover B: the record page waited for as pack pushed it, written to page
    // rank of records, then freed.
    template <typename Records>
    inline bool note_writer(
        km::Movement& movement,
        const km_command_t& command,
        const Records& records,
        uint32_t tiles
    ) {
        static_cast<void>(tiles);
        const uint32_t rank = movement.me() - km::first(command);
        if (
            !movement.wait_front(kFieldRecord, kFieldRecordPages) ||
            !km::write_pages(
                movement, command, records, rank, 1, get_read_ptr(kFieldRecord)
            )
        ) {
            return false;
        }
        noc_async_write_barrier();
        cb_pop_front(kFieldRecord, kFieldRecordPages);
        return true;
    }

}  // namespace note_k
