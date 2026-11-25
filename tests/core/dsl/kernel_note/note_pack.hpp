// NOTE on the pack CPU for test_shared_buffers.py (as example 33's): the raw field's
// page filled with the core's words and pushed.
#pragma once

namespace note_k {

    constexpr uint32_t kNoteMark = 0x4B4D5241u;  // word 2 of each record
    constexpr uint32_t kNoteWords = kFieldRecordPageBytes / sizeof(uint32_t);

    // NOTE on the pack CPU: word 0 the core's rank, word 1 `tiles`, word 2 the mark,
    // word i = i from 3 on; pushed for mover B.
    inline void note_pack(
        km::Compute& compute, const km_command_t& command, uint32_t tiles
    ) {
        if (!compute.reserve_back(kFieldRecord, kFieldRecordPages)) {
            return;
        }
        volatile tt_l1_ptr uint32_t* words = (
            km::l1(get_local_cb_interface(kFieldRecord).fifo_wr_ptr << 4)
        );
        words[0] = compute.me() - km::first(command);
        words[1] = tiles;
        words[2] = kNoteMark;
        for (uint32_t i = 3; i < kNoteWords; i++) {
            words[i] = i;
        }
        cb_push_back(kFieldRecord, kFieldRecordPages);
    }

}  // namespace note_k
