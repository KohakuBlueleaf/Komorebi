// Attend, writer: this core's cached values (the root appends the token's value), then
// its state up the group's tree; the root writes the group's block of a.
#include "attend.hpp"
#if defined(SPLIT_CHAIN)
#include "../../../dense/kernels/split/chain.hpp"
#endif

uint32_t km_overlay(km::Writer& writer, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    const Attend at = attend_of(writer, block);
    if (!attend::read_cache(
        writer, command, at, DC_CB_VALUES, command.a7, km::kStWriter
    ) ||
        (
            at.rank == 0 &&
            !attend::append(
                writer, command, at, DC_CB_VALUES, command.a7, true, km::kStWriter
            )
        )) {
        return 0;
    }
    attend::pad(at, DC_CB_VALUES);
    if ((at.flags & DC_ATTEND_READ_ONLY) != 0) {
        return 1;
    }
    if (at.rank != 0 && merges(at)) {
        const km::ReduceStep step = km::reduce_send(at.group, split_form(command));
#if defined(SPLIT_CHAIN)
        if (!chain_credit(
            writer
        )) {  // in a chain: its receiver has ended the record before
            return 0;
        }
#endif
        return (
            km::emit(
                writer,
                command,
                attend_arrival(writer.arg(block.plan + kSpArrive)),
                DC_CB_OUTBOX,
                DC_STATE_ITEM,
                step.to,
                step.port,
                step.slot
            )
                ? 1
                : 0
        );
    }
    if (at.rank != 0) {  // no merge: the state is dropped
        if (!writer.wait_front(DC_CB_OUTBOX, DC_STATE_ITEM)) {
            return 0;
        }
        cb_pop_front(DC_CB_OUTBOX, DC_STATE_ITEM);
        return 1;
    }
    const uint32_t tiles = DC_HEAD_GROUP * DC_HEAD_TILES;
    const km::Tensor<0> output(command.a2);
    if (
        !km::reduce_result(writer, kCbSplitOut, tiles) ||
        !km::write_pages(
            writer, command, output, query_page(at), tiles, get_read_ptr(kCbSplitOut)
        ) ||
        !writer.flushed(km::kFlushSent, km::kStWriter)
    ) {
        return 0;
    }
    cb_pop_front(kCbSplitOut, tiles);
    return 1;
}
