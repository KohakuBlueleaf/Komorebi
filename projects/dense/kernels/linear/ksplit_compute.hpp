// Linear compute, the K-split plan: this rank's part over its K chunks into kCbKSum
// (fp32), then the parts it receives; the group's first rank packs the sum (bf16).
#include "ksplit.hpp"

namespace {

    // Tile `page` of fp32 buffer `source` into tile `target` of buffer `cb`
    // (accumulated when the packer's L1 accumulation is on), through Dst 0.
    void move_tile(uint32_t source, uint32_t page, uint32_t cb, uint32_t target) {
        tile_regs_acquire();
        copy_tile(source, page, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb, target);
        tile_regs_release();
    }

    // The whole-K plan's receiver: unpack's timed wait until X's marker in its port
    // counts `chunks` of the command's chunks landed; `seen` the count on all three.
    bool x_landed(
        const km::Compute& compute,
        const km_command_t& command,
        uint32_t chunks,
        uint32_t& seen
    ) {
        if (chunks <= seen) {
            return true;
        }
        uint32_t heard = 0;
        UNPACK(({
            volatile tt_l1_ptr uint32_t* marker = (
                km::l1(
                    compute.control() + KM_CONTROL_PORT + kPortKXGot * KM_PORT_STRIDE
                )
            );
            uint32_t word = 0;
            const uint32_t code = km::poll_live(
                compute.control(),
                km::kRoleUnpack,
                km::kWherePort,
                true,
                [&] {
                    invalidate_l1_cache();
                    word = marker[KM_PORT_COUNT];
                    return km::broadcast_reached(word, command.sequence, chunks);
                },
                compute.wait_limit(),
                0
            );
            if (code == km::kGaveNone) {
                heard = word & 0xFFFFu;
            } else {
                km::command_gave_up(
                    compute.control(),
                    km::kStCompute,
                    code,
                    km::kWherePort,
                    km::slot(command),
                    command.sequence
                );
            }
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, heard);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, heard);
        }));
        MATH((heard = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
        PACK((heard = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
        seen = heard;
        return heard != 0;
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    const auto plan = [&](uint32_t word) {
        return compute.arg(word);
    };
    const uint32_t block_height = plan(kKsOh), block_width = plan(kKsOw);
    const uint32_t chunk_tiles = plan(kKsKc), subblock_height = plan(kKsSh);
    const uint32_t subblock_width = plan(kKsSw);
    const uint32_t block_tiles = block_height * block_width;
    compute_kernel_hw_startup<SrcOrder::Reverse>(kCbKa, kCbKw, kCbKSum);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerLinear) {
            return;
        }
        const KPlace at = k_place(command, compute.me(), plan, km::island());
        if (!at.any || at.k_count == 0) {  // no K: the movers give up
            return;
        }
        uint32_t index = 0;
        const km::Group group = k_group(command, compute.me(), plan(kKsS), index);
        const km::Arrival arrival = {kCbKArrive, plan(kKsArrive), block_tiles};
        const uint32_t form = plan(kKsForm);
        const km::Pattern pattern = km::Pattern::of(plan(kKsPattern));
        const bool x_cast = plan(kKsXCast) != 0;  // X whole in kCbKX, by chunk
        const bool x_marked = plan(kKsXRole) == kKXReceiver;  // ... once marked
        uint32_t x_seen = 0;
#if defined(LINEAR_ACCOUNT) && !defined(TRISC_MATH)  // the math thread: no buffers
        volatile uint32_t* stamps = stamp_row();
#endif
        uint32_t round = 0;
        const bool room = compute.reserve_back(kCbKSum, block_tiles);
        bool ok = true;  // the same on all three threads (wait_front tells them)
        matmul_block_init(
            kCbKa, kCbKw, 0, subblock_width, subblock_height, chunk_tiles
        );
        pack_reconfig_data_format(kCbKSum);
        for (
            uint32_t offset = 0, chunk = 0; ok && offset < at.k_count;
            offset += chunk_tiles, chunk++
        ) {
            const uint32_t length = (
                at.k_count - offset < chunk_tiles ? at.k_count - offset : chunk_tiles
            );
            const uint32_t x_half = (
                x_cast ? kCbKX : km::block_half(kKChunksX, pattern, chunk)
            );
            const uint32_t w_half = km::block_half(kKChunksW, pattern, chunk);
            const uint32_t x_first = x_cast ? chunk * block_height * chunk_tiles : 0;
            ok = (
                (!x_marked || x_landed(compute, command, chunk + 1, x_seen)) &&
                compute.wait_front(x_half, x_first + block_height * chunk_tiles) &&
                compute.wait_front(w_half, chunk_tiles * block_width)
            );
            if (!ok) {
                break;
            }
            pack_reconfig_l1_acc(offset > 0 ? 1 : 0);
            for (
                uint32_t subblock_row = 0; subblock_row < block_height;
                subblock_row += subblock_height
            ) {
                for (
                    uint32_t subblock_column = 0; subblock_column < block_width;
                    subblock_column += subblock_width
                ) {
                    tile_regs_acquire();
                    for (uint32_t k_tile = 0; k_tile < length; ++k_tile) {
                        matmul_block(
                            x_half,
                            w_half,
                            x_first + subblock_row * chunk_tiles + k_tile,
                            k_tile * block_width + subblock_column,
                            0,
                            0,
                            subblock_width,
                            subblock_height,
                            chunk_tiles
                        );
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (
                        uint32_t tile_row = 0; room && tile_row < subblock_height;
                        ++tile_row
                    ) {
                        for (
                            uint32_t tile_column = 0; tile_column < subblock_width;
                            ++tile_column
                        ) {
                            pack_tile<true>(
                                tile_row * subblock_width + tile_column,
                                kCbKSum,
                                (subblock_row + tile_row) * block_width +
                                    subblock_column + tile_column
                            );
                        }
                    }
                    tile_regs_release();
                }
            }
            if (!x_cast) {
                cb_pop_front(x_half, block_height * chunk_tiles);
            }
            cb_pop_front(w_half, chunk_tiles * block_width);
        }
        if (ok && x_cast) {
            cb_pop_front(kCbKX, x_pages(plan, at));
        }
        LINEAR_ACCOUNT_DO(PACK((stamps[kAcKOwn] = km::wall())));
        reconfig_data_format_srca(kCbKw, kCbKArrive);
        copy_init(kCbKArrive);
        if (ok) {
            pack_reconfig_l1_acc(1);
            ok = km::reduce(compute, arrival, group, form, [&](uint32_t slot) {
                LINEAR_ACCOUNT_DO(if (round < kKRounds) {
                    UNPACK((stamps[kAcKTaken + round] = km::wall()));
                });
                for (uint32_t tile = 0; tile < block_tiles; ++tile) {
                    move_tile(
                        kCbKArrive, km::slot_page(arrival, slot) + tile, kCbKSum, tile
                    );
                }
                LINEAR_ACCOUNT_DO(if (round < kKRounds) {
                    PACK((stamps[kAcKAdded + round] = km::wall()));
                });
                round++;
            });
        }
        pack_reconfig_l1_acc(0);
        if (ok && room) {
            cb_push_back(kCbKSum, block_tiles);
        }
        if (ok && group.rank == 0) {  // the group's sum, in bf16, for the writer
            ok = compute.wait_front(kCbKSum, block_tiles);
            const bool out = compute.reserve_back(kCbKOut, block_tiles);
            pack_reconfig_data_format(kCbKOut);
            copy_init(kCbKSum);
            for (uint32_t tile = 0; ok && tile < block_tiles; ++tile) {
                tile_regs_acquire();
                copy_tile(kCbKSum, tile, 0);
                tile_regs_commit();
                tile_regs_wait();
                if (out) {
                    pack_tile(0, kCbKOut);
                }
                tile_regs_release();
            }
            if (ok && out) {
                cb_push_back(kCbKOut, block_tiles);
            }
            if (ok) {
                cb_pop_front(kCbKSum, block_tiles);
            }
        }
        // LINEAR_ADD (two islands, Y a copy): that sum was the island's partial; both
        // islands' blocks (bf16) come back, summed in fp32, the sum in bf16 for Y
        if (
            ok && group.rank == 0 && LINEAR_ADDS && km::islands() == 2 &&
            copied(command.a2)
        ) {
            ok = compute.wait_front(kCbKAdd, 2 * block_tiles, km::kBehindPort);
            const bool held = ok && compute.reserve_back(kCbKSum, block_tiles);
            reconfig_data_format_srca(kCbKSum, kCbKAdd);
            copy_init(kCbKAdd);
            pack_reconfig_data_format(kCbKSum);
            for (uint32_t half = 0; ok && half < 2; ++half) {
                pack_reconfig_l1_acc(half);
                for (uint32_t tile = 0; held && tile < block_tiles; ++tile) {
                    move_tile(kCbKAdd, half * block_tiles + tile, kCbKSum, tile);
                }
            }
            pack_reconfig_l1_acc(0);
            if (ok) {
                cb_pop_front(kCbKAdd, 2 * block_tiles);
            }
            if (ok && held) {
                cb_push_back(kCbKSum, block_tiles);
            }
            ok = ok && compute.wait_front(kCbKSum, block_tiles);
            const bool out = ok && compute.reserve_back(kCbKOut, block_tiles);
            reconfig_data_format_srca(kCbKAdd, kCbKSum);
            pack_reconfig_data_format(kCbKOut);
            copy_init(kCbKSum);
            for (uint32_t tile = 0; ok && tile < block_tiles; ++tile) {
                tile_regs_acquire();
                copy_tile(kCbKSum, tile, 0);
                tile_regs_commit();
                tile_regs_wait();
                if (out) {
                    pack_tile(0, kCbKOut);
                }
                tile_regs_release();
            }
            if (ok && out) {
                cb_push_back(kCbKOut, block_tiles);
            }
            if (ok) {
                cb_pop_front(kCbKSum, block_tiles);
            }
        }
        reconfig_data_format_srca(kCbKArrive, kCbKw);
    });
}
