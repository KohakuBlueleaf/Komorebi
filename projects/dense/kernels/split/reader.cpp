// K-split reader, NoC0 (mover A): its chunks of this core's 1/S of K, every other one
// (chunks.hpp, the rest on the writer); then the parts this core sums (owner or tree).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "chunks.hpp"
#include "komorebi/device/km_port.hpp"
#include "sets.hpp"
#include "split.h"
#include "split_op.hpp"

void kernel_main() {
    using A = km::Tensor<0>;
    using W = km::Tensor<A::next_compile_time_args_offset()>;
    km::Reader reader;
#if defined(SPLIT_CHAIN)
    chain_start();
#endif
    const auto product = [&](const km_command_t& command) SPLIT_PRODUCT {
        if (km::handler(command) != kHandlerSplit || command.a5 >= kSplitPlans) {
            return;
        }
        const auto plan = [&](uint32_t word) {
            return reader.arg(plan_words(command) + word);
        };
        const uint32_t reduce_form = split_form(command), mode = command.a4;
        const uint32_t block_height = plan(kSpOh), block_width = plan(kSpOw);
        const km::Group group = km::digit_group(command, reader.me(), 1, plan(kSpS));
        if (mode == kSplitOp) {
#if defined(SPLIT_CHAIN)
            if (!chain_input(reader)) {
                return;
            }
#endif
            split_op(reader, command);
            return;
        }
        if (!move_chunks(reader, command, A(command.a0), W(command.a1))) {
            return;
        }
        if (mode == kSplitFull &&
            arrive_set(
                reader,
                command,
                {kCbArrive, plan(kSpArrive), block_height * block_width},
                group,
                reduce_form
            ) &&
            group.rank == 0) {
            split_op(reader, command);
        }
    };
#if defined(SPLIT_CHAIN)
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerSplit && command.a4 == kChainMode) {
            chain_reader(reader, command, product);
        } else {
            product(command);
        }
    });
#else
    reader.run(product);
#endif
}
