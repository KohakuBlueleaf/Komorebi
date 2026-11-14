// Norm, reader: the constants, x's whole row (for the sums of squares) and this core's
// tiles of the weight, each as one item.
#include "../ops.hpp"

namespace {

    // Pages `first` to `first + pages - 1` of tensor `tensor` into `buffer`, unpushed.
    bool read(
        km::Reader& reader,
        const km_command_t& command,
        uint32_t buffer,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages
    ) {
        const km::Tensor<0> matrix(tensor);
        return (
            reader.reserve(buffer, pages) &&
            km::read_pages(reader, command, matrix, first, pages, get_write_ptr(buffer))
        );
    }

}  // namespace

uint32_t km_overlay(km::Reader& reader, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    if (km::count(command) != DC_NORM_CORES) {
        return 0;
    }
    const uint32_t first = (reader.me() - km::first(command)) * DC_NORM_SHARE;
    if (
        !read(reader, command, DC_CB_CONSTANT, command.a9, 0, DC_CONSTANTS) ||
        !read(reader, command, DC_CB_ROW, command.a7, 0, DC_ROW_TILES) ||
        !read(reader, command, DC_CB_WEIGHT, command.a8, first, DC_NORM_SHARE) ||
        !flushed(reader, km::kStReader, km::kFlushReads)
    ) {
        return 0;
    }
    cb_push_back(DC_CB_CONSTANT, DC_CONSTANTS);
    cb_push_back(DC_CB_ROW, DC_ROW_TILES);
    cb_push_back(DC_CB_WEIGHT, DC_NORM_SHARE);
    return 1;
}
