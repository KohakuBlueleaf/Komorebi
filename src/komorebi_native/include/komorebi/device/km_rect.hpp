// A rectangle given at each command (x0, y0, columns, rows): each core's place and
// BroadcastArgs, as komorebi.layout.broadcast_args; only its classes include it.
#pragma once

#include <stdint.h>

namespace km {

    // The grid as the load gives it: its width, each column's and row's NoC coordinate
    // (the core table's) and the panels' first columns (a row's runs stop there).
    struct RectGrid {
        uint32_t width;
        const uint8_t* noc_x;   // [width]
        const uint8_t* noc_y;   // [rows of the grid]
        const uint8_t* panels;  // first column of each panel, then 0xFF
    };

    // The grid from its program arguments' words: the width, then 16 bytes of each
    // column's NoC x, 16 of each row's NoC y, 8 of the panels' first columns.
    inline RectGrid rect_grid(const uint32_t* words) {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(words + 1);
        return {words[0], bytes, bytes + 16, bytes + 32};
    }

    // A core's place in the rectangle: its column and row in it, whether it is in it.
    struct RectPlace {
        uint32_t x, y, included, columns, rows;
    };

    inline RectPlace rect_place(uint32_t core, uint32_t width, const uint32_t* rect) {
        const uint32_t cx = core % width, cy = core / width;
        const bool in = (
            cx >= rect[0] && cx < rect[0] + rect[2] && cy >= rect[1] &&
            cy < rect[1] + rect[3]
        );
        return {
            in ? cx - rect[0] : 0u,
            in ? cy - rect[1] : 0u,
            in ? 1u : 0u,
            rect[2],
            rect[3]
        };
    }

    // The first and last cell of `core`'s run along a row or column, cut at panels and
    // in bands of `length` (0: none); false outside the rectangle.
    inline bool rect_run(
        uint32_t core,
        const RectGrid& grid,
        const uint32_t* rect,
        bool row,
        uint32_t length,
        uint32_t& first,
        uint32_t& last,
        uint32_t& at
    ) {
        const uint32_t cx = core % grid.width, cy = core / grid.width;
        if (
            cx < rect[0] || cx >= rect[0] + rect[2] || cy < rect[1] ||
            cy >= rect[1] + rect[3]
        ) {
            return false;
        }
        uint32_t low = row ? rect[0] : rect[1];
        uint32_t high = row ? rect[0] + rect[2] - 1u : rect[1] + rect[3] - 1u;
        at = row ? cx : cy;
        for (uint32_t i = 0; row && grid.panels[i] != 0xFFu; i++) {
            const uint32_t start = grid.panels[i];
            const uint32_t next = grid.panels[i + 1];
            const uint32_t end = next == 0xFFu ? grid.width - 1u : next - 1u;
            if (cx >= start && cx <= end) {
                low = low > start ? low : start;
                high = high < end ? high : end;
            }
        }
        if (length != 0u) {
            const uint32_t band = (at - low) / length;
            low += band * length;
            high = high < low + length - 1u ? high : low + length - 1u;
        }
        first = low;
        last = high;
        return true;
    }

    // One broadcast's arguments of `core`: rank, receivers, sender and the multicast
    // corners of the run it is in; `out`'s other fields are the build's.
    template <typename Args>
    inline void rect_broadcast(
        Args& out,
        uint32_t core,
        const RectGrid& grid,
        const uint32_t* rect,
        bool row,
        uint32_t length,
        uint32_t none
    ) {
        uint32_t first, last, at;
        if (!rect_run(core, grid, rect, row, length, first, last, at)) {
            out.rank = none;
            out.receivers = out.sender = out.mc_start = out.mc_end = 0u;
            return;
        }
        const bool back = out.noc == 1u;  // NoC 1 sends from the run's other end
        const uint32_t cx = core % grid.width, cy = core / grid.width;
        const uint32_t from = back ? last : first, to = back ? first : last;
        out.rank = back ? last - at : at - first;
        out.receivers = last - first;
        out.sender = row ? cy * grid.width + from : from * grid.width + cx;
        if (last == first) {
            out.mc_start = out.mc_end = 0u;
            return;
        }
        const uint32_t second = back ? from - 1u : from + 1u;
        const uint32_t sx = row ? second : cx, sy = row ? cy : second;
        const uint32_t ex = row ? to : cx, ey = row ? cy : to;
        out.mc_start = grid.noc_x[sx] | static_cast<uint32_t>(grid.noc_y[sy]) << 16;
        out.mc_end = grid.noc_x[ex] | static_cast<uint32_t>(grid.noc_y[ey]) << 16;
    }

}  // namespace km
