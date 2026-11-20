// The CPU framework the backend's emitted C++ is written against (portable C++17):
// workers, statuses, views, streams and sinks, tiles, and a module's entries.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "komo_cpu.h"

namespace komo_cpu {

    using u32 = std::uint32_t;
    using u64 = std::uint64_t;
    using f32 = float;

    // The numbers of the byte code's tables (docs/ir/bytecode.md, "The numbers").
    enum class Format : u32 { u32 = 2, f32 = 9, bf16 = 11 };

    enum class Failure : u32 {
        view_end = 0,
        empty_state = 1,
        send_refused = 2,
        capacity = 3,
        given_up = 4,
        dependency = 5,
        not_a_peer = 6,
        over_bound = 7,
        divide_by_zero = 8,
        out_of_range = 9,
    };

    // How a worker's part of an invocation ended: completed, failed by name, or not a
    // program (with what).
    struct Status {
        enum class Kind { completed, failed, invalid };

        Kind kind = Kind::completed;
        Failure failure = Failure::view_end;
        std::string what;

        static Status ok() {
            return Status{};
        }

        static Status fail(Failure f) {
            Status s;
            s.kind = Kind::failed;
            s.failure = f;
            return s;
        }

        static Status invalid(std::string w) {
            Status s;
            s.kind = Kind::invalid;
            s.what = std::move(w);
            return s;
        }

        bool is_ok() const {
            return kind == Kind::completed;
        }
    };

    // A value, or the status that stopped it.
    template <class T>
    struct Result {
        Status status;
        T value;
    };

    template <class T>
    Result<T> ok(T value) {
        return Result<T>{Status::ok(), std::move(value)};
    }

    // Binds `name` to the value of `expr` (a Result), or returns its status.
#define KOMO_LET(type, name, expr)       \
    auto name##_result = (expr);         \
    if (!name##_result.status.is_ok()) { \
        return name##_result.status;     \
    }                                    \
    type name = std::move(name##_result.value)

    // Runs `expr` (a Status), or returns it.
#define KOMO_DO(expr)                \
    do {                             \
        Status komo_status = (expr); \
        if (!komo_status.is_ok()) {  \
            return komo_status;      \
        }                            \
    } while (0)

    // ---- Elements ----

    struct bf16 {
        std::uint16_t bits;
    };

    // An element type's format, and its value as an f32 and back: the framework's own
    // conversion (a format is a notation; how this backend computes is its document).
    template <class T>
    struct Element;

    template <>
    struct Element<bf16> {
        static constexpr Format format = Format::bf16;

        static f32 get(bf16 x) {
            u32 word = u32(x.bits) << 16;
            f32 f;
            std::memcpy(&f, &word, 4);
            return f;
        }

        static bf16 put(f32 f) {
            u32 word;
            std::memcpy(&word, &f, 4);
            if ((word & 0x7fffffffu) > 0x7f800000u) {
                return bf16{std::uint16_t((word >> 16) | 0x40)};
            }
            u32 even = 0x7fffu + ((word >> 16) & 1u);
            return bf16{std::uint16_t((word + even) >> 16)};
        }
    };

    template <>
    struct Element<f32> {
        static constexpr Format format = Format::f32;

        static f32 get(f32 x) {
            return x;
        }

        static f32 put(f32 x) {
            return x;
        }
    };

    template <>
    struct Element<u32> {
        static constexpr Format format = Format::u32;

        static f32 get(u32 x) {
            return f32(x);
        }

        static u32 put(f32 x) {
            return u32(x);
        }
    };

    // ---- Workers and arguments ----

    // One worker's place in an invocation: its rank in the group, the group's size, and
    // its id among the machine's workers.
    struct Worker {
        u32 rank;
        u32 size;
        u32 id;
    };

    // The arguments of an invocation, as the host gave them.
    struct Args {
        const komo_arg* args;
        u32 count;

        const komo_tensor& tensor(u32 i) const {
            return *args[i].tensor;
        }

        u32 u32_at(u32 i) const {
            return u32(args[i].bits);
        }

        f32 f32_at(u32 i) const {
            u32 word = u32(args[i].bits);
            f32 f;
            std::memcpy(&f, &word, 4);
            return f;
        }
    };

    // ---- Tiles ----

    struct TileShape {
        u32 rows;
        u32 cols;
    };

    template <class T>
    struct Tile {
        TileShape shape;
        std::vector<T> elements;

        explicit Tile(TileShape s) : shape(s), elements(std::size_t(s.rows) * s.cols) {
        }
    };

    // A tile times a scalar, element by element.
    template <class T>
    Tile<T> mul(const Tile<T>& a, f32 k) {
        Tile<T> out(a.shape);
        for (std::size_t i = 0; i < a.elements.size(); ++i) {
            out.elements[i] = Element<T>::put(Element<T>::get(a.elements[i]) * k);
        }
        return out;
    }

    // ---- Views, streams, sinks ----

    // Tiles `first` to `first + n - 1` of a tensor, row of tiles after row of tiles;
    // a number past the tensor fails (`view_end`) only when it is read or written.
    struct View {
        const komo_tensor* tensor;
        u64 first;
        u64 n;
    };

    inline View tiles(const komo_tensor& t, u64 first, u64 n) {
        return View{&t, first, n};
    }

    // This worker's part of a view: the same rule on every backend (the parts as even
    // as they can be, the first ones one longer).
    inline Result<View> partition(const View& v, u64 rank, u64 size) {
        if (size == 0 || rank >= size) {
            return Result<View>{Status::invalid("a partition by rank of a group"), v};
        }
        u64 base = v.n / size;
        u64 extra = v.n % size;
        u64 first = rank * base + (rank < extra ? rank : extra);
        u64 n = base + (rank < extra ? 1 : 0);
        return ok(View{v.tensor, v.first + first, n});
    }

    inline u64 tile_count(const komo_tensor& t, TileShape s) {
        u64 grid_rows = (t.rows + s.rows - 1) / s.rows;
        u64 grid_cols = (t.cols + s.cols - 1) / s.cols;
        return grid_rows * grid_cols;
    }

    // Each element of tile `number` of `t`, with its place in the tensor; one past
    // the tensor's edge reads as zero and is not written.
    template <class F>
    void each_element(const komo_tensor& t, TileShape s, u64 number, F f) {
        u64 grid_cols = (t.cols + s.cols - 1) / s.cols;
        u64 row0 = (number / grid_cols) * s.rows;
        u64 col0 = (number % grid_cols) * s.cols;
        for (u32 i = 0; i < s.rows; ++i) {
            for (u32 j = 0; j < s.cols; ++j) {
                u64 r = row0 + i;
                u64 c = col0 + j;
                bool inside = r < t.rows && c < t.cols;
                f(std::size_t(i) * s.cols + j, inside ? r * t.cols + c : 0, inside);
            }
        }
    }

    // A view read in blocks of tiles, in order. At this stage a block is one tile.
    struct Stream {
        View view;
        u64 block;
        TileShape tile;
        u64 next = 0;

        // The next tile into `out`; false at the stream's end.
        template <class T>
        Result<bool> next_tile(Tile<T>& out) {
            if (view.tensor->format != u32(Element<T>::format)) {
                return Result<bool>{
                    Status::invalid("a stream of another format"), false
                };
            }
            if (next >= view.n) {
                return ok(false);
            }
            u64 number = view.first + next;
            if (number >= tile_count(*view.tensor, tile)) {
                return Result<bool>{Status::fail(Failure::view_end), false};
            }
            const T* data = static_cast<const T*>(view.tensor->data);
            each_element(
                *view.tensor, tile, number, [&](std::size_t k, u64 at, bool inside) {
                    out.elements[k] = inside ? data[at] : T{};
                }
            );
            next += 1;
            return ok(true);
        }
    };

    inline Result<Stream> stream(const View& v, u64 block, TileShape tile) {
        if (block != 1) {
            return Result<Stream>{
                Status::invalid("a stream of blocks of one tile"), {}
            };
        }
        return ok(Stream{v, block, tile});
    }

    // A view written in blocks of tiles, in order. At this stage a block is one tile.
    struct Sink {
        View view;
        u64 block;
        TileShape tile;
        u64 next = 0;

        template <class T>
        Status push(const Tile<T>& in) {
            if (view.tensor->format != u32(Element<T>::format)) {
                return Status::invalid("a sink of another format");
            }
            u64 number = view.first + next;
            if (next >= view.n || number >= tile_count(*view.tensor, tile)) {
                return Status::fail(Failure::view_end);
            }
            T* data = static_cast<T*>(view.tensor->data);
            each_element(
                *view.tensor, tile, number, [&](std::size_t k, u64 at, bool inside) {
                    if (inside) {
                        data[at] = in.elements[k];
                    }
                }
            );
            next += 1;
            return Status::ok();
        }
    };

    inline Result<Sink> sink(const View& v, u64 block, TileShape tile) {
        if (block != 1) {
            return Result<Sink>{Status::invalid("a sink of blocks of one tile"), {}};
        }
        return ok(Sink{v, block, tile});
    }

    // ---- A module's entries ----

    using Handler = Status (*)(const Worker&, const Args&);

    // A parameter: KOMO_ARG_TENSOR with a format's number, KOMO_ARG_SCALAR with a
    // scalar type's number, or KOMO_ARG_BOOL.
    struct ArgKind {
        u32 kind;
        u32 number;
    };

    struct EntryPoint {
        const char* name;
        Handler handler;
        u32 count_params;
        const ArgKind* params;
    };

    struct Module {
        u32 abi;
        const char* name;
        TileShape tile;
        u32 count_entries;
        const EntryPoint* entries;
    };

}  // namespace komo_cpu

// The module's table, which its emitted source defines.
extern "C" const komo_cpu::Module komo_module;
