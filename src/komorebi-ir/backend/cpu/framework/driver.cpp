// The CPU backend's driver: runs a run script (docs/ir/run-scripts.md) on the module it
// is linked with and prints the report as `komo run` does.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "komo_cpu.hpp"

namespace {

    using komo_cpu::u32;
    using komo_cpu::u64;

    struct Named {
        std::string name;
        std::vector<unsigned char> bytes;
        komo_tensor tensor;
    };

    [[noreturn]] void refuse(size_t line, const std::string& what) {
        std::fprintf(stderr, "line %zu: %s\n", line, what.c_str());
        std::exit(2);
    }

    // A format the framework holds: its number and the bytes of an element.
    bool format_of(const std::string& name, u32& number, u32& width) {
        if (name == "u32") {
            number = 2;
            width = 4;
        } else if (name == "f32") {
            number = 9;
            width = 4;
        } else if (name == "bf16") {
            number = 11;
            width = 2;
        } else {
            return false;
        }
        return true;
    }

    const char* format_name(u32 number) {
        return number == 2 ? "u32" : number == 9 ? "f32" : "bf16";
    }

    u64 count(size_t line, const std::string& word) {
        char* end = nullptr;
        unsigned long long v = std::strtoull(word.c_str(), &end, 10);
        if (word.empty() || *end != '\0' || word[0] == '-') {
            refuse(line, "`" + word + "` is not a count");
        }
        return v;
    }

    double real(size_t line, const std::string& word) {
        char* end = nullptr;
        double v = std::strtod(word.c_str(), &end);
        if (word.empty() || *end != '\0') {
            refuse(line, "`" + word + "` is not a number");
        }
        return v;
    }

    // Element `at` of a tensor's bytes as a double.
    double element(const Named& t, u64 at) {
        const unsigned char* p = t.bytes.data();
        if (t.tensor.format == 11) {
            komo_cpu::bf16 x;
            std::memcpy(&x.bits, p + at * 2, 2);
            return komo_cpu::Element<komo_cpu::bf16>::get(x);
        }
        if (t.tensor.format == 9) {
            float f;
            std::memcpy(&f, p + at * 4, 4);
            return f;
        }
        u32 v;
        std::memcpy(&v, p + at * 4, 4);
        return v;
    }

    void set_element(Named& t, u64 at, double v) {
        unsigned char* p = t.bytes.data();
        if (t.tensor.format == 11) {
            komo_cpu::bf16 x = komo_cpu::Element<komo_cpu::bf16>::put(float(v));
            std::memcpy(p + at * 2, &x.bits, 2);
        } else if (t.tensor.format == 9) {
            float f = float(v);
            std::memcpy(p + at * 4, &f, 4);
        } else {
            u32 u = u32(v);
            std::memcpy(p + at * 4, &u, 4);
        }
    }

    // A value as the report writes it: an integer as itself, a float as the shortest
    // text that reads back to its f32 value (`.0` added to a whole number).
    std::string text(double v, bool integer) {
        char buffer[128];
        if (integer) {
            std::snprintf(buffer, sizeof buffer, "%.0f", v);
            return buffer;
        }
        if (std::isnan(v)) {
            return "NaN";
        }
        if (std::isinf(v)) {
            return v > 0 ? "inf" : "-inf";
        }
        float f = float(v);
        double a = std::fabs(double(f));
        bool plain = a == 0.0 || (a >= 1e-5 && a < 1e16);
        for (int digits = plain ? 0 : 1; digits <= 48; ++digits) {
            const char* form = plain ? "%.*f" : "%.*g";
            std::snprintf(buffer, sizeof buffer, form, digits, double(f));
            if (std::strtof(buffer, nullptr) == f) {
                break;
            }
        }
        std::string s = buffer;
        if (s.find_first_of(".en") == std::string::npos) {
            s += ".0";
        }
        return s;
    }

    // Prints a tensor as the report shows it: each tile's elements, a run of one value
    // written once with its count. One past the tensor's edge reads as zero.
    void show(const Named& t, komo_cpu::TileShape tile) {
        std::printf(
            "tensor %s %s [%llu, %llu]\n",
            t.name.c_str(),
            format_name(t.tensor.format),
            (unsigned long long)t.tensor.rows,
            (unsigned long long)t.tensor.cols
        );
        bool integer = t.tensor.format == 2;
        u64 n = komo_cpu::tile_count(t.tensor, tile);
        for (u64 i = 0; i < n; ++i) {
            std::vector<std::string> elements(size_t(tile.rows) * tile.cols);
            komo_cpu::each_element(
                t.tensor, tile, i, [&](size_t k, u64 at, bool inside) {
                    elements[k] = text(inside ? element(t, at) : 0.0, integer);
                }
            );
            std::string line;
            for (size_t k = 0; k < elements.size();) {
                size_t run = k;
                while (run < elements.size() && elements[run] == elements[k]) {
                    ++run;
                }
                if (!line.empty()) {
                    line += ", ";
                }
                line += elements[k];
                if (run - k > 1) {
                    line += "*" + std::to_string(run - k);
                }
                k = run;
            }
            std::printf("  tile %llu: %s\n", (unsigned long long)i, line.c_str());
        }
    }

    Named* find(std::vector<Named>& tensors, size_t line, const std::string& name) {
        for (Named& t : tensors) {
            if (t.name == name) {
                return &t;
            }
        }
        refuse(line, "no tensor " + name);
    }

    // An argument: a tensor by name, `VALUE:TYPE`, `true` or `false`.
    komo_arg argument(std::vector<Named>& tensors, size_t line, const std::string& w) {
        if (w == "true" || w == "false") {
            return komo_arg{KOMO_ARG_BOOL, nullptr, w == "true" ? 1u : 0u};
        }
        size_t colon = w.rfind(':');
        if (colon == std::string::npos) {
            return komo_arg{KOMO_ARG_TENSOR, &find(tensors, line, w)->tensor, 0};
        }
        std::string value = w.substr(0, colon);
        std::string type = w.substr(colon + 1);
        if (type == "f32") {
            float f = float(real(line, value));
            u32 bits;
            std::memcpy(&bits, &f, 4);
            return komo_arg{KOMO_ARG_SCALAR, nullptr, bits};
        }
        if (type == "u32") {
            return komo_arg{KOMO_ARG_SCALAR, nullptr, count(line, value) & 0xffffffffu};
        }
        refuse(line, "the cpu backend passes no scalar of type " + type + " yet");
    }

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: driver <script>\n");
        return 2;
    }
    std::ifstream file(argv[1]);
    if (!file) {
        std::fprintf(stderr, "%s: cannot be read\n", argv[1]);
        return 2;
    }
    komo_cpu::TileShape tile = komo_module.tile;
    std::vector<Named> tensors;
    u32 workers = 1;
    komo_pool* pool = nullptr;
    u64 ended = 0;
    bool started = false;
    std::string raw;
    for (size_t n = 1; std::getline(file, raw); ++n) {
        std::istringstream in(raw.substr(0, raw.find('#')));
        std::vector<std::string> w;
        for (std::string word; in >> word;) {
            w.push_back(word);
        }
        if (w.empty()) {
            continue;
        }
        if (w[0] == "options") {
            if (started) {
                refuse(n, "`options` comes before every other line");
            }
            for (size_t i = 1; i < w.size(); i += 2) {
                if (i + 1 >= w.size()) {
                    refuse(n, "`" + w[i] + "` has no value");
                }
                if (w[i] == "workers") {
                    workers = u32(count(n, w[i + 1]));
                } else if (w[i] == "islands") {
                    if (w[i + 1].find(',') != std::string::npos) {
                        refuse(n, "the cpu backend runs one island");
                    }
                } else {
                    refuse(n, "`" + w[i] + "` is not an option");
                }
            }
            continue;
        }
        started = true;
        if (w[0] == "tensor" && (w.size() == 5 || w.size() == 7)) {
            Named t;
            t.name = w[1];
            u32 number = 0;
            u32 width = 0;
            if (!format_of(w[2], number, width)) {
                refuse(n, "the cpu backend holds no tensor of format " + w[2] + " yet");
            }
            u64 rows = count(n, w[3]);
            u64 cols = count(n, w[4]);
            t.bytes.assign(size_t(rows * cols * width), 0);
            t.tensor = komo_tensor{nullptr, number, rows, cols};
            if (w.size() == 7) {
                if (w[5] != "fill") {
                    refuse(n, "`" + w[5] + "` is not `fill`");
                }
                bool per_tile = w[6].rfind("tile", 0) == 0;
                double k = 0.0;
                if (per_tile && w[6].size() > 4) {
                    if (w[6][4] != '+') {
                        refuse(n, "`" + w[6] + "` is not `tile+K`");
                    }
                    k = real(n, w[6].substr(5));
                } else if (!per_tile) {
                    k = real(n, w[6]);
                }
                u64 grid_cols = (cols + tile.cols - 1) / tile.cols;
                for (u64 r = 0; r < rows; ++r) {
                    for (u64 c = 0; c < cols; ++c) {
                        u64 i = (r / tile.rows) * grid_cols + c / tile.cols;
                        set_element(t, r * cols + c, per_tile ? double(i) + k : k);
                    }
                }
            }
            tensors.push_back(std::move(t));
            tensors.back().tensor.data = tensors.back().bytes.data();
        } else if (w[0] == "invoke" && w.size() >= 5 && w[w.size() - 3] == "on") {
            if (pool == nullptr) {
                pool = komo_open(workers);
                if (pool == nullptr) {
                    refuse(n, "no pool of " + std::to_string(workers) + " workers");
                }
            }
            std::vector<komo_arg> args;
            for (size_t i = 2; i + 3 < w.size(); ++i) {
                args.push_back(argument(tensors, n, w[i]));
            }
            u64 first = count(n, w[w.size() - 2]);
            u64 group = count(n, w[w.size() - 1]);
            komo_end end = (
                komo_invoke(
                    pool,
                    komo_entry(w[1].c_str()),
                    args.data(),
                    u32(args.size()),
                    u32(first),
                    u32(group)
                )
            );
            std::string how = (
                end.state == KOMO_COMPLETED
                    ? "completed"
                    : end.state == KOMO_FAILED
                        ? std::string("failed ") + komo_cpu::failure_name(end.failure)
                        : "invalid: refused by the cpu backend"
            );
            std::printf(
                "#%llu %s group(%llu, %llu): %s\n",
                (unsigned long long)ended,
                w[1].c_str(),
                (unsigned long long)first,
                (unsigned long long)group,
                how.c_str()
            );
            ended += 1;
        } else if (w[0] == "control") {
            refuse(n, "the cpu backend runs no control program yet");
        } else if (w[0] == "read" && w.size() == 2) {
            show(*find(tensors, n, w[1]), tile);
        } else {
            refuse(n, "`" + raw + "` is not a line of a run script");
        }
    }
    komo_close(pool);
    return 0;
}
