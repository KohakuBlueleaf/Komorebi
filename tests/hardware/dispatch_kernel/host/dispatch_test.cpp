// _km_dispatch_test: host module of the dispatch kernel test harness. Opens the X280
// runtime, lays out the kernel's LIM and the pinned host buffer, starts the kernel on
// cluster 0, feeds the host command ring through a write-combining window and reads the
// completion events from host memory. Every wait is bounded.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <immintrin.h>
#include <x86intrin.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>
#include <umd/device/chip_helpers/sysmem_buffer.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/io_window/io_window.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/io_window_config.hpp>
#include "x280/device/x280_runtime.hpp"
#include "x280/device/x280_state.hpp"
#include "x280_address_map.h"

#include "komorebi/abi/km_abi.h"
#include "km_dispatch.h"

namespace nb = nanobind;
using tt::tt_metal::distributed::MeshDevice;
using Clock = std::chrono::steady_clock;

namespace {

    constexpr double kMtimeHz = 50e6;
    // One page: the largest buffer that maps to the NoC without an IOMMU.
    constexpr size_t kHostBufferBytes = 4096;

    static_assert(sizeof(km_record_t) == 64);
    static_assert(sizeof(km_event_t) == 16 && sizeof(kmd_args_t) <= KMD_LIM_MAILBOX);

    Clock::time_point after(double seconds) {
        return (
            Clock::now() +
            std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(seconds)
            )
        );
    }

    std::chrono::milliseconds millis(double seconds) {
        return std::chrono::milliseconds(static_cast<int64_t>(seconds * 1000.0));
    }

    struct Runtime {
        std::shared_ptr<MeshDevice> mesh;
        std::unique_ptr<x280::RuntimeOpenResult> opened;

        Runtime(std::shared_ptr<MeshDevice> mesh_device, uint32_t pll_mhz)
            : mesh(std::move(mesh_device)) {
            opened = std::make_unique<x280::RuntimeOpenResult>(x280::X280Runtime::open(
                mesh,
                x280::RuntimeOpenConfig{
                    .priming_policy = x280::LimEccPrimingPolicy::RequirePrepared
                }
            ));
            if (!opened->ready()) {
                throw std::runtime_error(
                    "X280 runtime not ready: " + opened->diagnostic
                );
            }
            auto& rt = runtime();
            bool all_idle = true;
            for (const auto& id : rt.tile_ids()) {
                all_idle = all_idle && rt.tile(id).inspect().resident_idle_live;
            }
            if (!all_idle) {
                x280::MultiTileBootRequest req;
                req.tiles = rt.tile_ids();
                req.idle_image_path =
                    (std::string(X280_IMAGE_DIR) + "/firmware-lim-idle.bin");
                req.allow_reset_release = true;
                if (!rt.boot_tiles(req).ready()) {
                    throw std::runtime_error("X280 boot_tiles failed");
                }
            }
            rt.set_pll(rt.tile_ids().front().chip_id, pll_mhz);
        }

        x280::X280Runtime& runtime() {
            return *opened->runtime;
        }

        // The largest active image cluster 0 accepts, by the runtime's own check
        // against the tile's ECC prime record.
        std::map<std::string, nb::object> image_limit() {
            auto& rt = runtime();
            const auto state = rt.tile(rt.tile_ids().front()).inspect();
            uint64_t lo = 0, hi = 4u << 20;
            while (lo + 1 < hi) {
                const uint64_t mid = (lo + hi) / 2;
                const auto check = (
                    x280::validate_active_image(
                        X280_ACTIVE_FW_LOAD_ADDR, mid, state.ecc_primed_bytes
                    )
                );
                (check.ok ? lo : hi) = mid;
            }
            std::map<std::string, nb::object> out;
            out["observation_valid"] = nb::bool_(state.observation_valid);
            out["ecc"] = nb::str(x280::to_string(state.ecc));
            out["raw_ecc_extent"] = nb::int_(state.raw_ecc_extent);
            out["primed_bytes"] = state.ecc_primed_bytes
                ? nb::object(nb::int_(*state.ecc_primed_bytes))
                : nb::none();
            out["max_image_bytes"] = nb::int_(lo);
            out["lim_backed_bytes"] = nb::int_(state.lim_backed_bytes);
            return out;
        }
    };

    struct Event {
        uint32_t index, req, status, ticks;
    };

    // What Plan::finish saw: the hart result, the cleanup and the result block.
    struct Finish {
        bool ready = false, cleaned = false;
        std::optional<uint64_t> result;
        std::string wait, cleanup;
        std::vector<uint32_t> words;
    };

    struct Plan {
        std::shared_ptr<MeshDevice> mesh;
        x280::X280Runtime* rt = nullptr;
        x280::X280Tile* tile = nullptr;
        uint32_t gx = 0, gy = 0, l2x = 0, l2y = 0;
        uint64_t base = 0;  // LIM address of the layout (the argument block)
        std::vector<std::array<uint32_t, 2>> panels;  // logical column ranges
        uint64_t run = 0;
        uint32_t tag = 1, depth = 1, host_depth = 16, event_depth = 252;

        std::shared_ptr<tt::umd::IoWindow> win;         // host -> LIM, write-combining
        std::shared_ptr<tt::umd::SysmemBuffer> buffer;  // kernel -> pinned host memory
        volatile uint32_t* host = nullptr;
        uint64_t host_noc = 0;
        uint32_t pcie_x = 0, pcie_y = 0;

        std::optional<x280::SessionId> session;
        uint32_t next = 1;  // number of the next command
        uint32_t seen = 0;  // events read from the ring
        std::vector<Event> pending;

        std::array<uint32_t, 2> translated(uint32_t x, uint32_t y) const {
            const auto t = x280::X280Runtime::virtual_core_from_logical_core(
                *mesh,
                tt::tt_metal::CoreCoord{x, y},
                tt::CoreType::WORKER,
                tt::tt_metal::NOC::NOC_0
            );
            return {static_cast<uint32_t>(t.x()), static_cast<uint32_t>(t.y())};
        }

        static Plan make(Runtime& r) {
            Plan p;
            p.mesh = r.mesh;
            p.rt = &r.runtime();
            const auto ids = p.rt->tile_ids();
            p.tile = &p.rt->tile(ids.front());
            const auto l2 = (
                p.rt->virtual_core_from_tile(ids.front(), tt::tt_metal::NOC::NOC_0)
            );
            p.l2x = static_cast<uint32_t>(l2.x());
            p.l2y = static_cast<uint32_t>(l2.y());
            p.base = p.tile->available_lim().begin;
            if (p.base % 64 != 0) {
                throw std::runtime_error(
                    "available LIM does not start on a 64-byte line"
                );
            }
            const auto g = p.mesh->compute_with_storage_grid_size();
            p.gx = g.x;
            p.gy = g.y;
            // Panels: runs of logical columns whose NoC0 x are consecutive.
            const auto& soc = (
                tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(
                    p.mesh->get_devices().front()->id()
                )
            );
            auto phys_x = [&](uint32_t lx) {
                const auto t = p.translated(lx, 0);
                return (
                    soc.translate_coord_to(
                                      tt::umd::CoreCoord(
                                          t[0],
                                          t[1],
                                          tt::CoreType::TENSIX,
                                          tt::CoordSystem::TRANSLATED
                                      ),
                                      tt::CoordSystem::NOC0
                    )
                    .x
                );
            };
            for (uint32_t x = 0; x < p.gx; x++) {
                if (!p.panels.empty() && phys_x(x) == phys_x(p.panels.back()[1]) + 1) {
                    p.panels.back()[1] = x;
                } else {
                    p.panels.push_back({x, x});
                }
            }
            if (p.panels.size() > 2 || p.gx > KMD_READY_MAX ||
                p.gx * p.gy > KM_CORES_MAX) {
                throw std::runtime_error("grid not supported: more than two panels");
            }
            return p;
        }

        // Core index -> translated NoC0 x | y << 16; core index = y * grid_x + x.
        std::vector<uint32_t> core_table() const {
            std::vector<uint32_t> t(KM_CORES_MAX, 0);
            for (uint32_t y = 0; y < gy; y++) {
                for (uint32_t x = 0; x < gx; x++) {
                    const auto c = translated(x, y);
                    t[y * gx + x] = c[0] | (c[1] << 16);
                }
            }
            return t;
        }

        // The stand-in worker's runtime arguments (kernels/worker.cpp, kArg*).
        std::vector<uint32_t> worker_args(
            uint32_t x, uint32_t y, uint32_t table, uint32_t out, uint32_t limit
        ) const {
            const auto leader = translated(x, gy - 1);
            const bool is_leader = y == gy - 1;
            return {
                y * gx + x,
                gx * gy,
                tag,
                l2x,
                l2y,
                static_cast<uint32_t>(base + KMD_LIM_MAILBOX),
                static_cast<uint32_t>(base + KMD_LIM_READY + 16u * x),
                static_cast<uint32_t>(base + KMD_LIM_ACKS),
                leader[0],
                leader[1],
                y,
                gy - 1,
                is_leader ? 1u : 0u,
                (x == 0 && y == 0) ? 1u : 0u,
                table,
                out,
                limit,
                depth
            };
        }

        // Picks the run and tag, zeroes the LIM layout and the host buffer, writes the
        // argument block. Call before building the worker arguments (they carry the
        // tag).
        void prepare(
            uint32_t depth_,
            uint32_t host_depth_,
            uint32_t event_depth_,
            double t_start_s,
            double t_record_s,
            double t_idle_s
        ) {
            if (session) {
                throw std::runtime_error("a kernel is running");
            }
            if (depth_ < 1 || depth_ > KM_RING_SLOTS || host_depth_ < 1 ||
                host_depth_ > KMD_HOST_DEPTH_MAX || event_depth_ < 1 ||
                KMD_HOST_RING + event_depth_ * sizeof(km_event_t) > kHostBufferBytes) {
                throw std::runtime_error(
                    "depth, host_depth or event_depth out of range"
                );
            }
            depth = depth_;
            host_depth = host_depth_;
            event_depth = event_depth_;
            std::random_device rd;
            run = (0x6b6d640000000000ull ^ (static_cast<uint64_t>(rd()) << 16)) + rd();
            do {
                tag = (rd() & 0xFFFFu) | 1u;
            } while (tag == 0xFFFFu);
            next = 1;
            seen = 0;
            pending.clear();
            open_host_side();
            std::memset(const_cast<uint32_t*>(host), 0, kHostBufferBytes);
            tile->write_available_lim(base, std::vector<uint8_t>(KMD_LIM_BYTES, 0));

            kmd_args_t a{};
            a.run_value = run;
            a.depth = depth;
            a.host_depth = host_depth;
            a.event_depth = event_depth;
            a.columns = gx;
            a.cores = gx * gy;
            a.panels = panels.size();
            for (size_t r = 0; r < panels.size(); r++) {
                const auto p0 = translated(panels[r][0], 0),
                           p1 = translated(panels[r][1], gy - 1);
                a.rect[r][0] = p0[0];
                a.rect[r][1] = p0[1];
                a.rect[r][2] = p1[0];
                a.rect[r][3] = p1[1];
            }
            a.host_noc_addr = host_noc;
            a.pcie_x = pcie_x;
            a.pcie_y = pcie_y;
            a.t_start = static_cast<uint64_t>(t_start_s * kMtimeHz);
            a.t_record = static_cast<uint64_t>(t_record_s * kMtimeHz);
            a.t_idle = static_cast<uint64_t>(t_idle_s * kMtimeHz);
            std::vector<uint8_t> block(256, 0);  // whole 64-byte lines
            std::memcpy(block.data(), &a, sizeof(a));
            tile->write_available_lim(base + KMD_LIM_ARGS, block);
        }

        // Enqueues the kernel image on cluster 0 with the prepared argument block.
        void start(const std::string& image) {
            if (session) {
                throw std::runtime_error("a kernel is running");
            }
            x280::ActiveRunRequest req;
            req.image_path = image;
            req.run_value = run;
            req.args_addr = base + KMD_LIM_ARGS;
            req.args_bytes = 256;
            const auto enq = rt->enqueue(*tile, req);
            if (!enq.session) {
                throw std::runtime_error("X280 enqueue failed: " + x280::describe(enq));
            }
            session = enq.session;
        }

        uint32_t host_word(uint32_t w) const {
            return host[w];
        }

        // Waits until the kernel reports READY or a final state; returns that state, or
        // 0 after the timeout.
        uint32_t wait_ready(double timeout_s) const {
            const auto by = after(timeout_s);
            for (;;) {
                const uint32_t s = host[KMD_HOST_STATE];
                if (s >= KMD_STATE_READY) {
                    return s;
                }
                if (Clock::now() > by) {
                    return 0;
                }
            }
        }

        // Moves the events that have arrived into `pending`; returns how many.
        uint32_t poll() {
            const auto* ring = (
                reinterpret_cast<const volatile uint32_t*>(
                    reinterpret_cast<const volatile uint8_t*>(host) + KMD_HOST_RING
                )
            );
            uint32_t n = 0;
            for (;;) {
                const uint32_t i = seen + 1;
                const volatile uint32_t* e = ring + (i % event_depth) * 4;
                if (e[0] != i) {
                    return n;
                }
                _mm_lfence();
                pending.push_back({i, e[1], e[2], e[3]});
                seen = i;
                n++;
            }
        }

        // Waits until command `i` fits: its ring slot is free (command i - host_depth
        // taken) and the event ring has room. Throws after `by` or if the kernel ended.
        void wait_room(uint32_t i, Clock::time_point by) {
            while (
                static_cast<int32_t>(i - host[KMD_HOST_CONSUMED]) >
                    static_cast<int32_t>(host_depth) ||
                i - seen > event_depth
            ) {
                poll();
                if (host[KMD_HOST_STATE] >= KMD_STATE_DONE) {
                    throw std::runtime_error(
                        "kernel finished with error " +
                        std::to_string(host[KMD_HOST_ERROR])
                    );
                }
                if (Clock::now() > by) {
                    throw std::runtime_error(
                        "command " + std::to_string(i) + " not accepted in time (" +
                        std::to_string(host[KMD_HOST_CONSUMED]) + " taken)"
                    );
                }
            }
        }

        // Stores command `i` (the record `rec` with i in seq and the landing words)
        // into its host ring slot, without a fence.
        void store_command(uint32_t i, const uint32_t* rec) {
            alignas(64) uint32_t w[16];
            std::memcpy(w, rec, sizeof(w));
            w[0] = w[4] = w[8] = w[12] = i;
            win->write_block(KMD_LIM_HRING + (i % host_depth) * 64u, w, 64);
        }

        // Stores command `next` once it fits, then a fence; returns its number. Throws
        // after timeout_s.
        uint32_t submit(const uint32_t* rec, double timeout_s) {
            const uint32_t i = next;
            wait_room(i, after(timeout_s));
            store_command(i, rec);
            _mm_sfence();
            next = i + 1;
            return i;
        }

        // Waits until `count` events in all have been read; returns the pending events
        // and clears them. Throws after timeout_s.
        std::vector<Event> take(uint32_t count, double timeout_s) {
            const auto by = after(timeout_s);
            while (static_cast<int32_t>(seen - count) < 0) {
                poll();
                if (static_cast<int32_t>(seen - count) >= 0) {
                    break;
                }
                if (host[KMD_HOST_STATE] >= KMD_STATE_DONE) {
                    break;
                }
                if (Clock::now() > by) {
                    throw std::runtime_error(
                        "events: " + std::to_string(seen) + " of " +
                        std::to_string(count)
                    );
                }
            }
            poll();
            std::vector<Event> out;
            out.swap(pending);
            return out;
        }

        // Issues the n records of `recs` (commands next .. next + n - 1) in batches of
        // up to `batch` (every command that fits, then one fence); collects the events.
        std::map<std::string, double> stream(
            const uint32_t* recs,
            size_t n,
            double timeout_s,
            size_t batch,
            std::vector<Event>& events
        ) {
            uint64_t store = 0;
            const uint32_t first_event = seen + 1;
            const auto t0 = Clock::now();
            const uint64_t c0 = __rdtsc();
            std::vector<uint64_t> each;  // store time per batch
            size_t r = 0;
            while (r < n) {
                const uint32_t i0 = next;
                wait_room(i0, after(timeout_s));
                size_t k = 1;
                while (
                    r + k < n && k < batch &&
                    static_cast<int32_t>(i0 + k - host[KMD_HOST_CONSUMED]) <=
                        static_cast<int32_t>(host_depth) &&
                    i0 + k - seen <= event_depth
                ) {
                    k++;
                }
                const uint64_t a = __rdtsc();
                for (size_t j = 0; j < k; j++) {
                    store_command(i0 + static_cast<uint32_t>(j), recs + 16 * (r + j));
                }
                _mm_sfence();
                each.push_back(__rdtsc() - a);
                store += each.back();
                next = i0 + static_cast<uint32_t>(k);
                r += k;
                if (pending.size() >= 4096) {
                    events.insert(events.end(), pending.begin(), pending.end());
                    pending.clear();
                }
            }
            const auto t1 = Clock::now();
            const uint64_t c1 = __rdtsc();
            auto rest = take(first_event + static_cast<uint32_t>(n) - 1u, timeout_s);
            const auto t2 = Clock::now();
            events.insert(events.end(), rest.begin(), rest.end());
            const double issue_s = std::chrono::duration<double>(t1 - t0).count();
            std::sort(each.begin(), each.end());
            const double tsc_hz = static_cast<double>(c1 - c0) / issue_s;
            return {
                {"issue_s", issue_s},
                {"total_s", std::chrono::duration<double>(t2 - t0).count()},
                {"store_s", static_cast<double>(store) / tsc_hz},
                {"tsc_hz", tsc_hz},
                {"batches", static_cast<double>(each.size())},
                {"batch_min_s", static_cast<double>(each.front()) / tsc_hz},
                {"batch_p50_s", static_cast<double>(each[each.size() / 2]) / tsc_hz},
                {"batch_max_s", static_cast<double>(each.back()) / tsc_hz},
            };
        }

        // Waits for the kernel's hart result, stops the session and reads the result
        // block.
        Finish finish(double timeout_s) {
            if (!session) {
                throw std::runtime_error("no kernel running");
            }
            Finish f;
            const auto res = (
                rt->session(*session).wait_until_result_ready(millis(timeout_s))
            );
            const auto cleaned = rt->cleanup(*session, std::chrono::seconds(5));
            session.reset();
            f.ready = res.succeeded();
            if (f.ready) {
                f.result = res.hartstates[0].state.result;
            } else {
                f.wait = x280::describe(res);
            }
            f.cleaned = cleaned.cleaned_up();
            if (!f.cleaned) {
                f.cleanup = x280::describe(cleaned);
            }
            const auto block = (
                tile->read_available_lim(base + KMD_LIM_RESULT, 4 * KMD_R_WORDS)
            );
            f.words.resize(KMD_R_WORDS);
            std::memcpy(f.words.data(), block.data(), 4 * KMD_R_WORDS);
            return f;
        }

        // Stops the session now if one runs (any failure path).
        bool abort() {
            if (!session) {
                return true;
            }
            const auto cleaned = rt->cleanup(*session, std::chrono::seconds(5));
            session.reset();
            return cleaned.cleaned_up();
        }

    private:
        void open_host_side() {
            if (win) {
                return;
            }
            const auto chip = mesh->get_devices().front()->id();
            auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            auto* dev = cluster.get_driver()->get_tt_device(chip);
            win = dev->create_io_window(
                tt::umd::TargetIoWindowConfig{
                    .core_start = {l2x, l2y}, .addr = base, .noc = tt::umd::NocId::NOC0
                },
                tt::umd::HostIoWindowConfig{.mapping = tt::umd::HostMemoryCaching::WC},
                tt::umd::IoOrdering::Strict
            );
            buffer = cluster.allocate_sysmem_buffer(chip, kHostBufferBytes, true);
            host = static_cast<volatile uint32_t*>(buffer->get_va());
            host_noc = buffer->get_noc_address().value();
            if ((host_noc & 0x1FFFFFu) + kHostBufferBytes > 0x200000u) {
                throw std::runtime_error("host buffer crosses a 2 MiB window");
            }
            const auto pcie = (
                cluster.get_soc_desc(chip).get_cores(
                    tt::CoreType::PCIE, tt::CoordSystem::TRANSLATED
                )
            );
            pcie_x = pcie.front().x;
            pcie_y = pcie.front().y;
        }
    };

    nb::list events_list(const std::vector<Event>& events) {
        nb::list out;
        for (const auto& e : events) {
            out.append(nb::make_tuple(e.index, e.req, e.status, e.ticks));
        }
        return out;
    }

}  // namespace

NB_MODULE(_km_dispatch_test, m) {
    m.attr("MTIME_HZ") = kMtimeHz;

    nb::class_<Runtime>(m, "Runtime")
        .def(
            nb::init<std::shared_ptr<MeshDevice>, uint32_t>(),
            nb::arg("mesh_device"),
            nb::arg("pll_mhz") = 1750
        )
        .def("image_limit", &Runtime::image_limit)
        .def(
            "plan",
            [](Runtime& self) {
                return Plan::make(self);
            },
            nb::keep_alive<0, 1>()
        );

    nb::class_<Plan>(m, "Plan")
        .def_ro("grid_x", &Plan::gx)
        .def_ro("grid_y", &Plan::gy)
        .def_ro("panels", &Plan::panels)
        .def_ro("tag", &Plan::tag)
        .def_ro("depth", &Plan::depth)
        .def_ro("host_depth", &Plan::host_depth)
        .def_ro("event_depth", &Plan::event_depth)
        .def_ro("next", &Plan::next)
        .def("core_table", &Plan::core_table)
        .def(
            "worker_args",
            &Plan::worker_args,
            nb::arg("x"),
            nb::arg("y"),
            nb::arg("table"),
            nb::arg("out"),
            nb::arg("limit")
        )
        .def(
            "prepare",
            &Plan::prepare,
            nb::arg("depth"),
            nb::arg("host_depth") = 16,
            nb::arg("event_depth") = 252,
            nb::arg("t_start_s") = 30.0,
            nb::arg("t_record_s") = 1.0,
            nb::arg("t_idle_s") = 30.0
        )
        .def("start", &Plan::start, nb::arg("image"))
        .def("host_word", &Plan::host_word, nb::arg("word"))
        .def(
            "wait_ready",
            [](const Plan& p, double timeout_s) {
                nb::gil_scoped_release release;
                return p.wait_ready(timeout_s);
            },
            nb::arg("timeout_s")
        )
        .def(
            "submit",
            [](Plan& p, const std::vector<uint32_t>& rec, double timeout_s) {
                if (rec.size() != 16) {
                    throw std::runtime_error("a record is 16 words");
                }
                return p.submit(rec.data(), timeout_s);
            },
            nb::arg("rec"),
            nb::arg("timeout_s") = 10.0
        )
        .def(
            "take",
            [](Plan& p, uint32_t count, double timeout_s) {
                return events_list(p.take(count, timeout_s));
            },
            nb::arg("count"),
            nb::arg("timeout_s") = 10.0
        )
        .def(
            "stream",
            [](Plan& p,
               nb::ndarray<const uint32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>
                   recs,
               double timeout_s,
               size_t batch) {
                if (recs.shape(1) != 16) {
                    throw std::runtime_error("recs is [n, 16]");
                }
                std::vector<Event> events;
                auto times = (
                    p.stream(recs.data(), recs.shape(0), timeout_s, batch, events)
                );
                return nb::make_tuple(events_list(events), times);
            },
            nb::arg("recs"),
            nb::arg("timeout_s") = 10.0,
            nb::arg("batch") = 1
        )
        .def(
            "finish",
            [](Plan& p, double timeout_s) {
                Finish f;
                {
                    nb::gil_scoped_release release;
                    f = p.finish(timeout_s);
                }
                nb::dict out;
                out["result_ready"] = f.ready;
                out["result"] = f.result ? nb::object(nb::int_(*f.result)) : nb::none();
                out["cleaned"] = f.cleaned;
                out["wait"] = f.wait;
                out["cleanup"] = f.cleanup;
                out["words"] = f.words;
                return out;
            },
            nb::arg("timeout_s") = 10.0
        )
        .def("abort", &Plan::abort)
        .def("running", [](const Plan& p) {
            return p.session.has_value();
        });
}
