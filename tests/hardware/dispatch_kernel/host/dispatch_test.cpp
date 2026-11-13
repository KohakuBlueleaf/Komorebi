// _km_dispatch_test, the harness's host module: X280 runtime, the kernel's LIM and
// host buffer, the host command ring, the events. Every wait is bounded.
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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <immintrin.h>
#include <x86intrin.h>

#include <tt-metalium/allocator.hpp>
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
#include "../kernels/dk_test.h"

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
        uint64_t base = 0;     // LIM address of the layout (the argument block)
        uint64_t lim_end = 0;  // end of the runtime's available LIM
        uint32_t upload_bytes = 0;
        std::vector<std::array<uint32_t, 2>> panels;  // logical column ranges
        uint64_t run = 0;
        uint32_t tag = 1, depth = 1, host_depth = 16, event_depth = 128;

        std::shared_ptr<tt::umd::IoWindow> win;         // host -> LIM, write-combining
        std::shared_ptr<tt::umd::IoWindow> l1_win;      // store probe only: host -> L1
        std::shared_ptr<tt::umd::SysmemBuffer> buffer;  // kernel -> pinned host memory
        volatile uint32_t* host = nullptr;
        uint64_t host_noc = 0;
        uint32_t pcie_x = 0, pcie_y = 0;

        std::optional<x280::SessionId> session;
        uint32_t next = 1;       // number of the next command
        uint32_t seen = 0;       // events read from the ring
        uint32_t published = 0;  // the read count last stored into LIM
        uint64_t fenced = 0;     // rdtsc at the last host-window fence
        double hz = 0;           // rdtsc ticks per second
        std::vector<Event> pending;
        // result_run: each core's expected run count, key sum and key mix
        std::vector<uint32_t> exp_runs, exp_sums, exp_mixes;

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
            p.lim_end = p.tile->available_lim().end;
            p.hz = tsc_hz();
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
            if ((table | out) % 16 != 0) {
                throw std::runtime_error("worker buffers not 16-byte aligned");
            }
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

        // Picks the run and tag, zeroes LIM layout and host buffer, writes the argument
        // block. Call before building the worker arguments (they carry the tag).
        void prepare(
            uint32_t depth_,
            uint32_t host_depth_,
            uint32_t event_depth_,
            double t_start_s,
            double t_record_s,
            double t_idle_s,
            uint64_t state_ddr,
            int64_t upload_bytes_
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
            const uint64_t room =
                (lim_end > base + KMD_LIM_UPLOAD
                     ? (lim_end - base - KMD_LIM_UPLOAD) & ~63ull
                     : 0);
            if (upload_bytes_ > static_cast<int64_t>(room) ||
                (upload_bytes_ >= 0 && upload_bytes_ % 64 != 0)) {
                throw std::runtime_error(
                    "upload_bytes: a multiple of 64, at most the room"
                );
            }
            depth = depth_;
            host_depth = host_depth_;
            upload_bytes = (
                static_cast<uint32_t>(upload_bytes_ < 0 ? room : upload_bytes_)
            );
            event_depth = event_depth_;
            std::random_device rd;
            run = (0x6b6d640000000000ull ^ (static_cast<uint64_t>(rd()) << 16)) + rd();
            do {
                tag = (rd() & 0xFFFFu) | 1u;
            } while (tag == 0xFFFFu);
            next = 1;
            seen = published = 0;
            exp_runs.assign(KM_CORES_MAX, 0);
            exp_sums.assign(KM_CORES_MAX, 0);
            exp_mixes.assign(KM_CORES_MAX, 0);
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
            a.state_ddr = state_ddr;
            a.upload_bytes = upload_bytes;
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

        // Stores `n` bytes (whole 64-byte lines) at `offset` of the upload region
        // through the host window, then a fence.
        void upload(uint64_t offset, const uint8_t* data, size_t n) {
            if (offset % 64 != 0 || n % 64 != 0 || offset > upload_bytes ||
                n > upload_bytes - offset) {
                throw std::runtime_error("upload: whole lines inside the region");
            }
            if (KMD_LIM_UPLOAD + offset + n > win->get_size()) {
                throw std::runtime_error("upload: beyond the host window");
            }
            win->write_block(KMD_LIM_UPLOAD + offset, data, n);
            _mm_sfence();
            fenced = __rdtsc();
        }

        // Reads `n` bytes of the upload region through the X280 runtime, once the
        // kernel has finished (a KMD_TRACE image's trace).
        std::vector<uint8_t> upload_read(uint64_t offset, size_t n) {
            if (session) {
                throw std::runtime_error("upload_read: the kernel is running");
            }
            if (offset > upload_bytes || n > upload_bytes - offset) {
                throw std::runtime_error("upload_read: inside the region");
            }
            return tile->read_available_lim(base + KMD_LIM_UPLOAD + offset, n);
        }

        uint32_t host_word(uint32_t w) const {
            return host[w];
        }

        // Waits until the kernel reports a final state, or READY and its ready block;
        // returns that state, or 0 after the timeout.
        uint32_t wait_ready(double timeout_s) const {
            const auto by = after(timeout_s);
            for (;;) {
                const uint32_t s = host[KMD_HOST_STATE];
                if (s > KMD_STATE_READY ||
                    (s == KMD_STATE_READY && host[KMD_HOST_READY] == KMD_READY_MARK)) {
                    return s;
                }
                if (Clock::now() > by) {
                    return 0;
                }
            }
        }

        // Moves the events that have arrived into `pending`, each read as a sequence
        // lock (index, fields, index again); returns how many. Throws on a lost event.
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
                const uint32_t index = e[0];
                if (index != i) {
                    if (static_cast<int32_t>(index - i) > 0) {
                        throw std::runtime_error(
                            "event " + std::to_string(i) + " overwritten by " +
                            std::to_string(index)
                        );
                    }
                    return n;
                }
                _mm_lfence();
                const Event ev{i, e[1], e[2], e[3]};
                _mm_lfence();
                if (e[0] != i) {
                    throw std::runtime_error(
                        "event " + std::to_string(i) + " changed while read"
                    );
                }
                pending.push_back(ev);
                seen = i;
                n++;
            }
        }

        // Stores the read count line into LIM (KMD_LIM_READ), without a fence.
        void store_read() {
            alignas(64) uint32_t w[16] = {seen};
            win->write_block(KMD_LIM_READ, w, 64);
            published = seen;
        }

        // While waiting: the read count and a fence, only if the kernel could hold a
        // command for it (command i can wait only if i - 1 - published >= depth).
        void publish() {
            if (seen != published && next - 1u - published >= event_depth &&
                __rdtsc() - fenced >= static_cast<uint64_t>(hz * 1e-6)) {
                store_read();
                _mm_sfence();
                fenced = __rdtsc();
            }
        }

        // Waits until command `i` fits: its ring slot is free (command i - host_depth
        // taken). Throws after `by` or if the kernel ended.
        void wait_room(uint32_t i, Clock::time_point by) {
            while (
                static_cast<int32_t>(i - host[KMD_HOST_CONSUMED]) >
                static_cast<int32_t>(host_depth)
            ) {
                poll();
                publish();
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
            poll();
            store_command(i, rec);
            store_read();
            _mm_sfence();
            fenced = __rdtsc();
            next = i + 1;
            return i;
        }

        // Waits until `count` events in all have been read; returns the pending events
        // and clears them. Throws after timeout_s.
        std::vector<Event> take(uint32_t count, double timeout_s) {
            const auto by = after(timeout_s);
            while (static_cast<int32_t>(seen - count) < 0) {
                poll();
                publish();
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

        // Issues `recs` as commands next.. in batches of up to `batch` plus the read
        // count, one fence each, a batch no sooner than settle_us after a fence.
        std::map<std::string, double> stream(
            const uint32_t* recs,
            size_t n,
            double timeout_s,
            size_t batch,
            double settle_us,
            std::vector<Event>& events
        ) {
            const auto settle = static_cast<uint64_t>(settle_us * 1e-6 * hz);
            uint64_t store = 0;
            const uint32_t first_event = seen + 1;
            const auto t0 = Clock::now();
            const uint64_t c0 = __rdtsc();
            std::vector<uint64_t> each;  // store time per batch
            size_t r = 0;
            while (r < n) {
                const uint32_t i0 = next;
                wait_room(i0, after(timeout_s));
                poll();  // the batch carries a fresh read count
                size_t k = 1;
                while (
                    r + k < n && k < batch &&
                    static_cast<int32_t>(i0 + k - host[KMD_HOST_CONSUMED]) <=
                        static_cast<int32_t>(host_depth)
                ) {
                    k++;
                }
                while (__rdtsc() - fenced < settle) {
                    _mm_pause();
                }
                const uint64_t a = __rdtsc();
                for (size_t j = 0; j < k; j++) {
                    store_command(i0 + static_cast<uint32_t>(j), recs + 16 * (r + j));
                }
                store_read();
                _mm_sfence();
                fenced = __rdtsc();
                each.push_back(fenced - a);
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

        // Streams n DK_SEG_RESULT records with KM_F_RESULT (key = command number, teams
        // of `count` cores); counts events other than DONE {first core, dk_result}.
        std::pair<std::map<std::string, double>, std::vector<Event>> result_run(
            uint32_t n, uint32_t count, uint32_t param, double timeout_s, size_t batch
        ) {
            const uint32_t cores = gx * gy;
            if (count < 1 || count > cores) {
                throw std::runtime_error("result_run: count 1 to the cores");
            }
            std::vector<uint32_t> recs(16 * static_cast<size_t>(n), 0);
            for (uint32_t j = 0; j < n; j++) {
                const uint32_t key = next + j, first = key % (cores - count + 1);
                uint32_t* w = recs.data() + 16 * static_cast<size_t>(j);
                w[1] = DK_SEG_RESULT | (KM_F_RESULT << 8);
                w[2] = first | (count << 16);
                w[3] = key;
                w[5] = param;
                for (uint32_t c = first; c < first + count; c++) {
                    exp_runs[c]++;
                    exp_sums[c] += key;
                    exp_mixes[c] ^= dk_mix(key);
                }
            }
            std::vector<Event> events, wrong;
            auto times = stream(recs.data(), n, timeout_s, batch, 0.0, events);
            uint32_t bad = 0;
            for (const auto& e : events) {
                const uint32_t first = e.req % (cores - count + 1);
                const uint32_t low = (e.status >> 8) &
                    KM_EV_AUX_LOW; /* not the seq bits */
                if ((e.status & 0xFFu) != KM_EV_DONE || low != first ||
                    e.ticks != dk_result(e.req, param)) {
                    if (bad++ < 8) {
                        wrong.push_back(e);
                    }
                }
            }
            times["events"] = static_cast<double>(events.size());
            times["wrong"] = bad;
            return {times, wrong};
        }

        // Runs a measuring image on cluster 0: argument block {run, words...}, a zeroed
        // result block; waits for the result and returns what finish() returns.
        Finish run_image(
            const std::string& image,
            const std::vector<uint64_t>& words,
            double timeout_s
        ) {
            if (session || words.size() > 31) {
                throw std::runtime_error("a kernel is running, or more than 31 words");
            }
            std::random_device rd;
            run = (0x6b6d640000000000ull ^ (static_cast<uint64_t>(rd()) << 16)) + rd();
            std::vector<uint8_t> block(KMD_LIM_RESULT + 4 * KMD_R_WORDS + 64 - 1, 0);
            block.resize(block.size() / 64 * 64);
            std::memcpy(block.data(), &run, 8);
            std::memcpy(block.data() + 8, words.data(), 8 * words.size());
            tile->write_available_lim(base, block);
            start(image);
            return finish(timeout_s);
        }

        // Waits for the kernel's hart result (or, with wait false, asks the runtime to
        // stop it at once), stops the session and reads the result block.
        Finish finish(double timeout_s, bool wait = true) {
            if (!session) {
                throw std::runtime_error("no kernel running");
            }
            Finish f;
            std::optional<x280::WaitResult> res;
            if (wait) {
                res = rt->session(*session).wait_until_result_ready(millis(timeout_s));
            }
            const auto cleaned = rt->cleanup(*session, std::chrono::seconds(5));
            session.reset();
            f.ready = wait ? res->succeeded() : cleaned.cleaned_up();
            if (f.ready) {
                f.result = (
                    (wait ? res->hartstates : cleaned.hartstates)[0].state.result
                );
            } else if (wait) {
                f.wait = x280::describe(*res);
            }
            f.cleaned = cleaned.cleaned_up();
            if (!f.cleaned) {
                f.cleanup = x280::describe(cleaned);
            }
            const auto block = (
                tile->read_available_lim(base + KMD_LIM_RESULT, 4 * KMD_R_WORDS)
            );
            const auto harts = (
                tile->read_available_lim(base + KMD_LIM_HARTS, 4 * KMD_H_WORDS)
            );
            f.words.resize(size_t{KMD_R_WORDS} + size_t{KMD_H_WORDS});
            std::memcpy(f.words.data(), block.data(), 4 * KMD_R_WORDS);
            std::memcpy(f.words.data() + KMD_R_WORDS, harts.data(), 4 * KMD_H_WORDS);
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

        // Store probe (no kernel): n bursts of `lines` lines and an sfence into LIM or
        // a Tensix L1, burst i at line lines * (i % spread); seconds in each part.
        std::vector<double> store_probe(
            uint32_t n,
            uint32_t lines,
            uint32_t spread,
            double gap_us,
            double busy_us,
            bool read,
            int target
        ) {
            if (session) {
                throw std::runtime_error("a kernel is running");
            }
            if (lines < 1 || spread < 1 || lines * spread > 64 || target < 0 ||
                target > 1) {
                throw std::runtime_error("lines * spread is 1 to 64, target 0 or 1");
            }
            open_host_side();
            tt::umd::IoWindow* w = win.get();
            uint64_t off = KMD_LIM_HRING, busy_off = KMD_LIM_ACKS;
            if (target == 1) {
                w = l1_window();
                off = 0;
                busy_off = 64u * 64u;
            }
            const double hz = tsc_hz();
            const auto ticks = [hz](double us) {
                return static_cast<uint64_t>(us * 1e-6 * hz);
            };
            // Keep-busy thread: one line and an sfence every busy_us until the end.
            std::atomic<bool> stop{false};
            std::thread busy;
            if (busy_us > 0) {
                busy = std::thread([&, period = ticks(busy_us)] {
                    alignas(64) uint32_t line[16] = {};
                    uint64_t at = __rdtsc();
                    while (!stop.load(std::memory_order_relaxed)) {
                        at += period;
                        while (__rdtsc() < at) {
                            _mm_pause();
                        }
                        line[0]++;
                        w->write_block(busy_off, line, 64);
                        _mm_sfence();
                    }
                });
            }
            alignas(64) uint32_t line[16] = {};
            std::vector<double> t(2 * n);
            const uint64_t gap = ticks(gap_us);
            uint64_t end = __rdtsc();
            for (uint32_t i = 0; i < n; i++) {
                while (__rdtsc() - end < gap) {
                    _mm_pause();
                }
                if (read) {
                    tile->read_available_lim(base + KMD_LIM_RESULT, 64);
                }
                const uint64_t a = __rdtsc();
                for (uint32_t j = 0; j < lines; j++) {
                    line[0] = i;
                    w->write_block(off + 64u * (lines * (i % spread) + j), line, 64);
                }
                const uint64_t b = __rdtsc();
                _mm_sfence();
                end = __rdtsc();
                t[2 * i] = static_cast<double>(b - a) / hz;
                t[2 * i + 1] = static_cast<double>(end - b) / hz;
            }
            stop = true;
            if (busy.joinable()) {
                busy.join();
            }
            return t;
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

        // A window onto logical core (0, 0)'s L1 at the allocator's base: free while no
        // program runs.
        tt::umd::IoWindow* l1_window() {
            if (!l1_win) {
                const auto chip = mesh->get_devices().front()->id();
                auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
                const auto c = translated(0, 0);
                l1_win = cluster.get_driver()->get_tt_device(chip)->create_io_window(
                    tt::umd::TargetIoWindowConfig{
                        .core_start = {c[0], c[1]},
                        .addr = mesh->allocator()->get_base_allocator_addr(
                            tt::tt_metal::HalMemType::L1
                        ),
                        .noc = tt::umd::NocId::NOC0
                    },
                    tt::umd::HostIoWindowConfig{
                        .mapping = tt::umd::HostMemoryCaching::WC
                    },
                    tt::umd::IoOrdering::Strict
                );
            }
            return l1_win.get();
        }

        // rdtsc ticks per second, over 20 ms.
        static double tsc_hz() {
            const auto w0 = Clock::now();
            const uint64_t c0 = __rdtsc();
            while (Clock::now() - w0 < std::chrono::milliseconds(20)) {
                ;
            }
            return (
                static_cast<double>(__rdtsc() - c0) /
                std::chrono::duration<double>(Clock::now() - w0).count()
            );
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
        .def_ro("seen", &Plan::seen)
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
            nb::arg("event_depth") = 128,
            nb::arg("t_start_s") = 30.0,
            nb::arg("t_record_s") = 1.0,
            nb::arg("t_idle_s") = 0.0,
            nb::arg("state_ddr") = 0,
            nb::arg("upload_bytes") = -1
        )
        .def_ro("upload_bytes", &Plan::upload_bytes)
        .def_ro("lim_base", &Plan::base)
        .def(
            "upload",
            [](Plan& p,
               uint64_t offset,
               nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>
                   data) {
                nb::gil_scoped_release release;
                p.upload(offset, data.data(), data.shape(0));
            },
            nb::arg("offset"),
            nb::arg("data")
        )
        .def(
            "upload_read",
            [](Plan& p, uint64_t offset, size_t n) {
                std::vector<uint8_t> data;
                {
                    nb::gil_scoped_release release;
                    data = p.upload_read(offset, n);
                }
                return (
                    nb::bytes(reinterpret_cast<const char*>(data.data()), data.size())
                );
            },
            nb::arg("offset"),
            nb::arg("n")
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
               size_t batch,
               double settle_us) {
                if (recs.shape(1) != 16) {
                    throw std::runtime_error("recs is [n, 16]");
                }
                std::vector<Event> events;
                auto times = (
                    p.stream(
                        recs.data(), recs.shape(0), timeout_s, batch, settle_us, events
                    )
                );
                return nb::make_tuple(events_list(events), times);
            },
            nb::arg("recs"),
            nb::arg("timeout_s") = 10.0,
            nb::arg("batch") = 1,
            nb::arg("settle_us") = 0.0
        )
        .def(
            "finish",
            [](Plan& p, double timeout_s, bool wait) {
                Finish f;
                {
                    nb::gil_scoped_release release;
                    f = p.finish(timeout_s, wait);
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
            nb::arg("timeout_s") = 10.0,
            nb::arg("wait") = true
        )
        .def(
            "run_image",
            [](Plan& p,
               const std::string& image,
               const std::vector<uint64_t>& words,
               double timeout_s) {
                Finish f;
                {
                    nb::gil_scoped_release release;
                    f = p.run_image(image, words, timeout_s);
                }
                nb::dict out;
                out["result_ready"] = f.ready;
                out["result"] = f.result ? nb::object(nb::int_(*f.result)) : nb::none();
                out["cleaned"] = f.cleaned;
                out["words"] = f.words;
                return out;
            },
            nb::arg("image"),
            nb::arg("words"),
            nb::arg("timeout_s") = 10.0
        )
        .def("abort", &Plan::abort)
        .def(
            "store_probe",
            [](Plan& p,
               uint32_t n,
               uint32_t lines,
               uint32_t spread,
               double gap_us,
               double busy_us,
               bool read,
               int target) {
                nb::gil_scoped_release release;
                return p.store_probe(n, lines, spread, gap_us, busy_us, read, target);
            },
            nb::arg("n"),
            nb::arg("lines") = 2,
            nb::arg("spread") = 1,
            nb::arg("gap_us") = 0.0,
            nb::arg("busy_us") = 0.0,
            nb::arg("read") = false,
            nb::arg("target") = 0
        )
        .def(
            "result_run",
            [](Plan& p,
               uint32_t n,
               uint32_t count,
               uint32_t param,
               double timeout_s,
               size_t batch) {
                std::pair<std::map<std::string, double>, std::vector<Event>> r;
                {
                    nb::gil_scoped_release release;
                    r = p.result_run(n, count, param, timeout_s, batch);
                }
                return nb::make_tuple(r.first, events_list(r.second));
            },
            nb::arg("n"),
            nb::arg("count") = 1,
            nb::arg("param") = 0,
            nb::arg("timeout_s") = 30.0,
            nb::arg("batch") = 64
        )
        .def(
            "expected",
            [](const Plan& p) {
                return nb::make_tuple(p.exp_runs, p.exp_sums, p.exp_mixes);
            }
        )
        .def("running", [](const Plan& p) {
            return p.session.has_value();
        });
}
