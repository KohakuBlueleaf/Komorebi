// _km_dispatch_test, the harness's host module: X280 runtime, the kernel's LIM and
// host buffer, the host command ring, the events. Every wait has a timeout.
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

    // The device in a komorebi Device.mesh_capsule().
    std::shared_ptr<MeshDevice> mesh_of(const nb::capsule& device) {
        const char* name = device.name();
        if (name == nullptr || std::strcmp(name, "komorebi.mesh_device") != 0) {
            throw std::invalid_argument("not a Device.mesh_capsule() capsule");
        }
        return *static_cast<std::shared_ptr<MeshDevice>*>(device.data());
    }

    constexpr double kMtimeHz = 50e6;
    // One page: the largest buffer that maps to the NoC without an IOMMU.
    constexpr size_t kHostBufferBytes = 4096;

    static_assert(sizeof(km_command_t) == 64);
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

        Runtime(std::shared_ptr<MeshDevice> mesh_device, uint32_t pll_mhz) :
            mesh(std::move(mesh_device)) {
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
            auto& x280_runtime = runtime();
            bool all_idle = true;
            for (const auto& id : x280_runtime.tile_ids()) {
                all_idle = (
                    all_idle && x280_runtime.tile(id).inspect().resident_idle_live
                );
            }
            if (!all_idle) {
                x280::MultiTileBootRequest boot_request;
                boot_request.tiles = x280_runtime.tile_ids();
                boot_request.idle_image_path = (
                    std::string(X280_IMAGE_DIR) + "/firmware-lim-idle.bin"
                );
                boot_request.allow_reset_release = true;
                if (!x280_runtime.boot_tiles(boot_request).ready()) {
                    throw std::runtime_error("X280 boot_tiles failed");
                }
            }
            x280_runtime.set_pll(x280_runtime.tile_ids().front().chip_id, pll_mhz);
        }

        x280::X280Runtime& runtime() {
            return *opened->runtime;
        }

        // The largest active image cluster 0 accepts, by the runtime's own check
        // against the tile's ECC prime command.
        std::map<std::string, nb::object> image_limit() {
            auto& x280_runtime = runtime();
            const auto state = (
                x280_runtime.tile(x280_runtime.tile_ids().front()).inspect()
            );
            uint64_t low = 0, high = 4u << 20;
            while (low + 1 < high) {
                const uint64_t mid = (low + high) / 2;
                const auto check = (
                    x280::validate_active_image(
                        X280_ACTIVE_FW_LOAD_ADDR, mid, state.ecc_primed_bytes
                    )
                );
                (check.ok ? low : high) = mid;
            }
            std::map<std::string, nb::object> out;
            out["observation_valid"] = nb::bool_(state.observation_valid);
            out["ecc"] = nb::str(x280::to_string(state.ecc));
            out["raw_ecc_extent"] = nb::int_(state.raw_ecc_extent);
            out["primed_bytes"] = state.ecc_primed_bytes
                ? nb::object(nb::int_(*state.ecc_primed_bytes))
                : nb::none();
            out["max_image_bytes"] = nb::int_(low);
            out["lim_backed_bytes"] = nb::int_(state.lim_backed_bytes);
            return out;
        }
    };

    struct Event {
        uint32_t index, req, status, ticks;
    };

    // What Plan::finish saw: the core result, the cleanup and the result block.
    struct Finish {
        bool ready = false, cleaned = false;
        std::optional<uint64_t> result;
        std::string wait, cleanup;
        std::vector<uint32_t> words;
    };

    struct Plan {
        std::shared_ptr<MeshDevice> mesh;
        x280::X280Runtime* x280_runtime = nullptr;
        x280::X280Tile* tile = nullptr;
        uint32_t grid_x = 0, grid_y = 0, l2_x = 0, l2_y = 0;
        uint64_t lim_base = 0;  // LIM address of the layout (the argument block)
        uint64_t lim_end = 0;   // end of the runtime's available LIM
        uint32_t upload_bytes = 0;
        std::vector<std::array<uint32_t, 2>> panels;  // logical column ranges
        uint64_t run = 0;
        uint32_t tag = 1, depth = 1, host_depth = 16, event_depth = 128;

        std::shared_ptr<tt::umd::IoWindow> lim_window;  // host -> LIM, write-combining
        std::shared_ptr<tt::umd::IoWindow>
            l1_io_window;                               // store probe only: host -> L1
        std::shared_ptr<tt::umd::SysmemBuffer> buffer;  // kernel -> pinned host memory
        volatile uint32_t* host = nullptr;
        uint64_t host_noc = 0;
        uint32_t pcie_x = 0, pcie_y = 0;

        std::optional<x280::SessionId> session;
        uint32_t next = 1;            // number of the next command
        uint32_t seen = 0;            // events read from the ring
        uint32_t published = 0;       // the read count last stored into LIM
        uint64_t fenced = 0;          // rdtsc at the last host-window fence
        double ticks_per_second = 0;  // rdtsc ticks per second
        std::vector<Event> pending;
        // result_run: each core's expected run count, key sum and key mix
        std::vector<uint32_t> expected_runs, expected_sums, expected_mixes;

        std::array<uint32_t, 2> translated(uint32_t x, uint32_t y) const {
            const auto translated_core = x280::X280Runtime::virtual_core_from_logical_core(
                *mesh,
                tt::tt_metal::CoreCoord{x, y},
                tt::CoreType::WORKER,
                tt::tt_metal::NOC::NOC_0
            );
            return {
                static_cast<uint32_t>(translated_core.x()),
                static_cast<uint32_t>(translated_core.y())
            };
        }

        static Plan make(Runtime& runtime) {
            Plan plan;
            plan.mesh = runtime.mesh;
            plan.x280_runtime = &runtime.runtime();
            const auto ids = plan.x280_runtime->tile_ids();
            plan.tile = &plan.x280_runtime->tile(ids.front());
            const auto l2 = (
                plan.x280_runtime->virtual_core_from_tile(
                    ids.front(), tt::tt_metal::NOC::NOC_0
                )
            );
            plan.l2_x = static_cast<uint32_t>(l2.x());
            plan.l2_y = static_cast<uint32_t>(l2.y());
            plan.lim_base = plan.tile->available_lim().begin;
            plan.lim_end = plan.tile->available_lim().end;
            plan.ticks_per_second = tsc_hz();
            if (plan.lim_base % 64 != 0) {
                throw std::runtime_error(
                    "available LIM does not start on a 64-byte line"
                );
            }
            const auto grid = plan.mesh->compute_with_storage_grid_size();
            plan.grid_x = grid.x;
            plan.grid_y = grid.y;
            // Panels: runs of logical columns whose NoC0 x are consecutive.
            const auto& soc = (
                tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(
                    plan.mesh->get_devices().front()->id()
                )
            );
            auto phys_x = [&](uint32_t logical_x) {
                const auto core = plan.translated(logical_x, 0);
                return (
                    soc.translate_coord_to(
                        tt::umd::CoreCoord(
                            core[0],
                            core[1],
                            tt::CoreType::TENSIX,
                            tt::CoordSystem::TRANSLATED
                        ),
                        tt::CoordSystem::NOC0
                    )
                    .x
                );
            };
            for (uint32_t x = 0; x < plan.grid_x; x++) {
                if (
                    !plan.panels.empty() &&
                    phys_x(x) == phys_x(plan.panels.back()[1]) + 1
                ) {
                    plan.panels.back()[1] = x;
                } else {
                    plan.panels.push_back({x, x});
                }
            }
            if (
                plan.panels.size() > 2 || plan.grid_x > KMD_READY_MAX ||
                plan.grid_x * plan.grid_y > KM_CORES_MAX
            ) {
                throw std::runtime_error("grid not supported: more than two panels");
            }
            return plan;
        }

        // Core index -> translated NoC0 x | y << 16; core index = y * grid_x + x.
        std::vector<uint32_t> core_table() const {
            std::vector<uint32_t> table(KM_CORES_MAX, 0);
            for (uint32_t y = 0; y < grid_y; y++) {
                for (uint32_t x = 0; x < grid_x; x++) {
                    const auto core = translated(x, y);
                    table[y * grid_x + x] = core[0] | (core[1] << 16);
                }
            }
            return table;
        }

        // The stand-in worker's runtime arguments (kernels/worker.cpp, kArg*).
        std::vector<uint32_t> worker_args(
            uint32_t x, uint32_t y, uint32_t table, uint32_t out, uint32_t limit
        ) const {
            if ((table | out) % 16 != 0) {
                throw std::runtime_error("worker buffers not 16-byte aligned");
            }
            const auto leader = translated(x, grid_y - 1);
            const bool is_leader = y == grid_y - 1;
            return {
                y * grid_x + x,
                grid_x * grid_y,
                tag,
                l2_x,
                l2_y,
                static_cast<uint32_t>(lim_base + KMD_LIM_MAILBOX),
                static_cast<uint32_t>(lim_base + KMD_LIM_READY + 16u * x),
                static_cast<uint32_t>(lim_base + KMD_LIM_ACKS),
                leader[0],
                leader[1],
                y,
                grid_y - 1,
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
            double t_command_s,
            double t_idle_s,
            uint64_t state_ddr,
            int64_t upload_bytes_
        ) {
            if (session) {
                throw std::runtime_error("a kernel is running");
            }
            if (
                depth_ < 1 || depth_ > KM_RING_SLOTS || host_depth_ < 1 ||
                host_depth_ > KMD_HOST_DEPTH_MAX || event_depth_ < 1 ||
                KMD_HOST_RING + event_depth_ * sizeof(km_event_t) > kHostBufferBytes
            ) {
                throw std::runtime_error(
                    "depth, host_depth or event_depth out of range"
                );
            }
            const uint64_t room = (
                lim_end > lim_base + KMD_LIM_UPLOAD
                    ? (lim_end - lim_base - KMD_LIM_UPLOAD) & ~63ull
                    : 0
            );
            if (
                upload_bytes_ > static_cast<int64_t>(room) ||
                (upload_bytes_ >= 0 && upload_bytes_ % 64 != 0)
            ) {
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
            std::random_device entropy;
            run = (
                (0x6b6d640000000000ull ^ (static_cast<uint64_t>(entropy()) << 16)) +
                entropy()
            );
            do {
                tag = (entropy() & 0xFFFFu) | 1u;
            } while (tag == 0xFFFFu);
            next = 1;
            seen = published = 0;
            expected_runs.assign(KM_CORES_MAX, 0);
            expected_sums.assign(KM_CORES_MAX, 0);
            expected_mixes.assign(KM_CORES_MAX, 0);
            pending.clear();
            open_host_side();
            std::memset(const_cast<uint32_t*>(host), 0, kHostBufferBytes);
            tile->write_available_lim(lim_base, std::vector<uint8_t>(KMD_LIM_BYTES, 0));

            kmd_args_t arguments{};
            arguments.run_value = run;
            arguments.depth = depth;
            arguments.host_depth = host_depth;
            arguments.event_depth = event_depth;
            arguments.columns = grid_x;
            arguments.cores = grid_x * grid_y;
            arguments.panels = panels.size();
            for (size_t panel = 0; panel < panels.size(); panel++) {
                const auto first_corner = translated(panels[panel][0], 0);
                const auto last_corner = translated(panels[panel][1], grid_y - 1);
                arguments.rect[panel][0] = first_corner[0];
                arguments.rect[panel][1] = first_corner[1];
                arguments.rect[panel][2] = last_corner[0];
                arguments.rect[panel][3] = last_corner[1];
            }
            arguments.host_noc_addr = host_noc;
            arguments.pcie_x = pcie_x;
            arguments.pcie_y = pcie_y;
            arguments.t_start = static_cast<uint64_t>(t_start_s * kMtimeHz);
            arguments.t_command = static_cast<uint64_t>(t_command_s * kMtimeHz);
            arguments.t_idle = static_cast<uint64_t>(t_idle_s * kMtimeHz);
            arguments.state_ddr = state_ddr;
            arguments.upload_bytes = upload_bytes;
            arguments.version = KMD_ARGS_VERSION;
            std::vector<uint8_t> block(256, 0);  // whole 64-byte lines
            std::memcpy(block.data(), &arguments, sizeof(arguments));
            tile->write_available_lim(lim_base + KMD_LIM_ARGS, block);
        }

        // Enqueues the kernel image on cluster 0 with the prepared argument block.
        void start(const std::string& image) {
            if (session) {
                throw std::runtime_error("a kernel is running");
            }
            x280::ActiveRunRequest run_request;
            run_request.image_path = image;
            run_request.run_value = run;
            run_request.args_addr = lim_base + KMD_LIM_ARGS;
            run_request.args_bytes = 256;
            const auto enqueued = x280_runtime->enqueue(*tile, run_request);
            if (!enqueued.session) {
                throw std::runtime_error(
                    "X280 enqueue failed: " + x280::describe(enqueued)
                );
            }
            session = enqueued.session;
        }

        // Stores `byte_count` bytes (whole 64-byte lines) at `offset` of the upload
        // region through the host window, then a fence.
        void upload(uint64_t offset, const uint8_t* data, size_t byte_count) {
            if (
                offset % 64 != 0 || byte_count % 64 != 0 || offset > upload_bytes ||
                byte_count > upload_bytes - offset
            ) {
                throw std::runtime_error("upload: whole lines inside the region");
            }
            if (KMD_LIM_UPLOAD + offset + byte_count > lim_window->get_size()) {
                throw std::runtime_error("upload: beyond the host window");
            }
            lim_window->write_block(KMD_LIM_UPLOAD + offset, data, byte_count);
            _mm_sfence();
            fenced = __rdtsc();
        }

        // Reads `byte_count` bytes of the upload region through the X280 runtime, once
        // the kernel has finished (a KMD_TRACE image's trace).
        std::vector<uint8_t> upload_read(uint64_t offset, size_t byte_count) {
            if (session) {
                throw std::runtime_error("upload_read: the kernel is running");
            }
            if (offset > upload_bytes || byte_count > upload_bytes - offset) {
                throw std::runtime_error("upload_read: inside the region");
            }
            return (
                tile->read_available_lim(lim_base + KMD_LIM_UPLOAD + offset, byte_count)
            );
        }

        uint32_t host_word(uint32_t word) const {
            return host[word];
        }

        // Waits until the kernel reports a final state, or READY and its ready block;
        // returns that state, or 0 after the timeout.
        uint32_t wait_ready(double timeout_s) const {
            const auto deadline = after(timeout_s);
            for (;;) {
                const uint32_t state = host[KMD_HOST_STATE];
                if (
                    state > KMD_STATE_READY ||
                    (state == KMD_STATE_READY && host[KMD_HOST_READY] == KMD_READY_MARK)
                ) {
                    return state;
                }
                if (Clock::now() > deadline) {
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
            uint32_t count = 0;
            for (;;) {
                const uint32_t i = seen + 1;
                const volatile uint32_t* event_words = ring + (i % event_depth) * 4;
                const uint32_t index = event_words[0];
                if (index != i) {
                    if (static_cast<int32_t>(index - i) > 0) {
                        throw std::runtime_error(
                            "event " + std::to_string(i) + " overwritten by " +
                            std::to_string(index)
                        );
                    }
                    return count;
                }
                _mm_lfence();
                const Event event{i, event_words[1], event_words[2], event_words[3]};
                _mm_lfence();
                if (event_words[0] != i) {
                    throw std::runtime_error(
                        "event " + std::to_string(i) + " changed while read"
                    );
                }
                pending.push_back(event);
                seen = i;
                count++;
            }
        }

        // Stores the read count line into LIM (KMD_LIM_READ), without a fence.
        void store_read() {
            alignas(64) uint32_t line[16] = {seen};
            lim_window->write_block(KMD_LIM_READ, line, 64);
            published = seen;
        }

        // While waiting: the read count and a fence, only if the kernel could hold a
        // command for it (command i can wait only if i - 1 - published >= depth).
        void publish() {
            if (
                seen != published && next - 1u - published >= event_depth &&
                __rdtsc() - fenced >= static_cast<uint64_t>(ticks_per_second * 1e-6)
            ) {
                store_read();
                _mm_sfence();
                fenced = __rdtsc();
            }
        }

        // Waits until command `i` fits: its ring slot is free (command i - host_depth
        // taken). Throws after `deadline` or if the kernel ended.
        void wait_room(uint32_t i, Clock::time_point deadline) {
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
                if (Clock::now() > deadline) {
                    throw std::runtime_error(
                        "command " + std::to_string(i) + " not accepted in time (" +
                        std::to_string(host[KMD_HOST_CONSUMED]) + " taken)"
                    );
                }
            }
        }

        // Stores command `i` (the command `command` with i in sequence and the landing
        // words) into its host ring slot, without a fence.
        void store_command(uint32_t i, const uint32_t* command) {
            alignas(64) uint32_t words[16];
            std::memcpy(words, command, sizeof(words));
            words[0] = words[4] = words[8] = words[12] = i;
            lim_window->write_block(KMD_LIM_HRING + (i % host_depth) * 64u, words, 64);
        }

        // Stores command `next` once it fits, then a fence; returns its number. Throws
        // after timeout_s.
        uint32_t submit(const uint32_t* command, double timeout_s) {
            const uint32_t i = next;
            wait_room(i, after(timeout_s));
            poll();
            store_command(i, command);
            store_read();
            _mm_sfence();
            fenced = __rdtsc();
            next = i + 1;
            return i;
        }

        // Waits until `count` events in all have been read; returns the pending events
        // and clears them. Throws after timeout_s.
        std::vector<Event> take(uint32_t count, double timeout_s) {
            const auto deadline = after(timeout_s);
            while (static_cast<int32_t>(seen - count) < 0) {
                poll();
                publish();
                if (static_cast<int32_t>(seen - count) >= 0) {
                    break;
                }
                if (host[KMD_HOST_STATE] >= KMD_STATE_DONE) {
                    break;
                }
                if (Clock::now() > deadline) {
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

        // Submits `commands` as commands next.. in batches of up to `batch` plus the
        // read count, one fence each, no sooner than settle_microseconds after one.
        std::map<std::string, double> stream(
            const uint32_t* commands,
            size_t command_count,
            double timeout_s,
            size_t batch,
            double settle_microseconds,
            std::vector<Event>& events
        ) {
            const auto settle = (
                static_cast<uint64_t>(settle_microseconds * 1e-6 * ticks_per_second)
            );
            uint64_t store = 0;
            const uint32_t first_event = seen + 1;
            const auto started = Clock::now();
            const uint64_t ticks_started = __rdtsc();
            std::vector<uint64_t> each;  // store time per batch
            size_t next_command = 0;
            while (next_command < command_count) {
                const uint32_t first_number = next;
                wait_room(first_number, after(timeout_s));
                poll();  // the batch carries a fresh read count
                size_t batch_size = 1;
                while (
                    next_command + batch_size < command_count && batch_size < batch &&
                    static_cast<int32_t>(
                        first_number + batch_size - host[KMD_HOST_CONSUMED]
                    ) <= static_cast<int32_t>(host_depth)
                ) {
                    batch_size++;
                }
                while (__rdtsc() - fenced < settle) {
                    _mm_pause();
                }
                const uint64_t store_started = __rdtsc();
                for (size_t j = 0; j < batch_size; j++) {
                    store_command(
                        first_number + static_cast<uint32_t>(j),
                        commands + 16 * (next_command + j)
                    );
                }
                store_read();
                _mm_sfence();
                fenced = __rdtsc();
                each.push_back(fenced - store_started);
                store += each.back();
                next = first_number + static_cast<uint32_t>(batch_size);
                next_command += batch_size;
                if (pending.size() >= 4096) {
                    events.insert(events.end(), pending.begin(), pending.end());
                    pending.clear();
                }
            }
            const auto submitted_at = Clock::now();
            const uint64_t ticks_submitted = __rdtsc();
            auto rest = (
                take(first_event + static_cast<uint32_t>(command_count) - 1u, timeout_s)
            );
            const auto finished_at = Clock::now();
            events.insert(events.end(), rest.begin(), rest.end());
            const double submit_s = (
                std::chrono::duration<double>(submitted_at - started).count()
            );
            std::sort(each.begin(), each.end());
            const double tsc_hz = (
                static_cast<double>(ticks_submitted - ticks_started) / submit_s
            );
            return {
                {"submit_s", submit_s},
                {
                    "total_s",
                    std::chrono::duration<double>(finished_at - started).count()
                },
                {"store_s", static_cast<double>(store) / tsc_hz},
                {"tsc_hz", tsc_hz},
                {"batches", static_cast<double>(each.size())},
                {"batch_min_s", static_cast<double>(each.front()) / tsc_hz},
                {"batch_p50_s", static_cast<double>(each[each.size() / 2]) / tsc_hz},
                {"batch_max_s", static_cast<double>(each.back()) / tsc_hz},
            };
        }

        // Streams `command_count` RESULT commands with KM_FLAG_RESULT (key = command
        // number, groups of `count` cores); counts events but DONE {first, result}.
        std::pair<std::map<std::string, double>, std::vector<Event>> result_run(
            uint32_t command_count,
            uint32_t count,
            uint32_t param,
            double timeout_s,
            size_t batch
        ) {
            const uint32_t cores = grid_x * grid_y;
            if (count < 1 || count > cores) {
                throw std::runtime_error("result_run: count 1 to the cores");
            }
            std::vector<uint32_t> commands(16 * static_cast<size_t>(command_count), 0);
            for (uint32_t j = 0; j < command_count; j++) {
                const uint32_t key = next + j, first = key % (cores - count + 1);
                uint32_t* words = commands.data() + 16 * static_cast<size_t>(j);
                words[1] = DK_HANDLER_RESULT | (KM_FLAG_RESULT << 8);
                words[2] = first | (count << 16);
                words[3] = key;
                words[5] = param;
                for (uint32_t core = first; core < first + count; core++) {
                    expected_runs[core]++;
                    expected_sums[core] += key;
                    expected_mixes[core] ^= dk_mix(key);
                }
            }
            std::vector<Event> events, wrong;
            auto times = (
                stream(commands.data(), command_count, timeout_s, batch, 0.0, events)
            );
            uint32_t bad = 0;
            for (const auto& event : events) {
                const uint32_t first = event.req % (cores - count + 1);
                const uint32_t low = (event.status >> 8) &
                    KM_EVENT_AUXILIARY_LOW; /* not the sequence bits */
                if (
                    (event.status & 0xFFu) != KM_EVENT_DONE || low != first ||
                    event.ticks != dk_result(event.req, param)
                ) {
                    if (bad++ < 8) {
                        wrong.push_back(event);
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
            std::random_device entropy;
            run = (
                (0x6b6d640000000000ull ^ (static_cast<uint64_t>(entropy()) << 16)) +
                entropy()
            );
            std::vector<uint8_t> block(KMD_LIM_RESULT + 4 * KMD_R_WORDS + 64 - 1, 0);
            block.resize(block.size() / 64 * 64);
            std::memcpy(block.data(), &run, 8);
            std::memcpy(block.data() + 8, words.data(), 8 * words.size());
            tile->write_available_lim(lim_base, block);
            start(image);
            return finish(timeout_s);
        }

        // Waits for the kernel's core result (or, with wait false, asks the runtime to
        // stop it at once), stops the session and reads the result block.
        Finish finish(double timeout_s, bool wait = true) {
            if (!session) {
                throw std::runtime_error("no kernel running");
            }
            Finish finished;
            std::optional<x280::WaitResult> wait_result;
            if (wait) {
                wait_result = (
                    x280_runtime->session(*session).wait_until_result_ready(
                        millis(timeout_s)
                    )
                );
            }
            const auto cleaned = (
                x280_runtime->cleanup(*session, std::chrono::seconds(5))
            );
            session.reset();
            finished.ready = wait ? wait_result->succeeded() : cleaned.cleaned_up();
            if (finished.ready) {
                finished.result = (
                    (wait ? wait_result->hartstates : cleaned.hartstates)[0]
                    .state.result
                );
            } else if (wait) {
                finished.wait = x280::describe(*wait_result);
            }
            finished.cleaned = cleaned.cleaned_up();
            if (!finished.cleaned) {
                finished.cleanup = x280::describe(cleaned);
            }
            const auto block = (
                tile->read_available_lim(lim_base + KMD_LIM_RESULT, 4 * KMD_R_WORDS)
            );
            const auto x280_cores = (
                tile->read_available_lim(lim_base + KMD_LIM_CORES, 4 * KMD_H_WORDS)
            );
            finished.words.resize(size_t{KMD_R_WORDS} + size_t{KMD_H_WORDS});
            std::memcpy(finished.words.data(), block.data(), 4 * KMD_R_WORDS);
            std::memcpy(
                finished.words.data() + KMD_R_WORDS, x280_cores.data(), 4 * KMD_H_WORDS
            );
            return finished;
        }

        // Stops the session now if one runs (any failure path).
        bool abort() {
            if (!session) {
                return true;
            }
            const auto cleaned = (
                x280_runtime->cleanup(*session, std::chrono::seconds(5))
            );
            session.reset();
            return cleaned.cleaned_up();
        }

        // Store probe (no kernel): `bursts` bursts of `lines` lines and an sfence into
        // LIM or a Tensix L1, burst i at line lines * (i % spread); seconds per part.
        std::vector<double> store_probe(
            uint32_t bursts,
            uint32_t lines,
            uint32_t spread,
            double gap_microseconds,
            double busy_microseconds,
            bool read,
            int target
        ) {
            if (session) {
                throw std::runtime_error("a kernel is running");
            }
            if (
                lines < 1 || spread < 1 || lines * spread > 64 || target < 0 ||
                target > 1
            ) {
                throw std::runtime_error("lines * spread is 1 to 64, target 0 or 1");
            }
            open_host_side();
            tt::umd::IoWindow* window = lim_window.get();
            uint64_t offset = KMD_LIM_HRING, busy_offset = KMD_LIM_ACKS;
            if (target == 1) {
                window = l1_window();
                offset = 0;
                busy_offset = 64u * 64u;
            }
            const double ticks_per_second = tsc_hz();
            const auto ticks = [ticks_per_second](double microseconds) {
                return static_cast<uint64_t>(microseconds * 1e-6 * ticks_per_second);
            };
            // Keep-busy thread: one line and an sfence every busy_microseconds until
            // the end.
            std::atomic<bool> stop{false};
            std::thread busy;
            if (busy_microseconds > 0) {
                busy = std::thread([&, period = ticks(busy_microseconds)] {
                    alignas(64) uint32_t line[16] = {};
                    uint64_t due = __rdtsc();
                    while (!stop.load(std::memory_order_relaxed)) {
                        due += period;
                        while (__rdtsc() < due) {
                            _mm_pause();
                        }
                        line[0]++;
                        window->write_block(busy_offset, line, 64);
                        _mm_sfence();
                    }
                });
            }
            alignas(64) uint32_t line[16] = {};
            std::vector<double> durations(2 * bursts);
            const uint64_t gap = ticks(gap_microseconds);
            uint64_t end = __rdtsc();
            for (uint32_t i = 0; i < bursts; i++) {
                while (__rdtsc() - end < gap) {
                    _mm_pause();
                }
                if (read) {
                    tile->read_available_lim(lim_base + KMD_LIM_RESULT, 64);
                }
                const uint64_t start = __rdtsc();
                for (uint32_t j = 0; j < lines; j++) {
                    line[0] = i;
                    window->write_block(
                        offset + 64u * (lines * (i % spread) + j), line, 64
                    );
                }
                const uint64_t stored = __rdtsc();
                _mm_sfence();
                end = __rdtsc();
                durations
                    [2 * i] = (static_cast<double>(stored - start) / ticks_per_second);
                durations
                    [
                        2 * i + 1
                    ] = (static_cast<double>(end - stored) / ticks_per_second);
            }
            stop = true;
            if (busy.joinable()) {
                busy.join();
            }
            return durations;
        }

    private:
        void open_host_side() {
            if (lim_window) {
                return;
            }
            const auto chip = mesh->get_devices().front()->id();
            auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            auto* device = cluster.get_driver()->get_tt_device(chip);
            lim_window = device->create_io_window(
                tt::umd::TargetIoWindowConfig{
                    .core_start = {l2_x, l2_y},
                    .addr = lim_base,
                    .noc = tt::umd::NocId::NOC0
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
            if (!l1_io_window) {
                const auto chip = mesh->get_devices().front()->id();
                auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
                const auto core = translated(0, 0);
                l1_io_window = cluster.get_driver()
                                   ->get_tt_device(chip)
                                   ->create_io_window(
                                       tt::umd::TargetIoWindowConfig{
                                           .core_start = {core[0], core[1]},
                                           .addr = mesh->allocator()
                                                       ->get_base_allocator_addr(
                                                           tt::tt_metal::HalMemType::L1
                                                       ),.noc = tt::umd::NocId::NOC0
                                       },
                                       tt::umd::HostIoWindowConfig{
                                           .mapping = tt::umd::HostMemoryCaching::WC
                                       },
                                       tt::umd::IoOrdering::Strict
                                   );
            }
            return l1_io_window.get();
        }

        // rdtsc ticks per second, over 20 ms.
        static double tsc_hz() {
            const auto wall_start = Clock::now();
            const uint64_t ticks_start = __rdtsc();
            while (Clock::now() - wall_start < std::chrono::milliseconds(20)) {
                ;
            }
            return (
                static_cast<double>(__rdtsc() - ticks_start) /
                std::chrono::duration<double>(Clock::now() - wall_start).count()
            );
        }
    };

    nb::list events_list(const std::vector<Event>& events) {
        nb::list out;
        for (const auto& event : events) {
            out.append(
                nb::make_tuple(event.index, event.req, event.status, event.ticks)
            );
        }
        return out;
    }

}  // namespace

NB_MODULE(_km_dispatch_test, python_module) {
    python_module.attr("MTIME_HZ") = kMtimeHz;

    nb::class_<Runtime>(python_module, "Runtime")
        .def(
            "__init__",
            [](Runtime* self, const nb::capsule& device, uint32_t pll_mhz) {
                new (self) Runtime(mesh_of(device), pll_mhz);
            },
            nb::arg("device"),
            nb::arg("pll_mhz") = 1000
        ).def("image_limit", &Runtime::image_limit).def(
            "plan",
            [](Runtime& self) {
                return Plan::make(self);
            },
            nb::keep_alive<0, 1>()
        );

    nb::class_<Plan>(python_module, "Plan")
        .def_ro("grid_x", &Plan::grid_x)
        .def_ro("grid_y", &Plan::grid_y)
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
        ).def(
            "prepare",
            &Plan::prepare,
            nb::arg("depth"),
            nb::arg("host_depth") = 16,
            nb::arg("event_depth") = 128,
            nb::arg("t_start_s") = 30.0,
            nb::arg("t_command_s") = 1.0,
            nb::arg("t_idle_s") = 0.0,
            nb::arg("state_ddr") = 0,
            nb::arg("upload_bytes") = -1
        ).def_ro("upload_bytes", &Plan::upload_bytes)
        .def_ro("lim_base", &Plan::lim_base)
        .def(
            "upload",
            [](
                Plan & plan,
                uint64_t offset,
                nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>
                    data
            ) {
                nb::gil_scoped_release release;
                plan.upload(offset, data.data(), data.shape(0));
            },
            nb::arg("offset"),
            nb::arg("data")
        ).def(
            "upload_read",
            [](Plan& plan, uint64_t offset, size_t byte_count) {
                std::vector<uint8_t> data;
                {
                    nb::gil_scoped_release release;
                    data = plan.upload_read(offset, byte_count);
                }
                return (
                    nb::bytes(reinterpret_cast<const char*>(data.data()), data.size())
                );
            },
            nb::arg("offset"),
            nb::arg("n")
        ).def("start", &Plan::start, nb::arg("image"))
        .def("host_word", &Plan::host_word, nb::arg("word"))
        .def(
            "wait_ready",
            [](const Plan& plan, double timeout_s) {
                nb::gil_scoped_release release;
                return plan.wait_ready(timeout_s);
            },
            nb::arg("timeout_s")
        ).def(
            "submit",
            [](Plan& plan, const std::vector<uint32_t>& command, double timeout_s) {
                if (command.size() != 16) {
                    throw std::runtime_error("a command is 16 words");
                }
                return plan.submit(command.data(), timeout_s);
            },
            nb::arg("rec"),
            nb::arg("timeout_s") = 10.0
        ).def(
            "take",
            [](Plan& plan, uint32_t count, double timeout_s) {
                return events_list(plan.take(count, timeout_s));
            },
            nb::arg("count"),
            nb::arg("timeout_s") = 10.0
        ).def(
            "stream",
            [](
                Plan & plan,
                nb::ndarray<const uint32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>
                    commands,
                double timeout_s,
                size_t batch,
                double settle_microseconds
            ) {
                if (commands.shape(1) != 16) {
                    throw std::runtime_error("recs is [n, 16]");
                }
                std::vector<Event> events;
                auto times = (
                    plan.stream(
                        commands.data(),
                        commands.shape(0),
                        timeout_s,
                        batch,
                        settle_microseconds,
                        events
                    )
                );
                return nb::make_tuple(events_list(events), times);
            },
            nb::arg("recs"),
            nb::arg("timeout_s") = 10.0,
            nb::arg("batch") = 1,
            nb::arg("settle_microseconds") = 0.0
        ).def(
            "finish",
            [](Plan& plan, double timeout_s, bool wait) {
                Finish finished;
                {
                    nb::gil_scoped_release release;
                    finished = plan.finish(timeout_s, wait);
                }
                nb::dict out;
                out["result_ready"] = finished.ready;
                out["result"] = finished.result ? nb::object(nb::int_(*finished.result))
                                                : nb::none();
                out["cleaned"] = finished.cleaned;
                out["wait"] = finished.wait;
                out["cleanup"] = finished.cleanup;
                out["words"] = finished.words;
                return out;
            },
            nb::arg("timeout_s") = 10.0,
            nb::arg("wait") = true
        ).def(
            "run_image",
            [](
                Plan & plan,
                const std::string& image,
                const std::vector<uint64_t>& words,
                double timeout_s
            ) {
                Finish finished;
                {
                    nb::gil_scoped_release release;
                    finished = plan.run_image(image, words, timeout_s);
                }
                nb::dict out;
                out["result_ready"] = finished.ready;
                out["result"] = finished.result ? nb::object(nb::int_(*finished.result))
                                                : nb::none();
                out["cleaned"] = finished.cleaned;
                out["words"] = finished.words;
                return out;
            },
            nb::arg("image"),
            nb::arg("words"),
            nb::arg("timeout_s") = 10.0
        ).def("abort", &Plan::abort).def(
            "store_probe",
            [](
                Plan & plan,
                uint32_t bursts,
                uint32_t lines,
                uint32_t spread,
                double gap_microseconds,
                double busy_microseconds,
                bool read,
                int target
            ) {
                nb::gil_scoped_release release;
                return (
                    plan.store_probe(
                        bursts,
                        lines,
                        spread,
                        gap_microseconds,
                        busy_microseconds,
                        read,
                        target
                    )
                );
            },
            nb::arg("n"),
            nb::arg("lines") = 2,
            nb::arg("spread") = 1,
            nb::arg("gap_microseconds") = 0.0,
            nb::arg("busy_microseconds") = 0.0,
            nb::arg("read") = false,
            nb::arg("target") = 0
        ).def(
            "result_run",
            [](
                Plan & plan,
                uint32_t command_count,
                uint32_t count,
                uint32_t param,
                double timeout_s,
                size_t batch
            ) {
                std::pair<std::map<std::string, double>, std::vector<Event>> outcome;
                {
                    nb::gil_scoped_release release;
                    outcome = (
                        plan.result_run(
                            command_count, count, param, timeout_s, batch
                        )
                    );
                }
                return nb::make_tuple(outcome.first, events_list(outcome.second));
            },
            nb::arg("n"),
            nb::arg("count") = 1,
            nb::arg("param") = 0,
            nb::arg("timeout_s") = 30.0,
            nb::arg("batch") = 64
        ).def(
            "expected",
            [](const Plan& plan) {
                return (
                    nb::make_tuple(
                        plan.expected_runs, plan.expected_sums, plan.expected_mixes
                    )
                );
            }
        ).def("running", [](const Plan& plan) {
            return plan.session.has_value();
        });
}
