// Rasterizer on the host CPU: setup and binning over triangles, fill over tiles.
// Usage: raster_cpu --scene DIR --out DIR [--threads N] [--reps R]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "host.hpp"
#include "raster_core.hpp"

using namespace kr;

struct Buffers {
    std::vector<Record> records;  // (2T): first pieces, then second pieces
    std::vector<uint8_t> pieces;  // per triangle, setup_triangle's bits
    std::vector<int32_t> counts, offsets, list;
    std::vector<int32_t> chunk_counts;  // (tiles, chunks)
    std::vector<float> colour, depth;
    std::vector<int32_t> owner;
};

struct Times {
    double setup, bin, fill;
};

static void for_tiles_of(
    const Record& record,
    int& first_tile_x,
    int& last_tile_x,
    int& first_tile_y,
    int& last_tile_y
) {
    first_tile_x = record.box[0] / TILE;
    last_tile_x = record.box[1] / TILE;
    first_tile_y = record.box[2] / TILE;
    last_tile_y = record.box[3] / TILE;
}

static void fill_tile(Buffers& buffers, int tile) {
    const int tile_x = tile % TILES_X, tile_y = tile / TILES_X;
    const int left = tile_x * TILE, top = tile_y * TILE;
    float depth[TILE * TILE];
    int32_t owner[TILE * TILE];
    float colour[TILE * TILE][3];
    for (int i = 0; i < TILE * TILE; i++) {
        depth[i] = 0.0f;
        owner[i] = -1;
        for (int channel = 0; channel < 3; channel++) {
            colour[i][channel] = background(channel);
        }
    }
    for (
        int entry = buffers.offsets[tile]; entry < buffers.offsets[tile + 1]; entry++
    ) {
        const Record& record = buffers.records[buffers.list[entry]];
        int first_x = std::max(record.box[0], left);
        int last_x = std::min(record.box[1], left + TILE - 1);
        int first_y = std::max(record.box[2], top);
        int last_y = std::min(record.box[3], top + TILE - 1);
        for (int y = first_y; y <= last_y; y++) {
            float py = (float)y + 0.5f;
            for (int x = first_x; x <= last_x; x++) {
                float px = (float)x + 0.5f;
                if (!covers(record, px, py)) {
                    continue;
                }
                int i = (y - top) * TILE + (x - left);
                float inverse_z = depth_at(record, px, py);
                if (nearer(inverse_z, record.triangle, depth[i], owner[i])) {
                    depth[i] = inverse_z;
                    owner[i] = record.triangle;
                    colour_at(record, px, py, colour[i]);
                }
            }
        }
    }
    for (int y = 0; y < TILE; y++) {
        for (int x = 0; x < TILE; x++) {
            int i = y * TILE + x;
            size_t pixel = (size_t)(top + y) * WIDTH + (left + x);
            buffers.depth[pixel] = depth[i];
            buffers.owner[pixel] = owner[i];
            for (int channel = 0; channel < 3; channel++) {
                buffers.colour[3 * pixel + channel] = colour[i][channel];
            }
        }
    }
}

static Times run(const Mesh& mesh, Pool& pool, Buffers& buffers) {
    const int thread_count = pool.size();
    const int tile_count = TILES_X * TILES_Y;
    const int chunk_count = thread_count * 2;
    const int64_t per_chunk = (mesh.triangle_count + chunk_count - 1) / chunk_count;
    Times times;
    double started = now_ms();
    pool.for_chunks(mesh.triangle_count, 256, [&](int64_t first, int64_t end) {
        for (int64_t i = first; i < end; i++) {
            buffers.pieces[i] = (
                (uint8_t)setup_triangle(
                    mesh,
                    (int)i,
                    buffers.records[i],
                    buffers.records[mesh.triangle_count + i]
                )
            );
        }
    });
    double setup_done = now_ms();
    std::fill(buffers.chunk_counts.begin(), buffers.chunk_counts.end(), 0);
    pool.for_chunks(chunk_count, 1, [&](int64_t chunk, int64_t) {
        int32_t* count = &buffers.chunk_counts[chunk];
        int64_t end = std::min<int64_t>((chunk + 1) * per_chunk, mesh.triangle_count);
        for (int64_t i = chunk * per_chunk; i < end; i++) {
            for (int piece = 0; piece < 2; piece++) {
                if (!((buffers.pieces[i] >> piece) & 1)) {
                    continue;
                }
                const Record& record = (
                    buffers.records[i + piece * (int64_t)mesh.triangle_count]
                );
                int first_tile_x, last_tile_x, first_tile_y, last_tile_y;
                for_tiles_of(
                    record, first_tile_x, last_tile_x, first_tile_y, last_tile_y
                );
                for (int tile_y = first_tile_y; tile_y <= last_tile_y; tile_y++) {
                    for (int tile_x = first_tile_x; tile_x <= last_tile_x; tile_x++) {
                        count[(tile_y * TILES_X + tile_x) * chunk_count]++;
                    }
                }
            }
        }
    });
    int32_t total = 0;
    for (int tile = 0; tile < tile_count; tile++) {
        buffers.offsets[tile] = total;
        for (int chunk = 0; chunk < chunk_count; chunk++) {
            int32_t entries = buffers.chunk_counts[tile * chunk_count + chunk];
            buffers.chunk_counts[tile * chunk_count + chunk] = total;
            total += entries;
        }
        buffers.counts[tile] = total - buffers.offsets[tile];
    }
    buffers.offsets[tile_count] = total;
    buffers.list.resize(total);
    pool.for_chunks(chunk_count, 1, [&](int64_t chunk, int64_t) {
        int32_t* cursor = &buffers.chunk_counts[chunk];
        int64_t end = std::min<int64_t>((chunk + 1) * per_chunk, mesh.triangle_count);
        for (int64_t i = chunk * per_chunk; i < end; i++) {
            for (int piece = 0; piece < 2; piece++) {
                if (!((buffers.pieces[i] >> piece) & 1)) {
                    continue;
                }
                const int64_t record_index = i + piece * (int64_t)mesh.triangle_count;
                const Record& record = buffers.records[record_index];
                int first_tile_x, last_tile_x, first_tile_y, last_tile_y;
                for_tiles_of(
                    record, first_tile_x, last_tile_x, first_tile_y, last_tile_y
                );
                for (int tile_y = first_tile_y; tile_y <= last_tile_y; tile_y++) {
                    for (int tile_x = first_tile_x; tile_x <= last_tile_x; tile_x++) {
                        buffers.list[
                            cursor [(tile_y * TILES_X + tile_x) * chunk_count]++
                        ] = (int32_t)record_index;
                    }
                }
            }
        }
    });
    double bin_done = now_ms();
    std::vector<int> order(tile_count);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int first_tile, int second_tile) {
        return buffers.counts[first_tile] > buffers.counts[second_tile];
    });
    pool.for_chunks(tile_count, 1, [&](int64_t rank, int64_t) {
        fill_tile(buffers, order[rank]);
    });
    double fill_done = now_ms();
    times.setup = setup_done - started;
    times.bin = bin_done - setup_done;
    times.fill = fill_done - bin_done;
    return times;
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    std::string scene = args.get("scene"), out_folder = args.get("out");
    int thread_count = args.geti("threads", (int)std::thread::hardware_concurrency());
    int repetitions = args.geti("reps", 3);
    Array positions = load_npy(scene + "/positions.npy", 4);
    Array colours = load_npy(scene + "/colors.npy", 4);
    Array indices = load_npy(scene + "/indices.npy", 4);
    Array camera = load_npy(scene + "/camera.npy", 4);
    Mesh mesh;
    mesh.positions = positions.data<float>();
    mesh.colors = colours.data<float>();
    mesh.indices = indices.data<int32_t>();
    mesh.triangle_count = (int)indices.shape[0];
    std::copy(camera.data<float>(), camera.data<float>() + 16, mesh.camera);

    Pool pool(thread_count);
    Buffers buffers;
    const int tile_count = TILES_X * TILES_Y;
    buffers.records.resize(2 * (size_t)mesh.triangle_count);
    buffers.pieces.resize(mesh.triangle_count);
    buffers.counts.resize(tile_count);
    buffers.offsets.resize(tile_count + 1);
    buffers.chunk_counts.resize((size_t)thread_count * 2 * tile_count);
    buffers.colour.resize((size_t)WIDTH * HEIGHT * 3);
    buffers.depth.resize((size_t)WIDTH * HEIGHT);
    buffers.owner.resize((size_t)WIDTH * HEIGHT);

    run(mesh, pool, buffers);  // warm-up: page faults, thread start
    RepLog log(repetitions);
    for (int repetition = 0; repetition < repetitions; repetition++) {
        Times times = run(mesh, pool, buffers);
        char record[256];
        std::snprintf(
            record,
            sizeof record,
            "{\"setup_ms\": %.4f, \"bin_ms\": %.4f, \"fill_ms\": %.4f, \"total_ms\": "
            "%.4f}",
            times.setup,
            times.bin,
            times.fill,
            times.setup + times.bin + times.fill
        );
        log.add(record);
    }
    double loop_ms = log.finish();
    int valid = 0;
    for (uint8_t bits : buffers.pieces) {
        valid += (bits & 1) + (bits >> 1);
    }
    std::printf(
        "{\"impl\": \"cpu\", \"threads\": %d, \"triangles\": %d, \"reps_n\": %d, "
        "\"loop_ms\": %.4f, \"valid\": %d, \"entries\": %d, \"reps\": [%s]}\n",
        thread_count,
        mesh.triangle_count,
        repetitions,
        loop_ms,
        valid,
        buffers.offsets[tile_count],
        log.body.c_str()
    );
    if (!out_folder.empty()) {
        save_f32(out_folder + "/colour.npy", buffers.colour.data(), {HEIGHT, WIDTH, 3});
        save_f32(out_folder + "/depth.npy", buffers.depth.data(), {HEIGHT, WIDTH});
        save_i32(out_folder + "/tri.npy", buffers.owner.data(), {HEIGHT, WIDTH});
        save_i32(
            out_folder + "/tile_counts.npy", buffers.counts.data(), {TILES_Y, TILES_X}
        );
    }
    return 0;
}
