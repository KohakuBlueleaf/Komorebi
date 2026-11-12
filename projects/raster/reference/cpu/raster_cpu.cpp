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
    std::vector<Record> records;
    std::vector<int32_t> counts, offsets, list;
    std::vector<int32_t> chunk_counts;  // (tiles, chunks)
    std::vector<float> colour, depth;
    std::vector<int32_t> owner;
};

struct Times {
    double setup, bin, fill;
};

static void for_tiles_of(const Record& r, int& tx0, int& tx1, int& ty0, int& ty1) {
    tx0 = r.box[0] / TILE;
    tx1 = r.box[1] / TILE;
    ty0 = r.box[2] / TILE;
    ty1 = r.box[3] / TILE;
}

static void fill_tile(Buffers& b, int tile) {
    const int tx = tile % TILES_X, ty = tile / TILES_X;
    const int x0 = tx * TILE, y0 = ty * TILE;
    float depth[TILE * TILE];
    int32_t owner[TILE * TILE];
    float colour[TILE * TILE][3];
    for (int i = 0; i < TILE * TILE; i++) {
        depth[i] = 0.0f;
        owner[i] = -1;
        for (int ch = 0; ch < 3; ch++) {
            colour[i][ch] = background(ch);
        }
    }
    for (int k = b.offsets[tile]; k < b.offsets[tile + 1]; k++) {
        const Record& r = b.records[b.list[k]];
        int xa = std::max(r.box[0], x0), xb = std::min(r.box[1], x0 + TILE - 1);
        int ya = std::max(r.box[2], y0), yb = std::min(r.box[3], y0 + TILE - 1);
        for (int y = ya; y <= yb; y++) {
            float py = (float)y + 0.5f;
            for (int x = xa; x <= xb; x++) {
                float px = (float)x + 0.5f;
                if (!covers(r, px, py)) {
                    continue;
                }
                int i = (y - y0) * TILE + (x - x0);
                float w = depth_at(r, px, py);
                if (nearer(w, r.tri, depth[i], owner[i])) {
                    depth[i] = w;
                    owner[i] = r.tri;
                    colour_at(r, px, py, colour[i]);
                }
            }
        }
    }
    for (int y = 0; y < TILE; y++) {
        for (int x = 0; x < TILE; x++) {
            int i = y * TILE + x;
            size_t g = (size_t)(y0 + y) * WIDTH + (x0 + x);
            b.depth[g] = depth[i];
            b.owner[g] = owner[i];
            for (int ch = 0; ch < 3; ch++) {
                b.colour[3 * g + ch] = colour[i][ch];
            }
        }
    }
}

static Times run(const Mesh& mesh, Pool& pool, Buffers& b) {
    const int n_threads = pool.size();
    const int n_tiles = TILES_X * TILES_Y;
    const int n_chunks = n_threads * 2;
    const int64_t per_chunk = (mesh.n_tris + n_chunks - 1) / n_chunks;
    Times t;
    double t0 = now_ms();
    pool.for_chunks(mesh.n_tris, 256, [&](int64_t a, int64_t e) {
        for (int64_t i = a; i < e; i++) {
            setup_triangle(mesh, (int)i, b.records[i]);
        }
    });
    double t1 = now_ms();
    std::fill(b.chunk_counts.begin(), b.chunk_counts.end(), 0);
    pool.for_chunks(n_chunks, 1, [&](int64_t c, int64_t) {
        int32_t* count = &b.chunk_counts[c];
        int64_t end = std::min<int64_t>((c + 1) * per_chunk, mesh.n_tris);
        for (int64_t i = c * per_chunk; i < end; i++) {
            const Record& r = b.records[i];
            if (r.tri < 0) {
                continue;
            }
            int tx0, tx1, ty0, ty1;
            for_tiles_of(r, tx0, tx1, ty0, ty1);
            for (int ty = ty0; ty <= ty1; ty++) {
                for (int tx = tx0; tx <= tx1; tx++) {
                    count[(ty * TILES_X + tx) * n_chunks]++;
                }
            }
        }
    });
    int32_t total = 0;
    for (int tile = 0; tile < n_tiles; tile++) {
        b.offsets[tile] = total;
        for (int c = 0; c < n_chunks; c++) {
            int32_t n = b.chunk_counts[tile * n_chunks + c];
            b.chunk_counts[tile * n_chunks + c] = total;
            total += n;
        }
        b.counts[tile] = total - b.offsets[tile];
    }
    b.offsets[n_tiles] = total;
    b.list.resize(total);
    pool.for_chunks(n_chunks, 1, [&](int64_t c, int64_t) {
        int32_t* cursor = &b.chunk_counts[c];
        int64_t end = std::min<int64_t>((c + 1) * per_chunk, mesh.n_tris);
        for (int64_t i = c * per_chunk; i < end; i++) {
            const Record& r = b.records[i];
            if (r.tri < 0) {
                continue;
            }
            int tx0, tx1, ty0, ty1;
            for_tiles_of(r, tx0, tx1, ty0, ty1);
            for (int ty = ty0; ty <= ty1; ty++) {
                for (int tx = tx0; tx <= tx1; tx++) {
                    b.list[cursor[(ty * TILES_X + tx) * n_chunks]++] = (int32_t)i;
                }
            }
        }
    });
    double t2 = now_ms();
    std::vector<int> order(n_tiles);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int p, int q) {
        return b.counts[p] > b.counts[q];
    });
    pool.for_chunks(n_tiles, 1, [&](int64_t k, int64_t) {
        fill_tile(b, order[k]);
    });
    double t3 = now_ms();
    t.setup = t1 - t0;
    t.bin = t2 - t1;
    t.fill = t3 - t2;
    return t;
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    std::string scene = args.get("scene"), out = args.get("out");
    int n_threads = args.geti("threads", (int)std::thread::hardware_concurrency());
    int reps = args.geti("reps", 3);
    Array pos = load_npy(scene + "/positions.npy", 4);
    Array col = load_npy(scene + "/colors.npy", 4);
    Array idx = load_npy(scene + "/indices.npy", 4);
    Array cam = load_npy(scene + "/camera.npy", 4);
    Mesh mesh;
    mesh.positions = pos.data<float>();
    mesh.colors = col.data<float>();
    mesh.indices = idx.data<int32_t>();
    mesh.n_tris = (int)idx.shape[0];
    std::copy(cam.data<float>(), cam.data<float>() + 16, mesh.camera);

    Pool pool(n_threads);
    Buffers b;
    const int n_tiles = TILES_X * TILES_Y;
    b.records.resize(mesh.n_tris);
    b.counts.resize(n_tiles);
    b.offsets.resize(n_tiles + 1);
    b.chunk_counts.resize((size_t)n_threads * 2 * n_tiles);
    b.colour.resize((size_t)WIDTH * HEIGHT * 3);
    b.depth.resize((size_t)WIDTH * HEIGHT);
    b.owner.resize((size_t)WIDTH * HEIGHT);

    run(mesh, pool, b);  // warm-up: page faults, thread start
    RepLog log(reps);
    for (int rep = 0; rep < reps; rep++) {
        Times t = run(mesh, pool, b);
        char record[256];
        std::snprintf(
            record,
            sizeof record,
            "{\"setup_ms\": %.4f, \"bin_ms\": %.4f, \"fill_ms\": %.4f, \"total_ms\": "
            "%.4f}",
            t.setup,
            t.bin,
            t.fill,
            t.setup + t.bin + t.fill
        );
        log.add(record);
    }
    double loop_ms = log.finish();
    int valid = 0;
    for (const Record& r : b.records) {
        valid += r.tri >= 0;
    }
    std::printf(
        "{\"impl\": \"cpu\", \"threads\": %d, \"triangles\": %d, \"reps_n\": %d, "
        "\"loop_ms\": %.4f, \"valid\": %d, \"entries\": %d, \"reps\": [%s]}\n",
        n_threads,
        mesh.n_tris,
        reps,
        loop_ms,
        valid,
        b.offsets[n_tiles],
        log.body.c_str()
    );
    if (!out.empty()) {
        save_f32(out + "/colour.npy", b.colour.data(), {HEIGHT, WIDTH, 3});
        save_f32(out + "/depth.npy", b.depth.data(), {HEIGHT, WIDTH});
        save_i32(out + "/tri.npy", b.owner.data(), {HEIGHT, WIDTH});
        save_i32(out + "/tile_counts.npy", b.counts.data(), {TILES_Y, TILES_X});
    }
    return 0;
}
