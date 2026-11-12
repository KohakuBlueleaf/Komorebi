// Rasterizer on one GPU: setup and binning with one thread per triangle, fill with one
// block per tile. Usage: raster_gpu --scene DIR --out DIR [--reps R]
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "host.hpp"
#include "raster_core.hpp"

using namespace kr;

#define CHECK(call)                                                                 \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            std::fprintf(                                                           \
                stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_) \
            );                                                                      \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

constexpr int N_TILES = TILES_X * TILES_Y;
constexpr int BATCH = 64;

__global__ void setup_kernel(Mesh mesh, Record* records, int32_t* counts) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= mesh.n_tris) {
        return;
    }
    Record& r = records[t];
    setup_triangle(mesh, t, r);
    if (r.tri < 0) {
        return;
    }
    for (int ty = r.box[2] / TILE; ty <= r.box[3] / TILE; ty++) {
        for (int tx = r.box[0] / TILE; tx <= r.box[1] / TILE; tx++) {
            atomicAdd(&counts[ty * TILES_X + tx], 1);
        }
    }
}

// Exclusive scan of the 1024 tile counts by one block; cursor starts as the offsets.
__global__ void scan_kernel(const int32_t* counts, int32_t* offsets, int32_t* cursor) {
    __shared__ int32_t s[N_TILES];
    int i = threadIdx.x;
    s[i] = counts[i];
    __syncthreads();
    for (int d = 1; d < N_TILES; d *= 2) {
        int32_t add = i >= d ? s[i - d] : 0;
        __syncthreads();
        s[i] += add;
        __syncthreads();
    }
    int32_t before = s[i] - counts[i];
    offsets[i] = before;
    cursor[i] = before;
    if (i == N_TILES - 1) {
        offsets[N_TILES] = s[i];
    }
}

__global__ void scatter_kernel(
    Mesh mesh, const Record* records, int32_t* cursor, int32_t* list
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= mesh.n_tris || records[t].tri < 0) {
        return;
    }
    const Record& r = records[t];
    for (int ty = r.box[2] / TILE; ty <= r.box[3] / TILE; ty++) {
        for (int tx = r.box[0] / TILE; tx <= r.box[1] / TILE; tx++) {
            list[atomicAdd(&cursor[ty * TILES_X + tx], 1)] = t;
        }
    }
}

// Bitonic sort of the tiles by list size, largest first (ties: lower tile first).
__global__ void order_kernel(const int32_t* counts, int32_t* order) {
    __shared__ uint32_t key[N_TILES];
    int i = threadIdx.x;
    key[i] = (uint32_t)counts[i] * 1024u + (uint32_t)(N_TILES - 1 - i);
    __syncthreads();
    for (int k = 2; k <= N_TILES; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int partner = i ^ j;
            if (partner > i) {
                bool descending = (i & k) == 0;
                uint32_t a = key[i], b = key[partner];
                if ((a < b) == descending) {
                    key[i] = b;
                    key[partner] = a;
                }
            }
            __syncthreads();
        }
    }
    order[i] = N_TILES - 1 - (int)(key[i] & 1023u);
}

__global__ void fill_kernel(
    const Record* records,
    const int32_t* list,
    const int32_t* offsets,
    const int32_t* order,
    float* colour,
    float* depth_out,
    int32_t* owner_out
) {
    __shared__ Record batch[BATCH];
    const int tile = order[blockIdx.x];
    const int x = (tile % TILES_X) * TILE + threadIdx.x;
    const int y0 = (tile / TILES_X) * TILE + threadIdx.y;
    const int tid = threadIdx.y * TILE + threadIdx.x;
    const float px = (float)x + 0.5f;
    float depth[4], col[4][3];
    int owner[4];
    for (int k = 0; k < 4; k++) {
        depth[k] = 0.0f;
        owner[k] = -1;
        for (int ch = 0; ch < 3; ch++) {
            col[k][ch] = background(ch);
        }
    }
    const int begin = offsets[tile], end = offsets[tile + 1];
    for (int base = begin; base < end; base += BATCH) {
        int n = min(BATCH, end - base);
        __syncthreads();
        if (tid < n) {
            batch[tid] = records[list[base + tid]];
        }
        __syncthreads();
        for (int j = 0; j < n; j++) {
            const Record& r = batch[j];
            if (x < r.box[0] || x > r.box[1]) {
                continue;
            }
            for (int k = 0; k < 4; k++) {
                int y = y0 + 8 * k;
                if (y < r.box[2] || y > r.box[3]) {
                    continue;
                }
                float py = (float)y + 0.5f;
                if (!covers(r, px, py)) {
                    continue;
                }
                float w = depth_at(r, px, py);
                if (nearer(w, r.tri, depth[k], owner[k])) {
                    depth[k] = w;
                    owner[k] = r.tri;
                    colour_at(r, px, py, col[k]);
                }
            }
        }
    }
    for (int k = 0; k < 4; k++) {
        size_t g = (size_t)(y0 + 8 * k) * WIDTH + x;
        depth_out[g] = depth[k];
        owner_out[g] = owner[k];
        for (int ch = 0; ch < 3; ch++) {
            colour[3 * g + ch] = col[k][ch];
        }
    }
}

struct Device {
    Record* records;
    int32_t *counts, *offsets, *cursor, *order, *list, *owner;
    float *colour, *depth;
    size_t list_capacity;
};

struct Times {
    float setup, bin, fill;
};

static Times run(const Mesh& mesh, Device& d, bool grow) {
    cudaEvent_t e[4];
    for (auto& ev : e) {
        CHECK(cudaEventCreate(&ev));
    }
    const int blocks = (mesh.n_tris + 255) / 256;
    CHECK(cudaEventRecord(e[0]));
    CHECK(cudaMemset(d.counts, 0, N_TILES * sizeof(int32_t)));
    setup_kernel<<<blocks, 256>>>(mesh, d.records, d.counts);
    CHECK(cudaEventRecord(e[1]));
    scan_kernel<<<1, N_TILES>>>(d.counts, d.offsets, d.cursor);
    int32_t total = 0;
    CHECK(cudaMemcpy(&total, d.offsets + N_TILES, 4, cudaMemcpyDeviceToHost));
    if ((size_t)total > d.list_capacity) {
        if (!grow) {
            std::fprintf(stderr, "list buffer too small\n");
            std::exit(1);
        }
        CHECK(cudaFree(d.list));
        CHECK(cudaMalloc(&d.list, (size_t)total * 4));
        d.list_capacity = total;
    }
    scatter_kernel<<<blocks, 256>>>(mesh, d.records, d.cursor, d.list);
    order_kernel<<<1, N_TILES>>>(d.counts, d.order);
    CHECK(cudaEventRecord(e[2]));
    dim3 block(TILE, 8);
    fill_kernel<<<N_TILES, block>>>(
        d.records, d.list, d.offsets, d.order, d.colour, d.depth, d.owner
    );
    CHECK(cudaEventRecord(e[3]));
    CHECK(cudaEventSynchronize(e[3]));
    CHECK(cudaGetLastError());
    Times t;
    CHECK(cudaEventElapsedTime(&t.setup, e[0], e[1]));
    CHECK(cudaEventElapsedTime(&t.bin, e[1], e[2]));
    CHECK(cudaEventElapsedTime(&t.fill, e[2], e[3]));
    for (auto& ev : e) {
        CHECK(cudaEventDestroy(ev));
    }
    return t;
}

template <class T>
static T* upload(const Array& a) {
    T* p;
    CHECK(cudaMalloc(&p, a.bytes.size()));
    CHECK(cudaMemcpy(p, a.bytes.data(), a.bytes.size(), cudaMemcpyHostToDevice));
    return p;
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    std::string scene = args.get("scene"), out = args.get("out");
    int reps = args.geti("reps", 3);
    Array pos = load_npy(scene + "/positions.npy", 4);
    Array col = load_npy(scene + "/colors.npy", 4);
    Array idx = load_npy(scene + "/indices.npy", 4);
    Array cam = load_npy(scene + "/camera.npy", 4);
    Mesh mesh;
    mesh.positions = upload<float>(pos);
    mesh.colors = upload<float>(col);
    mesh.indices = upload<int32_t>(idx);
    mesh.n_tris = (int)idx.shape[0];
    std::copy(cam.data<float>(), cam.data<float>() + 16, mesh.camera);

    Device d{};
    CHECK(cudaMalloc(&d.records, (size_t)mesh.n_tris * sizeof(Record)));
    CHECK(cudaMalloc(&d.counts, N_TILES * 4));
    CHECK(cudaMalloc(&d.offsets, (N_TILES + 1) * 4));
    CHECK(cudaMalloc(&d.cursor, N_TILES * 4));
    CHECK(cudaMalloc(&d.order, N_TILES * 4));
    CHECK(cudaMalloc(&d.list, 4));
    CHECK(cudaMalloc(&d.colour, (size_t)WIDTH * HEIGHT * 12));
    CHECK(cudaMalloc(&d.depth, (size_t)WIDTH * HEIGHT * 4));
    CHECK(cudaMalloc(&d.owner, (size_t)WIDTH * HEIGHT * 4));
    d.list_capacity = 1;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    run(mesh, d, true);  // warm-up: sizes the tile lists, loads the code
    RepLog log(reps);
    for (int rep = 0; rep < reps; rep++) {
        Times t = run(mesh, d, false);
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
    std::vector<int32_t> counts(N_TILES), offsets(N_TILES + 1);
    CHECK(cudaMemcpy(counts.data(), d.counts, N_TILES * 4, cudaMemcpyDeviceToHost));
    CHECK(
        cudaMemcpy(offsets.data(), d.offsets, (N_TILES + 1) * 4, cudaMemcpyDeviceToHost)
    );
    std::printf(
        "{\"impl\": \"gpu\", \"device\": \"%s\", \"triangles\": %d, \"reps_n\": %d, "
        "\"loop_ms\": %.4f, \"entries\": %d, \"reps\": [%s]}\n",
        prop.name,
        mesh.n_tris,
        reps,
        loop_ms,
        offsets[N_TILES],
        log.body.c_str()
    );
    if (!out.empty()) {
        std::vector<float> colour((size_t)WIDTH * HEIGHT * 3),
            depth((size_t)WIDTH * HEIGHT);
        std::vector<int32_t> owner((size_t)WIDTH * HEIGHT);
        CHECK(cudaMemcpy(
            colour.data(), d.colour, colour.size() * 4, cudaMemcpyDeviceToHost
        ));
        CHECK(
            cudaMemcpy(depth.data(), d.depth, depth.size() * 4, cudaMemcpyDeviceToHost)
        );
        CHECK(
            cudaMemcpy(owner.data(), d.owner, owner.size() * 4, cudaMemcpyDeviceToHost)
        );
        save_f32(out + "/colour.npy", colour.data(), {HEIGHT, WIDTH, 3});
        save_f32(out + "/depth.npy", depth.data(), {HEIGHT, WIDTH});
        save_i32(out + "/tri.npy", owner.data(), {HEIGHT, WIDTH});
        save_i32(out + "/tile_counts.npy", counts.data(), {TILES_Y, TILES_X});
    }
    return 0;
}
