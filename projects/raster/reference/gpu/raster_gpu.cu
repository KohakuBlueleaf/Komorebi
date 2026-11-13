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

constexpr int TILE_COUNT = TILES_X * TILES_Y;
constexpr int BATCH = 64;

// records: (2T), the first piece of triangle t at t and a second one at T + t.
__global__ void setup_kernel(
    Mesh mesh, Record* records, uint8_t* pieces, int32_t* counts
) {
    int triangle = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle >= mesh.triangle_count) {
        return;
    }
    int bits = setup_triangle(mesh, triangle, records[triangle], records[mesh.triangle_count + triangle]);
    pieces[triangle] = (uint8_t)bits;
    for (int piece = 0; piece < 2; piece++) {
        if (!((bits >> piece) & 1)) {
            continue;
        }
        const Record& record = records[triangle + piece * mesh.triangle_count];
        for (int tile_y = record.box[2] / TILE; tile_y <= record.box[3] / TILE; tile_y++) {
            for (int tile_x = record.box[0] / TILE; tile_x <= record.box[1] / TILE; tile_x++) {
                atomicAdd(&counts[tile_y * TILES_X + tile_x], 1);
            }
        }
    }
}

// Exclusive scan of the 1024 tile counts by one block; cursor starts as the offsets.
__global__ void scan_kernel(const int32_t* counts, int32_t* offsets, int32_t* cursor) {
    __shared__ int32_t sums[TILE_COUNT];
    int i = threadIdx.x;
    sums[i] = counts[i];
    __syncthreads();
    for (int distance = 1; distance < TILE_COUNT; distance *= 2) {
        int32_t add = i >= distance ? sums[i - distance] : 0;
        __syncthreads();
        sums[i] += add;
        __syncthreads();
    }
    int32_t before = sums[i] - counts[i];
    offsets[i] = before;
    cursor[i] = before;
    if (i == TILE_COUNT - 1) {
        offsets[TILE_COUNT] = sums[i];
    }
}

__global__ void scatter_kernel(
    Mesh mesh,
    const Record* records,
    const uint8_t* pieces,
    int32_t* cursor,
    int32_t* list
) {
    int triangle = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle >= mesh.triangle_count) {
        return;
    }
    int bits = pieces[triangle];
    for (int piece = 0; piece < 2; piece++) {
        if (!((bits >> piece) & 1)) {
            continue;
        }
        const int record_index = triangle + piece * mesh.triangle_count;
        const Record& record = records[record_index];
        for (int tile_y = record.box[2] / TILE; tile_y <= record.box[3] / TILE; tile_y++) {
            for (int tile_x = record.box[0] / TILE; tile_x <= record.box[1] / TILE; tile_x++) {
                list[atomicAdd(&cursor[tile_y * TILES_X + tile_x], 1)] = record_index;
            }
        }
    }
}

// Bitonic sort of the tiles by list size, largest first (ties: lower tile first).
__global__ void order_kernel(const int32_t* counts, int32_t* order) {
    __shared__ uint32_t key[TILE_COUNT];
    int i = threadIdx.x;
    key[i] = (uint32_t)counts[i] * 1024u + (uint32_t)(TILE_COUNT - 1 - i);
    __syncthreads();
    for (int size = 2; size <= TILE_COUNT; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
            int partner = i ^ stride;
            if (partner > i) {
                bool descending = (i & size) == 0;
                uint32_t own_key = key[i], partner_key = key[partner];
                if ((own_key < partner_key) == descending) {
                    key[i] = partner_key;
                    key[partner] = own_key;
                }
            }
            __syncthreads();
        }
    }
    order[i] = TILE_COUNT - 1 - (int)(key[i] & 1023u);
}

__global__ void fill_kernel(
    const Record* records,
    const int32_t* list,
    const int32_t* offsets,
    const int32_t* order,
    float* colour_out,
    float* depth_out,
    int32_t* owner_out
) {
    __shared__ Record batch[BATCH];
    const int tile = order[blockIdx.x];
    const int x = (tile % TILES_X) * TILE + threadIdx.x;
    const int top_y = (tile / TILES_X) * TILE + threadIdx.y;
    const int thread_number = threadIdx.y * TILE + threadIdx.x;
    const float px = (float)x + 0.5f;
    float depth[4], colours[4][3];
    int owner[4];
    for (int row = 0; row < 4; row++) {
        depth[row] = 0.0f;
        owner[row] = -1;
        for (int channel = 0; channel < 3; channel++) {
            colours[row][channel] = background(channel);
        }
    }
    const int begin = offsets[tile], end = offsets[tile + 1];
    for (int base = begin; base < end; base += BATCH) {
        int batch_count = min(BATCH, end - base);
        __syncthreads();
        if (thread_number < batch_count) {
            batch[thread_number] = records[list[base + thread_number]];
        }
        __syncthreads();
        for (int entry = 0; entry < batch_count; entry++) {
            const Record& record = batch[entry];
            if (x < record.box[0] || x > record.box[1]) {
                continue;
            }
            for (int row = 0; row < 4; row++) {
                int y = top_y + 8 * row;
                if (y < record.box[2] || y > record.box[3]) {
                    continue;
                }
                float py = (float)y + 0.5f;
                if (!covers(record, px, py)) {
                    continue;
                }
                float inverse_z = depth_at(record, px, py);
                if (nearer(inverse_z, record.triangle, depth[row], owner[row])) {
                    depth[row] = inverse_z;
                    owner[row] = record.triangle;
                    colour_at(record, px, py, colours[row]);
                }
            }
        }
    }
    for (int row = 0; row < 4; row++) {
        size_t pixel = (size_t)(top_y + 8 * row) * WIDTH + x;
        depth_out[pixel] = depth[row];
        owner_out[pixel] = owner[row];
        for (int channel = 0; channel < 3; channel++) {
            colour_out[3 * pixel + channel] = colours[row][channel];
        }
    }
}

struct Device {
    Record* records;
    uint8_t* pieces;
    int32_t *counts, *offsets, *cursor, *order, *list, *owner;
    float *colour, *depth;
    size_t list_capacity;
};

struct Times {
    float setup, bin, fill;
};

static Times run(const Mesh& mesh, Device& device, bool grow) {
    cudaEvent_t events[4];
    for (auto& event : events) {
        CHECK(cudaEventCreate(&event));
    }
    const int blocks = (mesh.triangle_count + 255) / 256;
    CHECK(cudaEventRecord(events[0]));
    CHECK(cudaMemset(device.counts, 0, TILE_COUNT * sizeof(int32_t)));
    setup_kernel<<<blocks, 256>>>(mesh, device.records, device.pieces, device.counts);
    CHECK(cudaEventRecord(events[1]));
    scan_kernel<<<1, TILE_COUNT>>>(device.counts, device.offsets, device.cursor);
    int32_t total = 0;
    CHECK(cudaMemcpy(&total, device.offsets + TILE_COUNT, 4, cudaMemcpyDeviceToHost));
    if ((size_t)total > device.list_capacity) {
        if (!grow) {
            std::fprintf(stderr, "list buffer too small\n");
            std::exit(1);
        }
        CHECK(cudaFree(device.list));
        CHECK(cudaMalloc(&device.list, (size_t)total * 4));
        device.list_capacity = total;
    }
    scatter_kernel<<<blocks, 256>>>(mesh, device.records, device.pieces, device.cursor, device.list);
    order_kernel<<<1, TILE_COUNT>>>(device.counts, device.order);
    CHECK(cudaEventRecord(events[2]));
    dim3 block(TILE, 8);
    fill_kernel<<<TILE_COUNT, block>>>(
        device.records, device.list, device.offsets, device.order, device.colour, device.depth, device.owner
    );
    CHECK(cudaEventRecord(events[3]));
    CHECK(cudaEventSynchronize(events[3]));
    CHECK(cudaGetLastError());
    Times times;
    CHECK(cudaEventElapsedTime(&times.setup, events[0], events[1]));
    CHECK(cudaEventElapsedTime(&times.bin, events[1], events[2]));
    CHECK(cudaEventElapsedTime(&times.fill, events[2], events[3]));
    for (auto& event : events) {
        CHECK(cudaEventDestroy(event));
    }
    return times;
}

template <class T>
static T* upload(const Array& array) {
    T* pointer;
    CHECK(cudaMalloc(&pointer, array.bytes.size()));
    CHECK(cudaMemcpy(pointer, array.bytes.data(), array.bytes.size(), cudaMemcpyHostToDevice));
    return pointer;
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    std::string scene = args.get("scene"), out_folder = args.get("out");
    int repetitions = args.geti("reps", 3);
    Array positions = load_npy(scene + "/positions.npy", 4);
    Array colours = load_npy(scene + "/colors.npy", 4);
    Array indices = load_npy(scene + "/indices.npy", 4);
    Array camera = load_npy(scene + "/camera.npy", 4);
    Mesh mesh;
    mesh.positions = upload<float>(positions);
    mesh.colors = upload<float>(colours);
    mesh.indices = upload<int32_t>(indices);
    mesh.triangle_count = (int)indices.shape[0];
    std::copy(camera.data<float>(), camera.data<float>() + 16, mesh.camera);

    Device device{};
    CHECK(cudaMalloc(&device.records, 2 * (size_t)mesh.triangle_count * sizeof(Record)));
    CHECK(cudaMalloc(&device.pieces, (size_t)mesh.triangle_count));
    CHECK(cudaMalloc(&device.counts, TILE_COUNT * 4));
    CHECK(cudaMalloc(&device.offsets, (TILE_COUNT + 1) * 4));
    CHECK(cudaMalloc(&device.cursor, TILE_COUNT * 4));
    CHECK(cudaMalloc(&device.order, TILE_COUNT * 4));
    CHECK(cudaMalloc(&device.list, 4));
    CHECK(cudaMalloc(&device.colour, (size_t)WIDTH * HEIGHT * 12));
    CHECK(cudaMalloc(&device.depth, (size_t)WIDTH * HEIGHT * 4));
    CHECK(cudaMalloc(&device.owner, (size_t)WIDTH * HEIGHT * 4));
    device.list_capacity = 1;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    run(mesh, device, true);  // warm-up: sizes the tile lists, loads the code
    RepLog log(repetitions);
    for (int repetition = 0; repetition < repetitions; repetition++) {
        Times times = run(mesh, device, false);
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
    std::vector<int32_t> counts(TILE_COUNT), offsets(TILE_COUNT + 1);
    CHECK(cudaMemcpy(counts.data(), device.counts, TILE_COUNT * 4, cudaMemcpyDeviceToHost));
    CHECK(
        cudaMemcpy(offsets.data(), device.offsets, (TILE_COUNT + 1) * 4, cudaMemcpyDeviceToHost)
    );
    std::printf(
        "{\"impl\": \"gpu\", \"device\": \"%s\", \"triangles\": %d, \"reps_n\": %d, "
        "\"loop_ms\": %.4f, \"entries\": %d, \"reps\": [%s]}\n",
        prop.name,
        mesh.triangle_count,
        repetitions,
        loop_ms,
        offsets[TILE_COUNT],
        log.body.c_str()
    );
    if (!out_folder.empty()) {
        std::vector<float> colour((size_t)WIDTH * HEIGHT * 3),
            depth((size_t)WIDTH * HEIGHT);
        std::vector<int32_t> owner((size_t)WIDTH * HEIGHT);
        CHECK(cudaMemcpy(
            colour.data(), device.colour, colour.size() * 4, cudaMemcpyDeviceToHost
        ));
        CHECK(
            cudaMemcpy(depth.data(), device.depth, depth.size() * 4, cudaMemcpyDeviceToHost)
        );
        CHECK(
            cudaMemcpy(owner.data(), device.owner, owner.size() * 4, cudaMemcpyDeviceToHost)
        );
        save_f32(out_folder + "/colour.npy", colour.data(), {HEIGHT, WIDTH, 3});
        save_f32(out_folder + "/depth.npy", depth.data(), {HEIGHT, WIDTH});
        save_i32(out_folder + "/tri.npy", owner.data(), {HEIGHT, WIDTH});
        save_i32(out_folder + "/tile_counts.npy", counts.data(), {TILES_Y, TILES_X});
    }
    return 0;
}
