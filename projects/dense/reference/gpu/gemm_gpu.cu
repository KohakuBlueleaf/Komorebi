// Matrix product Y[m, n] = A[m, k] W[k, n] (row major) with cuBLAS on one GPU.
// Usage: gemm_gpu --m M --k K --n N --type fp32|fp16 [--seconds S]
// fp32: float in and out, float accumulation. fp16: half in and out, float accumulation
// (tensor cores). Prints one JSON line; LOOP_START and LOOP_END on stderr.
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(call)                                                                 \
    do {                                                                            \
        cudaError_t error_ = (call);                                                  \
        if (error_ != cudaSuccess) {                                                  \
            std::fprintf(                                                           \
                stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_) \
            );                                                                      \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

#define CHECK_BLAS(call)                                                          \
    do {                                                                          \
        cublasStatus_t status_ = (call);                                              \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                       \
            std::fprintf(stderr, "%s:%d: cuBLAS %d\n", __FILE__, __LINE__, status_); \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

namespace {

    // Uniform in [-1, 1) from a 64-bit linear congruential generator.
    std::vector<float> uniform(size_t count, uint64_t seed) {
        std::vector<float> values(count);
        for (auto& value : values) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            value = static_cast<float>((seed >> 40) * (2.0 / 16777216.0) - 1.0);
        }
        return values;
    }

    struct Gemm {
        cublasHandle_t handle;
        int m_rows, k_depth, n_columns;
        bool half;
        void *a_device, *w_device, *y_device;

        void call() const {
            const float one = 1.0f, zero = 0.0f;
            const cudaDataType data_type = half ? CUDA_R_16F : CUDA_R_32F;
            // Row-major Y = A W is column-major Y^T = W^T A^T.
            CHECK_BLAS(cublasGemmEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, n_columns, m_rows, k_depth, &one, w_device, data_type, n_columns, a_device, data_type, k_depth, &zero, y_device, data_type,
                n_columns, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
            ));
        }
    };

    double median(std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

}  // namespace

int main(int argc, char** argv) {
    int m_rows = 0, k_depth = 0, n_columns = 0;
    double seconds = 4.0;
    std::string type = "fp32";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        if (key == "--m") {
            m_rows = std::atoi(argv[i + 1]);
        } else if (key == "--k") {
            k_depth = std::atoi(argv[i + 1]);
        } else if (key == "--n") {
            n_columns = std::atoi(argv[i + 1]);
        } else if (key == "--type") {
            type = argv[i + 1];
        } else if (key == "--seconds") {
            seconds = std::atof(argv[i + 1]);
        }
    }
    if (m_rows <= 0 || k_depth <= 0 || n_columns <= 0 || (type != "fp32" && type != "fp16")) {
        std::fprintf(stderr, "usage: gemm_gpu --m M --k K --n N --type fp32|fp16\n");
        return 2;
    }
    const bool half = type == "fp16";
    const size_t element_bytes = half ? 2 : 4;
    std::vector<float> a_values = uniform(size_t(m_rows) * k_depth, 1), w_values = uniform(size_t(k_depth) * n_columns, 2);
    std::vector<__half> a_half, w_half;
    if (half) {
        // The values the GPU sees, so the check below uses them too.
        for (auto* values : {&a_values, &w_values}) {
            auto& half_values = values == &a_values ? a_half : w_half;
            half_values.resize(values->size());
            for (size_t i = 0; i < values->size(); ++i) {
                half_values[i] = __float2half((*values)[i]);
                (*values)[i] = __half2float(half_values[i]);
            }
        }
    }
    Gemm gemm{nullptr, m_rows, k_depth, n_columns, half, nullptr, nullptr, nullptr};
    CHECK(cudaMalloc(&gemm.a_device, a_values.size() * element_bytes));
    CHECK(cudaMalloc(&gemm.w_device, w_values.size() * element_bytes));
    CHECK(cudaMalloc(&gemm.y_device, size_t(m_rows) * n_columns * element_bytes));
    CHECK(cudaMemcpy(gemm.a_device, half ? (void*)a_half.data() : a_values.data(), a_values.size() * element_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gemm.w_device, half ? (void*)w_half.data() : w_values.data(), w_values.size() * element_bytes, cudaMemcpyHostToDevice));
    CHECK_BLAS(cublasCreate(&gemm.handle));
    cudaEvent_t start_event, stop_event;
    CHECK(cudaEventCreate(&start_event));
    CHECK(cudaEventCreate(&stop_event));
    float milliseconds = 0;
    CHECK(cudaEventRecord(start_event));
    gemm.call();
    CHECK(cudaEventRecord(stop_event));
    CHECK(cudaEventSynchronize(stop_event));
    CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    const double first_ms = milliseconds;
    // Check 256 entries against a double-precision dot product of the same values.
    std::vector<uint8_t> y_bytes(size_t(m_rows) * n_columns * element_bytes);
    CHECK(cudaMemcpy(y_bytes.data(), gemm.y_device, y_bytes.size(), cudaMemcpyDeviceToHost));
    double max_error = 0, scale = 0;
    for (int sample = 0; sample < 256; ++sample) {
        const size_t row = (size_t(sample) * 2654435761ULL) % m_rows, column = (size_t(sample) * 40503ULL) % n_columns;
        double want = 0;
        for (int inner = 0; inner < k_depth; ++inner) {
            want += double(a_values[row * k_depth + inner]) * w_values[size_t(inner) * n_columns + column];
        }
        const size_t position = row * n_columns + column;
        const double got = half ? double(__half2float(reinterpret_cast<__half*>(y_bytes.data())[position]))
                                : double(reinterpret_cast<float*>(y_bytes.data())[position]);
        max_error = std::max(max_error, std::fabs(got - want));
        scale = std::max(scale, std::fabs(want));
    }
    // Synced calls, then a back-to-back loop of about `seconds`.
    std::vector<double> synced;
    for (int repetition = 0; repetition < 20; ++repetition) {
        CHECK(cudaEventRecord(start_event));
        gemm.call();
        CHECK(cudaEventRecord(stop_event));
        CHECK(cudaEventSynchronize(stop_event));
        CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        synced.push_back(milliseconds);
    }
    const double one_ms = median(synced);
    const int reps = std::max(10, int(seconds * 1e3 / std::max(one_ms, 1e-3)));
    std::fprintf(stderr, "LOOP_START\n");
    std::fflush(stderr);
    CHECK(cudaEventRecord(start_event));
    for (int repetition = 0; repetition < reps; ++repetition) {
        gemm.call();
    }
    CHECK(cudaEventRecord(stop_event));
    CHECK(cudaEventSynchronize(stop_event));
    std::fprintf(stderr, "LOOP_END\n");
    std::fflush(stderr);
    CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    cudaDeviceProp properties;
    CHECK(cudaGetDeviceProperties(&properties, 0));
    std::printf(
        "{\"device\": \"%s\", \"type\": \"%s\", \"m\": %d, \"k\": %d, \"n\": %d, "
        "\"first_ms\": %.4f, \"synced_ms_median\": %.5f, \"synced_ms_min\": %.5f, "
        "\"reps_n\": %d, \"loop_ms\": %.4f, \"microseconds_per_call\": %.4f, \"tflops\": %.3f, "
        "\"sample_max_error_over_max\": %.3g}\n",
        properties.name, type.c_str(), m_rows, k_depth, n_columns, first_ms, one_ms,
        *std::min_element(synced.begin(), synced.end()), reps, milliseconds, milliseconds * 1e3 / reps,
        2.0 * m_rows * k_depth * n_columns / (milliseconds * 1e-3 / reps) / 1e12, max_error / scale
    );
    cublasDestroy(gemm.handle);
    return 0;
}
