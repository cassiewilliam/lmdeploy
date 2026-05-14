// Copyright (c) OpenMMLab. All rights reserved.

#include "flashinfer_gemm_wrapper.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace {

void check_cuda(cudaError_t err, const char* expr)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", expr, cudaGetErrorString(err));
        std::exit(1);
    }
}

float a_value(int row, int col)
{
    static constexpr float values[] = {0.5f, 1.f, 2.f, 4.f};
    return values[(row * 17 + col * 13) & 3];
}

float b_value(int row, int col)
{
    static constexpr float values[] = {0.25f, 0.5f, 1.f, 2.f};
    return values[(row * 11 + col * 7 + 1) & 3];
}

float a_scale_value(int group, int row)
{
    return 0.03125f * static_cast<float>(group + 1) + 0.00390625f * static_cast<float>((row % 7) + 1);
}

float b_scale_value(int group, int block)
{
    return 0.0625f * static_cast<float>(group + 1) + 0.015625f * static_cast<float>(block + 1);
}

float reference_value(int row, int col, int m, int n, int k)
{
    const int n_blocks = n / 128;
    const int col_blk  = col / 128;
    float     acc      = 0.f;
    for (int kk = 0; kk < k; ++kk) {
        const int group = kk / 128;
        acc += a_value(row, kk) * a_scale_value(group, row) * b_value(col, kk)
               * b_scale_value(group, col_blk);
    }
    (void)m;
    (void)n_blocks;
    return acc;
}

int run_case(int m, int n, int k, int mma_sm)
{
    const int k_blocks = k / 128;
    const int n_blocks = n / 128;

    std::vector<turbomind::fp8_e4m3_t> h_a(m * k);
    std::vector<turbomind::fp8_e4m3_t> h_b(n * k);
    std::vector<float>                 h_a_scale(k_blocks * m);
    std::vector<float>                 h_b_scale(k_blocks * n_blocks);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            h_a[row * k + col] = turbomind::fp8_e4m3_t(a_value(row, col));
        }
    }
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < k; ++col) {
            h_b[row * k + col] = turbomind::fp8_e4m3_t(b_value(row, col));
        }
    }
    for (int group = 0; group < k_blocks; ++group) {
        for (int row = 0; row < m; ++row) {
            h_a_scale[group * m + row] = a_scale_value(group, row);
        }
        for (int block = 0; block < n_blocks; ++block) {
            h_b_scale[group * n_blocks + block] = b_scale_value(group, block);
        }
    }

    turbomind::fp8_e4m3_t* a{};
    float*                 a_scale{};
    turbomind::fp8_e4m3_t* b{};
    float*                 b_scale{};
    __nv_bfloat16*         out{};
    void*                  workspace{};
    constexpr size_t       workspace_bytes = 32u << 20;

    check_cuda(cudaMalloc(&a, h_a.size() * sizeof(*a)), "cudaMalloc(a)");
    check_cuda(cudaMalloc(&a_scale, h_a_scale.size() * sizeof(*a_scale)), "cudaMalloc(a_scale)");
    check_cuda(cudaMalloc(&b, h_b.size() * sizeof(*b)), "cudaMalloc(b)");
    check_cuda(cudaMalloc(&b_scale, h_b_scale.size() * sizeof(*b_scale)), "cudaMalloc(b_scale)");
    check_cuda(cudaMalloc(&out, static_cast<size_t>(m) * n * sizeof(*out)), "cudaMalloc(out)");
    check_cuda(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc(workspace)");
    check_cuda(cudaMemcpy(a, h_a.data(), h_a.size() * sizeof(*a), cudaMemcpyHostToDevice), "cudaMemcpy(a)");
    check_cuda(cudaMemcpy(a_scale, h_a_scale.data(), h_a_scale.size() * sizeof(*a_scale), cudaMemcpyHostToDevice),
               "cudaMemcpy(a_scale)");
    check_cuda(cudaMemcpy(b, h_b.data(), h_b.size() * sizeof(*b), cudaMemcpyHostToDevice), "cudaMemcpy(b)");
    check_cuda(cudaMemcpy(b_scale, h_b_scale.data(), h_b_scale.size() * sizeof(*b_scale), cudaMemcpyHostToDevice),
               "cudaMemcpy(b_scale)");
    check_cuda(cudaMemset(out, 0, static_cast<size_t>(m) * n * sizeof(*out)), "cudaMemset(out)");

    turbomind::flashinfer_gemm::Fp8GroupwiseGemmParams params{};
    params.input           = a;
    params.input_scale     = a_scale;
    params.weight          = b;
    params.weight_scale    = b_scale;
    params.output          = out;
    params.workspace       = workspace;
    params.workspace_bytes = workspace_bytes;
    params.m               = m;
    params.n               = n;
    params.k               = k;
    params.output_dtype    = turbomind::flashinfer_gemm::DType::kBF16;
    params.mma_sm          = mma_sm;
    params.stream          = nullptr;

    if (!turbomind::flashinfer_gemm::dispatch_fp8_groupwise(params)) {
        std::fprintf(stderr, "dispatch_fp8_groupwise failed\n");
        return 1;
    }
    turbomind::sync_check_cuda_error();

    std::vector<__nv_bfloat16> h_out(static_cast<size_t>(m) * n);
    check_cuda(cudaMemcpy(h_out.data(), out, h_out.size() * sizeof(*out), cudaMemcpyDeviceToHost),
               "cudaMemcpy(out)");

    float max_abs = 0.f;
    float max_rel = 0.f;
    int   bad_row = -1;
    int   bad_col = -1;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            const float got = __bfloat162float(h_out[row * n + col]);
            const float ref = reference_value(row, col, m, n, k);
            const float abs = std::fabs(got - ref);
            const float rel = abs / std::max(std::fabs(ref), 1.0e-6f);
            if (abs > max_abs) {
                max_abs = abs;
                max_rel = rel;
                bad_row = row;
                bad_col = col;
            }
        }
    }

    cudaFree(a);
    cudaFree(a_scale);
    cudaFree(b);
    cudaFree(b_scale);
    cudaFree(out);
    cudaFree(workspace);

    if (max_abs > 0.25f && max_rel > 0.02f) {
        std::fprintf(stderr,
                     "unexpected output: m=%d n=%d k=%d mma_sm=%d row=%d col=%d max_abs=%f max_rel=%f\n",
                     m,
                     n,
                     k,
                     mma_sm,
                     bad_row,
                     bad_col,
                     max_abs,
                     max_rel);
        return 2;
    }
    std::printf("flashinfer_gemm_smoke case ok: m=%d n=%d k=%d mma_sm=%d max_abs=%.6f max_rel=%.6f\n",
                m,
                n,
                k,
                mma_sm,
                max_abs,
                max_rel);
    return 0;
}

int round_up(int x, int y)
{
    return ((x + y - 1) / y) * y;
}

int64_t fp4_sf_128x4_offset(int row, int col, int num_cols)
{
    const int inner_k     = col & 3;
    const int inner_m     = (row & 127) >> 5;
    const int outer_m     = row & 31;
    const int k_tile      = col >> 2;
    const int m_tile      = row >> 7;
    const int num_k_tiles = (num_cols + 3) >> 2;
    return (static_cast<int64_t>(m_tile) * num_k_tiles + k_tile) * 512 + outer_m * 16 + inner_m * 4 + inner_k;
}

uint8_t fp4_code(const std::vector<uint8_t>& data, int64_t idx)
{
    const uint8_t byte = data[idx >> 1];
    return (idx & 1) ? (byte >> 4) : (byte & 0x0f);
}

float fp4_value(uint8_t code)
{
    static constexpr float lut[] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    return lut[code & 0x0f];
}

float fp8_e4m3_value(uint8_t code)
{
    __nv_fp8_e4m3 value;
    value.__x = code;
    return static_cast<float>(value);
}

float nvfp4_a_value(int row, int col)
{
    static constexpr float values[] = {-1.5f, -0.75f, -0.25f, 0.25f, 0.75f, 1.5f};
    return values[(row * 19 + col * 7) % 6];
}

float nvfp4_b_value(int row, int col)
{
    static constexpr float values[] = {-1.0f, -0.5f, -0.125f, 0.125f, 0.5f, 1.0f};
    return values[(row * 13 + col * 11 + 3) % 6];
}

int run_nvfp4_case(int m, int n, int k)
{
    const int gemm_m      = round_up(m, 128);
    const int scale_k     = round_up(k / 16, 4);
    const int weight_rows = round_up(n, 128);

    std::vector<__nv_bfloat16> h_a(static_cast<size_t>(m) * k);
    std::vector<__nv_bfloat16> h_b(static_cast<size_t>(n) * k);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            h_a[static_cast<size_t>(row) * k + col] = __float2bfloat16(nvfp4_a_value(row, col));
        }
    }
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < k; ++col) {
            h_b[static_cast<size_t>(row) * k + col] = __float2bfloat16(nvfp4_b_value(row, col));
        }
    }

    turbomind::Tensor A_src{{m, k}, turbomind::kBfloat16, turbomind::kDEVICE};
    turbomind::Tensor B_src{{n, k}, turbomind::kBfloat16, turbomind::kDEVICE};
    turbomind::Tensor input_scale{{1, 1}, turbomind::kFloat, turbomind::kDEVICE};
    turbomind::Tensor weight_scale{{1, 1}, turbomind::kFloat, turbomind::kDEVICE};
    turbomind::Tensor alpha{{1}, turbomind::kFloat, turbomind::kDEVICE};
    const float       h_input_scale  = 0.01f;
    const float       h_weight_scale = 0.02f;
    const float       h_alpha        = h_input_scale * h_weight_scale;

    check_cuda(cudaMemcpy(A_src.raw_data(), h_a.data(), h_a.size() * sizeof(h_a[0]), cudaMemcpyHostToDevice),
               "cudaMemcpy(A_src)");
    check_cuda(cudaMemcpy(B_src.raw_data(), h_b.data(), h_b.size() * sizeof(h_b[0]), cudaMemcpyHostToDevice),
               "cudaMemcpy(B_src)");
    check_cuda(cudaMemcpy(input_scale.raw_data(), &h_input_scale, sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy(input_scale)");
    check_cuda(cudaMemcpy(weight_scale.raw_data(), &h_weight_scale, sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy(weight_scale)");
    check_cuda(cudaMemcpy(alpha.raw_data(), &h_alpha, sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(alpha)");

    turbomind::Tensor A;
    turbomind::Tensor A_scale;
    turbomind::Tensor B;
    turbomind::Tensor B_scale;
    turbomind::flashinfer_gemm::QuantizeNvfp4(A, A_scale, A_src, input_scale, nullptr, gemm_m);
    turbomind::flashinfer_gemm::QuantizeNvfp4(B, B_scale, B_src, weight_scale, nullptr, n);
    turbomind::sync_check_cuda_error();

    turbomind::Tensor D{{gemm_m, n}, turbomind::kBfloat16, turbomind::kDEVICE};
    void*             workspace{};
    constexpr size_t  workspace_bytes = 32u << 20;
    check_cuda(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc(workspace)");

    turbomind::flashinfer_gemm::Nvfp4GemmParams params{};
    params.input           = A.raw_data();
    params.input_scale     = A_scale.raw_data();
    params.weight          = B.raw_data();
    params.weight_scale    = B_scale.raw_data();
    params.alpha           = alpha.raw_data();
    params.output          = D.raw_data();
    params.workspace       = workspace;
    params.workspace_bytes = workspace_bytes;
    params.m               = gemm_m;
    params.n               = n;
    params.k               = k;
    params.output_dtype    = turbomind::flashinfer_gemm::DType::kBF16;
    params.stream          = nullptr;
    if (!turbomind::flashinfer_gemm::dispatch_nvfp4(params)) {
        std::fprintf(stderr, "dispatch_nvfp4 failed\n");
        return 1;
    }
    turbomind::sync_check_cuda_error();

    std::vector<uint8_t>       h_a_fp4(static_cast<size_t>(gemm_m) * k / 2);
    std::vector<uint8_t>       h_b_fp4(static_cast<size_t>(n) * k / 2);
    std::vector<uint8_t>       h_a_scale(static_cast<size_t>(gemm_m) * scale_k);
    std::vector<uint8_t>       h_b_scale(static_cast<size_t>(weight_rows) * scale_k);
    std::vector<__nv_bfloat16> h_out(static_cast<size_t>(gemm_m) * n);
    check_cuda(cudaMemcpy(h_a_fp4.data(), A.raw_data(), h_a_fp4.size(), cudaMemcpyDeviceToHost),
               "cudaMemcpy(A)");
    check_cuda(cudaMemcpy(h_b_fp4.data(), B.raw_data(), h_b_fp4.size(), cudaMemcpyDeviceToHost),
               "cudaMemcpy(B)");
    check_cuda(cudaMemcpy(h_a_scale.data(), A_scale.raw_data(), h_a_scale.size(), cudaMemcpyDeviceToHost),
               "cudaMemcpy(A_scale)");
    check_cuda(cudaMemcpy(h_b_scale.data(), B_scale.raw_data(), h_b_scale.size(), cudaMemcpyDeviceToHost),
               "cudaMemcpy(B_scale)");
    check_cuda(cudaMemcpy(h_out.data(), D.raw_data(), h_out.size() * sizeof(h_out[0]), cudaMemcpyDeviceToHost),
               "cudaMemcpy(D)");
    cudaFree(workspace);

    float max_abs = 0.f;
    float max_rel = 0.f;
    int   bad_row = -1;
    int   bad_col = -1;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float ref = 0.f;
            for (int kk = 0; kk < k; ++kk) {
                const int block = kk / 16;
                const float av = fp4_value(fp4_code(h_a_fp4, static_cast<int64_t>(row) * k + kk));
                const float bv = fp4_value(fp4_code(h_b_fp4, static_cast<int64_t>(col) * k + kk));
                const float as =
                    fp8_e4m3_value(h_a_scale[fp4_sf_128x4_offset(row, block, scale_k)]);
                const float bs =
                    fp8_e4m3_value(h_b_scale[fp4_sf_128x4_offset(col, block, scale_k)]);
                ref += av * bv * as * bs * h_alpha;
            }
            const float got = __bfloat162float(h_out[static_cast<size_t>(row) * n + col]);
            const float abs = std::fabs(got - ref);
            const float rel = abs / std::max(std::fabs(ref), 1.0e-6f);
            if (abs > max_abs) {
                max_abs = abs;
                max_rel = rel;
                bad_row = row;
                bad_col = col;
            }
        }
    }

    if (max_abs > 0.25f && max_rel > 0.05f) {
        std::fprintf(stderr,
                     "unexpected NVFP4 output: m=%d n=%d k=%d row=%d col=%d max_abs=%f max_rel=%f\n",
                     m,
                     n,
                     k,
                     bad_row,
                     bad_col,
                     max_abs,
                     max_rel);
        return 2;
    }
    std::printf("flashinfer_gemm_smoke NVFP4 case ok: m=%d n=%d k=%d max_abs=%.6f max_rel=%.6f\n",
                m,
                n,
                k,
                max_abs,
                max_rel);
    return 0;
}

}  // namespace

int main()
{
    auto stream = turbomind::core::Stream::create();
    turbomind::core::ContextGuard ctx{
        stream, turbomind::core::Allocator{turbomind::kCPU}, turbomind::core::Allocator{stream, false}};

    if (int rc = run_case(17, 256, 384, 1)) {
        return rc;
    }
    if (int rc = run_case(128, 256, 384, 1)) {
        return rc;
    }
    if (int rc = run_case(256, 256, 384, 2)) {
        return rc;
    }
    if (int rc = run_nvfp4_case(17, 256, 128)) {
        return rc;
    }
    if (int rc = run_nvfp4_case(128, 256, 128)) {
        return rc;
    }
    return 0;
}
