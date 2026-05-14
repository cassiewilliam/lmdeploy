// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"

namespace turbomind::flashinfer_gemm {

enum class DType {
    kFP16,
    kBF16,
    kFP8E4M3,
};

struct Fp8GroupwiseGemmParams {
    const void* input{};         // [m, k] fp8_e4m3, row-major
    const void* input_scale{};   // [k / 128, m] fp32, row-major (scale_major_mode = MN)
    const void* weight{};        // logical [n, k] fp8_e4m3, column-major
    const void* weight_scale{};  // [k / 128, n / 128] fp32, row-major (scale_major_mode = MN)
    void* output{};        // [m, n] fp16/bf16, row-major
    void* workspace{};     // scratch buffer

    int64_t workspace_bytes{};
    int     m{};
    int     n{};
    int     k{};
    DType   output_dtype{DType::kBF16};
    int     mma_sm{1};

    cudaStream_t stream{};
};

struct Nvfp4GemmParams {
    const void* input{};         // [m, k / 2] packed e2m1, row-major
    const void* input_scale{};   // [ceil(m / 128) * 128, align4(k / 16)] fp8_e4m3, 128x4 swizzled
    const void* weight{};        // [n, k / 2] packed e2m1, row-major
    const void* weight_scale{};  // [ceil(n / 128) * 128, align4(k / 16)] fp8_e4m3, 128x4 swizzled
    const void* alpha{};         // [1] fp32, input_global_scale * weight_global_scale
    void*       output{};        // [m, n] fp16/bf16, row-major
    void*       workspace{};     // scratch buffer

    int64_t workspace_bytes{};
    int     m{};
    int     n{};
    int     k{};
    DType   output_dtype{DType::kBF16};
    // -1 means "autotune/cache in the wrapper".  Non-negative values are
    // forwarded directly to FlashInfer for debugging or reproducing a tactic.
    int64_t tactic{-1};

    cudaStream_t stream{};
};

bool is_available();

bool is_nvfp4_available();

// Enabled by TurboMind warmup so FlashInfer tactic probing happens during
// startup, not on the first real user request.  The service parameter
// `disable_flashinfer_tuning` decides whether warmup enables this.
void SetTuningActive(bool active);

bool dispatch_fp8_groupwise(const Fp8GroupwiseGemmParams& p);

bool dispatch_nvfp4(const Nvfp4GemmParams& p);

// FlashInfer CUTLASS FP8 groupwise GEMM uses scale_major_mode="MN".
// Activation is therefore quantized to:
//   out         : [m, k] fp8_e4m3, row-major
//   scale       : [k / 128, m] fp32, row-major
void QuantizeFp8Groupwise(Tensor& out, Tensor& scale, const Tensor& src, cudaStream_t stream, int padded_m = 0);

// FlashInfer CUTLASS FP4 GEMM consumes the same NVFP4 layout as TRT-LLM:
//   out         : [m, k / 2] packed e2m1, row-major
//   scale       : [ceil(m / 128) * 128, align4(k / 16)] fp8_e4m3, 128x4 swizzled
// `input_scale` is the ModelOpt/TensorRT-LLM global activation scale; the
// quantizer internally uses 1 / input_scale as the FP4 SF scale.
void QuantizeNvfp4(Tensor&      out,
                   Tensor&      scale,
                   const Tensor& src,
                   const Tensor& input_scale,
                   cudaStream_t  stream,
                   int           padded_m = 0);

// Convert ModelOpt/TurboMind checkpoint layout:
//   weight: [k, n] packed e2m1, scales: [k / 16, n] fp8_e4m3
// into FlashInfer/TRT-LLM GEMM layout:
//   weight: [n, k] packed e2m1, scales: [ceil(n / 128) * 128, align4(k / 16)] swizzled.
void PrepareNvfp4WeightForGemm(Tensor& weight, Tensor& scales, int input_dim, int output_dim, cudaStream_t stream);

}  // namespace turbomind::flashinfer_gemm
