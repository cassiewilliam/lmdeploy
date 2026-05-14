// Copyright (c) OpenMMLab. All rights reserved.

#include "flashinfer_gemm_wrapper.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <type_traits>

#include <cub/block/block_reduce.cuh>

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::flashinfer_gemm {

namespace {

constexpr int kBlockSize = 128;
constexpr int kNvfp4GroupSize = 16;

template<class T>
__device__ __forceinline__ float as_float(T x)
{
    return static_cast<float>(x);
}

template<class T>
__global__ void quantize_fp8_groupwise_kernel(fp8_e4m3_t* out,
                                              int          out_ld,
                                              float*       scale,
                                              const T*     src,
                                              int          src_ld,
                                              int          src_m,
                                              int          out_m,
                                              int          k)
{
    const int row   = blockIdx.x;
    const int group = blockIdx.y;
    const int col   = group * kBlockSize + threadIdx.x;

    float x = 0.f;
    if (row < src_m && col < k) {
        x = as_float(src[static_cast<int64_t>(row) * src_ld + col]);
    }

    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float                             block_scale;

    float amax = fabsf(x);
    amax       = BlockReduce{temp_storage}.Reduce(amax, [](float a, float b) { return fmaxf(a, b); });

    if (threadIdx.x == 0) {
        block_scale              = fmaxf(amax, 1.0e-8f) / 448.f;
        scale[group * out_m + row] = block_scale;
    }
    __syncthreads();

    if (row < out_m && col < k) {
        out[static_cast<int64_t>(row) * out_ld + col] = fp8_e4m3_t(x / block_scale);
    }
}

__device__ __forceinline__ int64_t fp4_sf_128x4_offset(int row, int col, int num_cols)
{
    const int inner_k = col & 3;
    const int inner_m = (row & 127) >> 5;
    const int outer_m = row & 31;
    const int k_tile  = col >> 2;
    const int m_tile  = row >> 7;
    const int num_k_tiles = (num_cols + 3) >> 2;
    return (static_cast<int64_t>(m_tile) * num_k_tiles + k_tile) * 512
           + outer_m * 16 + inner_m * 4 + inner_k;
}

__device__ __forceinline__ uint8_t get_nibble(const uint8_t* data, int64_t idx)
{
    const uint8_t byte = data[idx >> 1];
    return (idx & 1) ? (byte >> 4) : (byte & 0x0f);
}

__device__ __forceinline__ void set_nibble(uint8_t* data, int64_t idx, uint8_t val)
{
    uint8_t* ptr = data + (idx >> 1);
    const uint8_t old = *ptr;
    if (idx & 1) {
        *ptr = (old & 0x0f) | (val << 4);
    }
    else {
        *ptr = (old & 0xf0) | (val & 0x0f);
    }
}

__global__ void prepare_nvfp4_weight_kernel(uint8_t* dst, const uint8_t* src, int k, int n)
{
    const int64_t total = static_cast<int64_t>(k) * n;
    int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    for (; idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int kk = idx / n;
        const int nn = idx - static_cast<int64_t>(kk) * n;
        const auto val = get_nibble(src, static_cast<int64_t>(kk) * n + nn);
        set_nibble(dst, static_cast<int64_t>(nn) * k + kk, val);
    }
}

__global__ void prepare_nvfp4_scale_kernel(uint8_t* dst, const uint8_t* src, int k_groups, int n, int padded_cols)
{
    const int64_t total = static_cast<int64_t>(k_groups) * n;
    int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    for (; idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int kg = idx / n;
        const int nn = idx - static_cast<int64_t>(kg) * n;
        dst[fp4_sf_128x4_offset(nn, kg, padded_cols)] = src[static_cast<int64_t>(kg) * n + nn];
    }
}

__device__ __forceinline__ float reciprocal_approximate_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ __nv_bfloat162 cuda_abs(__nv_bfloat162 a)
{
    __nv_bfloat162 result;
    result.x = __float2bfloat16(fabsf(__bfloat162float(a.x)));
    result.y = __float2bfloat16(fabsf(__bfloat162float(a.y)));
    return result;
}

__device__ __forceinline__ half2 cuda_abs(half2 a)
{
    return __habs2(a);
}

__device__ __forceinline__ __nv_bfloat162 cuda_max(__nv_bfloat162 a, __nv_bfloat162 b)
{
    __nv_bfloat162 result;
    result.x = __bfloat162float(a.x) > __bfloat162float(b.x) ? a.x : b.x;
    result.y = __bfloat162float(a.y) > __bfloat162float(b.y) ? a.y : b.y;
    return result;
}

__device__ __forceinline__ half2 cuda_max(half2 a, half2 b)
{
    return __hmax2(a, b);
}

inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t val;
    asm volatile("{\n"
                 ".reg .b8 byte0;\n"
                 ".reg .b8 byte1;\n"
                 ".reg .b8 byte2;\n"
                 ".reg .b8 byte3;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
                 "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
                 "}"
                 : "=r"(val)
                 : "f"(array[0].x),
                   "f"(array[0].y),
                   "f"(array[1].x),
                   "f"(array[1].y),
                   "f"(array[2].x),
                   "f"(array[2].y),
                   "f"(array[3].x),
                   "f"(array[3].y));
    return val;
#else
    __trap();
    return 0;
#endif
}

template<typename T>
struct fp16_traits;

template<>
struct fp16_traits<__nv_bfloat16> {
    using packed_type = __nv_bfloat162;
    static __device__ __forceinline__ __nv_bfloat16 zero()
    {
        return __float2bfloat16(0.0f);
    }
    static __device__ __forceinline__ __nv_bfloat162 zero2()
    {
        return __float2bfloat162_rn(0.0f);
    }
};

template<>
struct fp16_traits<half> {
    using packed_type = half2;
    static __device__ __forceinline__ half zero()
    {
        return __float2half(0.0f);
    }
    static __device__ __forceinline__ half2 zero2()
    {
        return __float2half2_rn(0.0f);
    }
};

template<typename InType>
__device__ uint32_t
quantize_fp16_to_e2m1_with_scaling(typename fp16_traits<InType>::packed_type (&vec)[4],
                                   float global_scale,
                                   uint8_t* block_scale_out)
{
    using PackedType = typename fp16_traits<InType>::packed_type;

    constexpr int kElementsPerThread = 8;

    PackedType local_max = cuda_abs(vec[0]);
#pragma unroll
    for (int i = 1; i < kElementsPerThread / 2; ++i) {
        local_max = cuda_max(local_max, cuda_abs(vec[i]));
    }

    local_max = cuda_max(__shfl_xor_sync(uint32_t(-1), local_max, 1), local_max);

    float vec_max;
    if constexpr (std::is_same_v<InType, __nv_bfloat16>) {
        auto max_single = __bfloat162float(local_max.x) > __bfloat162float(local_max.y) ? local_max.x : local_max.y;
        vec_max         = __bfloat162float(max_single);
    }
    else {
        vec_max = fmaxf(__half2float(local_max.x), __half2float(local_max.y));
    }

    auto sf_value = reciprocal_approximate_ftz(global_scale) * (vec_max * reciprocal_approximate_ftz(6.0f));

    __nv_fp8_e4m3 fp8_scale = __nv_fp8_e4m3(sf_value);
    if (block_scale_out) {
        *block_scale_out = fp8_scale.__x;
    }
    sf_value = static_cast<float>(fp8_scale);

    const float output_scale = vec_max != 0.f ? reciprocal_approximate_ftz(sf_value * global_scale) : 0.f;

    float2 fp2_vals[kElementsPerThread / 2];
#pragma unroll
    for (int i = 0; i < kElementsPerThread / 2; ++i) {
        if constexpr (std::is_same_v<InType, __nv_bfloat16>) {
            fp2_vals[i] = __bfloat1622float2(vec[i]);
        }
        else {
            fp2_vals[i] = __half22float2(vec[i]);
        }
        fp2_vals[i].x *= output_scale;
        fp2_vals[i].y *= output_scale;
    }

    return fp32_vec_to_e2m1(fp2_vals);
}

template<typename InType, int kThreads = 128>
__global__ void quantize_nvfp4_kernel(const InType* __restrict__ input,
                                      int src_ld,
                                      int src_m,
                                      const float* __restrict__ input_global_scale,
                                      uint8_t* __restrict__ fp4_output,
                                      uint8_t* __restrict__ block_scales,
                                      int out_m,
                                      int k,
                                      int scale_cols)
{
    using traits     = fp16_traits<InType>;
    using PackedType = typename traits::packed_type;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float global_scale;
    if (tid == 0) {
        global_scale = fmaxf(input_global_scale[0], 1.0e-20f);
    }
    __syncthreads();

    constexpr int kElementsPerThread = 8;
    constexpr int kPackedPerThread   = kElementsPerThread / 2;
    const int     elements_per_cta   = kThreads * kElementsPerThread;

    const InType* row_input = input + static_cast<int64_t>(row) * src_ld;
    uint8_t*      row_fp4   = fp4_output + static_cast<int64_t>(row) * (k / 2);

    for (int base_col = 0; base_col < k; base_col += elements_per_cta) {
        const int col_start = base_col + tid * kElementsPerThread;
        if (col_start >= k) {
            break;
        }

        PackedType vec[4];
#pragma unroll
        for (int i = 0; i < kPackedPerThread; ++i) {
            const int col = col_start + i * 2;
            if (row < src_m && col + 1 < k) {
                vec[i] = *reinterpret_cast<const PackedType*>(&row_input[col]);
            }
            else if (row < src_m && col < k) {
                vec[i].x = row_input[col];
                vec[i].y = traits::zero();
            }
            else {
                vec[i] = traits::zero2();
            }
        }

        const int block_idx = col_start / kNvfp4GroupSize;
        uint8_t*  scale_out =
            (tid % 2 == 0) ? block_scales + fp4_sf_128x4_offset(row, block_idx, scale_cols) : nullptr;

        uint32_t e2m1_vals = quantize_fp16_to_e2m1_with_scaling<InType>(vec, global_scale, scale_out);

        const int packed_idx = col_start / 2;
        if (packed_idx + 3 < k / 2) {
            *reinterpret_cast<uint32_t*>(&row_fp4[packed_idx]) = e2m1_vals;
        }
        else {
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&e2m1_vals);
            for (int i = 0; i < 4 && packed_idx + i < k / 2; ++i) {
                row_fp4[packed_idx + i] = bytes[i];
            }
        }
    }
}

}  // namespace

void QuantizeFp8Groupwise(Tensor& out, Tensor& scale, const Tensor& src, cudaStream_t stream, int padded_m)
{
    TM_CHECK_EQ(src.ndim(), 2);
    TM_CHECK_EQ(src.stride(1), 1);
    TM_CHECK_EQ(src.shape(1) % kBlockSize, 0);

    const int src_m = static_cast<int>(src.shape(0));
    const int out_m = padded_m > 0 ? padded_m : src_m;
    const int k     = static_cast<int>(src.shape(1));
    TM_CHECK_GE(out_m, src_m);

    if (!out) {
        out = Tensor{{out_m, k}, kFloat8_e4m3, kDEVICE};
    }
    else {
        TM_CHECK(std::make_tuple(out_m, k) == out.shapes(0, 1));
        TM_CHECK_EQ(out.dtype(), kFloat8_e4m3);
    }

    const int k_blocks = k / kBlockSize;
    if (!scale) {
        scale = Tensor{{k_blocks, out_m}, kFloat, kDEVICE};
    }
    else {
        TM_CHECK(std::make_tuple(k_blocks, out_m) == scale.shapes(0, 1));
        TM_CHECK_EQ(scale.dtype(), kFloat);
    }

    const dim3 grid(out_m, k_blocks);
    auto invoke = [&](auto t) {
        using T = decltype(t);
        quantize_fp8_groupwise_kernel<<<grid, kBlockSize, 0, stream>>>(
            out.data<fp8_e4m3_t>(), out.stride(0), scale.data<float>(), src.data<T>(), src.stride(0), src_m, out_m, k);
    };
    TM_DISPATCH_PRIMARY_DTYPES(src.dtype(), invoke);
}

void QuantizeNvfp4(Tensor& out, Tensor& scale, const Tensor& src, const Tensor& input_scale, cudaStream_t stream, int padded_m)
{
    TM_CHECK_EQ(src.ndim(), 2);
    TM_CHECK_EQ(src.stride(1), 1);
    TM_CHECK_EQ(src.shape(1) % kNvfp4GroupSize, 0);
    TM_CHECK(input_scale);
    TM_CHECK_EQ(input_scale.dtype(), kFloat);

    const int src_m    = static_cast<int>(src.shape(0));
    const int out_m    = padded_m > 0 ? padded_m : src_m;
    const int k        = static_cast<int>(src.shape(1));
    const int k_groups = k / kNvfp4GroupSize;
    const int scale_m  = static_cast<int>(((static_cast<int64_t>(out_m) + 127) / 128) * 128);
    const int scale_k  = static_cast<int>(((k_groups + 3) / 4) * 4);
    TM_CHECK_GE(out_m, src_m);

    if (!out) {
        out = Tensor{{out_m, k}, kFloat4_e2m1, kDEVICE};
    }
    else {
        TM_CHECK(std::make_tuple(out_m, k) == out.shapes(0, 1));
        TM_CHECK_EQ(out.dtype(), kFloat4_e2m1);
    }
    check_cuda_error(cudaMemsetAsync(out.raw_data(), 0, out.byte_size(), stream));

    if (!scale) {
        scale = Tensor{{scale_m, scale_k}, kFloat8_e4m3, kDEVICE};
    }
    else {
        TM_CHECK(std::make_tuple(scale_m, scale_k) == scale.shapes(0, 1));
        TM_CHECK_EQ(scale.dtype(), kFloat8_e4m3);
    }
    check_cuda_error(cudaMemsetAsync(scale.raw_data(), 0, scale.byte_size(), stream));

    const dim3 grid(out_m);
    const dim3 block(128);
    if (src.dtype() == kBfloat16) {
        quantize_nvfp4_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(src.raw_data()),
            static_cast<int>(src.stride(0)),
            src_m,
            input_scale.data<float>(),
            static_cast<uint8_t*>(out.raw_data()),
            static_cast<uint8_t*>(scale.raw_data()),
            out_m,
            k,
            scale_k);
    }
    else if (src.dtype() == kHalf) {
        quantize_nvfp4_kernel<half><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(src.raw_data()),
                                                                static_cast<int>(src.stride(0)),
                                                                src_m,
                                                                input_scale.data<float>(),
                                                                static_cast<uint8_t*>(out.raw_data()),
                                                                static_cast<uint8_t*>(scale.raw_data()),
                                                                out_m,
                                                                k,
                                                                scale_k);
    }
    else {
        TM_CHECK(0) << "FlashInfer NVFP4 quantization only supports fp16/bf16 input, got " << to_string(src.dtype());
    }
}

void PrepareNvfp4WeightForGemm(Tensor& weight, Tensor& scales, int input_dim, int output_dim, cudaStream_t stream)
{
    TM_CHECK(weight);
    TM_CHECK(scales);
    TM_CHECK_EQ(weight.dtype(), kFloat4_e2m1);
    TM_CHECK_EQ(scales.dtype(), kFloat8_e4m3);
    TM_CHECK_EQ(input_dim % kNvfp4GroupSize, 0);
    TM_CHECK_EQ(input_dim % 2, 0);
    TM_CHECK_EQ(weight.shape(0), input_dim);
    TM_CHECK_EQ(weight.shape(1), output_dim);
    TM_CHECK_EQ(scales.shape(0), input_dim / kNvfp4GroupSize);
    TM_CHECK_EQ(scales.shape(1), output_dim);

    const int k_groups  = input_dim / kNvfp4GroupSize;
    const int scale_m   = static_cast<int>(((static_cast<int64_t>(output_dim) + 127) / 128) * 128);
    const int scale_k   = static_cast<int>(((k_groups + 3) / 4) * 4);
    Tensor    new_w{{output_dim, input_dim}, kFloat4_e2m1, kDEVICE};
    Tensor    new_s{{scale_m, scale_k}, kFloat8_e4m3, kDEVICE};
    check_cuda_error(cudaMemsetAsync(new_s.raw_data(), 0, new_s.byte_size(), stream));

    Buffer_<uint8_t> unpacked{weight.size(), kDEVICE};
    extend_to_u8(unpacked.data(), reinterpret_cast<const uint4_t*>(weight.raw_data()), weight.size(), stream);
    Tensor_<uint8_t> trans{{output_dim, input_dim}, kDEVICE};
    invokeTransposeAxis01(trans.data(), unpacked.data(), input_dim, output_dim, 1, stream);
    compact_to_u4(reinterpret_cast<uint4_t*>(new_w.raw_data()), trans.data(), new_w.size(), stream);

    constexpr int block = 256;
    const int64_t s_total = static_cast<int64_t>(k_groups) * output_dim;
    const int s_grid = std::min<int64_t>((s_total + block - 1) / block, 65535);
    prepare_nvfp4_scale_kernel<<<s_grid, block, 0, stream>>>(
        static_cast<uint8_t*>(new_s.raw_data()),
        static_cast<const uint8_t*>(scales.raw_data()),
        k_groups,
        output_dim,
        scale_k);

    weight = std::move(new_w);
    scales = std::move(new_s);
}

}  // namespace turbomind::flashinfer_gemm
