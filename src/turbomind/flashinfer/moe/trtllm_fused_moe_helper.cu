// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/flashinfer/moe/trtllm_fused_moe_helper.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/turbomind/core/context.h"
#include "src/turbomind/flashinfer/moe/trtllm_fused_moe_wrapper.h"
#include "src/turbomind/kernels/quantization.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/core/logger.h"

namespace turbomind {

namespace {

__device__ __forceinline__ int64_t trtllm_sf_out_offset_128x4(int row, int col, int num_cols)
{
    const int inner_k     = col & 3;
    const int inner_m     = (row & 127) >> 5;
    const int outer_m     = row & 31;
    const int k_tile      = col >> 2;
    const int m_tile      = row >> 7;
    const int num_k_tiles = (num_cols + 3) >> 2;
    return (static_cast<int64_t>(m_tile) * num_k_tiles + k_tile) * 512 + outer_m * 16 + inner_m * 4 + inner_k;
}

__device__ __forceinline__ uint8_t get_u4(const uint8_t* in, int64_t idx)
{
    const uint8_t byte = in[idx >> 1];
    return (idx & 1) ? (byte >> 4) : (byte & 0x0f);
}

template<bool ReverseK>
__global__ void packCutlassFp4MoeFc1Kernel(int64_t* out, const uint8_t* in, int hidden_dim, int inter_size)
{
    const int     packed_k = hidden_dim / 16;
    const int64_t total    = static_cast<int64_t>(2) * inter_size * packed_k;
    int64_t       idx      = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    for (; idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int k_pack  = idx % packed_k;
        const int out_row = idx / packed_k;
        // CUTLASS gated activation reads [linear/up | gate], while TurboMind
        // stores fused experts as [gate | up].
        const int     src_row       = out_row < inter_size ? out_row + inter_size : out_row - inter_size;
        const int64_t row_stride_u4 = static_cast<int64_t>(2) * inter_size;
        uint64_t      packed        = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int     h     = k_pack * 16 + j;
            const uint8_t v     = get_u4(in, static_cast<int64_t>(h) * row_stride_u4 + src_row);
            const int     dst_j = ReverseK ? (15 - j) : j;
            packed |= static_cast<uint64_t>(v & 0x0f) << (4 * dst_j);
        }
        out[idx] = static_cast<int64_t>(packed);
    }
}

template<bool ReverseK>
__global__ void packCutlassFp4MoeFc2Kernel(int64_t* out, const uint8_t* in, int inter_size, int hidden_dim)
{
    const int     packed_k = inter_size / 16;
    const int64_t total    = static_cast<int64_t>(hidden_dim) * packed_k;
    int64_t       idx      = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    for (; idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int     k_pack        = idx % packed_k;
        const int     h             = idx / packed_k;
        const int64_t row_stride_u4 = hidden_dim;
        uint64_t      packed        = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int     i     = k_pack * 16 + j;
            const uint8_t v     = get_u4(in, static_cast<int64_t>(i) * row_stride_u4 + h);
            const int     dst_j = ReverseK ? (15 - j) : j;
            packed |= static_cast<uint64_t>(v & 0x0f) << (4 * dst_j);
        }
        out[idx] = static_cast<int64_t>(packed);
    }
}

__device__ __forceinline__ uint8_t e4m3_scale_to_e8m0(uint8_t x)
{
    __nv_fp8_e4m3 v;
    v.__x             = x;
    const float scale = static_cast<float>(v);
    if (!(scale > 0.f)) {
        return 0;
    }
    int exp = static_cast<int>(ceilf(log2f(scale))) + 127;
    exp     = exp < 0 ? 0 : (exp > 255 ? 255 : exp);
    return static_cast<uint8_t>(exp);
}

__device__ __forceinline__ uint8_t e4m3_scale_to_e8m0_round(uint8_t x)
{
    __nv_fp8_e4m3 v;
    v.__x             = x;
    const float scale = static_cast<float>(v);
    if (!(scale > 0.f)) {
        return 0;
    }
    int exp = static_cast<int>(nearbyintf(log2f(scale))) + 127;
    exp     = exp < 0 ? 0 : (exp > 255 ? 255 : exp);
    return static_cast<uint8_t>(exp);
}

template<int ScaleMode>
__global__ void
packCutlassFp4MoeFc1ScaleKernel(int32_t* out, const uint8_t* in, int hidden_dim, int inter_size, int group_size)
{
    const int     scale_k   = hidden_dim / group_size;
    const int64_t total     = static_cast<int64_t>(2) * inter_size * scale_k;
    auto*         out_bytes = reinterpret_cast<uint8_t*>(out);
    int64_t       idx       = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    for (; idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int     k_block = idx % scale_k;
        const int     out_row = idx / scale_k;
        const int     src_row = out_row < inter_size ? out_row + inter_size : out_row - inter_size;
        const uint8_t src     = in[static_cast<int64_t>(k_block) * 2 * inter_size + src_row];
        const int64_t dst     = trtllm_sf_out_offset_128x4(out_row, k_block, scale_k);
        if constexpr (ScaleMode == 1) {
            out_bytes[dst] = e4m3_scale_to_e8m0(src);
        }
        else if constexpr (ScaleMode == 2) {
            out_bytes[dst] = e4m3_scale_to_e8m0_round(src);
        }
        else {
            out_bytes[dst] = src;
        }
    }
}

template<int ScaleMode>
__global__ void
packCutlassFp4MoeFc2ScaleKernel(int32_t* out, const uint8_t* in, int inter_size, int hidden_dim, int group_size)
{
    const int     scale_k   = inter_size / group_size;
    const int64_t total     = static_cast<int64_t>(hidden_dim) * scale_k;
    auto*         out_bytes = reinterpret_cast<uint8_t*>(out);
    int64_t       idx       = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    for (; idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int     k_block = idx % scale_k;
        const int     h       = idx / scale_k;
        const uint8_t src     = in[static_cast<int64_t>(k_block) * hidden_dim + h];
        const int64_t dst     = trtllm_sf_out_offset_128x4(h, k_block, scale_k);
        if constexpr (ScaleMode == 1) {
            out_bytes[dst] = e4m3_scale_to_e8m0(src);
        }
        else if constexpr (ScaleMode == 2) {
            out_bytes[dst] = e4m3_scale_to_e8m0_round(src);
        }
        else {
            out_bytes[dst] = src;
        }
    }
}

__global__ void buildMoeTopKForCutlassKernel(int*          out_ids,
                                             float*        out_scales,
                                             const int8_t* masks,
                                             const float*  in_scales,
                                             int           tokens,
                                             int           tokens_padded,
                                             int           experts,
                                             int           top_k)
{
    const int total = tokens * experts;
    int       idx   = threadIdx.x + blockIdx.x * blockDim.x;
    for (; idx < total; idx += blockDim.x * gridDim.x) {
        const int token  = idx % tokens;
        const int expert = idx / tokens;
        const int k      = static_cast<int>(masks[expert * tokens_padded + token]);
        if (0 <= k && k < top_k) {
            out_ids[token * top_k + k]    = expert;
            out_scales[token * top_k + k] = in_scales[k * tokens + token];
        }
    }
}

void invokePackCutlassFp4MoeFc1(int64_t* out, const uint8_t* in, int hidden_dim, int inter_size, cudaStream_t stream)
{
    TM_CHECK_EQ(hidden_dim % 16, 0);
    constexpr int block = 512;
    const int64_t total = static_cast<int64_t>(2) * inter_size * (hidden_dim / 16);
    const int     grid  = std::min<int64_t>((total + block - 1) / block, 65535);
    const char*   mode  = std::getenv("LMDEPLOY_W4A8_WEIGHT_PACK_MODE");
    if (mode && std::string_view(mode) == "reverse") {
        packCutlassFp4MoeFc1Kernel<true><<<grid, block, 0, stream>>>(out, in, hidden_dim, inter_size);
    }
    else {
        packCutlassFp4MoeFc1Kernel<false><<<grid, block, 0, stream>>>(out, in, hidden_dim, inter_size);
    }
}

void invokePackCutlassFp4MoeFc2(int64_t* out, const uint8_t* in, int inter_size, int hidden_dim, cudaStream_t stream)
{
    TM_CHECK_EQ(inter_size % 16, 0);
    constexpr int block = 512;
    const int64_t total = static_cast<int64_t>(hidden_dim) * (inter_size / 16);
    const int     grid  = std::min<int64_t>((total + block - 1) / block, 65535);
    const char*   mode  = std::getenv("LMDEPLOY_W4A8_WEIGHT_PACK_MODE");
    if (mode && std::string_view(mode) == "reverse") {
        packCutlassFp4MoeFc2Kernel<true><<<grid, block, 0, stream>>>(out, in, inter_size, hidden_dim);
    }
    else {
        packCutlassFp4MoeFc2Kernel<false><<<grid, block, 0, stream>>>(out, in, inter_size, hidden_dim);
    }
}

void invokePackCutlassFp4MoeFc1Scale(
    int32_t* out, const uint8_t* in, int hidden_dim, int inter_size, int group_size, cudaStream_t stream)
{
    TM_CHECK_EQ(group_size, 32);
    TM_CHECK_EQ((2 * inter_size) % 128, 0);
    TM_CHECK_EQ(hidden_dim % group_size, 0);
    TM_CHECK_EQ((hidden_dim / group_size) % 4, 0);
    constexpr int block = 512;
    const int64_t total = static_cast<int64_t>(2) * inter_size * (hidden_dim / group_size);
    const int     grid  = std::min<int64_t>((total + block - 1) / block, 65535);
    const char*   mode  = std::getenv("LMDEPLOY_W4A8_SCALE_BLOCK_MODE");
    if (mode && std::string_view(mode) == "raw") {
        packCutlassFp4MoeFc1ScaleKernel<0><<<grid, block, 0, stream>>>(out, in, hidden_dim, inter_size, group_size);
    }
    else if (mode && std::string_view(mode) == "e8_ceil") {
        packCutlassFp4MoeFc1ScaleKernel<1><<<grid, block, 0, stream>>>(out, in, hidden_dim, inter_size, group_size);
    }
    else {
        packCutlassFp4MoeFc1ScaleKernel<2><<<grid, block, 0, stream>>>(out, in, hidden_dim, inter_size, group_size);
    }
}

void invokePackCutlassFp4MoeFc2Scale(
    int32_t* out, const uint8_t* in, int inter_size, int hidden_dim, int group_size, cudaStream_t stream)
{
    TM_CHECK_EQ(group_size, 32);
    TM_CHECK_EQ(hidden_dim % 128, 0);
    TM_CHECK_EQ(inter_size % group_size, 0);
    TM_CHECK_EQ((inter_size / group_size) % 4, 0);
    constexpr int block = 512;
    const int64_t total = static_cast<int64_t>(hidden_dim) * (inter_size / group_size);
    const int     grid  = std::min<int64_t>((total + block - 1) / block, 65535);
    const char*   mode  = std::getenv("LMDEPLOY_W4A8_SCALE_BLOCK_MODE");
    if (mode && std::string_view(mode) == "raw") {
        packCutlassFp4MoeFc2ScaleKernel<0><<<grid, block, 0, stream>>>(out, in, inter_size, hidden_dim, group_size);
    }
    else if (mode && std::string_view(mode) == "e8_ceil") {
        packCutlassFp4MoeFc2ScaleKernel<1><<<grid, block, 0, stream>>>(out, in, inter_size, hidden_dim, group_size);
    }
    else {
        packCutlassFp4MoeFc2ScaleKernel<2><<<grid, block, 0, stream>>>(out, in, inter_size, hidden_dim, group_size);
    }
}

void invokeBuildMoeTopKForCutlass(int*          out_ids,
                                  float*        out_scales,
                                  const int8_t* masks,
                                  const float*  in_scales,
                                  int           tokens,
                                  int           tokens_padded,
                                  int           experts,
                                  int           top_k,
                                  cudaStream_t  stream)
{
    constexpr int block = 256;
    const int     total = tokens * experts;
    const int     grid  = std::min<int>((total + block - 1) / block, 65535);
    buildMoeTopKForCutlassKernel<<<grid, block, 0, stream>>>(
        out_ids, out_scales, masks, in_scales, tokens, tokens_padded, experts, top_k);
}

void DumpFirstBf16(const Tensor& t, const char* tag, int layer_id)
{
    if (layer_id != 0 || !std::getenv("LMDEPLOY_DEBUG_L0_MOE") || t.size() == 0) {
        return;
    }
    if (t.dtype() != kBfloat16) {
        std::cerr << "[L0MOE] " << tag << " dtype=" << (int)t.dtype() << " (skip bf16 dump)\n";
        return;
    }
    const int                n = std::min<int>(8, t.size());
    std::vector<nv_bfloat16> host(n);
    auto                     stream = core::Context::stream().handle();
    cudaMemcpyAsync(host.data(), t.raw_data(), n * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::ostringstream oss;
    oss << "[L0MOE] " << tag << " =";
    for (auto v : host) {
        oss << " " << (float)v;
    }
    std::cerr << oss.str() << std::endl;
}

void DumpFirstFloatTensor(const Tensor& t, const char* tag, int layer_id)
{
    if (layer_id != 0 || !std::getenv("LMDEPLOY_DEBUG_L0_MOE") || t.size() == 0) {
        return;
    }
    if (t.dtype() != kFloat) {
        std::cerr << "[L0MOE] " << tag << " dtype=" << (int)t.dtype() << " (skip float dump)\n";
        return;
    }
    const int          n = std::min<int>(16, t.size());
    std::vector<float> host(n);
    auto               stream = core::Context::stream().handle();
    cudaMemcpyAsync(host.data(), t.raw_data(), n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::ostringstream oss;
    oss << "[L0MOE] " << tag << " =";
    for (auto v : host) {
        oss << " " << v;
    }
    std::cerr << oss.str() << std::endl;
}

void DumpFirstIntTensor(const Tensor& t, const char* tag, int layer_id)
{
    if (layer_id != 0 || !std::getenv("LMDEPLOY_DEBUG_L0_MOE") || t.size() == 0) {
        return;
    }
    if (t.dtype() != kInt32) {
        std::cerr << "[L0MOE] " << tag << " dtype=" << (int)t.dtype() << " (skip int dump)\n";
        return;
    }
    const int        n = std::min<int>(16, t.size());
    std::vector<int> host(n);
    auto             stream = core::Context::stream().handle();
    cudaMemcpyAsync(host.data(), t.raw_data(), n * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::ostringstream oss;
    oss << "[L0MOE] " << tag << " =";
    for (auto v : host) {
        oss << " " << v;
    }
    std::cerr << oss.str() << std::endl;
}

}  // namespace

TrtllmFusedMoeHelper::TrtllmFusedMoeHelper(int layer_num, int inter_size, int hidden_dim):
    inter_size_(inter_size), hidden_dim_(hidden_dim), layer_cache_(layer_num)
{
}

TrtllmFusedMoeHelper::LayerCache&
TrtllmFusedMoeHelper::PrepareScales(const MoeFfnWeight& moe, int layer_id, cudaStream_t st)
{
    TM_CHECK_GE(layer_id, 0);
    if (static_cast<size_t>(layer_id) >= layer_cache_.size()) {
        layer_cache_.resize(layer_id + 1);
    }
    auto& cache = layer_cache_[layer_id];
    if (cache.prepared) {
        return cache;
    }

    constexpr int kW4A8GroupSize = 32;
    const int     E              = static_cast<int>(moe.experts.size());
    cache.fc1_weights            = Tensor{{E, 2 * inter_size_, hidden_dim_ / 16}, kInt64, kDEVICE};
    cache.fc2_weights            = Tensor{{E, hidden_dim_, inter_size_ / 16}, kInt64, kDEVICE};
    cache.fc1_scale_blocks       = Tensor{{E, 2 * inter_size_, hidden_dim_ / 128}, kInt32, kDEVICE};
    cache.fc2_scale_blocks       = Tensor{{E, hidden_dim_, inter_size_ / 128}, kInt32, kDEVICE};
    cache.fc1_global_scales      = Buffer_<float>{E, kDEVICE};
    cache.fc2_act_global_scales  = Buffer_<float>{E, kDEVICE};
    cache.fc2_global_scales      = Buffer_<float>{E, kDEVICE};
    cache.fc1_input_scale        = Tensor{{1, 1}, kFloat, kDEVICE};

    std::vector<float> h_fc1_global(E);
    std::vector<float> h_fc2_act_global(E);
    std::vector<float> h_fc2_global(E);
    std::vector<float> h_fc1_input(E);
    std::vector<float> h_fc2_input(E);
    std::vector<float> h_fc1_scale_2(E);
    std::vector<float> h_fc2_scale_2(E);
    float              h_fc1_input_scale = 0.f;
    float              h_fc2_input_scale = 0.f;

    const char*       scale_mode_env = std::getenv("LMDEPLOY_W4A8_SCALE_MODE");
    const std::string scale_mode     = scale_mode_env && *scale_mode_env ? scale_mode_env : "plain";

    auto close_enough = [](float a, float b) {
        const float denom = std::max(std::max(std::abs(a), std::abs(b)), 1.0e-12f);
        return std::abs(a - b) / denom < 1.0e-3f;
    };

    for (int i = 0; i < E; ++i) {
        const auto& fc1 = moe.experts[i]->fused_gating_intermediate;
        const auto& fc2 = moe.experts[i]->output;
        TM_CHECK(fc1.weight && fc2.weight && fc1.scales && fc2.scales && fc1.scale_2 && fc2.scale_2 && fc1.input_scales
                 && fc2.input_scales)
            << "W4A8_NVFP4_FP8 MoE requires FP4 weights, FP8 block scales, global scales, and input scales.";
        TM_CHECK_EQ(fc1.group_size, kW4A8GroupSize);
        TM_CHECK_EQ(fc2.group_size, kW4A8GroupSize);
        TM_CHECK_EQ(fc1.scales.dtype(), kFloat8_e4m3);
        TM_CHECK_EQ(fc2.scales.dtype(), kFloat8_e4m3);

        auto* fc1_w_dst = static_cast<int64_t*>(cache.fc1_weights.raw_data())
                          + static_cast<size_t>(i) * 2 * inter_size_ * (hidden_dim_ / 16);
        invokePackCutlassFp4MoeFc1(
            fc1_w_dst, static_cast<const uint8_t*>(fc1.weight.raw_data()), hidden_dim_, inter_size_, st);

        auto* fc2_w_dst = static_cast<int64_t*>(cache.fc2_weights.raw_data())
                          + static_cast<size_t>(i) * hidden_dim_ * (inter_size_ / 16);
        invokePackCutlassFp4MoeFc2(
            fc2_w_dst, static_cast<const uint8_t*>(fc2.weight.raw_data()), inter_size_, hidden_dim_, st);

        auto* fc1_s_dst = static_cast<int32_t*>(cache.fc1_scale_blocks.raw_data())
                          + static_cast<size_t>(i) * 2 * inter_size_ * (hidden_dim_ / 128);
        invokePackCutlassFp4MoeFc1Scale(fc1_s_dst,
                                        static_cast<const uint8_t*>(fc1.scales.raw_data()),
                                        hidden_dim_,
                                        inter_size_,
                                        kW4A8GroupSize,
                                        st);

        auto* fc2_s_dst = static_cast<int32_t*>(cache.fc2_scale_blocks.raw_data())
                          + static_cast<size_t>(i) * hidden_dim_ * (inter_size_ / 128);
        invokePackCutlassFp4MoeFc2Scale(fc2_s_dst,
                                        static_cast<const uint8_t*>(fc2.scales.raw_data()),
                                        inter_size_,
                                        hidden_dim_,
                                        kW4A8GroupSize,
                                        st);

        float fc1_scale_2_host[2]{};
        float fc1_input_scale_host[2]{};
        float fc2_scale_2_host{};
        float fc2_input_scale_host{};
        check_cuda_error(
            cudaMemcpy(fc1_scale_2_host, fc1.scale_2.raw_data(), sizeof(fc1_scale_2_host), cudaMemcpyDeviceToHost));
        check_cuda_error(cudaMemcpy(
            fc1_input_scale_host, fc1.input_scales.raw_data(), sizeof(fc1_input_scale_host), cudaMemcpyDeviceToHost));
        check_cuda_error(
            cudaMemcpy(&fc2_scale_2_host, fc2.scale_2.raw_data(), sizeof(fc2_scale_2_host), cudaMemcpyDeviceToHost));
        check_cuda_error(cudaMemcpy(
            &fc2_input_scale_host, fc2.input_scales.raw_data(), sizeof(fc2_input_scale_host), cudaMemcpyDeviceToHost));

        TM_CHECK(close_enough(fc1_scale_2_host[0], fc1_scale_2_host[1]))
            << "CUTLASS W4A8 fused fc1 has one global scale, but gate/up scales differ: " << fc1_scale_2_host[0]
            << " vs " << fc1_scale_2_host[1];
        TM_CHECK(close_enough(fc1_input_scale_host[0], fc1_input_scale_host[1]))
            << "CUTLASS W4A8 fused fc1 has one input scale, but gate/up input scales differ: "
            << fc1_input_scale_host[0] << " vs " << fc1_input_scale_host[1];

        h_fc1_input[i]    = fc1_input_scale_host[0];
        h_fc2_input[i]    = fc2_input_scale_host;
        h_fc1_scale_2[i]  = fc1_scale_2_host[0];
        h_fc2_scale_2[i]  = fc2_scale_2_host;
        h_fc1_input_scale = std::max(h_fc1_input_scale, h_fc1_input[i]);
        h_fc2_input_scale = std::max(h_fc2_input_scale, h_fc2_input[i]);
    }

    TM_CHECK_GT(h_fc1_input_scale, 0.f);
    TM_CHECK_GT(h_fc2_input_scale, 0.f);

    for (int i = 0; i < E; ++i) {
        if (scale_mode == "modelopt_trtllm") {
            h_fc1_global[i]     = h_fc1_scale_2[i] * h_fc1_input_scale;
            h_fc2_act_global[i] = 1.f / h_fc2_input_scale;
            h_fc2_global[i]     = h_fc2_scale_2[i] * h_fc2_input_scale;
        }
        else if (scale_mode == "trtllm") {
            h_fc1_global[i]     = h_fc1_input_scale;
            h_fc2_act_global[i] = 1.f / h_fc2_input_scale;
            h_fc2_global[i]     = h_fc2_input_scale;
        }
        else if (scale_mode == "modelopt") {
            h_fc1_input_scale   = h_fc1_input[i];
            h_fc1_global[i]     = h_fc1_scale_2[i] * h_fc1_input[i];
            h_fc2_act_global[i] = h_fc2_input[i];
            h_fc2_global[i]     = h_fc2_scale_2[i];
        }
        else if (scale_mode == "fc2_act") {
            h_fc1_input_scale   = 1.f;
            h_fc1_global[i]     = h_fc1_scale_2[i];
            h_fc2_act_global[i] = h_fc2_input[i];
            h_fc2_global[i]     = h_fc2_scale_2[i];
        }
        else if (scale_mode == "fc1_quant") {
            h_fc1_input_scale   = h_fc1_input[i];
            h_fc1_global[i]     = h_fc1_scale_2[i] * h_fc1_input[i];
            h_fc2_act_global[i] = 1.f;
            h_fc2_global[i]     = h_fc2_scale_2[i];
        }
        else {
            h_fc1_input_scale   = 1.f;
            h_fc1_global[i]     = h_fc1_scale_2[i];
            h_fc2_act_global[i] = 1.f;
            h_fc2_global[i]     = h_fc2_scale_2[i];
        }
    }

    TM_CHECK_GT(h_fc1_input_scale, 0.f);

    check_cuda_error(cudaMemcpyAsync(
        cache.fc1_global_scales.data(), h_fc1_global.data(), sizeof(float) * E, cudaMemcpyHostToDevice, st));
    check_cuda_error(cudaMemcpyAsync(
        cache.fc2_act_global_scales.data(), h_fc2_act_global.data(), sizeof(float) * E, cudaMemcpyHostToDevice, st));
    check_cuda_error(cudaMemcpyAsync(
        cache.fc2_global_scales.data(), h_fc2_global.data(), sizeof(float) * E, cudaMemcpyHostToDevice, st));
    check_cuda_error(cudaMemcpyAsync(
        cache.fc1_input_scale.raw_data(), &h_fc1_input_scale, sizeof(float), cudaMemcpyHostToDevice, st));
    sync_check_cuda_error();

    cache.prepared = true;
    TM_LOG_INFO("[moe] CUTLASS W4A8_NVFP4_FP8 weights/scales prepared for layer %d (%d experts, scale_mode=%s)",
                layer_id,
                E,
                scale_mode.c_str());
    return cache;
}

void TrtllmFusedMoeHelper::PrepareWeights(const MoeFfnWeight& moe, int layer_id, cudaStream_t st)
{
    (void)PrepareScales(moe, layer_id, st);
}

bool TrtllmFusedMoeHelper::Dispatch(const MoeFfnWeight&   moe,
                                    const MoeParam&       param,
                                    const Tensor&         input,
                                    Tensor&               output,
                                    const Buffer_<int>&   masks,
                                    const Buffer_<float>& scales,
                                    int                   num_tokens,
                                    int                   tokens_padded,
                                    int                   layer_id,
                                    bool                  enable_pdl,
                                    cudaStream_t          st)
{
    if (input.dtype() != kBfloat16 && input.dtype() != kFloat16) {
        TM_LOG_WARNING("[moe] CUTLASS W4A8_NVFP4_FP8 path expects fp16/bf16 hidden states.");
        return false;
    }
    if (output.dtype() != kBfloat16) {
        TM_LOG_WARNING("[moe] CUTLASS W4A8_NVFP4_FP8 path currently writes BF16 output.");
        return false;
    }
    TM_CHECK_EQ(input.shape(0), num_tokens);
    TM_CHECK_EQ(input.shape(1), hidden_dim_);

    const int expert_num = static_cast<int>(moe.experts.size());
    TM_CHECK_GE(layer_id, 0);
    TM_CHECK_LT(static_cast<size_t>(layer_id), layer_cache_.size());
    auto& cache = layer_cache_[layer_id];
    TM_CHECK(cache.prepared) << "trtllm fused MoE W4A8 weights/scales were not prepared during model initialization.";

    if (!hidden_states_ || hidden_states_.shape(0) < num_tokens || hidden_states_.shape(1) != hidden_dim_) {
        hidden_states_ = Tensor{{num_tokens, hidden_dim_}, kFloat8_e4m3, kDEVICE};
    }
    const bool use_mxfp8_act_scaling = [] {
        const char* env = std::getenv("LMDEPLOY_W4A8_MXFP8_ACT");
        return env && *env && std::string_view(env) != "0";
    }();
    if (use_mxfp8_act_scaling
        && (!hidden_scales_ || hidden_scales_.shape(0) < num_tokens || hidden_scales_.shape(1) != hidden_dim_ / 32)) {
        hidden_scales_ = Tensor{{num_tokens, hidden_dim_ / 32}, kUint8, kDEVICE};
    }
    if (!topk_ids_ || topk_ids_.shape(0) < num_tokens || topk_ids_.shape(1) != param.experts_per_token) {
        topk_ids_       = Tensor{{num_tokens, param.experts_per_token}, kInt32, kDEVICE};
        expert_weights_ = Tensor{{num_tokens, param.experts_per_token}, kFloat, kDEVICE};
    }

    auto fp8_hidden_states = hidden_states_.slice({0, 0}, {num_tokens, hidden_dim_});
    auto fp8_hidden_scales =
        use_mxfp8_act_scaling ? hidden_scales_.slice({0, 0}, {num_tokens, hidden_dim_ / 32}) : Tensor{};
    auto topk_ids       = topk_ids_.slice({0, 0}, {num_tokens, param.experts_per_token});
    auto expert_weights = expert_weights_.slice({0, 0}, {num_tokens, param.experts_per_token});

    QuantizeStatic(fp8_hidden_states, input, cache.fc1_input_scale, st);
    if (use_mxfp8_act_scaling) {
        // UE8M0(1.0) is exponent-bias 127.  Keeping scale factors at 1 lets
        // this mode exercise the MXFP8 runner without changing activation scale.
        check_cuda_error(cudaMemsetAsync(fp8_hidden_scales.raw_data(), 127, fp8_hidden_scales.byte_size(), st));
    }
    invokeBuildMoeTopKForCutlass(topk_ids.data<int>(),
                                 expert_weights.data<float>(),
                                 reinterpret_cast<const int8_t*>(masks.data()),
                                 scales.data(),
                                 num_tokens,
                                 tokens_padded,
                                 expert_num,
                                 param.experts_per_token,
                                 st);
    sync_check_cuda_error();
    DumpFirstIntTensor(topk_ids, "cutlass_topk_ids", layer_id);
    DumpFirstFloatTensor(expert_weights, "cutlass_topk_scales", layer_id);

    namespace fi = turbomind::trtllm_fused_moe;
    fi::CutlassW4A8Nvfp4Fp8MoeParams cw4{};
    cw4.input                  = fp8_hidden_states.raw_data();
    cw4.token_selected_experts = topk_ids.raw_data();
    cw4.token_final_scales     = expert_weights.raw_data();
    cw4.fc1_expert_weights     = cache.fc1_weights.raw_data();
    cw4.fc2_expert_weights     = cache.fc2_weights.raw_data();
    cw4.fc1_weight_block_scale = cache.fc1_scale_blocks.raw_data();
    cw4.fc1_global_scale       = cache.fc1_global_scales.data();
    cw4.fc2_act_global_scale   = cache.fc2_act_global_scales.data();
    cw4.fc2_weight_block_scale = cache.fc2_scale_blocks.raw_data();
    cw4.fc2_global_scale       = cache.fc2_global_scales.data();
    cw4.input_sf               = use_mxfp8_act_scaling ? fp8_hidden_scales.raw_data() : nullptr;
    cw4.output                 = output.raw_data();
    cw4.num_tokens             = num_tokens;
    cw4.num_experts            = expert_num;
    cw4.hidden_size            = hidden_dim_;
    cw4.intermediate_size      = inter_size_;
    cw4.top_k                  = param.experts_per_token;
    cw4.enable_pdl             = enable_pdl;
    cw4.use_mxfp8_act_scaling  = use_mxfp8_act_scaling;
    cw4.gemm1_tactic           = -1;
    cw4.gemm2_tactic           = -1;
    cw4.stream                 = st;

    if (!fi::dispatch_cutlass_w4a8_nvfp4_fp8(cw4)) {
        return false;
    }
    sync_check_cuda_error();
    DumpFirstBf16(output, "output", layer_id);
    return true;
}

}  // namespace turbomind
