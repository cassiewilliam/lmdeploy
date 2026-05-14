// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/trtllm_fused_moe_backend.h"

#include <algorithm>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "src/turbomind/core/context.h"
#include "src/turbomind/flashinfer/gemm/flashinfer_gemm_wrapper.h"
#include "src/turbomind/flashinfer/moe/trtllm_fused_moe_helper.h"
#include "src/turbomind/flashinfer/moe/trtllm_fused_moe_wrapper.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/kernels/quantization.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/core/logger.h"

namespace turbomind {

namespace {

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

turbomind::trtllm_fused_moe::RoutingMethod GetRoutingMethod(const MoeParam& param)
{
    namespace fi = turbomind::trtllm_fused_moe;
    if (param.topk_method == "noaux_tc") {
        return fi::RoutingMethod::kDeepSeekV3;
    }
    return param.norm_topk_prob ? fi::RoutingMethod::kRenormalizeNaive : fi::RoutingMethod::kDefault;
}

size_t TensorBytes(const Tensor& tensor)
{
    return tensor ? static_cast<size_t>(tensor.byte_size()) : 0;
}

size_t DensePayloadBytes(const LlamaDenseWeight& dense)
{
    return TensorBytes(dense.weight) + TensorBytes(dense.scales) + TensorBytes(dense.scale_2) + TensorBytes(dense.zeros)
           + TensorBytes(dense.input_scales);
}

void ReleaseDensePayload(LlamaDenseWeight& dense)
{
    // Clear both the view AND the original LinearWeight so GPU memory is freed.
    if (dense.src) {
        dense.src->weight       = {};
        dense.src->scales       = {};
        dense.src->scale_2      = {};
        dense.src->zeros        = {};
        dense.src->input_scales = {};
    }
    dense.weight       = {};
    dense.scales       = {};
    dense.scale_2      = {};
    dense.zeros        = {};
    dense.input_scales = {};
}

}  // namespace

bool TrtllmFusedMoeBackend::ShouldUse(MoeBackend backend)
{
    return backend == MoeBackend::kTrtllmFusedMoe && trtllm_fused_moe::is_sm10x_capable();
}

TrtllmFusedMoeBackend::TrtllmFusedMoeBackend(const ModelParam& model, const MoeParam& param, const EngineParam& engine):
    inter_size_(param.inter_size / engine.mlp_tp_size),
    hidden_dim_(model.hidden_units),
    param_(param),
    moe_quant_w4a8_nvfp4_fp8_(model.moe_quant_algo == "w4a8_nvfp4_fp8"),
    enable_pdl_(engine.enable_pdl),
    fp8_layer_cache_(model.layer_num),
    fp8_blockscale_layer_cache_(model.layer_num),
    fp4_layer_cache_(model.layer_num),
    bf16_layer_cache_(model.layer_num),
    w4a8_helper_(moe_quant_w4a8_nvfp4_fp8_ ?
                     std::make_unique<TrtllmFusedMoeHelper>(model.layer_num, inter_size_, hidden_dim_) :
                     nullptr)
{
    const int max_expert_num = *std::max_element(param.expert_num.begin(), param.expert_num.end());
    const int max_token_num  = engine.max_forward_token_num * engine.attn_dp_size;

    // Keep the TRTLLM path allocation-free during forward and CUDA-graph
    // capture.  This mirrors the FMHA wrapper shape: allocate backend-owned
    // scratch once, then feed slices to the dispatch call.
    const DataType out_dtype = kBfloat16;
    gate_logits_buf_         = Tensor{{max_token_num, max_expert_num}, kFloat, kDEVICE};
    temp_buf_                = Tensor{{max_token_num, hidden_dim_}, out_dtype, kDEVICE};

    trtllm_fp8_hidden_states_ = Tensor{{max_token_num, hidden_dim_}, kFloat8_e4m3, kDEVICE};
    if (hidden_dim_ % 128 == 0) {
        trtllm_fp8_hidden_scales_ = Tensor{{hidden_dim_ / 128, max_token_num}, kFloat, kDEVICE};
        const int scale_rows      = hidden_dim_ / 128;
        const int max_bs          = std::max(engine.max_batch_size, 1);
        for (int n = 1; n <= max_bs; ++n) {
            trtllm_fp8_block_scales_per_n_.emplace(n, Tensor{{scale_rows, n}, kFloat, kDEVICE});
        }
    }

    if (hidden_dim_ % 16 == 0) {
        trtllm_fp4_hidden_states_  = Tensor{{max_token_num, hidden_dim_ / 2}, kUint8, kDEVICE};
        trtllm_fp4_hidden_scales_  = Tensor{{max_token_num, hidden_dim_ / 16}, kFloat8_e4m3, kDEVICE};
        trtllm_fp4_topk_ids_       = Tensor{{max_token_num, param_.experts_per_token}, kInt32, kDEVICE};
        trtllm_fp4_expert_weights_ = Tensor{{max_token_num, param_.experts_per_token}, kFloat, kDEVICE};
    }

    const size_t topk    = static_cast<size_t>(param_.experts_per_token);
    const size_t per_tok = topk * (3 * static_cast<size_t>(inter_size_) + 2 * static_cast<size_t>(hidden_dim_));
    const size_t arena_b = static_cast<size_t>(max_token_num) * per_tok + (size_t{256} << 20);
    TM_LOG_INFO("[moe] trtllm_fused_moe dispatch arena capacity = %zu MiB (max_tok=%d, topk=%zu, hidden=%d, inter=%d)",
                arena_b >> 20,
                max_token_num,
                topk,
                hidden_dim_,
                inter_size_);
    trtllm_fused_moe::set_dispatch_arena_capacity(arena_b);
}

TrtllmFusedMoeBackend::~TrtllmFusedMoeBackend() = default;

Tensor TrtllmFusedMoeBackend::RoutedOutput(Tensor output, int num_tokens)
{
    if (temp_buf_ && temp_buf_.shape(0) >= num_tokens && temp_buf_.shape(1) == hidden_dim_) {
        temp_ = temp_buf_.slice({0, 0}, {num_tokens, hidden_dim_});
    }
    else {
        temp_ = Tensor{{num_tokens, hidden_dim_}, output.dtype(), output.device()};
    }
    return temp_;
}

bool TrtllmFusedMoeBackend::NeedsNativeRouting(const MoeFfnWeight& moe) const
{
    if (param_.method != MoeParam::kFused) {
        return false;
    }
    const auto& fc1 = moe.block.fused_gating_intermediate;
    const auto& fc2 = moe.block.output;
    return moe_quant_w4a8_nvfp4_fp8_ && fc1.weight_type == kFloat4_e2m1 && fc2.weight_type == kFloat4_e2m1
           && fc1.group_size == 32 && fc2.group_size == 32;
}

bool TrtllmFusedMoeBackend::RequiresDispatch(const MoeFfnWeight& moe) const
{
    if (param_.method != MoeParam::kFused) {
        return false;
    }
    const auto& fc1 = moe.block.fused_gating_intermediate;
    const auto& fc2 = moe.block.output;
    return (fc1.weight_type == kBfloat16 && fc2.weight_type == kBfloat16)
           || (fc1.weight_type == kFloat8_e4m3 && fc2.weight_type == kFloat8_e4m3)
           || (fc1.weight_type == kFloat4_e2m1 && fc2.weight_type == kFloat4_e2m1
               && ((fc1.group_size == 16 && fc2.group_size == 16)
                   || (moe_quant_w4a8_nvfp4_fp8_ && fc1.group_size == 32 && fc2.group_size == 32)));
}

void TrtllmFusedMoeBackend::PrepareWeights(MoeFfnWeight& moe, int layer_id)
{
    if (moe.method != MoeParam::kFused) {
        return;
    }

    const auto& fc1           = moe.block.fused_gating_intermediate;
    const auto& fc2           = moe.block.output;
    auto        st            = core::Context::stream().handle();
    const char* prepared_path = nullptr;

    if (moe_quant_w4a8_nvfp4_fp8_ && fc1.weight_type == kFloat4_e2m1 && fc2.weight_type == kFloat4_e2m1
        && fc1.group_size == 32 && fc2.group_size == 32) {
        TM_CHECK(w4a8_helper_);
        w4a8_helper_->PrepareWeights(moe, layer_id, st);
        prepared_path = "W4A8_NVFP4_FP8";
    }
    else if (fc1.weight_type == kFloat4_e2m1 && fc2.weight_type == kFloat4_e2m1 && fc1.group_size == 16
             && fc2.group_size == 16) {
        (void)PrepareFp4Scales(moe, layer_id);
        prepared_path = "NVFP4";
    }
    else if (fc1.weight_type == kFloat8_e4m3 && fc2.weight_type == kFloat8_e4m3) {
        if (fc1.group_size == 1) {
            (void)PrepareFp8Scales(moe, layer_id);
            prepared_path = "FP8 per-tensor";
        }
        else if (fc1.group_size == 128 && fc2.group_size == 128) {
            (void)PrepareFp8BlockScale(moe, layer_id);
            prepared_path = "FP8 block-scale";
        }
    }
    else if (fc1.weight_type == kBfloat16 && fc2.weight_type == kBfloat16) {
        (void)PrepareBf16Weights(moe, layer_id);
        prepared_path = "BF16";
    }

    if (prepared_path) {
        ReleaseOriginalExpertWeights(moe, layer_id, prepared_path);
    }
}

void TrtllmFusedMoeBackend::ReleaseOriginalExpertWeights(MoeFfnWeight& moe, int layer_id, const char* backend_path)
{
    size_t released_bytes = 0;
    for (auto& expert : moe.experts) {
        if (!expert) {
            continue;
        }
        released_bytes += DensePayloadBytes(expert->fused_gating_intermediate);
        released_bytes += DensePayloadBytes(expert->output);
        ReleaseDensePayload(expert->fused_gating_intermediate);
        ReleaseDensePayload(expert->output);
    }

    released_bytes += DensePayloadBytes(moe.block.fused_gating_intermediate);
    released_bytes += DensePayloadBytes(moe.block.output);
    ReleaseDensePayload(moe.block.fused_gating_intermediate);
    ReleaseDensePayload(moe.block.output);

    TM_LOG_INFO("[moe] trtllm_fused_moe released original %s expert tensors for layer %d: %.2f MiB",
                backend_path,
                layer_id,
                static_cast<double>(released_bytes) / (1024.0 * 1024.0));
}

TrtllmFusedMoeBackend::TrtllmFp8LayerCache& TrtllmFusedMoeBackend::PrepareFp8Scales(MoeFfnWeight& moe, int layer_id)
{
    TM_CHECK_GE(layer_id, 0);
    if (static_cast<size_t>(layer_id) >= fp8_layer_cache_.size()) {
        fp8_layer_cache_.resize(layer_id + 1);
    }
    auto& cache = fp8_layer_cache_[layer_id];
    if (cache.prepared) {
        return cache;
    }

    const int E           = static_cast<int>(moe.experts.size());
    cache.fc1_weights     = Tensor{{E, 2 * inter_size_, hidden_dim_}, kFloat8_e4m3, kDEVICE};
    cache.fc2_weights     = Tensor{{E, hidden_dim_, inter_size_}, kFloat8_e4m3, kDEVICE};
    cache.w1_scales       = Buffer_<float>{E, kDEVICE};
    cache.w3_scales       = Buffer_<float>{E, kDEVICE};
    cache.w2_scales       = Buffer_<float>{E, kDEVICE};
    cache.fc1_input_scale = Tensor{{1, 1}, kFloat, kDEVICE};

    auto               st = core::Context::stream().handle();
    std::vector<float> h_w1_scales(E);
    std::vector<float> h_w3_scales(E);
    std::vector<float> h_w2_scales(E);
    float              h_fc1_input_scale = 0.f;

    for (int i = 0; i < E; ++i) {
        const auto& fc1        = moe.experts[i]->fused_gating_intermediate;
        const auto& fc2        = moe.experts[i]->output;
        const auto& fc1_scales = fc1.scales;
        const auto& fc2_scales = fc2.scales;
        TM_CHECK(fc1_scales && fc2_scales && fc1.input_scales)
            << "FP8 per-tensor MoE requires expert weight scales and fc1 input scale.";
        TM_CHECK_EQ(fc1.weight.shape(0), hidden_dim_);
        TM_CHECK_EQ(fc1.weight.shape(1), 2 * inter_size_);
        TM_CHECK_EQ(fc2.weight.shape(0), inter_size_);
        TM_CHECK_EQ(fc2.weight.shape(1), hidden_dim_);
        TM_CHECK_GE(fc1_scales.size(), 2);
        TM_CHECK_GE(fc2_scales.size(), 1);

        auto* fc1_dst = static_cast<uint8_t*>(cache.fc1_weights.raw_data())
                        + static_cast<size_t>(i) * 2 * inter_size_ * hidden_dim_;
        auto* fc1_src = static_cast<const uint8_t*>(fc1.weight.raw_data());
        invokePackTrtllmFp8MoeFc1(fc1_dst, fc1_src, hidden_dim_, inter_size_, st);

        auto* fc2_dst =
            static_cast<uint8_t*>(cache.fc2_weights.raw_data()) + static_cast<size_t>(i) * hidden_dim_ * inter_size_;
        auto* fc2_src = static_cast<const uint8_t*>(fc2.weight.raw_data());
        invokePackTrtllmFp8MoeFc2(fc2_dst, fc2_src, inter_size_, hidden_dim_, st);

        float fc1_scales_host[2]{};
        float fc2_scale_host{};
        float fc1_input_scale_host{};
        check_cuda_error(
            cudaMemcpy(fc1_scales_host, fc1_scales.raw_data(), sizeof(fc1_scales_host), cudaMemcpyDeviceToHost));
        check_cuda_error(
            cudaMemcpy(&fc2_scale_host, fc2_scales.raw_data(), sizeof(fc2_scale_host), cudaMemcpyDeviceToHost));
        check_cuda_error(cudaMemcpy(
            &fc1_input_scale_host, fc1.input_scales.raw_data(), sizeof(fc1_input_scale_host), cudaMemcpyDeviceToHost));

        h_fc1_input_scale = std::max(h_fc1_input_scale, fc1_input_scale_host);
        h_w1_scales[i]    = fc1_scales_host[0];
        h_w3_scales[i]    = fc1_scales_host[1];
        h_w2_scales[i]    = fc2_scale_host;
    }

    TM_CHECK_GT(h_fc1_input_scale, 0.f);
    for (int i = 0; i < E; ++i) {
        h_w1_scales[i] *= h_fc1_input_scale;
        h_w3_scales[i] *= h_fc1_input_scale;
    }

    check_cuda_error(
        cudaMemcpyAsync(cache.w1_scales.data(), h_w1_scales.data(), sizeof(float) * E, cudaMemcpyHostToDevice, st));
    check_cuda_error(
        cudaMemcpyAsync(cache.w3_scales.data(), h_w3_scales.data(), sizeof(float) * E, cudaMemcpyHostToDevice, st));
    check_cuda_error(
        cudaMemcpyAsync(cache.w2_scales.data(), h_w2_scales.data(), sizeof(float) * E, cudaMemcpyHostToDevice, st));
    check_cuda_error(cudaMemcpyAsync(
        cache.fc1_input_scale.raw_data(), &h_fc1_input_scale, sizeof(float), cudaMemcpyHostToDevice, st));
    sync_check_cuda_error();
    cache.prepared = true;
    TM_LOG_INFO("[moe] trtllm FP8 per-tensor weights/scales prepared for layer %d (%d experts, fc1_input_scale=%.6g)",
                layer_id,
                E,
                h_fc1_input_scale);
    return cache;
}

TrtllmFusedMoeBackend::TrtllmBf16LayerCache& TrtllmFusedMoeBackend::PrepareBf16Weights(MoeFfnWeight& moe, int layer_id)
{
    TM_CHECK_GE(layer_id, 0);
    if (static_cast<size_t>(layer_id) >= bf16_layer_cache_.size()) {
        bf16_layer_cache_.resize(layer_id + 1);
    }
    auto& cache = bf16_layer_cache_[layer_id];
    if (cache.prepared) {
        return cache;
    }

    const int E       = static_cast<int>(moe.experts.size());
    cache.fc1_weights = Tensor{{E, 2 * inter_size_, hidden_dim_}, kBfloat16, kDEVICE};
    cache.fc2_weights = Tensor{{E, hidden_dim_, inter_size_}, kBfloat16, kDEVICE};

    auto st = core::Context::stream().handle();
    for (int i = 0; i < E; ++i) {
        const auto& fc1 = moe.experts[i]->fused_gating_intermediate;
        const auto& fc2 = moe.experts[i]->output;
        TM_CHECK_EQ(fc1.weight_type, kBfloat16) << "BF16 MoE prepare expects BF16 fc1 weight";
        TM_CHECK_EQ(fc2.weight_type, kBfloat16) << "BF16 MoE prepare expects BF16 fc2 weight";
        TM_CHECK_EQ(fc1.weight.shape(0), hidden_dim_);
        TM_CHECK_EQ(fc1.weight.shape(1), 2 * inter_size_);
        TM_CHECK_EQ(fc2.weight.shape(0), inter_size_);
        TM_CHECK_EQ(fc2.weight.shape(1), hidden_dim_);

        auto* fc1_dst = static_cast<char*>(cache.fc1_weights.raw_data())
                        + static_cast<size_t>(i) * 2 * inter_size_ * hidden_dim_ * sizeof(__nv_bfloat16);
        invokePackTrtllmBf16MoeFc1(fc1_dst, fc1.weight.raw_data(), hidden_dim_, inter_size_, st);

        auto* fc2_dst = static_cast<char*>(cache.fc2_weights.raw_data())
                        + static_cast<size_t>(i) * hidden_dim_ * inter_size_ * sizeof(__nv_bfloat16);
        invokePackTrtllmBf16MoeFc2(fc2_dst, fc2.weight.raw_data(), inter_size_, hidden_dim_, st);
    }

    sync_check_cuda_error();
    cache.prepared = true;
    TM_LOG_INFO("[moe] trtllm BF16 weights packed (BlockMajorK) for layer %d (%d experts)", layer_id, E);
    return cache;
}

TrtllmFusedMoeBackend::TrtllmFp8BlockScaleLayerCache& TrtllmFusedMoeBackend::PrepareFp8BlockScale(MoeFfnWeight& moe,
                                                                                                  int layer_id)
{
    TM_CHECK_GE(layer_id, 0);
    if (static_cast<size_t>(layer_id) >= fp8_blockscale_layer_cache_.size()) {
        fp8_blockscale_layer_cache_.resize(layer_id + 1);
    }
    auto& cache = fp8_blockscale_layer_cache_[layer_id];
    if (cache.prepared) {
        return cache;
    }

    const int E = static_cast<int>(moe.experts.size());
    TM_CHECK_EQ(hidden_dim_ % 128, 0);
    TM_CHECK_EQ(inter_size_ % 128, 0);

    cache.fc1_weights = Tensor{{E, 2 * inter_size_, hidden_dim_}, kFloat8_e4m3, kDEVICE};
    cache.fc2_weights = Tensor{{E, hidden_dim_, inter_size_}, kFloat8_e4m3, kDEVICE};
    cache.fc1_scales  = Tensor{{E, (2 * inter_size_) / 128, hidden_dim_ / 128}, kFloat, kDEVICE};
    cache.fc2_scales  = Tensor{{E, hidden_dim_ / 128, inter_size_ / 128}, kFloat, kDEVICE};

    auto st = core::Context::stream().handle();
    for (int i = 0; i < E; ++i) {
        const auto& fc1 = moe.experts[i]->fused_gating_intermediate;
        const auto& fc2 = moe.experts[i]->output;
        TM_CHECK(fc1.weight && fc2.weight && fc1.scales && fc2.scales)
            << "FP8 block-scale MoE requires expert weights and block scales.";
        TM_CHECK_EQ(fc1.weight.dtype(), kFloat8_e4m3);
        TM_CHECK_EQ(fc2.weight.dtype(), kFloat8_e4m3);
        TM_CHECK_EQ(fc1.scales.dtype(), kFloat);
        TM_CHECK_EQ(fc2.scales.dtype(), kFloat);
        TM_CHECK_EQ(fc1.group_size, 128);
        TM_CHECK_EQ(fc2.group_size, 128);
        TM_CHECK_EQ(fc1.weight.shape(0), 2 * inter_size_);
        TM_CHECK_EQ(fc1.weight.shape(1), hidden_dim_);
        TM_CHECK_EQ(fc2.weight.shape(0), hidden_dim_);
        TM_CHECK_EQ(fc2.weight.shape(1), inter_size_);
        TM_CHECK_EQ(fc1.scales.shape(0), (2 * inter_size_) / 128);
        TM_CHECK_EQ(fc1.scales.shape(1), hidden_dim_ / 128);
        TM_CHECK_EQ(fc2.scales.shape(0), hidden_dim_ / 128);
        TM_CHECK_EQ(fc2.scales.shape(1), inter_size_ / 128);

        auto* fc1_w_dst = static_cast<uint8_t*>(cache.fc1_weights.raw_data())
                          + static_cast<size_t>(i) * 2 * inter_size_ * hidden_dim_;
        const auto* fc1_w_src       = static_cast<const uint8_t*>(fc1.weight.raw_data());
        const auto  fc1_w_half_size = static_cast<size_t>(inter_size_) * hidden_dim_;
        check_cuda_error(
            cudaMemcpyAsync(fc1_w_dst, fc1_w_src + fc1_w_half_size, fc1_w_half_size, cudaMemcpyDefault, st));
        check_cuda_error(
            cudaMemcpyAsync(fc1_w_dst + fc1_w_half_size, fc1_w_src, fc1_w_half_size, cudaMemcpyDefault, st));

        auto* fc1_s_dst = static_cast<float*>(cache.fc1_scales.raw_data())
                          + static_cast<size_t>(i) * (2 * inter_size_ / 128) * (hidden_dim_ / 128);
        const auto* fc1_s_src       = static_cast<const float*>(fc1.scales.raw_data());
        const auto  fc1_s_half_size = static_cast<size_t>(inter_size_ / 128) * (hidden_dim_ / 128);
        check_cuda_error(cudaMemcpyAsync(
            fc1_s_dst, fc1_s_src + fc1_s_half_size, fc1_s_half_size * sizeof(float), cudaMemcpyDefault, st));
        check_cuda_error(cudaMemcpyAsync(
            fc1_s_dst + fc1_s_half_size, fc1_s_src, fc1_s_half_size * sizeof(float), cudaMemcpyDefault, st));

        Copy(fc2.weight, cache.fc2_weights.slice(i, 1).squeeze(0));
        Copy(fc2.scales, cache.fc2_scales.slice(i, 1).squeeze(0));
    }
    sync_check_cuda_error();

    cache.prepared = true;
    TM_LOG_INFO("[moe] trtllm FP8 block-scale weights/scales prepared for layer %d (%d experts)", layer_id, E);
    return cache;
}

TrtllmFusedMoeBackend::TrtllmFp4LayerCache& TrtllmFusedMoeBackend::PrepareFp4Scales(MoeFfnWeight& moe, int layer_id)
{
    TM_CHECK_GE(layer_id, 0);
    if (static_cast<size_t>(layer_id) >= fp4_layer_cache_.size()) {
        fp4_layer_cache_.resize(layer_id + 1);
    }
    auto& cache = fp4_layer_cache_[layer_id];
    if (cache.prepared) {
        return cache;
    }

    constexpr int kNvfp4GroupSize = 16;
    const int     E               = static_cast<int>(moe.experts.size());
    cache.fc1_weights             = Tensor{{E, 2 * inter_size_, hidden_dim_ / 2}, kUint8, kDEVICE};
    cache.fc2_weights             = Tensor{{E, hidden_dim_, inter_size_ / 2}, kUint8, kDEVICE};
    cache.fc1_scales              = Tensor{{E, 2 * inter_size_, hidden_dim_ / kNvfp4GroupSize}, kFloat8_e4m3, kDEVICE};
    cache.fc2_scales              = Tensor{{E, hidden_dim_, inter_size_ / kNvfp4GroupSize}, kFloat8_e4m3, kDEVICE};
    cache.w1_scale_2              = Buffer_<float>{E, kDEVICE};
    cache.w3_scale_2              = Buffer_<float>{E, kDEVICE};
    cache.w2_scale_2              = Buffer_<float>{E, kDEVICE};

    auto               st = core::Context::stream().handle();
    std::vector<float> h_w1_scale_2(E);
    std::vector<float> h_w3_scale_2(E);
    std::vector<float> h_w2_scale_2(E);
    float              h_fc1_input_scale = 1.f;

    for (int i = 0; i < E; ++i) {
        const auto& fc1 = moe.experts[i]->fused_gating_intermediate;
        const auto& fc2 = moe.experts[i]->output;
        TM_CHECK(fc1.weight && fc2.weight && fc1.scales && fc2.scales && fc1.scale_2 && fc2.scale_2 && fc1.input_scales
                 && fc2.input_scales)
            << "NVFP4 MoE requires expert weights, fp8 block scales, second-level scales, and input scales.";
        TM_CHECK_EQ(fc1.weight.shape(0), hidden_dim_);
        TM_CHECK_EQ(fc1.weight.shape(1), 2 * inter_size_);
        TM_CHECK_EQ(fc2.weight.shape(0), inter_size_);
        TM_CHECK_EQ(fc2.weight.shape(1), hidden_dim_);
        TM_CHECK_EQ(fc1.scales.dtype(), kFloat8_e4m3);
        TM_CHECK_EQ(fc2.scales.dtype(), kFloat8_e4m3);

        auto* fc1_w_dst = static_cast<uint8_t*>(cache.fc1_weights.raw_data())
                          + static_cast<size_t>(i) * 2 * inter_size_ * (hidden_dim_ / 2);
        invokePackTrtllmFp4MoeFc1(
            fc1_w_dst, static_cast<const uint8_t*>(fc1.weight.raw_data()), hidden_dim_, inter_size_, st);

        auto* fc2_w_dst = static_cast<uint8_t*>(cache.fc2_weights.raw_data())
                          + static_cast<size_t>(i) * hidden_dim_ * (inter_size_ / 2);
        invokePackTrtllmFp4MoeFc2(
            fc2_w_dst, static_cast<const uint8_t*>(fc2.weight.raw_data()), inter_size_, hidden_dim_, st);

        auto* fc1_s_dst = static_cast<uint8_t*>(cache.fc1_scales.raw_data())
                          + static_cast<size_t>(i) * 2 * inter_size_ * (hidden_dim_ / kNvfp4GroupSize);
        invokePackTrtllmFp4MoeFc1Scale(fc1_s_dst,
                                       static_cast<const uint8_t*>(fc1.scales.raw_data()),
                                       hidden_dim_,
                                       inter_size_,
                                       kNvfp4GroupSize,
                                       st);

        auto* fc2_s_dst = static_cast<uint8_t*>(cache.fc2_scales.raw_data())
                          + static_cast<size_t>(i) * hidden_dim_ * (inter_size_ / kNvfp4GroupSize);
        invokePackTrtllmFp4MoeFc2Scale(fc2_s_dst,
                                       static_cast<const uint8_t*>(fc2.scales.raw_data()),
                                       inter_size_,
                                       hidden_dim_,
                                       kNvfp4GroupSize,
                                       st);

        float fc1_scale_2_host[2]{};
        float fc2_scale_2_host{};
        check_cuda_error(
            cudaMemcpy(fc1_scale_2_host, fc1.scale_2.raw_data(), sizeof(fc1_scale_2_host), cudaMemcpyDeviceToHost));
        check_cuda_error(
            cudaMemcpy(&fc2_scale_2_host, fc2.scale_2.raw_data(), sizeof(fc2_scale_2_host), cudaMemcpyDeviceToHost));
        h_w1_scale_2[i] = fc1_scale_2_host[0];
        h_w3_scale_2[i] = fc1_scale_2_host[1];
        h_w2_scale_2[i] = fc2_scale_2_host;
    }
    cache.fc1_input_scale = h_fc1_input_scale;

    check_cuda_error(
        cudaMemcpyAsync(cache.w1_scale_2.data(), h_w1_scale_2.data(), sizeof(float) * E, cudaMemcpyHostToDevice, st));
    check_cuda_error(
        cudaMemcpyAsync(cache.w3_scale_2.data(), h_w3_scale_2.data(), sizeof(float) * E, cudaMemcpyHostToDevice, st));
    check_cuda_error(
        cudaMemcpyAsync(cache.w2_scale_2.data(), h_w2_scale_2.data(), sizeof(float) * E, cudaMemcpyHostToDevice, st));
    sync_check_cuda_error();

    cache.prepared = true;
    TM_LOG_INFO("[moe] trtllm NVFP4 weights/scales prepared for layer %d (%d experts)", layer_id, E);
    return cache;
}

TrtllmFusedMoeBackend::DispatchResult TrtllmFusedMoeBackend::Dispatch(const DispatchParam& p)
{
    DispatchResult result{};
    if (param_.method != MoeParam::kFused) {
        return result;
    }

    const auto& moe               = *p.weights;
    const auto& fc1               = moe.block.fused_gating_intermediate;
    const auto& fc2               = moe.block.output;
    auto&       logits            = *p.logits;
    const bool  has_shared_expert = static_cast<bool>(moe.shared_gate_weight);

    namespace fi = turbomind::trtllm_fused_moe;
    fi::AutoMoeRequest             req{};
    Tensor                         fp8_hidden_states;
    Tensor                         fp8_hidden_scales;
    TrtllmFp8LayerCache*           fp8_cache       = nullptr;
    TrtllmFp8BlockScaleLayerCache* fp8_block_cache = nullptr;

    if (fc1.weight_type == kBfloat16 && fc2.weight_type == kBfloat16) {
        if (static_cast<size_t>(p.layer_id) >= bf16_layer_cache_.size() || !bf16_layer_cache_[p.layer_id].prepared) {
            return result;
        }
        const auto& bf16_cache = bf16_layer_cache_[p.layer_id];
        const int   num_tokens = p.input.shape(0);
        Tensor      routed     = has_shared_expert ? RoutedOutput(p.output, num_tokens) : p.output;

        fi::Bf16MoeParams pb{};
        pb.routing_logits       = const_cast<void*>(logits.raw_data());
        pb.routing_logits_dtype = fi::DType::kFP32;
        pb.routing_bias =
            (moe.score_correction_bias.size() > 0) ? const_cast<void*>(moe.score_correction_bias.raw_data()) : nullptr;
        pb.hidden_states       = const_cast<void*>(p.input.raw_data());
        pb.gemm1_weights       = const_cast<void*>(bf16_cache.fc1_weights.raw_data());
        pb.gemm2_weights       = const_cast<void*>(bf16_cache.fc2_weights.raw_data());
        pb.output              = routed.raw_data();
        pb.num_tokens          = num_tokens;
        pb.num_experts         = static_cast<int>(moe.experts.size());
        pb.hidden_size         = hidden_dim_;
        pb.intermediate_size   = inter_size_;
        pb.top_k               = param_.experts_per_token;
        pb.n_group             = param_.n_group;
        pb.topk_group          = param_.topk_group;
        pb.local_expert_offset = 0;
        pb.local_num_experts   = pb.num_experts;
        pb.routing_method      = GetRoutingMethod(param_);
        pb.use_shuffled_weight = true;
        pb.weight_layout       = 2;
        pb.enable_pdl          = enable_pdl_;
        pb.tile_n              = -1;
        pb.config_index        = -1;
        pb.stream              = core::Context::stream().handle();

        if (!fi::dispatch_bf16(pb)) {
            temp_ = {};
            return result;
        }
        sync_check_cuda_error();
        result.ok            = true;
        result.finalized     = !has_shared_expert;
        result.routed        = has_shared_expert;
        result.routed_output = routed;
        return result;
    }

    if (moe_quant_w4a8_nvfp4_fp8_ && fc1.weight_type == kFloat4_e2m1 && fc2.weight_type == kFloat4_e2m1
        && fc1.group_size == 32 && fc2.group_size == 32) {
        if (!w4a8_helper_ || !p.masks || !p.scales) {
            return result;
        }
        const int num_tokens = p.input.shape(0);
        Tensor    routed     = has_shared_expert ? RoutedOutput(p.output, num_tokens) : p.output;
        if (!w4a8_helper_->Dispatch(moe,
                                    param_,
                                    p.input,
                                    routed,
                                    *p.masks,
                                    *p.scales,
                                    num_tokens,
                                    p.tokens_padded,
                                    p.layer_id,
                                    enable_pdl_,
                                    core::Context::stream().handle())) {
            temp_ = {};
            return result;
        }
        result.ok            = true;
        result.finalized     = !has_shared_expert;
        result.routed        = has_shared_expert;
        result.routed_output = routed;
        return result;
    }

    if (fc1.weight_type == kFloat4_e2m1 && fc2.weight_type == kFloat4_e2m1 && fc1.group_size == 16
        && fc2.group_size == 16) {
        if (p.input.dtype() != kBfloat16 && p.input.dtype() != kFloat16) {
            TM_LOG_WARNING("[moe] trtllm NVFP4 path expects fp16/bf16 hidden states.");
            return result;
        }
        if (p.output.dtype() != kBfloat16) {
            TM_LOG_WARNING("[moe] trtllm NVFP4 path currently writes BF16 output.");
            return result;
        }

        TM_CHECK_GE(p.layer_id, 0);
        TM_CHECK_LT(static_cast<size_t>(p.layer_id), fp4_layer_cache_.size());
        auto& fp4_cache = fp4_layer_cache_[p.layer_id];
        TM_CHECK(fp4_cache.prepared)
            << "trtllm NVFP4 MoE weights/scales were not prepared during model initialization.";
        const auto [num_tokens, hidden_dim] = p.input.shapes(0, 1);
        if (!trtllm_fp4_hidden_states_ || trtllm_fp4_hidden_states_.shape(0) < num_tokens
            || trtllm_fp4_hidden_states_.shape(1) != hidden_dim / 2) {
            trtllm_fp4_hidden_states_ = Tensor{{num_tokens, hidden_dim / 2}, kUint8, kDEVICE};
            trtllm_fp4_hidden_scales_ = Tensor{{num_tokens, hidden_dim / 16}, kFloat8_e4m3, kDEVICE};
        }
        if (!trtllm_fp4_topk_ids_ || trtllm_fp4_topk_ids_.shape(0) < num_tokens
            || trtllm_fp4_topk_ids_.shape(1) != param_.experts_per_token) {
            trtllm_fp4_topk_ids_       = Tensor{{num_tokens, param_.experts_per_token}, kInt32, kDEVICE};
            trtllm_fp4_expert_weights_ = Tensor{{num_tokens, param_.experts_per_token}, kFloat, kDEVICE};
        }
        auto fp4_hidden_states  = trtllm_fp4_hidden_states_.slice({0, 0}, {num_tokens, hidden_dim / 2});
        auto fp4_hidden_scales  = trtllm_fp4_hidden_scales_.slice({0, 0}, {num_tokens, hidden_dim / 16});
        auto fp4_topk_ids       = trtllm_fp4_topk_ids_.slice({0, 0}, {num_tokens, param_.experts_per_token});
        auto fp4_expert_weights = trtllm_fp4_expert_weights_.slice({0, 0}, {num_tokens, param_.experts_per_token});
        invokeQuantizeTrtllmFp4MoeActivation(
            fp4_hidden_states, fp4_hidden_scales, p.input, fp4_cache.fc1_input_scale, core::Context::stream().handle());
        sync_check_cuda_error();

        const fi::AutotuneKey    key{/*variant=*/3,
                                  static_cast<int>(num_tokens),
                                  hidden_dim_,
                                  inter_size_,
                                  static_cast<int>(moe.experts.size()),
                                  param_.experts_per_token,
                                  static_cast<int>(moe.experts.size())};
        const fi::AutotuneTactic tactic = fi::lookup_tactic(key);
        Tensor                   routed = has_shared_expert ? RoutedOutput(p.output, num_tokens) : p.output;

        fi::Fp4BlockScaleMoeParamsRouting p4{};
        p4.routing_logits           = const_cast<void*>(logits.raw_data());
        p4.routing_logits_dtype     = fi::DType::kFP32;
        p4.topk_ids_workspace       = fp4_topk_ids.raw_data();
        p4.expert_weights_workspace = fp4_expert_weights.raw_data();
        p4.routing_bias =
            (moe.score_correction_bias.size() > 0) ? const_cast<void*>(moe.score_correction_bias.raw_data()) : nullptr;
        p4.hidden_states              = fp4_hidden_states.raw_data();
        p4.hidden_states_scale        = fp4_hidden_scales.raw_data();
        p4.gemm1_weights              = fp4_cache.fc1_weights.raw_data();
        p4.gemm1_weights_scale        = fp4_cache.fc1_scales.raw_data();
        p4.gemm2_weights              = fp4_cache.fc2_weights.raw_data();
        p4.gemm2_weights_scale        = fp4_cache.fc2_scales.raw_data();
        p4.output1_scales_gate_scalar = fp4_cache.w1_scale_2.data();
        p4.output1_scales_scalar      = fp4_cache.w3_scale_2.data();
        p4.output2_scales_scalar      = fp4_cache.w2_scale_2.data();
        p4.output                     = routed.raw_data();
        p4.num_tokens                 = num_tokens;
        p4.num_experts                = moe.experts.size();
        p4.hidden_size                = hidden_dim_;
        p4.intermediate_size          = inter_size_;
        p4.top_k                      = param_.experts_per_token;
        p4.n_group                    = param_.n_group;
        p4.topk_group                 = param_.topk_group;
        p4.local_expert_offset        = 0;
        p4.local_num_experts          = p4.num_experts;
        p4.weight_scale_vec_size      = 16;
        p4.routed_scaling_factor      = param_.routed_scale;
        p4.routing_method             = GetRoutingMethod(param_);
        p4.gated_act_type             = fi::GatedActType::kSwiGlu;
        p4.hidden_states_dtype        = fi::DType::kE2M1U8;
        p4.norm_topk_prob             = param_.norm_topk_prob;
        p4.do_finalize                = true;
        p4.enable_pdl                 = enable_pdl_;
        p4.tile_n                     = tactic.tile_n;
        p4.config_index               = tactic.config_index;
        p4.stream                     = core::Context::stream().handle();

        if (!fi::dispatch_fp4_block_scale(p4)) {
            return result;
        }
        sync_check_cuda_error();
        DumpFirstBf16(routed, "output", p.layer_id);
        result.ok            = true;
        result.finalized     = !has_shared_expert;
        result.routed        = has_shared_expert;
        result.routed_output = routed;
        return result;
    }

    if (fc1.weight_type != kFloat8_e4m3 || fc2.weight_type != kFloat8_e4m3) {
        return result;
    }

    req.weight_dtype      = fi::DType::kFP8E4M3;
    req.weight_block_size = fc1.group_size;
    if (fc1.group_size == 1) {
        TM_CHECK_GE(p.layer_id, 0);
        TM_CHECK_LT(static_cast<size_t>(p.layer_id), fp8_layer_cache_.size());
        fp8_cache = &fp8_layer_cache_[p.layer_id];
        TM_CHECK(fp8_cache->prepared)
            << "trtllm FP8 per-tensor MoE weights/scales were not prepared during model initialization.";
        if (!fp8_cache->fc1_input_scale) {
            TM_LOG_WARNING("[moe] trtllm FP8 per-tensor path requires fc1 input scale.");
            return result;
        }
        if (p.input.dtype() != kBfloat16 && p.input.dtype() != kFloat16) {
            TM_LOG_WARNING("[moe] trtllm FP8 per-tensor pre-quant expects fp16/bf16 hidden states.");
            return result;
        }
        const auto [num_tokens, hidden_dim] = p.input.shapes(0, 1);
        if (!trtllm_fp8_hidden_states_ || trtllm_fp8_hidden_states_.shape(0) < num_tokens
            || trtllm_fp8_hidden_states_.shape(1) != hidden_dim) {
            trtllm_fp8_hidden_states_ = Tensor{{num_tokens, hidden_dim}, kFloat8_e4m3, kDEVICE};
        }
        fp8_hidden_states = trtllm_fp8_hidden_states_.slice({0, 0}, {num_tokens, hidden_dim});
        QuantizeStatic(fp8_hidden_states, p.input, fp8_cache->fc1_input_scale, core::Context::stream().handle());
        sync_check_cuda_error();
    }
    else if (fc1.group_size == 128 && fc2.group_size == 128) {
        TM_CHECK_GE(p.layer_id, 0);
        TM_CHECK_LT(static_cast<size_t>(p.layer_id), fp8_blockscale_layer_cache_.size());
        fp8_block_cache = &fp8_blockscale_layer_cache_[p.layer_id];
        TM_CHECK(fp8_block_cache->prepared)
            << "trtllm FP8 block-scale MoE weights/scales were not prepared during model initialization.";
        if (p.input.dtype() != kBfloat16 && p.input.dtype() != kFloat16) {
            TM_LOG_WARNING("[moe] trtllm FP8 block-scale pre-quant expects fp16/bf16 hidden states.");
            return result;
        }
        const auto [num_tokens, hidden_dim] = p.input.shapes(0, 1);
        const int scale_rows                = hidden_dim / 128;
        if (!trtllm_fp8_hidden_states_ || trtllm_fp8_hidden_states_.shape(0) < num_tokens
            || trtllm_fp8_hidden_states_.shape(1) != hidden_dim) {
            trtllm_fp8_hidden_states_ = Tensor{{num_tokens, hidden_dim}, kFloat8_e4m3, kDEVICE};
        }
        auto scales_it = trtllm_fp8_block_scales_per_n_.find(num_tokens);
        if (scales_it == trtllm_fp8_block_scales_per_n_.end()) {
            scales_it =
                trtllm_fp8_block_scales_per_n_.emplace(num_tokens, Tensor{{scale_rows, num_tokens}, kFloat, kDEVICE})
                    .first;
        }
        fp8_hidden_states = trtllm_fp8_hidden_states_.slice({0, 0}, {num_tokens, hidden_dim});
        fp8_hidden_scales = scales_it->second;
        flashinfer_gemm::QuantizeFp8Groupwise(
            fp8_hidden_states, fp8_hidden_scales, p.input, core::Context::stream().handle());
        sync_check_cuda_error();
    }
    else {
        TM_LOG_WARNING("[moe] unsupported FP8 block size combination for trtllm path: fc1=%d, fc2=%d.",
                       fc1.group_size,
                       fc2.group_size);
        return result;
    }

    req.routing_logits       = const_cast<void*>(logits.raw_data());
    req.routing_logits_dtype = fi::DType::kFP32;
    req.routing_bias =
        (moe.score_correction_bias.size() > 0) ? const_cast<void*>(moe.score_correction_bias.raw_data()) : nullptr;
    req.hidden_states       = fp8_hidden_states ? fp8_hidden_states.raw_data() : const_cast<void*>(p.input.raw_data());
    req.hidden_states_scale = fp8_hidden_scales ? fp8_hidden_scales.raw_data() : nullptr;
    req.gemm1_weights       = fp8_cache && fp8_cache->fc1_weights ? fp8_cache->fc1_weights.raw_data() :
                              fp8_block_cache && fp8_block_cache->fc1_weights ? fp8_block_cache->fc1_weights.raw_data() :
                                                                                const_cast<void*>(fc1.weight.raw_data());
    req.gemm1_weights_scale = fp8_block_cache && fp8_block_cache->fc1_scales ?
                                  fp8_block_cache->fc1_scales.raw_data() :
                                  (fc1.scales ? const_cast<void*>(fc1.scales.raw_data()) : nullptr);
    req.gemm2_weights       = fp8_cache && fp8_cache->fc2_weights ? fp8_cache->fc2_weights.raw_data() :
                              fp8_block_cache && fp8_block_cache->fc2_weights ? fp8_block_cache->fc2_weights.raw_data() :
                                                                                const_cast<void*>(fc2.weight.raw_data());
    req.gemm2_weights_scale = fp8_block_cache && fp8_block_cache->fc2_scales ?
                                  fp8_block_cache->fc2_scales.raw_data() :
                                  (fc2.scales ? const_cast<void*>(fc2.scales.raw_data()) : nullptr);

    Tensor routed = has_shared_expert ? RoutedOutput(p.output, p.input.shape(0)) : p.output;
    req.output    = routed.raw_data();

    if (req.weight_block_size == 1 && fp8_cache && fp8_cache->prepared) {
        req.output1_scales_gate_scalar = fp8_cache->w1_scales.data();
        req.output1_scales_scalar      = fp8_cache->w3_scales.data();
        req.output2_scales_scalar      = fp8_cache->w2_scales.data();
    }
    req.use_routing_scales_on_input = false;
    req.num_tokens                  = p.input.shape(0);
    req.num_experts                 = moe.experts.size();
    req.hidden_size                 = hidden_dim_;
    req.intermediate_size           = inter_size_;
    req.top_k                       = param_.experts_per_token;
    req.n_group                     = param_.n_group;
    req.topk_group                  = param_.topk_group;
    req.local_expert_offset         = 0;
    req.local_num_experts           = req.num_experts;
    req.hidden_states_dtype =
        fp8_hidden_states ? fi::DType::kFP8E4M3 : ((p.input.dtype() == kFloat16) ? fi::DType::kFP16 : fi::DType::kBF16);
    req.topk_method           = param_.topk_method.c_str();
    req.norm_topk_prob        = param_.norm_topk_prob;
    req.routed_scaling_factor = param_.routed_scale;
    req.use_shuffled_weight   = req.weight_block_size != 128;
    req.weight_layout         = 0;
    req.enable_pdl            = enable_pdl_;
    req.stream                = core::Context::stream().handle();

    if (!fi::dispatch_auto(req)) {
        return result;
    }
    sync_check_cuda_error();
    DumpFirstBf16(routed, "output", p.layer_id);
    result.ok            = true;
    result.finalized     = !has_shared_expert;
    result.routed        = has_shared_expert;
    result.routed_output = routed;
    return result;
}

}  // namespace turbomind
