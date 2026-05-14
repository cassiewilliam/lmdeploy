// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class TrtllmFusedMoeHelper;

class TrtllmFusedMoeBackend {
public:
    static bool ShouldUse(MoeBackend backend);

    TrtllmFusedMoeBackend(const ModelParam& model, const MoeParam& param, const EngineParam& engine);
    ~TrtllmFusedMoeBackend();

    TrtllmFusedMoeBackend(const TrtllmFusedMoeBackend&)            = delete;
    TrtllmFusedMoeBackend& operator=(const TrtllmFusedMoeBackend&) = delete;

    void PrepareWeights(MoeFfnWeight& moe, int layer_id);

    bool RequiresDispatch(const MoeFfnWeight& moe) const;
    bool NeedsNativeRouting(const MoeFfnWeight& moe) const;

    struct DispatchParam {
        Tensor                input;
        Tensor                output;
        const MoeFfnWeight*   weights;
        Tensor_<float>*       logits;
        const Buffer_<int>*   masks;
        const Buffer_<float>* scales;
        int                   tokens_padded;
        int                   layer_id;
    };

    struct DispatchResult {
        bool   ok{false};
        bool   finalized{false};
        bool   routed{false};
        Tensor routed_output;
    };

    DispatchResult Dispatch(const DispatchParam& p);

private:
    struct TrtllmFp8LayerCache {
        Tensor         fc1_weights;
        Tensor         fc2_weights;
        Buffer_<float> w1_scales;
        Buffer_<float> w3_scales;
        Buffer_<float> w2_scales;
        Tensor         fc1_input_scale;
        bool           prepared{false};
    };

    struct TrtllmFp8BlockScaleLayerCache {
        Tensor fc1_weights;
        Tensor fc2_weights;
        Tensor fc1_scales;
        Tensor fc2_scales;
        bool   prepared{false};
    };

    struct TrtllmFp4LayerCache {
        Tensor         fc1_weights;
        Tensor         fc2_weights;
        Tensor         fc1_scales;
        Tensor         fc2_scales;
        Buffer_<float> w1_scale_2;
        Buffer_<float> w3_scale_2;
        Buffer_<float> w2_scale_2;
        float          fc1_input_scale{1.f};
        bool           prepared{false};
    };

    struct TrtllmBf16LayerCache {
        Tensor fc1_weights;
        Tensor fc2_weights;
        bool   prepared{false};
    };

    TrtllmFp8LayerCache&           PrepareFp8Scales(MoeFfnWeight& moe, int layer_id);
    TrtllmFp8BlockScaleLayerCache& PrepareFp8BlockScale(MoeFfnWeight& moe, int layer_id);
    TrtllmFp4LayerCache&           PrepareFp4Scales(MoeFfnWeight& moe, int layer_id);
    TrtllmBf16LayerCache&          PrepareBf16Weights(MoeFfnWeight& moe, int layer_id);

    void ReleaseOriginalExpertWeights(MoeFfnWeight& moe, int layer_id, const char* backend_path);

    Tensor RoutedOutput(Tensor output, int num_tokens);

    const int      inter_size_;
    const int      hidden_dim_;
    const MoeParam param_;
    const bool     moe_quant_w4a8_nvfp4_fp8_;
    const bool     enable_pdl_;

    std::vector<TrtllmFp8LayerCache>           fp8_layer_cache_;
    std::vector<TrtllmFp8BlockScaleLayerCache> fp8_blockscale_layer_cache_;
    std::vector<TrtllmFp4LayerCache>           fp4_layer_cache_;
    std::vector<TrtllmBf16LayerCache>          bf16_layer_cache_;

    Tensor gate_logits_buf_;
    Tensor temp_buf_;
    Tensor temp_;

    Tensor                          trtllm_fp8_hidden_states_;
    Tensor                          trtllm_fp8_hidden_scales_;
    std::unordered_map<int, Tensor> trtllm_fp8_block_scales_per_n_;
    Tensor                          trtllm_fp4_hidden_states_;
    Tensor                          trtllm_fp4_hidden_scales_;
    Tensor                          trtllm_fp4_topk_ids_;
    Tensor                          trtllm_fp4_expert_weights_;

    std::unique_ptr<TrtllmFusedMoeHelper> w4a8_helper_;
};

}  // namespace turbomind
