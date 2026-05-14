// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <unordered_map>

#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/trtllm_fused_moe_backend.h"
#include "src/turbomind/models/moe_weight.h"

namespace turbomind {

class MoeFfnLayer {
public:
    MoeFfnLayer(const EngineParam& engine, const Context& ctx);

    struct ForwardParam {
        Tensor           input;
        Tensor           output;
        const MoeWeight* weights;
        float            scale;
        int              layer_id;
    };

    void Forward(ForwardParam& p);

    void Combine(ForwardParam& p);

private:
    void Init(ForwardParam& p);

    Tensor_<float> Gate(const Tensor& input, const LinearWeight& gate);

    void dump_logits(int token_num, int layer_id, int expert_num);

    const int tp_size_;
    const int max_token_num_;
    int&      is_warm_up_;

    LlamaLinear& linear_;

    std::unique_ptr<LlamaFfnLayer> expert_ffn_;

    bool initialized_ = false;

    ///////////////////////////////////////////////////////
    /// runtime states
    Buffer_<int> h_offsets_;

    Buffer_<int>   masks_;
    Buffer_<int>   f2n_;
    Buffer_<int>   f2E_;
    Buffer_<int>   en2f_;
    Buffer_<float> scales_;
    Buffer_<int>   accum_;
    Buffer_<int>   offsets_;

    Tensor         temp_;
    Tensor_<float> shared_scales_;
    ///////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////
    /// trtllm fused MoE dispatch
    EngineParam                            engine_param_;
    std::unique_ptr<TrtllmFusedMoeBackend> trtllm_backend_;
    std::unordered_map<int, MoeFfnWeight>  trtllm_weight_cache_;  // keyed by layer_id
    bool                                   trtllm_handled_{false};  // trtllm fully handled this forward step
    ///////////////////////////////////////////////////////
};

}  // namespace turbomind
