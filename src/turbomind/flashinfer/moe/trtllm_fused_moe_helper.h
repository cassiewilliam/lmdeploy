// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class TrtllmFusedMoeHelper {
public:
    TrtllmFusedMoeHelper(int layer_num, int inter_size, int hidden_dim);

    void PrepareWeights(const MoeFfnWeight& moe, int layer_id, cudaStream_t stream);

    bool Dispatch(const MoeFfnWeight&   moe,
                  const MoeParam&       param,
                  const Tensor&         input,
                  Tensor&               output,
                  const Buffer_<int>&   masks,
                  const Buffer_<float>& scales,
                  int                   num_tokens,
                  int                   tokens_padded,
                  int                   layer_id,
                  bool                  enable_pdl,
                  cudaStream_t          stream);

private:
    struct LayerCache {
        Tensor         fc1_weights;
        Tensor         fc2_weights;
        Tensor         fc1_scale_blocks;
        Tensor         fc2_scale_blocks;
        Buffer_<float> fc1_global_scales;
        Buffer_<float> fc2_act_global_scales;
        Buffer_<float> fc2_global_scales;
        Tensor         fc1_input_scale;
        bool           prepared{false};
    };

    LayerCache& PrepareScales(const MoeFfnWeight& moe, int layer_id, cudaStream_t stream);

    const int inter_size_;
    const int hidden_dim_;

    std::vector<LayerCache> layer_cache_;

    Tensor hidden_states_;
    Tensor hidden_scales_;
    Tensor topk_ids_;
    Tensor expert_weights_;
};

}  // namespace turbomind
