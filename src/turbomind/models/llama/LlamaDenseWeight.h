// Copyright (c) OpenMMLab. All rights reserved.
//
// Compatibility layer: provides ModelParam, MoeParam, LlamaDenseWeight, and
// MoeFfnWeight types that map to the current open-source LinearWeight /
// MoeWeight / EngineParam API used by trtllm_fused_moe_backend.cc.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/moe_weight.h"

namespace turbomind {

// ---------------------------------------------------------------------------
// ModelParam — subset of ModelWeight fields needed by TrtllmFusedMoeBackend
// ---------------------------------------------------------------------------
struct ModelParam {
    int         hidden_units{};
    int         layer_num{};
    std::string moe_quant_algo;  // e.g. "w4a8_nvfp4_fp8", empty = none
};

// ---------------------------------------------------------------------------
// MoeParam — routing/expert layout params needed by TrtllmFusedMoeBackend
// ---------------------------------------------------------------------------
struct MoeParam {
    enum Method { kDefault = 0, kFused = 1 };

    int         inter_size{};
    std::vector<int> expert_num;  // per-layer expert counts
    int         experts_per_token{};
    int         n_group{};
    int         topk_group{};
    std::string topk_method;
    bool        norm_topk_prob{};
    float       routed_scale{};
    std::string scoring_func;
    Method      method{kDefault};
    bool        w4a8_nvfp4_fp8{};
};

// ---------------------------------------------------------------------------
// LlamaDenseWeight — thin view over a LinearWeight's Tensor fields.
//
// Holds non-owning copies of the Tensors (ref-counted, share GPU memory).
// ReleaseDensePayload() clears both this view AND the original LinearWeight
// so the GPU memory is actually freed.
// ---------------------------------------------------------------------------
struct LlamaDenseWeight {
    Tensor weight;
    Tensor bias;
    Tensor scales;
    Tensor zeros;
    Tensor scale_2;       // NVFP4 second-level scalar scale
    Tensor input_scales;  // per-layer activation quantization scale

    DataType weight_type{};
    int      group_size{1};

    // Back-pointer so ReleaseDensePayload can clear the original.
    LinearWeight* src{nullptr};

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(weight);
    }

    // Populate from a LinearWeight; reads block_sizes[0] for group_size.
    static LlamaDenseWeight from(LinearWeight* lw)
    {
        if (!lw) {
            return {};
        }
        LlamaDenseWeight d;
        d.weight       = lw->weight;
        d.bias         = lw->bias;
        d.scales       = lw->scales;
        d.zeros        = lw->zeros;
        d.scale_2      = lw->scale_2;
        d.input_scales = lw->input_scales;
        d.weight_type  = lw->weight_format.dtype;
        d.group_size   = lw->weight_format.block_sizes.empty()
                             ? 1
                             : lw->weight_format.block_sizes[0];
        d.src          = lw;
        return d;
    }
};

// Release GPU tensors from this view AND from the original LinearWeight.
inline void ReleaseDensePayload(LlamaDenseWeight& d)
{
    if (d.src) {
        d.src->weight       = {};
        d.src->scales       = {};
        d.src->scale_2      = {};
        d.src->zeros        = {};
        d.src->input_scales = {};
    }
    d.weight       = {};
    d.scales       = {};
    d.scale_2      = {};
    d.zeros        = {};
    d.input_scales = {};
}

// ---------------------------------------------------------------------------
// MoeFfnBlock — fused-weight block (fc1=w1w3, fc2=w2)
// ---------------------------------------------------------------------------
struct MoeFfnBlock {
    LlamaDenseWeight fused_gating_intermediate;  // w1w3
    LlamaDenseWeight output;                      // w2
};

// ---------------------------------------------------------------------------
// MoeFfnExpert — per-expert weights (back-pointed to FfnWeight)
// ---------------------------------------------------------------------------
struct MoeFfnExpert {
    LlamaDenseWeight fused_gating_intermediate;
    LlamaDenseWeight output;
    FfnWeight*       src{nullptr};
};

// ---------------------------------------------------------------------------
// MoeFfnWeight — mirrors internal-fork MoeFfnWeight structure.
// Populated from MoeWeight via MoeFfnWeight::from().
// ---------------------------------------------------------------------------
struct MoeFfnWeight {
    MoeFfnBlock                               block;
    std::vector<std::unique_ptr<MoeFfnExpert>> experts;
    Tensor                                    score_correction_bias;
    // Weight tensor from MoeWeight::shared_gate; non-null means a shared expert exists.
    Tensor                                    shared_gate_weight;
    MoeParam::Method                          method{MoeParam::kDefault};

    // Build from current MoeWeight.  Caller owns `moe`; this must not outlive it.
    static MoeFfnWeight from(MoeWeight& moe, MoeParam::Method method = MoeParam::kFused)
    {
        MoeFfnWeight w;
        w.method = method;

        FfnWeight* blk = moe.block();
        if (blk) {
            w.block.fused_gating_intermediate = LlamaDenseWeight::from(blk->w1w3.get());
            w.block.output                    = LlamaDenseWeight::from(blk->w2.get());
        }

        const int E = moe.num_experts();
        w.experts.reserve(E);
        for (int i = 0; i < E; ++i) {
            auto       e      = std::make_unique<MoeFfnExpert>();
            FfnWeight* expert = moe.expert(i);
            if (expert) {
                e->fused_gating_intermediate = LlamaDenseWeight::from(expert->w1w3.get());
                e->output                    = LlamaDenseWeight::from(expert->w2.get());
                e->src                       = expert;
            }
            w.experts.push_back(std::move(e));
        }

        if (moe.score_correction_bias) {
            w.score_correction_bias = moe.score_correction_bias;
        }

        // Capture just the weight tensor of shared_gate (LinearWeight is non-copyable).
        if (moe.shared_gate && moe.shared_gate->weight) {
            w.shared_gate_weight = moe.shared_gate->weight;
        }
        return w;
    }
};

}  // namespace turbomind
