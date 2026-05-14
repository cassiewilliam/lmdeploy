// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <climits>
#include <cstdint>
#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
struct TrtllmFmhaParams {
    void* kv_cache_pool_ptr = nullptr;
    int   page_nums         = 0;

    const int* page_tables              = nullptr;
    const int* seq_lens_kv              = nullptr;
    int        max_num_pages_per_seq_kv = 0;

    void*   kv_cache_value_pool_ptr    = nullptr;
    int64_t kv_cache_page_stride_bytes = 0;
    int64_t kv_cache_head_stride_bytes = 0;

    T* raw_q = nullptr;
    T* raw_k = nullptr;
    T* raw_v = nullptr;

    const int* token2batch = nullptr;
    const int* kv_len = nullptr;

    int layer_idx = 0;
    int layer_num = 0;

    float* partial_M = nullptr;
    float* partial_L = nullptr;
    int*   locks     = nullptr;

    float  fp8_qscale        = 0.0f;
    float  host_bmm1_scale   = 1.0f;
    float* qkv_scale_orig    = nullptr;
    float* o_scale_orig      = nullptr;
    int*   fmha_tile_counter = nullptr;

    bool        is_qk_norm  = false;
    float       qk_norm_eps = 0.0f;
    const void* q_weight    = nullptr;
    const void* k_weight    = nullptr;

    int sum_q_len  = 0;
    int sum_kv_len = 0;

    float* attention_sinks_ptr = nullptr;

    bool    use_sparse_mla       = false;
    float2* softmax_stats_Ptr    = nullptr;
    bool    use_sparse_attention = false;

    int max_attention_window_size    = 40961;
    int cyclic_attention_window_size = 40961;
    int chunked_attention_size       = INT_MAX;

    float* scale_bmm1_ptr = nullptr;
    float* scale_bmm2_ptr = nullptr;
    void*  key_block_scales = nullptr;
    void*  value_block_scales = nullptr;
    int64_t kv_scale_stride_heads = 0;
    int64_t kv_scale_stride_batch = 0;
    float  q_scaling      = 1.0f;
    float* osf_scale_Ptr  = nullptr;
    void*  output_sf_ptr  = nullptr;

    const float* fp4_out_sf_scale   = nullptr;
    int          start_token_idx_sf = 0;

    bool multi_block_mode   = false;
    bool multi_query_tokens = false;
    bool is_spec_dec_tree   = true;

    void*   workspace_buffer = nullptr;
    int64_t workspace_size   = 0;

    bool enable_pdl                   = false;
    int  sm_count                     = 0;
    bool enable_prefill_fp8_attention = false;
};

}  // namespace turbomind
