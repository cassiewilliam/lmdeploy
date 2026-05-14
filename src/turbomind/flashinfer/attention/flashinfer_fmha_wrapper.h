// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace turbomind::flashinfer_fmha {

enum class DType : int {
    kFP16 = 0,
    kBF16 = 1,
    kFP8E4M3 = 2,
    kFP4E2M1 = 3,
};

struct FmhaParams {
    void*  out;
    void*  query;
    void*  key_cache;
    void*  value_cache;
    void*  workspace_buffer;
    int*   block_tables;
    int*   seq_lens;
    int*   cum_seq_lens_q;
    int*   cum_seq_lens_kv;

    DType  q_dtype;
    DType  kv_dtype;
    DType  o_dtype;

    int    batch_size;
    int    max_q_len;
    int    max_kv_len;
    int    num_qo_heads;
    int    num_kv_heads;
    int    head_dim_qk;
    int    head_dim_vo;
    int    page_size;
    int    max_num_blocks_per_seq;
    int    num_pages_in_pool;

    int64_t kv_stride_keys_values;
    int64_t kv_stride_heads;
    int64_t kv_stride_batch;

    double bmm1_scale;
    double bmm2_scale;
    float* bmm1_scale_log2_ptr;
    float* bmm2_scale_ptr;
    void*  key_block_scales;
    void*  value_block_scales;
    int64_t kv_scale_stride_heads;
    int64_t kv_scale_stride_batch;

    int    window_left;
    int    sm_count;
    bool   enable_pdl;  // forwarded to flashinfer trtllm-gen FMHA launcher
    int    workspace_size;

    cudaStream_t stream;
};

bool dispatch_decode(const FmhaParams& p);
bool dispatch_context(const FmhaParams& p);

void initialize();
bool is_available();

}  // namespace turbomind::flashinfer_fmha
