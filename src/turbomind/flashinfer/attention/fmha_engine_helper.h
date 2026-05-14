// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind::flashinfer_attention {

class FmhaEngineHelper {
public:
    struct SequenceInfo {
        const int* block_ids{nullptr};
        int        block_count{0};
        int        kv_len{0};
    };

    using GetBlockPtr = std::function<void*(int)>;

    static bool ShouldUse(AttentionBackend backend);

    FmhaEngineHelper(int max_batch_size, int max_pages_cap);

    void Publish(TensorMap&                      env,
                 const std::vector<SequenceInfo>& sequences,
                 GetBlockPtr                     get_block_ptr,
                 int                             total_block_count,
                 const ModelParam&               model_param,
                 DataType                        runtime_dtype,
                 int                             cache_block_seq_len,
                 int                             attn_tp_size,
                 cudaStream_t                    stream);

    // Engine-init publish: same as Publish() but skips the per-request
    // sequence data (page_tables_dev_, seq_lens_kv_dev_).  Populates the
    // model-derived constants (kv_cache_pool, page_nums, layer/head/page
    // strides) into the engine TensorMap so downstream wrappers can
    // ingest them BEFORE the first request arrives — needed by CUDA-Graph
    // warmup which captures attention kernels but has no real batch yet.
    void PublishConstants(TensorMap&        env,
                          GetBlockPtr       get_block_ptr,
                          int               total_block_count,
                          const ModelParam& model_param,
                          DataType          runtime_dtype,
                          int               cache_block_seq_len,
                          int               attn_tp_size);

private:
    struct PoolInfo {
        void*  base{nullptr};
        size_t block_size{0};
        int    num_blocks{0};
        bool   is_valid{false};
    };

    PoolInfo ResolvePool(GetBlockPtr       get_block_ptr,
                         int               total_block_count,
                         const ModelParam& model_param,
                         DataType          runtime_dtype,
                         int               cache_block_seq_len,
                         int               attn_tp_size);

    Buffer_<int>     page_tables_host_;
    Buffer_<int>     page_tables_dev_;
    Buffer_<int>     seq_lens_kv_host_;
    Buffer_<int>     seq_lens_kv_dev_;
    Buffer_<int>     max_pages_buf_;
    Buffer_<void*>   pool_base_buf_;
    Buffer_<int>     page_nums_buf_;
    Buffer_<int64_t> stride_layer_buf_;
    Buffer_<int64_t> kv_value_offset_buf_;
    Buffer_<int64_t> kv_page_stride_buf_;
    Buffer_<int64_t> kv_head_stride_buf_;
    int              max_pages_cap_{0};
    PoolInfo         pool_info_;
    int              pool_total_block_count_{0};
};

}  // namespace turbomind::flashinfer_attention
