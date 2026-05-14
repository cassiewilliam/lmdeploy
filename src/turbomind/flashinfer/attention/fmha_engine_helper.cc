// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/flashinfer/attention/fmha_engine_helper.h"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "src/turbomind/core/check.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::flashinfer_attention {

namespace {

int count_cache_layers(const ModelParam& model_param)
{
    int cache_layer_num = static_cast<int>(model_param.layer_num);
    for (const auto type : model_param.layer_types) {
        if (type == 1) {
            --cache_layer_num;
        }
    }
    return cache_layer_num;
}

size_t native_page_block_size(const ModelParam& model_param,
                              DataType          runtime_dtype,
                              int               cache_block_seq_len,
                              int               attn_tp_size)
{
    const int cache_layer_num = count_cache_layers(model_param);
    const int local_kv_heads  = static_cast<int>(model_param.kv_head_num) / attn_tp_size;
    const int dbits           = static_cast<int>(byte_size(runtime_dtype, 8));
    const int elem_bits       = KvCacheElemBits(model_param.quant_policy, dbits);

    TM_CHECK_GT(cache_layer_num, 0);
    TM_CHECK_GT(local_kv_heads, 0);
    TM_CHECK_GT(cache_block_seq_len, 0);
    TM_CHECK_GT(elem_bits, 0);

    const int64_t token_bits = static_cast<int64_t>(model_param.head_dim) * elem_bits;
    TM_CHECK_EQ(token_bits % 8, 0);

    const int64_t head_data_bytes = static_cast<int64_t>(cache_block_seq_len) * token_bits / 8;
    const int64_t block_size      = 2LL * cache_layer_num * local_kv_heads * head_data_bytes;
    TM_CHECK_GT(block_size, 0);
    return static_cast<size_t>(block_size);
}

}  // namespace

bool FmhaEngineHelper::ShouldUse(AttentionBackend backend)
{
    return backend == AttentionBackend::kTrtllmFmha && isSM10x();
}

FmhaEngineHelper::FmhaEngineHelper(int max_batch_size, int max_pages_cap): max_pages_cap_{max_pages_cap}
{
    TM_CHECK_GT(max_batch_size, 0);
    TM_CHECK_GT(max_pages_cap_, 0);

    const ssize_t page_table_elems = static_cast<ssize_t>(max_batch_size) * 2 * max_pages_cap_;
    page_tables_host_             = {page_table_elems, kCPUpinned};
    page_tables_dev_              = {page_table_elems, kDEVICE};
    seq_lens_kv_host_             = {max_batch_size, kCPUpinned};
    seq_lens_kv_dev_              = {max_batch_size, kDEVICE};
    max_pages_buf_                = {1, kCPUpinned};
    pool_base_buf_                = {1, kCPUpinned};
    page_nums_buf_                = {1, kCPUpinned};
    stride_layer_buf_             = {1, kCPUpinned};
    kv_value_offset_buf_          = {1, kCPUpinned};
    kv_page_stride_buf_           = {1, kCPUpinned};
    kv_head_stride_buf_           = {1, kCPUpinned};
}

auto FmhaEngineHelper::ResolvePool(GetBlockPtr       get_block_ptr,
                                   int               total_block_count,
                                   const ModelParam& model_param,
                                   DataType          runtime_dtype,
                                   int               cache_block_seq_len,
                                   int               attn_tp_size) -> PoolInfo
{
    const size_t block_size =
        native_page_block_size(model_param, runtime_dtype, cache_block_seq_len, attn_tp_size);

    if (pool_info_.is_valid && pool_info_.block_size == block_size
        && pool_total_block_count_ == total_block_count) {
        return pool_info_;
    }

    pool_info_              = {};
    pool_total_block_count_ = total_block_count;

    if (total_block_count <= 0 || !get_block_ptr) {
        return pool_info_;
    }

    auto* const base = static_cast<std::byte*>(get_block_ptr(0));
    if (!base) {
        return pool_info_;
    }

    for (int block_id = 1; block_id < total_block_count; ++block_id) {
        auto* const block_ptr = static_cast<std::byte*>(get_block_ptr(block_id));
        const auto  offset =
            static_cast<std::ptrdiff_t>(block_id) * static_cast<std::ptrdiff_t>(block_size);
        if (block_ptr != base + offset) {
            return pool_info_;
        }
    }

    pool_info_.base       = base;
    pool_info_.block_size = block_size;
    pool_info_.num_blocks = total_block_count;
    pool_info_.is_valid   = true;
    return pool_info_;
}

void FmhaEngineHelper::Publish(TensorMap&                       env,
                               const std::vector<SequenceInfo>& sequences,
                               GetBlockPtr                      get_block_ptr,
                               int                              total_block_count,
                               const ModelParam&                model_param,
                               DataType                         runtime_dtype,
                               int                              cache_block_seq_len,
                               int                              attn_tp_size,
                               cudaStream_t                     stream)
{
    const auto pool = ResolvePool(
        std::move(get_block_ptr), total_block_count, model_param, runtime_dtype, cache_block_seq_len, attn_tp_size);

    if (!pool.is_valid || sequences.empty()) {
        return;
    }

    // Fixed worst-case stride: the page_tables layout MUST match what CG
    // captured (PublishConstants set max_pages_per_seq = max_pages_cap_).
    // PopulateExtensionFields computes element addresses as
    //   page_tables + batch_offset * 2 * env.max_num_pages_per_seq_kv
    // so the stride per batch row is baked into every captured kernel.
    // If runtime shrinks `table_pages` below max_pages_cap_, the captured
    // graph reads from the wrong row.  Always use max_pages_cap_; the
    // unused trailing slots stay zero-padded (no semantic effect — kernel
    // bounds it by seq_lens_kv anyway).
    const int table_pages = std::max(max_pages_cap_, 1);

    auto* table    = page_tables_host_.data();
    auto* seq_lens = seq_lens_kv_host_.data();
    const int batch_size = static_cast<int>(sequences.size());
    std::fill_n(table, static_cast<ssize_t>(batch_size) * 2 * table_pages, 0);

    for (int b = 0; b < batch_size; ++b) {
        const auto& seq  = sequences[b];
        const int   nblk = std::min(seq.block_count, table_pages);
        for (int i = 0; i < nblk; ++i) {
            const int block_id = seq.block_ids[i];
            table[b * 2 * table_pages + i]               = block_id;
            table[b * 2 * table_pages + table_pages + i] = block_id;
        }
        seq_lens[b] = seq.kv_len;
    }

    const ssize_t copy_elems = static_cast<ssize_t>(batch_size) * 2 * table_pages;
    check_cuda_error(cudaMemcpyAsync(page_tables_dev_.data(),
                                     page_tables_host_.data(),
                                     copy_elems * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream));
    check_cuda_error(cudaMemcpyAsync(seq_lens_kv_dev_.data(),
                                     seq_lens_kv_host_.data(),
                                     batch_size * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream));

    *max_pages_buf_.data() = table_pages;
    *pool_base_buf_.data() = pool.base;
    *page_nums_buf_.data() = pool.num_blocks;

    const int cache_layer_num = count_cache_layers(model_param);
    const int local_kv_heads  = static_cast<int>(model_param.kv_head_num) / attn_tp_size;
    TM_CHECK_GT(cache_layer_num, 0);
    TM_CHECK_GT(local_kv_heads, 0);

    const int64_t layer_stride_bytes = static_cast<int64_t>(pool.block_size) / cache_layer_num;
    const int64_t head_data_bytes    = layer_stride_bytes / (2 * local_kv_heads);
    TM_CHECK_EQ(layer_stride_bytes * cache_layer_num, static_cast<int64_t>(pool.block_size));
    TM_CHECK_EQ(head_data_bytes * 2 * local_kv_heads, layer_stride_bytes);

    *stride_layer_buf_.data()    = layer_stride_bytes;
    *kv_value_offset_buf_.data() = head_data_bytes;
    *kv_page_stride_buf_.data()  = static_cast<int64_t>(pool.block_size);
    *kv_head_stride_buf_.data()  = 2 * head_data_bytes;

    env.emplace("fmha_page_tables", page_tables_dev_);
    env.emplace("fmha_seq_lens_kv", seq_lens_kv_dev_);
    env.emplace("fmha_max_pages_per_seq", max_pages_buf_);
    env.emplace("fmha_kv_cache_pool", pool_base_buf_);
    env.emplace("fmha_page_nums", page_nums_buf_);
    env.emplace("fmha_stride_layer_bytes", stride_layer_buf_);
    env.emplace("fmha_kv_cache_value_offset_bytes", kv_value_offset_buf_);
    env.emplace("fmha_kv_cache_page_stride_bytes", kv_page_stride_buf_);
    env.emplace("fmha_kv_cache_head_stride_bytes", kv_head_stride_buf_);
}

void FmhaEngineHelper::PublishConstants(TensorMap&        env,
                                        GetBlockPtr       get_block_ptr,
                                        int               total_block_count,
                                        const ModelParam& model_param,
                                        DataType          runtime_dtype,
                                        int               cache_block_seq_len,
                                        int               attn_tp_size)
{
    const auto pool = ResolvePool(
        std::move(get_block_ptr), total_block_count, model_param, runtime_dtype, cache_block_seq_len, attn_tp_size);
    if (!pool.is_valid) {
        return;
    }

    *pool_base_buf_.data() = pool.base;
    *page_nums_buf_.data() = pool.num_blocks;

    const int cache_layer_num = count_cache_layers(model_param);
    const int local_kv_heads  = static_cast<int>(model_param.kv_head_num) / attn_tp_size;
    TM_CHECK_GT(cache_layer_num, 0);
    TM_CHECK_GT(local_kv_heads, 0);

    const int64_t layer_stride_bytes = static_cast<int64_t>(pool.block_size) / cache_layer_num;
    const int64_t head_data_bytes    = layer_stride_bytes / (2 * local_kv_heads);
    TM_CHECK_EQ(layer_stride_bytes * cache_layer_num, static_cast<int64_t>(pool.block_size));
    TM_CHECK_EQ(head_data_bytes * 2 * local_kv_heads, layer_stride_bytes);

    *stride_layer_buf_.data()    = layer_stride_bytes;
    *kv_value_offset_buf_.data() = head_data_bytes;
    *kv_page_stride_buf_.data()  = static_cast<int64_t>(pool.block_size);
    *kv_head_stride_buf_.data()  = 2 * head_data_bytes;

    // Cap max_pages_per_seq at the engine-known maximum so CG warmup
    // sees a non-zero value (per-request Publish refines this from the
    // observed sequence's block_count).
    *max_pages_buf_.data() = std::max(max_pages_cap_, 1);

    env.emplace("fmha_page_tables", page_tables_dev_);
    env.emplace("fmha_seq_lens_kv", seq_lens_kv_dev_);
    env.emplace("fmha_max_pages_per_seq", max_pages_buf_);
    env.emplace("fmha_kv_cache_pool", pool_base_buf_);
    env.emplace("fmha_page_nums", page_nums_buf_);
    env.emplace("fmha_stride_layer_bytes", stride_layer_buf_);
    env.emplace("fmha_kv_cache_value_offset_bytes", kv_value_offset_buf_);
    env.emplace("fmha_kv_cache_page_stride_bytes", kv_page_stride_buf_);
    env.emplace("fmha_kv_cache_head_stride_bytes", kv_head_stride_buf_);
}

}  // namespace turbomind::flashinfer_attention
