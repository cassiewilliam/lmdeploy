/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.cc

#include <algorithm>
#include <math.h>
#include <vector>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/macro.h"

#include "src/turbomind/flashinfer/attention/flashinfer_fmha_wrapper.h"
#include "src/turbomind/flashinfer/attention/kv_cache_utils_v3.h"
#include "src/turbomind/flashinfer/attention/trtllm_fmha_params.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_rope.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/mla_utils.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

// #include "dbg.h"

namespace turbomind {

struct AttentionData {
    struct Stat {
        int n;
        int q_sum;
        int q_max;
        int k_sum;
        int k_max;
    } decode, prefill;

    Buffer_<void*> block_ptrs;
    Buffer_<int>   block_ptrs_offsets;

    Buffer_<float> rope_base;

    Tensor_<int> mrope_position_ids;
    Buffer_<int> mrope_position_delta;
    Buffer_<int> mrope_length;

    // borrowed from env
    Buffer_<bool> finished;
    Buffer_<int>  q_offsets;
    Buffer_<int>  k_offsets;

    // FMHA per-phase state (zero-size when FMHA is disabled)
    Buffer_<int> fmha_page_tables;
    Buffer_<int> fmha_seq_lens_kv;
    Buffer_<int> fmha_token2batch;   // [2*total_tokens]: (batch_idx, token_idx) pairs
    Buffer_<int> fmha_pf_cu_q_len;  // [prefill_count+1] cumulative Q lengths
    Buffer_<int> fmha_pf_cu_k_len;  // [prefill_count+1] cumulative K lengths
    Tensor       fmha_processed_q;  // [max_tokens, head_num*head_dim] pre-processed Q

    // Cached engine layout from fmha_engine_helper
    void*    fmha_kv_pool_ptr{nullptr};
    int      fmha_page_nums{0};
    int64_t  fmha_stride_layer_bytes{0};
    int64_t  fmha_kv_value_offset{0};
    int64_t  fmha_kv_page_stride{0};
    int64_t  fmha_kv_head_stride{0};
    int      fmha_max_pages_per_seq{0};
    int      fmha_total_tokens{0};

    // int dbg_offset;
    // int dbg_size;
};

namespace {

// ---------------------------------------------------------------------------
// FMHA helper functions (inlined from TrtllmAttentionWrapper)
// ---------------------------------------------------------------------------

inline void fmha_sync_stream(cudaStream_t stream)
{
    cudaStreamCaptureStatus st{cudaStreamCaptureStatusNone};
    if (cudaStreamIsCapturing(stream, &st) == cudaSuccess && st != cudaStreamCaptureStatusNone) {
        return;
    }
    check_cuda_error(cudaStreamSynchronize(stream));
}

inline int64_t fmha_decode_workspace_size(int batch_size)
{
    constexpr int64_t kMin = 128LL * 1024 * 1024;
    constexpr int64_t kPer = 8LL  * 1024 * 1024;
    constexpr int64_t kMax = 1024LL * 1024 * 1024;
    return std::min(std::max(static_cast<int64_t>(std::max(batch_size, 1)) * kPer, kMin), kMax);
}

template<class T>
constexpr ::turbomind::flashinfer_fmha::DType fmha_model_dtype()
{
    if constexpr (std::is_same_v<T, half>) return ::turbomind::flashinfer_fmha::DType::kFP16;
    else                                   return ::turbomind::flashinfer_fmha::DType::kBF16;
}

inline int fmha_dtype_nbytes(::turbomind::flashinfer_fmha::DType d)
{
    using ::turbomind::flashinfer_fmha::DType;
    if (d == DType::kFP8E4M3 || d == DType::kFP4E2M1) return 1;
    return 2;
}

inline int64_t fmha_stride_elems(::turbomind::flashinfer_fmha::DType d, int64_t bytes)
{
    using ::turbomind::flashinfer_fmha::DType;
    if (d == DType::kFP4E2M1) return bytes * 2;
    return bytes / fmha_dtype_nbytes(d);
}

template<class T>
void fmha_resolve_dtypes(int quant_policy,
                         bool fuse_out,
                         ::turbomind::flashinfer_fmha::DType& q,
                         ::turbomind::flashinfer_fmha::DType& kv,
                         ::turbomind::flashinfer_fmha::DType& o)
{
    using ::turbomind::flashinfer_fmha::DType;
    q = kv = o = fmha_model_dtype<T>();
    if (quant_policy == 0) return;
    if (IsCacheKVFP4(quant_policy)) {
        q = DType::kFP8E4M3; kv = DType::kFP4E2M1;
        if (fuse_out) o = DType::kFP8E4M3;
    } else if (IsCacheKVFP8(quant_policy)) {
        q = kv = DType::kFP8E4M3;
        if (fuse_out) o = DType::kFP8E4M3;
    }
}

template<class T>
::turbomind::flashinfer_fmha::FmhaParams fmha_build_params(
    const AttentionParams<T>& p,
    const AttentionData&      d,
    int                       batch_offset,
    int                       sm_count,
    bool                      enable_pdl)
{
    using ::turbomind::flashinfer_fmha::FmhaParams;
    using ::turbomind::flashinfer_fmha::DType;

    FmhaParams fp{};
    fp.out   = p.out;
    fp.query = p.q;

    const int64_t layer_offset = static_cast<int64_t>(p.layer_id) * d.fmha_stride_layer_bytes;
    auto* layer_base = reinterpret_cast<char*>(d.fmha_kv_pool_ptr) + layer_offset;
    fp.key_cache   = layer_base;
    fp.value_cache = layer_base + d.fmha_kv_value_offset;

    fp.workspace_buffer  = p.fmha.workspace_buffer;
    fp.block_tables      = const_cast<int*>(p.fmha.page_tables);
    fp.seq_lens          = const_cast<int*>(p.fmha.seq_lens_kv);
    fp.cum_seq_lens_q    = const_cast<int*>(reinterpret_cast<const int*>(p.cu_q_len));
    fp.cum_seq_lens_kv   = const_cast<int*>(reinterpret_cast<const int*>(p.cu_k_len));

    fmha_resolve_dtypes<T>(p.quant_policy, p.fuse_attention_quant, fp.q_dtype, fp.kv_dtype, fp.o_dtype);

    fp.batch_size             = p.batch_size;
    fp.max_q_len              = p.max_q_len;
    fp.max_kv_len             = p.max_k_len;
    fp.num_qo_heads           = p.num_heads;
    fp.num_kv_heads           = p.num_kv_heads;
    fp.head_dim_qk            = p.size_per_head;
    fp.head_dim_vo            = p.size_per_head;
    fp.page_size              = p.block_iter_params.block_len;
    fp.max_num_blocks_per_seq = p.fmha.max_num_pages_per_seq_kv;
    fp.num_pages_in_pool      = p.fmha.page_nums;

    fp.kv_stride_keys_values = p.size_per_head;
    fp.kv_stride_heads       = d.fmha_kv_head_stride > 0
                                   ? fmha_stride_elems(fp.kv_dtype, d.fmha_kv_head_stride)
                                   : static_cast<int64_t>(p.block_iter_params.block_len) * p.size_per_head;
    fp.kv_stride_batch       = d.fmha_kv_page_stride > 0
                                   ? fmha_stride_elems(fp.kv_dtype, d.fmha_kv_page_stride)
                                   : static_cast<int64_t>(p.num_kv_heads) * p.block_iter_params.block_len * p.size_per_head;

    const double host_scale = static_cast<double>(p.fmha.host_bmm1_scale);
    fp.bmm1_scale = (host_scale > 0.0 && host_scale != 1.0)
                        ? host_scale
                        : 1.0 / std::sqrt(static_cast<double>(p.size_per_head));
    fp.bmm2_scale         = 1.0;
    const bool is_fp_kv   = IsCacheKVFP(p.quant_policy);
    fp.bmm1_scale_log2_ptr = (is_fp_kv && p.fmha.scale_bmm1_ptr) ? p.fmha.scale_bmm1_ptr + 1 : nullptr;
    fp.bmm2_scale_ptr      = is_fp_kv ? p.fmha.scale_bmm2_ptr : nullptr;
    fp.key_block_scales    = p.fmha.key_block_scales;
    fp.value_block_scales  = p.fmha.value_block_scales;
    fp.kv_scale_stride_heads = p.fmha.kv_scale_stride_heads;
    fp.kv_scale_stride_batch = p.fmha.kv_scale_stride_batch;

    fp.window_left    = (p.fmha.cyclic_attention_window_size > 0)
                            ? p.fmha.cyclic_attention_window_size - 1 : -1;
    fp.sm_count       = sm_count;
    fp.enable_pdl     = enable_pdl;
    fp.workspace_size = static_cast<int>(p.fmha.workspace_size);
    fp.stream         = p.stream;

    return fp;
}

template<class T>
void fmha_populate_ext(AttentionParams<T>&  p,
                       const AttentionData& d,
                       int                  batch_offset,
                       Tensor_<float>&      scale_bmm1,
                       Tensor_<float>&      scale_bmm2,
                       Tensor_<float>&      fp8_scale_orig,
                       Tensor_<int8_t>&     fp4_k_scales,
                       Tensor_<int8_t>&     fp4_v_scales,
                       int64_t              fp4_scale_stride)
{
    p.fmha.kv_cache_pool_ptr          = nullptr;  // computed in fmha_build_params via layer_offset
    p.fmha.kv_cache_value_pool_ptr    = nullptr;
    p.fmha.page_nums                  = d.fmha_page_nums;
    p.fmha.kv_cache_page_stride_bytes = d.fmha_kv_page_stride;
    p.fmha.kv_cache_head_stride_bytes = d.fmha_kv_head_stride;
    p.fmha.page_tables                = d.fmha_page_tables.data() + batch_offset * 2 * d.fmha_max_pages_per_seq;
    p.fmha.seq_lens_kv                = d.fmha_seq_lens_kv.data() ? d.fmha_seq_lens_kv.data() + batch_offset : nullptr;
    p.fmha.max_num_pages_per_seq_kv   = d.fmha_max_pages_per_seq;
    p.fmha.layer_idx                  = p.layer_id;
    p.fmha.kv_len                     = p.fmha.seq_lens_kv;
    p.fmha.fmha_tile_counter          = nullptr;  // not used in simplified path
    p.fmha.scale_bmm1_ptr             = scale_bmm1.data();
    p.fmha.scale_bmm2_ptr             = scale_bmm2.data();
    p.fmha.attention_sinks_ptr        = nullptr;
    p.block_iter_params.layer_id      = 0;

    if (IsCacheKVFP(p.quant_policy) && p.fmha.qkv_scale_orig == nullptr) {
        p.fmha.qkv_scale_orig = fp8_scale_orig.data();
    }

    if (IsCacheKVFP4(p.quant_policy) && fp4_scale_stride > 0) {
        const int64_t scale_dim    = p.size_per_head / 16;
        const int64_t layer_offset = static_cast<int64_t>(p.layer_id) * fp4_scale_stride;
        p.fmha.key_block_scales       = fp4_k_scales.data() + layer_offset;
        p.fmha.value_block_scales     = fp4_v_scales.data() + layer_offset;
        p.fmha.kv_scale_stride_heads  = p.block_iter_params.block_len * scale_dim;
        p.fmha.kv_scale_stride_batch  = p.num_kv_heads * p.fmha.kv_scale_stride_heads;
    }
}

}  // namespace

UnifiedAttentionLayer::~UnifiedAttentionLayer()
{

    check_cuda_error(cudaEventDestroy(aux_event_));
    check_cuda_error(cudaEventDestroy(qkv_event_));
    check_cuda_error(cudaStreamDestroy(aux_stream_));

    aux_event_ = qkv_event_ = {};
    aux_stream_             = {};
}

UnifiedAttentionLayer::UnifiedAttentionLayer(int                           quant_policy,
                                             const std::vector<int>&       layer_types,
                                             int                           layer_num,
                                             std::vector<AttentionWeight*> attn_weights,
                                             const EngineParam&            engine,
                                             const Context&                ctx,
                                             int                           phases,
                                             bool                          init):
    quant_policy_{quant_policy},
    rope_{attn_weights[0]->rope},
    engine_param_{engine},
    layer_num_{layer_num},
    cp_fn_ctx_{ctx.comm.d_comm, ctx.comm.d_cp_group},
    is_warm_up_{*ctx.is_warm_up},
    context_{ctx},
    init_{init},
    linear_(*ctx.linear),
    arch_{getSMVersion()}
{
    TM_CHECK(!attn_weights.empty()) << "attn_weights must not be empty";
    TM_CHECK(attn_weights[0]) << "attn_weights[0] must not be null";

    check_cuda_error(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    check_cuda_error(cudaEventCreateWithFlags(&qkv_event_, cudaEventDisableTiming));
    check_cuda_error(cudaEventCreateWithFlags(&aux_event_, cudaEventDisableTiming));

    init_rope_kernel_param(rope_, rope_param_);

    // Skip other attention layer types
    std::vector<int> types = layer_types;
    types.resize(layer_num);
    cache_layer_ids_.resize(types.size(), -1);
    int next_cache_id = 0;
    for (size_t i = 0; i < types.size(); ++i) {
        if (types[i] == 0) {
            cache_layer_ids_[i] = next_cache_id++;
        }
    }

    const int bsz = engine.max_batch_size;

    if (rope_param_.type == RopeType::kDynamic) {
        rope_base_buf_ = {bsz + 1, kCPUpinned};
    }
    else if (rope_param_.type == RopeType::kMrope) {
        // `mrope_position_ids` is not buffered
        mrope_position_delta_buf_ = {bsz, kCPUpinned};
        mrope_length_buf_         = {bsz, kCPUpinned};
    }
    const int max_blocks = bsz * cdiv(engine.session_len, engine_param_.cache_block_seq_len);
    for (int i = 0; i < phases; ++i) {
        auto& d               = data_.emplace_back(std::make_shared<AttentionData>());
        d->block_ptrs         = {max_blocks + 16, kDEVICE};
        d->block_ptrs_offsets = {bsz + 1, kDEVICE};
        if (rope_param_.type == RopeType::kDynamic) {
            d->rope_base = empty_like(rope_base_buf_, kDEVICE);
        }
        else if (rope_param_.type == RopeType::kMrope) {
            /// TODO: total space for `mrope_position_ids` can be reduced to (max_fwd_tokens, 3)
            d->mrope_position_ids    = {{bsz, engine.session_len, 3}, kDEVICE};
            d->mrope_position_delta  = empty_like(mrope_position_delta_buf_, kDEVICE);
            d->mrope_length          = empty_like(mrope_length_buf_, kDEVICE);
            rope_param_.mrope.stride = d->mrope_position_ids.stride(0);
        }
    }

    // Eagerly initialize workspace buffers (was previously lazy in Init())
    {
        const auto& w              = *attn_weights[0];
        const int   tp_size        = w.tp_size;
        const int   local_head_num = w.head_num / tp_size;
        const int   size_per_head  = w.head_dim;

        TM_CHECK_EQ(w.head_num % tp_size, 0) << w.head_num << " " << tp_size;
        TM_CHECK_EQ(w.head_num % w.kv_head_num, 0) << w.head_num << " " << w.kv_head_num;

        ssize_t   workspace_tokens = kMaxWorkspaceTokens;
        Allocator alloc            = core::Context::device_alloc();
        if (engine_param_.attn_cp_size > 1) {
            alloc = GetSymmAllocator(context_.comm.d_comm);
            workspace_tokens += engine_param_.max_forward_token_num;
        }

        partial_O_  = Tensor_<float>({workspace_tokens, local_head_num, size_per_head}, kDEVICE);
        partial_ML_ = Tensor_<float>({engine_param_.attn_cp_size, workspace_tokens, local_head_num, 2}, alloc);
        split_cnt_  = Tensor_<int>({workspace_tokens}, kDEVICE);
        if (init_) {
            const int dim = local_head_num * size_per_head;
            tmp_attn_     = Tensor{{engine_param_.max_forward_token_num, dim}, w.data_type, kDEVICE};
        }

        Clear(split_cnt_.buffer());
    }

    const bool use_fmha = engine_param_.attention_backend == AttentionBackend::kTrtllmFmha && isSM10x();
    fmha_use_ = use_fmha;

    if (use_fmha) {
        ::turbomind::flashinfer_fmha::initialize();

        const auto& w0            = *attn_weights[0];
        const int   local_head_num = w0.head_num / w0.tp_size;
        const int   size_per_head  = w0.head_dim;
        const int   max_tokens     = engine_param_.max_forward_token_num + engine_param_.max_batch_size;

        // Pre-allocate global FMHA buffers
        fmha_sm_count_        = getSMCount();
        fmha_scale_bmm1_      = Tensor_<float>({2}, kDEVICE);
        fmha_scale_bmm2_      = Tensor_<float>({1}, kDEVICE);
        fmha_tile_counter_    = Tensor_<int>({1}, kDEVICE);
        check_cuda_error(cudaMemset(fmha_tile_counter_.data(), 0, sizeof(int)));
        fmha_fp8_scale_orig_  = Tensor_<float>({4}, kDEVICE);
        const float def_scales[4] = {1.f, 1.f, 1.f, 1.f};
        check_cuda_error(cudaMemcpy(fmha_fp8_scale_orig_.data(), def_scales, sizeof(def_scales), cudaMemcpyHostToDevice));
        const int64_t ws = fmha_decode_workspace_size(bsz);
        fmha_decode_workspace_ = Tensor_<int8_t>({ws}, kDEVICE);
        check_cuda_error(cudaMemset(fmha_decode_workspace_.data(), 0, ws));

        // Per-phase FMHA buffers
        for (auto& sp : data_) {
            auto& d = *sp;
            d.fmha_page_tables  = {2 * max_blocks, kDEVICE};
            d.fmha_seq_lens_kv  = {bsz, kDEVICE};
            d.fmha_token2batch  = {2 * max_tokens, kCPUpinned};
            d.fmha_pf_cu_q_len  = {bsz + 1, kCPUpinned};
            d.fmha_pf_cu_k_len  = {bsz + 1, kCPUpinned};
            d.fmha_processed_q  = Tensor{{max_tokens, local_head_num * size_per_head}, w0.data_type, kDEVICE};
        }

        if (!tmp_attn_) {
            const int attn_dim = local_head_num * size_per_head;
            tmp_attn_          = Tensor{{max_tokens, attn_dim}, w0.data_type, kDEVICE};
        }
        TM_LOG_INFO("FMHA dispatch enabled: sm_count=%d heads=%d/%d head_dim=%d",
                    fmha_sm_count_, local_head_num, w0.kv_head_num / w0.tp_size, size_per_head);
    }
}

static void init_dynamic_ntk(RequestCache& cache, const core::RopeConfig& rope)
{
    cache.rope_base = rope.base;
    if (auto scaling_factor = rope.factor; scaling_factor > 1.f) {
        const auto max_seq_len = cache.prompt_len;
        const auto max_pos_emb = rope.max_position_embeddings;
        if (max_seq_len > max_pos_emb) {
            scaling_factor = scaling_factor * max_seq_len / max_pos_emb - (scaling_factor - 1);
            cache.rope_base *= powf(scaling_factor, rope.dim / (rope.dim - 2.f));
            // clang-format off
            TM_LOG_INFO("{} rope_scaling_factor: {}, rope_theta = {}",
                        cache.req->id, scaling_factor, cache.rope_base);
            // clang-format on
        }
    }
}

void UnifiedAttentionLayer::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op == BatchOp::kAdd) {
        Buffer_<RequestCache*> rc = env.at("requests").buffer();
        if (rope_param_.type == RopeType::kDynamic) {
            for (int i = 0; i < rc.size(); ++i) {
                init_dynamic_ntk(*rc[i], rope_);
            }
        }
    }
    else if (op == BatchOp::kSetup) {
        Setup(phase, env);
    }
    else if (op == BatchOp::kPrepare) {
        data_.at(phase)->finished  = env.at("finished").buffer().borrow();
        data_.at(phase)->q_offsets = env.at("q_offsets").buffer().borrow();
        data_.at(phase)->k_offsets = env.at("k_offsets").buffer().borrow();

        // This is needed in async mode to clear the `attn` buffer for the finished sequences. Ohterwise random NaNs
        // will crash the MoE router later
        /// TODO: use better solution, this increase memory usage and heterogenous attention layers may still break it
        if (tmp_attn_) {
            auto& d = data_.at(phase);
            Clear(tmp_attn_.slice(0, d->decode.n + d->prefill.q_sum));
            Clear(split_cnt_);
        }
    }
}

void UnifiedAttentionLayer::Setup(int phase, TensorMap& env)
{
    const auto& rc  = env.at("batch").data<BatchData*>()[0]->rc;
    const int   bsz = rc.size();

    auto& d    = *data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    {  /// Upload KV cache ptrs
        const Buffer_<int> offsets = env.at("block_ptrs_offsets").buffer();
        copy(env.at("block_ptrs").buffer(), offsets[bsz], d.block_ptrs);
        copy(offsets, bsz + 1, d.block_ptrs_offsets);
    }

    /// prepare Q/K stats for decode/prefill
    d.decode = d.prefill = {};

    d.decode.n  = std::find_if(rc.begin(), rc.end(), [](auto r) { return r->input_len > 1; }) - rc.begin();
    d.prefill.n = bsz - d.decode.n;

    // d.dbg_offset = d.dbg_size = 0;

    for (int i = 0; i < bsz; ++i) {
        const auto& c = *rc[i];

        // if (c.request->id == 4 && c.input_len > 1) {
        //     d.dbg_offset = d.decode.q_sum + d.prefill.q_sum;
        //     d.dbg_size   = c.input_len;
        // }

        auto& s = i < d.decode.n ? d.decode : d.prefill;
        s.q_sum += c.input_len;
        s.k_sum += c.history_len + c.alpha + c.input_len;
        s.q_max = std::max(s.q_max, c.input_len);
        s.k_max = std::max(s.k_max, c.history_len + c.alpha + c.input_len);
    }

    // auto &D = d.decode, &P = d.prefill;
    // dbg(D.n, D.k_sum, D.k_max, P.n, P.q_sum, P.q_max, P.k_sum, P.k_max);

    /// handling different RoPE types
    if (rope_param_.type == RopeType::kDynamic) {
        for (int i = 0; i < bsz; ++i) {
            rope_base_buf_[i] = rc[i]->rope_base;
        }
        copy(rope_base_buf_, bsz, d.rope_base);
    }
    else if (rope_param_.type == RopeType::kMrope) {
        const auto stride = d.mrope_position_ids.stride(0);
        for (int i = 0; i < rc.size(); ++i) {
            auto& c = *rc[i];
            auto& r = *c.req;
            if (auto pos_ids = r.inputs.try_("mrope_position_ids")) {
                int length                   = pos_ids->shape(0);
                mrope_length_buf_[i]         = length;
                mrope_position_delta_buf_[i] = *r.inputs.at("mrope_position_delta").data<int>();
                if (auto o = Interval{0, length} & Interval{c.history_len + c.alpha, Interval::Size{c.input_len}}) {
                    copy(pos_ids->data<int>() + o.begin() * 3,
                         (int)o.size() * 3,
                         d.mrope_position_ids.data() + i * stride + o.begin() * 3);
                }
            }
            else {
                mrope_length_buf_[i] = mrope_position_delta_buf_[i] = 0;
            }
        }
        copy(mrope_length_buf_, rc.size(), d.mrope_length);
        copy(mrope_position_delta_buf_, rc.size(), d.mrope_position_delta);
    }

    if (fmha_use_) {
        const cudaStream_t st = core::Context::stream().handle();

        // Pull KV pool layout from env (published by FmhaEngineHelper)
        auto pull_ptr = [&](const char* key, void*& dst) {
            if (auto it = env.find(key); it != env.end() && it->second) {
                dst = *it->second.data<void*>();
            }
        };
        auto pull_int = [&](const char* key, int& dst) {
            if (auto it = env.find(key); it != env.end() && it->second) {
                dst = *it->second.data<int>();
            }
        };
        auto pull_i64 = [&](const char* key, int64_t& dst) {
            if (auto it = env.find(key); it != env.end() && it->second) {
                dst = *it->second.data<int64_t>();
            }
        };
        pull_ptr("fmha_kv_cache_pool", d.fmha_kv_pool_ptr);
        pull_int("fmha_page_nums",      d.fmha_page_nums);
        pull_int("fmha_max_pages_per_seq", d.fmha_max_pages_per_seq);
        pull_i64("fmha_stride_layer_bytes",          d.fmha_stride_layer_bytes);
        pull_i64("fmha_kv_cache_value_offset_bytes", d.fmha_kv_value_offset);
        pull_i64("fmha_kv_cache_page_stride_bytes",  d.fmha_kv_page_stride);
        pull_i64("fmha_kv_cache_head_stride_bytes",  d.fmha_kv_head_stride);

        // Copy page tables and seq_lens from env to stable per-phase buffers
        const bool can_dispatch = d.fmha_kv_pool_ptr && d.fmha_page_nums > 0
                                  && d.fmha_max_pages_per_seq > 0 && d.fmha_kv_page_stride > 0;
        if (can_dispatch) {
            const int* src_pt = nullptr;
            const int* src_sl = nullptr;
            if (auto it = env.find("fmha_page_tables"); it != env.end() && it->second) {
                src_pt = it->second.data<int>();
            }
            if (auto it = env.find("fmha_seq_lens_kv"); it != env.end() && it->second) {
                src_sl = it->second.data<int>();
            }
            if (src_pt) {
                const int table_elems = bsz * 2 * d.fmha_max_pages_per_seq;
                check_cuda_error(cudaMemcpyAsync(d.fmha_page_tables.data(), src_pt,
                                                 table_elems * sizeof(int), cudaMemcpyDeviceToDevice, st));
            }
            if (src_sl) {
                check_cuda_error(cudaMemcpyAsync(d.fmha_seq_lens_kv.data(), src_sl,
                                                 bsz * sizeof(int), cudaMemcpyDeviceToDevice, st));
            }
        }

        // Build token2batch mapping (for append KV update)
        int token_idx = 0;
        for (int b = 0; b < bsz; ++b) {
            for (int i = 0; i < rc[b]->input_len; ++i) {
                d.fmha_token2batch[token_idx * 2]     = b;
                d.fmha_token2batch[token_idx * 2 + 1] = i;
                ++token_idx;
            }
        }
        d.fmha_total_tokens = token_idx;

        // Build cumulative Q/K lengths for prefill dispatch
        const int prefill_count = bsz - d.decode.n;
        if (d.decode.n > 0 && prefill_count > 0) {
            d.fmha_pf_cu_q_len[0] = 0;
            d.fmha_pf_cu_k_len[0] = 0;
            for (int i = 0; i < prefill_count; ++i) {
                const auto& c             = *rc[d.decode.n + i];
                d.fmha_pf_cu_q_len[i + 1] = d.fmha_pf_cu_q_len[i] + c.input_len;
                d.fmha_pf_cu_k_len[i + 1] = d.fmha_pf_cu_k_len[i] + c.history_len + c.alpha + c.input_len;
            }
        }
    }
}

void UnifiedAttentionLayer::Forward(ForwardParam p)
{
    TM_LOG_DEBUG("{}", __PRETTY_FUNCTION__);

    /////////////////////////////////////////////
    /// parse inputs
    const int token_num = p.input.shape(0);

    if (token_num == 0) {
        return;
    }

    const int layer_id = p.layer_id;

    const auto& weights = *p.weights;

    TM_LOG_DEBUG("layer=%d, token_num=%d", layer_id, token_num);

    Tensor qkv;

    auto& d = *data_.at(p.phase);

    // if (d.dbg_size) {
    //     DebugTensor(p.input.slice(d.dbg_offset, d.dbg_size), Concat("attn_in", p.layer_id), 0);
    // }

    if (weights.w_qkv && weights.w_qkv->output_dim) {
        // [token_num, hidden_dim] -> [token_num, local_q_kv_head_num, head_dim]
        qkv = linear_.Forward(p.input, *weights.w_qkv);
        sync_check_cuda_error();

        qk_norm(qkv, weights);
    }
    else {
        qkv = forward_mla(p.input, weights);
    }

    TM_DEBUG_TENSOR(qkv, Concat("qkv", layer_id), 3);

    auto invoke = [&](auto t) -> Tensor {
        using T = decltype(t);
        if constexpr (sizeof(T) == 2) {
            auto& d = *data_.at(p.phase);
            const bool fmha_ready = fmha_use_ && d.fmha_kv_pool_ptr != nullptr
                                    && d.fmha_total_tokens > 0 && !is_warm_up_;
            if (fmha_ready) {
                return core_attention_fmha<T>(qkv, p, weights);
            }
        }
        return core_attention<T>(qkv, p, weights);
    };

    Tensor attn = [&]() -> Tensor { TM_DISPATCH_PRIMARY_DTYPES_RET(qkv.dtype(), invoke); }();

    // Apply sigmoid gating: attn *= sigmoid(gate)
    // Gate is stored at the end of each token's QKV: [Q|K|V|Gate]
    if (weights.output_gate) {
        const int  tp_size           = weights.tp_size;
        const int  local_head_num    = weights.head_num / tp_size;
        const int  local_kv_head_num = weights.kv_head_num / tp_size;
        const int  size_per_head     = weights.head_dim;
        const int  q_count           = qkv.shape(0);
        const int  attn_dim          = local_head_num * size_per_head;
        const int  gate_offset       = (local_head_num + 2 * local_kv_head_num) * size_per_head;
        const int  qkv_stride        = (2 * local_head_num + 2 * local_kv_head_num) * size_per_head;
        const auto stream            = core::Context::stream().handle();
        invokeSigmoidGateMultiply(attn.raw_data(),
                                  (const char*)qkv.raw_data() + gate_offset * byte_size(qkv.dtype(), 1),
                                  attn_dim,
                                  qkv_stride,
                                  q_count,
                                  qkv.dtype(),
                                  stream);
        sync_check_cuda_error();
    }

    TM_DEBUG_TENSOR(attn, Concat("attn", layer_id), 3);

    // if (d.dbg_size) {
    //     DebugTensor(attn.slice(d.dbg_offset, d.dbg_size), Concat("attn_out", p.layer_id), 0);
    // }

    //////////////////////////////////////////////
    /// output gemm <Bs,HD> -> <Bs,HD>
    (void)linear_.Forward(attn, *weights.wo, p.output);
    sync_check_cuda_error();
}

template<class T>
Tensor UnifiedAttentionLayer::core_attention(Tensor& qkv, const ForwardParam& p, const WeightType& weights)
{
    const int tp_size           = weights.tp_size;
    const int local_head_num    = weights.head_num / tp_size;
    const int local_kv_head_num = weights.kv_head_num / tp_size;
    const int size_per_head     = weights.head_dim;

    const auto device = qkv.device();
    const auto dtype  = qkv.dtype();

    auto& d = *data_.at(p.phase);

    const int batch_size = d.decode.n + d.prefill.n;
    const int q_count    = qkv.shape(0);

    TM_CHECK_EQ(d.prefill.q_sum + d.decode.n, q_count);

    const int local_q_kv_head_num = local_head_num + 2 * local_kv_head_num;

    Tensor attn;
    if (tmp_attn_) {
        attn = tmp_attn_.slice(0, q_count);
    }
    else {
        attn = {{q_count, local_head_num * size_per_head}, dtype, device};
    }

    const bool is_mla = weights.is_mla();

    Tensor tmp_kv{{local_kv_head_num, is_mla ? 1 : 2, d.prefill.k_sum + MAX_CTA_S, size_per_head}, dtype, device};

    const int cache_layer_id = cache_layer_ids_[p.layer_id];

    auto CreateParams = [&](int offset, AttentionData::Stat stat, int max_kv_splits, cudaStream_t stream) {
        AttentionParams<T> params{};

        // Batch offset for `out` and `q` are computed inside the kernel
        params.out = (T*)attn.raw_data();

        params.q = (T*)qkv.raw_data();
        params.k = params.q + local_head_num * size_per_head;
        if (is_mla) {
            params.v      = params.k;
            params.stride = (local_head_num + 1 * local_kv_head_num) * size_per_head;
        }
        else {
            params.v = params.k + local_kv_head_num * size_per_head;
            // When attn_output_gate, QKV layout is [Q|K|V|Gate] per token
            // stride must account for the extra gate portion at the end
            if (weights.output_gate) {
                params.stride = (2 * local_head_num + 2 * local_kv_head_num) * size_per_head;
            }
            else {
                params.stride = (local_head_num + 2 * local_kv_head_num) * size_per_head;
            }
        }

        if (!is_mla && weights.w_qkv && weights.w_qkv->bias) {
            params.q_bias = (T*)weights.w_qkv->bias.data_or<T>(nullptr);
            params.k_bias = params.q_bias + local_head_num * size_per_head;
            params.v_bias = params.k_bias + local_kv_head_num * size_per_head;
        }

        params.batch_size = stat.n;

        params.token_num = stat.q_sum;
        params.max_q_len = stat.q_max;
        params.max_k_len = stat.k_max;

        // decode only
        params.block_iter_params = BlockIteratorParams{(char**)d.block_ptrs.data(),  //
                                                       d.block_ptrs_offsets.data() + offset,
                                                       cache_layer_id,
                                                       engine_param_.cache_block_seq_len};

        // prefill only
        if (is_mla) {
            params.linear_iter_params = LinearIteratorParams{
                tmp_kv.raw_data(),           // flattened KV
                stat.k_sum * size_per_head,  // stride to next head
                0                            // stride from K to V
            };
        }
        else {
            params.linear_iter_params = LinearIteratorParams{
                tmp_kv.raw_data(),               // flattened KV
                stat.k_sum * size_per_head * 2,  // stride to next head
                stat.k_sum * size_per_head       // stride from K to V
            };
        }

        params.finished = d.finished.data() + offset;
        params.cu_q_len = d.q_offsets.data() + offset;
        params.cu_k_len = d.k_offsets.data() + offset;

        params.num_heads     = local_head_num;
        params.num_kv_heads  = local_kv_head_num;
        params.size_per_head = size_per_head;
        params.layer_id      = cache_layer_id;

        double scaling = 1.;
        if (weights.softmax_scale) {  // model predefined softmax scale
            scaling *= weights.softmax_scale;
        }
        else {  // default value
            scaling /= std::sqrt((float)params.size_per_head);
        }
        params.inv_sqrt_dh = scaling * std::log2(std::exp(1.));

        params.sinks       = weights.sinks ? weights.sinks.data_or((T*)nullptr) : (T*)nullptr;
        params.scale_sinks = scaling;

        params.window_size = weights.window_size;
        if (!params.window_size) {
            params.window_size = 256 << 20;  // 256 M
        }

        params.rope_param = rope_param_;
        if (rope_param_.type == RopeType::kDynamic) {
            params.rope_param.base = d.rope_base.data() + offset;
        }
        else if (rope_param_.type == RopeType::kMrope) {
            params.rope_param.mrope.position_ids   = d.mrope_position_ids.data() + offset * rope_param_.mrope.stride;
            params.rope_param.mrope.position_delta = d.mrope_position_delta.data() + offset;
            params.rope_param.mrope.length         = d.mrope_length.data() + offset;
        }

        // logn attn
        params.use_logn_attn           = weights.use_logn_attn;
        params.max_position_embeddings = weights.rope.max_position_embeddings;

        // Decoding use only for now
        params.split_cnt   = split_cnt_.data();
        params.partial_ML  = partial_ML_.data();
        params.partial_O   = partial_O_.data();
        params.max_split_k = std::min(std::max(1, kMaxWorkspaceTokens / params.token_num), max_kv_splits);

        // context parallel
        params.cp_rank = engine_param_.attn_cp_rank;
        params.cp_size = engine_param_.attn_cp_size;
        if (params.cp_size > 1) {
            params.cp_size = cutlass::FastDivmod(params.cp_size);

            // update ML,O offset if both prefill and decode present
            const int offset_ML_stage =
                engine_param_.attn_cp_size * (offset ? kMaxWorkspaceTokens * local_head_num * 2 : 0);
            const int offset_ML_rank = params.cp_rank * params.token_num * local_head_num * params.max_split_k * 2;
            const int offset_O       = offset ? kMaxWorkspaceTokens * local_head_num * size_per_head : 0;

            params.partial_ML = partial_ML_.data() + offset_ML_stage + offset_ML_rank;
            params.partial_O  = partial_O_.data() + offset_O;
            params.offset_q   = offset;

            // postprocess func
            params.cp_fn          = CpPost;
            params.cp_fn_ctx      = (void*)&cp_fn_ctx_;
            cp_fn_ctx_.cp_rank    = params.cp_rank;
            cp_fn_ctx_.count      = params.token_num * local_head_num * params.max_split_k * 2;
            cp_fn_ctx_.partial_ML = partial_ML_.data() + offset_ML_stage;
            cp_fn_ctx_.stream     = stream;
        }

        params.arch   = arch_;
        params.stream = stream;

        params.quant_policy = quant_policy_;
        return params;
    };

    const cudaStream_t stream = core::Context::stream().handle();

    cudaStream_t pf_stream = stream;
    cudaStream_t dc_stream = pf_stream;

    if (d.decode.n && d.prefill.n) {
        pf_stream = aux_stream_;
        check_cuda_error(cudaEventRecord(qkv_event_, stream));
        check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));
    }

    if (d.prefill.n && !is_warm_up_) {
        const int offset = d.decode.n;
        // We are executing prefill & decoding kernels concurrently, but only have 1 workspace
        // disable split kv for prefill for now
        auto params = CreateParams(offset, d.prefill, 1, pf_stream);
        if constexpr (sizeof(T) == 2) {
            invokeProcessKV_v2_(params);
            sync_check_cuda_error();

            /// TODO: skip flattening for `sm_80`
            invokeFlattenKV_v2_(params, d.prefill.k_sum);
            sync_check_cuda_error();

            dispatchAttention(params);
            sync_check_cuda_error();
        }
    }

    if (d.decode.n && !is_warm_up_) {
        auto params = CreateParams(0, d.decode, kMaxKVSplits, dc_stream);
        if constexpr (sizeof(T) == 2) {
            dispatchDecoding<T>(params);
            sync_check_cuda_error();
        }
    }

    if (d.decode.n && d.prefill.n) {
        check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
        check_cuda_error(cudaStreamWaitEvent(stream, aux_event_));
    }

    if (is_warm_up_) {
        rng_.set_stream(stream);
        rng_.GenerateUniform(attn.data<T>(), attn.size(), .02f, -.01f);
    }

    return attn;
}

template<class T>
Tensor UnifiedAttentionLayer::core_attention_fmha(Tensor& qkv, const ForwardParam& p, const WeightType& weights)
{
    static_assert(sizeof(T) == 2, "core_attention_fmha is BF16/FP16 only");

    const int local_head_num    = weights.head_num / weights.tp_size;
    const int local_kv_head_num = weights.kv_head_num / weights.tp_size;
    const int size_per_head     = weights.head_dim;
    const auto device           = qkv.device();

    auto& d = *data_.at(p.phase);
    TM_CHECK_EQ(d.prefill.q_sum + d.decode.n, (int)qkv.shape(0));

    const bool is_mla          = weights.kv_lora_rank > 0;
    const int  cache_layer_id  = cache_layer_ids_[p.layer_id];
    const bool full_fp8        = IsCacheKVFP(quant_policy_);
    const bool fuse_fp8_out    = weights.output.input_scales && !weights.output_gate && full_fp8;
    const int  attn_dim        = local_head_num * size_per_head;
    const int  q_count         = static_cast<int>(qkv.shape(0));

    TM_CHECK(tmp_attn_) << "FMHA path requires pre-allocated tmp_attn_";
    Tensor attn;
    if (fuse_fp8_out) {
        attn = Tensor{tmp_attn_.raw_data(), {q_count, attn_dim}, kFloat8_e4m3, device};
    } else {
        attn = tmp_attn_.slice(0, q_count);
    }

    const cudaStream_t stream = core::Context::stream().handle();
    check_cuda_error(cudaMemsetAsync(attn.raw_data(), 0, attn.byte_size(), stream));

    // Build AttentionParams<T> for a given batch segment
    auto make_params = [&](int offset, int n, int q_sum, int q_max, int k_sum, int k_max, int splits) {
        AttentionParams<T> params{};
        params.out = reinterpret_cast<T*>(attn.raw_data());
        params.q   = reinterpret_cast<T*>(qkv.raw_data());
        params.k   = params.q + local_head_num * size_per_head;
        if (is_mla) {
            params.v      = params.k;
            params.stride = (local_head_num + local_kv_head_num) * size_per_head;
        } else {
            params.v = params.k + local_kv_head_num * size_per_head;
            params.stride = weights.output_gate
                                ? (2 * local_head_num + 2 * local_kv_head_num) * size_per_head
                                : (local_head_num + 2 * local_kv_head_num) * size_per_head;
        }
        if (weights.w_qkv && weights.w_qkv->bias) {
            params.q_bias = reinterpret_cast<T*>(weights.w_qkv->bias.data_or<T>(nullptr));
            params.k_bias = params.q_bias + local_head_num * size_per_head;
            params.v_bias = params.k_bias + local_kv_head_num * size_per_head;
        }
        params.batch_size = n;
        params.token_num  = q_sum;
        params.max_q_len  = q_max;
        params.max_k_len  = k_max;
        params.block_iter_params = BlockIteratorParams{reinterpret_cast<char**>(d.block_ptrs.data()),
                                                       d.block_ptrs_offsets.data() + offset,
                                                       cache_layer_id,
                                                       static_cast<int>(engine_param_.cache_block_seq_len)};
        params.finished = d.finished.data() + offset;
        params.cu_q_len = d.q_offsets.data() + offset;
        params.cu_k_len = d.k_offsets.data() + offset;
        params.num_heads     = local_head_num;
        params.num_kv_heads  = local_kv_head_num;
        params.size_per_head = size_per_head;
        params.layer_id      = cache_layer_id;
        const double scaling = weights.softmax_scale
                                   ? static_cast<double>(weights.softmax_scale)
                                   : 1.0 / std::sqrt(static_cast<double>(size_per_head));
        params.inv_sqrt_dh   = scaling * std::log2(std::exp(1.));
        params.sinks         = weights.sinks ? weights.sinks.data_or(static_cast<T*>(nullptr)) : nullptr;
        params.scale_sinks   = scaling;
        params.window_size   = weights.window_size ? weights.window_size : (256 << 20);
        params.rope_param    = rope_param_;
        if (rope_param_.type == RopeType::kDynamic) {
            params.rope_param.base = d.rope_base.data() + offset;
        } else if (rope_param_.type == RopeType::kMrope) {
            params.rope_param.mrope.position_ids   = d.mrope_position_ids.data() + offset * rope_param_.mrope.stride;
            params.rope_param.mrope.position_delta = d.mrope_position_delta.data() + offset;
            params.rope_param.mrope.length         = d.mrope_length.data() + offset;
        }
        params.use_logn_attn           = weights.use_logn_attn;
        params.max_position_embeddings = weights.rope.max_position_embeddings;
        params.split_cnt   = split_cnt_.data();
        params.partial_ML  = partial_ML_.data();
        params.partial_O   = partial_O_.data();
        params.max_split_k = std::min(std::max(1, kMaxWorkspaceTokens / std::max(q_sum, 1)), splits);
        params.cp_rank     = engine_param_.attn_cp_rank;
        params.cp_size     = engine_param_.attn_cp_size;
        params.arch        = arch_;
        params.stream      = stream;
        params.quant_policy = quant_policy_;
        if (full_fp8 && weights.qkv_scale_orig) {
            params.fmha.qkv_scale_orig = const_cast<float*>(weights.qkv_scale_orig.data<float>());
        }
        if (fuse_fp8_out) {
            params.fuse_attention_quant = true;
            params.fp8_scale_inv_ptr    = weights.output.input_scales.data<float>();
            params.fmha.o_scale_orig    = const_cast<float*>(weights.output.input_scales.data<float>());
        }
        // FMHA-specific fmha sub-struct
        params.fmha.layer_num                    = layer_num_;
        params.fmha.sum_q_len                    = q_sum;
        params.fmha.sum_kv_len                   = k_sum;
        params.fmha.max_attention_window_size    = engine_param_.session_len;
        params.fmha.cyclic_attention_window_size = engine_param_.session_len;
        params.fmha.is_qk_norm  = false;
        params.fmha.qk_norm_eps = 1e-5f;
        params.fmha.q_scaling   = weights.softmax_scale
                                       ? 1.0f / (std::sqrt(static_cast<float>(size_per_head)) * weights.softmax_scale)
                                       : 1.0f;
        params.fmha.workspace_buffer = fmha_decode_workspace_.data();
        params.fmha.workspace_size   = static_cast<int64_t>(fmha_decode_workspace_.size());
        params.fmha.multi_block_mode = false;
        return params;
    };

    auto set_raw_qkv = [&](AttentionParams<T>& params) {
        auto* raw = reinterpret_cast<T*>(qkv.raw_data());
        params.fmha.raw_q = raw;
        params.fmha.raw_k = raw + local_head_num * size_per_head;
        params.fmha.raw_v = params.fmha.raw_k + local_kv_head_num * size_per_head;
    };

    auto set_q_output = [&](AttentionParams<T>& params, int q_byte_offset) {
        const bool fp8_q = IsCacheKVFP(params.quant_policy);
        const int  q_bytes = fp8_q ? 1 : static_cast<int>(sizeof(T));
        params.q = reinterpret_cast<T*>(
            reinterpret_cast<char*>(d.fmha_processed_q.raw_data()) + q_byte_offset);
        params.k = nullptr;
        params.v = nullptr;
    };

    auto set_attn_output_offset = [&](AttentionParams<T>& params, int q_byte_offset) {
        const int attn_bytes = (attn.dtype() == kFloat8_e4m3) ? 1 : static_cast<int>(sizeof(T));
        params.out = reinterpret_cast<T*>(
            reinterpret_cast<char*>(attn.raw_data()) + q_byte_offset);
    };

    auto populate_ext = [&](AttentionParams<T>& params, int batch_offset) {
        fmha_populate_ext(params, d, batch_offset,
                          fmha_scale_bmm1_, fmha_scale_bmm2_, fmha_fp8_scale_orig_,
                          fmha_fp4_k_scales_, fmha_fp4_v_scales_, fmha_fp4_scale_stride_);
    };

    // Helper: upload token2batch to device (happens before KV update kernels)
    auto upload_token2batch = [&]() {
        if (d.fmha_total_tokens > 0) {
            // Allocate a small device buffer for the upload
            // token2batch is in host pinned memory; we pass data directly to the kernel
        }
    };

    // Dispatch: mixed decode + prefill
    if (d.decode.n && d.prefill.n) {
        // 1. Append KV update for ALL tokens (decode + prefill)
        auto prep = make_params(0, d.decode.n + d.prefill.n,
                                d.decode.q_sum + d.prefill.q_sum,
                                std::max(d.decode.q_max, d.prefill.q_max),
                                d.decode.k_sum + d.prefill.k_sum,
                                std::max(d.decode.k_max, d.prefill.k_max), 1);
        set_raw_qkv(prep);
        set_q_output(prep, 0);
        prep.fmha.token2batch = d.fmha_token2batch.data();
        prep.fmha.host_bmm1_scale = 1.0f / (std::sqrt(static_cast<float>(size_per_head)) * prep.fmha.q_scaling);
        populate_ext(prep, 0);
        invokeAppendQKNormRopeKVUpdate_<T>(prep);
        fmha_sync_stream(stream);

        // 2. Decode attention
        auto dc = make_params(0, d.decode.n, d.decode.q_sum, d.decode.q_max, d.decode.k_sum, d.decode.k_max, 128);
        set_q_output(dc, 0);
        populate_ext(dc, 0);
        dc.fmha.token2batch = d.fmha_token2batch.data();
        dc.fmha.host_bmm1_scale = 1.0f / (std::sqrt(static_cast<float>(size_per_head)) * dc.fmha.q_scaling);
        dc.cu_q_len = nullptr;
        dc.cu_k_len = nullptr;
        {
            auto fp = fmha_build_params<T>(dc, d, 0, fmha_sm_count_, false);
            fp.cum_seq_lens_q = nullptr;
            fp.cum_seq_lens_kv = nullptr;
            if (!::turbomind::flashinfer_fmha::dispatch_decode(fp)) {
                FT_THROW("FlashInfer FMHA decode dispatch failed.");
            }
            fmha_sync_stream(stream);
        }

        // 3. Prefill attention
        const int pf_offset     = d.decode.n;
        const int pf_q_bytes    = (IsCacheKVFP(quant_policy_) ? 1 : static_cast<int>(sizeof(T)));
        const int pf_q_byte_off = d.decode.q_sum * local_head_num * size_per_head * pf_q_bytes;
        const int pf_attn_bytes = (attn.dtype() == kFloat8_e4m3) ? 1 : static_cast<int>(sizeof(T));
        const int pf_attn_off   = d.decode.q_sum * local_head_num * size_per_head * pf_attn_bytes;
        auto pf = make_params(pf_offset, d.prefill.n, d.prefill.q_sum, d.prefill.q_max, d.prefill.k_sum, d.prefill.k_max, 1);
        set_q_output(pf, pf_q_byte_off);
        set_attn_output_offset(pf, pf_attn_off);
        pf.fmha.token2batch      = d.fmha_token2batch.data() + d.decode.q_sum * 2;
        pf.fmha.host_bmm1_scale  = 1.0f / (std::sqrt(static_cast<float>(size_per_head)) * pf.fmha.q_scaling);
        pf.fmha.workspace_buffer = nullptr;
        pf.fmha.workspace_size   = 0;
        pf.cu_q_len              = d.fmha_pf_cu_q_len.data();
        pf.cu_k_len              = d.fmha_pf_cu_k_len.data();
        populate_ext(pf, pf_offset);
        {
            auto fp = fmha_build_params<T>(pf, d, pf_offset, fmha_sm_count_, false);
            if (!::turbomind::flashinfer_fmha::dispatch_context(fp)) {
                FT_THROW("FlashInfer FMHA context dispatch failed.");
            }
            fmha_sync_stream(stream);
        }
    }
    else if (d.prefill.n) {
        // Pure prefill
        auto params = make_params(0, d.prefill.n, d.prefill.q_sum, d.prefill.q_max, d.prefill.k_sum, d.prefill.k_max, 1);
        set_raw_qkv(params);
        set_q_output(params, 0);
        params.fmha.token2batch  = d.fmha_token2batch.data();
        params.fmha.host_bmm1_scale = 1.0f / (std::sqrt(static_cast<float>(size_per_head)) * params.fmha.q_scaling);
        populate_ext(params, 0);
        invokeAppendQKNormRopeKVUpdate_<T>(params);
        fmha_sync_stream(stream);

        params.fmha.workspace_buffer = nullptr;
        params.fmha.workspace_size   = 0;
        {
            auto fp = fmha_build_params<T>(params, d, 0, fmha_sm_count_, false);
            if (!::turbomind::flashinfer_fmha::dispatch_context(fp)) {
                FT_THROW("FlashInfer FMHA context dispatch failed.");
            }
            fmha_sync_stream(stream);
        }
    }
    else if (d.decode.n) {
        // Pure decode
        auto params = make_params(0, d.decode.n, d.decode.q_sum, d.decode.q_max, d.decode.k_sum, d.decode.k_max, 128);
        set_raw_qkv(params);
        set_q_output(params, 0);
        params.fmha.token2batch  = d.fmha_token2batch.data();
        params.fmha.host_bmm1_scale = 1.0f / (std::sqrt(static_cast<float>(size_per_head)) * params.fmha.q_scaling);
        populate_ext(params, 0);
        invokeDecodingQKNormRopeKVUpdate_<T>(params);
        fmha_sync_stream(stream);

        params.cu_q_len = nullptr;
        params.cu_k_len = nullptr;
        {
            auto fp = fmha_build_params<T>(params, d, 0, fmha_sm_count_, false);
            fp.cum_seq_lens_q = nullptr;
            fp.cum_seq_lens_kv = nullptr;
            if (!::turbomind::flashinfer_fmha::dispatch_decode(fp)) {
                FT_THROW("FlashInfer FMHA decode dispatch failed.");
            }
            fmha_sync_stream(stream);
        }
    }

    // FP8 output dequantize if needed but output is not fused
    if (full_fp8 && !fuse_fp8_out && attn.dtype() == kFloat8_e4m3) {
        // Should not happen in the simplified path — log a warning
        TM_LOG_WARNING("core_attention_fmha: FP8 output without fusion; unexpected");
    }

    return attn;
}

template Tensor UnifiedAttentionLayer::core_attention_fmha<half>(Tensor&, const ForwardParam&, const WeightType&);
template Tensor UnifiedAttentionLayer::core_attention_fmha<nv_bfloat16>(Tensor&, const ForwardParam&, const WeightType&);

Tensor UnifiedAttentionLayer::forward_mla(const Tensor& hidden_state, const WeightType& w)
{

    const int tp_size           = w.tp_size;
    const int local_head_num    = w.head_num / tp_size;
    const int local_kv_head_num = w.kv_head_num / tp_size;
    const int size_per_head     = w.head_dim;

    const auto token_num = hidden_state.shape(0);
    const auto dtype     = hidden_state.dtype();

    const int q_lora_rank  = w.q_a_proj->output_dim;
    const int kv_lora_rank = w.kv_a_layernorm->weight.size();
    const int qk_rope_dim  = w.kv_a_proj->output_dim - kv_lora_rank;

    Tensor q;

    const auto stream = core::Context::stream().handle();

    if (w.q_proj && w.q_proj->weight) {
        q = linear_.Forward(hidden_state, *w.q_proj);
        sync_check_cuda_error();
    }
    else {
        Tensor q_a = linear_.Forward(hidden_state, *w.q_a_proj);
        sync_check_cuda_error();

        invokeRMSNorm(q_a, q_a, w.q_a_layernorm->weight, w.q_a_layernorm->norm_eps_, stream);
        sync_check_cuda_error();

        q = linear_.Forward(q_a, *w.q_b_proj);
        sync_check_cuda_error();
    }

    Tensor kv_a_k_pe = linear_.Forward(hidden_state, *w.kv_a_proj);
    sync_check_cuda_error();

    auto kv_a = kv_a_k_pe.slice({0, 0}, {-1, kv_lora_rank});
    invokeRMSNorm(kv_a, kv_a, w.kv_a_layernorm->weight, w.kv_a_layernorm->norm_eps_, stream);
    sync_check_cuda_error();

    const int local_q_kv_head_num = local_head_num + 1 * local_kv_head_num;

    Tensor qkv{{token_num, local_q_kv_head_num, size_per_head}, dtype, hidden_state.device()};
    MLACopyQKV(dtype,
               qkv.raw_data(),
               q.raw_data(),
               kv_a_k_pe.raw_data(),
               token_num,
               local_head_num,
               kv_lora_rank,
               qk_rope_dim,
               stream);
    sync_check_cuda_error();

    return qkv;
}

void UnifiedAttentionLayer::qk_norm(Tensor& qkv, const WeightType& weights)
{
    if (!(weights.q_norm || weights.k_norm)) {
        return;
    }

    TM_CHECK(weights.q_norm && weights.k_norm);

    const int tp_size           = weights.tp_size;
    const int local_head_num    = weights.head_num / tp_size;
    const int local_kv_head_num = weights.kv_head_num / tp_size;
    const int size_per_head     = weights.head_dim;

    const auto stream = core::Context::stream().handle();

    check_cuda_error(cudaEventRecord(qkv_event_, stream));
    check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));

    const auto token_num = qkv.shape(0);

    auto qkv3 = qkv.view({token_num, -1, size_per_head});

    auto q = qkv3.slice({0, 0, 0}, {-1, local_head_num, -1});
    invokeRMSNormQK(q, weights.q_norm->weight, weights.q_norm->norm_eps_, stream);
    sync_check_cuda_error();

    auto k = qkv3.slice({0, local_head_num, 0}, {-1, local_kv_head_num, -1});
    invokeRMSNormQK(k, weights.k_norm->weight, weights.k_norm->norm_eps_, aux_stream_);
    sync_check_cuda_error();

    check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
    check_cuda_error(cudaStreamWaitEvent(stream, aux_event_));
}

}  // namespace turbomind
