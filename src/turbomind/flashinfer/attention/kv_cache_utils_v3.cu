// Copyright (c) OpenMMLab. All rights reserved.

#include <type_traits>

#include "src/turbomind/flashinfer/attention/kv_cache_utils_v3.h"
#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/attention/rotary_embedding.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/thread_map.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<class F>
void DispatchKvUpdateHeadDim(int head_dim, F&& f)
{
    switch (head_dim) {
        case 64:
            f(std::integral_constant<int, 64>{});
            return;
        case 128:
            f(std::integral_constant<int, 128>{});
            return;
        case 192:
            f(std::integral_constant<int, 192>{});
            return;
        case 256:
            f(std::integral_constant<int, 256>{});
            return;
        case 576:
            f(std::integral_constant<int, 576>{});
            return;
        default:
            FT_CHECK_WITH_INFO(false, "FlashInfer TRTLLM FMHA KV update supports head_dim in {64,128,192,256,576}");
    }
}

template<int HeadDim, int VecSize>
struct AppendKvUpdateBlockSize {
    static_assert(HeadDim % VecSize == 0, "HeadDim must be multiple of vector size");

    static constexpr int kVectorPerHead = HeadDim / VecSize;
    static constexpr int value          = (256 % kVectorPerHead == 0) ? 256 :
                                          (240 % kVectorPerHead == 0) ? 240 :
                                          (224 % kVectorPerHead == 0) ? 224 :
                                          (288 % kVectorPerHead == 0) ? 288 :
                                          (320 % kVectorPerHead == 0) ? 320 :
                                          (384 % kVectorPerHead == 0) ? 384 :
                                          (448 % kVectorPerHead == 0) ? 448 :
                                          (512 % kVectorPerHead == 0) ? 512 :
                                                                        kVectorPerHead;
};

template<int ActiveLanes>
struct WarpActiveMask {
    static_assert(ActiveLanes > 0 && ActiveLanes < WARP_SIZE);

    static constexpr unsigned value = (1u << ActiveLanes) - 1u;
};

template<>
struct WarpActiveMask<WARP_SIZE> {
    static constexpr unsigned value = 0xffffffffu;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PERFORMANCE OPTIMIZATION UTILITIES
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Optimized squared sum computation using FMA instructions
template<int kVecSize, typename T, typename Vec>
__device__ __forceinline__ float compute_squared_sum_fma(const Vec* vecs, int iter_count)
{
    float sum = 0.0f;
#pragma unroll
    for (int c = 0; c < iter_count; ++c) {
#pragma unroll
        for (int v = 0; v < kVecSize; ++v) {
            float val = static_cast<float>(vecs[c][v]);
            sum       = fmaf(val, val, sum);  // FMA: sum += val * val
        }
    }
    return sum;
}

// Optimized RMSNorm application with weight scaling
template<int kVecSize, typename T, typename Vec>
__device__ __forceinline__ void apply_rmsnorm_with_weight_fma(Vec* vec, const Vec* weight, float inv_rms)
{
#pragma unroll
    for (int v = 0; v < kVecSize; ++v) {
        float val = static_cast<float>((*vec)[v]);
        float w   = static_cast<float>((*weight)[v]);
        (*vec)[v] = static_cast<T>(val * inv_rms * w);
    }
}

__device__ __forceinline__ void initialize_fmha_scales(float fmha_host_bmm1_scale,
                                                       float* __restrict__ qkv_scale_orig,
                                                       float* __restrict__ o_scale_orig,
                                                       float* __restrict__ fmha_bmm1_scale,
                                                       float* __restrict__ fmha_bmm2_scale,
                                                       int* __restrict__ fmha_tile_counter)
{
    const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        if (fmha_tile_counter) {
            fmha_tile_counter[0] = 0u;
        }

        float q_scale_orig_quant = qkv_scale_orig ? qkv_scale_orig[0] : 1.f;
        float k_scale_orig_quant = qkv_scale_orig ? qkv_scale_orig[1] : 1.f;
        float v_scale_orig_quant = qkv_scale_orig ? qkv_scale_orig[2] : 1.f;
        float o_scale_orig_quant = o_scale_orig ? o_scale_orig[0] : 1.f;

        if (fmha_bmm1_scale) {
            // The scale after fmha bmm1
            fmha_bmm1_scale[0] = q_scale_orig_quant * k_scale_orig_quant * fmha_host_bmm1_scale;
            // The scale prepared for log2 optimization
            constexpr float kLog2e = 1.4426950408889634074f;
            fmha_bmm1_scale[1]     = fmha_bmm1_scale[0] * kLog2e;
        }
        if (fmha_bmm2_scale) {
            fmha_bmm2_scale[0] = 1.f / o_scale_orig_quant * v_scale_orig_quant;
        }
    }
}

template<class T, int N>
__device__ __forceinline__ float vec_absmax(const Array<T, N>& vec)
{
    float amax = 0.f;
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        amax = fmaxf(amax, fabsf(static_cast<float>(vec[i])));
    }
    return amax;
}

__device__ __forceinline__ float
store_fp4_scale(float amax, float global_sf, uint8_t* scales, int scale_idx, bool write)
{
    const float   safe_global = global_sf != 0.f ? global_sf : 1.f;
    const float   sf          = amax / (6.f * safe_global);
    __nv_fp8_e4m3 fp8_sf{sf};
    if (amax > 0.f && fp8_sf.__x == 0) {
        fp8_sf.__x = 0x01;
    }
    if (scales && write) {
        scales[scale_idx] = fp8_sf.__x;
    }
    return static_cast<float>(fp8_sf) * safe_global;
}

__device__ __forceinline__ int fp4_k_scale_index(int token_idx, int scale_idx, int scale_dim)
{
    return token_idx * scale_dim + scale_idx;
}

__device__ __forceinline__ int fp4_v_scale_index(int token_idx, int scale_idx, int scale_dim)
{
    return (token_idx / 4) * 4 * scale_dim + scale_idx * 4 + (token_idx % 4);
}

template<class T, class Tkv, int N>
__device__ __forceinline__ Array<Tkv, N>
convert_kv_cache(const Array<T, N>& vec, float global_sf, uint8_t* scales, int scale_idx, int vec_idx)
{
    if constexpr (std::is_same_v<Tkv, fp4_e2m1_t>) {
        const float            local_amax = vec_absmax(vec);
        const float            block_amax = fmaxf(local_amax, __shfl_xor_sync(uint32_t(-1), local_amax, 1));
        const float            scale = store_fp4_scale(block_amax, global_sf, scales, scale_idx, (vec_idx & 1) == 0);
        ConvertKvCache<T, Tkv> conv{scale, 0.f};
        return conv(vec);
    }
    else {
        ConvertKvCache<T, Tkv> conv{global_sf, 0.f};
        return conv(vec);
    }
}

template<class T, class Tkv, class Layout>
class HeadBlockPagedKV {
public:
    TM_HOST_DEVICE HeadBlockPagedKV(Layout     layout,
                                    int        head_id,
                                    const int* page_tables,
                                    int        batch_idx,
                                    int        max_num_pages_per_seq_kv,
                                    char*      kv_cache_buffer,
                                    int64_t    page_stride_bytes,
                                    void*      key_block_scales      = nullptr,
                                    void*      value_block_scales    = nullptr,
                                    int64_t    kv_scale_stride_heads = 0,
                                    int64_t    kv_scale_stride_batch = 0,
                                    int        scale_dim             = 0):
        layout_{layout},
        head_id_{head_id},
        page_tables_{page_tables},
        batch_idx_{batch_idx},
        max_num_pages_per_seq_kv_{max_num_pages_per_seq_kv},
        kv_cache_buffer_{kv_cache_buffer},
        page_stride_bytes_{page_stride_bytes},
        key_block_scales_{reinterpret_cast<uint8_t*>(key_block_scales)},
        value_block_scales_{reinterpret_cast<uint8_t*>(value_block_scales)},
        kv_scale_stride_heads_{kv_scale_stride_heads},
        kv_scale_stride_batch_{kv_scale_stride_batch},
        scale_dim_{scale_dim}
    {
    }

    TM_HOST_DEVICE void get_block_coord(int seq_ti, int& block_idx, int& block_ti) const
    {
        block_idx = seq_ti / block_len();
        block_ti  = seq_ti % block_len();
    }

    TM_HOST_DEVICE auto block_len() const
    {
        return layout_.config().block_len();
    }

    template<class Func>
    TM_HOST_DEVICE auto with(int ti, Func&& func) const
    {
        int block_idx;
        int block_ti;
        get_block_coord(ti, block_idx, block_ti);

        if (block_idx < 0 || block_idx >= max_num_pages_per_seq_kv_) [[unlikely]] {
            return;
        }

        const size_t base_offset      = static_cast<size_t>(batch_idx_) * 2 * max_num_pages_per_seq_kv_;
        const int    k_physical_block = page_tables_[base_offset + block_idx];
        const int    v_physical_block = page_tables_[base_offset + max_num_pages_per_seq_kv_ + block_idx];

        if (k_physical_block < 0 || v_physical_block < 0) [[unlikely]] {
            return;
        }

        const int64_t k_block_offset = static_cast<int64_t>(k_physical_block) * page_stride_bytes_;
        const int64_t v_block_offset = static_cast<int64_t>(v_physical_block) * page_stride_bytes_;

        auto* k_ptr = kv_cache_buffer_ + k_block_offset + layout_.k_data(0, head_id_, block_ti);
        auto* v_ptr = kv_cache_buffer_ + v_block_offset + layout_.v_data(0, head_id_, block_ti);

        if constexpr (bitsof<Tkv> % bitsof<char> != 0) {
            return ((Func&&)func)(SubBytePtr<Tkv>{k_ptr}, SubBytePtr<Tkv>{v_ptr});
        }
        else {
            return ((Func&&)func)(reinterpret_cast<Tkv*>(k_ptr), reinterpret_cast<Tkv*>(v_ptr));
        }
    }

    template<class Func>
    TM_HOST_DEVICE auto with_scales(int ti, Func&& func) const
    {
        int block_idx;
        int block_ti;
        get_block_coord(ti, block_idx, block_ti);

        if (block_idx < 0 || block_idx >= max_num_pages_per_seq_kv_) [[unlikely]] {
            return;
        }

        const size_t base_offset      = static_cast<size_t>(batch_idx_) * 2 * max_num_pages_per_seq_kv_;
        const int    k_physical_block = page_tables_[base_offset + block_idx];
        const int    v_physical_block = page_tables_[base_offset + max_num_pages_per_seq_kv_ + block_idx];

        if (k_physical_block < 0 || v_physical_block < 0) [[unlikely]] {
            return;
        }

        const int64_t k_block_offset = static_cast<int64_t>(k_physical_block) * page_stride_bytes_;
        const int64_t v_block_offset = static_cast<int64_t>(v_physical_block) * page_stride_bytes_;

        auto* k_ptr   = kv_cache_buffer_ + k_block_offset + layout_.k_data(0, head_id_, block_ti);
        auto* v_ptr   = kv_cache_buffer_ + v_block_offset + layout_.v_data(0, head_id_, block_ti);
        auto* k_scale = key_block_scales_ ?
                            key_block_scales_ + static_cast<int64_t>(k_physical_block) * kv_scale_stride_batch_
                                + static_cast<int64_t>(head_id_) * kv_scale_stride_heads_ :
                            nullptr;
        auto* v_scale = value_block_scales_ ?
                            value_block_scales_ + static_cast<int64_t>(v_physical_block) * kv_scale_stride_batch_
                                + static_cast<int64_t>(head_id_) * kv_scale_stride_heads_ :
                            nullptr;

        if constexpr (bitsof<Tkv> % bitsof<char> != 0) {
            return ((Func&&)func)(SubBytePtr<Tkv>{k_ptr}, SubBytePtr<Tkv>{v_ptr}, k_scale, v_scale, block_ti);
        }
        else {
            return ((Func&&)func)(
                reinterpret_cast<Tkv*>(k_ptr), reinterpret_cast<Tkv*>(v_ptr), k_scale, v_scale, block_ti);
        }
    }

private:
    Layout     layout_;
    int        head_id_;
    const int* page_tables_;
    int        batch_idx_;
    int        max_num_pages_per_seq_kv_;
    char*      kv_cache_buffer_;
    int64_t    page_stride_bytes_;
    uint8_t*   key_block_scales_;
    uint8_t*   value_block_scales_;
    int64_t    kv_scale_stride_heads_;
    int64_t    kv_scale_stride_batch_;
    int        scale_dim_;
};

template<class TOut, class Tkv, int HeadDim, class T, class BlockLayout>
__global__ void decodingQKNormRopeKVUpdate_Kernel_v2(void*           processed_q,
                                                     void*           kv_cache_buffer,
                                                     const T*        q,
                                                     const T*        k,
                                                     const T*        v,
                                                     const int*      cu_q_len,
                                                     const int*      cu_k_len,
                                                     RopeKernelParam rope_param,
                                                     int64_t         stride_b,
                                                     int64_t         stride_c,
                                                     int64_t         stride_h,
                                                     int64_t         stride_s,
                                                     int64_t         output_stride_s,
                                                     int             layer_id,
                                                     BlockLayout     block_layout,
                                                     bool            use_logn_attn,
                                                     bool            is_qk_norm,
                                                     float           qk_norm_eps,
                                                     float           inv_head_dim,
                                                     const void*     q_weight,
                                                     const void*     k_weight,
                                                     int             max_position_embeddings,
                                                     float           fmha_host_bmm1_scale,
                                                     float*          qkv_scale_orig,
                                                     float*          o_scale_orig,
                                                     float*          fmha_bmm1_scale,
                                                     float*          fmha_bmm2_scale,
                                                     int*            fmha_tile_counter,
                                                     const int*      page_tables,
                                                     int             max_num_pages_per_seq_kv,
                                                     int64_t         page_stride_bytes,
                                                     int             query_group_sz)  // NEW: GQA group size
{
    constexpr int kVecSize        = sizeof(uint4) / sizeof(T);  // 8 for fp16/bf16
    constexpr int kThreadsPerHead = HeadDim / kVecSize;         // 16 for HeadDim=128, 24 for HeadDim=192
    static_assert(HeadDim % kVecSize == 0, "HeadDim must be multiple of vector size");
    static_assert(kThreadsPerHead <= WARP_SIZE, "The default decode KV update path supports one warp per head");

    constexpr unsigned kActiveMask = WarpActiveMask<kThreadsPerHead>::value;
    using Vec                      = Array<T, kVecSize>;

    const int  q_head_idx          = blockIdx.y;  // Each block handles 1 Q head
    const int  batch_idx           = blockIdx.z;
    const int  kv_head_idx         = q_head_idx / query_group_sz;  // Corresponding KV head
    const bool is_first_q_in_group = (q_head_idx % query_group_sz == 0);

    const int qi_beg      = cu_q_len[batch_idx];
    const int qi_end      = cu_q_len[batch_idx + 1];
    const int q_len       = qi_end - qi_beg;  // For decoding, q_len = 1
    const int k_len       = cu_k_len[batch_idx + 1] - cu_k_len[batch_idx];
    const int history_len = k_len - q_len;

    if (q_len == 0)
        return;

    const int tid = threadIdx.x;

    // Each thread handles one kVecSize chunk of head_dim
    // tid 0 -> di=0, tid 1 -> di=8, ..., tid 15 -> di=120 (for HeadDim=128)
    const int di = tid * kVecSize;

    // Only threads with di < HeadDim participate
    if (di >= HeadDim)
        return;

    // Load Q data for this head
    Vec vec_Q;
    {
        const int64_t q_index = (batch_idx * stride_b + qi_beg * stride_c + q_head_idx * stride_h) * HeadDim + di;
        Ldg(vec_Q, &q[q_index]);
    }

    // Load K/V data (same for all Q heads in GQA group)
    Vec vec_K, vec_V;
    {
        const int64_t kv_index = (batch_idx * stride_b + qi_beg * stride_c + kv_head_idx * stride_h) * HeadDim + di;
        Ldg(vec_K, &k[kv_index]);
        Ldg(vec_V, &v[kv_index]);
    }

    if (is_qk_norm) {
        const T* q_weight_ptr = static_cast<const T*>(q_weight);
        const T* k_weight_ptr = static_cast<const T*>(k_weight);

        Vec w_q, w_k;
        Ldg(w_q, &q_weight_ptr[di]);
        Ldg(w_k, &k_weight_ptr[di]);

        float sum_q = 0.0f, sum_k = 0.0f;
        PRAGMA_UNROLL
        for (int i = 0; i < kVecSize; ++i) {
            float qv = (float)vec_Q[i];
            float kv = (float)vec_K[i];
            sum_q    = fmaf(qv, qv, sum_q);
            sum_k    = fmaf(kv, kv, sum_k);
        }

        // Warp reduction across active lanes. kThreadsPerHead is not always a
        // power of two, e.g. head_dim=80 with fp16 uses 10 vector lanes.
        PRAGMA_UNROLL
        for (int offset = WARP_SIZE / 2; offset >= 1; offset /= 2) {
            const float q_other = __shfl_down_sync(kActiveMask, sum_q, offset);
            const float k_other = __shfl_down_sync(kActiveMask, sum_k, offset);
            if (tid + offset < kThreadsPerHead) {
                sum_q += q_other;
                sum_k += k_other;
            }
        }
        sum_q = __shfl_sync(kActiveMask, sum_q, 0);
        sum_k = __shfl_sync(kActiveMask, sum_k, 0);

        float inv_rms_q = rsqrtf(fmaf(sum_q, inv_head_dim, qk_norm_eps));
        float inv_rms_k = rsqrtf(fmaf(sum_k, inv_head_dim, qk_norm_eps));

        PRAGMA_UNROLL
        for (int i = 0; i < kVecSize; ++i) {
            float q_val   = (float)vec_Q[i];
            float w_q_val = (float)w_q[i];
            float k_val   = (float)vec_K[i];
            float w_k_val = (float)w_k[i];
            vec_Q[i]      = (T)(q_val * inv_rms_q * w_q_val);
            vec_K[i]      = (T)(k_val * inv_rms_k * w_k_val);
        }
    }

    // Step 2: RoPE
    if (rope_param.type != RopeType::kNull) {
        FastRoPE  rope(rope_param, batch_idx, std::integral_constant<int, kVecSize>{});
        const int ti = history_len;  // token position

        rope.init(di);
        rope.apply(vec_Q, ti);
        rope.apply(vec_K, ti);
    }

    // Step 3: LogN Scaling (Q only)
    if (use_logn_attn) {
        LogNScaling logn_scaling(history_len, max_position_embeddings);
        logn_scaling.apply(vec_Q);
    }

    // Step 4 & 5: Convert and update KV cache (only first Q head in GQA group)
    if (is_first_q_in_group) {
        Array<Tkv, kVecSize> out_K, out_V;

        if (qkv_scale_orig) {
            ConvertKvCache<T, Tkv> conv_K{qkv_scale_orig[1], 0.0};
            ConvertKvCache<T, Tkv> conv_V{qkv_scale_orig[2], 0.0};
            out_K = conv_K(vec_K);
            out_V = conv_V(vec_V);
        }
        else {
            out_K = ConvertKvCache<T, Tkv>::convert(vec_K);
            out_V = ConvertKvCache<T, Tkv>::convert(vec_V);
        }

        // Write to KV cache
        HeadBlockPagedKV<T, Tkv, BlockLayout> block_head{block_layout,
                                                         kv_head_idx,
                                                         page_tables,
                                                         batch_idx,
                                                         max_num_pages_per_seq_kv,
                                                         (char*)kv_cache_buffer,
                                                         page_stride_bytes};

        const int ti = history_len;  // global timestep
        block_head.with(ti, [&](auto k_cache, auto v_cache) {
            Store(&k_cache[di], out_K);
            Store(&v_cache[di], out_V);
        });
    }

    // Step 6: Write processed Q to output buffer
    {
        auto                  q_output_ptr = reinterpret_cast<TOut*>(processed_q);
        Array<TOut, kVecSize> out_Q;

        if (qkv_scale_orig) {
            ConvertKvCache<T, TOut> conv_Q{qkv_scale_orig[0], 0.0};
            out_Q = conv_Q(vec_Q);
        }
        else {
            out_Q = ConvertKvCache<T, TOut>::convert(vec_Q);
        }

        const int64_t out_index = (batch_idx * output_stride_s + q_head_idx) * HeadDim + di;
        Store(&q_output_ptr[out_index], out_Q);
    }

    initialize_fmha_scales(
        fmha_host_bmm1_scale, qkv_scale_orig, o_scale_orig, fmha_bmm1_scale, fmha_bmm2_scale, fmha_tile_counter);
}

// Legacy kernel for backward compatibility
template<class TOut, class Tkv, int CTA_S, int CTA_H, int HeadDim, int WarpCnt, class T, class BlockLayout>
__global__ void decodingQKNormRopeKVUpdate_Kernel(void*           processed_q,
                                                  void*           kv_cache_buffer,
                                                  const T*        q,
                                                  const T*        k,
                                                  const T*        v,
                                                  const int*      cu_q_len,
                                                  const int*      cu_k_len,
                                                  RopeKernelParam rope_param,
                                                  int64_t         stride_b,
                                                  int64_t         stride_c,
                                                  int64_t         stride_h,
                                                  int64_t         stride_s,
                                                  int64_t         output_stride_s,
                                                  int             layer_id,
                                                  BlockLayout     block_layout,
                                                  bool            use_logn_attn,
                                                  bool            is_qk_norm,
                                                  float           qk_norm_eps,
                                                  float           inv_head_dim,
                                                  const void*     q_weight,
                                                  const void*     k_weight,
                                                  int             max_position_embeddings,
                                                  float           fmha_host_bmm1_scale,
                                                  float*          qkv_scale_orig,
                                                  float*          o_scale_orig,
                                                  float*          fmha_bmm1_scale,
                                                  float*          fmha_bmm2_scale,
                                                  int*            fmha_tile_counter,
                                                  const int*      page_tables,
                                                  int             max_num_pages_per_seq_kv,
                                                  int64_t         page_stride_bytes,
                                                  void*           key_block_scales,
                                                  void*           value_block_scales,
                                                  int64_t         kv_scale_stride_heads,
                                                  int64_t         kv_scale_stride_batch,
                                                  int             query_group_sz,
                                                  bool            grid_by_q_head)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    using Vec = Array<T, kVecSize>;
    using Map = RakedThreadMap<HeadDim, CTA_S, kVecSize, WarpCnt>;

    constexpr int ITER_C = Map::kIterC;

    const int  token_idx     = blockIdx.x * CTA_S;  // local offset, for decoding always 0
    const int  grid_head_idx = blockIdx.y;
    const int  kv_head_idx   = grid_by_q_head ? grid_head_idx / query_group_sz : grid_head_idx;
    const int  q_head_base   = grid_by_q_head ? grid_head_idx : grid_head_idx * CTA_H;
    const bool write_kv      = !grid_by_q_head || (grid_head_idx % query_group_sz == 0);
    const int  batch_idx     = blockIdx.z;

    const int qi_beg = cu_q_len[batch_idx];
    const int qi_end = cu_q_len[batch_idx + 1];
    const int q_len  = qi_end - qi_beg;  // For decoding, q_len = 1

    const int k_len       = cu_k_len[batch_idx + 1] - cu_k_len[batch_idx];
    const int history_len = k_len - q_len;

    // Early exit for empty tiles (rarely taken in decoding since CTA_S=1)
    if (qi_beg + token_idx >= qi_end) {
        return;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int2 offset = Map::get_offset(warp_id, lane_id);

    const int qi = offset.y + token_idx;
    if (qi >= q_len)
        return;

    Vec __align__(16) vec_Q[CTA_H][ITER_C];
    Vec __align__(16) vec_K[ITER_C];
    Vec __align__(16) vec_V[ITER_C];

    // Decoding Optimization: Load QKV data
    // ITER_S=1: Only process 1 token, so s is always 0
    // Simplified loop structure for better performance
    PRAGMA_UNROLL
    for (int c = 0; c < ITER_C; ++c) {
        const int     qi = offset.y + token_idx;  // s * Map::kDeltaS = 0 for ITER_S=1
        const int     di = offset.x + c * Map::kDeltaC;
        const int64_t index =
            (batch_idx * stride_b + qi_beg * stride_c + qi * stride_s + kv_head_idx * stride_h) * HeadDim + di;
        if (qi < q_len) {
            Ldg(vec_K[c], &k[index]);
            Ldg(vec_V[c], &v[index]);
        }
        else {
            // Rarely taken in decoding
            clear(vec_K[c]);
            clear(vec_V[c]);
        }

        for (int h = 0; h < CTA_H; ++h) {
            const int64_t q_index =
                (batch_idx * stride_b + qi_beg * stride_c + qi * stride_s + (q_head_base + h) * stride_h) * HeadDim
                + di;
            if (qi < q_len) {
                Ldg(vec_Q[h][c], &q[q_index]);
            }
            else {
                clear(vec_Q[h][c]);
            }
        }
    }

    // Step 1: QK Norm - 在整个 head_dim 维度上进行归一化
    // Decoding Optimization: With 2 warps instead of 4, reduction is more efficient
    if (is_qk_norm) {
        // 类型转换为 T* 以便访问权重数据
        const T* q_weight_ptr = static_cast<const T*>(q_weight);
        const T* k_weight_ptr = static_cast<const T*>(k_weight);

        constexpr int thr_per_qk = Map::kWarpThreadC;  // 参与同一个 head_dim 归一化的线程数

        // Phase 0: 预加载权重（权重与 token 无关，只需加载一次）
        Array<T, kVecSize> w_q[ITER_C];
        Array<T, kVecSize> w_k[ITER_C];

        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            Ldg(w_q[c], &q_weight_ptr[di]);
            Ldg(w_k[c], &k_weight_ptr[di]);
        }

        float sum_q[CTA_H];
        PRAGMA_UNROLL
        for (int h = 0; h < CTA_H; ++h) {
            sum_q[h] = 0.0f;
        }
        float sum_k = 0.0f;

        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            PRAGMA_UNROLL
            for (int v = 0; v < kVecSize; ++v) {
                float k_val = (float)vec_K[c][v];
                sum_k       = fmaf(k_val, k_val, sum_k);
            }

            PRAGMA_UNROLL
            for (int h = 0; h < CTA_H; ++h) {
                PRAGMA_UNROLL
                for (int v = 0; v < kVecSize; ++v) {
                    float q_val = (float)vec_Q[h][c][v];
                    sum_q[h]    = fmaf(q_val, q_val, sum_q[h]);
                }
            }
        }

        PRAGMA_UNROLL
        for (int mask = thr_per_qk / 2; mask >= 1; mask /= 2) {
            PRAGMA_UNROLL
            for (int h = 0; h < CTA_H; ++h) {
                sum_q[h] += __shfl_xor_sync(uint32_t(-1), sum_q[h], mask);
            }
            sum_k += __shfl_xor_sync(uint32_t(-1), sum_k, mask);
        }

        float inv_rms_q[CTA_H];
        PRAGMA_UNROLL
        for (int h = 0; h < CTA_H; ++h) {
            inv_rms_q[h] = rsqrtf(fmaf(sum_q[h], inv_head_dim, qk_norm_eps));
        }
        float inv_rms_k = rsqrtf(fmaf(sum_k, inv_head_dim, qk_norm_eps));

        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            PRAGMA_UNROLL
            for (int v = 0; v < kVecSize; ++v) {
                float k_val   = (float)vec_K[c][v];
                float w_k_val = (float)w_k[c][v];
                vec_K[c][v]   = (T)(k_val * inv_rms_k * w_k_val);
            }

            PRAGMA_UNROLL
            for (int h = 0; h < CTA_H; ++h) {
                PRAGMA_UNROLL
                for (int v = 0; v < kVecSize; ++v) {
                    float q_val    = (float)vec_Q[h][c][v];
                    float w_q_val  = (float)w_q[c][v];
                    vec_Q[h][c][v] = (T)(q_val * inv_rms_q[h] * w_q_val);
                }
            }
        }
    }

    // Step 2: RoPE计算
    // ITER_S=1: Only 1 token, s=0
    if (rope_param.type != RopeType::kNull) {
        FastRoPE  rope(rope_param, batch_idx, std::integral_constant<int, kVecSize>{});
        const int ti = history_len + offset.y + token_idx;  // s * Map::kDeltaS = 0

        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            rope.init(di);
            rope.apply(vec_K[c], ti);

            PRAGMA_UNROLL
            for (int h = 0; h < CTA_H; ++h) {
                rope.apply(vec_Q[h][c], ti);
            }
        }
    }

    // Step 3: LogN Scaling计算
    // ITER_S=1: Only 1 token, s=0
    if (use_logn_attn) {
        const int   ti = history_len + offset.y + token_idx;  // s * Map::kDeltaS = 0
        LogNScaling logn_scaling(ti, max_position_embeddings);

        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            PRAGMA_UNROLL
            for (int h = 0; h < CTA_H; ++h) {
                logn_scaling.apply(vec_Q[h][c]);
            }
        }
    }

    // Step 5: KVCache Quant and Update
    // ITER_S=1: Only 1 token, s=0, qi=0 for decoding
    if (write_kv) {
        HeadBlockPagedKV<T, Tkv, BlockLayout> block_head{block_layout,
                                                         kv_head_idx,
                                                         page_tables,
                                                         batch_idx,
                                                         max_num_pages_per_seq_kv,
                                                         (char*)kv_cache_buffer,
                                                         page_stride_bytes,
                                                         key_block_scales,
                                                         value_block_scales,
                                                         kv_scale_stride_heads,
                                                         kv_scale_stride_batch,
                                                         HeadDim / 16};

        const int qi = offset.y + token_idx;  // s * Map::kDeltaS = 0
        if (qi < q_len) {
            const int ti = history_len + qi;  // global timestep
            block_head.with_scales(ti, [&](auto k_cache, auto v_cache, auto k_scale, auto v_scale, int block_ti) {
                PRAGMA_UNROLL
                for (int c = 0; c < ITER_C; ++c) {
                    int        di          = offset.x + c * Map::kDeltaC;
                    const int  vec_idx     = di / kVecSize;
                    const int  scale_idx   = di / 16;
                    const int  k_scale_idx = fp4_k_scale_index(block_ti, scale_idx, HeadDim / 16);
                    const int  v_scale_idx = fp4_v_scale_index(block_ti, scale_idx, HeadDim / 16);
                    const auto out_K       = convert_kv_cache<T, Tkv>(
                        vec_K[c], qkv_scale_orig ? qkv_scale_orig[1] : 1.f, k_scale, k_scale_idx, vec_idx);
                    const auto out_V = convert_kv_cache<T, Tkv>(
                        vec_V[c], qkv_scale_orig ? qkv_scale_orig[2] : 1.f, v_scale, v_scale_idx, vec_idx);
                    Store(&k_cache[di], out_K);
                    Store(&v_cache[di], out_V);
                }
            });
        }
    }

    // Step 6: Write processed Q back to linear buffer
    // Decoding Optimization: Only write Q, skip K/V writes since they're already in KV cache
    // ITER_S=1: Only 1 token, s=0
    auto q_output_ptr = reinterpret_cast<TOut*>(processed_q);

    // Output to regular 4D tensor: [batch_size, 1, num_q_heads, head_dim]

    Array<TOut, kVecSize> out_Q[CTA_H][ITER_C];

    // Step 4: 转换QKV的数据类型
    // ITER_S=1: Only 1 token, s=0
    if (qkv_scale_orig) {
        ConvertKvCache<T, TOut> conv_Q{qkv_scale_orig[0], 0.0};

        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            PRAGMA_UNROLL
            for (int h = 0; h < CTA_H; ++h) {
                out_Q[h][c] = conv_Q(vec_Q[h][c]);
            }
        }
    }
    else {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            PRAGMA_UNROLL
            for (int h = 0; h < CTA_H; ++h) {
                out_Q[h][c] = ConvertKvCache<T, TOut>::convert(vec_Q[h][c]);
            }
        }
    }

    PRAGMA_UNROLL
    for (int c = 0; c < ITER_C; ++c) {
        const int di = offset.x + c * Map::kDeltaC;
        for (int h = 0; h < CTA_H; ++h) {
            const int64_t index = (batch_idx * output_stride_s + q_head_base + h) * HeadDim + di;
            Store(&q_output_ptr[index], out_Q[h][c]);
        }
    }

    initialize_fmha_scales(
        fmha_host_bmm1_scale, qkv_scale_orig, o_scale_orig, fmha_bmm1_scale, fmha_bmm2_scale, fmha_tile_counter);
}

template<class T>
void invokeDecodingQKNormRopeKVUpdate_(const AttentionParams<T>& params)
{
    constexpr int WARPS = 2;
    constexpr int CTA_S = 1;

    int  grouped_block = WARPS * WARP_SIZE;
    dim3 grouped_grid((params.max_q_len + CTA_S - 1) / CTA_S, params.num_kv_heads, params.batch_size);
    dim3 default_block(grouped_block, 1, 1);
    dim3 default_grid(1, params.num_heads, params.batch_size);

    // Calculate strides
    const int64_t stride_b        = 0;                                     // not used
    const int64_t stride_c        = params.stride / params.size_per_head;  // num_heads for input QKV
    const int64_t stride_h        = 1;                                     // heads are contiguous
    const int64_t stride_s        = params.stride / params.size_per_head;
    const int64_t output_stride_s = params.num_heads;

    FT_CHECK_WITH_INFO(params.num_kv_heads > 0 && params.num_heads % params.num_kv_heads == 0,
                       "FlashInfer TRTLLM FMHA KV update expects num_heads to be divisible by num_kv_heads");

    const int   query_group_sz = params.num_heads / params.num_kv_heads;
    const float inv_head_dim   = 1.f / params.size_per_head;

    auto invoke_grouped = [&](auto tkv, auto tout, const auto dim, const auto cta_h) {
        using Tkv  = decltype(tkv);
        using TOut = decltype(tout);

        constexpr int kHeadDim = dim;
        FT_CHECK(params.size_per_head == kHeadDim);

        constexpr int CTA_H = cta_h;

        block::Layout<block::Config<T, Tkv, kHeadDim, false>> block_layout{
            block::Config<T, Tkv, kHeadDim, false>{params.num_kv_heads, params.block_iter_params.block_len}};
        const int64_t page_stride_bytes = params.fmha.kv_cache_page_stride_bytes ?
                                              params.fmha.kv_cache_page_stride_bytes :
                                              block_layout.block_size(params.fmha.layer_num);

        decodingQKNormRopeKVUpdate_Kernel<TOut, Tkv, CTA_S, CTA_H, kHeadDim, WARPS>
            <<<grouped_grid, grouped_block, 0, params.stream>>>((void*)params.q,  // processed_q output
                                                                (void*)params.fmha.kv_cache_pool_ptr,
                                                                params.fmha.raw_q,
                                                                params.fmha.raw_k,
                                                                params.fmha.raw_v,
                                                                params.cu_q_len,
                                                                params.cu_k_len,
                                                                params.rope_param,
                                                                stride_b,
                                                                stride_c,
                                                                stride_h,
                                                                stride_s,
                                                                output_stride_s,
                                                                params.block_iter_params.layer_id,
                                                                block_layout,
                                                                params.use_logn_attn,
                                                                params.fmha.is_qk_norm,
                                                                params.fmha.qk_norm_eps,
                                                                inv_head_dim,
                                                                params.fmha.q_weight,
                                                                params.fmha.k_weight,
                                                                params.max_position_embeddings,
                                                                params.fmha.host_bmm1_scale,
                                                                params.fmha.qkv_scale_orig,
                                                                params.fmha.o_scale_orig,
                                                                params.fmha.scale_bmm1_ptr,
                                                                params.fmha.scale_bmm2_ptr,
                                                                params.fmha.fmha_tile_counter,
                                                                params.fmha.page_tables,
                                                                params.fmha.max_num_pages_per_seq_kv,
                                                                page_stride_bytes,
                                                                params.fmha.key_block_scales,
                                                                params.fmha.value_block_scales,
                                                                params.fmha.kv_scale_stride_heads,
                                                                params.fmha.kv_scale_stride_batch,
                                                                query_group_sz,
                                                                false);
    };

    auto invoke_default = [&](auto tkv, auto tout, const auto dim) {
        using Tkv  = decltype(tkv);
        using TOut = decltype(tout);

        constexpr int kHeadDim = dim;
        FT_CHECK(params.size_per_head == kHeadDim);

        block::Layout<block::Config<T, Tkv, kHeadDim, false>> block_layout{
            block::Config<T, Tkv, kHeadDim, false>{params.num_kv_heads, params.block_iter_params.block_len}};
        const int64_t page_stride_bytes = params.fmha.kv_cache_page_stride_bytes ?
                                              params.fmha.kv_cache_page_stride_bytes :
                                              block_layout.block_size(params.fmha.layer_num);

        decodingQKNormRopeKVUpdate_Kernel<TOut, Tkv, CTA_S, 1, kHeadDim, WARPS>
            <<<default_grid, default_block, 0, params.stream>>>((void*)params.q,
                                                                (void*)params.fmha.kv_cache_pool_ptr,
                                                                params.fmha.raw_q,
                                                                params.fmha.raw_k,
                                                                params.fmha.raw_v,
                                                                params.cu_q_len,
                                                                params.cu_k_len,
                                                                params.rope_param,
                                                                stride_b,
                                                                stride_c,
                                                                stride_h,
                                                                stride_s,
                                                                output_stride_s,
                                                                params.block_iter_params.layer_id,
                                                                block_layout,
                                                                params.use_logn_attn,
                                                                params.fmha.is_qk_norm,
                                                                params.fmha.qk_norm_eps,
                                                                inv_head_dim,
                                                                params.fmha.q_weight,
                                                                params.fmha.k_weight,
                                                                params.max_position_embeddings,
                                                                params.fmha.host_bmm1_scale,
                                                                params.fmha.qkv_scale_orig,
                                                                params.fmha.o_scale_orig,
                                                                params.fmha.scale_bmm1_ptr,
                                                                params.fmha.scale_bmm2_ptr,
                                                                params.fmha.fmha_tile_counter,
                                                                params.fmha.page_tables,
                                                                params.fmha.max_num_pages_per_seq_kv,
                                                                page_stride_bytes,
                                                                params.fmha.key_block_scales,
                                                                params.fmha.value_block_scales,
                                                                params.fmha.kv_scale_stride_heads,
                                                                params.fmha.kv_scale_stride_batch,
                                                                query_group_sz,
                                                                true);
    };

    auto dispatch = [&](auto tkv, auto tout) {
        DispatchKvUpdateHeadDim(params.size_per_head, [&](auto dim) {
            constexpr int kHeadDim = decltype(dim)::value;
            if constexpr (kHeadDim == 576) {
                return invoke_default(tkv, tout, dim);
            }
            if (query_group_sz == 1) {
                return invoke_grouped(tkv, tout, dim, std::integral_constant<int, 1>{});
            }
            if (query_group_sz == 2) {
                return invoke_grouped(tkv, tout, dim, std::integral_constant<int, 2>{});
            }
            if (query_group_sz == 4) {
                return invoke_grouped(tkv, tout, dim, std::integral_constant<int, 4>{});
            }
            if (query_group_sz == 5) {
                return invoke_grouped(tkv, tout, dim, std::integral_constant<int, 5>{});
            }
            if (query_group_sz == 8) {
                return invoke_grouped(tkv, tout, dim, std::integral_constant<int, 8>{});
            }
            return invoke_default(tkv, tout, dim);
        });
    };

#ifdef ENABLE_FP8
    if (params.quant_policy & QuantPolicy::kCacheKVFP8) {
        dispatch(__nv_fp8_e4m3{}, __nv_fp8_e4m3{});
    }
#ifdef ENABLE_FP4
    else if (params.quant_policy & QuantPolicy::kCacheKVFP4) {
        dispatch(fp4_e2m1_t{}, __nv_fp8_e4m3{});
    }
#endif
    else
#endif
    {
        dispatch(T{}, T{});
    }
}

#define INSTANTIATE_invokeDecodingQKNormRopeKVUpdate(type)                                                             \
    template void invokeDecodingQKNormRopeKVUpdate_(const AttentionParams<type>& params);

INSTANTIATE_invokeDecodingQKNormRopeKVUpdate(half);
#if ENABLE_BF16
INSTANTIATE_invokeDecodingQKNormRopeKVUpdate(nv_bfloat16);
#endif

// ----------------------------------------------------------------------------
// Append KV writer.
// ----------------------------------------------------------------------------

// Token-level load balancing inspired by TensorRT-LLM updateKVCacheV2Prefill.
// Uses token2batch[global_token_idx * 2 + {0,1}] to map a flattened token
// to (batch_id, relative token position).
template<class TOut, class Tkv, int BlockSize, int HeadDim, class T, class BlockLayout>
__global__ void appendQKNormRopeKVUpdate_Kernel(void* __restrict__ processed_q,
                                                void* __restrict__ kv_cache_buffer,
                                                const T* __restrict__ q,
                                                const T* __restrict__ k,
                                                const T* __restrict__ v,
                                                const int* __restrict__ cu_q_len,
                                                const int* __restrict__ cu_k_len,
                                                const int* __restrict__ token2batch,
                                                RopeKernelParam rope_param,
                                                int64_t         stride_b,
                                                int64_t         stride_c,
                                                int64_t         stride_h,
                                                int64_t         stride_s,
                                                int64_t         output_stride_s,
                                                int             layer_id,
                                                BlockLayout     block_layout,
                                                float           fmha_host_bmm1_scale,
                                                float* __restrict__ qkv_scale_orig,
                                                float* __restrict__ o_scale_orig,
                                                float* __restrict__ fmha_bmm1_scale,
                                                float* __restrict__ fmha_bmm2_scale,
                                                int* __restrict__ fmha_tile_counter,
                                                const int* __restrict__ page_tables,
                                                int     max_num_pages_per_seq_kv,
                                                int64_t page_stride_bytes,
                                                void*   key_block_scales,
                                                void*   value_block_scales,
                                                int64_t kv_scale_stride_heads,
                                                int64_t kv_scale_stride_batch,
                                                int     token_nums,
                                                int     query_group_sz)
{
    // Constants (all compile-time)
    constexpr int kVecSize        = sizeof(uint4) / sizeof(T);   // 16 bytes / sizeof(T), e.g., 8 for half
    constexpr int kVectorPerHead  = HeadDim / kVecSize;          // Vectors needed per head, e.g., 128/8=16
    constexpr int kTokensPerBlock = BlockSize / kVectorPerHead;  // Tokens processed per CUDA block, e.g., 256/16=16

    static_assert(HeadDim % kVecSize == 0, "HeadDim must be multiple of kVecSize");
    static_assert(BlockSize % kVectorPerHead == 0, "BlockSize must be multiple of kVectorPerHead");

    using Vec = Array<T, kVecSize>;

    // Thread organization (similar to TensorRT-LLM)
    const int  q_head_idx          = blockIdx.y;                   // Q head index (gridDim.y = num_heads)
    const int  kv_head_idx         = q_head_idx / query_group_sz;  // Derive KV head from Q head (GQA)
    const bool is_first_q_in_group = (q_head_idx % query_group_sz == 0);

    // Vector index within head (which vector of the head this thread handles)
    const int head_dim_vec_idx = threadIdx.x % kVectorPerHead;
    // Element index within head
    const int di = head_dim_vec_idx * kVecSize;

    // Make sure all threads in the block process aligned tokens (avoid syncthreads deadlock)
    const int tokens_loop_end = ((token_nums + kTokensPerBlock - 1) / kTokensPerBlock) * kTokensPerBlock;

    // Grid-stride loop over tokens (key for load balancing)
    // Each iteration, this thread processes tokens at indices:
    //   - (threadIdx.x / kVectorPerHead) + blockIdx.x * kTokensPerBlock
    //   - (threadIdx.x / kVectorPerHead) + (blockIdx.x + gridDim.x) * kTokensPerBlock
    //   - ...
    for (int global_token_idx = (threadIdx.x / kVectorPerHead) + blockIdx.x * kTokensPerBlock;
         global_token_idx < tokens_loop_end;
         global_token_idx += kTokensPerBlock * gridDim.x) {

        // Check if this is a valid token BEFORE accessing token2batch
        const bool is_valid_idx = (global_token_idx < token_nums);

        // Bound the token index for safe memory access (out-of-bound threads use last valid token)
        const int bounded_token_idx = min(global_token_idx, token_nums - 1);

        // Get batch_id and relative token position from token2batch
        // token2batch layout: [batch_id, relative_token_idx, batch_id, relative_token_idx, ...]
        // Only access if within valid range (to avoid reading uninitialized memory)
        const int batch_idx = token2batch[bounded_token_idx * 2 + 0];  // Global batch ID
        const int qi        = token2batch[bounded_token_idx * 2 + 1];  // Token index within batch

        const int qi_beg = cu_q_len[batch_idx];
        const int qi_end = cu_q_len[batch_idx + 1];
        const int q_len  = qi_end - qi_beg;

        const int k_len       = cu_k_len[batch_idx + 1] - cu_k_len[batch_idx];
        const int history_len = k_len - q_len;

        // Final validity check: is this a real token (not padding for alignment)?
        const bool valid_token = is_valid_idx && (qi < q_len);

        // Global timestep for RoPE
        const int ti = history_len + qi;

        // Calculate linear indices for Q/K/V
        // Layout: [batch, context, seq, head, head_dim]
        const int64_t kv_index =
            (batch_idx * stride_b + qi_beg * stride_c + qi * stride_s + kv_head_idx * stride_h) * HeadDim + di;
        const int64_t q_index =
            (batch_idx * stride_b + qi_beg * stride_c + qi * stride_s + q_head_idx * stride_h) * HeadDim + di;

        // Load QKV vectors
        Vec __align__(16) vec_Q;
        Vec __align__(16) vec_K;
        Vec __align__(16) vec_V;

        if (valid_token) {
            Ldg(vec_Q, &q[q_index]);
            Ldg(vec_K, &k[kv_index]);
            Ldg(vec_V, &v[kv_index]);
        }
        else {
            clear(vec_Q);
            clear(vec_K);
            clear(vec_V);
        }

        // Initialize RoPE for this batch and head dimension
        FastRoPE rope(rope_param, batch_idx, std::integral_constant<int, kVecSize>{});
        rope.init(di);

        // Apply RoPE to Q and K (no need for syncthreads - each thread is independent)
        if (rope_param.type != RopeType::kNull && valid_token) {
            rope.apply(vec_Q, ti);
            rope.apply(vec_K, ti);
        }

        // Convert Q to output type
        Array<TOut, kVecSize> out_Q;
        if (qkv_scale_orig) {
            ConvertKvCache<T, TOut> conv_Q{qkv_scale_orig[0], 0.0};
            out_Q = conv_Q(vec_Q);
        }
        else {
            out_Q = ConvertKvCache<T, TOut>::convert(vec_Q);
        }

        // Write processed Q to output buffer
        if (valid_token) {
            auto q_output_ptr = reinterpret_cast<TOut*>(processed_q);
            // Use output_stride_s for output buffer, not input stride_s
            // Output layout: [B, S, H, D]
            // Use qi_beg + qi so the output is contiguous in flattened-token order.
            const int64_t out_q_index =
                (batch_idx * 0 + (qi_beg + qi) * output_stride_s + q_head_idx * 1) * HeadDim + di;
            Store(&q_output_ptr[out_q_index], out_Q);
        }

        // Update KV cache (only first Q head in GQA group to avoid redundant writes)
        if (is_first_q_in_group && valid_token) {
            HeadBlockPagedKV<T, Tkv, BlockLayout> block_head{block_layout,
                                                             kv_head_idx,
                                                             page_tables,
                                                             batch_idx,
                                                             max_num_pages_per_seq_kv,
                                                             (char*)kv_cache_buffer,
                                                             page_stride_bytes,
                                                             key_block_scales,
                                                             value_block_scales,
                                                             kv_scale_stride_heads,
                                                             kv_scale_stride_batch,
                                                             HeadDim / 16};

            block_head.with_scales(ti, [&](auto k_cache, auto v_cache, auto k_scale, auto v_scale, int block_ti) {
                const int  scale_idx   = di / 16;
                const int  k_scale_idx = fp4_k_scale_index(block_ti, scale_idx, HeadDim / 16);
                const int  v_scale_idx = fp4_v_scale_index(block_ti, scale_idx, HeadDim / 16);
                const auto out_K       = convert_kv_cache<T, Tkv>(
                    vec_K, qkv_scale_orig ? qkv_scale_orig[1] : 1.f, k_scale, k_scale_idx, head_dim_vec_idx);
                const auto out_V = convert_kv_cache<T, Tkv>(
                    vec_V, qkv_scale_orig ? qkv_scale_orig[2] : 1.f, v_scale, v_scale_idx, head_dim_vec_idx);
                Store(&k_cache[di], out_K);
                Store(&v_cache[di], out_V);
            });
        }
    }

    initialize_fmha_scales(
        fmha_host_bmm1_scale, qkv_scale_orig, o_scale_orig, fmha_bmm1_scale, fmha_bmm2_scale, fmha_tile_counter);
}

inline void calGridSizeWithBestEfficiency(
    dim3 const block, dim3& grid, int numNeededBlockX, int multiProcessorCount, int availableThreadsPerSm)
{
    int numThreadsPerBlock        = block.x * block.y * block.z;
    int numAvailableBlocksPerWave = int(availableThreadsPerSm / numThreadsPerBlock) * multiProcessorCount;
    int numBatchBlocks            = grid.y * grid.z;

    // The best wave efficiency it can achieve with different number of blocks.
    float bestEfficiency = 0.f;
    int   selectedBlockX = 0;
    // Iterate over all possible number of blocks in the x dimension.
    for (int blockX = 1; blockX <= numNeededBlockX; ++blockX) {
        float numWaves   = float(blockX * numBatchBlocks) / numAvailableBlocksPerWave;
        float efficiency = numWaves / std::ceil(numWaves);
        if (efficiency > bestEfficiency) {
            bestEfficiency = efficiency;
            selectedBlockX = blockX;
        }
    }
    // Update grid.x.
    grid.x = selectedBlockX;
}

template<class T>
void invokeAppendQKNormRopeKVUpdate_(const AttentionParams<T>& params)
{
    // Calculate strides
    const int64_t stride_b = 0;                                     // stride b
    const int64_t stride_c = params.stride / params.size_per_head;  // stride c
    const int64_t stride_h = 1;                                     // stride h
    const int64_t stride_s = params.stride / params.size_per_head;  // stride s

    // Append: Output Q stride is different from input QKV stride
    // Input:  stride = (num_q_heads + 2*num_kv_heads) * head_dim  -> stride_s = num_q_heads + 2*num_kv_heads
    // Output: only Q, so stride = num_q_heads * head_dim          -> output_stride_s = num_q_heads
    const int64_t output_stride_s = params.num_heads;  // Only Q heads, no K/V

    FT_CHECK_WITH_INFO(params.num_kv_heads > 0 && params.num_heads % params.num_kv_heads == 0,
                       "FlashInfer TRTLLM FMHA KV update expects num_heads to be divisible by num_kv_heads");

    const int query_group_sz = params.num_heads / params.num_kv_heads;

    auto invoke = [&](auto tkv, auto tout, const auto dim) {
        using Tkv  = decltype(tkv);
        using TOut = decltype(tout);

        constexpr int kHeadDim = dim;
        FT_CHECK(params.size_per_head == kHeadDim);

        block::Layout<block::Config<T, Tkv, kHeadDim, false>> block_layout{
            block::Config<T, Tkv, kHeadDim, false>{params.num_kv_heads, params.block_iter_params.block_len}};
        const int64_t page_stride_bytes = params.fmha.kv_cache_page_stride_bytes ?
                                              params.fmha.kv_cache_page_stride_bytes :
                                              block_layout.block_size(params.fmha.layer_num);

        auto output_q = params.q;

        // Grid-stride over flattened tokens; each thread handles one vector element.

        // Choose BlockSize based on HeadDim to ensure divisibility by
        // kVectorPerHead = HeadDim / kVecSize.
        constexpr int kVecSize   = sizeof(uint4) / sizeof(T);
        constexpr int kBlockSize = AppendKvUpdateBlockSize<kHeadDim, kVecSize>::value;

        const int vecs_per_head         = params.size_per_head / kVecSize;
        const int tokens_per_cuda_block = kBlockSize / vecs_per_head;

        // Block: kBlockSize threads, where each handles one vector element
        // Grid: grid.x covers tokens (with grid-stride loop), grid.y = num_heads
        auto block = dim3(kBlockSize, 1, 1);
        auto grid  = dim3(1, params.num_heads, 1);  // grid.x will be set by calGridSizeWithBestEfficiency

        // Calculate optimal grid.x for wave efficiency.
        // Global token index spans all batches, so use total token_num.
        const int num_needed_blocks_x = (params.token_num + tokens_per_cuda_block - 1) / tokens_per_cuda_block;

        // Was previously hardcoded for B200 (192 SMs, 2048 threads/SM),
        // which silently yielded suboptimal grid sizing on B300 / future
        // SKUs.  Helpers in `utils/cuda_utils.h` query
        // cudaDevAttrMultiProcessorCount + MaxThreadsPerMultiProcessor.
        // Cache once per process to avoid the per-call CUDA driver hop.
        static const int kMultiProcessorCount   = getSMCount();
        static const int kAvailableThreadsPerSm = getMaxThreadsPerSM();

        calGridSizeWithBestEfficiency(block, grid, num_needed_blocks_x, kMultiProcessorCount, kAvailableThreadsPerSm);

        appendQKNormRopeKVUpdate_Kernel<TOut, Tkv, kBlockSize, kHeadDim>
            <<<grid, block, 0, params.stream>>>((void*)output_q,
                                                (void*)params.fmha.kv_cache_pool_ptr,
                                                params.fmha.raw_q,
                                                params.fmha.raw_k,
                                                params.fmha.raw_v,
                                                params.cu_q_len,
                                                params.cu_k_len,
                                                params.fmha.token2batch,
                                                params.rope_param,
                                                stride_b,
                                                stride_c,
                                                stride_h,
                                                stride_s,
                                                output_stride_s,
                                                params.block_iter_params.layer_id,
                                                block_layout,
                                                params.fmha.host_bmm1_scale,
                                                params.fmha.qkv_scale_orig,
                                                params.fmha.o_scale_orig,
                                                params.fmha.scale_bmm1_ptr,
                                                params.fmha.scale_bmm2_ptr,
                                                params.fmha.fmha_tile_counter,
                                                params.fmha.page_tables,
                                                params.fmha.max_num_pages_per_seq_kv,
                                                page_stride_bytes,
                                                params.fmha.key_block_scales,
                                                params.fmha.value_block_scales,
                                                params.fmha.kv_scale_stride_heads,
                                                params.fmha.kv_scale_stride_batch,
                                                params.token_num,  // token_num: total tokens across all batches
                                                query_group_sz);
    };
    auto dispatch = [&](auto tkv, auto tout) {
        DispatchKvUpdateHeadDim(params.size_per_head, [&](auto dim) { return invoke(tkv, tout, dim); });
    };

#ifdef ENABLE_FP8
    if (params.quant_policy & QuantPolicy::kCacheKVFP8) {
        dispatch(__nv_fp8_e4m3{}, __nv_fp8_e4m3{});
    }
#ifdef ENABLE_FP4
    else if (params.quant_policy & QuantPolicy::kCacheKVFP4) {
        dispatch(fp4_e2m1_t{}, __nv_fp8_e4m3{});
    }
#endif
    else
#endif
    {
        dispatch(T{}, T{});
    }
}

#define INSTANTIATE_invokeAppendQKNormRopeKVUpdate(type)                                                               \
    template void invokeAppendQKNormRopeKVUpdate_(const AttentionParams<type>& params);

INSTANTIATE_invokeAppendQKNormRopeKVUpdate(half);
#if ENABLE_BF16
INSTANTIATE_invokeAppendQKNormRopeKVUpdate(nv_bfloat16);
#endif
}  // namespace turbomind
