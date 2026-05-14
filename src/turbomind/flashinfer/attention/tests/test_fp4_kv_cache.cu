// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "src/turbomind/flashinfer/attention/flashinfer_fmha_wrapper.h"
#include "src/turbomind/flashinfer/attention/kv_cache_utils_v3.h"
#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/models/llama/llama_rope.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace {

#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t e = (call);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(e) << "\n";    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

constexpr int kBatchSize = 2;
constexpr int kSeqLen    = 4;
constexpr int kQHeads    = 2;
constexpr int kKvHeads   = 1;
constexpr int kHeadDim   = 128;
constexpr int kPageSize  = 32;
constexpr int kLayerNum  = 1;

using T = half;

enum class QMode {
    kZero,
    kNonZero,
};

struct TestCase {
    const char* name;
    QMode       q_mode;
    float       quant_tol;
    bool        require_byte_exact;
};

float fp8_e4m3_to_float_host(uint8_t raw)
{
    __nv_fp8_e4m3 fp8;
    fp8.__x = raw;
    return static_cast<float>(fp8);
}

uint8_t fp8_e4m3_from_float_host(float val)
{
    __nv_fp8_e4m3 fp8{val};
    return fp8.__x;
}

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t raw)
{
    __nv_fp8_e4m3 fp8;
    fp8.__x = raw;
    return static_cast<float>(fp8);
}

__device__ __forceinline__ float fp4_e2m1_lut(uint8_t raw)
{
    constexpr float lut[16] = {
        0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return lut[raw & 0xF];
}

template<bool IsValue>
__global__ void dequant_fp4_cache_kernel(const uint8_t* __restrict__ cache,
                                         const uint8_t* __restrict__ scales,
                                         T* __restrict__ out,
                                         int   page_stride_bytes,
                                         int   head_data_bytes,
                                         int   scale_stride_batch,
                                         int   scale_stride_head,
                                         float global_scale)
{
    constexpr int kElemsPerPack = 8;
    constexpr int kScaleDim     = kHeadDim / 16;

    const int pack_id     = blockIdx.x * blockDim.x + threadIdx.x;
    const int packs_total = kBatchSize * kSeqLen * kKvHeads * (kHeadDim / kElemsPerPack);
    if (pack_id >= packs_total) {
        return;
    }

    int       tmp = pack_id;
    const int d8  = tmp % (kHeadDim / kElemsPerPack);
    tmp /= (kHeadDim / kElemsPerPack);
    const int head = tmp % kKvHeads;
    tmp /= kKvHeads;
    const int tok = tmp % kSeqLen;
    const int b   = tmp / kSeqLen;
    const int d   = d8 * kElemsPerPack;

    const int page        = b;
    const int byte_offset = page * page_stride_bytes + head * 2 * head_data_bytes + (IsValue ? head_data_bytes : 0)
                            + tok * (kHeadDim / 2) + d / 2;

    const uint32_t packed = *reinterpret_cast<const uint32_t*>(cache + byte_offset);

    const int scale_idx = d / 16;
    const int token_scale_idx =
        IsValue ? (tok / 4) * 4 * kScaleDim + scale_idx * 4 + (tok % 4) : tok * kScaleDim + scale_idx;
    const int   scale_offset = page * scale_stride_batch + head * scale_stride_head + token_scale_idx;
    const float scale        = fp8_e4m3_to_float(scales[scale_offset]) * global_scale;

    const int out_base = ((b * kSeqLen + tok) * kKvHeads + head) * kHeadDim + d;
#pragma unroll
    for (int i = 0; i < kElemsPerPack; ++i) {
        const uint8_t nibble = (packed >> (4 * i)) & 0xF;
        out[out_base + i]    = __float2half(fp4_e2m1_lut(nibble) * scale);
    }
}

float fp4_exact_pattern(int d, int salt)
{
    static constexpr float vals[16] = {
        6.f, -6.f, 4.f, -4.f, 3.f, -3.f, 2.f, -2.f, 1.5f, -1.5f, 1.f, -1.f, 0.5f, -0.5f, 0.f, 0.f};
    static constexpr float amps[4]   = {0.5f, 1.f, 2.f, 4.f};
    const int              scale_idx = d / 16;
    const float            amp       = amps[(salt + scale_idx) & 3];
    return vals[(d + salt) & 15] * amp;
}

int qkv_index(int b, int s, int h, int d)
{
    constexpr int kQkvHeads = kQHeads + 2 * kKvHeads;
    return ((b * kSeqLen + s) * kQkvHeads + h) * kHeadDim + d;
}

int kv_index(int b, int s, int h, int d)
{
    return ((b * kSeqLen + s) * kKvHeads + h) * kHeadDim + d;
}

void fill_qkv(std::vector<T>& qkv, QMode q_mode)
{
    for (int b = 0; b < kBatchSize; ++b) {
        for (int s = 0; s < kSeqLen; ++s) {
            for (int h = 0; h < kQHeads; ++h) {
                for (int d = 0; d < kHeadDim; ++d) {
                    const float q              = q_mode == QMode::kZero ? 0.f : 0.01f * static_cast<float>((d % 7) - 3);
                    qkv[qkv_index(b, s, h, d)] = __float2half(q);
                }
            }
            for (int h = 0; h < kKvHeads; ++h) {
                for (int d = 0; d < kHeadDim; ++d) {
                    qkv[qkv_index(b, s, kQHeads + h, d)] = __float2half(fp4_exact_pattern(d, b + s + h));
                    qkv[qkv_index(b, s, kQHeads + kKvHeads + h, d)] =
                        __float2half(fp4_exact_pattern(d, b * 3 + s + h + 5));
                }
            }
        }
    }
}

bool compare_tensor(const char* name, const std::vector<T>& got, const std::vector<T>& qkv, bool is_value)
{
    float max_abs_err = 0.f;
    int   bad_count   = 0;
    for (int b = 0; b < kBatchSize; ++b) {
        for (int s = 0; s < kSeqLen; ++s) {
            for (int h = 0; h < kKvHeads; ++h) {
                for (int d = 0; d < kHeadDim; ++d) {
                    const int   qh  = kQHeads + (is_value ? kKvHeads : 0) + h;
                    const float ref = __half2float(qkv[qkv_index(b, s, qh, d)]);
                    const float val = __half2float(got[kv_index(b, s, h, d)]);
                    const float err = std::fabs(ref - val);
                    max_abs_err     = std::max(max_abs_err, err);
                    if (err > 1e-2f) {
                        if (bad_count < 8) {
                            std::cerr << name << " mismatch b=" << b << " s=" << s << " h=" << h << " d=" << d
                                      << " ref=" << ref << " got=" << val << " err=" << err << "\n";
                        }
                        ++bad_count;
                    }
                }
            }
        }
    }
    std::cout << name << " max_abs_err=" << max_abs_err << " bad_count=" << bad_count << "\n";
    return bad_count == 0;
}

std::vector<T> reference_context_attention(const std::vector<uint8_t>& processed_q,
                                           const std::vector<T>&       k_cache,
                                           const std::vector<T>&       v_cache,
                                           float                       q_global_scale)
{
    std::vector<T>     out(kBatchSize * kSeqLen * kQHeads * kHeadDim);
    std::vector<float> scores(kSeqLen);

    for (int b = 0; b < kBatchSize; ++b) {
        for (int q = 0; q < kSeqLen; ++q) {
            for (int h = 0; h < kQHeads; ++h) {
                const int kv_h      = h / (kQHeads / kKvHeads);
                float     max_score = -std::numeric_limits<float>::infinity();
                for (int k = 0; k < kSeqLen; ++k) {
                    if (k > q) {
                        scores[k] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    float dot = 0.f;
                    for (int d = 0; d < kHeadDim; ++d) {
                        const int   q_idx = ((b * kSeqLen + q) * kQHeads + h) * kHeadDim + d;
                        const float q_val = fp8_e4m3_to_float_host(processed_q[q_idx]) * q_global_scale;
                        const float k_val = __half2float(k_cache[kv_index(b, k, kv_h, d)]);
                        dot += q_val * k_val;
                    }
                    scores[k] = dot / std::sqrt(static_cast<float>(kHeadDim));
                    max_score = std::max(max_score, scores[k]);
                }

                float denom = 0.f;
                for (int k = 0; k <= q; ++k) {
                    scores[k] = std::exp(scores[k] - max_score);
                    denom += scores[k];
                }

                for (int d = 0; d < kHeadDim; ++d) {
                    float val = 0.f;
                    for (int k = 0; k <= q; ++k) {
                        const float prob  = scores[k] / denom;
                        const float v_val = __half2float(v_cache[kv_index(b, k, kv_h, d)]);
                        val += prob * v_val;
                    }
                    out[((b * kSeqLen + q) * kQHeads + h) * kHeadDim + d] = __float2half(val);
                }
            }
        }
    }
    return out;
}

bool compare_attention_output(const std::vector<uint8_t>& got,
                              const std::vector<T>&       ref,
                              float                       o_global_scale,
                              float                       quant_tol,
                              bool                        require_byte_exact)
{
    float max_raw_abs_err   = 0.f;
    float max_quant_abs_err = 0.f;
    int   byte_diff_count   = 0;
    int   bad_count         = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float   g             = fp8_e4m3_to_float_host(got[i]) * o_global_scale;
        const float   r             = __half2float(ref[i]);
        const uint8_t rq_raw        = fp8_e4m3_from_float_host(r / o_global_scale);
        const float   rq            = fp8_e4m3_to_float_host(rq_raw) * o_global_scale;
        const float   raw_abs_err   = std::fabs(g - r);
        const float   quant_abs_err = std::fabs(g - rq);
        max_raw_abs_err             = std::max(max_raw_abs_err, raw_abs_err);
        max_quant_abs_err           = std::max(max_quant_abs_err, quant_abs_err);
        byte_diff_count += got[i] != rq_raw;
        if (quant_abs_err > quant_tol) {
            if (bad_count < 8) {
                const int d   = static_cast<int>(i % kHeadDim);
                int       tmp = static_cast<int>(i / kHeadDim);
                const int h   = tmp % kQHeads;
                tmp /= kQHeads;
                const int q = tmp % kSeqLen;
                const int b = tmp / kSeqLen;
                std::cerr << "O mismatch b=" << b << " q=" << q << " h=" << h << " d=" << d << " ref=" << r
                          << " ref_q=" << rq << " got=" << g << " got_raw=" << static_cast<int>(got[i])
                          << " ref_raw=" << static_cast<int>(rq_raw) << " q_abs=" << quant_abs_err
                          << " raw_abs=" << raw_abs_err << "\n";
            }
            ++bad_count;
        }
    }
    std::cout << "O max_raw_abs_err=" << max_raw_abs_err << " max_quant_abs_err=" << max_quant_abs_err
              << " byte_diff_count=" << byte_diff_count << " bad_count=" << bad_count << "\n";
    if (require_byte_exact && byte_diff_count != 0) {
        std::cerr << "O byte-exact check failed: byte_diff_count=" << byte_diff_count << "\n";
        return false;
    }
    return bad_count == 0;
}

}  // namespace

int main()
{
    if (!turbomind::isSM10x()) {
        std::cout << "[test_fp4_kv_cache] skip: SM10x is required for NVFP4 conversion\n";
        return 0;
    }

    constexpr int kQkvHeads = kQHeads + 2 * kKvHeads;
    constexpr int kTokenNum = kBatchSize * kSeqLen;
    constexpr int kScaleDim = kHeadDim / 16;

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));

    turbomind::block::Layout<turbomind::block::Config<T, turbomind::fp4_e2m1_t, kHeadDim, false>> layout{
        turbomind::block::Config<T, turbomind::fp4_e2m1_t, kHeadDim, false>{kKvHeads, kPageSize}};
    const int page_stride_bytes = layout.block_size(kLayerNum);
    const int head_data_bytes   = kPageSize * kHeadDim / 2;

    const TestCase test_cases[] = {
        {"zero_q", QMode::kZero, 2.6e-1f, false},
        {"nonzero_q", QMode::kNonZero, 1.01f, false},
    };
    bool all_ok = true;

    for (const auto& test_case : test_cases) {
        std::cout << "[test_fp4_kv_cache] case=" << test_case.name << "\n";

        std::vector<T> h_qkv(kTokenNum * kQkvHeads * kHeadDim);
        fill_qkv(h_qkv, test_case.q_mode);

        T* d_qkv{};
        CHECK_CUDA(cudaMalloc(&d_qkv, h_qkv.size() * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(d_qkv, h_qkv.data(), h_qkv.size() * sizeof(T), cudaMemcpyHostToDevice));

        uint8_t* d_processed_q{};
        CHECK_CUDA(cudaMalloc(&d_processed_q, kTokenNum * kQHeads * kHeadDim));
        CHECK_CUDA(cudaMemset(d_processed_q, 0, kTokenNum * kQHeads * kHeadDim));

        uint8_t* d_cache{};
        CHECK_CUDA(cudaMalloc(&d_cache, kBatchSize * page_stride_bytes));
        CHECK_CUDA(cudaMemset(d_cache, 0, kBatchSize * page_stride_bytes));

        uint8_t*  d_k_scales{};
        uint8_t*  d_v_scales{};
        const int scale_stride_head  = kPageSize * kScaleDim;
        const int scale_stride_batch = kKvHeads * scale_stride_head;
        CHECK_CUDA(cudaMalloc(&d_k_scales, kBatchSize * scale_stride_batch));
        CHECK_CUDA(cudaMalloc(&d_v_scales, kBatchSize * scale_stride_batch));
        CHECK_CUDA(cudaMemset(d_k_scales, 0xCC, kBatchSize * scale_stride_batch));
        CHECK_CUDA(cudaMemset(d_v_scales, 0xCC, kBatchSize * scale_stride_batch));

        std::vector<int> h_cu_len(kBatchSize + 1);
        for (int i = 0; i <= kBatchSize; ++i) {
            h_cu_len[i] = i * kSeqLen;
        }
        int* d_cu_len{};
        CHECK_CUDA(cudaMalloc(&d_cu_len, h_cu_len.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_cu_len, h_cu_len.data(), h_cu_len.size() * sizeof(int), cudaMemcpyHostToDevice));

        std::vector<int> h_token2batch(kTokenNum * 2);
        for (int b = 0, t = 0; b < kBatchSize; ++b) {
            for (int s = 0; s < kSeqLen; ++s, ++t) {
                h_token2batch[t * 2 + 0] = b;
                h_token2batch[t * 2 + 1] = s;
            }
        }
        int* d_token2batch{};
        CHECK_CUDA(cudaMalloc(&d_token2batch, h_token2batch.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(
            d_token2batch, h_token2batch.data(), h_token2batch.size() * sizeof(int), cudaMemcpyHostToDevice));

        std::vector<int> h_page_tables(kBatchSize * 2);
        for (int b = 0; b < kBatchSize; ++b) {
            h_page_tables[b * 2 + 0] = b;
            h_page_tables[b * 2 + 1] = b;
        }
        int* d_page_tables{};
        CHECK_CUDA(cudaMalloc(&d_page_tables, h_page_tables.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(
            d_page_tables, h_page_tables.data(), h_page_tables.size() * sizeof(int), cudaMemcpyHostToDevice));

        float*      d_qkv_scale{};
        const float h_qkv_scale[3] = {0.25f, 2.f, 0.5f};
        CHECK_CUDA(cudaMalloc(&d_qkv_scale, sizeof(h_qkv_scale)));
        CHECK_CUDA(cudaMemcpy(d_qkv_scale, h_qkv_scale, sizeof(h_qkv_scale), cudaMemcpyHostToDevice));

        float*      d_o_scale{};
        const float h_o_scale[1] = {0.0625f};
        CHECK_CUDA(cudaMalloc(&d_o_scale, sizeof(h_o_scale)));
        CHECK_CUDA(cudaMemcpy(d_o_scale, h_o_scale, sizeof(h_o_scale), cudaMemcpyHostToDevice));

        float* d_bmm1_scale{};
        float* d_bmm2_scale{};
        int*   d_tile_counter{};
        CHECK_CUDA(cudaMalloc(&d_bmm1_scale, 2 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_bmm2_scale, sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_tile_counter, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_bmm1_scale, 0, 2 * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_bmm2_scale, 0, sizeof(float)));
        CHECK_CUDA(cudaMemset(d_tile_counter, 0, sizeof(int)));

        turbomind::AttentionParams<T> params{};
        params.fmha.raw_q = d_qkv;
        params.fmha.raw_k = d_qkv + kQHeads * kHeadDim;
        params.fmha.raw_v = params.fmha.raw_k + kKvHeads * kHeadDim;
        params.q          = reinterpret_cast<T*>(d_processed_q);
        params.k          = nullptr;
        params.v          = nullptr;
        params.stride     = kQkvHeads * kHeadDim;

        params.batch_size    = kBatchSize;
        params.token_num     = kTokenNum;
        params.max_q_len     = kSeqLen;
        params.max_k_len     = kSeqLen;
        params.num_heads     = kQHeads;
        params.num_kv_heads  = kKvHeads;
        params.size_per_head = kHeadDim;
        params.cu_q_len      = d_cu_len;
        params.cu_k_len      = d_cu_len;
        params.quant_policy  = turbomind::QuantPolicy::kCacheKVFP4;
        params.rope_param    = turbomind::RopeKernelParam{turbomind::RopeType::kNull};
        params.stream        = stream;

        params.block_iter_params               = turbomind::BlockIteratorParams{nullptr, nullptr, 0, kPageSize};
        params.fmha.layer_num                  = kLayerNum;
        params.fmha.kv_cache_pool_ptr          = d_cache;
        params.fmha.page_tables                = d_page_tables;
        params.fmha.max_num_pages_per_seq_kv   = 1;
        params.fmha.page_nums                  = kBatchSize;
        params.fmha.kv_cache_page_stride_bytes = page_stride_bytes;
        params.fmha.key_block_scales           = d_k_scales;
        params.fmha.value_block_scales         = d_v_scales;
        params.fmha.kv_scale_stride_heads      = scale_stride_head;
        params.fmha.kv_scale_stride_batch      = scale_stride_batch;
        params.fmha.token2batch                = d_token2batch;
        params.fmha.qkv_scale_orig             = d_qkv_scale;
        params.fmha.o_scale_orig               = d_o_scale;
        params.fmha.host_bmm1_scale            = 1.f / std::sqrt(static_cast<float>(kHeadDim));
        params.fmha.scale_bmm1_ptr             = d_bmm1_scale;
        params.fmha.scale_bmm2_ptr             = d_bmm2_scale;
        params.fmha.fmha_tile_counter          = d_tile_counter;

        turbomind::invokeAppendQKNormRopeKVUpdate_(params);
        CHECK_CUDA(cudaDeviceSynchronize());

        T*        d_k_deq{};
        T*        d_v_deq{};
        const int kv_elems = kBatchSize * kSeqLen * kKvHeads * kHeadDim;
        CHECK_CUDA(cudaMalloc(&d_k_deq, kv_elems * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&d_v_deq, kv_elems * sizeof(T)));
        const int packs = kBatchSize * kSeqLen * kKvHeads * (kHeadDim / 8);
        dequant_fp4_cache_kernel<false><<<dim3((packs + 127) / 128), dim3(128)>>>(d_cache,
                                                                                  d_k_scales,
                                                                                  d_k_deq,
                                                                                  page_stride_bytes,
                                                                                  head_data_bytes,
                                                                                  scale_stride_batch,
                                                                                  scale_stride_head,
                                                                                  h_qkv_scale[1]);
        dequant_fp4_cache_kernel<true><<<dim3((packs + 127) / 128), dim3(128)>>>(d_cache,
                                                                                 d_v_scales,
                                                                                 d_v_deq,
                                                                                 page_stride_bytes,
                                                                                 head_data_bytes,
                                                                                 scale_stride_batch,
                                                                                 scale_stride_head,
                                                                                 h_qkv_scale[2]);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<T> h_k_deq(kv_elems);
        std::vector<T> h_v_deq(kv_elems);
        CHECK_CUDA(cudaMemcpy(h_k_deq.data(), d_k_deq, kv_elems * sizeof(T), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_v_deq.data(), d_v_deq, kv_elems * sizeof(T), cudaMemcpyDeviceToHost));

        const bool k_ok = compare_tensor("K", h_k_deq, h_qkv, false);
        const bool v_ok = compare_tensor("V", h_v_deq, h_qkv, true);

        int*             d_seq_lens{};
        std::vector<int> h_seq_lens(kBatchSize, kSeqLen);
        CHECK_CUDA(cudaMalloc(&d_seq_lens, h_seq_lens.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_seq_lens, h_seq_lens.data(), h_seq_lens.size() * sizeof(int), cudaMemcpyHostToDevice));

        uint8_t*  d_attn_out{};
        const int out_elems = kBatchSize * kSeqLen * kQHeads * kHeadDim;
        CHECK_CUDA(cudaMalloc(&d_attn_out, out_elems));
        CHECK_CUDA(cudaMemset(d_attn_out, 0, out_elems));

        turbomind::flashinfer_fmha::initialize();
        turbomind::flashinfer_fmha::FmhaParams fmha{};
        fmha.out                    = d_attn_out;
        fmha.query                  = d_processed_q;
        fmha.key_cache              = d_cache;
        fmha.value_cache            = d_cache + head_data_bytes;
        fmha.workspace_buffer       = nullptr;
        fmha.block_tables           = d_page_tables;
        fmha.seq_lens               = d_seq_lens;
        fmha.cum_seq_lens_q         = d_cu_len;
        fmha.cum_seq_lens_kv        = d_cu_len;
        fmha.q_dtype                = turbomind::flashinfer_fmha::DType::kFP8E4M3;
        fmha.kv_dtype               = turbomind::flashinfer_fmha::DType::kFP4E2M1;
        fmha.o_dtype                = turbomind::flashinfer_fmha::DType::kFP8E4M3;
        fmha.batch_size             = kBatchSize;
        fmha.max_q_len              = kSeqLen;
        fmha.max_kv_len             = kSeqLen;
        fmha.num_qo_heads           = kQHeads;
        fmha.num_kv_heads           = kKvHeads;
        fmha.head_dim_qk            = kHeadDim;
        fmha.head_dim_vo            = kHeadDim;
        fmha.page_size              = kPageSize;
        fmha.max_num_blocks_per_seq = 1;
        fmha.num_pages_in_pool      = kBatchSize;
        fmha.kv_stride_keys_values  = kHeadDim;
        fmha.kv_stride_heads        = 2 * head_data_bytes * 2;
        fmha.kv_stride_batch        = page_stride_bytes * 2;
        fmha.bmm1_scale             = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
        fmha.bmm2_scale             = 1.0;
        fmha.bmm1_scale_log2_ptr    = d_bmm1_scale + 1;
        fmha.bmm2_scale_ptr         = d_bmm2_scale;
        fmha.key_block_scales       = d_k_scales;
        fmha.value_block_scales     = d_v_scales;
        fmha.kv_scale_stride_heads  = scale_stride_head;
        fmha.kv_scale_stride_batch  = scale_stride_batch;
        fmha.window_left            = -1;
        fmha.sm_count               = turbomind::getSMCount();
        fmha.workspace_size         = 0;
        fmha.stream                 = stream;
        if (!turbomind::flashinfer_fmha::dispatch_context(fmha)) {
            std::cerr << "[test_fp4_kv_cache] FlashInfer context dispatch returned false\n";
            return 1;
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<uint8_t> h_processed_q(kTokenNum * kQHeads * kHeadDim);
        std::vector<uint8_t> h_attn_out(out_elems);
        CHECK_CUDA(cudaMemcpy(h_processed_q.data(), d_processed_q, h_processed_q.size(), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_attn_out.data(), d_attn_out, out_elems, cudaMemcpyDeviceToHost));

        const auto h_attn_ref = reference_context_attention(h_processed_q, h_k_deq, h_v_deq, h_qkv_scale[0]);
        const bool o_ok       = compare_attention_output(
            h_attn_out, h_attn_ref, h_o_scale[0], test_case.quant_tol, test_case.require_byte_exact);

        std::vector<uint8_t> h_decode_q(kBatchSize * kQHeads * kHeadDim);
        std::vector<T>       h_decode_ref(kBatchSize * kQHeads * kHeadDim);
        for (int b = 0; b < kBatchSize; ++b) {
            const int src_token = b * kSeqLen + (kSeqLen - 1);
            const int dst_token = b;
            for (int h = 0; h < kQHeads; ++h) {
                for (int d = 0; d < kHeadDim; ++d) {
                    const int src_idx = (src_token * kQHeads + h) * kHeadDim + d;
                    const int dst_idx = (dst_token * kQHeads + h) * kHeadDim + d;
                    h_decode_q[dst_idx]   = h_processed_q[src_idx];
                    h_decode_ref[dst_idx] = h_attn_ref[src_idx];
                }
            }
        }

        uint8_t* d_decode_q{};
        uint8_t* d_decode_out{};
        void*    d_decode_workspace{};
        CHECK_CUDA(cudaMalloc(&d_decode_q, h_decode_q.size()));
        CHECK_CUDA(cudaMemcpy(d_decode_q, h_decode_q.data(), h_decode_q.size(), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&d_decode_out, h_decode_q.size()));
        CHECK_CUDA(cudaMemset(d_decode_out, 0, h_decode_q.size()));
        CHECK_CUDA(cudaMalloc(&d_decode_workspace, 16 * 1024 * 1024));

        auto decode_fmha              = fmha;
        decode_fmha.out               = d_decode_out;
        decode_fmha.query             = d_decode_q;
        decode_fmha.workspace_buffer  = d_decode_workspace;
        decode_fmha.workspace_size    = 16 * 1024 * 1024;
        decode_fmha.max_q_len         = 1;
        decode_fmha.cum_seq_lens_q    = nullptr;
        decode_fmha.cum_seq_lens_kv   = nullptr;
        if (!turbomind::flashinfer_fmha::dispatch_decode(decode_fmha)) {
            std::cerr << "[test_fp4_kv_cache] FlashInfer decode dispatch returned false\n";
            return 1;
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<uint8_t> h_decode_out(h_decode_q.size());
        CHECK_CUDA(cudaMemcpy(h_decode_out.data(), d_decode_out, h_decode_out.size(), cudaMemcpyDeviceToHost));
        const bool decode_ok =
            compare_attention_output(h_decode_out, h_decode_ref, h_o_scale[0], test_case.quant_tol, false);

        cudaFree(d_qkv);
        cudaFree(d_processed_q);
        cudaFree(d_cache);
        cudaFree(d_k_scales);
        cudaFree(d_v_scales);
        cudaFree(d_cu_len);
        cudaFree(d_token2batch);
        cudaFree(d_page_tables);
        cudaFree(d_qkv_scale);
        cudaFree(d_o_scale);
        cudaFree(d_bmm1_scale);
        cudaFree(d_bmm2_scale);
        cudaFree(d_tile_counter);
        cudaFree(d_k_deq);
        cudaFree(d_v_deq);
        cudaFree(d_seq_lens);
        cudaFree(d_attn_out);
        cudaFree(d_decode_q);
        cudaFree(d_decode_out);
        cudaFree(d_decode_workspace);
        all_ok = all_ok && k_ok && v_ok && o_ok && decode_ok;
    }
    cudaStreamDestroy(stream);

    if (!all_ok) {
        std::cerr << "[test_fp4_kv_cache] FAILED\n";
        return 1;
    }
    std::cout << "[test_fp4_kv_cache] PASSED\n";
    return 0;
}
