#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/attention_params.h"
#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/attention/reference.h"
#include "src/turbomind/kernels/attention/test_utils.h"
#include "src/turbomind/flashinfer/attention/kv_cache_utils_v3.h"
#include "src/turbomind/flashinfer/attention/kv_cache_utils_v3.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/flashinfer/attention/trtllm_attention_wrapper.h"

using namespace turbomind;

// [b, h, s, d] : current -> stride_h=s, stride_s=1, stride_b=hs
// [cu_q, h, d] : qkvgemm -> stride_h=1, stride_s=h, stride_b=0
// [h, cu_s, d] : prefill -> stride_h=s, stride_s=1, stride_b=0

// Test configuration for different scenarios
struct TestConfig {
    const char* name;
    size_t      head_num;
    size_t      kv_head_num;  // 0 means same as head_num
    size_t      batch_size;
    size_t      input_len;
    size_t      sequence_len;
    int         max_split_k;
    int         block_len;

    size_t get_kv_head_num() const
    {
        return kv_head_num == 0 ? head_num : kv_head_num;
    }
};

// Predefined test configurations
namespace TestConfigs {
// Decoding stage configurations
//           name, head_num, kv_head_num, batch_size input_len, sequence_len, max_split_k, block_len
// block_len=32 matches the FP16 PagedKv P32 Context cubins shipped here.
constexpr TestConfig Append_Small = {"Append_Small", 32, 8, 64, 12, 63, 128, 32};
constexpr TestConfig Append_Mid   = {"Append_Mid", 32, 8, 128, 123, 8191, 128, 32};
constexpr TestConfig Append_Large = {"Append_Large", 32, 8, 128, 1051, 32767, 128, 32};

}  // namespace TestConfigs

template<class T, class Tkv>
struct Config {
    int head_dim_;
    int head_num_;
    int block_len_;
    int num_layers_;

    TM_HOST_DEVICE constexpr int t_bits() const
    {
        if constexpr (std::is_same_v<T, Tkv>) {
            return 0;
        }
        else {
            return bitsof<T>;
        }
    }

    TM_HOST_DEVICE constexpr int q_bits() const
    {
        return bitsof<Tkv>;
    }

    TM_HOST_DEVICE constexpr int head_dim() const
    {
        return head_dim_;
    }

    TM_HOST_DEVICE int head_num() const
    {
        return head_num_;
    }

    TM_HOST_DEVICE constexpr int block_len() const
    {
        return block_len_;
    }

    TM_HOST_DEVICE int layer_num() const
    {
        return num_layers_;
    }
};

// Seed paged KV cache with the same append writer used by the runtime path.
template<class Tkv, class T>
void processQAndKVUpdateForAppend(thrust::universal_vector<T>&    dc_processed_q,  // 输出用于decoding attn的q数据
                                  thrust::universal_vector<T>&    pf_processed_qkv,
                                  thrust::universal_vector<T>&    pf_qkv,
                                  thrust::universal_vector<T>&    dc_qkv,
                                  thrust::universal_vector<char>& blocks,  // 写入用于decoding attn的kv数据
                                  thrust::universal_vector<int>&  page_tables,
                                  const size_t                    total_num_pages,
                                  const size_t                    head_num,
                                  const size_t                    kv_head_num,
                                  const size_t                    head_dim,
                                  const size_t                    block_seq_len,
                                  const size_t                    pf_batch_size,
                                  const size_t                    dc_batch_size,
                                  int                             quant_policy,
                                  const size_t                    kv_seq_len,
                                  const size_t                    q_seq_len,
                                  const RopeKernelParam&          rope_param)
{
    // 前一次的Prefill阶段，将输入的prefill qkv，进行前处理，并将其写入到kv cache中
    {

        AttentionParams<T> params{};

        thrust::universal_vector<int> pf_seq_lens(pf_batch_size);
        thrust::universal_vector<int> pf_cu_seq_lens(pf_batch_size + 1);
        thrust::fill(pf_seq_lens.begin(), pf_seq_lens.end(), kv_seq_len);

        for (size_t i = 0; i <= pf_batch_size; ++i) {
            pf_cu_seq_lens[i] = i * kv_seq_len;
        }

        // QKV pointers: qkv layout is [batch, seq_len, (Q_heads + KV_heads * 2), head_dim]
        params.fmha.raw_q = pf_qkv.data().get();
        params.fmha.raw_k = params.fmha.raw_q + head_num * head_dim;
        params.fmha.raw_v = params.fmha.raw_k + kv_head_num * head_dim;

        params.q = pf_processed_qkv.data().get();
        params.k = nullptr;
        params.v = nullptr;

        params.stride = (head_num + 2 * kv_head_num) * head_dim;

        params.batch_size    = pf_batch_size;
        params.num_heads     = head_num;
        params.num_kv_heads  = kv_head_num;
        params.size_per_head = head_dim;
        params.max_q_len     = kv_seq_len;
        params.max_k_len     = kv_seq_len;
        params.token_num     = pf_batch_size * kv_seq_len;

        params.cu_q_len = pf_cu_seq_lens.data().get();
        params.cu_k_len = pf_cu_seq_lens.data().get();

        params.quant_policy = quant_policy;
        params.rope_param   = rope_param;

        // Initialize token2batch for V4 kernel
        // Layout: [batch_id_0, token_idx_0, batch_id_1, token_idx_1, ...]
        // global_token_idx is a counter across all batches, starting from 0
        thrust::universal_vector<int> pf_token2batch_vec(pf_batch_size * kv_seq_len * 2);
        int                           global_token_idx = 0;
        for (size_t b = 0; b < pf_batch_size; ++b) {
            for (size_t s = 0; s < kv_seq_len; ++s) {
                pf_token2batch_vec[global_token_idx * 2 + 0] = b;  // batch_id
                pf_token2batch_vec[global_token_idx * 2 + 1] = s;  // relative token_idx within batch
                ++global_token_idx;
            }
        }
        params.fmha.token2batch = pf_token2batch_vec.data().get();

        const size_t max_num_pages_per_seq_kv = div_up(q_seq_len + kv_seq_len, block_seq_len);

        params.block_iter_params             = BlockIteratorParams{nullptr, nullptr, 0, (int)block_seq_len};
        params.fmha.kv_cache_pool_ptr        = blocks.data().get();
        params.fmha.page_nums                = (int)total_num_pages;
        params.fmha.page_tables              = page_tables.data().get();
        params.fmha.seq_lens_kv              = nullptr;
        params.fmha.max_num_pages_per_seq_kv = (int)max_num_pages_per_seq_kv;

        invokeAppendQKNormRopeKVUpdate_(params);
        cudaDeviceSynchronize();
    }

    // 当前的Prefill阶段，将输入的prefill qkv，进行前处理，并将其写入到kv cache中
    {

        AttentionParams<T> params{};

        thrust::universal_vector<int> dc_q_lens(dc_batch_size);
        thrust::universal_vector<int> dc_cu_q_lens(dc_batch_size + 1);
        thrust::fill(dc_q_lens.begin(), dc_q_lens.end(), q_seq_len);

        dc_cu_q_lens[0] = 0;
        for (size_t i = 1; i <= dc_batch_size; ++i) {
            dc_cu_q_lens[i] = dc_cu_q_lens[i - 1] + dc_q_lens[i - 1];
        }

        thrust::universal_vector<int> dc_k_lens(dc_batch_size);
        thrust::universal_vector<int> dc_cu_k_lens(dc_batch_size + 1);
        thrust::fill(dc_k_lens.begin(), dc_k_lens.end(), kv_seq_len + q_seq_len);

        dc_cu_k_lens[0] = 0;
        for (size_t i = 1; i <= dc_batch_size; ++i) {
            dc_cu_k_lens[i] = dc_cu_k_lens[i - 1] + dc_k_lens[i - 1];
        }

        // QKV pointers: qkv layout is [batch, seq_len, (Q_heads + KV_heads * 2), head_dim]
        params.fmha.raw_q = dc_qkv.data().get();
        params.fmha.raw_k = params.fmha.raw_q + head_num * head_dim;
        params.fmha.raw_v = params.fmha.raw_k + kv_head_num * head_dim;

        params.q = dc_processed_q.data().get();

        params.stride = (head_num + 2 * kv_head_num) * head_dim;

        params.batch_size    = dc_batch_size;
        params.num_heads     = head_num;
        params.num_kv_heads  = kv_head_num;
        params.size_per_head = head_dim;
        params.max_q_len     = q_seq_len;
        params.max_k_len     = q_seq_len + kv_seq_len;
        params.token_num     = q_seq_len * dc_batch_size;

        params.cu_q_len = dc_cu_q_lens.data().get();
        params.cu_k_len = dc_cu_k_lens.data().get();

        params.quant_policy = quant_policy;
        params.rope_param   = rope_param;

        thrust::universal_vector<int> token2batch_vec(dc_batch_size * q_seq_len * 2);
        int                           global_token_idx = 0;
        for (size_t b = 0; b < dc_batch_size; ++b) {
            for (size_t s = 0; s < q_seq_len; ++s) {
                token2batch_vec[global_token_idx * 2 + 0] = b;  // batch_id
                token2batch_vec[global_token_idx * 2 + 1] = s;  // relative token_idx within batch
                ++global_token_idx;
            }
        }
        params.fmha.token2batch = token2batch_vec.data().get();

        const size_t max_num_pages_per_seq_kv = (q_seq_len + kv_seq_len + block_seq_len - 1) / block_seq_len;

        params.block_iter_params             = BlockIteratorParams{nullptr, nullptr, 0, (int)block_seq_len};
        params.fmha.kv_cache_pool_ptr        = blocks.data().get();
        params.fmha.page_nums                = (int)total_num_pages;
        params.fmha.page_tables              = page_tables.data().get();
        params.fmha.seq_lens_kv              = dc_k_lens.data().get();
        params.fmha.max_num_pages_per_seq_kv = (int)max_num_pages_per_seq_kv;

        invokeAppendQKNormRopeKVUpdate_(params);
        cudaDeviceSynchronize();

        auto cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            std::cout << "[ERROR] KV update failed: " << cudaGetErrorString(cuda_err) << std::endl;
        }
    }
}

double get_memory_bandwidth()  // -> GB/s
{
    int clock_rate_khz{};
    int bus_width_bits{};
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    return 2. * (double)clock_rate_khz / 1e6 * (double)bus_width_bits / 8.;
}

// 从 paged blocks 中提取 KV cache 用于对比
template<class T, class Tkv>
void extractKVCacheFromBlocks(thrust::universal_vector<T>&          k_cache_extracted,  // 输出: [B*H, S, D]
                              thrust::universal_vector<T>&          v_cache_extracted,  // 输出: [B*H, S, D]
                              const thrust::universal_vector<char>& blocks,             // 输入: paged blocks
                              const thrust::universal_vector<int>&  page_tables,        // 输入: [B, 2, max_pages]
                              size_t                                batch_size,
                              size_t                                kv_head_num,
                              size_t                                head_dim,
                              size_t                                block_len,
                              size_t                                seq_len,  // 要提取的序列长度
                              int                                   num_layers)
{
    // 配置 layout
    Config<T, Tkv>            config_layer{(int)head_dim, (int)kv_head_num, (int)block_len, num_layers};
    block::LayoutLayerPagedKV layer_paged_layout{config_layer};

    const size_t n_blocks                 = (seq_len + block_len - 1) / block_len;
    const int    max_num_pages_per_seq_kv = n_blocks;

    // 确保输出缓冲区大小正确: [B*H, S, D]
    k_cache_extracted.resize(batch_size * kv_head_num * seq_len * head_dim);
    v_cache_extracted.resize(batch_size * kv_head_num * seq_len * head_dim);

    const char* kv_cache_pool = blocks.data().get();
    const int*  page_table    = page_tables.data().get();

    // 遍历每个 batch
    for (size_t b = 0; b < batch_size; ++b) {
        // 遍历每个 head
        for (size_t h = 0; h < kv_head_num; ++h) {
            // 遍历序列中的每个 token
            for (size_t s = 0; s < seq_len; ++s) {
                // 计算这个 token 在哪个 block 和 block 内的偏移
                const size_t block_idx       = s / block_len;
                const size_t offset_in_block = s % block_len;

                // 从 page_table 中获取物理 block 索引
                // page_table layout: [batch_size, 2, max_num_pages_per_seq_kv]
                const int k_physical_block =
                    page_table[b * 2 * max_num_pages_per_seq_kv + 0 * max_num_pages_per_seq_kv + block_idx];
                const int v_physical_block =
                    page_table[b * 2 * max_num_pages_per_seq_kv + 1 * max_num_pages_per_seq_kv + block_idx];

                // 计算在 K cache pool 中的地址
                // LayoutLayerPagedKV: [num_blocks, num_heads, block_len, head_dim]
                // offset = block_data(block_id) + head_data(head_id) + token_data(token_id)
                const size_t k_block_offset = layer_paged_layout.block_data(k_physical_block)
                                              + layer_paged_layout.head_data(h)
                                              + layer_paged_layout.token_data(offset_in_block);
                const size_t v_block_offset = layer_paged_layout.block_data(v_physical_block)
                                              + layer_paged_layout.head_data(h)
                                              + layer_paged_layout.token_data(offset_in_block);

                // 计算在输出数组中的位置: [B*H, S, D]
                const size_t bh            = b * kv_head_num + h;
                const size_t output_offset = bh * seq_len * head_dim + s * head_dim;

                // 复制数据
                cudaMemcpy(k_cache_extracted.data().get() + output_offset,
                           kv_cache_pool + k_block_offset,
                           head_dim * sizeof(T),
                           cudaMemcpyDeviceToDevice);

                cudaMemcpy(v_cache_extracted.data().get() + output_offset,
                           kv_cache_pool + v_block_offset,
                           head_dim * sizeof(T),
                           cudaMemcpyDeviceToDevice);
            }
        }
    }

    cudaDeviceSynchronize();
}

template<class T, class Tkv>
int test_trtllm_attention_impl(const TestConfig& config, int kQuantPolicy, int kTestIter = 10)
{
    AttentionParams<T> params{};

    constexpr size_t kHeadDim   = 128;
    constexpr int    kNumLayers = 1;

    // Use test configuration parameters
    const size_t kHeadNum     = config.head_num;
    const size_t KvHeadNum    = config.get_kv_head_num();
    const size_t kBatchSize   = config.batch_size;
    const size_t kInputLen    = config.input_len;
    const size_t kSequenceLen = config.sequence_len;
    const int    kMaxSplitK   = config.max_split_k;
    const int    kBlockLen    = config.block_len;

    const int kTpSize = 1;
    const int kTpRank = 0;

    std::cout << "\n=== Testing (TRT-LLM): " << config.name << " ===" << std::endl;
    std::cout << "HeadNum=" << kHeadNum << ", KvHeadNum=" << KvHeadNum << ", BatchSize=" << kBatchSize
              << ", InputLen=" << kInputLen << ", SeqLen=" << kSequenceLen << std::endl;

    if (KvHeadNum == 0 || kHeadNum % KvHeadNum != 0) {
        std::cerr << "Error: Invalid KvHeadNum configuration!" << std::endl;
        return -1;
    }

    std::cout << "\n=== Begin init TrtllmAttentionWrapper ===" << std::endl;
    constexpr DataType kModelDtype = std::is_same_v<T, half> ? DataType::kFloat16 : DataType::kBfloat16;
    auto attention_wrapper_ = std::make_unique<TrtllmAttentionWrapper>(
        kHeadNum, KvHeadNum, kHeadDim, turbomind::getSMVersion(), kModelDtype);


    const size_t kContextLen = kSequenceLen + kInputLen;
    const size_t kTokenNum   = kBatchSize * kInputLen;

    constexpr float kRoPEBase = 10000.f;
    constexpr int   kRoPEDim  = kHeadDim / 2;
    constexpr int   kDump     = 0;

    RNG rng{};

    // 用于reference的其中kv的最后batch_size的最后一个token会进行覆盖，所以需要分配 kContextLen + 1 个token的空间
    thrust::universal_vector<T> k_cache(kBatchSize * KvHeadNum * kContextLen * kHeadDim);
    thrust::universal_vector<T> v_cache(kBatchSize * KvHeadNum * kContextLen * kHeadDim);

    thrust::universal_vector<T> pf_qkv(kBatchSize * kSequenceLen * (kHeadNum + KvHeadNum * 2) * kHeadDim);
    thrust::universal_vector<T> pf_processed_qkv(kBatchSize * kSequenceLen * (kHeadNum + KvHeadNum * 2) * kHeadDim);

    thrust::universal_vector<T> dc_qkv(kBatchSize * kInputLen * (kHeadNum + KvHeadNum * 2) * kHeadDim);
    thrust::universal_vector<T> dc_q(kBatchSize * kInputLen * kHeadNum * kHeadDim);

    thrust::universal_vector<T> output(kBatchSize * kInputLen * kHeadNum * kHeadDim);

    thrust::universal_vector<bool>  finished(kBatchSize);
    thrust::universal_vector<int>   sequence_length(kBatchSize);
    thrust::universal_vector<int>   input_length(kBatchSize);
    thrust::universal_vector<int>   context_length(kBatchSize);
    thrust::universal_vector<float> rope_base(kBatchSize);
    thrust::universal_vector<int>   cu_seqlens(kBatchSize + 1);
    thrust::universal_vector<int>   cu_kv_lens(kBatchSize + 1);

    thrust::device_vector<float> partial_M(kTokenNum * kHeadNum * kMaxSplitK);
    thrust::device_vector<float> partial_L(kTokenNum * kHeadNum * kMaxSplitK);
    thrust::device_vector<float> partial_O(kTokenNum * kHeadNum * kMaxSplitK * kHeadDim);
    thrust::device_vector<int>   split_cnt(kTokenNum);
    thrust::device_vector<int>   semaphores(kTokenNum * kHeadNum * kMaxSplitK);

    thrust::universal_vector<float> qk_buf((size_t)kDump * kBatchSize * kHeadNum * kInputLen * kContextLen);
    thrust::universal_vector<T>     pr_buf((size_t)kDump * kBatchSize * kHeadNum * kInputLen * kContextLen);

    // Allocate workspace buffer for multi-block mode (decode phase)
    // 为 multi-block mode 分配 workspace buffer (64MB)
    constexpr size_t            kWorkspaceSize = 64 * 1024 * 1024;  // 64 MB
    thrust::device_vector<char> workspace_buffer(kWorkspaceSize);

    thrust::fill(semaphores.begin(), semaphores.end(), 0);

    rng.GenerateNormal(dc_qkv.data().get(), dc_qkv.size(), 1.f, 0.f);

    rng.GenerateNormal(k_cache.data().get(), kBatchSize * KvHeadNum * kContextLen * kHeadDim);
    rng.GenerateNormal(v_cache.data().get(), kBatchSize * KvHeadNum * kContextLen * kHeadDim);

    if (0) {
        // Set input range to zero
        // (BH, SD)
        cudaMemset2DAsync(k_cache.data().get() + kSequenceLen * kHeadDim,
                          sizeof(T) * kContextLen * kHeadDim,
                          0,
                          sizeof(T) * kInputLen * kHeadDim,
                          kBatchSize * KvHeadNum);
        cudaMemset2DAsync(v_cache.data().get() + kSequenceLen * kHeadDim,
                          sizeof(T) * kContextLen * kHeadDim,
                          0,
                          sizeof(T) * kInputLen * kHeadDim,
                          kBatchSize * KvHeadNum);
    }

    // Initialize rope_param first
    float           scale_factor = -std::log2f(kRoPEBase) / kRoPEDim;
    RopeKernelParam rope_param_value{RopeType::kDefault, nullptr, kRoPEDim, scale_factor, 1.f, {}, {}, {}};

    thrust::universal_vector<char> blocks;
    thrust::universal_vector<int>  cu_block_cnts;
    thrust::universal_vector<int>  page_tables;

    const size_t n_blocks                 = (kContextLen + kBlockLen - 1) / kBlockLen;
    const int    max_num_pages_per_seq_kv = n_blocks;

    // TRT-LLM LayoutLayerPagedKV: [num_blocks, num_heads, block_len, head_dim]
    Config<T, Tkv>            config_layer{(int)kHeadDim, (int)KvHeadNum, (int)kBlockLen, kNumLayers};
    block::LayoutLayerPagedKV layer_paged_layout{config_layer};

    std::cout << "Using LayoutLayerPagedKV for TRT-LLM attention\n";
    std::cout << "  head_dim:  " << layer_paged_layout.config().head_dim() << "\n"
              << "  head_num:  " << layer_paged_layout.config().head_num() << "\n"
              << "  block_len: " << layer_paged_layout.config().block_len() << "\n"
              << "  layer_num: " << layer_paged_layout.config().layer_num() << "\n";

    const size_t total_blocks = kBatchSize * n_blocks;

    const size_t block_size = layer_paged_layout.block_size();
    blocks.resize(total_blocks * block_size * 2);     // K + V
    thrust::fill(blocks.begin(), blocks.end(), NAN);

    page_tables.resize(kBatchSize * 2 * max_num_pages_per_seq_kv);

    std::vector<size_t> k_idxs(total_blocks);
    std::iota(k_idxs.begin(), k_idxs.end(), 0);  // K blocks: [0, total_blocks)

    std::random_device rd;
    std::mt19937       g(rd());
    std::shuffle(k_idxs.begin(), k_idxs.end(), g);

    for (size_t b = 0; b < kBatchSize; ++b) {
        for (size_t i = 0; i < n_blocks; ++i) {
            size_t logical_idx = b * n_blocks + i;
            int    k_physical  = k_idxs[logical_idx];
            page_tables[b * 2 * max_num_pages_per_seq_kv + 0 * max_num_pages_per_seq_kv + i] = k_physical;
            page_tables[b * 2 * max_num_pages_per_seq_kv + 1 * max_num_pages_per_seq_kv + i] =
                k_physical + total_blocks;
        }
    }

    std::vector<int> n_blocks_vec(kBatchSize + 1, n_blocks);
    cu_block_cnts.resize(kBatchSize + 1);
    std::exclusive_scan(n_blocks_vec.begin(), n_blocks_vec.end(), cu_block_cnts.begin(), 0);

    std::cout << "Allocated " << total_blocks << " blocks (K+V shuffled), block_size=" << block_size
              << ", total_memory=" << (blocks.size() / 1024.0 / 1024.0) << " MB\n";

    if (kSequenceLen > 0) {
        std::cout << "Initializing history KV cache (" << kSequenceLen << " tokens) into paged blocks...\n";

        thrust::fill(pf_qkv.begin(), pf_qkv.end(), T(0.0f));

        for (size_t b = 0; b < kBatchSize; ++b) {
            for (size_t s = 0; s < kSequenceLen; ++s) {
                for (size_t h = 0; h < KvHeadNum; ++h) {
                    size_t src_k_idx =
                        b * KvHeadNum * kContextLen * kHeadDim + h * kContextLen * kHeadDim + s * kHeadDim;
                    size_t dst_k_idx = b * kSequenceLen * (kHeadNum + KvHeadNum * 2) * kHeadDim
                                       + s * (kHeadNum + KvHeadNum * 2) * kHeadDim + kHeadNum * kHeadDim + h * kHeadDim;
                    cudaMemcpy(pf_qkv.data().get() + dst_k_idx,
                               k_cache.data().get() + src_k_idx,
                               kHeadDim * sizeof(T),
                               cudaMemcpyDeviceToDevice);

                    size_t src_v_idx =
                        b * KvHeadNum * kContextLen * kHeadDim + h * kContextLen * kHeadDim + s * kHeadDim;
                    size_t dst_v_idx = b * kSequenceLen * (kHeadNum + KvHeadNum * 2) * kHeadDim
                                       + s * (kHeadNum + KvHeadNum * 2) * kHeadDim + (kHeadNum + KvHeadNum) * kHeadDim
                                       + h * kHeadDim;
                    cudaMemcpy(pf_qkv.data().get() + dst_v_idx,
                               v_cache.data().get() + src_v_idx,
                               kHeadDim * sizeof(T),
                               cudaMemcpyDeviceToDevice);
                }
            }
        }
        cudaDeviceSynchronize();

        // 调用kernel将历史KV写入paged blocks
        processQAndKVUpdateForAppend<Tkv>(dc_q,
                                          pf_processed_qkv,
                                          pf_qkv,
                                          dc_qkv,
                                          blocks,
                                          page_tables,
                                          total_blocks * 2,   // total_num_pages
                                          kHeadNum,           // head_num
                                          KvHeadNum,          // kv_head_num
                                          kHeadDim,           // head_dim
                                          kBlockLen,          // block_seq_len
                                          kBatchSize,         // pf_batch_size
                                          kBatchSize,         // dc_batch_size
                                          kQuantPolicy,       // quant_policy
                                          kSequenceLen,       // kv_seq_len: 历史序列长度
                                          kInputLen,          // q_seq_len: 当前查询序列长度
                                          rope_param_value);  // rope_param

        std::cout << "History KV cache initialization completed\n";
    }

    thrust::universal_vector<T>     output_ref = output;
    thrust::universal_vector<void*> k_cache_ref_ptrs(kBatchSize);
    thrust::universal_vector<void*> v_cache_ref_ptrs(kBatchSize);
    cudaDeviceSynchronize();

    Reference<T> reference(0);  // Use default CUDA stream
    // window_size=INT_MAX = full causal (no SWA).  Reference's mask uses
    // `0 <= w && w < window_size`; passing 0 silently masks everything
    // and the subsequent softmax becomes 0/0 = NaN.
    reference.Reshape(kInputLen, kContextLen, kHeadNum, kHeadDim, KvHeadNum, kBatchSize,
                      /*window_size=*/INT_MAX);

    // For decoding: k_cache already contains history with RoPE applied
    // For prefill: k_cache is fresh, RoPE will be applied during ProcessKV
    if (kSequenceLen > 0) {
        // Apply RoPE to existing history for decoding scenario
        invokeApplyRotaryEmbedding(
            k_cache.data().get(), kContextLen, KvHeadNum, kHeadDim, kRoPEBase, kRoPEDim, kBatchSize);
    }

    thrust::universal_vector<T> k_cache_ref = k_cache;
    thrust::universal_vector<T> v_cache_ref = v_cache;

    for (int i = 0; i < 1; ++i) {
        reference.Execute(output_ref.data().get(),  // 使用有效的输出缓冲区
                          k_cache_ref.data().get(),
                          v_cache_ref.data().get(),
                          dc_qkv.data().get(),
                          nullptr,        // qkv_bias
                          nullptr,        // sinks
                          (float)kRoPEBase,
                          (int)kRoPEDim);
    }

    cudaDeviceSynchronize();

    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "\n";
        return -1;
    }
    std::cout << "---------------------------------------------------\n";

    params.out = output.data().get();

    std::vector<thrust::universal_vector<T>> outputs;

    std::vector<cudaEvent_t> ev_start(kTestIter);
    std::vector<cudaEvent_t> ev_end(kTestIter);

    for (int i = 0; i < kTestIter; ++i) {
        cudaEventCreate(&ev_start[i]);
        cudaEventCreate(&ev_end[i]);
    }

    params.q = dc_q.data().get();

    params.out = output.data().get();

    params.q_bias = nullptr;
    params.k_bias = nullptr;
    params.v_bias = nullptr;

    params.stride = (kHeadNum + 2 * KvHeadNum) * kHeadDim;

    params.token_num  = kTokenNum;
    params.batch_size = kBatchSize;
    params.max_q_len  = kInputLen;
    params.max_k_len  = kContextLen;

    int totalQSeqLen  = 0;
    int totalKvSeqLen = 0;
    for (size_t b = 0; b < kBatchSize; ++b) {
        input_length[b]     = kInputLen;
        sequence_length[b]  = kSequenceLen;
        context_length[b]   = kContextLen;
        k_cache_ref_ptrs[b] = k_cache_ref.data().get() + b * k_cache_ref.size() / kBatchSize;
        v_cache_ref_ptrs[b] = v_cache_ref.data().get() + b * v_cache_ref.size() / kBatchSize;
        rope_base[b]        = kRoPEBase;

        totalQSeqLen += kInputLen;
        totalKvSeqLen += kContextLen;
    }

    params.block_iter_params             = BlockIteratorParams{nullptr, nullptr, 0, kBlockLen};
    params.fmha.kv_cache_pool_ptr        = blocks.data().get();
    params.fmha.page_nums                = (int)(total_blocks * 2);
    params.fmha.page_tables              = page_tables.data().get();
    params.fmha.seq_lens_kv              = context_length.data().get();
    params.fmha.max_num_pages_per_seq_kv = max_num_pages_per_seq_kv;

    for (size_t idx = 0; idx <= kBatchSize; ++idx) {
        cu_seqlens[idx] = idx * kInputLen;
        cu_kv_lens[idx] = idx * kContextLen;
    }

    params.quant_policy = kQuantPolicy;

    params.cu_q_len = cu_seqlens.data().get();
    params.cu_k_len = cu_kv_lens.data().get();

    params.fmha.sum_q_len  = totalQSeqLen;
    params.fmha.sum_kv_len = totalKvSeqLen;

    params.num_heads     = kHeadNum;
    params.num_kv_heads  = KvHeadNum;
    params.size_per_head = kHeadDim;
    params.inv_sqrt_dh   = (float)std::log2(expf(1.)) / std::sqrt((float)params.size_per_head);

    params.arch = getSMVersion();

    params.fmha.workspace_buffer = workspace_buffer.data().get();
    params.fmha.workspace_size   = kWorkspaceSize;
    params.fmha.multi_block_mode = false;

    for (int i = 0; i < std::max(kTestIter, 1); ++i) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        cudaEventRecord(ev_start[i]);
        attention_wrapper_->DispatchDecoding(params);
        cudaEventRecord(ev_end[i]);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        if (auto err = cudaGetLastError(); err != cudaSuccess) {
            std::cout << cudaGetErrorString(err) << "\n";
            return -1;
        }
        if (1) {
            outputs.push_back(output);
        }
    }

    cudaDeviceSynchronize();

    const size_t nbytes = blocks.size();

    const size_t ops = 2 * kInputLen * kContextLen * kHeadDim * kHeadNum * kBatchSize;

    const float peak_bw = get_memory_bandwidth();

    std::cout << "Device peak global memory bandwidth: " << peak_bw << " GB/s\n";

    for (int i = 0; i < kTestIter; ++i) {
        float ms{};
        cudaEventElapsedTime(&ms, ev_start[i], ev_end[i]);
        const float bw      = nbytes / 1e9f / ms * 1000.f;
        const float flops   = ops / 1e12f / ms * 1000.f;
        const float percent = bw / peak_bw * 100.f;
        printf("time %.3f ms, bw %.3f GB/s, %.3f %%, tflops %.3f \n", ms, bw, percent, flops);
    }

    if (outputs.size() > 1) {
        std::cout << "Evaluating consistency..." << std::endl;
        for (size_t i = 1; i < outputs.size(); ++i) {
            Compare(outputs[i].data().get(), outputs[i - 1].data().get(), kHeadDim, kHeadDim, kHeadNum, 0, 0, 0);
        }
    }

    std::cout << "---------------------------------------------------\n";

    std::cout << ">>> Compare Q (processed)\n";
    // dc_q layout: [B, s, H, D] (output from invokeAppendQKNormRopeKVUpdate_)
    // reference.q() layout: [B, H, s, D] (output from Reference::processQKV)
    // 需要进行 layout 转换后再比较
    {
        size_t      outliers      = 0;
        size_t      total         = kBatchSize * kInputLen * kHeadNum * kHeadDim;
        float       max_abs_diff  = 0.f;
        float       max_rel_diff  = 0.f;
        std::string max_diff_info = "";

        thrust::host_vector<T> dc_q_host(dc_q.size());
        thrust::host_vector<T> ref_q_host(dc_q.size());

        cudaMemcpy(dc_q_host.data(), dc_q.data().get(), dc_q.size() * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(ref_q_host.data(), reference.q(), dc_q.size() * sizeof(T), cudaMemcpyDeviceToHost);

        for (size_t b = 0; b < kBatchSize; ++b) {
            for (size_t s = 0; s < kInputLen; ++s) {
                for (size_t h = 0; h < kHeadNum; ++h) {
                    for (size_t d = 0; d < kHeadDim; ++d) {
                        // dc_q layout: [B, s, H, D]
                        size_t dc_idx = ((b * kInputLen + s) * kHeadNum + h) * kHeadDim + d;
                        // ref_q layout: [B, H, s, D]
                        size_t ref_idx = ((b * kHeadNum + h) * kInputLen + s) * kHeadDim + d;

                        float dc_val  = float(dc_q_host[dc_idx]);
                        float ref_val = float(ref_q_host[ref_idx]);

                        float abs_diff = std::abs(dc_val - ref_val);
                        float rel_diff = abs_diff / (std::abs(ref_val) + 1e-6f);

                        if (abs_diff > 0.01f) {
                            outliers++;
                            if (abs_diff > max_abs_diff) {
                                max_abs_diff  = abs_diff;
                                max_rel_diff  = rel_diff;
                                max_diff_info = "at [" + std::to_string(b) + "," + std::to_string(s) + ","
                                                + std::to_string(h) + "," + std::to_string(d)
                                                + "] dc=" + std::to_string(dc_val) + " ref=" + std::to_string(ref_val);
                            }
                        }
                    }
                }
            }
        }

        std::cout << "outliers = " << outliers << " (" << (100.0 * outliers / total) << "%)\n";
        if (outliers > 0) {
            std::cout << "max_abs_diff = " << max_abs_diff << " (rel=" << max_rel_diff << ") " << max_diff_info << "\n";
        }
    }

    std::cout << "\n>>> Compare output\n";
    // [B, S, H, D]
    Compare(output.data().get(),  //
            output_ref.data().get(),
            kHeadNum * kHeadDim,
            kHeadNum * kHeadDim,
            kBatchSize * kInputLen,
            0);

    {
        // 从 paged blocks 中提取 KV cache
        thrust::universal_vector<T> k_cache_extracted;
        thrust::universal_vector<T> v_cache_extracted;

        extractKVCacheFromBlocks<T, Tkv>(k_cache_extracted,
                                         v_cache_extracted,
                                         blocks,
                                         page_tables,
                                         kBatchSize,
                                         KvHeadNum,
                                         kHeadDim,
                                         kBlockLen,
                                         kContextLen,
                                         kNumLayers);

        std::cout << "\n>>> Compare k_cache (extracted from blocks) - History part (first " << kSequenceLen
                  << " tokens)\n";
        // 提取的 k_cache_extracted layout: [B*H, S, D]
        // k_cache_ref layout: [B*H, S, D]
        // 对比历史部分 (kSequenceLen)
        Compare(k_cache_extracted.data().get(),
                k_cache_ref.data().get(),
                kSequenceLen * kHeadDim,
                kSequenceLen * kHeadDim,
                kBatchSize * KvHeadNum,
                0);

        std::cout << "\n>>> Compare v_cache (extracted from blocks) - History part (first " << kSequenceLen
                  << " tokens)\n";
        Compare(v_cache_extracted.data().get(),
                v_cache_ref.data().get(),
                kSequenceLen * kHeadDim,
                kSequenceLen * kHeadDim,
                kBatchSize * KvHeadNum,
                0);

        std::cout << "\n>>> Compare k_cache (extracted from blocks) - New tokens (input_len=" << kInputLen
                  << " tokens)\n";
        // 对比新增的 kInputLen 个 tokens，位置范围: [kSequenceLen, kSequenceLen + kInputLen)
        // Layout: [B*H, S, D]
        // 总共对比: kBatchSize * KvHeadNum * kInputLen 个样本，每个样本是 kHeadDim 个元素
        size_t total_outliers_k = 0;
        size_t total_elements_k = 0;
        for (size_t bh = 0; bh < kBatchSize * KvHeadNum; ++bh) {
            for (size_t token_idx = 0; token_idx < kInputLen; ++token_idx) {
                const size_t offset = bh * kContextLen * kHeadDim + (kSequenceLen + token_idx) * kHeadDim;
                const T*     src    = k_cache_extracted.data().get() + offset;
                const T*     ref    = k_cache_ref.data().get() + offset;
                for (size_t i = 0; i < kHeadDim; ++i) {
                    float abs_diff = fabs(float(src[i]) - float(ref[i]));
                    if (abs_diff > 0.01f) {
                        total_outliers_k++;
                    }
                    total_elements_k++;
                }
            }
        }
        std::cout << "outliers = " << total_outliers_k << " (" << (100.0 * total_outliers_k / total_elements_k) << "%)"
                  << std::endl;

        std::cout << "\n>>> Compare v_cache (extracted from blocks) - New tokens (input_len=" << kInputLen
                  << " tokens)\n";
        size_t total_outliers_v = 0;
        size_t total_elements_v = 0;
        // Shape: [B*H, S, D]
        for (size_t bh = 0; bh < kBatchSize * KvHeadNum; ++bh) {
            for (size_t token_idx = 0; token_idx < kInputLen; ++token_idx) {
                const size_t offset = bh * kContextLen * kHeadDim + (kSequenceLen + token_idx) * kHeadDim;
                const T*     src    = v_cache_extracted.data().get() + offset;
                const T*     ref    = v_cache_ref.data().get() + offset;
                for (size_t i = 0; i < kHeadDim; ++i) {
                    float abs_diff = fabs(float(src[i]) - float(ref[i]));
                    if (abs_diff > 0.01f) {
                        total_outliers_v++;
                    }
                    total_elements_v++;
                }
            }
        }
        std::cout << "outliers = " << total_outliers_v << " (" << (100.0 * total_outliers_v / total_elements_v) << "%)"
                  << std::endl;
    }

    return 0;
}

// Dispatch function based on quant_policy
template<class T>
int test_trtllm_attention(const TestConfig& config, QuantPolicy quant_policy, int test_iter = 10)
{
    // Dispatch based on quant_policy to select appropriate Tkv type
    if (quant_policy == QuantPolicy::kNone) {
        std::cout << ">>> Testing QuantPolicy::kNone\n";
        return test_trtllm_attention_impl<T, T>(config, QuantPolicy::kNone, test_iter);
    }
    else if (quant_policy == QuantPolicy::kCacheKVFP8) {
        std::cout << ">>> Testing QuantPolicy::kCacheKVFP8\n";
        return test_trtllm_attention_impl<T, __nv_fp8_e4m3>(config, QuantPolicy::kCacheKVFP8, test_iter);
    }
    else if (quant_policy == QuantPolicy::kCacheKVFP4) {
        std::cout << ">>> Testing QuantPolicy::kCacheKVFP4\n";
        return test_trtllm_attention_impl<T, __nv_fp4_e2m1>(config, QuantPolicy::kCacheKVFP4, test_iter);
    }
    else {
        std::cerr << "Error: Unknown quantization policy!" << std::endl;
        return -1;
    }
}

int main(int argc, char* argv[])
{
    bool is_trtllm_attention = turbomind::isSM10x();

    // Parse command line arguments
    int         test_iter = 10;  // Default value
    std::string mode      = "default";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--full") {
            mode = "full";
        }
        else if (arg == "--iter" && i + 1 < argc) {
            test_iter = std::atoi(argv[i + 1]);
            ++i;  // Skip next argument
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n";
            std::cout << "Options:\n";
            std::cout << "  --full          Run full test suite\n";
            std::cout << "  --iter N        Set number of test iterations (default: 10)\n";
            std::cout << "  --help, -h      Show this help message\n";
            return 0;
        }
    }

    std::cout << "Test iterations: " << test_iter << std::endl;

    // Define test configurations to run
    std::vector<TestConfig> test_configs;

    if (mode == "full") {
        // Full test suite
        test_configs = {TestConfigs::Append_Small, TestConfigs::Append_Mid, TestConfigs::Append_Large};
    }
    else {
        // Default: run essential tests
        test_configs = {
            TestConfigs::Append_Small,
        };
    }

    std::cout << "\n========== TRT-LLM FMHA Tests ==========" << std::endl;

    for (const auto& config : test_configs) {
        // Test with FP16
        std::cout << "\n---------- Testing FP16 ----------" << std::endl;
        test_trtllm_attention<half>(config, QuantPolicy::kNone, test_iter);
        // std::cout << "---------------------------------------------------\n";
        // test_trtllm_attention<half>(config, QuantPolicy::kCacheKVFP8, test_iter);
        // std::cout << "---------------------------------------------------\n";
        // test_trtllm_attention<half>(config, QuantPolicy::kCacheKVFP4, test_iter);
        // std::cout << "---------------------------------------------------\n";

        // // Test with BF16
        // std::cout << "\n---------- Testing BF16 ----------" << std::endl;
        // test_trtllm_attention<__nv_bfloat16>(config, QuantPolicy::kNone, test_iter);
        // std::cout << "---------------------------------------------------\n";
        // test_trtllm_attention<__nv_bfloat16>(config, QuantPolicy::kCacheKVFP8, test_iter);
        // std::cout << "---------------------------------------------------\n";
        // test_trtllm_attention<__nv_bfloat16>(config, QuantPolicy::kCacheKVFP4, test_iter);
        // std::cout << "---------------------------------------------------\n";
    }

    std::cout << "\n========== All Tests Completed ==========" << std::endl;
    return 0;
}
