// Copyright (c) OpenMMLab. All rights reserved.
//
// Smoke test: allocate dummy GPU tensors, call dispatch_decode through the
// wrapper, sanity-check the FlashInfer .so actually loads and the call
// returns without throwing.  Invoke with:
//   ./bin/flashinfer_fmha_smoke   (no env required)
//
// We don't validate output values here — that's `test_dc_fmha`'s job.
// Goal is to confirm the FFI plumbing.

#include "flashinfer_fmha_wrapper.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t e = (call);                                        \
        if (e != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " __FILE__ ":" << __LINE__     \
                      << " - " << cudaGetErrorString(e) << "\n";       \
            std::exit(1);                                              \
        }                                                              \
    } while (0)

int main() {
    namespace fi = turbomind::flashinfer_fmha;

    if (!fi::is_available()) {
        std::cerr << "[smoke] FlashInfer FMHA not available — load failed.\n";
        return 1;
    }
    std::cout << "[smoke] FlashInfer FMHA loaded ok\n";

    // Minimal Qwen3-0.6B-ish decode shape: bs=1, q_len=1, NH=16, NKV=8, D=128, history=64, page=64.
    constexpr int B = 1, Q = 1, NH = 16, NKV = 8, D = 128, H = 64, PAGE = 64;
    const int max_pages = (H + Q + PAGE - 1) / PAGE;  // == 1

    void *q, *k, *v, *o, *ws;
    CHECK(cudaMalloc(&q, sizeof(uint16_t) * B * Q * NH * D));
    CHECK(cudaMalloc(&k, sizeof(uint16_t) * max_pages * B * NKV * PAGE * D));
    CHECK(cudaMalloc(&v, sizeof(uint16_t) * max_pages * B * NKV * PAGE * D));
    CHECK(cudaMalloc(&o, sizeof(uint16_t) * B * Q * NH * D));
    CHECK(cudaMalloc(&ws, 64 * 1024 * 1024));
    CHECK(cudaMemset(q, 0, sizeof(uint16_t) * B * Q * NH * D));
    CHECK(cudaMemset(k, 0, sizeof(uint16_t) * max_pages * B * NKV * PAGE * D));
    CHECK(cudaMemset(v, 0, sizeof(uint16_t) * max_pages * B * NKV * PAGE * D));
    CHECK(cudaMemset(o, 0xCC, sizeof(uint16_t) * B * Q * NH * D));  // sentinel
    CHECK(cudaMemset(ws, 0, 64 * 1024 * 1024));

    int *bt, *sl;
    CHECK(cudaMalloc(&bt, sizeof(int) * B * max_pages));
    CHECK(cudaMalloc(&sl, sizeof(int) * B));
    std::vector<int> bt_h(B * max_pages, 0);
    std::vector<int> sl_h(B, H);
    CHECK(cudaMemcpy(bt, bt_h.data(), bt_h.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sl, sl_h.data(), sl_h.size() * sizeof(int), cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    fi::FmhaParams p{};
    p.out = o;
    p.query = q;
    p.key_cache = k;
    p.value_cache = v;
    p.workspace_buffer = ws;
    p.block_tables = bt;
    p.seq_lens = sl;
    p.q_dtype = fi::DType::kBF16;
    p.kv_dtype = fi::DType::kBF16;
    p.o_dtype = fi::DType::kBF16;
    p.batch_size = B;
    p.max_q_len = Q;
    p.max_kv_len = H;
    p.num_qo_heads = NH;
    p.num_kv_heads = NKV;
    p.head_dim_qk = D;
    p.head_dim_vo = D;
    p.page_size = PAGE;
    p.max_num_blocks_per_seq = max_pages;
    p.kv_stride_batch = NKV * PAGE * D;
    p.kv_stride_heads = PAGE * D;
    p.kv_stride_keys_values = D;
    p.bmm1_scale = 1.0 / 11.31371;  // 1/sqrt(128)
    p.bmm2_scale = 1.0;
    p.window_left = -1;
    p.sm_count = prop.multiProcessorCount;
    p.workspace_size = 64 * 1024 * 1024;
    p.stream = nullptr;

    bool ok = fi::dispatch_decode(p);
    CHECK(cudaDeviceSynchronize());

    std::cout << "[smoke] dispatch_decode returned " << (ok ? "TRUE" : "FALSE") << "\n";
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
    cudaFree(ws); cudaFree(bt); cudaFree(sl);
    return ok ? 0 : 2;
}
