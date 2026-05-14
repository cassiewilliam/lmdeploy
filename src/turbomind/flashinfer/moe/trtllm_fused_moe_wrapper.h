// Copyright (c) OpenMMLab. All rights reserved.
//
// Lightweight wrapper around FlashInfer's tvm-ffi-exported trtllm-gen MoE
// entry points (FP8 per-tensor, FP8 block-scale, NVFP4 / W-NVFP4-A-FP8).
// The .so is resolved by `flashinfer/moe/CMakeLists.txt` from
// flashinfer-python's JIT cache — no vendoring or source-level coupling.
// At runtime we dlopen the .so via tvm-ffi, look up
//   * `trtllm_fp8_per_tensor_scale_moe`
//   * `trtllm_fp8_block_scale_moe`
//   * `trtllm_fp4_block_scale_moe`
// and invoke them with our raw GPU pointers wrapped in `tvm::ffi::Tensor`
// views (no extra copy, no host roundtrip).
//
// Activation: CUDA >= 12.8 builds the bridge, and
// `engine.moe_backend == kTrtllmFusedMoe` decides whether the call-site MoE
// layer dispatches here.
//
// Parameter naming mirrors FlashInfer's Python wrapper
// `flashinfer.fused_moe.trtllm_{fp8_per_tensor,fp8_block,fp4_block}_scale_moe`
// (see `flashinfer/fused_moe/core.py`) so callers familiar with the
// reference `linear_.forward_cutlass_moe`-style integration find the
// same names — `routing_logits`, `gemm{1,2}_weights`, `gemm{1,2}_weights_scale`,
// `routed_scaling_factor`, `routing_method_type`, `local_expert_offset`,
// `local_num_experts`, `intermediate_size`, `top_k`, `n_group`, `topk_group`.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace turbomind::trtllm_fused_moe {

// Mirrors `RoutingMethodType` in flashinfer's csrc.  Keep in sync with
// `flashinfer/fused_moe/core.py:trtllm_fp4_block_scale_moe` docstring.
enum class RoutingMethod : int {
    kDefault          = 0,  // Softmax -> TopK
    kRenormalize      = 1,  // TopK -> Softmax
    kDeepSeekV3       = 2,  // Sigmoid -> bias-add -> Top2-in-group -> Top4-groups -> Top8
    kLlama4           = 3,  // Top1 -> Sigmoid
    kRenormalizeNaive = 4,  // Softmax -> TopK -> Renormalize
};

// Gated activation in the FC1 -> activation -> FC2 path.  Only consumed
// by the FP4 entry; the FP8 entries hard-code SwiGLU upstream.
enum class GatedActType : int {
    kSwiGlu = 3,
    kGeGlu  = 4,
};

// Element type tag for routing tensors / hidden states.  The MoE entries
// only consume the dtypes the upstream kernel accepts:
//   * routing_logits — bf16 or fp32
//   * routing_bias   — bf16
//   * hidden_states  — bf16/fp16 (and fp8_e4m3 for fp8 per-tensor)
//   * weights        — fp8_e4m3 (per-tensor + block-scale) / uint8 packed
//                      e2m1 (NVFP4 / mxfp4)
enum class DType : int {
    kFP16    = 0,
    kBF16    = 1,
    kFP32    = 2,
    kFP8E4M3 = 3,
    kE2M1U8  = 4,  // NVFP4 packed two-per-byte uint8
};

// ---------------------------------------------------------------------------
// BF16 MoE — unquantized routed-expert MoE.
//
// The simplest variant: weights and activations are all bf16, no scale
// tensors anywhere.  Useful first as a "parameter-passing alignment"
// vehicle — once the BF16 path runs end-to-end you've validated the FFI
// plumbing (DLPack views, routing-method enum mapping, autotuner key,
// `loaded()` cubin callback chain) without dragging in the per-tensor /
// block-scale FP8 quantization machinery.
//
// One quirk vs the FP8 entries: the upstream `trtllm_bf16_moe` FFI does
// NOT take an `output` TensorView; it allocates its own and returns it.
// To keep our raw-pointer API uniform with the FP8 variants, the wrapper
// captures the returned ffi::Tensor and `cudaMemcpyAsync`s its bytes
// into the caller-supplied `output` buffer.  That's one extra D2D copy
// per call — acceptable for bring-up, can be removed later by patching
// the upstream signature or pre-allocating via a hook.
// ---------------------------------------------------------------------------
struct Bf16MoeParams {
    void* routing_logits;  // [num_tokens, num_experts]   bf16 / fp32
    DType routing_logits_dtype{DType::kBF16};
    void* routing_bias;  // [num_experts] or nullptr     bf16

    void* hidden_states;  // [num_tokens, hidden_size]    bf16
    void* gemm1_weights;  // [num_experts, 2*intermediate_size, hidden_size]  bf16
    void* gemm2_weights;  // [num_experts, hidden_size, intermediate_size]    bf16

    void* output;  // [num_tokens, hidden_size]    bf16

    int num_tokens;
    int num_experts;
    int hidden_size;
    int intermediate_size;
    int top_k;
    int n_group;     // 0 if N/A
    int topk_group;  // 0 if N/A
    int local_expert_offset;
    int local_num_experts;

    RoutingMethod routing_method;

    bool use_shuffled_weight;  // upstream defaults to true
    int  weight_layout;        // 0 = MajorK
    bool enable_pdl;

    int64_t tile_n;
    int64_t config_index;

    cudaStream_t stream;
};

// ---------------------------------------------------------------------------
// FP8 per-tensor scale MoE.
//
// `output1_scales_scalar` / `output1_scales_gate_scalar` / `output2_scales_scalar`
// are device tensors of shape `[local_num_experts]` (fp32).  Hidden states
// can be bf16/fp16 (auto-quant inside the kernel) or pre-quantised fp8.
// ---------------------------------------------------------------------------
struct Fp8PerTensorMoeParams {
    // routing
    void* routing_logits;  // [num_tokens, num_experts]   bf16 / fp32
    DType routing_logits_dtype{DType::kBF16};
    void* routing_bias;  // [num_experts] or nullptr     bf16

    // activations
    void* hidden_states;  // [num_tokens, hidden_size]    bf16/fp16/fp8_e4m3

    // gemm1 (fc1)
    void* gemm1_weights;               // [num_experts, 2*intermediate_size, hidden_size]   fp8_e4m3
    void* output1_scales_scalar;       // [local_num_experts]          fp32
    void* output1_scales_gate_scalar;  // [local_num_experts]          fp32

    // gemm2 (fc2)
    void* gemm2_weights;          // [num_experts, hidden_size, intermediate_size]     fp8_e4m3
    void* output2_scales_scalar;  // [local_num_experts]          fp32

    // output
    void* output;  // [num_tokens, hidden_size]    bf16

    // shape / routing config
    int num_tokens;
    int num_experts;
    int hidden_size;
    int intermediate_size;
    int top_k;
    int n_group;     // 0 if N/A
    int topk_group;  // 0 if N/A
    int local_expert_offset;
    int local_num_experts;

    double routed_scaling_factor;  // 1.0 if N/A
    bool   use_routing_scales_on_input;

    RoutingMethod routing_method;
    DType         hidden_states_dtype;

    bool enable_pdl;

    // Tactic selection (autotuner key).  `{-1, -1}` lets the wrapper pick
    // the smallest supported tile — same default as the Python entry's
    // `tune_max_num_tokens=8192` heuristic when the AOT autotuner record
    // is missing.
    int64_t tile_n;
    int64_t config_index;

    cudaStream_t stream;
};

// ---------------------------------------------------------------------------
// FP8 block-scale (DeepSeek-V3 style 128x128 block scale) MoE.
// ---------------------------------------------------------------------------
struct Fp8BlockScaleMoeParams {
    void* routing_logits;  // [num_tokens, num_experts]    bf16 / fp32 (DSv3)
    DType routing_logits_dtype{DType::kBF16};
    void* routing_bias;  // [num_experts] or nullptr     bf16

    void* hidden_states;        // [num_tokens, hidden_size]    bf16/fp16/fp8
    void* hidden_states_scale;  // [hidden_size/128, num_tokens]   fp32

    void* gemm1_weights;        // [num_experts, 2*intermediate_size, hidden_size]            fp8
    void* gemm1_weights_scale;  // [num_experts, 2*intermediate_size/128, hidden_size/128]    fp32

    void* gemm2_weights;        // [num_experts, hidden_size, intermediate_size]              fp8
    void* gemm2_weights_scale;  // [num_experts, hidden_size/128, intermediate_size/128]      fp32

    void* output;  // [num_tokens, hidden_size]    bf16

    int num_tokens;
    int num_experts;
    int hidden_size;
    int intermediate_size;
    int top_k;
    int n_group;
    int topk_group;
    int local_expert_offset;
    int local_num_experts;

    double routed_scaling_factor;

    RoutingMethod routing_method;
    DType         hidden_states_dtype;
    bool          use_shuffled_weight;
    int           weight_layout;  // 0 = MajorK (default upstream)
    bool          norm_topk_prob;

    bool enable_pdl;

    int64_t tile_n;
    int64_t config_index;

    cudaStream_t stream;
};

// ---------------------------------------------------------------------------
// NVFP4 / W-NVFP4-A-FP8 MoE (block-scale FP4 weights) — common fields.
//
// `weight_scale_vec_size` selects between NVFP4 (16) and MXFP4 (32).  The
// kernel infers it from `gemm1_weights_scale.numel()` upstream — we mirror
// that inference but expose the resolved value here for traceability.
//
// Activation can be bf16, mxfp8, or NVFP4-packed (uint8).  Weights are
// always uint8-packed e2m1.
//
// The upstream FFI takes both `routing_logits` AND `topk_ids+expert_weights`
// in a single signature, but only one branch is exercised per call (the
// kernel switches on `routing_logits.has_value()`).  We split it into two
// strongly-typed param structs so the call site can't pass nonsense
// combinations.  Common fields go in `Fp4MoeBase`; the two derived structs
// add the discriminating tensor pointers.
// ---------------------------------------------------------------------------
struct Fp4MoeBase {
    void* routing_bias;  // optional; bf16

    void* hidden_states;        // [num_tokens, hidden_size/2] uint8 (nvfp4)
                                // or [num_tokens, hidden_size] bf16/mxfp8
    void* hidden_states_scale;  // optional; fp8 scale tensor

    // gemm1
    void* gemm1_weights;        // [num_experts, 2*intermediate_size, hidden_size/2] uint8 e2m1
    void* gemm1_weights_scale;  // [num_experts, 2*intermediate_size, hidden_size/16 or /32] fp8
    void* gemm1_bias;           // optional, [num_experts, 2*intermediate_size] fp32
    void* gemm1_alpha;          // optional, [num_experts] fp32 (swiglu alpha)
    void* gemm1_beta;           // optional, [num_experts] fp32 (swiglu beta)
    void* gemm1_clamp_limit;    // optional, [num_experts] fp32

    // gemm2
    void* gemm2_weights;  // [num_experts, hidden_size, intermediate_size]
    void* gemm2_weights_scale;
    void* gemm2_bias;  // optional, [num_experts, hidden_size] fp32

    // optional output scaling for W-NVFP4-A-FP8 path
    void* output1_scales_scalar;       // optional; [local_num_experts] fp32
    void* output1_scales_gate_scalar;  // optional; [local_num_experts] fp32
    void* output2_scales_scalar;       // optional; [local_num_experts] fp32

    void* output;  // [num_tokens, hidden_size] bf16

    int num_tokens;
    int num_experts;
    int hidden_size;
    int intermediate_size;
    int top_k;
    int n_group;
    int topk_group;
    int local_expert_offset;
    int local_num_experts;

    int weight_scale_vec_size;  // 16 (NVFP4) or 32 (MXFP4)

    double routed_scaling_factor;

    RoutingMethod routing_method;
    GatedActType  gated_act_type;
    DType         hidden_states_dtype;
    bool          norm_topk_prob;

    bool do_finalize;  // false → return intermediate gemm2_output / weights
    bool enable_pdl;

    int64_t tile_n;
    int64_t config_index;

    cudaStream_t stream;
};

// NVFP4 entry — caller supplies softmax/sigmoid logits, kernel does
// the routing internally.  This is the path qwen3-MoE / DeepSeek-V3-style
// layers take.
struct Fp4BlockScaleMoeParamsRouting: Fp4MoeBase {
    void* routing_logits;  // [num_tokens, num_experts] bf16 / fp32 (DSv3)
    DType routing_logits_dtype{DType::kBF16};
    void* topk_ids_workspace;        // [num_tokens, top_k] int32 scratch, written by routing
    void* expert_weights_workspace;  // [num_tokens, top_k] fp32 scratch, written by routing
};

// NVFP4 entry — caller already ran routing externally and supplies the
// chosen expert IDs + per-token expert weights directly.  This is the
// path `MoeFfnLayer` takes (it has `f2E_` / `scales_` already populated
// by `invokeMoeGate_V2` / `invokeMoeGate_NoAuxTC`).
struct Fp4BlockScaleMoeParamsTopK: Fp4MoeBase {
    void* topk_ids;        // [num_tokens, top_k] int32
    void* expert_weights;  // [num_tokens, top_k] fp32
};

// All five return true on success.  When the .so failed to load, or the
// kernel choked on the supplied parameter combination, they return false;
// callers should fall back to the native MoE path (e.g. cutlass-based
// `MoeFfnLayer` in `src/turbomind/models/llama/moe_ffn_layer.cc`).
bool dispatch_bf16(const Bf16MoeParams& p);
bool dispatch_fp8_per_tensor(const Fp8PerTensorMoeParams& p);
bool dispatch_fp8_block_scale(const Fp8BlockScaleMoeParams& p);
bool dispatch_fp4_block_scale(const Fp4BlockScaleMoeParamsRouting& p);
bool dispatch_fp4_block_scale(const Fp4BlockScaleMoeParamsTopK& p);

// ---------------------------------------------------------------------------
// CUTLASS W4A8_NVFP4_FP8 MoE.
//
// This uses FlashInfer's cutlass_fused_moe runner, not the trtllm-gen routing
// entry above.  The caller must provide already-routed top-k ids/scales in
// contiguous [num_tokens, top_k] layout and pre-quantized FP8 input.
// Weights are packed int64 e2m1:
//   fc1: [E, 2 * intermediate_size, hidden_size / 16]
//   fc2: [E, hidden_size, intermediate_size / 16]
// Weight block scales are packed int32 FP8 groups:
//   fc1: [E, 2 * intermediate_size, hidden_size / 128]
//   fc2: [E, hidden_size, intermediate_size / 128]
// ---------------------------------------------------------------------------
struct CutlassW4A8Nvfp4Fp8MoeParams {
    void* input;                   // [num_tokens, hidden_size] fp8_e4m3
    void* token_selected_experts;  // [num_tokens, top_k] int32
    void* token_final_scales;      // [num_tokens, top_k] fp32

    void* fc1_expert_weights;  // [E, 2I, H/16] int64 packed e2m1
    void* fc2_expert_weights;  // [E, H, I/16] int64 packed e2m1

    void* fc1_weight_block_scale;  // [E, 2I, H/128] int32 packed fp8 scale bytes
    void* fc1_global_scale;        // [E] fp32
    void* fc2_act_global_scale;    // [E] or [1] fp32 quant scale
    void* fc2_weight_block_scale;  // [E, H, I/128] int32 packed fp8 scale bytes
    void* fc2_global_scale;        // [E] fp32
    void* input_sf;                // optional [num_tokens, hidden_size / 32] uint8/ue8m0

    void* output;  // [num_tokens, hidden_size] bf16

    int num_tokens;
    int num_experts;
    int hidden_size;
    int intermediate_size;
    int top_k;

    bool    enable_pdl;
    bool    use_mxfp8_act_scaling;
    int64_t gemm1_tactic;
    int64_t gemm2_tactic;

    cudaStream_t stream;
};

bool dispatch_cutlass_w4a8_nvfp4_fp8(const CutlassW4A8Nvfp4Fp8MoeParams& p);

// Enabled by TurboMind warmup so FlashInfer tactic probing happens during
// startup, not on the first real request.  The service parameter
// `disable_flashinfer_tuning` decides whether warmup enables this.
void SetTuningActive(bool active);

// Set the maximum bytes the per-thread dispatch arena may use.  Called
// once at engine init from `MoeFfnLayer`'s ctor with a value derived
// from `max_token_num`, hidden_dim, inter_size, num_experts, top_k and
// the largest activation dtype the engine will run.  The arena backs
// `dlpack_cuda_alloc` (FlashInfer FFI scratch tensors) so per-call
// `cudaMalloc` / `cudaFree` overhead — measured at ~80 % CPU time on
// FP8 PT-MODELOPT MoE decode — is replaced with a thread-local bump
// pointer.  Idempotent: only the first non-zero call sticks.
void set_dispatch_arena_capacity(size_t bytes);

// True iff the FlashInfer .so loaded successfully and at least one MoE
// entry resolved.  Useful for once-at-init compatibility checks.
bool is_available();

// True iff the host GPU is SM10x (SM100 / SM103 — B200 / B300).  The
// trtllm-gen cubins target that family only; callers should refuse to
// dispatch on anything else.  Cached once per process (the SM family
// doesn't change at runtime).  This is a *capability* probe, not a
// "did the user opt in" check — backend selection is the caller's
// responsibility (see `EngineParam::moe_backend`).
bool is_sm10x_capable();

// ---------------------------------------------------------------------------
// High-level entry — caller-friendly façade for BF16 + FP8 paths.
//
// Owns every decision the per-variant entries above leave to the caller:
//
//   * `is_available()` short-circuit (no need for callers to gate themselves)
//   * variant selection from `weight_dtype` + `weight_block_size`:
//       weight_dtype == kBF16                            → BF16
//       weight_dtype == kFP8E4M3, weight_block_size==1   → per-tensor FP8
//       weight_dtype == kFP8E4M3, weight_block_size==128 → block-scale FP8
//   * routing-method mapping: `topk_method` string + `norm_topk_prob` flag
//                             → `RoutingMethod` enum
//   * autotuner cache: looks up `{tile_n, config_index}` by shape, falls
//                      back to launcher defaults `{-1, -1}` on miss
//   * per-variant param-struct assembly (no need for callers to know
//                                        about `Bf16MoeParams` /
//                                        `Fp8PerTensorMoeParams` /
//                                        `Fp8BlockScaleMoeParams`)
//
// Returns true when the kernel ran end-to-end (output finalised in the
// caller's `output` buffer).  Returns false on any of:
//   * .so not loaded
//   * unsupported (weight_dtype, weight_block_size) combination
//   * required scale tensor is null for the FP8 paths
//   * underlying dispatch call threw
// On false, callers should fall back to the native MoE path (the wrapper
// has already logged the specific reason).
//
// NVFP4 / W-NVFP4-A-FP8 are NOT routed through here today; their FFI
// surface is meaningfully different (separate routing-logits vs topk-ids
// entries, optional swiglu α/β/clamp, packed-uint8 layout) and warrants
// its own façade — use `dispatch_fp4_block_scale(...)` directly until
// that lands.
// ---------------------------------------------------------------------------
struct AutoMoeRequest {
    // Tensors (raw GPU pointers; caller owns memory).  See the per-variant
    // structs above for the exact shape/dtype expectations.
    void* routing_logits;  // [num_tokens, num_experts]
    DType routing_logits_dtype{DType::kBF16};
    void* routing_bias;   // [num_experts] or nullptr (only consumed for noaux_tc)
    void* hidden_states;  // [num_tokens, hidden_size]
    // FP8 block-scale only — required there, ignored for BF16 / per-tensor FP8.
    void* hidden_states_scale;  // [hidden_size/128, num_tokens] fp32
    void* gemm1_weights;
    // FP8 block-scale only — required there, ignored for BF16 / per-tensor FP8.
    void* gemm1_weights_scale;
    void* gemm2_weights;
    void* gemm2_weights_scale;
    void* output;  // [num_tokens, hidden_size] bf16

    // Per-tensor FP8 only — ignored when `weight_dtype != kFP8E4M3` or
    // `weight_block_size != 1`.  Each is `[local_num_experts]` fp32.
    void* output1_scales_scalar;
    void* output1_scales_gate_scalar;
    void* output2_scales_scalar;
    bool  use_routing_scales_on_input;

    // Shape + routing config.
    int num_tokens;
    int num_experts;
    int hidden_size;
    int intermediate_size;
    int top_k;
    int n_group;     // 0 if N/A
    int topk_group;  // 0 if N/A
    int local_expert_offset;
    int local_num_experts;

    DType hidden_states_dtype;  // kFP16 / kBF16 / kFP8E4M3
    DType weight_dtype;         // kBF16 / kFP8E4M3
    // Only consulted when weight_dtype == kFP8E4M3:
    //   1   → per-tensor FP8
    //   128 → block-scale FP8
    int weight_block_size;

    // Routing config — strings/bools as the caller has them.  Wrapper
    // maps to `RoutingMethod` internally (so callers don't need to
    // include flashinfer-specific enums).  Recognised `topk_method`:
    //   * "noaux_tc"   → kDeepSeekV3
    //   * everything else falls back to: kRenormalizeNaive when
    //                                    `norm_topk_prob == true`,
    //                                    kDefault otherwise.
    const char* topk_method;
    bool        norm_topk_prob;
    double      routed_scaling_factor;

    // Block-scale FP8 / BF16 only — caller may signal pre-shuffled
    // weights.  Most turbomind callers leave at `{true, 0}`.  Per-tensor
    // FP8 entry has these fields hard-coded upstream.
    bool use_shuffled_weight;
    int  weight_layout;  // 0 = MajorK

    bool         enable_pdl;
    cudaStream_t stream;
};

bool dispatch_auto(const AutoMoeRequest& req);

// ---------------------------------------------------------------------------
// Autotuner cache.
//
// Each `dispatch_*` call carries `tile_n` / `config_index`; passing
// `{-1, -1}` lets the launcher pick its smallest-supported tile (= the
// safest but rarely the fastest).  In production we want per-shape
// autotuning + cache so the second call picks up the winning config.
//
// `AutotuneKey` is the cache key — same for all three MoE variants.
// `lookup_or_default` returns `{-1, -1}` until the caller calls `record`.
//
// The cache is process-wide (static unordered_map under a mutex) and not
// persistent — a fresh process re-tunes.  Persistence to disk is left
// to the engine layer (mirrors how the FlashInfer Python wrapper drives
// its `tune_max_num_tokens=8192` AOT autotuner separately).
// ---------------------------------------------------------------------------
struct AutotuneKey {
    int variant;  // 0 = bf16, 1 = fp8_per_tensor, 2 = fp8_block,
                  // 3 = fp4_routing, 4 = fp4_topk
    int num_tokens;
    int hidden_size;
    int intermediate_size;
    int num_experts;
    int top_k;
    int local_num_experts;

    bool operator==(const AutotuneKey& o) const noexcept;
};

struct AutotuneTactic {
    int64_t tile_n;
    int64_t config_index;
};

AutotuneTactic lookup_tactic(const AutotuneKey& k);
void           record_tactic(const AutotuneKey& k, AutotuneTactic v);

}  // namespace turbomind::trtllm_fused_moe
