// Copyright (c) OpenMMLab. All rights reserved.

#include "flashinfer_fmha_wrapper.h"
#include "lmdeploy/flashinfer_fmha_config.h"

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace turbomind::flashinfer_fmha {

namespace ffi = tvm::ffi;

namespace {

struct Loaded {
    std::optional<ffi::Module>   mod;
    std::optional<ffi::Function> decode;
    std::optional<ffi::Function> context;
    bool                         ok{false};
    std::string                  error;
};

namespace {

using SetCubinCallbackFn  = void (*)(void (*)(const char*, const char*));
using SetCurrentCubinFn   = void (*)(const char*, int);

std::string cubin_cache_root() {
    if (const char* env = std::getenv("LMDEPLOY_FLASHINFER_FMHA_CUBIN_CACHE"); env && *env) {
        return env;
    }
    if (const char* env = std::getenv("FLASHINFER_CUBIN_DIR"); env && *env) {
        return env;
    }
    if (const char* baked = LMDEPLOY_FLASHINFER_FMHA_CUBIN_CACHE_PATH; baked && *baked) {
        return baked;
    }
    if (const char* home = std::getenv("HOME"); home && *home) {
        return std::string(home) + "/.cache/flashinfer/cubins";
    }
    return "/tmp/flashinfer_cubins";
}

struct CubinCtx {
    SetCurrentCubinFn  set_current{nullptr};
    std::string         last_blob;
} g_cubin_ctx;

void cubin_callback(const char* path, const char* sha256) {
    static const std::string root = cubin_cache_root();
    std::string full = root;
    if (!full.empty() && full.back() != '/') full += '/';
    full += path;

    std::ifstream f(full, std::ios::binary);
    if (!f) {
        std::cerr << "[flashinfer-fmha] cubin not found at " << full << "\n"
                  << "  set LMDEPLOY_FLASHINFER_FMHA_CUBIN_CACHE to a directory "
                     "containing the FlashInfer trtllm-gen cubins.\n";
        if (g_cubin_ctx.set_current) g_cubin_ctx.set_current("", 0);
        (void)sha256;
        return;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    g_cubin_ctx.last_blob = ss.str();
    if (g_cubin_ctx.set_current) {
        g_cubin_ctx.set_current(g_cubin_ctx.last_blob.data(),
                                static_cast<int>(g_cubin_ctx.last_blob.size()));
    }
}

}  // namespace

std::string resolve_so_path() {
    if (const char* env = std::getenv("LMDEPLOY_FLASHINFER_FMHA_SO"); env && *env) {
        return env;
    }
    if (const char* baked = LMDEPLOY_FLASHINFER_FMHA_SO_PATH; baked && *baked) {
        return baked;
    }
    return {};
}

void promote_tvm_ffi_symbols() {
    void* tvm_ffi = dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (!tvm_ffi) {
        tvm_ffi = dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL);
    }
}

Loaded& loaded_state() {
    static Loaded L;
    return L;
}

Loaded& load_once() {
    auto& L = loaded_state();
    static std::once_flag flag;
    std::call_once(flag, [&]() {
        try {
            const std::string so_path = resolve_so_path();
            if (so_path.empty()) {
                throw std::runtime_error(
                    "no trtllm-gen FMHA .so path resolved.  "
                    "Set LMDEPLOY_FLASHINFER_FMHA_SO to the .so produced by "
                    "`flashinfer-python`'s JIT compile, or rebuild with the "
                    "host's flashinfer-python install reachable so CMake's "
                    "`scripts/resolve_so.py` can bake the path at configure time.");
            }

            promote_tvm_ffi_symbols();
            L.mod.emplace(ffi::Module::LoadFromFile(so_path));
            auto dec = (*L.mod)->GetFunction("trtllm_paged_attention_decode");
            auto ctx = (*L.mod)->GetFunction("trtllm_paged_attention_context");
            if (!dec.has_value() || !ctx.has_value()) {
                throw std::runtime_error("missing FFI exports in " + so_path
                                         + " — required: trtllm_paged_attention_decode/"
                                           "context.");
            }
            L.decode.emplace(*dec);
            L.context.emplace(*ctx);

            // Register the cubin callback — must be done after the .so is in
            // memory so that the C symbols inside it are resolvable.
            void* lib = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
            if (!lib) {
                lib = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
            }
            if (!lib) {
                throw std::runtime_error(std::string("dlopen failed for ") + so_path
                                         + ": " + dlerror());
            }
            auto set_cb = reinterpret_cast<SetCubinCallbackFn>(
                dlsym(lib, "FlashInferSetCubinCallback"));
            auto set_cur = reinterpret_cast<SetCurrentCubinFn>(
                dlsym(lib, "FlashInferSetCurrentCubin"));
            if (!set_cb || !set_cur) {
                throw std::runtime_error("missing FlashInferSetCubin* C entries in " + so_path);
            }
            g_cubin_ctx.set_current = set_cur;
            set_cb(cubin_callback);
            L.ok = true;
        } catch (const std::exception& e) {
            L.error = std::string("[flashinfer-fmha] init failed: ") + e.what();
            std::cerr << L.error << std::endl;
        }
    });
    return L;
}

// Helper used by dispatch_* — throws with the saved error message when the
// .so didn't load.  Inline so each dispatch path picks up its own
// `__FILE__:__LINE__`-level traceback.
[[noreturn]] inline void throw_unavailable(const Loaded& L, const char* api) {
    throw std::runtime_error(L.error.empty()
                                 ? std::string("[flashinfer-fmha] ") + api
                                       + " called before successful init"
                                 : L.error + " (called from " + api + ")");
}

DLDataType to_dl(DType d) {
    switch (d) {
        case DType::kFP16:    return DLDataType{kDLFloat,  16, 1};
        case DType::kBF16:    return DLDataType{kDLBfloat, 16, 1};
        case DType::kFP8E4M3: return DLDataType{kDLFloat8_e4m3fn, 8, 1};
        case DType::kFP4E2M1: return DLDataType{kDLUInt, 8, 1};
    }
    return DLDataType{kDLFloat, 16, 1};
}

inline int current_device_id() {
    int dev_id = 0;
    cudaError_t err = cudaGetDevice(&dev_id);
    if (err != cudaSuccess) {
        std::cerr << "[flashinfer-fmha] cudaGetDevice failed: "
                  << cudaGetErrorString(err) << " — falling back to device 0\n";
        return 0;
    }
    return dev_id;
}

class TvmFfiStreamScope {
public:
    explicit TvmFfiStreamScope(cudaStream_t stream): device_id_{current_device_id()} {
        if (!stream) {
            std::cerr << "[flashinfer-fmha] WARNING: TvmFfiStreamScope got nullptr stream "
                         "— FMHA kernel would run on stream 0; check upstream wiring\n";
        }
        int rc = TVMFFIEnvSetStream(kDLCUDA, device_id_, stream, &prev_);
        if (rc != 0) {
            std::cerr << "[flashinfer-fmha] TVMFFIEnvSetStream(set) rc=" << rc
                      << " device=" << device_id_ << "\n";
        }
    }
    ~TvmFfiStreamScope() {
        TVMFFIEnvSetStream(kDLCUDA, device_id_, prev_, nullptr);
    }
    TvmFfiStreamScope(const TvmFfiStreamScope&) = delete;
    TvmFfiStreamScope& operator=(const TvmFfiStreamScope&) = delete;

private:
    int                device_id_{0};
    TVMFFIStreamHandle prev_{nullptr};
};

// A heap-allocated bundle that backs an `ffi::Tensor` view of an existing
// GPU buffer.  We pass it to `FromDLPack`; the deleter `delete`s the bundle
// (which frees shape/strides storage) but never touches the user's data.
struct DLBundle {
    DLManagedTensor       managed;
    std::vector<int64_t>  shape;
    std::vector<int64_t>  strides;

    static void Deleter(DLManagedTensor* mt) {
        delete reinterpret_cast<DLBundle*>(mt->manager_ctx);
    }
};

ffi::Tensor make_view(void* data,
                      std::vector<int64_t> shape,
                      std::vector<int64_t> strides,
                      DLDataType dtype,
                      int dev_id = -1) {
    if (dev_id < 0) {
        dev_id = current_device_id();
    }
    auto bundle = new DLBundle;
    bundle->shape = std::move(shape);
    bundle->strides = std::move(strides);
    bundle->managed.dl_tensor.data = data;
    bundle->managed.dl_tensor.device = DLDevice{kDLCUDA, dev_id};
    bundle->managed.dl_tensor.ndim = static_cast<int32_t>(bundle->shape.size());
    bundle->managed.dl_tensor.dtype = dtype;
    bundle->managed.dl_tensor.shape = bundle->shape.data();
    bundle->managed.dl_tensor.strides = bundle->strides.empty() ? nullptr : bundle->strides.data();
    bundle->managed.dl_tensor.byte_offset = 0;
    bundle->managed.manager_ctx = bundle;
    bundle->managed.deleter = &DLBundle::Deleter;
    return ffi::Tensor::FromDLPack(&bundle->managed);
}

ffi::Variant<double, ffi::Tensor> scale_variant(double scalar, float* tensor_ptr, int64_t n) {
    if (tensor_ptr == nullptr) {
        return ffi::Variant<double, ffi::Tensor>(scalar);
    }
    return ffi::Variant<double, ffi::Tensor>(
        make_view(tensor_ptr, {n}, {}, DLDataType{kDLFloat, 32, 1}));
}

ffi::Optional<ffi::Tensor> block_scale_view(const FmhaParams& p, void* data, int64_t num_pages) {
    if (!data) {
        return ffi::Optional<ffi::Tensor>();
    }
    const int64_t scale_dim = p.head_dim_qk / 16;
    return make_view(data,
                     {num_pages, p.num_kv_heads, p.page_size, scale_dim},
                     {p.kv_scale_stride_batch, p.kv_scale_stride_heads, scale_dim, 1},
                     DLDataType{kDLFloat8_e4m3fn, 8, 1});
}

ffi::Tensor kv_cache_view(const FmhaParams& p, void* data, int64_t num_pages, int head_dim) {
    int64_t view_head_dim = head_dim;
    int64_t stride_batch  = p.kv_stride_batch;
    int64_t stride_heads  = p.kv_stride_heads;
    int64_t stride_tokens = p.kv_stride_keys_values;
    if (p.kv_dtype == DType::kFP4E2M1) {
        if (head_dim % 2 || stride_batch % 2 || stride_heads % 2 || stride_tokens % 2) {
            throw std::runtime_error("[flashinfer-fmha] FP4 KV cache expects even head_dim and strides");
        }
        view_head_dim = head_dim / 2;
        stride_batch /= 2;
        stride_heads /= 2;
        stride_tokens /= 2;
    }
    return make_view(data,
                     {num_pages, p.num_kv_heads, p.page_size, view_head_dim},
                     {stride_batch, stride_heads, stride_tokens, 1},
                     to_dl(p.kv_dtype));
}

}  // namespace

void initialize() {
    auto& L = load_once();
    if (!L.ok) {
        throw_unavailable(L, "initialize");
    }
}

bool is_available() {
    return load_once().ok;
}

bool dispatch_decode(const FmhaParams& p) {
    auto& L = loaded_state();
    if (!L.ok) throw_unavailable(L, "dispatch_decode");

    TvmFfiStreamScope ffi_stream{p.stream};

    const int64_t sum_q = static_cast<int64_t>(p.batch_size) * p.max_q_len;
    auto qry = make_view(
        p.query, {sum_q, p.num_qo_heads, p.head_dim_qk}, {},
        to_dl(p.q_dtype));
    int64_t num_pages = p.num_pages_in_pool > 0
                            ? static_cast<int64_t>(p.num_pages_in_pool)
                            : static_cast<int64_t>(p.max_num_blocks_per_seq) * p.batch_size;
    auto kc = kv_cache_view(p, p.key_cache, num_pages, p.head_dim_qk);
    auto vc = kv_cache_view(p, p.value_cache, num_pages, p.head_dim_vo);
    auto out = make_view(
        p.out, {sum_q, p.num_qo_heads, p.head_dim_vo}, {},
        to_dl(p.o_dtype));
    auto ws = make_view(
        p.workspace_buffer, {static_cast<int64_t>(p.workspace_size)}, {},
        DLDataType{kDLInt, 8, 1});
    auto bt = make_view(
        p.block_tables, {p.batch_size, 2, p.max_num_blocks_per_seq}, {},
        DLDataType{kDLInt, 32, 1});
    auto sl = make_view(
        p.seq_lens, {p.batch_size}, {}, DLDataType{kDLInt, 32, 1});
    auto k_scales = block_scale_view(p, p.key_block_scales, num_pages);
    auto v_scales = block_scale_view(p, p.value_block_scales, num_pages);

    ffi::Optional<ffi::Tensor> cuq_opt;
    if (p.cum_seq_lens_q != nullptr) {
        cuq_opt = make_view(
            p.cum_seq_lens_q, {p.batch_size + 1}, {}, DLDataType{kDLInt, 32, 1});
    }

    try {
        (*L.decode)(
            out,
            ffi::Optional<ffi::Tensor>(),                               // out_scale_factor
            qry, kc, vc, ws, bt, sl,
            static_cast<int64_t>(p.max_q_len),
            static_cast<int64_t>(p.max_kv_len),
            scale_variant(p.bmm1_scale, p.bmm1_scale_log2_ptr, 1),
            scale_variant(p.bmm2_scale, p.bmm2_scale_ptr,      1),
            static_cast<double>(0.0),                                   // o_sf_scale
            static_cast<int64_t>(-1),                                   // o_sf_vec_size
            static_cast<int64_t>(0),                                    // o_sf_start_index
            static_cast<int64_t>(p.batch_size),
            static_cast<int64_t>(p.window_left),
            static_cast<int64_t>(0),                                    // sparse_mla_top_k
            static_cast<int64_t>(p.sm_count),
            p.enable_pdl,                                               // enable_pdl
            static_cast<int64_t>(p.workspace_size),
            ffi::Optional<ffi::Tensor>(),                               // attention_sinks
            cuq_opt,                                                    // cum_seq_lens_q
            k_scales,                                                   // key_block_scales
            v_scales,                                                   // value_block_scales
            ffi::Optional<float>(),                                     // skip_softmax_threshold
            ffi::Optional<bool>(false));                                // uses_shared_paged_kv_idx
        return true;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("[flashinfer-fmha] decode call failed: ")
                                 + e.what());
    }
}

bool dispatch_context(const FmhaParams& p) {
    auto& L = loaded_state();
    if (!L.ok) throw_unavailable(L, "dispatch_context");

    TvmFfiStreamScope ffi_stream{p.stream};

    auto qry = make_view(
        p.query,
        {static_cast<int64_t>(p.max_q_len) * p.batch_size, p.num_qo_heads, p.head_dim_qk},
        {},
        to_dl(p.q_dtype));
    int64_t num_pages = p.num_pages_in_pool > 0
                            ? static_cast<int64_t>(p.num_pages_in_pool)
                            : static_cast<int64_t>(p.max_num_blocks_per_seq) * p.batch_size;
    auto kc = kv_cache_view(p, p.key_cache, num_pages, p.head_dim_qk);
    auto vc = kv_cache_view(p, p.value_cache, num_pages, p.head_dim_vo);
    auto out = make_view(
        p.out,
        {static_cast<int64_t>(p.max_q_len) * p.batch_size, p.num_qo_heads, p.head_dim_vo},
        {},
        to_dl(p.o_dtype));
    auto ws = make_view(
        p.workspace_buffer, {static_cast<int64_t>(p.workspace_size)}, {},
        DLDataType{kDLInt, 8, 1});
    auto bt = make_view(
        p.block_tables, {p.batch_size, 2, p.max_num_blocks_per_seq}, {},
        DLDataType{kDLInt, 32, 1});
    auto sl = make_view(
        p.seq_lens, {p.batch_size}, {}, DLDataType{kDLInt, 32, 1});
    auto cuq = make_view(
        p.cum_seq_lens_q, {p.batch_size + 1}, {}, DLDataType{kDLInt, 32, 1});
    auto cuk = make_view(
        p.cum_seq_lens_kv, {p.batch_size + 1}, {}, DLDataType{kDLInt, 32, 1});
    auto k_scales = block_scale_view(p, p.key_block_scales, num_pages);
    auto v_scales = block_scale_view(p, p.value_block_scales, num_pages);

    try {
        (*L.context)(
            out,
            ffi::Optional<ffi::Tensor>(),                               // out_scale_factor
            qry, kc, vc, ws, bt, sl,
            static_cast<int64_t>(p.max_q_len),
            static_cast<int64_t>(p.max_kv_len),
            scale_variant(p.bmm1_scale, p.bmm1_scale_log2_ptr, 1),
            scale_variant(p.bmm2_scale, p.bmm2_scale_ptr,      1),
            static_cast<double>(0.0),
            static_cast<int64_t>(-1),
            static_cast<int64_t>(0),
            static_cast<int64_t>(p.batch_size),
            static_cast<int64_t>(p.window_left),
            cuq, cuk,
            static_cast<int64_t>(p.sm_count),
            p.enable_pdl,                                               // enable_pdl
            static_cast<int64_t>(p.workspace_size),
            ffi::Optional<ffi::Tensor>(),                               // attention_sinks
            k_scales,                                                   // key_block_scales
            v_scales,                                                   // value_block_scales
            ffi::Optional<float>(),                                     // skip_softmax_threshold
            ffi::Optional<bool>(false));                                // uses_shared_paged_kv_idx
        return true;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("[flashinfer-fmha] context call failed: ")
                                 + e.what());
    }
}

}  // namespace turbomind::flashinfer_fmha
