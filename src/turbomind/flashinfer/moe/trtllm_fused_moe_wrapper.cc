// Copyright (c) OpenMMLab. All rights reserved.

#include "trtllm_fused_moe_wrapper.h"
#include "lmdeploy/trtllm_fused_moe_config.h"
#include "src/turbomind/utils/logger.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/tvm_ffi.h>

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace turbomind::trtllm_fused_moe {

namespace ffi = tvm::ffi;

namespace {

std::atomic<bool> g_tuning_active{false};

// Holds the loaded .so + the four FFI entry points for the trtllm-gen
// MoE kernels.  ffi::Module isn't default-constructible, so wrap behind
// an Optional + lazy populate (mirrors flashinfer/attention wrapper).
struct Loaded {
    std::optional<ffi::Module>   mod;
    std::optional<ffi::Function> bf16;
    std::optional<ffi::Function> fp8_per_tensor;
    std::optional<ffi::Function> fp8_block_scale;
    std::optional<ffi::Function> fp4_block_scale;
    std::optional<ffi::Function> valid_configs;
    bool                         ok{false};
};

struct CutlassLoaded {
    std::optional<ffi::Module>   mod;
    std::optional<ffi::Module>   runner;
    std::optional<ffi::Function> run_moe;
    std::optional<ffi::Function> run_gemm_profile;
    std::optional<ffi::Function> get_gemm1_tactic_count;
    std::optional<ffi::Function> get_gemm2_tactic_count;
    bool                         ok{false};
};

// FlashInfer's cubin loader is callback-driven — see
// `flashinfer/attention/flashinfer_fmha_wrapper.cc` for the full
// rationale.  Same machinery is needed for the MoE/BMM cubin set; the
// only difference is the cache root: we look at
// `LMDEPLOY_FLASHINFER_MOE_CUBIN_CACHE`, then `FLASHINFER_CUBIN_DIR`,
// then the standard pip cache location.

namespace {

using SetCubinCallbackFn = void (*)(void (*)(const char*, const char*));
using SetCurrentCubinFn  = void (*)(const char*, int);

std::string cubin_cache_root()
{
    if (const char* env = std::getenv("LMDEPLOY_FLASHINFER_MOE_CUBIN_CACHE"); env && *env) {
        return env;
    }
    if (const char* env = std::getenv("FLASHINFER_CUBIN_DIR"); env && *env) {
        return env;
    }
    if (const char* home = std::getenv("HOME"); home && *home) {
        return std::string(home) + "/.cache/flashinfer/cubins";
    }
    return "/tmp/flashinfer_cubins";
}

struct CubinCtx {
    SetCurrentCubinFn set_current{nullptr};
    std::string       last_blob;
} g_cubin_ctx;

std::string shell_quote(const std::string& s)
{
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        }
        else {
            out += c;
        }
    }
    out += "'";
    return out;
}

bool valid_cubin_path(const char* path)
{
    if (!path || !*path || path[0] == '/') {
        return false;
    }
    std::string prev;
    const char* p = path;
    while (*p) {
        const unsigned char c = static_cast<unsigned char>(*p++);
        if (!(std::isalnum(c) || c == '_' || c == '-' || c == '.' || c == '/')) {
            return false;
        }
    }
    return std::string(path).find("..") == std::string::npos;
}

bool auto_download_cubin(const std::string& full, const char* path, const char* sha256)
{
    if (!valid_cubin_path(path)) {
        return false;
    }
    if (const char* env = std::getenv("FLASHINFER_NO_DOWNLOAD"); env && *env) {
        return false;
    }
    if (const char* env = std::getenv("LMDEPLOY_FLASHINFER_MOE_NO_DOWNLOAD"); env && *env) {
        return false;
    }

    std::string repo = "https://edge.urm.nvidia.com/artifactory/"
                       "sw-kernelinferencelibrary-public-generic-local/";
    if (const char* env = std::getenv("FLASHINFER_CUBINS_REPOSITORY"); env && *env) {
        repo = env;
    }
    if (!repo.empty() && repo.back() != '/') {
        repo += '/';
    }
    static std::atomic<unsigned long> download_seq{0};
    const auto                        seq = download_seq.fetch_add(1, std::memory_order_relaxed);
    const std::string                 url = repo + path;
    const std::string                 tmp =
        full + ".tmp." + std::to_string(static_cast<long long>(getpid())) + "." + std::to_string(seq);

    std::string cmd =
        "curl -fsSL --retry 3 --connect-timeout 10 --create-dirs -o " + shell_quote(tmp) + " " + shell_quote(url);
    if (std::system(cmd.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }

    if (sha256 && *sha256) {
        const std::string line = std::string(sha256) + "  " + tmp + "\n";
        cmd                    = "printf %s " + shell_quote(line) + " | sha256sum -c - >/dev/null";
        if (std::system(cmd.c_str()) != 0) {
            std::cerr << "[flashinfer-moe] cubin checksum mismatch for " << full << "\n";
            std::remove(tmp.c_str());
            return false;
        }
    }

    if (std::rename(tmp.c_str(), full.c_str()) != 0) {
        std::cerr << "[flashinfer-moe] failed to move downloaded cubin to " << full << "\n";
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

void cubin_callback(const char* path, const char* sha256)
{
    static const std::string root  = cubin_cache_root();
    static const bool        debug = [] {
        const char* env = std::getenv("LMDEPLOY_FLASHINFER_MOE_CUBIN_DEBUG");
        return env && *env && std::string(env) != "0";
    }();
    std::string full = root;
    if (!full.empty() && full.back() != '/')
        full += '/';
    full += path;

    std::ifstream f(full, std::ios::binary);
    if (!f) {
        std::cerr << "[flashinfer-moe] cubin not found at " << full << ", fetching...\n";
        if (auto_download_cubin(full, path, sha256)) {
            f.open(full, std::ios::binary);
        }
        if (!f) {
            std::cerr << "[flashinfer-moe] cubin unavailable at " << full << "\n"
                      << "  set LMDEPLOY_FLASHINFER_MOE_CUBIN_CACHE to a directory "
                         "containing the FlashInfer trtllm-gen BMM cubins.\n";
            if (g_cubin_ctx.set_current)
                g_cubin_ctx.set_current("", 0);
            return;
        }
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    g_cubin_ctx.last_blob = ss.str();
    if (debug) {
        std::cerr << "[flashinfer-moe] cubin loaded " << full << " (" << g_cubin_ctx.last_blob.size() << " bytes)\n";
        const auto            slash   = full.find_last_of('/');
        std::string           func    = slash == std::string::npos ? full : full.substr(slash + 1);
        constexpr const char* kSuffix = ".cubin";
        if (func.size() > std::strlen(kSuffix)
            && func.compare(func.size() - std::strlen(kSuffix), std::strlen(kSuffix), kSuffix) == 0) {
            func.resize(func.size() - std::strlen(kSuffix));
            if (!func.empty())
                func[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(func[0])));
            CUmodule       mod{};
            CUfunction     fn{};
            const CUresult load_rc = cuModuleLoadData(&mod, g_cubin_ctx.last_blob.data());
            CUresult       func_rc = CUDA_ERROR_INVALID_HANDLE;
            if (load_rc == CUDA_SUCCESS) {
                func_rc = cuModuleGetFunction(&fn, mod, func.c_str());
                cuModuleUnload(mod);
            }
            std::cerr << "[flashinfer-moe] cubin driver probe load_rc=" << load_rc << " func_rc=" << func_rc
                      << " func=" << func << "\n";
        }
    }
    if (g_cubin_ctx.set_current) {
        g_cubin_ctx.set_current(g_cubin_ctx.last_blob.data(), static_cast<int>(g_cubin_ctx.last_blob.size()));
    }
}

}  // namespace

// DLPack tensor allocator — backs `TVMFFIEnvTensorAlloc`, which the
// flashinfer-python 0.6.9+ `trtllm_bf16_moe` (and other newer FFI
// entries) call to allocate intermediate workspace tensors.
//
// PROBLEM: the previous implementation used synchronous cudaMalloc /
// cudaFree.  Decode-phase profiling on Qwen3-30B-A3B FP8 PT-MODELOPT
// showed ~16 k cudaMalloc + ~16 k cudaFree per second consuming ~80 %
// of CPU time and leaving the GPU 96 % idle.  Each MoE call allocates
// several scratch tensors (routing tables, BMM workspaces, finalize
// buffers); over 48 layers × 8 parallel decode steps this dominates.
//
// SOLUTION: pre-allocate one large GPU arena per worker thread on
// first use, and serve allocations as bump allocations into that
// arena.  The arena is reset at the top of every FFI dispatch (via
// `TvmFfiStreamScope`'s ctor — that's the entry point of every
// trtllm_*_moe FFI call we make), so the arena's lifetime spans
// exactly one dispatch.  Hot path becomes a thread-local pointer
// bump — no cuda API call, no sync, no host blocking.  Allocations
// that overflow the arena fall back to cudaMallocAsync on the
// dispatch stream (which still beats the sync path).
namespace {

thread_local cudaStream_t g_tls_dispatch_stream = nullptr;

// Capacity for the per-thread dispatch arena.  Set once at engine
// init via `set_dispatch_arena_capacity` (called from MoeFfnLayer's
// ctor with a size derived from real model params: max_token_num,
// hidden_dim, inter_size, num_experts, top_k).  Arena is allocated
// lazily on the first dispatch in each worker thread so processes
// that never dispatch pay nothing.
inline std::atomic<size_t>& g_arena_capacity_bytes()
{
    static std::atomic<size_t> v{0};
    return v;
}

// Singleton arena: one device-resident base buffer shared by all worker
// threads (and the cuda-graph capture thread).  The base pointer is
// eagerly allocated by `set_dispatch_arena_capacity()` on the engine init
// thread — by the time the first dispatch / cuda-graph capture runs,
// `g_arena_base` is already set, so no `cudaMalloc` ever happens on a
// hot stream.  Bump-pointer state stays per-thread so threads don't stomp
// each other and so within one captured forward pass the offsets are
// deterministic across replays.
struct GlobalArena {
    std::mutex mu;
    void*      base   = nullptr;
    size_t     size   = 0;
    int        dev_id = -1;

    bool ensure(size_t want)
    {
        std::lock_guard<std::mutex> g(mu);
        if (base) {
            return true;
        }
        if (want == 0) {
            return false;
        }
        if (cudaGetDevice(&dev_id) != cudaSuccess) {
            return false;
        }
        const cudaError_t err = cudaMalloc(&base, want);
        if (err != cudaSuccess) {
            std::cerr << "[flashinfer-moe] global arena cudaMalloc(" << want << ") failed: "
                      << cudaGetErrorString(err) << " — falling back to per-call mallocs.\n";
            base = nullptr;
            cudaGetLastError();
            return false;
        }
        size = want;
        return true;
    }
};

inline GlobalArena& g_arena()
{
    static GlobalArena a;
    return a;
}

// Per-thread bump pointer.  Reset at the top of every FFI call by
// TvmFfiStreamScope::ctor — the previous dispatch's allocations are
// reclaimable because subsequent kernels on the same stream happen-after.
thread_local size_t g_tls_arena_used = 0;

// Returns a stable pointer into the global arena, or nullptr if it would
// overflow.  Aligned to 256 B.  The pointer is stable across cuda-graph
// replays because `g_arena().base` is allocated once and never freed.
inline void* arena_alloc(size_t bytes)
{
    auto& a = g_arena();
    if (!a.base) {
        return nullptr;
    }
    constexpr size_t kAlign = 256;
    const size_t     start  = (g_tls_arena_used + kAlign - 1) & ~(kAlign - 1);
    if (start + bytes > a.size) {
        return nullptr;
    }
    g_tls_arena_used = start + bytes;
    return static_cast<char*>(a.base) + start;
}

struct DLPackAllocBundle {
    DLManagedTensorVersioned managed;
    std::vector<int64_t>     shape;
    bool                     from_arena = false;
    cudaStream_t             stream     = nullptr;  // for async free fallback
};

int dlpack_cuda_alloc(DLTensor*                  prototype,
                      DLManagedTensorVersioned** out,
                      void* /*error_ctx*/,
                      void (*SetError)(void*, const char*, const char*))
{
    int64_t numel = 1;
    for (int i = 0; i < prototype->ndim; ++i) {
        numel *= prototype->shape[i];
    }
    const int    bits_per_elem = prototype->dtype.bits * prototype->dtype.lanes;
    const size_t bytes         = (static_cast<size_t>(numel) * bits_per_elem + 7) / 8;

    void* data       = nullptr;
    bool  from_arena = false;
    if (bytes > 0) {
        // Hot path: bump-allocate from the eager global arena.  Pointer
        // is stable across cuda-graph replays because the base buffer
        // was allocated once at engine init.
        data = arena_alloc(bytes);
        if (data) {
            from_arena = true;
        }
        else {
            // Cold/fallback path: arena overflow or arena init failed.
            // Use cudaMallocAsync on the dispatch stream so we still
            // avoid the host-blocking sync of plain cudaMalloc.  This
            // path is graph-capture-incompatible (each replay would
            // realloc) — overflow during capture is a sizing bug.
            cudaStream_t s   = g_tls_dispatch_stream;
            cudaError_t  err = s ? cudaMallocAsync(&data, bytes, s) : cudaMalloc(&data, bytes);
            if (err != cudaSuccess) {
                SetError(nullptr, "RuntimeError", cudaGetErrorString(err));
                return -1;
            }
        }
    }

    auto* bundle                = new DLPackAllocBundle;
    bundle->shape.assign(prototype->shape, prototype->shape + prototype->ndim);
    bundle->from_arena          = from_arena;
    bundle->stream              = g_tls_dispatch_stream;
    bundle->managed.version     = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    bundle->managed.manager_ctx = bundle;
    bundle->managed.deleter     = [](DLManagedTensorVersioned* self) {
        auto* b = static_cast<DLPackAllocBundle*>(self->manager_ctx);
        // Arena allocations live until the next reset; nothing to free
        // here except the bookkeeping bundle itself.
        if (self->dl_tensor.data && !b->from_arena) {
            if (b->stream) {
                cudaFreeAsync(self->dl_tensor.data, b->stream);
            }
            else {
                cudaFree(self->dl_tensor.data);
            }
        }
        delete b;
    };
    bundle->managed.flags                 = 0;
    bundle->managed.dl_tensor.data        = data;
    bundle->managed.dl_tensor.device      = prototype->device;
    bundle->managed.dl_tensor.ndim        = prototype->ndim;
    bundle->managed.dl_tensor.dtype       = prototype->dtype;
    bundle->managed.dl_tensor.shape       = bundle->shape.data();
    bundle->managed.dl_tensor.strides     = nullptr;  // contiguous
    bundle->managed.dl_tensor.byte_offset = 0;

    *out = &bundle->managed;
    return 0;
}

}  // namespace

void promote_tvm_ffi_symbols()
{
    // FlashInfer's JIT'd MoE .so lazily calls TVMFFIEnvTensorAlloc from
    // libtvm_ffi.  _turbomind links libtvm_ffi, but Python extension
    // loading does not guarantee its symbols are globally visible to a
    // later dlopen'd JIT .so.  Re-open/promote it with RTLD_GLOBAL before
    // resolving FlashInfer entry points.
    void* tvm_ffi = dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (!tvm_ffi) {
        tvm_ffi = dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!tvm_ffi) {
        std::cerr << "[flashinfer-moe] warning: failed to promote libtvm_ffi.so: " << dlerror() << "\n";
    }
}

bool ensure_dlpack_allocator()
{
    static std::once_flag flag;
    static int            rc = -1;
    std::call_once(flag, [&]() {
        // Register the DLPack allocator for the FFI's internal scratch
        // tensors (`TVMFFIEnvTensorAlloc` calls back into this).  Both the
        // trtllm-gen and CUTLASS fused-MoE modules may allocate temporaries.
        // `write_to_global_context=1` keeps the allocator visible outside the
        // current TLS scope.
        rc = TVMFFIEnvSetDLPackManagedTensorAllocator(dlpack_cuda_alloc, /*write_to_global_context=*/1, nullptr);
        if (rc != 0) {
            std::cerr << "[flashinfer-moe] TVMFFIEnvSetDLPackManagedTensorAllocator rc=" << rc << "\n";
        }
    });
    return rc == 0;
}

Loaded& loaded()
{
    static Loaded         L;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
        try {
            const char*       override_path = std::getenv("LMDEPLOY_FLASHINFER_MOE_SO");
            const std::string so_path =
                override_path && *override_path ? override_path : LMDEPLOY_FLASHINFER_MOE_SO_PATH;

            promote_tvm_ffi_symbols();

            L.mod.emplace(ffi::Module::LoadFromFile(so_path));
            auto bf16    = (*L.mod)->GetFunction("trtllm_bf16_moe");
            auto fp8_pt  = (*L.mod)->GetFunction("trtllm_fp8_per_tensor_scale_moe");
            auto fp8_blk = (*L.mod)->GetFunction("trtllm_fp8_block_scale_moe");
            auto fp4_blk = (*L.mod)->GetFunction("trtllm_fp4_block_scale_moe");
            auto valid   = (*L.mod)->GetFunction("trtllm_get_valid_moe_configs");
            // We accept a partial export set: a build that only enables a
            // subset of dtypes is still useful.  Mark `ok` true when at
            // least one entry is callable; per-call dispatch verifies its
            // specific entry is present.
            if (bf16.has_value())
                L.bf16.emplace(*bf16);
            if (fp8_pt.has_value())
                L.fp8_per_tensor.emplace(*fp8_pt);
            if (fp8_blk.has_value())
                L.fp8_block_scale.emplace(*fp8_blk);
            if (fp4_blk.has_value())
                L.fp4_block_scale.emplace(*fp4_blk);
            if (valid.has_value())
                L.valid_configs.emplace(*valid);

            if (!L.bf16.has_value() && !L.fp8_per_tensor.has_value() && !L.fp8_block_scale.has_value()
                && !L.fp4_block_scale.has_value()) {
                std::cerr << "[flashinfer-moe] no MoE exports found in " << so_path << std::endl;
                return;
            }

            // Cubin callback registration — must be done after the .so is
            // mapped into the global namespace by tvm-ffi.
            void* lib = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
            if (!lib) {
                lib = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
            }
            if (!lib) {
                std::cerr << "[flashinfer-moe] dlopen failed: " << dlerror() << std::endl;
                return;
            }
            auto set_cb  = reinterpret_cast<SetCubinCallbackFn>(dlsym(lib, "FlashInferSetCubinCallback"));
            auto set_cur = reinterpret_cast<SetCurrentCubinFn>(dlsym(lib, "FlashInferSetCurrentCubin"));
            if (!set_cb || !set_cur) {
                std::cerr << "[flashinfer-moe] missing C cubin entries in " << so_path << "\n";
                return;
            }
            g_cubin_ctx.set_current = set_cur;
            set_cb(cubin_callback);

            if (!ensure_dlpack_allocator()) {
                return;
            }
            L.ok = true;
        }
        catch (const std::exception& e) {
            std::cerr << "[flashinfer-moe] load failed: " << e.what() << std::endl;
        }
    });
    return L;
}

CutlassLoaded& cutlass_loaded()
{
    static CutlassLoaded  L;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
        try {
            const char*       override_path = std::getenv("LMDEPLOY_FLASHINFER_CUTLASS_MOE_SO");
            const std::string so_path =
                override_path && *override_path ? override_path : LMDEPLOY_FLASHINFER_CUTLASS_MOE_SO_PATH;

            promote_tvm_ffi_symbols();
            if (!ensure_dlpack_allocator()) {
                return;
            }

            L.mod.emplace(ffi::Module::LoadFromFile(so_path));
            auto init = (*L.mod)->GetFunction("init");
            if (!init.has_value()) {
                std::cerr << "[flashinfer-moe] CUTLASS init export not found in " << so_path << "\n";
                return;
            }

            const bool use_mxfp8_act_scaling = [] {
                const char* env = std::getenv("LMDEPLOY_W4A8_MXFP8_ACT");
                return env && *env && std::string_view(env) != "0";
            }();

            auto runner = (*init)(DLDataType{kDLFloat8_e4m3fn, 8, 1},
                                  DLDataType{kDLInt, 64, 1},
                                  DLDataType{kDLBfloat, 16, 1},
                                  false,  // use_deepseek_fp8_block_scale
                                  false,  // use_w4_group_scaling
                                  use_mxfp8_act_scaling,
                                  false)  // use_packed_weights
                              .cast<ffi::Module>();
            L.runner.emplace(std::move(runner));
            auto run_moe = (*L.runner)->GetFunction("run_moe");
            if (!run_moe.has_value()) {
                std::cerr << "[flashinfer-moe] CUTLASS run_moe export not found in " << so_path << "\n";
                return;
            }
            L.run_moe.emplace(*run_moe);
            if (auto run_profile = (*L.runner)->GetFunction("run_gemm_profile"); run_profile.has_value()) {
                L.run_gemm_profile.emplace(*run_profile);
            }
            if (auto count = (*L.runner)->GetFunction("get_gemm1_tactic_count"); count.has_value()) {
                L.get_gemm1_tactic_count.emplace(*count);
            }
            if (auto count = (*L.runner)->GetFunction("get_gemm2_tactic_count"); count.has_value()) {
                L.get_gemm2_tactic_count.emplace(*count);
            }
            L.ok = true;
        }
        catch (const std::exception& e) {
            std::cerr << "[flashinfer-moe] CUTLASS load failed: " << e.what() << std::endl;
        }
    });
    return L;
}

DLDataType to_dl(DType d)
{
    switch (d) {
        case DType::kFP16:
            return DLDataType{kDLFloat, 16, 1};
        case DType::kBF16:
            return DLDataType{kDLBfloat, 16, 1};
        case DType::kFP32:
            return DLDataType{kDLFloat, 32, 1};
        case DType::kFP8E4M3:
            return DLDataType{kDLFloat8_e4m3fn, 8, 1};
        case DType::kE2M1U8:
            return DLDataType{kDLUInt, 8, 1};  // packed nvfp4
    }
    return DLDataType{kDLBfloat, 16, 1};
}

DLDataType routing_logits_dl(DType dtype, RoutingMethod method)
{
    // DeepSeek/noaux_tc always routes from fp32 logits in TurboMind.  For
    // other methods, honor the caller's actual Gate() output dtype instead
    // of assuming bf16; Qwen3-MoE currently keeps gate logits in fp32.
    if (method == RoutingMethod::kDeepSeekV3 || dtype == DType::kFP32) {
        return DLDataType{kDLFloat, 32, 1};
    }
    return to_dl(dtype);
}

inline int current_device_id()
{
    int         dev_id = 0;
    cudaError_t err    = cudaGetDevice(&dev_id);
    if (err != cudaSuccess) {
        std::cerr << "[flashinfer-moe] cudaGetDevice failed: " << cudaGetErrorString(err)
                  << " — falling back to device 0\n";
        return 0;
    }
    return dev_id;
}

// Same RAII stream-scope as the FMHA wrapper — see that file for the
// full reasoning on why we have to set tvm-ffi's "current stream" so the
// FlashInfer launcher's `get_stream(tensor.device())` returns ours.
//
// In addition to setting tvm-ffi's stream, this scope:
//   1. Publishes the dispatch stream into a thread-local so the DLPack
//      allocator (`dlpack_cuda_alloc`) knows what stream to use for
//      its async-malloc/async-free fallback path and so the per-thread
//      bump arena can attribute its allocations to the right stream.
//   2. Resets the per-thread arena's bump pointer so the new dispatch
//      starts fresh — the previous dispatch's allocations are
//      guaranteed completed because the arena hands out memory only
//      to FFI work that gets submitted on this same stream.
class TvmFfiStreamScope {
public:
    explicit TvmFfiStreamScope(cudaStream_t stream): device_id_{current_device_id()}
    {
        // The trtllm-gen BMM path uses CUDA Driver API module loading
        // (cuModuleLoadData/cuModuleGetFunction).  Turbomind runs model
        // execution on its own worker thread, so make the runtime primary
        // context current on this thread before FlashInfer asks the driver
        // to load cubins.
        if (cudaError_t err = cudaSetDevice(device_id_); err != cudaSuccess) {
            std::cerr << "[flashinfer-moe] cudaSetDevice(" << device_id_ << ") failed: " << cudaGetErrorString(err)
                      << "\n";
        }
        if (!stream) {
            std::cerr << "[flashinfer-moe] WARNING: TvmFfiStreamScope got nullptr stream "
                         "— MoE kernel would run on stream 0; check upstream wiring\n";
        }
        int rc = TVMFFIEnvSetStream(kDLCUDA, device_id_, stream, &prev_);
        if (rc != 0) {
            std::cerr << "[flashinfer-moe] TVMFFIEnvSetStream(set) rc=" << rc << " device=" << device_id_ << "\n";
        }
        // Publish stream + reset arena for this dispatch.
        prev_dispatch_stream_   = g_tls_dispatch_stream;
        g_tls_dispatch_stream   = stream;
        g_tls_arena_used        = 0;
    }
    ~TvmFfiStreamScope()
    {
        TVMFFIEnvSetStream(kDLCUDA, device_id_, prev_, nullptr);
        g_tls_dispatch_stream = prev_dispatch_stream_;
    }
    TvmFfiStreamScope(const TvmFfiStreamScope&)            = delete;
    TvmFfiStreamScope& operator=(const TvmFfiStreamScope&) = delete;

private:
    int                device_id_{0};
    TVMFFIStreamHandle prev_{nullptr};
    cudaStream_t       prev_dispatch_stream_{nullptr};
};

// Heap-allocated DLPack bundle backing an ffi::Tensor view of a raw GPU
// pointer.  Identical pattern to the FMHA wrapper — no copy, no host
// roundtrip; the bundle is freed by the DLPack deleter when the last
// ffi::Tensor reference drops.
struct DLBundle {
    DLManagedTensor      managed;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;

    static void Deleter(DLManagedTensor* mt)
    {
        delete reinterpret_cast<DLBundle*>(mt->manager_ctx);
    }
};

ffi::Tensor
make_view(void* data, std::vector<int64_t> shape, std::vector<int64_t> strides, DLDataType dtype, int dev_id = -1)
{
    if (dev_id < 0) {
        dev_id = current_device_id();
    }
    auto bundle                           = new DLBundle;
    bundle->shape                         = std::move(shape);
    bundle->strides                       = std::move(strides);
    bundle->managed.dl_tensor.data        = data;
    bundle->managed.dl_tensor.device      = DLDevice{kDLCUDA, dev_id};
    bundle->managed.dl_tensor.ndim        = static_cast<int32_t>(bundle->shape.size());
    bundle->managed.dl_tensor.dtype       = dtype;
    bundle->managed.dl_tensor.shape       = bundle->shape.data();
    bundle->managed.dl_tensor.strides     = bundle->strides.empty() ? nullptr : bundle->strides.data();
    bundle->managed.dl_tensor.byte_offset = 0;
    bundle->managed.manager_ctx           = bundle;
    bundle->managed.deleter               = &DLBundle::Deleter;
    return ffi::Tensor::FromDLPack(&bundle->managed);
}

inline ffi::Optional<ffi::Tensor> opt_view(void* data, std::vector<int64_t> shape, DLDataType dtype)
{
    if (!data)
        return ffi::Optional<ffi::Tensor>();
    return make_view(data, std::move(shape), {}, dtype);
}

}  // namespace

bool is_available()
{
    return loaded().ok;
}

bool is_sm10x_capable()
{
    // Cached once per process — the SM family doesn't change at runtime.
    // We do the probe inline (no cuda_utils.h dep) so the wrapper stays
    // self-contained.
    static const bool kCapable = []() {
        int device   = 0;
        int sm_major = 0;
        int sm_minor = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            return false;
        }
        cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
        const int sm = sm_major * 10 + sm_minor;
        return sm == 100 || sm == 103;
    }();
    return kCapable;
}

void SetTuningActive(bool active)
{
    g_tuning_active.store(active, std::memory_order_relaxed);
}

void set_dispatch_arena_capacity(size_t bytes)
{
    // Idempotent setter: only the first non-zero value sticks.  We
    // intentionally do NOT realloc an existing arena.  Caller is
    // expected to pick a conservative upper bound based on
    // max_token_num × hidden × top_k × dtype + headroom for routing /
    // BMM workspaces.
    //
    // EAGERLY allocate the global arena on the calling (engine init)
    // thread — this matters for cuda-graph capture, which can't run
    // cudaMalloc during cudaStreamBeginCapture.  By the time capture
    // starts, `g_arena().base` is already a stable, capture-safe
    // pointer that all dispatch threads share.
    auto&  cap      = g_arena_capacity_bytes();
    size_t expected = 0;
    if (cap.compare_exchange_strong(expected, bytes, std::memory_order_relaxed)) {
        g_arena().ensure(bytes);
    }
}

namespace {

constexpr int64_t trtllm_gen_dtype(int block_format_bit, int signed_bit, int integer_bit, int num_bits, int uid)
{
    return (static_cast<int64_t>(block_format_bit) << 24) | (static_cast<int64_t>(signed_bit) << 20)
           | (static_cast<int64_t>(integer_bit) << 16) | (static_cast<int64_t>(num_bits) << 8)
           | static_cast<int64_t>(uid);
}

constexpr int64_t kTrtllmBfloat16    = trtllm_gen_dtype(0, 1, 0, 16, 0);
constexpr int64_t kTrtllmE4m3        = trtllm_gen_dtype(0, 1, 0, 8, 5);
constexpr int64_t kTrtllmFp16        = trtllm_gen_dtype(0, 1, 0, 16, 7);
constexpr int64_t kTrtllmFp32        = trtllm_gen_dtype(0, 1, 0, 32, 8);
constexpr int64_t kTrtllmE2m1        = trtllm_gen_dtype(1, 1, 0, 4, 2);
constexpr int64_t kTrtllmMxE2m1      = trtllm_gen_dtype(1, 1, 0, 4, 12);
constexpr int64_t kActivationSwiglu  = 3;
constexpr int64_t kFp8QuantizationNone     = 0;
constexpr int64_t kFp8QuantizationDeepSeek = 1;

constexpr int64_t trtllm_dtype(DType dtype, int64_t packed_fp4_dtype = kTrtllmE2m1)
{
    switch (dtype) {
        case DType::kFP16:
            return kTrtllmFp16;
        case DType::kBF16:
            return kTrtllmBfloat16;
        case DType::kFP32:
            return kTrtllmFp32;
        case DType::kFP8E4M3:
            return kTrtllmE4m3;
        case DType::kE2M1U8:
            return packed_fp4_dtype;
    }
    return kTrtllmBfloat16;
}

std::vector<AutotuneTactic> get_valid_moe_tactics(const AutotuneKey& k,
                                                  int64_t            fp8_quantization_type,
                                                  bool               use_shuffled_weight,
                                                  int64_t            weight_layout,
                                                  int64_t            dtype_act,
                                                  int64_t            dtype_weights);

AutotuneTactic select_tactic(const AutotuneKey&                         k,
                             std::vector<AutotuneTactic>&&              candidates,
                             cudaStream_t                               stream,
                             const std::function<bool(AutotuneTactic)>& run_candidate);

int64_t select_profile_id(const AutotuneKey&                  k,
                          std::vector<int64_t>&&              candidates,
                          cudaStream_t                        stream,
                          const std::function<bool(int64_t)>& run_candidate,
                          const char*                         tag);

bool tuning_enabled();

std::optional<AutotuneTactic> lookup_cached_tactic(const AutotuneKey& k);

// Bucketize num_tokens for the autotune cache key.  Mirrors
// TensorRT-LLM upstream (`tensorrt_llm/_torch/utils.py::last_positive_power_of_2`,
// used inside `FP4BlockScaleMoERunner.get_dynamic_tensor_specs`):
// every distinct num_tokens would otherwise hash to a different cache slot,
// so steady-state inference (where batch size flaps between 1/2/4/8/16 etc.)
// keeps missing the cache and re-invokes the FFI tactic enumeration each
// MoE call.  Bucketing collapses the cache axis to ~13 entries (powers of 2
// up to 8192) without changing kernel behavior — the launcher picks tile_n
// from the actual num_tokens at submission time, the cached tactic
// (tile_n, config_index) is shape-stable across the bucket.
inline int bucket_num_tokens_for_autotune(int n)
{
    constexpr int kMaxBucket = 8192;
    if (n <= 1) {
        return 1;
    }
    if (n >= kMaxBucket) {
        return kMaxBucket;
    }
    int v = 1;
    while (v < n) {
        v <<= 1;
    }
    return v;
}

class ProfileOutputScratch {
public:
    ProfileOutputScratch(int num_tokens, int hidden_size, cudaStream_t stream): stream_(stream)
    {
        if (num_tokens <= 0 || hidden_size <= 0) {
            return;
        }
        const size_t elems = static_cast<size_t>(num_tokens) * static_cast<size_t>(hidden_size);
        if (elems > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
            std::cerr << "[flashinfer-moe] skip tuning scratch allocation: output size overflow"
                      << " tokens=" << num_tokens << " hidden=" << hidden_size << "\n";
            return;
        }
        const cudaError_t err = cudaMallocAsync(&data_, elems * sizeof(uint16_t), stream_);
        if (err != cudaSuccess) {
            std::cerr << "[flashinfer-moe] skip tuning scratch allocation: " << cudaGetErrorString(err)
                      << " tokens=" << num_tokens << " hidden=" << hidden_size << "\n";
            data_ = nullptr;
            cudaGetLastError();
        }
    }

    ~ProfileOutputScratch()
    {
        if (data_) {
            const cudaError_t err = cudaFreeAsync(data_, stream_);
            if (err != cudaSuccess) {
                cudaGetLastError();
            }
        }
    }

    ProfileOutputScratch(const ProfileOutputScratch&)            = delete;
    ProfileOutputScratch& operator=(const ProfileOutputScratch&) = delete;

    void* data() const
    {
        return data_;
    }

private:
    void*        data_{nullptr};
    cudaStream_t stream_{};
};

}  // namespace

bool dispatch_bf16(const Bf16MoeParams& p)
{
    auto& L = loaded();
    if (!L.ok || !L.bf16.has_value())
        return false;
    TvmFfiStreamScope ffi_stream{p.stream};

    // The flashinfer-python 0.6.9+ (post pre-tag refactor) `trtllm_bf16_moe`
    // FFI takes 25 positional args and writes into a caller-supplied
    // `output` TensorView (no more allocate-and-return-Tensor).  Signature
    // (see `csrc/trtllm_fused_moe_kernel_launcher.cu:1768`):
    //   ( Optional routing_logits, Optional routing_bias,
    //     TensorView expert_indices, TensorView expert_weights,
    //     TensorView hidden_states, TensorView gemm1_weights,
    //     TensorView gemm2_weights, TensorView output,
    //     int num_experts, int top_k, Optional n_group, Optional topk_group,
    //     int intermediate_size, int local_expert_offset,
    //     int local_num_experts, Optional routed_scaling_factor,
    //     int routing_method_type, bool use_shuffled_weight,
    //     int weight_layout, bool do_finalize, bool enable_pdl,
    //     Array<int64> moe_tactic, int activation_type, bool norm_topk_prob,
    //     Optional routing_replay_out )
    //
    // We always go through the routing-logits path (caller's `routing_logits`
    // is required for now), so `expert_indices` / `expert_weights` are
    // EMPTY 0-element tensor views (the launcher branches on
    // `routing_logits.has_value()` and ignores them).
    auto routing_logits_dt = routing_logits_dl(p.routing_logits_dtype, p.routing_method);
    auto routing_logits    = make_view(p.routing_logits, {p.num_tokens, p.num_experts}, {}, routing_logits_dt);

    auto routing_bias_opt = opt_view(p.routing_bias, {p.num_experts}, DLDataType{kDLBfloat, 16, 1});

    // Empty placeholder views for the unused expert_indices / expert_weights
    // (kernel ignores them when routing_logits has a value).  Backing data
    // pointer must be non-null (DLPack rejects null) — use any valid GPU
    // address; we reuse `p.routing_logits` since we know it's valid.
    auto expert_indices = make_view(p.routing_logits, {0, p.top_k}, {}, DLDataType{kDLInt, 32, 1});
    auto expert_weights = make_view(p.routing_logits, {0, p.top_k}, {}, DLDataType{kDLFloat, 32, 1});

    auto hidden_states = make_view(p.hidden_states, {p.num_tokens, p.hidden_size}, {}, DLDataType{kDLBfloat, 16, 1});

    // BlockMajorK weights: 4D shape [num_experts, K/block_k, M, block_k].
    // For BF16, block_k = 128 bytes / 2 = 64 elements.  When the caller
    // sets `weight_layout = MajorK (0)`, fall back to 3D [E, M, K] view.
    constexpr int kBf16BlockK = 64;
    ffi::Tensor   gemm1_weights;
    ffi::Tensor   gemm2_weights;
    if (p.weight_layout == 2 /* BlockMajorK */) {
        if (p.hidden_size % kBf16BlockK != 0 || p.intermediate_size % kBf16BlockK != 0) {
            std::cerr << "[flashinfer-moe] bf16 BlockMajorK: hidden/intermediate must be % "
                      << kBf16BlockK << " == 0 (hidden=" << p.hidden_size
                      << " inter=" << p.intermediate_size << ")\n";
            return false;
        }
        gemm1_weights = make_view(p.gemm1_weights,
                                  {p.num_experts,
                                   p.hidden_size / kBf16BlockK,
                                   2 * p.intermediate_size,
                                   kBf16BlockK},
                                  {},
                                  DLDataType{kDLBfloat, 16, 1});
        gemm2_weights = make_view(p.gemm2_weights,
                                  {p.num_experts,
                                   p.intermediate_size / kBf16BlockK,
                                   p.hidden_size,
                                   kBf16BlockK},
                                  {},
                                  DLDataType{kDLBfloat, 16, 1});
    }
    else {
        gemm1_weights = make_view(p.gemm1_weights,
                                  {p.num_experts, 2 * p.intermediate_size, p.hidden_size},
                                  {},
                                  DLDataType{kDLBfloat, 16, 1});
        gemm2_weights = make_view(p.gemm2_weights,
                                  {p.num_experts, p.hidden_size, p.intermediate_size},
                                  {},
                                  DLDataType{kDLBfloat, 16, 1});
    }

    auto output = make_view(p.output, {p.num_tokens, p.hidden_size}, {}, DLDataType{kDLBfloat, 16, 1});

    ffi::Array<int64_t> moe_tactic;
    moe_tactic.push_back(p.tile_n);
    moe_tactic.push_back(p.config_index);

    // Activation type: 0 = SwiGLU (Qwen3-MoE default), 1 = GeGLU.  We don't
    // expose this through `Bf16MoeParams` yet — Qwen3 / DeepSeek are both
    // SwiGLU so 0 is correct for current callers.
    // Upstream MoeActivationType (csrc/nv_internal/.../moeUtils.h):
    //   Gelu=0, Relu=1, Silu=2, Swiglu=3, Geglu=4, Identity=5.
    // Qwen3-MoE uses SwiGLU.
    constexpr int64_t kActivationSwiglu = 3;

    try {
        // Returns `Array<Tensor>` (we ignore — output already in our buffer).
        (*L.bf16)(ffi::Optional<ffi::Tensor>(routing_logits),
                  routing_bias_opt,
                  expert_indices,
                  expert_weights,
                  hidden_states,
                  gemm1_weights,
                  gemm2_weights,
                  output,
                  static_cast<int64_t>(p.num_experts),
                  static_cast<int64_t>(p.top_k),
                  p.n_group > 0 ? ffi::Optional<int64_t>(static_cast<int64_t>(p.n_group)) : ffi::Optional<int64_t>(),
                  p.topk_group > 0 ? ffi::Optional<int64_t>(static_cast<int64_t>(p.topk_group)) :
                                     ffi::Optional<int64_t>(),
                  static_cast<int64_t>(p.intermediate_size),
                  static_cast<int64_t>(p.local_expert_offset),
                  static_cast<int64_t>(p.local_num_experts),
                  ffi::Optional<double>(),  // routed_scaling_factor (unused for default routing)
                  static_cast<int64_t>(p.routing_method),
                  p.use_shuffled_weight,
                  static_cast<int64_t>(p.weight_layout),
                  true,  // do_finalize — write final reduction into `output`
                  p.enable_pdl,
                  moe_tactic,
                  kActivationSwiglu,              // activation_type (SwiGLU)
                  true,                           // norm_topk_prob (Qwen3-MoE default)
                  ffi::Optional<ffi::Tensor>());  // routing_replay_out (unused)
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[flashinfer-moe] bf16 call failed: " << e.what() << std::endl;
        return false;
    }
}

bool dispatch_fp8_per_tensor(const Fp8PerTensorMoeParams& p)
{
    auto& L = loaded();
    if (!L.ok || !L.fp8_per_tensor.has_value())
        return false;
    TvmFfiStreamScope ffi_stream{p.stream};

    // routing_logits: [num_tokens, num_experts] bf16/fp32
    auto routing_logits_dt = routing_logits_dl(p.routing_logits_dtype, p.routing_method);
    auto routing_logits    = make_view(p.routing_logits, {p.num_tokens, p.num_experts}, {}, routing_logits_dt);

    auto routing_bias = opt_view(p.routing_bias, {p.num_experts}, DLDataType{kDLBfloat, 16, 1});

    auto hidden_states = make_view(p.hidden_states, {p.num_tokens, p.hidden_size}, {}, to_dl(p.hidden_states_dtype));

    auto gemm1_weights = make_view(p.gemm1_weights,
                                   {p.num_experts, 2 * p.intermediate_size, p.hidden_size},
                                   {},
                                   DLDataType{kDLFloat8_e4m3fn, 8, 1});
    auto output1_scales_scalar =
        make_view(p.output1_scales_scalar, {p.local_num_experts}, {}, DLDataType{kDLFloat, 32, 1});
    auto output1_scales_gate_scalar =
        make_view(p.output1_scales_gate_scalar, {p.local_num_experts}, {}, DLDataType{kDLFloat, 32, 1});

    auto gemm2_weights = make_view(
        p.gemm2_weights, {p.num_experts, p.hidden_size, p.intermediate_size}, {}, DLDataType{kDLFloat8_e4m3fn, 8, 1});
    auto output2_scales_scalar =
        make_view(p.output2_scales_scalar, {p.local_num_experts}, {}, DLDataType{kDLFloat, 32, 1});

    auto output = make_view(p.output, {p.num_tokens, p.hidden_size}, {}, DLDataType{kDLBfloat, 16, 1});

    // config_index: [tile_n, config].  The kernel launcher selects defaults
    // when either is -1.  See `trtllm_fused_moe_kernel_launcher.cu:1549`.
    ffi::Array<int64_t> config_index;
    config_index.push_back(p.tile_n);
    config_index.push_back(p.config_index);

    try {
        // flashinfer-python 0.6.9+ signature (25 positional args; returns
        // Array<Tensor> — `output` is now passed in, no more allocate-and-copy).
        // See `csrc/trtllm_fused_moe_kernel_launcher.cu:1853`.  Added vs the
        // older 21-arg shape: do_finalize, activation_type, norm_topk_prob,
        // routing_replay_out (Optional).
        // Upstream MoeActivationType (csrc/nv_internal/.../moeUtils.h):
        //   Gelu=0, Relu=1, Silu=2, Swiglu=3, Geglu=4, Identity=5.
        // Qwen3-MoE uses SwiGLU.
        constexpr int64_t kActivationSwiglu = 3;
        (*L.fp8_per_tensor)(
            routing_logits,
            routing_bias,
            hidden_states,
            gemm1_weights,
            output1_scales_scalar,
            output1_scales_gate_scalar,
            gemm2_weights,
            output2_scales_scalar,
            output,
            static_cast<int64_t>(p.num_experts),
            static_cast<int64_t>(p.top_k),
            p.n_group > 0 ? ffi::Optional<int64_t>(static_cast<int64_t>(p.n_group)) : ffi::Optional<int64_t>(),
            p.topk_group > 0 ? ffi::Optional<int64_t>(static_cast<int64_t>(p.topk_group)) : ffi::Optional<int64_t>(),
            static_cast<int64_t>(p.intermediate_size),
            static_cast<int64_t>(p.local_expert_offset),
            static_cast<int64_t>(p.local_num_experts),
            ffi::Optional<double>(p.routed_scaling_factor),
            p.use_routing_scales_on_input,
            static_cast<int64_t>(p.routing_method),
            true,  // do_finalize — write final reduction
            p.enable_pdl,
            config_index,
            kActivationSwiglu,              // activation_type (SwiGLU)
            true,                           // norm_topk_prob (Qwen3-MoE default)
            ffi::Optional<ffi::Tensor>());  // routing_replay_out (unused)
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[flashinfer-moe] fp8_per_tensor call failed: " << e.what() << std::endl;
        return false;
    }
}

bool dispatch_fp8_block_scale(const Fp8BlockScaleMoeParams& p)
{
    auto& L = loaded();
    if (!L.ok || !L.fp8_block_scale.has_value())
        return false;
    // Upstream signature treats `hidden_states_scale` as a required
    // TensorView (not Optional) — see csrc/trtllm_fused_moe_kernel_launcher.cu:1567.
    // Forwarding a nullptr would crash inside `make_view` / dl_tensor.data
    // dereference downstream; surface a clean false so the caller falls
    // back to its native MoE path.
    if (!p.hidden_states_scale) {
        std::cerr << "[flashinfer-moe] fp8_block_scale: hidden_states_scale is required; "
                     "caller must pre-quantize bf16/fp16 input.\n";
        return false;
    }
    TvmFfiStreamScope ffi_stream{p.stream};

    auto routing_logits_dt = routing_logits_dl(p.routing_logits_dtype, p.routing_method);
    auto routing_logits    = make_view(p.routing_logits, {p.num_tokens, p.num_experts}, {}, routing_logits_dt);

    auto routing_bias = opt_view(p.routing_bias, {p.num_experts}, DLDataType{kDLBfloat, 16, 1});

    // The 0.6.9+ SM100 launcher takes expert_indices/expert_weights as
    // non-optional placeholder TensorViews even when routing_logits is used.
    // It ignores them when routing_logits.has_value() is true, but they must
    // still be valid DLPack views.
    auto expert_indices = make_view(p.routing_logits, {0, p.top_k}, {}, DLDataType{kDLInt, 32, 1});
    auto expert_weights = make_view(p.routing_logits, {0, p.top_k}, {}, DLDataType{kDLFloat, 32, 1});

    auto hidden_states = make_view(p.hidden_states, {p.num_tokens, p.hidden_size}, {}, to_dl(p.hidden_states_dtype));
    // hidden_states_scale upstream is `[hidden_size/128, num_tokens]` (yes,
    // transposed) — see flashinfer/fused_moe/core.py:trtllm_fp8_block_scale_moe.
    auto hidden_states_scale =
        make_view(p.hidden_states_scale, {p.hidden_size / 128, p.num_tokens}, {}, DLDataType{kDLFloat, 32, 1});

    auto gemm1_weights       = make_view(p.gemm1_weights,
                                         {p.num_experts, 2 * p.intermediate_size, p.hidden_size},
                                         {},
                                   DLDataType{kDLFloat8_e4m3fn, 8, 1});
    auto gemm1_weights_scale = make_view(p.gemm1_weights_scale,
                                         {p.num_experts, (2 * p.intermediate_size) / 128, p.hidden_size / 128},
                                         {},
                                         DLDataType{kDLFloat, 32, 1});

    auto gemm2_weights = make_view(
        p.gemm2_weights, {p.num_experts, p.hidden_size, p.intermediate_size}, {}, DLDataType{kDLFloat8_e4m3fn, 8, 1});
    auto gemm2_weights_scale = make_view(p.gemm2_weights_scale,
                                         {p.num_experts, p.hidden_size / 128, p.intermediate_size / 128},
                                         {},
                                         DLDataType{kDLFloat, 32, 1});

    auto output = make_view(p.output, {p.num_tokens, p.hidden_size}, {}, DLDataType{kDLBfloat, 16, 1});

    ffi::Array<int64_t> config_index;
    config_index.push_back(p.tile_n);
    config_index.push_back(p.config_index);

    try {
        // FlashInfer 0.6.9+ SM100 signature:
        //   routing_logits, expert_indices, expert_weights, routing_bias,
        //   hidden/scales/weights, output, routing/meta, do_finalize,
        //   enable_pdl, config_index, fp8_quantization_type,
        //   activation_type, norm_topk_prob, routing_replay_out.
        // Fp8QuantizationType: None=0, DeepSeekFp8=1, MxFp8=2,
        // PerTensorFp8=3.  HF block-scale Qwen uses DeepSeekFp8 layout
        // (float32 block scales).
        constexpr int64_t kFp8QuantizationDeepSeek = 1;
        constexpr int64_t kActivationSwiglu        = 3;
        (*L.fp8_block_scale)(
            ffi::Optional<ffi::Tensor>(routing_logits),
            expert_indices,
            expert_weights,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            output,
            static_cast<int64_t>(p.num_experts),
            static_cast<int64_t>(p.top_k),
            p.n_group > 0 ? ffi::Optional<int64_t>(static_cast<int64_t>(p.n_group)) : ffi::Optional<int64_t>(),
            p.topk_group > 0 ? ffi::Optional<int64_t>(static_cast<int64_t>(p.topk_group)) : ffi::Optional<int64_t>(),
            static_cast<int64_t>(p.intermediate_size),
            static_cast<int64_t>(p.local_expert_offset),
            static_cast<int64_t>(p.local_num_experts),
            ffi::Optional<double>(p.routed_scaling_factor),
            static_cast<int64_t>(p.routing_method),
            p.use_shuffled_weight,
            static_cast<int64_t>(p.weight_layout),
            true,  // do_finalize — write final reduction into `output`
            p.enable_pdl,
            config_index,
            kFp8QuantizationDeepSeek,
            kActivationSwiglu,
            p.norm_topk_prob,
            ffi::Optional<ffi::Tensor>());  // routing_replay_out (unused)
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[flashinfer-moe] fp8_block_scale call failed: " << e.what() << std::endl;
        return false;
    }
}

namespace {

// Shared body for both NVFP4 dispatch entries.  `routing_logits_opt` and
// the (`topk_ids`, `expert_weights`) pair are mutually exclusive — the
// upstream launcher branches on `routing_logits.has_value()`, but we
// guarantee at most one is supplied via the strongly-typed param structs
// in the header, so this private helper just plumbs whichever is set.
bool dispatch_fp4_impl(const Fp4MoeBase&          p,
                       ffi::Optional<ffi::Tensor> routing_logits_opt,
                       ffi::Optional<ffi::Tensor> topk_ids_opt,
                       ffi::Optional<ffi::Tensor> expert_weights_opt)
{
    auto& L = loaded();
    if (!L.ok || !L.fp4_block_scale.has_value())
        return false;
    TvmFfiStreamScope ffi_stream{p.stream};

    auto routing_bias = opt_view(p.routing_bias, {p.num_experts}, DLDataType{kDLBfloat, 16, 1});

    // Hidden states layout depends on input dtype:
    //   nvfp4 (uint8 packed) → [num_tokens, hidden_size/2]
    //   bf16 / mxfp8         → [num_tokens, hidden_size]
    bool hidden_is_packed = (p.hidden_states_dtype == DType::kE2M1U8);
    auto hidden_states    = make_view(p.hidden_states,
                                      {p.num_tokens, hidden_is_packed ? p.hidden_size / 2 : p.hidden_size},
                                      {},
                                   to_dl(p.hidden_states_dtype));

    ffi::Optional<ffi::Tensor> hidden_states_scale_opt;
    if (p.hidden_states_scale) {
        // shape: [num_tokens, hidden_size / (32 if mxfp8 else 16 if mxfp4)]
        // We don't know the vec_size at this layer with full precision (the
        // launcher infers it from `numel()`), but for the scale tensor view
        // any 2D shape with the right total is accepted because the launcher
        // re-slices internally.  Use the conservative `weight_scale_vec_size`
        // (= 16 for NVFP4, 32 for MXFP4) for activation scale too.
        const int act_sf_vec    = p.weight_scale_vec_size;
        hidden_states_scale_opt = make_view(
            p.hidden_states_scale, {p.num_tokens, p.hidden_size / act_sf_vec}, {}, DLDataType{kDLFloat8_e4m3fn, 8, 1});
    }

    // gemm1_weights: packed nvfp4, [num_experts, 2*intermediate_size, hidden_size/2]
    auto gemm1_weights = make_view(
        p.gemm1_weights, {p.num_experts, 2 * p.intermediate_size, p.hidden_size / 2}, {}, DLDataType{kDLUInt, 8, 1});
    auto gemm1_weights_scale =
        make_view(p.gemm1_weights_scale,
                  {p.num_experts, 2 * p.intermediate_size, p.hidden_size / p.weight_scale_vec_size},
                  {},
                  DLDataType{kDLFloat8_e4m3fn, 8, 1});

    auto gemm1_bias_opt = opt_view(p.gemm1_bias, {p.num_experts, 2 * p.intermediate_size}, DLDataType{kDLFloat, 32, 1});
    auto gemm1_alpha_opt = opt_view(p.gemm1_alpha, {p.num_experts}, DLDataType{kDLFloat, 32, 1});
    auto gemm1_beta_opt  = opt_view(p.gemm1_beta, {p.num_experts}, DLDataType{kDLFloat, 32, 1});
    auto gemm1_clamp_opt = opt_view(p.gemm1_clamp_limit, {p.num_experts}, DLDataType{kDLFloat, 32, 1});

    auto gemm2_weights = make_view(
        p.gemm2_weights, {p.num_experts, p.hidden_size, p.intermediate_size / 2}, {}, DLDataType{kDLUInt, 8, 1});
    auto gemm2_weights_scale = make_view(p.gemm2_weights_scale,
                                         {p.num_experts, p.hidden_size, p.intermediate_size / p.weight_scale_vec_size},
                                         {},
                                         DLDataType{kDLFloat8_e4m3fn, 8, 1});

    auto gemm2_bias_opt = opt_view(p.gemm2_bias, {p.num_experts, p.hidden_size}, DLDataType{kDLFloat, 32, 1});

    auto out1_sc_opt      = opt_view(p.output1_scales_scalar, {p.local_num_experts}, DLDataType{kDLFloat, 32, 1});
    auto out1_sc_gate_opt = opt_view(p.output1_scales_gate_scalar, {p.local_num_experts}, DLDataType{kDLFloat, 32, 1});
    auto out2_sc_opt      = opt_view(p.output2_scales_scalar, {p.local_num_experts}, DLDataType{kDLFloat, 32, 1});

    auto make_output_view = [&](void* data) {
        return make_view(data, {p.num_tokens, p.hidden_size}, {}, DLDataType{kDLBfloat, 16, 1});
    };

    // Upstream signature: csrc/trtllm_fused_moe_kernel_launcher.cu
    // (0.6.10+ includes per_token_scales; returns Array<Tensor>; we discard the extras).
    // The launcher accepts `topk_ids` / `expert_weights` as Optional in
    // newer flashinfer (where one branch is exercised at a time).  In
    // 0.6.9+ they are positional `TensorView`s but the launcher only
    // touches them when `routing_logits.has_value() == false`.  We pass
    // empty placeholders when the branch isn't taken.
    ffi::Tensor topk_ids_tv =
        topk_ids_opt.has_value() ?
            topk_ids_opt.value() :
            make_view(p.routing_bias ? p.routing_bias : p.gemm1_weights, {0, p.top_k}, {}, DLDataType{kDLInt, 32, 1});
    ffi::Tensor expert_weights_tv =
        expert_weights_opt.has_value() ?
            expert_weights_opt.value() :
            make_view(
                p.routing_bias ? p.routing_bias : p.gemm1_weights, {0, p.top_k}, {}, DLDataType{kDLFloat, 32, 1});

    void* profile_output_ptr = nullptr;
    auto  run_with_tactic    = [&](AutotuneTactic tactic) {
        auto output = make_output_view(profile_output_ptr ? profile_output_ptr : p.output);
        ffi::Array<int64_t> config_index;
        config_index.push_back(tactic.tile_n);
        config_index.push_back(tactic.config_index);
        try {
            (*L.fp4_block_scale)(
                routing_logits_opt,
                topk_ids_tv,
                expert_weights_tv,
                routing_bias,
                hidden_states,
                hidden_states_scale_opt,
                gemm1_weights,
                gemm1_weights_scale,
                gemm1_bias_opt,
                gemm1_alpha_opt,
                gemm1_beta_opt,
                gemm1_clamp_opt,
                gemm2_weights,
                gemm2_weights_scale,
                gemm2_bias_opt,
                out1_sc_opt,
                out1_sc_gate_opt,
                out2_sc_opt,
                // NOTE: flashinfer 0.6.10.post1 still uses the 34-arg
                // signature (no per_token_scales).  Verified against
                // flashinfer/data/csrc/trtllm_fused_moe_kernel_launcher.cu
                // line 2068.  Older comment claimed "0.6.10+" added the
                // arg, but that's a future-version annotation — adding it
                // here triggers "Mismatched number of arguments ...
                // Expected 34 but got 35" at first NVFP4 MoE dispatch.
                static_cast<int64_t>(p.num_experts),
                static_cast<int64_t>(p.top_k),
                p.n_group > 0 ? ffi::Optional<int64_t>(static_cast<int64_t>(p.n_group)) : ffi::Optional<int64_t>(),
                p.topk_group > 0 ? ffi::Optional<int64_t>(static_cast<int64_t>(p.topk_group)) :
                                    ffi::Optional<int64_t>(),
                static_cast<int64_t>(p.intermediate_size),
                static_cast<int64_t>(p.local_expert_offset),
                static_cast<int64_t>(p.local_num_experts),
                ffi::Optional<double>(p.routed_scaling_factor),
                static_cast<int64_t>(p.routing_method),
                p.do_finalize,
                p.enable_pdl,
                static_cast<int64_t>(p.gated_act_type),
                output,
                config_index,
                p.norm_topk_prob,
                ffi::Optional<ffi::Tensor>());
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "[flashinfer-moe] fp4_block_scale call failed: " << e.what() << std::endl;
            return false;
        }
    };

    const int64_t dtype_weights = p.weight_scale_vec_size == 32 ? kTrtllmMxE2m1 : kTrtllmE2m1;
    const int64_t dtype_act = trtllm_dtype(p.hidden_states_dtype, dtype_weights);
    const AutotuneKey key{/*variant=*/3,
                          bucket_num_tokens_for_autotune(p.num_tokens),
                          p.hidden_size,
                          p.intermediate_size,
                          p.num_experts,
                          p.top_k,
                          p.local_num_experts};
    const bool has_cached_tactic = lookup_cached_tactic(key).has_value();
    // Skip the FFI tactic-enumeration roundtrip when the autotuner is off
    // (steady-state inference): there is nothing to time, the launcher will
    // accept tile_n=-1/config_index=-1 and pick its own defaults.
    auto       candidates        = (has_cached_tactic || !tuning_enabled()) ?
                                       std::vector<AutotuneTactic>{} :
                                       get_valid_moe_tactics(key,
                                                             kFp8QuantizationNone,
                                                             /*use_shuffled_weight=*/false,
                                                             /*weight_layout=*/0,
                                                             dtype_act,
                                                             dtype_weights);
    ProfileOutputScratch profile_output(tuning_enabled() && !has_cached_tactic && !candidates.empty() ?
                                            p.num_tokens :
                                            0,
                                        p.hidden_size,
                                        p.stream);
    const AutotuneTactic selected = select_tactic(key, std::move(candidates), p.stream, [&](AutotuneTactic tactic) {
        if (!profile_output.data()) {
            return false;
        }
        profile_output_ptr = profile_output.data();
        const bool ok      = run_with_tactic(tactic);
        profile_output_ptr = nullptr;
        return ok;
    });
    return run_with_tactic(selected);
}

}  // namespace

bool dispatch_fp4_block_scale(const Fp4BlockScaleMoeParamsRouting& p)
{
    auto routing_logits_dt = routing_logits_dl(p.routing_logits_dtype, p.routing_method);
    auto routing_logits    = make_view(p.routing_logits, {p.num_tokens, p.num_experts}, {}, routing_logits_dt);
    ffi::Optional<ffi::Tensor> topk_ids;
    ffi::Optional<ffi::Tensor> expert_weights;
    if (p.topk_ids_workspace && p.expert_weights_workspace) {
        topk_ids = make_view(p.topk_ids_workspace, {p.num_tokens, p.top_k}, {}, DLDataType{kDLInt, 32, 1});
        expert_weights =
            make_view(p.expert_weights_workspace, {p.num_tokens, p.top_k}, {}, DLDataType{kDLFloat, 32, 1});
    }
    return dispatch_fp4_impl(
        static_cast<const Fp4MoeBase&>(p), ffi::Optional<ffi::Tensor>(routing_logits), topk_ids, expert_weights);
}

bool dispatch_fp4_block_scale(const Fp4BlockScaleMoeParamsTopK& p)
{
    auto topk_ids       = make_view(p.topk_ids, {p.num_tokens, p.top_k}, {}, DLDataType{kDLInt, 32, 1});
    auto expert_weights = make_view(p.expert_weights, {p.num_tokens, p.top_k}, {}, DLDataType{kDLFloat, 32, 1});
    return dispatch_fp4_impl(static_cast<const Fp4MoeBase&>(p),
                             ffi::Optional<ffi::Tensor>(),
                             ffi::Optional<ffi::Tensor>(topk_ids),
                             ffi::Optional<ffi::Tensor>(expert_weights));
}

bool dispatch_cutlass_w4a8_nvfp4_fp8(const CutlassW4A8Nvfp4Fp8MoeParams& p)
{
    auto& L = cutlass_loaded();
    if (!L.ok || !L.run_moe.has_value()) {
        std::cerr << "[flashinfer-moe] CUTLASS W4A8 dispatch: fused_moe runner not loaded.\n";
        return false;
    }
    TvmFfiStreamScope ffi_stream{p.stream};

    auto output = make_view(p.output, {p.num_tokens, p.hidden_size}, {}, DLDataType{kDLBfloat, 16, 1});
    auto input  = make_view(p.input, {p.num_tokens, p.hidden_size}, {}, DLDataType{kDLFloat8_e4m3fn, 8, 1});
    auto token_selected_experts =
        make_view(p.token_selected_experts, {p.num_tokens, p.top_k}, {}, DLDataType{kDLInt, 32, 1});
    auto token_final_scales = make_view(p.token_final_scales, {p.num_tokens, p.top_k}, {}, DLDataType{kDLFloat, 32, 1});

    auto fc1_expert_weights = make_view(p.fc1_expert_weights,
                                        {p.num_experts, 2 * p.intermediate_size, p.hidden_size / 16},
                                        {},
                                        DLDataType{kDLInt, 64, 1});
    auto fc2_expert_weights = make_view(
        p.fc2_expert_weights, {p.num_experts, p.hidden_size, p.intermediate_size / 16}, {}, DLDataType{kDLInt, 64, 1});

    ffi::Array<ffi::Tensor> quant_scales;
    quant_scales.push_back(make_view(p.fc1_weight_block_scale,
                                     {p.num_experts, 2 * p.intermediate_size, p.hidden_size / 128},
                                     {},
                                     DLDataType{kDLInt, 32, 1}));
    quant_scales.push_back(make_view(p.fc1_global_scale, {p.num_experts}, {}, DLDataType{kDLFloat, 32, 1}));
    if (!p.use_mxfp8_act_scaling) {
        quant_scales.push_back(make_view(p.fc2_act_global_scale, {p.num_experts}, {}, DLDataType{kDLFloat, 32, 1}));
    }
    quant_scales.push_back(make_view(p.fc2_weight_block_scale,
                                     {p.num_experts, p.hidden_size, p.intermediate_size / 128},
                                     {},
                                     DLDataType{kDLInt, 32, 1}));
    quant_scales.push_back(make_view(p.fc2_global_scale, {p.num_experts}, {}, DLDataType{kDLFloat, 32, 1}));

    auto run_profile = [&](int64_t gemm_idx, int64_t profile_id, bool do_preparation) {
        if (!L.run_gemm_profile.has_value()) {
            return false;
        }
        try {
            (*L.run_gemm_profile)(input,
                                  fc1_expert_weights,
                                  ffi::Optional<ffi::Tensor>(),
                                  fc2_expert_weights,
                                  ffi::Optional<ffi::Tensor>(),
                                  static_cast<int64_t>(p.top_k),
                                  static_cast<int64_t>(1),   // tp_size
                                  static_cast<int64_t>(0),   // tp_rank
                                  static_cast<int64_t>(1),   // ep_size
                                  static_cast<int64_t>(0),   // ep_rank
                                  static_cast<int64_t>(1),   // cluster_size
                                  static_cast<int64_t>(0),   // cluster_rank
                                  false,                     // enable_alltoall
                                  false,                     // min_latency_mode
                                  gemm_idx,
                                  profile_id,
                                  do_preparation,
                                  p.enable_pdl,
                                  kActivationSwiglu);
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "[flashinfer-moe] CUTLASS W4A8 profile failed: " << e.what()
                      << " gemm_idx=" << gemm_idx << " profile=" << profile_id
                      << " preparation=" << do_preparation << "\n";
            return false;
        }
    };

    auto make_range = [](int64_t begin, int64_t count) {
        std::vector<int64_t> values;
        if (count > 0) {
            values.reserve(static_cast<size_t>(count));
            for (int64_t i = 0; i < count; ++i) {
                values.push_back(begin + i);
            }
        }
        return values;
    };

    int64_t gemm1_tactic = p.gemm1_tactic;
    int64_t gemm2_tactic = p.gemm2_tactic;

    // Cache lookup BEFORE the run_profile(do_preparation=true) call: that
    // preparation pass executes a real cutlass grouped-GEMM kernel and is
    // NOT capture-safe (returns "Error Internal" inside
    // cudaStreamBeginCapture).  When the autotune cache already has a tactic
    // for this bucket — populated by an earlier non-capture WarmUp call — we
    // can use it directly and skip preparation entirely, keeping CG capture
    // safe AND retaining the full autotune perf benefit.
    const AutotuneKey gemm1_key{/*variant=*/4,
                                bucket_num_tokens_for_autotune(p.num_tokens),
                                p.hidden_size,
                                p.intermediate_size,
                                p.num_experts,
                                p.top_k,
                                p.num_experts};
    const AutotuneKey gemm2_key{/*variant=*/5,
                                bucket_num_tokens_for_autotune(p.num_tokens),
                                p.hidden_size,
                                p.intermediate_size,
                                p.num_experts,
                                p.top_k,
                                p.num_experts};
    // First: cache lookup — if WarmUp's autotune populated this bucket, use
    // the cached tactic directly and skip the (capture-unsafe) preparation
    // call entirely.  Mirrors `dispatch_fp4_impl` at line ~1380 and
    // TRT-LLM's `AutoTuner.choose_one` short-circuit pattern.
    if (gemm1_tactic < 0) {
        if (auto cached = lookup_cached_tactic(gemm1_key)) {
            gemm1_tactic = cached->tile_n;
        }
    }
    if (gemm2_tactic < 0) {
        if (auto cached = lookup_cached_tactic(gemm2_key)) {
            gemm2_tactic = cached->tile_n;
        }
    }

    // Run preparation + tactic selection ONLY when autotune is active (i.e.
    // during the WarmUp pass at engine init).  Mirrors TRT-LLM
    // `AutoTuner.is_tuning_mode` gating: outside the autotune window —
    // including CG capture and steady-state inference — `tuning_enabled()`
    // returns false and any uncached bucket falls through to tactic=-1,
    // which the launcher resolves to the default cutlass kernel.  This
    // avoids the otherwise-unconditional `run_gemm_profile(preparation=true)`
    // call, which is NOT capture-safe (returns "Error Internal" inside
    // cudaStreamBeginCapture) AND blows up at very small num_tokens (the
    // M128 group-GEMM kernel can't handle bs=1/2/4 with many experts).
    if (gemm1_tactic < 0 && tuning_enabled() && L.get_gemm1_tactic_count.has_value()
        && L.run_gemm_profile.has_value()) {
        int64_t count = 0;
        try {
            count = (*L.get_gemm1_tactic_count)().cast<int64_t>();
        }
        catch (const std::exception& e) {
            std::cerr << "[flashinfer-moe] get_gemm1_tactic_count failed: " << e.what() << "\n";
        }
        run_profile(/*gemm_idx=*/1, /*profile_id=*/-1, /*do_preparation=*/true);
        gemm1_tactic = select_profile_id(
            gemm1_key, make_range(0, count), p.stream, [&](int64_t profile_id) {
                return run_profile(/*gemm_idx=*/1, profile_id, /*do_preparation=*/false);
            }, "cutlass_w4a8_gemm1");
    }
    if (gemm2_tactic < 0 && tuning_enabled() && L.get_gemm1_tactic_count.has_value()
        && L.get_gemm2_tactic_count.has_value() && L.run_gemm_profile.has_value()) {
        int64_t gemm1_count = 0;
        int64_t gemm2_count = 0;
        try {
            gemm1_count = (*L.get_gemm1_tactic_count)().cast<int64_t>();
            gemm2_count = (*L.get_gemm2_tactic_count)().cast<int64_t>();
        }
        catch (const std::exception& e) {
            std::cerr << "[flashinfer-moe] get_gemm{1,2}_tactic_count failed: " << e.what() << "\n";
        }
        run_profile(/*gemm_idx=*/2, /*profile_id=*/-1, /*do_preparation=*/true);
        gemm2_tactic = select_profile_id(
            gemm2_key, make_range(gemm1_count, gemm2_count), p.stream, [&](int64_t profile_id) {
                return run_profile(/*gemm_idx=*/2, profile_id, /*do_preparation=*/false);
            }, "cutlass_w4a8_gemm2");
    }

    ffi::Array<int64_t> profile_ids;
    profile_ids.push_back(gemm1_tactic);
    profile_ids.push_back(gemm2_tactic);

    // Upstream ActivationType::Swiglu.

    try {
        (*L.run_moe)(output,
                     input,
                     token_selected_experts,
                     ffi::Optional<ffi::Tensor>(token_final_scales),
                     fc1_expert_weights,
                     ffi::Optional<ffi::Tensor>(),
                     fc2_expert_weights,
                     ffi::Optional<ffi::Tensor>(),
                     ffi::Optional<ffi::Array<ffi::Tensor>>(quant_scales),
                     p.use_mxfp8_act_scaling ?
                         ffi::Optional<ffi::Tensor>(
                             make_view(p.input_sf, {p.num_tokens, p.hidden_size / 32}, {}, DLDataType{kDLUInt, 8, 1})) :
                         ffi::Optional<ffi::Tensor>(),  // input_sf
                     ffi::Optional<ffi::Tensor>(),      // swiglu_alpha
                     ffi::Optional<ffi::Tensor>(),      // swiglu_beta
                     ffi::Optional<ffi::Tensor>(),      // swiglu_limit
                     false,                             // swizzled_input_sf
                     static_cast<int64_t>(1),           // tp_size
                     static_cast<int64_t>(0),           // tp_rank
                     static_cast<int64_t>(1),           // ep_size
                     static_cast<int64_t>(0),           // ep_rank
                     static_cast<int64_t>(1),           // cluster_size
                     static_cast<int64_t>(0),           // cluster_rank
                     false,                             // enable_alltoall
                     false,                             // min_latency_mode
                     ffi::Optional<ffi::Array<int64_t>>(profile_ids),
                     p.enable_pdl,
                     kActivationSwiglu);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[flashinfer-moe] CUTLASS W4A8 call failed: " << e.what() << std::endl;
        return false;
    }
}

// ---------------------------------------------------------------------------
// Autotuner cache.
// ---------------------------------------------------------------------------

bool AutotuneKey::operator==(const AutotuneKey& o) const noexcept
{
    return variant == o.variant && num_tokens == o.num_tokens && hidden_size == o.hidden_size
           && intermediate_size == o.intermediate_size && num_experts == o.num_experts && top_k == o.top_k
           && local_num_experts == o.local_num_experts;
}

namespace {

struct AutotuneKeyHash {
    size_t operator()(const AutotuneKey& k) const noexcept
    {
        // Cheap mix; shapes won't collide in practice (per-process cache).
        size_t h   = static_cast<size_t>(k.variant);
        auto   mix = [&](size_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
        mix(static_cast<size_t>(k.num_tokens));
        mix(static_cast<size_t>(k.hidden_size));
        mix(static_cast<size_t>(k.intermediate_size));
        mix(static_cast<size_t>(k.num_experts));
        mix(static_cast<size_t>(k.top_k));
        mix(static_cast<size_t>(k.local_num_experts));
        return h;
    }
};

std::mutex& tactic_mutex()
{
    static std::mutex m;
    return m;
}

std::unordered_map<AutotuneKey, AutotuneTactic, AutotuneKeyHash>& tactic_cache()
{
    static std::unordered_map<AutotuneKey, AutotuneTactic, AutotuneKeyHash> cache;
    return cache;
}

bool tuning_enabled()
{
    return g_tuning_active.load(std::memory_order_relaxed);
}

bool tuning_log_enabled()
{
    // Tuning is enabled by default; detailed tactic logs are only useful when debugging.
    return g_tuning_active.load(std::memory_order_relaxed)
           && turbomind::Logger::getLogger().getLevel() <= turbomind::Logger::DEBUG;
}

std::optional<AutotuneTactic> lookup_cached_tactic(const AutotuneKey& k)
{
    std::lock_guard<std::mutex> g(tactic_mutex());
    auto&                       c  = tactic_cache();
    auto                        it = c.find(k);
    if (it == c.end()) {
        return std::nullopt;
    }
    return it->second;
}

template<class Fn>
bool profile_once(cudaStream_t stream, Fn&& fn, float& ms)
{
    cudaEvent_t start{};
    cudaEvent_t stop{};
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        if (start)
            cudaEventDestroy(start);
        if (stop)
            cudaEventDestroy(stop);
        return fn();
    }

    bool ok = false;
    try {
        cudaEventRecord(start, stream);
        ok = fn();
        cudaEventRecord(stop, stream);
        const cudaError_t sync_err = cudaEventSynchronize(stop);
        if (sync_err != cudaSuccess) {
            std::cerr << "[flashinfer-moe] tactic profiling failed: " << cudaGetErrorString(sync_err) << "\n";
            ok = false;
            cudaGetLastError();
        }
        else {
            cudaEventElapsedTime(&ms, start, stop);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[flashinfer-moe] tactic profiling threw: " << e.what() << "\n";
        ok = false;
        cudaGetLastError();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ok;
}

std::vector<AutotuneTactic> get_valid_moe_tactics(const AutotuneKey& k,
                                                  int64_t            fp8_quantization_type,
                                                  bool               use_shuffled_weight,
                                                  int64_t            weight_layout,
                                                  int64_t            dtype_act,
                                                  int64_t            dtype_weights)
{
    auto& L = loaded();
    if (!L.ok || !L.valid_configs.has_value()) {
        return {};
    }

    std::vector<AutotuneTactic> tactics;
    try {
        auto configs =
            (*L.valid_configs)(dtype_act,
                               dtype_weights,
                               fp8_quantization_type,
                               static_cast<int64_t>(k.top_k),
                               static_cast<int64_t>(k.hidden_size),
                               static_cast<int64_t>(k.intermediate_size),
                               static_cast<int64_t>(k.local_num_experts),
                               kActivationSwiglu,
                               use_shuffled_weight,
                               weight_layout,
                               static_cast<int64_t>(k.num_tokens))
                .cast<ffi::Array<ffi::Array<int64_t>>>();
        tactics.reserve(configs.size());
        for (const auto& cfg : configs) {
            if (cfg.size() >= 2) {
                tactics.push_back(AutotuneTactic{cfg[0], cfg[1]});
            }
        }
    }
    catch (const std::exception& e) {
        // Log unconditionally: a silent failure here means we silently
        // fall back to the default tactic, which masks perf regressions
        // and FFI signature drift (e.g. flashinfer-python adding/removing
        // an arg).  Throttle by hashing into a small set of seen messages.
        static std::mutex                seen_mu;
        static std::unordered_set<std::string> seen;
        std::string                       msg = e.what();
        bool                              fresh = false;
        {
            std::lock_guard<std::mutex> g(seen_mu);
            fresh = seen.insert(msg).second;
        }
        if (fresh) {
            std::cerr << "[flashinfer-moe] get_valid_moe_tactics FFI failed: " << msg
                      << "  (tuning will fall back to default tactic for this shape)\n";
        }
    }
    return tactics;
}

AutotuneTactic select_tactic(const AutotuneKey&              k,
                             std::vector<AutotuneTactic>&&   candidates,
                             cudaStream_t                    stream,
                             const std::function<bool(AutotuneTactic)>& run_candidate)
{
    if (auto cached = lookup_cached_tactic(k)) {
        if (tuning_log_enabled()) {
            std::cerr << "[flashinfer-moe][tuning] cache hit tactic tile_n=" << cached->tile_n
                      << " config=" << cached->config_index << " variant=" << k.variant
                      << " tokens=" << k.num_tokens << "\n";
        }
        return *cached;
    }
    if (!tuning_enabled() || candidates.empty()) {
        if (tuning_log_enabled()) {
            std::cerr << "[flashinfer-moe][tuning] skip MoE tuning; use default tactic variant=" << k.variant
                      << " tokens=" << k.num_tokens << " hidden=" << k.hidden_size
                      << " inter=" << k.intermediate_size << " experts=" << k.num_experts
                      << " candidates=" << candidates.size() << " active=" << tuning_enabled() << "\n";
        }
        return AutotuneTactic{-1, -1};
    }

    if (tuning_log_enabled()) {
        std::cerr << "[flashinfer-moe][tuning] start candidates=" << candidates.size()
                  << " variant=" << k.variant << " tokens=" << k.num_tokens
                  << " hidden=" << k.hidden_size << " inter=" << k.intermediate_size
                  << " experts=" << k.num_experts << " top_k=" << k.top_k << "\n";
    }

    AutotuneTactic best{-1, -1};
    float          best_ms = std::numeric_limits<float>::infinity();
    for (const auto& tactic : candidates) {
        float ms = 0.0f;
        const bool ok = profile_once(stream, [&]() { return run_candidate(tactic); }, ms);
        if (ok && ms < best_ms) {
            best    = tactic;
            best_ms = ms;
        }
        if (tuning_log_enabled()) {
            std::cerr << "[flashinfer-moe][tuning] candidate tile_n=" << tactic.tile_n
                      << " config=" << tactic.config_index << (ok ? " ok" : " failed")
                      << " time_ms=" << ms << "\n";
        }
    }
    if (best.tile_n >= 0 && best.config_index >= 0) {
        record_tactic(k, best);
        if (tuning_log_enabled()) {
            std::cerr << "[flashinfer-moe][tuning] selected tactic tile_n=" << best.tile_n
                      << " config=" << best.config_index << " variant=" << k.variant << " tokens=" << k.num_tokens
                      << " hidden=" << k.hidden_size << " inter=" << k.intermediate_size
                      << " experts=" << k.num_experts << " top_k=" << k.top_k << " time_ms=" << best_ms << "\n";
        }
    }
    return best;
}

int64_t select_profile_id(const AutotuneKey&                   k,
                          std::vector<int64_t>&&               candidates,
                          cudaStream_t                         stream,
                          const std::function<bool(int64_t)>&  run_candidate,
                          const char*                          tag)
{
    if (auto cached = lookup_cached_tactic(k)) {
        if (tuning_log_enabled()) {
            std::cerr << "[flashinfer-moe][tuning] cache hit " << tag << " profile=" << cached->tile_n
                      << " variant=" << k.variant << " tokens=" << k.num_tokens << "\n";
        }
        return cached->tile_n;
    }
    if (!tuning_enabled() || candidates.empty()) {
        if (tuning_log_enabled()) {
            std::cerr << "[flashinfer-moe][tuning] skip " << tag << " tuning; use default profile=-1"
                      << " tokens=" << k.num_tokens << " candidates=" << candidates.size()
                      << " active=" << tuning_enabled() << "\n";
        }
        return -1;
    }

    if (tuning_log_enabled()) {
        std::cerr << "[flashinfer-moe][tuning] start " << tag << " profiles=" << candidates.size()
                  << " tokens=" << k.num_tokens << " hidden=" << k.hidden_size
                  << " inter=" << k.intermediate_size << " experts=" << k.num_experts << "\n";
    }

    int64_t best    = -1;
    float   best_ms = std::numeric_limits<float>::infinity();
    for (const auto profile_id : candidates) {
        float ms = 0.0f;
        const bool ok = profile_once(stream, [&]() { return run_candidate(profile_id); }, ms);
        if (ok && ms < best_ms) {
            best    = profile_id;
            best_ms = ms;
        }
        if (tuning_log_enabled()) {
            std::cerr << "[flashinfer-moe][tuning] " << tag << " candidate profile=" << profile_id
                      << (ok ? " ok" : " failed") << " time_ms=" << ms << "\n";
        }
    }
    if (best >= 0) {
        record_tactic(k, AutotuneTactic{best, 0});
        if (tuning_log_enabled()) {
            std::cerr << "[flashinfer-moe][tuning] selected " << tag << " profile=" << best
                      << " tokens=" << k.num_tokens << " time_ms=" << best_ms << "\n";
        }
    }
    return best;
}

}  // namespace

AutotuneTactic lookup_tactic(const AutotuneKey& k)
{
    if (auto cached = lookup_cached_tactic(k)) {
        return *cached;
    }
    {
        // Sentinel: launcher selects defaults.  Caller can probe a small
        // number of tile/config combinations once + `record_tactic` the
        // winner; later calls hit the cache.
        return AutotuneTactic{-1, -1};
    }
}

void record_tactic(const AutotuneKey& k, AutotuneTactic v)
{
    std::lock_guard<std::mutex> g(tactic_mutex());
    tactic_cache()[k] = v;
}

// ---------------------------------------------------------------------------
// High-level façade — picks BF16 / FP8 per-tensor / FP8 block-scale based
// on `weight_dtype` + `weight_block_size`, hides the routing-method enum
// mapping, and consults the autotuner cache.
// ---------------------------------------------------------------------------

namespace {

// Map turbomind's `topk_method` string + `norm_topk_prob` flag onto the
// FlashInfer `RoutingMethod` enum.  Centralised here so callers don't
// need to include flashinfer-specific headers to figure out the value.
RoutingMethod resolve_routing_method(const char* topk_method, bool norm_topk_prob)
{
    const std::string m = topk_method ? std::string(topk_method) : std::string();
    if (m == "noaux_tc") {
        // Sigmoid → bias-add → top-k-in-group → top-groups → top-k experts;
        // matches DeepSeek-V3.
        return RoutingMethod::kDeepSeekV3;
    }
    return norm_topk_prob ? RoutingMethod::kRenormalizeNaive : RoutingMethod::kDefault;
}

}  // namespace

bool dispatch_auto(const AutoMoeRequest& r)
{
    if (!is_available()) {
        std::cerr << "[flashinfer-moe] dispatch_auto: libflashinfer_moe.so not loaded.\n";
        return false;
    }

    const RoutingMethod routing = resolve_routing_method(r.topk_method, r.norm_topk_prob);

    // ---------- BF16 ------------------------------------------------------
    if (r.weight_dtype == DType::kBF16) {
        const AutotuneKey    key{/*variant=*/0,
                              bucket_num_tokens_for_autotune(r.num_tokens),
                              r.hidden_size,
                              r.intermediate_size,
                              r.num_experts,
                              r.top_k,
                              r.local_num_experts};

        Bf16MoeParams p{};
        p.routing_logits       = r.routing_logits;
        p.routing_logits_dtype = r.routing_logits_dtype;
        p.routing_bias         = r.routing_bias;
        p.hidden_states        = r.hidden_states;
        p.gemm1_weights        = r.gemm1_weights;
        p.gemm2_weights        = r.gemm2_weights;
        p.output               = r.output;

        p.num_tokens          = r.num_tokens;
        p.num_experts         = r.num_experts;
        p.hidden_size         = r.hidden_size;
        p.intermediate_size   = r.intermediate_size;
        p.top_k               = r.top_k;
        p.n_group             = r.n_group;
        p.topk_group          = r.topk_group;
        p.local_expert_offset = r.local_expert_offset;
        p.local_num_experts   = r.local_num_experts;

        p.routing_method      = routing;
        p.use_shuffled_weight = r.use_shuffled_weight;
        p.weight_layout       = r.weight_layout;
        p.enable_pdl          = r.enable_pdl;

        p.tile_n       = -1;
        p.config_index = -1;
        p.stream       = r.stream;

        const bool has_cached_tactic = lookup_cached_tactic(key).has_value();
        // See note in dispatch_fp4_impl: skip the FFI tactic enumeration when
        // the autotuner is off — `select_tactic` returns the {-1,-1} sentinel
        // and the launcher picks defaults itself.
        auto       candidates        = (has_cached_tactic || !tuning_enabled()) ?
                                           std::vector<AutotuneTactic>{} :
                                           get_valid_moe_tactics(key,
                                                                 kFp8QuantizationNone,
                                                                 r.use_shuffled_weight,
                                                                 r.weight_layout,
                                                                 kTrtllmBfloat16,
                                                                 kTrtllmBfloat16);
        ProfileOutputScratch profile_output(tuning_enabled() && !has_cached_tactic && !candidates.empty() ?
                                                r.num_tokens :
                                                0,
                                            r.hidden_size,
                                            r.stream);
        const AutotuneTactic selected = select_tactic(key, std::move(candidates), r.stream, [&](AutotuneTactic tactic) {
            p.tile_n       = tactic.tile_n;
            p.config_index = tactic.config_index;
            if (!profile_output.data()) {
                return false;
            }
            void* const real_output = p.output;
            p.output                = profile_output.data();
            const bool ok = dispatch_bf16(p);
            p.output      = real_output;
            return ok;
        });
        p.tile_n       = selected.tile_n;
        p.config_index = selected.config_index;
        return dispatch_bf16(p);
    }

    // ---------- FP8 -------------------------------------------------------
    if (r.weight_dtype != DType::kFP8E4M3) {
        std::cerr << "[flashinfer-moe] dispatch_auto: unsupported weight_dtype " << static_cast<int>(r.weight_dtype)
                  << "; expected kBF16 or kFP8E4M3 (NVFP4 callers should use "
                     "dispatch_fp4_block_scale directly).\n";
        return false;
    }

    if (r.weight_block_size != 1 && r.weight_block_size != 128) {
        std::cerr << "[flashinfer-moe] dispatch_auto: FP8 weight_block_size must be 1 "
                     "(per-tensor) or 128 (block-scale), got "
                  << r.weight_block_size << ".\n";
        return false;
    }

    const AutotuneKey key{/*variant=*/r.weight_block_size == 1 ? 1 : 2,
                          bucket_num_tokens_for_autotune(r.num_tokens),
                          r.hidden_size,
                          r.intermediate_size,
                          r.num_experts,
                          r.top_k,
                          r.local_num_experts};

    if (r.weight_block_size == 1) {
        // Per-tensor FP8 — extra scalar scale tensors required.
        if (!r.output1_scales_scalar || !r.output1_scales_gate_scalar || !r.output2_scales_scalar) {
            std::cerr << "[flashinfer-moe] dispatch_auto: per-tensor FP8 requires "
                         "output{1,2}_scales_scalar + output1_scales_gate_scalar.\n";
            return false;
        }
        Fp8PerTensorMoeParams p{};
        p.routing_logits             = r.routing_logits;
        p.routing_logits_dtype       = r.routing_logits_dtype;
        p.routing_bias               = r.routing_bias;
        p.hidden_states              = r.hidden_states;
        p.gemm1_weights              = r.gemm1_weights;
        p.output1_scales_scalar      = r.output1_scales_scalar;
        p.output1_scales_gate_scalar = r.output1_scales_gate_scalar;
        p.gemm2_weights              = r.gemm2_weights;
        p.output2_scales_scalar      = r.output2_scales_scalar;
        p.output                     = r.output;

        p.num_tokens          = r.num_tokens;
        p.num_experts         = r.num_experts;
        p.hidden_size         = r.hidden_size;
        p.intermediate_size   = r.intermediate_size;
        p.top_k               = r.top_k;
        p.n_group             = r.n_group;
        p.topk_group          = r.topk_group;
        p.local_expert_offset = r.local_expert_offset;
        p.local_num_experts   = r.local_num_experts;

        p.routed_scaling_factor       = r.routed_scaling_factor;
        p.use_routing_scales_on_input = r.use_routing_scales_on_input;
        p.routing_method              = routing;
        p.hidden_states_dtype         = r.hidden_states_dtype;
        p.enable_pdl                  = r.enable_pdl;

        p.tile_n       = -1;
        p.config_index = -1;
        p.stream       = r.stream;

        const bool has_cached_tactic = lookup_cached_tactic(key).has_value();
        // Skip the FFI tactic enumeration when the autotuner is off; see
        // dispatch_fp4_impl for the rationale.
        auto       candidates        = (has_cached_tactic || !tuning_enabled()) ?
                                           std::vector<AutotuneTactic>{} :
                                           get_valid_moe_tactics(key,
                                                                 kFp8QuantizationNone,
                                                                 /*use_shuffled_weight=*/true,
                                                                 /*weight_layout=*/0,
                                                                 trtllm_dtype(r.hidden_states_dtype),
                                                                 kTrtllmE4m3);
        ProfileOutputScratch profile_output(tuning_enabled() && !has_cached_tactic && !candidates.empty() ?
                                                r.num_tokens :
                                                0,
                                            r.hidden_size,
                                            r.stream);
        const AutotuneTactic selected = select_tactic(key, std::move(candidates), r.stream, [&](AutotuneTactic tactic) {
            p.tile_n       = tactic.tile_n;
            p.config_index = tactic.config_index;
            if (!profile_output.data()) {
                return false;
            }
            void* const real_output = p.output;
            p.output                = profile_output.data();
            const bool ok = dispatch_fp8_per_tensor(p);
            p.output      = real_output;
            return ok;
        });
        p.tile_n       = selected.tile_n;
        p.config_index = selected.config_index;
        return dispatch_fp8_per_tensor(p);
    }

    // Block-scale FP8 — `hidden_states_scale` is mandatory upstream
    // (see csrc/trtllm_fused_moe_kernel_launcher.cu:1567).
    if (!r.gemm1_weights_scale || !r.gemm2_weights_scale) {
        std::cerr << "[flashinfer-moe] dispatch_auto: block-scale FP8 requires "
                     "gemm{1,2}_weights_scale tensors.\n";
        return false;
    }
    if (!r.hidden_states_scale) {
        std::cerr << "[flashinfer-moe] dispatch_auto: block-scale FP8 requires "
                     "hidden_states_scale (caller must pre-quantise bf16/fp16 input).\n";
        return false;
    }

    Fp8BlockScaleMoeParams p{};
    p.routing_logits       = r.routing_logits;
    p.routing_logits_dtype = r.routing_logits_dtype;
    p.routing_bias         = r.routing_bias;
    p.hidden_states        = r.hidden_states;
    p.hidden_states_scale  = r.hidden_states_scale;
    p.gemm1_weights        = r.gemm1_weights;
    p.gemm1_weights_scale  = r.gemm1_weights_scale;
    p.gemm2_weights        = r.gemm2_weights;
    p.gemm2_weights_scale  = r.gemm2_weights_scale;
    p.output               = r.output;

    p.num_tokens          = r.num_tokens;
    p.num_experts         = r.num_experts;
    p.hidden_size         = r.hidden_size;
    p.intermediate_size   = r.intermediate_size;
    p.top_k               = r.top_k;
    p.n_group             = r.n_group;
    p.topk_group          = r.topk_group;
    p.local_expert_offset = r.local_expert_offset;
    p.local_num_experts   = r.local_num_experts;

    p.routed_scaling_factor = r.routed_scaling_factor;
    p.routing_method        = routing;
    p.hidden_states_dtype   = r.hidden_states_dtype;
    p.use_shuffled_weight   = r.use_shuffled_weight;
    p.weight_layout         = r.weight_layout;
    p.norm_topk_prob        = r.norm_topk_prob;
    p.enable_pdl            = r.enable_pdl;

    p.tile_n       = -1;
    p.config_index = -1;
    p.stream       = r.stream;

    const bool has_cached_tactic = lookup_cached_tactic(key).has_value();
    // Skip the FFI tactic enumeration when the autotuner is off; see
    // dispatch_fp4_impl for the rationale.
    auto       candidates        = (has_cached_tactic || !tuning_enabled()) ?
                                       std::vector<AutotuneTactic>{} :
                                       get_valid_moe_tactics(key,
                                                             kFp8QuantizationDeepSeek,
                                                             r.use_shuffled_weight,
                                                             r.weight_layout,
                                                             trtllm_dtype(r.hidden_states_dtype),
                                                             kTrtllmE4m3);
    ProfileOutputScratch profile_output(tuning_enabled() && !has_cached_tactic && !candidates.empty() ?
                                            r.num_tokens :
                                            0,
                                        r.hidden_size,
                                        r.stream);
    const AutotuneTactic selected = select_tactic(key, std::move(candidates), r.stream, [&](AutotuneTactic tactic) {
        p.tile_n       = tactic.tile_n;
        p.config_index = tactic.config_index;
        if (!profile_output.data()) {
            return false;
        }
        void* const real_output = p.output;
        p.output                = profile_output.data();
        const bool ok = dispatch_fp8_block_scale(p);
        p.output      = real_output;
        return ok;
    });
    p.tile_n       = selected.tile_n;
    p.config_index = selected.config_index;
    return dispatch_fp8_block_scale(p);
}

}  // namespace turbomind::trtllm_fused_moe
