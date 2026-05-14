// Copyright (c) OpenMMLab. All rights reserved.

#include "flashinfer_gemm_wrapper.h"
#include "lmdeploy/flashinfer_gemm_config.h"
#include "src/turbomind/utils/logger.h"

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <dlfcn.h>

#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace turbomind::flashinfer_gemm {

namespace ffi = tvm::ffi;

namespace {

std::atomic<bool> g_tuning_active{false};

struct Loaded {
    std::optional<ffi::Module>   mod;
    std::optional<ffi::Module>   fp4_mod;
    std::optional<ffi::Function> fp8_groupwise;
    std::optional<ffi::Function> fp4_gemm;
    std::optional<ffi::Function> fp4_tactic_num;
    bool                         ok{false};
    bool                         fp4_ok{false};
};

void promote_tvm_ffi_symbols()
{
    void* tvm_ffi = dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (!tvm_ffi) {
        tvm_ffi = dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!tvm_ffi) {
        std::cerr << "[flashinfer-gemm] warning: failed to promote libtvm_ffi.so: " << dlerror() << "\n";
    }
}

inline int current_device_id()
{
    int         dev_id = 0;
    cudaError_t err    = cudaGetDevice(&dev_id);
    if (err != cudaSuccess) {
        std::cerr << "[flashinfer-gemm] cudaGetDevice failed: " << cudaGetErrorString(err) << "\n";
        return 0;
    }
    return dev_id;
}

class TvmFfiStreamScope {
public:
    explicit TvmFfiStreamScope(cudaStream_t stream): device_id_{current_device_id()}
    {
        if (cudaError_t err = cudaSetDevice(device_id_); err != cudaSuccess) {
            std::cerr << "[flashinfer-gemm] cudaSetDevice(" << device_id_ << ") failed: " << cudaGetErrorString(err)
                      << "\n";
        }
        int rc = TVMFFIEnvSetStream(kDLCUDA, device_id_, stream, &prev_);
        if (rc != 0) {
            std::cerr << "[flashinfer-gemm] TVMFFIEnvSetStream(set) rc=" << rc << " device=" << device_id_ << "\n";
        }
    }

    ~TvmFfiStreamScope()
    {
        TVMFFIEnvSetStream(kDLCUDA, device_id_, prev_, nullptr);
    }

    TvmFfiStreamScope(const TvmFfiStreamScope&)            = delete;
    TvmFfiStreamScope& operator=(const TvmFfiStreamScope&) = delete;

private:
    int                device_id_{0};
    TVMFFIStreamHandle prev_{nullptr};
};

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
    auto* bundle                           = new DLBundle;
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

DLDataType to_dl(DType dtype)
{
    switch (dtype) {
        case DType::kFP16:
            return DLDataType{kDLFloat, 16, 1};
        case DType::kBF16:
            return DLDataType{kDLBfloat, 16, 1};
        case DType::kFP8E4M3:
            return DLDataType{kDLFloat8_e4m3fn, 8, 1};
    }
    return DLDataType{kDLBfloat, 16, 1};
}

int64_t round_up(int64_t x, int64_t y)
{
    return ((x + y - 1) / y) * y;
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

struct Nvfp4TuningKey {
    int m;
    int n;
    int k;
    int output_dtype;
    int device;

    bool operator==(const Nvfp4TuningKey& o) const noexcept
    {
        return m == o.m && n == o.n && k == o.k && output_dtype == o.output_dtype && device == o.device;
    }
};

struct Nvfp4TuningKeyHash {
    size_t operator()(const Nvfp4TuningKey& k) const noexcept
    {
        size_t h   = static_cast<size_t>(k.m);
        auto   mix = [&](size_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
        mix(static_cast<size_t>(k.n));
        mix(static_cast<size_t>(k.k));
        mix(static_cast<size_t>(k.output_dtype));
        mix(static_cast<size_t>(k.device));
        return h;
    }
};

std::mutex& nvfp4_tactic_mutex()
{
    static std::mutex m;
    return m;
}

std::unordered_map<Nvfp4TuningKey, int64_t, Nvfp4TuningKeyHash>& nvfp4_tactic_cache()
{
    static std::unordered_map<Nvfp4TuningKey, int64_t, Nvfp4TuningKeyHash> cache;
    return cache;
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
            std::cerr << "[flashinfer-gemm] tactic profiling failed: " << cudaGetErrorString(sync_err) << "\n";
            ok = false;
            cudaGetLastError();
        }
        else {
            cudaEventElapsedTime(&ms, start, stop);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[flashinfer-gemm] tactic profiling threw: " << e.what() << "\n";
        ok = false;
        cudaGetLastError();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ok;
}

std::string resolve_fp4_so_path()
{
    if (const char* override_path = std::getenv("LMDEPLOY_FLASHINFER_FP4_GEMM_SO"); override_path && *override_path) {
        return override_path;
    }

    cudaDeviceProp prop{};
    const int      dev_id = current_device_id();
    const auto     err    = cudaGetDeviceProperties(&prop, dev_id);
    const bool     sm103  = err == cudaSuccess && prop.major == 10 && prop.minor == 3;

    if (sm103) {
        if (const char* path = std::getenv("LMDEPLOY_FLASHINFER_FP4_GEMM_SM103_SO"); path && *path) {
            return path;
        }
        if (const char* path = LMDEPLOY_FLASHINFER_FP4_GEMM_SM103_SO_PATH; path && *path) {
            return path;
        }
    }
    else {
        if (const char* path = std::getenv("LMDEPLOY_FLASHINFER_FP4_GEMM_SM100_SO"); path && *path) {
            return path;
        }
        if (const char* path = LMDEPLOY_FLASHINFER_FP4_GEMM_SM100_SO_PATH; path && *path) {
            return path;
        }
    }

    if (const char* path = LMDEPLOY_FLASHINFER_FP4_GEMM_SO_PATH; path && *path) {
        return path;
    }
    return {};
}

Loaded& loaded()
{
    static Loaded         L;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
        try {
            const char*       override_path = std::getenv("LMDEPLOY_FLASHINFER_GEMM_SO");
            const std::string so_path =
                override_path && *override_path ? override_path : LMDEPLOY_FLASHINFER_GEMM_SO_PATH;

            promote_tvm_ffi_symbols();

            L.mod.emplace(ffi::Module::LoadFromFile(so_path));
            auto fn = (*L.mod)->GetFunction("gemm_fp8_nt_groupwise");
            if (!fn.has_value()) {
                std::cerr << "[flashinfer-gemm] gemm_fp8_nt_groupwise export not found in " << so_path << "\n";
                return;
            }
            L.fp8_groupwise.emplace(*fn);
            L.ok = true;
        }
        catch (const std::exception& e) {
            std::cerr << "[flashinfer-gemm] load failed: " << e.what() << std::endl;
        }

        try {
            const std::string so_path = resolve_fp4_so_path();

            promote_tvm_ffi_symbols();

            L.fp4_mod.emplace(ffi::Module::LoadFromFile(so_path));
            auto fn = (*L.fp4_mod)->GetFunction("fp4_gemm");
            if (!fn.has_value()) {
                std::cerr << "[flashinfer-gemm] fp4_gemm export not found in " << so_path << "\n";
                return;
            }
            L.fp4_gemm.emplace(*fn);
            if (auto tactic_num = (*L.fp4_mod)->GetFunction("fp4_gemm_tactic_num"); tactic_num.has_value()) {
                L.fp4_tactic_num.emplace(*tactic_num);
            }
            L.fp4_ok = true;
        }
        catch (const std::exception& e) {
            std::cerr << "[flashinfer-gemm] FP4 load failed: " << e.what() << std::endl;
        }
    });
    return L;
}

int64_t select_nvfp4_tactic(Loaded&       L,
                            const Nvfp4GemmParams& p,
                            const Nvfp4TuningKey&  key,
                            ffi::Tensor&           input,
                            ffi::Tensor&           weight,
                            ffi::Tensor&           input_scale,
                            ffi::Tensor&           weight_scale,
                            ffi::Tensor&           alpha,
                            ffi::Tensor&           output,
                            ffi::Tensor&           workspace)
{
    if (p.tactic >= 0) {
        return p.tactic;
    }
    {
        std::lock_guard<std::mutex> g(nvfp4_tactic_mutex());
        auto&                       cache = nvfp4_tactic_cache();
        if (auto it = cache.find(key); it != cache.end()) {
            if (tuning_log_enabled()) {
                std::cerr << "[flashinfer-gemm][tuning] cache hit NVFP4 tactic=" << it->second << " m=" << p.m
                          << " n=" << p.n << " k=" << p.k << "\n";
            }
            return it->second;
        }
    }

    if (!tuning_enabled() || !L.fp4_tactic_num.has_value()) {
        if (tuning_log_enabled()) {
            std::cerr << "[flashinfer-gemm][tuning] skip NVFP4 tuning; use tactic=0 m=" << p.m << " n=" << p.n
                      << " k=" << p.k << " active=" << tuning_enabled()
                      << " has_tactic_num=" << L.fp4_tactic_num.has_value() << "\n";
        }
        return 0;
    }

    int64_t tactic_count = 0;
    try {
        tactic_count = (*L.fp4_tactic_num)().cast<int64_t>();
    }
    catch (const std::exception& e) {
        std::cerr << "[flashinfer-gemm] fp4_gemm_tactic_num failed: " << e.what() << "\n";
        return 0;
    }

    if (tactic_count <= 0) {
        return 0;
    }

    if (tuning_log_enabled()) {
        std::cerr << "[flashinfer-gemm][tuning] start NVFP4 candidates=" << tactic_count << " m=" << p.m
                  << " n=" << p.n << " k=" << p.k << " output_dtype=" << static_cast<int>(p.output_dtype) << "\n";
    }

    int64_t best_tactic = 0;
    float   best_ms     = std::numeric_limits<float>::infinity();
    for (int64_t tactic = 0; tactic < tactic_count; ++tactic) {
        float ms = 0.0f;
        const bool ok = profile_once(p.stream, [&]() {
            (*L.fp4_gemm)(input, weight, input_scale, weight_scale, alpha, output, workspace, tactic);
            return true;
        }, ms);
        if (ok && ms < best_ms) {
            best_ms     = ms;
            best_tactic = tactic;
        }
        if (tuning_log_enabled()) {
            std::cerr << "[flashinfer-gemm][tuning] NVFP4 candidate tactic=" << tactic
                      << (ok ? " ok" : " failed") << " time_ms=" << ms << "\n";
        }
    }

    {
        std::lock_guard<std::mutex> g(nvfp4_tactic_mutex());
        nvfp4_tactic_cache()[key] = best_tactic;
    }

    if (tuning_log_enabled()) {
        std::cerr << "[flashinfer-gemm][tuning] selected NVFP4 tactic=" << best_tactic << " m=" << p.m << " n=" << p.n
                  << " k=" << p.k << " time_ms=" << best_ms << "\n";
    }
    return best_tactic;
}

}  // namespace

bool is_available()
{
    return loaded().ok;
}

bool is_nvfp4_available()
{
    return loaded().fp4_ok;
}

void SetTuningActive(bool active)
{
    g_tuning_active.store(active, std::memory_order_relaxed);
}

bool dispatch_fp8_groupwise(const Fp8GroupwiseGemmParams& p)
{
    auto& L = loaded();
    if (!L.ok || !L.fp8_groupwise.has_value()) {
        return false;
    }
    if (!p.input || !p.input_scale || !p.weight || !p.weight_scale || !p.output || !p.workspace) {
        std::cerr << "[flashinfer-gemm] null pointer in FP8 groupwise GEMM params\n";
        return false;
    }
    if (p.m <= 0 || p.n <= 0 || p.k <= 0 || p.k % 128 != 0 || p.n % 128 != 0 || p.workspace_bytes <= 0) {
        std::cerr << "[flashinfer-gemm] invalid FP8 groupwise GEMM shape: m=" << p.m << " n=" << p.n
                  << " k=" << p.k << " workspace=" << p.workspace_bytes << "\n";
        return false;
    }
    if (tuning_log_enabled()) {
        std::cerr << "[flashinfer-gemm][tuning] FP8 groupwise GEMM uses FlashInfer kernel selection"
                  << " (no external tactic parameter in current FFI) m=" << p.m << " n=" << p.n
                  << " k=" << p.k << " mma_sm=" << p.mma_sm << "\n";
    }

    try {
        TvmFfiStreamScope scope{p.stream};
        const int         k_blocks = p.k / 128;
        const int         n_blocks = p.n / 128;

        auto workspace =
            make_view(p.workspace, {p.workspace_bytes}, {}, DLDataType{kDLUInt, 8, 1});
        auto input  = make_view(const_cast<void*>(p.input), {p.m, p.k}, {}, to_dl(DType::kFP8E4M3));
        auto weight = make_view(const_cast<void*>(p.weight), {p.n, p.k}, {}, to_dl(DType::kFP8E4M3));
        auto input_scale =
            make_view(const_cast<void*>(p.input_scale), {k_blocks, p.m}, {}, DLDataType{kDLFloat, 32, 1});
        auto weight_scale =
            make_view(const_cast<void*>(p.weight_scale), {k_blocks, n_blocks}, {}, DLDataType{kDLFloat, 32, 1});
        auto output = make_view(p.output, {p.m, p.n}, {}, to_dl(p.output_dtype));

        (*L.fp8_groupwise)(workspace,
                           input,
                           weight,
                           input_scale,
                           weight_scale,
                           output,
                           int64_t{1},
                           int64_t{128},
                           int64_t{128},
                           std::string{"MN"},
                           static_cast<int64_t>(p.mma_sm));
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[flashinfer-gemm] fp8 groupwise call failed: " << e.what() << " (m=" << p.m
                  << " n=" << p.n << " k=" << p.k << " mma_sm=" << p.mma_sm
                  << " workspace=" << p.workspace_bytes << " output_dtype=" << static_cast<int>(p.output_dtype)
                  << ")" << std::endl;
        return false;
    }
}

bool dispatch_nvfp4(const Nvfp4GemmParams& p)
{
    auto& L = loaded();
    if (!L.fp4_ok || !L.fp4_gemm.has_value()) {
        return false;
    }
    if (!p.input || !p.input_scale || !p.weight || !p.weight_scale || !p.alpha || !p.output || !p.workspace) {
        std::cerr << "[flashinfer-gemm] null pointer in NVFP4 GEMM params\n";
        return false;
    }
    if (p.m <= 0 || p.n <= 0 || p.k <= 0 || p.k % 32 != 0 || p.n % 8 != 0 || p.workspace_bytes <= 0) {
        std::cerr << "[flashinfer-gemm] invalid NVFP4 GEMM shape: m=" << p.m << " n=" << p.n << " k=" << p.k
                  << " workspace=" << p.workspace_bytes << "\n";
        return false;
    }

    try {
        TvmFfiStreamScope scope{p.stream};
        const int64_t scale_cols = round_up(p.k / 16, 4);
        const int64_t scale_m    = round_up(p.m, 128);
        const int64_t scale_n    = round_up(p.n, 128);

        auto input = make_view(
            const_cast<void*>(p.input), {p.m, p.k / 2}, {}, DLDataType{kDLUInt, 8, 1});
        auto weight = make_view(
            const_cast<void*>(p.weight), {p.n, p.k / 2}, {}, DLDataType{kDLUInt, 8, 1});
        auto input_scale =
            make_view(const_cast<void*>(p.input_scale), {scale_m, scale_cols}, {}, DLDataType{kDLUInt, 8, 1});
        auto weight_scale =
            make_view(const_cast<void*>(p.weight_scale), {scale_n, scale_cols}, {}, DLDataType{kDLUInt, 8, 1});
        auto alpha = make_view(const_cast<void*>(p.alpha), {1}, {}, DLDataType{kDLFloat, 32, 1});
        auto output = make_view(p.output, {p.m, p.n}, {}, to_dl(p.output_dtype));
        auto workspace =
            make_view(p.workspace, {p.workspace_bytes}, {}, DLDataType{kDLUInt, 8, 1});

        const Nvfp4TuningKey key{p.m, p.n, p.k, static_cast<int>(p.output_dtype), current_device_id()};
        const int64_t tactic =
            select_nvfp4_tactic(L, p, key, input, weight, input_scale, weight_scale, alpha, output, workspace);

        (*L.fp4_gemm)(input, weight, input_scale, weight_scale, alpha, output, workspace, tactic);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[flashinfer-gemm] NVFP4 call failed: " << e.what() << " (m=" << p.m << " n=" << p.n
                  << " k=" << p.k << " workspace=" << p.workspace_bytes
                  << " output_dtype=" << static_cast<int>(p.output_dtype) << ")" << std::endl;
        return false;
    }
}

}  // namespace turbomind::flashinfer_gemm
