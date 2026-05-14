// Copyright (c) OpenMMLab. All rights reserved.

#pragma once
#include "src/turbomind/utils/nvtx_utils.h"
#include <cuda_runtime.h>
#include <sstream>
#include <string>
#include <vector>

namespace turbomind {

enum QuantPolicy
{
    kNone = 0x00,
    // reserve 0x01 and 0x02 for backward compatibility
    kReserve1 = 0x01,
    kReserve2 = 0x02,
    // quantize cache kv
    kCacheKVInt8 = 0x08,
    kCacheKVInt4 = 0x04,
    kCacheKVFP8 = 0x10,
    kCacheKVFP4 = 0x20,
};

inline bool IsCacheKVFP8(int quant_policy)
{
    return (quant_policy & QuantPolicy::kCacheKVFP8) != 0;
}

inline bool IsCacheKVFP4(int quant_policy)
{
    return (quant_policy & QuantPolicy::kCacheKVFP4) != 0;
}

inline bool IsCacheKVFP(int quant_policy)
{
    return IsCacheKVFP8(quant_policy) || IsCacheKVFP4(quant_policy);
}

inline bool IsSupportedQuantPolicy(int quant_policy)
{
    return quant_policy == QuantPolicy::kNone || quant_policy == QuantPolicy::kCacheKVInt4
           || quant_policy == QuantPolicy::kCacheKVInt8 || quant_policy == QuantPolicy::kCacheKVFP8
           || quant_policy == QuantPolicy::kCacheKVFP4;
}

inline int KvCacheElemBits(int quant_policy, int default_bits)
{
    if (quant_policy & (QuantPolicy::kCacheKVInt4 | QuantPolicy::kCacheKVFP4)) {
        return 4;
    }
    if (quant_policy & (QuantPolicy::kCacheKVInt8 | QuantPolicy::kCacheKVFP8)) {
        return 8;
    }
    return default_bits;
}

inline const char* QuantPolicyName(int quant_policy)
{
    switch (quant_policy) {
        case QuantPolicy::kNone: return "none";
        case QuantPolicy::kCacheKVInt4: return "kCacheKVInt4";
        case QuantPolicy::kCacheKVInt8: return "kCacheKVInt8";
        case QuantPolicy::kCacheKVFP8: return "kCacheKVFP8";
        case QuantPolicy::kCacheKVFP4: return "kCacheKVFP4";
        default: return "unknown";
    }
}

enum CmpMode
{
    kCmpNone,
    kCmpRead,
    kCmpWrite,
};

extern CmpMode compare_mode;

template<typename T>
void Compare(T* ptr, size_t size, std::string key, CmpMode mode, cudaStream_t stream);

template<typename T>
void CheckNan(const T* ptr, size_t size, std::string key, cudaStream_t stream);

namespace detail {

template<typename T>
std::string to_string(T x)
{
    return std::to_string(x);
}

inline std::string to_string(std::string x)
{
    return x;
}

}  // namespace detail

template<typename... Args>
std::string Concat(std::string key, Args&&... args)
{
    std::vector<std::string> args_str{detail::to_string((Args &&) args)...};
    for (const auto& s : args_str) {
        key.append("_");
        key.append(s);
    }
    return key;
}

size_t curandStateGetSize();

bool isDebug();

struct NvtxScope {
    explicit NvtxScope(const std::string& name)
    {
        PUSH_RANGE(name.c_str());
    }

    ~NvtxScope()
    {
        POP_RANGE;
    }
};

int64_t& gSequenceIds(int batch_idx);

}  // namespace turbomind
