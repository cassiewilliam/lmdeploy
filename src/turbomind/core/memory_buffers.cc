// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/memory_buffers.h"

#include <algorithm>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/common.h"
#include "src/turbomind/core/context.h"

namespace turbomind::core {

namespace {

// 256-byte alignment matches cuBLAS / FlashInfer device-pointer
// alignment expectations.
constexpr ssize_t kAlignBytes = 256;

inline ssize_t round_up(ssize_t n, ssize_t a)
{
    return ((n + a - 1) / a) * a;
}

}  // namespace

Tensor MemoryBuffers::view_as(const Buffer&               storage,
                              const std::vector<ssize_t>& shape,
                              DataType                    dtype) const
{
    // Slice the byte storage to exactly the required size, then
    // reinterpret as the target dtype with the requested shape.
    ssize_t n = 1;
    for (auto d : shape) {
        n *= d;
    }
    const ssize_t need_bytes = turbomind::byte_size(dtype, n);
    TM_CHECK_GE((ssize_t)storage.byte_size(), need_bytes)
        << "MemoryBuffers internal: storage too small for view";
    auto bytes_view = storage.slice(0, need_bytes);  // uint8 view
    auto typed_view = bytes_view.view(dtype);        // reinterpret bytes as dtype
    return Tensor(typed_view, Layout(shape));
}

Tensor MemoryBuffers::get_buffer(const std::vector<ssize_t>& shape,
                                 DataType                    dtype,
                                 const std::string&          name,
                                 bool                        reserve)
{
    ssize_t n = 1;
    for (auto d : shape) {
        n *= d;
    }
    const ssize_t need_bytes = round_up(turbomind::byte_size(dtype, n), kAlignBytes);

    std::lock_guard<std::mutex> g(mu_);
    auto&                       blocks = pool_[name];

    // Best-fit search: smallest block that's still large enough.
    // Prefer non-reserved blocks (so reserved persistent slots aren't
    // contended).
    Block*  best      = nullptr;
    ssize_t best_size = 0;
    for (auto& blk : blocks) {
        const ssize_t bsz = (ssize_t)blk.storage.byte_size();
        if (bsz < need_bytes) {
            continue;
        }
        // Skip reserved blocks unless we're explicitly reserving (then
        // reuse same-name reserved is allowed for the FIRST call; later
        // calls fall through and may allocate fresh).
        if (blk.reserved && !reserve) {
            continue;
        }
        if (best == nullptr || bsz < best_size) {
            best      = &blk;
            best_size = bsz;
        }
    }

    if (best != nullptr) {
        if (reserve) {
            best->reserved = true;
        }
        return view_as(best->storage, shape, dtype);
    }

    // No fit — allocate a new uint8 byte buffer at the rounded size.
    Buffer storage(need_bytes, kUint8, kDEVICE);
    blocks.push_back(Block{storage, reserve});
    return view_as(blocks.back().storage, shape, dtype);
}

void MemoryBuffers::clear_unreserved()
{
    std::lock_guard<std::mutex> g(mu_);
    for (auto& [name, blocks] : pool_) {
        blocks.erase(std::remove_if(blocks.begin(),
                                    blocks.end(),
                                    [](const Block& b) { return !b.reserved; }),
                     blocks.end());
    }
}

size_t MemoryBuffers::total_bytes() const
{
    std::lock_guard<std::mutex> g(mu_);
    size_t                      total = 0;
    for (const auto& [name, blocks] : pool_) {
        for (const auto& b : blocks) {
            total += (size_t)b.storage.byte_size();
        }
    }
    return total;
}

MemoryBuffers& global_memory_buffers()
{
    static MemoryBuffers inst;
    return inst;
}

}  // namespace turbomind::core
