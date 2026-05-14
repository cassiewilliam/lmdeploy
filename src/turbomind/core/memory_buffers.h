// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"

namespace turbomind::core {

// Manages and reuses CUDA memory buffers to keep device pointers stable
// across capture/replay/eager dispatches.  Mirrors TensorRT-LLM's
// `Buffers` class (memory_buffer_utils.py:23-123): each named buffer is
// stored as a uint8 storage block; `get_buffer` returns the smallest
// existing block that fits the requested (shape, dtype), or allocates a
// new one and adds it to the pool.  Repeated calls with the same name
// and same-or-smaller required bytes return a Tensor view of the SAME
// underlying storage — so its `.raw_data()` is stable.
//
// This is the foundation of CG correctness: the captured graph bakes
// the buffer's device pointer into kernel arg buffers; if a later call
// reallocated the storage (different address), replay would silently
// read garbage.  Reusing a stable storage block by name fixes that.
//
// Thread-safety: the pool is mutex-guarded; concurrent get_buffer
// calls from different worker threads are serialised.
class MemoryBuffers {
public:
    MemoryBuffers() = default;

    // Returns a Tensor view backed by an internally-pooled uint8 buffer.
    //  - shape   : desired tensor shape
    //  - dtype   : desired element dtype (Tensor view will reinterpret
    //              the underlying uint8 storage)
    //  - name    : pool key.  Repeated calls with the same name try to
    //              reuse an existing block.  Distinct names always get
    //              distinct storage.
    //  - reserve : if true, mark the chosen block as "reserved" so
    //              subsequent get_buffer calls under the same name pick
    //              a different (new or unreserved) block.  Use this for
    //              persistent allocations that must outlive the call
    //              site (e.g. graph-baked tensors).
    Tensor get_buffer(const std::vector<ssize_t>& shape,
                      DataType                    dtype,
                      const std::string&          name,
                      bool                        reserve = false);

    // Free all unreserved blocks (does NOT touch reserved ones).
    void clear_unreserved();

    // Diagnostic: total bytes currently held across all blocks.
    size_t total_bytes() const;

private:
    struct Block {
        // Underlying storage as a raw uint8 buffer.  Its lifetime is
        // tied to this Block via shared ownership.
        Buffer storage;
        bool   reserved{false};
    };

    Tensor view_as(const Buffer&               storage,
                   const std::vector<ssize_t>& shape,
                   DataType                    dtype) const;

    mutable std::mutex                                  mu_;
    std::unordered_map<std::string, std::vector<Block>> pool_;
};

// Process-wide singleton (one logical pool per device).  Use this from
// any layer / kernel wrapper that wants stable device addresses.
MemoryBuffers& global_memory_buffers();

}  // namespace turbomind::core
