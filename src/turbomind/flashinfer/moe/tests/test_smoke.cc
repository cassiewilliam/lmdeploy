// Copyright (c) OpenMMLab. All rights reserved.
//
// Smoke test for the FlashInfer MoE bridge: loads the .so that the
// configure-time `build_moe_so.py` script built (or the env-var
// override), verifies the FFI entries resolve, and exits 0.  Doesn't run
// a real kernel — that requires valid weight tensors which only the
// in-engine MoE layer constructs.

#include "src/turbomind/flashinfer/moe/trtllm_fused_moe_wrapper.h"

#include <cstdlib>
#include <iostream>

int main() {
    if (!turbomind::trtllm_fused_moe::is_available()) {
        std::cerr << "[smoke] flashinfer-moe failed to load.\n"
                  << "  Override path with: LMDEPLOY_FLASHINFER_MOE_SO=<path/to/.so>\n"
                  << "  Cubin loading is owned by flashinfer-python (its own JIT cache\n"
                  << "  under ~/.cache/flashinfer/...); no lmdeploy-side override here.\n";
        return EXIT_FAILURE;
    }
    if (!turbomind::trtllm_fused_moe::is_sm10x_capable()) {
        std::cerr << "[smoke] WARN: device is not SM10x; .so is loaded but kernels won't run.\n";
    }
    std::cout << "[smoke] flashinfer-moe loaded OK; FFI entries resolved.\n";
    return EXIT_SUCCESS;
}
