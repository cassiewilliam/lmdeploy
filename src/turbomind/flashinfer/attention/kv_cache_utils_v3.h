// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/attention/attention_params.h"

namespace turbomind {

// TRT-LLM GEN Attention paged-KV writers with RoPE / QK-norm fusion.
// HND layout: [Page Size, Block Length, Num Heads, Head Dim].

template<class T>
void invokeDecodingQKNormRopeKVUpdate_(const AttentionParams<T>& params);

// Append writes RoPE'd Q to `params.q` and K/V to the paged pool in one
// fused launch driven by `token2batch`.
template<class T>
void invokeAppendQKNormRopeKVUpdate_(const AttentionParams<T>& params);

}  // namespace turbomind
