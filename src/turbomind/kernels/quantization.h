#include "src/turbomind/core/core.h"

namespace turbomind {

void QuantizeSymm(Tensor& out, Tensor& scale, const Tensor& src, cudaStream_t st);

void DequantizeSymm(Tensor& out, const Tensor& src, const Tensor& scale, cudaStream_t st);

void QuantizeSymmBlock(Ref<Tensor> out_, Ref<Tensor> scale_, const Tensor& src, cudaStream_t st);

void DequantizeSymmBlock(Ref<Tensor> out_, Ref<Tensor> src_, const Tensor& scale, cudaStream_t st);

void QuantizeGroupwise(Tensor            quant,    // (m,k)
                       Tensor            scales,   // (m,k/g)
                       Tensor            zeros,    // (m,k/g)
                       Tensor            dequant,  // (m,k)
                       Tensor            src,      // (m,k)
                       Buffer_<unsigned> rbits,    // (m*k)
                       int               group_size);

// Quantize BF16 input to FP8 e4m3 using a pre-determined (static) per-tensor scale.
// out.dtype() must be kFloat8_e4m3; scale is a device scalar Tensor of dtype kFloat.
void QuantizeStatic(Tensor& out, const Tensor& in, const Tensor& scale, cudaStream_t st);

// Quantize BF16 activations to NVFP4 (e2m1) with per-group FP8 e4m3 block scales.
// group_size is fixed at 16 (kNvfp4GroupSize).
// out:       [num_tokens, hidden_dim/2]  kUint8       (2 FP4 nibbles per byte, lo=even col)
// scale_out: [num_tokens, hidden_dim/16] kFloat8_e4m3 (per-group scale)
// in:        [num_tokens, hidden_dim]    kBfloat16
// global_scale: second-level (tensor-wide) scale applied before quantization
void invokeQuantizeTrtllmFp4MoeActivation(Tensor&      out,
                                          Tensor&      scale_out,
                                          const Tensor& in,
                                          float        global_scale,
                                          cudaStream_t st);

}  // namespace turbomind
