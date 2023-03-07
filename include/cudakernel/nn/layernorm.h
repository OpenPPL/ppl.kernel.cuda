#ifndef PPLCUDA_KERNEL_INCLUDE_LAYERNORM_H_
#define PPLCUDA_KERNEL_INCLUDE_LAYERNORM_H_
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDALayerNormForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const void* input,
    const void* scale,
    const void* shift,
    void* output,
    int outer,
    int inner,
    bool elementwise_affine,
    float epsilon,
    float in_scale,
    float out_scale);
#endif // #define PPLCUDA_KERNEL_INCLUDE_LAYERNORM_H_
