#ifndef PPLCUDA_KERNEL_INCLUDE_GELU_H_
#define PPLCUDA_KERNEL_INCLUDE_GELU_H_
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDAGeluForwardImp(
    cudaStream_t stream,
    const void* input,
    ppl::common::TensorShape* input_shape,
    void* output);
#endif // #define PPLCUDA_KERNEL_INCLUDE_GELU_H_