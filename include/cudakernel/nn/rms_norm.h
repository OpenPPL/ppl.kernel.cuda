#ifndef PPLCUDA_KERNEL_INCLUDE_RMSNORM_H_
#define PPLCUDA_KERNEL_INCLUDE_RMSNORM_H_
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDARmsNormForwardImp(
    cudaStream_t stream,
    const void* input,
    const void* skip,
    const void* weight,
    const float eps,
    ppl::common::TensorShape* input_shape,
    void* output1,
    void* output2);
#endif // #define PPLCUDA_KERNEL_INCLUDE_PRELU_H_
