#ifndef PPLCUDA_KERNEL_INCLUDE_PPLLOG_H_
#define PPLCUDA_KERNEL_INCLUDE_PPLLOG_H_
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDAPPLLogForwardImp(
    cudaStream_t stream,
    const void* input,
    ppl::common::TensorShape* input_shape,
    float base,
    float scale,
    float shift,
    void* output);
#endif // #define PPLCUDA_KERNEL_INCLUDE_PPLLOG_H_