#include "cudakernel/nn/log.h"
#include "ppl/common/tensor_shape.h"
#include <stdint.h>
#include <cuda_fp16.h>

__global__ void ppl_cukernel_log_fp32(const size_t count, float base, float scale, float shift,
            const float *input, float *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count) return;
    float val = shift + scale * input[index];
    if(base > 0)
        output[index] = logf(val) / logf(base);
    else
        output[index] = logf(val);
}

__global__ void ppl_cukernel_log_fp16(const size_t count, float base, float scale, float shift,
            const half *input, half *output) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count) return;
    float val = shift + scale * __half2float(input[index]);
    float out_val = 0.f;
    if(base > 0)
        out_val = logf(val) / logf(base);
    else
        out_val = logf(val);
    output[index] = __float2half_rn(out_val);
#endif
}


ppl::common::RetCode PPLCUDAPPLLogForwardImp(
    cudaStream_t stream,
    const void* input,
    ppl::common::TensorShape* input_shape,
    float base,
    float scale,
    float shift,
    void* output){

        int BS = 256;
        uint64_t elemCount = input_shape->CalcElementsIncludingPadding();
        uint64_t GS = (elemCount + BS - 1) / BS;
        if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            ppl_cukernel_log_fp32<<<GS, BS, 0, stream>>>(
                elemCount, base, scale, shift, (const float*)input, (float*)output);
        
        } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            ppl_cukernel_log_fp16<<<GS, BS, 0, stream>>>(
                elemCount, base, scale, shift, (const half*)input, (half*)output);

        } else {
            return ppl::common::RC_UNSUPPORTED;  //TODO
        }

    return ppl::common::RC_SUCCESS;
}
