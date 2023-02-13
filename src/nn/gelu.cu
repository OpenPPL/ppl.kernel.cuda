#include "cudakernel/nn/log.h"
#include "ppl/common/tensor_shape.h"
#include <stdint.h>
#include <cuda_fp16.h>


__global__ void ppl_cukernel_gelu_fp16(const size_t count, const half *input, half *output) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count) return;
    float val = __half2float(input[index]);
    float out_val = val * 0.5 * (1 + erff(val * 0.707106781f));
    output[index] = __float2half_rn(out_val);
#endif
}


__global__ void ppl_cukernel_gelu_fp16_pack(const size_t count, const half2 *input, half2 *output) {
    #if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= count) return;
        half2 h_val = input[index];
        float2 f_val = __half22float2(h_val);

        half2 t_val;
        t_val.x = erff(f_val.x * 0.707106781f);
        t_val.y = erff(f_val.y * 0.707106781f);
        half2 one_constant = {1,1};
        half2 half_constant = {0.5,0.5};
        t_val = __hmul2(half_constant, __hmul2(h_val, __hadd2(one_constant, t_val)));

        output[index] = t_val;
    #endif
}


__global__ void ppl_cukernel_gelu_fp32(const size_t count, const float *input, float *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count) return;
    float val = input[index];
    float out_val = val * 0.5f * (1.0f + erff(val * 0.707106781f));
    output[index] = out_val;
}

__global__ void ppl_cukernel_gelu_fp32_pack(const size_t count, const float4 *input, float4 *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count) return;
    float4 val = input[index];
    float4 out_val;
    out_val.x = val.x * 0.5f * (1.0f + erff(val.x * 0.707106781f));
    out_val.y = val.y * 0.5f * (1.0f + erff(val.y * 0.707106781f));
    out_val.z = val.z * 0.5f * (1.0f + erff(val.z * 0.707106781f));
    out_val.w = val.w * 0.5f * (1.0f + erff(val.w * 0.707106781f));

    output[index] = out_val;
}

ppl::common::RetCode PPLCUDAGeluForwardImp(
    cudaStream_t stream,
    const void* input,
    ppl::common::TensorShape* input_shape,
    void* output){

        int BS = 256;
        uint64_t elemCount = input_shape->CalcElementsIncludingPadding();
        uint64_t GS = (elemCount + BS - 1) / BS;
        if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            if (elemCount % 4 == 0) {
                GS = ((elemCount >> 2) + BS - 1) / BS;
                ppl_cukernel_gelu_fp32_pack<<<GS, BS, 0, stream>>>(
                    elemCount >> 2, (const float4*)input, (float4*)output);
            } else {
            ppl_cukernel_gelu_fp32<<<GS, BS, 0, stream>>>(
                elemCount, (const float*)input, (float*)output);
            }
        
        } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            if (elemCount % 2 == 0) {
                GS = ((elemCount >> 1) + BS - 1) / BS;
                ppl_cukernel_gelu_fp16_pack<<<GS, BS, 0, stream>>>(
                    elemCount >> 1, (const half2*)input, (half2*)output);                
            } else {
            ppl_cukernel_gelu_fp16<<<GS, BS, 0, stream>>>(
                elemCount, (const half*)input, (half*)output);
            }

        } else {
            return ppl::common::RC_UNSUPPORTED;  //TODO
        }

    return ppl::common::RC_SUCCESS;
}
