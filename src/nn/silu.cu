#include "cudakernel/nn/silu.h"
#include "ppl/common/tensor_shape.h"
#include "cudakernel/common/cuda_check.h"
#include <cuda_fp16.h>

__global__ void ppl_cukernel_silu_fp16(const size_t count, const half *input, half *output) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count) return;
    float val = __half2float(input[index]);
    float out_val = val / (1 + __expf(-val));
    output[index] = __float2half_rn(out_val);
#endif
}


__global__ void ppl_cukernel_silu_fp16_pack(const size_t count, const half2 *input, half2 *output) {
    #if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= count) return;
        half2 h_val = input[index];
        float2 f_val = __half22float2(h_val);

        half2 t_val;
        t_val.x = __float2half_rn(f_val.x / (1 + __expf(-f_val.x)));
        t_val.y = __float2half_rn(f_val.y / (1 + __expf(-f_val.y)));
        output[index] = t_val;
    #endif
}


ppl::common::RetCode PPLCUDASiluForwardImp(
    cudaStream_t stream,
    const void* input,
    ppl::common::TensorShape* input_shape,
    void* output) 
{
    PPL_CHECK(input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16, "Silu only support fp16");
    int BS = 256;
    uint64_t elemCount = input_shape->CalcElementsIncludingPadding();
    uint64_t GS = (elemCount + BS - 1) / BS;
    if (elemCount % 2 == 0) {
        GS = ((elemCount >> 1) + BS - 1) / BS;
        ppl_cukernel_silu_fp16_pack<<<GS, BS, 0, stream>>>(
            elemCount >> 1, (const half2*)input, (half2*)output);                
    } else {
        ppl_cukernel_silu_fp16<<<GS, BS, 0, stream>>>(
            elemCount, (const half*)input, (half*)output);
    }
    return ppl::common::RC_SUCCESS;
}