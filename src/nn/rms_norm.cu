#include "cudakernel/nn/rms_norm.h"
#include "ppl/common/tensor_shape.h"
#include "cudakernel/common/cuda_check.h"
#include "cudakernel/common/common.cuh"
#include <cuda_fp16.h>

/**
 * RMSNorm Cuda impl template.
 *
 * @param VPT: Value processed per thread.
 * @param TPB: Thread per block.

 * @param x data pointer of input.
 * @param weight parameter of this RmsNorm.
 * @param eps
 * @param normalize_shape num of elements within last dimension of input.
 * @param o1 data pointer of output.
 * @param o2 input 
 */
template <int VPT, int TPB>
__global__
void _RmsNormForward_fp16(
  const half *x,
  const half *weight,
  const float eps,
  const int32_t normalize_shape,
  half *o1,
  half *o2
){
  const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
  half inLocal[VPT]; half weightLocal[VPT];

  copy<sizeof(half) * VPT>(&x[idx], inLocal);
  float accumulator = 0.0f; // accumulator
  float r_normalize_shape = 1.0f / (float)(normalize_shape);

#pragma unroll
  for (int32_t it = 0; it < VPT; it++)
    accumulator = accumulator + (__half2float(inLocal[it]) * __half2float(inLocal[it]));
  copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);

  #if (__CUDACC_VER_MAJOR__ >= 11)
      const float reduced = BlockAllReduce<SumOp, float, TPB>(accumulator) * r_normalize_shape;
  #else
      const float reduced = blockReduceSum<float>(accumulator) * r_normalize_shape;
  #endif
  __shared__ float r_reduced;

  if (threadIdx.x == 0)
    r_reduced = rsqrt(reduced + eps);
  __syncthreads();

  half outLocal[VPT];
#pragma unroll
  for (int32_t it = 0; it < VPT; it++)
    outLocal[it] = __float2half(__half2float(inLocal[it]) * r_reduced) * weightLocal[it];
  copy<sizeof(half) * VPT>(outLocal, &o1[idx]);
  copy<sizeof(half) * VPT>(inLocal, &o2[idx]);
};


template <int TPB>
__global__
void _RmsNormForward_fp16_default(
  const half *x,
  const half *weight,
  const float eps,
  const int32_t normalize_shape,
  half *o1,
  half *o2
){
  auto cur_x = x + normalize_shape * blockIdx.x;
  auto cur_o1 = o1 + normalize_shape * blockIdx.x;
  auto cur_o2 = o2 + normalize_shape * blockIdx.x;

  float accumulator = 0.0f; // accumulator
  float r_normalize_shape = 1.0f / (float)(normalize_shape);

  for(int idx = threadIdx.x; idx < normalize_shape; idx += TPB) {
    cur_o2[idx] = cur_x[idx];
    accumulator = accumulator + (__half2float(cur_x[idx]) * __half2float(cur_x[idx]));
  }

  #if (__CUDACC_VER_MAJOR__ >= 11)
      const float reduced = BlockAllReduce<SumOp, float, TPB>(accumulator) * r_normalize_shape;
  #else
      const float reduced = blockReduceSum<float>(accumulator) * r_normalize_shape;
  #endif
  __shared__ float r_reduced;

  if (threadIdx.x == 0)
    r_reduced = rsqrt(reduced + eps);
  __syncthreads();

  for(int idx = threadIdx.x; idx < normalize_shape; idx += TPB) {
    cur_o1[idx] = __float2half(__half2float(cur_x[idx]) * r_reduced) * weight[idx];
  }
};


/**
 * RMSNorm Cuda impl template(with skip connection).
 *
 * @param VPT: Value processed per thread.
 * @param TPB: Thread per block.

 * @param x data pointer of input.
 * @param weight parameter of this RmsNorm.
 * @param skip skip connection of this RmsNorm.
 * @param eps
 * @param normalize_shape num of elements within last dimension of input.
 * @param output data pointer of output.
 * input and output should share a same size.
 */
 template <int VPT, int TPB>
__global__
void _SkipRmsNormForward_fp16(
  const half *x,
  const half *weight,
  const half *skip,
  const float eps,
  const int32_t normalize_shape,
  half *o1,
  half *o2
){
  const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
  half inLocal[VPT]; half weightLocal[VPT];
  float inLocal_fp32[VPT];

  copy<sizeof(half) * VPT>(&x[idx], inLocal);
  copy<sizeof(half) * VPT>(&skip[idx], weightLocal);
  float accumulator = 0.0f; // accumulator
  float r_normalize_shape = 1.0f / (float)(normalize_shape);

// step 1. compute x + skip
#pragma unroll
  for (int32_t it = 0; it < VPT; it++) 
    inLocal[it] = inLocal[it] + weightLocal[it];

#pragma unroll
  for (int32_t it = 0; it < VPT; it++) 
    inLocal_fp32[it] = __half2float(inLocal[it]);

  copy<sizeof(half) * VPT>(inLocal, &o2[idx]);

#pragma unroll
  for (int32_t it = 0; it < VPT; it++)
    accumulator = accumulator + (inLocal_fp32[it] * inLocal_fp32[it]);
  copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);

  #if (__CUDACC_VER_MAJOR__ >= 11)
      const float reduced = BlockAllReduce<SumOp, float, TPB>(accumulator) * r_normalize_shape;
  #else
      const float reduced = blockReduceSum<float>(accumulator) * r_normalize_shape;
  #endif
  __shared__ float r_reduced;

  if (threadIdx.x == 0)
    r_reduced = rsqrt(reduced + eps);
  __syncthreads();

  half outLocal[VPT];
#pragma unroll
  for (int32_t it = 0; it < VPT; it++) 
    outLocal[it] = __float2half(inLocal_fp32[it] * r_reduced) * weightLocal[it];
  
  copy<sizeof(half) * VPT>(outLocal, &o1[idx]);
};



 template <int TPB>
__global__
void _SkipRmsNormForward_fp16_default(
  const half *x,
  const half *weight,
  const half *skip,
  const float eps,
  const int32_t normalize_shape,
  half *o1,
  half *o2
){
  auto cur_x = x + normalize_shape * blockIdx.x;
  auto cur_skip = skip + normalize_shape * blockIdx.x;
  auto cur_o1 = o1 + normalize_shape * blockIdx.x;
  auto cur_o2 = o2 + normalize_shape * blockIdx.x;

  float accumulator = 0.0f; // accumulator
  float r_normalize_shape = 1.0f / (float)(normalize_shape);

// step 1. compute x + skip

  for(int idx = threadIdx.x; idx < normalize_shape; idx += TPB) {
    half temp = cur_x[idx] + cur_skip[idx];
    cur_o2[idx] = temp;
    accumulator = accumulator + (__half2float(temp) * __half2float(temp));
  }

  #if (__CUDACC_VER_MAJOR__ >= 11)
      const float reduced = BlockAllReduce<SumOp, float, TPB>(accumulator) * r_normalize_shape;
  #else
      const float reduced = blockReduceSum<float>(accumulator) * r_normalize_shape;
  #endif
  __shared__ float r_reduced;

  if (threadIdx.x == 0)
    r_reduced = rsqrt(reduced + eps);
  __syncthreads();

  for(int idx = threadIdx.x; idx < normalize_shape; idx += TPB) {
    float temp = __half2float(cur_x[idx] + cur_skip[idx]);
    cur_o1[idx] = __float2half(temp * r_reduced) * weight[idx];
  }
};



ppl::common::RetCode PPLCUDARmsNormForwardImp(
    cudaStream_t stream,
    const void* input,
    const void* skip,
    const void* weight,
    const float eps,
    ppl::common::TensorShape* input_shape,
    void* output1,
    void* output2)
{
    if(input_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
      printf("RmsNorm only support fp16, but got %d \n", input_shape->GetDataType());
    }
    constexpr int32_t VPT = 16 / sizeof(half);

    const int32_t normalize_shape = input_shape->GetDim(input_shape->GetDimCount() - 1);
    const int32_t grid_size = input_shape->CalcElementsIncludingPadding() / normalize_shape;
    
    if (skip == nullptr) {
      switch (normalize_shape)
      {
      case 768:
        _RmsNormForward_fp16<VPT, 768 / VPT>
        <<<grid_size, 768 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          eps, normalize_shape,
          (half*)(output1),
          (half*)(output2));
        break;
      case 1024:
        _RmsNormForward_fp16<VPT, 1024 / VPT>
        <<<grid_size, 1024 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          eps, normalize_shape, 
          (half*)(output1),
          (half*)(output2));
        break;
      case 4096:
        _RmsNormForward_fp16<VPT, 4096 / VPT>
        <<<grid_size, 4096 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          eps, normalize_shape, 
          (half*)(output1),
          (half*)(output2));
        break;
      case 5120:
        _RmsNormForward_fp16<VPT, 5120 / VPT>
        <<<grid_size, 5120 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          eps, normalize_shape, 
          (half*)(output1),
          (half*)(output2));
        break;
      case 8192:
        _RmsNormForward_fp16<VPT, 8192 / VPT>
        <<<grid_size, 8192 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          eps, normalize_shape, 
          (half*)(output1),
          (half*)(output2));
        break;
      default:
        _RmsNormForward_fp16_default<512>
        <<<grid_size, 512, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          eps, normalize_shape, 
          (half*)(output1),
          (half*)(output2));
      };
    } else {
      switch (normalize_shape)
      {
      case 768:
        _SkipRmsNormForward_fp16<VPT, 768 / VPT>
        <<<grid_size, 768 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          (half*)(skip), 
          eps, normalize_shape, 
          (half*)(output1),
          (half*)(output2));
        break;
      case 1024:
      _SkipRmsNormForward_fp16<VPT, 1024 / VPT>
        <<<grid_size, 1024 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          (half*)(skip), 
          eps, normalize_shape, 
          (half*)(output1),
          (half*)(output2));
        break;
      case 4096:
      _SkipRmsNormForward_fp16<VPT, 4096 / VPT>
        <<<grid_size, 4096 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          (half*)(skip), 
          eps, normalize_shape, 
          (half*)(output1),
          (half*)(output2));
        break;
      case 5120:
      _SkipRmsNormForward_fp16<VPT, 5120 / VPT>
        <<<grid_size, 5120 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          (half*)(skip), 
          eps, normalize_shape, 
          (half*)(output1),
          (half*)(output2));
        break;
      case 8192:
        _SkipRmsNormForward_fp16<VPT, 8192 / VPT>
          <<<grid_size, 8192 / VPT, 0, stream>>>(
            (half*)(input), 
            (half*)(weight), 
            (half*)(skip), 
            eps, normalize_shape, 
            (half*)(output1),
            (half*)(output2));
        break;
      default:
        _SkipRmsNormForward_fp16_default<512>
          <<<grid_size, 512, 0, stream>>>(
            (half*)(input), 
            (half*)(weight), 
            (half*)(skip), 
            eps, normalize_shape, 
            (half*)(output1),
            (half*)(output2));
      };
    }
    
    return ppl::common::RC_SUCCESS;
    


}
