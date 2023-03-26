// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "cudakernel/nn/softmax.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/reduce/reduce.h"
#include "cudakernel/arithmetic/arithmetic.h"
#include "cudakernel/unary/exp.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "cudakernel/common/common.cuh"
#include "cudakernel/common/common.h"
#include "../reformat/cvt_int8_float.cuh"

template <typename T>
__device__ inline T __ldg_ver_ctrl(T* ptr) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

uint64_t PPLSoftmaxGetTempBufferSize(
    const ppl::common::TensorShape* input_shape,
    int axis)
{
    int N = input_shape->CalcElementsIncludingPadding() / input_shape->GetDim(axis);
    return N * ppl::common::GetSizeOfDataType(input_shape->GetDataType());
}

ppl::common::RetCode PPLCUDASoftmaxForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    int axis)
{
    int N = input_shape->CalcElementsToDimensionIncludingPadding(axis);
    int R = input_shape->GetDim(axis);
    int D = input_shape->CalcElementsFromDimensionIncludingPadding(axis + 1);
    // reduce max
    PPLReduceDimDes reduce_desc(D, R, N);
    ReduceParam reduce_max = ReduceMax;
    void* max_sum_output   = temp_buffer;
    ppl::common::TensorShape max_sum_shape(*input_shape);
    max_sum_shape.SetDimCount(3);
    max_sum_shape.SetDim(0, N);
    max_sum_shape.SetDim(1, 1);
    max_sum_shape.SetDim(2, D);

    auto status = PPLCUDAReduceForwardImp(stream, reduce_max, reduce_desc, input_shape, input, &max_sum_shape, max_sum_output);
    // sub
    ppl::common::TensorShape nd_shape(*input_shape);
    nd_shape.SetDimCount(3);
    nd_shape.SetDim(0, N);
    nd_shape.SetDim(1, R);
    nd_shape.SetDim(2, D);
    status = PPLCUDAArithMeticSubForwardImp(stream, &nd_shape, input, &max_sum_shape, max_sum_output, &nd_shape, output);
    // exp
    status                 = PPLCUDAExpForwardImp(stream, &nd_shape, output, &nd_shape, output);
    // reduce sum
    ReduceParam reduce_sum = ReduceSum;
    status = PPLCUDAReduceForwardImp(stream, reduce_sum, reduce_desc, &nd_shape, output, &max_sum_shape, max_sum_output);
    //div
    status = PPLCUDAArithMeticDivForwardImp(stream, &nd_shape, output, &max_sum_shape, max_sum_output, &nd_shape, output);
    return status;
}

__global__ void __launch_bounds__(256) ppl_cukernel_softmax_int8(
    const int8_t* input, int8_t* output, int max_int8,
    int outer, int axis_width, int inner,
    QuantKernelParamCuda qparam) {
    int tid = threadIdx.x;
    int inner_idx = blockIdx.x;
    int out_idx = blockIdx.y;
    __shared__ float shared[256];
    shared[tid] = 0.f;
    float max_val = _int82float(max_int8, qparam.i_step, qparam.i_zero_point);
    for(int id = tid; id < axis_width; id += blockDim.x) {
        if(id < axis_width) {
            uint64_t in_index = out_idx * axis_width * inner +
                id * inner + inner_idx;
            float in_val  = _int82float(input[in_index], qparam.i_step, qparam.i_zero_point);
            //calculate each c exp sum
            shared[tid] += expf(in_val - max_val);

        }
    }
    //accumulate all c exp sum
    __syncwarp();
    float exp_sum = BlockReduceSum(shared[tid]);

    for(int id = tid; id < axis_width; id += blockDim.x) {
        if(id < axis_width) {
            uint64_t in_index = out_idx * axis_width * inner +
                id * inner + inner_idx;
            //calculate output
            float in_val  = _int82float(input[in_index], qparam.i_step, qparam.i_zero_point);
            float out_val = expf(in_val - max_val) / exp_sum;
            output[in_index] = _float2int8(out_val, qparam.o_step, qparam.o_zero_point);
        }
    }
    __syncthreads();
}

ppl::common::RetCode PPLCUDASoftmaxForwardImpInt8(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    int axis,
    const QuantKernelParamCuda* qparam)
{
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    int outer = input_shape->CalcElementsToDimensionIncludingPadding(axis);
    int axis_width = input_shape->GetDim(axis);
    int inner = input_shape->CalcElementsFromDimensionIncludingPadding(axis + 1);
    // for int8 case, use 127 as the max_val
    int max_int8 = 127;
    int block_size = 256;
    dim3 grid_size(inner, outer, 1);
    ppl_cukernel_softmax_int8<<<grid_size, block_size, 0, stream>>>((const int8_t*)input,
            (int8_t*)output, max_int8, outer, axis_width, inner, *qparam);

    return status;
}


template <typename Tin, typename Tout>                                                              
__global__ void SoftmaxScoreKernel32Mask(                                                    
    const Tin* in, const bool* key_padding_mask, Tout* out,                                        
    const int mask_scale, const int T) {                                                        
        auto cur_in = in + blockIdx.x * T;                                                          
        auto cur_out = out + blockIdx.x * T;                                                        
        auto cur_mask = key_padding_mask + blockIdx.x / mask_scale * T;                                               
        float log_sum = CudaLogZero<float>();                                                  
        for (auto tid = threadIdx.x; tid < T; tid += blockDim.x) {                                  
            float maskv = (float) static_cast<float>(_Ldg(cur_mask + tid));  
            maskv = 0;                 
            log_sum =                                                                               
                    _LogAdd((float)__ldg_ver_ctrl(cur_in + tid) * ((float)1.0f - maskv) +              
                            CudaLogZero<float>() * maskv, log_sum);                              
        }                                                                                           
        log_sum = WarpReduceLogAddSum(log_sum);                                                     
        for (auto tid = threadIdx.x; tid < T; tid += blockDim.x) {                                  
            float maskv = (float) static_cast<float>(_Ldg(cur_mask + tid));   
            maskv = 0;                
            cur_out[tid] = (Tout)(_Exp((float)__ldg_ver_ctrl(cur_in + tid) - log_sum) *   
                            (float)(1.0f - maskv));                                             
        }                                                                                           
    }


template <typename Tin, typename Tout>                                                              
__global__ void SoftmaxScoreKernel64Mask(                                                    
    const Tin* in, const bool* key_padding_mask, Tout* out,                                        
    const int mask_scale, const int T) {                                                        
        auto cur_in = in + blockIdx.x * T;                                                          
        auto cur_out = out + blockIdx.x * T;                                                        
        auto cur_mask = key_padding_mask + blockIdx.x / mask_scale * T;                                                  
        __shared__ float sm[2];                                                                  
        float log_sum = CudaLogZero<float>();                                                 
        for (auto tid = threadIdx.x; tid < T; tid += blockDim.x) {                                  
            float maskv = (float) static_cast<float>(_Ldg(cur_mask + tid));
            maskv = 0; 
            log_sum =                                                                               
                    _LogAdd((float)__ldg_ver_ctrl(cur_in + tid) * ((float)1.0f - maskv) +              
                            CudaLogZero<float>() * maskv, log_sum);                              
        }                                                                                           
        auto lane_id = threadIdx.x & 0x1f;                                                          
        auto wid = threadIdx.x >> 5;                                                                
        log_sum = WarpReduceLogAddSum(log_sum);                                                     
        if(lane_id == 0) {                                                                          
            sm[wid] = log_sum;                                                                      
        }                                                                                           
        __syncthreads();                                                                            
        if (lane_id == 0) {                                                                         
            log_sum = _LogAdd(sm[0], sm[1]);                                                        
        }                                                                                           
        __syncthreads();                                                                            
        log_sum = WARP_SHFL(log_sum, 0);                                                            
        for (auto tid = threadIdx.x; tid < T; tid += blockDim.x) {                                  
            float maskv = (float) static_cast<float>(_Ldg(cur_mask + tid));  
            maskv = 0;                  
            cur_out[tid] = (Tout)(_Exp((float)__ldg_ver_ctrl(cur_in + tid) - log_sum) *                   
                            (float)(1.0f - maskv));                                              
        }                                                                                           
    }




template<typename Tin, typename Tout>
__global__ void SoftmaxScoreKernel32(const Tin* in, Tout* out, const int T) {
    auto cur_in = in + blockIdx.x * T;
    auto cur_out = out + blockIdx.x * T;
    // reduce log sum
    float log_sum = CudaLogZero<float>();
    for(auto tid = threadIdx.x; tid < T; tid += blockDim.x) {
        log_sum = _LogAdd((float)__ldg_ver_ctrl(cur_in + tid), log_sum);
    }
    log_sum = WarpReduceLogAddSum(log_sum);
    for(auto tid = threadIdx.x; tid < T; tid += blockDim.x) {
        cur_out[tid] = _Exp((float)__ldg_ver_ctrl(cur_in + tid) - log_sum);
    }
}

template<typename Tin, typename Tout>
__global__ void SoftmaxScoreKernel64(const Tin* in, Tout* out, const int T) {
    auto cur_in = in + blockIdx.x * T;
    auto cur_out = out + blockIdx.x * T;
    __shared__ float sm[2];
    // reduce log sum
    float log_sum = CudaLogZero<float>();
    for(auto tid = threadIdx.x; tid < T; tid += blockDim.x) {
        log_sum = _LogAdd((float)__ldg_ver_ctrl(cur_in + tid), log_sum);
    }
    auto lane_id = threadIdx.x & 0x1f;
    auto wid = threadIdx.x >> 5;
    log_sum = WarpReduceLogAddSum(log_sum);
    if(lane_id == 0) {
        sm[wid] = log_sum;
    }
    __syncthreads();
    if (lane_id == 0) {
        log_sum = _LogAdd(sm[0], sm[1]);
    }
    __syncthreads();
    log_sum = WARP_SHFL(log_sum, 0);
    for(auto tid = threadIdx.x; tid < T; tid += blockDim.x) {
        cur_out[tid] = _Exp((float)__ldg_ver_ctrl(cur_in + tid) - log_sum);
    }
}

/*
par:
    in & out : [BHTT]
    key_padding_mask : [B, H, T, T] or [B, 1, T, T] or [B, 1, 1, T], or [1, 1, T, T]
*/
template<typename Tin, typename Tout>
ppl::common::RetCode PPLCUDAFastSoftmaxForwardImp(
    cudaStream_t stream,
    const Tin* input,
    Tout* output,
    const bool* key_padding_mask,
    const int mask_scale,
    const int outer,
    const int inner)
{
    dim3 grid(outer, 1, 1);
    if (key_padding_mask != nullptr) {
        if(outer < 512) {
            dim3 block(32);
            SoftmaxScoreKernel32Mask<Tin, Tout>
                <<<grid, block, 0, stream>>>(input, key_padding_mask, output, mask_scale, inner);
        } else {
            dim3 block(64);
            SoftmaxScoreKernel64Mask<Tin, Tout>
                <<<grid, block, 0, stream>>>(input, key_padding_mask, output, mask_scale, inner);
        }
    } else {
        if (outer < 512) {
            dim3 block(32);
            SoftmaxScoreKernel32<Tin, Tout>
                <<<grid, block, 0, stream>>>(input, output, inner);
        } else {
            dim3 block(64);
            SoftmaxScoreKernel64<Tin, Tout>
                <<<grid, block, 0, stream>>>(input, output, inner);
        }
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAFastSoftmax(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const void* key_padding_mask) {
        int dim_cnt = input_shape->GetDimCount();
        int outer = 1;
        int inner = input_shape->GetDim(dim_cnt - 1);
        for(int i = 0; i < dim_cnt - 1; i++) {
            outer *= input_shape->GetDim(i);
        }
        int mask_scale = outer / input_shape->GetDim(0);
        if(output_shape->GetDataType() == 5)  {
            return PPLCUDAFastSoftmaxForwardImp<half, half>(stream, (const half*)input, (half*)output, (const bool*)key_padding_mask, mask_scale, outer, inner);
        } else if(output_shape->GetDataType() == 6) {
            return PPLCUDAFastSoftmaxForwardImp<float, float>(stream, (const float*)input, (float*)output, (const bool*)key_padding_mask, mask_scale, outer, inner);
        }
        return ppl::common::RC_UNSUPPORTED;
    }
