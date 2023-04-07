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
#include "ppl/common/log.h"

#define _HLAF_MIN -65504
#define _FLT_MIN  -3.40282346638528859811704183484516925e+38F

template <typename T>
T get_min();

template <>
inline __host__ __device__ float get_min<float>()
{
    return _FLT_MIN;
}
template <>
inline __host__ __device__ half get_min<half>()
{
    return _HLAF_MIN;
}

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


template <typename T, typename acc_t, int log2_ceil>
__global__ void SoftmaxWarpImpl(const T* X, T* Y, int o_dim, int i_dim) {
    constexpr int next_power_of_two = 1 << log2_ceil;
    constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    int local_batches = o_dim - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    int tid = threadIdx.x;

    uint offset = first_batch * i_dim + tid;
    X += offset;
    Y += offset;

    acc_t x[WARP_BATCH][WARP_ITERATIONS];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : i_dim;
        #pragma unroll
        for (int j = 0;  j < WARP_ITERATIONS;  ++j) {
            int element_index = tid + j * WARP_SIZE;
            if (element_index < batch_element_count) {
                x[i][j] = X[i * i_dim + j * WARP_SIZE];
            } else {
                x[i][j] = get_min<acc_t>();
            }
        }
    }

    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        max_value[i] = x[i][0];
        #pragma unroll
        for (int j = 1;  j < WARP_ITERATIONS;  ++j) {
            max_value[i] = (max_value[i] > x[i][j]) ? max_value[i] : x[i][j];
        }
    }
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    #pragma unroll
        for (int i = 0; i < WARP_BATCH; ++i) {
            acc_t b = __shfl_xor_sync(0xffffffff, max_value[i], offset);
            max_value[i] = (max_value[i] > b) ? max_value[i] : b;
        }
    }

    acc_t sum[WARP_BATCH] { 0.0f };
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        #pragma unroll
        for (int j = 0;  j < WARP_ITERATIONS;  ++j) {
            x[i][j] = std::exp(x[i][j] - max_value[i]);
            sum[i] += x[i][j];
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    #pragma unroll
        for (int i = 0; i < WARP_BATCH; ++i) {
            sum[i] += __shfl_xor_sync(0xffffffff, sum[i], offset);
        }
    }
    

    #pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        sum[i] = 1.f / sum[i];
    }

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int j = 0;  j < WARP_ITERATIONS;  ++j) {
            int element_index = tid + j * WARP_SIZE;
            if (element_index < i_dim) {
                Y[i*i_dim+j*WARP_SIZE] =  x[i][j] * sum[i];
            } else {
                break;
            }
        }
    }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void SoftmaxBlockSMemImpl(LOAD load, STORE store, const int64_t rows,
                                     const int64_t cols) {

  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  __shared__ ComputeType row_sum_r;
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_max = -80.0f;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        thread_max = max(thread_max, pack[i]);
      }
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
    #else
        const ComputeType row_max = blockReduceMax<ComputeType>(thread_max);
    #endif
    ComputeType thread_sum = 0;
    for (int col = tid; col < cols; col += block_size) {
        const ComputeType exp_x = std::exp(buf[col] - row_max);
        buf[col] = exp_x;
        thread_sum += exp_x;
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    #else
        const ComputeType row_sum = blockReduceSum<ComputeType>(thread_sum);
    #endif
    if(threadIdx.x == 0) row_sum_r = 1.f / row_sum;
    __syncthreads();

    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
          pack[i] = buf[i * num_packs + pack_id] * row_sum_r;
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void SoftmaxBlockUncachedImpl(LOAD load, STORE store, const int64_t rows,
                                         const int64_t cols) {
  const int tid = threadIdx.x;
  const int num_packs = cols / pack_size;
  __shared__ ComputeType row_sum_r;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_max = -80.0f;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_max = max(thread_max, pack[i]); }
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
    #else
        const ComputeType row_max = blockReduceMax<ComputeType>(thread_max);
    #endif
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_sum += std::exp(pack[i] - row_max); }
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    #else
        const ComputeType row_sum = blockReduceSum<ComputeType>(thread_sum);
    #endif
    if(threadIdx.x == 0) row_sum_r = 1.f / row_sum;
    __syncthreads();
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        pack[i] = std::exp(pack[i] - row_max) * row_sum_r;
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

/*
par:
    in & out : [BHTT]
    key_padding_mask : [B, H, T, T] or [B, 1, T, T] or [B, 1, 1, T], or [1, 1, T, T]
*/
template<typename T, int pack_size>
ppl::common::RetCode PPLCUDAFastSoftmaxForwardImp(
    cudaStream_t stream,
    const T* input,
    T* output,
    const bool* key_padding_mask,
    const int mask_scale,
    const int outer_dim,
    const int inner_dim)
{
    using ComputeType = typename DefaultComputeType<T>::type;
    DirectLoad<T, ComputeType> load(input, inner_dim);
    DirectStore<ComputeType, T> store(output, inner_dim);

    if(inner_dim <= 1024)
	{
        int log2_ceil = 0;
        while((1 << log2_ceil) < inner_dim) log2_ceil++;
        const int next_power_of_two = 1 << log2_ceil;


        int warp_size = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;

        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int gridSize = (outer_dim + batches_per_block - 1) / batches_per_block;
		dim3 blockSize(warp_size, warps_per_block, 1);

        switch(log2_ceil) {
            case 0:
                SoftmaxWarpImpl<T,ComputeType,0><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 1:
                SoftmaxWarpImpl<T,ComputeType,1><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 2:
                SoftmaxWarpImpl<T,ComputeType,2><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 3:
                SoftmaxWarpImpl<T,ComputeType,3><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 4:
                SoftmaxWarpImpl<T,ComputeType,4><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 5:
                SoftmaxWarpImpl<T,ComputeType,5><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 6:
                SoftmaxWarpImpl<T,ComputeType,6><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 7:
                SoftmaxWarpImpl<T,ComputeType,7><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 8:
                SoftmaxWarpImpl<T,ComputeType,8><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 9:
                SoftmaxWarpImpl<T,ComputeType,9><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            case 10:
                SoftmaxWarpImpl<T,ComputeType,10><<<gridSize, blockSize, 0, stream>>>(
                        input, output, outer_dim, inner_dim);
                break;
            default:
                break;
        }
		
	} else {
        int grid = outer_dim;
        constexpr int block_size_conf = 256;
        const size_t smem = inner_dim * sizeof(ComputeType);
        int max_active_blocks_conf_1;
        {
            cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_active_blocks_conf_1,
                SoftmaxBlockSMemImpl<decltype(load), decltype(store), ComputeType, pack_size, block_size_conf>,
                block_size_conf, smem);
            if (err != cudaSuccess) { LOG(ERROR) << "cudaOccupancyMaxActiveBlocksPerMultiprocessor error"; }
        }
        if (max_active_blocks_conf_1 <= 0) {
            SoftmaxBlockUncachedImpl<decltype(load), decltype(store), ComputeType, pack_size, 1024><<<grid, 1024, 0, stream>>>(load, store, outer_dim, inner_dim);
        }

        SoftmaxBlockSMemImpl<decltype(load),decltype(store),ComputeType,pack_size,block_size_conf><<<grid, block_size_conf, smem, stream>>>(load, store, outer_dim, inner_dim);
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAFastSoftmax(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const void* key_padding_mask) 
{
    int dim_cnt = input_shape->GetDimCount();
    int outer_dim = 1;
    int inner_dim = input_shape->GetDim(dim_cnt - 1);
    for(int i = 0; i < dim_cnt - 1; i++) {
        outer_dim *= input_shape->GetDim(i);
    }
    int mask_scale = outer_dim / input_shape->GetDim(0);
    switch(output_shape->GetDataType()) {
        case ppl::common::DATATYPE_FLOAT32: {
            if (inner_dim % 2 == 0)
                PPLCUDAFastSoftmaxForwardImp<float, 2>(stream, (const float*)input, (float*)output, (const bool*)key_padding_mask, mask_scale, outer_dim, inner_dim);
            else
                PPLCUDAFastSoftmaxForwardImp<float, 1>(stream, (const float*)input, (float*)output, (const bool*)key_padding_mask, mask_scale, outer_dim, inner_dim);
            break;
        }
        case ppl::common::DATATYPE_FLOAT16: {
            if (inner_dim % 2 == 0)
                PPLCUDAFastSoftmaxForwardImp<half, 2>(stream, (const half*)input, (half*)output, (const bool*)key_padding_mask, mask_scale, outer_dim, inner_dim);
            else
                PPLCUDAFastSoftmaxForwardImp<half, 1>(stream, (const half*)input, (half*)output, (const bool*)key_padding_mask, mask_scale, outer_dim, inner_dim);
            break;
        }
        default:
            break;
    }
    return ppl::common::RC_SUCCESS;
}
