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

#include "cudakernel/arithmetic/einsum.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>

template <typename T>
__global__ void ppl_cukernel_einsum_nbdce(const T* input0, const T* input1, T* output, uint64_t outer, uint64_t inner,
                                        uint64_t n, uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e){
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
        // nbac * ndae --> nbdce
        int tid = threadIdx.x;
        int outer_id = blockIdx.x; //nbd
        int inner_id = blockIdx.y; //ce
        int productDim = a;

        int nb_id = outer_id / d;
        int d_id = outer_id % d;

        int n_id = nb_id / b;
        int b_id = nb_id % b;  // outer_id = n_id*b*d + b_id*d + d_id

        int c_id = inner_id / e;
        int e_id = inner_id % e;

        __shared__ T tmp[256];
        tmp[tid] = 0;
        for(int id = tid; id < productDim; id += blockDim.x){
            if(id < productDim){
                uint64_t input0_offset = n_id * b * a * c + b_id * a * c + id * c + c_id;
                uint64_t input1_offset = n_id * d * a * e + d_id * a * e + id * e + e_id;

                tmp[id] += input0[input0_offset] * input1[input1_offset];
            }
        }
        __syncthreads();

        // for(int stride=blockDim.x/2; stride>0; stride >>= 1){
        //     if(tid < stride)
        //         tmp[tid] += tmp[tid + stride];
        //     __syncthreads();
        // }
        if(tid<128) tmp[tid] += tmp[tid+128];
        __syncthreads();
        if(tid<64) tmp[tid] += tmp[tid+64];
        __syncthreads();
        if(tid<32) tmp[tid] += tmp[tid+32];
        __syncthreads();
        if(tid<16) tmp[tid] += tmp[tid+16];
        __syncthreads();
        if(tid<8) tmp[tid] += tmp[tid+8];
        __syncthreads();
        if(tid<4) tmp[tid] += tmp[tid+4];
        __syncthreads();
        if(tid<2) tmp[tid] += tmp[tid+2];
        __syncthreads();
        if(tid<1) tmp[tid] += tmp[tid+1];
        __syncthreads();

        uint64_t output_offset = outer_id * inner + inner_id;
        output[output_offset] = tmp[0];
#endif
}
template <typename T>
__global__ void ppl_cukernel_einsum_nbdc(const T* input0, const T* input1, T* output, uint64_t outer, uint64_t inner,
                                        uint64_t n, uint64_t a, uint64_t b, uint64_t c, uint64_t d){
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
        // nabc * nadc -> n(a)bdc
        int tid = threadIdx.x;
        int outer_id = blockIdx.x; //nc
        int inner_id = blockIdx.y; //bd
        int productDim = a;

        int c_id = outer_id % c;
        int n_id = outer_id / c;

        // inner_id = b_id *d + d_id
        int d_id = inner_id % d;
        int b_id = inner_id / d;

        __shared__ T tmp[256];
        tmp[tid] = 0;
        for(int id = tid; id < productDim; id += blockDim.x){
            if(id < productDim){
                uint64_t input0_offset = n_id * a * b * c + id * b * c + b_id * c + c_id;
                uint64_t input1_offset = n_id * a * d * c + id * d * c + d_id * c + c_id;

                tmp[id] += input0[input0_offset] * input1[input1_offset];
            }
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride >>= 1){
            if(tid < stride)
                tmp[tid] += tmp[tid + stride];
            __syncthreads();
        }

        // ncbd = n_id*c*b*d + c_id*b*d + b_id*d + d_id = outer_id * inner + inner_id
        // nbdc
        uint64_t output_offset = n_id*b*d*c + b_id*d*c + d_id*c + c_id;
        output[output_offset] = tmp[0];
#endif
}

ppl::common::RetCode PPLCUDAEinSum_nbac_ndae_nbdce_ForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape0,
    const void* input0,
    const ppl::common::TensorShape* input_shape1,
    const void* input1,
    const ppl::common::TensorShape* output_shape,
    void* output,
    std::string equation)
{
    // nbac * ndae -> nbd(a)ce
    auto n = input_shape0->GetDim(0);
    auto b = input_shape0->GetDim(1);
    auto a = input_shape0->GetDim(2);
    auto c = input_shape0->GetDim(3);
    auto d = input_shape1->GetDim(1);
    auto e = input_shape1->GetDim(3);

    auto outer = n * b * d; // nbd
    auto inner = c * e; //ce

    int block_size     = 256;
    dim3 block(block_size);
    dim3 grid(outer, inner);

    auto datatype = output_shape->GetDataType();
    auto dataformat = output_shape->GetDataFormat();

    switch(datatype){
        case ppl::common::DATATYPE_FLOAT32:{
            ppl_cukernel_einsum_nbdce<float><<<grid, block, 0, stream>>>((const float*)input0, (const float*)input1, (float*)output, outer, inner, n, a, b, c, d, e);
            break;
        }
        case ppl::common::DATATYPE_FLOAT16:{
            ppl_cukernel_einsum_nbdce<half><<<grid, block, 0, stream>>>((const half*)input0, (const half*)input1, (half*)output, outer, inner, n, a, b, c, d, e);
            break;
        }
        case ppl::common::DATATYPE_INT64:{
            ppl_cukernel_einsum_nbdce<int64_t><<<grid, block, 0, stream>>>((const int64_t*)input0, (const int64_t*)input1, (int64_t*)output, outer, inner, n, a, b, c, d, e);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }


    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode PPLCUDAEinSum_nabc_nadc_nbdc_ForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape0,
    const void* input0,
    const ppl::common::TensorShape* input_shape1,
    const void* input1,
    const ppl::common::TensorShape* output_shape,
    void* output,
    std::string equation)
{
    // nabc * nadc -> n(a)bdc
    auto n = input_shape0->GetDim(0);
    auto a = input_shape0->GetDim(1);
    auto b = input_shape0->GetDim(2);
    auto c = input_shape0->GetDim(3);
    auto d = input_shape1->GetDim(2);

    auto outer = n * c; // nc
    auto inner = b * d; // bd

    int block_size     = 256;
    dim3 block(block_size);
    dim3 grid(outer, inner);

    auto datatype = output_shape->GetDataType();
    auto dataformat = output_shape->GetDataFormat();

    switch(datatype){
        case ppl::common::DATATYPE_FLOAT32:{
            ppl_cukernel_einsum_nbdc<float><<<grid, block, 0, stream>>>((const float*)input0, (const float*)input1, (float*)output, outer, inner, n, a, b, c, d);
            break;
        }
        case ppl::common::DATATYPE_FLOAT16:{
            ppl_cukernel_einsum_nbdc<half><<<grid, block, 0, stream>>>((const half*)input0, (const half*)input1, (half*)output, outer, inner, n, a, b, c, d);
            break;
        }
        case ppl::common::DATATYPE_INT64:{
            ppl_cukernel_einsum_nbdc<int64_t><<<grid, block, 0, stream>>>((const int64_t*)input0, (const int64_t*)input1, (int64_t*)output, outer, inner, n, a, b, c, d);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }


    return ppl::common::RC_SUCCESS;
}

template <typename T>
__global__ void ppl_cukernel_einsum_nbdce2(const T* input0, const T* input1, T* output, uint64_t outer, uint64_t inner,
                                        uint64_t n, uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e){
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
        // nbac * ndae --> nbdce
        // grid(d, b, n) block(e, c)
        // int ix = threadIdx.x + blockDim.x * blockIdx.x;
        // int iy = threadIdx.y + blockDim.y * blockIdx.y;

        int g_blockId = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
                        // n_id * b * d + b_id * d + d_id
        int g_threadId = g_blockId * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
                        // () * c * e + c_id * e + e_id

        int n_id = blockIdx.z;
        int b_id = blockIdx.y;
        int d_id = blockIdx.x;
        int c_id = threadIdx.y;
        int e_id = threadIdx.x;
        int productDim = a;

        T sum=0; // when T is unit64,  sum may exceed max(uint64) if preductDim is too big
        #pragma unroll
        for(int i=0;i<productDim;++i){
            uint64_t input0_offset = n_id * b * a * c + b_id * a * c + i * c + c_id;
            uint64_t input1_offset = n_id * d * a * e + d_id * a * e + i * e + e_id;
            sum += input0[input0_offset] * input1[input1_offset];
        }
        // __syncthreads();

        output[g_threadId] = sum;
#endif
}


ppl::common::RetCode PPLCUDAEinSum_nbac_ndae_nbdce_2_ForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape0,
    const void* input0,
    const ppl::common::TensorShape* input_shape1,
    const void* input1,
    const ppl::common::TensorShape* output_shape,
    void* output,
    std::string equation)
{
    // nbac * ndae -> nbd(a)ce
    auto n = input_shape0->GetDim(0);
    auto b = input_shape0->GetDim(1);
    auto a = input_shape0->GetDim(2);
    auto c = input_shape0->GetDim(3);
    auto d = input_shape1->GetDim(1);
    auto e = input_shape1->GetDim(3);

    dim3 block(e, c);
    dim3 grid(d, b, n);
    int outer=1;
    int inner=1;

    auto datatype = output_shape->GetDataType();
    auto dataformat = output_shape->GetDataFormat();

    switch(datatype){
        case ppl::common::DATATYPE_FLOAT32:{
            ppl_cukernel_einsum_nbdce2<float><<<grid, block, 0, stream>>>((const float*)input0, (const float*)input1, (float*)output, outer, inner, n, a, b, c, d, e);
            break;
        }
        case ppl::common::DATATYPE_FLOAT16:{
            ppl_cukernel_einsum_nbdce2<half><<<grid, block, 0, stream>>>((const half*)input0, (const half*)input1, (half*)output, outer, inner, n, a, b, c, d, e);
            break;
        }
        case ppl::common::DATATYPE_INT64:{
            ppl_cukernel_einsum_nbdce2<int64_t><<<grid, block, 0, stream>>>((const int64_t*)input0, (const int64_t*)input1, (int64_t*)output, outer, inner, n, a, b, c, d, e);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}


template <typename T>
__global__ void ppl_cukernel_einsum_i_j_ij(const T* input0, const T* input1, T* output, uint64_t i, uint64_t j){
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
        // int g_blockId = blockIdx.y * gridDim.x + blockIdx.x;
        // int g_threadId = g_blockId * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
       
        int index_j = threadIdx.x + blockDim.x * blockIdx.x;
        int index_i = threadIdx.y + blockDim.y * blockIdx.y;
        int id = index_i * j + index_j;

        if(index_i >= i || index_j >= j) return;

        output[id] = input0[index_i] * input1[index_j];
#endif
}

ppl::common::RetCode PPLCUDAEinSum_i_j_ij_ForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape0,
    const void* input0,
    const ppl::common::TensorShape* input_shape1,
    const void* input1,
    const ppl::common::TensorShape* output_shape,
    void* output,
    std::string equation)
{
    // i * j -> ij
    // ni * nj -> nij
    auto i = input_shape0->GetDim(0);
    auto j = input_shape1->GetDim(0);

    // int num_elems = output_shape->CalcElementsIncludingPadding();

    int block_size     = 32;
    int grid_x = (j + block_size - 1) / block_size ;
    int grid_y = (i + block_size - 1) / block_size;

    dim3 block(block_size, block_size);
    dim3 grid(grid_x, grid_y);

    auto datatype = output_shape->GetDataType();
    // auto dataformat = output_shape->GetDataFormat();

    switch(datatype){
        case ppl::common::DATATYPE_FLOAT32:{
            ppl_cukernel_einsum_i_j_ij<float><<<grid, block, 0, stream>>>((const float*)input0, (const float*)input1, (float*)output, i, j);
            break;
        }
        case ppl::common::DATATYPE_FLOAT16:{
            ppl_cukernel_einsum_i_j_ij<half><<<grid, block, 0, stream>>>((const half*)input0, (const half*)input1, (half*)output, i, j);
            break;
        }
        case ppl::common::DATATYPE_INT64:{
            ppl_cukernel_einsum_i_j_ij<int64_t><<<grid, block, 0, stream>>>((const int64_t*)input0, (const int64_t*)input1, (int64_t*)output, i, j);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }


    return ppl::common::RC_SUCCESS;
}

