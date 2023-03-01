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

#include "cudakernel/memory/tile.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void ppl_cukernel_tile(
    int64_t num_elems,
    int num_dims,
    GArray<DivModFast> input_dims_fast,
    GArray<int64_t> input_strides,
    const T* input,
    GArray<DivModFast> output_strides_fast,
    T* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    int64_t input_offset = 0;
    int idx, remain = index;
    for (int it = 0; it < num_dims; ++it) {
        output_strides_fast[it].divmod(remain, idx, remain);
        int quo, in_idx;
        input_dims_fast[it].divmod(idx, quo, in_idx);
        input_offset += input_strides[it] * in_idx;
    }
    output[index] = input[input_offset];
}

template <typename srcT, typename dstT>
__global__ void ppl_cukernel_tile_last_dim_vec(
    int64_t num_elems,
    int divisor,
    int vec_factor,
    const srcT* input,
    dstT* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    int64_t input_offset = index / divisor;
    srcT src_val = input[input_offset];
    dstT dst_val;
    srcT* dst_val_ptr = reinterpret_cast<srcT*>(&dst_val);
    for (int i = 0; i < vec_factor; ++i) {
        dst_val_ptr[i] = src_val;
    }
    output[index] = dst_val;
}

static bool isPowerOfTwo(int n) {
    return (n & (n - 1)) == 0;
}

static bool isScalar2VecOnLastDim(
    TileParam param,
    ppl::common::TensorShape* input_shape)
{
    int dim_count = input_shape->GetDimCount();
    bool pre_is_one = true;
    for (int i = 0; i < (dim_count - 1); ++i) {
        pre_is_one = pre_is_one && (param.repeats[i] == 1);
    }
    bool last_is_power_of2 = isPowerOfTwo(param.repeats[dim_count - 1]);
    bool input_last_is_scalar = (input_shape->GetDim(dim_count - 1) == 1);
    
    if (pre_is_one && last_is_power_of2 && input_last_is_scalar) return true;
    return false;
}

ppl::common::RetCode PPLCUDATileScalar2VecOnLastDimImp(
    cudaStream_t stream,
    TileParam param,
    ppl::common::TensorShape* input_shape,
    const void* input,
    ppl::common::TensorShape* output_shape,
    void* output) {
    int dim_count = input_shape->GetDimCount();
    constexpr int float4_as_bytes = 16;
    constexpr int float2_as_bytes = 8;
    constexpr int float_as_bytes = 4;
    constexpr int half_as_bytes = 2;
    const int input_type_as_bytes = ppl::common::GetSizeOfDataType(input_shape->GetDataType());
    const int expand_factor = param.repeats[dim_count - 1];
    const int expand_size = input_type_as_bytes * expand_factor;

    int divisor = 1;
    int output_type_as_bytes = input_type_as_bytes;
    if (expand_size >= float4_as_bytes) {
        output_type_as_bytes = 16;
        divisor = expand_size / output_type_as_bytes;
    } else if (expand_size >= float2_as_bytes) {
        output_type_as_bytes = 8;
    } else if (expand_size >= float_as_bytes) {
        output_type_as_bytes = 4;
    } else if (expand_size >= half_as_bytes) {
        output_type_as_bytes = 2;
    }
    int vec_factor = output_type_as_bytes / input_type_as_bytes;
    int block_size     = 256;
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding() / vec_factor;
    int grid_size      = (num_elems + block_size - 1) / block_size;

    #define SWITCH_CASE(TYPE)                                                                                               \
    case sizeof(TYPE): {                                                                                                    \
        if (output_type_as_bytes == 16) {                                                                                   \
            ppl_cukernel_tile_last_dim_vec<TYPE, float4><<<grid_size, block_size, 0, stream>>>(num_elems,                   \
                divisor, vec_factor, (const TYPE*)input, (float4*)output);                                                  \
        } else if (output_type_as_bytes == 8) {                                                                             \
            ppl_cukernel_tile_last_dim_vec<TYPE, float2><<<grid_size, block_size, 0, stream>>>(num_elems,                   \
                divisor, vec_factor, (const TYPE*)input, (float2*)output);                                                   \
        } else if (output_type_as_bytes == 4) {                                                                             \
            ppl_cukernel_tile_last_dim_vec<TYPE, float><<<grid_size, block_size, 0, stream>>>(num_elems,                    \
                divisor, vec_factor, (const TYPE*)input, (float*)output);                                                    \
        } else if (output_type_as_bytes == 2) {                                                                             \
            ppl_cukernel_tile_last_dim_vec<TYPE, int16_t><<<grid_size, block_size, 0, stream>>>(num_elems,                     \
                divisor, vec_factor, (const TYPE*)input, (int16_t*)output);                                                 \
        } else {                                                                                                            \
            return ppl::common::RC_UNSUPPORTED;                                                                             \
        }                                                                                                                   \
        return ppl::common::RC_SUCCESS;                                                                                     \
    }

    switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        SWITCH_CASE(int8_t);
        SWITCH_CASE(int16_t);
        SWITCH_CASE(int32_t);
        SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
#undef SWITCH_CASE
}

ppl::common::RetCode PPLCUDATileForwardImp(
    cudaStream_t stream,
    TileParam param,
    ppl::common::TensorShape* input_shape,
    const void* input,
    ppl::common::TensorShape* output_shape,
    void* output)
{
    if (isScalar2VecOnLastDim(param, input_shape)) {
        return PPLCUDATileScalar2VecOnLastDimImp(stream, param,
            input_shape, input, output_shape, output);
    }
    int block_size     = 256;
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    int grid_size      = (num_elems + block_size - 1) / block_size;
    int num_dims       = output_shape->GetDimCount();
    GArray<int64_t> input_strides(num_dims);
    GArray<DivModFast> input_dims_fast(num_dims);
    GArray<DivModFast> output_strides_fast(num_dims);
    int64_t acc_output_stride = 1;
    int64_t acc_input_stride  = 1;
    for (int it = num_dims - 1; it >= 0; --it) {
        input_strides[it]       = acc_input_stride;
        input_dims_fast[it]     = input_shape->GetDim(it);
        output_strides_fast[it] = DivModFast(acc_output_stride);
        acc_input_stride *= input_shape->GetDim(it);
        acc_output_stride *= output_shape->GetDim(it);
    }

#define SWITCH_CASE(TYPE)                                                                                                 \
    case sizeof(TYPE): {                                                                                                  \
        ppl_cukernel_tile<<<grid_size, block_size, 0, stream>>>(                                                          \
            num_elems, num_dims, input_dims_fast, input_strides, (const TYPE*)input, output_strides_fast, (TYPE*)output); \
        return ppl::common::RC_SUCCESS;                                                                                   \
    }

    switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        SWITCH_CASE(int8_t);
        SWITCH_CASE(int16_t);
        SWITCH_CASE(int32_t);
        SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
#undef SWITCH_CASE
}
