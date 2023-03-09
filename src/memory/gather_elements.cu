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

#include "cudakernel/memory/gather_elements.h"
#include "cudakernel/common/divmod_fast.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>
#include <memory>
#include <stdio.h>

static __host__ __device__ __inline__ int get_indices_val(
    int indices_element_size,
    int offset,
    const void* indices)
{
    int res = 0;
    switch (indices_element_size) {
        case sizeof(int32_t):
            res = static_cast<const int32_t*>(indices)[offset];
            break;
        case sizeof(int64_t):
            res = static_cast<const int64_t*>(indices)[offset];
            break;
        default:
            break;
    }
    return res;
}

template <typename T>
__global__ void ppl_cukernel_gather_elements(
    int64_t num_elems,
    DivModFast output_outer_block_fast,
    int output_axis_size,
    DivModFast output_inner_block_fast,
    const T* input,
    T* output,
    int indices_element_size,
    const void* indices)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int outer_idx, block_offset;
    output_outer_block_fast.divmod(index, outer_idx, block_offset);
    int indices_offset, inner_idx;
    output_inner_block_fast.divmod(block_offset, indices_offset, inner_idx);
    int64_t indices_idx = get_indices_val(indices_element_size, index, indices);
    int64_t input_idx = (outer_idx * output_axis_size + indices_idx) *
                            output_inner_block_fast.d_ +
                        inner_idx;
    output[index] = input[input_idx];
}

ppl::common::RetCode PPLCUDAGatherElementsForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* indices_shape,
    const void* indices,
    const ppl::common::TensorShape* output_shape,
    void* output,
    int axis)
{
    int indices_element_size = ppl::common::GetSizeOfDataType(indices_shape->GetDataType());
    int64_t num_elems      = output_shape->CalcElementsIncludingPadding();
    int block_size         = 256;
    int grid_size          = (num_elems + block_size - 1) / block_size;
    // output dimension can be partitioned as outer--indices--inner. (before axis, axis, after axis)
    int output_inner_block = output_shape->CalcElementsFromDimensionIncludingPadding(axis + 1);
    int output_axis_size   = output_shape->GetDim(axis);
    int indices_block_size = output_shape->CalcElementsIncludingPadding();
    int output_outer_block = output_axis_size * output_inner_block;

    DivModFast output_outer_block_fast(output_outer_block);
    DivModFast output_inner_block_fast(output_inner_block);

#define SWITCH_CASE(TYPE)                                                                                                                                                                                                       \
    case sizeof(TYPE): {                                                                                                                                                                                                        \
        ppl_cukernel_gather_elements<<<grid_size, block_size, 0, stream>>>(num_elems, output_outer_block_fast, output_axis_size, output_inner_block_fast, (const TYPE*)input, (TYPE*)output, indices_element_size, (const void*)indices); \
        return ppl::common::RC_SUCCESS;                                                                                                                                                                                         \
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
