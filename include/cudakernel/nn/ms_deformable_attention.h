#include "ppl/common/tensor_shape.h"

#include "ppl/common/retcode.h"

ppl::common::RetCode PPLCUDAMSDeformAttnForwardImp(
    const cudaStream_t &stream,
    const ppl::common::TensorShape *input_shape,
    const ppl::common::TensorShape *output_shape,
    const void *data, // float
    const void *spatial_shapes, // int64_t
    const void *level_start_index, // int64_t
    const void *sampling_loc, // float
    const void *attn_weight,  // float
    void* output,                  // float
    const int batch,
    int64_t im2col_step_, 
    const int spatial_size, 
    const int num_heads, 
    const int channels, 
    const int num_levels, 
    const int num_query, 
    const int num_point);
