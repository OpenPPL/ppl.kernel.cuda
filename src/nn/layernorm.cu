#include "cudakernel/nn/layernorm.h"
#include "ppl/common/tensor_shape.h"
#include <cuda_fp16.h>
#include "cudakernel/common/common.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
// 4 使用float4 读入 + shared mem 保存
__global__ __launch_bounds__(256) void ppl3_cukernel_layernorm_fp32_float4(
    const float* input, const float* scale, const float* shift, float* output,
    int B, int N, bool has_affine, float eps) {
    // N = 256, blockDim.x = 64
    int tid = threadIdx.x;
    float* cur_in = const_cast<float*>(input + blockIdx.x * N);
    float* cur_out = output + blockIdx.x * N;
    float sumx=0.0f;
    float sumy=0.0f;
    __shared__ float data[256];

    float4 v =  FETCH_FLOAT4(cur_in[4*tid]);
    data[4*tid] = v.x;
    data[4*tid+1] = v.y;
    data[4*tid+2] = v.z;
    data[4*tid+3] = v.w;

    sumx = v.x + v.y + v.z + v.w;
    sumy = v.x * v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    //BlockReduceSum
    float sumx_ = BlockReduceSum(sumx);
    float sumy_ = BlockReduceSum(sumy);
    float mean = sumx_ / N;
    float rstd = rsqrtf(sumy_ / N - mean * mean + eps);
    float4 val;
    val.x = (data[4*tid+0] - mean) * rstd;
    val.y = (data[4*tid+1] - mean) * rstd;
    val.z = (data[4*tid+2] - mean) * rstd;
    val.w = (data[4*tid+3] - mean) * rstd;
    if(has_affine){
        val.x = val.x * (float)__ldg(scale + 4*tid+0) + (float)__ldg(shift + 4*tid+0);
        val.y = val.y * (float)__ldg(scale + 4*tid+1) + (float)__ldg(shift + 4*tid+1);
        val.z = val.z * (float)__ldg(scale + 4*tid+2) + (float)__ldg(shift + 4*tid+2);
        val.w = val.w * (float)__ldg(scale + 4*tid+3) + (float)__ldg(shift + 4*tid+3);
    }

    FETCH_FLOAT4(cur_out[4*tid]) =  val;
}


__global__ __launch_bounds__(256) void ppl3_cukernel_layernorm_fp32_opt(
    const float* input, const float* scale, const float* shift, float* output,
    int B, int N, bool has_affine, float eps) {
    auto cur_in = input + blockIdx.x * N;
    auto cur_out = output + blockIdx.x * N;
    float sumx=0;
    float sumy=0;
    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        float v = (float)__ldg(cur_in + tid);
        sumx += v;
        sumy += v* v;
    }
    //BlockReduceSum
    float sumx_ = BlockReduceSum(sumx);
    float sumy_ = BlockReduceSum(sumy);
    float mean = sumx_ / N;
    float rstd = rsqrtf(sumy_ / N - mean * mean + eps);
    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        float val = ((float)__ldg(cur_in + tid) - mean) * rstd;
        if(has_affine){
            val = val * (float)__ldg(scale + tid) + (float)__ldg(shift + tid);
        }
        cur_out[tid] =  val;
    }
}

__global__ __launch_bounds__(256) void ppl3_cukernel_layernorm_fp32(
    const float* input, const float* weight, const float* bias, float* output,
    int outer, int inner, bool has_affine, float eps) {
    __shared__ float acc_val[256];
    const int outer_idx = blockIdx.x;
    const float* input_ptr = input + outer_idx * inner;
    float* output_ptr = output + outer_idx * inner;

    const unsigned int tid = threadIdx.x;
    //! calculate mean_val
    acc_val[tid] = 0.f;
    for (int i = tid; i < inner; i += blockDim.x) {
        acc_val[tid] += input_ptr[i];
    }
    __syncthreads();

    unsigned int i = 1, j = 2;
    while (j <= blockDim.x) {
        if (tid % j == 0) {
            acc_val[tid] += acc_val[tid + i];
        }
        i <<= 1; j <<= 1;
        __syncthreads();
    }
    const float mean_val = acc_val[0] / inner;
    __syncthreads();

    acc_val[tid] = 0.f;
    for (int i = tid; i < inner; i += blockDim.x) {
        acc_val[tid] += (input_ptr[i] - mean_val) * (input_ptr[i] - mean_val);
    }
    __syncthreads();

    i = 1; j = 2;
    while (j <= blockDim.x) {
        if (tid % j == 0) {
            acc_val[tid] += acc_val[tid + i];
        }
        i <<= 1; j <<= 1;
        __syncthreads();
    }
    float std = sqrtf(acc_val[0] / inner + eps);
    float r_std = 1.0f / std;

    if (has_affine) {
        for (int i = tid; i < inner; i += blockDim.x) {
            float weight_val = weight[tid]; float bias_val = bias[tid];
            float out_val = weight_val * (input_ptr[i] - mean_val) * r_std  + bias_val;
            // if(if_relu) out_val = (out_val > 0) ? out_val : 0;
            output_ptr[i] =  out_val;
        }
    } else {
        for (int i = tid; i < inner; i += blockDim.x) {
            float out_val = (input_ptr[i] - mean_val) * r_std;
            // if(if_relu) out_val = (out_val > 0) ? out_val : 0;
            output_ptr[i] =  out_val;
        }
    }
}
__global__ __launch_bounds__(256) void ppl3_cukernel_layernorm_fp16_float2(
    const half* input, const half* scale, const half* shift, half* output,
    int B, int N, bool has_affine, float eps) {
        // outer=B inner=N
        // N shoud be 256, blockDim.x = 64
        // TODO add float2 opt
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
#endif
}

__global__ __launch_bounds__(256) void ppl3_cukernel_layernorm_fp16_opt(
    const half* input, const half* scale, const half* shift, half* output,
    int B, int N, bool has_affine, float eps) { // outer=B inner=N
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    auto cur_in = input + blockIdx.x * N;
    auto cur_out = output + blockIdx.x * N;
    float sumx = 0;
    float sumy = 0;

    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x){
        float v = __half2float(cur_in[tid]);
        sumx += v;
        sumy += v * v;
    }
    // BlockReduceSum;
    float sumx_ = BlockReduceSum(sumx);
    float sumy_ = BlockReduceSum(sumy);
    float mean = sumx_ / N;
    float rstd = rsqrtf(sumy_ / N - mean * mean + eps);

    half mean_h = __float2half(mean);
    half rstd_h = __float2half(rstd);

    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        half val = (cur_in[tid] - mean_h) * rstd_h;
        if(has_affine)
            val = val * scale[tid] + shift[tid];
        cur_out[tid] = val;
    }
#endif
}
__global__ __launch_bounds__(256) void ppl3_cukernel_layernorm_fp16(
    const half* input, const half* weight, const half* bias, half* output,
    int outer, int inner, bool has_affine, float eps) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    __shared__ float acc_val[256];
    const int outer_idx = blockIdx.x;
    const half* input_ptr = input + outer_idx * inner;
    half* output_ptr = output + outer_idx * inner;

    const unsigned int tid = threadIdx.x;
    //! calculate mean_val
    acc_val[tid] = 0.f;
    for (int i = tid; i < inner; i += blockDim.x) {
        acc_val[tid] += __half2float(input_ptr[i]);
    }
    __syncthreads();

    unsigned int i = 1, j = 2;
    while (j <= blockDim.x) {
        if (tid % j == 0) {
            acc_val[tid] += acc_val[tid + i];
        }
        i <<= 1; j <<= 1;
        __syncthreads();
    }
    const float mean_val = acc_val[0] / inner;
    __syncthreads();

    acc_val[tid] = 0.f;
    for (int i = tid; i < inner; i += blockDim.x) {
        acc_val[tid] += (__half2float(input_ptr[i]) - mean_val) * (__half2float(input_ptr[i]) - mean_val);
    }
    __syncthreads();

    i = 1; j = 2;
    while (j <= blockDim.x) {
        if (tid % j == 0) {
            acc_val[tid] += acc_val[tid + i];
        }
        i <<= 1; j <<= 1;
        __syncthreads();
    }
    float std = sqrtf(acc_val[0] / inner + eps);
    float r_std = 1.0f / std;

    half mean_val_h = __float2half(mean_val);
    half r_std_h = __float2half(r_std);

    if (has_affine) {
        for (int i = tid; i < inner; i += blockDim.x) {
            half weight_val = weight[tid]; half bias_val = bias[tid];
            half out_val = weight_val * (input_ptr[i] - mean_val_h) * r_std_h  + bias_val;
            // if(if_relu) out_val = (out_val > 0) ? out_val : 0;
            output_ptr[i] =  out_val;
        }
    } else {
        for (int i = tid; i < inner; i += blockDim.x) {
            half out_val = (input_ptr[i] - mean_val_h) * r_std_h;
            // if(if_relu) out_val = (out_val > 0) ? out_val : 0;
            output_ptr[i] =  out_val;
        }
    }
#endif
}


__global__ __launch_bounds__(256) void ppl_cudakernel_layernorm_int8(
    const char* input, const float* weight, const float* bias, char* output,
    int outer, int inner, bool has_affine, float eps, float in_scale, float out_scale) {

    __shared__ float acc_val[256];
    const int outer_idx = blockIdx.x;
    const unsigned int tid = threadIdx.x;

    //! calculate mean_val
    acc_val[tid] = 0.f;
    for (int i = tid; i < inner; i += blockDim.x) {
        acc_val[tid] += (float)input[outer_idx * inner + i] * in_scale;
    }
    __syncthreads();

    unsigned int i = 1, j = 2;
    while (j <= blockDim.x) {
        if (tid % j == 0) {
            acc_val[tid] += acc_val[tid + i];
        }
        i <<= 1; j <<= 1;
        __syncthreads();
    }
    const float mean_val = acc_val[0] / inner ;
    __syncthreads();

    acc_val[tid] = 0.f;
    for (int i = tid; i < inner; i += blockDim.x) {
        acc_val[tid] += ((float)input[outer_idx * inner + i] * in_scale - mean_val) * ((float)input[outer_idx * inner + i]  *in_scale - mean_val);
    }

    __syncthreads();

    i = 1; j = 2;
    while (j <= blockDim.x) {
        if (tid % j == 0) {
            acc_val[tid] += acc_val[tid + i];
        }
        i <<= 1; j <<= 1;
        __syncthreads();
    }
    float std = sqrtf(acc_val[0] / inner + eps);
    float r_std = 1.0f / std;

    for (int i=tid; i < inner; i += blockDim.x){
        float weight_val = 1.0f;  float bias_val=0;
        if (has_affine){
             weight_val = weight[tid];
             bias_val = bias[tid];
        }

        float out_val = weight_val * ((float)input[outer_idx * inner + i] *in_scale - mean_val) * r_std + bias_val;
        int int_val = __float2int_rn(out_val * out_scale);
        char dst = int_val < -128 ? -128 : int_val > 127 ? 127 : (char)int_val;
        output[outer_idx * inner + i] = dst;
    }
}

ppl::common::RetCode PPLCUDALayerNormForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const void* input,
    const void* scale,
    const void* shift,
    void* output,
    int outer,
    int inner,
    bool elementwise_affine,
    float epsilon,
    float in_scale,
    float out_scale){

    int block_size = 256;
    int grid_size = outer;

    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        ppl3_cukernel_layernorm_fp32_opt<<<grid_size, block_size, 0, stream>>>(
                    (const float*)input,(const float*)scale, (const float*)shift,
                    (float*)output, outer, inner, elementwise_affine, epsilon);
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16){
        ppl3_cukernel_layernorm_fp16_opt<<<grid_size, block_size, 0, stream>>>(
                    (const half*)input,(const half*)scale, (const half*)shift,
                    (half*)output, outer, inner, elementwise_affine, epsilon);
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        ppl_cudakernel_layernorm_int8<<<grid_size, block_size, 0, stream>>>(
                    (const char*)input, (const float*)scale, (const float*)shift,
                    (char*)output, outer, inner, elementwise_affine, epsilon, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}
ppl::common::RetCode PPLCUDALayerNormForwardImp256(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const void* input,
    const void* scale,
    const void* shift,
    void* output,
    int outer,
    int inner,
    bool elementwise_affine,
    float epsilon,
    float in_scale,
    float out_scale){

    int block_size = 64;
    int grid_size = outer;
    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32)
            ppl3_cukernel_layernorm_fp32_float4<<<grid_size, block_size, 0, stream>>>(
                    (const float*)input,(const float*)scale, (const float*)shift,
                    (float*)output, outer, inner, elementwise_affine, epsilon);
    else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}
