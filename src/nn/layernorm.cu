#include "cudakernel/nn/layernorm.h"
#include "ppl/common/tensor_shape.h"
#include <cuda_fp16.h>
#include "cudakernel/common/common.cuh"

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}


template <int VPT, int TPB>
__global__ void LayernormForward_fp16(
    const half *x,
    const half *weight,
    const half *bias,
    const float eps,
    const int32_t normalize_shape,
    half *output
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    half inLocal[VPT]; half weightLocal[VPT]; half biasLocal[VPT];

    copy<sizeof(half) * VPT>(&x[idx], inLocal);
    half2 loc = __floats2half2_rn(0.f, 0.f); // accumulator
    half r_normalize_shape = __float2half_rn(1 / (float)(normalize_shape));

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        loc.x = __hfma(r_normalize_shape, inLocal[it], loc.x); 
        loc.y = __hfma(loc.x, inLocal[it], loc.y);
    }

    copy<sizeof(half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    __shared__ half mu;     // mean
    __shared__ half rsigma; // 1 / std.dev.
    #if (__CUDACC_VER_MAJOR__ >= 11)
        const half2 reduced = BlockAllReduce<SumOp, half2, TPB>(loc);
    #else
        half2 reduced;
        reduced.x = blockReduceSum<half>(loc.x);
        reduced.y = blockReduceSum<half>(loc.y);
    #endif

    if (threadIdx.x == 0)
    {
        mu = __low2half(reduced);
        rsigma = rsqrt(__high2half(reduced) - mu * mu + __float2half(eps));
    }
    __syncthreads();

    half outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        outLocal[it] = (inLocal[it] - mu) * rsigma * weightLocal[it] + biasLocal[it];
    }
    copy<sizeof(half) * VPT>(outLocal, &output[idx]);
};

template <int VPT, int TPB>
__global__ void LayernormForward_fp32(
    const float *x,
    const float *weight,
    const float *bias,
    const float eps,
    const int32_t normalize_shape,
    float *output
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    float inLocal[VPT]; float weightLocal[VPT]; float biasLocal[VPT];

    copy<sizeof(float) * VPT>(&x[idx], inLocal);
    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        loc.x = fmaf(r_normalize_shape, inLocal[it], loc.x); 
        loc.y = fmaf(loc.x, inLocal[it], loc.y);
    }

    copy<sizeof(float) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(float) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    __shared__ float mu;     // mean
    __shared__ float rsigma; // 1 / std.dev.
    #if (__CUDACC_VER_MAJOR__ >= 11)
        const float reduced_x = BlockAllReduce<SumOp, float, TPB>(loc.x);
        const float reduced_y = BlockAllReduce<SumOp, float, TPB>(loc.y);
    #else
        const float reduced_x = blockReduceSum<float>(loc.x);
        const float reduced_y = blockReduceSum<float>(loc.y);
    #endif
    if (threadIdx.x == 0)
    {
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);
    }
    __syncthreads();

    float outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        outLocal[it] = (inLocal[it] - mu) * rsigma * weightLocal[it] + biasLocal[it];
    }
    copy<sizeof(float) * VPT>(outLocal, &output[idx]);
};



__global__ __launch_bounds__(256) void LayernormForwardDefault_fp16(
    const half* input, const half* scale, const half* shift, half* output,
    int N, bool has_affine, float eps) {
    auto cur_in = input + blockIdx.x * N;
    auto cur_out = output + blockIdx.x * N;
    float2 loc = make_float2(0.f, 0.f);

    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x){
        float v = __half2float(cur_in[tid]);
        loc.x += v;
        loc.y += v * v;
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        loc.x =  BlockAllReduce<SumOp, float, 256>(loc.x);
        loc.y =  BlockAllReduce<SumOp, float, 256>(loc.y);
    #else
        BlockDoubleReduceSum(loc.x, loc.y);
    #endif

    float mean = loc.x / N;
    float rstd = rsqrtf(loc.y / N - mean * mean + eps);

    half mean_h = __float2half(mean);
    half rstd_h = __float2half(rstd);

    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        half val = (cur_in[tid] - mean_h) * rstd_h;
        if(has_affine)
            val = val * scale[tid] + shift[tid];
        cur_out[tid] = val;
    }
}



__global__ __launch_bounds__(256) void LayernormForwardDefault_fp32(
    const float* input, const float* scale, const float* shift, float* output,
    int N, bool has_affine, float eps) {
    auto cur_in = input + blockIdx.x * N;
    auto cur_out = output + blockIdx.x * N;
    float2 loc = make_float2(0.f, 0.f);

    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x){
        float v = cur_in[tid];
        loc.x += v;
        loc.y += v * v;
    }

    #if (__CUDACC_VER_MAJOR__ >= 11)
        loc.x =  BlockAllReduce<SumOp, float, 256>(loc.x);
        loc.y =  BlockAllReduce<SumOp, float, 256>(loc.y);
    #else
        BlockDoubleReduceSum(loc.x, loc.y);
    #endif

    float mean = loc.x / N;
    float rstd = rsqrtf(loc.y / N - mean * mean + eps);

    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        float val = (cur_in[tid] - mean) * rstd;
        if(has_affine)
            val = val * scale[tid] + shift[tid];
        cur_out[tid] = val;
    }
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
    float eps,
    float in_scale,
    float out_scale){

    const int64_t norm_size = inner;
    const int32_t grid_size = outer;
    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        constexpr int32_t VPT = 16 / sizeof(float);
        switch (norm_size)
        {
            case 128:
                LayernormForward_fp32<VPT, 128 / VPT><<<grid_size, 128 / VPT, 0, stream>>>
                ((float*)input, (float*)scale, (float*)shift, eps, norm_size, (float*)output);
                break;
            case 256:
                LayernormForward_fp32<VPT, 256 / VPT><<<grid_size, 256 / VPT, 0, stream>>>
                ((float*)input, (float*)scale, (float*)shift, eps, norm_size, (float*)output);
                break;
            case 320:
                LayernormForward_fp32<VPT, 320 / VPT><<<grid_size, 320 / VPT, 0, stream>>>
                ((float*)input, (float*)scale, (float*)shift, eps, norm_size, (float*)output);
                break;
            case 512:
                LayernormForward_fp32<VPT, 512 / VPT><<<grid_size, 512 / VPT, 0, stream>>>
                ((float*)input, (float*)scale, (float*)shift, eps, norm_size, (float*)output);
                break;
            case 640:
                LayernormForward_fp32<VPT, 640 / VPT><<<grid_size, 640 / VPT, 0, stream>>>
                ((float*)input, (float*)scale, (float*)shift, eps, norm_size, (float*)output);
                break;
            case 768:
                LayernormForward_fp32<VPT, 768 / VPT><<<grid_size, 768 / VPT, 0, stream>>>
                ((float*)input, (float*)scale, (float*)shift, eps, norm_size, (float*)output);
                break;
            case 1024:
                LayernormForward_fp32<VPT, 1024 / VPT><<<grid_size, 1024 / VPT, 0, stream>>>
                ((float*)input, (float*)scale, (float*)shift, eps, norm_size, (float*)output);
                break;
            case 1280:
                LayernormForward_fp32<VPT, 1280 / VPT><<<grid_size, 1280 / VPT, 0, stream>>>
                ((float*)input, (float*)scale, (float*)shift, eps, norm_size, (float*)output);
                break;
            case 2048:
                LayernormForward_fp32<VPT, 2048 / VPT><<<grid_size, 2048 / VPT, 0, stream>>>
                ((float*)input, (float*)scale, (float*)shift, eps, norm_size, (float*)output);
                break;
            case 4096:
                LayernormForward_fp32<VPT, 4096 / VPT><<<grid_size, 4096 / VPT, 0, stream>>>
                ((float*)input, (float*)scale, (float*)shift, eps, norm_size, (float*)output);
                break;
            default :
                LayernormForwardDefault_fp32<<<grid_size, 256, 0, stream>>>
                ((float*)input, (float*)scale, (float*)shift, (float*)output, norm_size, true, eps);
        }
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        constexpr int32_t VPT = 16 / sizeof(half);
        switch (norm_size)
        {
            case 128:
                LayernormForward_fp16<VPT, 128 / VPT><<<grid_size, 128 / VPT, 0, stream>>>
                ((half*)input, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            case 256:
                LayernormForward_fp16<VPT, 256 / VPT><<<grid_size, 256 / VPT, 0, stream>>>
                ((half*)input, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            // case 320:
                // LayernormForward_fp16<VPT, 320 / VPT><<<grid_size, 320 / VPT, 0, stream>>>
                // ((half*)input, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                // break;
            case 512:
                LayernormForward_fp16<VPT, 512 / VPT><<<grid_size, 512 / VPT, 0, stream>>>
                ((half*)input, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            // case 640:
                // LayernormForward_fp16<VPT, 640 / VPT><<<grid_size, 640 / VPT, 0, stream>>>
                // ((half*)input, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                // break;
            case 768:
                LayernormForward_fp16<VPT, 768 / VPT><<<grid_size, 768 / VPT, 0, stream>>>
                ((half*)input, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            case 1024:
                LayernormForward_fp16<VPT, 1024 / VPT><<<grid_size, 1024 / VPT, 0, stream>>>
                ((half*)input, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            // case 1280:
                // LayernormForward_fp16<VPT, 1280 / VPT><<<grid_size, 1280 / VPT, 0, stream>>>
                // ((half*)input, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                // break;
            case 2048:
                LayernormForward_fp16<VPT, 2048 / VPT><<<grid_size, 2048 / VPT, 0, stream>>>
                ((half*)input, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            case 4096:
                LayernormForward_fp16<VPT, 4096 / VPT><<<grid_size, 4096 / VPT, 0, stream>>>
                ((half*)input, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            default :
                LayernormForwardDefault_fp16<<<grid_size, 256, 0, stream>>>
                ((half*)input, (half*)scale, (half*)shift, (half*)output, norm_size, true, eps);
        }
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        ppl_cudakernel_layernorm_int8<<<grid_size, 256, 0, stream>>>(
                    (const char*)input, (const float*)scale, (const float*)shift,
                    (char*)output, outer, inner, elementwise_affine, eps, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}