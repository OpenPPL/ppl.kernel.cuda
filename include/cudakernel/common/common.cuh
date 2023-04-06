#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <memory>

#define GPU_WARP_SIZE 32

template<typename T>
struct DefaultComputeType {
  using type = T;
};

template<>
struct DefaultComputeType<half> {
  using type = float;
};

template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};

template<typename SRC, typename DST>
struct DirectLoad {
  DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
  const SRC* src;
  int64_t row_size;
};

template<typename SRC, typename DST>
struct DirectStore {
  DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<DST, N> pack;
    const int64_t offset = (row * row_size + col) / N;
#pragma unroll
    for (int i = 0; i < N; ++i) { pack.elem[i] = static_cast<DST>(src[i]); }
    *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
  }
  DST* dst;
  int64_t row_size;
};

#if (__CUDACC_VER_MAJOR__ >= 11)
  #include <cub/cub.cuh>
  template<typename T>
  struct SumOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
  };

  template<typename T>
  struct MaxOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
  };

  template<template<typename> class ReductionOp, typename T, int block_size>
  __inline__ __device__ T BlockAllReduce(T val) {
    typedef cub::BlockReduce<T, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T result_broadcast;
    T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
    if (threadIdx.x == 0) { result_broadcast = result; }
    __syncthreads();
    return result_broadcast;
  }
#endif

inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves,
                                int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks =
      std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
  return cudaSuccess;
}

template<typename T>
__device__ __forceinline__ T _Exp(T a);

template<>
__device__ __forceinline__ float _Exp<float>(float a) {
    return expf(a);
}

template<>
__device__ __forceinline__ double _Exp<double>(double a) {
    return exp(a);
}

template<>
__device__ __forceinline__ half _Exp<half>(half a) {
    return __float2half(exp(__half2float(a)));
}

template<typename T>
__device__ __forceinline__ T _Ldg(const T* p) {
    return __ldg(p);
}

template<>
__device__ __forceinline__ bool _Ldg<bool>(const bool* p) {
    return *p;
}

template<typename T>
__device__ __forceinline__ T _ExpMax() {
    return (T)20.0f;
}

template<>
__device__ __forceinline__ float _ExpMax<float>() {
    return 80.0f;
}

template<>
__device__ __forceinline__ double _ExpMax<double>() {
    return 800.0;
}

template<typename T>
__device__ __forceinline__ T CudaLogZero() {
    return (T)-_ExpMax<T>();
}

template<typename T>
__device__ __forceinline__ T _SafeExp(const T v) {
    return _Exp(min(v, _ExpMax<T>()));
}

template<typename T>
__device__ __forceinline__ T _LogAdd(T x, T y) {
    x = min(x, _ExpMax<T>());
    x = max(x, CudaLogZero<T>());
    y = min(y, _ExpMax<T>());
    y = max(y, CudaLogZero<T>());
    return x + max(log(_SafeExp(y - x) + (T)1.0f), y - x);
}

#define FINAL_MASK 0xffffffff
template<typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane,
                                        int width = 32, unsigned int mask = FINAL_MASK) {
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9000
    return __shfl_sync(mask, value, srcLane, width);
#else
    return __shfl(value, srcLane, width);
#endif
}


template<typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask,
                                            int width = 32, unsigned int mask = FINAL_MASK) {
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template<typename T>
__device__ __forceinline__ T WarpReduceSum(T val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += WARP_SHFL_XOR(val, mask, 32, FINAL_MASK);
    return val;
}

template<typename T>
__device__ __forceinline__ T WarpReduceMax(T val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(WARP_SHFL_XOR(val, mask, 32, FINAL_MASK), val);
    return val;
}

template<typename T>
__device__ __forceinline__ T WarpReduceLogAddSum(T val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = _LogAdd(WARP_SHFL_XOR(val, mask, 32, FINAL_MASK), val);
    return val;
}

template <typename T>
__forceinline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = WarpReduceSum<T>(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  if (wid == 0) {
      val = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? shared[lane] : (T)0.0f;
      val = WarpReduceSum<T>(val);
      return val;
  }
  return (T)0.0f;
}

template <typename T>
__forceinline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = WarpReduceMax<T>(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  if (wid == 0) {
      val = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? shared[lane] : (T)-99999;
      val = WarpReduceMax<T>(val);
      return val;
  }
  return (T)0.0f;
}


template<typename T>
__device__ __forceinline__ T BlockReduceSum(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = WarpReduceSum(val);
    if(lane == 0) shared[wid] = val;
    __syncthreads();

    val = (lane < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    __syncthreads();
    val = WarpReduceSum(val);
    return val;
}

inline int GetBlockSize(const int n, const int max_size = 1024) {
    int ret = 32;
    while(ret < n && ret < max_size) {
        ret <<= 1;
    }
    return ret;
}
