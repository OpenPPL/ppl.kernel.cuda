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

#ifndef __PPLCUDA_IMPLICITGEMM_CONV_H_
#define __PPLCUDA_IMPLICITGEMM_CONV_H_

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include "ppl/common/types.h"
#include "ppl/common/retcode.h"
#include "ppl/common/cuda/conv_param.h"


std::string GetConvShapeString(const conv_param_t& conv_param);

uint64_t PPLCUDAConvolutionGetCompilationBufSize(
    ppl::common::datatype_t type,
    conv_param_t& conv_param,
    uint64_t workspace = ((uint64_t)8) * 1024 * 1024 * 1024);

uint64_t PPLCUDAConvolutionGetRuntimeBufSize(
    ppl::common::datatype_t type,
    conv_param_t& conv_param,
    unsigned int splitk,
    unsigned int splitf,
    uint64_t workspace = ((uint64_t)8) * 1024 * 1024 * 1024);

ppl::common::RetCode GetInt8ConvKernelNominees(
    const cudaDeviceProp& device_prop,
    ppl::common::datatype_t type,
    conv_param_t &conv_param,
    std::vector<std::string> & knames,
    std::vector<algo_param_t> & params,
    std::string & sources,
    bool spk_only); // gemm and matmul only support 2spk kernel

ppl::common::RetCode GetFp16ConvKernelNominees(
    const cudaDeviceProp& device_prop,
    ppl::common::datatype_t type,
    conv_param_t &conv_param,
    std::vector<std::string> & knames,
    std::vector<algo_param_t> & params,
    std::string & sources,
    bool spk_only); // gemm and matmul only support 2spk kernel

double PPLCUDAConvolutionSelectKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t& stream,
    ppl::common::datatype_t type,
    int4* d_input,
    int4* d_flt,
    int4* d_output,
    int4* bias,
    int4* d_temp_buf,
    algo_param_t& algo_param,
    conv_param_t& conv_param,
    fuse_param_t& fuse_param,
    uint64_t workspace = (uint64_t)8 * 1024 * 1024 * 1024);

double PPLCUDAConvolutionJitSelectKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t& stream,
    ppl::common::datatype_t type,
    int4* d_input,
    int4* d_flt,
    int4* d_output,
    int4* bias,
    int4* d_temp_buf,
    algo_param_t& algo_param,
    conv_param_t& conv_param,
    fuse_param_t& fuse_param,
    uint64_t workspace = (uint64_t)8 * 1024 * 1024 * 1024);

float AlgoForwardTime(
    const cudaDeviceProp& device_prop,
    cudaStream_t& stream,
    std::vector<std::string> name,
    std::string code,
    int& idx,
    std::vector<const char*> compile_params,
    bool include,
    ppl::common::datatype_t type,
    int4* d_input,
    int4* d_flt,
    int4* d_output,
    int4* bias,
    int4* d_temp_buf,
    std::vector<algo_param_t>& algo_param,
    conv_param_t& conv_param,
    fuse_param_t& fuse_param,
    uint64_t workspace);

void PPLCUDAConvolutionForwardImp(
    const cudaDeviceProp& device_prop,
    cudaStream_t& stream,
    ppl::common::datatype_t type,
    int4* d_input,
    int4* d_flt,
    int4* d_output,
    int4* bias,
    int4* d_temp_buf,
    algo_param_t& algo_param,
    conv_param_t& conv_param,
    fuse_param_t& fuse_param);

void PPLCUDAConvolutionForwardJitImp(
    const cudaDeviceProp& device_prop,
    cudaStream_t& stream,
    CUfunction function,
    ppl::common::datatype_t type,
    int4* d_input,
    int4* d_flt,
    int4* d_output,
    int4* bias,
    int4* d_temp_buf,
    algo_param_t& algo_param,
    conv_param_t& conv_param,
    fuse_param_t& fuse_param);

double PPLCUDAConvolutionSelectKernelInt8(
    const cudaDeviceProp& device_prop,
    cudaStream_t &stream, 
    ppl::common::datatype_t type,
    int4* d_input,
    int4* d_flt,
    int4* d_output,
    int4* bias,
    int4* d_temp_buf, 
    algo_param_t &algo_param,
    conv_param_t &conv_param, 
    quant_param_t &quant_param,
    fuse_param_t &fuse_param,
    uint64_t workspace = (uint64_t)8*1024*1024*1024);

void PPLCUDAConvolutionForwardImpInt8(
    const cudaDeviceProp& device_prop,
    cudaStream_t &stream, 
    ppl::common::datatype_t type,
    int4* d_input,
    int4* d_flt,
    int4* d_output,
    int4* bias,
    int4* d_temp_buf, 
    algo_param_t &algo_param,
    conv_param_t &conv_param, 
    quant_param_t &quant_param,
    fuse_param_t &fuse_param);

float AlgoForwardTimeInt8(
    const cudaDeviceProp& device_prop,
    cudaStream_t& stream,
    std::vector<std::string> name,
    std::string code,
    int& idx,
    std::vector<const char*> compile_params,
    bool include,
    ppl::common::datatype_t type,
    int4* d_input,
    int4* d_flt,
    int4* d_output,
    int4* bias,
    int4* d_temp_buf,
    std::vector<algo_param_t>& algo_param,
    conv_param_t& conv_param,
    quant_param_t& quant_param,
    fuse_param_t& fuse_param,
    uint64_t workspace);
    
double PPLCUDAConvolutionJitSelectKernelInt8(
    const cudaDeviceProp& device_prop,
    cudaStream_t &stream, 
    ppl::common::datatype_t type,
    int4* d_input,
    int4* d_flt,
    int4* d_output,
    int4* bias,
    int4* d_temp_buf, 
    algo_param_t &algo_param,
    conv_param_t &conv_param, 
    quant_param_t &quant_param,
    fuse_param_t &fuse_param,
    uint64_t workspace = (uint64_t)8*1024*1024*1024);

void PPLCUDAConvolutionForwardJitImpInt8(
    const cudaDeviceProp& device_prop,
    cudaStream_t &stream,
    CUfunction function,
    ppl::common::datatype_t type,
    int4* d_input,
    int4* d_flt,
    int4* d_output,
    int4* bias,
    int4* d_temp_buf, 
    algo_param_t &algo_param,
    conv_param_t &conv_param, 
    quant_param_t &quant_param,
    fuse_param_t &fuse_param);

#endif// __PPLCUDA_IMPLICITGEMM_CONV_H_
