// nnetflow/core/blas_helpers.hpp
#pragma once
#include "device.hpp"
#ifdef __CUDACC__
#include <cublas_v2.h>
#endif
extern "C" {
#include <cblas.h>
}
#include <vector>

// Matrix multiplication (float32)
inline void matmul(const float* A, const float* B, float* C, int M, int N, int K, DeviceType device) {
    if (device == DeviceType::CUDA) {
#ifdef __CUDACC__
        cublasHandle_t handle;
        cublasCreate(&handle);
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
        cublasDestroy(handle);
#endif
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
}
