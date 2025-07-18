// nnetflow/core/tensor_cuda.cu
// CUDA Tensor implementation inspired by the Python Tensor class in engine.py
// This class is designed to mimic PyTorch's Tensor with strides, views, and device support


#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cassert>
#include <numeric>
#include <algorithm>
#include "device.hpp"
#include "blas_helpers.hpp"


class CudaTensor {
public:
    float* data;
    float* grad;
    std::vector<int> shape;
    std::vector<int> strides;
    int ndim;
    size_t numel;
    bool require_grad;
    Device device;

    // Constructors
    CudaTensor(const std::vector<int>& shape_, bool require_grad_ = true, Device device_ = Device::detect_best())
        : shape(shape_), require_grad(require_grad_), device(device_) {
        ndim = shape.size();
        numel = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        compute_strides();
        if (device.type == DeviceType::CUDA) {
            cudaMalloc(&data, numel * sizeof(float));
            if (require_grad) {
                cudaMalloc(&grad, numel * sizeof(float));
                cudaMemset(grad, 0, numel * sizeof(float));
            } else {
                grad = nullptr;
            }
        } else {
            data = new float[numel];
            if (require_grad) {
                grad = new float[numel]();
            } else {
                grad = nullptr;
            }
        }
    }


    // Destructor
    ~CudaTensor() {
        if (device.type == DeviceType::CUDA) {
            cudaFree(data);
            if (grad) cudaFree(grad);
        } else {
            delete[] data;
            if (grad) delete[] grad;
        }
    }

    // Compute strides for the tensor (row-major)
    void compute_strides() {
        strides.resize(ndim);
        int stride = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    // View: create a new tensor sharing the same data but with a different shape/strides
    CudaTensor view(const std::vector<int>& new_shape) const {
        CudaTensor t = *this;
        t.shape = new_shape;
        t.ndim = new_shape.size();
        t.compute_strides();
        // Data pointer is shared (shallow copy)
        return t;
    }


    // Copy data from host
    void from_host(const std::vector<float>& host_data) {
        assert(host_data.size() == numel);
        if (device.type == DeviceType::CUDA) {
            cudaMemcpy(data, host_data.data(), numel * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            std::copy(host_data.begin(), host_data.end(), data);
        }
    }

    // Copy data to host
    std::vector<float> to_host() const {
        std::vector<float> host_data(numel);
        if (device.type == DeviceType::CUDA) {
            cudaMemcpy(host_data.data(), data, numel * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            std::copy(data, data + numel, host_data.begin());
        }
        return host_data;
    }


    // Zero grad
    void zero_grad() {
        if (require_grad && grad) {
            if (device.type == DeviceType::CUDA) {
                cudaMemset(grad, 0, numel * sizeof(float));
            } else {
                std::fill(grad, grad + numel, 0.0f);
            }
        }
    }


    // Print tensor (for debugging)
    void print() const {
        std::vector<float> host = to_host();
        std::cout << "CudaTensor(device=" << device.str() << ", shape=[";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i + 1 < shape.size()) std::cout << ", ";
        }
        std::cout << "], data=[";
        for (size_t i = 0; i < host.size(); ++i) {
            std::cout << host[i];
            if (i + 1 < host.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Move tensor to another device
    CudaTensor to(DeviceType new_device) const {
        if (device.type == new_device) return *this;
        CudaTensor t(shape, require_grad, Device(new_device, 0));
        std::vector<float> host = to_host();
        t.from_host(host);
        return t;
    }

    // Matrix multiplication (matmul)
    CudaTensor matmul(const CudaTensor& other) const {
        assert(ndim == 2 && other.ndim == 2);
        int M = shape[0], K = shape[1], N = other.shape[1];
        CudaTensor out({M, N}, require_grad, device);
        ::matmul(data, other.data, out.data, M, N, K, device.type);
        return out;
    }
};
};

// More CUDA kernels and operations (add, mul, matmul, etc.) can be added as needed.
