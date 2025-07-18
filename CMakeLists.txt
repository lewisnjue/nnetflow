# CMake configuration for nnetflow CUDA extension
cmake_minimum_required(VERSION 3.18)
project(nnetflow LANGUAGES CXX CUDA)

# Find pybind11
find_package(pybind11 REQUIRED)
find_package(BLAS REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Add core sources
set(CORE_SRC
    nnetflow/core/tensor_cuda.cu
    nnetflow/core/tensor_cuda_pybind.cpp
)

pybind11_add_module(tensor_cuda MODULE ${CORE_SRC})
target_link_libraries(tensor_cuda PRIVATE CUDA::cudart CUDA::cublas ${BLAS_LIBRARIES})
set_target_properties(tensor_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(tensor_cuda PRIVATE nnetflow/core)

if (UNIX)
    set_target_properties(tensor_cuda PROPERTIES PREFIX "" SUFFIX ".so")
endif()
