
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor_cuda.cu"
#include "device.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tensor_cuda, m) {
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA);

    py::class_<Device>(m, "Device")
        .def(py::init<DeviceType, int>(), py::arg("type") = DeviceType::CPU, py::arg("index") = 0)
        .def("str", &Device::str)
        .def_static("detect_best", &Device::detect_best);

    py::class_<CudaTensor>(m, "CudaTensor")
        .def(py::init<const std::vector<int>&, bool, Device>(), py::arg("shape"), py::arg("require_grad") = true, py::arg("device") = Device(DeviceType::CPU, 0))
        .def("from_host", &CudaTensor::from_host)
        .def("to_host", &CudaTensor::to_host)
        .def("zero_grad", &CudaTensor::zero_grad)
        .def("view", &CudaTensor::view)
        .def("print", &CudaTensor::print)
        .def("to", &CudaTensor::to)
        .def("matmul", &CudaTensor::matmul)
        .def_property_readonly("shape", [](const CudaTensor& t) { return t.shape; })
        .def_property_readonly("strides", [](const CudaTensor& t) { return t.strides; })
        .def_property_readonly("ndim", [](const CudaTensor& t) { return t.ndim; })
        .def_property_readonly("numel", [](const CudaTensor& t) { return t.numel; })
        .def_property_readonly("require_grad", [](const CudaTensor& t) { return t.require_grad; })
        .def_property_readonly("device", [](const CudaTensor& t) { return t.device; });
}
