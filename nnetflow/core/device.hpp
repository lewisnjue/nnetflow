// nnetflow/core/device.hpp
#pragma once
#include <string>
#include <stdexcept>

enum class DeviceType { CPU, CUDA };

class Device {
public:
    DeviceType type;
    int index;
    Device(DeviceType t = DeviceType::CPU, int idx = 0) : type(t), index(idx) {}
    std::string str() const {
        if (type == DeviceType::CPU) return "cpu";
        else return "cuda:" + std::to_string(index);
    }
    static Device detect_best();
};

// Device detection (simple version)
inline Device Device::detect_best() {
#ifdef __CUDACC__
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count > 0) return Device(DeviceType::CUDA, 0);
#endif
    return Device(DeviceType::CPU, 0);
}
