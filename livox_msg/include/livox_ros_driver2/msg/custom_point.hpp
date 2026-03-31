#pragma once

#include <memory>
#include <cstdint>

namespace livox_ros_driver2 {
namespace msg {

struct CustomPoint {
    using SharedPtr = std::shared_ptr<CustomPoint>;
    using ConstSharedPtr = std::shared_ptr<const CustomPoint>;
    using UniquePtr = std::unique_ptr<CustomPoint>;

    uint32_t offset_time;
    float x;
    float y;
    float z;
    uint8_t reflectivity;
    uint8_t tag;
    uint8_t line;
};

} // namespace msg
} // namespace livox_ros_driver2