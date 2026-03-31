#pragma once

#include <memory>
#include <vector>
#include <array>
#include <cstdint>

#include <std_msgs/msg/header.hpp>
#include "livox_ros_driver2/msg/custom_point.hpp"

namespace livox_ros_driver2 {
namespace msg {

struct CustomMsg {
    using SharedPtr = std::shared_ptr<CustomMsg>;
    using ConstSharedPtr = std::shared_ptr<const CustomMsg>;
    using UniquePtr = std::unique_ptr<CustomMsg>;

    std_msgs::msg::Header header;
    uint64_t timebase;
    uint32_t point_num;
    uint8_t lidar_id;
    std::array<uint8_t, 3> rsvd;
    
    std::vector<CustomPoint> points;
};

} // namespace msg
} // namespace livox_ros_driver2