#pragma once

// Bridges old (livox_ros_driver2/msg) and new (livox_interfaces2/msg) driver layouts.
#ifdef LIVOX_INTERFACES2
#include <livox_interfaces2/msg/custom_msg.hpp>
namespace livox_ros_driver2 {
namespace msg {
using livox_interfaces2::msg::CustomMsg;
using livox_interfaces2::msg::CustomPoint;
}  // namespace msg
}  // namespace livox_ros_driver2
#else
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif