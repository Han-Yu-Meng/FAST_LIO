/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include <atomic>
#include <condition_variable>
#include <fins/node.hpp>
#include <memory>
#include <mutex>
#include <thread>

#include "laserMapping.cpp"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

class FastLIO : public fins::Node {
public:
  void define() override {
    set_name("FastLIO");
    set_description("Fast LIO-2 SLAM Node");
    set_category("SLAM");

    register_input<0, sensor_msgs::msg::Imu>("imu", &FastLIO::on_imu);

    register_input<1, livox_ros_driver2::msg::CustomMsg>("lidar",
                                                         &FastLIO::on_livox);

    register_input<2, sensor_msgs::msg::PointCloud2>("lidar_standard",
                                                     &FastLIO::on_lidar);

    register_output<0, sensor_msgs::msg::PointCloud2>("cloud");
    register_output<1, nav_msgs::msg::Path>("path");
    register_output<2, nav_msgs::msg::Odometry>("odometry");
    register_output<3, geometry_msgs::msg::TransformStamped>("transform");
  }

  void initialize() override {
    logger->info("Initializing FastLIO Node...");
    mapper_ = std::make_shared<LaserMapping>(this);
    mapper_->initialize();

    is_running_ = true;
    has_new_data_ = false;
    mapping_thread_ = std::thread(&FastLIO::mapping_worker_loop, this);

    logger->info("FastLIO Node initialized with independent mapping thread.");
  }

  void deinitialize() {
    is_running_ = false;
    trigger_cv_.notify_all();

    if (mapping_thread_.joinable()) {
      mapping_thread_.join();
    }

    mapper_.reset();
  }

  ~FastLIO() { deinitialize(); }

  void run() override {}
  void pause() override {}
  void reset() override {}

  void on_lidar(const fins::Msg<sensor_msgs::msg::PointCloud2> &msg) {
    if (mapper_) {
      // FINS_TIME_BLOCK(logger, "Lidar Callback");
      mapper_->standard_pcl_cbk(msg.ptr());
    }
  }

  void on_livox(const fins::Msg<livox_ros_driver2::msg::CustomMsg> &msg) {
    if (mapper_) {
      // FINS_TIME_BLOCK(logger, "Livox Callback");
      mapper_->livox_pcl_cbk(msg.ptr());
    }
  }

  void on_imu(const fins::Msg<sensor_msgs::msg::Imu> &msg) {
    if (mapper_) {
      // FINS_TIME_BLOCK(logger, "IMU Callback");
      mapper_->imu_cbk(msg.ptr());
    }
  }

private:
  void notify_backend() {
    {
      std::lock_guard<std::mutex> lock(trigger_mtx_);
      has_new_data_ = true;
    }
    trigger_cv_.notify_one();
  }

  void mapping_worker_loop() {
    logger->info("Backend mapping worker started.");

    while (is_running_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));

      if (!is_running_)
        break;

      if (mapper_) {
        // FINS_TIME_BLOCK(logger, "LIVMapper Trigger");
        mapper_->loop_once();
      }
    }
    logger->info("Backend mapping worker exiting.");
  }

  std::shared_ptr<LaserMapping> mapper_;

  std::thread mapping_thread_;
  std::mutex trigger_mtx_;
  std::condition_variable trigger_cv_;
  std::atomic<bool> is_running_{false};
  bool has_new_data_{false};
};

EXPORT_NODE(FastLIO)