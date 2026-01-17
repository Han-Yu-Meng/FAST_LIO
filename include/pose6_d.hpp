#pragma once

struct Pose6D
{
  double offset_time; // the offset time of IMU measurement w.r.t the first lidar point
  double acc[3];     // the preintegrated total acceleration (global frame) at the Lidar origin
  double gyr[3];     // the unbiased angular velocity (body frame) at the Lidar origin
  double vel[3];     // the preintegrated velocity (global frame) at the Lidar origin
  double pos[3];     // the preintegrated position (global frame) at the Lidar origin
  double rot[9];     // the preintegrated rotation (global frame) at the Lidar origin
};