// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <chrono>
#include <unistd.h>
#include <so3_math.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include <fins/agent/parameter_server.hpp>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define PUBFRAME_PERIOD     (20)

class LaserMapping {

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string initial_frame, body_frame;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, scan_body_pub_en = false, time_sync_en = false;
bool extrinsic_est_en = false;
int lidar_type;
bool use_imu_odometry_ = true;
int imu_window_size = 200;
std::deque<Eigen::Vector3d> imu_ang_vel_window_;
Eigen::Vector3d avg_ang_vel_ = Eigen::Vector3d::Zero();

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT{0.0, 0.0, 0.0};
vector<double>       extrinR{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

double epsi[23];

PointCloudXYZI::Ptr featsFromMap{new PointCloudXYZI()};
PointCloudXYZI::Ptr feats_undistort{new PointCloudXYZI()};
PointCloudXYZI::Ptr feats_down_body{new PointCloudXYZI()};
PointCloudXYZI::Ptr feats_down_world{new PointCloudXYZI()};
PointCloudXYZI::Ptr normvec{new PointCloudXYZI(100000, 1)};
PointCloudXYZI::Ptr laserCloudOri{new PointCloudXYZI(100000, 1)};
PointCloudXYZI::Ptr corr_normvect{new PointCloudXYZI(100000, 1)};

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body{LIDAR_SP_LEN, 0.0, 0.0}, XAxisPoint_world{LIDAR_SP_LEN, 0.0, 0.0};
V3D euler_cur;
V3D position_last{Zero3d};
V3D Lidar_T_wrt_IMU{Zero3d};
M3D Lidar_R_wrt_IMU{Eye3d};

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf_copy;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre;
shared_ptr<ImuProcess> p_imu;

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = pos_LiD(i) - LocalMap_Points.vertex_min[i];
        dist_to_map_edge[i][1] = LocalMap_Points.vertex_max[i] - pos_LiD(i);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    if(cub_needrm.size() > 0) {
        ikdtree.Delete_Point_Boxes(cub_needrm);
    }
}

public:
void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg, fins::AcqTime t) 
{
    mtx_buffer.lock();
    scan_count ++;
    // fins_node->logger->info("Received standard lidar point cloud with timestamp {}, points {}", get_time_sec(msg->header.stamp), msg->width * msg->height);
    if (get_time_sec(msg->header.stamp) < last_timestamp_lidar)
    {
        fins_node->logger->error("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(get_time_sec(msg->header.stamp));
    last_timestamp_lidar = get_time_sec(msg->header.stamp);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;

void livox_pcl_cbk(const livox_driver2::msg::CustomMsg::ConstSharedPtr &msg, fins::AcqTime t) 
{
    mtx_buffer.lock();
    scan_count ++;
    if (get_time_sec(msg->header.stamp) < last_timestamp_lidar)
    {
        fins_node->logger->error("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = get_time_sec(msg->header.stamp);
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        fins_node->logger->warn("IMU and LiDAR not Synced, IMU time: {}, lidar header time: {}", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        fins_node->logger->info("Self sync IMU and LiDAR, time diff is {}", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void process_and_publish_imu_odometry(
    const sensor_msgs::msg::Imu::ConstSharedPtr& msg_in,
    double dt,
    const input_ikfom& in,
    bool use_imu_odometry,
    fins::AcqTime t
) {
    if (!use_imu_odometry) return;

    kf_copy.predict(dt, p_imu->Q, in);
    auto imu_state = kf_copy.get_x();

    // 从IMU消息中获取原始角速度
    Eigen::Vector3d current_ang_vel(
        msg_in->angular_velocity.x,
        msg_in->angular_velocity.y,
        msg_in->angular_velocity.z
    );

    // 对角速度进行滑动窗口平滑
    imu_ang_vel_window_.push_back(current_ang_vel);
    avg_ang_vel_ += current_ang_vel;

    // 如果窗口已满，移除最旧的数据
    if (imu_ang_vel_window_.size() > imu_window_size) {
        avg_ang_vel_ -= imu_ang_vel_window_.front();
        imu_ang_vel_window_.pop_front();
    }

    // 计算平滑后的角速度
    Eigen::Vector3d smoothed_ang_vel = avg_ang_vel_ / imu_ang_vel_window_.size();

    nav_msgs::msg::Odometry imu_odometry;
    imu_odometry.header.stamp = msg_in->header.stamp;
    imu_odometry.header.frame_id = initial_frame;
    imu_odometry.child_frame_id = body_frame;

    imu_odometry.pose.pose.position.x = imu_state.pos(0);
    imu_odometry.pose.pose.position.y = imu_state.pos(1);
    imu_odometry.pose.pose.position.z = imu_state.pos(2);
    imu_odometry.pose.pose.orientation.x = imu_state.rot.coeffs()(0);
    imu_odometry.pose.pose.orientation.y = imu_state.rot.coeffs()(1);
    imu_odometry.pose.pose.orientation.z = imu_state.rot.coeffs()(2);
    imu_odometry.pose.pose.orientation.w = imu_state.rot.coeffs()(3);

    vect3 vel_body = imu_state.rot.conjugate() * imu_state.vel;
    imu_odometry.twist.twist.linear.x = vel_body(0);
    imu_odometry.twist.twist.linear.y = vel_body(1);
    imu_odometry.twist.twist.linear.z = vel_body(2);

    // 设置平滑后的角速度
    imu_odometry.twist.twist.angular.x = smoothed_ang_vel(0);
    imu_odometry.twist.twist.angular.y = smoothed_ang_vel(1);
    imu_odometry.twist.twist.angular.z = smoothed_ang_vel(2);

    // 发布里程计
    publish_imu_odometry(imu_odometry, t);
}

void imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr &msg_in, fins::AcqTime t) 
{
    // fins_node->logger->info("Received IMU message with timestamp {}", get_time_sec(msg_in->header.stamp));
    publish_count ++;

    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        get_ros_time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
    }

    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        fins_node->logger->warn("IMU loop back, clear buffer");
        imu_buffer.clear();
    }

    double dt = 0.0;
    if (last_timestamp_imu < 0.0) {
        dt = 0.01;
    } else {
        dt = timestamp - last_timestamp_imu;
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    if (p_imu->imu_need_init()) {
        return;
    }

    V3D ang_vel, acc;
    ang_vel << msg_in->angular_velocity.x, msg_in->angular_velocity.y, msg_in->angular_velocity.z;
    acc << msg_in->linear_acceleration.x, msg_in->linear_acceleration.y, msg_in->linear_acceleration.z;
    
    // Normalize acc if possible, but we need mean_acc from imu_process
    acc = acc * G_m_s2 / p_imu->get_mean_acc().norm();

    input_ikfom in;
    in.acc = acc;
    in.gyro = ang_vel;

    process_and_publish_imu_odometry(msg_in, dt, in, use_imu_odometry_, t);
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();


        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            fins_node->logger->warn("Too few input point cloud!");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }
        if(lidar_type == MARSIM)
            lidar_end_time = meas.lidar_beg_time;

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
}

PointCloudXYZI::Ptr pcl_wait_pub{new PointCloudXYZI(500000, 1)};
PointCloudXYZI::Ptr pcl_wait_save{new PointCloudXYZI()};

void publish_frame_world()
{
    if(fins_node->required("cloud"))
    {
        PointCloudXYZI::Ptr laserCloudFullRes(feats_undistort);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = initial_frame;
        fins_node->send("cloud", laserCloudmsg, fins::from_seconds(lidar_end_time));
    }
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_imu_odometry(const nav_msgs::msg::Odometry &odom, fins::AcqTime t)
{
    if (fins_node->required("odometry")) {
        fins_node->send("odometry", odom, t);
    }

    if (fins_node->required("transform")) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = odom.header.stamp;
        tf.header.frame_id = odom.header.frame_id;
        tf.child_frame_id = odom.child_frame_id;
        tf.transform.translation.x = odom.pose.pose.position.x;
        tf.transform.translation.y = odom.pose.pose.position.y;
        tf.transform.translation.z = odom.pose.pose.position.z;
        tf.transform.rotation = odom.pose.pose.orientation;
        fins_node->send("transform", tf, t);
    }
}

void publish_lidar_odometry() {
    if (fins_node->required("odometry") || fins_node->required("transform")) {
        odomAftMapped.header.frame_id = initial_frame;
        odomAftMapped.child_frame_id = body_frame;
        odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
        set_posestamp(odomAftMapped.pose);

        // 计算线速度
        vect3 vel_body = state_point.rot.conjugate() * state_point.vel;
        odomAftMapped.twist.twist.linear.x = vel_body(0);
        odomAftMapped.twist.twist.linear.y = vel_body(1);
        odomAftMapped.twist.twist.linear.z = vel_body(2);

        // 计算角速度
        static double last_lidar_end_time = 0;
        static SO3 last_rot = SO3::Identity();
        SO3 delta_rot =  state_point.rot * last_rot.conjugate();
        vect3 ang_vel_body;
        ang_vel_body.setZero();
        if (last_lidar_end_time > 0) {
            double dt = lidar_end_time - last_lidar_end_time;
            ang_vel_body = SO3::log(delta_rot) / dt;
        }
        last_rot = state_point.rot;
        last_lidar_end_time = lidar_end_time;

        odomAftMapped.twist.twist.angular.x = ang_vel_body(0);
        odomAftMapped.twist.twist.angular.y = ang_vel_body(1);
        odomAftMapped.twist.twist.angular.z = ang_vel_body(2);

        if (fins_node->required("odometry")) {
            fins_node->send("odometry", odomAftMapped, fins::from_seconds(lidar_end_time));
        }

        if (fins_node->required("transform")) {
            geometry_msgs::msg::TransformStamped tf;
            tf.header = odomAftMapped.header;
            tf.child_frame_id = odomAftMapped.child_frame_id;
            tf.transform.translation.x = odomAftMapped.pose.pose.position.x;
            tf.transform.translation.y = odomAftMapped.pose.pose.position.y;
            tf.transform.translation.z = odomAftMapped.pose.pose.position.z;
            tf.transform.rotation = odomAftMapped.pose.pose.orientation;
            fins_node->send("transform", tf, fins::from_seconds(lidar_end_time));
        }
    }
}

void publish_path()
{
    if (fins_node->required("path")) {
        set_posestamp(msg_body_pose);
        msg_body_pose.header.stamp = get_ros_time(lidar_end_time);
        msg_body_pose.header.frame_id = initial_frame;

        path.header.stamp = msg_body_pose.header.stamp;
        path.header.frame_id = initial_frame;
        path.poses.push_back(msg_body_pose);
        fins_node->send("path", path, fins::from_seconds(lidar_end_time));
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        fins_node->logger->warn("No Effective Points!");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
}

fins::Node* fins_node;

public:
LaserMapping(fins::Node* node_ptr)
    : fins_node(node_ptr)
{
    p_pre = make_shared<Preprocess>(fins_node);
    p_imu = make_shared<ImuProcess>(fins_node);
}

void initialize() {
    fins_node->logger->info("User Initialization Start");

    std::fill(epsi, epsi + 23, 0.001);

    NUM_MAX_ITERATIONS = fins::param_server().get("FastLIO.max_iteration", 4);
    filter_size_surf_min = fins::param_server().get("FastLIO.filter_size_surf", 0.5);
    filter_size_map_min = fins::param_server().get("FastLIO.filter_size_map", 0.5);
    cube_len = fins::param_server().get("FastLIO.cube_side_length", 200.0);

    fins::ParamLoader common("FastLIO.common");
    initial_frame = common.get("initial_frame", "lidar_odom");
    body_frame = common.get("body_frame", "livox_frame");
    time_sync_en = common.get("time_sync_en", false);
    time_diff_lidar_to_imu = common.get("time_offset_lidar_to_imu", 0.0);
    use_imu_odometry_ = common.get("use_imu_odometry", false);
    imu_window_size = common.get("imu_window_size", 10);

    fins::ParamLoader preprocess("FastLIO.preprocess");
    p_pre->blind = preprocess.get("blind", 0.5);
    p_pre->lidar_type = static_cast<LID_TYPE>(preprocess.get<int>("lidar_type", AVIA));
    lidar_type = p_pre->lidar_type;
    p_pre->N_SCANS = preprocess.get("scan_line", 16);
    p_pre->time_unit = static_cast<TIME_UNIT>(preprocess.get<int>("timestamp_unit", US));
    p_pre->SCAN_RATE = preprocess.get("scan_rate", 10);
    p_pre->feature_enabled = preprocess.get("feature_extract_enable", false);
    p_pre->point_filter_num = preprocess.get("point_filter_num", 2);

    fins::ParamLoader mapping("FastLIO.mapping");
    acc_cov = mapping.get("acc_cov", 0.1);
    gyr_cov = mapping.get("gyr_cov", 0.1);
    b_acc_cov = mapping.get("b_acc_cov", 0.0001);
    b_gyr_cov = mapping.get("b_gyr_cov", 0.0001);
    fov_deg = mapping.get("fov_degree", 180.0);
    DET_RANGE = mapping.get("det_range", 300.0f);
    extrinsic_est_en = mapping.get("extrinsic_est_en", false);
    extrinT = mapping.get("extrinsic_T", std::vector<double>{-0.011, -0.02329, 0.04412});
    extrinR = mapping.get("extrinsic_R", std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
    
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);

    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    
    kf.init_dyn_share(
        get_f, df_dx, df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { h_share_model(s, ekfom_data); },
        NUM_MAX_ITERATIONS, epsi);
    
    kf_copy.init_dyn_share(
        get_f, df_dx, df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { h_share_model(s, ekfom_data); },
        NUM_MAX_ITERATIONS, epsi);

    fins_node->logger->info("User Initialization Finished");
}

void loop_once() {
    if(sync_packages(Measures))  {
        if (flg_first_scan)
        {
            first_lidar_time = Measures.lidar_beg_time;
            p_imu->first_lidar_time = first_lidar_time;
            flg_first_scan = false;
            return ;
        }

        double t0 = omp_get_wtime();
        p_imu->Process(Measures, kf, feats_undistort);
        state_point = kf.get_x();
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

        if (feats_undistort->empty() || (feats_undistort == NULL))
        {
            fins_node->logger->warn("No point, skip this scan!\n");
            return ;
        }

        flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                        false : true;
        /*** Segment the map in lidar FOV ***/
        lasermap_fov_segment();

        /*** downsample the feature points in a scan ***/
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);
        feats_down_size = feats_down_body->points.size();
        /*** initialize the map kdtree ***/
        if(ikdtree.Root_Node == nullptr)
        {
            if(feats_down_size > 5)
            {
                ikdtree.set_downsample_param(filter_size_map_min);
                feats_down_world->resize(feats_down_size);
                for(int i = 0; i < feats_down_size; i++)
                {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                }
                ikdtree.Build(feats_down_world->points);
            }
            return ;
        }

        /*** ICP and iterated Kalman filter update ***/
        if (feats_down_size < 5)
        {
            fins_node->logger->warn("No point, skip this scan!\n");
            return ;
        }
        
        normvec->resize(feats_down_size);
        feats_down_world->resize(feats_down_size);

        pointSearchInd_surf.resize(feats_down_size);
        Nearest_Points.resize(feats_down_size);
        
        /*** iterated state estimation ***/
        double t_update_start = omp_get_wtime();
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV);
        state_point = kf.get_x();
        double t_update_end = omp_get_wtime();

        euler_cur = SO3ToEuler(state_point.rot);
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        geoQuat.x = state_point.rot.coeffs()[0];
        geoQuat.y = state_point.rot.coeffs()[1];
        geoQuat.z = state_point.rot.coeffs()[2];
        geoQuat.w = state_point.rot.coeffs()[3];

        if (use_imu_odometry_) {
            kf_copy = kf;
        }

        /******* Publish odometry *******/
        if (!use_imu_odometry_) {
            publish_lidar_odometry();
        }
        publish_path();

        /*** add the feature points to map kdtree ***/
        double t_incre_start = omp_get_wtime();
        map_incremental();
        double t_incre_end = omp_get_wtime();
        
        publish_frame_world();

        double t_total = omp_get_wtime() - t0;
        static int total_frame = 0;
        static double total_time = 0;
        total_frame++;
        total_time += t_total;
        if (total_frame % 10 == 0) {
            fins_node->logger->info("FAST_LIO Statistics: [Points: {}] [Update: {:.3f}ms] [Incremental: {:.3f}ms] [Total: {:.3f}ms] [Avg Total: {:.3f}ms]", 
                feats_down_size, (t_update_end - t_update_start) * 1000.0, (t_incre_end - t_incre_start) * 1000.0, t_total * 1000.0, (total_time / total_frame) * 1000.0);
        }
    }
}
};