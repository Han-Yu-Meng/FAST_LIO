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
#include <algorithm>
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
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64.hpp>
#include <pcl/common/transforms.h>
#include <limits>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

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

string initial_frame, base_link_frame, base_lidar_frame;
M3D base_R_lidar = Eye3d;
V3D base_T_lidar = Zero3d;
bool static_transform_received = false;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
double imu_time_tolerance = 0.01;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, scan_body_pub_en = false, time_sync_en = false;
bool extrinsic_est_en = false;
int lidar_type;
bool use_imu_odometry_ = true;
bool planar_constraint_ = false;
int imu_window_size = 200;
std::deque<Eigen::Vector3d> imu_ang_vel_window_;
Eigen::Vector3d avg_ang_vel_ = Eigen::Vector3d::Zero();

// ---- 定位模式 IMU 高频输出（开关默认开） ----
// fixed_map 定位时，kf_copy 同时被 IMU 回调线程（process_and_publish_imu_odometry
// 里 predict / get_x）与后端 mapping 线程（kf_copy = kf）访问。用一把锁保护
// kf_copy 及高频发布要读的共享状态快照，消除两线程间的数据竞争。
bool imu_rate_output_ = true;          // FastLIO.localization.imu_rate_output，默认 true
std::mutex kf_copy_mutex;
// 高频发布需要的共享状态，由后端线程在锁内刷新，IMU 线程在锁内拷贝后使用：
struct ImuRateSnapshot {
    bool fixed_map_enabled = false;
    bool locked = false;               // fixed_map_active && LOCKED && bad_matches==0
    // 是否启用 IMU 高频输出。建图模式恒 true（沿用旧行为）；定位模式仅 LOCKED
    // 之后为 true——初始化/未 LOCKED 期间必须保持 scan-rate，否则 BBS 依赖的
    // $T_{odom}^{base}$ TF 时间戳会被 IMU 时间戳顶掉，导致永远 WAITING_INITIALIZATION。
    bool imu_rate_live = false;
    Eigen::Isometry3d map_to_odom = Eigen::Isometry3d::Identity();
    M3D base_R_lidar = Eye3d;
    V3D base_T_lidar = Zero3d;
};
ImuRateSnapshot imu_rate_snapshot;

// 组装 map 系下带 6x6 协方差的定位位姿消息。
// scan-rate 路径传 kf.get_x()/get_P()，IMU 高频路径传 kf_copy 的副本，
// 两条路径共用同一套位姿/协方差公式，避免不一致。定义在文件前部，供两处调用。
template <typename CovT>
geometry_msgs::msg::PoseWithCovarianceStamped compose_localization_pose(
    const state_ikfom &s, const CovT &P,
    const Eigen::Isometry3d &map_to_odom_,
    const M3D &R_bl, const V3D &t_bl,
    const fins::AcqTime &ts)
{
    const M3D R = s.rot.toRotationMatrix();
    const M3D R_L_I = s.offset_R_L_I.toRotationMatrix();
    const V3D lever = R_L_I * (-R_bl.transpose() * t_bl) + s.offset_T_L_I;
    Eigen::Isometry3d odom_base = Eigen::Isometry3d::Identity();
    odom_base.linear() = R_bl * R * R_L_I * R_bl.transpose();
    odom_base.translation() = R_bl * (R * lever + s.pos) + t_bl;
    Eigen::Isometry3d global = map_to_odom_ * odom_base;

    geometry_msgs::msg::PoseWithCovarianceStamped msg;
    msg.header.stamp = fins::to_ros_msg_time(ts);
    msg.header.frame_id = "map";
    msg.pose.pose.position.x = global.translation().x();
    msg.pose.pose.position.y = global.translation().y();
    msg.pose.pose.position.z = global.translation().z();
    Eigen::Quaterniond q(global.linear());
    msg.pose.pose.orientation.w = q.w();
    msg.pose.pose.orientation.x = q.x();
    msg.pose.pose.orientation.y = q.y();
    msg.pose.pose.orientation.z = q.z();

    Eigen::Matrix<double,6,23> J = Eigen::Matrix<double,6,23>::Zero();
    M3D R_map_world = map_to_odom_.linear() * R_bl;
    J.block<3,3>(0,0) = R_map_world;
    M3D lever_cross;
    lever_cross << SKEW_SYM_MATRX(lever);
    J.block<3,3>(0,3) = -R_map_world * R * lever_cross;
    J.block<3,3>(3,3) = R_map_world * R;
    Eigen::Matrix<double,6,6> covariance = J * P * J.transpose();
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            msg.pose.covariance[r*6+c] = covariance(r,c);
    return msg;
}

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT{0.0, 0.0, 0.0};
vector<double>       extrinR{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;
deque<fins::AcqTime>              acq_time_buffer;

double epsi[23];

PointCloudXYZI::Ptr featsFromMap{new PointCloudXYZI()};
PointCloudXYZI::Ptr feats_undistort{new PointCloudXYZI()};
PointCloudXYZI::Ptr feats_down_body{new PointCloudXYZI()};
PointCloudXYZI::Ptr feats_down_world{new PointCloudXYZI()};
PointCloudXYZI::Ptr normvec{new PointCloudXYZI(30000, 1)};
PointCloudXYZI::Ptr laserCloudOri{new PointCloudXYZI(30000, 1)};
PointCloudXYZI::Ptr corr_normvect{new PointCloudXYZI(30000, 1)};

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

std::unique_ptr<KD_TREE<PointType>> ikdtree{new KD_TREE<PointType>()};
std::unique_ptr<KD_TREE<PointType>> fixed_tree;
PointCloudXYZI::Ptr prior_map{new PointCloudXYZI()};
bool fixed_map_enabled = false, fixed_map_active = false;
bool has_alignment = false, alignment_pending = false;
std::mutex alignment_mutex;
geometry_msgs::msg::TransformStamped pending_alignment;
Eigen::Isometry3d map_to_odom = Eigen::Isometry3d::Identity();
std::string tracking_status = "WAITING_INITIALIZATION";
int bad_matches = 0, good_matches = 0;
int min_effective_points = 30, failures_before_lost = 5, good_frames_to_lock = 3;
double max_mean_residual = 0.2, max_update_translation = 0.3, max_update_rotation_deg = 10.0;
double max_alignment_age = 10.0;

// ---- 定位置信度 [0,1]（std_msgs/Float64 /localization_confidence）----
// 由本帧 scan-to-map 匹配的现成指标加权合成，复用上面的质量门限做归一化分母
// （1 表示「离门限越远越好」），再乘状态系数并做非对称 EMA：变差快速下降、
// 恢复平滑上升。非 fixed_map / 未 LOCKED 归零，供下游判断位姿是否可信。
double localization_confidence_ = 0.0;
double conf_jump_ = 0.0, conf_turn_ = 0.0;   // 质量门控块内的本帧跳变，供发布读取
bool   conf_has_quality_ = false;            // 是否已跑过一次质量门控
static constexpr double kConfAlphaUp = 0.2;  // 上升（恢复）平滑系数
static constexpr double kConfAlphaDown = 0.5;// 下降（退化）平滑系数，下降更快


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

void RGBpointBodyLidarToBase(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_base(base_R_lidar * p_body_lidar + base_T_lidar);
    po->x = p_body_base(0);
    po->y = p_body_base(1);
    po->z = p_body_base(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree->acquire_removed_points(points_history);
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
        ikdtree->Delete_Point_Boxes(cub_needrm);
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
        acq_time_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    publish_raw_body(ptr, t);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(get_time_sec(msg->header.stamp));
    acq_time_buffer.push_back(t);
    last_timestamp_lidar = get_time_sec(msg->header.stamp);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;

void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg, fins::AcqTime t) 
{
    mtx_buffer.lock();
    scan_count ++;
    if (get_time_sec(msg->header.stamp) < last_timestamp_lidar)
    {
        fins_node->logger->error("lidar loop back, clear buffer");
        lidar_buffer.clear();
        acq_time_buffer.clear();
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
    publish_raw_body(ptr, t);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    acq_time_buffer.push_back(t);
    
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

    // kf_copy 同时被本 IMU 回调线程与后端 mapping 线程（kf_copy = kf）访问，
    // 必须在同一把锁内完成 predict / 取状态 / 取协方差 / 读定位快照 / 组装消息。
    // 锁内只做纯 Eigen 矩阵运算（微秒级，无 I/O），锁外才发布，临界区极短。
    state_ikfom imu_state;
    bool pub_localization_pose = false;
    geometry_msgs::msg::PoseWithCovarianceStamped loc_msg;
    {
        std::lock_guard<std::mutex> lock(kf_copy_mutex);
        // 定位模式未 LOCKED 时不发布任何 IMU 频率的 odom/TF/localization_pose：
        // 该阶段由 scan-rate 的 publish_lidar_odometry 独占输出，避免用 IMU 时间戳
        // 覆盖 BBS 初始化所需的扫描时间戳 TF 配对（否则会卡在 WAITING_INITIALIZATION）。
        // 建图模式（fixed_map_enabled=false）不受此限制，保持原有 IMU 高频行为。
        bool imu_rate_gate = !imu_rate_snapshot.fixed_map_enabled
                             || imu_rate_snapshot.imu_rate_live;
        if (!imu_rate_gate) return;
        kf_copy.predict(dt, p_imu->Q, in);
        imu_state = kf_copy.get_x();
        // fixed_map 定位且已 LOCKED：用 IMU 预测状态 + 常量 map->odom 外推，
        // 让 /localization_pose 也达到 IMU 频率；协方差用 kf_copy 传播的 P。
        pub_localization_pose = imu_rate_output_ && imu_rate_snapshot.fixed_map_enabled
            && imu_rate_snapshot.locked && fins_node->required("localization_pose");
        if (pub_localization_pose) {
            loc_msg = compose_localization_pose(
                imu_state, kf_copy.get_P(), imu_rate_snapshot.map_to_odom,
                imu_rate_snapshot.base_R_lidar, imu_rate_snapshot.base_T_lidar, t);
        }
    }

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

    // Remove bias from angular velocity for odometry output
    smoothed_ang_vel -= imu_state.bg;

    // Transform imu_state.pos and imu_state.rot to base_link_frame
    // T_odom_base = T_base_lidar * T_lidar_init_lidar_curr * T_lidar_base
    M3D R_L_I = imu_state.offset_R_L_I.toRotationMatrix();
    M3D R_I_W = imu_state.rot.toRotationMatrix();
    M3D R_base_inv = base_R_lidar.transpose();

    M3D R_odom_base = base_R_lidar * R_I_W * R_L_I * R_base_inv;
    V3D t_odom_base = base_R_lidar * (R_I_W * (R_L_I * (-R_base_inv * base_T_lidar) + imu_state.offset_T_L_I) + imu_state.pos) + base_T_lidar;
    apply_planar_constraint(R_odom_base, t_odom_base);
    Eigen::Quaterniond q_base(R_odom_base);

    nav_msgs::msg::Odometry imu_odometry;
    imu_odometry.header.stamp = fins::to_ros_msg_time(t);
    imu_odometry.header.frame_id = initial_frame;
    imu_odometry.child_frame_id = base_link_frame;

    imu_odometry.pose.pose.position.x = t_odom_base(0);
    imu_odometry.pose.pose.position.y = t_odom_base(1);
    imu_odometry.pose.pose.position.z = t_odom_base(2);
    imu_odometry.pose.pose.orientation.x = q_base.x();
    imu_odometry.pose.pose.orientation.y = q_base.y();
    imu_odometry.pose.pose.orientation.z = q_base.z();
    imu_odometry.pose.pose.orientation.w = q_base.w();

    // 计算线速度
    vect3 vel_imu_in_IMU = imu_state.rot.conjugate() * imu_state.vel; // vel_imu

    // 刚体速度修正：v_base = v_imu + omega_IMU × (p_base - p_imu)，均在 IMU 体系下
    V3D t_base_in_IMU = imu_state.offset_R_L_I.toRotationMatrix() * (-base_R_lidar.transpose() * base_T_lidar)
                        + imu_state.offset_T_L_I;
    vect3 vel_base_in_IMU = vel_imu_in_IMU + smoothed_ang_vel.cross(t_base_in_IMU);

    // 再从 IMU 体系旋转到 base_link 体系
    M3D R_L_I_T = R_L_I.transpose();
    vect3 vel_base = base_R_lidar * R_L_I_T * vel_base_in_IMU; 
    imu_odometry.twist.twist.linear.x = vel_base(0);
    imu_odometry.twist.twist.linear.y = vel_base(1);
    imu_odometry.twist.twist.linear.z = planar_constraint_ ? 0.0 : vel_base(2);

    // 角速度转到 base_link_frame
    vect3 ang_vel_base = base_R_lidar * R_L_I_T * smoothed_ang_vel;
    imu_odometry.twist.twist.angular.x = planar_constraint_ ? 0.0 : ang_vel_base(0);
    imu_odometry.twist.twist.angular.y = planar_constraint_ ? 0.0 : ang_vel_base(1);
    imu_odometry.twist.twist.angular.z = ang_vel_base(2);

    // 发布里程计（IMU 频率）
    publish_imu_odometry(imu_odometry, t);
    // 发布 map 系全局定位位姿（IMU 频率，仅 fixed_map 且 LOCKED 时）
    if (pub_localization_pose) fins_node->send("localization_pose", loc_msg, t);
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

    if (!use_imu_odometry_) return;
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
        meas.acq_time = acq_time_buffer.front();


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

    if (last_timestamp_imu < lidar_end_time - imu_time_tolerance) // IMU Time Tolerance from config
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
    acq_time_buffer.pop_front();
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

    ikdtree->Add_Points(PointToAdd, true);
    ikdtree->Add_Points(PointNoNeedDownsample, false);
}

void publish_frame_world(const fins::AcqTime &acq_time)
{   
    auto t = fins_node->recorder("frame", acq_time);
    if(fins_node->required("cloud"))
    {
        PointCloudXYZI::Ptr laserCloudFullRes(feats_undistort);
        int size = laserCloudFullRes->points.size();
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudWorld(new pcl::PointCloud<pcl::PointXYZI>(size, 1));

        M3D R_L_I = state_point.offset_R_L_I.toRotationMatrix();
        M3D R_I_W = state_point.rot.toRotationMatrix();
        M3D R_odom_lidar = base_R_lidar * R_I_W * R_L_I;
        V3D t_odom_lidar = base_R_lidar * (R_I_W * state_point.offset_T_L_I + state_point.pos) + base_T_lidar;

        // #pragma omp parallel for schedule(dynamic, 512)
        for (int i = 0; i < size; i++)
        {
            const auto &p_in = laserCloudFullRes->points[i];
            auto &p_out = laserCloudWorld->points[i];
            
            p_out.x = R_odom_lidar(0,0) * p_in.x + R_odom_lidar(0,1) * p_in.y + R_odom_lidar(0,2) * p_in.z + t_odom_lidar(0);
            p_out.y = R_odom_lidar(1,0) * p_in.x + R_odom_lidar(1,1) * p_in.y + R_odom_lidar(1,2) * p_in.z + t_odom_lidar(1);
            p_out.z = R_odom_lidar(2,0) * p_in.x + R_odom_lidar(2,1) * p_in.y + R_odom_lidar(2,2) * p_in.z + t_odom_lidar(2);
            p_out.intensity = p_in.intensity;
        }
        
        laserCloudWorld->header.frame_id = initial_frame;
        laserCloudWorld->header.stamp = fins::to_microseconds(acq_time);
        
        fins_node->send("cloud", laserCloudWorld, acq_time);
    }
}

void publish_raw_body(const PointCloudXYZI::Ptr& ptr, const fins::AcqTime &acq_time)
{
    if(fins_node->required("cloud_body") && static_transform_received)
    {
        int size = ptr->points.size();
        pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudBody(new pcl::PointCloud<pcl::PointXYZI>(size, 1));

        for (int i = 0; i < size; i++)
        {
            const auto &p_in = ptr->points[i];
            auto &p_out = laserCloudBody->points[i];
            
            V3D p_L(p_in.x, p_in.y, p_in.z);
            V3D p_B = base_R_lidar * p_L + base_T_lidar;
            
            p_out.x = p_B(0);
            p_out.y = p_B(1);
            p_out.z = p_B(2);
            p_out.intensity = p_in.intensity;
        }
        
        laserCloudBody->header.frame_id = base_link_frame;
        laserCloudBody->header.stamp = fins::to_microseconds(acq_time);
        fins_node->send("cloud_body", laserCloudBody, acq_time);
    }
}

void apply_planar_constraint(M3D &R, V3D &t) {
	if (!planar_constraint_) return;
	double yaw = std::atan2(R(1,0), R(0,0));
	R = M3D::Identity();
	R(0,0) = std::cos(yaw); R(0,1) = -std::sin(yaw);
	R(1,0) = std::sin(yaw); R(1,1) =  std::cos(yaw);
	t(2) = 0.0;
}

template<typename T>
void set_posestamp(T & out)
{
    if (planar_constraint_) {
        // Force z=0 and use only yaw from quaternion
        out.pose.position.x = state_point.pos(0);
        out.pose.position.y = state_point.pos(1);
        out.pose.position.z = 0.0;

        // Extract yaw from geoQuat and set roll=pitch=0
        double siny_cosp = 2.0 * (geoQuat.w * geoQuat.z + geoQuat.x * geoQuat.y);
        double cosy_cosp = 1.0 - 2.0 * (geoQuat.y * geoQuat.y + geoQuat.z * geoQuat.z);
        double yaw = std::atan2(siny_cosp, cosy_cosp);

        double cy = std::cos(yaw * 0.5);
        double sy = std::sin(yaw * 0.5);
        out.pose.orientation.x = 0.0;
        out.pose.orientation.y = 0.0;
        out.pose.orientation.z = sy;
        out.pose.orientation.w = cy;
    } else {
        out.pose.position.x = state_point.pos(0);
        out.pose.position.y = state_point.pos(1);
        out.pose.position.z = state_point.pos(2);
        out.pose.orientation.x = geoQuat.x;
        out.pose.orientation.y = geoQuat.y;
        out.pose.orientation.z = geoQuat.z;
        out.pose.orientation.w = geoQuat.w;
    }
}

void publish_imu_odometry(const nav_msgs::msg::Odometry &odom, fins::AcqTime t)
{
    if (fins_node->required("odometry")) {
        fins_node->send("odometry", odom, t);
    }

    if (fins_node->required("$T_{odom}^{base}$")) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = odom.header.stamp;
        tf.header.frame_id = odom.header.frame_id;
        tf.child_frame_id = odom.child_frame_id;
        tf.transform.translation.x = odom.pose.pose.position.x;
        tf.transform.translation.y = odom.pose.pose.position.y;
        tf.transform.translation.z = odom.pose.pose.position.z;
        tf.transform.rotation = odom.pose.pose.orientation;
        fins_node->send("$T_{odom}^{base}$", tf, t);
    }
}

void publish_lidar_odometry(const fins::AcqTime &acq_time) {
    if (fins_node->required("odometry") || fins_node->required("$T_{odom}^{base}$")) {
        odomAftMapped.header.frame_id = initial_frame;
        odomAftMapped.child_frame_id = base_link_frame;
        odomAftMapped.header.stamp = fins::to_ros_msg_time(acq_time);

        // Transform state_point.pos and state_point.rot to base_link_frame
        // T_odom_base = T_base_lidar * T_lidar_init_lidar_curr * T_lidar_base
        M3D R_odom_base = base_R_lidar * state_point.rot.toRotationMatrix() * state_point.offset_R_L_I.toRotationMatrix() * base_R_lidar.transpose();
        V3D t_odom_base = base_R_lidar * (state_point.rot.toRotationMatrix() * (state_point.offset_R_L_I.toRotationMatrix() * (-base_R_lidar.transpose() * base_T_lidar) + state_point.offset_T_L_I) + state_point.pos) + base_T_lidar;
        apply_planar_constraint(R_odom_base, t_odom_base);
        Eigen::Quaterniond q_base(R_odom_base);

        odomAftMapped.pose.pose.position.x = t_odom_base(0);
        odomAftMapped.pose.pose.position.y = t_odom_base(1);
        odomAftMapped.pose.pose.position.z = t_odom_base(2);
        odomAftMapped.pose.pose.orientation.x = q_base.x();
        odomAftMapped.pose.pose.orientation.y = q_base.y();
        odomAftMapped.pose.pose.orientation.z = q_base.z();
        odomAftMapped.pose.pose.orientation.w = q_base.w();

        // 计算线速度
        // state_point.vel 是 IMU 原点在世界系下的速度，先转到 IMU 体系
        vect3 vel_imu_in_IMU = state_point.rot.conjugate() * state_point.vel;

        // 计算角速度：使用相邻两帧 lidar 时间戳计算 dt，从而得到真实角速度 (rad/s)
        static SO3 last_rot = SO3::Identity();
        static double last_lidar_time = 0;
        double dt_ang = lidar_end_time - last_lidar_time;
        SO3 delta_rot = state_point.rot * last_rot.conjugate();
        last_rot = state_point.rot;
        double current_lidar_time = lidar_end_time;
        last_lidar_time = current_lidar_time;

        // SO3::log(R_new * R_old^T) / dt 得到世界系角速度，再转到 IMU 体系
        vect3 ang_vel_world = (dt_ang > 1e-6) ? vect3(SO3::log(delta_rot) / dt_ang) : vect3::Zero();
        vect3 omega_IMU = state_point.rot.toRotationMatrix().transpose() * ang_vel_world;

        // 刚体速度修正：v_base = v_imu + omega_IMU × (p_base - p_imu)，均在 IMU 体系下
        // p_base 在 IMU 体系中的位置：先从 base 转到 lidar，再从 lidar 转到 IMU
        V3D t_base_in_IMU = state_point.offset_R_L_I.toRotationMatrix() * (-base_R_lidar.transpose() * base_T_lidar)
                            + state_point.offset_T_L_I;
        vect3 vel_base_in_IMU = vel_imu_in_IMU + omega_IMU.cross(t_base_in_IMU);

        // 再从 IMU 体系旋转到 base_link 体系：R_base_IMU = base_R_lidar * R_LI^T
        vect3 vel_base = base_R_lidar * state_point.offset_R_L_I.toRotationMatrix().transpose() * vel_base_in_IMU;

        odomAftMapped.twist.twist.linear.x = vel_base(0);
        odomAftMapped.twist.twist.linear.y = vel_base(1);
        odomAftMapped.twist.twist.linear.z = planar_constraint_ ? 0.0 : vel_base(2);

        // 角速度从 IMU 体系转到 base_link_frame：R_BI = R_BL * R_LI^T
        vect3 ang_vel_base = base_R_lidar * state_point.offset_R_L_I.toRotationMatrix().transpose() * omega_IMU;

        odomAftMapped.twist.twist.angular.x = planar_constraint_ ? 0.0 : ang_vel_base(0);
        odomAftMapped.twist.twist.angular.y = planar_constraint_ ? 0.0 : ang_vel_base(1);
        odomAftMapped.twist.twist.angular.z = ang_vel_base(2);

        if (fins_node->required("odometry")) {
            fins_node->send("odometry", odomAftMapped, acq_time);
        }

        if (fins_node->required("$T_{odom}^{base}$")) {
            geometry_msgs::msg::TransformStamped tf;
            tf.header = odomAftMapped.header;
            tf.child_frame_id = odomAftMapped.child_frame_id;
            tf.transform.translation.x = odomAftMapped.pose.pose.position.x;
            tf.transform.translation.y = odomAftMapped.pose.pose.position.y;
            tf.transform.translation.z = odomAftMapped.pose.pose.position.z;
            tf.transform.rotation = odomAftMapped.pose.pose.orientation;
            fins_node->send("$T_{odom}^{base}$", tf, acq_time);
        }
    }
}

void publish_path(const fins::AcqTime &acq_time)
{
    if (fins_node->required("path")) {
        set_posestamp(msg_body_pose);
        msg_body_pose.header.stamp = fins::to_ros_msg_time(acq_time);
        msg_body_pose.header.frame_id = initial_frame;

        path.header.stamp = msg_body_pose.header.stamp;
        path.header.frame_id = initial_frame;
        path.poses.push_back(msg_body_pose);
        fins_node->send("path", path, acq_time);
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

        static thread_local vector<float> pointSearchSqDis;
        if (pointSearchSqDis.size() != NUM_MATCH_POINTS) {
            pointSearchSqDis.resize(NUM_MATCH_POINTS);
        }

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            (fixed_map_active ? fixed_tree.get() : ikdtree.get())->Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
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
            laserCloudOri->push_back(feats_down_body->points[i]);
            corr_normvect->push_back(normvec->points[i]);
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

void update_transform(const geometry_msgs::msg::TransformStamped &msg)
{
    base_T_lidar = V3D(msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z);
    base_R_lidar = Eigen::Quaterniond(msg.transform.rotation.w, msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z).toRotationMatrix();
    static_transform_received = true;
    // fins_node->logger->info("Static transform from {} to {} updated.", msg.header.frame_id, msg.child_frame_id);
}

// Runs on the FINS input thread: defer all tree/filter operations to the backend.
void queue_initial_alignment(const geometry_msgs::msg::TransformStamped &msg) {
    std::lock_guard<std::mutex> lock(alignment_mutex);
    pending_alignment = msg;
    alignment_pending = true;
}

void publish_tracking_status(const fins::AcqTime &ts) {
    if (!fixed_map_enabled) return;
    std_msgs::msg::String msg;
    msg.data = tracking_status;
    fins_node->send("localization_status", msg, ts);
}

void lose_tracking(const fins::AcqTime &ts) {
    fixed_map_active = false;
    good_matches = bad_matches = 0;
    tracking_status = "LOST";
    // A new temporary local map is needed for fresh BBS accumulation. Destroying
    // the old tree joins its rebuild worker; do not Build() over an active tree.
    ikdtree.reset(new KD_TREE<PointType>());
    Localmap_Initialized = false;
    fins_node->logger->warn("[FixedMap] LOST; suppressing global pose, requesting BBS initialization");
    publish_tracking_status(ts);
}

void reject_empty_scan(const fins::AcqTime &ts) {
    if (!fixed_map_active) return;
    good_matches = 0;
    tracking_status = "DEGRADED";
    if (++bad_matches >= failures_before_lost) lose_tracking(ts);
    publish_tracking_status(ts);
}

void apply_pending_alignment(const fins::AcqTime &ts) {
    if (!fixed_map_enabled) return;
    geometry_msgs::msg::TransformStamped msg;
    {
        std::lock_guard<std::mutex> lock(alignment_mutex);
        if (!alignment_pending) return;
        msg = pending_alignment;
        alignment_pending = false;
    }
    if (fixed_map_active) return;  // no periodic external correction while tracking
    double age = fins::to_seconds(ts) - get_time_sec(msg.header.stamp);
    Eigen::Quaterniond q(msg.transform.rotation.w, msg.transform.rotation.x,
                         msg.transform.rotation.y, msg.transform.rotation.z);
    V3D t(msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z);
    if (msg.header.frame_id != "map" || msg.child_frame_id != initial_frame ||
        !q.coeffs().allFinite() || q.norm() < 1e-6 || !t.allFinite() ||
        age < -0.1 || age > max_alignment_age || !static_transform_received) {
        fins_node->logger->warn("[FixedMap] Rejected initialization: invalid frame/transform, missing extrinsics or age={}s", age);
        tracking_status = "INITIALIZATION_REJECTED";
        publish_tracking_status(ts);
        tracking_status = "WAITING_INITIALIZATION";
        return;
    }
    Eigen::Isometry3d alignment = Eigen::Isometry3d::Identity();
    alignment.linear() = q.normalized().toRotationMatrix();
    alignment.translation() = t;
    // Existing FAST-LIO publishes odom points as B * p_internal, where B is
    // base->lidar. Express the immutable prior in that SAME internal world:
    // p_internal = B^-1 * (map->odom)^-1 * p_map.
    // The EKF state, velocity, gravity, covariance and IMU history stay intact.
    Eigen::Isometry3d B = Eigen::Isometry3d::Identity();
    B.linear() = base_R_lidar; B.translation() = base_T_lidar;
    PointCloudXYZI local_prior;
    pcl::transformPointCloud(*prior_map, local_prior, (B.inverse()*alignment.inverse()).matrix().cast<float>());
    auto tree = std::make_unique<KD_TREE<PointType>>();
    tree->Build(local_prior.points);
    fixed_tree = std::move(tree);
    map_to_odom = alignment;
    has_alignment = true;
    fixed_map_active = true;
    good_matches = bad_matches = 0;
    tracking_status = "TRACKING";
    fins_node->logger->info("[FixedMap] Initialized fixed prior: {} points, alignment age {:.3f}s; periodic GICP disabled", prior_map->size(), age);
    publish_tracking_status(ts);
}

void publish_fixed_pose(const fins::AcqTime &ts) {
    if (!fixed_map_enabled) return;
    publish_tracking_status(ts);
    // Keep the existing single-parent TF tree. This transform stays constant
    // throughout tracking; corrections now enter the scan-to-map EKF itself.
    if (has_alignment) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = fins::to_ros_msg_time(ts);
        tf.header.frame_id = "map"; tf.child_frame_id = initial_frame;
        tf.transform.translation.x = map_to_odom.translation().x();
        tf.transform.translation.y = map_to_odom.translation().y();
        tf.transform.translation.z = map_to_odom.translation().z();
        Eigen::Quaterniond q(map_to_odom.linear());
        tf.transform.rotation.w=q.w(); tf.transform.rotation.x=q.x();
        tf.transform.rotation.y=q.y(); tf.transform.rotation.z=q.z();
        fins_node->send("$T_{map}^{odom}$", tf, ts);
    }
    // 仅当 IMU 高频真正生效（LOCKED，imu_rate_live）时，localization_pose 才由
    // IMU 回调按 IMU 频率发布，这里跳过 scan-rate 版本避免同一话题两种频率。
    // 未 LOCKED / DEGRADED / LOST 期间 imu_rate_live=false，IMU 回调不发布，
    // 此时保留 scan-rate 路径，保持与旧版一致（不出现发布空窗）。
    if (imu_rate_snapshot.imu_rate_live) return;
    if (!fixed_map_active || tracking_status != "LOCKED" || bad_matches != 0) return;
    auto msg = compose_localization_pose(state_point, kf.get_P(), map_to_odom,
                                         base_R_lidar, base_T_lidar, ts);
    fins_node->send("localization_pose", msg, ts);
}

// 发布定位置信度 [0,1]。仅用现成的 scan-to-map 匹配指标与状态，无新增传感器量。
// 归一化分母直接复用质量门限：点数 >= min_effective_points 记满分；残差/平移跳变/
// 转角相对各自门限越小分越高；连续有效帧按 good_frames_to_lock 计。状态系数使
// 未 LOCKED 不到满值、DEGRADED/LOST/WAITING 直接归零。
void publish_localization_confidence(const fins::AcqTime &ts) {
    if (!fins_node->required("localization_confidence")) return;

    auto clamp01 = [](double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); };
    double target = 0.0;
    if (fixed_map_enabled && conf_has_quality_) {
        if (tracking_status == "LOCKED" || tracking_status == "TRACKING") {
            // 指标可能因空扫描/EKF 发散而是 inf/NaN，先做有限性保护，别让置信度变 NaN
            double feat   = std::isfinite(double(effct_feat_num)) ? double(effct_feat_num) : 0.0;
            double res    = std::isfinite(res_mean_last) ? res_mean_last : max_mean_residual;
            double jump   = std::isfinite(conf_jump_) ? conf_jump_ : max_update_translation;
            double turn   = std::isfinite(conf_turn_) ? conf_turn_ : max_update_rotation_deg;
            double f_feat   = clamp01(feat / std::max(1, min_effective_points));
            double f_res    = clamp01(1.0 - res / std::max(1e-6, max_mean_residual));
            double f_jump   = clamp01(1.0 - jump / std::max(1e-6, max_update_translation));
            double f_turn   = clamp01(1.0 - turn / std::max(1e-6, max_update_rotation_deg));
            double f_streak = clamp01(double(good_matches) /
                                      std::max(1, good_frames_to_lock));
            double quality = 0.25 * f_feat + 0.30 * f_res + 0.15 * f_jump
                           + 0.15 * f_turn + 0.15 * f_streak;
            // LOCKED 才给满权重；TRACKING 打折，反映尚未稳定
            double state_factor = (tracking_status == "LOCKED") ? 1.0 : 0.6;
            target = clamp01(quality) * state_factor;
        }
        // DEGRADED / LOST / WAITING_INITIALIZATION / INITIALIZATION_REJECTED -> 0
    }

    // 非对称 EMA：退化/失锁时快速掉到 0，恢复时平滑爬升，避免逐帧抖动
    double alpha = (target < localization_confidence_) ? kConfAlphaDown : kConfAlphaUp;
    localization_confidence_ += alpha * (target - localization_confidence_);
    if (localization_confidence_ < 1e-3) localization_confidence_ = 0.0;
    if (localization_confidence_ > 1.0)  localization_confidence_ = 1.0;

    std_msgs::msg::Float64 msg;
    msg.data = localization_confidence_;
    fins_node->send("localization_confidence", msg, ts);
}

void initialize() {
    fins_node->logger->info("User Initialization Start");

    std::fill(epsi, epsi + 23, 0.001);

    NUM_MAX_ITERATIONS = fins::param_server().get("FastLIO.max_iteration", 4)
                            .with_description("max iteration")
                            .greater_than(0);
    
    filter_size_surf_min = fins::param_server().get("FastLIO.filter_size_surf", 0.5)
                            .with_description("filter size of surface")
                            .greater_than(0.0);
    
    filter_size_map_min = fins::param_server().get("FastLIO.filter_size_map", 0.5)
                            .with_description("filter size of map")
                            .greater_than(0.0);
        
    cube_len = fins::param_server().get("FastLIO.cube_side_length", 200.0)
                            .with_description("cube side length")
                            .greater_than(0.00);

    fins::ParamLoader localization("FastLIO.localization");
    fixed_map_enabled = localization.get("enabled", false).with_description("Track immutable prior map after BBS initialization");
    min_effective_points = localization.get("min_effective_points", 30).greater_than(5);
    max_mean_residual = localization.get("max_mean_residual", 0.2).greater_than(0.0);
    max_update_translation = localization.get("max_update_translation", 0.3).greater_than(0.0);
    max_update_rotation_deg = localization.get("max_update_rotation_deg", 10.0).greater_than(0.0);
    failures_before_lost = localization.get("failures_before_lost", 5).greater_than(0);
    good_frames_to_lock = localization.get("good_frames_to_lock", 3).greater_than(0);
    max_alignment_age = localization.get("max_alignment_age", 10.0).greater_than(0.0);
    imu_rate_output_ = localization.get("imu_rate_output", true)
                           .with_description("publish odometry/localization_pose at IMU rate in fixed-map localization");
    if (fixed_map_enabled) {
        std::string map_path = localization.get("map_path", std::string(""));
        double leaf = localization.get("map_leaf_size", 0.15).greater_than(0.0);
        PointCloudXYZI::Ptr raw(new PointCloudXYZI());
        if (pcl::io::loadPCDFile(map_path, *raw) < 0) throw std::runtime_error("FixedMap: cannot load prior " + map_path);
        PointCloudXYZI::Ptr clean(new PointCloudXYZI());
        for (const auto &point : raw->points) if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) clean->push_back(point);
        pcl::VoxelGrid<PointType> filter;
        filter.setLeafSize(leaf,leaf,leaf); filter.setInputCloud(clean); filter.filter(*prior_map);
        if (prior_map->size() < 100) throw std::runtime_error("FixedMap: prior has fewer than 100 valid points");
    }
    fins::ParamLoader common("FastLIO.common");

    initial_frame = common.get("initial_frame", "odom")
                          .with_description("initial frame");

    base_link_frame = common.get("base_link_frame", "base_link")
                            .with_description("base link frame");

    base_lidar_frame = common.get("base_lidar_frame", "base_lidar")
                             .with_description("lidar frame");

    time_sync_en = common.get("time_sync_en", false)
                          .with_description("enable time synchronization");

    time_diff_lidar_to_imu = common.get("time_offset_lidar_to_imu", 0.0)
                                   .with_description("time offset between lidar and imu")
                                   .greater_than(0.0);

    use_imu_odometry_ = common.get("use_imu_odometry", false)
                              .with_description("use imu odometry");

    planar_constraint_ = common.get("planar_constraint", false)
                                .with_description("force planar output: roll=0, pitch=0, z=0");

    if (fixed_map_enabled) {
        // 平面约束沿用旧行为（定位不做强制平面归零）。
        planar_constraint_ = false;
        // 定位模式默认开启 IMU 高频输出：/Odometry、odom->base TF、
        // /localization_pose 都按 IMU 频率发布。kf_copy 已由 kf_copy_mutex
        // 保护，不再需要为了规避并发而关闭异步 IMU 预测。
        // 开关关闭时退回旧的 scan-rate(10Hz) 行为，完全等价于旧版。
        if (!imu_rate_output_) use_imu_odometry_ = false;
    }
    imu_window_size = common.get("imu_window_size", 10)
                            .with_description("imu window size")
                            .greater_than(0);
    
    imu_time_tolerance = common.get("imu_time_tolerance", 0.00)
                               .with_description("imu time tolerance")
                               .greater_than(0.00);

    fins::ParamLoader preprocess("FastLIO.preprocess");

    p_pre->blind = preprocess.get("blind", 0.5)
                             .with_description("blind radius")
                             .greater_than(0.0);

    p_pre->lidar_type = preprocess.get<int>("lidar_type", AVIA)
                             .with_description("AVIA = 1, VELO16 = 2, OUST64 = 3, MARSIM = 4")
                             .one_of({LID_TYPE::AVIA, LID_TYPE::VELO16, LID_TYPE::OUST64, LID_TYPE::MARSIM});
    lidar_type = p_pre->lidar_type;

    p_pre->N_SCANS = preprocess.get("scan_line", 16)
                               .with_description("scan line")
                               .greater_than(0);

    p_pre->time_unit = preprocess.get<int>("timestamp_unit", US)
                             .with_description("SEC = 0, MS = 1, US = 2, NS = 3")
                             .one_of({TIME_UNIT::SEC, TIME_UNIT::MS, TIME_UNIT::US, TIME_UNIT::NS});

    p_pre->SCAN_RATE = preprocess.get("scan_rate", 10)
                                 .with_description("scan rate")
                                 .greater_than(0);

    p_pre->feature_enabled = preprocess.get("feature_extract_enable", false)
                                       .with_description("enable feature extraction");

    p_pre->point_filter_num = preprocess.get("point_filter_num", 2)
                               .with_description("point filter number")
                               .greater_than(0);

    fins::ParamLoader mapping("FastLIO.mapping");
    acc_cov = mapping.get("acc_cov", 0.1)
               .with_description("acceleration covariance")
               .greater_than(0.0);
    gyr_cov = mapping.get("gyr_cov", 0.1)
               .with_description("gyroscope covariance")
               .greater_than(0.0);
    b_acc_cov = mapping.get("b_acc_cov", 0.0001)
               .with_description("acceleration bias covariance")
               .greater_than(0.0);
    b_gyr_cov = mapping.get("b_gyr_cov", 0.0001)
               .with_description("gyroscope bias covariance")
               .greater_than(0.0);
    fov_deg = mapping.get("fov_degree", 180.0)
               .with_description("FOV degree")
               .greater_than(0.0);
    DET_RANGE = mapping.get("det_range", 300.0f)
               .with_description("detection range")
               .greater_than(0.0f);

    extrinsic_est_en = mapping.get("extrinsic_est_en", false)
                              .with_description("enable extrinsic estimation");
    if (fixed_map_enabled) extrinsic_est_en = false;
    extrinT = mapping.get("extrinsic_T", std::vector<double>{-0.011, -0.02329, 0.04412})
               .with_description("extrinsic translation");
    extrinR = mapping.get("extrinsic_R", std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1})
               .with_description("extrinsic rotation");
    
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
        // 每帧必达（含所有 early-return 分支之前），保证置信度持续更新；
        // 用截至上一帧质量门控的指标，scan-rate 发布，滞后一帧可接受。
        publish_localization_confidence(Measures.acq_time);
        if (flg_first_scan)
        {
            first_lidar_time = Measures.lidar_beg_time;
            p_imu->first_lidar_time = first_lidar_time;
            flg_first_scan = false;
            return ;
        }
        
        apply_pending_alignment(Measures.acq_time);
        double t0;
        {
            auto t = fins_node->recorder("preprocessing", Measures.acq_time);
            t0 = omp_get_wtime();
            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                fins_node->logger->warn("No point, skip this scan!");
                reject_empty_scan(Measures.acq_time);
                return ;
            }
        }

        {
            auto t = fins_node->recorder("spatial_search", Measures.acq_time);
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            if (!fixed_map_active) lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(!fixed_map_active && ikdtree->Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree->set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree->Build(feats_down_world->points);
                }
                return ;
            }

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                fins_node->logger->warn("No point, skip this scan!");
                reject_empty_scan(Measures.acq_time);
                return ;
            }
            
            if (feats_down_size > 100000) {
                feats_down_body->resize(100000); feats_down_size = 100000;
            }
            laserCloudOri->resize(feats_down_size);
            corr_normvect->resize(feats_down_size);
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
        }
        
        /*** iterated state estimation ***/
        double t_update_start, t_update_end;
        {
            auto t = fins_node->recorder("state_estimation", Measures.acq_time);
            t_update_start = omp_get_wtime();
            auto predicted_state = kf.get_x();
            auto predicted_covariance = kf.get_P();
            effct_feat_num = 0;
            res_mean_last = std::numeric_limits<double>::infinity();
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV);
            state_point = kf.get_x();
            if (fixed_map_active) {
                const double jump = (state_point.pos-predicted_state.pos).norm();
                const double turn = SO3::log(predicted_state.rot.conjugate()*state_point.rot).norm()*180.0/M_PI;
                // 供下一帧 publish_localization_confidence 读取（本帧跳变指标）
                conf_jump_ = jump;
                conf_turn_ = turn;
                conf_has_quality_ = true;
                bool valid = effct_feat_num >= min_effective_points && std::isfinite(res_mean_last) &&
                    res_mean_last <= max_mean_residual && state_point.pos.allFinite() &&
                    std::isfinite(turn) && jump <= max_update_translation && turn <= max_update_rotation_deg;
                if (!valid) {
                    kf.change_x(predicted_state); kf.change_P(predicted_covariance); state_point = predicted_state;
                    good_matches = 0; ++bad_matches;
                    tracking_status = "DEGRADED";
                    fins_node->logger->warn("[FixedMap] Rejected update: features={} residual={} step={} angle={} failures={}",
                        effct_feat_num, res_mean_last, jump, turn, bad_matches);
                    if (bad_matches >= failures_before_lost) lose_tracking(Measures.acq_time);
                } else {
                    bad_matches = 0;
                    tracking_status = ++good_matches >= good_frames_to_lock ? "LOCKED" : "TRACKING";
                }
            }
            t_update_end = omp_get_wtime();

            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            {
                // 与 IMU 回调共用 kf_copy_mutex：同步滤波器副本 + 刷新高频定位快照。
                // 快照字段无条件刷新（不依赖 use_imu_odometry_），避免 IMU 回调的
                // gate 隐含依赖别处的返回前置条件。kf_copy=kf 仅在启用时同步。
                std::lock_guard<std::mutex> lock(kf_copy_mutex);
                if (use_imu_odometry_) kf_copy = kf;
                imu_rate_snapshot.fixed_map_enabled = fixed_map_enabled;
                imu_rate_snapshot.locked = (fixed_map_active &&
                                            tracking_status == "LOCKED" && bad_matches == 0);
                // 定位模式仅 LOCKED 后才启用 IMU 高频。初始化/未 LOCKED/LOST 期间
                // 必须保持 scan-rate：BBS 初始化的 on_cloud_callback 依赖
                // $T_{odom}^{base}$ TF 与 /cloud_registered 相同的扫描时间戳配对，
                // 只有 publish_lidar_odometry(Measures.acq_time) 能满足；若此时改用
                // IMU 时间戳发 TF，配对永远失败，会卡死在 WAITING_INITIALIZATION。
                imu_rate_snapshot.imu_rate_live = use_imu_odometry_ && imu_rate_output_
                    && fixed_map_enabled && fixed_map_active
                    && tracking_status == "LOCKED" && bad_matches == 0;
                imu_rate_snapshot.map_to_odom = map_to_odom;
                imu_rate_snapshot.base_R_lidar = base_R_lidar;
                imu_rate_snapshot.base_T_lidar = base_T_lidar;
            }
        }

        /******* Publish odometry *******/
        // scan-rate 路径在两种情况下必须保留：
        //   1) 配置关闭 IMU odom（!use_imu_odometry_）；
        //   2) 定位模式尚未 LOCKED（BBS 初始化依赖扫描时间戳的 odom->base TF）。
        // LOCKED 后停掉 scan-rate，交由 IMU 回调按 IMU 频率发布。
        if (!use_imu_odometry_
            || (fixed_map_enabled && !imu_rate_snapshot.imu_rate_live)) {
            publish_lidar_odometry(Measures.acq_time);
        }
        publish_path(Measures.acq_time);
        publish_fixed_pose(Measures.acq_time);

        /*** add the feature points to map kdtree ***/
        double t_incre_start, t_incre_end;
        {
            auto t = fins_node->recorder("map_incremental", Measures.acq_time);
            t_incre_start = omp_get_wtime();
            if (!fixed_map_active) {
                if (ikdtree->Root_Node == nullptr) {
                    PointVector seed;
                    for (const auto &point : feats_down_body->points) { PointType world; pointBodyToWorld(&point, &world); seed.push_back(world); }
                    ikdtree->set_downsample_param(filter_size_map_min); ikdtree->Build(seed);
                } else map_incremental();
            }
            t_incre_end = omp_get_wtime();
        }
        
        publish_frame_world(Measures.acq_time);

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