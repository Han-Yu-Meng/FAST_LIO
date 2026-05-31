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
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>

#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
bool   airy_ned_to_flu_en = false;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic, initial_frame, base_link_frame, base_lidar_frame;
M3D base_R_lidar = Eye3d;
V3D base_T_lidar = Zero3d;
bool static_transform_received = false;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
int    num_sub_cloud = 1;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false, tf_pub_en = false;
bool    is_first_lidar = true;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

inline M3D get_ned_to_flu_rotation()
{
    M3D ned_to_flu;
    ned_to_flu << 1,  0,  0,
                  0, -1,  0,
                  0,  0, -1;
    return ned_to_flu;
}

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf_copy; // 在每次观测更新后，继承kf中的状态，用于IMU预测
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

/*** Modified for Fines Robot ***/
// ADD LINES BELOW
geometry_msgs::msg::TransformStamped transform_lidar2base;
// ADD LINES ABOVE

void SigHandle(int sig)
{
    flg_exit = true;
    std::cout << "catch sig %d" << sig << std::endl;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

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
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
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
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
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
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;

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
            std::cerr << "Too few input point cloud!\n";
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
            BoxPointType Box_of_Point;
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

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            // 1. 原本：将点转换到 FAST-LIO 内部的 Lidar 世界坐标系
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
            
            // 2. 多加一步：应用 base_link -> base_lidar 变换，将其转换到 odom (base_link 初始) 坐标系
            V3D p_world(laserCloudWorld->points[i].x, laserCloudWorld->points[i].y, laserCloudWorld->points[i].z);
            V3D p_odom = base_R_lidar * p_world + base_T_lidar;
            
            laserCloudWorld->points[i].x = p_odom(0);
            laserCloudWorld->points[i].y = p_odom(1);
            laserCloudWorld->points[i].z = p_odom(2);
        }
        
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = initial_frame;
        pubLaserCloudFull->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    /*
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
    */
}

void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudBaseBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToBase(&feats_undistort->points[i], \
                            &laserCloudBaseBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudBaseBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = base_link_frame;
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
    laserCloudFullRes3.header.frame_id = initial_frame;
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                            &laserCloudWorld->points[i]);
    }
    *pcl_wait_pub += *laserCloudWorld;

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
    // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = initial_frame;
    pubLaserCloudMap->publish(laserCloudmsg);

    // sensor_msgs::msg::PointCloud2 laserCloudMap;
    // pcl::toROSMsg(*featsFromMap, laserCloudMap);
    // laserCloudMap.header.stamp = get_ros_time(lidar_end_time);
    // laserCloudMap.header.frame_id = initial_frame;
    // pubLaserCloudMap->publish(laserCloudMap);
}

void save_to_pcd()
{
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
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

inline void compute_odom_to_base_pose(
    const M3D &R_odom_imu,
    const V3D &t_odom_imu,
    const M3D &R_imu_lidar,
    const V3D &t_imu_lidar,
    M3D &R_odom_base,
    V3D &t_odom_base)
{
    // odom is defined as the initial base_link frame:
    // T_(odom<-base) = T_BL * inv(T_IL) * T_OI * T_IL * inv(T_BL)
    const M3D R_lidar_imu = R_imu_lidar.transpose();
    const V3D t_lidar_imu = -R_lidar_imu * t_imu_lidar;
    const M3D R_lidar_base = base_R_lidar.transpose();
    const V3D t_lidar_base = -R_lidar_base * base_T_lidar;

    const M3D R1 = base_R_lidar * R_lidar_imu;
    const V3D t1 = base_R_lidar * t_lidar_imu + base_T_lidar;

    const M3D R2 = R1 * R_odom_imu;
    const V3D t2 = R1 * t_odom_imu + t1;

    const M3D R3 = R2 * R_imu_lidar;
    const V3D t3 = R2 * t_imu_lidar + t2;

    R_odom_base = R3 * R_lidar_base;
    t_odom_base = R3 * t_lidar_base + t3;
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    odomAftMapped.header.frame_id = initial_frame;
    odomAftMapped.child_frame_id = base_link_frame;
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    
    M3D R_odom_base;
    V3D t_odom_base;
    compute_odom_to_base_pose(
        state_point.rot.toRotationMatrix(),
        state_point.pos,
        state_point.offset_R_L_I.toRotationMatrix(),
        state_point.offset_T_L_I,
        R_odom_base,
        t_odom_base);
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
    static double last_lidar_time = lidar_end_time;
    double dt_ang = lidar_end_time - last_lidar_time;
    SO3 delta_rot = state_point.rot * last_rot.conjugate();
    last_rot = state_point.rot;
    last_lidar_time = lidar_end_time;
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
    odomAftMapped.twist.twist.linear.z = vel_base(2);

    // 角速度从 IMU 体系转到 base_link_frame：R_BI = R_BL * R_LI^T
    vect3 ang_vel_base = base_R_lidar * state_point.offset_R_L_I.toRotationMatrix().transpose() * omega_IMU;

    odomAftMapped.twist.twist.angular.x = ang_vel_base(0);
    odomAftMapped.twist.twist.angular.y = ang_vel_base(1);
    odomAftMapped.twist.twist.angular.z = ang_vel_base(2);

    // 计算协方差 (keep as is as an approximation)
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }
    pubOdomAftMapped->publish(odomAftMapped);

    if (tf_pub_en) {
        geometry_msgs::msg::TransformStamped trans;
        trans.header.frame_id = initial_frame;
        trans.child_frame_id = base_link_frame;
        trans.header.stamp = get_ros_time(lidar_end_time);
        trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
        trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
        trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
        trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
        trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
        trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
        trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
        tf_br->sendTransform(trans);
    }

}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = initial_frame;

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
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
        std::cerr << "No Effective Points!" << std::endl;
        // ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

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
    solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<bool>("publish.tf_en", false);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<int>("common.imu_window_size", 10);                            // 250727 szl imu_window_size_
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<string>("common.initial_frame", "odom");
        this->declare_parameter<string>("common.base_link_frame", "base_link");
        this->declare_parameter<string>("common.base_lidar_frame", "base_lidar");
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<bool>("common.use_imu_odometry", true);
        this->declare_parameter<bool>("common.airy_ned_to_flu_en", false);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<int>("mapping.num_sub_cloud", 1);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());


        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<bool>("publish.tf_en", tf_pub_en, false);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<int>("common.imu_window_size", imu_window_size , 10);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/livox/imu");
        this->get_parameter_or<string>("common.initial_frame", initial_frame, "camera_init");
        this->get_parameter_or<string>("common.base_link_frame", base_link_frame, "base_link");
        this->get_parameter_or<string>("common.base_lidar_frame", base_lidar_frame, "base_lidar");
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<bool>("common.use_imu_odometry", use_imu_odometry_, true);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<bool>("common.airy_ned_to_flu_en", airy_ned_to_flu_en, false);
        this->get_parameter_or<double>("filter_size_corner",filter_size_corner_min,0.5);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.fov_degree",fov_deg,180.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<int>("mapping.num_sub_cloud", num_sub_cloud, 1);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());

        RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id =initial_frame;

        // /*** variables definition ***/
        // int effect_feat_num = 0, frame_num = 0;
        // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        // bool flg_EKF_converged, EKF_stop_flg = 0;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        if (airy_ned_to_flu_en)
        {
            const M3D ned_to_flu = get_ned_to_flu_rotation();
            Lidar_R_wrt_IMU = ned_to_flu * Lidar_R_wrt_IMU;
            Lidar_T_wrt_IMU = ned_to_flu * Lidar_T_wrt_IMU;
        }
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        if (use_imu_odometry_) {
            kf_copy.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);
        }

        /*** debug record ***/
        // FILE *fp;
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");

        // ofstream fout_pre, fout_out, fout_dbg;
        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
        if (fout_pre && fout_out)
            cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
        else
            cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

        cb_group_imu_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_lidar_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_timer_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        auto imu_opt = rclcpp::SubscriptionOptions();
        imu_opt.callback_group = cb_group_imu_;

        auto lidar_opt = rclcpp::SubscriptionOptions();
        lidar_opt.callback_group = cb_group_lidar_;

        /*** ROS subscribe initialization ***/
        if (p_pre->lidar_type == AVIA)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, std::bind(&LaserMappingNode::livox_pcl_cbk, this, std::placeholders::_1), lidar_opt);
        }
        else
        {
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), std::bind(&LaserMappingNode::standard_pcl_cbk, this, std::placeholders::_1), lidar_opt);
        }

        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 10, std::bind(&LaserMappingNode::imu_cbk, this, std::placeholders::_1), imu_opt);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        //------------------------------------------------------------------------------------------------------
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this), cb_group_timer_);

        auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
        map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

    ~LaserMappingNode()
    {
        fout_out.close();
        fout_pre.close();
        fclose(fp);
    }

private:
    void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg) 
    {
        mtx_buffer.lock();
        scan_count ++;
        double cur_time = get_time_sec(msg->header.stamp);
        double preprocess_start_time = omp_get_wtime();
        if (!is_first_lidar && cur_time < last_timestamp_lidar)
        {
            std::cerr << "lidar loop back, clear buffer" << std::endl;
            lidar_buffer.clear();
        }
        if (is_first_lidar)
        {
            is_first_lidar = false;
        }

        if (p_pre->lidar_type == RSM1_BREAK) {
            double start_time, end_time;
            for (int i_sub_cloud = 0; i_sub_cloud < num_sub_cloud; i_sub_cloud++) {
                PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
                p_pre->process(msg, ptr, i_sub_cloud, num_sub_cloud, start_time, end_time);
                lidar_buffer.push_back(ptr);
                time_buffer.push_back(start_time);
                last_timestamp_lidar = start_time;
                sig_buffer.notify_all();
            }
            lidar_end_time = end_time;
        } else {
            PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
            p_pre->process(msg, ptr);
            lidar_buffer.push_back(ptr);
            time_buffer.push_back(cur_time);
            last_timestamp_lidar = cur_time;
        }
        s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
        mtx_buffer.unlock();
        sig_buffer.notify_all();
    }

    void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg) 
    {
        mtx_buffer.lock();
        double cur_time = get_time_sec(msg->header.stamp);
        double preprocess_start_time = omp_get_wtime();
        scan_count ++;
        if (!is_first_lidar && cur_time < last_timestamp_lidar)
        {
            std::cerr << "lidar loop back, clear buffer" << std::endl;
            lidar_buffer.clear();
        }
        if(is_first_lidar)
        {
            is_first_lidar = false;
        }
        last_timestamp_lidar = cur_time;
        
        if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
        {
            printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
        }

        if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
        {
            timediff_set_flg = true;
            timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
            printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
        }

        PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
        p_pre->process(msg, ptr);
        lidar_buffer.push_back(ptr);
        time_buffer.push_back(last_timestamp_lidar);
        
        s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
        mtx_buffer.unlock();
        sig_buffer.notify_all();
    }

    void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in) {
        auto t0 = omp_get_wtime();
        publish_count ++;

          // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
          sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

        if (airy_ned_to_flu_en)
        {
            // Airy IMU vectors are reported in NED; FAST-LIO expects FLU.
            msg->angular_velocity.y *= -1.0;
            msg->angular_velocity.z *= -1.0;
            msg->linear_acceleration.y *= -1.0;
            msg->linear_acceleration.z *= -1.0;
        }

          msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
          if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
          {
              msg->header.stamp = \
              rclcpp::Time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
          }

          double timestamp = get_time_sec(msg->header.stamp);

          mtx_buffer.lock();

          if (timestamp < last_timestamp_imu)
          {
              std::cerr << "lidar loop back, clear buffer" << std::endl;
              imu_buffer.clear();
          }
          auto dt = timestamp - last_timestamp_imu;

          last_timestamp_imu = timestamp;

          imu_buffer.push_back(msg);
          mtx_buffer.unlock();
          sig_buffer.notify_all();

        auto t1 = omp_get_wtime();

        // IMU 预测
        if (p_imu->imu_need_init()) {
            return;
        }

        V3D ang_vel, acc;
        // Use the ned_to_flu-converted msg (not raw msg_in) so that kf_copy.predict() receives
        // data consistent with the FLU extrinsics used by FAST-LIO when airy_ned_to_flu_en=true.
        ang_vel << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
        acc << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
        acc = acc * G_m_s2 / p_imu->get_mean_acc().norm();

        input_ikfom in;
        in.acc = acc;
        in.gyro = ang_vel;

        // Pass the converted msg (SharedPtr) so that process_and_publish_imu_odometry also
        // works on FLU-converted angular velocity for smoothing and omega_IMU computation.
        process_and_publish_imu_odometry(msg, dt, in, use_imu_odometry_);

        auto t2 = omp_get_wtime();
        // std::cout << "[IMU Process] Add to buffter, time cost " << t1 - t0 << "s" << std::endl;
        // std::cout << "[IMU Process] Predict imu state, time cost: " << t2 - t1 << "s" << std::endl;
    }

    void timer_callback()
    {
        if(sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;

                if (!static_transform_received) {
                    try {
                        geometry_msgs::msg::TransformStamped transformStamped = tf_buffer_->lookupTransform(base_link_frame, base_lidar_frame, tf2::TimePointZero);
                        base_T_lidar = V3D(transformStamped.transform.translation.x, transformStamped.transform.translation.y, transformStamped.transform.translation.z);
                        base_R_lidar = Eigen::Quaterniond(transformStamped.transform.rotation.w, transformStamped.transform.rotation.x, transformStamped.transform.rotation.y, transformStamped.transform.rotation.z).toRotationMatrix();
                        static_transform_received = true;
                        RCLCPP_INFO(this->get_logger(), "Static transform from %s to %s received.", base_link_frame.c_str(), base_lidar_frame.c_str());
                    } catch (tf2::TransformException &ex) {
                        RCLCPP_WARN(this->get_logger(), "Could not get static transform from %s to %s: %s", base_link_frame.c_str(), base_lidar_frame.c_str(), ex.what());
                    }
                }
                return;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
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
                return;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            // 更新 kf_copy 与 kf 状态一致
            if (use_imu_odometry_) {
                kf_copy = kf;
            }


            /******* Publish odometry *******/
            if (!use_imu_odometry_) {
                publish_odometry(pubOdomAftMapped_, tf_broadcaster_);
            }

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath_);
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull_);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_);
            if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_);
            // if (map_pub_en) publish_map(pubLaserCloudMap_);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                // time_log_counter ++;
                // printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }
    }

    void map_publish_callback()
    {
        if (map_pub_en) publish_map(pubLaserCloudMap_);
    }

    void map_save_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        if (pcd_save_en)
        {
            save_to_pcd();
            res->success = true;
            res->message = "Map saved.";
        }
        else
        {
            res->success = false;
            res->message = "Map save disabled.";
        }
    }

    void process_and_publish_imu_odometry(
        const sensor_msgs::msg::Imu::SharedPtr& msg_in,
        double dt,
        const input_ikfom& in,
        bool use_imu_odometry
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

        M3D R_odom_base;
        V3D t_odom_base;
        compute_odom_to_base_pose(
            imu_state.rot.toRotationMatrix(),
            imu_state.pos,
            imu_state.offset_R_L_I.toRotationMatrix(),
            imu_state.offset_T_L_I,
            R_odom_base,
            t_odom_base);
        Eigen::Quaterniond q_base(R_odom_base);

        nav_msgs::msg::Odometry imu_odometry;
        imu_odometry.header.stamp = msg_in->header.stamp;
        imu_odometry.header.frame_id = initial_frame;
        imu_odometry.child_frame_id = base_link_frame;

        imu_odometry.pose.pose.position.x = t_odom_base(0);
        imu_odometry.pose.pose.position.y = t_odom_base(1);
        imu_odometry.pose.pose.position.z = t_odom_base(2);
        imu_odometry.pose.pose.orientation.x = q_base.x();
        imu_odometry.pose.pose.orientation.y = q_base.y();
        imu_odometry.pose.pose.orientation.z = q_base.z();
        imu_odometry.pose.pose.orientation.w = q_base.w();

        // 线速度在 base_link_frame 下，需修正 IMU 与 base_link 偏移引起的速度差
        vect3 vel_imu_in_IMU = imu_state.rot.conjugate() * imu_state.vel; // IMU 原点速度在 IMU 体系

        // IMU 体系角速度（减去陀螺仪偏置）
        vect3 omega_IMU = smoothed_ang_vel - imu_state.bg;

        // 刚体速度修正：p_base 在 IMU 体系中的偏移
        V3D t_base_in_IMU = imu_state.offset_R_L_I.toRotationMatrix() * (-base_R_lidar.transpose() * base_T_lidar)
                            + imu_state.offset_T_L_I;
        // v_base = v_imu + omega_IMU × (p_base - p_imu)，均在 IMU 体系下
        vect3 vel_base_in_IMU = vel_imu_in_IMU + omega_IMU.cross(t_base_in_IMU);
        // 旋转到 base_link 体系
        vect3 vel_base = base_R_lidar * imu_state.offset_R_L_I.toRotationMatrix().transpose() * vel_base_in_IMU;

        imu_odometry.twist.twist.linear.x = vel_base(0);
        imu_odometry.twist.twist.linear.y = vel_base(1);
        imu_odometry.twist.twist.linear.z = vel_base(2);

        // 角速度转到 base_link_frame
        vect3 ang_vel_base = base_R_lidar * imu_state.offset_R_L_I.toRotationMatrix().transpose() * smoothed_ang_vel;
        imu_odometry.twist.twist.angular.x = ang_vel_base(0);
        imu_odometry.twist.twist.angular.y = ang_vel_base(1);
        imu_odometry.twist.twist.angular.z = ang_vel_base(2);

        // 发布里程计
        pubOdomAftMapped_->publish(imu_odometry);

        if (tf_pub_en) {
            geometry_msgs::msg::TransformStamped trans;
            trans.header.frame_id = initial_frame;
            trans.child_frame_id = base_link_frame;
            trans.header.stamp = msg_in->header.stamp;
            trans.transform.translation.x = imu_odometry.pose.pose.position.x;
            trans.transform.translation.y = imu_odometry.pose.pose.position.y;
            trans.transform.translation.z = imu_odometry.pose.pose.position.z;
            trans.transform.rotation.w = imu_odometry.pose.pose.orientation.w;
            trans.transform.rotation.x = imu_odometry.pose.pose.orientation.x;
            trans.transform.rotation.y = imu_odometry.pose.pose.orientation.y;
            trans.transform.rotation.z = imu_odometry.pose.pose.orientation.z;
            tf_broadcaster_->sendTransform(trans);
        }
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

    rclcpp::CallbackGroup::SharedPtr cb_group_imu_;
    rclcpp::CallbackGroup::SharedPtr cb_group_lidar_;
    rclcpp::CallbackGroup::SharedPtr cb_group_timer_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    bool effect_pub_en = false, map_pub_en = false;
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[23] = {0.001};
    bool use_imu_odometry_;

    FILE *fp;
    ofstream fout_pre, fout_out, fout_dbg;

    // 滑动窗口平滑所需变量
    int imu_window_size = 41 ; // 默认值
    std::deque<Eigen::Vector3d> imu_ang_vel_window_;  // 仅存储角速度历史数据
    Eigen::Vector3d avg_ang_vel_ = Eigen::Vector3d::Zero(); // 对角速度求和取平均
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    auto node = std::make_shared<LaserMappingNode>();
    
    // 使用多线程执行器以解决回调阻塞问题
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();

    if (rclcpp::ok())
        rclcpp::shutdown();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
