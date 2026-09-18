#pragma once

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace turtlmap {
struct SensorList
{
  bool isDvlUsed;
  bool isImuUsed;
  bool isBaroUsed;
  bool isSonarUsed;
  bool areCamsUsed;
};

struct BarometerParameters
{
  double atm_press;
  bool enable_interframe_integration;
  double noise_sigma;
  double min_correction_threshold;
  bool enable_velocity_correction;
};

struct ImuParameters
{
  double acc_max;
  double gyro_max;
  double gyro_noise_density;
  double acc_noise_density;
  double acc_random_walk;
  double gyro_random_walk;
  std::vector<double> acc_bias_prior;
  std::vector<double> gyro_bias_prior;
  double imu_rate;
  double integration_covariance;
  double g;
  double dvl_bias_prior;
  double dt_imu;
};

struct Camera
{
  std::vector<int> image_dimension;
  std::vector<double> distortion_coefficients;
  std::string distortion_type;
  std::vector<double> focal_length;
  std::vector<double> principal_point;
};

struct SensorTopics
{
  std::string imu_topic, dvl_topic, baro_topic, sonar_topic, left_cam_topic, right_cam_topic, dvl_local_position_topic;
};

struct StereoParameters
{
  double translation_sigma;
  double rotation_sigma;
  double loop_translation_sigma;
  double loop_rotation_sigma;
  double time_tolerance;
};

struct Extrinsics
{
  Eigen::Matrix4d T_SD;    // dvl to imu
  Eigen::Matrix4d T_SSo;   // sonar to imu
  Eigen::Matrix4d T_BS;    // imu to body
  Eigen::Matrix4d T_SLc;   // left cam to imu
  Eigen::Matrix4d T_SRc;   // right cam to imu
  Eigen::Matrix4d T_SBa;   // barometer to imu
  Eigen::Matrix4d T_W_WD;  // world to dvl world
};

struct OptimizationParameters
{
  double lambdaUpperBound;
  double lambdaLowerBound;
  double initialLambda;
  int maxIterations;
  double relativeErrorTol;
  double absoluteErrorTol;
};

struct ImuPreintegrationParams
{
  double gap_time;
};

class PoseGraphParameters
{
 public:
  PoseGraphParameters() {};
  ~PoseGraphParameters() {};

  // IMU Parameters
  ImuParameters imu_params_;
  Extrinsics extrinsics_;
  SensorList sensor_list_;
  SensorTopics sensor_topics_;
  OptimizationParameters optimization_params_;
  ImuPreintegrationParams imu_preintegration_params_;
  std::vector<std::string> rosbag_topics_;
  BarometerParameters baro_params_;
  StereoParameters stereo_params_;
  bool using_orbslam_;
  int num_iters_;
  double baro_atm_pressure_;
  bool using_smoother_;  // Deprecated: use optimization_mode_ instead
  double kf_gap_time_;
  double dvl_fom_threshold_;
  bool using_pseudo_dvl_;
  bool compute_marginals_;  // Flag to enable/disable marginal covariance computation
  double min_kf_delta_time_;

  bool use_cam_for_kf_;
  bool external_kf_only_;

  // Optimization backend configuration
  std::string optimization_mode_;  // "isam2", "batch", or "fixed_lag"
};
}  // namespace turtlmap
