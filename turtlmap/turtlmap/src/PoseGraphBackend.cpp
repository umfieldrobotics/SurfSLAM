// Non-ROS implementation of posegraph backend

#include "PoseGraphBackend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <thread>

#include "QuaternionAveraging.h"
#include "gtsam/nonlinear/Marginals.h"

namespace turtlmap {

PoseGraphBackend::PoseGraphBackend(const std::string& config_file,
                                                         std::shared_ptr<turtlmap::IDataProvider> data_provider)
    : data_provider_(data_provider)
{
  logger_.debug() << "Initializing PoseGraphBackend..." << std::endl;

  std::string config = config_file;
  posegraph_ = std::make_unique<AUVPoseGraph>(config);

  // Initialize pim_dvl with same params as main pim
  pim_dvl_ = std::make_unique<PreintegratedCombinedMeasurements>(posegraph_->pim_params_, posegraph_->priorImuBias_);

  // Initialize barometer integrator with parameters from config
  if (posegraph_->params_->baro_params_.enable_interframe_integration) {
    BarometerIntegrator::Params baro_params;
    baro_params.baro_noise_sigma = posegraph_->params_->baro_params_.noise_sigma;
    baro_params.min_correction_threshold = posegraph_->params_->baro_params_.min_correction_threshold;
    baro_params.enable_velocity_correction = posegraph_->params_->baro_params_.enable_velocity_correction;

    // Pass IMU parameters to compute drift rate
    baro_params.acc_noise_density = posegraph_->params_->imu_params_.acc_noise_density;
    baro_params.acc_random_walk = posegraph_->params_->imu_params_.acc_random_walk;
    baro_params.imu_rate = posegraph_->params_->imu_params_.imu_rate;

    baro_integrator_ = std::make_unique<BarometerIntegrator>(baro_params, logger_);

    logger_.debug() << "Barometer inter-frame integration ENABLED (Kalman filter mode)" << std::endl;
    logger_.debug() << "  - Baro noise: " << baro_params.baro_noise_sigma << " m" << std::endl;
  } else {
    logger_.debug() << "Barometer inter-frame integration DISABLED (factor-only mode)" << std::endl;
    logger_.debug() << "  - Barometer will only constrain pose at keyframes" << std::endl;
  }

  kf_gap_time_ = posegraph_->params_->kf_gap_time_;
  frame_count_ = 0;

  try {
    gtsam::Rot3 R_BS(posegraph_->params_->extrinsics_.T_BS.block<3, 3>(0, 0));
    gtsam::Point3 t_BS(posegraph_->params_->extrinsics_.T_BS.block<3, 1>(0, 3));
    gtsam::Pose3 T_BS(R_BS, t_BS);

    gtsam::Rot3 R_S_Lc(posegraph_->params_->extrinsics_.T_SLc.block(0, 0, 3, 3));
    gtsam::Point3 t_S_Lc(posegraph_->params_->extrinsics_.T_SLc.block(0, 3, 3, 1));
    gtsam::Pose3 T_S_Lc(R_S_Lc, t_S_Lc);

    gtsam::Pose3 T_NED(gtsam::Rot3(0.0, 1.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));
    gtsam::Pose3 A = T_NED.inverse();
    gtsam::Pose3 B = T_BS.inverse() * T_S_Lc.inverse();

    gtsam::Matrix Ad_A = A.AdjointMap();
    gtsam::Matrix Ad_Binv = B.inverse().AdjointMap();
    J_world_to_ned_ = Ad_Binv * Ad_A;  // 6x6
  } catch (...) {
    J_world_to_ned_ = gtsam::Matrix::Identity(6, 6);
  }
}

void PoseGraphBackend::run()
{
  if (running_) {
    std::cerr << "Backend is already running!" << std::endl;
    return;
  }

  logger_.info() << "Running PoseGraphBackend..." << std::endl;

  data_provider_->setImuCallback([this](const turtlmap::ImuMeasurement& msg) { this->insertIMU(msg); });

  if (posegraph_->params_->sensor_list_.isDvlUsed) {
    data_provider_->setDvlCallback([this](const turtlmap::DvlMeasurement& msg) { this->insertDVL(msg); });
  }

  data_provider_->setBarometerCallback([this](const turtlmap::BarometerMeasurement& msg) { this->insertBaro(msg); });

  data_provider_->setStereoCallback([this](const turtlmap::StereoMeasurement& msg) { this->insertStereo(msg); });

  if (live_vis_enabled_) {
    std::ofstream ofs(live_traj_file_, std::ios::trunc);
    if (ofs.is_open()) {
      logger_.debug() << "Live trajectory will be written to: " << live_traj_file_ << std::endl;
      ofs.flush();  // Ensure file is created on disk
    } else {
      std::cerr << "Warning: cannot open live trajectory file: " << live_traj_file_ << std::endl;
      live_vis_enabled_ = false;
    }

    // Initialize sensor data file
    std::ofstream sensor_ofs(live_sensor_file_, std::ios::trunc);
    if (sensor_ofs.is_open()) {
      logger_.debug() << "Live sensor data will be written to: " << live_sensor_file_ << std::endl;
      sensor_ofs.flush();
    }

    // Initialize covariance file (XY plane translation covariance at keyframes)
    std::ofstream cov_ofs(live_traj_file_ + ".cov", std::ios::trunc);
    if (cov_ofs.is_open()) {
      logger_.debug() << "Live covariance will be written to: " << (live_traj_file_ + ".cov") << std::endl;
      // Optional header (commented): # ts x y z cov_xx cov_xy cov_yy
      cov_ofs.flush();
    }

    // Initialize optimization timing file
    std::ofstream timing_ofs(optimization_timing_file_, std::ios::trunc);
    if (timing_ofs.is_open()) {
      logger_.debug() << "Optimization timing will be written to: " << optimization_timing_file_ << std::endl;
      // Write header
      timing_ofs << "# timestamp keyframe_index elapsed_time_ms optimization_mode" << std::endl;
      timing_ofs.flush();
    }
  }

  running_ = true;
  output_thread_ = std::thread(&PoseGraphBackend::outputLoop, this);
  processing_thread_ = std::thread(&PoseGraphBackend::processingLoop, this);
  kf_thread_ = std::thread(&PoseGraphBackend::kfLoop, this);

  logger_.debug() << "Backend ready to consume data (callbacks registered)" << std::endl;
}

void PoseGraphBackend::stop()
{
  if (!running_) {
    return;  // Already stopped
  }

  logger_.info() << "Stopping PoseGraphBackend..." << std::endl;

  running_ = false;
  kf_cv_.notify_all();
  processing_cv_.notify_all();
  output_cv_.notify_all();

  if (data_provider_) {
    data_provider_->stop();
  }

  if (output_thread_.joinable()) {
    output_thread_.join();
  }

  if (processing_thread_.joinable()) {
    processing_thread_.join();
  }

  if (kf_thread_.joinable()) {
    kf_thread_.join();
  }

  logger_.debug() << "Backend stopped" << std::endl;
}

PoseGraphBackend::~PoseGraphBackend()
{
  logger_.debug() << "Shutting down PoseGraphBackend..." << std::endl;

  // Ensure stop is called if not already
  if (running_) {
    stop();
  }
}

void PoseGraphBackend::insertIMU(const turtlmap::ImuMeasurement& imu_msg)
{
  std::lock_guard<std::mutex> lock(queue_mutex_);
  data_queue_.emplace(imu_msg);
  processing_cv_.notify_one();
}

void PoseGraphBackend::insertDVL(const turtlmap::DvlMeasurement& dvl_msg)
{
  std::lock_guard<std::mutex> lock(queue_mutex_);
  data_queue_.emplace(dvl_msg);
  processing_cv_.notify_one();
}

void PoseGraphBackend::insertBaro(const turtlmap::BarometerMeasurement& baro_msg)
{
  std::lock_guard<std::mutex> lock(queue_mutex_);
  data_queue_.emplace(baro_msg);
  processing_cv_.notify_one();
}

void PoseGraphBackend::insertStereo(const turtlmap::StereoMeasurement& stereo_msg)
{
  std::lock_guard<std::mutex> lock(queue_mutex_);
  data_queue_.emplace(stereo_msg);
  processing_cv_.notify_one();
}

void PoseGraphBackend::outputLoop()
{
  logger_.debug() << "Output loop started" << std::endl;

  while (running_) {
    std::unique_lock<std::mutex> lock(output_mutex_);
    output_cv_.wait_for(lock, std::chrono::milliseconds(50), [this] { return !output_queue_.empty() || !running_; });

    if (!running_ && output_queue_.empty()) break;

    while (!output_queue_.empty()) {
      OutputState state = output_queue_.front();
      output_queue_.pop();
      lock.unlock();

      if (live_vis_enabled_) {
        std::ofstream sensor_ofs(live_sensor_file_, std::ios::app);
        if (sensor_ofs.is_open()) {
          sensor_ofs << std::fixed << std::setprecision(6) << state.timestamp << "," << state.imu_ax << ","
                     << state.imu_ay << "," << state.imu_az << "," << state.imu_wx << "," << state.imu_wy << ","
                     << state.imu_wz << "," << state.dvl_vx << "," << state.dvl_vy << "," << state.dvl_vz << ","
                     << -state.baro_depth << "\n";
          sensor_ofs.flush();
        }

        std::ofstream traj_ofs(live_traj_file_, std::ios::app);
        if (traj_ofs.is_open()) {
          const auto& p = state.pose_ned;
          auto q = p.rotation().toQuaternion();
          traj_ofs << std::fixed << std::setprecision(6) << state.timestamp << ' ' << p.translation().x() << ' '
                   << p.translation().y() << ' ' << p.translation().z() << ' ' << ' ' << q.x() << ' ' << q.y() << ' '
                   << q.z() << ' ' << q.w() << '\n';
          traj_ofs.flush();
        }

        if (state.is_keyframe) {
          std::ofstream kf_ofs(live_traj_file_ + ".keyframes", std::ios::app);
          if (kf_ofs.is_open()) {
            const auto& p = state.pose_ned;
            auto q = p.rotation().toQuaternion();
            kf_ofs << std::fixed << std::setprecision(6) << state.timestamp << ' ' << p.translation().x() << ' '
                   << p.translation().y() << ' ' << p.translation().z() << ' ' << q.x() << ' ' << q.y() << ' ' << q.z()
                   << ' ' << q.w() << '\n';
            kf_ofs.flush();
          }
        }
      }

      if (pose_callback_) {
        pose_callback_(state.timestamp, state.pose_ned);
      }

      lock.lock();
    }
  }

  logger_.debug() << "Output loop ended" << std::endl;
}

void PoseGraphBackend::processingLoop()
{
  logger_.debug() << "Processing loop started" << std::endl;

  while (running_) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    processing_cv_.wait_for(lock, std::chrono::milliseconds(10), [this] { return !data_queue_.empty() || !running_; });

    if (!running_ && data_queue_.empty()) break;

    std::optional<double> input_kf_time = std::nullopt;

    bool make_kf_here = false;

    while (!data_queue_.empty()) {
      turtlmap::SensorData data = data_queue_.top();
      data_queue_.pop();
      lock.unlock();

      make_kf_here = std::visit(
          [&](const auto& msg) -> bool {
            using T = std::decay_t<decltype(msg)>;
            if constexpr (std::is_same_v<T, turtlmap::ImuMeasurement>) {
              processIMU(msg);
              return msg.make_keyframe;
            } else if constexpr (std::is_same_v<T, turtlmap::DvlMeasurement>) {
              processDVL(msg);
              return false;
            } else if constexpr (std::is_same_v<T, turtlmap::BarometerMeasurement>) {
              processBarometer(msg);
              return false;
            } else if constexpr (std::is_same_v<T, turtlmap::StereoMeasurement>) {
              processStereo(msg);
              return false;
            }
          },
          data);

      lock.lock();

      if (make_kf_here) {
        break;
      }
    }

    lock.unlock();

    // Check if keyframe is needed (holds mtx_ internally)
    if (shouldCreateKeyframe(make_kf_here)) {
      // Set flag to block processing until keyframe setup is complete
      kf_opt_in_progress_ = true;

      // Signal keyframe thread
      kf_cv_.notify_one();

      // Wait until keyframe setup (symbols and factors) is complete
      // This ensures no additional data is processed between KF creation and factor addition
      std::unique_lock<std::mutex> setup_lock(mtx_);
      kf_opt_done_cv_.wait(setup_lock, [this] { return !kf_opt_in_progress_; });
    }
  }

  logger_.debug() << "Processing loop ended" << std::endl;
}

gtsam::Pose3 PoseGraphBackend::transformImuToCamera(const gtsam::Pose3& T_W_S)
{
  // Transform base_link -> IMU -> Left camera -> NED
  // gtsam::Pose3 T_BS(posegraph_->params_->extrinsics_.T_BS);
  // gtsam::Pose3 T_W_B = T_W_S * T_BS.inverse();

  gtsam::Pose3 T_S_Lc(posegraph_->params_->extrinsics_.T_SLc); // camera to imu
  gtsam::Pose3 pose_Lc = T_W_S * T_S_Lc.inverse();

  gtsam::Pose3 T_NED = gtsam::Pose3(gtsam::Rot3(0.0, 1.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));
  return T_NED.inverse() * pose_Lc;
}

gtsam::Pose3 PoseGraphBackend::transformLeftCameraToIMU(const gtsam::Pose3& T_lc1_lc2)
{
  gtsam::Pose3 T_S_Lc(posegraph_->params_->extrinsics_.T_SLc); // camera to imu
  gtsam::Pose3 T_S1_S2 = T_S_Lc.inverse() * T_lc1_lc2 * T_S_Lc;
  return T_S1_S2;
}

void PoseGraphBackend::publishState(const gtsam::Pose3& pose_imu, bool is_keyframe, double timestamp)
{
  if (!live_vis_enabled_ && !pose_callback_) return;

  OutputState state;
  state.timestamp = (timestamp >= 0.0) ? timestamp : latest_sensor_time_;
  state.pose_ned = transformImuToCamera(pose_imu);
  state.velocity = latest_dvl_vel_;
  state.imu_ax = latest_imu_ax_;
  state.imu_ay = latest_imu_ay_;
  state.imu_az = latest_imu_az_;
  state.imu_wx = latest_imu_wx_;
  state.imu_wy = latest_imu_wy_;
  state.imu_wz = latest_imu_wz_;
  state.dvl_vx = latest_dvl_vx_;
  state.dvl_vy = latest_dvl_vy_;
  state.dvl_vz = latest_dvl_vz_;
  state.baro_depth = latest_baro_depth_;
  state.is_keyframe = is_keyframe;

  // Update trajectory queue for pose queries (thread-safe)
  {
    std::lock_guard<std::mutex> traj_lock(traj_mtx_);
    traj_pose_queue_.push_back({state.timestamp, state.pose_ned});

    // Keep queue size manageable
    while (traj_pose_queue_.size() > traj_pose_queue_max_size_) {
      traj_pose_queue_.pop_front();
    }
  }

  std::lock_guard<std::mutex> lock(output_mutex_);
  output_queue_.push(state);
  output_cv_.notify_one();
}

bool PoseGraphBackend::shouldCreateKeyframe(const bool force /*=false*/)
{
  std::lock_guard<std::mutex> lock(mtx_);

  const bool cams_used = posegraph_->params_->sensor_list_.areCamsUsed;
  const bool external_kf_only = posegraph_->params_->external_kf_only_;
  const bool cams_used_for_kf = !external_kf_only && cams_used && posegraph_->params_->use_cam_for_kf_;

  // ========== Stereo-triggered keyframes (cameras enabled) ==========
  if (cams_used_for_kf && stereo_measurement_ready_) {
    logger_.debug() << "[shouldCreateKeyframe] Stereo keyframe ready:" << std::endl;
    logger_.debug() << "  Previous KF time: " << std::fixed << std::setprecision(9) << prev_kf_time_ << std::endl;
    logger_.debug() << "  Current KF time: " << current_kf_time_ << std::endl;
    logger_.debug() << "  Stereo trigger time: " << stereo_measurement_time_ << std::endl;
    logger_.debug() << "  Time since last KF: " << (stereo_measurement_time_ - current_kf_time_) << " s" << std::endl;
    logger_.debug() << "  PIM delta time: " << posegraph_->pim_->deltaTij() << " s" << std::endl;

    // Add pseudo-DVL if needed (same as time-based path)
    if (posegraph_->params_->sensor_list_.isDvlUsed && posegraph_->params_->using_pseudo_dvl_) {
      gtsam::NavState pseudo_state = pim_dvl_->predict(
          gtsam::NavState(latest_dvl_pose_, latest_dvl_pose_.rotation() * latest_dvl_vel_), posegraph_->priorImuBias_);
      gtsam::Vector3 pseudo_dvl_vel = pseudo_state.pose().rotation().inverse() * pseudo_state.velocity();

      posegraph_->addDvlVelocity(stereo_measurement_time_, pseudo_dvl_vel);
      posegraph_->imu_rot_list_.push_back(imu_latest_rot_);
      posegraph_->current_dvl_foms_.push_back(0.02);
      prev_dvl_time_ = stereo_measurement_time_;
    }

    prev_kf_time_ = current_kf_time_;
    current_kf_time_ = stereo_measurement_time_;
    stereo_measurement_ready_ = false;
    new_kf_flag_ = true;

    return true;
  }

  const double reference_time = latest_sensor_time_;

  if (reference_time <= 0.0) {
    return false;
  }

  // Initialize timing on first call
  if (current_kf_time_ == 0.0) {
    first_kf_time_ = reference_time;
    current_kf_time_ = reference_time;
    return false;
  }

  const double dt = reference_time - current_kf_time_;
  if (dt <= 0.0) {
    return false;
  }

  // Check if time threshold exceeded
  if (force || (dt > kf_gap_time_ && !external_kf_only)) {
    // Cameras enabled but stereo hasn't triggered - fall back to time-based
    if (!force && cams_used) {
      logger_.debug() << "No stereo for " << dt << "s > " << kf_gap_time_ << "s, using time-based trigger" << std::endl;
    }

    // Add pseudo-DVL if needed
    if (posegraph_->params_->sensor_list_.isDvlUsed && posegraph_->params_->using_pseudo_dvl_) {
      gtsam::NavState pseudo_state = pim_dvl_->predict(
          gtsam::NavState(latest_dvl_pose_, latest_dvl_pose_.rotation() * latest_dvl_vel_), posegraph_->priorImuBias_);
      gtsam::Vector3 pseudo_dvl_vel = pseudo_state.pose().rotation().inverse() * pseudo_state.velocity();

      posegraph_->addDvlVelocity(latest_sensor_time_, pseudo_dvl_vel);
      posegraph_->imu_rot_list_.push_back(imu_latest_rot_);
      posegraph_->current_dvl_foms_.push_back(0.02);
      prev_dvl_time_ = latest_sensor_time_;
    }

    prev_kf_time_ = current_kf_time_;
    current_kf_time_ = reference_time;
    new_kf_flag_ = true;

    return true;
  }

  return false;
}

void PoseGraphBackend::predictInterFramePose()
{
  gtsam::Pose3 base_pose = latest_dvl_pose_;
  gtsam::Vector3 base_vel = latest_dvl_vel_;
  gtsam::NavState seed(base_pose, base_pose.rotation() * base_vel);
  latest_imu_prop_state_ = pim_dvl_->predict(seed, posegraph_->priorImuBias_);
  imu_latest_rot_ = latest_imu_prop_state_.pose().rotation();

  if (posegraph_->params_->sensor_list_.isBaroUsed && posegraph_->params_->baro_params_.enable_interframe_integration &&
      posegraph_->getDepthValid()) {
    double dt = (last_baro_time_ > 0) ? (latest_sensor_time_ - last_baro_time_) : 0.1;
    latest_imu_prop_state_ =
        baro_integrator_->correctState(latest_imu_prop_state_, posegraph_->getDepthMeasurement(), dt);
  }

  if (!posegraph_->params_->sensor_list_.isDvlUsed) {
    publishState(latest_imu_prop_state_.pose(), false);
  }
}

void PoseGraphBackend::initializeFromIMU(const turtlmap::ImuMeasurement& imu_msg)
{
  // Check if already initialized (race condition protection)
  if (is_rot_initialized_) return;

  imu_init_count_++;
  gtsam::Rot3 imu_rot_sample(imu_msg.qw, imu_msg.qx, imu_msg.qy, imu_msg.qz);

  // Store as quaternion for proper averaging
  auto quat = imu_rot_sample.toQuaternion();
  Eigen::Quaterniond eigen_quat(quat.w(), quat.x(), quat.y(), quat.z());
  imu_init_quats_.push_back(eigen_quat);

  if (imu_init_count_ == 29)  // Just before averaging
  {
    logger_.debug() << "IMU init samples: " << std::endl;
    for (size_t i = 0; i < imu_init_quats_.size(); ++i) {
      const auto& q = imu_init_quats_[i];
      logger_.debug() << "  [" << i << "] qw=" << q.w() << " qx=" << q.x() << " qy=" << q.y() << " qz=" << q.z()
                      << std::endl;
    }
  }

  if (imu_init_count_ < 30) return;

  std::vector<gtsam::Rot3> rotations_to_average;
  int N = std::min<int>(static_cast<int>(imu_init_quats_.size()), 30);
  for (int i = imu_init_quats_.size() - N; i < (int)imu_init_quats_.size(); ++i) {
    const auto& q = imu_init_quats_[i];
    rotations_to_average.push_back(gtsam::Rot3(q.w(), q.x(), q.y(), q.z()));
  }

  // Average on the Lie algebra (tangent space at identity)
  gtsam::Rot3 imu_rot = averageRotationsSO3(rotations_to_average);

  // ADD DETAILED LOGGING HERE
  logger_.info() << "=== IMU INITIALIZATION COMPLETE ===" << std::endl;
  logger_.info() << "  Averaged rotation (quat): w=" << imu_rot.toQuaternion().w()
                 << " x=" << imu_rot.toQuaternion().x() << " y=" << imu_rot.toQuaternion().y()
                 << " z=" << imu_rot.toQuaternion().z() << std::endl;
  logger_.info() << "  IMU timestamp: " << std::fixed << std::setprecision(9) << imu_msg.timestamp << std::endl;

  // One-time graph initialization from IMU
  imu_latest_rot_ = imu_rot;
  posegraph_->imu_prev_rot_ = imu_rot;
  imu_init_quats_.clear();

  // Initialize pose graph with IMU attitude
  posegraph_->initializePoseGraphFromImu(imu_rot);

  // Anchor initial timing to IMU timestamp
  current_kf_time_ = imu_msg.timestamp;
  if (current_kf_time_ == 0.0) {
    current_kf_time_ = 0.0;
  }

  if (!posegraph_->initial_->exists(gtsam::Symbol('v', 0))) {
    posegraph_->initial_->insert(gtsam::Symbol('v', 0), gtsam::Vector3(0, 0, 0));
  }
  auto prior_vel_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 0.01, 0.01, 0.01).finished());
  posegraph_->graph->add(
      gtsam::PriorFactor<gtsam::Vector3>(gtsam::Symbol('v', 0), gtsam::Vector3(0, 0, 0), prior_vel_noise));

  if (!posegraph_->initial_->exists(gtsam::Symbol('b', 0))) {
    posegraph_->initial_->insert(gtsam::Symbol('b', 0), posegraph_->priorImuBias_);
  }
  {
    gtsam::Vector6 sigmas;
    sigmas << 0.01, 0.01, 0.01, 0.001, 0.001, 0.001;
    auto bias_prior_noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    posegraph_->graph->add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(
        gtsam::Symbol('b', 0), posegraph_->priorImuBias_, bias_prior_noise));
  }

  if (posegraph_->params_->using_smoother_) {
    posegraph_->smootherTimestamps[gtsam::Symbol('x', posegraph_->index_)] = 0.0;
    posegraph_->smootherTimestamps[gtsam::Symbol('v', posegraph_->index_)] = 0.0;
    posegraph_->smootherTimestamps[gtsam::Symbol('b', posegraph_->index_)] = 0.0;
    if (posegraph_->params_->sensor_list_.isDvlUsed) {
      if (!posegraph_->initial_->exists(gtsam::Symbol('d', posegraph_->index_))) {
        posegraph_->initial_->insert(gtsam::Symbol('d', posegraph_->index_), posegraph_->priorDvlBias_);
      }
      posegraph_->smootherTimestamps[gtsam::Symbol('d', posegraph_->index_)] = 0.0;
    }
  }

  first_kf_time_ = current_kf_time_;
  posegraph_->prev_kf_time_ = current_kf_time_;
  kf_timestamps_.push_back(current_kf_time_);
  latest_kf_pose_ = posegraph_->initial_->at<gtsam::Pose3>(gtsam::Symbol('x', 0));
  latest_publish_pose_ = latest_kf_pose_;
  latest_dvl_pose_ = latest_publish_pose_;
  latest_dvl_vel_ = gtsam::Vector3(0, 0, 0);
  first_depth_ = 0.0;

  logger_.info() << "=== POSE GRAPH INITIALIZED ===" << std::endl;
  logger_.info() << "  Current KF time: " << std::fixed << std::setprecision(9) << current_kf_time_ << std::endl;
  logger_.info() << "  First KF time: " << first_kf_time_ << std::endl;
  logger_.info() << "  First depth: " << first_depth_ << std::endl;
  if (posegraph_->initial_->exists(gtsam::Symbol('x', 0))) {
    auto x0 = posegraph_->initial_->at<gtsam::Pose3>(gtsam::Symbol('x', 0));
    logger_.info() << "  Initial pose x0 translation: [" << x0.translation().x() << ", " << x0.translation().y() << ", "
                   << x0.translation().z() << "]" << std::endl;
    auto q = x0.rotation().toQuaternion();
    logger_.info() << "  Initial pose x0 rotation (quat): w=" << q.w() << " x=" << q.x() << " y=" << q.y()
                   << " z=" << q.z() << std::endl;
  }

  is_rot_initialized_ = true;
  logger_.debug() << "=== GRAPH READY FOR STEREO PROCESSING ===" << std::endl;
  logger_.debug() << "  Python can now start processing stereo measurements" << std::endl;
  logger_.debug() << "  First stereo will anchor to x0, subsequent stereo will "
                     "create keyframes"
                  << std::endl;
  logger_.info() << "IMU initialization complete with " << N << " samples averaged" << std::endl;
}

void PoseGraphBackend::processIMU(const turtlmap::ImuMeasurement& imu_msg, bool unsafe /*=false*/)
{
  if (!is_rot_initialized_) {
    initializeFromIMU(imu_msg);
    logger_.debug() << "IMU not init..." << std::endl;
    return;
  }

  std::unique_lock<std::mutex> lock(mtx_, std::defer_lock);
  if (!unsafe) {
    lock.lock();  // Only acquire lock if caller doesn't already hold it
  }

  imu_latest_rot_ = gtsam::Rot3(imu_msg.qw, imu_msg.qx, imu_msg.qy, imu_msg.qz);

  latest_imu_ax_ = imu_msg.ax;
  latest_imu_ay_ = imu_msg.ay;
  latest_imu_az_ = imu_msg.az;
  latest_imu_wx_ = imu_msg.wx;
  latest_imu_wy_ = imu_msg.wy;
  latest_imu_wz_ = imu_msg.wz;
  latest_sensor_time_ = imu_msg.timestamp;

  double dt = posegraph_->params_->imu_params_.dt_imu;
  // update w real dt
  if (prev_imu_time_ > 0.0) {
    double measured_dt = imu_msg.timestamp - prev_imu_time_;
    if (measured_dt > 0.001 && measured_dt < 0.1) {
      dt = measured_dt;
    }
  }
  prev_imu_time_ = imu_msg.timestamp;

  gtsam::Vector3 acc(imu_msg.ax, imu_msg.ay, imu_msg.az);
  gtsam::Vector3 gyro(imu_msg.wx, imu_msg.wy, imu_msg.wz);

  // Track IMU integration count for debugging
  imu_integration_count_++;

  posegraph_->pim_->integrateMeasurement(acc, gyro, dt);
  pim_dvl_->integrateMeasurement(acc, gyro, dt);

  const int N = 5;
  imu_count_++;
  if (imu_count_ >= N) {
    predictInterFramePose();
    imu_count_ = 0;
  }
}

void PoseGraphBackend::processDVL(const turtlmap::DvlMeasurement& dvl_msg, bool unsafe /*=false*/)
{
  if (!posegraph_->params_->sensor_list_.isDvlUsed || !is_rot_initialized_) {
    return;
  }

  std::unique_lock<std::mutex> lock(mtx_, std::defer_lock);
  if (!unsafe) {
    lock.lock();  // Only acquire lock if caller doesn't already hold it
  }

  double dvl_current_time = dvl_msg.timestamp;
  gtsam::Vector3 dvl_vel_imu_frame = posegraph_->T_SD_.block(0, 0, 3, 3) * gtsam::Vector3(dvl_msg.vx, dvl_msg.vy, dvl_msg.vz);

  latest_dvl_vx_ = dvl_vel_imu_frame.x();
  latest_dvl_vy_ = dvl_vel_imu_frame.y();
  latest_dvl_vz_ = dvl_vel_imu_frame.z();

  if (prev_dvl_time_ == 0.0) {
    prev_dvl_time_ = dvl_current_time;
    first_dvl_vel_ = dvl_vel_imu_frame;

    logger_.debug() << "First DVL velocity set: [" << dvl_vel_imu_frame.x() << ", " << dvl_vel_imu_frame.y() << ", " << dvl_vel_imu_frame.z() << "]"
                    << std::endl;

    if (!posegraph_->initial_->exists(gtsam::Symbol('v', 0))) {
      posegraph_->initial_->insert(gtsam::Symbol('v', 0), first_dvl_vel_);
      auto prior_vel_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 0.1, 0.1, 0.1).finished());
      posegraph_->graph->add(
          gtsam::PriorFactor<gtsam::Vector3>(gtsam::Symbol('v', 0), first_dvl_vel_, prior_vel_noise));
    }

    latest_dvl_vel_ = first_dvl_vel_;
    latest_dvl_pose_ = latest_publish_pose_;
    return;
  }

  bool use_pseudo = (dvl_msg.fom >= posegraph_->params_->dvl_fom_threshold_ && !dvl_msg.velocity_valid);
  gtsam::Vector3 vel_to_use = dvl_vel_imu_frame;

  if (use_pseudo) {
    logger_.debug() << "Invalid DVL measurement (FOM: " << dvl_msg.fom << ")" << std::endl;
    gtsam::NavState T_world_to_imu_prop = pim_dvl_->predict(
        gtsam::NavState(latest_dvl_pose_, latest_dvl_pose_.rotation() * latest_dvl_vel_), posegraph_->priorImuBias_);
    vel_to_use = T_world_to_imu_prop.pose().rotation().inverse() * T_world_to_imu_prop.velocity();
    // R_imu_world * v_world
    // = v_imu_frame
  }

  // Already holding mtx_ lock, no need for separate lock
  posegraph_->addDvlVelocity(dvl_current_time, vel_to_use);
  posegraph_->current_dvl_foms_.push_back(dvl_msg.fom * 4);
  posegraph_->imu_rot_list_.push_back(imu_latest_rot_);

  latest_dvl_vel_ = vel_to_use;

  double dt_dvl = dvl_current_time - prev_dvl_time_;
  gtsam::Rot3 d_rot = dvl_prev_rot_.inverse() * imu_latest_rot_;
  gtsam::Point3 d_pos =
      d_rot.matrix() * gtsam::Point3(vel_to_use.x() * dt_dvl, vel_to_use.y() * dt_dvl, vel_to_use.z() * dt_dvl);
  gtsam::Pose3 d_pose(d_rot, d_pos);
  latest_publish_pose_ = latest_publish_pose_ * d_pose;

  if (posegraph_->params_->sensor_list_.isBaroUsed && posegraph_->params_->baro_params_.enable_interframe_integration &&
      posegraph_->getDepthValid()) {
    double dt = (last_baro_time_ > 0) ? (dvl_current_time - last_baro_time_) : dt_dvl;
    latest_publish_pose_ = baro_integrator_->correctPose(latest_publish_pose_, posegraph_->getDepthMeasurement(), dt);
  }

  latest_dvl_pose_ = latest_publish_pose_;
  pim_dvl_->resetIntegration();

  prev_dvl_time_ = dvl_current_time;
  dvl_prev_rot_ = imu_latest_rot_;

  // Store dense trajectory entry: accumulate relative pose from anchor keyframe
  dense_traj_relative_pose_ = dense_traj_relative_pose_ * d_pose;
  dense_trajectory_.push_back({dvl_current_time, dense_traj_relative_pose_, current_dense_kf_index_});

  publishState(latest_publish_pose_, false, dvl_current_time);
}

void PoseGraphBackend::processBarometer(const turtlmap::BarometerMeasurement& baro_msg,
                                                   bool unsafe /*=false*/)
{
  if (!is_rot_initialized_) {
    return;
  }

  std::unique_lock<std::mutex> lock(mtx_, std::defer_lock);
  if (!unsafe) {
    lock.lock();  // Only acquire lock if caller doesn't already hold it
  }

  double depth = ((baro_msg.pressure - posegraph_->params_->baro_params_.atm_press) * 100 / 9.81 / 997.0);

  if (first_depth_ == 0.0) {
    first_depth_ = depth;
    logger_.debug() << "First depth set: " << first_depth_ << " m" << std::endl;
  }

  posegraph_->setDepthMeasurement(depth - first_depth_);
  last_baro_time_ = baro_msg.timestamp;
  latest_baro_depth_ = depth - first_depth_;
}

void PoseGraphBackend::processStereo(const turtlmap::StereoMeasurement& stereo_msg, bool unsafe /*=false*/)
{
  if (!is_rot_initialized_ || !posegraph_->params_->sensor_list_.areCamsUsed) {
    return;
  }

  double prev_time = stereo_msg.timestamp;
  if (!unsafe) {
    const std::lock_guard<std::mutex> lock(mtx_);
    prev_time = (current_kf_time_ > 0.0) ? current_kf_time_ : stereo_msg.timestamp;
  } else {
    // Caller already holds mtx_
    prev_time = (current_kf_time_ > 0.0) ? current_kf_time_ : stereo_msg.timestamp;
  }
  // Debug: print the received 4x4 transformation (row-major)
  std::ostringstream stereo_mat_ss;
  stereo_mat_ss << "[Stereo] transformation matrix:\n";
  stereo_mat_ss << stereo_msg.transformation << std::endl;
  logger_.debug() << stereo_mat_ss.str();

  gtsam::Pose3 meas_pose_cam(stereo_msg.transformation);
  gtsam::Pose3 meas_pose_imu = transformLeftCameraToIMU(meas_pose_cam);

  logger_.debug() << "[Stereo] received stereo measurement between poses at times " << prev_time << " and "
                  << stereo_msg.timestamp << std::endl;

  stereo_measurement_time_ = stereo_msg.timestamp;
  // stereo_measurement_ready_ = true;
  latest_stereo_time_ = stereo_msg.timestamp;

  const auto prev_id_opt = getKFIdByTimestamp(stereo_msg.prev_kf_timestamp);
  const auto curr_id_opt = getKFIdByTimestamp(stereo_msg.curr_kf_timestamp);

  if (!prev_id_opt.has_value() || !curr_id_opt.has_value()) {
    return;
  }

  int prev_id = *prev_id_opt;
  int curr_id = *curr_id_opt;

  if (prev_id >= 0 && curr_id >= 0) {
    // Only lock if we don't already have it
    std::unique_lock<std::mutex> stereo_lock(mtx_, std::defer_lock);
    if (!unsafe) {
      stereo_lock.lock();
    }
    const bool have_prev = posegraph_->initial_->exists(gtsam::Symbol('x', prev_id)) ||
                           (posegraph_->result_ && posegraph_->result_->exists(gtsam::Symbol('x', prev_id)));
    const bool have_curr = posegraph_->initial_->exists(gtsam::Symbol('x', curr_id)) ||
                           (posegraph_->result_ && posegraph_->result_->exists(gtsam::Symbol('x', curr_id)));

    if (have_prev && have_curr) {
      // Retrieve poses from the factor graph
      gtsam::Pose3 pose_prev, pose_curr;

      // Get previous pose (prefer result over initial)
      if (posegraph_->result_ && posegraph_->result_->exists(gtsam::Symbol('x', prev_id)))
        pose_prev = posegraph_->result_->at<gtsam::Pose3>(gtsam::Symbol('x', prev_id));
      else
        pose_prev = posegraph_->initial_->at<gtsam::Pose3>(gtsam::Symbol('x', prev_id));

      // Get current pose (prefer result over initial)
      if (posegraph_->result_ && posegraph_->result_->exists(gtsam::Symbol('x', curr_id)))
        pose_curr = posegraph_->result_->at<gtsam::Pose3>(gtsam::Symbol('x', curr_id));
      else
        pose_curr = posegraph_->initial_->at<gtsam::Pose3>(gtsam::Symbol('x', curr_id));

      // Compute the delta from factor graph
      gtsam::Pose3 graph_delta = pose_prev.inverse() * pose_curr;

      // Print comparison
      logger_.debug() << "[Stereo] Comparison for x" << prev_id << " -> x" << curr_id << ":" << std::endl;
      logger_.debug() << "  Measurement (base frame):" << std::endl;
      logger_.debug() << "    Translation: [" << meas_pose_imu.translation().x() << ", "
                      << meas_pose_imu.translation().y() << ", " << meas_pose_imu.translation().z() << "]"
                      << std::endl;
      auto meas_quat = meas_pose_imu.rotation().toQuaternion();
      logger_.debug() << "    Rotation (quat): [w=" << meas_quat.w() << ", x=" << meas_quat.x()
                      << ", y=" << meas_quat.y() << ", z=" << meas_quat.z() << "]" << std::endl;

      logger_.debug() << "  Graph delta (base frame):" << std::endl;
      logger_.debug() << "    Translation: [" << graph_delta.translation().x() << ", " << graph_delta.translation().y()
                      << ", " << graph_delta.translation().z() << "]" << std::endl;
      auto graph_quat = graph_delta.rotation().toQuaternion();
      logger_.debug() << "    Rotation (quat): [w=" << graph_quat.w() << ", x=" << graph_quat.x()
                      << ", y=" << graph_quat.y() << ", z=" << graph_quat.z() << "]" << std::endl;
      
      const bool is_loop = stereo_msg.timestamp < 1;
      const auto& stereo_params = posegraph_->params_->stereo_params_;
      if(is_loop) {
        logger_.info() << "=== Adding Loop Closure to Graph!! ===" << std::endl;
      }
      
      posegraph_->addStereoBetweenFactor(
          meas_pose_imu,
          is_loop ? stereo_params.loop_translation_sigma : stereo_params.translation_sigma,
          is_loop ? stereo_params.loop_rotation_sigma : stereo_params.rotation_sigma,
          prev_id, curr_id);
    } else {
      logger_.debug() << "[Stereo] Skipping between factor; missing pose(s) for x" << prev_id << " or x" << curr_id
                      << std::endl;
    }
  }

  if (!unsafe) {
    const std::lock_guard<std::mutex> lock(mtx_);
    latest_stereo_time_ = stereo_msg.timestamp;
  } else {
    // Caller already holds mtx_
    latest_stereo_time_ = stereo_msg.timestamp;
  }
}

void PoseGraphBackend::triggerStereoKeyframe(double timestamp)
{
  if (!is_rot_initialized_ || !posegraph_->params_->sensor_list_.areCamsUsed) {
    return;
  }
  if (posegraph_->params_->external_kf_only_) {
    logger_.warn() << "[WARNING] Stereo KF triggered, but the system is configured to only accept external time-based "
                      "KFs (sent with the IMU)."
                   << std::endl;
    return;
  }

  {
    const std::lock_guard<std::mutex> lock(mtx_);

    stereo_measurement_time_ = timestamp;
    stereo_measurement_ready_ = true;
    latest_stereo_time_ = timestamp;

    logger_.debug() << "Stereo keyframe triggered at t=" << std::fixed << std::setprecision(6) << timestamp
                    << " (dt=" << (timestamp - current_kf_time_) * 1000.0 << " ms)"
                    << " next_idx=" << (posegraph_->index_ + 1) << std::endl;
  }

  // Check if keyframe is needed and notify keyframe thread
  if (shouldCreateKeyframe()) {
    kf_cv_.notify_one();
  }
}

void PoseGraphBackend::restoreBaroIntegrator(std::shared_ptr<BarometerIntegratorState> baro_integrator_state)
{
  baro_integrator_->loadState(*baro_integrator_state);
}

void PoseGraphBackend::initializeKeyframeState()
{
  if (posegraph_->index_ == 0) {
    logger_.debug() << "Creating first keyframe at t=" << std::fixed << std::setprecision(6) << current_kf_time_
                    << std::endl;
  } else {
    logger_.debug() << "Creating keyframe at t=" << std::fixed << std::setprecision(6) << current_kf_time_
                    << std::endl;
  }
  posegraph_->index_++;

  // Clear initial values for iSAM2 after first keyframe
  // iSAM2 maintains state internally, so we only pass new variables
  if (posegraph_->params_->optimization_mode_ == "isam2" && posegraph_->isam2_initialized_) {
    posegraph_->initial_->clear();
  }

  kf_timestamps_.push_back(current_kf_time_);
}

void PoseGraphBackend::addDvlBiasSymbol(double relative_time)
{
  gtsam::Symbol sym_d('d', posegraph_->index_);

  if (!posegraph_->initial_->exists(sym_d) && !(posegraph_->result_ && posegraph_->result_->exists(sym_d))) {
    posegraph_->initial_->insert(sym_d, posegraph_->priorDvlBias_);
  } else {
    std::cerr << "  NOTE: DVL bias key " << sym_d << " already exists; skipping re-insert" << std::endl;
  }

  posegraph_->smootherTimestamps[sym_d] = relative_time;

  // Add anchor prior for marginals computation
  if (posegraph_->params_->compute_marginals_) {
    const gtsam::Key d_anchor_key = gtsam::Symbol('d', posegraph_->index_ <= 1 ? posegraph_->index_ : 1);

    if (posegraph_->initial_->exists(d_anchor_key) ||
        (posegraph_->result_ && posegraph_->result_->exists(d_anchor_key))) {
      gtsam::imuBias::ConstantBias d_anchor_mean =
          (posegraph_->result_ && posegraph_->result_->exists(d_anchor_key))
              ? posegraph_->result_->at<gtsam::imuBias::ConstantBias>(d_anchor_key)
              : posegraph_->initial_->at<gtsam::imuBias::ConstantBias>(d_anchor_key);

      gtsam::Vector6 sigmas;
      sigmas << 1e-3, 1e-3, 1e-3, 1e6, 1e6, 1e6;
      auto noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

      posegraph_->graph->add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(d_anchor_key, d_anchor_mean, noise));
      logger_.debug() << "  Added DVL bias anchor prior at d" << (posegraph_->index_ <= 1 ? posegraph_->index_ : 1)
                      << " with current estimate" << std::endl;
    }
  }
}

void PoseGraphBackend::addVelocitySymbol(double relative_time)
{
  if (posegraph_->index_ > 1) {
    posegraph_->initial_->insert(gtsam::Symbol('v', posegraph_->index_),
                                 posegraph_->result_->at<gtsam::Vector3>(gtsam::Symbol('v', posegraph_->index_ - 1)));
  } else {
    posegraph_->initial_->insert(gtsam::Symbol('v', posegraph_->index_), gtsam::Vector3(0, 0, 0));
  }

  posegraph_->smootherTimestamps[gtsam::Symbol('v', posegraph_->index_)] = relative_time;
}

void PoseGraphBackend::addPoseSymbolWithPrediction()
{
  if (posegraph_->initial_->exists(gtsam::Symbol('x', posegraph_->index_))) {
    return;  // Already exists
  }

  try {
    // Get previous pose
    gtsam::Pose3 prevPose;
    if (posegraph_->result_ && posegraph_->result_->exists(gtsam::Symbol('x', posegraph_->index_ - 1))) {
      prevPose = posegraph_->result_->at<gtsam::Pose3>(gtsam::Symbol('x', posegraph_->index_ - 1));
    } else if (posegraph_->initial_->exists(gtsam::Symbol('x', posegraph_->index_ - 1))) {
      prevPose = posegraph_->initial_->at<gtsam::Pose3>(gtsam::Symbol('x', posegraph_->index_ - 1));
    } else {
      prevPose = posegraph_->prev_pose_;
    }

    // Get previous velocity
    gtsam::Vector3 prevVel(0, 0, 0);
    if (posegraph_->result_ && posegraph_->result_->exists(gtsam::Symbol('v', posegraph_->index_ - 1))) {
      prevVel = posegraph_->result_->at<gtsam::Vector3>(gtsam::Symbol('v', posegraph_->index_ - 1));
    } else if (posegraph_->initial_->exists(gtsam::Symbol('v', posegraph_->index_ - 1))) {
      prevVel = posegraph_->initial_->at<gtsam::Vector3>(gtsam::Symbol('v', posegraph_->index_ - 1));
    }

    // Predict new pose using IMU preintegration
    gtsam::NavState seed(prevPose, prevVel);
    gtsam::NavState pred = posegraph_->pim_->predict(seed, posegraph_->priorImuBias_);
    posegraph_->initial_->insert(gtsam::Symbol('x', posegraph_->index_), pred.pose());
  } catch (...) {
    posegraph_->initial_->insert(gtsam::Symbol('x', posegraph_->index_), posegraph_->prev_pose_);
  }
}

void PoseGraphBackend::savePostOptimizationState()
{
  prev_baro_integrator_state_ = std::make_shared<BarometerIntegratorState>(std::move(baro_integrator_->saveState()));
  prev_imu_integrator_state_ =
      std::make_shared<PreintegratedImuState>(std::move(savePreintegratedImuState(*posegraph_->pim_)));
  prev_dvl_integrator_state_ = std::make_shared<PreintegratedVelocityState>(std::move(posegraph_->pvm_->saveState()));
}

void PoseGraphBackend::addKeyframeSymbolsToGraph(double relative_time)
{
  const bool dvl_active = posegraph_->params_->sensor_list_.isDvlUsed;

  // Add DVL bias symbol if DVL is active
  if (dvl_active) {
    addDvlBiasSymbol(relative_time);
  }

  // Add IMU bias symbol
  posegraph_->initial_->insert(gtsam::Symbol('b', posegraph_->index_), posegraph_->priorImuBias_);
  posegraph_->smootherTimestamps[gtsam::Symbol('b', posegraph_->index_)] = relative_time;
  posegraph_->smootherTimestamps[gtsam::Symbol('x', posegraph_->index_)] = relative_time;

  // Add velocity symbol
  addVelocitySymbol(relative_time);

  // Add pose symbol with IMU prediction
  addPoseSymbolWithPrediction();
}

void PoseGraphBackend::addSensorFactorsToGraph()
{
  // Add barometric factor if enabled
  if (posegraph_->params_->sensor_list_.isBaroUsed) {
    last_baro_factor_ = posegraph_->addBarometricFactor(posegraph_->getDepthMeasurement(), 0.01, posegraph_->index_);
  }

  // Add IMU factor (always present)
  last_imu_factor_ = posegraph_->addImuFactor();


  // Add DVL factor if active and measurements available
  const bool dvl_active = posegraph_->params_->sensor_list_.isDvlUsed;
  if (dvl_active) {
    const bool has_dvl_measurement =
        !posegraph_->current_dvl_vels_.empty() && !posegraph_->current_dvl_timestamps_.empty();

    if (has_dvl_measurement && is_using_dvl_v2_factor) {
      auto dvl_factor = posegraph_->addDvlFactorImuRot();
    } else if (!has_dvl_measurement) {
      logger_.debug() << "Skipping DVL factor at kf " << posegraph_->index_ << " (no DVL samples)" << std::endl;
    } else {
      throw std::runtime_error("DVL factor path not implemented (expected v2)");
    }
  }
}

void PoseGraphBackend::optimizeGraph()
{
  // Log graph state before optimization (for debugging)
  if (posegraph_->index_ <= 5) {
    logger_.debug() << "Graph before optimization (index=" << posegraph_->index_ << "): " << posegraph_->graph->size()
                    << " factors" << std::endl;
  }

  // Start timing
  auto t_start = std::chrono::high_resolution_clock::now();

  // Select optimization backend based on configuration
  const std::string& opt_mode = posegraph_->params_->optimization_mode_;

  if (opt_mode == "isam2") {
    posegraph_->optimizePoseGraphISAM2();
  } else if (opt_mode == "batch") {
    posegraph_->optimizePoseGraph();
  } else if (opt_mode == "fixed_lag") {
    posegraph_->optimizePoseGraphSmoother();
  } else {
    throw std::runtime_error("Unknown optimization_mode: '" + opt_mode + "'");
  }

  // End timing and log
  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

  // Write timing to file: timestamp, keyframe_index, elapsed_time_ms, optimization_mode
  std::ofstream timing_ofs(optimization_timing_file_, std::ios::app);
  if (timing_ofs.is_open()) {
    timing_ofs << std::fixed << std::setprecision(6) << current_kf_time_ << " "
               << posegraph_->index_ << " "
               << std::setprecision(3) << elapsed_ms << " "
               << opt_mode << std::endl;
  }

  // Log completion
  if (posegraph_->index_ <= 5) {
    logger_.debug() << "Optimization complete (index=" << posegraph_->index_ << ") in "
                    << elapsed_ms << " ms" << std::endl;
  }
}

void PoseGraphBackend::computeAndStoreMarginals()
{
  const gtsam::Symbol pose_symbol('x', posegraph_->index_);

  // Only compute marginals if enabled in config
  if (!posegraph_->params_->compute_marginals_) {
    logger_.debug() << "Marginals computation disabled (compute_marginals=false)" << std::endl;
    return;
  }

  try {
    const gtsam::NonlinearFactorGraph* graph_ptr = nullptr;
    const std::string& opt_mode = posegraph_->params_->optimization_mode_;

    // Select the appropriate graph based on optimization mode
    if (opt_mode == "isam2") {
      graph_ptr = &posegraph_->isam2_->getFactorsUnsafe();
    } else if (opt_mode == "batch") {
      graph_ptr = posegraph_->graph.get();
    } else if (opt_mode == "fixed_lag") {
      graph_ptr = &posegraph_->smootherISAM2.getFactors();
    } else {
      // Fallback to main graph
      graph_ptr = posegraph_->graph.get();
    }

    // Sanity checks
    bool in_values = (posegraph_->result_ && posegraph_->result_->exists(pose_symbol));
    bool in_graph = false;

    if (graph_ptr) {
      for (const auto& f : *graph_ptr) {
        if (!f) continue;
        for (gtsam::Key k : f->keys()) {
          if (k == pose_symbol.key()) {
            in_graph = true;
            break;
          }
        }
        if (in_graph) break;
      }
    }

    if (in_values && in_graph) {
      logger_.debug() << "Computing marginals for key x" << posegraph_->index_ << std::endl;
      gtsam::Marginals marginals(*graph_ptr, *posegraph_->result_);
      gtsam::Matrix covariance = marginals.marginalCovariance(pose_symbol);
      latest_kf_pose_cov_ = covariance;

      if (kf_pose_covariances_world_.size() <= static_cast<size_t>(posegraph_->index_))
        kf_pose_covariances_world_.resize(posegraph_->index_ + 1);
      kf_pose_covariances_world_[posegraph_->index_] = covariance;

      // Store first-optimization marginal and pose (never overwritten by later optimizations)
      if (kf_first_opt_covariances_world_.size() <= static_cast<size_t>(posegraph_->index_))
        kf_first_opt_covariances_world_.resize(posegraph_->index_ + 1);
      if (kf_first_opt_poses_.size() <= static_cast<size_t>(posegraph_->index_))
        kf_first_opt_poses_.resize(posegraph_->index_ + 1);
      if (kf_first_opt_covariances_world_[posegraph_->index_].rows() == 0) {
        kf_first_opt_covariances_world_[posegraph_->index_] = covariance;
        kf_first_opt_poses_[posegraph_->index_] = posegraph_->result_->at<gtsam::Pose3>(pose_symbol);
      }

      logger_.debug() << "Marginal at key x" << posegraph_->index_ << std::endl;
      logger_.debug() << covariance << std::endl;

      // Append XY translation covariance to live covariance file
      if (live_vis_enabled_) {
        // Extract top-left 3x3 for translation; covariance ordering assumed [rx
        // ry rz tx ty tz] If ordering differs, adjust indices accordingly.
        double cov_xx = covariance(3, 3);
        double cov_xy = covariance(3, 4);
        double cov_yy = covariance(4, 4);
        // Position in the exact same frame as the live trajectory file
        // (transformImuToCamera)
        gtsam::Pose3 kf_pose_imu = posegraph_->result_->at<gtsam::Pose3>(pose_symbol);
        gtsam::Pose3 kf_pose_cam = transformImuToCamera(kf_pose_imu);
        std::ofstream cov_ofs(live_traj_file_ + ".cov", std::ios::app);
        if (cov_ofs.is_open()) {
          cov_ofs << std::fixed << std::setprecision(6) << current_kf_time_ << ' ' << kf_pose_cam.translation().x()
                  << ' ' << kf_pose_cam.translation().y() << ' ' << kf_pose_cam.translation().z() << ' ' << cov_xx
                  << ' ' << cov_xy << ' ' << cov_yy << '\n';
          cov_ofs.flush();
        }
      }
    } else {
      std::cerr << "Skipping marginals: key x" << posegraph_->index_ << (in_values ? " present" : " missing")
                << " in Values," << (in_graph ? " present" : " missing") << " in factor graph" << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error computing marginals at key x" << posegraph_->index_ << ": " << e.what()
              << " (no first-opt pose will be available for this keyframe)" << std::endl;
  }
}

void PoseGraphBackend::updateStateAfterOptimization()
{
  // Update IMU bias and reset preintegration
  posegraph_->priorImuBias_ =
      posegraph_->result_->at<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', posegraph_->index_));
  posegraph_->pim_->resetIntegrationAndSetBias(posegraph_->priorImuBias_);
  pim_dvl_->resetIntegrationAndSetBias(posegraph_->priorImuBias_);

  // Reset barometer integrator at keyframe - we have a new optimized reference
  if (posegraph_->params_->sensor_list_.isBaroUsed && posegraph_->params_->baro_params_.enable_interframe_integration &&
      baro_integrator_) {
    baro_integrator_->reset();
  }

  // Update latest pose and velocity estimates
  latest_kf_pose_ = posegraph_->result_->at<gtsam::Pose3>(gtsam::Symbol('x', posegraph_->index_));
  latest_publish_pose_ = latest_kf_pose_;
  latest_dvl_pose_ = latest_kf_pose_;
  latest_dvl_vel_ = posegraph_->result_->at<gtsam::Vector3>(gtsam::Symbol('v', posegraph_->index_));

  // Update DVL bias if using DVL v2 factor
  const bool dvl_active = posegraph_->params_->sensor_list_.isDvlUsed;
  if (is_using_dvl_v2_factor && dvl_active) {
    posegraph_->priorDvlBias_ =
        posegraph_->result_->at<gtsam::imuBias::ConstantBias>(gtsam::Symbol('d', posegraph_->index_));
    posegraph_->pvm_->resetIntegrationAndBias(posegraph_->priorDvlBias_);
  }

  // Update previous pose and time
  posegraph_->prev_pose_ = posegraph_->result_->at<gtsam::Pose3>(gtsam::Symbol('x', posegraph_->index_));
  posegraph_->prev_kf_time_ = current_kf_time_;

  // Reset dense trajectory accumulator for new keyframe interval
  dense_traj_relative_pose_ = gtsam::Pose3::Identity();
  current_dense_kf_index_ = posegraph_->index_;
}

void PoseGraphBackend::publishKeyframePose()
{
  // Transform the pose into the IMU frame from the abstract "base_link"
  // gtsam::Rot3 R_BS(posegraph_->params_->extrinsics_.T_BS.block<3, 3>(0, 0));
  // gtsam::Point3 t_BS(posegraph_->params_->extrinsics_.T_BS.block<3, 1>(0, 3));
  // gtsam::Pose3 T_BS(R_BS, t_BS);
  // gtsam::Pose3 T_W_S = latest_kf_pose_ * T_BS.inverse();

  // Transform the pose in IMU into the left camera frame
  // gtsam::Rot3 R_S_Lc(posegraph_->params_->extrinsics_.T_SLc.block(0, 0, 3, 3));
  // gtsam::Point3 t_S_Lc(posegraph_->params_->extrinsics_.T_SLc.block(0, 3, 3, 1));
  gtsam::Pose3 T_S_Lc(posegraph_->params_->extrinsics_.T_SLc);
  gtsam::Pose3 latest_kf_pose_Lc = transformImuToCamera(latest_kf_pose_);  ;

  // // Transform to NED frame
  // gtsam::Pose3 T_NED = gtsam::Pose3(gtsam::Rot3(0.0, 1.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));
  // gtsam::Pose3 latest_kf_pose_NED = T_NED.inverse() * latest_kf_pose_Lc;

  const auto& p = latest_kf_pose_Lc;
  auto q = p.rotation().toQuaternion();

  // Log keyframe pose
  logger_.debug() << std::fixed << std::setprecision(6) << "KF pose | t: " << current_kf_time_
                  << ", idx: " << posegraph_->index_ << ", pos: [" << p.translation().x() << ", " << p.translation().y()
                  << ", " << p.translation().z() << "]" << std::endl;

  // Store in trajectory history
  traj_timestamps_.push_back(current_kf_time_);
  traj_poses_.push_back(p);

  // Publish keyframe through output thread with its own timestamp
  // Note: The .keyframes file will be written by the output thread
  publishState(latest_kf_pose_, true, current_kf_time_);
}

bool PoseGraphBackend::shouldMergeWithPreviousKeyframe(double new_kf_time)
{
  // Need at least 2 keyframes to merge (can't merge with x0 which has priors)
  if (posegraph_->index_ < 1) {
    logger_.debug() << "[KeyframeMerge] Not merging; posegraph index < 1" << std::endl;
    return false;
  }

  double delta = new_kf_time - prev_kf_time_;
  double min_delta = posegraph_->params_->min_kf_delta_time_;

  if (delta < min_delta && delta > 0) {
    logger_.info() << "[KeyframeMerge] Delta " << delta << "s < min " << min_delta
                   << "s, will merge with previous keyframe" << std::endl;
    return true;
  }
  logger_.debug() << "[KeyframeMerge] delta >= min_delta or delta <= 0. Delta:" << delta << " min_delta: " << min_delta
                  << std::endl;

  return false;
}

std::optional<size_t> PoseGraphBackend::getKFIdByTimestamp(double timestamp, double eps)
{
  // Binary search for the closest timestamp within eps tolerance
  auto it = std::lower_bound(kf_timestamps_.begin(), kf_timestamps_.end(), timestamp);

  // Check the element at the lower_bound position
  if (it != kf_timestamps_.end() && std::abs(*it - timestamp) < eps) {
    return std::distance(kf_timestamps_.begin(), it);
  }

  // Check the previous element if we're not at the beginning
  if (it != kf_timestamps_.begin()) {
    --it;
    if (std::abs(*it - timestamp) < eps) {
      return std::distance(kf_timestamps_.begin(), it);
    }
  }

  return std::nullopt;
}

void PoseGraphBackend::kfLoop()
{
  logger_.debug() << "Keyframe loop started" << std::endl;

  while (running_) {
    std::unique_lock<std::mutex> lk(mtx_);
    kf_cv_.wait(lk, [this] { return new_kf_flag_ || !running_; });

    if (!running_) break;

    initializeKeyframeState();
    const double relative_time = current_kf_time_ - first_kf_time_;

    addKeyframeSymbolsToGraph(relative_time);
    addSensorFactorsToGraph();    
    optimizeGraph();
    computeAndStoreMarginals();
    updateStateAfterOptimization();
    publishKeyframePose();


    // Keyframe setup is complete - resume processing loop
    kf_opt_in_progress_ = false;
    kf_opt_done_cv_.notify_one();

    new_kf_flag_ = false;
    lk.unlock();
  }

  logger_.debug() << "Keyframe loop ended" << std::endl;
}

// bool PoseGraphBackend::saveTrajectory(const std::string& filename)
// {
//   std::string pose_file_name = filename + ".txt";

//   std::ofstream pose_file;
//   pose_file.open(pose_file_name);

//   if (!pose_file.is_open()) {
//     std::cerr << "Failed to open file: " << pose_file_name << std::endl;
//     return false;
//   }

//   for (size_t i = 0; i < traj_poses_.size(); i++) {
//     pose_file << std::fixed << std::setprecision(12) << traj_timestamps_[i] << " " << traj_poses_[i].translation().x()
//               << " " << traj_poses_[i].translation().y() << " " << traj_poses_[i].translation().z() << " "
//               << traj_poses_[i].rotation().toQuaternion().x() << " " << traj_poses_[i].rotation().toQuaternion().y()
//               << " " << traj_poses_[i].rotation().toQuaternion().z() << " "
//               << traj_poses_[i].rotation().toQuaternion().w() << std::endl;
//   }

//   pose_file.close();
//   logger_.info() << "Trajectory saved to " << pose_file_name << " (" << traj_poses_.size() << " poses)" << std::endl;

//   return true;
// }

bool PoseGraphBackend::saveTrajectory(const std::string& filename)
{
  std::lock_guard<std::mutex> lock(mtx_);
  
  std::string pose_file_name = filename + ".txt";

  std::ofstream pose_file;
  pose_file.open(pose_file_name);

  if (!pose_file.is_open()) {
    std::cerr << "Failed to open file: " << pose_file_name << std::endl;
    return false;
  }

  // Save keyframe poses (x0, x1, x2, ...) with their timestamps
  const std::string& opt_mode = posegraph_->params_->optimization_mode_;

  if (opt_mode == "isam2" || opt_mode == "batch") {

    for (size_t i = 0; i < kf_timestamps_.size(); i++) {
      gtsam::Symbol sym_x('x', static_cast<int>(i));
      
      gtsam::Pose3 pose_imu;
      bool found = false;
      
      if (posegraph_->result_ && posegraph_->result_->exists(sym_x)) {
        pose_imu = posegraph_->result_->at<gtsam::Pose3>(sym_x);
        found = true;
      } else if (posegraph_->initial_->exists(sym_x)) {
        pose_imu = posegraph_->initial_->at<gtsam::Pose3>(sym_x);
        found = true;
      }
      
      if (!found) {
        logger_.warn() << "Keyframe pose x" << i << " not found, skipping" << std::endl;
        continue;
      }
      
      // Transform to camera/NED frame (same as publishKeyframePose)
      gtsam::Pose3 T_S_Lc(posegraph_->params_->extrinsics_.T_SLc);
      gtsam::Pose3 pose_cam = transformImuToCamera(pose_imu);
      
      // Write in TUM format: timestamp x y z qx qy qz qw
      pose_file << std::fixed << std::setprecision(12) << kf_timestamps_[i] << " "
                << pose_cam.translation().x() << " "
                << pose_cam.translation().y() << " "
                << pose_cam.translation().z() << " "
                << pose_cam.rotation().toQuaternion().x() << " "
                << pose_cam.rotation().toQuaternion().y() << " "
                << pose_cam.rotation().toQuaternion().z() << " "
                << pose_cam.rotation().toQuaternion().w() << std::endl;
    }
  
    pose_file.close();
    logger_.info() << "Trajectory saved to " << pose_file_name
                   << " (" << kf_timestamps_.size() << " keyframe poses)" << std::endl;
  
    return true;
  } else if (opt_mode == "fixed_lag") {
      for (size_t i = 0; i < traj_poses_.size(); i++) {
        pose_file << std::fixed << std::setprecision(12) << traj_timestamps_[i] << " " 
                  << traj_poses_[i].translation().x() << " " 
                  << traj_poses_[i].translation().y() << " " 
                  << traj_poses_[i].translation().z() << " "
                  << traj_poses_[i].rotation().toQuaternion().x() << " " 
                  << traj_poses_[i].rotation().toQuaternion().y() << " " 
                  << traj_poses_[i].rotation().toQuaternion().z() << " "
                  << traj_poses_[i].rotation().toQuaternion().w() << std::endl;
      }

      pose_file.close();
      logger_.info() << "Trajectory saved to " << pose_file_name << " (" << traj_poses_.size() << " poses)" << std::endl;

      return true;
  } else {
    throw std::runtime_error("Unknown optimization_mode: '" + opt_mode + "'");
  }
  return false;
}

bool PoseGraphBackend::saveDenseTrajectory(const std::string& filename)
{
  std::lock_guard<std::mutex> lock(mtx_);

  std::string dense_file_name = filename + "_dense.txt";
  std::ofstream file(dense_file_name);

  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << dense_file_name << std::endl;
    return false;
  }

  size_t poses_written = 0;
  for (const auto& entry : dense_trajectory_) {
    // Look up the FINAL optimized pose for this entry's anchor keyframe
    gtsam::Symbol sym_x('x', entry.kf_index);

    gtsam::Pose3 T_world_kf;
    bool found = false;

    if (posegraph_->result_ && posegraph_->result_->exists(sym_x)) {
      T_world_kf = posegraph_->result_->at<gtsam::Pose3>(sym_x);
      found = true;
    } else if (posegraph_->initial_ && posegraph_->initial_->exists(sym_x)) {
      T_world_kf = posegraph_->initial_->at<gtsam::Pose3>(sym_x);
      found = true;
    }

    if (!found) {
      logger_.warn() << "Anchor keyframe x" << entry.kf_index
                     << " not found for dense pose at t=" << entry.timestamp << std::endl;
      continue;
    }

    // Reconstruct absolute pose: T_world_pose = T_world_kf * T_kf_pose
    gtsam::Pose3 T_world_pose = T_world_kf * entry.T_kf_pose;

    // Transform to camera frame for output (same as saveTrajectory)
    gtsam::Pose3 pose_cam = transformImuToCamera(T_world_pose);
    auto q = pose_cam.rotation().toQuaternion();

    // Write in TUM format: timestamp x y z qx qy qz qw
    file << std::fixed << std::setprecision(12)
         << entry.timestamp << " "
         << pose_cam.translation().x() << " "
         << pose_cam.translation().y() << " "
         << pose_cam.translation().z() << " "
         << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";

    poses_written++;
  }

  file.close();
  logger_.info() << "Dense trajectory saved to " << dense_file_name
                 << " (" << poses_written << " poses from " << dense_trajectory_.size() << " entries)" << std::endl;

  return true;
}

void PoseGraphBackend::waitForCompletion()
{
  while (data_provider_->isRunning()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  logger_.debug() << "Data streaming completed" << std::endl;
}

std::optional<Eigen::Matrix4d> PoseGraphBackend::getPoseAtTime(double t)
{
  std::lock_guard<std::mutex> lock(traj_mtx_);

  if (traj_pose_queue_.size() < 2) {
    return std::nullopt;
  }

  const double front_ts = traj_pose_queue_.front().timestamp;
  if (front_ts > t) {
    return std::nullopt;
  }

  // Use std::lower_bound to find first element with timestamp >= t
  auto it = std::lower_bound(
      traj_pose_queue_.begin(), traj_pose_queue_.end(), t,
      [](const auto& pose, double ts) { return pose.timestamp < ts; });

  // If t is beyond all timestamps, no valid interval
  if (it == traj_pose_queue_.end()) {
    return std::nullopt;
  }

  // If lower_bound points to the first element, t is before or at the first timestamp
  // but we already checked front_ts > t above, so this means t == front_ts exactly
  if (it == traj_pose_queue_.begin()) {
    return it->pose.matrix();
  }

  const auto& b = *it;
  const auto& a = *std::prev(it);

  double dt = b.timestamp - a.timestamp;
  if (dt <= 0.0) return std::nullopt;

  double alpha = (t - a.timestamp) / dt;
  gtsam::Pose3 delta = a.pose.inverse() * b.pose;
  gtsam::Vector6 logv = gtsam::Pose3::Logmap(delta);
  gtsam::Pose3 interp = a.pose * gtsam::Pose3::Expmap(logv * alpha);

  return interp.matrix();
}

bool PoseGraphBackend::hasPoseForTime(double t)
{
  std::lock_guard<std::mutex> lock(traj_mtx_);
  if (traj_pose_queue_.size() < 2) return false;
  return traj_pose_queue_.front().timestamp <= t && traj_pose_queue_.back().timestamp >= t;
}

void PoseGraphBackend::setTrajPoseQueueMaxSize(size_t n)
{
  std::lock_guard<std::mutex> lock(traj_mtx_);
  traj_pose_queue_max_size_ = (n < 2) ? 2 : n;
  while (traj_pose_queue_.size() > traj_pose_queue_max_size_) {
    traj_pose_queue_.pop_front();
  }
}

std::optional<Eigen::Matrix4d> PoseGraphBackend::getPoseAtKeyframe(int index)
{
  std::lock_guard<std::mutex> lock(mtx_);

  // Allow initial keyframe at index 0
  if (index < 0) {
    return std::nullopt;
  }

  gtsam::Symbol sym_x('x', index);

  if (posegraph_->result_ && posegraph_->result_->exists(sym_x)) {
    return transformImuToCamera(posegraph_->result_->at<gtsam::Pose3>(sym_x)).matrix();
  }

  if (posegraph_->initial_->exists(sym_x)) {
    return transformImuToCamera(posegraph_->initial_->at<gtsam::Pose3>(sym_x)).matrix();
  }

  return std::nullopt;
}

std::optional<Eigen::Matrix4d> PoseGraphBackend::getPoseAtKeyframePublished(int index)
{
  // Allow initial keyframe at index 0
  if (index < 0) {
    return std::nullopt;
  }

  gtsam::Symbol sym_x('x', index);

  std::lock_guard<std::mutex> lock(mtx_);

  bool in_result = posegraph_->result_ && posegraph_->result_->exists(sym_x);
  bool in_initial = posegraph_->initial_->exists(sym_x);

  if (!in_result && !in_initial) {
    return std::nullopt;
  }

  gtsam::Pose3 pose_imu;
  if (in_result) {
    pose_imu = posegraph_->result_->at<gtsam::Pose3>(sym_x);
  } else {
    pose_imu = posegraph_->initial_->at<gtsam::Pose3>(sym_x);
  }

  gtsam::Pose3 pose_ned = transformImuToCamera(pose_imu);
  return pose_ned.matrix();
}

std::vector<std::pair<int, Eigen::Matrix4d>> PoseGraphBackend::getUpdatedGraph()
{
  std::lock_guard<std::mutex> lock(mtx_);
  std::vector<std::pair<int, Eigen::Matrix4d>> updated_graph;

  if (!posegraph_->result_) {
    return updated_graph;
  }

  updated_graph.reserve(kf_timestamps_.size());
  for (size_t i = 0; i < kf_timestamps_.size(); ++i) {
    gtsam::Symbol sym_x('x', static_cast<int>(i));
    if (!posegraph_->result_->exists(sym_x)) {
      continue;
    }
    Pose3 updated_pose(posegraph_->result_->at<gtsam::Pose3>(sym_x));
    updated_graph.emplace_back(static_cast<int>(i),
                               transformImuToCamera(updated_pose).matrix());
  }

  return updated_graph;
}

std::optional<Eigen::Matrix4d> PoseGraphBackend::getLatestPose()
{
  std::lock_guard<std::mutex> lock(traj_mtx_);
  if (traj_pose_queue_.empty()) {
    return std::nullopt;
  }
  return traj_pose_queue_.back().pose.matrix();
}

std::optional<Eigen::Matrix4d> PoseGraphBackend::getFirstOptimizedKeyframePose(double kf_timestamp, double eps/*=1e-2*/)
{
  std::lock_guard<std::mutex> lock(mtx_);

  if (kf_timestamps_.empty()) {
    logger_.warn() << "No keyframes available" << std::endl;
    return std::nullopt;
  }

  // Check if query is out of range
  if (kf_timestamp < kf_timestamps_.front() - eps || kf_timestamp > kf_timestamps_.back() + eps) {
    logger_.warn() << "Query timestamp " << kf_timestamp << " is outside keyframe range ["
                      << kf_timestamps_.front() << ", " << kf_timestamps_.back() << "]" << std::endl;
    return std::nullopt;
  }

  // Find first keyframe with timestamp >= kf_timestamp
  auto it = std::lower_bound(kf_timestamps_.begin(), kf_timestamps_.end(), kf_timestamp);

  // Check both the element at it and the one before to find the closest
  size_t closest_idx = 0;
  double min_diff = std::numeric_limits<double>::max();

  if (it != kf_timestamps_.end()) {
    double diff = *it - kf_timestamp;
    if (diff < min_diff) {
      min_diff = diff;
      closest_idx = std::distance(kf_timestamps_.begin(), it);
    }
  }

  if (it != kf_timestamps_.begin()) {
    auto prev = std::prev(it);
    double diff = kf_timestamp - *prev;
    if (diff < min_diff) {
      min_diff = diff;
      closest_idx = std::distance(kf_timestamps_.begin(), prev);
    }
  }

  if (min_diff > eps) {
    logger_.warn() << "Closest keyframe is " << min_diff << "s away from query timestamp "
                      << kf_timestamp << " (threshold: " << eps << "s)" << std::endl;
    return std::nullopt;
  }

  // kf_first_opt_poses_ is indexed by keyframe index but only written when marginal
  // computation succeeds (computeAndStoreMarginals), which happens after the keyframe's
  // timestamp is already in kf_timestamps_. So closest_idx can point past the end (newest
  // keyframe, store not yet done) or at a slot a failed marginal left default-constructed.
  // Both are "no first-optimized pose available" -- report that instead of throwing or
  // silently returning identity.
  if (closest_idx >= kf_first_opt_poses_.size()) {
    logger_.warn() << "First-opt pose for keyframe " << closest_idx
                   << " not yet stored (have " << kf_first_opt_poses_.size() << ")" << std::endl;
    return std::nullopt;
  }
  if (closest_idx >= kf_first_opt_covariances_world_.size() ||
      kf_first_opt_covariances_world_[closest_idx].rows() == 0) {
    logger_.warn() << "First-opt pose for keyframe " << closest_idx
                   << " was never stored (marginal computation failed)" << std::endl;
    return std::nullopt;
  }

  return transformImuToCamera(kf_first_opt_poses_.at(closest_idx)).matrix();
}

std::optional<double> PoseGraphBackend::getFirstPoseTime()
{
  std::lock_guard<std::mutex> lock(traj_mtx_);
  if (traj_pose_queue_.empty()) {
    return std::nullopt;
  }
  return traj_pose_queue_.front().timestamp;
}

std::optional<Matrix6> PoseGraphBackend::getCovAtTime(double timestamp, bool use_first_opt)
{
  gtsam::Pose3 interp_pose_ned;

  // Use traj_mtx_ for trajectory queue access
  {
    std::lock_guard<std::mutex> traj_lock(traj_mtx_);

    if (traj_pose_queue_.size() < 2) {
      logger_.debug() << "Traj Pose Queue too small" << std::endl;
      return std::nullopt;
    }

    const double front_ts = traj_pose_queue_.front().timestamp;
    const double back_ts = traj_pose_queue_.back().timestamp;

    if (timestamp < front_ts || timestamp > back_ts) {
      logger_.debug() << "timestamp="<<timestamp << "out of range ["<<front_ts << ','<<back_ts << "]" << std::endl;
      logger_.debug() << "KF Times: ";
      for(auto kf_t : kf_timestamps_) logger_.debug() << kf_t << ", ";
      logger_.debug() << std::endl; 

      return std::nullopt;
    }

    bool found = false;
    for (size_t i = 1; i < traj_pose_queue_.size(); ++i) {
      const auto& a = traj_pose_queue_.at(i - 1);
      const auto& b = traj_pose_queue_.at(i);
      if (b.timestamp < timestamp) continue;
      double dt = b.timestamp - a.timestamp;
      if (dt <= 0.0) throw std::runtime_error("Invalid time spacing for interpolation");
      double alpha = (timestamp - a.timestamp) / dt;
      gtsam::Pose3 delta = a.pose.inverse() * b.pose;
      gtsam::Vector6 logv = gtsam::Pose3::Logmap(delta);
      interp_pose_ned = a.pose * gtsam::Pose3::Expmap(logv * alpha);
      found = true;
      break;
    }
    
    if (!found) {
      logger_.debug() << "couldn't find timestamp pair straddling="<< timestamp <<". options: ";
      for(auto e : traj_pose_queue_) {
        logger_.debug() << e.timestamp << ", ";
      }
      logger_.debug() << std::endl;
      
      return std::nullopt;
    }
  }

  // Use mtx_ for posegraph state access
  std::lock_guard<std::mutex> lock(mtx_);

  if (kf_timestamps_.empty()) {
    logger_.debug() << "kf timestamps empty." << std::endl;
    return std::nullopt;
  }

  size_t k = 0;
  {
    auto it = std::upper_bound(kf_timestamps_.begin(), kf_timestamps_.end(), timestamp);
    if (it == kf_timestamps_.begin()) throw std::runtime_error("No prior keyframe for timestamp");
    k = static_cast<size_t>((it - kf_timestamps_.begin()) - 1);
  }

  gtsam::Symbol pose_key('x', static_cast<int>(k));
  // Retrieve the keyframe pose in the pose graph
  gtsam::Pose3 T_WB_k;
  if (posegraph_->result_ && posegraph_->result_->exists(pose_key))
    T_WB_k = posegraph_->result_->at<gtsam::Pose3>(pose_key);
  else if (posegraph_->initial_ && posegraph_->initial_->exists(pose_key))
    T_WB_k = posegraph_->initial_->at<gtsam::Pose3>(pose_key);
  else
    throw std::runtime_error("Keyframe pose not available");

  // compute pose in NED
  gtsam::Rot3 R_BS(posegraph_->params_->extrinsics_.T_BS.block<3, 3>(0, 0));
  gtsam::Point3 t_BS(posegraph_->params_->extrinsics_.T_BS.block<3, 1>(0, 3));
  gtsam::Pose3 T_BS(R_BS, t_BS);
  gtsam::Rot3 R_S_Lc(posegraph_->params_->extrinsics_.T_SLc.block(0, 0, 3, 3));
  gtsam::Point3 t_S_Lc(posegraph_->params_->extrinsics_.T_SLc.block(0, 3, 3, 1));
  gtsam::Pose3 T_S_Lc(R_S_Lc, t_S_Lc);
  gtsam::Pose3 T_NED(gtsam::Rot3(0.0, 1.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));
  gtsam::Pose3 B = T_BS.inverse() * T_S_Lc.inverse();
  // queried pose in the NED frame
  gtsam::Pose3 T_WB_t = T_NED * interp_pose_ned * B.inverse();

  gtsam::Symbol pose_k('x', static_cast<int>(k));
  gtsam::Matrix Sigma_W_k;

  // Check for first-optimization covariance if requested
  if (use_first_opt && kf_first_opt_covariances_world_.size() > k &&
      kf_first_opt_covariances_world_[k].rows() == 6) {
    Sigma_W_k = kf_first_opt_covariances_world_[k];
  } else if (kf_pose_covariances_world_.size() > k && kf_pose_covariances_world_[k].rows() == 6) {
    Sigma_W_k = kf_pose_covariances_world_[k];
  } else {
    gtsam::NonlinearFactorGraph posegraph;
    bool posegraph_valid = true;

    if (!posegraph_->params_->using_smoother_) {
      if (posegraph_->graph) {
        posegraph = *(posegraph_->graph);
      } else {
        posegraph_valid = false;
      }
    } else
      posegraph = posegraph_->smootherISAM2.getFactors();

    bool in_values = (posegraph_->result_ && posegraph_->result_->exists(pose_k));

    bool in_graph = false;
    if (posegraph_valid) {
      for (const auto& f : posegraph) {
        if (!f) continue;
        for (gtsam::Key key : f->keys()) {
          if (key == pose_k.key()) {
            in_graph = true;
            break;
          }
        }
        if (in_graph) break;
      }
    }
    if (!(in_values && in_graph)) throw std::runtime_error("Keyframe marginal not available");

    gtsam::Marginals marginals(posegraph, *posegraph_->result_);
    Sigma_W_k = marginals.marginalCovariance(pose_k);
    if (kf_pose_covariances_world_.size() <= k) kf_pose_covariances_world_.resize(k + 1);
    if(!use_first_opt) kf_pose_covariances_world_[k] = Sigma_W_k;
  }

  // finally, the marginal at k to t using the pose transformation
  gtsam::Pose3 Delta_k_to_t = T_WB_k.inverse() * T_WB_t;
  gtsam::Matrix Ad_Delta_inv = Delta_k_to_t.inverse().AdjointMap();
  gtsam::Matrix Sigma_W_t = Ad_Delta_inv * Sigma_W_k * Ad_Delta_inv.transpose();

  gtsam::Matrix Ad_T_WB_t = T_WB_t.AdjointMap();
  gtsam::Matrix Sigma_L_t = Ad_T_WB_t * Sigma_W_t * Ad_T_WB_t.transpose();
  Sigma_L_t = 0.5 * (Sigma_L_t + Sigma_L_t.transpose());

  // convert Sigma_L_t to the desired return format matrix6
  Matrix6 Sigma_L_t_eigen = Sigma_L_t;
  return Sigma_L_t_eigen;
}

}  // namespace turtlmap
