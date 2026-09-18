// Non-ROS backend for pose graph optimization
// Uses HDF5 data streaming instead of ROS topics

#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "BarometerIntegrator.h"
#include "DataProvider.h"
#include "Logger.h"
#include "Posegraph.h"
#include "SensorData.h"
#include "PreintegratedImuState.h"

struct TrajPoseQueueEntry
{
  double timestamp;
  gtsam::Pose3 pose;
};

inline bool operator<(const TrajPoseQueueEntry& a, const TrajPoseQueueEntry& b) { return a.timestamp < b.timestamp; };

// Dense trajectory entry storing pose relative to anchor keyframe
struct DenseTrajectoryEntry
{
  double timestamp;
  gtsam::Pose3 T_kf_pose;  // Pose RELATIVE to anchor keyframe (not absolute)
  int kf_index;            // Anchor keyframe index
};

using Matrix6 = Eigen::Matrix<double, 6, 6>;

namespace turtlmap {
// Structure for output state to be written by output thread
struct OutputState
{
  double timestamp;
  gtsam::Pose3 pose_ned;
  gtsam::Vector3 velocity;
  double imu_ax, imu_ay, imu_az;
  double imu_wx, imu_wy, imu_wz;
  double dvl_vx, dvl_vy, dvl_vz;
  double baro_depth;
  bool has_covariance = false;
  double cov2_a = 0.0, cov2_b = 0.0, cov2_angle = 0.0;
  bool has_covariance3d = false;
  double cov3_a = 0.0, cov3_b = 0.0, cov3_c = 0.0;
  Eigen::Matrix3d cov3_R = Eigen::Matrix3d::Identity();
  bool is_keyframe;
};

class PoseGraphBackend
{
 private:
  std::unique_ptr<AUVPoseGraph> posegraph_;

  // Data provider (HDF5 streamer)
  std::shared_ptr<turtlmap::IDataProvider> data_provider_;

  // State tracking
  std::vector<double> kf_timestamps_;
  std::vector<double> traj_timestamps_;
  std::vector<gtsam::Pose3> traj_poses_;
  std::vector<gtsam::Pose3> kf_first_opt_poses_;

  std::int64_t frame_count_;

  void insertIMU(const turtlmap::ImuMeasurement& imu_msg);
  void insertDVL(const turtlmap::DvlMeasurement& dvl_msg);
  void insertBaro(const turtlmap::BarometerMeasurement& baro_msg);
  void insertStereo(const turtlmap::StereoMeasurement& stereo_msg);

  void processIMU(const turtlmap::ImuMeasurement& imu_msg, const bool unsafe=false);
  void processDVL(const turtlmap::DvlMeasurement& dvl_msg, const bool unsafe=false);
  void processBarometer(const turtlmap::BarometerMeasurement& baro_msg, const bool unsafe=false);
  void processStereo(const turtlmap::StereoMeasurement& stereo_msg, const bool unsafe=false);
  void initializeFromIMU(const turtlmap::ImuMeasurement& imu_msg);

  void kfLoop();
  void processingLoop();
  void outputLoop();

  // Helper methods for kfLoop to improve readability
  void initializeKeyframeState();
  void processImuQueue();
  void processDvlQueue();
  void processBarometerQueue();
  void processStereoQueue();
  void savePreOptimizationState();
  void savePostOptimizationState();
  void addKeyframeSymbolsToGraph(double relative_time);
  void addDvlBiasSymbol(double relative_time);
  void addVelocitySymbol(double relative_time);
  void addPoseSymbolWithPrediction();
  void addSensorFactorsToGraph();
  void optimizeGraph();
  void computeAndStoreMarginals();
  void updateStateAfterOptimization();
  void publishKeyframePose();

  bool shouldCreateKeyframe(const bool force = false);
  void predictInterFramePose();
  bool shouldMergeWithPreviousKeyframe(double new_kf_time);

  std::optional<size_t> getKFIdByTimestamp(double timestamp, double eps=1e-5);

  gtsam::Pose3 transformImuToCamera(const gtsam::Pose3& pose_base);
  gtsam::Pose3 transformLeftCameraToIMU(const gtsam::Pose3& pose_cam);
  // Publish current state with an explicit timestamp; falls back to latest_sensor_time_ if not provided.
  void publishState(const gtsam::Pose3& pose_base, bool is_keyframe, double timestamp = -1.0);

  double first_depth_ = 0.0;
  double prev_dvl_time_ = 0.0;
  double current_kf_time_ = 0.0;
  double prev_kf_time_ = 0.0;
  bool new_kf_flag_ = false;
  std::atomic<bool> kf_opt_in_progress_{false};  // True when KF is being created and factors are being added

  // Optimization state tracking for debugging race conditions
  // States: 0=idle, 1=adding_factors, 2=optimizing, 3=computing_marginals, 4=updating_state
  std::atomic<int> optimization_state_{0};
  std::atomic<int> current_optimizing_kf_{-1};  // Which keyframe index is currently being optimized
  std::atomic<bool> optimization_in_progress_{false};  // For synchronization fix
  std::condition_variable optimization_complete_cv_;  // For synchronization fix

  gtsam::Pose3 prev_dvl_local_pose_;
  gtsam::Pose3 T_w_wd_;
  double kf_gap_time_;

  std::mutex mtx_;
  std::condition_variable kf_cv_;
  std::condition_variable kf_opt_done_cv_;  // Signals when KF setup is complete and processing can resume

  // Threading for sensor processing
  std::mutex queue_mutex_;
  std::condition_variable processing_cv_;
  turtlmap::SensorDataQueue data_queue_;
  std::shared_ptr<PreintegratedImuState> prev_imu_integrator_state_;
  std::shared_ptr<BarometerIntegratorState> prev_baro_integrator_state_;
  std::shared_ptr<PreintegratedVelocityState> prev_dvl_integrator_state_;


  // Threading for output
  std::mutex output_mutex_;
  std::condition_variable output_cv_;
  std::queue<OutputState> output_queue_;
  std::mutex traj_mtx_;  // Mutex for trajectory queue (thread-safe pose queries)

  bool is_using_dvl_v2_factor = true;
  bool is_rot_initialized_ = false;
  bool first_stereo_processed_ = false;  // Track if first stereo after graph init has been processed

  gtsam::Rot3 imu_latest_rot_;
  gtsam::Rot3 pim_latest_rot_;
  gtsam::Rot3 dvl_prev_rot_;

  int imu_init_count_ = 0;
  std::vector<gtsam::Vector3> imu_init_acc_;
  std::vector<gtsam::Vector3> imu_init_rot_;
  std::vector<Eigen::Quaterniond> imu_init_quats_;  // Store quaternions for averaging

  gtsam::Pose3 latest_kf_pose_;
  gtsam::Pose3 latest_publish_pose_;
  gtsam::Vector3 first_dvl_vel_;

  double first_kf_time_;

  std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> pim_dvl_;
  gtsam::Vector3 latest_dvl_vel_;
  gtsam::Pose3 latest_dvl_pose_;
  int imu_count_ = 0;
  gtsam::NavState latest_imu_prop_state_;

  // Barometer interframe integration
  std::unique_ptr<BarometerIntegrator> baro_integrator_;
  double last_baro_time_ = 0.0;

  std::atomic<bool> running_{false};
  std::thread kf_thread_;
  std::thread processing_thread_;
  std::thread output_thread_;

  double prev_imu_time_ = 0.0;

  // Output file for trajectory
  std::string output_file_;

  // Live visualization support
  bool live_vis_enabled_ = true;                        // enable/disable live trajectory logging
  std::string live_traj_file_ = "trajectory_live.txt";  // path to live trajectory file (x y z per line)
  std::string live_sensor_file_ = "sensors_live.txt";   // path to live sensor data file
  std::string optimization_timing_file_ = "optimization_timing.txt";  // path to optimization timing log

  // Latest sensor data for live streaming
  double latest_imu_ax_ = 0.0, latest_imu_ay_ = 0.0, latest_imu_az_ = 0.0;
  double latest_imu_wx_ = 0.0, latest_imu_wy_ = 0.0, latest_imu_wz_ = 0.0;
  double latest_dvl_vx_ = 0.0, latest_dvl_vy_ = 0.0, latest_dvl_vz_ = 0.0;
  double latest_baro_depth_ = 0.0;
  double latest_sensor_time_ = 0.0;
  double latest_stereo_time_ = 0.0;

  std::function<void(double, const gtsam::Pose3&)> pose_callback_;
  std::deque<TrajPoseQueueEntry> traj_pose_queue_;
  // Bounds how far back getPoseAtTime/getCovAtTime can answer. Queries older than
  // the retained window fail permanently, so this also bounds the age of a loop
  // closure the tracker is able to validate.
  size_t traj_pose_queue_max_size_ = 10000;

  // computing covariance
  gtsam::Matrix latest_kf_pose_cov_;
  std::vector<gtsam::Matrix> kf_pose_covariances_world_;
  std::vector<gtsam::Matrix> kf_first_opt_covariances_world_;  // Marginals from first optimization only
  gtsam::Matrix J_world_to_ned_;

  bool stereo_measurement_ready_ = false;
  double stereo_measurement_time_ = 0.0;

  // Debug tracking
  int imu_integration_count_ = 0;
  double last_imu_log_time_ = 0.0;

  Logger logger_;

  BluerovBarometerFactor last_baro_factor_;
  gtsam::CombinedImuFactor last_imu_factor_;

  // Dense trajectory storage: relative poses from anchor keyframe
  std::vector<DenseTrajectoryEntry> dense_trajectory_;
  gtsam::Pose3 dense_traj_relative_pose_;  // Accumulated relative pose from current KF
  int current_dense_kf_index_ = 0;         // Current anchor keyframe index

 public:
  PoseGraphBackend(const std::string& config_file, std::shared_ptr<turtlmap::IDataProvider> data_provider);
  ~PoseGraphBackend();
  void run();
  void stop();
  bool isRunning() const { return running_; }
  bool saveTrajectory(const std::string& filename);
  bool saveDenseTrajectory(const std::string& filename);
  void waitForCompletion();

  std::optional<Eigen::Matrix4d> getPoseAtTime(double timestamp);
  std::optional<Eigen::Matrix4d> getPoseAtKeyframe(int index);
  std::optional<Eigen::Matrix4d> getPoseAtKeyframePublished(int index);
  std::optional<Eigen::Matrix4d> getLatestPose();
  std::optional<Eigen::Matrix4d> getFirstOptimizedKeyframePose(double kf_timestamp, double eps=1e-2);
  std::optional<Matrix6> getCovAtTime(double timestamp, bool use_first_opt = false);
  std::optional<double> getFirstPoseTime();
  bool hasPoseForTime(double t);
  void setTrajPoseQueueMaxSize(size_t n);
  std::vector<std::pair<int, Eigen::Matrix4d>> getUpdatedGraph();

  // Graph initialization status query (for Python frontend to know when to start stereo processing)
  bool isGraphInitialized() const { return is_rot_initialized_; }

  int getProcessedKeyframeCount() const { return static_cast<int>(traj_poses_.size()); }
  double getLatestTimestamp() const { return traj_timestamps_.empty() ? 0.0 : traj_timestamps_.back(); }
  double getLatestKeyFrameTime() const { return kf_timestamps_.empty() ? 0 : kf_timestamps_.back(); }

  // visualization related functions
  void configureLiveVisualization(bool enabled, const std::string& filepath)
  {
    live_vis_enabled_ = enabled;
    if (!filepath.empty()) {
      live_traj_file_ = filepath;
      // Derive companion files in the same directory: <base>_optimization_timing.txt, <base>_sensors.txt
      size_t last_dot = filepath.find_last_of('.');
      std::string base = (last_dot != std::string::npos) ? filepath.substr(0, last_dot) : filepath;
      optimization_timing_file_ = base + "_optimization_timing.txt";
      live_sensor_file_ = base + "_sensors.txt";
    }
    if (live_vis_enabled_) {
      std::ofstream ofs(live_traj_file_, std::ios::trunc);
    }
  }

  void setPoseCallback(std::function<void(double, const gtsam::Pose3&)> cb) { pose_callback_ = std::move(cb); }
  void triggerStereoKeyframe(double timestamp);

  void restoreBaroIntegrator(std::shared_ptr<BarometerIntegratorState> prev_state);
};

}  // namespace turtlmap
