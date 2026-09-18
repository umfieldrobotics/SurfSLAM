#include "Posegraph.h"

#include <algorithm>  // for std::clamp, std::max, std::min

namespace turtlmap {
std::shared_ptr<PoseGraphParameters> readConfigFile(const std::string& config_file)
{
  YAML::Node config = YAML::LoadFile(config_file);

  // Check if the config file is empty.
  if (config.IsNull()) {
    std::cout << "Config file is empty!" << std::endl;
    return nullptr;
  }

  auto params = std::make_shared<PoseGraphParameters>();

  ImuParameters imu_params;
  imu_params.acc_max = config["imu_params"]["acc_max"].as<double>();
  imu_params.gyro_max = config["imu_params"]["gyro_max"].as<double>();
  imu_params.gyro_noise_density = config["imu_params"]["gyro_noise_density"].as<double>();
  imu_params.acc_noise_density = config["imu_params"]["acc_noise_density"].as<double>();
  imu_params.acc_random_walk = config["imu_params"]["acc_random_walk"].as<double>();
  imu_params.gyro_random_walk = config["imu_params"]["gyro_random_walk"].as<double>();
  // imu_params.acc_bias_prior = config["imu_params"]["acc_bias_prior"].as<double>();
  // imu_params.gyro_bias_prior = config["imu_params"]["gyro_bias_prior"].as<double>();
  imu_params.integration_covariance = config["imu_params"]["integration_covariance"].as<double>();
  imu_params.imu_rate = config["imu_params"]["imu_rate"].as<double>();
  imu_params.g = config["imu_params"]["g"].as<double>();
  imu_params.dvl_bias_prior = config["dvl_params"]["prior_bias"].as<double>();
  imu_params.gyro_random_walk = config["imu_params"]["gyro_random_walk"].as<double>();
  std::vector<double> acc_bias_prior_flat = config["imu_params"]["acc_bias_prior"].as<std::vector<double>>();
  for (int i = 0; i < 3; ++i) {
    imu_params.acc_bias_prior.push_back(acc_bias_prior_flat[i]);
  }
  std::vector<double> gyro_bias_prior_flat = config["imu_params"]["gyro_bias_prior"].as<std::vector<double>>();
  for (int i = 0; i < 3; ++i) {
    imu_params.gyro_bias_prior.push_back(gyro_bias_prior_flat[i]);
  }

  imu_params.dt_imu = 1 / imu_params.imu_rate;

  params->imu_params_ = imu_params;

  Extrinsics extrinsics;
  std::vector<double> T_SD_flat = config["dvl_params"]["T_SD"].as<std::vector<double>>();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      extrinsics.T_SD(i, j) = T_SD_flat[i * 4 + j];
    }
  }
  std::vector<double> T_SSo_flat = config["sonar_params"]["T_SSo"].as<std::vector<double>>();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      extrinsics.T_SSo(i, j) = T_SSo_flat[i * 4 + j];
    }
  }

  std::vector<double> T_BS_flat = config["imu_params"]["T_BS"].as<std::vector<double>>();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      extrinsics.T_BS(i, j) = T_BS_flat[i * 4 + j];
    }
  }

  std::vector<double> T_SLc_flat = config["left_cam_params"]["T_SC"].as<std::vector<double>>();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      extrinsics.T_SLc(i, j) = T_SLc_flat[i * 4 + j];
    }
  }
  std::vector<double> T_SRc_flat = config["right_cam_params"]["T_SC"].as<std::vector<double>>();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      extrinsics.T_SRc(i, j) = T_SRc_flat[i * 4 + j];
    }
  }
  std::vector<double> T_SBa_flat = config["barometer_params"]["T_SBa"].as<std::vector<double>>();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      extrinsics.T_SBa(i, j) = T_SBa_flat[i * 4 + j];
    }
  }
  std::vector<double> T_W_WD_flat = config["dvl_params"]["T_W_WD"].as<std::vector<double>>();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      extrinsics.T_W_WD(i, j) = T_W_WD_flat[i * 4 + j];
    }
  }

  params->extrinsics_ = extrinsics;

  BarometerParameters baro_params;
  baro_params.atm_press = config["barometer_params"]["atm_press"].as<double>();
  baro_params.enable_interframe_integration = config["barometer_params"]["enable_interframe_integration"].as<bool>();
  baro_params.noise_sigma = config["barometer_params"]["noise_sigma"].as<double>();
  baro_params.min_correction_threshold = config["barometer_params"]["min_correction_threshold"].as<double>();
  baro_params.enable_velocity_correction = config["barometer_params"]["enable_velocity_correction"].as<bool>();
  params->baro_params_ = baro_params;

  // BACKWARDS COMPATIBILITY: Also set the old field name
  params->baro_atm_pressure_ = baro_params.atm_press;

  SensorList sensor_list;
  auto isDvlUsed = config["sensor_list"]["isDvlUsed"];
  auto isImuUsed = config["sensor_list"]["isImuUsed"];
  auto isBaroUsed = config["sensor_list"]["isBaroUsed"];
  auto isSonarUsed = config["sensor_list"]["isSonarUsed"];
  auto areCamsUsed = config["sensor_list"]["areCamsUsed"];
  sensor_list.isDvlUsed = isDvlUsed.as<bool>();
  sensor_list.isImuUsed = isImuUsed.as<bool>();
  sensor_list.isBaroUsed = isBaroUsed.as<bool>();
  sensor_list.isSonarUsed = isSonarUsed.as<bool>();
  sensor_list.areCamsUsed = areCamsUsed.as<bool>();
  params->sensor_list_ = sensor_list;

  StereoParameters stereo_params;
  if (config["stereo_params"]) {
    auto stereo_node = config["stereo_params"];
    stereo_params.rotation_sigma = stereo_node["rotation_sigma"].as<double>();
    stereo_params.translation_sigma = stereo_node["translation_sigma"].as<double>();

    stereo_params.loop_translation_sigma = stereo_node["loop_translation_sigma"].as<double>();
    stereo_params.loop_rotation_sigma = stereo_node["loop_rotation_sigma"].as<double>();
    stereo_params.time_tolerance = stereo_node["time_tolerance"].as<double>();
  } else {
    stereo_params.translation_sigma = 0.1;
    stereo_params.rotation_sigma = 0.05;
    stereo_params.time_tolerance = 0.1;
  }
  params->stereo_params_ = stereo_params;

  SensorTopics sensor_topics;
  sensor_topics.imu_topic = config["sensor_topics"]["imu"].as<std::string>();
  sensor_topics.dvl_topic = config["sensor_topics"]["dvl"].as<std::string>();
  sensor_topics.baro_topic = config["sensor_topics"]["barometer"].as<std::string>();
  sensor_topics.sonar_topic = config["sensor_topics"]["sonar"].as<std::string>();
  sensor_topics.left_cam_topic = config["sensor_topics"]["left_cam"].as<std::string>();
  sensor_topics.right_cam_topic = config["sensor_topics"]["right_cam"].as<std::string>();
  sensor_topics.dvl_local_position_topic = config["sensor_topics"]["dvl_local_position"].as<std::string>();
  params->sensor_topics_ = sensor_topics;

  std::vector<std::string> rosbag_topics;
  rosbag_topics.push_back(sensor_topics.dvl_topic);
  rosbag_topics.push_back(sensor_topics.dvl_local_position_topic);
  rosbag_topics.push_back(sensor_topics.imu_topic);
  rosbag_topics.push_back(sensor_topics.baro_topic);
  rosbag_topics.push_back(sensor_topics.sonar_topic);
  rosbag_topics.push_back(sensor_topics.left_cam_topic);
  rosbag_topics.push_back(sensor_topics.right_cam_topic);
  params->rosbag_topics_ = rosbag_topics;

  auto useOrbSlam = config["useOrbSlam"];
  params->using_orbslam_ = useOrbSlam.as<bool>();

  // Read legacy using_smoother flag (optional, for backward compatibility)
  if (config["using_smoother"]) {
    auto using_smoother = config["using_smoother"];
    params->using_smoother_ = using_smoother.as<bool>();
  } else {
    params->using_smoother_ = false;
  }

  // Read optimization_mode parameter (takes precedence over legacy flags)
  if (config["optimization_mode"]) {
    params->optimization_mode_ = config["optimization_mode"].as<std::string>();

    // Validate the optimization mode
    if (params->optimization_mode_ != "isam2" && params->optimization_mode_ != "batch" &&
        params->optimization_mode_ != "fixed_lag") {
      throw std::runtime_error("Invalid optimization_mode: '" + params->optimization_mode_ +
                               "'. "
                               "Valid options are: 'isam2', 'batch', 'fixed_lag'");
    }

    std::cout << "✓ Optimization mode set to: " << params->optimization_mode_ << std::endl;
  } else {
    // Fallback to legacy flags for backward compatibility
    std::cout << "WARNING: optimization_mode not specified in config file." << std::endl;
    if (params->using_smoother_) {
      params->optimization_mode_ = "fixed_lag";
    } else {
      params->optimization_mode_ = "batch";
    }
    std::cout << "WARNING: Using legacy flags. Setting to: " << params->optimization_mode_ << std::endl;
  }

  OptimizationParameters optimization_params;
  optimization_params.lambdaUpperBound = config["optimization_params"]["lambdaUpperBound"].as<double>();
  optimization_params.lambdaLowerBound = config["optimization_params"]["lambdaLowerBound"].as<double>();
  optimization_params.initialLambda = config["optimization_params"]["initialLambda"].as<double>();
  optimization_params.maxIterations = config["optimization_params"]["maxIterations"].as<int>();
  optimization_params.relativeErrorTol = config["optimization_params"]["relativeErrorTol"].as<double>();
  optimization_params.absoluteErrorTol = config["optimization_params"]["absoluteErrorTol"].as<double>();
  params->optimization_params_ = optimization_params;

  ImuPreintegrationParams imu_preintegration_params;
  imu_preintegration_params.gap_time = config["imu_preintegration_params"]["gap_time"].as<double>();
  params->imu_preintegration_params_ = imu_preintegration_params;

  int num_iters = config["num_iters"].as<int>();
  params->num_iters_ = num_iters;
  params->dvl_fom_threshold_ = config["dvl_params"]["fom_threshold"].as<double>();
  params->kf_gap_time_ = config["kf_gap_time"].as<double>();
  params->using_pseudo_dvl_ = config["using_pseudo_dvl"].as<bool>();
  params->min_kf_delta_time_ = config["min_kf_delta_time"].as<double>();
  params->use_cam_for_kf_ = config["use_cam_for_kf"].as<bool>();
  params->external_kf_only_ = config["external_kf_only"].as<bool>();

  // Read compute_marginals flag with default value of false
  if (config["compute_marginals"]) {
    params->compute_marginals_ = config["compute_marginals"].as<bool>();
  } else {
    params->compute_marginals_ = false;
    std::cout << "compute_marginals not found in config, defaulting to false" << std::endl;
  }

  return params;
}

AUVPoseGraph::AUVPoseGraph() {}
AUVPoseGraph::AUVPoseGraph(std::string& config_file)
{
  // addition of the smoother
  smootherParameters.relinearizeThreshold = 0.01;
  smootherParameters.relinearizeSkip = 1;
  smootherISAM2 = gtsam::BatchFixedLagSmoother(smootherLag);

  graph = std::make_shared<gtsam::NonlinearFactorGraph>();
  initial_ = std::make_shared<gtsam::Values>();
  result_ = std::make_shared<gtsam::Values>();
  index_ = 0;

  // Initialize iSAM2
  isam2_params_.relinearizeThreshold = 0.01;
  isam2_params_.relinearizeSkip = 1;
  isam2_params_.enableRelinearization = true;
  isam2_params_.enablePartialRelinearizationCheck = true;
  isam2_ = new gtsam::ISAM2(isam2_params_);
  isam2_initialized_ = false;

  // Call imu_params() to initialize imu_params_.
  params_ = readConfigFile(config_file);
  setImuParams();
  pvm_ = std::make_shared<PreintegratedVelocityMeasurementsDvlOnly>();

  setVisualGapTime(params_->imu_preintegration_params_.gap_time);
  defineTransforms();

  // PreintegratedVelocityMeasurements(const dvlBias &vBias = dvlBias()) : biasDvl_(vBias) { preintMeasCov_.setZero(); }

  // turtlmap::dvlBias biasDvl_;
  // preintegrated_velocity_measurements_ = new turtlmap::PreintegratedVelocityMeasurements(biasDvl_);
  // preintegrated_velocity_measurements_ = new turtlmap::PreintegratedVelocityMeasurements();
  // preintegrated_velocity_measurements_->resetIntegration();
}

AUVPoseGraph::~AUVPoseGraph() {}

void AUVPoseGraph::setImuParams()
{
  pim_params_ = boost::make_shared<gtsam::PreintegratedCombinedMeasurements::Params>(
      gtsam::Vector3(0, 0, params_->imu_params_.g));
  // Make sharedD
  // pim_params_->MakeSharedU(9.81); // TODO (@Onur): is this correct?
  pim_params_->setAccelerometerCovariance(gtsam::I_3x3 * std::pow(params_->imu_params_.acc_noise_density, 2));
  pim_params_->setGyroscopeCovariance(gtsam::I_3x3 * std::pow(params_->imu_params_.gyro_noise_density, 2));
  pim_params_->setIntegrationCovariance(gtsam::I_3x3 * 1e-8);
  pim_params_->setBiasAccCovariance(gtsam::I_3x3 * std::pow(params_->imu_params_.acc_random_walk, 2));
  pim_params_->setBiasOmegaCovariance(gtsam::I_3x3 * std::pow(params_->imu_params_.gyro_random_walk, 2));

  // double prior_accbias = params_->imu_params_.acc_bias_prior;
  // double prior_gyrobias = params_->imu_params_.gyro_bias_prior;
  // double prior_dvl_bias = params_->imu_params_.dvl_bias_prior;
  std::vector<double> prior_accbias = params_->imu_params_.acc_bias_prior;
  std::vector<double> prior_gyrobias = params_->imu_params_.gyro_bias_prior;
  double prior_dvl_bias = params_->imu_params_.dvl_bias_prior;

  // Convert the prior biases to gtsam::Vector3
  gtsam::Vector3 prior_accbias_vec =
      (gtsam::Vector3() << prior_accbias[0], prior_accbias[1], prior_accbias[2]).finished();
  gtsam::Vector3 prior_gyrobias_vec =
      (gtsam::Vector3() << prior_gyrobias[0], prior_gyrobias[1], prior_gyrobias[2]).finished();
  gtsam::Vector3 prior_dvlbias_vec = (Vector(3) << prior_dvl_bias, prior_dvl_bias, prior_dvl_bias).finished();

  // Vector of zeros
  gtsam::Vector3 prior_zeros_vec = (Vector(3) << 0, 0, 0).finished();

  priorImuBias_ = gtsam::imuBias::ConstantBias(prior_accbias_vec, prior_gyrobias_vec);
  priorDvlBias_ = gtsam::imuBias::ConstantBias(prior_dvlbias_vec, prior_zeros_vec);

  // Convert std::shared_ptr to boost::shared_ptr for GTSAM compatibility
  boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> boost_pim_params(
      pim_params_.get(), [pim_params_copy = pim_params_](auto*) {});
  pim_ = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(boost_pim_params, priorImuBias_);
}

BluerovBarometerFactor AUVPoseGraph::addBarometricFactor(double W_measurement_z, double measurement_noise, int baro_id)
{
  // gtsam::SharedNoiseModel barometer_model = gtsam::noiseModel::Isotropic::Sigma(1, measurement_noise);
  const auto new_factor = BluerovBarometerFactor(gtsam::Symbol('x', baro_id), W_measurement_z,
                                                 gtsam::noiseModel::Isotropic::Sigma(1, measurement_noise));
  graph->add(new_factor);
  return new_factor;
  // barometer_factor.print("Barometer factor: ");
}

gtsam::CombinedImuFactor AUVPoseGraph::addImuFactor()
{
  gtsam::CombinedImuFactor imu_factor(gtsam::Symbol('x', index_ - 1), gtsam::Symbol('v', index_ - 1),
                                      gtsam::Symbol('x', index_), gtsam::Symbol('v', index_),
                                      gtsam::Symbol('b', index_ - 1), gtsam::Symbol('b', index_), *pim_);
  graph->add(imu_factor);
  return imu_factor;
}

std::optional<gtsam::BetweenFactor<gtsam::Pose3>> AUVPoseGraph::addDvlFactorImuRot()
{
  int dvl_size = current_dvl_timestamps_.size();
  assert(current_dvl_vels_.size() == imu_rot_list_.size());  // ensure the size is the same
  double dt_dvl;

  // almost the same strategy as addDvlFactorV2 but using the imu rotation instead of DVL odom rotation
  gtsam::Rot3 integrated_pose_rotation = imu_prev_rot_.inverse() * imu_rot_list_[dvl_size - 1];
  gtsam::Point3 integrated_pose_translation = gtsam::Point3(0.0, 0.0, 0.0);

  // TODO: replace the integration with IMU rot
  gtsam::Matrix3 integrated_rot_matrix = gtsam::Matrix3::Identity();

  for (int i = 0; i < dvl_size; i++) {
    if (i == 0) {
      dt_dvl = current_dvl_timestamps_[i] - prev_kf_time_;  // TODO: check with Onur
    } else {
      dt_dvl = current_dvl_timestamps_[i] - current_dvl_timestamps_[i - 1];
    }

    // do pvm integration
    integrated_rot_matrix = imu_prev_rot_.matrix().inverse() * imu_rot_list_[i].matrix();
    pvm_->integrateMeasurementsNoise(current_dvl_vels_[i], gtsam::Rot3(integrated_rot_matrix), dt_dvl,
                                     current_dvl_foms_[i]);
    integrated_pose_translation += integrated_rot_matrix.matrix() * current_dvl_vels_[i] * dt_dvl;
  }

  gtsam::Pose3 integrated_pose(integrated_pose_rotation,
                               integrated_pose_translation);  // TODO Jingyu: use PVM integration results instead

  std::optional<gtsam::BetweenFactor<gtsam::Pose3>> integrated_pose_factor;
  if (index_ > 0) {
    if (!initial_->exists(gtsam::Symbol('x', index_))) {
      initial_->insert(gtsam::Symbol('x', index_), prev_pose_ * integrated_pose);
    }
    gtsam::Vector6 rot_noise_vec;
    rot_noise_vec << 0.01, 0.01, 0.01, 0.1, 0.1, 0.1;  // TODO: Jingyu: find the best parameters to reduce drift
    gtsam::SharedNoiseModel integrated_pose_noise_model = gtsam::noiseModel::Diagonal::Sigmas(rot_noise_vec);
    if (index_ < 10000)  // for testing IMU without the DVL factor
    {
      DvlOnlyFactor dvl_only_factor(gtsam::Symbol('x', index_ - 1), gtsam::Symbol('x', index_),
                                    gtsam::Symbol('d', index_ - 1), *pvm_);  // use the bias with current index
      graph->add(dvl_only_factor);

      integrated_pose_factor = std::make_optional<gtsam::BetweenFactor<gtsam::Pose3>>(
          gtsam::Symbol('x', index_ - 1), gtsam::Symbol('x', index_), integrated_pose, integrated_pose_noise_model);
      graph->add(*integrated_pose_factor);
    } else {
      std::cout << "no dvl factor added" << std::endl;
    }

    imuBias::ConstantBias zero_bias(Vector3(0, 0, 0), Vector3(0, 0, 0));
    graph->add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(gtsam::Symbol('d', index_ - 1),
                                                                  gtsam::Symbol('d', index_), zero_bias,
                                                                  gtsam::noiseModel::Isotropic::Sigma(6, 1e-1)));
  }

  current_dvl_timestamps_.clear();
  current_dvl_vels_.clear();
  current_dvl_poses_.clear();
  current_dvl_foms_.clear();
  current_dvl_local_timestamps_.clear();  // TODO: Jingyu - not needing these dvl local
  imu_prev_rot_ = imu_rot_list_[dvl_size - 1];
  imu_rot_list_.clear();

  // Jingyu reset integration
  pvm_->resetIntegration();
  return integrated_pose_factor;
}

std::optional<gtsam::BetweenFactor<gtsam::Pose3>> AUVPoseGraph::addVelocityIntegrationFactor()
{
  // Get the last keyframe's pose and velocity
  // gtsam::Pose3 prev_pose = initial_->at<gtsam::Pose3>(gtsam::Symbol('x', index_ - 1));
  // gtsam::Vector3 prev_vel = initial_->at<gtsam::Vector3>(gtsam::Symbol('v', index_ - 1));
  // current timestamp

  // Get the size of the current dvl measurements
  int dvl_size = current_dvl_timestamps_.size();
  int dvl_local_size = current_dvl_local_timestamps_.size();

  // if any of the size is zero assert an error
  if (dvl_size == 0) {
    // std::cout << "DVL size is zero!" << std::endl;
    assert(false);
  }

  // init a translation with zero for dvl velocity integration
  // gtsam::Pose3 integrated_pose = gtsam::Pose3::identity();
  gtsam::Point3 integrated_pose_translation = gtsam::Point3(0.0, 0.0, 0.0);
  // go through the dvl velocity measurements current_dvl_vels_
  for (int i = 0; i < dvl_size; i++) {
    // integrate over time to get the pose
    double dt;
    if (i == 0) {
      dt = current_dvl_timestamps_[i] - prev_kf_time_;  // TODO: check with Onur
    } else {
      dt = current_dvl_timestamps_[i] - current_dvl_timestamps_[i - 1];
    }

    gtsam::Rot3 current_rot = findCurrentPoseForDvlVel(current_dvl_timestamps_[i]);
    gtsam::Vector3 current_vel = current_dvl_vels_[i];

    gtsam::Vector3 current_vel_world = current_rot.matrix() * current_vel;  // TODO: triple check the transform!!!

    integrated_pose_translation += current_vel_world * dt;
  }
  std::cout << "integrated_pose_translation: " << integrated_pose_translation << std::endl;

  gtsam::Rot3 integrated_pose_rotation = gtsam::Rot3();
  for (int i = 0; i < dvl_local_size; i++) {
    // std::cout << "current_dvl_poses_[i]: " << current_dvl_poses_[i] << std::endl;
    integrated_pose_rotation = integrated_pose_rotation * current_dvl_poses_[i].rotation();
  }

  gtsam::Pose3 integrated_pose(integrated_pose_rotation, integrated_pose_translation);

  std::cout << "current idex: " << index_ << std::endl;
  std::optional<gtsam::BetweenFactor<gtsam::Pose3>> integrated_pose_factor;

  if (index_ > 0) {
    if (!initial_->exists(gtsam::Symbol('x', index_))) {
      initial_->insert(gtsam::Symbol('x', index_), prev_pose_ * integrated_pose);
    }
    gtsam::SharedNoiseModel integrated_pose_noise_model =
        gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(0.01));
    integrated_pose_factor = std::make_optional<gtsam::BetweenFactor<gtsam::Pose3>>(
        gtsam::Symbol('x', index_ - 1), gtsam::Symbol('x', index_), integrated_pose, integrated_pose_noise_model);
    graph->add(*integrated_pose_factor);
  }
  current_dvl_timestamps_.clear();
  current_dvl_vels_.clear();
  current_dvl_poses_.clear();
  current_dvl_local_timestamps_.clear();
  return integrated_pose_factor;
}

gtsam::Rot3 AUVPoseGraph::findCurrentPoseForDvlVel(double dvlTimeStamp)
{
  // diff to previous time and current time
  gtsam::Pose3 currKfPose = prev_pose_;
  double diffCurr = dvlTimeStamp - prev_kf_time_;
  // find min diff and idx in dvl_local_timestamps_
  double minDiff = 1000000;
  int minIdx = -1;
  for (int i = 0; i < current_dvl_local_timestamps_.size(); i++) {
    double diff = abs(dvlTimeStamp - current_dvl_local_timestamps_[i]);
    if (diff < minDiff) {
      minDiff = diff;
      minIdx = i;
    }
  }

  // std::cout << "size of current_dvl_poses_: " << current_dvl_poses_.size() << std::endl;
  // std::cout << "minIdx: " << minIdx << std::endl;

  gtsam::Rot3 accumulatedRotation = gtsam::Rot3();
  if (diffCurr < minDiff) {
    // return current pose
    // std::cout << "current pose " << currKfPose << std::endl;
    return accumulatedRotation;
  } else {
    for (int i = 0; i < minIdx; i++) {
      accumulatedRotation * current_dvl_poses_[i].rotation();
    }
    // return the pose in current_dvl_poses_[minIdx]
    // std::cout << "current_dvl_poses_[minIdx] " << current_dvl_poses_[minIdx] << std::endl;
    // return current_dvl_poses_[minIdx];
    return accumulatedRotation;
  }
}

gtsam::BetweenFactor<gtsam::Pose3> AUVPoseGraph::addVisualConstraintFactor(gtsam::Pose3 between_pose, double weight,
                                                                           int prev_idx, int curr_idx)
{
  gtsam::Matrix66 informationMatrix = gtsam::Matrix66::Identity() * weight * 1e-6;
  gtsam::noiseModel::Gaussian::shared_ptr noise =
      gtsam::noiseModel::Gaussian::Information(informationMatrix);  // * 1e-3);
  gtsam::BetweenFactor<gtsam::Pose3> visual_factor(gtsam::Symbol('x', prev_idx), gtsam::Symbol('x', curr_idx),
                                                   between_pose, noise);
  graph->add(visual_factor);
  std::cout << "(Visual) Adding symbols: x" << prev_idx << ", x" << curr_idx << std::endl;
  return visual_factor;
}

gtsam::PriorFactor<gtsam::Pose3> AUVPoseGraph::addPriorFactor(gtsam::Pose3 initial_pose, gtsam::Vector initial_vel,
                                                              double pose_noise)
{
  // Differentiate rotation vs translation priors based on barometer availability
  // Roll/Pitch must be well-constrained to define gravity direction (critical for observability)
  // Yaw, X, Y have weaker priors
  // Z prior depends on barometer availability

  double roll_pitch_noise = 0.001;  // Very tight - gravity direction must be known
  double yaw_noise = 0.01;          // Looser - yaw is less constrained by IMU
  double xy_noise = 0.1;            // Loose - no absolute XY reference
  double z_noise;

  if (params_->sensor_list_.isBaroUsed) {
    // Barometer provides absolute Z - use tight prior
    z_noise = 0.01;  // 1cm uncertainty
  } else {
    // No barometer - use very loose Z prior to avoid over-constraining
    // This prevents the optimizer from forcing incorrect Z values
    z_noise = 10.0;  // 10m uncertainty - essentially unconstrained
  }

  gtsam::noiseModel::Diagonal::shared_ptr prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << roll_pitch_noise, roll_pitch_noise, yaw_noise, xy_noise, xy_noise, z_noise).finished());
  const auto new_factor = gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', 0), initial_pose, prior_noise);
  graph->add(new_factor);

  gtsam::noiseModel::Diagonal::shared_ptr prior_vel_noise =
      gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << pose_noise, pose_noise, pose_noise).finished());
  // graph->add(gtsam::PriorFactor<gtsam::Vector3>(gtsam::Symbol('v', 0), initial_vel, prior_vel_noise));

  std::cout << "Pose graph initial prior (roll/pitch/yaw | x/y/z): " << roll_pitch_noise << "/" << roll_pitch_noise
            << "/" << yaw_noise << " | " << xy_noise << "/" << xy_noise << "/" << z_noise << std::endl;
  return new_factor;
}

void AUVPoseGraph::addInitialEstimate(gtsam::Pose3 initial_pose, gtsam::Vector3 initial_vel)
{
  initial_->insert(gtsam::Symbol('x', index_), initial_pose);

  if (params_->sensor_list_.isImuUsed) {
    // remove to dvl callback when first measurement is received
    // initial_->insert(gtsam::Symbol('v', index_), initial_vel);

    initial_->insert(gtsam::Symbol('b', index_), priorImuBias_);
  }
  if (params_->sensor_list_.isDvlUsed) {
    initial_->insert(gtsam::Symbol('d', index_), priorDvlBias_);
  }
  std::cout << "(Initial Estimate) Adding symbols: x, v, b, d " << index_ << std::endl;
}

void AUVPoseGraph::defineTransforms()
{
  T_SD_ = params_->extrinsics_.T_SD;

  T_SB_ = params_->extrinsics_.T_BS;

  T_W_WD_ = params_->extrinsics_.T_W_WD;
}

void AUVPoseGraph::addEstimateSimple(double dt, double noise)
{
  // generate a random vector of dimension 3
  gtsam::Vector3 noisy =
      noise * gtsam::Vector3(normal_distribution_(rng_), normal_distribution_(rng_), normal_distribution_(rng_));

  if (params_->sensor_list_.isImuUsed) {
    gtsam::Pose3 pose_i = initial_->at<gtsam::Pose3>(gtsam::Symbol('x', index_ - 1));
    gtsam::Vector3 vel_i = initial_->at<gtsam::Vector3>(gtsam::Symbol('v', index_ - 1));
    gtsam::imuBias::ConstantBias bias_i = initial_->at<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', index_ - 1));
    // turtlmap::dvlBias bias_i_dvl = initial_->at<turtlmap::dvlBias>(gtsam::Symbol('d', index_ -
    // 1));

    // Compute translation update
    gtsam::Vector3 W_dp_S = pose_i.rotation().matrix() * (vel_i * dt + 0.5 * std::pow(dt, 2) * noisy);
    gtsam::Vector3 W_p_S = pose_i.translation() + W_dp_S;

    // Compute Rotation update
    gtsam::Rot3 R_W_S;
    // Perturb the rotation with a small angle about a random axis
    gtsam::Vector3 axis =
        gtsam::Vector3(normal_distribution_(rng_), normal_distribution_(rng_), normal_distribution_(rng_));
    axis.normalize();
    double angle = normal_distribution_(rng_);
    R_W_S = gtsam::Rot3::Rodrigues(axis * angle) * pose_i.rotation();
    // Pose
    gtsam::Pose3 pred_pose(R_W_S, W_p_S);

    // Compute velocity update
    gtsam::Vector3 B_v_S = vel_i + dt * noisy;
    gtsam::Vector3 pred_vel = B_v_S;

    // Compute velocity update
    gtsam::Vector3 priorBiasGyro =
        priorImuBias_.gyroscope() +
        gtsam::Vector3(normal_distribution_(rng_), normal_distribution_(rng_), normal_distribution_(rng_));
    gtsam::Vector3 priorBiasAcc =
        priorImuBias_.accelerometer() +
        gtsam::Vector3(normal_distribution_(rng_), normal_distribution_(rng_), normal_distribution_(rng_));
    gtsam::imuBias::ConstantBias pred_bias = gtsam::imuBias::ConstantBias(priorBiasAcc, priorBiasGyro);
    initial_->insert(gtsam::Symbol('x', index_), pred_pose);
    initial_->insert(gtsam::Symbol('v', index_), pred_vel);
    initial_->insert(gtsam::Symbol('b', index_), pred_bias);
    initial_->insert(gtsam::Symbol('d', index_), priorDvlBias_);

    std::cout << "Simple Estimate Adding symbols: x, v, b, d " << index_ << std::endl;

    // gtsam::Vector3 priorDvlBias = bias_i_dvl.getBias() + gtsam::Vector3(normal_distribution_(rng_),
    // normal_distribution_(rng_), normal_distribution_(rng_));
  }
}

void AUVPoseGraph::optimizePoseGraph()
{
  // Create the optimizer
  gtsam::LevenbergMarquardtParams params;
  // params.setVerbosity("ERROR");
  // params.setVerbosityLM("ERROR");
  // params.setlambdaUpperBound(params_->optimization_params_.lambdaUpperBound);
  // params.setlambdaLowerBound(params_->optimization_params_.lambdaLowerBound);
  // params.setlambdaInitial(params_->optimization_params_.initialLambda);
  // params.setMaxIterations(params_->optimization_params_.maxIterations);
  // params.setRelativeErrorTol(params_->optimization_params_.relativeErrorTol);
  // params.setAbsoluteErrorTol(params_->optimization_params_.absoluteErrorTol);
  gtsam::LevenbergMarquardtOptimizer optimizer(*graph, *initial_, params);

  // print initials and graphs
  // initial_->print("Initial Estimate:\n");
  // graph->print("Initial Graph:\n");

  // Optimize
  *result_ = optimizer.optimize();

  // Print results
  // std::cout << "Initial Error = " << graph->error(*initial_) << std::endl;

  *initial_ = *result_;

  // std::cout << "Final Error = " << graph->error(*result_) << std::endl;
}

void AUVPoseGraph::optimizePoseGraphSmoother()
{
  // use smootherISAM2 to optimize the graph
  // graph->print("Initial Graph:\n");
  // initial_->print("Initial Estimate:\n");

  // print smootherTimestamps
  // std::cout << "smootherTimestamps: " << std::endl;
  // for (auto it = smootherTimestamps.begin(); it != smootherTimestamps.end(); ++it)
  // {
  //     std::cout << it->first << " " << it->second << std::endl;
  // }
  // initial_->print("Initial Estimate:\n");
  gtsam::FactorIndices delete_slots;
  smootherISAM2.update(*graph, *initial_, smootherTimestamps, delete_slots);

  // cout delete_slots
  // for (auto it = delete_slots.begin(); it != delete_slots.end(); ++it)
  // {
  //     std::cout << "delete_slots: " << *it << std::endl;
  // }

  // print everything inside the smootherISAM2
  // std::cout << "smootherISAM2: " << std::endl;
  // smootherISAM2.print("smootherISAM2: ");

  // for(const FixedLagSmoother::KeyTimestampMap::value_type& key_timestamp: smootherISAM2.timestamps()) {
  // std::cout << "    Key: " << key_timestamp.first << "  Time: " << key_timestamp.second << std::endl;
  // }

  // print initial
  // smootherISAM2.getFactors().print("smootherISAM2 factors: ");

  // directly taken from the example
  // for(size_t i = 1; i < 2; ++i) { // Optionally perform multiple iSAM2 iterations
  //   smootherISAM2.update();}

  // smootherISAM2.calculateEstimate<gtsam::Pose3>(index_).print("iSAM2 Estimate:");
  *result_ = smootherISAM2.calculateEstimate();

  // calculate error
  double error = smootherISAM2.getFactors().error(*result_);
  if (std::isnan(error)) {
    std::cout << "Error is NaN" << std::endl;
    assert(false);
  }
  // std::cout << "Final Error = " << smootherISAM2.getFactors().error(*result_) << std::endl;

  // result_->print("Final Estimate:\n");
  smootherTimestamps.clear();
  initial_->clear();
  graph->resize(0);
}

void AUVPoseGraph::optimizePoseGraphISAM2()
{
  // Update iSAM2 with new factors and initial values
  gtsam::ISAM2Result result = isam2_->update(*graph, *initial_);

  // Mark as initialized after first update
  if (!isam2_initialized_) {
    isam2_initialized_ = true;
  }

  // Get the optimized result
  *result_ = isam2_->calculateEstimate();

  // Calculate and check error
  double error = isam2_->getFactorsUnsafe().error(*result_);
  if (std::isnan(error)) {
    std::cout << "iSAM2 Error is NaN" << std::endl;
    assert(false);
  }

  // Clear for next iteration (iSAM2 maintains state internally)
  initial_->clear();
  graph->resize(0);
}

void AUVPoseGraph::addEstimateVisual(const std::vector<double> vertex)
{
  Eigen::Quaterniond quat(vertex[8], vertex[5], vertex[5], vertex[7]);
  Eigen::Vector3d trans(vertex[2], vertex[3], vertex[4]);
  Eigen::Vector3d v(vertex[9], vertex[10], vertex[11]);
  Eigen::Vector3d bacc(vertex[12], vertex[13], vertex[14]);
  Eigen::Vector3d bgyro(vertex[15], vertex[16], vertex[17]);

  // Convert quaternion ro Rot3
  gtsam::Rot3 R(quat.normalized().toRotationMatrix());
  // Convert translation to Point3
  gtsam::Point3 p(trans);
  // Create Pose3
  gtsam::Pose3 W_visualEstPose(R, p);
  gtsam::imuBias::ConstantBias visualEstBias(bacc, bgyro);
  gtsam::Point3 B_visualEstVel(v);
  initial_->insert(gtsam::Symbol('x', index_), W_visualEstPose);

  // update the prev_pose_ as the current pose
  prev_pose_ = W_visualEstPose;
  if (params_->sensor_list_.isImuUsed) {
    initial_->insert(gtsam::Symbol('v', index_), B_visualEstVel);
    initial_->insert(gtsam::Symbol('b', index_), visualEstBias);
  }
  if (params_->sensor_list_.isDvlUsed) {
    initial_->insert(gtsam::Symbol('d', index_), priorDvlBias_);
  }
  std::cout << "(Visual Estimate) Adding symbols: x, v, b, d " << index_ << std::endl;
}

void AUVPoseGraph::initializePoseGraph()
{
  gtsam::Pose3 priorPose = gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0, 0, 0));
  if (params_->using_orbslam_) {
    Eigen::Quaterniond q_prior(-0.0602337, 0.743171, -0.162309, -0.646316);
    Eigen::Matrix3d R_prior = q_prior.normalized().toRotationMatrix();
    priorPose = gtsam::Pose3(gtsam::Rot3(R_prior), gtsam::Point3(0.0023731, 0.0249478, 0.0354701));
  } else {
    // if not using orbslam then start from identity
    // priorPose = gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0, 0, 0));

    // Jingyu update: use mocap pose to initialize the pose graph
    // T_set_for_slam:  [[ 0.94334495 -0.32362494  0.07326098  0.        ]
    // [ 0.31379384  0.94186878  0.12006906  0.        ]
    // [-0.10785957 -0.09027771  0.99005872  0.        ]
    // [ 0.          0.          0.          1.        ]]
    priorPose = gtsam::Pose3(gtsam::Rot3(0.94334495, -0.32362494, 0.07326098, 0.31379384, 0.94186878, 0.12006906,
                                         -0.10785957, -0.09027771, 0.99005872),
                             gtsam::Point3(0, 0, 0));
  }
  addInitialEstimate(priorPose, gtsam::Vector3());

  // Jingyu add: update the initial pose as the previous pose
  prev_pose_ = priorPose;
  addPriorFactor(priorPose, gtsam::Vector3(), 0.001);
}

void AUVPoseGraph::initializePoseGraphFromImu(gtsam::Rot3 initial_rotation)
{
  gtsam::Pose3 priorPose = gtsam::Pose3(initial_rotation, gtsam::Point3(0, 0, 0));
  addInitialEstimate(priorPose, gtsam::Vector3());
  prev_pose_ = priorPose;
  addPriorFactor(priorPose, gtsam::Vector3(), 0.001);  // TODO: check the noise
}

void AUVPoseGraph::addDvlVelocity(double timeStamp, gtsam::Vector3 velocity)
{
  current_dvl_timestamps_.push_back(timeStamp);
  current_dvl_vels_.push_back(velocity);
}

void AUVPoseGraph::addDvlPose(double timeStamp, gtsam::Pose3 pose)
{
  current_dvl_local_timestamps_.push_back(timeStamp);
  current_dvl_poses_.push_back(pose);
}

void AUVPoseGraph::addDvlRotation(double timeStamp, gtsam::Rot3 rot)
{
  current_dvl_local_timestamps_.push_back(timeStamp);
  current_dvl_rotations_.push_back(rot);
}

// TODO: add functions to clean the dvl related vectors

void AUVPoseGraph::addDvlOdometryFactor(double noise)
{
  // add a between factor for the current pose and the previous pose
  gtsam::Pose3 between_pose = gtsam::Pose3();
  gtsam::SharedNoiseModel between_pose_noise_model =
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(noise));

  // go through all the current_dvl_poses_ and multiply them to get the between_pose
  for (int i = 0; i < current_dvl_poses_.size(); i++) {
    between_pose = between_pose * current_dvl_poses_[i];
  }

  // add the between factor
  if (index_ > 0) {
    gtsam::BetweenFactor<gtsam::Pose3> between_pose_factor(gtsam::Symbol('x', index_ - 1), gtsam::Symbol('x', index_),
                                                           between_pose, between_pose_noise_model);
    graph->add(between_pose_factor);
  }
}

std::optional<gtsam::BetweenFactor<gtsam::Pose3>> AUVPoseGraph::addStereoBetweenFactor(const gtsam::Pose3& between_pose,
                                                                                       double translation_sigma,
                                                                                       double rotation_sigma,
                                                                                       int prev_idx, int curr_idx)
{
  gtsam::Vector6 sigmas;
  sigmas << rotation_sigma, rotation_sigma, rotation_sigma, translation_sigma, translation_sigma, translation_sigma;
  gtsam::SharedNoiseModel gaussian_noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
  gtsam::noiseModel::Robust::shared_ptr between_pose_noise_model =
      gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(2.0), gaussian_noise);
  // if past 0 and using cameras
  std::optional<gtsam::BetweenFactor<gtsam::Pose3>> between_pose_factor;
  if (prev_idx > 0 && params_->sensor_list_.areCamsUsed) {
    std::cout << "Adding stereo factor" << std::endl;
    between_pose_factor = std::make_optional<gtsam::BetweenFactor<gtsam::Pose3>>(
        gtsam::Symbol('x', prev_idx), gtsam::Symbol('x', curr_idx), between_pose, between_pose_noise_model);
    graph->add(*between_pose_factor);
  }
  return between_pose_factor;
}

void AUVPoseGraph::restoreImuIntegrator(std::shared_ptr<PreintegratedImuState> imu_integrator_state)
{
  auto restored_imu_integrator = loadPreintegratedImuState(*imu_integrator_state);
  *pim_ = std::move(restored_imu_integrator);
}

void AUVPoseGraph::restoreDvlIntegrator(std::shared_ptr<PreintegratedVelocityState> dvl_integrator_state)
{
  pvm_->loadState(*dvl_integrator_state);
}

}  // namespace turtlmap