#include "BarometerIntegrator.h"

#include <cmath>
#include <iostream>

namespace turtlmap {

BarometerIntegrator::BarometerIntegrator(const Params& params, Logger& logger)
    : params_(params),
      logger_(logger),
      z_variance_(params.baro_noise_sigma * params.baro_noise_sigma)  // Initialize variance with measurement noise
      ,
      last_innovation_(0.0),
      last_update_time_(0.0)
{
  // Log the process noise parameters
  double process_noise_rate = params_.computeProcessNoiseRate();
  logger_.debug() << "  - IMU process noise rate (from IMU params): " << process_noise_rate << " m/s²/√Hz" << std::endl;
  logger_.debug() << "  - Initial z-uncertainty: " << std::sqrt(z_variance_) << " m" << std::endl;
}

void BarometerIntegrator::predictVariance(double dt)
{
  // Proper power-law integration of IMU errors
  // Reference: "Aided Navigation: GPS with High Rate Sensors" by Jay A. Farrell

  // White noise contribution: Variance grows as t³/3
  double noise_variance = std::pow(params_.acc_noise_density * dt, 2) * dt / 3.0;

  // Bias random walk contribution: Variance grows as t⁵/20
  // (This is the dominant term for typical keyframe intervals)
  double bias_variance = std::pow(params_.acc_random_walk * dt, 2) * std::pow(dt, 3) / 20.0;

  // Process noise: Q = noise_variance + bias_variance
  double process_noise = noise_variance + bias_variance;

  // Prediction: P_{t|t-1} = P_{t-1|t-1} + Q
  z_variance_ += process_noise;
}

void BarometerIntegrator::updateVariance(double kalman_gain, double measurement_variance)
{
  // Joseph form (numerically stable) Kalman covariance update
  // For scalar case: P_{t|t} = (1 - K)² P_{t|t-1} + K² R
  // where H = 1 (direct measurement of position)

  double predicted_variance = z_variance_;
  double innovation_factor = 1.0 - kalman_gain;

  // Joseph form for scalar case
  z_variance_ =
      innovation_factor * innovation_factor * predicted_variance + kalman_gain * kalman_gain * measurement_variance;

  // Ensure variance doesn't go below measurement noise (numerical stability)
  z_variance_ = std::max(z_variance_, measurement_variance);
}

Eigen::VectorXd BarometerIntegrator::computeKalmanGain(double dt) const
{
  // Predict variance growth using proper power-law integration
  double noise_variance = std::pow(params_.acc_noise_density * dt, 2) * dt / 3.0;
  double bias_variance = std::pow(params_.acc_random_walk * dt, 2) * std::pow(dt, 3) / 20.0;
  double process_noise = noise_variance + bias_variance;

  double predicted_variance = z_variance_ + process_noise;

  // Measurement noise variance
  double measurement_variance = params_.baro_noise_sigma * params_.baro_noise_sigma;

  // Kalman gain: K = P_{t|t-1} H^T / (H P_{t|t-1} H^T + R)
  // For scalar case with H = 1: K = P_{t|t-1} / (P_{t|t-1} + R)
  double position_gain = predicted_variance / (predicted_variance + measurement_variance);

  if (params_.enable_velocity_correction) {
    // For velocity correction, use a scaled gain
    // Velocity innovation = (position_error) / dt
    // Apply a conservative velocity correction
    double velocity_gain = position_gain * 0.5;  // More conservative

    Eigen::VectorXd gains(2);
    gains << position_gain, velocity_gain;
    return gains;
  } else {
    Eigen::VectorXd gains(1);
    gains << position_gain;
    return gains;
  }
}

gtsam::NavState BarometerIntegrator::correctState(const gtsam::NavState& propagated_state, double barometer_depth,
                                                  double dt)
{
  // Compute innovation (measurement residual)
  double predicted_z = propagated_state.pose().z();
  last_innovation_ = barometer_depth - predicted_z;

  // Check if correction is needed
  if (std::abs(last_innovation_) < params_.min_correction_threshold) {
    // Error is small, no correction needed
    return propagated_state;
  }

  // Predict variance growth
  predictVariance(dt);

  // Compute Kalman gains
  Eigen::VectorXd gains = computeKalmanGain(dt);
  double position_gain = gains(0);

  // Correct z-position
  double corrected_z = predicted_z + position_gain * last_innovation_;

  // Build corrected pose
  gtsam::Point3 corrected_position(propagated_state.pose().x(), propagated_state.pose().y(), corrected_z);
  gtsam::Pose3 corrected_pose(propagated_state.pose().rotation(), corrected_position);

  // Correct z-velocity if enabled
  gtsam::Vector3 corrected_velocity = propagated_state.velocity();
  if (params_.enable_velocity_correction && gains.size() > 1) {
    double velocity_gain = gains(1);
    // Estimate z-velocity innovation from position innovation and dt
    double velocity_innovation = last_innovation_ / std::max(dt, 0.01);
    corrected_velocity(2) += velocity_gain * velocity_innovation;
  }

  // Update variance using measurement
  double measurement_variance = params_.baro_noise_sigma * params_.baro_noise_sigma;
  updateVariance(position_gain, measurement_variance);

  // Log significant corrections
  if (std::abs(last_innovation_) > 0.05)  // > 5cm
  {
    logger_.debug() << "[BaroInt] Large correction: innovation=" << last_innovation_ << "m, gain=" << position_gain
              << ", uncertainty=" << std::sqrt(z_variance_) << "m" << std::endl;
  }

  return gtsam::NavState(corrected_pose, corrected_velocity);
}

gtsam::Pose3 BarometerIntegrator::correctPose(const gtsam::Pose3& propagated_pose, double barometer_depth, double dt)
{
  // Compute innovation
  double predicted_z = propagated_pose.z();
  last_innovation_ = barometer_depth - predicted_z;

  // Check if correction is needed
  if (std::abs(last_innovation_) < params_.min_correction_threshold) {
    return propagated_pose;
  }

  // Predict variance growth
  predictVariance(dt);

  // Compute Kalman gain (position only)
  Eigen::VectorXd gains = computeKalmanGain(dt);
  double position_gain = gains(0);

  // Correct z-position
  double corrected_z = predicted_z + position_gain * last_innovation_;

  // Build corrected pose
  gtsam::Point3 corrected_position(propagated_pose.x(), propagated_pose.y(), corrected_z);

  // Update variance using measurement
  double measurement_variance = params_.baro_noise_sigma * params_.baro_noise_sigma;
  updateVariance(position_gain, measurement_variance);

  return gtsam::Pose3(propagated_pose.rotation(), corrected_position);
}

void BarometerIntegrator::reset()
{
  // Reset to initial variance (σ²)
  z_variance_ = params_.baro_noise_sigma * params_.baro_noise_sigma;
  last_innovation_ = 0.0;
  last_update_time_ = 0.0;
}

BarometerIntegratorState BarometerIntegrator::saveState() const
{
  return BarometerIntegratorState(z_variance_, last_innovation_, last_update_time_);
}

void BarometerIntegrator::loadState(const BarometerIntegratorState& state)
{
  z_variance_ = state.z_variance;
  last_innovation_ = state.last_innovation;
  last_update_time_ = state.last_update_time;
}

}  // namespace turtlmap
