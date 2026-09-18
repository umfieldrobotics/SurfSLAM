#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>

#include <Eigen/Dense>
#include <cmath>

#include "Logger.h"

namespace turtlmap {

/**
 * @brief Saved state for BarometerIntegrator
 *
 * Contains all internal state variables needed to restore the integrator
 * to a previous state. Useful for keyframe management and state rollback.
 */
struct BarometerIntegratorState
{
  double z_variance;        // Z-variance estimate (meters²)
  double last_innovation;   // Last measurement residual (meters)
  double last_update_time;  // Time of last barometer measurement

  BarometerIntegratorState()
      : z_variance(0.0), last_innovation(0.0), last_update_time(0.0)
  {}

  BarometerIntegratorState(double variance, double innovation, double update_time)
      : z_variance(variance), last_innovation(innovation), last_update_time(update_time)
  {}
};

/**
 * @brief Integrates barometer measurements into navigation state propagation
 *
 * Uses an EKF-like update to correct the z-component of the propagated state
 * while maintaining proper uncertainty quantification. This prevents z-drift
 * accumulation between keyframes without introducing discontinuous jumps.
 */
class BarometerIntegrator
{
 public:
  /**
   * @brief Configuration parameters for barometer integration
   */
  struct Params
  {
    double baro_noise_sigma;          // Barometer measurement noise (meters)
    double min_correction_threshold;  // Minimum error before correction (meters)
    bool enable_velocity_correction;  // Also correct z-velocity

    // IMU noise parameters (used to compute z-drift rate)
    double acc_noise_density;  // Accelerometer noise density [m/s^2/sqrt(Hz)]
    double acc_random_walk;    // Accelerometer bias random walk [m/s^2/sqrt(Hz)]
    double imu_rate;           // IMU sampling rate [Hz]

    Params()
        : baro_noise_sigma(0.01)  // 1cm noise
          ,
          min_correction_threshold(0.005)  // 5mm threshold
          ,
          enable_velocity_correction(true),
          acc_noise_density(0.001)  // Default IMU noise
          ,
          acc_random_walk(0.0001)  // Default IMU bias drift
          ,
          imu_rate(200.0)  // Default IMU rate
    {}

    /**
     * @brief Compute IMU z-drift rate from noise parameters
     *
     * This computes the process noise (uncertainty growth) based on IMU error propagation.
     *
     * For IMU-based position estimation:
     * - Accelerometer white noise integrates to position uncertainty: σ_z ∝ t^(3/2)
     * - Accelerometer bias random walk integrates to: σ_z ∝ t^(5/2)
     *
     * We return the power spectral density that will be used with proper
     * time-dependent integration in predictUncertainty().
     */
    double computeProcessNoiseRate() const
    {
      // The acc_random_walk and acc_noise_density are already in the correct units
      // for computing position variance growth rate
      // These will be integrated with proper power laws in predictUncertainty()
      return acc_random_walk + acc_noise_density;
    }
  };

  BarometerIntegrator(const Params& params, Logger& logger);

  /**
   * @brief Correct a propagated navigation state using barometer measurement
   *
   * @param propagated_state State from IMU preintegration
   * @param barometer_depth Measured depth from barometer (meters, down positive)
   * @param dt Time since last barometer update (seconds)
   * @return Corrected navigation state
   */
  gtsam::NavState correctState(const gtsam::NavState& propagated_state, double barometer_depth, double dt);

  /**
   * @brief Correct pose with barometer, simpler interface
   *
   * @param propagated_pose Pose from dead reckoning
   * @param barometer_depth Measured depth (meters)
   * @param dt Time since last update (seconds)
   * @return Corrected pose
   */
  gtsam::Pose3 correctPose(const gtsam::Pose3& propagated_pose, double barometer_depth, double dt);

  /**
   * @brief Compute Kalman gain for z-correction
   *
   * @param dt Time since last measurement
   * @return Kalman gain for position (and velocity if enabled)
   */
  Eigen::VectorXd computeKalmanGain(double dt) const;

  /**
   * @brief Get innovation (measurement residual)
   */
  double getLastInnovation() const { return last_innovation_; }

  /**
   * @brief Reset internal state (e.g., at keyframe)
   */
  void reset();

  /**
   * @brief Get current uncertainty estimate
   */
  double getCurrentUncertainty() const { return std::sqrt(z_variance_); }

  /**
   * @brief Get current variance estimate
   */
  double getCurrentVariance() const { return z_variance_; }

  /**
   * @brief Save current state for later restoration
   *
   * @return Current state of the integrator
   */
  BarometerIntegratorState saveState() const;

  /**
   * @brief Load a previously saved state
   *
   * @param state State to restore
   */
  void loadState(const BarometerIntegratorState& state);

 private:
  Params params_;
  Logger& logger_;
  double z_variance_;        // Current z-variance estimate (meters²) - working in variance space!
  double last_innovation_;   // Last measurement residual (meters)
  double last_update_time_;  // Time of last barometer measurement

  /**
   * @brief Predict variance growth due to IMU drift (proper power-law integration)
   */
  void predictVariance(double dt);

  /**
   * @brief Update variance after measurement correction (proper Kalman update)
   */
  void updateVariance(double kalman_gain, double measurement_variance);
};

}  // namespace turtlmap
