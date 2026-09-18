#pragma once

#include <gtsam/geometry/Rot3.h>

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace turtlmap {

/**
 * @brief Average rotations using SO(3) Lie algebra (preferred method)
 *
 * This method averages rotations in the tangent space at the identity
 * using the exponential map (ExpMap) and logarithm map (LogMap).
 *
 * Algorithm:
 * 1. Choose initial estimate (first rotation or identity)
 * 2. For each rotation R_i, compute the relative rotation: ΔR_i = R_mean^{-1} * R_i
 * 3. Map to tangent space: ω_i = LogMap(ΔR_i) ∈ ℝ³
 * 4. Average in tangent space: ω_avg = (1/N) * Σ ω_i
 * 5. Update estimate: R_mean = R_mean * ExpMap(ω_avg)
 * 6. Iterate until convergence
 *
 * @param rotations Vector of gtsam::Rot3 rotations to average
 * @return Average rotation
 */
inline gtsam::Rot3 averageRotationsSO3(const std::vector<gtsam::Rot3>& rotations)
{
  if (rotations.empty()) {
    return gtsam::Rot3::Identity();
  }

  if (rotations.size() == 1) {
    return rotations[0];
  }

  // Initialize with first rotation as reference
  gtsam::Rot3 R_mean = rotations[0];

  const int max_iterations = 10;
  const double convergence_threshold = 1e-6;

  for (int iter = 0; iter < max_iterations; ++iter) {
    // Accumulate tangent vectors
    gtsam::Vector3 omega_sum = gtsam::Vector3::Zero();

    for (const auto& R_i : rotations) {
      // Compute relative rotation: ΔR = R_mean^{-1} * R_i
      gtsam::Rot3 delta_R = R_mean.inverse() * R_i;

      // Map to tangent space using LogMap
      // LogMap converts SO(3) rotation to so(3) Lie algebra (axis-angle in ℝ³)
      gtsam::Vector3 omega_i = gtsam::Rot3::Logmap(delta_R);

      omega_sum += omega_i;
    }

    // Average in tangent space
    gtsam::Vector3 omega_avg = omega_sum / rotations.size();

    // Check convergence
    double error_norm = omega_avg.norm();
    if (error_norm < convergence_threshold) {
      break;
    }

    // Update estimate: R_mean = R_mean * ExpMap(ω_avg)
    // ExpMap converts so(3) Lie algebra back to SO(3) rotation
    gtsam::Rot3 delta_R_update = gtsam::Rot3::Expmap(omega_avg);
    R_mean = R_mean * delta_R_update;
  }

  return R_mean;
}

/**
 * @brief Average quaternions using iterative method (legacy/alternative method)
 *
 * This implements quaternion averaging using tangent space on S³.
 * Less preferred than SO(3) method but kept for reference.
 *
 * @param quats Input quaternions to average (must be non-empty)
 * @return Average quaternion (normalized)
 */
inline Eigen::Quaterniond averageQuaternions(const std::vector<Eigen::Quaterniond>& quats)
{
  if (quats.empty()) {
    return Eigen::Quaterniond::Identity();
  }

  if (quats.size() == 1) {
    return quats[0].normalized();
  }

  // Method 1: Simple iterative averaging (works well for small spreads)
  // Use first quaternion as initial reference
  Eigen::Quaterniond q_avg = quats[0].normalized();

  const int max_iterations = 10;
  const double convergence_threshold = 1e-6;

  for (int iter = 0; iter < max_iterations; ++iter) {
    Eigen::Vector3d error_sum = Eigen::Vector3d::Zero();

    for (const auto& q : quats) {
      // Ensure we take the shorter path (quaternion double cover)
      Eigen::Quaterniond q_normalized = q.normalized();
      if (q_avg.dot(q_normalized) < 0.0) {
        q_normalized.coeffs() *= -1.0;
      }

      // Compute quaternion difference in tangent space
      Eigen::Quaterniond q_diff = q_avg.inverse() * q_normalized;

      // Convert to axis-angle representation (small angle approximation)
      double angle = 2.0 * std::acos(std::min(1.0, std::abs(q_diff.w())));
      if (angle > 1e-10) {
        Eigen::Vector3d axis = q_diff.vec() / std::sin(angle / 2.0);
        error_sum += angle * axis;
      }
    }

    // Average error
    error_sum /= quats.size();

    // Check convergence
    double error_norm = error_sum.norm();
    if (error_norm < convergence_threshold) {
      break;
    }

    // Update estimate using exponential map
    if (error_norm > 1e-10) {
      Eigen::Vector3d axis = error_sum / error_norm;
      double angle = error_norm;

      Eigen::Quaterniond q_update;
      q_update.w() = std::cos(angle / 2.0);
      q_update.vec() = std::sin(angle / 2.0) * axis;

      q_avg = (q_avg * q_update).normalized();
    }
  }

  return q_avg.normalized();
}

/**
 * @brief Average quaternions using covariance matrix method (more accurate)
 *
 * This method builds the covariance matrix and finds the eigenvector
 * with the largest eigenvalue. More accurate but computationally expensive.
 *
 * @param quats Input quaternions
 * @return Average quaternion
 */
inline Eigen::Quaterniond averageQuaternionsCovarianceMethod(const std::vector<Eigen::Quaterniond>& quats)
{
  if (quats.empty()) {
    return Eigen::Quaterniond::Identity();
  }

  if (quats.size() == 1) {
    return quats[0].normalized();
  }

  // Build 4x4 covariance matrix
  Eigen::Matrix4d M = Eigen::Matrix4d::Zero();

  for (const auto& q : quats) {
    Eigen::Quaterniond q_normalized = q.normalized();
    Eigen::Vector4d q_vec;
    q_vec << q_normalized.w(), q_normalized.x(), q_normalized.y(), q_normalized.z();

    M += q_vec * q_vec.transpose();
  }

  M /= quats.size();

  // Find eigenvector with largest eigenvalue
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(M);

  if (eigensolver.info() != Eigen::Success) {
    std::cerr << "Eigenvalue decomposition failed in quaternion averaging!" << std::endl;
    return quats[0].normalized();
  }

  // Largest eigenvalue corresponds to last column
  Eigen::Vector4d q_avg_vec = eigensolver.eigenvectors().col(3);

  Eigen::Quaterniond q_avg(q_avg_vec(0), q_avg_vec(1), q_avg_vec(2), q_avg_vec(3));

  return q_avg.normalized();
}

}  // namespace turtlmap