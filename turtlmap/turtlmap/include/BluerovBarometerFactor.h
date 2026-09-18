#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/BarometricFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/dataset.h>

#include <Eigen/Dense>

using namespace gtsam;

class BluerovBarometerFactor : public NoiseModelFactor1<Pose3>
{
 private:
  /* data */
  double measured_;

 public:
  /// Constructor
  BluerovBarometerFactor() {}
  virtual ~BluerovBarometerFactor() {}
  BluerovBarometerFactor(Key key, double measured, const SharedNoiseModel& model)
      : NoiseModelFactor1<Pose3>(model, key), measured_(measured)
  {}
  // BluerovBarometerFactor(/* args */);
  // ~BluerovBarometerFactor();
  // virtual ~BluerovBarometerFactor() override {}
  Vector evaluateError(const Pose3& pose, boost::optional<gtsam::Matrix&> H) const;
};
