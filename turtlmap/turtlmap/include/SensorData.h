#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <queue>
#include <concepts>
#include <Eigen/Dense>
#include <variant>

namespace turtlmap {

struct ImuMeasurement
{
  double timestamp;
  double qw, qx, qy, qz;  // orientation quaternion
  double wx, wy, wz;      // angular velocity (rad/s)
  double ax, ay, az;      // linear acceleration (m/s^2)
  bool make_keyframe{false}; 
  bool skip_optimize{false};
};

struct DvlMeasurement
{
  double timestamp;
  double time;                                // DVL internal time
  double vx, vy, vz;                          // velocity (m/s)
  std::array<double, 9> velocity_covariance;  // 3x3 covariance
  double fom;                                 // figure of merit
  double altitude;                            // altitude from seafloor
  bool velocity_valid;
  int status;
};

struct BarometerMeasurement
{
  double timestamp;
  double pressure;  // Pa
  double variance;
};

struct StereoMeasurement
{
  double timestamp;
  double prev_kf_timestamp;
  double curr_kf_timestamp;
  Eigen::Matrix4d transformation;
};

using SensorData = std::variant<ImuMeasurement,
                                DvlMeasurement, 
                                BarometerMeasurement,
                                StereoMeasurement>;
struct SensorDataCmp
{
  bool operator()(const SensorData &l, const SensorData &r) const
  {
    auto get_ts = [](const auto &m) { return m.timestamp; };
    double tl = std::visit(get_ts, l);
    double tr = std::visit(get_ts, r);
    return tl > tr;  // oldest (smallest) timestamp becomes highest priority
  }
};

using SensorDataQueue = std::priority_queue<SensorData, std::vector<SensorData>, SensorDataCmp>;

}  // namespace turtlmap
