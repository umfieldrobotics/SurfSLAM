#pragma once

#include <string>
#include <vector>

namespace hdf5 {

struct Event {
  enum Type { IMU, PRESSURE, DVL, STEREO, GENERIC } type;
  std::string group;   // group name
  size_t index;        // row index
  double stamp;        // seconds
};

struct ImuData {
  std::vector<double> stamps;              // N
  std::vector<double> orientation;         // N x 4 (row-major)
  std::vector<double> angular_velocity;    // N x 3
  std::vector<double> linear_acceleration; // N x 3
};

struct PressureData {
  std::vector<double> stamps;   // N
  std::vector<double> pressure; // N
  std::vector<double> variance; // N
};

struct DvlData {
  std::vector<double> stamps;              // N
  std::vector<double> time;                // N
  std::vector<double> velocity;            // N x 3
  std::vector<double> velocity_covariance; // N x 9
  std::vector<double> fom;                 // N
  std::vector<double> altitude;            // N
  std::vector<double> velocity_valid;      // N
  std::vector<double> status;              // N
};

struct StereoData {
  std::vector<double> stamps;              // N
  std::vector<double> transformation;      // N x 7 (xyz + quaternion)
};

class Streamer {
public:
  explicit Streamer(const std::string& h5_path);
  ~Streamer();

  // Optionally restrict groups before load(); if empty, auto-detect known groups
  void setIncludeGroups(const std::vector<std::string>& groups);

  // Load datasets and build an event timeline (sorted by stamp)
  void load();

  // Accessors (valid after load)
  const std::vector<Event>& events() const; // merged sorted events
  const ImuData& imu() const;               // may be empty
  const PressureData& pressure() const;     // may be empty
  const DvlData& dvl() const;               // may be empty
  const StereoData& stereo() const;         // may be empty

  // Convenience: return earliest timestamp among all events (NAN if none)
  double firstStamp() const;

private:
  struct Impl; // PIMPL to keep header small
  Impl* pimpl_;
};

} // namespace hdf5
