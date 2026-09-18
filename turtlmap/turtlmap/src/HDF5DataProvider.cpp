#include "HDF5DataProvider.h"

#include <cstring>
#include <iostream>

namespace turtlmap {

HDF5DataProvider::HDF5DataProvider(const std::string& h5_path, double playback_rate)
    : streamer_(h5_path), playback_rate_(playback_rate)
{}

HDF5DataProvider::~HDF5DataProvider() { stop(); }

void HDF5DataProvider::setImuCallback(std::function<void(const ImuMeasurement&)> callback) { imu_callback_ = callback; }

void HDF5DataProvider::setDvlCallback(std::function<void(const DvlMeasurement&)> callback) { dvl_callback_ = callback; }

void HDF5DataProvider::setBarometerCallback(std::function<void(const BarometerMeasurement&)> callback)
{
  baro_callback_ = callback;
}

void HDF5DataProvider::setStereoCallback(std::function<void(const StereoMeasurement&)> callback)
{
  // Not implemented for HDF5DataProvider
}

void HDF5DataProvider::start()
{
  if (running_) {
    std::cerr << "HDF5DataProvider already running" << std::endl;
    return;
  }

  std::cout << "Loading HDF5 data..." << std::endl;
  streamer_.load();
  std::cout << "HDF5 data loaded, starting replay..." << std::endl;

  running_ = true;
  replay_thread_ = std::thread(&HDF5DataProvider::replayLoop, this);
}

void HDF5DataProvider::stop()
{
  if (!running_) {
    return;
  }

  running_ = false;
  if (replay_thread_.joinable()) {
    replay_thread_.join();
  }
}

void HDF5DataProvider::replayLoop()
{
  const auto& events = streamer_.events();

  if (events.empty()) {
    std::cerr << "No events in HDF5 file" << std::endl;
    running_ = false;
    return;
  }

  double start_time = events[0].stamp;
  auto replay_start = std::chrono::steady_clock::now();

  for (const auto& event : events) {
    if (!running_) {
      break;
    }

    // Real-time pacing (if playback_rate > 0)
    if (playback_rate_ > 0) {
      double elapsed_sim = (event.stamp - start_time) / playback_rate_;
      auto elapsed_real = std::chrono::steady_clock::now() - replay_start;
      auto target = std::chrono::duration<double>(elapsed_sim);

      if (elapsed_real < target) {
        std::this_thread::sleep_for(target - elapsed_real);
      }
    }

    // Dispatch to appropriate callback
    try {
      if (event.type == hdf5::Event::IMU && imu_callback_) {
        ImuMeasurement imu = convertImu(event);
        imu_callback_(imu);
      } else if (event.type == hdf5::Event::DVL && dvl_callback_) {
        DvlMeasurement dvl = convertDvl(event);
        dvl_callback_(dvl);
      } else if (event.type == hdf5::Event::PRESSURE && baro_callback_) {
        BarometerMeasurement baro = convertBaro(event);
        baro_callback_(baro);
      } else if (event.type == hdf5::Event::STEREO && stereo_callback_) {
        std::cerr << "Got Stereo Event!, not doing anything with it" << std::endl;
      }
    } catch (const std::exception& e) {
      std::cerr << "Error processing event: " << e.what() << std::endl;
    }
  }

  std::cout << "HDF5 replay complete" << std::endl;
  running_ = false;
}

ImuMeasurement HDF5DataProvider::convertImu(const hdf5::Event& event)
{
  const auto& imu_data = streamer_.imu();
  size_t idx = event.index;

  ImuMeasurement msg;
  msg.timestamp = imu_data.stamps[idx];

  // Orientation (quaternion): qx, qy, qz, qw
  msg.qx = imu_data.orientation[idx * 4 + 0];
  msg.qy = imu_data.orientation[idx * 4 + 1];
  msg.qz = imu_data.orientation[idx * 4 + 2];
  msg.qw = imu_data.orientation[idx * 4 + 3];

  // Angular velocity
  msg.wx = imu_data.angular_velocity[idx * 3 + 0];
  msg.wy = imu_data.angular_velocity[idx * 3 + 1];
  msg.wz = imu_data.angular_velocity[idx * 3 + 2];

  // Linear acceleration
  msg.ax = imu_data.linear_acceleration[idx * 3 + 0];
  msg.ay = imu_data.linear_acceleration[idx * 3 + 1];
  msg.az = imu_data.linear_acceleration[idx * 3 + 2];

  return msg;
}

DvlMeasurement HDF5DataProvider::convertDvl(const hdf5::Event& event)
{
  const auto& dvl_data = streamer_.dvl();
  size_t idx = event.index;

  DvlMeasurement msg;
  msg.timestamp = dvl_data.stamps.empty() ? 0.0 : dvl_data.stamps[idx];
  msg.time = dvl_data.time.empty() ? 0.0 : dvl_data.time[idx];

  // Velocity
  if (!dvl_data.velocity.empty()) {
    msg.vx = dvl_data.velocity[idx * 3 + 0];
    msg.vy = dvl_data.velocity[idx * 3 + 1];
    msg.vz = dvl_data.velocity[idx * 3 + 2];
  } else {
    msg.vx = msg.vy = msg.vz = 0.0;
  }

  // Velocity covariance
  if (!dvl_data.velocity_covariance.empty()) {
    for (int i = 0; i < 9; ++i) {
      msg.velocity_covariance[i] = dvl_data.velocity_covariance[idx * 9 + i];
    }
  } else {
    std::memset(msg.velocity_covariance.data(), 0, sizeof(msg.velocity_covariance));
  }

  // FOM
  msg.fom = dvl_data.fom.empty() ? 0.0 : dvl_data.fom[idx];

  // Altitude
  msg.altitude = dvl_data.altitude.empty() ? 0.0 : dvl_data.altitude[idx];

  // Velocity valid
  msg.velocity_valid = dvl_data.velocity_valid.empty() ? false : (dvl_data.velocity_valid[idx] > 0.5);

  // Status
  msg.status = dvl_data.status.empty() ? 0 : static_cast<int>(dvl_data.status[idx]);

  return msg;
}

BarometerMeasurement HDF5DataProvider::convertBaro(const hdf5::Event& event)
{
  const auto& pres_data = streamer_.pressure();
  size_t idx = event.index;

  BarometerMeasurement msg;
  msg.timestamp = pres_data.stamps[idx];
  msg.pressure = pres_data.pressure[idx];
  msg.variance = pres_data.variance[idx];

  return msg;
}

}  // namespace turtlmap
