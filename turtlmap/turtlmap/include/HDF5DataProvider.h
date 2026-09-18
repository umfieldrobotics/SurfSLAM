#pragma once

#include <atomic>
#include <chrono>
#include <thread>

#include "DataProvider.h"
#include "hdf5_streamer.hpp"

namespace turtlmap {

class HDF5DataProvider : public IDataProvider
{
 public:
  explicit HDF5DataProvider(const std::string& h5_path, double playback_rate = 1.0);
  ~HDF5DataProvider() override;

  void setImuCallback(std::function<void(const ImuMeasurement&)> callback) override;
  void setDvlCallback(std::function<void(const DvlMeasurement&)> callback) override;
  void setBarometerCallback(std::function<void(const BarometerMeasurement&)> callback) override;
  void setStereoCallback(std::function<void(const StereoMeasurement&)> callback) override;

  void start() override;
  void stop() override;
  bool isRunning() const override { return running_; }

 private:
  void replayLoop();

  ImuMeasurement convertImu(const hdf5::Event& event);
  DvlMeasurement convertDvl(const hdf5::Event& event);
  BarometerMeasurement convertBaro(const hdf5::Event& event);
  hdf5::Streamer streamer_;
  std::thread replay_thread_;
  std::atomic<bool> running_{false};
  double playback_rate_;

  std::function<void(const ImuMeasurement&)> imu_callback_;
  std::function<void(const DvlMeasurement&)> dvl_callback_;
  std::function<void(const BarometerMeasurement&)> baro_callback_;
  std::function<void(const StereoMeasurement&)> stereo_callback_;
};

}  // namespace turtlmap
