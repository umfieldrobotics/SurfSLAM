#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include "DataProvider.h"
#include "hdf5_streamer.hpp"

namespace turtlmap {

class LiveFeedDataProvider : public IDataProvider
{
 public:
  explicit LiveFeedDataProvider() : running_(false) {};
  ~LiveFeedDataProvider() override;

  void setImuCallback(std::function<void(const ImuMeasurement&)> callback) override;
  void setDvlCallback(std::function<void(const DvlMeasurement&)> callback) override;
  void setBarometerCallback(std::function<void(const BarometerMeasurement&)> callback) override;
  void setStereoCallback(std::function<void(const StereoMeasurement&)> callback) override;

  void start() override;
  void stop() override;
  bool isRunning() const override { return running_; }

  void insertIMU(const ImuMeasurement& meas);
  void insertDVL(const DvlMeasurement& meas);
  void insertBaro(const BarometerMeasurement& meas);
  void insertStereo(const StereoMeasurement& meas);

 private:
  std::atomic<bool> running_{false};
  std::function<void(const ImuMeasurement&)> imu_callback_;
  std::function<void(const DvlMeasurement&)> dvl_callback_;
  std::function<void(const BarometerMeasurement&)> baro_callback_;
  std::function<void(const StereoMeasurement&)> stereo_callback_;
};

}  // namespace turtlmap