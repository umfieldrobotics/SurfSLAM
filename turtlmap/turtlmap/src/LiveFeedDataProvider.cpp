#include "LiveFeedDataProvider.h"

#include <iostream>

namespace turtlmap {

LiveFeedDataProvider::~LiveFeedDataProvider() { stop(); }

void LiveFeedDataProvider::start()
{
  if (running_) {
    std::cerr << "LiveFeedDataProvider already running" << std::endl;
    return;
  }

  running_ = true;
}

void LiveFeedDataProvider::stop()
{
  if (!running_) {
    return;
  }

  running_ = false;
}

void LiveFeedDataProvider::insertIMU(const ImuMeasurement& meas) { imu_callback_(meas); }

void LiveFeedDataProvider::insertDVL(const DvlMeasurement& meas) { dvl_callback_(meas); }

void LiveFeedDataProvider::insertBaro(const BarometerMeasurement& meas) { baro_callback_(meas); }

void LiveFeedDataProvider::insertStereo(const StereoMeasurement& meas) { stereo_callback_(meas); }

void LiveFeedDataProvider::setImuCallback(std::function<void(const ImuMeasurement&)> callback)
{
  imu_callback_ = callback;
}

void LiveFeedDataProvider::setDvlCallback(std::function<void(const DvlMeasurement&)> callback)
{
  dvl_callback_ = callback;
}

void LiveFeedDataProvider::setBarometerCallback(std::function<void(const BarometerMeasurement&)> callback)
{
  baro_callback_ = callback;
}

void LiveFeedDataProvider::setStereoCallback(std::function<void(const StereoMeasurement&)> callback)
{
  stereo_callback_ = callback;
}

}  // namespace turtlmap