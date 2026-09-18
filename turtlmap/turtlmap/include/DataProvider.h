#pragma once

#include <functional>
#include <memory>

#include "SensorData.h"

namespace turtlmap {

class IDataProvider
{
 public:
  virtual ~IDataProvider() = default;

  // Set callbacks to be invoked when new data arrives
  virtual void setImuCallback(std::function<void(const ImuMeasurement&)> callback) = 0;
  virtual void setDvlCallback(std::function<void(const DvlMeasurement&)> callback) = 0;
  virtual void setBarometerCallback(std::function<void(const BarometerMeasurement&)> callback) = 0;
  virtual void setStereoCallback(std::function<void(const StereoMeasurement&)> callback) = 0;

  // Control streaming
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual bool isRunning() const = 0;
};

}  // namespace turtlmap
