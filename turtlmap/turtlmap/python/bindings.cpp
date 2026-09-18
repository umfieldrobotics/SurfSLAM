#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "SensorData.h"
#include "PoseGraphBackend.h"
#include "HDF5DataProvider.h"
#include "LiveFeedDataProvider.h"

namespace py = pybind11;
using namespace turtlmap;

PYBIND11_MODULE(pyturtlmap, m)
{
  // Register exception translator to preserve C++ stack traces
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const std::exception &e) {
      // Print the exception type and message to stderr for better debugging
      std::cerr << "C++ exception: " << typeid(e).name() << ": " << e.what() << std::endl;
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
  });

  py::class_<turtlmap::ImuMeasurement>(m, "ImuMeasurement")
      .def(py::init<>())
      .def_readwrite("timestamp", &turtlmap::ImuMeasurement::timestamp)
      .def_readwrite("qw", &turtlmap::ImuMeasurement::qw)
      .def_readwrite("qx", &turtlmap::ImuMeasurement::qx)
      .def_readwrite("qy", &turtlmap::ImuMeasurement::qy)
      .def_readwrite("qz", &turtlmap::ImuMeasurement::qz)
      .def_readwrite("wx", &turtlmap::ImuMeasurement::wx)
      .def_readwrite("wy", &turtlmap::ImuMeasurement::wy)
      .def_readwrite("wz", &turtlmap::ImuMeasurement::wz)
      .def_readwrite("ax", &turtlmap::ImuMeasurement::ax)
      .def_readwrite("ay", &turtlmap::ImuMeasurement::ay)
      .def_readwrite("az", &turtlmap::ImuMeasurement::az)
      .def_readwrite("make_keyframe", &turtlmap::ImuMeasurement::make_keyframe)
      .def_readwrite("skip_optimize", &turtlmap::ImuMeasurement::skip_optimize)
      .def(py::pickle(
          [](const turtlmap::ImuMeasurement &o)
          {
            return py::make_tuple(
                o.timestamp,
                o.qw, o.qx, o.qy, o.qz,
                o.wx, o.wy, o.wz,
                o.ax, o.ay, o.az,
                o.make_keyframe,
                o.skip_optimize);
          },
          [](py::tuple t)
          {
            if (t.size() != 13)
              throw std::runtime_error("Invalid state!");
            turtlmap::ImuMeasurement o;
            o.timestamp = t[0].cast<double>();
            o.qw = t[1].cast<double>();
            o.qx = t[2].cast<double>();
            o.qy = t[3].cast<double>();
            o.qz = t[4].cast<double>();
            o.wx = t[5].cast<double>();
            o.wy = t[6].cast<double>();
            o.wz = t[7].cast<double>();
            o.ax = t[8].cast<double>();
            o.ay = t[9].cast<double>();
            o.az = t[10].cast<double>();
            o.make_keyframe = t[11].cast<bool>();
            o.skip_optimize = t[12].cast<bool>();
            return o;
          }));

  py::class_<turtlmap::DvlMeasurement>(m, "DvlMeasurement")
      .def(py::init<>())
      .def_readwrite("timestamp", &turtlmap::DvlMeasurement::timestamp)
      .def_readwrite("time", &turtlmap::DvlMeasurement::time)
      .def_readwrite("vx", &turtlmap::DvlMeasurement::vx)
      .def_readwrite("vy", &turtlmap::DvlMeasurement::vy)
      .def_readwrite("vz", &turtlmap::DvlMeasurement::vz)
      .def_readwrite("velocity_covariance", &turtlmap::DvlMeasurement::velocity_covariance)
      .def_readwrite("fom", &turtlmap::DvlMeasurement::fom)
      .def_readwrite("altitude", &turtlmap::DvlMeasurement::altitude)
      .def_readwrite("velocity_valid", &turtlmap::DvlMeasurement::velocity_valid)
      .def_readwrite("status", &turtlmap::DvlMeasurement::status)
      .def(py::pickle(
          [](const turtlmap::DvlMeasurement &o)
          {
            return py::make_tuple(
                o.timestamp, o.time,
                o.vx, o.vy, o.vz,
                py::cast(o.velocity_covariance), // convert array to Python list
                o.fom, o.altitude,
                o.velocity_valid, o.status);
          },
          [](py::tuple t)
          {
            if (t.size() != 10)
              throw std::runtime_error("Invalid state!");
            turtlmap::DvlMeasurement o;
            o.timestamp = t[0].cast<double>();
            o.time = t[1].cast<double>();
            o.vx = t[2].cast<double>();
            o.vy = t[3].cast<double>();
            o.vz = t[4].cast<double>();

            auto cov = t[5].cast<std::vector<double>>();
            if (cov.size() != 9)
              throw std::runtime_error("velocity_covariance must have size 9");
            for (size_t i = 0; i < 9; ++i)
              o.velocity_covariance[i] = cov[i];

            o.fom = t[6].cast<double>();
            o.altitude = t[7].cast<double>();
            o.velocity_valid = t[8].cast<bool>();
            o.status = t[9].cast<int>();
            return o;
          }));

  py::class_<turtlmap::BarometerMeasurement>(m, "BarometerMeasurement")
      .def(py::init<>())
      .def_readwrite("timestamp", &turtlmap::BarometerMeasurement::timestamp)
      .def_readwrite("pressure", &turtlmap::BarometerMeasurement::pressure)
      .def_readwrite("variance", &turtlmap::BarometerMeasurement::variance)
      .def(py::pickle(
          [](const turtlmap::BarometerMeasurement &o)
          {
            return py::make_tuple(
                o.timestamp, o.pressure, o.variance);
          },
          [](py::tuple t)
          {
            if (t.size() != 3)
              throw std::runtime_error("Invalid state!");
            turtlmap::BarometerMeasurement o;
            o.timestamp = t[0].cast<double>();
            o.pressure = t[1].cast<double>();
            o.variance = t[2].cast<double>();
            return o;
          }));

  py::class_<turtlmap::StereoMeasurement>(m, "StereoMeasurement")
      .def(py::init<>())
      .def_readwrite("timestamp", &turtlmap::StereoMeasurement::timestamp)
      .def_readwrite("prev_kf_timestamp", &turtlmap::StereoMeasurement::prev_kf_timestamp)
      .def_readwrite("curr_kf_timestamp", &turtlmap::StereoMeasurement::curr_kf_timestamp)
      .def_readwrite("transformation", &turtlmap::StereoMeasurement::transformation)
      .def(py::pickle(
          [](const turtlmap::StereoMeasurement &o) {
            return py::make_tuple(
                o.timestamp,
                o.prev_kf_timestamp, o.curr_kf_timestamp,
                o.transformation);
          },
          [](py::tuple t) {
            if (t.size() != 4) throw std::runtime_error("Invalid state!");
            turtlmap::StereoMeasurement o;
            o.timestamp = t[0].cast<double>();
            o.prev_kf_timestamp = t[1].cast<int>();
            o.curr_kf_timestamp = t[2].cast<int>();
            o.transformation = t[3].cast<Eigen::Matrix4d>();
            return o;
          }));

  py::class_<PoseGraphBackend, std::shared_ptr<PoseGraphBackend>>(m, "PoseGraphBackend")
      .def(py::init<const std::string &, std::shared_ptr<turtlmap::IDataProvider>>(),
           py::arg("config_file"), py::arg("data_provider"))

      .def("run", &PoseGraphBackend::run)
      .def("stop", &PoseGraphBackend::stop)
      .def("is_running", &PoseGraphBackend::isRunning)
      .def("save_trajectory", &PoseGraphBackend::saveTrajectory, py::arg("filename"))
      .def("save_dense_trajectory", &PoseGraphBackend::saveDenseTrajectory, py::arg("filename"))
      .def("wait_for_completion", &PoseGraphBackend::waitForCompletion)

      .def("get_processed_keyframe_count", &PoseGraphBackend::getProcessedKeyframeCount)
      .def("get_latest_timestamp", &PoseGraphBackend::getLatestTimestamp)
      .def("get_latest_kf_timestamp", &PoseGraphBackend::getLatestKeyFrameTime)
      .def("configure_live_visualization",
           &PoseGraphBackend::configureLiveVisualization,
           py::arg("enabled"), py::arg("filepath") = "")
      .def("get_pose_at_time",
           &PoseGraphBackend::getPoseAtTime,
           py::arg("timestamp"))
      .def("get_pose_at_keyframe",
           &PoseGraphBackend::getPoseAtKeyframe,
           py::arg("index"))
      .def("get_pose_at_keyframe_published",
           &PoseGraphBackend::getPoseAtKeyframePublished,
           py::arg("index"))
      .def("get_latest_pose", &PoseGraphBackend::getLatestPose)
      .def("get_updated_graph", &PoseGraphBackend::getUpdatedGraph)
      .def("get_latest_pose", &PoseGraphBackend::getLatestPose)
      .def("has_pose_for_time", &PoseGraphBackend::hasPoseForTime)
      .def("set_traj_pose_queue_max_size",
           &PoseGraphBackend::setTrajPoseQueueMaxSize,
           py::arg("n"))
      .def("get_first_pose_time", &PoseGraphBackend::getFirstPoseTime)
      .def("get_cov_at_time",
           &PoseGraphBackend::getCovAtTime,
           py::arg("timestamp"), py::arg("use_first_opt") = false)
      .def("get_first_opt_kf_pose", &PoseGraphBackend::getFirstOptimizedKeyframePose,
           py::arg("kf_timestamp"), py::arg("eps")=1e-2)
      .def("is_graph_initialized", &PoseGraphBackend::isGraphInitialized)
      .def("trigger_stereo_keyframe", &PoseGraphBackend::triggerStereoKeyframe,
           py::arg("timestamp"));

  // expose the abstract base
  py::class_<turtlmap::IDataProvider, std::shared_ptr<turtlmap::IDataProvider>>(m, "IDataProvider");

  // Live feed provider
  py::class_<turtlmap::LiveFeedDataProvider, turtlmap::IDataProvider,
             std::shared_ptr<turtlmap::LiveFeedDataProvider>>(m, "LiveFeedDataProvider")
      .def(py::init<>())
      .def("start", [](turtlmap::LiveFeedDataProvider &self)
           { py::gil_scoped_release r; self.start(); })
      .def("stop", [](turtlmap::LiveFeedDataProvider &self)
           { py::gil_scoped_release r; self.stop(); })
      .def("insert_imu", &turtlmap::LiveFeedDataProvider::insertIMU)
      .def("insert_dvl", &turtlmap::LiveFeedDataProvider::insertDVL)
      .def("insert_baro", &turtlmap::LiveFeedDataProvider::insertBaro)
      .def("insert_stereo", &turtlmap::LiveFeedDataProvider::insertStereo);
}