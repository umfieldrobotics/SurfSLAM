// Main entry point for the HDF5-driven pose graph backend (no ROS)

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

#include "HDF5DataProvider.h"
#include "LiveFeedDataProvider.h"
#include "LiveWebServer.h"
#include "PoseGraphBackend.h"

// Global pointer for signal handler
static turtlmap::PoseGraphBackend* g_backend = nullptr;
static turtlmap::LiveWebServer* g_web_server = nullptr;

void signal_handler(int signal)
{
  std::cout << "\nInterrupt signal (" << signal << ") received. Shutting down..." << std::endl;

  // Stop backend if it exists
  if (g_backend) {
    g_backend->stop();
  }

  // Stop web server if running
  if (g_web_server) {
    g_web_server->stop();
  }
}

// How measurements reach the backend:
//   Pull - the backend is driven directly by the HDF5 streamer.
//   Push - the HDF5 streamer feeds a LiveFeedDataProvider, exercising the same
//          push path that live hardware uses.
enum class FeedMode
{
  Pull,
  Push
};

// Derive the companion sensor file name the backend writes alongside the live trajectory.
static std::string sensorFileFor(const std::string& live_file)
{
  size_t pos = live_file.find_last_of('.');
  if (pos != std::string::npos) {
    return live_file.substr(0, pos) + "_sensors.txt";
  }
  return live_file + "_sensors.txt";
}

int main(int argc, char** argv)
{
  // Set up signal handler for graceful shutdown
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  // Parse command line arguments
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <config.yaml> <data.h5> [playback_rate] [output_trajectory.txt] [--live-vis=on|off] "
                 "[--live-file=path] [--web-port=PORT] [--feed=pull|push]"
              << std::endl;
    std::cerr << "  config.yaml       - Path to configuration file" << std::endl;
    std::cerr << "  data.h5           - Path to HDF5 data file" << std::endl;
    std::cerr << "  playback_rate     - Optional playback rate (default: 1.0, 0 = as fast as possible)" << std::endl;
    std::cerr << "  output_trajectory - Optional output trajectory filename (default: trajectory)" << std::endl;
    std::cerr << "  --live-vis        - Enable/disable live trajectory logging to file (default: on)" << std::endl;
    std::cerr << "  --live-file       - Path to live trajectory file (default: trajectory_live.txt)" << std::endl;
    std::cerr << "  --web-port        - If set (>0), starts a built-in WebGL viewer server on this port" << std::endl;
    std::cerr << "  --feed            - pull: backend reads the HDF5 stream directly (default)" << std::endl;
    std::cerr << "                      push: HDF5 stream is pushed through a LiveFeedDataProvider" << std::endl;
    return 1;
  }

  std::string config_file = argv[1];
  std::string h5_file = argv[2];
  double playback_rate = 1.0;
  std::string output_file = "trajectory";

  if (argc >= 4) {
    playback_rate = std::atof(argv[3]);
  }

  if (argc >= 5) {
    output_file = argv[4];
  }
  bool live_vis = true;  // default on
  std::string live_file = "trajectory_live.txt";
  int web_port = 0;  // disabled by default
  FeedMode feed_mode = FeedMode::Pull;

  // Parse optional flags
  for (int i = 5; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--live-vis=", 0) == 0) {
      std::string v = arg.substr(11);
      if (v == "on" || v == "ON" || v == "1")
        live_vis = true;
      else if (v == "off" || v == "OFF" || v == "0")
        live_vis = false;
    } else if (arg.rfind("--live-file=", 0) == 0) {
      live_file = arg.substr(12);
    } else if (arg.rfind("--web-port=", 0) == 0) {
      web_port = std::atoi(arg.substr(11).c_str());
    } else if (arg.rfind("--feed=", 0) == 0) {
      std::string v = arg.substr(7);
      if (v == "pull") {
        feed_mode = FeedMode::Pull;
      } else if (v == "push") {
        feed_mode = FeedMode::Push;
      } else {
        std::cerr << "Unknown --feed mode '" << v << "' (expected pull or push)" << std::endl;
        return 1;
      }
    }
  }

  std::cout << "=== TurtlMap Pose Graph Backend ===" << std::endl;
  std::cout << "Config file:     " << config_file << std::endl;
  std::cout << "HDF5 data file:  " << h5_file << std::endl;
  std::cout << "Playback rate:   " << playback_rate << "x" << std::endl;
  std::cout << "Output file:     " << output_file << std::endl;
  std::cout << "Live vis:        " << (live_vis ? "on" : "off") << std::endl;
  std::cout << "Live file:       " << live_file << std::endl;
  std::cout << "Web port:        " << (web_port > 0 ? std::to_string(web_port) : "disabled") << std::endl;
  std::cout << "Feed mode:       " << (feed_mode == FeedMode::Pull ? "pull" : "push") << std::endl;
  if (web_port > 0) std::cout << "Web viewer:      http://localhost:" << web_port << std::endl;
  std::cout << "===================================" << std::endl;

  try {
    auto h5_provider = std::make_shared<turtlmap::HDF5DataProvider>(h5_file, playback_rate);

    // In push mode the backend consumes a live feed that the HDF5 stream fills in.
    std::shared_ptr<turtlmap::LiveFeedDataProvider> live_provider;
    std::shared_ptr<turtlmap::IDataProvider> backend_provider = h5_provider;
    if (feed_mode == FeedMode::Push) {
      live_provider = std::make_shared<turtlmap::LiveFeedDataProvider>();
      h5_provider->setImuCallback([&](const turtlmap::ImuMeasurement& m) { live_provider->insertIMU(m); });
      h5_provider->setDvlCallback([&](const turtlmap::DvlMeasurement& m) { live_provider->insertDVL(m); });
      h5_provider->setBarometerCallback(
          [&](const turtlmap::BarometerMeasurement& m) { live_provider->insertBaro(m); });
      backend_provider = live_provider;
    }

    // Create pose graph backend (but don't start processing yet)
    turtlmap::PoseGraphBackend backend(config_file, backend_provider);
    g_backend = &backend;  // Set global pointer for signal handler

    // Configure live visualization
    backend.configureLiveVisualization(live_vis, live_file);

    // Optionally start embedded web viewer
    turtlmap::LiveWebServer web;
    if (web_port > 0) {
      g_web_server = &web;  // Set global pointer for signal handler
      web.start(web_port, live_file, sensorFileFor(live_file));
    }

    backend.run();
    if (live_provider) {
      live_provider->start();
      h5_provider->start();
      while (h5_provider->isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      live_provider->stop();
    } else {
      h5_provider->start();
    }
    backend.waitForCompletion();
    if (backend.saveTrajectory(output_file)) {
      std::cout << "Trajectory saved successfully to " << output_file << std::endl;
    } else {
      std::cerr << "Failed to save trajectory" << std::endl;
    }
    if (web_port > 0) {
      web.stop();
    }
    g_backend = nullptr;
    g_web_server = nullptr;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    g_backend = nullptr;
    g_web_server = nullptr;
    return 1;
  }

  std::cout << "Done!" << std::endl;
  return 0;
}
