#pragma once

#include <atomic>
#include <string>
#include <thread>

namespace turtlmap {

// A minimal HTTP server for serving a live 3D WebGL trajectory viewer and the
// current trajectory data. No external deps; blocking accept loop in a thread.
class LiveWebServer
{
 public:
  LiveWebServer();
  ~LiveWebServer();

  // Starts server on given port. The server will serve:
  //  - GET /           -> embedded HTML for WebGL viewer
  //  - GET /index.html -> same as /
  //  - GET /traj       -> contents of trajectory file (text/plain)
  //  - GET /sensors    -> contents of sensor data file (text/plain)
  bool start(int port, const std::string& trajectory_file, const std::string& sensor_file = "");

  void stop();

 private:
  void run(int port, std::string trajectory_file, std::string sensor_file);
  void cleanup();

  std::thread server_thread_;
  std::atomic<bool> running_{false};
  std::atomic<int> server_fd_{-1};         // Socket file descriptor
  std::atomic<size_t> last_file_size_{0};  // Track file size for incremental reads
};

}  // namespace turtlmap
