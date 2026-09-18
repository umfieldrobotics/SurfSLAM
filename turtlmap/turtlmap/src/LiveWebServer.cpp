#include "LiveWebServer.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
// Global pointer for signal handler
static turtlmap::LiveWebServer* g_server_instance = nullptr;

// Signal handler
void signal_handler(int signum)
{
  if (g_server_instance) {
    g_server_instance->stop();
  }
  exit(signum);
}
// Frontend HTML extracted to disk: web/index.html
// The embedded fallback was removed to require an editable file on disk.

std::string http_ok(const std::string& content_type, const std::string& body)
{
  std::string hdr =
      "HTTP/1.1 200 OK\r\n"
      "Connection: close\r\n"
      "Content-Type: " +
      content_type +
      "\r\n"
      "Content-Length: " +
      std::to_string(body.size()) +
      "\r\n"
      "Access-Control-Allow-Origin: *\r\n"
      "Cross-Origin-Opener-Policy: same-origin\r\n"
      "Cross-Origin-Embedder-Policy: require-corp\r\n\r\n";
  return hdr + body;
}

std::string http_404()
{
  std::string body = "Not Found";
  return "HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Type: text/plain\r\nContent-Length: " +
         std::to_string(body.size()) + "\r\n\r\n" + body;
}

std::string read_file(const std::string& path)
{
  std::ifstream ifs(path);
  if (!ifs.is_open()) return std::string();
  return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

// Returns the directory containing the running executable, or empty on error.
std::string get_exe_dir()
{
  char buf[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (len <= 0) return std::string();
  buf[len] = '\0';
  std::string full(buf);
  size_t pos = full.find_last_of('/');
  if (pos == std::string::npos) return std::string();
  return full.substr(0, pos);
}

// Try to locate web directory using a few sensible paths relative to the
// executable. Returns the web directory path or empty if not found.
std::string find_web_dir()
{
  std::string exe_dir = get_exe_dir();
  std::vector<std::string> candidates;
  if (!exe_dir.empty()) {
    // If executable is in build/turtlmap, ../turtlmap/web points to source web dir
    candidates.push_back(exe_dir + "/web");
    candidates.push_back(exe_dir + "/../turtlmap/web");
    candidates.push_back(exe_dir + "/../web");
    candidates.push_back(exe_dir + "/../../turtlmap/web");
  }
  // Also try relative to current working directory for backwards compatibility
  candidates.push_back(std::string("web"));

  for (const auto& dir : candidates) {
    // Check if index.html exists in this directory
    std::string index_path = dir + "/index.html";
    std::ifstream ifs(index_path);
    if (ifs.is_open()) return dir;
  }
  return std::string();
}

// Get MIME type for a file based on its extension
std::string get_mime_type(const std::string& path)
{
  size_t dot = path.find_last_of('.');
  if (dot == std::string::npos) return "application/octet-stream";
  std::string ext = path.substr(dot);
  if (ext == ".html") return "text/html; charset=utf-8";
  if (ext == ".js") return "application/javascript; charset=utf-8";
  if (ext == ".mjs") return "application/javascript; charset=utf-8";
  if (ext == ".map") return "application/json; charset=utf-8";
  if (ext == ".json") return "application/json; charset=utf-8";
  if (ext == ".css") return "text/css; charset=utf-8";
  if (ext == ".png") return "image/png";
  if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
  if (ext == ".gif") return "image/gif";
  if (ext == ".svg") return "image/svg+xml";
  if (ext == ".woff" || ext == ".woff2") return "font/woff2";
  if (ext == ".ttf") return "font/ttf";
  return "application/octet-stream";
}

// Check if a path is safe (no directory traversal)
bool is_safe_path(const std::string& base, const std::string& path)
{
  if (path.find("..") != std::string::npos) return false;
  char real_base[PATH_MAX];
  char real_full[PATH_MAX];
  if (!realpath(base.c_str(), real_base)) return false;
  std::string full = base + path;
  if (!realpath(full.c_str(), real_full)) return false;
  return strncmp(real_base, real_full, strlen(real_base)) == 0;
}

}  // namespace

namespace turtlmap {

LiveWebServer::LiveWebServer()
{
  // Register signal handlers
  g_server_instance = this;
  std::signal(SIGINT, signal_handler);   // Ctrl+C
  std::signal(SIGTERM, signal_handler);  // Termination request
  std::signal(SIGSEGV, signal_handler);  // Segmentation fault
  std::signal(SIGABRT, signal_handler);  // Abort
}

LiveWebServer::~LiveWebServer()
{
  stop();
  g_server_instance = nullptr;
}

bool LiveWebServer::start(int port, const std::string& trajectory_file, const std::string& sensor_file)
{
  if (running_) return true;
  server_fd_ = -1;  // Reset server file descriptor
  running_ = true;
  server_thread_ = std::thread(&LiveWebServer::run, this, port, trajectory_file, sensor_file);
  return true;
}

void LiveWebServer::cleanup()
{
  int fd = server_fd_.load();
  if (fd >= 0) {
    close(fd);
    server_fd_.store(-1);
  }
}

void LiveWebServer::stop()
{
  if (!running_) return;
  running_ = false;
  if (server_thread_.joinable()) {
    server_thread_.join();
  }
  cleanup();
}

void LiveWebServer::run(int port, std::string trajectory_file, std::string sensor_file)
{
  const int MAX_PORT_ATTEMPTS = 10;
  int current_port = port;
  int server_fd = -1;
  bool bind_success = false;

  for (int attempt = 0; attempt < MAX_PORT_ATTEMPTS && !bind_success; attempt++) {
    current_port = port + attempt;

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) continue;

    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
      close(server_fd);
      continue;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(current_port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
      close(server_fd);
      continue;
    }

    bind_success = true;
  }

  if (!bind_success) {
    running_ = false;
    return;
  }

  server_fd_.store(server_fd);

  if (listen(server_fd, 16) < 0) {
    close(server_fd);
    running_ = false;
    return;
  }

  std::cout << "Live web server started on http://localhost:" << current_port << std::endl;
  const std::string web_dir = find_web_dir();
  if (web_dir.empty()) {
    std::cerr << "Error: Web directory not found" << std::endl;
  }

  while (running_) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(server_fd, &fds);
    timeval tv{1, 0};
    int rv = select(server_fd + 1, &fds, nullptr, nullptr, &tv);
    if (rv <= 0) continue;
    int client = accept(server_fd, nullptr, nullptr);
    if (client < 0) continue;

    char buf[4096];
    int n = read(client, buf, sizeof(buf) - 1);
    if (n <= 0) {
      close(client);
      continue;
    }
    buf[n] = 0;

    std::string req(buf);
    std::string path = "/";
    size_t sp1 = req.find(' ');
    if (sp1 != std::string::npos) {
      size_t sp2 = req.find(' ', sp1 + 1);
      if (sp2 != std::string::npos) {
        path = req.substr(sp1 + 1, sp2 - (sp1 + 1));
      }
    }

    if (path == "/") {
      path = "/index.html";
    }

    if (path == "/traj") {
      // Incremental read: only return new data since last read
      // On first call, last_file_size_ is 0, so all data is returned
      // On subsequent calls, only new appended data is returned
      std::string body;
      std::ifstream ifs(trajectory_file, std::ios::binary | std::ios::ate);
      if (ifs.is_open()) {
        size_t current_size = ifs.tellg();
        size_t last_size = last_file_size_.load();

        if (current_size > last_size) {
          // File has grown, read only new data
          size_t new_data_size = current_size - last_size;
          ifs.seekg(last_size);
          body.resize(new_data_size);
          ifs.read(&body[0], new_data_size);
          last_file_size_.store(current_size);
        }
        // else: no new data, body remains empty
        ifs.close();
      }
      std::string resp = http_ok("text/plain; charset=utf-8", body);
      send(client, resp.data(), resp.size(), 0);
    } else if (path == "/sensors") {
      // Return sensor data file (full content each time for simplicity)
      std::string body;
      if (!sensor_file.empty()) {
        std::ifstream ifs(sensor_file);
        if (ifs.is_open()) {
          body = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
          ifs.close();
        }
      }
      std::string resp = http_ok("text/plain; charset=utf-8", body);
      send(client, resp.data(), resp.size(), 0);
    } else if (path == "/cov") {
      // Return covariance data file (full content each time)
      std::string body;
      std::string cov_file = trajectory_file + ".cov";
      std::ifstream ifs(cov_file);
      if (ifs.is_open()) {
        body = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        ifs.close();
      }
      std::string resp = http_ok("text/plain; charset=utf-8", body);
      send(client, resp.data(), resp.size(), 0);
    } else if (!web_dir.empty() && is_safe_path(web_dir, path)) {
      std::string full_path = web_dir + path;
      std::string content = read_file(full_path);
      if (!content.empty()) {
        std::string mime = get_mime_type(full_path);
        std::string resp = http_ok(mime, content);
        send(client, resp.data(), resp.size(), 0);
      } else {
        std::string resp = http_404();
        send(client, resp.data(), resp.size(), 0);
      }
    } else {
      std::string resp = http_404();
      send(client, resp.data(), resp.size(), 0);
    }
    close(client);
  }
  cleanup();  // This will close server_fd
}

}  // namespace turtlmap
