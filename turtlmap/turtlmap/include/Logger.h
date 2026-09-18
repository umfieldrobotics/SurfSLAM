#pragma once
#include <algorithm>
#include <cstdlib>
#include <ostream>
#include <string>

enum class LogLevel
{
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  FATAL = 5,
};

inline LogLevel parseLogLevel(const char* s)
{
  if (!s) return LogLevel::INFO;

  std::string v(s);
  std::transform(v.begin(), v.end(), v.begin(), ::toupper);

  if (v == "TRACE") return LogLevel::TRACE;
  if (v == "DEBUG") return LogLevel::DEBUG;
  if (v == "INFO") return LogLevel::INFO;
  if (v == "WARN") return LogLevel::WARN;
  if (v == "ERROR") return LogLevel::ERROR;
  if (v == "FATAL") return LogLevel::FATAL;

  return LogLevel::INFO;
}

class LogStream
{
 public:
  LogStream(std::ostream& out, LogLevel current, LogLevel threshold)
      : out_((int)current >= (int)threshold ? &out : nullptr)
  {}

  template <typename T>
  LogStream& operator<<(const T& v)
  {
    if (out_) (*out_) << v;
    return *this;
  }

  LogStream& operator<<(std::ostream& (*manip)(std::ostream&))
  {
    if (out_) (*out_) << manip;
    return *this;
  }

 private:
  std::ostream* out_;
};

class Logger
{
 public:
  Logger() : threshold_(parseLogLevel(std::getenv("TURTLMAP_LOG_LEVEL"))) {}

  LogStream trace() { return LogStream(std::cout, LogLevel::TRACE, threshold_); }
  LogStream debug() { return LogStream(std::cout, LogLevel::DEBUG, threshold_); }
  LogStream info() { return LogStream(std::cout, LogLevel::INFO, threshold_); }
  LogStream warn() { return LogStream(std::cout, LogLevel::WARN, threshold_); }
  LogStream error() { return LogStream(std::cout, LogLevel::ERROR, threshold_); }
  LogStream fatal() { return LogStream(std::cout, LogLevel::FATAL, threshold_); }

 private:
  LogLevel threshold_;
};
