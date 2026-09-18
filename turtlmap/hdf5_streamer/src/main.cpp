#include "hdf5_streamer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct Args {
  std::string h5;
  double rate = 1.0;      // 1.0 = realtime, 2.0 = 2x faster, 0 = as fast as possible
  bool realtime = true;   // if false, publish as fast as possible
  double start = 0.0;     // seconds from min stamp
  double duration = -1.0; // seconds, negative until end
  std::vector<std::string> include_groups; // empty = auto
  std::string format = "jsonl"; // jsonl or csv
};

static void usage(const char* prog){
  std::cerr << "Usage: " << prog << " --h5 path [--rate 1.0] [--realtime 1|0] [--start 0.0] [--duration 10.0] [--include g1,g2] [--format jsonl|csv]\n";
}

static Args parse_args(int argc, char** argv){
  Args a; for (int i=1;i<argc;++i){ std::string k=argv[i]; auto need=[&](int n){ if(i+n>=argc) throw std::runtime_error("Missing value for "+k); };
    if(k=="--h5"){ need(1); a.h5=argv[++i]; }
    else if(k=="--rate"){ need(1); a.rate=std::atof(argv[++i]); }
    else if(k=="--realtime"){ need(1); a.realtime=std::atoi(argv[++i])!=0; }
    else if(k=="--start"){ need(1); a.start=std::atof(argv[++i]); }
    else if(k=="--duration"){ need(1); a.duration=std::atof(argv[++i]); }
    else if(k=="--include"){ need(1); std::string v=argv[++i]; size_t pos=0; while(true){ size_t c=v.find(',',pos); a.include_groups.push_back(v.substr(pos, c==std::string::npos?std::string::npos:c-pos)); if(c==std::string::npos) break; pos=c+1; }}
    else if(k=="--format"){ need(1); a.format=argv[++i]; }
    else { usage(argv[0]); throw std::runtime_error("Unknown argument: "+k); }
  }
  if(a.h5.empty()){ usage(argv[0]); throw std::runtime_error("--h5 is required"); }
  if(a.start<0.0) throw std::runtime_error("--start must be >= 0");
  if(a.duration==0.0) throw std::runtime_error("--duration must be > 0 or omitted");
  return a;
}

static void sleep_until(double target){ using namespace std::chrono; while(true){ double now=duration<double>(steady_clock::now().time_since_epoch()).count(); if(now+0.0005>=target) break; std::this_thread::sleep_for(std::chrono::milliseconds(1)); } }

int main(int argc, char** argv){
  try{
    Args args = parse_args(argc, argv);
    hdf5::Streamer streamer(args.h5);
    if(!args.include_groups.empty()) streamer.setIncludeGroups(args.include_groups);
    streamer.load();
    const auto& events = streamer.events();
    const auto& imu = streamer.imu();
    const auto& pres = streamer.pressure();
    const bool have_imu = !imu.stamps.empty();
    const bool have_pressure = !pres.stamps.empty();
    if(events.empty()){ std::cerr << "No events to stream.\n"; return 0; }

    const double first_stamp = events.front().stamp;
    double start_abs = first_stamp + args.start;
    double end_abs = (args.duration>0) ? (start_abs + args.duration) : std::numeric_limits<double>::infinity();

    std::cout.setf(std::ios::fixed); std::cout << std::setprecision(9);
    const double start_wall = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    const double sim0 = start_abs;

    size_t out_count=0;
    for(const auto& e: events){
      if(e.stamp < start_abs) continue; if(e.stamp > end_abs) break;
      if(args.realtime && args.rate>0.0){ double sim_elapsed=(e.stamp - sim0)/args.rate; double target = start_wall + sim_elapsed; sleep_until(target); }
      const std::string topic = std::string("/") + e.group;
      if(args.format=="csv"){
        if(e.type==hdf5::Event::IMU && have_imu){ size_t i=e.index; std::cout<<topic<<','<<e.stamp<<','<<imu.orientation[i*4+0]<<','<<imu.orientation[i*4+1]<<','<<imu.orientation[i*4+2]<<','<<imu.orientation[i*4+3]
          <<','<<imu.angular_velocity[i*3+0]<<','<<imu.angular_velocity[i*3+1]<<','<<imu.angular_velocity[i*3+2]
          <<','<<imu.linear_acceleration[i*3+0]<<','<<imu.linear_acceleration[i*3+1]<<','<<imu.linear_acceleration[i*3+2]<<'\n'; }
        else if(e.type==hdf5::Event::PRESSURE && have_pressure){ size_t i=e.index; std::cout<<topic<<','<<e.stamp<<','<<pres.pressure[i]<<','<<pres.variance[i]<<'\n'; }
      } else {
        if(e.type==hdf5::Event::IMU && have_imu){ size_t i=e.index; std::cout<<"{\"topic\":\""<<topic<<"\",\"stamp\":"<<e.stamp<<",\"data\":{\"orientation\":["<<imu.orientation[i*4+0]<<','<<imu.orientation[i*4+1]<<','<<imu.orientation[i*4+2]<<','<<imu.orientation[i*4+3]
          <<"],\"angular_velocity\":["<<imu.angular_velocity[i*3+0]<<','<<imu.angular_velocity[i*3+1]<<','<<imu.angular_velocity[i*3+2]
          <<"],\"linear_acceleration\":["<<imu.linear_acceleration[i*3+0]<<','<<imu.linear_acceleration[i*3+1]<<','<<imu.linear_acceleration[i*3+2]<<"]}}\n"; }
        else if(e.type==hdf5::Event::PRESSURE && have_pressure){ size_t i=e.index; std::cout<<"{\"topic\":\""<<topic<<"\",\"stamp\":"<<e.stamp<<",\"data\":{\"pressure\":"<<pres.pressure[i]<<",\"variance\":"<<pres.variance[i]<<"}}\n"; }
      }
      std::cout.flush(); ++out_count;
    }
    std::cerr << "Streamed "<< out_count << " messages." << std::endl; return 0;
  } catch(const std::exception& e){ std::cerr << "Error: "<< e.what() << std::endl; return 1; }
}
