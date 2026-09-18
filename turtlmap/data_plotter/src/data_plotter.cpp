#include "hdf5_streamer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void usage(const char* prog){
  std::cerr << "Usage: " << prog << " --h5 path [--include g1,g2] [--out png_path]\n";
}

static void write_inline_series(std::ostringstream& ss, const std::vector<double>& t, const std::vector<double>& y){
  const size_t n = std::min(t.size(), y.size());
  for(size_t i=0;i<n;++i){ ss << std::fixed << std::setprecision(9) << t[i] << ' ' << y[i] << '\n'; }
  ss << "e\n";
}

static bool run_gnuplot(const std::string& script){
  FILE* gp = popen("gnuplot -persist", "w");
  if(!gp){ std::perror("gnuplot"); return false; }
  fwrite(script.data(), 1, script.size(), gp);
  int rc = pclose(gp);
  return rc == 0;
}

int main(int argc, char** argv){
  try{
    std::string h5; std::vector<std::string> includes; std::string outpng;
    for(int i=1;i<argc;++i){ std::string k=argv[i]; auto need=[&](int n){ if(i+n>=argc) throw std::runtime_error("Missing value for "+k); };
      if(k=="--h5"){ need(1); h5=argv[++i]; }
      else if(k=="--include"){ need(1); std::string v=argv[++i]; size_t pos=0; while(true){ size_t c=v.find(',',pos); includes.push_back(v.substr(pos, c==std::string::npos?std::string::npos:c-pos)); if(c==std::string::npos) break; pos=c+1; } }
      else if(k=="--out"){ need(1); outpng=argv[++i]; }
      else { usage(argv[0]); return 1; }
    }
    if(h5.empty()){ usage(argv[0]); return 1; }

    hdf5::Streamer s(h5);
    if(!includes.empty()) s.setIncludeGroups(includes);
    s.load();

    const auto& imu = s.imu();
    const auto& pr = s.pressure();

    // Prepare gnuplot script
    std::ostringstream gp;
    gp << "set key outside\n";
    if(!outpng.empty()) gp << "set terminal pngcairo size 1600,900\nset output '" << outpng << "'\n";
    gp << "set multiplot layout 4,1 title 'hdf5 Measurements' margins 0.08,0.98,0.08,0.95 spacing 0.07,0.05\n";

    // Panel 1: IMU Orientation (qx,qy,qz,qw)
    if(!imu.stamps.empty() && imu.orientation.size()>=4){
      gp << "set title 'IMU Orientation'\nset xlabel 't (s)'\nset ylabel 'quat'\nplot '-' using 1:2 with lines title 'qx', '-' using 1:2 with lines title 'qy', '-' using 1:2 with lines title 'qz', '-' using 1:2 with lines title 'qw'\n";
      // Build series
      std::vector<double> qx, qy, qz, qw; qx.reserve(imu.stamps.size()); qy.reserve(imu.stamps.size()); qz.reserve(imu.stamps.size()); qw.reserve(imu.stamps.size());
      for(size_t i=0;i<imu.stamps.size();++i){ qx.push_back(imu.orientation[i*4+0]); qy.push_back(imu.orientation[i*4+1]); qz.push_back(imu.orientation[i*4+2]); qw.push_back(imu.orientation[i*4+3]); }
      write_inline_series(gp, imu.stamps, qx);
      write_inline_series(gp, imu.stamps, qy);
      write_inline_series(gp, imu.stamps, qz);
      write_inline_series(gp, imu.stamps, qw);
    } else {
      gp << "set title 'IMU Orientation (no data)'\nplot NaN notitle\n";
    }

    // Panel 2: IMU Angular Velocity (wx,wy,wz)
    if(!imu.stamps.empty() && imu.angular_velocity.size()>=3){
      gp << "set title 'IMU Angular Velocity'\nset xlabel 't (s)'\nset ylabel 'rad/s'\nplot '-' using 1:2 with lines title 'wx', '-' using 1:2 with lines title 'wy', '-' using 1:2 with lines title 'wz'\n";
      std::vector<double> wx, wy, wz; wx.reserve(imu.stamps.size()); wy.reserve(imu.stamps.size()); wz.reserve(imu.stamps.size());
      for(size_t i=0;i<imu.stamps.size();++i){ wx.push_back(imu.angular_velocity[i*3+0]); wy.push_back(imu.angular_velocity[i*3+1]); wz.push_back(imu.angular_velocity[i*3+2]); }
      write_inline_series(gp, imu.stamps, wx);
      write_inline_series(gp, imu.stamps, wy);
      write_inline_series(gp, imu.stamps, wz);
    } else {
      gp << "set title 'IMU Angular Velocity (no data)'\nplot NaN notitle\n";
    }

    // Panel 3: IMU Linear Acceleration (ax,ay,az)
    if(!imu.stamps.empty() && imu.linear_acceleration.size()>=3){
      gp << "set title 'IMU Linear Acceleration'\nset xlabel 't (s)'\nset ylabel 'm/s^2'\nplot '-' using 1:2 with lines title 'ax', '-' using 1:2 with lines title 'ay', '-' using 1:2 with lines title 'az'\n";
      std::vector<double> ax, ay, az; ax.reserve(imu.stamps.size()); ay.reserve(imu.stamps.size()); az.reserve(imu.stamps.size());
      for(size_t i=0;i<imu.stamps.size();++i){ ax.push_back(imu.linear_acceleration[i*3+0]); ay.push_back(imu.linear_acceleration[i*3+1]); az.push_back(imu.linear_acceleration[i*3+2]); }
      write_inline_series(gp, imu.stamps, ax);
      write_inline_series(gp, imu.stamps, ay);
      write_inline_series(gp, imu.stamps, az);
    } else {
      gp << "set title 'IMU Linear Acceleration (no data)'\nplot NaN notitle\n";
    }

    // Panel 4: Pressure
    if(!pr.stamps.empty()){
      gp << "set title 'Fluid Pressure'\nset xlabel 't (s)'\nset ylabel 'Pa'\nplot '-' using 1:2 with lines title 'pressure'\n";
      write_inline_series(gp, pr.stamps, pr.pressure);
    } else {
      gp << "set title 'Fluid Pressure (no data)'\nplot NaN notitle\n";
    }

    gp << "unset multiplot\n";

    if(!run_gnuplot(gp.str())){
      std::cerr << "Failed to run gnuplot. Ensure it is installed (e.g., apt-get install gnuplot).\n";
      return 2;
    }
    if(!outpng.empty()) std::cerr << "Wrote plot: " << outpng << std::endl;
    return 0;
  } catch(const std::exception& e){
    std::cerr << "Error: " << e.what() << std::endl; return 1;
  }
}
