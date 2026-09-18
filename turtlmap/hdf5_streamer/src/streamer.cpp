#include "hdf5_streamer.hpp"

#include <H5Cpp.h>
#include <hdf5.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <limits>

namespace hdf5 {

static bool has_group(hid_t file_id, const std::string &name)
{
    H5O_info_t info;
    herr_t st = H5Oget_info_by_name(file_id, name.c_str(), &info, H5P_DEFAULT);
    return st >= 0 && info.type == H5O_TYPE_GROUP;
}

static void read1d_double(const H5::Group &g, const std::string &name, std::vector<double> &out)
{
    H5::DataSet d = g.openDataSet(name);
    H5::DataSpace s = d.getSpace();
    if (s.getSimpleExtentNdims() != 1)
        throw std::runtime_error("Dataset '" + name + "' not 1D");
    hsize_t dims[1];
    s.getSimpleExtentDims(dims);
    out.resize(static_cast<size_t>(dims[0]));
    d.read(out.data(), H5::PredType::NATIVE_DOUBLE);
}

static void read2d_double(const H5::Group &g, const std::string &name, size_t cols, std::vector<double> &out)
{
    H5::DataSet d = g.openDataSet(name);
    H5::DataSpace s = d.getSpace();
    if (s.getSimpleExtentNdims() != 2)
        throw std::runtime_error("Dataset '" + name + "' not 2D");
    hsize_t dims[2];
    s.getSimpleExtentDims(dims);
    if (static_cast<size_t>(dims[1]) != cols)
        throw std::runtime_error("Dataset '" + name + "' wrong cols");
    const size_t rows = static_cast<size_t>(dims[0]);
    out.resize(rows * cols);
    d.read(out.data(), H5::PredType::NATIVE_DOUBLE);
}

struct Streamer::Impl {
    std::string path;
    std::vector<std::string> include_groups;

    std::vector<Event> evs;
    ImuData imu_data;
    PressureData pres_data;
    DvlData dvl_data;
    StereoData stereo_data;
    
    bool have_imu = false, have_pressure = false, have_dvl = false;

    explicit Impl(std::string p) : path(std::move(p)) {}

    static ImuData load_imu(const H5::H5File &f, const std::string &group)
    {
        ImuData imu; H5::Group g = f.openGroup(group);
        read2d_double(g, "orientation", 4, imu.orientation);
        read2d_double(g, "angular_velocity", 3, imu.angular_velocity);
        read2d_double(g, "linear_acceleration", 3, imu.linear_acceleration);
        read1d_double(g, "stamp", imu.stamps);
        const size_t n = imu.stamps.size();
        if (imu.orientation.size() != n * 4 || imu.angular_velocity.size() != n * 3 || imu.linear_acceleration.size() != n * 3)
            throw std::runtime_error("IMU dataset sizes inconsistent");
        return imu;
    }

    static PressureData load_pressure(const H5::H5File &f, const std::string &group)
    {
        PressureData p; H5::Group g = f.openGroup(group);
        read1d_double(g, "stamp", p.stamps);
        read1d_double(g, "pressure", p.pressure);
        read1d_double(g, "variance", p.variance);
        if (p.pressure.size() != p.stamps.size() || p.variance.size() != p.stamps.size())
            throw std::runtime_error("Pressure dataset sizes inconsistent");
        return p;
    }

    static DvlData load_dvl(const H5::H5File &f, const std::string &group)
    {
        DvlData dvl; H5::Group g = f.openGroup(group);
        try { read1d_double(g, "stamp", dvl.stamps); } catch (...) {}
        try { read1d_double(g, "time", dvl.time); } catch (...) {}
        try { read2d_double(g, "velocity", 3, dvl.velocity); } catch (...) {}
        try { read2d_double(g, "velocity_covariance", 9, dvl.velocity_covariance); } catch (...) {}
        try { read1d_double(g, "fom", dvl.fom); } catch (...) {}
        try { read1d_double(g, "altitude", dvl.altitude); } catch (...) {}
        try { read1d_double(g, "velocity_valid", dvl.velocity_valid); } catch (...) {}
        try { read1d_double(g, "status", dvl.status); } catch (...) {}
        return dvl;
    }

    void load() {
        H5::H5File f(path, H5F_ACC_RDONLY);
        hid_t fid = f.getId();

        std::vector<std::string> groups;
        if (!include_groups.empty()) {
            groups = include_groups;
        } else {
            if (has_group(fid, "vectornav_IMU")) groups.push_back("vectornav_IMU");
            if (has_group(fid, "BlueROV_pressure2_fluid")) groups.push_back("BlueROV_pressure2_fluid");
            if (has_group(fid, "dvl_data")) groups.push_back("dvl_data");
        }
        if (groups.empty()) {
            throw std::runtime_error("No known groups found. Use setIncludeGroups().");
        }

        for (const auto& gname : groups) {
            if (gname == "vectornav_IMU") { imu_data = load_imu(f, gname); have_imu = !imu_data.stamps.empty(); }
            else if (gname == "BlueROV_pressure2_fluid") { pres_data = load_pressure(f, gname); have_pressure = !pres_data.stamps.empty(); }
            else if (gname == "dvl_data") { dvl_data = load_dvl(f, gname); have_dvl = !dvl_data.stamps.empty(); }
        }

        evs.clear();
        auto add_events = [&](const std::vector<double>& stamps, Event::Type t, const std::string& grp){
            for (size_t i = 0; i < stamps.size(); ++i) evs.push_back({t, grp, i, stamps[i]});
        };
        if (have_imu) add_events(imu_data.stamps, Event::IMU, "vectornav_IMU");
        if (have_pressure) add_events(pres_data.stamps, Event::PRESSURE, "BlueROV_pressure2_fluid");
        if (have_dvl) add_events(dvl_data.stamps, Event::DVL, "dvl_data");
        std::sort(evs.begin(), evs.end(), [](const Event& a, const Event& b){ return a.stamp < b.stamp; });
    }
};

Streamer::Streamer(const std::string& h5_path) : pimpl_(new Impl(h5_path)) {}
Streamer::~Streamer() { delete pimpl_; }

void Streamer::setIncludeGroups(const std::vector<std::string>& groups) { pimpl_->include_groups = groups; }
void Streamer::load() { pimpl_->load(); }
const std::vector<Event>& Streamer::events() const { return pimpl_->evs; }
const ImuData& Streamer::imu() const { return pimpl_->imu_data; }
const PressureData& Streamer::pressure() const { return pimpl_->pres_data; }
const DvlData& Streamer::dvl() const { return pimpl_->dvl_data; }
const StereoData& Streamer::stereo() const { return pimpl_->stereo_data; }

double Streamer::firstStamp() const { return pimpl_->evs.empty() ? std::numeric_limits<double>::quiet_NaN() : pimpl_->evs.front().stamp; }

} // namespace hdf5
