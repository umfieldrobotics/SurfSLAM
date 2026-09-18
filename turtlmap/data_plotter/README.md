# data_plotter

Standalone plotting utility that uses the local `hdf5_streamer` library (no ROS) to read an HDF5 produced by `bag_to_hdf5.py` and render plots via gnuplot.

## Build

Build from the parent directory so CMake can see both the streamer library and this tool:

```bash
cd turtlmap   # from the SurfSLAM repo root
mkdir -p build && cd build
cmake ..
make -j
```

This builds:
- `hdf5_streamer/hdf5_streamer` (CLI streamer)
- `data_plotter/data_plotter` (this plotting tool)

Dependencies:
- C++17, CMake >= 3.10
- HDF5 (dev headers)
- gnuplot (runtime; `sudo apt-get install gnuplot`)

## Run

```bash
./data_plotter/data_plotter --h5 /path/to/output.h5
```

Options:
- `--h5 PATH` (required) the HDF5 file
- `--include g1,g2` optional list of groups to include (default auto-detects: `vectornav_IMU`, `BlueROV_pressure2_fluid`, `dvl_data`)
- `--out plot.png` write plots to a PNG instead of opening a GUI window

Notes:
- The tool creates a 4-panel multiplot: IMU orientation, angular velocity, linear acceleration, and fluid pressure vs. time. DVL can be added similarly; let me know if you want a DVL panel.
- If you want PDF/SVG, change `--out` extension; gnuplot will pick the terminal accordingly if supported.
