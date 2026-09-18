# TurtlMap backend

C++ opti-acoustic-inertial posegraph backend used by SurfSLAM, based on
[TURTLMap](https://github.com/umfieldrobotics/TURTLMap). Exposed to the Python
pipeline as the `pyturtlmap` bindings.

## Building

### In Docker (automatic)
If you use the provided Docker setup (`docker/run.sh` or the devcontainer),
nothing to do: on first launch the container entrypoint runs
`docker/install_pyturtlmap.sh`, which builds this project with
`-DBUILD_PYTHON=ON` (in `turtlmap/build-docker/`, kept separate from any
host-side build) and installs the `pyturtlmap` module into the container
user's `~/.local` site-packages. Later launches skip the build if
`pyturtlmap` already imports. To rebuild after changing the C++ code, run
`FORCE=1 docker/install_pyturtlmap.sh` from the repo root inside the
container.

### Manual build
Dependencies: CMake ≥ 3.10, Eigen3, GTSAM (+ gtsam_unstable), Boost,
yaml-cpp, HDF5, and pybind11 for the Python bindings (all preinstalled in
the Docker image).

```bash
cd turtlmap
mkdir build; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON
make -j
make install   # installs pyturtlmap into ~/.local/lib/python3.X/site-packages
```
`-DBUILD_PYTHON=ON` builds the `pyturtlmap` bindings used by the Python
pipeline (`surfslam`, `examples/run_backend.py`); omit it to build only the
C++ targets. If `CONDA_PREFIX` is set, the bindings install against that
Python.

Verify with:
```bash
python3 -c "import pyturtlmap"
```

Now install three.js to handle web visualization
```bash
cd ../turtlmap/web
npm install
```

This will build the target `turtlmap_backend`. **In the build directory**:
```bash
./turtlmap/turtlmap_backend --help
    Usage: ./turtlmap/turtlmap_backend <config.yaml> <data.h5> [playback_rate] [output_trajectory.txt] [--live-vis=on|off] [--live-file=path] [--web-port=PORT] [--feed=pull|push]
    config.yaml       - Path to configuration file
    data.h5           - Path to HDF5 data file
    playback_rate     - Optional playback rate (default: 1.0, 0 = as fast as possible)
    output_trajectory - Optional output trajectory filename (default: trajectory)
    --live-vis        - Enable/disable live trajectory logging to file (default: on)
    --live-file       - Path to live trajectory file (default: trajectory_live.txt)
    --web-port        - If set (>0), starts a built-in WebGL viewer server on this port
    --feed            - pull: backend reads the HDF5 stream directly (default)
                        push: HDF5 stream is pushed through a LiveFeedDataProvider
```

Sample data: `/path/to/sequence.hdf5`

Sample param file: `cfg/calibration/calib_vi_optimized_surfslam.yaml` (repo root)

i.e.: `./turtlmap/turtlmap_backend ../../cfg/calibration/calib_vi_optimized_surfslam.yaml /path/to/sequence.hdf5 1 trajectory --live-file=trajectory_live.txt --web-port=8080`
