# hdf5_streamer

Standalone C++ streamer that replays an HDF5 file produced by `bag_to_hdf5.py` as a time-ordered stream on stdout, without any ROS dependency.

- Input: HDF5 with groups like `vectornav_IMU`, `BlueROV_pressure2_fluid`, `dvl_data`, containing datasets such as `stamp`, `orientation`, `angular_velocity`, etc.
- Output: newline-delimited JSON (default) or CSV lines to stdout, in timestamp order, with optional real-time pacing to mimic rosbag playback.

## Build

Dependencies:
- A C++17 compiler
- CMake >= 3.10
- HDF5 development libraries (Ubuntu/Debian: `sudo apt-get install libhdf5-dev`)

Commands (from this folder):

```bash
mkdir -p build && cd build
cmake ..
make -j
```

This produces the `hdf5_streamer` executable in `build/`.

## Run

Basic usage:

```bash
./hdf5_streamer --h5 /path/to/output.h5
```

Options:
- `--h5 PATH` (required) path to the `.h5` file
- `--rate R` playback rate multiplier (default 1.0). Use 2.0 for 2x faster; `<=0` or `--realtime 0` streams as fast as possible.
- `--realtime 1|0` enable real-time pacing (default 1). If 0, no sleeping is performed.
- `--start S` start offset in seconds from the earliest timestamp in the file (default 0.0)
- `--duration D` duration in seconds to stream (default: until end)
- `--include g1,g2` only include specific HDF5 groups (by name). If omitted, the tool auto-detects known groups: `vectornav_IMU`, `BlueROV_pressure2_fluid`, `dvl_data` if present.
- `--format jsonl|csv` output format (default `jsonl`)

Examples:

- Stream IMU and pressure in real-time JSONL:
```bash
./hdf5_streamer --h5 output.h5
```

- Stream 10s starting at t=30s, 2x faster, only IMU:
```bash
./hdf5_streamer --h5 output.h5 --start 30 --duration 10 --rate 2 --include vectornav_IMU
```

- CSV output, as fast as possible:
```bash
./hdf5_streamer --h5 output.h5 --format csv --realtime 0
```

## Output format

- JSONL (one JSON object per line):
```json
{"topic":"/vectornav/IMU","stamp":1697654321.123456789,"data":{"orientation":[qx,qy,qz,qw],"angular_velocity":[wx,wy,wz],"linear_acceleration":[ax,ay,az]}}
{"topic":"/BlueROV/pressure2_fluid","stamp":1697654321.234,"data":{"pressure":101325.0,"variance":0.1}}
```

- CSV:
```
/vectornav/IMU,1697654321.123456789,qx,qy,qz,qw,wx,wy,wz,ax,ay,az
/BlueROV/pressure2_fluid,1697654321.234,pressure,variance
```

DVL (if `dvl_data` group is present) prints velocity components and optional fields when available.

## Notes

- Topic labels: this tool uses `"/" + group` as the topic label to avoid ambiguity (underscores in original names are preserved). Example: `vectornav_IMU` -> `/vectornav_IMU`.
- The HDF5 must contain a `stamp` dataset (double seconds). Datasets are read as `double` for simplicity.
- For large files, this tool loads datasets into memory before streaming for simplicity and speed. If you need true streaming from disk in constant memory, open an issue and we can switch to chunked reads.
