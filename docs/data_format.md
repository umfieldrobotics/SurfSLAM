# Data Format

Each sequence is a single HDF5 file. Each sensor is a top-level group; each group holds parallel datasets indexed by measurement, including a `stamp` dataset of epoch seconds (float64). Measurements across groups are merged and replayed in `stamp` order.

Group names are configurable (`system.h5_keys` in the config). Defaults:

| Sensor    | Group                    |
|-----------|--------------------------|
| IMU       | `vectornav_IMU`          |
| DVL       | `dvl_data`               |
| Barometer | `BlueROV_pressure2_fluid`|
| Stereo    | `zed`                    |

## IMU (`vectornav_IMU`)

| Dataset               | Shape  | Notes                  |
|-----------------------|--------|------------------------|
| `stamp`               | (N,)   | seconds                |
| `orientation`         | (N, 4) | quaternion x, y, z, w  |
| `angular_velocity`    | (N, 3) | rad/s                  |
| `linear_acceleration` | (N, 3) | m/s²                   |

## DVL (`dvl_data`)

Waterlinked A50 fields. Only `stamp` and `velocity` are required; the rest are optional.

| Dataset               | Shape  | Notes                       |
|-----------------------|--------|-----------------------------|
| `stamp`               | (N,)   | seconds                     |
| `velocity`            | (N, 3) | body-frame m/s              |
| `velocity_covariance` | (N, 9) | row-major 3×3               |
| `fom`                 | (N,)   | figure of merit             |
| `altitude`            | (N,)   | meters above bottom         |
| `velocity_valid`      | (N,)   | bool/int                    |
| `status`              | (N,)   | sensor status code          |
| `time`                | (N,)   | sensor-reported time        |

## Barometer (`BlueROV_pressure2_fluid`)

| Dataset    | Shape | Notes   |
|------------|-------|---------|
| `stamp`    | (N,)  | seconds |
| `pressure` | (N,)  | Pa      |
| `variance` | (N,)  |         |

## Stereo (`zed`)

Group attributes: `height`, `width`, `channels` (ints).

| Dataset | Shape                | Notes                             |
|---------|----------------------|-----------------------------------|
| `stamp` | (N,)                 | seconds                           |
| `left`  | (N, H·W·C) uint8     | flattened; reshape to (H, W, C)   |
| `right` | (N, H·W·C) uint8     | flattened; reshape to (H, W, C)   |

## Converting your data

[scripts/bag_to_hdf5.py](../scripts/bag_to_hdf5.py) converts a ROS1 bag (IMU, DVL, barometer topics) to this format; group names are the topic names with `/` replaced by `_`:

```bash
python scripts/bag_to_hdf5.py --bag input.bag --out output.h5 \
    --topics /dvl/data /vectornav/IMU /BlueROV/pressure2_fluid
```

For other sources, write the datasets above with `h5py` and point `system.h5_keys` at your group names. [examples/utils/h5_log_reader.py](../examples/utils/h5_log_reader.py) is the reference reader.