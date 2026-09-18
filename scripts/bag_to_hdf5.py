#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Serialize selected ROS topics from a .bag file into an HDF5 file.

Each topic is stored as a group, with datasets for timestamps and message fields.

Example:
    python bag_to_hdf5.py --bag input.bag --out output.h5 \
        --topics /dvl/data /vectornav/IMU /BlueROV/pressure2_fluid
"""

import rosbag
import h5py
import numpy as np
import argparse
from tqdm import tqdm


def write_dataset(group, name, data):
    """Write or extend a dataset in HDF5."""
    data = np.array(data)
    if name in group:
        dset = group[name]
        dset.resize((dset.shape[0] + len(data)), axis=0)
        dset[-len(data):] = data
    else:
        maxshape = (None,) + data.shape[1:]
        group.create_dataset(name, data=data, maxshape=maxshape, chunks=True)


def parse_dvl_msg(msg):
    """Extract fields from waterlinked_a50_ros_driver/DVL."""
    return dict(
        time=msg.time,
        velocity=[msg.velocity.x, msg.velocity.y, msg.velocity.z],
        fom=msg.fom,
        altitude=msg.altitude,
        velocity_valid=msg.velocity_valid,
        status=msg.status,
        velocity_covariance=list(msg.velocity_covariance),
        stamp=msg.header.stamp.to_sec(),
    )


def parse_imu_msg(msg):
    """Extract fields from sensor_msgs/Imu."""
    ori = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    ang = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
    lin = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
    return dict(
        orientation=ori,
        angular_velocity=ang,
        linear_acceleration=lin,
        stamp=msg.header.stamp.to_sec(),
    )


def parse_pressure_msg(msg):
    """Extract fields from sensor_msgs/FluidPressure."""
    return dict(
        pressure=msg.fluid_pressure,
        variance=msg.variance,
        stamp=msg.header.stamp.to_sec(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Input ROS bag file")
    parser.add_argument("--out", required=True, help="Output HDF5 file")
    parser.add_argument("--topics", nargs="+", default=["/dvl/data", "/vectornav/IMU", "/BlueROV/pressure2_fluid"], help="Topics to serialize")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start offset in seconds from bag start (default 0)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration in seconds to extract (default until end)")
    args = parser.parse_args()

    parsers = {
        "/dvl/data": parse_dvl_msg,
        "/vectornav/IMU": parse_imu_msg,
        "/BlueROV/pressure2_fluid": parse_pressure_msg,
    }
    print("[INFO]")
    with rosbag.Bag(args.bag, "r") as bag, h5py.File(args.out, "w") as h5f:
        # compute absolute start/end times (seconds since epoch) based on bag start
        bag_start = bag.get_start_time()
        if args.start < 0:
            raise ValueError("--start must be non-negative")
        start_abs = bag_start + args.start
        if args.duration is not None:
            if args.duration <= 0:
                raise ValueError("--duration must be positive")
            end_abs = start_abs + args.duration
        else:
            # until end of bag
            try:
                end_abs = bag.get_end_time()
            except Exception:
                # fallback: use a very large end (treat as 'until end')
                end_abs = float("inf")

        for topic in args.topics:
            if topic not in parsers:
                print(f"[WARN] No parser defined for {topic}")
                continue

            parser_fn = parsers[topic]
            records = []

            print(f"[INFO] Processing topic: {topic} (window {start_abs:.3f} -> {end_abs:.3f})")
            # iterate messages and keep only those in the requested time window
            for entry in tqdm(bag.read_messages(topics=[topic])):
                # rosbag.read_messages sometimes returns (topic,msg,t) or
                # (topic,msg,t,connection_header). Handle both.
                try:
                    tpc, msg, ros_time = entry
                except ValueError:
                    tpc, msg, ros_time, _ = entry
                t_sec = ros_time.to_sec()
                if t_sec < start_abs:
                    continue
                # if duration was provided and we've passed the end, stop scanning this topic
                if args.duration is not None and t_sec > end_abs:
                    break

                parsed = parser_fn(msg)
                records.append(parsed)

            if not records:
                print(f"[WARN] No messages for {topic} in the requested window; skipping.")
                continue

            # create group and stack numeric data per field
            group = h5f.create_group(topic.replace("/", "_")[1:])
            keys = records[0].keys()
            for k in keys:
                data = np.array([r[k] for r in records])
                group.create_dataset(k, data=data)

    # Try to open the produced HDF5 file and print an informative summary
    try:
        with h5py.File(args.out, "r") as out_h5:
            groups = list(out_h5.keys())
            print(f"[DONE] Saved serialized topics to {args.out}")
            if not groups:
                print("[SUMMARY] HDF5 file is empty (no topic groups found)")
            else:
                print("[SUMMARY] HDF5 contents:")
                for g in groups:
                    print(f"  {g}/")
                    grp = out_h5[g]
                    for name, dset in grp.items():
                        try:
                            shape = dset.shape
                            dtype = dset.dtype
                        except Exception:
                            # not a dataset
                            continue
                        print(f"    - {name}: shape={shape}, dtype={dtype}")
    except Exception as e:
        # fallback to original message if we can't open/read the file
        print(f"[DONE] Saved serialized topics to {args.out} (could not summarize: {e})")


if __name__ == "__main__":
    main()
