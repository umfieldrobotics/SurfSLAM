#!/usr/bin/env python3
"""
Extract raw (unrectified) and rectified frames from a ZED .svo file between two
timestamps **expressed since the Unix epoch**.

Example (epoch seconds):
    python extract_svo_frames.py \
        --svo   /data/recording.svo \
        --out   /data/extracted_frames \
        --start 1718637600.0 \
        --end   1718637630.0

Example (epoch nanoseconds):
    python extract_svo_frames.py \
        --start-ns 1718637600000000000 \
        --end-ns   1718637630000000000 \
        --svo my.svo --out frames
"""
import os
import cv2
import time
import argparse
from datetime import datetime, timezone
import pyzed.sl as sl


# ───────────────────────────── Helpers ──────────────────────────────
def mkdir_p(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_image(mat: sl.Mat, path: str) -> None:
    img = mat.get_data()
    if img.shape[-1] == 4:          # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:                           # RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def to_ns(value: float | int) -> int:
    """Accept seconds (float / int) or nanoseconds (int ≥ 1e12) → int [nanoseconds]."""
    value = float(value)
    return int(value if value > 1e12 else value * 1e9)


def fmt_dt(ns: int) -> str:
    """Readable UTC time string for logging/debug."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat(timespec="microseconds")


# ───────────────────────────── Main ──────────────────────────────
def main(args: argparse.Namespace) -> None:
    # ----------- normalise input timestamps to nanoseconds -----------
    start_ns = to_ns(args.start_ns if args.start_ns is not None else args.start)
    end_ns   = to_ns(args.end_ns   if args.end_ns   is not None else args.end)

    if end_ns is not None and end_ns < start_ns:
        raise ValueError("end timestamp must be ≥ start timestamp")

    print(f"[INFO] extracting from {fmt_dt(start_ns)} → {fmt_dt(end_ns) if end_ns else 'EOF'}")

    # ----------- open SVO -----------
    ipt = sl.InputType(); ipt.set_from_svo_file(args.svo)
    init = sl.InitParameters(input_t=ipt,
                             svo_real_time_mode=False,
                             depth_mode=sl.DEPTH_MODE.NONE)
    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Could not open SVO")

    fps        = cam.get_camera_information().camera_configuration.fps
    total_frms = cam.get_svo_number_of_frames()
    frame_ns   = int(1e9 / fps)            # nominal frame period in ns

    # ----------- find start frame (coarse estimate then refine) -----------
    cam.set_svo_position(0)
    cam.grab()                             # first frame
    first_ts = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

    est_start_idx = max(0, int((start_ns - first_ts) / frame_ns))
    if est_start_idx >= total_frms:
        raise ValueError("start timestamp is beyond the end of the SVO")

    cam.set_svo_position(est_start_idx)

    # ----------- prepare output -----------
    subdirs = ["raw_left", "raw_right", "rect_left", "rect_right"]
    for s in subdirs:
        mkdir_p(os.path.join(args.out, s))

    raw_l, raw_r, rect_l, rect_r = sl.Mat(), sl.Mat(), sl.Mat(), sl.Mat()
    frames_written = 0

    # ----------- extraction loop -----------
    while cam.grab() == sl.ERROR_CODE.SUCCESS:
        ts_ns = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

        if ts_ns < start_ns:
            continue            # not yet inside the interval
        if end_ns is not None and ts_ns > end_ns:
            break               # finished

        # retrieve & save
        cam.retrieve_image(raw_l,  sl.VIEW.LEFT_UNRECTIFIED)
        cam.retrieve_image(raw_r,  sl.VIEW.RIGHT_UNRECTIFIED)
        cam.retrieve_image(rect_l, sl.VIEW.LEFT)          # default = rectified
        cam.retrieve_image(rect_r, sl.VIEW.RIGHT)

        stem = f"{ts_ns}"
        save_image(raw_l,  f"{args.out}/raw_left/{stem}.png")
        save_image(raw_r,  f"{args.out}/raw_right/{stem}.png")
        save_image(rect_l, f"{args.out}/rect_left/{stem}.png")
        save_image(rect_r, f"{args.out}/rect_right/{stem}.png")
        frames_written += 1

    cam.close()
    print(f"[INFO] wrote {frames_written} frame pairs to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract raw & rectified ZED frames by epoch timestamp")
    p.add_argument("--svo",  required=True, help="Path to input .svo file")
    p.add_argument("--out",  required=True, help="Output directory")
    # epoch seconds (float) – convenience flags kept for backward compatibility
    p.add_argument("--start",   type=float, default=None,
                   help="Start time in **epoch seconds** (float).")
    p.add_argument("--end",     type=float, default=None,
                   help="End time in **epoch seconds** (float). Omit to go to EOF.")
    # explicit nanoseconds
    p.add_argument("--start-ns", type=int, default=None,
                   help="Start time in epoch nanoseconds (overrides --start).")
    p.add_argument("--end-ns",   type=int, default=None,
                   help="End time in epoch nanoseconds (overrides --end).")
    main(p.parse_args())
