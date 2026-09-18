import h5py
import numpy as np
from pyturtlmap import ImuMeasurement, DvlMeasurement, BarometerMeasurement
from typing import List
from tqdm import tqdm


class Event:
    def __init__(self, stamp, group, index, typ):
        self.stamp = stamp
        self.group = group
        self.index = index
        self.type = typ


class DatasetView:
    """
    Wrapper giving Python-indexable access to a dataset in HDF5.
    Values are constructed lazily from the file on demand.
    """
    def __init__(self, f, group, loader, length):
        self.f = f
        self.group = f[group]
        self.loader = loader  # fn(group, index) -> Measurement
        self._len = length

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.loader(self.group, idx)


class H5LogReader:
    def __init__(self, path, imu_key, dvl_key, baro_key, stereo_key):
        self.path = path
        self.imu_key = imu_key
        self.dvl_key = dvl_key
        self.baro_key = baro_key
        self.stereo_key = stereo_key

        self.f = h5py.File(self.path, "r")   # <--- KEEP OPEN

        self.include_groups = []
        self.events_list = []

        self.imu_view = None
        self.dvl_view = None
        self.baro_view = None
        self.stereo_view = None

        self.have_imu = False
        self.have_dvl = False
        self.have_pressure = False
        self.have_stereo = False

        self._load_metadata()    # uses self.f


    def set_include_groups(self, groups):
        self.include_groups = groups


    ## ---- Lazy measurement generators ------------------------------------

    def _load_imu_one(self, g, i):
        m = ImuMeasurement()
        m.timestamp = float(g["stamp"][i])
        m.qx, m.qy, m.qz, m.qw = g["orientation"][i]
        m.wx, m.wy, m.wz = g["angular_velocity"][i]
        m.ax, m.ay, m.az = g["linear_acceleration"][i]
        return m


    def _load_baro_one(self, g, i):
        m = BarometerMeasurement()
        m.timestamp = float(g["stamp"][i])
        m.pressure = float(g["pressure"][i])
        m.variance = float(g["variance"][i])
        return m


    def _load_dvl_one(self, g, i):
        m = DvlMeasurement()
        if "stamp" in g:
            m.timestamp = float(g["stamp"][i])
        if "time" in g:
            m.time = float(g["time"][i])
        if "velocity" in g:
            m.vx, m.vy, m.vz = g["velocity"][i]

        if "velocity_covariance" in g:
            m.velocity_covariance = g["velocity_covariance"][i].tolist()
        if "fom" in g:
            m.fom = float(g["fom"][i])
        if "altitude" in g:
            m.altitude = float(g["altitude"][i])
        if "velocity_valid" in g:
            m.velocity_valid = int(g["velocity_valid"][i])
        if "status" in g:
            m.status = int(g["status"][i])
        return m


    def _load_stereo_one(self, g, i):
        H = int(g.attrs["height"])
        W = int(g.attrs["width"])
        C = int(g.attrs["channels"])

        stamp = float(g["stamp"][i])
        left = np.asarray(g["left"][i], dtype=np.uint8).reshape(H, W, C)
        right = np.asarray(g["right"][i], dtype=np.uint8).reshape(H, W, C)
        return (stamp, left, right)


    ## ---- Metadata pass (no large loads) ------------------------------------

    def _load_metadata(self):
        f = self.f
        if self.include_groups:
            groups = self.include_groups
        else:
            groups = []
            for k in (self.imu_key, self.dvl_key, self.baro_key, self.stereo_key):
                if k in f:
                    groups.append(k)

        if not groups:
            raise RuntimeError("No known groups found. Use set_include_groups().")

        # Count total events only
        total = 0
        for grp in groups:
            g = f[grp]
            if grp in (self.imu_key, self.baro_key, self.stereo_key) and "stamp" in g:
                total += len(g["stamp"])
            elif grp == self.dvl_key:
                for k in ("stamp", "time", "velocity"):
                    if k in g:
                        total += len(g[k])
                        break

        pbar = tqdm(total=total, desc="Indexing HDF5", dynamic_ncols=True)

        # Build dataset views and event list
        if self.imu_key in groups:
            g = f[self.imu_key]
            n = len(g["stamp"])
            self.imu_view = DatasetView(f, self.imu_key, self._load_imu_one, n)
            self.have_imu = n > 0
            for i in range(n):
                s = float(g["stamp"][i])
                self.events_list.append(Event(s, self.imu_key, i, "IMU"))
                pbar.update(1)

        if self.baro_key in groups:
            g = f[self.baro_key]
            n = len(g["stamp"])
            self.baro_view = DatasetView(f, self.baro_key, self._load_baro_one, n)
            self.have_pressure = n > 0
            for i in range(n):
                s = float(g["stamp"][i])
                self.events_list.append(Event(s, self.baro_key, i, "PRESSURE"))
                pbar.update(1)

        if self.dvl_key in groups:
            g = f[self.dvl_key]
            # choose longest available dataset
            key = None
            for cand in ("stamp", "time", "velocity"):
                if cand in g:
                    key = cand
                    break
            n = len(g[key])
            self.dvl_view = DatasetView(f, self.dvl_key, self._load_dvl_one, n)
            self.have_dvl = n > 0
            stamps = g["stamp"] if "stamp" in g else np.zeros(n)
            for i in range(n):
                s = float(stamps[i])
                self.events_list.append(Event(s, self.dvl_key, i, "DVL"))
                pbar.update(1)

        if self.stereo_key in groups:
            g = f[self.stereo_key]
            n = len(g["stamp"])
            self.stereo_view = DatasetView(f, self.stereo_key, self._load_stereo_one, n)
            self.have_stereo = n > 0
            for i in range(n):
                s = float(g["stamp"][i])
                self.events_list.append(Event(s, self.stereo_key, i, "STEREO"))
                pbar.update(1)

        pbar.close()

        # Sort small event list (cheap)
        self.events_list.sort(key=lambda e: e.stamp)
        print("Done indexing HDF5 (streaming mode)")


    ## ---- Public API: now streaming -----------------------------------------

    def events(self) -> List[Event]:
        return self.events_list

    def imu(self):
        return self.imu_view

    def pressure(self):
        return self.baro_view

    def dvl(self):
        return self.dvl_view

    def stereo(self):
        return self.stereo_view

    def first_stamp(self):
        if not self.events_list:
            return float("nan")
        return self.events_list[0].stamp
