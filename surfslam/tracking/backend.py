import time
import pyturtlmap as tm
import yaml

import torch
from common.pose import Pose
from common.sensors import BackendData
from pyturtlmap import ImuMeasurement, DvlMeasurement, BarometerMeasurement, StereoMeasurement
import numpy as np
from typing import Union, Optional
from pathlib import Path


# The C++ backend serves poses from a bounded sliding window (`traj_pose_queue_`).
# A query that has aged out of that window can never be satisfied, so waiting on it
# is an infinite loop; a query that is merely not optimized yet resolves in ms.
_POSE_QUERY_TIMEOUT_SEC = 2.0
_POSE_QUERY_POLL_SEC = 0.01
_POSE_QUERY_SUMMARY_SEC = 10.0


class Backend:
    """Wrapper around the TurtlMap C++ posegraph backend (`pyturtlmap` bindings).

    Fuses IMU, DVL, and barometer measurements, and accepts stereo-driven
    keyframe triggers and relative-pose constraints from the tracker.
    """

    @staticmethod
    def _get_dvl_enabled(config_path: str) -> bool:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            sensor_list = cfg.get("sensor_list", {})
            return bool(sensor_list.get("isDvlUsed", True))
        except Exception as exc:
            print(
                f"Warning: could not determine DVL enablement from {config_path}: {exc}. "
                "Defaulting to DVL enabled."
            )
            return True

    @staticmethod
    def _get_barometer_enabled(config_path: str) -> bool:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            sensor_list = cfg.get("sensor_list", {})
            return bool(sensor_list.get("isBaroUsed", True))
        except Exception as exc:
            print(
                f"Warning: could not determine barometer enablement from {config_path}: {exc}. "
                "Defaulting to barometer enabled."
            )
            return True

    def __init__(
        self,
        config_file: Union[str, Path],
        live_log_file: Optional[str] = None,
        traj_pose_queue_max_size: Optional[int] = None,
        pose_query_timeout_sec: Optional[float] = None,
    ) -> None:
        self._data_provider = tm.LiveFeedDataProvider()
        config_path: str = (
            config_file.as_posix() if isinstance(config_file, Path) else config_file
        )
        self._dvl_enabled = self._get_dvl_enabled(config_path)
        self._barometer_enabled = self._get_barometer_enabled(config_path)
        self._pg_backend = tm.PoseGraphBackend(config_path, self._data_provider)
        if traj_pose_queue_max_size is not None:
            self._pg_backend.set_traj_pose_queue_max_size(int(traj_pose_queue_max_size))
            print(
                "[Backend] trajectory pose window capped at "
                f"{int(traj_pose_queue_max_size)} entries"
            )
        if live_log_file:
            self._pg_backend.configure_live_visualization(True, live_log_file)
        else:
            self._pg_backend.configure_live_visualization(False)

        self._pose_query_timeout_sec = (
            _POSE_QUERY_TIMEOUT_SEC
            if pose_query_timeout_sec is None
            else float(pose_query_timeout_sec)
        )
        # Counters for the rate-limited query-failure census (see _log_query_stats).
        self._query_stale_count = 0
        self._query_timeout_count = 0
        self._query_stale_total = 0
        self._query_timeout_total = 0
        self._query_max_staleness = 0.0
        self._last_query_summary_time = time.time()

        self._graph_initialized = False
        self._first_keyframe_created = False

        # Map keyframe IDs to timestamps
        self._keyframe_id_to_timestamp = {}
        self._next_keyframe_id = 0  # x0 is id 0, first created keyframe is id 1

    def start(self) -> None:
        self._pg_backend.run()
        self._data_provider.start()

    def is_graph_initialized(self) -> bool:
        """Check if the graph is ready for stereo processing"""
        if not self._graph_initialized:
            self._graph_initialized = self._pg_backend.is_graph_initialized()
        return self._graph_initialized

    def get_last_kf_timestamp(self) -> float:
        return self._pg_backend.get_latest_kf_timestamp()

    def process(self, data: BackendData) -> None:
        if isinstance(data, ImuMeasurement):
            self._data_provider.insert_imu(data)
        elif isinstance(data, DvlMeasurement):
            if not self._dvl_enabled:
                return
            self._data_provider.insert_dvl(data)
        elif isinstance(data, BarometerMeasurement):
            self._data_provider.insert_baro(data)
        else:
            raise ValueError(
                "Unknown Data Type. Expected ImuMeasurement, DvlMeasurement, or BarometerMeasurement. Got:",
                data,
            )

    def get_latest_pose(self):
        print("Getting latest pose from backend...")
        return self._pg_backend.get_latest_pose()

    def get_updated_graph(self):
        return self._pg_backend.get_updated_graph()

    def compute_mahalanobis_distance(
        self,
        T_world_a: Pose,
        T_world_b: Pose,
        T_a_b_est: Pose,
        T_world_b_cov: torch.Tensor,
        min_diag_cov=1e-8,
    ) -> int | float | bool:
        """Compute mahalanobis_distance

        Args:
            T_world_a (Pose): pose at kf a
            T_world_b (Pose): pose at kf b
            T_a_b_est (Pose): estimated transformation from a to b
            T_world_b_cov (torch.Tensor): covariance of pose at kf b
            min_diag_cov (_type_, optional): _description_. Defaults to 1e-8.

        Returns:
            int | float | bool: _description_
        """
        T_world_b_est: Pose = T_world_a * T_a_b_est
        delta_pose: Pose = T_world_b_est.inv() * T_world_b
        error: torch.Tensor = delta_pose.get_pose_tensor()
        cov_inv: torch.Tensor = (T_world_b_cov + torch.eye(6) * min_diag_cov).inverse()
        mahalanobis_dist: int | float | bool = torch.sqrt(
            error.view(1, -1) @ cov_inv.float() @ error.view(-1, 1)
        ).item()
        return mahalanobis_dist

    def trigger_stereo_keyframe(self, timestamp: float, keyframe_id: int) -> None:
        """
        Trigger the backend to create a stereo-driven keyframe at the given timestamp.

        This is just a trigger signal - no stereo measurement is sent.
        The actual stereo pose constraint will be added later via process_registration_result().

        Args:
            timestamp: Time at which to create the keyframe
            keyframe_id: SurfSLAM keyframe ID to associate with this timestamp
        """
        if not self._first_keyframe_created:
            # First stereo frame becomes SurfSLAM id 1 (not id 0)
            # This triggers x1 creation in the backend
            adjusted_id = 1
            self._keyframe_id_to_timestamp[adjusted_id] = timestamp

            # Also store x0's timestamp (IMU-initialized pose) for potential future use
            first_pose_time = self._pg_backend.get_first_pose_time()
            if first_pose_time is not None:
                self._keyframe_id_to_timestamp[0] = first_pose_time

            # Trigger x1 creation
            self._pg_backend.trigger_stereo_keyframe(timestamp)

            self._first_keyframe_created = True
            self._next_keyframe_id = 1  # This was x1
        else:
            # Subsequent keyframes - use keyframe_id + 1 to maintain the offset
            adjusted_id: int = keyframe_id + 1
            self._keyframe_id_to_timestamp[adjusted_id] = timestamp

            # Trigger keyframe creation in the C++ backend
            self._pg_backend.trigger_stereo_keyframe(timestamp)
            self._next_keyframe_id += 1


    def process_registration_result(self,
                                    timestamp: float,
                                    transformation: list,
                                    prev_kf_timestamp: int,
                                    curr_kf_timestamp: int) -> None:

        # Create stereo measurement with actual pose
        meas = StereoMeasurement()
        meas.timestamp = float(timestamp)
        meas.prev_kf_timestamp = prev_kf_timestamp
        meas.curr_kf_timestamp = curr_kf_timestamp
        meas.transformation = transformation
        self._data_provider.insert_stereo(meas)

    def save_traj(self, fname) -> None:
        self._log_query_stats(force=True)
        self._pg_backend.save_trajectory(fname)
        self._pg_backend.save_dense_trajectory(fname)

    def _staleness(self, query_time: float) -> Optional[float]:
        """Seconds by which `query_time` predates the backend's trajectory window.

        The C++ backend keeps only a bounded deque of dense poses. Once a timestamp
        falls off the front there is no way for it to come back, so a query older
        than the window is permanently unsatisfiable rather than merely early.
        Returns None when the query is still inside (or ahead of) the window.
        """
        first_pose_time = self._pg_backend.get_first_pose_time()
        if first_pose_time is None or query_time >= first_pose_time:
            return None
        return first_pose_time - query_time

    def _log_query_stats(self, force: bool = False) -> None:
        """Rate-limited summary of pose queries that could not be answered."""
        now = time.time()
        elapsed = now - self._last_query_summary_time
        pending = self._query_stale_count + self._query_timeout_count
        if pending == 0 or not (force or elapsed >= _POSE_QUERY_SUMMARY_SEC):
            return

        print(
            f"[Backend] skipped {pending} pose queries in last {elapsed:.1f}s "
            f"({self._query_stale_count} aged out of the trajectory window, "
            f"max staleness {self._query_max_staleness:.1f}s; "
            f"{self._query_timeout_count} timed out) -- totals: "
            f"{self._query_stale_total} stale, {self._query_timeout_total} timeout"
        )
        self._query_stale_count = 0
        self._query_timeout_count = 0
        self._query_max_staleness = 0.0
        self._last_query_summary_time = now

    def _poll_optional(self, fetch, query_time: float, block: bool,
                       check_traj_window: bool, label: str):
        """Fetch an optional backend value, waiting only when waiting can help.

        `fetch` is retried every `_POSE_QUERY_POLL_SEC` until it returns a value or
        the wait is judged futile: either the query has aged out of the trajectory
        window (permanent -- checked every iteration, since the window front keeps
        advancing while we wait) or the deadline expires. Returns None on give-up.
        """
        value = fetch()
        if value is not None or not block:
            return value

        deadline = time.monotonic() + self._pose_query_timeout_sec
        while True:
            if check_traj_window:
                staleness = self._staleness(query_time)
                if staleness is not None:
                    self._query_stale_count += 1
                    self._query_stale_total += 1
                    self._query_max_staleness = max(
                        self._query_max_staleness, staleness)
                    if self._query_stale_total == 1:
                        print(
                            f"[Backend] {label} query t={query_time:.6f} is "
                            f"{staleness:.1f}s older than the trajectory window; "
                            "skipping (this will not be repeated for every query)"
                        )
                    self._log_query_stats()
                    return None

            if time.monotonic() >= deadline:
                self._query_timeout_count += 1
                self._query_timeout_total += 1
                if self._query_timeout_total == 1:
                    print(
                        f"[Backend] {label} query t={query_time:.6f} timed out after "
                        f"{self._pose_query_timeout_sec:.1f}s; skipping "
                        "(this will not be repeated for every query)"
                    )
                self._log_query_stats()
                return None

            time.sleep(_POSE_QUERY_POLL_SEC)
            value = fetch()
            if value is not None:
                return value

    def get_pose_at_time(self, query_time: float, return_cov=False, first_opt=False, block=True):
        """Pose (and optionally covariance) at `query_time`, or None if unavailable.

        With `return_cov=True` the return is always a 2-tuple -- callers unpack it.
        """
        if first_opt:
            # Backed by kf_timestamps_, which is append-only and never trimmed, so
            # the trajectory-window check does not apply; the deadline is the guard.
            def fetch_pose():
                return self._pg_backend.get_first_opt_kf_pose(query_time)
            pose_uses_window = False
        else:
            def fetch_pose():
                return self._pg_backend.get_pose_at_time(query_time)
            pose_uses_window = True

        T_world_rob_NED: Optional[np.ndarray] = self._poll_optional(
            fetch_pose, query_time, block, pose_uses_window, "pose")

        if not return_cov:
            return T_world_rob_NED

        if T_world_rob_NED is None:
            # No point paying a second wait for a timestamp already known to be lost.
            return None, None

        pose_cov = self._poll_optional(
            lambda: self._pg_backend.get_cov_at_time(query_time, first_opt),
            query_time, block, True, "pose covariance")

        if pose_cov is None:
            return T_world_rob_NED, pose_cov

        # swap [rx, ry, rz, tx, ty, tz] for [tx, ty, tz, rx, ry, rz]
        reorder_indices = [3, 4, 5, 0, 1, 2]
        pose_cov = pose_cov[np.ix_(reorder_indices, reorder_indices)]

        return T_world_rob_NED, pose_cov

    def get_pose_at_keyframe(self, index: int):
        return self._pg_backend.get_pose_at_keyframe(index)

    def get_pose_at_keyframe_published(self, index: int):
        return self._pg_backend.get_pose_at_keyframe_published(index)
