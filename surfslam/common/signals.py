"""
File: src/common/signals.py

Extremely simple signals/slots implementation for multiprocessing information sharing

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

import time
import torch.multiprocessing as mp
import copy
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TimestampedData:
    """Wrapper that attaches wall-clock emission time to data."""
    wall_time: float
    data: Any


class StopSignal:
    """ Dummy class used to signal processes to stop by inserting this into MP queues.
    """
    pass


class SimpleQueue:
    """
    A very simple queue to mimic the interface of MP queue for single-threaded operation
    """
    def __init__(self):
        self._data = []

    def put(self, value):
        self._data.append(copy.deepcopy(value))

    def get(self):
        return self._data.pop(0)

    def empty(self):
        return len(self._data) == 0

    def full(self):
        return False

    def qsize(self):
        return len(self._data)


class Slot:
    """ A Slot is a listener which listens to data on a particular signal.

    This is analogous to a subscriber in ROS
    """

    # This should not be called directly. Instead, call Signal.register
    def __init__(self, single_process: bool,
                 max_age_seconds: Optional[float] = None,
                 signal_name: str = "unnamed",
                 warn_on_drop: bool = False,
                 summary_interval_seconds: float = 10.0):

        if single_process:
            self._queue = SimpleQueue()
        else:
            # The use of Manager().Queue() instead of mp.Queue() here is quite important.
            self._queue = mp.Queue()

        # Staleness configuration
        self._max_age_seconds = max_age_seconds
        self._signal_name = signal_name
        self._warn_on_drop = warn_on_drop
        self._summary_interval_seconds = summary_interval_seconds

        # Drop tracking - use shared memory for multiprocessing
        self._single_process = single_process
        if single_process:
            self._drop_count = 0
            self._total_drop_count = 0
        else:
            self._drop_count = mp.Value('i', 0)
            self._total_drop_count = mp.Value('i', 0)
        self._last_summary_time = time.time()
        self._max_dropped_age = 0.0

    # Checks whether a value is available
    def has_value(self) -> bool:
        return not self._queue.empty()

    # Returns a value if available, and otherwise None
    # Drops stale items based on wall-clock age if max_age_seconds is configured
    def get_value(self):
        while self.has_value():
            item = self._queue.get()

            # Handle unwrapping timestamped data
            if isinstance(item, TimestampedData):
                wall_time, data = item.wall_time, item.data
            else:
                # Backwards compatibility: unwrapped data has no age check
                return item

            # Never drop stop signals
            if isinstance(data, StopSignal):
                self._maybe_log_summary(force=True)  # Final summary before stop
                return data

            # Check staleness if configured
            if self._max_age_seconds is not None:
                age = time.time() - wall_time
                if age > self._max_age_seconds:
                    self._record_drop(age)
                    continue  # Drop and try next item

            return data

        return None

    def _record_drop(self, age: float):
        """Record a dropped item and potentially log summary."""
        if self._single_process:
            self._drop_count += 1
            self._total_drop_count += 1
        else:
            self._drop_count.value += 1
            self._total_drop_count.value += 1
        self._max_dropped_age = max(self._max_dropped_age, age)
        self._maybe_log_summary()

    def _maybe_log_summary(self, force: bool = False):
        """Log drop summary if interval has elapsed or forced."""
        now = time.time()
        elapsed = now - self._last_summary_time

        drop_count = self._drop_count if self._single_process else self._drop_count.value
        total_drop_count = self._total_drop_count if self._single_process else self._total_drop_count.value

        if drop_count > 0 and (force or elapsed >= self._summary_interval_seconds):
            msg = (f"[{self._signal_name}] Dropped {drop_count} items in last "
                   f"{elapsed:.1f}s (max age: {self._max_dropped_age:.2f}s, "
                   f"total dropped: {total_drop_count})")

            if self._warn_on_drop:
                print(f"WARNING: {msg}")
            else:
                print(msg)

            # Reset counters
            if self._single_process:
                self._drop_count = 0
            else:
                self._drop_count.value = 0
            self._max_dropped_age = 0.0
            self._last_summary_time = now

    def get_drop_stats(self) -> dict:
        """Return current drop statistics for monitoring."""
        total_dropped = self._total_drop_count if self._single_process else self._total_drop_count.value
        pending_drops = self._drop_count if self._single_process else self._drop_count.value
        return {
            "signal_name": self._signal_name,
            "total_dropped": total_dropped,
            "pending_summary_drops": pending_drops,
            "max_age_seconds": self._max_age_seconds,
        }

    def __len__(self):
        return self._queue.qsize()

    # Used by Signal to send data. Don't call directly.
    def _insert(self, value):
        self._queue.put(value)


class Signal:
    """ A Signal defines a channel for communication.

    A Signal object is analogous to a topic and publisher in ROS, all in one.

    Calling @m register returns a slot, which functions as a subscriber.
    """

    # Constructor: An empty signal is just an empty list of slots
    # If @p synchronous is True, then emit will block until each item has been removed
    # If @p single_process is true, this will not use MP queues, and will just use a normal queue
    # @p name is used for identifying the signal in log messages
    # @p signal_type is "input" or "internal" - input signals show warnings when dropping
    # @p max_age_seconds is the maximum age (wall-clock) before data is dropped
    def __init__(self, synchronous: bool = False, single_process: bool = False,
                 name: str = "unnamed", signal_type: str = "internal",
                 max_age_seconds: Optional[float] = None,
                 warn_on_drop: Optional[bool] = None,
                 summary_interval_seconds: float = 10.0):

        # Stores Slot objects to write to when data is emitted
        self._slots = []

        self._synchronous = synchronous
        self._single_process = single_process

        # Staleness configuration
        self._name = name
        self._signal_type = signal_type
        self._max_age_seconds = max_age_seconds
        self._summary_interval_seconds = summary_interval_seconds

        # Default warn_on_drop based on signal type if not specified
        if warn_on_drop is None:
            self._warn_on_drop = (signal_type == "input")
        else:
            self._warn_on_drop = warn_on_drop

    # Creates and returns a Slot which listens on the Signal
    def register(self) -> Slot:
        slot = Slot(
            self._single_process,
            max_age_seconds=self._max_age_seconds,
            signal_name=self._name,
            warn_on_drop=self._warn_on_drop,
            summary_interval_seconds=self._summary_interval_seconds
        )
        self._slots.append(slot)
        return slot

    # Sends the given value to all the registered Slots
    # Wraps the value with a wall-clock timestamp for staleness detection
    def emit(self, value) -> None:
        timestamped = TimestampedData(wall_time=time.time(), data=value)
        for s in self._slots:
            while self._synchronous and s.has_value():
                time.sleep(1e-5)
            s._insert(timestamped)

    # Get drop statistics from all slots
    def get_drop_stats(self) -> dict:
        """Return drop statistics aggregated across all slots."""
        total_dropped = 0
        for s in self._slots:
            if s._single_process:
                total_dropped += s._total_drop_count
            else:
                total_dropped += s._total_drop_count.value
        return {
            "signal_name": self._name,
            "total_dropped": total_dropped,
            "num_slots": len(self._slots),
            "max_age_seconds": self._max_age_seconds,
        }
