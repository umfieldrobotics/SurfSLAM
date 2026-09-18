import csv
import json
import os
import time
from typing import Any, Dict, Optional

_TIMING_FILENAME = "process_stereo_timing.csv"
_FIELDNAMES = ["wall_time", "stage", "duration_sec", "metadata"]


def _serialize_metadata(metadata: Optional[Dict[str, Any]]) -> str:
    if metadata is None:
        return "{}"

    # Convert unsupported values to strings to keep JSON dumps simple.
    safe_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (int, float, str)) or value is None:
            safe_metadata[key] = value
        else:
            safe_metadata[key] = str(value)

    return json.dumps(safe_metadata, ensure_ascii=True)


def log_process_timing(
    log_directory: Optional[str],
    stage: str,
    duration_sec: float,
    metadata: Optional[Dict[str, Any]] = None,
    emit_stdout: bool = False,
) -> None:
    """Append a timing measurement to the shared profiling log."""
    if log_directory is None:
        return

    os.makedirs(log_directory, exist_ok=True)
    log_path = os.path.join(log_directory, _TIMING_FILENAME)

    write_header = not os.path.exists(log_path)
    metadata_blob = _serialize_metadata(metadata)

    with open(log_path, "a", newline="") as timing_file:
        writer = csv.DictWriter(timing_file, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "wall_time": f"{time.time():.6f}",
                "stage": stage,
                "duration_sec": f"{duration_sec:.6f}",
                "metadata": metadata_blob,
            }
        )

    if emit_stdout:
        metadata_text = ""
        if metadata:
            metadata_pairs = [f"{k}={v}" for k, v in metadata.items()]
            metadata_text = f" ({', '.join(metadata_pairs)})"
        print(
            f"[Timing] {stage}: {duration_sec:.3f}s{metadata_text}"
        )
