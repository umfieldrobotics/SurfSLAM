"""Where the datasets live on this machine.

Copy ``cfg/dataset_paths.example.yaml`` to ``cfg/dataset_paths.yaml`` and
fill in the paths you have.

Config files reference these roots with ``${key}`` tokens (e.g.
``dataset: ${suds_slam}/long/measurements.hdf5``); every config loader in the
repo runs :func:`expand_tokens` before parsing. Resolution order for a key
(first hit wins):

1. an explicit ``override`` argument,
2. the environment variable ``SURF_<KEY>_DIR`` (handy for containers and cluster jobs),
3. ``cfg/dataset_paths.yaml`` (or ``$SURF_DATASET_PATHS``).

Inside Docker, host paths from ``cfg/dataset_paths.yaml`` do not exist. ``docker/run.sh``
bind-mounts every registered dataset at ``/data/<key>``, so a path read from the yaml file
is re-pointed there when we are running in a container and that mount is present. Explicit
overrides (1) and ``SURF_<KEY>_DIR`` (2) are container-aware by assumption and never remapped.

Run ``python surfslam/common/dataset_paths.py`` to print what currently resolves.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Default location of the paths file. Override with ``$SURF_DATASET_PATHS``.
DEFAULT_PATHS_FILE = REPO_ROOT / "cfg" / "dataset_paths.yaml"

EXAMPLE_PATHS_FILE = REPO_ROOT / "cfg" / "dataset_paths.example.yaml"

ENV_PATHS_FILE = "SURF_DATASET_PATHS"

#: Set by ``docker/run.sh``; forces the in-container branch even if the heuristics below miss.
ENV_IN_DOCKER = "SURF_IN_DOCKER"

#: Where ``docker/run.sh`` mounts the datasets. Must match CONTAINER_DATA_ROOT in that script.
ENV_DOCKER_DATA_ROOT = "SURF_DOCKER_DATA_ROOT"
DEFAULT_DOCKER_DATA_ROOT = "/data"


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset key, and how to tell whether a path really points at it."""

    key: str
    description: str
    #: Relative path that must exist under the root for it to look valid.
    sentinel: Optional[str] = None
    #: Glob (relative to the root) used when a single sentinel file isn't meaningful.
    sentinel_glob: Optional[str] = None


DATASET_SPECS: Dict[str, DatasetSpec] = {
    spec.key: spec
    for spec in [
        DatasetSpec(
            key="suds_slam",
            description="SUDS-SLAM release scenes (DeepBlue): <scene>/measurements.hdf5 + ground_truth/",
            sentinel="long/measurements.hdf5",
        ),
        DatasetSpec(
            key="surfslam_data",
            description="SurfSLAM release (DeepBlue): DEFOM checkpoint + ground-truth maps",
            sentinel="ours_vits_slam.pth",
        ),
        DatasetSpec(
            key="slam_results",
            # An OUTPUT root, so it legitimately starts empty: no sentinel, which makes
            # looks_valid() accept any existing directory.
            description="Where run_all_ablations.sh parks finished runs (results, not inputs)",
        ),
        DatasetSpec(
            key="baseline_results",
            description="Baseline SLAM result roots for analysis/ comparisons (paper tables only)",
            sentinel_glob="*",
        ),
        DatasetSpec(
            key="flsea",
            description="FLSea_VI image tree (only for retraining the NetVLAD vocabulary)",
            sentinel_glob="*",
        ),
        DatasetSpec(
            key="acfr_slam",
            description="ACFR_SLAM image tree (only for retraining the NetVLAD vocabulary)",
            sentinel_glob="*",
        ),
        DatasetSpec(
            key="raw_tbnms",
            description="Raw TBNMS image tree (only for retraining the NetVLAD vocabulary)",
            sentinel_glob="*",
        ),
    ]
}


class DatasetNotRegisteredError(RuntimeError):
    """Raised when a dataset root cannot be resolved from any source."""


def _env_var_for(name: str) -> str:
    return f"SURF_{name.upper()}_DIR"


def paths_file() -> Path:
    """Path to ``cfg/dataset_paths.yaml``, honouring ``$SURF_DATASET_PATHS``."""
    override = os.environ.get(ENV_PATHS_FILE)
    return Path(override).expanduser() if override else DEFAULT_PATHS_FILE


def in_docker() -> bool:
    """True if this process looks like it is running inside a container."""
    flag = os.environ.get(ENV_IN_DOCKER, "")
    if flag:
        return flag.lower() not in ("0", "false", "no")
    if Path("/.dockerenv").exists():
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            return any(marker in f.read() for marker in ("docker", "containerd", "kubepods"))
    except OSError:
        return False


def docker_data_root() -> Path:
    """Directory under which ``docker/run.sh`` mounts the registered datasets."""
    return Path(os.environ.get(ENV_DOCKER_DATA_ROOT) or DEFAULT_DOCKER_DATA_ROOT)


def docker_path_for(name: str) -> Optional[Path]:
    """The in-container mount for ``name``, or ``None`` if it isn't there.

    Returns a path only when we are in a container *and* the mount actually exists, so a
    container started without ``docker/run.sh`` falls back to whatever the yaml file says.
    """
    if not in_docker():
        return None
    candidate = docker_data_root() / name
    return candidate if candidate.is_dir() else None


def _check_known(name: str) -> DatasetSpec:
    if name not in DATASET_SPECS:
        known = ", ".join(sorted(DATASET_SPECS))
        raise KeyError(f"Unknown dataset key {name!r}. Known keys: {known}")
    return DATASET_SPECS[name]


def load_paths() -> Dict[str, str]:
    """Read ``cfg/dataset_paths.yaml``. Missing file or unfilled (null) keys are fine."""
    path = paths_file()
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping of dataset name -> path")

    unknown = sorted(set(data) - set(DATASET_SPECS))
    if unknown:
        print(
            f"warning: {path} has unrecognized keys: {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(DATASET_SPECS))}",
            file=sys.stderr,
        )
    return {str(k): str(v) for k, v in data.items() if v is not None}


def looks_valid(name: str, path: Path) -> bool:
    """True if ``path`` plausibly contains the dataset identified by ``name``."""
    spec = _check_known(name)
    if not path.is_dir():
        return False
    if spec.sentinel is not None:
        return (path / spec.sentinel).exists()
    if spec.sentinel_glob is not None:
        return next(path.glob(spec.sentinel_glob), None) is not None
    return True


_WARNED: set = set()


def get_dataset_root(name: str, override: Optional[str] = None) -> Path:
    """Resolve the root directory for ``name``.

    Args:
        name: a key from :data:`DATASET_SPECS`.
        override: explicit path that wins over the environment and the paths file.
            ``None`` or an empty string means "not set".

    Raises:
        DatasetNotRegisteredError: if no source provides a path.
    """
    spec = _check_known(name)

    if override:
        raw, source = str(override), "override"
    elif os.environ.get(_env_var_for(name)):
        raw, source = os.environ[_env_var_for(name)], f"${_env_var_for(name)}"
    else:
        raw, source = load_paths().get(name), str(paths_file())
        mounted = docker_path_for(name)
        if mounted is not None:
            # The yaml path is a host path; use the bind mount docker/run.sh made from it.
            raw, source = str(mounted), f"docker mount for {paths_file()}"

    if raw is None:
        raise DatasetNotRegisteredError(
            f"No path set for '{name}' ({spec.description}).\n"
            f"  Add it to {paths_file()}:  {name}: /path/on/this/machine\n"
            f"  (copy {EXAMPLE_PATHS_FILE.relative_to(REPO_ROOT)} if that file doesn't exist yet)"
        )

    path = Path(raw).expanduser().resolve()

    if not path.is_dir():
        raise DatasetNotRegisteredError(
            f"'{name}' points at {path} (from {source}), which is not a directory.\n"
            f"  Fix the '{name}' entry in {paths_file()}."
        )

    if name not in _WARNED and not looks_valid(name, path):
        _WARNED.add(name)
        expected = spec.sentinel or spec.sentinel_glob
        print(
            f"warning: {path} (from {source}) does not look like '{name}' "
            f"- expected to find {expected!r} inside it. Continuing anyway.",
            file=sys.stderr,
        )

    return path


_TOKEN_RE = re.compile(r"\$\{([a-z_][a-z0-9_]*)\}")


def expand_tokens(text: str) -> str:
    """Expand ``${key}`` dataset-root tokens and the ``PROJECT_ROOT`` token in config text.

    Only keys that actually appear in ``text`` are resolved, so an unfilled
    dataset_paths.yaml entry never breaks a run that doesn't need it.
    """
    text = text.replace("PROJECT_ROOT", str(REPO_ROOT))
    return _TOKEN_RE.sub(lambda m: str(get_dataset_root(m.group(1))), text)


def load_config(path) -> dict:
    """Read a yaml config, expanding ``${key}`` and ``PROJECT_ROOT`` tokens first."""
    with open(os.path.expanduser(str(path)), "r") as f:
        return yaml.safe_load(expand_tokens(f.read()))


def resolve_config_path(name_or_path) -> str:
    """Resolve a config reference (e.g. a ``baseline:`` value) to a real file.

    Tries the literal (expanduser'd) path first, then relative to ``cfg/`` in the
    repo. Returns the original string if neither exists, so callers fail with a
    useful open() error.
    """
    literal = os.path.expanduser(str(name_or_path))
    if os.path.exists(literal):
        return literal
    candidate = REPO_ROOT / "cfg" / str(name_or_path)
    if candidate.exists():
        return str(candidate)
    return str(name_or_path)


def _main() -> int:
    """Print every key and what it currently resolves to."""
    entries = load_paths()
    exists = paths_file().exists()
    print(f"paths file: {paths_file()}" + ("" if exists else "  (does not exist yet)"))
    if not exists:
        print(f"            copy {EXAMPLE_PATHS_FILE.relative_to(REPO_ROOT)} to it and fill it in")
    if in_docker():
        print(f"in docker: yes, yaml paths re-pointed under {docker_data_root()}/<key> when mounted")

    width = max(len(k) for k in DATASET_SPECS)
    for name in sorted(DATASET_SPECS):
        env_val = os.environ.get(_env_var_for(name))
        mounted = None if env_val else docker_path_for(name)
        raw = env_val or (str(mounted) if mounted else entries.get(name))
        if not raw:
            status = "unset"
        elif not Path(raw).expanduser().is_dir():
            status = "missing"
        elif not looks_valid(name, Path(raw).expanduser().resolve()):
            status = "suspect"
        else:
            status = "ok"
        if env_val:
            suffix = f"  [{_env_var_for(name)}]"
        elif mounted:
            suffix = "  [docker mount]"
        else:
            suffix = ""
        print(f"  {name:<{width}}  {status:<8}  {raw or '-'}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
