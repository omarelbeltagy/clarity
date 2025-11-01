"""General utility functions used across the project.

This module contains small helpers for filesystem cleanup, environment
detection and safe type conversion with logging fallbacks.
"""

import os
import shutil

from utils.logger import logger


def cleanup_checkpoints(output_dir: str):
    """
    Delete checkpoint directories in a given output directory.

    Scans `output_dir` for subdirectories that begin with the prefix
    "checkpoint-" and removes them recursively.

    Parameters
    ----------
    output_dir : str
        Path to the directory that should be scanned for checkpoint folders.

    Returns
    -------
    None

    Notes
    -----
    Uses shutil.rmtree(..., ignore_errors=True) so failures are logged by the
    underlying system but do not raise exceptions here.
    """
    for item in os.listdir(output_dir):
        path = os.path.join(output_dir, item)
        if os.path.isdir(path) and item.startswith("checkpoint-"):
            logger.info(f"Deleting checkpoint directory: {path}")
            shutil.rmtree(path, ignore_errors=True)


def is_running_in_docker() -> bool:
    """
    Detect whether the current process runs inside a Docker container.

    The detection performs several checks:
    - Presence of the /.dockerenv file.
    - Inspecting /proc/1/cgroup and /proc/self/cgroup for container identifiers.
    - Checking the DOCKER_CONTAINER environment variable.

    Returns
    -------
    bool
        True if any heuristic indicates a containerized (Docker) environment,
        False otherwise.

    Notes
    -----
    Heuristics are not perfect; false positives/negatives are possible depending
    on host configuration and container runtime.
    """

    # 1: Check for .dockerenv file (most reliable)
    if os.path.exists('/.dockerenv'):
        return True

    # 2: Check /proc/1/cgroup for docker or containerd
    try:
        with open('/proc/1/cgroup', 'r') as f:
            content = f.read()
            if 'docker' in content or 'containerd' in content:
                return True
    except (FileNotFoundError, PermissionError):
        pass

    # 3: Check if /proc/self/cgroup contains docker
    try:
        with open('/proc/self/cgroup', 'r') as f:
            content = f.read()
            if 'docker' in content or 'containerd' in content:
                return True
    except (FileNotFoundError, PermissionError):
        pass

    # 4: Check environment variables
    if os.getenv('DOCKER_CONTAINER'):
        return True

    return False


def get_execution_environment() -> str:
    """
    Return a short description of the current execution environment.

    The returned string includes whether the code runs in Docker and basic
    platform information such as OS, machine architecture and Python version.

    Returns
    -------
    str
        Human-readable description, e.g. "Docker Container on Linux (x86_64)"
        or "Native Darwin (arm64)".
    """
    import platform

    env_info = {
        'is_docker': is_running_in_docker(),
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version()
    }

    if env_info['is_docker']:
        return f"Docker Container on {env_info['platform']} ({env_info['machine']})"
    else:
        return f"Native {env_info['platform']} ({env_info['machine']})"


def as_int(val, default):
    """
    Safely convert a value to int with a fallback default.

    Parameters
    ----------
    val : any
        Value to convert to int.
    default : any
        Value to return if conversion fails. If both val and default are None,
        None is returned.

    Returns
    -------
    int or None
        Converted integer or the provided default.

    Notes
    -----
    Logs a warning when conversion fails.
    """
    try:
        return int(val)
    except (TypeError, ValueError):
        if val is None and default is None:
            return None
        logger.warning(f"Failed to convert {val} to int, returning default {default}")
        return default


def as_float(val, default):
    """
    Safely convert a value to float with a fallback default.

    Parameters
    ----------
    val : any
        Value to convert to float.
    default : any
        Value to return if conversion fails. If both val and default are None,
        None is returned.

    Returns
    -------
    float or None
        Converted float or the provided default.

    Notes
    -----
    Logs a warning when conversion fails.
    """
    try:
        return float(val)
    except (TypeError, ValueError):
        if val is None and default is None:
            return None
        logger.warning(f"Failed to convert {val} to float, returning default {default}")
        return default


def as_bool(val, default):
    """
    Convert a value to boolean with reasonable interpretations.

    Parameters
    ----------
    val : any
        Value to convert. Strings like "true", "1", "yes" (case-insensitive)
        are interpreted as True. Actual bool values are returned unchanged.
    default : any
        Value returned if conversion is ambiguous. If both val and default
        are None, None is returned.

    Returns
    -------
    bool or None
        Converted boolean or the provided default.

    Notes
    -----
    This helper is intentionally permissive to allow configuration values
    coming from environment variables or configuration files.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    if val is None and default is None:
        return None
    logger.warning(f"Failed to convert {val} to bool, returning default {default}")
    return default


def as_str(val, default):
    """
    Safely convert a value to string with a fallback default.

    Parameters
    ----------
    val : any
        Value to convert to str.
    default : any
        Value returned if conversion fails. If both val and default are None,
        None is returned.

    Returns
    -------
    str or None
        Converted string or the provided default.

    Notes
    -----
    Logs a warning when conversion fails.
    """
    try:
        return str(val)
    except (TypeError, ValueError):
        if val is None and default is None:
            return None
        logger.warning(f"Failed to convert {val} to str, returning default {default}")
        return default
