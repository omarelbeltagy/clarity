import os
import pathlib
from utils.logger import logger


def is_running_in_docker() -> bool:
    """
    Check if the code is running inside a Docker container.
    Uses multiple detection methods for reliability.
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
    Get a description of the current execution environment.
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
    try:
        return int(val)
    except (TypeError, ValueError):
        if val is None and default is None:
            return None
        logger.warning(f"Failed to convert {val} to int, returning default {default}")
        return default


def as_float(val, default):
    try:
        return float(val)
    except (TypeError, ValueError):
        if val is None and default is None:
            return None
        logger.warning(f"Failed to convert {val} to float, returning default {default}")
        return default


def as_bool(val, default):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    if val is None and default is None:
        return None
    logger.warning(f"Failed to convert {val} to bool, returning default {default}")
    return default


def as_str(val, default):
    try:
        return str(val)
    except (TypeError, ValueError):
        if val is None and default is None:
            return None
        logger.warning(f"Failed to convert {val} to str, returning default {default}")
        return default
