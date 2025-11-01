"""Utilities to manage a TensorBoard subprocess.

This module exposes TensorboardManager which can start and stop an external
TensorBoard process given a TensorboardConfig. The manager captures stdout/
stderr and provides basic lifecycle handling.
"""

import subprocess
import time

from models.config.tensorboard_config import TensorboardConfig
from utils.logger import logger


class TensorboardManager:
    """
    Manages the lifecycle of an external TensorBoard subprocess.

    Parameters
    ----------
    config : TensorboardConfig
        Configuration object controlling auto-start, host and port.

    Attributes
    ----------
    config : TensorboardConfig
        The configuration passed to the manager.
    logdir : str or None
        Last used log directory passed to start().
    process : subprocess.Popen or None
        Handle to the running TensorBoard process, if any.

    Notes
    -----
    - The manager uses `subprocess.Popen` to start the `tensorboard` binary.
    - STDOUT/STDERR are captured but not processed further by this class.
    - The start() method waits a short time to determine whether startup
      succeeded (process still running).
    """

    def __init__(self, config: TensorboardConfig):
        self.config = config
        self.logdir = None
        self.process = None

    def start(self, logdir: str):
        """
        Start TensorBoard pointing to a given log directory.

        Parameters
        ----------
        logdir : str
            Path to the TensorBoard log directory (event files).

        Returns
        -------
        bool
            True if TensorBoard was launched successfully and appears to be
            running after a short wait, False otherwise.

        Raises
        ------
        FileNotFoundError
            If the `tensorboard` executable is not found on PATH (caught and
            logged inside the method).
        """

        if not self.config.auto_start:
            return False

        self.logdir = logdir

        try:
            logger.info(f"Starting Tensorboard on port {self.config.port}...")

            self.process = subprocess.Popen(
                [
                    "tensorboard",
                    "--logdir", self.logdir,
                    "--port", str(self.config.port),
                    "--host", self.config.host
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            time.sleep(3)

            if self.process.poll() is None:
                url = f"http://localhost:{self.config.port}"
                logger.info(f"Tensorboard started successfully at {url}")
                return True
            else:
                logger.error("Tensorboard failed to start")
                return False

        except FileNotFoundError:
            logger.error("Tensorboard not found.")
            return False
        except Exception as e:
            logger.error(f"Error starting Tensorboard: {e}")
            return False

    def stop(self):
        """
        Stop the TensorBoard subprocess if it is running.

        Returns
        -------
        None

        Notes
        -----
        Terminates the process and waits until it exits. If no process is
        running this is a no-op.
        """

        if self.process and self.process.poll() is None:
            logger.info("Stopping Tensorboard...")
            self.process.terminate()
            self.process.wait()
            logger.info("Tensorboard stopped")

    def __del__(self):
        """
        Ensure TensorBoard process is terminated on object deletion.

        Notes
        -----
        Destructor best-effort stops the process; do not rely on this for
        deterministic shutdown in long-running applications (call stop()).
        """

        self.stop()
