import subprocess
import time
from models.config.tensorboard_config import TensorboardConfig
from utils.logger import logger


class TensorboardManager:
    """Manages Tensorboard process."""

    def __init__(self, config: TensorboardConfig):
        self.config = config
        self.logdir = None
        self.process = None

    def start(self, logdir: str):
        """Start Tensorboard server."""
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
        """Stop Tensorboard server."""
        if self.process and self.process.poll() is None:
            logger.info("Stopping Tensorboard...")
            self.process.terminate()
            self.process.wait()
            logger.info("Tensorboard stopped")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
