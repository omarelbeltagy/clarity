"""Tensorboard configuration dataclass.

This module provides a simple dataclass wrapper to hold TensorBoard
configuration options and a helper to instantiate it from a dictionary.

Examples
--------
>>> cfg = TensorboardConfig.from_dict({'auto_start': False, 'port': 7007})
>>> cfg.port
7007
"""

from dataclasses import dataclass
from typing import Dict

from utils.general_utils import (
    as_int,
    as_float,
    as_bool,
    as_str
)


@dataclass
class TensorboardConfig:
    """
    Configuration container for launching TensorBoard.

    Parameters
    ----------
    auto_start : bool, optional
        If True the TensorBoard process should be auto-started by the manager,
        by default True.
    port : int, optional
        TCP port on which TensorBoard should listen, by default 6006.
    host : str, optional
        Host interface for TensorBoard, by default "0.0.0.0".

    Notes
    -----
    This is a simple typed container used by the TensorboardManager
    to configure the external TensorBoard process.
    """
    auto_start: bool = True
    port: int = 6006
    host: str = "0.0.0.0"

    @classmethod
    def from_dict(cls, cfg: Dict) -> "TensorboardConfig":
        """
        Create a TensorboardConfig from a dictionary, with safe conversions.

        Parameters
        ----------
        cfg : dict
            Mapping that can contain keys "auto_start", "port" and "host".
            Values are coerced using helper conversion functions and fallbacks
            are applied if conversion fails or keys are missing.

        Returns
        -------
        TensorboardConfig
            Instantiated configuration object.

        Examples
        --------
        >>> TensorboardConfig.from_dict({'auto_start': 'false', 'port': '6007'})
        TensorboardConfig(auto_start=False, port=6007, host='0.0.0.0')
        """
        return cls(
            auto_start=as_bool(cfg.get("auto_start", True), True),
            port=as_int(cfg.get("port", 6006), 6006),
            host=as_str(cfg.get("host", "0.0.0.0"), "0.0.0.0"),
        )
