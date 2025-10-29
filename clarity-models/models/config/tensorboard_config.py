from dataclasses import dataclass, field
from typing import Dict, Optional
from utils.general_utils import (
    as_int,
    as_float,
    as_bool,
    as_str
)


@dataclass
class TensorboardConfig:
    """Tensorboard configuration."""
    auto_start: bool = True
    port: int = 6006
    host: str = "0.0.0.0"

    @classmethod
    def from_dict(cls, cfg: Dict) -> "TensorboardConfig":
        """Create TensorboardConfig from dictionary."""
        return cls(
            auto_start=as_bool(cfg.get("auto_start", True), True),
            port=as_int(cfg.get("port", 6006), 6006),
            host=as_str(cfg.get("host", "0.0.0.0"), "0.0.0.0"),
        )
