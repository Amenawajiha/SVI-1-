""" Config loader. """

import json
from commons.src.log_helper import logger


class ConfigLoader:
    """Config loader Singleton"""

    _instances = {}

    @classmethod
    def get_config(cls, config_path: str):
        """Get config."""
        logger.info("Loading config from %s", config_path)

        if config_path not in cls._instances:
            with open(config_path, "r", encoding="utf-8") as f:
                cls._instances[config_path] = json.load(f)
        return cls._instances[config_path]
