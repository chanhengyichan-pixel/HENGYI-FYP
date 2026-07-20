import yaml
from pathlib import Path


def load_config(config_path="config/config.yaml"):
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse configuration file '{path}': {e}") from e

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    return config
