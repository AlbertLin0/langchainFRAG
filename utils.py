import yaml
import os
import logging

logger = logging.getLogger(__name__)

def load_yaml_config(file_path: str):
    """

    """
    if not os.path.exists(file_path):
        logger.error(f"Config file not found: {file_path}")
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {file_path}: {e}")
            raise yaml.YAMLError(f"Failed to parse YAML file {file_path}: {e}")
