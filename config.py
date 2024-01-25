import os
import yaml
import logging

logger = logging.getLogger(__name__)


def read_yaml(file_path):
    config_yaml = {}
    try:
        with open(file_path, "r") as f:
            config_yaml = yaml.safe_load(f)
    except Exception as e:
        logger.error(f'Error occurred while loading config.yaml {e}')

    return config_yaml


config_path = os.getenv('CONFIG_PATH') or '/app/config.yaml'
config = read_yaml(config_path)
