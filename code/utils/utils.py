import os
import json
from logging.config import dictConfig

import yaml


def load_config():
    """config.json 파싱 후 config 반환

    Returns:
        dict: db 정보, log 정보
    """
    try:
        config = "./cfg/config.json"
        with open(config) as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"Config load fail: {e}")

    return cfg


def setting_logger():
    log_config_file = "./cfg/logger.yaml"
    if not os.path.isfile(log_config_file):
        raise FileNotFoundError

    with open(log_config_file, 'rt') as f:
        config = yaml.safe_load(f.read())
        dictConfig(config)
