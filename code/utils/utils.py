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

    # 1. logger.yaml 확인
    if not os.path.isfile(log_config_file):
        raise FileNotFoundError(f"Logger config not found: {log_config_file}")

    # 2. log 폴더 자동 생성
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 3. YAML 로드 후 설정 적용
    with open(log_config_file, "rt", encoding="utf8") as f:
        config = yaml.safe_load(f)
        dictConfig(config)
