# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from ..utils.logger import logger
from ..utils.utils import read_yaml


def load_default_generation_config(model_name: str) -> dict:
    cur_dir = Path(__file__).parent / "generation.yaml"
    generation_config = read_yaml(cur_dir)
    if model_name in generation_config:
        return generation_config[model_name]

    logger.warning(
        f"No default generation config found for model {model_name}, using default config"
    )
    return generation_config["default"]
