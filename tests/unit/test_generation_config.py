# -*- encoding: utf-8 -*-
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from rapidvlm_ocr import EngineType, ModelName, RapidVLMOCR


def test_default_generation_config_can_be_overridden() -> None:
    app = RapidVLMOCR(
        model_name=ModelName.QIANFAN_OCR,
        engine=EngineType.MOCK,
    )

    merged = app.merge_generation_config({"max_tokens": 1024})

    assert merged["temperature"] == 0.2
    assert merged["max_tokens"] == 1024
    assert merged["stop"] == ["<|endoftext|>", "###"]


def test_model_read_belong_generate_config():
    app = RapidVLMOCR(
        model_name=ModelName.QIANFAN_OCR,
        engine=EngineType.MOCK,
    )

    config = app.default_generation_config
    assert config["temperature"] == 0.2
    assert config["max_tokens"] == 2048
    assert config["stop"][0] == "<|endoftext|>"


def test_unknown_model_name_fallback_default_generation_config():
    class FakeModelName(Enum):
        UNKNOWN_MODEL = "unknown_model"

    app = RapidVLMOCR(
        model_name=FakeModelName.UNKNOWN_MODEL,  # type: ignore[arg-type]
        engine=EngineType.MOCK,
    )

    config = app.default_generation_config
    assert config["temperature"] == 0.2
    assert config["max_tokens"] == 2048
    assert config["stop"][0] == "<|endoftext|>"
