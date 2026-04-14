# -*- encoding: utf-8 -*-
from __future__ import annotations

import sys
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
