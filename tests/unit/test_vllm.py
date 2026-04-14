# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import importlib
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


def test_vllm_package_is_importable() -> None:
    vllm = importlib.import_module("vllm")

    assert vllm is not None
    assert hasattr(vllm, "LLM")
    assert hasattr(vllm, "SamplingParams")


def test_project_vllm_engine_dependencies_are_available() -> None:
    module = importlib.import_module("rapidvlm_ocr.engine.vllm_engine")

    assert hasattr(module, "VLLMEngine")
