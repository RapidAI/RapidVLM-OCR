# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidvlm_ocr import EngineType, InferenceResponse, RapidVLMOCR, TaskType

TEST_FILES_DIR = root_dir / "tests" / "test_files" / "QianFan_OCR"

app = RapidVLMOCR(engine=EngineType.MOCK)
