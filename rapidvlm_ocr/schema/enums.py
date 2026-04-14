# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from enum import Enum


class TaskType(str, Enum):
    TEXT_EXTRACTION = "text_extraction"
    FORMULA_PARSING = "formula_parsing"
    TABLE_PARSING = "table_parsing"
    DOCUMENT_PARSING = "document_parsing"
    MULTI_SCENE_REC = "multilingual_scene_text_recognition"
    KIE = "key_information_extraction"


class ModelName(str, Enum):
    QIANFAN_OCR = "qianfan_ocr"


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


class EngineType(str, Enum):
    VLLM = "vllm"
    MOCK = "mock"
