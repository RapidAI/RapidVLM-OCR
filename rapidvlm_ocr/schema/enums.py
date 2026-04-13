# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from enum import Enum


class TaskType(str, Enum):
    OCR = "ocr"
    DOCUMENT_PARSE = "document_parse"
    KIE = "kie"


class ModelName(str, Enum):
    QIANFAN_OCR = "qianfan_ocr"


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


class EngineType(str, Enum):
    VLLM = "vllm"
    MOCK = "mock"
