# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from ..preprocess.text import normalize_text
from ..schema.enums import OutputFormat


class OCRTask:
    output_format = OutputFormat.TEXT

    def postprocess(self, text: str):
        return normalize_text(text)
