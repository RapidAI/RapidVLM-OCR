# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from ..preprocess.text import normalize_text


class MarkdownPostprocessor:
    def parse(self, text: str) -> str:
        return normalize_text(text)
