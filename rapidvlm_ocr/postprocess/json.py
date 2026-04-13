# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import json

from ..preprocess.text import normalize_text


class JsonPostprocessor:
    def parse(self, text: str):
        cleaned = normalize_text(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"text": cleaned}
