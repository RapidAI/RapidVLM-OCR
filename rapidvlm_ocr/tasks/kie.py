# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from ..postprocess.json import JsonPostprocessor
from ..schema.enums import OutputFormat


class KIETask:
    output_format = OutputFormat.JSON

    def __init__(self):
        self.postprocessor = JsonPostprocessor()

    def postprocess(self, text: str):
        return self.postprocessor.parse(text)
