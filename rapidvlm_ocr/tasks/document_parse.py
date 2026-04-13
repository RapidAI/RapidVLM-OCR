# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from ..postprocess.markdown import MarkdownPostprocessor
from ..schema.enums import OutputFormat


class DocumentParseTask:
    output_format = OutputFormat.MARKDOWN

    def __init__(self):
        self.postprocessor = MarkdownPostprocessor()

    def postprocess(self, text: str):
        return self.postprocessor.parse(text)
