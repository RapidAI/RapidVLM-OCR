# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")
