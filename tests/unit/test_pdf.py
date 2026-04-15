# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from rapidvlm_ocr.preprocess.pdf import load_pdf_images


def test_load_pdf_images_returns_rendered_pages() -> None:
    images = load_pdf_images("tests/test_files/test.pdf")

    assert len(images) == 2
    assert all(isinstance(image, Image.Image) for image in images)
    assert all(image.mode == "RGB" for image in images)
    assert all(image.size[0] > 0 and image.size[1] > 0 for image in images)
