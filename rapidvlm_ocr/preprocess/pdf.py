# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


def load_pdf_images(pdf_path: str | Path, scale: int = 2) -> list[Image.Image]:
    images = []

    with pdfium.PdfDocument(pdf_path) as pdf:
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            try:
                images.append(bitmap.to_pil())
            finally:
                bitmap.close()
                page.close()

    return images
