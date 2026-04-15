# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapidvlm_ocr import EngineType, ModelName, RapidVLMOCR, TaskType

model_path = "models/Qianfan-OCR"
app = RapidVLMOCR(
    model_name=ModelName.QIANFAN_OCR,
    model_path=model_path,
    engine=EngineType.VLLM,
)

img_paths = [
    "tests/test_files/QianFan_OCR/general_1.jpeg",
    "tests/test_files/QianFan_OCR/general.jpg",
    "tests/test_files/QianFan_OCR/document.png",
]
input_path = "tests/test_files/test.pdf"
result = app(task_type=TaskType.DOCUMENT_PARSING, input_path=input_path, batch_size=2)

print(result)
