# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapidvlm_ocr import EngineType, ModelName, RapidVLMOCR, TaskType

# app = RapidVLMOCR(engine=EngineType.MOCK)

model_path = "models/Qianfan-OCR"
app = RapidVLMOCR(
    engine=EngineType.VLLM, model_name=ModelName.QIANFAN_OCR, model_path=model_path
)

img_path = "tests/test_files/QianFan_OCR/document.png"
result = app(
    task_type=TaskType.DOCUMENT_PARSE,
    image_path=img_path,
    generation_config={
        "temperature": 0.2,
        "max_tokens": 2048,
        "stop": ["<|endoftext|>", "###"],
    },
)

print(result)
