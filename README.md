<div align="center">
  <h1><b>RapidVLM-OCR</b></h1>
  <b><font size="4"><i> 基于 vLLM 加速的端到端 VLM-OCR 统一推理框架 </i></font></b>
  <div>&nbsp;</div>

<a href=""><img src="https://img.shields.io/badge/Python->=3.6-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

</div>

### 📝 简介

RpaidVLM-OCR：基于 vLLM 加速的端到端 VLM-OCR 统一推理框架，多模型兼容、一套接口、开箱即用。

项目定位：只集成端到端 OCR 领域大模型，力争在效果和模型大小之间找到一个平衡。

与现有项目区别：

- [RapidOCR](https://github.com/RapidAI/RapidOCRDocs): 仅集成单纯从图像中提取文字功能，目前仅包括两阶段传统方案。
- [RapidDoc](https://github.com/RapidAI/RapidDoc): 基于 MinerU，两阶段方案，集成多个小模型来做文档解析任务。
- RapidVLM-OCR: 专注端到端 VLM-OCR，解决综合性任务。

## 下载模型

```bash
huggingface-cli download --resume-download baidu/Qianfan-OCR --local-dir Qianfan-OCR
```

## 使用

### 图像批处理

```python
from rapidvlm_ocr import EngineType, ModelName, RapidVLMOCR, TaskType

model_path = "models/Qianfan-OCR"
app = RapidVLMOCR(
    model_name=ModelName.QIANFAN_OCR,
    model_path=model_path,
    engine=EngineType.VLLM,
)

input_paths = [
    "tests/test_files/QianFan_OCR/general_1.jpeg",
    "tests/test_files/QianFan_OCR/general.jpg",
    "tests/test_files/QianFan_OCR/document.png",
]
result = app(task_type=TaskType.DOCUMENT_PARSING, input_path=input_paths, batch_size=2)

print(result)
```

### PDF 输入

```python
from rapidvlm_ocr import EngineType, ModelName, RapidVLMOCR, TaskType

model_path = "models/Qianfan-OCR"
app = RapidVLMOCR(
    model_name=ModelName.QIANFAN_OCR,
    model_path=model_path,
    engine=EngineType.VLLM,
)

result = app(
    task_type=TaskType.DOCUMENT_PARSING,
    input_path="tests/test_files/test.pdf",
    batch_size=4,
)

print(result)
```

说明：

- 公开接口参数已统一为 `input_path` / `input_paths`。
- `app(input_path="xxx.pdf")` 会使用 `pypdfium2` 将 PDF 按页渲染为图像，并默认走批处理。
- `run()` 只用于单个图像输入；PDF 请使用 `app(...)` 或 `run_batch(...)`。
