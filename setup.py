# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path
from typing import List, Union

import setuptools
from get_pypi_latest_version import GetPyPiLatestVersion


def read_txt(txt_path: Union[Path, str]) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        data = [v.rstrip("\n") for v in f]
    return data


def get_readme():
    root_dir = Path(__file__).resolve().parent
    readme_path = str(root_dir / "docs" / "doc_whl_rapidvlm_ocr.md")
    print(readme_path)
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
    return readme


MODULE_NAME = "rapidvlm_ocr"

obtainer = GetPyPiLatestVersion()
try:
    latest_version = obtainer(MODULE_NAME)
except Exception as e:
    latest_version = "0.0.0"
VERSION_NUM = obtainer.version_add_one(latest_version, add_patch=True)

if len(sys.argv) > 2:
    match_str = " ".join(sys.argv[2:])
    matched_versions = obtainer.extract_version(match_str)
    if matched_versions:
        VERSION_NUM = matched_versions
sys.argv = sys.argv[:2]

setuptools.setup(
    name=MODULE_NAME,
    version=VERSION_NUM,
    platforms="Any",
    description="End-to-end VLM-OCR unified inference framework accelerated by vLLM, featuring multi-model compatibility, a unified set of interfaces, and out-of-the-box usability.",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    author="SWHL",
    author_email="liekkaskono@163.com",
    url="https://github.com/RapidAI/RapidVLM-OCR",
    license="Apache-2.0",
    include_package_data=True,
    install_requires=read_txt("requirements.txt"),
    packages=setuptools.find_packages(include=[MODULE_NAME, f"{MODULE_NAME}.*"]),
    package_data={MODULE_NAME: ["*.onnx", "*.yaml", "*.txt", "config/*.yaml"]},
    entry_points={
        "console_scripts": [
            "rapidvlm_ocr=rapidvlm_ocr.cli:main",
        ]
    },
    keywords=[
        "ocr,text_detection,text_recognition,db,onnxruntime,paddleocr,openvino,rapidvlm_ocr"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    python_requires=">=3.10,<4",
)
