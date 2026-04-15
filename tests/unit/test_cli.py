# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from rapidvlm_ocr import cli


def test_cli_check_returns_zero_and_prints_success(capsys) -> None:
    exit_code = cli.main(["check"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "rapidvlm_ocr check passed" in captured.out
    assert "app: RapidVLMOCR" in captured.out


def test_cli_check_returns_one_when_dependency_import_fails(monkeypatch, capsys) -> None:
    original_import_module = cli.importlib.import_module

    def fake_import_module(name: str):
        if name == "pypdfium2":
            raise ModuleNotFoundError("No module named 'pypdfium2'")
        return original_import_module(name)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    exit_code = cli.main(["check"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "rapidvlm_ocr check failed" in captured.err
    assert "pypdfium2" in captured.err
