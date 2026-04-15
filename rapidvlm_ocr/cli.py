# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence, TextIO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rapidvlm_ocr")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "check",
        help="Verify that rapidvlm_ocr and its key runtime dependencies are available.",
    )

    return parser


def run_check(stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    try:
        importlib.import_module("PIL")
        importlib.import_module("pypdfium2")
        importlib.import_module("vllm")
        package = importlib.import_module("rapidvlm_ocr")
        app = package.RapidVLMOCR(engine=package.EngineType.MOCK)
    except Exception as exc:
        stderr.write(f"rapidvlm_ocr check failed: {exc}\n")
        stderr.flush()
        return 1

    stdout.write("rapidvlm_ocr check passed\n")
    stdout.write(f"app: {app.__class__.__name__}\n")
    stdout.flush()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return run_check()

    parser.print_help()
    return 1
