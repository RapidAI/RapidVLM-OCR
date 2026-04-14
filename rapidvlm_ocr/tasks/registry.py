# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from ..schema.enums import TaskType
from .document_parse import DocumentParseTask
from .kie import KIETask
from .ocr import OCRTask

TASK_REGISTRY = {
    TaskType.TEXT_EXTRACTION: OCRTask(),
    TaskType.FORMULA_PARSING: OCRTask(),
    TaskType.TABLE_PARSING: OCRTask(),
    TaskType.DOCUMENT_PARSING: DocumentParseTask(),
    TaskType.MULTI_SCENE_REC: OCRTask(),
    TaskType.KIE: KIETask(),
}


def get_task_handler(task_type: TaskType):
    task_handler = TASK_REGISTRY.get(task_type)
    if task_handler is None:
        raise KeyError(f"No task handler registered for task: {task_type.value}")
    return task_handler
