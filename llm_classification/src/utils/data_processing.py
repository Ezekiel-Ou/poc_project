# coding=utf-8
# Filename:    data_processing.py
# Author:      ZENGGUANRONG
# Date:        2024-06-25
# description: 数据处理函数

import json


def normalize_record(record, idx=None):
    """标准化样本结构，输出统一 dict。"""
    text = str(record.get("text", "")).strip()
    if not text:
        return None

    item_id = record.get("id")
    if item_id is None:
        item_id = str(idx) if idx is not None else ""

    label = record.get("label")
    label_name = record.get("label_name")
    if not label_name:
        label_name = label

    normalized = {
        "id": str(item_id),
        "text": text,
        "label": label,
        "label_name": label_name,
        "l1": record.get("l1"),
        "meta": record.get("meta", {}),
    }
    return normalized


def load_jsonl_data(path):
    """加载符合 INTENT_DATA_SPEC 的 jsonl 数据。"""
    source_data = []
    with open(path, encoding="utf8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            normalized = normalize_record(data, idx=line_no)
            if normalized is not None:
                source_data.append(normalized)
    return source_data


def load_legacy_delimited_data(path):
    """
    兼容旧的头条格式，输出统一结构。
    原始格式：新闻ID_!_分类code_!_分类名称_!_标题_!_关键词
    """
    source_data = []
    with open(path, encoding="utf8") as f:
        for line_no, line in enumerate(f, start=1):
            ll = line.strip().split("_!_")
            if len(ll) < 4:
                continue
            record = {
                "id": ll[0],
                "text": ll[3],
                "label": ll[1] if len(ll) > 1 else None,
                "label_name": ll[2] if len(ll) > 2 else None,
                "l1": None,
                "meta": {
                    "keywords": ll[4] if len(ll) > 4 else "",
                    "raw": ll,
                    "line_no": line_no,
                },
            }
            normalized = normalize_record(record, idx=line_no)
            if normalized is not None:
                source_data.append(normalized)
    return source_data


# backward compatible alias
load_toutiao_data = load_legacy_delimited_data


def load_class_def(path):
    source_data = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ll = line.split("\t", maxsplit=1)
            if len(ll) == 1:
                source_data[ll[0]] = ""
            else:
                source_data[ll[0]] = ll[1]
    return source_data
