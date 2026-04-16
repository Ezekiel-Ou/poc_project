# coding=utf-8
# Filename:    build_vec_index.py
# Author:      ZENGGUANRONG
# Date:        2023-12-12
# description: 构造向量索引脚本（通用格式）

import json
import os
import random
import sys
from pathlib import Path

import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

LLM_CLASSIFICATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(LLM_CLASSIFICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_CLASSIFICATION_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import VEC_DB_TYPE, VEC_MODEL_PATH
from src.models.vec_model.vec_model import VectorizeModel
from src.searcher.vec_searcher.backend_factory import get_vec_searcher_class, normalize_vec_db_type
from src.utils.data_processing import load_jsonl_data, load_legacy_delimited_data
from src.utils.device import resolve_device


def _load_source_data(source_path, data_format):
    fmt = (data_format or "auto").lower()
    if fmt == "auto":
        fmt = "jsonl" if str(source_path).endswith(".jsonl") else "legacy"

    if fmt == "jsonl":
        return load_jsonl_data(source_path)
    if fmt in {"legacy", "delimited"}:
        return load_legacy_delimited_data(source_path)
    raise ValueError("Unsupported DATA_FORMAT: {}".format(data_format))


def _few_shot_by_label(data, per_label):
    class_count = {}
    selected = []
    for item in data:
        label = item.get("label")
        if not label:
            continue
        if class_count.get(label, 0) >= per_label:
            continue
        class_count[label] = class_count.get(label, 0) + 1
        selected.append(item)
    return selected


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    llm_classification_root = project_root / "llm_classification"

    mode = os.getenv("BUILD_MODE", "FEW").upper()  # DEBUG / FEW / PRO
    version = os.getenv("DATA_VERSION", "intent")
    source_index_data_path = os.getenv(
        "SOURCE_INDEX_DATA_PATH",
        str(llm_classification_root / "data" / "intent_data" / "train.jsonl"),
    )
    data_format = os.getenv("DATA_FORMAT", "auto")  # auto / jsonl / legacy
    vec_index_data = os.getenv("VEC_INDEX_DATA", "vec_index_{}_{}".format(version, mode.lower()))
    test_data_path = os.getenv(
        "TEST_DATA_PATH",
        str(llm_classification_root / "data" / "intent_data" / "test_set_{}_{}.jsonl".format(version, mode.lower())),
    )

    vec_db_type = normalize_vec_db_type(VEC_DB_TYPE)
    logger.info("Using vector DB backend: {}".format(vec_db_type))
    vec_searcher_class = get_vec_searcher_class(vec_db_type)

    random_seed = int(os.getenv("RANDOM_SEED", "100"))
    test_size = float(os.getenv("TEST_SIZE", "0.1"))
    few_per_label = int(os.getenv("FEW_PER_LABEL", "10"))

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    device = resolve_device(None)

    vec_model = VectorizeModel(VEC_MODEL_PATH, device)
    index_dim = len(vec_model.predict_vec("你好啊")[0])

    all_data = _load_source_data(source_index_data_path, data_format)
    source_index_data = [x for x in all_data if x.get("text")]
    logger.info("load data done: {}".format(len(source_index_data)))

    if mode == "DEBUG":
        random.shuffle(source_index_data)
        source_index_data = source_index_data[:10000]
    elif mode == "FEW":
        source_index_data = _few_shot_by_label(source_index_data, few_per_label)

    if mode != "FEW":
        train_list, test_list = train_test_split(source_index_data, test_size=test_size, random_state=random_seed)
    else:
        train_list = source_index_data
        random.shuffle(all_data)
        test_list = all_data[:1000]

    vec_searcher = vec_searcher_class()
    vec_searcher.build(index_dim, vec_index_data)

    for item in tqdm(train_list, desc="INSERT INTO INDEX"):
        vec = vec_model.predict_vec(item["text"]).cpu().numpy()
        doc = {
            "id": item.get("id", ""),
            "text": item.get("text", ""),
            "label": item.get("label"),
            "label_name": item.get("label_name") or item.get("label"),
            "l1": item.get("l1"),
            "meta": item.get("meta", {}),
        }
        vec_searcher.insert(vec, doc)

    vec_searcher.save()

    os.makedirs(Path(test_data_path).parent, exist_ok=True)
    with open(test_data_path, "w", encoding="utf8") as f:
        for item in test_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
