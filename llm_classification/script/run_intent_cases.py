import json
import os
import sys
from pathlib import Path

from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

LLM_CLASSIFICATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(LLM_CLASSIFICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_CLASSIFICATION_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import VecLlmClassifier
from src.utils.data_processing import load_jsonl_data, load_legacy_delimited_data


def load_eval_data(test_data_path, data_format="auto"):
    fmt = (data_format or "auto").lower()
    if fmt == "auto":
        fmt = "jsonl" if test_data_path.endswith(".jsonl") else "legacy"

    if fmt == "jsonl":
        return load_jsonl_data(test_data_path)
    if fmt in {"legacy", "delimited"}:
        return load_legacy_delimited_data(test_data_path)
    raise ValueError("Unsupported TEST_DATA_FORMAT: {}".format(data_format))


if __name__ == "__main__":
    version = os.getenv("EVAL_VERSION", "intent_few")
    test_data_path = os.getenv("TEST_DATA_PATH", "data/intent_data/test_set_{}.jsonl".format(version))
    output_data_path = os.getenv("OUTPUT_DATA_PATH", "data/intent_data/test_set_{}_result.jsonl".format(version))
    test_data_format = os.getenv("TEST_DATA_FORMAT", "auto")

    test_data = load_eval_data(test_data_path, test_data_format)

    vlc = VecLlmClassifier()
    gold_list = []
    pred_list = []
    labels = set()

    for item in tqdm(test_data, desc="RUNNING TEST"):
        text = item.get("text", "")
        gold_label = item.get("label_name") or item.get("label")
        if not text or not gold_label:
            continue
        pred = vlc.predict(text)
        gold_list.append(gold_label)
        pred_list.append(pred)
        labels.add(gold_label)

    labels = list(labels)
    logger.info("\n{}".format(classification_report(gold_list, pred_list, labels=labels)))
    logger.info("\n{}".format(confusion_matrix(gold_list, pred_list, labels=labels)))

    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    with open(output_data_path, "w", encoding="utf8") as fout:
        for idx in range(len(gold_list)):
            fout.write(json.dumps({
                "text": test_data[idx].get("text", ""),
                "label": gold_list[idx],
                "prediction": pred_list[idx],
                "correct": gold_list[idx] == pred_list[idx],
            }, ensure_ascii=False) + "\n")
