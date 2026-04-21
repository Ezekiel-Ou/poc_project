# coding=utf-8
# Filename:    classifier.py
# Author:      ZENGGUANRONG
# Date:        2024-06-25
# description: 分类器主函数

import copy
import json
import sys
from pathlib import Path
from loguru import logger
from tqdm import tqdm

LLM_CLASSIFICATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(LLM_CLASSIFICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_CLASSIFICATION_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (VEC_INDEX_DATA, VEC_MODEL_PATH,
                                   LLM_CONFIG, LLM_PATH, PROMPT_TEMPLATE, CLASS_DEF_PATH, VEC_DB_TYPE,
                                   BATCH_SIZE)
from src.searcher.searcher import Searcher
from src.models.llm.llm_model import QWen3Model
from src.utils.data_processing import load_class_def
from src.utils.device import resolve_device

class VecLlmClassifier:
    def __init__(self) -> None:
        self.device = resolve_device(None)

        self.searcher = Searcher(VEC_MODEL_PATH, VEC_INDEX_DATA, vec_db_type=VEC_DB_TYPE)
        self.llm = QWen3Model(LLM_PATH, LLM_CONFIG, self.device)
        self.PROMPT_TEMPLATE = PROMPT_TEMPLATE
        self.class_def = load_class_def(CLASS_DEF_PATH)

    def _build_prompt(self, query, recall_result):
        request_prompt = copy.deepcopy(self.PROMPT_TEMPLATE)
        examples = []
        options = []
        options_detail = []
        for item in recall_result:
            doc = item[1]

            if isinstance(doc, dict):
                doc_text = str(doc.get("text", "")).strip()
                label = str(doc.get("label_name") or doc.get("label") or "").strip()
            else:
                doc_text = doc[0] if len(doc) > 0 else ""
                label = ""
                if len(doc) > 1 and isinstance(doc[1], list) and len(doc[1]) > 5:
                    label = doc[1][5]

            if not label:
                continue

            tmp_examples = "——".join([doc_text, label])
            if tmp_examples not in examples:
                examples.append(tmp_examples)

            if label not in options:
                options.append(label)
                opt_detail = self.class_def.get(label, "")
                opt_detail_str = "：".join(["【" + label + "】", opt_detail]) if opt_detail else "【{}】".format(label)
                options_detail.append(opt_detail_str)

        examples_str = "\n".join(examples)
        options_str = "，".join(options)
        options_detail_str = "\n".join(options_detail)

        request_prompt = request_prompt.replace("<examples>", examples_str)
        request_prompt = request_prompt.replace("<options>", options_str)
        request_prompt = request_prompt.replace("<options_detail>", options_detail_str)
        request_prompt = request_prompt.replace("<query>", query)

        return request_prompt, options

    def _parse_response(self, llm_response, options):
        if not options:
            return "拒识"
        for option in options:
            if option in llm_response:
                return option
        return "拒识"

    def predict_batch(self, queries):
        if not queries:
            return []

        prompts = []
        options_per_query = []
        fallback_mask = []
        for q in queries:
            logger.info("request: {}".format(q))
            recall_result = self.searcher.search(q, nums=5)
            prompt, options = self._build_prompt(q, recall_result)
            prompts.append(prompt)
            options_per_query.append(options)
            fallback_mask.append(not options)
            if not options:
                logger.warning("empty recall labels for query: {}, fallback to 拒识".format(q))

        # 只对有候选类目的 query 送 LLM，节省算力
        to_infer_idx = [i for i, skip in enumerate(fallback_mask) if not skip]
        results = ["拒识"] * len(queries)
        if to_infer_idx:
            batch_prompts = [prompts[i] for i in to_infer_idx]
            llm_responses = self.llm.predict_batch(batch_prompts)
            for idx, resp in zip(to_infer_idx, llm_responses):
                results[idx] = self._parse_response(resp, options_per_query[idx])

        for q, r in zip(queries, results):
            logger.info("response: {} -> {}".format(q, r))
        return results

    def predict(self, query):
        return self.predict_batch([query])[0]


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def run_jsonl_inference(classifier, input_jsonl_path, output_jsonl_path, batch_size=None):
    batch_size = batch_size or BATCH_SIZE
    items = []
    with open(input_jsonl_path, "r", encoding="utf8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item.get("text", "")
            if not text:
                logger.warning("skip line {}: missing field 'text'".format(line_no))
                continue
            items.append(text)

    results = []
    for batch in tqdm(list(_chunked(items, batch_size)), desc="BATCH INFER"):
        preds = classifier.predict_batch(batch)
        for text, pred in zip(batch, preds):
            results.append({"text": text, "prediction": pred})

    with open(output_jsonl_path, "w", encoding="utf8") as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("jsonl inference done: {} items -> {}".format(len(results), output_jsonl_path))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="", help="single input text")
    parser.add_argument("--input_jsonl", type=str, default="", help="jsonl input path, each line like {\"text\": \"...\"}")
    parser.add_argument("--output_jsonl", type=str, default="", help="jsonl output path")
    parser.add_argument("--batch_size", type=int, default=0, help="override BATCH_SIZE env var")
    args, unknown_args = parser.parse_known_args()

    vlc = VecLlmClassifier()

    if args.input_jsonl:
        output_path = args.output_jsonl or args.input_jsonl.replace(".jsonl", "_pred.jsonl")
        run_jsonl_inference(vlc, args.input_jsonl, output_path, batch_size=args.batch_size or None)
    elif args.text:
        logger.info(vlc.predict(args.text))
    elif len(unknown_args) > 0:
        # backward compatible with previous positional text input
        logger.info(vlc.predict("".join(unknown_args)))
