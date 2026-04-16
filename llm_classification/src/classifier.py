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

LLM_CLASSIFICATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(LLM_CLASSIFICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_CLASSIFICATION_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (VEC_INDEX_DATA, VEC_MODEL_PATH,
                                   LLM_CONFIG, LLM_PATH, PROMPT_TEMPLATE, CLASS_DEF_PATH, VEC_DB_TYPE)
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

    def predict(self, query):
        # 1. query预处理
        logger.info("request: {}".format(query))
        # 2. query向量召回
        recall_result = self.searcher.search(query, nums=5)
        # logger.debug(recall_result)

        # 3. 请求大模型
        # 3.1 PROMPT拼接
        request_prompt= copy.deepcopy(self.PROMPT_TEMPLATE)
        # 3.1.1 子模块拼接
        examples = []
        options = []
        options_detail = []
        for item in recall_result:
            doc = item[1]

            if isinstance(doc, dict):
                doc_text = str(doc.get("text", "")).strip()
                label = str(doc.get("label_name") or doc.get("label") or "").strip()
            else:
                # backward compatibility with old list format
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
        # options.append("拒识：含义不明或用户query所属类目不在列举内时，分为此类")
        examples_str = "\n".join(examples)
        options_str = "，".join(options)
        options_detail_str = "\n".join(options_detail)

        # 3.1.2 整体组装
        request_prompt = request_prompt.replace("<examples>", examples_str)
        request_prompt = request_prompt.replace("<options>", options_str)
        request_prompt = request_prompt.replace("<options_detail>", options_detail_str)
        request_prompt = request_prompt.replace("<query>", query)
        # logger.info(request_prompt)

        if not options:
            logger.warning("empty recall labels, fallback to 拒识")
            return "拒识"

        # 3.2 请求大模型
        llm_response = self.llm.predict(request_prompt)
        # logger.info("llm response: {}".format(llm_response))

        # 3.3 大模型结果解析
        result = "拒识"
        for option in options:
            if option in llm_response:
                result = option
                break
        # logger.info("parse result: {}".format(result))

        # 4. 返回结果
        logger.info("response: {}".format(result))
        return result


def run_jsonl_inference(classifier, input_jsonl_path, output_jsonl_path):
    results = []
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
            pred = classifier.predict(text)
            results.append({"text": text, "prediction": pred})

    with open(output_jsonl_path, "w", encoding="utf8") as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("jsonl inference done: {} items -> {}".format(len(results), output_jsonl_path))

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="", help="single input text")
    parser.add_argument("--input_jsonl", type=str, default="", help="jsonl input path, each line like {\"text\": \"...\"}")
    parser.add_argument("--output_jsonl", type=str, default="", help="jsonl output path")
    args, unknown_args = parser.parse_known_args()

    vlc = VecLlmClassifier()

    if args.input_jsonl:
        output_path = args.output_jsonl or args.input_jsonl.replace(".jsonl", "_pred.jsonl")
        run_jsonl_inference(vlc, args.input_jsonl, output_path)
    elif args.text:
        logger.info(vlc.predict(args.text))
    elif len(unknown_args) > 0:
        # backward compatible with previous positional text input
        logger.info(vlc.predict("".join(unknown_args)))

    # # 性能测试
    # from tqdm import tqdm
    # for i in tqdm(range(20), desc="warm up"):
    #     vlc.predict("感冒发烧怎么治疗")
    # for i in tqdm(range(20), desc="running speed"):
    #     vlc.predict("王阳明到底顿悟了什么？")
