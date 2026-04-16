# coding=utf-8
# Filename:    searcher.py
# Author:      ZENGGUANRONG
# Date:        2023-12-12
# description: 核心检索器

import json,requests,copy
import numpy as np
from loguru import logger
from src.models.vec_model.vec_model import VectorizeModel
from src.utils.device import resolve_device
from src.searcher.vec_searcher.backend_factory import create_vec_searcher, normalize_vec_db_type

class Searcher:
    def __init__(self, model_path, vec_search_path, vec_db_type="lancedb"):
        device = resolve_device(None)
        self.vec_db_type = normalize_vec_db_type(vec_db_type)

        self.vec_model = VectorizeModel(model_path, device=device)
        logger.info("load vec_model done")

        self.vec_searcher = create_vec_searcher(self.vec_db_type)
        logger.info("vec db backend: {}".format(self.vec_db_type))
            
        self.vec_searcher.load(vec_search_path)
        logger.info("load vec_searcher done")

    def rank(self, query, recall_result):
        rank_result = []
        for idx in range(len(recall_result)):
            doc = recall_result[idx][1]
            if isinstance(doc, dict):
                doc_text = doc.get("text", "")
            else:
                doc_text = doc[0] if len(doc) > 0 else ""
            new_sim = self.vec_model.predict_sim(query, doc_text)
            rank_item = copy.deepcopy(recall_result[idx])
            rank_item.append(new_sim)
            rank_result.append(copy.deepcopy(rank_item))
        rank_result.sort(key=lambda x: x[3], reverse=True)
        return rank_result
    
    def search(self, query, nums=3):
        # logger.info("request: {}".format(query))

        q_vec = self.vec_model.predict_vec(query).cpu().numpy()

        recall_result = self.vec_searcher.search(q_vec, nums)

        rank_result = self.rank(query, recall_result)
        # rank_result = list(filter(lambda x:x[4] > 0.8, rank_result))

        # logger.info("response: {}".format(rank_result))
        return rank_result

if __name__ == "__main__":
    from config.project_config import VEC_MODEL_PATH, VEC_INDEX_DATA

    searcher = Searcher(VEC_MODEL_PATH, VEC_INDEX_DATA)
    q = "什么人不能吃花生"
    print(searcher.search(q))
