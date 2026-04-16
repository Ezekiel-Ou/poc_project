# coding=utf-8
# Filename:    lancedb_index.py
# Author:      ZENGGUANRONG
# Date:        2025-01-01
# description: 向量召回索引-LanceDB

import lancedb
import numpy as np
from loguru import logger


class LanceDBIndex:
    def __init__(self) -> None:
        self.db = None
        self.table = None
        self.index_name = ""
    
    def build(self, index_path, index_name, data_schema=None):
        """初始化 LanceDB 实例和表"""
        self.index_name = index_name
        # LanceDB 支持本地 DB 路径
        self.db = lancedb.connect(index_path)
        logger.info(f"LanceDB connected to {index_path}")
    
    def insert(self, vec, doc_dict):
        """
        插入单条数据：向量 + 元数据
        Args:
            vec: numpy array，embedding 向量
            doc_dict: dict，包含 text、metadata 等
        """
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim > 1:
            vec = vec[0]

        if self.table is None:
            # 首次插入时创建表
            record = {"vector": vec, **doc_dict}
            self.table = self.db.create_table(self.index_name, data=[record], mode="overwrite")
        else:
            record = {"vector": vec, **doc_dict}
            self.table.add([record])
    
    def batch_insert(self, vecs, docs):
        """批量插入数据"""
        records = []
        for vec, doc in zip(vecs, docs):
            records.append({"vector": vec, **doc})
        
        if self.table is None:
            self.table = self.db.create_table(self.index_name, data=records, mode="overwrite")
        else:
            self.table.add(records)
    
    def load(self, index_path, index_name):
        """从本地加载 LanceDB"""
        self.index_name = index_name
        self.db = lancedb.connect(index_path)
        self.table = self.db.open_table(index_name)
        logger.info(f"Loaded LanceDB table {index_name} from {index_path}")
    
    def save(self):
        """LanceDB 自动持久化，无需手动保存"""
        logger.info(f"LanceDB table {self.index_name} auto-persisted")
    
    def search(self, vec, num):
        """
        相似度搜索
        Args:
            vec: numpy array，查询向量
            num: int，返回结果数
        
        Returns:
            results: list of dict，每条包含 id, text, distance 等
        """
        if self.table is None:
            logger.warning("table not initialized")
            return []
        
        results = self.table.search(vec).limit(num).to_list()
        return results
