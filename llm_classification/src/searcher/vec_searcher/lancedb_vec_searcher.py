import os, json
from pathlib import Path
from loguru import logger
from src.searcher.vec_searcher.lancedb_index import LanceDBIndex


class LanceDBVecSearcher:
    """LanceDB 版本的向量检索器，接口兼容 VecSearcher"""
    def __init__(self):
        self.index = LanceDBIndex()  # 使用 LanceDB 后端
        self.index_name = ""
        llm_classification_root = Path(__file__).resolve().parents[3]
        index_root = Path(os.getenv("VEC_INDEX_ROOT", str(llm_classification_root / "data" / "index")))
        self.INDEX_FOLDER_PATH_TEMPLATE = str(index_root / "{}")

    def build(self, index_dim, index_name):
        """初始化新索引"""
        self.index_name = index_name
        self.index_folder_path = self.INDEX_FOLDER_PATH_TEMPLATE.format(index_name)
        os.makedirs(self.index_folder_path, exist_ok=True)
        
        # LanceDB 初始化
        self.index.build(self.index_folder_path, index_name)
        logger.info(f"LanceDB index built at {self.index_folder_path}")
    
    def insert(self, vec, doc):
        """
        插入单条数据
        Args:
            vec: numpy array，向量
            doc: list or dict，文档数据
        """
        # 统一把 doc 转成 dict 格式
        if isinstance(doc, list):
            doc_dict = {
                "id": str(hash(str(doc)))[:10],
                "text": doc[0] if len(doc) > 0 else "",
                "label": None,
                "label_name": None,
                "l1": None,
                "metadata": json.dumps({"raw": doc}, ensure_ascii=False),
            }
        else:
            meta = doc.get("meta", {})
            doc_dict = {
                "id": str(doc.get("id", str(hash(str(doc)))[:10])),
                "text": doc.get("text", ""),
                "label": doc.get("label"),
                "label_name": doc.get("label_name") or doc.get("label"),
                "l1": doc.get("l1"),
                "metadata": json.dumps(meta, ensure_ascii=False),
            }
        
        self.index.insert(vec, doc_dict)
    
    def save(self):
        """保存索引（LanceDB 自动持久化）"""
        self.index.save()
    
    def load(self, index_name):
        """加载已存在的索引"""
        self.index_name = index_name
        self.index_folder_path = self.INDEX_FOLDER_PATH_TEMPLATE.format(index_name)
        self.index.load(self.index_folder_path, index_name)
        logger.info(f"LanceDB index loaded from {self.index_folder_path}")
    
    def search(self, vecs, nums=5):
        """
        相似度搜索
        Args:
            vecs: numpy array，形状 (1, dim)，查询向量
            nums: int，返回结果数
        
        Returns:
            recall_list: list，每条格式 [item_id, item_metadata, distance]
        """
        results = self.index.search(vecs, nums)
        
        recall_list = []
        for idx, item in enumerate(results):
            item_id = item.get("id", idx)
            if isinstance(item.get("metadata"), str):
                try:
                    item_meta = json.loads(item.get("metadata", "{}"))
                except Exception:
                    item_meta = {}
            else:
                item_meta = item.get("metadata", {})

            doc = {
                "id": item_id,
                "text": item.get("text", ""),
                "label": item.get("label"),
                "label_name": item.get("label_name") or item.get("label"),
                "l1": item.get("l1"),
                "meta": item_meta,
            }
            distance = item.get("_distance", 0.0)

            recall_list.append([item_id, doc, distance])
        
        return recall_list
