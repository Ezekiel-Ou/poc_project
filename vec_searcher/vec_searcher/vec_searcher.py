import os, json
from pathlib import Path
from loguru import logger
from vec_searcher.vec_index import VecIndex

class VecSearcher:
    def __init__(self):
        self.invert_index = VecIndex() # 检索倒排，使用的是索引是VecIndex
        self.forward_index = [] # 检索正排，实质上只是个list，通过ID获取对应的内容
        vec_searcher_root = Path(__file__).resolve().parents[1]
        index_root = Path(os.getenv("VEC_INDEX_ROOT", str(vec_searcher_root / "data" / "index")))
        self.INDEX_FOLDER_PATH_TEMPLATE = str(index_root / "{}")

    def build(self, index_dim, index_name):
        self.index_name = index_name
        self.index_folder_path = self.INDEX_FOLDER_PATH_TEMPLATE.format(index_name)
        os.makedirs(self.index_folder_path, exist_ok=True)

        self.invert_index = VecIndex()
        self.invert_index.build(index_dim)

        self.forward_index = []
    
    def insert(self, vec, doc):
        self.invert_index.insert(vec)
        # self.invert_index.batch_insert(vecs)

        self.forward_index.append(doc)
    
    def save(self):
        with open(self.index_folder_path + "/forward_index.txt", "w", encoding="utf8") as f:
            for data in self.forward_index:
                f.write("{}\n".format(json.dumps(data, ensure_ascii=False)))

        self.invert_index.save(self.index_folder_path + "/invert_index.faiss")
    
    def load(self, index_name):
        self.index_name = index_name
        self.index_folder_path = self.INDEX_FOLDER_PATH_TEMPLATE.format(index_name)

        self.invert_index = VecIndex()
        self.invert_index.load(self.index_folder_path + "/invert_index.faiss")

        self.forward_index = []
        with open(self.index_folder_path + "/forward_index.txt", encoding="utf8") as f:
            for line in f:
                self.forward_index.append(json.loads(line.strip()))
    
    def search(self, vecs, nums = 5):
        search_res = self.invert_index.search(vecs, nums)
        recall_list = []
        for idx in range(nums):
            # recall_list_idx, recall_list_detail, distance
            recall_list.append([search_res[1][0][idx], self.forward_index[search_res[1][0][idx]], search_res[0][0][idx]])
        # recall_list = list(filter(lambda x: x[2] < 100, result))

        return recall_list