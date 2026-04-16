from src.searcher.vec_searcher.lancedb_vec_searcher import LanceDBVecSearcher


def normalize_vec_db_type(vec_db_type):
    db_type = (vec_db_type or "lancedb").strip().lower()
    if db_type in {"lance", "lancedb"}:
        return "lancedb"
    if db_type in {"faiss"}:
        return "faiss"
    raise ValueError("Unsupported VEC_DB_TYPE: {}. Use 'lancedb' or 'faiss'.".format(vec_db_type))


def get_vec_searcher_class(vec_db_type):
    db_type = normalize_vec_db_type(vec_db_type)
    if db_type == "lancedb":
        return LanceDBVecSearcher

    from src.searcher.vec_searcher.vec_searcher import VecSearcher

    return VecSearcher


def create_vec_searcher(vec_db_type):
    return get_vec_searcher_class(vec_db_type)()
