#!/usr/bin/env bash
# 向量索引构建入口脚本
# 用法：./build_index.sh <train.jsonl> [index_name] [test_snapshot.jsonl]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/default.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/default.env"
    set +a
fi

PYTHON=${PYTHON:-python}

usage() {
    cat <<'EOF'
用法: ./build_index.sh <train.jsonl> [index_name] [test_snapshot.jsonl]

功能:
  读取训练锚点 jsonl，构建并充实向量索引

参数:
  train.jsonl          锚点数据集路径
  index_name           可选，索引名，默认使用环境变量 VEC_INDEX_DATA
  test_snapshot.jsonl  可选，构建时顺手生成的测试集快照路径

示例:
  ./build_index.sh llm_classification/data/intent_data/train.jsonl
  ./build_index.sh llm_classification/data/intent_data/train.jsonl vec_index_intent_few
  ./build_index.sh llm_classification/data/intent_data/train.jsonl vec_index_intent_few llm_classification/data/intent_data/test_set_intent_few.jsonl

临时切换后端:
  VEC_DB_TYPE=faiss ./build_index.sh train.jsonl
EOF
}

if [[ $# -lt 1 ]] || [[ "${1:-}" == "help" ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

INPUT_TRAIN="$1"
INDEX_NAME="${2:-${VEC_INDEX_DATA:-vec_index_intent_few}}"
TEST_SNAPSHOT="${3:-${TEST_DATA_PATH:-llm_classification/data/intent_data/test_set_intent_few.jsonl}}"

SOURCE_INDEX_DATA_PATH="$INPUT_TRAIN" \
VEC_INDEX_DATA="$INDEX_NAME" \
TEST_DATA_PATH="$TEST_SNAPSHOT" \
$PYTHON llm_classification/script/build_vec_index.py
