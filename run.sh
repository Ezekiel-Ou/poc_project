#!/usr/bin/env bash
# 项目统一入口脚本
# 用法：./run.sh <command> [args...]
set -euo pipefail

# 总是以项目根目录作为 CWD，避免脚本在任何地方执行都能工作
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 自动加载 default.env（如果存在）
if [[ -f "$PROJECT_ROOT/default.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/default.env"
    set +a
fi

PYTHON=${PYTHON:-python}

usage() {
    cat <<'EOF'
用法: ./run.sh <command> [args...]

命令:
  download-models               下载向量模型 (SimCSE)
  build-index [lancedb|faiss]   构建向量索引，默认 lancedb
  predict "<query>"             单条推理
  predict-batch <input.jsonl> [output.jsonl]
                                批量推理 jsonl
  eval                          跑测试集评估（使用 TEST_DATA_PATH）
  help                          显示本帮助

环境变量（可通过 default.env 或命令行前置设置）:
  VEC_DB_TYPE           lancedb | faiss (默认 lancedb)
  BATCH_SIZE            批量推理 batch 大小 (默认 8)
  VEC_MODEL_PATH        向量模型路径
  LLM_PATH              LLM 模型路径
  VEC_INDEX_DATA        索引名
  TEST_DATA_PATH        测试集 jsonl 路径
  OUTPUT_DATA_PATH      评估结果输出路径

示例:
  # 首次环境准备
  ./run.sh download-models
  ./run.sh build-index lancedb

  # 切到 FAISS 后端并重建索引
  ./run.sh build-index faiss

  # 单条推理
  ./run.sh predict "帮我查一下订单状态"

  # 批量推理（输出自动写到 *_pred.jsonl）
  ./run.sh predict-batch llm_classification/data/intent_data/test_set_intent_few.jsonl

  # 指定输出路径
  ./run.sh predict-batch input.jsonl output.jsonl

  # 跑评估，手动调 batch 大小
  BATCH_SIZE=4 ./run.sh eval
  BATCH_SIZE=16 ./run.sh eval

  # 切换 DB 后端（临时，不改 default.env）
  VEC_DB_TYPE=faiss ./run.sh eval
EOF
}

CMD=${1:-help}
shift || true

case "$CMD" in
    download-models)
        $PYTHON download_models.py "$@"
        ;;
    build-index)
        if [[ $# -ge 1 ]]; then
            export VEC_DB_TYPE="$1"
            shift
        fi
        $PYTHON llm_classification/script/build_vec_index.py "$@"
        ;;
    predict)
        if [[ $# -lt 1 ]]; then
            echo "错误：predict 需要一条 query" >&2
            echo "示例: ./run.sh predict \"帮我查一下订单状态\"" >&2
            exit 1
        fi
        $PYTHON llm_classification/src/classifier.py --text "$*"
        ;;
    predict-batch)
        if [[ $# -lt 1 ]]; then
            echo "错误：predict-batch 需要 input.jsonl 路径" >&2
            echo "示例: ./run.sh predict-batch input.jsonl [output.jsonl]" >&2
            exit 1
        fi
        INPUT="$1"; shift
        if [[ $# -ge 1 ]]; then
            $PYTHON llm_classification/src/classifier.py --input_jsonl "$INPUT" --output_jsonl "$1"
        else
            $PYTHON llm_classification/src/classifier.py --input_jsonl "$INPUT"
        fi
        ;;
    eval)
        $PYTHON llm_classification/script/run_intent_cases.py "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "未知命令: $CMD" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac
