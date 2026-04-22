#!/usr/bin/env bash
# 批量推理入口脚本
# 用法：./run.sh <input.jsonl> [output.jsonl]
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
用法: ./run.sh <input.jsonl> [output.jsonl]

功能:
  读取输入 jsonl，执行批量推理，并输出预测 jsonl

环境变量（可通过 default.env 或命令行前置设置）:
  VEC_DB_TYPE           lancedb | faiss (默认 lancedb)
  BATCH_SIZE            批量推理 batch 大小 (默认 8)
  VEC_MODEL_PATH        向量模型路径
  LLM_PATH              LLM 模型路径
  VEC_INDEX_DATA        索引名
  TEST_DATA_PATH        测试集 jsonl 路径
  OUTPUT_DATA_PATH      评估结果输出路径

示例:
  # 自动输出到 input_pred.jsonl
  ./run.sh llm_classification/data/intent_data/test_set_intent_few.jsonl

  # 指定输出路径
  ./run.sh input.jsonl output.jsonl

  # 临时切到 FAISS 后端
  VEC_DB_TYPE=faiss ./run.sh input.jsonl output.jsonl
EOF
}

if [[ $# -lt 1 ]] || [[ "${1:-}" == "help" ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.jsonl}_pred.jsonl}"

$PYTHON llm_classification/src/classifier.py --input_jsonl "$INPUT" --output_jsonl "$OUTPUT"
