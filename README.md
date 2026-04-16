# poc_project

通用文本意图分类项目（向量召回 + LLM 分类），当前默认向量库为 **LanceDB**。

## 目录说明

- `llm_classification/`：意图分类主流程
- `models/`：本地模型目录
- `default.env`：统一环境变量配置

## 当前默认配置（已对齐代码）

项目读取 `default.env` 后，默认使用：

- 向量库：`VEC_DB_TYPE=lancedb`（可选 `faiss`）
- 向量模型路径：`./models/simcse-chinese-roberta-wwm-ext`
- LLM 模型路径：`./models/qwen3-1.7b`
- 索引名：`vec_index_intent_few`
- 索引目录：`./llm_classification/data/index`
- 标签定义：`./llm_classification/data/intent_data/class_def.tsv`
- 建索引输入：`./llm_classification/data/intent_data/train.jsonl`
- 评估输入：`./llm_classification/data/intent_data/test_set_intent_few.jsonl`

## 环境准备

```bash
conda create -n poc_project python=3.13
conda activate poc_project
pip install -r requirements.txt
```

加载环境变量（必须）：

```bash
source default.env
```

## 模型准备

### 向量模型（自动下载）

```bash
python download_models.py
```

下载后默认落盘目录：

- `models/simcse-chinese-roberta-wwm-ext`

说明：项目会从 `default.env` 的 `VEC_MODEL_PATH` 读取向量模型路径。

## 数据格式规范

详细规范见：`llm_classification/INTENT_DATA_SPEC.md`

### 1) 训练/索引输入（JSONL）

路径：`llm_classification/data/intent_data/train.jsonl`

每行一个 JSON：

```json
{"id":"tr_001","text":"帮我查一下订单状态","label":"status_check","label_name":"status_check","l1":"operate_task","meta":{"source":"demo"}}
```

最少字段：`text`（建议完整字段都提供）。

### 2) 类目定义（TSV）

路径：`llm_classification/data/intent_data/class_def.tsv`

格式：`label<TAB>definition`

示例：

```tsv
status_check	用户希望查询对象当前状态或处理进度。
```

### 3) 评估输入（JSONL）

路径：`llm_classification/data/intent_data/test_set_intent_few.jsonl`

每行一个 JSON（建议含 `text` + `label`）：

```json
{"id":"te_001","text":"查一下订单B7788的处理进度","label":"status_check"}
```

## 流程说明

### 第一步：构建向量索引

```bash
python llm_classification/script/build_vec_index.py
```

可切换向量库：

```bash
# LanceDB
VEC_DB_TYPE=lancedb python llm_classification/script/build_vec_index.py

# FAISS
VEC_DB_TYPE=faiss python llm_classification/script/build_vec_index.py
```

输出：

- LanceDB 索引目录：`llm_classification/data/index/vec_index_intent_few`
- 测试集快照：由 `TEST_DATA_PATH` 指定（默认 `llm_classification/data/intent_data/test_set_intent_few.jsonl`）

### 第二步：单条预测

```bash
python llm_classification/src/classifier.py --text "帮我查一下订单状态"
```

预测会按当前 `VEC_DB_TYPE` 自动选择同后端索引（LanceDB 或 FAISS）。

输出位置：终端日志（不写文件）

### 第三步：批量预测（JSONL 输入）

```bash
python llm_classification/src/classifier.py --input_jsonl llm_classification/data/intent_data/test_set_intent_few.jsonl --output_jsonl llm_classification/data/intent_data/test_set_intent_few_pred.jsonl
```

输出文件格式（每行一个 JSON）：

```json
{"text":"帮我查一下订单状态","prediction":"status_check"}
```

若不传 `--output_jsonl`，默认输出到输入文件同目录并追加 `_pred.jsonl`。

### 第四步：评估

```bash
python llm_classification/script/run_intent_cases.py
```

默认输出结果文件：

- `llm_classification/data/intent_data/test_set_intent_few_result.jsonl`

输出文件格式（每行一个 JSON）：

```json
{"text":"查一下订单B7788的处理进度","label":"status_check","prediction":"status_check","correct":true}
```

## 常见问题

- 改了 `default.env` 但不生效：需要重新执行 `source default.env`
- 报索引不存在：先执行 `python llm_classification/script/build_vec_index.py`
- 想切换索引名：修改 `VEC_INDEX_DATA` 并重建索引
