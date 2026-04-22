# poc_project

通用文本意图分类项目，当前实现是“两阶段”流程：

1. 先用训练锚点数据构建向量索引
2. 再对待预测文本做“向量召回 + LLM 分类”

## 当前仓库结构

- `build_index.sh`：第一阶段入口，读取 `train.jsonl` 构建向量索引
- `run.sh`：第二阶段入口，读取输入 `jsonl` 批量推理并输出预测结果
- `default.env`：默认环境变量
- `llm_classification/`：主流程代码
- `models/`：本地模型目录

## 当前实现状态

代码层面的默认配置来自 [default.env](/Users/ezekiel/code/PY_codes/poc_project/default.env:1)：

- `VEC_DB_TYPE=lancedb`
- `VEC_MODEL_PATH=./models/simcse-chinese-roberta-wwm-ext`
- `LLM_PATH=./models/qwen3-1.7b`
- `VEC_INDEX_DATA=vec_index_intent_few`

但当前仓库里已经存在的索引文件是 FAISS 格式：

- `llm_classification/data/index/vec_index_intent_few/invert_index.faiss`
- `llm_classification/data/index/vec_index_intent_few/forward_index.txt`

所以如果直接复用这份现成索引，建议运行命令时显式带上：

```bash
VEC_DB_TYPE=faiss
```

如果你希望默认就用 FAISS，可以把 `default.env` 里的 `VEC_DB_TYPE` 改成 `faiss`。

## 环境准备

```bash
conda create -n poc_project python=3.13
conda activate poc_project
pip install -r requirements.txt
source default.env
```

## 模型目录

当前项目默认使用：

- 向量模型：`models/simcse-chinese-roberta-wwm-ext`
- LLM 模型：`models/qwen3-1.7b`

运行时真正生效的模型路径优先取 `default.env` 中的：

- `VEC_MODEL_PATH`
- `LLM_PATH`

## 两阶段运行逻辑

### 第一阶段：构建向量索引

`build_index.sh` 读取训练锚点数据集，把每条样本向量化后写入索引。

命令：

```bash
VEC_DB_TYPE=faiss ./build_index.sh llm_classification/data/intent_data/train.jsonl
```

可选写法：

```bash
VEC_DB_TYPE=faiss ./build_index.sh \
  llm_classification/data/intent_data/train.jsonl \
  vec_index_intent_few \
  llm_classification/data/intent_data/test_set_intent_few.jsonl
```

参数含义：

- 第 1 个参数：训练锚点文件路径
- 第 2 个参数：索引名，可选，默认取 `VEC_INDEX_DATA`
- 第 3 个参数：测试集快照路径，可选，默认取 `TEST_DATA_PATH`

构建完成后，索引会落在：

- `llm_classification/data/index/<索引名>/`

如果后端是 FAISS，当前保存内容是：

- `invert_index.faiss`：向量索引
- `forward_index.txt`：每条向量对应的原始文本、标签等信息

### 第二阶段：批量推理

`run.sh` 读取待预测 `jsonl`，对每条文本先做向量召回，再交给 LLM 做最终分类。

命令：

```bash
VEC_DB_TYPE=faiss ./run.sh \
  llm_classification/data/intent_data/test_set_intent_few.jsonl \
  llm_classification/data/intent_data/out_put.jsonl
```

如果不传输出文件，默认会生成：

```bash
<输入文件名去掉 .jsonl 后>_pred.jsonl
```

例如：

```bash
VEC_DB_TYPE=faiss ./run.sh llm_classification/data/intent_data/test_set_intent_few.jsonl
```

会默认输出到：

- `llm_classification/data/intent_data/test_set_intent_few_pred.jsonl`

## 数据流

完整链路如下：

```text
train.jsonl
  -> build_index.sh
  -> 向量化
  -> 写入 llm_classification/data/index/<index_name>/

test.jsonl
  -> run.sh
  -> 查询向量索引召回相似锚点
  -> LLM 从候选类目中选一个结果
  -> 输出 prediction jsonl
```

## 输入输出格式

详细格式见 [INTENT_DATA_SPEC.md](/Users/ezekiel/code/PY_codes/poc_project/llm_classification/INTENT_DATA_SPEC.md)，这里列当前代码真实使用的最小格式。

### 训练/建索引输入

文件：`llm_classification/data/intent_data/train.jsonl`

每行一个 JSON，至少需要：

- `text`
- `label`

建议完整字段：

```json
{"id":"tr_001","text":"帮我查一下订单状态","label":"status_check","label_name":"status_check","l1":"operate_task","meta":{"source":"demo"}}
```

### 推理输入

`run.sh` 当前只强依赖 `text` 字段：

```json
{"text":"帮我查一下订单状态"}
{"text":"我要退款"}
```

### 推理输出

当前实现的输出字段只有：

- `text`
- `prediction`

示例：

```json
{"text":"帮我查一下订单状态","prediction":"status_check"}
{"text":"我要退款","prediction":"refund"}
```

注意：

- 当前不会保留输入里的 `id`
- 当前不会输出 `confidence`
- 当前不会输出 `prediction_name`

## 当前分类逻辑

当前主流程在 [classifier.py](/Users/ezekiel/code/PY_codes/poc_project/llm_classification/src/classifier.py:25)：

1. 先用向量模型对输入 query 编码
2. 去向量索引里召回最相近的 5 条锚点
3. 从召回结果中整理候选标签和参考样本
4. 将候选标签、类目定义、参考样本拼到 prompt 中
5. 交给 `QWen3Model` 生成结果
6. 如果模型输出里包含某个候选标签，则返回该标签；否则返回 `拒识`

注意：

- 当前兜底标签是 `拒识`
- 不是 `unknown`

## 更换 LLM

当前分类器写死实例化的是 `QWen3Model`，位置在：

- [llm_model.py](/Users/ezekiel/code/PY_codes/poc_project/llm_classification/src/models/llm/llm_model.py:14)
- [classifier.py](/Users/ezekiel/code/PY_codes/poc_project/llm_classification/src/classifier.py:34)

如果新模型和当前加载方式兼容，通常只需要改：

- `default.env` 里的 `LLM_PATH`

如果新模型加载方式不同，还需要同时修改：

- `llm_classification/src/models/llm/llm_model.py`
- `llm_classification/src/classifier.py`

## 常见问题

- 推理时报索引加载失败：先确认 `VEC_DB_TYPE` 是否和现有索引格式一致
- 改了 `default.env` 不生效：需要重新执行 `source default.env`
- 输出文件没有立刻出现：批量推理会在整批处理完成后统一写文件
