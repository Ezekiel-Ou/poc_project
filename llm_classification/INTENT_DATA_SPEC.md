# 意图分类数据规范

本文档描述的是当前仓库代码实际兼容的数据格式，不写理想方案，只写当前实现真实支持的字段和行为。

## 1. 当前任务形态

当前项目是单标签意图分类：

- 输入：一条用户文本
- 输出：一个标签
- 兜底：`拒识`

主流程不是“纯 LLM 直接分类”，而是：

1. 先从向量索引中召回相似锚点
2. 再让 LLM 在候选类目中选一个标签

## 2. 当前代码使用到的数据文件

- 训练锚点数据：`llm_classification/data/intent_data/train.jsonl`
- 推理输入数据：任意 `jsonl`，例如 `test_set_intent_few.jsonl`
- 类目定义：`llm_classification/data/intent_data/class_def.tsv`

## 3. 训练/建索引数据格式

`build_index.sh` 最终调用 `build_vec_index.py`，当前建索引时真实会读取这些字段：

- `text`
- `label`
- `label_name`
- `l1`
- `meta`
- `id`

其中真正必需的是：

- `text`

但从当前逻辑上看，建议至少同时提供：

- `text`
- `label`

推荐样例：

```json
{"id":"tr_001","text":"帮我查一下订单状态","label":"status_check","label_name":"status_check","l1":"operate_task","meta":{"source":"demo"}}
```

字段说明：

- `id`：样本 ID，可选
- `text`：原始文本，必需
- `label`：标签，强烈建议提供
- `label_name`：标签展示名，可选；如果缺失，代码会回退到 `label`
- `l1`：一级类目，可选
- `meta`：扩展字段，可选

说明：

- 如果一条样本没有 `text`，建索引时会被跳过
- 如果没有 `label`，样本仍可能入索引，但后续召回到它时无法提供有效候选标签，不利于分类

## 4. 类目定义文件格式

文件：

- `llm_classification/data/intent_data/class_def.tsv`

格式：

```text
label<TAB>definition
```

示例：

```tsv
status_check	用户希望查询对象当前状态或处理进度。
refund	用户希望发起退款、查询退款规则或处理退款问题。
```

当前用途：

- 检索召回出候选标签后，代码会把这些标签对应的定义拼进 prompt

## 5. 推理输入格式

`run.sh` 最终调用 `classifier.py --input_jsonl ...`。

当前批量推理时，每行 JSON 实际只强依赖一个字段：

- `text`

最小可用样例：

```json
{"text":"请介绍一下会员权益"}
{"text":"帮我查一下订单A123现在到哪了"}
```

说明：

- 如果某一行没有 `text`，这行会被跳过
- 当前推理逻辑不会使用输入中的 `id`、`label`、`meta`

## 6. 推理输出格式

当前 `run_jsonl_inference` 的真实输出字段只有两个：

- `text`
- `prediction`

样例：

```json
{"text":"请介绍一下会员权益","prediction":"会员权益介绍"}
{"text":"帮我查一下订单A123现在到哪了","prediction":"status_check"}
```

当前不会输出：

- `id`
- `prediction_name`
- `confidence`

如果你在输入里传了 `id`，当前输出也不会原样带回。

## 7. 当前标签返回规则

在 `classifier.py` 里，返回标签的逻辑是：

1. 从召回样本里提取候选标签
2. 让 LLM 生成结果
3. 如果生成文本中包含某个候选标签，就返回这个标签
4. 如果没有匹配到任何候选标签，就返回 `拒识`

因此当前真实兜底值是：

```text
拒识
```

不是：

```text
unknown
```

## 8. 两阶段数据流

### 第一阶段：建索引

输入：

- `train.jsonl`

处理：

1. 读取每条训练样本
2. 对 `text` 做向量化
3. 将向量写入向量索引
4. 将原始文本、标签等信息写入正排文件

输出：

- `llm_classification/data/index/<index_name>/`

如果后端是 FAISS，典型文件是：

- `invert_index.faiss`
- `forward_index.txt`

### 第二阶段：批量推理

输入：

- 待预测 `jsonl`

处理：

1. 读取每条 `text`
2. 查询向量索引，召回相似锚点
3. 拼接 prompt
4. 调用 LLM 分类
5. 写出 `prediction`

输出：

- 预测结果 `jsonl`

## 9. 当前格式建议

为了和当前实现最稳地对齐，建议：

- 训练锚点文件至少包含 `text`、`label`
- 推理输入文件至少包含 `text`
- `class_def.tsv` 中为每个标签写清晰定义
- 统一保证标签值和 `class_def.tsv` 中的标签一致

## 10. 当前不支持或未实现的内容

下面这些在当前代码里没有实现，不建议文档或数据流程里默认假设已经支持：

- 多标签分类
- 置信度分数输出
- 输出保留输入 `id`
- 输出 `prediction_name`
- 兜底标签使用 `unknown`
