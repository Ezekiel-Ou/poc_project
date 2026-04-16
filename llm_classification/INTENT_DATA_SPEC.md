# 意图分类数据规范（Intent Classification Spec）

本文档定义意图分类任务的数据格式、标签体系和标注规则，目标是让项目可迁移到任意业务场景（客服、助手、工单、运营、办公自动化等）。

## 1. 任务定义

- 输入：用户原始文本（query）
- 输出：一个最合适的意图标签（单标签分类）
- 兜底：当文本不清晰或超出已知标签体系时，输出 `unknown`（拒识）

建议默认采用单标签模式（与当前召回+LLM判别流程一致），后续如有需要再扩展多标签。

## 2. 标签体系设计（推荐）

为保证可扩展性，采用两层结构：

- L1：意图大类（做什么）
- L2：具体意图（具体要做的动作）

### 2.1 常见 L1 大类示例

- `qa_info`：信息问答（解释、介绍、知识查询）
- `operate_task`：执行操作（创建、修改、删除、查询系统对象）
- `analysis_report`：分析总结（统计、对比、归因、报告生成）
- `plan_generate`：内容生成（方案、文案、邮件、脚本）
- `workflow_assist`：流程协助（审批、提醒、跟进、状态查询）
- `chitchat`：闲聊寒暄
- `unknown`：拒识/不在范围

### 2.2 L2 具体意图示例（可按业务定制）

- `create_ticket`：创建工单
- `query_order_status`：查询订单状态
- `cancel_order`：取消订单
- `book_meeting`：预约会议
- `generate_weekly_report`：生成周报
- `summarize_document`：总结文档
- `faq_refund_policy`：咨询退款规则

## 3. 标签命名规范

- 统一小写蛇形：`[a-z0-9_]+`，例如 `query_order_status`
- 标签语义应是可执行动作或稳定目的，不要混入情绪和措辞
- 不使用重叠标签（例如 `query_status` 与 `query_order_status` 同时存在会冲突）
- 所有线上预测标签必须在标签字典中有定义

## 4. 数据文件格式（必须）

统一使用 `jsonl`，每行一个 JSON 对象。

### 4.1 训练/索引样本格式

```json
{"id":"d1","text":"帮我查一下订单123现在到哪了","label":"query_order_status","label_name":"查询订单状态","l1":"operate_task","meta":{"source":"app_chat","lang":"zh"}}
{"id":"d2","text":"给我生成本周销售总结","label":"generate_weekly_report","label_name":"生成周报","l1":"analysis_report","meta":{"source":"web"}}
```

字段说明：

- `id`：样本唯一ID（字符串）
- `text`：用户原始文本
- `label`：L2标签（训练与评估主标签）
- `label_name`：标签中文名（可读）
- `l1`：L1大类（可选但强烈建议）
- `meta`：扩展字段（可选）

### 4.2 推理输入格式

```json
{"id":"q1","text":"把明天下午和客户的会约一下"}
{"id":"q2","text":"退款规则是什么"}
```

### 4.3 推理输出格式

```json
{"id":"q1","text":"把明天下午和客户的会约一下","prediction":"book_meeting","prediction_name":"预约会议","confidence":0.78}
{"id":"q2","text":"退款规则是什么","prediction":"faq_refund_policy","prediction_name":"退款规则咨询","confidence":0.73}
```

备注：`confidence` 可选。如果当前实现没有稳定置信度，可先不输出或输出 `null`。

## 5. 标签定义文件（class_def）

建议使用 `tsv`，格式：`label<TAB>definition`。

示例：

```tsv
query_order_status	用户希望查询订单处理进度、物流状态或当前节点
cancel_order	用户希望撤销未完成订单
book_meeting	用户希望安排、变更或取消会议日程
unknown	语义不明或不在已定义业务范围内
```

要求：

- 每个 `label` 必须有定义
- 定义要“边界清晰”，写出包含范围与排除范围

## 6. 标注规则（关键）

- 单标签优先：一句话只标一个“主要意图”
- 多意图句处理：按主诉求优先（若无法判定主次，标 `unknown` 并入复核池）
- 上下文缺失：仅根据当前句无法判断时标 `unknown`
- 不用情感/语气作为标签依据（如“急、烦、谢谢”）
- 对同义表达保持一致标注（建立同义短语清单）

## 7. 质量门槛

- 标签分布：头部标签不超过总样本 35%（建议）
- 每个标签最少样本数：`>= 50`（冷启动阶段可 `>= 20`）
- 标注一致性：抽样双标一致率 `>= 90%`
- 去重：`text` 完全重复样本需去重或保留一条

## 8. 评估与切分

- 数据切分：`train/valid/test = 8/1/1`（按标签分层）
- 核心指标：Macro-F1、每类 Recall、`unknown` 召回率
- 上线前必须检查：
  - 高频标签混淆矩阵
  - `unknown` 误分到具体标签的比例

## 9. 与当前项目的映射建议

当前项目可直接沿用“向量召回候选 + LLM从候选里选1类”的流程，推荐改造点：

- 索引入库时统一使用本规范的 `jsonl` 字段
- 召回结果统一输出 `text/label/label_name/meta`
- prompt 中的 `<options>` 使用 `label`，`<options_detail>` 使用 class_def 的定义
- 当 LLM 输出不在候选集合内时，强制回退 `unknown`

## 10. 最小可用标签集（MVP）建议

如果你先做“干什么事情”的基础意图分类，可先用以下 8 类：

- `query_info`（信息查询）
- `create_task`（创建任务/工单/日程）
- `update_task`（修改任务/订单/配置）
- `cancel_task`（取消/删除）
- `status_check`（进度/状态查询）
- `report_generate`（总结/报表生成）
- `smalltalk`（闲聊）
- `unknown`（拒识）

后续根据业务数据再把 `status_check`、`create_task` 等细分到垂直标签。
