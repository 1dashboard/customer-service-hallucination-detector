## Context

当前客服系统基于 LLM 生成回复，但缺乏系统性的幻觉检测机制。仅在用户投诉后被动抽查，效率低、覆盖度差。需要构建一个自动化检测工具，对比知识库（ground truth）与系统回复，识别编造、越界、遗漏等幻觉模式。

约束：
- 20 条标注数据（replies.json + ground_truth.json）作为初始数据集
- 需支持批量检测和 API 集成两种使用方式
- 主要用户：运营/审核人员（通过 Web Dashboard）、后端系统（通过 API）
- DeepSeek v3 API 作为 LLM 服务（兼容 OpenAI SDK）

## Goals / Non-Goals

**Goals:**
- 实现对 20 条样本的幻觉检测，检出率 ≥ 85%，误报率 ≤ 15%
- 提供可复用的检测管线（规则 + LLM），支持后续扩展新规则和切换 LLM
- 提供 Web Dashboard 供非技术人员使用
- 提供 REST API 供系统对接
- 自动对比 ground truth 计算评估指标

**Non-Goals:**
- 不实现实时在线拦截（本阶段仅做离线批量检测）
- 不训练/微调模型
- 不处理多语言回复
- 不接入客服系统的生产数据库

## Decisions

### 1. 两层分类体系

**选择**：检测层（面向方法）3 类 + 输出层（面向业务）8 类

**检测层（L1-L3）**：

| 检测层 | 定义 | 检测方法 | 置信度 |
|---|---|---|---|
| L1 直接矛盾 | KB 有明确数值/规则，reply 改动 | NER + 实体对齐 | HIGH |
| L2 能力越界 | KB 明确声明"无/未接入"，reply 声称执行 | 否定模式匹配 + 动作动词 | HIGH |
| L3 无据陈述 | KB 无信息/有风险提示，reply 做肯定陈述 | LLM 语义比对 | MEDIUM-LOW |

**输出层**：参数编造、政策编造、政策偏差、能力越界、优惠编造、信息编造、安全误导、信息遗漏

**原因**：检测驱动的分类使引擎实现清晰，L1/L2 规则覆盖确定性 case（约 50%），L3 送入 LLM 处理模糊 case。输出映射满足业务方的归因需求。

### 2. 双阶段 Pipeline 架构

**选择**：Stage 1 规则引擎（快速筛选）→ Stage 2 LLM（疑难判定）

```
replies → [Stage 1: Rules (L1/L2)] ──HIGH──→ 直接输出结果
                    │
                    └──LOW/UNCERTAIN──→ [Stage 2: DeepSeek (L3)] → 输出结果
```

**原因**：规则引擎零成本、毫秒级响应，能处理约一半 case。LLM 仅处理不确定 case，降低 API 开销。DeepSeek v3 兼容 OpenAI SDK，后续切换模型成本极低。

**替代方案**：全 LLM 方案（准确率可能更高但成本高、延迟大）；全规则方案（成本低但 h13、h20 等 case 漏检）。

### 3. Streamlit + FastAPI 分层架构

**选择**：FastAPI 做核心服务和 API，Streamlit 做 UI 层

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Streamlit UI │────▶│  FastAPI     │────▶│  Detection   │
│ (前端展示)    │     │  (服务层)     │     │  Engine      │
│ 8501         │     │  8000         │     │  (检测引擎)   │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   SQLite      │
                     │   (存储层)     │
                     └──────────────┘
```

**原因**：FastAPI 提供 REST API，既服务 Streamlit 前端也可独立调用。前后端分离避免未来迁移前端框架时重写业务逻辑。Streamlit 数据流通过 `requests` 调用 FastAPI，保持松耦合。

**替代方案**：纯 Streamlit（无 API 层，后续对接困难）；FastAPI + React（过度工程化，迭代慢）。

### 4. SQLite 数据库设计

**选择**：SQLite 单文件数据库，三张核心表

- `detection_results`: 检测结果（id, reply_id, is_hallucination, detection_layer, output_type, confidence, reason, created_at）
- `detection_batches`: 批次记录（id, filename, total_count, hallucination_count, created_at）
- `evaluation_runs`: 评估记录（id, accuracy, precision, recall, f1, false_positives, false_negatives, created_at）

**原因**：零配置，单文件备份，满足当前规模。后续可迁移到 PostgreSQL，SQL 语法兼容。

### 5. LLM 调用策略

**选择**：仅对 L3（无据陈述）case 调用 DeepSeek，Prompt 设计为"KB vs Reply 语义矛盾判定"

Prompt 模板核心要素：
- System: 幻觉检测专家角色，输出 JSON 格式
- Context: KB 原文 + Reply 原文
- Task: 判断是否存在矛盾/编造/遗漏，标注类型和原因
- Temperature: 0.0（保证判定一致性）

**原因**：DeepSeek v3 对结构化对比任务准确性高，JSON mode 保证可解析输出。Temperature=0 保证可复现。

## Risks / Trade-offs

| 风险 | 缓解 |
|---|---|
| h13（孕妇面膜）需外部医学知识，LLM 可能无法识别视黄醇衍生物的风险 | LLM 在安全相关 case 上表现通常较好（预训练包含医学常识）；若仍有误判，后期可加入安全关键词触发规则 |
| h20（鞋码遗漏）和 h16（色差措辞差异）容易误判 | Stage 2 的 LLM prompt 重点设计"遗漏"判定逻辑；评估阶段重点分析误判 case |
| DeepSeek API 可能不稳定或响应慢 | 代码中实现重试机制（max 3 retries）、超时控制（30s）；规则引擎可独立运行不依赖 API |
| Streamlit 与 FastAPI 双进程部署复杂度 | 提供 `run.py` 一键启动脚本和 Docker Compose |
| API key 安全 | 通过环境变量注入，.env 文件加入 .gitignore，代码中不硬编码 |
