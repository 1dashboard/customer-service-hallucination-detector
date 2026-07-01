## Why

团队的智能客服系统基于 LLM 生成回复时，经常"说瞎话"——编造不存在的优惠政策、杜撰产品参数、假装具备未接入的查询能力。业务方需要一个自动化工具来批量检测这些"幻觉"回复，替代当前的人工抽检（效率低、覆盖度差），及时发现并拦截错误回复，降低客诉风险。

## What Changes

- 新增两层幻觉分类体系（检测层 3 类 + 输出层 8 类），覆盖参数编造、能力越界、信息编造、安全误导等典型场景
- 新增双阶段检测引擎：Stage 1 规则引擎（jieba + 正则 + 实体对齐）处理确定性强的矛盾，Stage 2 LLM（DeepSeek v3）处理需语义推理的复杂 case
- 新增 Streamlit Web Dashboard，支持批量上传回复数据、查看检测结果、追溯每条回复的判定依据
- 新增 FastAPI REST API，支持外部系统（工单系统、客服后台）集成调用
- 新增评估工具，自动对比 ground truth 计算检出率、漏检率和误报率
- 新增 SQLite 存储检测历史，支持按类型、时间、状态查询

## Capabilities

### New Capabilities

- `hallucination-detection`: 核心检测引擎，包含规则层（L1 直接矛盾、L2 能力越界）和 LLM 语义层（L3 无据陈述），输出结构化的检测结果（是否幻觉、类型、置信度、判定依据）
- `detection-api`: FastAPI REST 接口，提供 POST /api/detect（单条/批量检测）、GET /api/results（查询历史结果）、POST /api/evaluate（对比 ground truth 评估）
- `detection-dashboard`: Streamlit Web 界面，提供数据上传、检测结果浏览与筛选、单条详情展开、评估指标可视化

### Modified Capabilities

<!-- 无已有 capability 需要修改 -->

## Impact

- 新增目录 `hallucination_detector/`，包含 engine/、api/、dashboard/、data/ 模块
- 新增依赖：fastapi、streamlit、jieba、openai（DeepSeek API 兼容）、sqlite3、pandas、uvicorn
- 敏感信息：DeepSeek API key 需通过环境变量 `DEEPSEEK_API_KEY` 注入，不可硬编码
- 数据文件：replies.json（待检测数据）和 ground_truth.json（人工标注）置于 `data/` 目录
