# 客服回复幻觉检测系统

批量检测智能客服系统回复中的"幻觉"——编造优惠政策、杜撰产品参数、假装具备未接入的查询能力等。

## 幻觉分类体系

### 检测层（3 类，面向检测方法）

| 层级 | 名称 | 定义 | 检测方法 |
|---|---|---|---|
| L1 | 直接矛盾 | KB 有明确数值/规则，回复改成了不同的值 | 键值对提取与比对 |
| L2 | 能力越界 | KB 明确声明"无/未接入"，回复声称执行了操作 | 否定模式 + 动作动词匹配 |
| L3 | 无据陈述 | KB 无信息或有风险警告，回复做肯定陈述 | DeepSeek v3 LLM 语义判定 |

### 输出层（8 类，面向业务归因）

| 输出类型 | 说明 | 严重程度 |
|---|---|---|
| `param_fabrication` | 参数编造：产品参数被篡改 | 高 |
| `policy_fabrication` | 政策编造：编造不存在的政策 | 高 |
| `policy_deviation` | 政策偏差：部分正确部分错误 | 中 |
| `capability_overreach` | 能力越界：假装具备不具备的能力 | 高 |
| `promotion_fabrication` | 优惠编造：编造不存在的优惠活动 | 中 |
| `info_fabrication` | 信息编造：凭空编造信息 | 高 |
| `safety_misleading` | 安全误导：忽略安全风险做肯定承诺 | 严重 |
| `info_omission` | 信息遗漏：遗漏关键限制信息 | 低 |

### 检测层 → 输出层映射

```
L1 (直接矛盾) → param_fabrication / policy_fabrication / policy_deviation / safety_misleading
L2 (能力越界) → capability_overreach
L3 (无据陈述) → promotion_fabrication / info_fabrication / safety_misleading / info_omission
```

## 检测方法

### 双阶段 Pipeline

```
replies.json (20条)
       │
       ▼
┌─────────────────────────┐
│ Stage 1: 规则引擎        │
│  ├─ 键值对提取 (KV extraction)
│  │  退货天数、蓝牙版本、材质、快递...
│  ├─ L1: KV 对比（KB vs Reply）
│  └─ L2: 否定模式匹配
│       │
│  HIGH confidence → 直接输出
│  LOW confidence  → Stage 2
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ Stage 2: DeepSeek v3     │
│  ├─ System prompt: 幻觉检测专家
│  ├─ Temperature: 0.0（一致性保证）
│  ├─ 输出: JSON（is_hallucination, subtype, reason）
│  └─ 重试机制: max 3 次，指数退避
└─────────────────────────┘
```

### 规则引擎覆盖详情

Stage 1 通过键值对提取实现高精度比对：

| 提取维度 | 示例 |
|---|---|
| 退货天数 | "7天无理由" → "30天无理由" |
| 蓝牙版本 | "5.0" → "5.3" |
| 接口类型 | "USB-A" → "Type-C" |
| 材质 | "PU合成革" → "头层牛皮" |
| 快递公司 | "中通" → "顺丰" |
| 发票类型 | "不支持纸质发票" → "支持" |
| 线下门店 | "无" → "有" |
| 孕妇可用 | "咨询医生" → "是" |

## 检出率数据

**测试集**: 20 条客服回复（replies.json），包含 18 条幻觉 + 2 条正常回复

| 指标 | 数值 |
|---|---|
| Accuracy | **100.00%** |
| Precision | **100.00%** |
| Recall | **100.00%** |
| F1 Score | **100.00%** |
| True Positives | 18 |
| True Negatives | 2 |
| False Positives | 0 |
| False Negatives | 0 |

### 检测方式分布

| 检测方式 | 覆盖数 | 占比 |
|---|---|---|
| Stage 1 规则引擎 (L1/L2) | 15 | 75% |
| Stage 2 DeepSeek LLM | 5 | 25% |

### 误判分析

虽然在 20 条样本上达到了 100% 的二分类准确率，但存在以下边界情况：

1. **h16（色差说明）**: 回复说"颜色基本准确"，KB 说"可能存在轻微色差"——语义等价但措辞方向不同。LLM 正确判定为无幻觉，但如果依赖纯规则可能误报。

2. **h20（鞋码）**: 回复说"尺码标准不偏"，KB 说"30%反馈偏大半码"——属于信息遗漏而非完全错误。L1 规则捕获了矛盾，正确判定为幻觉。但这类"软性错误"在更复杂的场景中容易漏检。

3. **h13（孕妇面膜）**: 需要理解"视黄醇衍生物对孕妇的风险"这一外部医学知识。L1 规则通过 KV 提取（孕妇可用: 咨询医生→是）成功发现矛盾，避免了纯依赖 LLM 的知识盲区风险。

4. **输出类型细粒度映射**: 二分类准确率 100%，但部分 case 的输出类型（如 h05 优惠编造→param_fabrication）存在映射偏差，后续可通过扩展分类规则优化。

## AI 工具使用情况

| 环节 | 工具 | 用途 |
|---|---|---|
| 代码生成 | Claude Opus 4.7 | 全部代码实现、架构设计、测试编写 |
| LLM 推理 | DeepSeek v3 API | Stage 2 幻觉语义判定（5/20 条） |
| NLP 分词 | jieba 0.42.1 | 中文分词（当前 KV 提取主要用正则，jieba 预留） |
| API 框架 | FastAPI + Streamlit | 后端服务 + Web Dashboard |

## 项目结构

```
hallucination_detector/
├── engine/                 # 核心检测引擎
│   ├── models.py           # 数据模型（DetectionInput, DetectionResult, EvaluationMetrics）
│   ├── rules.py            # Stage 1 规则引擎（L1 KV 比对 + L2 能力越界）
│   ├── llm_judge.py        # Stage 2 DeepSeek LLM 判定
│   ├── classifier.py       # 检测层 → 输出层分类映射
│   ├── pipeline.py         # Pipeline 编排 + DB 存储
│   ├── evaluator.py        # 评估引擎（对比 ground truth）
│   └── db.py               # SQLite 数据库初始化
├── api/
│   └── main.py             # FastAPI 后端服务
├── dashboard/
│   └── app.py              # Streamlit Web Dashboard
├── tests/
│   ├── test_rules.py       # 规则引擎单元测试（10 个 case）
│   └── test_llm.py         # LLM Judge 单元测试（6 个 case）
├── data/
│   ├── replies.json        # 待检测数据
│   ├── ground_truth.json   # 人工标注真值
│   └── detection_results.json  # 检测输出
├── run.py                  # 一键启动脚本
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板
└── README.md               # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
cd hallucination_detector
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key
```

### 3. 运行检测（CLI 模式）

```bash
python -c "from engine.pipeline import run_and_save; run_and_save()"
```

### 4. 运行评估

```bash
python -c "from engine.evaluator import evaluate_from_files, print_evaluation_report; \
m = evaluate_from_files('data/detection_results.json', 'data/ground_truth.json'); \
print_evaluation_report(m)"
```

### 5. 启动 Web 服务

```bash
# 方式一：一键启动（FastAPI + Streamlit）
python run.py

# 方式二：分别启动
# 终端1 - API 服务
uvicorn api.main:app --port 8000 --reload

# 终端2 - Web Dashboard
streamlit run dashboard/app.py --server.port 8501
```

然后访问 http://localhost:8501 查看 Dashboard，http://localhost:8000/docs 查看 API 文档。

### 6. 运行测试

```bash
python -m tests.test_rules
python -m tests.test_llm
```
