## ADDED Requirements

### Requirement: Two-layer hallucination classification
The system SHALL classify hallucinations using a detection layer (3 types) for engine decisions and an output layer (8 types) for business reporting.

Detection layer types:
- L1 (direct_contradiction): KB has explicit value/rule, reply changes it
- L2 (capability_overreach): KB declares no capability, reply claims execution
- L3 (unsupported_claim): KB has no info or has warnings, reply makes affirmative statement

Output layer types: policy_fabrication, param_fabrication, policy_deviation, capability_overreach, promotion_fabrication, info_fabrication, safety_misleading, info_omission

#### Scenario: L1 detection triggered
- **WHEN** KB states "7天无理由退货" and reply states "30天无理由退货"
- **THEN** system SHALL classify as L1 (direct_contradiction) with HIGH confidence and map output to policy_fabrication

#### Scenario: L2 detection triggered
- **WHEN** KB states "客服系统未接入物流查询接口" and reply claims "包裹目前在南京转运中心"
- **THEN** system SHALL classify as L2 (capability_overreach) with HIGH confidence and map output to capability_overreach

#### Scenario: L3 detection triggered
- **WHEN** KB states "含视黄醇衍生物，孕妇建议咨询医生" and reply states "孕妇可以放心使用"
- **THEN** system SHALL classify as L3 (unsupported_claim) with MEDIUM confidence and map output to safety_misleading

### Requirement: Stage 1 rule engine with entity alignment
The system SHALL implement a rule-based detection engine using jieba word segmentation, regex patterns, and NER entity extraction.

Stage 1 SHALL:
- Extract numeric entities (numbers, percentages, durations) from both KB and reply
- Extract named entities (product names, locations, person names) from both KB and reply
- Compare extracted entities: if KB entity != reply entity, flag as L1
- Match KB negation patterns ("无", "未接入", "不具备", "不支持") against reply action verbs ("已帮您", "我帮您查了", "已升级", "已修改")
- Flag matched negation patterns as L2
- Return confidence: "HIGH" for matches, "LOW" for uncertain cases

#### Scenario: Numeric value mismatch detected
- **WHEN** KB contains "蓝牙5.0" and reply contains "蓝牙5.3"
- **THEN** system SHALL detect the mismatch and flag as L1 direct_contradiction

#### Scenario: Negation pattern matched
- **WHEN** KB contains "未接入物流查询接口" and reply contains "我帮您查了"
- **THEN** system SHALL detect the capability mismatch and flag as L2 capability_overreach

#### Scenario: Entity alignment passes
- **WHEN** KB contains "不支持货到付款" and reply contains "不支持货到付款"
- **THEN** system SHALL NOT flag as hallucination

### Requirement: Stage 2 LLM semantic judgment
The system SHALL send LOW-confidence cases from Stage 1 to DeepSeek v3 API for semantic-level comparison.

The LLM prompt SHALL include:
- System message defining the hallucination detection task
- KB content and reply content as input
- Instruction to classify into: contradiction, fabrication, omission, or none
- Instruction to output structured JSON with fields: is_hallucination, type, reason, confidence

The LLM caller SHALL:
- Use temperature=0.0 for consistent results
- Implement retry logic (max 3 attempts, exponential backoff)
- Timeout after 30 seconds per request
- Return a fallback result on persistent failure (is_hallucination: null, reason: "LLM call failed")

#### Scenario: LLM identifies safety risk
- **WHEN** KB warns "含视黄醇衍生物" for pregnant women and reply says "可以放心使用"
- **THEN** LLM SHALL return is_hallucination=true, type=safety_misleading, confidence=MEDIUM

#### Scenario: LLM identifies information omission
- **WHEN** KB says "30%用户反馈偏大半码" and reply says "尺码标准不偏"
- **THEN** LLM SHALL return is_hallucination=true, type=info_omission

#### Scenario: LLM confirms no hallucination
- **WHEN** KB and reply convey the same meaning with different wording (e.g., both acknowledge color variation risk)
- **THEN** LLM SHALL return is_hallucination=false

#### Scenario: LLM call fails after retries
- **WHEN** DeepSeek API is unreachable after 3 retries
- **THEN** system SHALL return is_hallucination=null with reason="LLM call failed" and NOT crash

### Requirement: Detection-to-output layer mapping
The system SHALL map each detection-layer classification to the corresponding output-layer business label.

Mapping rules:
- L1 (direct_contradiction) with policy/number content → policy_fabrication or policy_deviation or param_fabrication
- L2 (capability_overreach) → capability_overreach
- L3 (unsupported_claim) → promotion_fabrication, info_fabrication, safety_misleading, or info_omission based on LLM-identified subtype

#### Scenario: Mixed correct and incorrect content
- **WHEN** reply correctly states "支持电子发票" but incorrectly claims "支持纸质发票"
- **THEN** system SHALL flag the reply as hallucination and detail which part is incorrect

### Requirement: Batch detection processing
The system SHALL accept a list of reply items (each containing id, user_question, system_reply, knowledge_base) and return detection results with the same IDs.

#### Scenario: Process 20 replies from file
- **WHEN** provided with 20 reply items from replies.json
- **THEN** system SHALL return 20 detection results each containing: id, is_hallucination, detection_layer (L1/L2/L3), output_type, confidence, reason

### Requirement: Evaluation against ground truth
The system SHALL compare detection results with ground_truth.json and compute accuracy, precision, recall, F1-score, false positives, and false negatives.

#### Scenario: Calculate evaluation metrics
- **WHEN** provided with 20 detection results and 20 ground truth labels
- **THEN** system SHALL output a report with: overall accuracy, precision, recall, F1-score, list of false positives, list of false negatives

#### Scenario: Misclassification analysis
- **WHEN** false positives or false negatives exist
- **THEN** system SHALL output the detailed reason for each misclassified case
