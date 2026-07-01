from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DetectionLayer(str, Enum):
    L1_DIRECT_CONTRADICTION = "直接矛盾"
    L2_CAPABILITY_OVERREACH = "能力越界"
    L3_UNSUPPORTED_CLAIM = "无据陈述"


class OutputType(str, Enum):
    POLICY_FABRICATION = "政策编造"
    PARAM_FABRICATION = "参数编造"
    POLICY_DEVIATION = "政策偏差"
    CAPABILITY_OVERREACH = "能力越界"
    PROMOTION_FABRICATION = "优惠编造"
    INFO_FABRICATION = "信息编造"
    SAFETY_MISLEADING = "安全误导"
    INFO_OMISSION = "信息遗漏"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class DetectionInput(BaseModel):
    id: Optional[str] = None
    user_question: str
    system_reply: str
    knowledge_base: str


class DetectionResult(BaseModel):
    id: str
    is_hallucination: Optional[bool] = None
    detection_layer: Optional[DetectionLayer] = None
    output_type: Optional[OutputType] = None
    confidence: Confidence = Confidence.LOW
    reason: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class BatchRecord(BaseModel):
    id: Optional[int] = None
    filename: str
    total_count: int
    hallucination_count: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class EvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    false_positive_cases: list[dict] = []
    false_negative_cases: list[dict] = []
