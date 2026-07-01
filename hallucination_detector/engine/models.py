from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DetectionLayer(str, Enum):
    L1_DIRECT_CONTRADICTION = "L1"
    L2_CAPABILITY_OVERREACH = "L2"
    L3_UNSUPPORTED_CLAIM = "L3"


class OutputType(str, Enum):
    POLICY_FABRICATION = "policy_fabrication"
    PARAM_FABRICATION = "param_fabrication"
    POLICY_DEVIATION = "policy_deviation"
    CAPABILITY_OVERREACH = "capability_overreach"
    PROMOTION_FABRICATION = "promotion_fabrication"
    INFO_FABRICATION = "info_fabrication"
    SAFETY_MISLEADING = "safety_misleading"
    INFO_OMISSION = "info_omission"


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
